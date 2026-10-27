"""Online multi-observation LiDAR voxel evidence and authority gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Iterable

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import FusionGates, load_reconstruction_config

from .observations import FilteredFrame, LidarObservationFrame


class VoxelState(str, Enum):
    CANDIDATE = "CANDIDATE"
    MEASURED = "MEASURED"
    UNSTABLE = "UNSTABLE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class VoxelEvidence:
    key: tuple[int, int, int]
    position: np.ndarray
    observation_count: int
    independent_views: int
    temporal_support_s: float
    principal_stddev_m: float
    point_to_plane_p95_m: float
    static_probability: float
    state: VoxelState
    pose_generation: str
    source_artifact_sha256: tuple[str, ...]
    incidence_cosine: float = 1.0

    def __post_init__(self) -> None:
        point = np.asarray(self.position, dtype=np.float64)
        point.setflags(write=False)
        object.__setattr__(self, "position", point)


@dataclass
class _MutableVoxel:
    key: tuple[int, int, int]
    points: list[np.ndarray]
    origins: list[np.ndarray]
    directions: list[np.ndarray]
    timestamps_ns: list[int]
    generations: set[str]
    artifacts: set[str]
    occupied_votes: int = 0
    free_votes: int = 0
    unstable_generations: set[str] | None = None
    state: VoxelState = VoxelState.CANDIDATE
    metrics_cache: dict[str, float | int] | None = None

    def __post_init__(self) -> None:
        if self.unstable_generations is None:
            self.unstable_generations = set()


class MeasuredVoxelMap:
    """Fuse accepted LiDAR observations without permitting RGB-D to move means."""

    def __init__(self, gates: FusionGates | None = None, voxel_size_m: float | None = None) -> None:
        self.gates = gates or load_reconstruction_config().fusion
        self.voxel_size_m = self.gates.surfacing.voxel_size_m if voxel_size_m is None else float(voxel_size_m)
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        self._voxels: dict[tuple[int, int, int], _MutableVoxel] = {}
        self._pose_generation = ""
        self._lock = threading.RLock()

    def integrate(
        self,
        frame: LidarObservationFrame,
        filtered: FilteredFrame | None = None,
    ) -> int:
        indices = np.arange(len(frame.points_xyz), dtype=np.int64) if filtered is None else filtered.accepted_indices
        if len(indices) == 0:
            return 0
        world_points = frame.points_xyz[indices] @ frame.pose_map_from_lidar[:3, :3].T + frame.pose_map_from_lidar[:3, 3]
        direction = world_points - frame.pose_map_from_lidar[:3, 3]
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(direction_norm, 1e-12)
        voxel_keys = np.floor(world_points / self.voxel_size_m).astype(np.int64)
        # Multiple returns in one LiDAR frame are not independent evidence.
        # Keep one deterministic observation per occupied voxel and reserve
        # promotion for repeat frames/viewpoints.
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        unique_indices.sort()
        world_points = world_points[unique_indices]
        direction = direction[unique_indices]
        voxel_keys = voxel_keys[unique_indices]
        with self._lock:
            self._pose_generation = frame.pose_generation
        for point, view_direction, voxel_key in zip(world_points, direction, voxel_keys):
            with self._lock:
                key = tuple(voxel_key.tolist())
                evidence = self._voxels.get(key)
                if evidence is None:
                    evidence = _MutableVoxel(key, [], [], [], [], set(), set())
                    self._voxels[key] = evidence
                evidence.metrics_cache = None
                evidence.points.append(np.asarray(point, dtype=np.float64))
                evidence.origins.append(np.asarray(frame.pose_map_from_lidar[:3, 3], dtype=np.float64))
                evidence.directions.append(np.asarray(view_direction, dtype=np.float64))
                evidence.timestamps_ns.append(frame.record_timestamp_ns)
                evidence.generations.add(frame.pose_generation)
                evidence.artifacts.add(frame.artifact_sha256)
                evidence.occupied_votes += 1
                self._update_state(evidence)
        return len(world_points)

    def observe_free_space(self, key: tuple[int, int, int], pose_generation: str) -> None:
        with self._lock:
            evidence = self._voxels.get(tuple(key))
            if evidence is None:
                return
            evidence.free_votes += 1
            if evidence.free_votes + evidence.occupied_votes >= self.gates.dynamic_evidence.minimum_votes:
                fraction = evidence.free_votes / max(evidence.free_votes + evidence.occupied_votes, 1)
                if self.gates.dynamic_evidence.is_unstable(
                    vote_count=evidence.free_votes + evidence.occupied_votes,
                    free_space_fraction=fraction,
                ):
                    evidence.unstable_generations.add(pose_generation)
                    evidence.state = (
                        VoxelState.REJECTED
                        if self.gates.dynamic_evidence.is_rejected(
                            unstable_generations=len(evidence.unstable_generations)
                        )
                        else VoxelState.UNSTABLE
                    )

    def voxel(self, key_or_position: tuple[int, int, int] | Iterable[float]) -> VoxelEvidence | None:
        values = tuple(float(value) for value in key_or_position)
        if len(values) != 3:
            raise ValueError("voxel key or position must have three values")
        key = tuple(int(round(value)) for value in values) if max(abs(value) for value in values) > 10 else tuple(
            np.floor(np.asarray(values) / self.voxel_size_m).astype(np.int64).tolist()
        )
        with self._lock:
            evidence = self._voxels.get(key)
            copied = None if evidence is None else self._copy_mutable(evidence)
            pose_generation = self._pose_generation
        return None if copied is None else self._snapshot(copied, pose_generation)

    def measured_points(self) -> np.ndarray:
        with self._lock:
            point_sets = [list(value.points) for value in self._voxels.values() if value.state is VoxelState.MEASURED]
        measured = [np.mean(np.asarray(points, dtype=np.float64), axis=0) for points in point_sets]
        return np.asarray(measured, dtype=np.float64).reshape(-1, 3)

    def evidence(self, state: VoxelState | None = None) -> tuple[VoxelEvidence, ...]:
        values, pose_generation = self._snapshot_inputs(state)
        snapshots = (self._snapshot(value, pose_generation) for value in values)
        return tuple(sorted(snapshots, key=lambda item: item.key))

    def update_measured_position_from_depth(self, key: tuple[int, int, int], position_xyz: Iterable[float]) -> None:
        with self._lock:
            evidence = self._voxels.get(tuple(key))
            if evidence is not None and evidence.state is VoxelState.MEASURED:
                raise PermissionError("RGB-D validation cannot move LIDAR_MEASURED geometry")

    def snapshot(self) -> dict:
        with self._lock:
            states = {state.value: 0 for state in VoxelState}
            for value in self._voxels.values():
                states[value.state.value] += 1
            return {
                "voxel_size_m": self.voxel_size_m,
                "voxel_count": len(self._voxels),
                "candidate_points": states[VoxelState.CANDIDATE.value],
                "measured_points": states[VoxelState.MEASURED.value],
                "unstable_points": states[VoxelState.UNSTABLE.value],
                "rejected_points": states[VoxelState.REJECTED.value],
                "pose_generation": self._pose_generation,
            }

    def _snapshot_inputs(self, state: VoxelState | None = None) -> tuple[list[_MutableVoxel], str]:
        with self._lock:
            values = (
                value for value in self._voxels.values()
                if state is None or value.state is state
            )
            return [self._copy_mutable(value) for value in values], self._pose_generation

    @staticmethod
    def _copy_mutable(evidence: _MutableVoxel) -> _MutableVoxel:
        return _MutableVoxel(
            key=evidence.key,
            points=list(evidence.points),
            origins=list(evidence.origins),
            directions=list(evidence.directions),
            timestamps_ns=list(evidence.timestamps_ns),
            generations=set(evidence.generations),
            artifacts=set(evidence.artifacts),
            occupied_votes=evidence.occupied_votes,
            free_votes=evidence.free_votes,
            unstable_generations=set(evidence.unstable_generations),
            state=evidence.state,
            metrics_cache=None if evidence.metrics_cache is None else dict(evidence.metrics_cache),
        )

    def _update_state(self, evidence: _MutableVoxel) -> None:
        if evidence.state is not VoxelState.CANDIDATE:
            return
        promotion = self.gates.measured_promotion
        if evidence.occupied_votes < promotion.minimum_observations:
            return
        temporal_support_s = (
            (max(evidence.timestamps_ns) - min(evidence.timestamps_ns)) / 1e9
            if evidence.timestamps_ns
            else 0.0
        )
        if temporal_support_s < promotion.minimum_temporal_support_s:
            return
        metrics = self._metrics(evidence)
        # Grazing-incidence returns are unstable for surface reconstruction;
        # retain them as candidates for diagnostics but never promote them to
        # measured geometry. The threshold is deliberately permissive for
        # walls/floors while rejecting near-tangent wave echoes.
        incidence_cosine = float(metrics.get("incidence_cosine", 0.0))
        # A zero value means the local normal/incidence diagnostic is
        # indeterminate (for example a degenerate single-point neighborhood),
        # not that a measured return was proven grazing. Preserve the formal
        # promotion gates in that case and reject only a finite positive value
        # that is explicitly below the grazing threshold.
        if 0.0 < incidence_cosine < 0.20:
            return
        # Keep diagnostic metrics (for example incidence_cosine) out of the
        # formal gate call so the gate contract remains explicit and stable.
        if promotion.accepts(
            observation_count=int(metrics["observation_count"]),
            independent_views=int(metrics["independent_views"]),
            temporal_support_s=float(metrics["temporal_support_s"]),
            principal_stddev_m=float(metrics["principal_stddev_m"]),
            point_to_plane_p95_m=float(metrics["point_to_plane_p95_m"]),
            static_probability=float(metrics["static_probability"]),
        ):
            evidence.state = VoxelState.MEASURED

    def _metrics(self, evidence: _MutableVoxel) -> dict[str, float | int]:
        if evidence.metrics_cache is not None:
            return dict(evidence.metrics_cache)
        points = np.asarray(evidence.points, dtype=np.float64)
        origins = np.asarray(evidence.origins, dtype=np.float64)
        directions = np.asarray(evidence.directions, dtype=np.float64)
        centered = points - points.mean(axis=0)
        principal_stddev = float(np.max(np.std(points, axis=0))) if len(points) else math.inf
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1] if len(points) >= 3 else np.array([0.0, 0.0, 1.0])
        residuals = np.abs(centered @ normal)
        incidence = np.abs(directions @ normal) if len(directions) else np.empty(0)
        independent = 1
        for index in range(len(origins)):
            for previous in range(index):
                baseline = float(np.linalg.norm(origins[index] - origins[previous]))
                cosine = float(np.clip(np.dot(directions[index], directions[previous]), -1.0, 1.0))
                separation = math.degrees(math.acos(cosine))
                if self.gates.independent_view.accepts(
                    origin_baseline_m=baseline,
                    direction_separation_deg=separation,
                ):
                    independent += 1
                    break
        metrics = {
            "observation_count": len(points),
            "independent_views": independent,
            "temporal_support_s": (max(evidence.timestamps_ns) - min(evidence.timestamps_ns)) / 1e9 if evidence.timestamps_ns else 0.0,
            "principal_stddev_m": principal_stddev,
            "point_to_plane_p95_m": float(np.percentile(residuals, 95)) if len(residuals) else math.inf,
            "static_probability": evidence.occupied_votes / max(evidence.occupied_votes + evidence.free_votes, 1),
            "incidence_cosine": float(np.median(incidence)) if len(incidence) else 0.0,
        }
        evidence.metrics_cache = metrics
        return dict(metrics)

    def _snapshot(self, evidence: _MutableVoxel, pose_generation: str | None = None) -> VoxelEvidence:
        metrics = (
            dict(evidence.metrics_cache)
            if evidence.metrics_cache is not None
            else self._metrics(evidence)
        )
        return VoxelEvidence(
            key=evidence.key,
            position=np.mean(np.asarray(evidence.points), axis=0),
            observation_count=int(metrics["observation_count"]),
            independent_views=int(metrics["independent_views"]),
            temporal_support_s=float(metrics["temporal_support_s"]),
            principal_stddev_m=float(metrics["principal_stddev_m"]),
            point_to_plane_p95_m=float(metrics["point_to_plane_p95_m"]),
            static_probability=float(metrics["static_probability"]),
            state=evidence.state,
            pose_generation=self._pose_generation if pose_generation is None else pose_generation,
            source_artifact_sha256=tuple(sorted(evidence.artifacts)),
            incidence_cosine=float(metrics.get("incidence_cosine", 1.0)),
        )
