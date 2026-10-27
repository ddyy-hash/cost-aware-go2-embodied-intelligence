"""Sparse witness-backed detailed surface state and atomic checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Optional
import uuid

import numpy as np

from .live_protocol import FrameIdentity, LiveKeyframeMessage
from .sensor_frame import (
    IDENTITY_SENSOR_FRAME_V1,
    get_sensor_frame_transform,
    normalize_sensor_frame_transform_id,
    transform_points,
)


class SupportState(IntEnum):
    OBSERVED_UNRESOLVED = 1
    SURFACE_SUPPORTED = 2


@dataclass(frozen=True, order=True)
class TileId:
    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class ObservationWitness:
    frame_identity: FrameIdentity
    sensor_origin: tuple[float, float, float]
    view_direction: tuple[float, float, float]


@dataclass(frozen=True)
class SurfelSnapshot:
    key: tuple[int, int, int]
    centroid: np.ndarray
    covariance_accumulator: np.ndarray
    observation_count: int
    independent_frame_support: int
    witnesses: tuple[ObservationWitness, ...]
    mean_view_direction: np.ndarray
    latest_sensor_origin: np.ndarray
    first_timestamp_ns: int
    last_timestamp_ns: int
    normal: np.ndarray
    curvature: float
    normal_confidence: float
    support_state: SupportState


@dataclass(frozen=True)
class MeshingSurfel:
    """Compact immutable surfel state transferred to the mesh worker."""

    key: tuple[int, int, int]
    centroid: tuple[float, float, float]
    observation_count: int
    independent_frame_support: int
    mean_view_direction: tuple[float, float, float]


@dataclass(frozen=True)
class IntegrationResult:
    frame_identity: FrameIdentity
    integrated_cells: int
    dirty_tiles: tuple[TileId, ...]
    duplicate: bool = False


@dataclass(frozen=True)
class SurfaceCheckpoint:
    generation_id: str
    pose_revision_id: str
    generation_path: Path
    last_integrated_identity: FrameIdentity
    sensor_frame_transform_id: str = IDENTITY_SENSOR_FRAME_V1


@dataclass
class _MutableSurfel:
    key: tuple[int, int, int]
    centroid: np.ndarray
    covariance_accumulator: np.ndarray
    observation_count: int
    witnesses: list[ObservationWitness]
    view_direction_sum: np.ndarray
    latest_sensor_origin: np.ndarray
    first_timestamp_ns: int
    last_timestamp_ns: int
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    curvature: float = math.inf
    normal_confidence: float = 0.0
    support_state: SupportState = SupportState.OBSERVED_UNRESOLVED


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class LiveSurfaceMap:
    JOURNAL_LIMIT = 4096
    _NEIGHBOR_OFFSETS = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )

    def __init__(
        self,
        *,
        map_id: str,
        source_config_sha256: str,
        sensor_frame_transform_id: str = IDENTITY_SENSOR_FRAME_V1,
        calibration_id: Optional[str] = None,
        cell_m: float = 0.005,
        tile_m: float = 0.25,
    ) -> None:
        if not map_id:
            raise ValueError("map_id is required")
        if len(source_config_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_config_sha256):
            raise ValueError("source_config_sha256 must be a lowercase SHA-256")
        if not math.isfinite(cell_m) or cell_m <= 0 or not math.isfinite(tile_m) or tile_m <= 0:
            raise ValueError("cell and tile sizes must be finite and positive")
        self.map_id = str(map_id)
        self.source_config_sha256 = source_config_sha256
        self.sensor_frame_transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)
        if calibration_id is not None and not str(calibration_id):
            raise ValueError("calibration_id must be non-empty when provided")
        self.calibration_id = None if calibration_id is None else str(calibration_id)
        self.cell_m = float(cell_m)
        self.tile_m = float(tile_m)
        self._surfels: dict[tuple[int, int, int], _MutableSurfel] = {}
        self._keys_by_tile: dict[TileId, set[tuple[int, int, int]]] = {}
        self._dirty_tiles: set[TileId] = set()
        self._journal: list[dict] = []
        self._journal_lookup: dict[tuple[str, str, str, str, int], tuple[str, str]] = {}
        self._last_integrated_identity: Optional[FrameIdentity] = None
        self._pose_revision_id: Optional[str] = None
        self._pose_revision_frame_count = 0
        self._lock = threading.RLock()

    @property
    def last_integrated_identity(self) -> Optional[FrameIdentity]:
        with self._lock:
            return self._last_integrated_identity

    @property
    def pose_revision_id(self) -> Optional[str]:
        with self._lock:
            return self._pose_revision_id

    def _tile_for_key(self, key: tuple[int, int, int]) -> TileId:
        centre = (np.asarray(key, dtype=np.float64) + 0.5) * self.cell_m
        tile = np.floor(centre / self.tile_m).astype(np.int64)
        return TileId(*(int(value) for value in tile))

    def _associated_key(
        self,
        raw_key: tuple[int, int, int],
        point: np.ndarray,
        updated_keys: set[tuple[int, int, int]],
    ) -> tuple[int, int, int] | None:
        maximum_distance2 = (self.cell_m * 1.5) ** 2
        owner_tile = self._tile_for_key(raw_key)
        nearest_key = None
        nearest_distance2 = maximum_distance2 + 1e-15
        for dx, dy, dz in self._NEIGHBOR_OFFSETS:
            candidate_key = (raw_key[0] + dx, raw_key[1] + dy, raw_key[2] + dz)
            candidate = self._surfels.get(candidate_key)
            if candidate is None or self._tile_for_key(candidate_key) != owner_tile:
                continue
            delta = point - candidate.centroid
            distance2 = float(np.dot(delta, delta))
            if distance2 <= nearest_distance2:
                nearest_key = candidate_key
                nearest_distance2 = distance2
        if nearest_key is not None:
            return None if nearest_key in updated_keys else nearest_key
        return None if raw_key in updated_keys else raw_key

    @staticmethod
    def _identity_payload(frame: LiveKeyframeMessage) -> dict:
        identity = frame.identity
        return {
            "building_id": identity.building_id,
            "session_id": identity.session_id,
            "map_id": identity.map_id,
            "producer_instance_id": identity.producer_instance_id,
            "sequence": identity.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "calibration_id": frame.calibration_id,
            "point_bytes_sha256": frame.point_bytes_sha256,
            "sensor_frame_transform_id": frame.sensor_frame_transform_id,
            "pose_map_from_lidar": [float(value) for value in frame.pose_map_from_lidar.reshape(-1)],
        }

    @staticmethod
    def _extend_pose_revision(previous: Optional[str], item: dict) -> str:
        pose_record = {
            "building_id": item["building_id"],
            "session_id": item["session_id"],
            "map_id": item["map_id"],
            "producer_instance_id": item["producer_instance_id"],
            "sequence": item["sequence"],
            "timestamp_ns": item["timestamp_ns"],
            "sensor_frame_transform_id": item.get("sensor_frame_transform_id", IDENTITY_SENSOR_FRAME_V1),
            "pose_map_from_lidar": item["pose_map_from_lidar"],
        }
        return hashlib.sha256(
            json.dumps(
                {
                    "schema_version": 1,
                    "previous_pose_revision_id": previous,
                    "canonical_pose": pose_record,
                },
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _legacy_pose_revision(journal: list[dict]) -> str:
        records = [
            {
                "building_id": item["building_id"],
                "session_id": item["session_id"],
                "map_id": item["map_id"],
                "producer_instance_id": item["producer_instance_id"],
                "sequence": item["sequence"],
                "timestamp_ns": item["timestamp_ns"],
                "pose_map_from_lidar": item["pose_map_from_lidar"],
            }
            for item in journal
        ]
        return hashlib.sha256(
            json.dumps(
                {"schema_version": 1, "canonical_poses": records},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _identity_from_payload(payload: dict) -> FrameIdentity:
        return FrameIdentity(
            str(payload["building_id"]),
            str(payload["session_id"]),
            str(payload["map_id"]),
            str(payload["producer_instance_id"]),
            int(payload["sequence"]),
        )

    def integrate(self, frame: LiveKeyframeMessage) -> IntegrationResult:
        if frame.identity.map_id != self.map_id:
            raise ValueError("keyframe map_id does not match the active surface")
        if frame.source_config_sha256 != self.source_config_sha256:
            raise ValueError("keyframe source configuration does not match the active surface")
        if frame.sensor_frame_transform_id != self.sensor_frame_transform_id:
            raise ValueError("keyframe sensor frame transform does not match the active surface")
        identity_tuple = frame.identity.as_tuple()
        with self._lock:
            if self.calibration_id is not None and frame.calibration_id != self.calibration_id:
                raise ValueError("keyframe calibration does not match the active surface")
            if self._last_integrated_identity is not None:
                if frame.identity.building_id != self._last_integrated_identity.building_id:
                    raise ValueError("keyframe building_id does not match the active surface")
                if frame.identity.session_id != self._last_integrated_identity.session_id:
                    raise ValueError("keyframe session_id does not match the active surface")
            existing = self._journal_lookup.get(identity_tuple)
            if existing is not None:
                existing_hash, existing_transform_id = existing
                if existing_hash != frame.point_bytes_sha256 or existing_transform_id != frame.sensor_frame_transform_id:
                    raise ValueError("keyframe identity conflicts with integrated surface evidence")
                return IntegrationResult(frame.identity, 0, (), duplicate=True)

        points = frame.points_xyz.astype(np.float64, copy=False)
        if frame.points_in_map_frame:
            world_points = points
            sensor_origin = frame.pose_map_from_lidar[:3, 3].astype(np.float64, copy=True)
        else:
            transform = get_sensor_frame_transform(frame.sensor_frame_transform_id)
            rectified = transform_points(points, frame.sensor_frame_transform_id)
            world_points = rectified @ frame.pose_map_from_lidar[:3, :3].T + frame.pose_map_from_lidar[:3, 3]
            sensor_origin = (
                frame.pose_map_from_lidar[:3, :3] @ transform.translation
                + frame.pose_map_from_lidar[:3, 3]
            )
        keys64 = np.floor(world_points / self.cell_m).astype(np.int64)
        int32 = np.iinfo(np.int32)
        if len(keys64) and (keys64.min() < int32.min or keys64.max() > int32.max):
            raise ValueError("surface cell key exceeds signed int32 storage")
        centres = (keys64.astype(np.float64) + 0.5) * self.cell_m
        distance2 = np.einsum("ij,ij->i", world_points - centres, world_points - centres)
        source_order = np.arange(len(world_points), dtype=np.int64)
        order = np.lexsort((source_order, distance2, keys64[:, 2], keys64[:, 1], keys64[:, 0]))
        sorted_keys = keys64[order]
        first = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        selected = order[first]
        world_points = world_points[selected]
        keys64 = keys64[selected]
        directions = world_points - sensor_origin
        lengths = np.linalg.norm(directions, axis=1)
        valid = lengths > 1e-12
        world_points = world_points[valid]
        keys64 = keys64[valid]
        directions = directions[valid] / lengths[valid, None]

        dirty: set[TileId] = set()
        with self._lock:
            if self.calibration_id is None:
                self.calibration_id = frame.calibration_id
            elif frame.calibration_id != self.calibration_id:
                raise ValueError("keyframe calibration does not match the active surface")
            updated_keys: set[tuple[int, int, int]] = set()
            for point, key_array, direction in zip(world_points, keys64, directions):
                raw_key = tuple(int(value) for value in key_array)
                key = self._associated_key(raw_key, point, updated_keys)
                if key is None:
                    continue
                surfel = self._surfels.get(key)
                witness = ObservationWitness(
                    frame.identity,
                    tuple(float(value) for value in sensor_origin),
                    tuple(float(value) for value in direction),
                )
                if surfel is None:
                    surfel = _MutableSurfel(
                        key=key,
                        centroid=point.copy(),
                        covariance_accumulator=np.zeros((3, 3), dtype=np.float64),
                        observation_count=1,
                        witnesses=[witness],
                        view_direction_sum=direction.copy(),
                        latest_sensor_origin=sensor_origin.copy(),
                        first_timestamp_ns=frame.timestamp_ns,
                        last_timestamp_ns=frame.timestamp_ns,
                    )
                    self._surfels[key] = surfel
                    self._keys_by_tile.setdefault(self._tile_for_key(key), set()).add(key)
                else:
                    count = surfel.observation_count + 1
                    delta = point - surfel.centroid
                    surfel.centroid = surfel.centroid + delta / count
                    surfel.covariance_accumulator += np.outer(delta, point - surfel.centroid)
                    surfel.observation_count = count
                    surfel.view_direction_sum += direction
                    surfel.latest_sensor_origin = sensor_origin.copy()
                    surfel.last_timestamp_ns = max(surfel.last_timestamp_ns, frame.timestamp_ns)
                    if len(surfel.witnesses) < 4 and self._is_independent(witness, surfel.witnesses):
                        surfel.witnesses.append(witness)
                surfel.support_state = (
                    SupportState.SURFACE_SUPPORTED
                    if len(surfel.witnesses) >= 2
                    else SupportState.OBSERVED_UNRESOLVED
                )
                dirty.add(self._tile_for_key(key))
                dirty.add(self._tile_for_key(raw_key))
                updated_keys.add(key)
            journal_item = self._identity_payload(frame)
            self._journal.append(journal_item)
            self._journal = self._journal[-self.JOURNAL_LIMIT:]
            self._journal_lookup = {
                self._identity_from_payload(item).as_tuple(): (
                    str(item["point_bytes_sha256"]),
                    normalize_sensor_frame_transform_id(item.get("sensor_frame_transform_id")),
                )
                for item in self._journal
            }
            self._pose_revision_id = self._extend_pose_revision(
                self._pose_revision_id, journal_item
            )
            self._pose_revision_frame_count += 1
            self._last_integrated_identity = frame.identity
            self._dirty_tiles.update(dirty)
        return IntegrationResult(frame.identity, len(world_points), tuple(sorted(dirty)))

    @staticmethod
    def _is_independent(candidate: ObservationWitness, existing: list[ObservationWitness]) -> bool:
        candidate_origin = np.asarray(candidate.sensor_origin, dtype=np.float64)
        candidate_direction = np.asarray(candidate.view_direction, dtype=np.float64)
        for witness in existing:
            if witness.frame_identity.sequence == candidate.frame_identity.sequence:
                return False
            origin = np.asarray(witness.sensor_origin, dtype=np.float64)
            direction = np.asarray(witness.view_direction, dtype=np.float64)
            baseline = float(np.linalg.norm(candidate_origin - origin))
            angle = math.degrees(math.acos(float(np.clip(np.dot(candidate_direction, direction), -1.0, 1.0))))
            if baseline < 0.05 and angle < 5.0:
                return False
        return True

    @staticmethod
    def _snapshot_surfel(surfel: _MutableSurfel) -> SurfelSnapshot:
        direction = surfel.view_direction_sum.astype(np.float64, copy=True)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction /= norm
        return SurfelSnapshot(
            key=surfel.key,
            centroid=surfel.centroid.copy(),
            covariance_accumulator=surfel.covariance_accumulator.copy(),
            observation_count=surfel.observation_count,
            independent_frame_support=len(surfel.witnesses),
            witnesses=tuple(surfel.witnesses),
            mean_view_direction=direction.astype(np.float32),
            latest_sensor_origin=surfel.latest_sensor_origin.copy(),
            first_timestamp_ns=surfel.first_timestamp_ns,
            last_timestamp_ns=surfel.last_timestamp_ns,
            normal=surfel.normal.copy(),
            curvature=float(surfel.curvature),
            normal_confidence=float(surfel.normal_confidence),
            support_state=surfel.support_state,
        )

    def surfel(self, key: tuple[int, int, int]) -> SurfelSnapshot:
        with self._lock:
            return self._snapshot_surfel(self._surfels[tuple(key)])

    def surfels(self) -> tuple[SurfelSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshot_surfel(self._surfels[key]) for key in sorted(self._surfels))

    def dirty_tiles(self) -> tuple[TileId, ...]:
        with self._lock:
            return tuple(sorted(self._dirty_tiles))

    def owned_surfels(self, tile: TileId) -> tuple[SurfelSnapshot, ...]:
        with self._lock:
            return tuple(
                self._snapshot_surfel(self._surfels[key])
                for key in sorted(self._keys_by_tile.get(tile, ()))
            )

    def surfels_for_tile(self, tile: TileId, *, halo_m: float = 0.05) -> tuple[SurfelSnapshot, ...]:
        return self.surfels_for_tiles({tile}, halo_m=halo_m)

    def surfels_for_tiles(
        self,
        tiles: set[TileId],
        *,
        halo_m: float = 0.05,
    ) -> tuple[SurfelSnapshot, ...]:
        if not math.isfinite(halo_m) or halo_m < 0:
            raise ValueError("halo_m must be finite and non-negative")
        if not tiles:
            return ()
        with self._lock:
            keys = self._keys_for_tiles_unlocked(tiles, halo_m=halo_m)
            return tuple(self._snapshot_surfel(self._surfels[key]) for key in keys)

    def _keys_for_tiles_unlocked(
        self,
        tiles: set[TileId],
        *,
        halo_m: float,
    ) -> tuple[tuple[int, int, int], ...]:
        bounds = tuple(
            (
                np.asarray(tile.as_tuple(), dtype=np.float64) * self.tile_m - halo_m,
                np.asarray(tile.as_tuple(), dtype=np.float64) * self.tile_m + self.tile_m + halo_m,
            )
            for tile in sorted(tiles)
        )
        radius = int(math.ceil(halo_m / self.tile_m)) + 1
        owner_tiles = {
            TileId(tile.x + dx, tile.y + dy, tile.z + dz)
            for tile in tiles
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        }
        keys = sorted({
            key
            for owner in owner_tiles
            for key in self._keys_by_tile.get(owner, ())
        })
        return tuple(
            key
            for key in keys
            if any(
                np.all(self._surfels[key].centroid >= lower)
                and np.all(self._surfels[key].centroid < upper)
                for lower, upper in bounds
            )
        )

    def meshing_surfels_for_tiles(
        self,
        tiles: set[TileId],
        *,
        halo_m: float = 0.05,
    ) -> tuple[MeshingSurfel, ...]:
        if not math.isfinite(halo_m) or halo_m < 0:
            raise ValueError("halo_m must be finite and non-negative")
        if not tiles:
            return ()
        with self._lock:
            frozen = []
            for key in self._keys_for_tiles_unlocked(tiles, halo_m=halo_m):
                surfel = self._surfels[key]
                direction = surfel.view_direction_sum.astype(np.float64, copy=True)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    direction /= norm
                frozen.append(
                    MeshingSurfel(
                        key=surfel.key,
                        centroid=tuple(float(value) for value in surfel.centroid),
                        observation_count=surfel.observation_count,
                        independent_frame_support=len(surfel.witnesses),
                        mean_view_direction=tuple(float(value) for value in direction),
                    )
                )
            return tuple(frozen)

    def rank_tiles_by_support(self, tiles: set[TileId]) -> tuple[TileId, ...]:
        with self._lock:
            counts = {
                tile: sum(
                    self._surfels[key].support_state == SupportState.SURFACE_SUPPORTED
                    for key in self._keys_by_tile.get(tile, ())
                )
                for tile in tiles
            }
        return tuple(sorted(tiles, key=lambda tile: (-counts[tile], tile.as_tuple())))

    def supported_tiles(self, tiles: set[TileId]) -> frozenset[TileId]:
        with self._lock:
            return frozenset(
                tile
                for tile in tiles
                if any(
                    self._surfels[key].support_state == SupportState.SURFACE_SUPPORTED
                    for key in self._keys_by_tile.get(tile, ())
                )
            )

    def tile_revision_seed(self, tile: TileId) -> str:
        digest = hashlib.sha256()
        digest.update(self.source_config_sha256.encode("ascii"))
        digest.update(json.dumps(tile.as_tuple(), separators=(",", ":")).encode("ascii"))
        for surfel in self.owned_surfels(tile):
            digest.update(np.asarray(surfel.key, dtype="<i4").tobytes())
            digest.update(np.asarray(surfel.centroid, dtype="<f8").tobytes())
            digest.update(np.asarray(surfel.covariance_accumulator, dtype="<f8").tobytes())
            digest.update(int(surfel.observation_count).to_bytes(4, "little"))
        return digest.hexdigest()

    def snapshot(self) -> dict:
        with self._lock:
            supported = sum(surfel.support_state == SupportState.SURFACE_SUPPORTED for surfel in self._surfels.values())
            return {
                "map_id": self.map_id,
                "source_config_sha256": self.source_config_sha256,
                "sensor_frame_transform_id": self.sensor_frame_transform_id,
                "calibration_id": self.calibration_id,
                "cell_m": self.cell_m,
                "tile_m": self.tile_m,
                "surfel_count": len(self._surfels),
                "supported_surfel_count": supported,
                "unresolved_surfel_count": len(self._surfels) - supported,
                "dirty_tile_count": len(self._dirty_tiles),
                "last_integrated_sequence": None if self._last_integrated_identity is None else self._last_integrated_identity.sequence,
            }

    @staticmethod
    def _witness_payload(witness: ObservationWitness) -> dict:
        identity = witness.frame_identity
        return {
            "frame_identity": {
                "building_id": identity.building_id,
                "session_id": identity.session_id,
                "map_id": identity.map_id,
                "producer_instance_id": identity.producer_instance_id,
                "sequence": identity.sequence,
            },
            "sensor_origin": list(witness.sensor_origin),
            "view_direction": list(witness.view_direction),
        }

    @staticmethod
    def _witness_from_payload(payload: dict) -> ObservationWitness:
        identity = payload["frame_identity"]
        return ObservationWitness(
            FrameIdentity(
                str(identity["building_id"]),
                str(identity["session_id"]),
                str(identity["map_id"]),
                str(identity["producer_instance_id"]),
                int(identity["sequence"]),
            ),
            tuple(float(value) for value in payload["sensor_origin"]),
            tuple(float(value) for value in payload["view_direction"]),
        )

    def checkpoint(
        self,
        root: str | Path,
        *,
        building_id: Optional[str] = None,
        session_id: Optional[str] = None,
        floor_id: Optional[str] = None,
        submap_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
    ) -> SurfaceCheckpoint:
        if self._last_integrated_identity is None:
            raise ValueError("cannot checkpoint an empty live surface")
        provenance = {
            "building_id": building_id,
            "session_id": session_id,
            "floor_id": floor_id,
            "submap_id": submap_id,
            "calibration_id": calibration_id,
        }
        if any(value is not None for value in provenance.values()):
            if not all(isinstance(value, str) and value for value in provenance.values()):
                raise ValueError("surface checkpoint provenance must be complete")
            if building_id != self._last_integrated_identity.building_id:
                raise ValueError("surface checkpoint building provenance mismatch")
            if session_id != self._last_integrated_identity.session_id:
                raise ValueError("surface checkpoint session provenance mismatch")
            if calibration_id != self.calibration_id:
                raise ValueError("surface checkpoint calibration provenance mismatch")
        root = Path(root)
        generations = root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        temporary = generations / f".partial-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            with self._lock:
                surfels = [self._surfels[key] for key in sorted(self._surfels)]
                last_integrated_identity = self._last_integrated_identity
                journal = list(self._journal)
                pose_revision_id = self._pose_revision_id
                pose_revision_frame_count = self._pose_revision_frame_count
                dirty_tiles = tuple(sorted(self._dirty_tiles))
            keys = np.asarray([surfel.key for surfel in surfels], dtype="<i4").reshape(-1, 3)
            centroids = np.asarray([surfel.centroid for surfel in surfels], dtype="<f8").reshape(-1, 3)
            covariance = np.asarray(
                [
                    (
                        surfel.covariance_accumulator[0, 0],
                        surfel.covariance_accumulator[0, 1],
                        surfel.covariance_accumulator[0, 2],
                        surfel.covariance_accumulator[1, 1],
                        surfel.covariance_accumulator[1, 2],
                        surfel.covariance_accumulator[2, 2],
                    )
                    for surfel in surfels
                ],
                dtype="<f8",
            ).reshape(-1, 6)
            witness_streams: list[tuple[str, str, str, str]] = []
            witness_stream_lookup: dict[tuple[str, str, str, str], int] = {}
            witness_offsets = np.zeros(len(surfels) + 1, dtype="<u4")
            flat_witnesses: list[ObservationWitness] = []
            for index, surfel in enumerate(surfels):
                flat_witnesses.extend(surfel.witnesses)
                witness_offsets[index + 1] = len(flat_witnesses)
            witness_stream_indices = np.empty(len(flat_witnesses), dtype="<u4")
            witness_sequences = np.empty(len(flat_witnesses), dtype="<i8")
            witness_sensor_origins = np.empty((len(flat_witnesses), 3), dtype="<f8")
            witness_view_directions = np.empty((len(flat_witnesses), 3), dtype="<f8")
            for index, witness in enumerate(flat_witnesses):
                stream = witness.frame_identity.stream_key().as_tuple()
                stream_index = witness_stream_lookup.get(stream)
                if stream_index is None:
                    stream_index = len(witness_streams)
                    witness_stream_lookup[stream] = stream_index
                    witness_streams.append(stream)
                witness_stream_indices[index] = stream_index
                witness_sequences[index] = witness.frame_identity.sequence
                witness_sensor_origins[index] = witness.sensor_origin
                witness_view_directions[index] = witness.view_direction
            arrays_path = temporary / "surfels.npz"
            np.savez_compressed(
                arrays_path,
                keys=keys,
                centroids=centroids,
                covariance=covariance,
                observation_counts=np.asarray([surfel.observation_count for surfel in surfels], dtype="<u4"),
                view_direction_sums=np.asarray(
                    [surfel.view_direction_sum for surfel in surfels], dtype="<f8"
                ).reshape(-1, 3),
                sensor_origins=np.asarray([surfel.latest_sensor_origin for surfel in surfels], dtype="<f8").reshape(-1, 3),
                first_timestamps=np.asarray([surfel.first_timestamp_ns for surfel in surfels], dtype="<i8"),
                last_timestamps=np.asarray([surfel.last_timestamp_ns for surfel in surfels], dtype="<i8"),
                normals=np.asarray([surfel.normal for surfel in surfels], dtype="<f4").reshape(-1, 3),
                curvatures=np.asarray([surfel.curvature for surfel in surfels], dtype="<f4"),
                normal_confidences=np.asarray([surfel.normal_confidence for surfel in surfels], dtype="<f4"),
                support_states=np.asarray([int(surfel.support_state) for surfel in surfels], dtype="u1"),
                witness_offsets=witness_offsets,
                witness_stream_indices=witness_stream_indices,
                witness_sequences=witness_sequences,
                witness_sensor_origins=witness_sensor_origins,
                witness_view_directions=witness_view_directions,
            )
            with arrays_path.open("r+b") as source:
                os.fsync(source.fileno())
            manifest_core = {
                "schema_version": 4,
                "map_id": self.map_id,
                "source_config_sha256": self.source_config_sha256,
                "sensor_frame_transform_id": self.sensor_frame_transform_id,
                "calibration_id": self.calibration_id,
                "cell_m": self.cell_m,
                "tile_m": self.tile_m,
                "surfel_count": len(surfels),
                "arrays_sha256": _sha256_file(arrays_path),
                "witness_streams": [list(stream) for stream in witness_streams],
                "last_integrated_identity": self._identity_payload_from_identity(last_integrated_identity),
                "integrated_journal": journal,
                "pose_revision_id": pose_revision_id,
                "pose_revision_frame_count": pose_revision_frame_count,
                "dirty_tiles": [list(tile.as_tuple()) for tile in dirty_tiles],
            }
            if all(value is not None for value in provenance.values()):
                manifest_core.update(provenance)
            generation_id = hashlib.sha256(
                json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            manifest = {**manifest_core, "generation_id": generation_id}
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as output:
                json.dump(manifest, output, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(temporary)
            generation_path = generations / generation_id
            if generation_path.exists():
                try:
                    self._validate_generation_directory(generation_path, generation_id)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("existing checkpoint generation is corrupt") from exc
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, generation_path)
                _fsync_directory(generations)
            _atomic_text(root / "CURRENT", generation_id + "\n")
            return SurfaceCheckpoint(
                generation_id,
                self._pose_revision_id,
                generation_path,
                self._last_integrated_identity,
                self.sensor_frame_transform_id,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _identity_payload_from_identity(identity: FrameIdentity) -> dict:
        return {
            "building_id": identity.building_id,
            "session_id": identity.session_id,
            "map_id": identity.map_id,
            "producer_instance_id": identity.producer_instance_id,
            "sequence": identity.sequence,
        }

    @classmethod
    def _validate_generation_directory(cls, generation: Path, generation_id: str) -> dict:
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("generation_id") != generation_id:
            raise ValueError("surface checkpoint generation identity mismatch")
        manifest_core = {key: value for key, value in manifest.items() if key != "generation_id"}
        expected_generation = hashlib.sha256(
            json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        if expected_generation != generation_id:
            raise ValueError("surface checkpoint manifest hash mismatch")
        schema_version = manifest.get("schema_version")
        if schema_version not in {1, 2, 3, 4}:
            raise ValueError("surface checkpoint schema is unsupported")
        try:
            normalize_sensor_frame_transform_id(manifest.get("sensor_frame_transform_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("surface checkpoint sensor frame transform is invalid") from exc
        if schema_version == 2:
            journal = manifest.get("integrated_journal")
            if (
                not isinstance(journal, list)
                or manifest.get("pose_revision_id") != cls._legacy_pose_revision(journal)
            ):
                raise ValueError("surface checkpoint pose revision mismatch")
        if schema_version in {3, 4}:
            journal = manifest.get("integrated_journal")
            revision = manifest.get("pose_revision_id")
            frame_count = manifest.get("pose_revision_frame_count")
            if (
                not isinstance(journal, list)
                or not isinstance(revision, str)
                or len(revision) != 64
                or any(character not in "0123456789abcdef" for character in revision)
                or isinstance(frame_count, bool)
                or not isinstance(frame_count, int)
                or frame_count < len(journal)
            ):
                raise ValueError("surface checkpoint pose revision chain is invalid")
        arrays_path = generation / "surfels.npz"
        if _sha256_file(arrays_path) != manifest.get("arrays_sha256"):
            raise ValueError("surface checkpoint array hash mismatch")
        if schema_version == 4:
            streams = manifest.get("witness_streams")
            if not isinstance(streams, list) or any(
                not isinstance(stream, list)
                or len(stream) != 4
                or any(not isinstance(value, str) or not value for value in stream)
                for stream in streams
            ):
                raise ValueError("surface checkpoint witness streams are invalid")
        else:
            witnesses_path = generation / "witnesses.json"
            if _sha256_file(witnesses_path) != manifest.get("witnesses_sha256"):
                raise ValueError("surface checkpoint witness hash mismatch")
        return manifest

    @classmethod
    def load_checkpoint(
        cls,
        root: str | Path,
        *,
        expected_map_id: str,
        expected_source_config_sha256: str,
        expected_generation_id: Optional[str] = None,
        expected_building_id: Optional[str] = None,
        expected_session_id: Optional[str] = None,
        expected_floor_id: Optional[str] = None,
        expected_submap_id: Optional[str] = None,
        expected_calibration_id: Optional[str] = None,
        expected_sensor_frame_transform_id: Optional[str] = None,
    ) -> "LiveSurfaceMap":
        root = Path(root)
        generation_id = (
            (root / "CURRENT").read_text(encoding="ascii").strip()
            if expected_generation_id is None
            else str(expected_generation_id)
        )
        if len(generation_id) != 64 or any(character not in "0123456789abcdef" for character in generation_id):
            raise ValueError("surface checkpoint pointer is invalid")
        generation = root / "generations" / generation_id
        manifest = cls._validate_generation_directory(generation, generation_id)
        if manifest.get("map_id") != expected_map_id:
            raise ValueError("surface checkpoint map does not match")
        if manifest.get("source_config_sha256") != expected_source_config_sha256:
            raise ValueError("surface checkpoint configuration does not match")
        checkpoint_transform_id = normalize_sensor_frame_transform_id(manifest.get("sensor_frame_transform_id"))
        if expected_sensor_frame_transform_id is not None:
            if normalize_sensor_frame_transform_id(expected_sensor_frame_transform_id) != checkpoint_transform_id:
                raise ValueError("surface checkpoint sensor frame transform does not match")
        provenance = {
            "building_id": expected_building_id,
            "session_id": expected_session_id,
            "floor_id": expected_floor_id,
            "submap_id": expected_submap_id,
            "calibration_id": expected_calibration_id,
        }
        if any(value is not None for value in provenance.values()):
            if not all(value is not None for value in provenance.values()):
                raise ValueError("surface checkpoint provenance expectation is incomplete")
            for name, expected in provenance.items():
                if manifest.get(name) != expected:
                    label = name[:-3] if name.endswith("_id") else name
                    raise ValueError(f"surface checkpoint {label} provenance does not match")
        arrays_path = generation / "surfels.npz"
        if _sha256_file(arrays_path) != manifest.get("arrays_sha256"):
            raise ValueError("surface checkpoint array hash mismatch")
        schema_version = int(manifest["schema_version"])
        witnesses_path = generation / "witnesses.json"
        if schema_version != 4 and _sha256_file(witnesses_path) != manifest.get("witnesses_sha256"):
            raise ValueError("surface checkpoint witness hash mismatch")
        surface = cls(
            map_id=expected_map_id,
            source_config_sha256=expected_source_config_sha256,
            sensor_frame_transform_id=checkpoint_transform_id,
            calibration_id=manifest.get("calibration_id"),
            cell_m=float(manifest["cell_m"]),
            tile_m=float(manifest["tile_m"]),
        )
        with np.load(arrays_path, allow_pickle=False) as arrays:
            keys = np.asarray(arrays["keys"], dtype="<i4")
            centroids = np.asarray(arrays["centroids"], dtype="<f8")
            covariance = np.asarray(arrays["covariance"], dtype="<f8")
            observation_counts = np.asarray(arrays["observation_counts"], dtype="<u4")
            view_direction_sums = np.asarray(arrays["view_direction_sums"], dtype="<f8")
            sensor_origins = np.asarray(arrays["sensor_origins"], dtype="<f8")
            first_timestamps = np.asarray(arrays["first_timestamps"], dtype="<i8")
            last_timestamps = np.asarray(arrays["last_timestamps"], dtype="<i8")
            normals = np.asarray(arrays["normals"], dtype="<f4")
            curvatures = np.asarray(arrays["curvatures"], dtype="<f4")
            normal_confidences = np.asarray(arrays["normal_confidences"], dtype="<f4")
            support_states = np.asarray(arrays["support_states"], dtype="u1")
            if schema_version == 4:
                witness_offsets = np.asarray(arrays["witness_offsets"], dtype="<u4")
                witness_stream_indices = np.asarray(arrays["witness_stream_indices"], dtype="<u4")
                witness_sequences = np.asarray(arrays["witness_sequences"], dtype="<i8")
                witness_sensor_origins = np.asarray(arrays["witness_sensor_origins"], dtype="<f8")
                witness_view_directions = np.asarray(arrays["witness_view_directions"], dtype="<f8")
                witness_streams = tuple(tuple(str(value) for value in stream) for stream in manifest["witness_streams"])
                witnesses = None
            else:
                witnesses = json.loads(witnesses_path.read_text(encoding="utf-8"))
            count = int(manifest["surfel_count"])
            expected_shapes = {
                "keys": (keys.shape, (count, 3)),
                "centroids": (centroids.shape, (count, 3)),
                "covariance": (covariance.shape, (count, 6)),
                "observation_counts": (observation_counts.shape, (count,)),
                "view_direction_sums": (view_direction_sums.shape, (count, 3)),
                "sensor_origins": (sensor_origins.shape, (count, 3)),
                "first_timestamps": (first_timestamps.shape, (count,)),
                "last_timestamps": (last_timestamps.shape, (count,)),
                "normals": (normals.shape, (count, 3)),
                "curvatures": (curvatures.shape, (count,)),
                "normal_confidences": (normal_confidences.shape, (count,)),
                "support_states": (support_states.shape, (count,)),
            }
            if schema_version == 4:
                witness_count = len(witness_stream_indices)
                expected_shapes.update({
                    "witness_offsets": (witness_offsets.shape, (count + 1,)),
                    "witness_sequences": (witness_sequences.shape, (witness_count,)),
                    "witness_sensor_origins": (witness_sensor_origins.shape, (witness_count, 3)),
                    "witness_view_directions": (witness_view_directions.shape, (witness_count, 3)),
                })
                invalid_witnesses = (
                    witness_offsets.shape != (count + 1,)
                    or witness_stream_indices.shape != (witness_count,)
                    or not len(witness_offsets)
                    or int(witness_offsets[0]) != 0
                    or int(witness_offsets[-1]) != witness_count
                    or np.any(witness_offsets[1:] < witness_offsets[:-1])
                    or np.any(witness_stream_indices >= len(witness_streams))
                    or np.any(witness_sequences < 0)
                )
            else:
                invalid_witnesses = len(witnesses) != count
            if any(actual != expected for actual, expected in expected_shapes.values()) or invalid_witnesses:
                raise ValueError("surface checkpoint arrays have invalid shapes")
            witness_by_key = None if schema_version == 4 else {
                tuple(item["key"]): item["witnesses"] for item in witnesses
            }
            for index in range(count):
                key = tuple(int(value) for value in keys[index])
                values = covariance[index]
                m2 = np.asarray(
                    ((values[0], values[1], values[2]), (values[1], values[3], values[4]), (values[2], values[4], values[5])),
                    dtype=np.float64,
                )
                direction_sum = np.asarray(view_direction_sums[index], dtype=np.float64)
                observation_count = int(observation_counts[index])
                if schema_version == 4:
                    start = int(witness_offsets[index])
                    end = int(witness_offsets[index + 1])
                    surfel_witnesses = []
                    for witness_index in range(start, end):
                        stream = witness_streams[int(witness_stream_indices[witness_index])]
                        surfel_witnesses.append(ObservationWitness(
                            FrameIdentity(*stream, int(witness_sequences[witness_index])),
                            tuple(float(value) for value in witness_sensor_origins[witness_index]),
                            tuple(float(value) for value in witness_view_directions[witness_index]),
                        ))
                else:
                    surfel_witnesses = [
                        cls._witness_from_payload(item) for item in witness_by_key[key]
                    ]
                surface._surfels[key] = _MutableSurfel(
                    key=key,
                    centroid=centroids[index].copy(),
                    covariance_accumulator=m2,
                    observation_count=observation_count,
                    witnesses=surfel_witnesses,
                    view_direction_sum=direction_sum,
                    latest_sensor_origin=np.asarray(sensor_origins[index], dtype=np.float64),
                    first_timestamp_ns=int(first_timestamps[index]),
                    last_timestamp_ns=int(last_timestamps[index]),
                    normal=np.asarray(normals[index], dtype=np.float32),
                    curvature=float(curvatures[index]),
                    normal_confidence=float(normal_confidences[index]),
                    support_state=SupportState(int(support_states[index])),
                )
                surface._keys_by_tile.setdefault(surface._tile_for_key(key), set()).add(key)
        surface._journal = list(manifest.get("integrated_journal", []))[-surface.JOURNAL_LIMIT:]
        surface._pose_revision_id = manifest.get("pose_revision_id")
        surface._pose_revision_frame_count = int(
            manifest.get("pose_revision_frame_count", len(surface._journal))
        )
        surface._journal_lookup = {
            surface._identity_from_payload(item).as_tuple(): (
                str(item["point_bytes_sha256"]),
                normalize_sensor_frame_transform_id(item.get("sensor_frame_transform_id")),
            )
            for item in surface._journal
        }
        surface._dirty_tiles = {TileId(*(int(value) for value in item)) for item in manifest.get("dirty_tiles", [])}
        last_identity = manifest["last_integrated_identity"]
        surface._last_integrated_identity = FrameIdentity(
            str(last_identity["building_id"]),
            str(last_identity["session_id"]),
            str(last_identity["map_id"]),
            str(last_identity["producer_instance_id"]),
            int(last_identity["sequence"]),
        )
        return surface
