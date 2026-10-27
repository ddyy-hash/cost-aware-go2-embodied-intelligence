"""Session-scoped, verified LiDAR surface reconstruction worker.

The worker deliberately owns one capture session.  It can publish a stable
observed-only surface while parts arrive, but it never advances the artifact
lifecycle to ``PC_RECONSTRUCTION_INGESTED``.  The immutable raw parts remain
available until the complete capture manifest is verified and the existing
offline replay has completed.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import zstandard

from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.high_fidelity_mapper import HighFidelityMapper
from mapping.pc.go2_mapping_live.sensor_frame import (
    get_sensor_frame_transform,
    normalize_sensor_frame_transform_id,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import (
    canonical_sha256,
    is_sha256,
    load_reconstruction_config,
)

from .artifacts import LayerArtifactStore
from .offline import (
    OFFLINE_DERIVATION_ALGORITHM,
    PREVIEW_UNCALIBRATED,
    RAW_LIDAR_WITH_RECORDED_LIO,
    RECORDED_LIO_REGISTERED_CLOUD,
    RECORDED_LIO_SESSION_ODOM_CLOUD,
    SourceArtifact,
    derive_capture,
)


_SESSION_RE = re.compile(r"^segment-[0-9]{8}T[0-9]{6}Z$", re.ASCII)
_PART_RE = re.compile(r"_(\d+)\.db3\.zstd$", re.ASCII)
_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})\s+\*?(.+)$", re.ASCII)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _part_sort_key(path: Path) -> tuple[int, str]:
    match = _PART_RE.search(path.name)
    return (int(match.group(1)), path.name) if match else (2**63 - 1, path.name)


@contextmanager
def _decompressed_part(source: Path, *, work_root: Path | None = None) -> Iterator[Path]:
    """Expose exactly one immutable compressed part as a temporary SQLite file."""

    root = None if work_root is None else Path(work_root)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="go2-session-part-",
        suffix=".db3",
        dir=None if root is None else str(root),
    )
    sqlite_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file, source.open("rb") as input_file:
            zstandard.ZstdDecompressor().copy_stream(input_file, output_file)
        yield sqlite_path
    except (OSError, zstandard.ZstdError) as exc:
        raise ValueError(f"cannot decompress verified capture part: {source}") from exc
    finally:
        try:
            sqlite_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_staging_directory(parent: Path, cache_key: str) -> Path:
    """Create an exclusive generation staging directory without Windows 0700 ACLs.

    Python's :class:`TemporaryDirectory` maps mode ``0700`` to a Windows ACL
    that the restricted desktop token cannot reopen.  The production WSL path
    retains a private mode; local Windows preview reconstruction instead needs
    an ordinary writable directory below the already-private workspace.
    """

    parent.mkdir(parents=True, exist_ok=True)
    mode = 0o755 if os.name == "nt" else 0o700
    for _ in range(32):
        candidate = parent / (
            f".{cache_key[:16]}-{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}.partial"
        )
        try:
            candidate.mkdir(mode=mode)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate an automatic reconstruction staging directory")


def _surface_from_snapshot(snapshot: Mapping[str, Any]) -> MeasuredSurface:
    source_layer = snapshot.get("source_layer")
    point_authority = snapshot.get("point_authority")
    if (source_layer, point_authority) not in {
        ("LIDAR_MEASURED", "LIDAR_MEASURED"),
        ("LIO_REGISTERED_LIDAR_OBSERVED", "LIO_REGISTERED_LIDAR_OBSERVED"),
    }:
        raise ValueError("incremental worker requires an allowed LiDAR observation surface")
    if snapshot.get("geometry_authority") is not False:
        raise ValueError("incremental worker cannot promote geometry authority")
    try:
        positions = np.asarray(snapshot.get("positions"), dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(snapshot.get("normals"), dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(snapshot.get("triangles"), dtype=np.int32).reshape(-1, 3)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("incremental measured surface arrays are invalid") from exc
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError("incremental measured surface vectors are non-finite")
    return MeasuredSurface(
        positions=positions,
        normals=normals,
        triangles=triangles,
        source_vertex_ids=tuple(snapshot.get("source_vertex_ids", ())),
        source_layer=str(source_layer),
        geometry_authority=False,
        point_authority=str(point_authority),
    )


class SessionReconstructionWorker:
    """Incrementally reconstruct one verified capture session.

    A worker has no discovery scope beyond ``capture_label/session_id``.  This
    prevents an old inbox artifact from silently becoming part of the active
    room or building reconstruction.
    """

    def __init__(
        self,
        *,
        session_id: str,
        inbox_root: str | Path,
        output_root: str | Path,
        status_path: str | Path,
        sensor_frame_transform_id: str,
        capture_label: str = "lidar-imu-rgbd",
        work_root: str | Path | None = None,
        voxel_m: float = 0.02,
        display_point_budget: int = 100_000,
        max_artifacts_per_pass: int = 1,
        mapper_factory: Callable[..., Any] = HighFidelityMapper,
        final_replayer: Callable[..., Any] = derive_capture,
        final_global_optimization: bool = True,
        source_mode: str = RAW_LIDAR_WITH_RECORDED_LIO,
        incremental_publish_interval_artifacts: int = 1,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(session_id, str) or _SESSION_RE.fullmatch(session_id) is None:
            raise ValueError("session_id must be a canonical segment timestamp")
        if not isinstance(capture_label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", capture_label):
            raise ValueError("capture_label is unsafe")
        if not isinstance(voxel_m, (float, int)) or isinstance(voxel_m, bool) or float(voxel_m) <= 0:
            raise ValueError("voxel_m must be positive")
        if not isinstance(display_point_budget, int) or isinstance(display_point_budget, bool) or display_point_budget < 100:
            raise ValueError("display_point_budget must be at least 100")
        if (
            not isinstance(max_artifacts_per_pass, int)
            or isinstance(max_artifacts_per_pass, bool)
            or max_artifacts_per_pass < 1
        ):
            raise ValueError("max_artifacts_per_pass must be at least 1")
        if not isinstance(final_global_optimization, bool):
            raise ValueError("final_global_optimization must be boolean")
        if source_mode not in {RAW_LIDAR_WITH_RECORDED_LIO, RECORDED_LIO_REGISTERED_CLOUD, RECORDED_LIO_SESSION_ODOM_CLOUD}:
            raise ValueError(f"unknown mapping source mode: {source_mode}")
        if incremental_publish_interval_artifacts < 1:
            raise ValueError("incremental_publish_interval_artifacts must be positive")

        self.session_id = session_id
        self.capture_label = capture_label
        self.inbox_root = Path(inbox_root).resolve(strict=False)
        self.output_root = Path(output_root).resolve(strict=False)
        self.status_path = Path(status_path).resolve(strict=False)
        self.source_root = (self.inbox_root / self.capture_label / self.session_id).resolve(strict=False)
        self.manifest_path = (self.inbox_root / self.capture_label / f"{self.session_id}.sha256").resolve(strict=False)
        self.work_root = (
            Path(work_root).resolve(strict=False)
            if work_root is not None
            else (
                Path(os.environ.get("GO2_MAPPING_WORK_ROOT", "/tmp/go2-mapping-reconstruction"))
                / self.session_id
                if os.name == "posix"
                else self.output_root / ".work" / self.session_id
            )
        )
        if self.output_root == self.inbox_root or self.inbox_root in self.output_root.parents:
            raise ValueError("automatic reconstruction output must be independent from the inbox")
        if self.source_root == self.output_root or self.source_root in self.output_root.parents:
            raise ValueError("automatic reconstruction output must be independent from the source session")

        self.sensor_frame_transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)
        self.voxel_m = float(voxel_m)
        self.display_point_budget = display_point_budget
        self.max_artifacts_per_pass = max_artifacts_per_pass
        self.final_global_optimization = final_global_optimization
        self.source_mode = source_mode
        self.incremental_publish_interval_artifacts = int(incremental_publish_interval_artifacts)
        self._lifecycle = ArtifactLifecycleStore(self.inbox_root)
        self._mapper = mapper_factory(
            voxel_m=self.voxel_m,
            display_point_budget=self.display_point_budget,
            source_mode=self.source_mode,
            sensor_frame_transform_id=self.sensor_frame_transform_id,
            reintegration_work_root=self.work_root,
            prewarm_surface_after_ingest=False,
        )
        self._final_replayer = final_replayer
        self._progress_callback = progress_callback
        self._ingested_digests: set[str] = set()
        self._finalized = False
        self._last_status: dict[str, Any] | None = None

    def process_once(self) -> dict[str, Any]:
        """Consume newly verified closed parts and optionally run the final replay."""

        existing_final = self._load_existing_final_status()
        if existing_final is not None:
            self._finalized = True
            self._last_status = existing_final
            return dict(existing_final)

        verified, invalid_artifacts = self._verified_parts()
        pending = [artifact for artifact in verified if artifact.sha256 not in self._ingested_digests]
        newly_ingested = 0
        ingest_results: list[dict[str, Any]] = []
        for artifact in pending[:self.max_artifacts_per_pass]:
            with _decompressed_part(artifact.path, work_root=self.work_root) as sqlite_path:
                result = dict(
                    self._mapper.ingest_sqlite_bag(
                        sqlite_path,
                        artifact_sha256=artifact.sha256,
                        source_artifact_path=artifact.path,
                    )
                )
            self._ingested_digests.add(artifact.sha256)
            newly_ingested += 1
            ingest_results.append(result)

        materialized = tuple(
            artifact for artifact in verified if artifact.sha256 in self._ingested_digests
        )
        remaining_verified_artifacts = len(verified) - len(materialized)
        if remaining_verified_artifacts:
            final_manifest_state = (
                "PRESENT_WAITING_FOR_INCREMENTAL_INGEST"
                if self.manifest_path.is_file()
                else "ABSENT"
            )
        else:
            final_manifest_state = self._validate_final_manifest(verified)
        surface_progress = {
            "final_manifest_state": final_manifest_state,
            "remaining_verified_artifacts": remaining_verified_artifacts,
        }
        publish_incremental = (
            final_manifest_state == "VERIFIED"
            or not newly_ingested
            or remaining_verified_artifacts == 0
            or len(materialized) % self.incremental_publish_interval_artifacts == 0
        )
        surface_status = (
            self._publish_incremental_surface(materialized, additional_status=surface_progress)
            if publish_incremental else None
        )
        if final_manifest_state == "VERIFIED":
            replay_kwargs: dict[str, Any] = {
                "sensor_frame_transform_id": self.sensor_frame_transform_id,
                "work_root": self.work_root,
            }
            if self.source_mode != RAW_LIDAR_WITH_RECORDED_LIO:
                replay_kwargs["source_mode"] = self.source_mode
            final = self._final_replayer(self.source_root, self.output_root, **replay_kwargs)
            surface_path = getattr(final, "surface_json_path", None)
            if surface_path is None:
                result = self._base_status(
                    state="FINAL_SURFACE_UNAVAILABLE",
                    mode="high_fidelity_batch_reconstruction",
                    finalized=True,
                    available=False,
                    blocker="LIDAR_MEASURED_SURFACE_UNAVAILABLE",
                )
                result.update({
                    "invalid_artifacts": invalid_artifacts,
                    "newly_ingested_artifacts": newly_ingested,
                    "ingest_results": ingest_results,
                    "final_manifest_state": final_manifest_state,
                    "remaining_verified_artifacts": remaining_verified_artifacts,
                })
                self._write_status(result)
                self._finalized = True
                return result
            result = self._publish_surface_pointer(
                Path(surface_path),
                mode="high_fidelity_batch_reconstruction",
                finalized=True,
                state="FINAL_SURFACE_AVAILABLE",
                additional_status={
                    "invalid_artifacts": invalid_artifacts,
                    "newly_ingested_artifacts": newly_ingested,
                    "ingest_results": ingest_results,
                    **surface_progress,
                },
            )
            self._finalized = True
            return result

        if surface_status is not None:
            result = dict(surface_status)
            result["state"] = "INCREMENTAL_SURFACE_AVAILABLE"
        elif not verified:
            result = self._base_status(
                state="WAITING_FOR_VERIFIED_ARTIFACTS",
                mode="high_fidelity_incremental_reconstruction",
                finalized=False,
                available=False,
                blocker="SHA256_VERIFIED_LIDAR_PART_UNAVAILABLE",
            )
        else:
            result = self._base_status(
                state="WAITING_FOR_MEASURED_SURFACE",
                mode="high_fidelity_incremental_reconstruction",
                finalized=False,
                available=False,
                blocker="LIDAR_MEASURED_SURFACE_UNAVAILABLE",
            )
        result.update({
            "invalid_artifacts": invalid_artifacts,
            "newly_ingested_artifacts": newly_ingested,
            "ingest_results": ingest_results,
            "final_manifest_state": final_manifest_state,
            "remaining_verified_artifacts": remaining_verified_artifacts,
        })
        if surface_status is None:
            self._write_status(result)
        else:
            # The pointer is already atomically published.  Do not write a
            # second status body after it, which could expose a half-updated
            # surface identity to the Dashboard.
            self._last_status = dict(result)
        return result

    def finalize_verified_session(self) -> dict[str, Any]:
        """Replay one closed verified session without waiting for incremental preview ingest.

        This is deliberately a separate operation from :meth:`process_once`.
        A live preview may be throttled or lag the capture, while a closed
        session can safely run the existing full read-only replay once its
        final manifest and every lifecycle-verified part agree.
        """

        existing_final = self._load_existing_final_status()
        if existing_final is not None:
            self._finalized = True
            self._last_status = existing_final
            return dict(existing_final)

        verified, invalid_artifacts = self._verified_parts()
        final_manifest_state = self._validate_final_manifest(verified)
        if final_manifest_state != "VERIFIED":
            result = self._base_status(
                state="FINAL_REPLAY_NOT_READY",
                mode="high_fidelity_batch_reconstruction",
                finalized=False,
                available=False,
                blocker="FINAL_MANIFEST_NOT_VERIFIED",
            )
            result.update({
                "invalid_artifacts": invalid_artifacts,
                "newly_ingested_artifacts": 0,
                "ingest_results": [],
                "final_manifest_state": final_manifest_state,
                "remaining_verified_artifacts": len(verified),
            })
            self._write_status(result)
            return result

        replay_kwargs: dict[str, Any] = {
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
            "work_root": self.work_root,
        }
        if self.source_mode != RAW_LIDAR_WITH_RECORDED_LIO:
            replay_kwargs["source_mode"] = self.source_mode
        if not self.final_global_optimization:
            replay_kwargs["global_optimization"] = False
        if self._progress_callback is not None:
            replay_kwargs["progress_callback"] = self._progress_callback
        final = self._final_replayer(self.source_root, self.output_root, **replay_kwargs)
        surface_path = getattr(final, "surface_json_path", None)
        if surface_path is None:
            result = self._base_status(
                state="FINAL_SURFACE_UNAVAILABLE",
                mode="high_fidelity_batch_reconstruction",
                finalized=True,
                available=False,
                blocker="LIDAR_MEASURED_SURFACE_UNAVAILABLE",
            )
            result.update({
                "invalid_artifacts": invalid_artifacts,
                "newly_ingested_artifacts": 0,
                "ingest_results": [],
                "final_manifest_state": final_manifest_state,
                "remaining_verified_artifacts": 0,
            })
            self._write_status(result)
            self._finalized = True
            return result
        result = self._publish_surface_pointer(
            Path(surface_path),
            mode="high_fidelity_batch_reconstruction",
            finalized=True,
            state="FINAL_SURFACE_AVAILABLE",
            additional_status={
                "invalid_artifacts": invalid_artifacts,
                "newly_ingested_artifacts": 0,
                "ingest_results": [],
                "final_manifest_state": final_manifest_state,
                "remaining_verified_artifacts": 0,
                "pose_graph_optimization": (
                    "GLOBAL_CONSTRAINT_OPTIMIZATION"
                    if self.final_global_optimization
                    else "RECORDED_LIO_POSES_ONLY"
                ),
            },
        )
        self._finalized = True
        return result

    def _verified_parts(self) -> tuple[tuple[SourceArtifact, ...], int]:
        registry = self._lifecycle.registry_root / self.session_id
        candidates: list[SourceArtifact] = []
        invalid = 0
        if not registry.is_dir():
            return (), 0
        for record_path in sorted(registry.glob("*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid += 1
                continue
            if not isinstance(record, dict):
                invalid += 1
                continue
            relative = record.get("relative_path")
            if not isinstance(relative, str) or not relative.endswith(".db3.zstd"):
                continue
            artifact = self._verified_part_from_record(record)
            if artifact is None:
                invalid += 1
                continue
            candidates.append(artifact)
        candidates.sort(key=lambda item: _part_sort_key(item.path))
        seen: set[str] = set()
        unique: list[SourceArtifact] = []
        for artifact in candidates:
            if artifact.sha256 in seen:
                invalid += 1
                continue
            seen.add(artifact.sha256)
            unique.append(artifact)
        return tuple(unique), invalid

    def _verified_part_from_record(self, record: Mapping[str, Any]) -> SourceArtifact | None:
        digest = record.get("artifact_sha256")
        artifact_id = record.get("artifact_id")
        relative = record.get("relative_path")
        size = record.get("size_bytes")
        if (
            record.get("schema_version") != 1
            or record.get("session_id") != self.session_id
            or record.get("state") != "SHA256_VERIFIED"
            or record.get("release_authorized") is not False
            or not isinstance(digest, str)
            or artifact_id != digest
            or not is_sha256(digest)
            or digest != digest.lower()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(relative, str)
        ):
            return None
        try:
            relative_path = PurePosixPath(relative)
        except TypeError:
            return None
        expected_relative = PurePosixPath(self.capture_label) / self.session_id
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parent != expected_relative
            or _PART_RE.search(relative_path.name) is None
        ):
            return None
        artifact_path = (self.inbox_root / Path(*relative_path.parts)).resolve(strict=False)
        try:
            artifact_path.relative_to(self.source_root)
        except ValueError:
            return None
        if not artifact_path.is_file() or artifact_path.stat().st_size != size:
            return None
        return SourceArtifact(
            relative_path=relative_path.name,
            path=artifact_path,
            sha256=digest,
            bytes=size,
        )

    def _publish_incremental_surface(
        self,
        artifacts: tuple[SourceArtifact, ...],
        *,
        additional_status: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not artifacts:
            return None
        snapshot = self._mapper.measured_surface_snapshot()
        if snapshot is None:
            return None
        snapshot = dict(snapshot)
        positions = snapshot.get("positions")
        triangles = snapshot.get("triangles")
        if (
            not isinstance(positions, list)
            or not isinstance(triangles, list)
            or len(positions) < 9
            or len(triangles) < 3
        ):
            return None
        source_hashes = [item.sha256 for item in artifacts]
        surface_hashes = snapshot.get("artifact_sha256")
        if not isinstance(surface_hashes, list) or sorted(surface_hashes) != sorted(source_hashes):
            raise ValueError("incremental surface source artifacts do not match verified session parts")
        if snapshot.get("sensor_frame_transform_id") != self.sensor_frame_transform_id:
            raise ValueError("incremental surface sensor transform does not match the worker")
        if snapshot.get("quality", PREVIEW_UNCALIBRATED) != PREVIEW_UNCALIBRATED:
            raise ValueError("incremental surface cannot claim calibrated geometry")

        mapper_state = dict(self._mapper.snapshot())
        pose_generation = str(
            snapshot.get("pose_generation")
            or mapper_state.get("optimized_pose_generation")
            or mapper_state.get("pose_generation")
            or ""
        )
        if not is_sha256(pose_generation) or pose_generation != pose_generation.lower():
            raise ValueError("incremental surface pose generation is not content addressed")
        config = load_reconstruction_config()
        identity_core = {
            "identity_schema_version": 1,
            "algorithm_version": OFFLINE_DERIVATION_ALGORITHM,
            "source_artifacts": [item.as_dict() for item in artifacts],
            "source_artifact_sha256": source_hashes,
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
            "mapping_source_mode": self.source_mode,
            "coordinate_scope": "SESSION_ODOM_LOCAL" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP",
            "coordinate_frame": "odom" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "map",
            "room_or_building": self.source_mode != RECORDED_LIO_SESSION_ODOM_CLOUD,
            "texture_authorized": False,
            "fusion_config_sha256": config.fusion_sha256,
            "pose_generation": pose_generation,
        }
        identity = {**identity_core, "cache_key": canonical_sha256(identity_core)}
        generation_root = (
            self.output_root
            / self.session_id
            / "incremental"
            / f"sensor-frame-{self.sensor_frame_transform_id}"
            / identity["cache_key"]
        )
        surface_path = generation_root / "measured-surface.json"
        if generation_root.exists():
            from mapping.pc.go2_mapping_dashboard.derived_surface import load_derived_surface_preview

            return self._publish_surface_pointer(
                load_derived_surface_preview(surface_path).surface_path,
                mode="high_fidelity_incremental_reconstruction",
                finalized=False,
                state="INCREMENTAL_SURFACE_AVAILABLE",
                additional_status=additional_status,
            )

        parent = generation_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging: Path | None = _create_staging_directory(parent, identity["cache_key"])
        try:
            surface = _surface_from_snapshot(snapshot)
            transform = get_sensor_frame_transform(self.sensor_frame_transform_id)
            sensor_frame_transform_applied = self.source_mode == RAW_LIDAR_WITH_RECORDED_LIO
            surface_payload = {
                **snapshot,
                "sensor_frame_transform_id": self.sensor_frame_transform_id,
                "geometry_authority": False,
                "source_layer": str(snapshot["source_layer"]),
                "point_authority": str(snapshot["point_authority"]),
                "mapping_source_mode": self.source_mode,
                "coordinate_scope": "SESSION_ODOM_LOCAL" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP",
                "coordinate_frame": "odom" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "map",
                "room_or_building": self.source_mode != RECORDED_LIO_SESSION_ODOM_CLOUD,
                "texture_authorized": False,
                "sensor_frame_transform_applied": sensor_frame_transform_applied,
                "quality": PREVIEW_UNCALIBRATED,
                "status": PREVIEW_UNCALIBRATED,
                "volume_filled": False,
                "watertight": False,
                "reconstruction_phase": "IDLE",
            }
            staged_surface_path = staging / "measured-surface.json"
            _write_json_atomic(staged_surface_path, surface_payload)
            published = LayerArtifactStore(staging / "artifacts").publish(
                surface,
                identity,
                annotations={
                    "quality": PREVIEW_UNCALIBRATED,
                    "status": PREVIEW_UNCALIBRATED,
                    "sensor_frame_transform_id": self.sensor_frame_transform_id,
                    "volume_filled": False,
                    "watertight": False,
                    "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
                    "mapping_source_mode": self.source_mode,
                    "coordinate_scope": "SESSION_ODOM_LOCAL" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP",
                    "coordinate_frame": "odom" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "map",
                    "room_or_building": self.source_mode != RECORDED_LIO_SESSION_ODOM_CLOUD,
                    "texture_authorized": False,
                    "sensor_frame_transform_applied": sensor_frame_transform_applied,
                },
            )
            manifest = {
                "schema_version": 1,
                "status": PREVIEW_UNCALIBRATED,
                "quality": PREVIEW_UNCALIBRATED,
                "geometry_authority": False,
                "source_layer": surface.source_layer,
                "point_authority": surface.point_authority,
                "volume_filled": False,
                "watertight": False,
                "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
                "sensor_frame_transform_id": self.sensor_frame_transform_id,
                "mapping_source_mode": self.source_mode,
                "coordinate_scope": "SESSION_ODOM_LOCAL" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP",
                "coordinate_frame": "odom" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "map",
                "room_or_building": self.source_mode != RECORDED_LIO_SESSION_ODOM_CLOUD,
                "texture_authorized": False,
                "sensor_frame_transform_applied": sensor_frame_transform_applied,
                "sensor_frame_transform": {
                    "id": transform.id,
                    "rotation": transform.rotation.tolist(),
                    "translation": transform.translation.tolist(),
                    "provisional": transform.provisional,
                    "description": transform.description,
                },
                "source_root_name": self.session_id,
                "source_artifacts": [item.as_dict() for item in artifacts],
                "derived_identity": identity,
                "mapper": {
                    key: mapper_state.get(key)
                    for key in (
                        "quality", "points", "artifacts", "frames", "source_points",
                        "skipped_pose_frames", "decode_errors", "reconstruction_phase",
                        "accepted_loop_closures", "raw_source_artifacts",
                    )
                },
                "ingest_results": [],
                "surface_available": True,
                "surface": {
                    "vertex_count": surface.vertex_count,
                    "triangle_count": surface.triangle_count,
                    "source_vertex_count": len(surface.source_vertex_ids),
                    "geometry_revision_id": snapshot.get("measured_surface_geometry_revision_id"),
                    "diagnostics": {
                        "hidden_surface_generated": False,
                        "volume_filled": False,
                        "watertight": False,
                    },
                },
                "artifact": {
                    "ply_name": published.ply_path.name,
                    "metadata_name": published.metadata_path.name,
                    "ply_sha256": published.ply_sha256,
                },
            }
            _write_json_atomic(staging / "manifest.json", manifest)
            try:
                os.replace(staging, generation_root)
            except FileExistsError:
                shutil.rmtree(staging, ignore_errors=True)
                staging = None
            else:
                staging = None
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        from mapping.pc.go2_mapping_dashboard.derived_surface import load_derived_surface_preview

        preview = load_derived_surface_preview(surface_path)
        return self._publish_surface_pointer(
            preview.surface_path,
            mode="high_fidelity_incremental_reconstruction",
            finalized=False,
            state="INCREMENTAL_SURFACE_AVAILABLE",
            additional_status=additional_status,
        )

    def _validate_final_manifest(self, verified_parts: tuple[SourceArtifact, ...]) -> str:
        if not self.manifest_path.is_file() or not self.source_root.is_dir():
            return "ABSENT"
        try:
            lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return "INVALID"
        expected: dict[str, str] = {}
        for line in lines:
            match = _MANIFEST_LINE_RE.fullmatch(line)
            if match is None:
                return "INVALID"
            digest, raw_path = match.groups()
            remote = PurePosixPath(raw_path)
            if (
                remote.is_absolute() is False
                or ".." in remote.parts
                or len(remote.parts) < 3
                or remote.parts[-3] != self.capture_label
                or remote.parts[-2] != self.session_id
                or not remote.name
            ):
                return "INVALID"
            if remote.name in expected:
                return "INVALID"
            expected[remote.name] = digest
        if not expected:
            return "INVALID"
        actual = {
            path.name: path
            for path in self.source_root.iterdir()
            if path.is_file() and not path.name.endswith(".partial")
        }
        if set(actual) != set(expected):
            return "INVALID"
        verified_by_name = {item.relative_path: item.sha256 for item in verified_parts}
        manifest_part_names = {name for name in expected if name.endswith(".db3.zstd")}
        if set(verified_by_name) != manifest_part_names:
            return "INVALID"
        for name, path in actual.items():
            if _sha256_file(path) != expected[name]:
                return "INVALID"
        return "VERIFIED"

    def _publish_surface_pointer(
        self,
        surface_path: Path,
        *,
        mode: str,
        finalized: bool,
        state: str,
        additional_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from mapping.pc.go2_mapping_dashboard.derived_surface import load_derived_surface_preview

        preview = load_derived_surface_preview(surface_path)
        result = self._base_status(
            state=state,
            mode=mode,
            finalized=finalized,
            available=True,
            blocker=None,
        )
        result.update({
            "surface_json_path": str(preview.surface_path),
            "surface_json_relative_path": preview.surface_path.relative_to(
                self.output_root
            ).as_posix(),
            "surface_sha256": preview.status["surface_sha256"],
            "derived_cache_key": preview.status["derived_cache_key"],
            "ply_sha256": preview.status["ply_sha256"],
            "metadata_sha256": preview.status["metadata_sha256"],
            "vertex_count": preview.status["vertex_count"],
            "triangle_count": preview.status["triangle_count"],
            "quality": PREVIEW_UNCALIBRATED,
            "geometry_authority": False,
            "source_layer": preview.status["source_layer"],
            "point_authority": preview.status["point_authority"],
            "mapping_source_mode": preview.status["mapping_source_mode"],
            "coordinate_scope": "SESSION_ODOM_LOCAL" if self.source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP",
            "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        })
        if additional_status:
            result.update(dict(additional_status))
        self._write_status(result)
        return result

    def _base_status(
        self,
        *,
        state: str,
        mode: str,
        finalized: bool,
        available: bool,
        blocker: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "state": state,
            "mode": mode,
            "finalized": finalized,
            "available": available,
            "blocker": blocker,
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
            "updated_at_ms": int(time.time() * 1000),
        }

    def _write_status(self, status: Mapping[str, Any]) -> None:
        _write_json_atomic(self.status_path, status)
        self._last_status = dict(status)

    def _load_existing_final_status(self) -> dict[str, Any] | None:
        if not self.status_path.is_file():
            return None
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(status, dict)
            or status.get("schema_version") != 1
            or status.get("session_id") != self.session_id
            or status.get("finalized") is not True
            or status.get("available") is not True
            or status.get("mode") != "high_fidelity_batch_reconstruction"
        ):
            return None
        path = status.get("surface_json_path")
        relative_path = status.get("surface_json_relative_path")
        migrated_legacy_pointer = False
        if relative_path is not None:
            if not isinstance(relative_path, str) or not relative_path:
                return None
            relative_surface_path = Path(relative_path)
            if relative_surface_path.is_absolute() or ".." in relative_surface_path.parts:
                return None
            surface_path = (self.output_root / relative_surface_path).resolve(strict=False)
        elif isinstance(path, str):
            surface_path = Path(path)
        else:
            return None
        try:
            from mapping.pc.go2_mapping_dashboard.derived_surface import load_derived_surface_preview

            preview = load_derived_surface_preview(surface_path)
        except (OSError, ValueError):
            return None
        if (
            status.get("surface_sha256") != preview.status["surface_sha256"]
            or status.get("derived_cache_key") != preview.status["derived_cache_key"]
        ):
            return None
        if relative_path is None:
            status = dict(status)
            status["surface_json_relative_path"] = preview.surface_path.relative_to(
                self.output_root
            ).as_posix()
            migrated_legacy_pointer = True
        if migrated_legacy_pointer:
            self._write_status(status)
        return status


__all__ = ["SessionReconstructionWorker"]
