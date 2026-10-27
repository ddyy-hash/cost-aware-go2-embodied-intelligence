"""Read-only RGB-D projection evidence for a verified derived LiDAR surface.

The final derived surface is immutable LiDAR geometry.  This module is a
separate inspection path: it replays the same raw LiDAR/LIO association used by
the final mapper, projects native LiDAR points using an operator-forced runtime
transform, and writes no geometry or runtime pointer.  It deliberately does
not use unverified RGB-D depth as a geometry or texture-delivery authority.
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import struct
import tempfile
from typing import Any, Iterator, Sequence

import numpy as np
import zstandard

from mapping.pc.go2_mapping_dashboard.derived_surface import load_automatic_reconstruction_preview
from mapping.pc.go2_mapping_live.high_fidelity_mapper import CdrDecodeError, _decode_pointcloud, _decode_pose
from mapping.pc.go2_mapping_live.sensor_frame import normalize_sensor_frame_transform_id
from mapping.pc.go2_mapping_quality.operator_override import load_operator_forced_calibration_override
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, is_sha256
from mapping.pc.go2_mapping_rgbd.decode import RgbdDecodeError, decode_camera_info, decode_image

from .derived_provisional_preview import (
    ProvisionalProjectionAccumulator,
    ProvisionalProjectionEvidence,
    _srgb_to_linear,
)


_SESSION_RE = __import__("re").compile(r"^segment-[0-9]{8}T[0-9]{6}Z$", __import__("re").ASCII)


@dataclass(frozen=True)
class ProvisionalSourceArtifact:
    relative_path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        value = self.relative_path
        try:
            path = PurePosixPath(value)
        except TypeError as exc:
            raise ValueError("source artifact relative path is invalid") from exc
        if (
            not isinstance(value, str)
            or not value
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 1
            or not value.endswith(".db3.zstd")
        ):
            raise ValueError("source artifact relative path is invalid")
        if not isinstance(self.sha256, str) or not is_sha256(self.sha256) or self.sha256 != self.sha256.lower():
            raise ValueError("source artifact SHA-256 is invalid")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes <= 0:
            raise ValueError("source artifact byte count is invalid")


@dataclass(frozen=True)
class DerivedSurfaceBinding:
    """The exact immutable mesh identity accepted by a preview replay."""

    session_id: str
    source_surface_sha256: str
    measured_surface_geometry_revision_id: str
    sensor_frame_transform_id: str
    voxel_size_m: float
    source_vertex_ids: tuple[str, ...]
    source_artifacts: tuple[ProvisionalSourceArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or _SESSION_RE.fullmatch(self.session_id) is None:
            raise ValueError("derived preview session ID is invalid")
        for field in ("source_surface_sha256", "measured_surface_geometry_revision_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
                raise ValueError(f"derived preview {field} is invalid")
        object.__setattr__(
            self,
            "sensor_frame_transform_id",
            normalize_sensor_frame_transform_id(self.sensor_frame_transform_id),
        )
        voxel_size = float(self.voxel_size_m)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("derived preview voxel size is invalid")
        object.__setattr__(self, "voxel_size_m", voxel_size)
        vertex_ids = tuple(self.source_vertex_ids)
        if not vertex_ids:
            raise ValueError("derived preview source vertex IDs are invalid")
        object.__setattr__(self, "source_vertex_ids", vertex_ids)
        artifacts = tuple(self.source_artifacts)
        if not artifacts or any(not isinstance(item, ProvisionalSourceArtifact) for item in artifacts):
            raise ValueError("derived preview source artifacts are invalid")
        if len({item.relative_path for item in artifacts}) != len(artifacts):
            raise ValueError("derived preview source artifact paths are duplicated")
        if len({item.sha256 for item in artifacts}) != len(artifacts):
            raise ValueError("derived preview source artifact hashes are duplicated")
        object.__setattr__(self, "source_artifacts", tuple(sorted(artifacts, key=lambda item: item.relative_path)))


@dataclass(frozen=True)
class RuntimeProjection:
    """The explicit operator-forced projection, distinct from measurement calibration."""

    calibration_id: str
    operator_override_sha256: str
    camera_from_native_lidar: np.ndarray
    runtime_projection_authorized: bool
    measurement_authority: bool

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("runtime projection calibration ID is required")
        if (
            not isinstance(self.operator_override_sha256, str)
            or not is_sha256(self.operator_override_sha256)
            or self.operator_override_sha256 != self.operator_override_sha256.lower()
        ):
            raise ValueError("runtime projection override SHA-256 is invalid")
        matrix = np.asarray(self.camera_from_native_lidar, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("runtime projection transform must be a finite 4 by 4 matrix")
        if self.runtime_projection_authorized is not True:
            raise ValueError("runtime projection is not authorized")
        if self.measurement_authority is not False:
            raise ValueError("runtime projection must not claim measurement authority")
        matrix = np.array(matrix, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "camera_from_native_lidar", matrix)


def load_active_derived_surface_binding(
    *,
    status_path: str | Path,
    output_root: str | Path,
) -> DerivedSurfaceBinding:
    """Load the current immutable surface and re-check all surface identities."""

    preview, pointer = load_automatic_reconstruction_preview(status_path, output_root=output_root)
    try:
        surface = json.loads(preview.surface_path.read_text(encoding="utf-8"))
        generation = json.loads(preview.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("active derived surface identity is unreadable") from exc
    if not isinstance(surface, dict) or not isinstance(generation, dict):
        raise ValueError("active derived surface identity is invalid")
    session_id = pointer.get("session_id")
    geometry = surface.get("measured_surface_geometry_revision_id")
    source_vertex_ids = surface.get("source_vertex_ids")
    if isinstance(source_vertex_ids, list) and source_vertex_ids and not all(isinstance(item, str) and item.count(":") == 2 for item in source_vertex_ids):
        positions = np.asarray(surface.get("positions"), dtype=np.float64)
        voxel = float(surface.get("voxel_size_m", 0.02))
        if positions.size == len(source_vertex_ids) * 3:
            positions = positions.reshape(len(source_vertex_ids), 3)
        if positions.ndim == 2 and positions.shape == (len(source_vertex_ids), 3) and np.isfinite(positions).all():
            cells = np.floor(positions / voxel).astype(np.int64)
            source_vertex_ids = [f"{int(x)}:{int(y)}:{int(z)}" for x, y, z in cells]
    artifacts = generation.get("source_artifacts")
    if (
        preview.status.get("surface_sha256") != pointer.get("surface_sha256")
        or surface.get("sensor_frame_transform_id") != preview.status.get("sensor_frame_transform_id")
        or surface.get("voxel_size_m") is None
        or not isinstance(source_vertex_ids, list)
        or len(source_vertex_ids) != preview.status.get("vertex_count")
        or not isinstance(artifacts, list)
    ):
        raise ValueError("active derived surface binding is inconsistent")
    source_hashes = surface.get("artifact_sha256")
    if not isinstance(source_hashes, list) or sorted(source_hashes) != sorted(
        item.get("sha256") for item in artifacts if isinstance(item, dict)
    ):
        raise ValueError("active derived surface source inventory is inconsistent")
    parsed_artifacts = tuple(
        ProvisionalSourceArtifact(
            relative_path=item.get("relative_path"),
            sha256=item.get("sha256"),
            bytes=item.get("bytes"),
        )
        for item in artifacts
        if isinstance(item, dict)
    )
    if len(parsed_artifacts) != len(artifacts):
        raise ValueError("active derived source artifact is invalid")
    return DerivedSurfaceBinding(
        session_id=session_id,
        source_surface_sha256=preview.status["surface_sha256"],
        measured_surface_geometry_revision_id=geometry,
        sensor_frame_transform_id=preview.status["sensor_frame_transform_id"],
        voxel_size_m=surface["voxel_size_m"],
        source_vertex_ids=tuple(source_vertex_ids),
        source_artifacts=parsed_artifacts,
    )


def load_operator_forced_runtime_projection(path: str | Path) -> RuntimeProjection:
    """Load a verified formal operator override without promoting its authority."""

    override = load_operator_forced_calibration_override(path)
    return RuntimeProjection(
        calibration_id=override.calibration_id,
        operator_override_sha256=override.override_sha256,
        camera_from_native_lidar=np.asarray(override.camera_from_lidar_row_major, dtype=np.float64).reshape(4, 4),
        runtime_projection_authorized=True,
        measurement_authority=False,
    )


def build_provisional_projection_evidence(
    *,
    binding: DerivedSurfaceBinding,
    source_root: str | Path,
    projection: RuntimeProjection,
    max_pair_delta_ns: int = 100_000_000,
    work_root: str | Path | None = None,
) -> ProvisionalProjectionEvidence:
    """Replay all bound raw artifacts and attach RGB observations to mesh IDs.

    All source artifacts in the binding must be present and hash-identical. A
    changed or substituted capture never produces a provisional preview.
    """

    if not isinstance(binding, DerivedSurfaceBinding):
        raise TypeError("binding must be DerivedSurfaceBinding")
    if not isinstance(projection, RuntimeProjection):
        raise TypeError("projection must be RuntimeProjection")
    if isinstance(max_pair_delta_ns, bool) or not isinstance(max_pair_delta_ns, int) or max_pair_delta_ns <= 0:
        raise ValueError("max_pair_delta_ns must be a positive integer")
    source = Path(source_root).resolve(strict=False)
    if not source.is_dir():
        raise ValueError("preview source root is unavailable")
    workspace = None if work_root is None else Path(work_root).resolve(strict=False)
    accumulator = ProvisionalProjectionAccumulator(
        source_vertex_ids=binding.source_vertex_ids,
        voxel_size_m=binding.voxel_size_m,
        sensor_frame_transform_id=binding.sensor_frame_transform_id,
    )
    camera_info_identities: dict[str, int] = {}
    lidar_frames = 0
    color_frames = 0
    paired_frames = 0
    for artifact in binding.source_artifacts:
        artifact_path = _source_artifact_path(source, artifact)
        _verify_artifact(artifact_path, artifact)
        with _decompressed_part(artifact_path, work_root=workspace) as sqlite_path:
            rows = _read_rows(sqlite_path)
        poses, clouds, colors, camera_infos = _decode_rows(rows)
        pose_times = [item[0] for item in poses]
        color_times = [item[0] for item in colors]
        info_times = [item[0] for item in camera_infos]
        for record_time, native_points, points_are_world in clouds:
            pose_match = _nearest(record_time, pose_times, poses, max_pair_delta_ns)
            if pose_match is None:
                continue
            _pose_time, pose = pose_match
            lidar_frames += 1
            color = _nearest(record_time, color_times, colors, max_pair_delta_ns)
            if color is None:
                continue
            color_time, color_record = color
            camera_info = _nearest(color_time, info_times, camera_infos, max_pair_delta_ns)
            if camera_info is None:
                continue
            _info_time, info_record = camera_info
            image_rgb = _rgb8(color_record)
            if image_rgb.shape[:2] != (info_record.height, info_record.width):
                raise ValueError("color image dimensions do not match its CameraInfo")
            identity = _camera_info_identity(info_record)
            camera_info_identities[identity] = camera_info_identities.get(identity, 0) + 1
            color_frames += 1
            paired_frames += 1
            accumulator.observe(
                points_native_lidar=native_points,
                pose_map_from_lidar=pose,
                color_image_rgb=image_rgb,
                color_camera_matrix=info_record.k,
                camera_from_native_lidar=projection.camera_from_native_lidar,
                color_frame_sha256=color_record.message_sha256,
                points_are_world=points_are_world,
                camera_from_points=(
                    projection.camera_from_native_lidar
                    @ np.linalg.inv(pose)
                    if points_are_world
                    else None
                ),
            )
        _verify_artifact(artifact_path, artifact)
    if not camera_info_identities:
        raise ValueError("preview requires at least one color CameraInfo identity")
    rgb, validity, color_frames_sha256 = accumulator.finalize()
    if not np.any(validity):
        raise ValueError("runtime projection produced no matching observed surface colors")
    return ProvisionalProjectionEvidence(
        session_id=binding.session_id,
        source_surface_sha256=binding.source_surface_sha256,
        measured_surface_geometry_revision_id=binding.measured_surface_geometry_revision_id,
        sensor_frame_transform_id=binding.sensor_frame_transform_id,
        voxel_size_m=binding.voxel_size_m,
        source_artifact_sha256=tuple(item.sha256 for item in binding.source_artifacts),
        source_vertex_ids=binding.source_vertex_ids,
        rgb=rgb,
        validity=validity,
        calibration_id=projection.calibration_id,
        operator_override_sha256=projection.operator_override_sha256,
        runtime_projection_authorized=True,
        measurement_authority=False,
        color_camera_info_sha256=max(camera_info_identities, key=camera_info_identities.get),
        color_source_frame_sha256=color_frames_sha256,
        lidar_frame_count=lidar_frames,
        color_frame_count=color_frames,
        paired_frame_count=paired_frames,
        projected_point_count=accumulator.projected_point_count,
        matched_surface_point_count=accumulator.matched_surface_point_count,
    )


@contextmanager
def _decompressed_part(source: Path, *, work_root: Path | None) -> Iterator[Path]:
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="go2-derived-provisional-preview-",
        suffix=".db3",
        dir=None if work_root is None else str(work_root),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            zstandard.ZstdDecompressor().copy_stream(input_file, output)
        yield temporary
    except (OSError, zstandard.ZstdError) as exc:
        raise ValueError(f"cannot decompress provisional preview source artifact: {source}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_rows(path: Path) -> list[tuple[str, str, int, bytes]]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
            database.execute("PRAGMA query_only=ON")
            topics = {row[0]: (row[1], row[2]) for row in database.execute("SELECT id, name, type FROM topics")}
            return [
                (topic[0], topic[1], int(timestamp), bytes(payload))
                for topic_id, timestamp, payload in database.execute(
                    "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp, id"
                )
                if (topic := topics.get(topic_id)) is not None
            ]
    except sqlite3.Error as exc:
        raise ValueError("provisional preview source SQLite is unreadable") from exc


def _decode_rows(rows: Sequence[tuple[str, str, int, bytes]]):
    poses: list[tuple[int, np.ndarray]] = []
    clouds: list[tuple[int, np.ndarray, bool]] = []
    colors: list[tuple[int, Any]] = []
    infos: list[tuple[int, Any]] = []
    for name, type_name, record_timestamp, payload in rows:
        try:
            if name == "/go2/lio/pose" and type_name == "geometry_msgs/msg/PoseWithCovarianceStamped":
                _header_timestamp, pose = _decode_pose(payload)
                poses.append((record_timestamp, pose))
            elif name in {"/livox/lidar", "/go2/lio/lidar", "/lio_sam_ros2/mapping/cloud_registered"} and type_name == "sensor_msgs/msg/PointCloud2":
                _header_timestamp, points = _decode_pointcloud(payload)
                clouds.append((record_timestamp, points, name == "/lio_sam_ros2/mapping/cloud_registered"))
            elif name == "/camera/color/image_raw" and type_name == "sensor_msgs/msg/Image":
                colors.append((record_timestamp, decode_image(payload)))
            elif name == "/camera/color/camera_info" and type_name == "sensor_msgs/msg/CameraInfo":
                infos.append((record_timestamp, decode_camera_info(payload)))
        except (CdrDecodeError, RgbdDecodeError, ValueError, struct.error) as exc:
            raise ValueError(f"provisional preview cannot decode {name}: {type(exc).__name__}") from exc
    return (
        sorted(poses, key=lambda item: item[0]),
        sorted(clouds, key=lambda item: item[0]),
        sorted(colors, key=lambda item: item[0]),
        sorted(infos, key=lambda item: item[0]),
    )


def _nearest(
    timestamp_ns: int,
    times: Sequence[int],
    values: Sequence[tuple[int, Any]],
    max_delta_ns: int,
) -> Any | None:
    if not values:
        return None
    index = bisect.bisect_left(times, timestamp_ns)
    candidates: list[tuple[int, Any]] = []
    if index < len(values):
        candidates.append(values[index])
    if index:
        candidates.append(values[index - 1])
    closest = min(candidates, key=lambda item: abs(item[0] - timestamp_ns))
    return closest if abs(closest[0] - timestamp_ns) <= max_delta_ns else None


def _source_artifact_path(root: Path, artifact: ProvisionalSourceArtifact) -> Path:
    candidate = (root / artifact.relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("source artifact path escapes the source root") from exc
    if not candidate.is_file() or candidate.stat().st_size != artifact.bytes:
        raise ValueError("source artifact is unavailable or its byte count changed")
    return candidate


def _verify_artifact(path: Path, artifact: ProvisionalSourceArtifact) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != artifact.sha256:
        raise ValueError("source artifact SHA-256 changed before provisional preview replay")


def _rgb8(record: Any) -> np.ndarray:
    image = np.asarray(record.image)
    if record.encoding == "rgb8":
        return image
    if record.encoding == "bgr8":
        return image[..., ::-1]
    if record.encoding == "mono8":
        return np.repeat(image[..., None], 3, axis=2)
    raise ValueError(f"unsupported provisional preview color encoding: {record.encoding}")


def _camera_info_identity(record: Any) -> str:
    return canonical_sha256(
        {
            "width": int(record.width),
            "height": int(record.height),
            "distortion_model": record.distortion_model,
            "k": np.asarray(record.k, dtype=np.float64).reshape(-1).tolist(),
            "d": np.asarray(record.d, dtype=np.float64).reshape(-1).tolist(),
            "r": np.asarray(record.r, dtype=np.float64).reshape(-1).tolist(),
            "p": np.asarray(record.p, dtype=np.float64).reshape(-1).tolist(),
            "rectified": not np.allclose(record.r, np.eye(3), atol=1e-12),
        }
    )


__all__ = [
    "DerivedSurfaceBinding",
    "ProvisionalSourceArtifact",
    "RuntimeProjection",
    "build_provisional_projection_evidence",
    "load_active_derived_surface_binding",
    "load_operator_forced_runtime_projection",
]
