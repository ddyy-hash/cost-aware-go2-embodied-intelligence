"""Validated read-only preview for the pre-ROS Go2 capture format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_SESSION_READ_ONLY = "LEGACY_SESSION_READ_ONLY"
PREVIEW_UNCALIBRATED = "PREVIEW_UNCALIBRATED"
LEGACY_SENSOR_TRANSFORM = "LEGACY_BODY_LIDAR_IDENTITY_UNCALIBRATED"
LEGACY_SURFACE_ENCODING = "PLY_BINARY_LITTLE_ENDIAN_OPEN3D_V1"

_VERTEX_PROPERTIES = (
    "property double x",
    "property double y",
    "property double z",
    "property double nx",
    "property double ny",
    "property double nz",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
)
_FACE_PROPERTY = "property list uchar uint vertex_indices"
_VERTEX_DTYPE = np.dtype([
    ("position", "<f8", (3,)),
    ("normal", "<f8", (3,)),
    ("color", "u1", (3,)),
])
_FACE_DTYPE = np.dtype([
    ("count", "u1"),
    ("indices", "<u4", (3,)),
])


@dataclass(frozen=True)
class LegacySessionPreview:
    session_root: Path
    surface_path: Path
    status: dict[str, Any]


@dataclass(frozen=True)
class _PlyLayout:
    data_offset: int
    vertex_count: int
    triangle_count: int


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"legacy {label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"legacy {label} must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _element_count(line: str, name: str) -> int:
    prefix = f"element {name} "
    if not line.startswith(prefix):
        raise ValueError(f"legacy PLY {name} element is missing")
    try:
        count = int(line[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"legacy PLY {name} count is invalid") from exc
    if count <= 0:
        raise ValueError(f"legacy PLY {name} count must be positive")
    return count


def _ply_layout(path: Path) -> _PlyLayout:
    with path.open("rb") as source:
        prefix = source.read(65536)
    terminator = b"end_header\n"
    end = prefix.find(terminator)
    if end < 0:
        raise ValueError("legacy PLY header is missing or too large")
    data_offset = end + len(terminator)
    try:
        lines = prefix[:data_offset].decode("ascii").replace("\r\n", "\n").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("legacy PLY header is not ASCII") from exc
    if len(lines) < 15 or lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
        raise ValueError("legacy PLY must use binary little-endian version 1.0")

    content = [line for line in lines[2:-1] if not line.startswith("comment ")]
    if not content or content[-1] == "end_header":
        raise ValueError("legacy PLY header layout is invalid")
    vertex_count = _element_count(content[0], "vertex")
    expected_vertex_end = 1 + len(_VERTEX_PROPERTIES)
    if tuple(content[1:expected_vertex_end]) != _VERTEX_PROPERTIES:
        raise ValueError("legacy PLY vertex layout is unsupported")
    triangle_count = _element_count(content[expected_vertex_end], "face")
    if content[expected_vertex_end + 1:] != [_FACE_PROPERTY]:
        raise ValueError("legacy PLY face layout is unsupported")

    expected_size = data_offset + vertex_count * _VERTEX_DTYPE.itemsize + triangle_count * _FACE_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError("legacy PLY byte count does not match its header")
    return _PlyLayout(data_offset, vertex_count, triangle_count)


def _validate_ply_arrays(path: Path, layout: _PlyLayout) -> None:
    vertices = np.memmap(
        path,
        dtype=_VERTEX_DTYPE,
        mode="r",
        offset=layout.data_offset,
        shape=(layout.vertex_count,),
    )
    try:
        if not np.all(np.isfinite(vertices["position"])) or not np.all(np.isfinite(vertices["normal"])):
            raise ValueError("legacy PLY contains non-finite geometry")
    finally:
        del vertices

    face_offset = layout.data_offset + layout.vertex_count * _VERTEX_DTYPE.itemsize
    faces = np.memmap(
        path,
        dtype=_FACE_DTYPE,
        mode="r",
        offset=face_offset,
        shape=(layout.triangle_count,),
    )
    try:
        if not np.all(faces["count"] == 3):
            raise ValueError("legacy PLY contains a non-triangular face")
        if np.any(faces["indices"] >= layout.vertex_count):
            raise ValueError("legacy PLY contains an out-of-range triangle index")
    finally:
        del faces


def load_legacy_session_preview(path: str | Path) -> LegacySessionPreview:
    session_root = Path(path).expanduser().resolve(strict=False)
    if not session_root.is_dir():
        raise ValueError("GO2_MAPPING_LEGACY_SESSION_ROOT must name an existing session directory")

    recording = _json_object(session_root / "metadata.json", "recording metadata")
    if recording.get("format_version") != 1 or recording.get("status") != "complete":
        raise ValueError("legacy recording is not a complete format-version-1 session")
    counts = recording.get("counts")
    required_counts = ("lidar", "motion", "rgb", "depth")
    if not isinstance(counts, dict) or any(
        isinstance(counts.get(name), bool) or not isinstance(counts.get(name), int) or counts[name] <= 0
        for name in required_counts
    ):
        raise ValueError("legacy recording counts are invalid")

    reconstruction_root = session_root / "reconstruction"
    audit = _json_object(reconstruction_root / "session_audit.json", "session audit")
    trajectory = _json_object(reconstruction_root / "trajectory_meta.json", "trajectory metadata")
    lidar = _json_object(reconstruction_root / "lidar_world_meta.json", "LiDAR metadata")
    extrinsic = _json_object(
        reconstruction_root / "linux_fusion" / "extrinsic_calibration.json",
        "extrinsic calibration",
    )
    if audit.get("calibration", {}).get("body_to_depth_available") is not False:
        raise ValueError("legacy body-to-depth calibration state is not explicitly unavailable")
    if trajectory.get("T_body_camera") != "identity_uncalibrated":
        raise ValueError("legacy body-to-camera transform state is unsupported")
    metrics = extrinsic.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("accepted") is not False:
        raise ValueError("legacy LiDAR-to-depth extrinsic must remain unaccepted")
    if "assumed identity" not in str(extrinsic.get("unresolved", "")):
        raise ValueError("legacy body-to-LiDAR uncertainty is missing")
    floor_height_offset_m = lidar.get("floor_height_offset_m")
    if isinstance(floor_height_offset_m, bool) or not isinstance(floor_height_offset_m, (int, float)):
        raise ValueError("legacy floor-height offset is invalid")
    floor_height_offset_m = float(floor_height_offset_m)
    if not np.isfinite(floor_height_offset_m):
        raise ValueError("legacy floor-height offset must be finite")

    surface_path = (reconstruction_root / "reconstruction.ply").resolve(strict=False)
    if not surface_path.is_file():
        raise ValueError("legacy reconstruction PLY is unavailable")
    layout = _ply_layout(surface_path)
    _validate_ply_arrays(surface_path, layout)
    status = {
        "schema_version": 1,
        "available": True,
        "blocker": None,
        "source_mode": LEGACY_SESSION_READ_ONLY,
        "read_only": True,
        "quality": PREVIEW_UNCALIBRATED,
        "geometry_authority": False,
        "hidden_surface_policy": "LEGACY_RECONSTRUCTION_UNVERIFIED",
        "sensor_frame_transform_id": LEGACY_SENSOR_TRANSFORM,
        "session_id": session_root.name,
        "surface_encoding": LEGACY_SURFACE_ENCODING,
        "surface_sha256": _sha256_file(surface_path),
        "vertex_count": layout.vertex_count,
        "triangle_count": layout.triangle_count,
        "has_vertex_colors": True,
        "body_to_depth_calibrated": False,
        "body_to_lidar_calibrated": False,
        "floor_height_authority": False,
        "floor_height_offset_m": floor_height_offset_m,
    }
    return LegacySessionPreview(session_root, surface_path, status)


__all__ = [
    "LEGACY_SESSION_READ_ONLY",
    "LegacySessionPreview",
    "load_legacy_session_preview",
]
