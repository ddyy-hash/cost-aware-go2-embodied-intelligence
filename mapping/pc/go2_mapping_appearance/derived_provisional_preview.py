"""Publish RGB-D color previews for an existing derived LiDAR surface.

Each preview is an inspection artifact tied to an immutable surface and an
operator-forced projection. It is separate from the validated appearance store
and cannot authorize texture delivery or final export.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any, Sequence
import uuid

import numpy as np

from mapping.pc.go2_mapping_live.sensor_frame import (
    normalize_sensor_frame_transform_id,
    transform_points,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, is_sha256


PROVISIONAL_VERTEX_COLOR_STATE = "PROVISIONAL_VERTEX_COLOR_UNCALIBRATED"
_MAGIC = b"G2PVCOL1"
_HEADER = struct.Struct("<8sHHI32s32s")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_SESSION_RE = re.compile(r"^segment-\d{8}T\d{6}Z$", re.ASCII)
_VOXEL_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


@dataclass(frozen=True)
class ProvisionalProjectionEvidence:
    """Color values and immutable provenance for one observed surface."""

    session_id: str
    source_surface_sha256: str
    measured_surface_geometry_revision_id: str
    sensor_frame_transform_id: str
    voxel_size_m: float
    source_artifact_sha256: tuple[str, ...]
    source_vertex_ids: tuple[str, ...]
    rgb: np.ndarray
    validity: np.ndarray
    calibration_id: str
    operator_override_sha256: str
    runtime_projection_authorized: bool
    measurement_authority: bool
    color_camera_info_sha256: str
    color_source_frame_sha256: tuple[str, ...]
    lidar_frame_count: int
    color_frame_count: int
    paired_frame_count: int
    projected_point_count: int
    matched_surface_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or _SESSION_RE.fullmatch(self.session_id) is None:
            raise ValueError("preview session ID is invalid")
        object.__setattr__(self, "source_surface_sha256", _sha256(self.source_surface_sha256, "source_surface_sha256"))
        object.__setattr__(
            self,
            "measured_surface_geometry_revision_id",
            _sha256(self.measured_surface_geometry_revision_id, "measured_surface_geometry_revision_id"),
        )
        object.__setattr__(self, "operator_override_sha256", _sha256(self.operator_override_sha256, "operator_override_sha256"))
        object.__setattr__(self, "color_camera_info_sha256", _sha256(self.color_camera_info_sha256, "color_camera_info_sha256"))
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("preview calibration_id is required")
        if self.runtime_projection_authorized is not True:
            raise ValueError("preview requires an authorized runtime projection")
        if self.measurement_authority is not False:
            raise ValueError("preview must not claim measurement authority")
        voxel_size = float(self.voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("preview voxel_size_m must be finite and positive")
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(
            self,
            "sensor_frame_transform_id",
            normalize_sensor_frame_transform_id(self.sensor_frame_transform_id),
        )
        artifacts = _sha256_tuple(self.source_artifact_sha256, "source_artifact_sha256")
        frames = _sha256_tuple(self.color_source_frame_sha256, "color_source_frame_sha256")
        if not artifacts:
            raise ValueError("preview requires source artifacts")
        if not frames:
            raise ValueError("preview requires color source frames")
        object.__setattr__(self, "source_artifact_sha256", artifacts)
        object.__setattr__(self, "color_source_frame_sha256", frames)
        vertex_ids = tuple(self.source_vertex_ids)
        if not vertex_ids:
            raise ValueError("preview source vertex IDs must be non-empty")
        _vertex_keys(vertex_ids)
        object.__setattr__(self, "source_vertex_ids", vertex_ids)
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        validity = np.asarray(self.validity, dtype=np.bool_).reshape(-1)
        if rgb.shape != (len(vertex_ids), 3) or len(validity) != len(vertex_ids):
            raise ValueError("preview color arrays do not match source vertex IDs")
        rgb = np.array(rgb, copy=True)
        if np.any(rgb[~validity] != 0):
            raise ValueError("invalid preview color records must have zero RGB")
        rgb.setflags(write=False)
        validity = np.array(validity, copy=True)
        validity.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "validity", validity)
        for field in (
            "lidar_frame_count",
            "color_frame_count",
            "paired_frame_count",
            "projected_point_count",
            "matched_surface_point_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"preview {field} must be a non-negative integer")
        if self.paired_frame_count > self.lidar_frame_count:
            raise ValueError("preview paired frame count exceeds LiDAR frame count")
        if self.matched_surface_point_count > self.projected_point_count:
            raise ValueError("preview matched points exceed projected points")


@dataclass(frozen=True)
class ParsedProvisionalVertexColor:
    vertex_count: int
    geometry_revision_id: str
    preview_revision_id: str
    rgb: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        validity = np.asarray(self.validity, dtype=np.uint8).reshape(-1)
        if len(rgb) != self.vertex_count or len(validity) != self.vertex_count:
            raise ValueError("provisional vertex-color arrays do not match vertex_count")
        rgb.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True)
class PublishedProvisionalSurfacePreview:
    preview_revision_id: str
    manifest: dict[str, Any]
    root_path: Path
    manifest_path: Path
    vertex_color_path: Path


class ProvisionalProjectionAccumulator:
    """Match projected native LiDAR points to transformed mapping-frame cells."""

    def __init__(
        self,
        *,
        source_vertex_ids: Sequence[str],
        voxel_size_m: float,
        sensor_frame_transform_id: str,
    ) -> None:
        vertex_ids = tuple(source_vertex_ids)
        if not vertex_ids:
            raise ValueError("source_vertex_ids must be non-empty")
        voxel_size = float(voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        keys = _vertex_keys(vertex_ids)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        self._sorted_keys = _voxel_records(keys[order])
        self._sorted_vertex_indices = order.astype(np.int64, copy=False)
        self._vertex_count = len(vertex_ids)
        self._voxel_size_m = voxel_size
        self._sensor_frame_transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)
        self._linear_sum = np.zeros((self._vertex_count, 3), dtype=np.float64)
        self._sample_count = np.zeros(self._vertex_count, dtype=np.uint32)
        self._frame_hashes: set[str] = set()
        self.projected_point_count = 0
        self.matched_surface_point_count = 0

    def observe(
        self,
        *,
        points_native_lidar: np.ndarray,
        pose_map_from_lidar: np.ndarray,
        color_image_rgb: np.ndarray,
        color_camera_matrix: np.ndarray,
        camera_from_native_lidar: np.ndarray,
        color_frame_sha256: str,
        points_are_world: bool = False,
        camera_from_points: np.ndarray | None = None,
    ) -> int:
        """Accumulate one color/LiDAR/LIO association without touching geometry.

        ``camera_from_native_lidar`` intentionally projects the untransformed
        raw LiDAR points.  The separate sensor-frame transform is applied only
        before world-cell matching, exactly as the mapper constructs its
        ``source_vertex_ids``.
        """

        frame_hash = _sha256(color_frame_sha256, "color_frame_sha256")
        points = _points(points_native_lidar, "points_native_lidar")
        if not len(points):
            return 0
        pose = _matrix(pose_map_from_lidar, "pose_map_from_lidar")
        image = np.asarray(color_image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
            raise ValueError("color_image_rgb must be a non-empty H by W by 3 image")
        camera = np.asarray(color_camera_matrix, dtype=np.float64)
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
            raise ValueError("color_camera_matrix must be a finite positive focal-length matrix")
        camera_from_lidar = _matrix(
            camera_from_native_lidar if camera_from_points is None else camera_from_points,
            "camera_from_native_lidar",
        )

        camera_points = points @ camera_from_lidar[:3, :3].T + camera_from_lidar[:3, 3]
        in_front = camera_points[:, 2] > 0.0
        if not np.any(in_front):
            return 0
        candidate_indices = np.flatnonzero(in_front)
        candidate_points = camera_points[candidate_indices]
        u = np.rint(camera[0, 0] * candidate_points[:, 0] / candidate_points[:, 2] + camera[0, 2]).astype(np.int64)
        v = np.rint(camera[1, 1] * candidate_points[:, 1] / candidate_points[:, 2] + camera[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
        if not np.any(inside):
            return 0
        candidate_indices = candidate_indices[inside]
        camera_points = candidate_points[inside]
        u, v = u[inside], v[inside]
        self.projected_point_count += int(len(candidate_indices))

        if points_are_world:
            world = points[candidate_indices]
        else:
            mapping_points = transform_points(points[candidate_indices], self._sensor_frame_transform_id)
            world = mapping_points @ pose[:3, :3].T + pose[:3, 3]
        cells = np.floor(world / self._voxel_size_m).astype(np.int64)
        incoming = _voxel_records(cells)
        positions = np.searchsorted(self._sorted_keys, incoming)
        target = positions < len(self._sorted_keys)
        if np.any(target):
            matched_positions = positions[target]
            target[target] = self._sorted_keys[matched_positions] == incoming[target]
        # Registered cloud and mesh can differ by one source voxel at room
        # boundaries. Probe the 26 neighboring cells before discarding a point.
        if np.any(~target):
            missing = np.flatnonzero(~target)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        if not len(missing):
                            break
                        shifted = _voxel_records(cells[missing] + np.array([dx, dy, dz], dtype=np.int64))
                        probe = np.searchsorted(self._sorted_keys, shifted)
                        valid = probe < len(self._sorted_keys)
                        if np.any(valid):
                            valid_positions = np.flatnonzero(valid)
                            valid[valid_positions] = self._sorted_keys[probe[valid_positions]] == shifted[valid_positions]
                        if np.any(valid):
                            accepted = missing[valid]
                            positions[accepted] = probe[valid]
                            target[accepted] = True
                            missing = missing[~valid]
                    if not len(missing):
                        break
                if not len(missing):
                    break
        if not np.any(target):
            return 0
        camera_points = camera_points[target]
        u, v = u[target], v[target]
        vertices = self._sorted_vertex_indices[positions[target]]

        # Keep the closest LiDAR return per image pixel, then one observation
        # per final mapping vertex for this color frame.
        pixels = v * image.shape[1] + u
        by_pixel = np.lexsort((camera_points[:, 2], pixels))
        first_pixel = np.empty(len(by_pixel), dtype=np.bool_)
        first_pixel[0] = True
        first_pixel[1:] = pixels[by_pixel][1:] != pixels[by_pixel][:-1]
        selected = by_pixel[first_pixel]
        by_vertex = np.lexsort((camera_points[selected, 2], vertices[selected]))
        selected = selected[by_vertex]
        first_vertex = np.empty(len(selected), dtype=np.bool_)
        first_vertex[0] = True
        first_vertex[1:] = vertices[selected][1:] != vertices[selected][:-1]
        selected = selected[first_vertex]
        if not len(selected):
            return 0

        sampled = image[v[selected], u[selected]].astype(np.float64) / 255.0
        linear = _srgb_to_linear(sampled)
        selected_vertices = vertices[selected]
        np.add.at(self._linear_sum, selected_vertices, linear)
        np.add.at(self._sample_count, selected_vertices, 1)
        self._frame_hashes.add(frame_hash)
        self.matched_surface_point_count += int(len(selected_vertices))
        return int(len(selected_vertices))

    def finalize(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        valid = self._sample_count > 0
        rgb = np.zeros((self._vertex_count, 3), dtype=np.uint8)
        if np.any(valid):
            mean_linear = self._linear_sum[valid] / self._sample_count[valid, None]
            rgb[valid] = _linear_to_srgb8(mean_linear)
        return rgb, valid, tuple(sorted(self._frame_hashes))


def publish_provisional_surface_preview(
    output_root: str | Path,
    evidence: ProvisionalProjectionEvidence,
) -> PublishedProvisionalSurfacePreview:
    """Atomically publish a content-addressed preview outside final stores."""

    if not isinstance(evidence, ProvisionalProjectionEvidence):
        raise TypeError("evidence must be ProvisionalProjectionEvidence")
    manifest, binary = _build_sidecar(evidence)
    revision = manifest["preview_revision_id"]
    root = Path(output_root).resolve(strict=False)
    destination = root / evidence.session_id / evidence.source_surface_sha256 / evidence.measured_surface_geometry_revision_id / revision
    if destination.exists():
        published = load_provisional_surface_preview(destination / "manifest.json")
        if published.manifest != manifest or published.vertex_color_path.read_bytes() != binary:
            raise ValueError("provisional preview revision conflicts with immutable content")
        return published
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".partial-{revision}-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        _atomic_write(staging / "vertex-color.bin", binary)
        _atomic_write(staging / "manifest.json", _canonical_bytes(manifest))
        try:
            os.replace(staging, destination)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
        return load_provisional_surface_preview(destination / "manifest.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def load_provisional_surface_preview(
    manifest_path: str | Path,
    *,
    expected_surface_sha256: str | None = None,
    expected_geometry_revision_id: str | None = None,
) -> PublishedProvisionalSurfacePreview:
    """Load and verify one explicitly non-deliverable vertex-color preview."""

    path = Path(manifest_path).resolve(strict=False)
    try:
        body = path.read_bytes()
        manifest = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provisional preview manifest is unreadable") from exc
    if not isinstance(manifest, dict) or body != _canonical_bytes(manifest):
        raise ValueError("provisional preview manifest is not canonical")
    _validate_manifest_authority(manifest)
    revision = _sha256(manifest.get("preview_revision_id"), "preview revision")
    if manifest.get("preview_revision_id") != revision:
        raise ValueError("provisional preview revision must be lowercase")
    core = manifest.get("revision_core")
    if not isinstance(core, dict) or canonical_sha256(core) != revision:
        raise ValueError("provisional preview revision identity is invalid")
    if manifest.get("source_surface_sha256") != core.get("source_surface_sha256"):
        raise ValueError("provisional preview source surface identity is invalid")
    if manifest.get("measured_surface_geometry_revision_id") != core.get("measured_surface_geometry_revision_id"):
        raise ValueError("provisional preview geometry revision identity is invalid")
    if expected_surface_sha256 is not None and manifest["source_surface_sha256"] != _sha256(
        expected_surface_sha256, "expected surface SHA-256"
    ):
        raise ValueError("provisional preview source surface does not match the requested surface")
    if expected_geometry_revision_id is not None and manifest["measured_surface_geometry_revision_id"] != _sha256(
        expected_geometry_revision_id, "expected geometry revision"
    ):
        raise ValueError("provisional preview geometry revision does not match the requested surface")
    vertex_color = manifest.get("vertex_color")
    if not isinstance(vertex_color, dict) or vertex_color.get("file_name") != "vertex-color.bin":
        raise ValueError("provisional preview vertex-color reference is invalid")
    payload_path = path.with_name("vertex-color.bin")
    try:
        payload = payload_path.read_bytes()
    except OSError as exc:
        raise ValueError("provisional preview vertex-color payload is unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != _sha256(vertex_color.get("content_sha256"), "vertex-color content SHA-256"):
        raise ValueError("provisional preview vertex-color checksum is invalid")
    parsed = parse_provisional_vertex_color_v1(
        payload,
        expected_geometry_revision_id=manifest["measured_surface_geometry_revision_id"],
        expected_preview_revision_id=revision,
        expected_vertex_count=manifest.get("vertex_count"),
    )
    colored = int(np.count_nonzero(parsed.validity))
    if colored != manifest.get("colored_vertex_count") or (colored / parsed.vertex_count) != manifest.get("surface_coverage"):
        raise ValueError("provisional preview color coverage is invalid")
    if manifest.get("vertex_color_payload_sha256") != hashlib.sha256(payload[_HEADER.size:]).hexdigest():
        raise ValueError("provisional preview color payload identity is invalid")
    return PublishedProvisionalSurfacePreview(
        preview_revision_id=revision,
        manifest=manifest,
        root_path=path.parent,
        manifest_path=path,
        vertex_color_path=payload_path,
    )


def parse_provisional_vertex_color_v1(
    payload: bytes,
    *,
    expected_geometry_revision_id: str | None = None,
    expected_preview_revision_id: str | None = None,
    expected_vertex_count: int | None = None,
) -> ParsedProvisionalVertexColor:
    """Parse the preview-only RGB8/validity payload without final-store reuse."""

    value = bytes(payload)
    if len(value) < _HEADER.size:
        raise ValueError("provisional vertex-color payload is shorter than its header")
    magic, version, header_bytes, vertex_count, geometry_raw, preview_raw = _HEADER.unpack_from(value)
    if magic != _MAGIC or version != 1 or header_bytes != _HEADER.size:
        raise ValueError("vertex-color header is not G2PVCOL1")
    if vertex_count <= 0 or len(value) != _HEADER.size + vertex_count * 4:
        raise ValueError("provisional vertex-color payload length does not match vertex_count")
    geometry = geometry_raw.hex()
    preview = preview_raw.hex()
    if expected_geometry_revision_id is not None and geometry != _sha256(
        expected_geometry_revision_id, "expected geometry revision"
    ):
        raise ValueError("provisional vertex-color geometry revision does not match")
    if expected_preview_revision_id is not None and preview != _sha256(
        expected_preview_revision_id, "expected preview revision"
    ):
        raise ValueError("provisional vertex-color preview revision does not match")
    if expected_vertex_count is not None and (
        isinstance(expected_vertex_count, bool) or expected_vertex_count != vertex_count
    ):
        raise ValueError("provisional vertex-color count does not match")
    records = np.frombuffer(value, dtype=np.uint8, offset=_HEADER.size).reshape(vertex_count, 4)
    validity = np.array(records[:, 3], copy=True)
    if np.any((validity != 0) & (validity != 255)):
        raise ValueError("provisional vertex-color validity must be 0 or 255")
    rgb = np.array(records[:, :3], copy=True)
    if np.any(rgb[validity == 0] != 0):
        raise ValueError("invalid provisional vertex-color records must have zero RGB")
    return ParsedProvisionalVertexColor(vertex_count, geometry, preview, rgb, validity)


def _build_sidecar(evidence: ProvisionalProjectionEvidence) -> tuple[dict[str, Any], bytes]:
    payload = bytearray()
    for color, valid in zip(evidence.rgb, evidence.validity):
        if valid:
            payload.extend(np.asarray(color, dtype=np.uint8).tobytes())
            payload.append(255)
        else:
            payload.extend(b"\0\0\0\0")
    payload_bytes = bytes(payload)
    colored = int(np.count_nonzero(evidence.validity))
    vertex_count = len(evidence.source_vertex_ids)
    source_vertex_ids_sha256 = canonical_sha256({"source_vertex_ids": list(evidence.source_vertex_ids)})
    revision_core = {
        "schema_version": 1,
        "state": PROVISIONAL_VERTEX_COLOR_STATE,
        "session_id": evidence.session_id,
        "source_surface_sha256": evidence.source_surface_sha256,
        "measured_surface_geometry_revision_id": evidence.measured_surface_geometry_revision_id,
        "sensor_frame_transform_id": evidence.sensor_frame_transform_id,
        "voxel_size_m": evidence.voxel_size_m,
        "source_artifact_sha256": list(evidence.source_artifact_sha256),
        "source_vertex_ids_sha256": source_vertex_ids_sha256,
        "vertex_count": vertex_count,
        "colored_vertex_count": colored,
        "vertex_color_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "calibration_id": evidence.calibration_id,
        "operator_override_sha256": evidence.operator_override_sha256,
        "runtime_projection_authorized": True,
        "measurement_authority": False,
        "color_camera_info_sha256": evidence.color_camera_info_sha256,
        "color_source_frame_sha256": list(evidence.color_source_frame_sha256),
        "lidar_frame_count": evidence.lidar_frame_count,
        "color_frame_count": evidence.color_frame_count,
        "paired_frame_count": evidence.paired_frame_count,
        "projected_point_count": evidence.projected_point_count,
        "matched_surface_point_count": evidence.matched_surface_point_count,
    }
    preview_revision_id = canonical_sha256(revision_core)
    binary = _HEADER.pack(
        _MAGIC,
        1,
        _HEADER.size,
        vertex_count,
        bytes.fromhex(evidence.measured_surface_geometry_revision_id),
        bytes.fromhex(preview_revision_id),
    ) + payload_bytes
    content_sha256 = hashlib.sha256(binary).hexdigest()
    manifest = {
        "schema_version": 1,
        "state": PROVISIONAL_VERTEX_COLOR_STATE,
        "preview_revision_id": preview_revision_id,
        "source_mode": "DERIVED_READ_ONLY",
        "read_only": True,
        "final_store_eligible": False,
        "delivery_authorized": False,
        "measurement_authority": False,
        "runtime_projection_authorized": True,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "session_id": evidence.session_id,
        "source_surface_sha256": evidence.source_surface_sha256,
        "measured_surface_geometry_revision_id": evidence.measured_surface_geometry_revision_id,
        "sensor_frame_transform_id": evidence.sensor_frame_transform_id,
        "voxel_size_m": evidence.voxel_size_m,
        "source_artifact_sha256": list(evidence.source_artifact_sha256),
        "source_vertex_ids_sha256": source_vertex_ids_sha256,
        "vertex_count": vertex_count,
        "colored_vertex_count": colored,
        "surface_coverage": colored / vertex_count,
        "calibration_id": evidence.calibration_id,
        "operator_override_sha256": evidence.operator_override_sha256,
        "color_camera_info_sha256": evidence.color_camera_info_sha256,
        "color_source_frame_sha256": list(evidence.color_source_frame_sha256),
        "lidar_frame_count": evidence.lidar_frame_count,
        "color_frame_count": evidence.color_frame_count,
        "paired_frame_count": evidence.paired_frame_count,
        "projected_point_count": evidence.projected_point_count,
        "matched_surface_point_count": evidence.matched_surface_point_count,
        "vertex_color_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "revision_core": revision_core,
        "vertex_color": {
            "file_name": "vertex-color.bin",
            "content_sha256": content_sha256,
            "format": "G2PVCOL1_RGB8_VALIDITY8",
        },
        "blockers": [
            "PROVISIONAL_RUNTIME_PROJECTION_ONLY",
            "MEASUREMENT_CALIBRATION_NOT_AUTHORIZED",
            "FINAL_TEXTURE_DELIVERY_NOT_AUTHORIZED",
        ],
    }
    return manifest, binary


def _validate_manifest_authority(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("state") != PROVISIONAL_VERTEX_COLOR_STATE:
        raise ValueError("provisional preview state is invalid")
    expected = {
        "source_mode": "DERIVED_READ_ONLY",
        "read_only": True,
        "final_store_eligible": False,
        "delivery_authorized": False,
        "measurement_authority": False,
        "runtime_projection_authorized": True,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("provisional preview authority is invalid")
    if not isinstance(manifest.get("session_id"), str) or _SESSION_RE.fullmatch(manifest["session_id"]) is None:
        raise ValueError("provisional preview session is invalid")
    for key in (
        "source_surface_sha256",
        "measured_surface_geometry_revision_id",
        "operator_override_sha256",
        "color_camera_info_sha256",
        "source_vertex_ids_sha256",
        "vertex_color_payload_sha256",
    ):
        if manifest.get(key) != _sha256(manifest.get(key), key):
            raise ValueError("provisional preview identity is invalid")
    if not isinstance(manifest.get("calibration_id"), str) or not manifest["calibration_id"].strip():
        raise ValueError("provisional preview calibration identity is invalid")
    normalize_sensor_frame_transform_id(manifest.get("sensor_frame_transform_id"))
    voxel = manifest.get("voxel_size_m")
    if isinstance(voxel, bool) or not isinstance(voxel, (int, float)) or not math.isfinite(float(voxel)) or float(voxel) <= 0.0:
        raise ValueError("provisional preview voxel size is invalid")
    artifacts = _sha256_tuple(manifest.get("source_artifact_sha256"), "source_artifact_sha256")
    frames = _sha256_tuple(manifest.get("color_source_frame_sha256"), "color_source_frame_sha256")
    if not artifacts or not frames:
        raise ValueError("provisional preview source evidence is invalid")
    vertex_count = manifest.get("vertex_count")
    colored = manifest.get("colored_vertex_count")
    if (
        isinstance(vertex_count, bool)
        or not isinstance(vertex_count, int)
        or vertex_count <= 0
        or isinstance(colored, bool)
        or not isinstance(colored, int)
        or not 0 <= colored <= vertex_count
        or manifest.get("surface_coverage") != colored / vertex_count
    ):
        raise ValueError("provisional preview vertex counts are invalid")
    for field in (
        "lidar_frame_count",
        "color_frame_count",
        "paired_frame_count",
        "projected_point_count",
        "matched_surface_point_count",
    ):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provisional preview metrics are invalid")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_tuple(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{label} must be a sequence of SHA-256 digests")
    normalized = tuple(sorted({_sha256(value, label) for value in values}))
    if len(normalized) != len(values):
        raise ValueError(f"{label} must not contain duplicate SHA-256 digests")
    return normalized


def _vertex_keys(vertex_ids: Sequence[str]) -> np.ndarray:
    keys = np.empty((len(vertex_ids), 3), dtype=np.int64)
    for index, vertex_id in enumerate(vertex_ids):
        if not isinstance(vertex_id, str):
            raise ValueError("source vertex ID must be a string")
        pieces = vertex_id.split(":")
        if len(pieces) != 3:
            raise ValueError("source vertex ID must contain three voxel coordinates")
        try:
            keys[index] = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError("source vertex ID contains an invalid voxel coordinate") from exc
    return keys


def _voxel_records(keys: np.ndarray) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    records = np.empty(len(values), dtype=_VOXEL_DTYPE)
    records["x"], records["y"], records["z"] = values[:, 0], values[:, 1], values[:, 2]
    return records


def _points(value: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError(f"{label} must be a finite N by 3 array")
    return points


def _matrix(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4 by 4 matrix")
    return matrix


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.floor(np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


__all__ = [
    "PROVISIONAL_VERTEX_COLOR_STATE",
    "ParsedProvisionalVertexColor",
    "ProvisionalProjectionAccumulator",
    "ProvisionalProjectionEvidence",
    "PublishedProvisionalSurfacePreview",
    "load_provisional_surface_preview",
    "parse_provisional_vertex_color_v1",
    "publish_provisional_surface_preview",
]
