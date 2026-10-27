"""Immutable RGB8 plus validity sidecars for historical measured surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
from typing import Any, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_rgbd.texture import VertexColor


_MAGIC = b"G2VCOLR1"
_HEADER = struct.Struct("<8sHHI32s32s")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$", re.ASCII)
_MINIMUM_COVERAGE_PPM = 800_000
_MINIMUM_INDEPENDENT_VIEWS = 2


@dataclass(frozen=True)
class VertexColorSidecar:
    manifest: dict[str, Any]
    binary: bytes


@dataclass(frozen=True)
class ParsedVertexColor:
    vertex_count: int
    geometry_revision_id: str
    appearance_revision_id: str
    rgb: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        validity = np.asarray(self.validity, dtype=np.uint8).reshape(-1)
        if len(rgb) != self.vertex_count or len(validity) != self.vertex_count:
            raise ValueError("vertex-color arrays do not match vertex_count")
        rgb.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "validity", validity)


def build_historical_vertex_color_sidecar(
    *,
    geometry_revision_id: str,
    pose_generation: str,
    calibration_id: str,
    calibration_sha256: str,
    physical_depth_validation_revision: str,
    appearance_config_sha256: str,
    vertex_colors: Sequence[VertexColor],
) -> VertexColorSidecar:
    geometry_hash = _sha256(geometry_revision_id, "geometry_revision_id")
    pose_hash = _sha256(pose_generation, "pose_generation")
    calibration_hash = _sha256(calibration_sha256, "calibration_sha256")
    validation_hash = _sha256(
        physical_depth_validation_revision,
        "physical_depth_validation_revision",
    )
    config_hash = _sha256(appearance_config_sha256, "appearance_config_sha256")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("calibration_id is required")
    colors = tuple(vertex_colors)
    if not colors:
        raise ValueError("vertex_colors must contain at least one vertex")

    vertex_ids: set[str] = set()
    source_hashes: set[str] = set()
    records = bytearray()
    colored_count = 0
    accepted_views: list[int] = []
    for color in colors:
        if not isinstance(color.vertex_id, str) or not color.vertex_id:
            raise ValueError("every vertex color needs a vertex_id")
        if color.vertex_id in vertex_ids:
            raise ValueError("vertex_colors must not contain duplicate vertex IDs")
        vertex_ids.add(color.vertex_id)
        source_hashes.update(_sha256(value, "source_frame_sha256") for value in color.source_frame_sha256)
        linear = np.asarray(color.mean_linear_rgb, dtype=np.float64)
        if linear.shape != (3,) or not np.all(np.isfinite(linear)):
            raise ValueError("mean_linear_rgb must contain three finite values")
        if color.accepted_for_texture:
            if np.any((linear < 0.0) | (linear > 1.0)):
                raise ValueError("authorized mean_linear_rgb must be within [0, 1]")
            records.extend(_linear_rgb8(linear))
            records.append(255)
            colored_count += 1
            accepted_views.append(int(color.independent_views))
        else:
            records.extend(b"\0\0\0\0")

    vertex_count = len(colors)
    coverage_ppm = colored_count * 1_000_000 // vertex_count
    minimum_views = min(accepted_views, default=0)
    payload = bytes(records)
    payload_hash = hashlib.sha256(payload).hexdigest()
    revision_core = {
        "schema_version": 1,
        "measured_surface_geometry_revision_id": geometry_hash,
        "pose_generation": pose_hash,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_hash,
        "physical_depth_validation_revision": validation_hash,
        "source_frame_sha256": sorted(source_hashes),
        "appearance_config_sha256": config_hash,
        "vertex_count": vertex_count,
        "colored_vertex_count": colored_count,
        "coverage_ppm": coverage_ppm,
        "vertex_color_payload_sha256": payload_hash,
    }
    appearance_revision_id = canonical_sha256(revision_core)
    binary = _HEADER.pack(
        _MAGIC,
        1,
        _HEADER.size,
        vertex_count,
        bytes.fromhex(geometry_hash),
        bytes.fromhex(appearance_revision_id),
    ) + payload
    content_hash = hashlib.sha256(binary).hexdigest()
    blockers: list[str] = []
    if coverage_ppm < _MINIMUM_COVERAGE_PPM:
        blockers.append("VERTEX_COLOR_COVERAGE_BELOW_800000_PPM")
    if minimum_views < _MINIMUM_INDEPENDENT_VIEWS:
        blockers.append("VERTEX_COLOR_INDEPENDENT_VIEWS_BELOW_2")
    delivery_authorized = not blockers
    manifest = {
        "schema_version": 1,
        "state": "VERTEX_COLOR_READY" if colored_count else "TEXTURE_NOT_AUTHORIZED",
        "appearance_revision_id": appearance_revision_id,
        "measured_surface_geometry_revision_id": geometry_hash,
        "pose_generation": pose_hash,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_hash,
        "vertex_count": vertex_count,
        "colored_vertex_count": colored_count,
        "surface_coverage": colored_count / vertex_count,
        "coverage_ppm": coverage_ppm,
        "minimum_independent_accepted_views": minimum_views,
        "minimum_required_coverage_ppm": _MINIMUM_COVERAGE_PPM,
        "minimum_required_independent_views": _MINIMUM_INDEPENDENT_VIEWS,
        "physical_depth_validation_revision": validation_hash,
        "appearance_config_sha256": config_hash,
        "source_frame_sha256": sorted(source_hashes),
        "vertex_color_payload_sha256": payload_hash,
        "revision_core": revision_core,
        "vertex_color": {
            "url": f"/api/reconstruction/appearance/{appearance_revision_id}/vertex-color.bin",
            "content_sha256": content_hash,
            "format": "RGB8_VALIDITY8_V1",
        },
        "immutable_cache_key": f"{geometry_hash}/{appearance_revision_id}/{content_hash}",
        "delivery_authorized": delivery_authorized,
        "blockers": blockers,
    }
    return VertexColorSidecar(manifest=manifest, binary=binary)


def parse_vertex_color_v1(payload: bytes) -> ParsedVertexColor:
    value = bytes(payload)
    if len(value) < _HEADER.size:
        raise ValueError("vertex-color payload is shorter than its header")
    magic, version, header_bytes, vertex_count, geometry_raw, appearance_raw = _HEADER.unpack_from(value)
    if magic != _MAGIC or version != 1 or header_bytes != _HEADER.size:
        raise ValueError("vertex-color header is not RGB8_VALIDITY8_V1")
    if vertex_count <= 0 or len(value) != _HEADER.size + vertex_count * 4:
        raise ValueError("vertex-color payload length does not match vertex_count")
    records = np.frombuffer(value, dtype=np.uint8, offset=_HEADER.size).reshape(vertex_count, 4)
    validity = np.array(records[:, 3], copy=True)
    if np.any((validity != 0) & (validity != 255)):
        raise ValueError("vertex-color validity must be 0 or 255")
    rgb = np.array(records[:, :3], copy=True)
    if np.any(rgb[validity == 0] != 0):
        raise ValueError("invalid vertex-color records must have zero RGB")
    return ParsedVertexColor(
        vertex_count=vertex_count,
        geometry_revision_id=geometry_raw.hex(),
        appearance_revision_id=appearance_raw.hex(),
        rgb=rgb,
        validity=validity,
    )


def _linear_rgb8(linear: np.ndarray) -> bytes:
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    quantized = np.floor(np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return quantized.tobytes()


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 ASCII hexadecimal characters")
    return value.lower()
