"""Canonical identity for separately owned RGB-D augmentation geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$", re.ASCII)


@dataclass(frozen=True)
class RgbdAugmentedIdentity:
    revision_id: str
    core: Mapping[str, Any]
    positions_sha256: str
    normals_sha256: str
    triangles_sha256: str
    uncertainty_sha256: str
    source_vertex_ids_sha256: str
    source_triangle_ids_sha256: str


def rgbd_augmented_identity(
    *,
    source_generation_id: str,
    source_revision_id: str,
    calibration_sha256: str,
    physical_depth_validation_revision: str,
    model_revision_id: str | None,
    assistance_config_sha256: str,
    source_frame_sha256: Sequence[str],
    positions: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uncertainty_m: np.ndarray,
    source_vertex_ids: Sequence[str],
    source_triangle_ids: Sequence[str],
    algorithm_version: str = "rgbd-bounded-augmentation-v1",
    uncertainty_policy: str = "RSS_METRES_FLOAT32_V1",
) -> RgbdAugmentedIdentity:
    """Hash exact geometry, provenance, and source ordering for one augmentation."""

    position_array = _float_xyz(positions, "positions")
    normal_array = _float_xyz(normals, "normals")
    if normal_array.shape != position_array.shape:
        raise ValueError("positions and normals must have the same shape")
    triangle_array = _triangles(triangles, len(position_array))
    uncertainty_array = _uncertainty(uncertainty_m, len(position_array))
    vertex_ids = _source_ids(source_vertex_ids, len(position_array), "source_vertex_ids")
    triangle_ids = _source_ids(source_triangle_ids, len(triangle_array), "source_triangle_ids")
    frames = tuple(sorted(_sha256(value, "source_frame_sha256") for value in source_frame_sha256))
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("source_frame_sha256 must contain unique digests")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise ValueError("algorithm_version must be a non-empty string")
    if not isinstance(uncertainty_policy, str) or not uncertainty_policy:
        raise ValueError("uncertainty_policy must be a non-empty string")

    positions_hash = _payload_sha256(position_array)
    normals_hash = _payload_sha256(normal_array)
    triangles_hash = _payload_sha256(triangle_array)
    uncertainty_hash = _payload_sha256(uncertainty_array)
    vertex_ids_hash = hashlib.sha256(encode_source_ids(vertex_ids)).hexdigest()
    triangle_ids_hash = hashlib.sha256(encode_source_ids(triangle_ids)).hexdigest()
    core: dict[str, Any] = {
        "schema_version": 1,
        "source_layer": "RGBD_AUGMENTED",
        "geometry_authority": False,
        "algorithm_version": algorithm_version,
        "source_generation_id": _sha256(source_generation_id, "source_generation_id"),
        "source_revision_id": _sha256(source_revision_id, "source_revision_id"),
        "calibration_sha256": _sha256(calibration_sha256, "calibration_sha256"),
        "physical_depth_validation_revision": _sha256(
            physical_depth_validation_revision, "physical_depth_validation_revision"
        ),
        "model_revision_id": None
        if model_revision_id is None
        else _sha256(model_revision_id, "model_revision_id"),
        "assistance_config_sha256": _sha256(
            assistance_config_sha256, "assistance_config_sha256"
        ),
        "source_frame_sha256": list(frames),
        "vertex_count": int(len(position_array)),
        "triangle_count": int(len(triangle_array)),
        "uncertainty_policy": uncertainty_policy,
        "positions_sha256": positions_hash,
        "normals_sha256": normals_hash,
        "triangles_sha256": triangles_hash,
        "uncertainty_sha256": uncertainty_hash,
        "source_vertex_ids_sha256": vertex_ids_hash,
        "source_triangle_ids_sha256": triangle_ids_hash,
    }
    return RgbdAugmentedIdentity(
        revision_id=canonical_sha256(core),
        core=core,
        positions_sha256=positions_hash,
        normals_sha256=normals_hash,
        triangles_sha256=triangles_hash,
        uncertainty_sha256=uncertainty_hash,
        source_vertex_ids_sha256=vertex_ids_hash,
        source_triangle_ids_sha256=triangle_ids_hash,
    )


def encode_source_ids(values: Sequence[str]) -> bytes:
    """Encode ordered UTF-8 IDs with unsigned little-endian length prefixes."""

    encoded = bytearray(struct.pack("<I", len(values)))
    for value in values:
        payload = value.encode("utf-8")
        encoded.extend(struct.pack("<I", len(payload)))
        encoded.extend(payload)
    return bytes(encoded)


def _float_xyz(value: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError(f"{name} must be an N by 3 array")
    result = np.ascontiguousarray(source, dtype="<f4")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite float32 values")
    return result


def _triangles(value: np.ndarray, vertex_count: int) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1:] != (3,) or source.dtype.kind not in "iu":
        raise ValueError("triangles must be an N by 3 integer array")
    if np.any(source < 0) or (source.size and int(np.max(source)) >= vertex_count):
        raise ValueError("triangle indices must reference existing vertices")
    return np.ascontiguousarray(source, dtype="<u4")


def _uncertainty(value: np.ndarray, vertex_count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f4").reshape(-1)
    if len(result) != vertex_count or not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("uncertainty_m must contain one finite non-negative value per vertex")
    return result


def _source_ids(values: Sequence[str], count: int, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != count or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain one non-empty string per element")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 ASCII hexadecimal characters")
    return value.lower()


def _payload_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
