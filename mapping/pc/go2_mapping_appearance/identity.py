"""Canonical identities for derived historical measured surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$", re.ASCII)


@dataclass(frozen=True)
class MeasuredSurfaceIdentity:
    revision_id: str
    core: Mapping[str, Any]
    positions_sha256: str
    normals_sha256: str
    triangles_sha256: str


def measured_surface_identity(
    *,
    positions: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    pose_generation: str,
    artifact_sha256: Sequence[str],
    fusion_config_sha256: str,
) -> MeasuredSurfaceIdentity:
    """Hash one exact measured-surface vertex and triangle ordering."""

    position_array = _float_xyz(positions, "positions")
    normal_array = _float_xyz(normals, "normals")
    if normal_array.shape != position_array.shape:
        raise ValueError("positions and normals must have the same shape")
    triangle_array = _triangles(triangles, len(position_array))
    pose_hash = _sha256(pose_generation, "pose_generation")
    fusion_hash = _sha256(fusion_config_sha256, "fusion_config_sha256")
    artifacts = tuple(sorted(_sha256(value, "artifact_sha256") for value in artifact_sha256))
    if not artifacts:
        raise ValueError("artifact_sha256 must contain at least one digest")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("artifact_sha256 must not contain duplicate digests")

    positions_hash = _payload_sha256(position_array)
    normals_hash = _payload_sha256(normal_array)
    triangles_hash = _payload_sha256(triangle_array)
    core: dict[str, Any] = {
        "schema_version": 1,
        "source_layer": "LIDAR_MEASURED",
        "pose_generation": pose_hash,
        "artifact_sha256": list(artifacts),
        "fusion_config_sha256": fusion_hash,
        "vertex_count": int(position_array.shape[0]),
        "triangle_count": int(triangle_array.shape[0]),
        "positions_sha256": positions_hash,
        "normals_sha256": normals_hash,
        "triangles_sha256": triangles_hash,
    }
    return MeasuredSurfaceIdentity(
        revision_id=canonical_sha256(core),
        core=core,
        positions_sha256=positions_hash,
        normals_sha256=normals_hash,
        triangles_sha256=triangles_hash,
    )


def _float_xyz(value: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError(f"{name} must be an N by 3 array")
    try:
        result = np.ascontiguousarray(source, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite float32 values")
    return result


def _triangles(value: np.ndarray, vertex_count: int) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("triangles must be an N by 3 array")
    if source.dtype.kind not in "iu":
        raise ValueError("triangles must contain integer triangle indices")
    if np.any(source < 0) or (source.size and int(np.max(source)) >= vertex_count):
        raise ValueError("triangle indices must reference existing vertices")
    if source.size and int(np.max(source)) > np.iinfo(np.uint32).max:
        raise ValueError("triangle indices exceed uint32 range")
    return np.ascontiguousarray(source, dtype="<u4")


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 ASCII hexadecimal characters")
    return value.lower()


def _payload_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
