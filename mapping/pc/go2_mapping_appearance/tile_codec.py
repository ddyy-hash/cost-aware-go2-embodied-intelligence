"""Backward-compatible read-only decoder for immutable G2TM mesh tiles."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


_HEADER = struct.Struct("<4sIIII")


@dataclass(frozen=True)
class DecodedTileV1:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    binary_schema_version: int = 1
    triangle_provenance: np.ndarray | None = None
    triangle_uncertainty_m: np.ndarray | None = None


def decode_g2tm_v1(body: bytes) -> DecodedTileV1:
    payload = bytes(body)
    if len(payload) < _HEADER.size:
        raise ValueError("G2TM body is truncated")
    magic, version, vertex_count, triangle_count, scale_count = _HEADER.unpack_from(payload)
    if magic != b"G2TM" or version not in {1, 2}:
        raise ValueError("G2TM header is invalid or unsupported")
    if scale_count != triangle_count:
        raise ValueError("G2TM triangle scale count must match triangle count")
    expected = _HEADER.size + vertex_count * 24 + triangle_count * 12 + scale_count * 4
    if version == 2:
        expected += triangle_count * 4 + triangle_count
    if len(payload) != expected:
        raise ValueError("G2TM body length does not match its exact counts")
    offset = _HEADER.size
    positions = _array(payload, "<f4", (vertex_count, 3), offset)
    offset += vertex_count * 12
    normals = _array(payload, "<f4", (vertex_count, 3), offset)
    offset += vertex_count * 12
    triangles = _array(payload, "<u4", (triangle_count, 3), offset)
    offset += triangle_count * 12
    scales = _array(payload, "<f4", (scale_count,), offset)
    offset += scale_count * 4
    if version == 2:
        uncertainty = _array(payload, "<f4", (triangle_count,), offset)
        offset += triangle_count * 4
        provenance = _array(payload, "u1", (triangle_count,), offset)
    else:
        uncertainty = scales.copy()
        uncertainty.setflags(write=False)
        provenance = np.full(triangle_count, 1, dtype="u1")
        provenance.setflags(write=False)
    if not np.all(np.isfinite(positions)):
        raise ValueError("G2TM positions must be finite")
    _require_unit_normals(normals, "G2TM normals")
    if triangle_count and int(np.max(triangles)) >= vertex_count:
        raise ValueError("G2TM triangle index is out of range")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("G2TM triangle scales must be finite and positive")
    if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0) or np.any(uncertainty > 1.0):
        raise ValueError("G2TM triangle uncertainty must be finite and bounded")
    if np.any(~np.isin(provenance, (1, 2, 3))):
        raise ValueError("G2TM triangle provenance is invalid")
    return DecodedTileV1(positions, normals, triangles, scales, version, provenance, uncertainty)


def _array(payload: bytes, dtype: str, shape: tuple[int, ...], offset: int) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    result = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
    result.setflags(write=False)
    return result


def _require_unit_normals(normals: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(normals)):
        raise ValueError(f"{name} must be finite unit vectors")
    lengths = np.linalg.norm(np.asarray(normals, dtype=np.float64), axis=1)
    if len(lengths) and not np.allclose(lengths, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must be unit vectors")
