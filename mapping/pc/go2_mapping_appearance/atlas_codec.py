"""Binary V1 codecs for immutable tile render geometry and UV sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from typing import Sequence

import numpy as np

from .tile_codec import _require_unit_normals


_HEADER = struct.Struct("<8sHHI32s32s")
_HEADER_BYTES = 80
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_VERTEX_DTYPE = np.dtype([("position", "<f4", (3,)), ("normal", "<f4", (3,)), ("source", "<u4")])
_INDEX_DTYPE = np.dtype([("triangle", "<u4", (3,)), ("source", "<u4")])
_UV_DTYPE = np.dtype([("uv", "<f4", (2,)), ("atlas", "<u2"), ("flags", "<u2")])


@dataclass(frozen=True)
class RenderVerticesV1:
    positions: np.ndarray
    normals: np.ndarray
    source_vertex_ids: np.ndarray


@dataclass(frozen=True)
class RenderIndicesV1:
    triangles: np.ndarray
    source_triangle_ids: np.ndarray


@dataclass(frozen=True)
class RenderUvsV1:
    uvs: np.ndarray
    atlas_indices: np.ndarray
    flags: np.ndarray


def encode_render_vertices_v1(
    tile_revision_id: str,
    appearance_revision_id: str,
    positions: np.ndarray,
    normals: np.ndarray,
    source_vertex_ids: Sequence[int] | np.ndarray,
) -> bytes:
    points = _float_array(positions, (-1, 3), "render positions")
    normal_array = _float_array(normals, (-1, 3), "render normals")
    if normal_array.shape != points.shape:
        raise ValueError("render normals must match render positions")
    _require_unit_normals(normal_array, "render normals")
    sources = _uint_array(source_vertex_ids, (len(points),), np.uint32, "source vertex IDs")
    records = np.empty(len(points), dtype=_VERTEX_DTYPE)
    records["position"] = points
    records["normal"] = normal_array
    records["source"] = sources
    return _header(b"G2RVERT1", len(records), tile_revision_id, appearance_revision_id) + records.tobytes()


def parse_render_vertices_v1(
    body: bytes,
    *,
    expected_tile_revision_id: str,
    expected_appearance_revision_id: str,
    source_vertex_count: int,
) -> RenderVerticesV1:
    records = _records(
        body,
        b"G2RVERT1",
        _VERTEX_DTYPE,
        expected_tile_revision_id,
        expected_appearance_revision_id,
    )
    positions = _readonly(records["position"], "<f4")
    normals = _readonly(records["normal"], "<f4")
    sources = _readonly(records["source"], "<u4")
    if not np.all(np.isfinite(positions)):
        raise ValueError("render positions must be finite")
    _require_unit_normals(normals, "render normals")
    _require_range(sources, source_vertex_count, "source vertex ID")
    return RenderVerticesV1(positions, normals, sources)


def encode_render_indices_v1(
    tile_revision_id: str,
    appearance_revision_id: str,
    triangles: np.ndarray,
    source_triangle_ids: Sequence[int] | np.ndarray,
) -> bytes:
    triangle_array = _uint_array(triangles, (-1, 3), np.uint32, "render triangles")
    sources = _uint_array(source_triangle_ids, (len(triangle_array),), np.uint32, "source triangle IDs")
    records = np.empty(len(triangle_array), dtype=_INDEX_DTYPE)
    records["triangle"] = triangle_array
    records["source"] = sources
    return _header(b"G2RIDX01", len(records), tile_revision_id, appearance_revision_id) + records.tobytes()


def parse_render_indices_v1(
    body: bytes,
    *,
    expected_tile_revision_id: str,
    expected_appearance_revision_id: str,
    render_vertex_count: int,
    source_triangle_count: int,
) -> RenderIndicesV1:
    records = _records(
        body,
        b"G2RIDX01",
        _INDEX_DTYPE,
        expected_tile_revision_id,
        expected_appearance_revision_id,
    )
    triangles = _readonly(records["triangle"], "<u4")
    sources = _readonly(records["source"], "<u4")
    _require_range(triangles.reshape(-1), render_vertex_count, "render vertex index")
    _require_range(sources, source_triangle_count, "source triangle ID")
    return RenderIndicesV1(triangles, sources)


def encode_render_uvs_v1(
    tile_revision_id: str,
    appearance_revision_id: str,
    uvs: np.ndarray,
    atlas_indices: Sequence[int] | np.ndarray,
    flags: Sequence[int] | np.ndarray,
) -> bytes:
    uv_array = _float_array(uvs, (-1, 2), "render UVs")
    atlases = _uint_array(atlas_indices, (len(uv_array),), np.uint16, "atlas indices")
    flag_array = _uint_array(flags, (len(uv_array),), np.uint16, "UV flags")
    _validate_uv_records(uv_array, atlases, flag_array, atlas_count=None)
    records = np.empty(len(uv_array), dtype=_UV_DTYPE)
    records["uv"] = uv_array
    records["atlas"] = atlases
    records["flags"] = flag_array
    return _header(b"G2RUV001", len(records), tile_revision_id, appearance_revision_id) + records.tobytes()


def parse_render_uvs_v1(
    body: bytes,
    *,
    expected_tile_revision_id: str,
    expected_appearance_revision_id: str,
    atlas_count: int,
) -> RenderUvsV1:
    records = _records(
        body,
        b"G2RUV001",
        _UV_DTYPE,
        expected_tile_revision_id,
        expected_appearance_revision_id,
    )
    uvs = _readonly(records["uv"], "<f4")
    atlases = _readonly(records["atlas"], "<u2")
    flags = _readonly(records["flags"], "<u2")
    _validate_uv_records(uvs, atlases, flags, atlas_count=atlas_count)
    return RenderUvsV1(uvs, atlases, flags)


def _header(magic: bytes, count: int, tile_revision_id: str, appearance_revision_id: str) -> bytes:
    tile = _sha256(tile_revision_id, "tile_revision_id")
    appearance = _sha256(appearance_revision_id, "appearance_revision_id")
    return _HEADER.pack(magic, 1, _HEADER_BYTES, int(count), bytes.fromhex(tile), bytes.fromhex(appearance))


def _records(
    body: bytes,
    magic: bytes,
    dtype: np.dtype,
    expected_tile_revision_id: str,
    expected_appearance_revision_id: str,
) -> np.ndarray:
    payload = bytes(body)
    if len(payload) < _HEADER_BYTES:
        raise ValueError("atlas binary header is truncated")
    actual_magic, version, header_bytes, count, tile_raw, appearance_raw = _HEADER.unpack_from(payload)
    if actual_magic != magic or version != 1 or header_bytes != _HEADER_BYTES:
        raise ValueError("atlas binary header is invalid or unsupported")
    expected_length = _HEADER_BYTES + count * dtype.itemsize
    if len(payload) != expected_length:
        raise ValueError("atlas binary length does not match its exact record count")
    if tile_raw.hex() != _sha256(expected_tile_revision_id, "expected_tile_revision_id"):
        raise ValueError("atlas binary tile revision identity mismatch")
    if appearance_raw.hex() != _sha256(expected_appearance_revision_id, "expected_appearance_revision_id"):
        raise ValueError("atlas binary appearance revision identity mismatch")
    return np.frombuffer(payload, dtype=dtype, count=count, offset=_HEADER_BYTES).copy()


def _float_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype="<f4")
    try:
        result = np.ascontiguousarray(result.reshape(shape), dtype="<f4")
    except ValueError as exc:
        raise ValueError(f"{name} have an invalid shape") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _uint_array(value: Sequence[int] | np.ndarray, shape: tuple[int, ...], dtype: type, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu" or np.any(source < 0):
        raise ValueError(f"{name} must contain non-negative integers")
    maximum = np.iinfo(dtype).max
    if source.size and int(np.max(source)) > maximum:
        raise ValueError(f"{name} exceed the binary range")
    try:
        return np.ascontiguousarray(source, dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape)
    except ValueError as exc:
        raise ValueError(f"{name} have an invalid shape") from exc


def _validate_uv_records(
    uvs: np.ndarray,
    atlas_indices: np.ndarray,
    flags: np.ndarray,
    *,
    atlas_count: int | None,
) -> None:
    if np.any(uvs < 0.0) or np.any(uvs > 1.0):
        raise ValueError("render UVs must be normalized to [0, 1]")
    if np.any(flags & np.uint16(0xFFFE)):
        raise ValueError("UV flags contain unsupported bits")
    textured = flags == 1
    if np.any(textured & (atlas_indices == 65535)) or np.any((~textured) & (atlas_indices != 65535)):
        raise ValueError("base-material sentinel and UV flags are inconsistent")
    if atlas_count is not None:
        if isinstance(atlas_count, bool) or not isinstance(atlas_count, int) or atlas_count < 0:
            raise ValueError("atlas_count must be a non-negative integer")
        if np.any(textured & (atlas_indices >= atlas_count)):
            raise ValueError("atlas index is out of range")


def _require_range(values: np.ndarray, count: int, name: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name} count is invalid")
    if values.size and int(np.max(values)) >= count:
        raise ValueError(f"{name} is out of range")


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _readonly(value: np.ndarray, dtype: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result
