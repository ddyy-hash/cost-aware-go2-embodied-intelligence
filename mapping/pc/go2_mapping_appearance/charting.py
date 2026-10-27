"""Deterministic xatlas normalization with exact render-to-source mapping."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Iterable, Sequence

import numpy as np
import xatlas

from .tile_codec import _require_unit_normals


@dataclass(frozen=True)
class ChartedMeshV1:
    render_positions: np.ndarray
    render_normals: np.ndarray
    render_triangles: np.ndarray
    uvs: np.ndarray
    source_vertex_ids: np.ndarray
    source_triangle_ids: np.ndarray
    atlas_indices: np.ndarray
    flags: np.ndarray
    triangle_materials: tuple[str, ...]
    triangle_source_layers: tuple[str, ...]
    triangle_primitive_ids: np.ndarray
    charting_identity: dict


def chart_mesh_v1(
    positions: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    *,
    source_vertex_ids: Sequence[int] | np.ndarray,
    source_triangle_ids: Sequence[int] | np.ndarray,
    triangle_materials: Sequence[str],
    triangle_source_layers: Sequence[str],
    allowed_augmented_vertex_ids: Iterable[int] | None = None,
    allowed_augmented_triangle_ids: Iterable[int] | None = None,
) -> ChartedMeshV1:
    points = _float_xyz(positions, "positions")
    normal_array = _float_xyz(normals, "normals")
    if normal_array.shape != points.shape:
        raise ValueError("positions and normals must have matching shapes")
    _require_unit_normals(normal_array, "chart normals")
    triangle_array = _uint_array(triangles, (-1, 3), "triangles")
    if triangle_array.size and int(np.max(triangle_array)) >= len(points):
        raise ValueError("triangle index is out of range")
    vertex_sources = _uint_array(source_vertex_ids, (len(points),), "source vertex IDs")
    triangle_sources = _uint_array(source_triangle_ids, (len(triangle_array),), "source triangle IDs")
    if len(set(vertex_sources.tolist())) != len(vertex_sources):
        raise ValueError("source vertex IDs must be unique")
    if len(set(triangle_sources.tolist())) != len(triangle_sources):
        raise ValueError("source triangle IDs must be unique")
    materials = tuple(triangle_materials)
    layers = tuple(triangle_source_layers)
    if len(materials) != len(triangle_array) or len(layers) != len(triangle_array):
        raise ValueError("triangle material and source-layer arrays must align")
    if any(value not in {"VALIDATED_UV_ATLAS", "BASE_MATERIAL"} for value in materials):
        raise ValueError("triangle material is invalid")
    if any(value not in {"LIDAR_MEASURED", "RGBD_AUGMENTED"} for value in layers):
        raise ValueError("triangle source layer is invalid")
    _validate_augmentation_sources(
        triangle_array,
        vertex_sources,
        triangle_sources,
        layers,
        allowed_augmented_vertex_ids,
        allowed_augmented_triangle_ids,
    )

    order = np.argsort(triangle_sources, kind="stable")
    sorted_triangles = np.ascontiguousarray(triangle_array[order], dtype=np.uint32)
    sorted_source_triangles = triangle_sources[order]
    sorted_materials = tuple(materials[int(index)] for index in order)
    sorted_layers = tuple(layers[int(index)] for index in order)
    try:
        xatlas_sources, xatlas_triangles, xatlas_uvs = xatlas.parametrize(
            points,
            sorted_triangles,
            normal_array,
        )
    except Exception as exc:
        raise ValueError(f"xatlas charting failed: {exc}") from exc
    xatlas_sources = np.asarray(xatlas_sources, dtype=np.uint32).reshape(-1)
    xatlas_triangles = np.asarray(xatlas_triangles, dtype=np.uint32).reshape(-1, 3)
    xatlas_uvs = np.asarray(xatlas_uvs, dtype=np.float32).reshape(-1, 2)
    if len(xatlas_triangles) != len(sorted_triangles):
        raise ValueError("xatlas changed the source triangle count")
    if len(xatlas_sources) != len(xatlas_uvs) or (
        xatlas_triangles.size and int(np.max(xatlas_triangles)) >= len(xatlas_sources)
    ):
        raise ValueError("xatlas returned inconsistent render mappings")
    if np.any(xatlas_sources >= len(points)) or not np.all(np.isfinite(xatlas_uvs)):
        raise ValueError("xatlas returned invalid source or UV records")
    xatlas_uvs = np.clip(xatlas_uvs, 0.0, 1.0)

    primitive_keys: list[tuple[str, str, int]] = []
    primitive_by_key: dict[tuple[str, str, int], int] = {}
    for material, layer in zip(sorted_materials, sorted_layers):
        atlas_index = 0 if material == "VALIDATED_UV_ATLAS" else 65535
        key = (material, layer, atlas_index)
        if key not in primitive_by_key:
            primitive_by_key[key] = len(primitive_keys)
            primitive_keys.append(key)

    render_positions: list[np.ndarray] = []
    render_normals: list[np.ndarray] = []
    render_uvs: list[np.ndarray] = []
    render_source_ids: list[int] = []
    render_atlases: list[int] = []
    render_flags: list[int] = []
    normalized_triangles: list[tuple[int, int, int]] = []
    triangle_primitive_ids: list[int] = []
    render_index_by_key: dict[tuple[int, int, int, int], int] = {}
    quantization = 1_000_000

    for face_index, old_triangle in enumerate(xatlas_triangles):
        material = sorted_materials[face_index]
        layer = sorted_layers[face_index]
        atlas_index = 0 if material == "VALIDATED_UV_ATLAS" else 65535
        primitive_id = primitive_by_key[(material, layer, atlas_index)]
        new_triangle: list[int] = []
        for old_render_id in old_triangle:
            old_render = int(old_render_id)
            source_index = int(xatlas_sources[old_render])
            source_id = int(vertex_sources[source_index])
            uv = xatlas_uvs[old_render]
            key = (
                primitive_id,
                source_id,
                int(np.rint(float(uv[0]) * quantization)),
                int(np.rint(float(uv[1]) * quantization)),
            )
            render_id = render_index_by_key.get(key)
            if render_id is None:
                render_id = len(render_positions)
                render_index_by_key[key] = render_id
                render_positions.append(points[source_index])
                render_normals.append(normal_array[source_index])
                render_uvs.append(uv)
                render_source_ids.append(source_id)
                render_atlases.append(atlas_index)
                render_flags.append(1 if material == "VALIDATED_UV_ATLAS" else 0)
            new_triangle.append(render_id)
        normalized_triangles.append(tuple(new_triangle))
        triangle_primitive_ids.append(primitive_id)

    return ChartedMeshV1(
        render_positions=_readonly(render_positions, "<f4", (-1, 3)),
        render_normals=_readonly(render_normals, "<f4", (-1, 3)),
        render_triangles=_readonly(normalized_triangles, "<u4", (-1, 3)),
        uvs=_readonly(render_uvs, "<f4", (-1, 2)),
        source_vertex_ids=_readonly(render_source_ids, "<u4", (-1,)),
        source_triangle_ids=_readonly(sorted_source_triangles, "<u4", (-1,)),
        atlas_indices=_readonly(render_atlases, "<u2", (-1,)),
        flags=_readonly(render_flags, "<u2", (-1,)),
        triangle_materials=sorted_materials,
        triangle_source_layers=sorted_layers,
        triangle_primitive_ids=_readonly(triangle_primitive_ids, "<u4", (-1,)),
        charting_identity={
            "implementation": "xatlas",
            "version": version("xatlas"),
            "adapter": "go2-deterministic-chart-v1",
            "uv_quantization": quantization,
            "options": {"api": "parametrize", "source_triangle_order": "ascending-source-id"},
        },
    )


def _validate_augmentation_sources(
    triangles: np.ndarray,
    source_vertex_ids: np.ndarray,
    source_triangle_ids: np.ndarray,
    layers: Sequence[str],
    allowed_vertex_ids: Iterable[int] | None,
    allowed_triangle_ids: Iterable[int] | None,
) -> None:
    augmented_indices = [index for index, layer in enumerate(layers) if layer == "RGBD_AUGMENTED"]
    if not augmented_indices:
        return
    allowed_vertices = set() if allowed_vertex_ids is None else {int(value) for value in allowed_vertex_ids}
    allowed_triangles = set() if allowed_triangle_ids is None else {int(value) for value in allowed_triangle_ids}
    for index in augmented_indices:
        source_triangle = int(source_triangle_ids[index])
        source_vertices = {int(source_vertex_ids[value]) for value in triangles[index]}
        if source_triangle not in allowed_triangles or not source_vertices.issubset(allowed_vertices):
            raise ValueError("augmentation source IDs do not resolve against the exact revision")


def _float_xyz(value: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError(f"{name} must be an N by 3 array")
    result = np.ascontiguousarray(source, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _uint_array(value: Sequence[int] | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu" or np.any(source < 0):
        raise ValueError(f"{name} must contain non-negative integers")
    if source.size and int(np.max(source)) > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} exceed uint32")
    try:
        return np.ascontiguousarray(source, dtype=np.uint32).reshape(shape)
    except ValueError as exc:
        raise ValueError(f"{name} have an invalid shape") from exc


def _readonly(value: Sequence, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result
