"""Deterministic RGBA8 atlas rasterization with binary authorization alpha."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RasterizedAtlas:
    png_bytes: bytes
    rgba: np.ndarray
    triangle_materials: tuple[str, ...]
    triangle_atlas_indices: np.ndarray


def rasterize_rgba8_atlas(
    uvs: np.ndarray,
    triangles: np.ndarray,
    triangle_linear_rgb: np.ndarray,
    triangle_authorized: Sequence[bool] | np.ndarray,
    *,
    width: int,
    height: int,
    padding_pixels: int = 0,
    mip_filter_radius_pixels: int = 0,
) -> RasterizedAtlas:
    width = _dimension(width, "width")
    height = _dimension(height, "height")
    padding = _nonnegative_integer(padding_pixels, "padding_pixels")
    mip_radius = _nonnegative_integer(mip_filter_radius_pixels, "mip_filter_radius_pixels")
    uv_array = np.asarray(uvs, dtype=np.float64)
    if uv_array.ndim != 2 or uv_array.shape[1:] != (2,) or not np.all(np.isfinite(uv_array)):
        raise ValueError("uvs must be a finite N by 2 array")
    if np.any(uv_array < 0.0) or np.any(uv_array > 1.0):
        raise ValueError("uvs must be normalized")
    source_triangles = np.asarray(triangles)
    if source_triangles.ndim != 2 or source_triangles.shape[1:] != (3,) or source_triangles.dtype.kind not in "iu":
        raise ValueError("triangles must be an integer N by 3 array")
    if np.any(source_triangles < 0) or (source_triangles.size and int(np.max(source_triangles)) >= len(uv_array)):
        raise ValueError("triangle index is out of range")
    source_triangles = np.asarray(source_triangles, dtype=np.uint32)
    colors = np.asarray(triangle_linear_rgb, dtype=np.float64)
    if colors.shape != (len(source_triangles), 3) or not np.all(np.isfinite(colors)) or np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("triangle colors must be normalized linear RGB")
    authorized = np.asarray(triangle_authorized)
    if authorized.shape != (len(source_triangles),) or authorized.dtype.kind != "b":
        authorized = np.asarray(triangle_authorized, dtype=bool)
        if authorized.shape != (len(source_triangles),):
            raise ValueError("triangle_authorized must align with triangles")

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    materials: list[str] = []
    atlas_indices: list[int] = []
    accepted_masks: list[np.ndarray] = []
    for triangle_id, triangle in enumerate(source_triangles):
        mask = _triangle_mask(uv_array[triangle], width, height)
        footprint_ok = _footprint_supported(mask, padding, mip_radius)
        accepted = bool(authorized[triangle_id] and np.any(mask) and footprint_ok)
        if accepted:
            srgb = _linear_to_srgb8(colors[triangle_id])
            rgba[mask, :3] = srgb
            rgba[mask, 3] = 255
            materials.append("VALIDATED_UV_ATLAS")
            atlas_indices.append(0)
            accepted_masks.append(mask)
        else:
            materials.append("BASE_MATERIAL")
            atlas_indices.append(65535)

    if padding:
        rgba = _dilate_rgb_preserve_alpha(rgba, padding)
    image = Image.fromarray(rgba, mode="RGBA")
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    immutable = np.ascontiguousarray(rgba)
    immutable.setflags(write=False)
    atlas_array = np.asarray(atlas_indices, dtype="<u2")
    atlas_array.setflags(write=False)
    return RasterizedAtlas(stream.getvalue(), immutable, tuple(materials), atlas_array)


def _triangle_mask(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.empty((3, 2), dtype=np.float64)
    points[:, 0] = uv[:, 0] * (width - 1)
    points[:, 1] = (1.0 - uv[:, 1]) * (height - 1)
    minimum_x = max(0, int(math.floor(float(np.min(points[:, 0])))))
    maximum_x = min(width - 1, int(math.ceil(float(np.max(points[:, 0])))))
    minimum_y = max(0, int(math.floor(float(np.min(points[:, 1])))))
    maximum_y = min(height - 1, int(math.ceil(float(np.max(points[:, 1])))))
    mask = np.zeros((height, width), dtype=bool)
    if minimum_x > maximum_x or minimum_y > maximum_y:
        return mask
    x_grid, y_grid = np.meshgrid(
        np.arange(minimum_x, maximum_x + 1, dtype=np.float64) + 0.5,
        np.arange(minimum_y, maximum_y + 1, dtype=np.float64) + 0.5,
    )
    a, b, c = points
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denominator)) <= 1e-12:
        return mask
    weight_a = ((b[1] - c[1]) * (x_grid - c[0]) + (c[0] - b[0]) * (y_grid - c[1])) / denominator
    weight_b = ((c[1] - a[1]) * (x_grid - c[0]) + (a[0] - c[0]) * (y_grid - c[1])) / denominator
    weight_c = 1.0 - weight_a - weight_b
    inside = (weight_a >= -1e-9) & (weight_b >= -1e-9) & (weight_c >= -1e-9)
    mask[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1] = inside
    return mask


def _footprint_supported(mask: np.ndarray, padding: int, radius: int) -> bool:
    if radius == 0:
        return True
    if padding < radius:
        return False
    y, x = np.nonzero(mask)
    if not len(x):
        return False
    return bool(
        int(np.min(x)) >= radius
        and int(np.min(y)) >= radius
        and int(np.max(x)) < mask.shape[1] - radius
        and int(np.max(y)) < mask.shape[0] - radius
    )


def _dilate_rgb_preserve_alpha(rgba: np.ndarray, iterations: int) -> np.ndarray:
    result = np.array(rgba, copy=True)
    frontier = result[..., 3] == 255
    for _ in range(iterations):
        new_pixels = np.zeros_like(frontier)
        rgb_sum = np.zeros(result.shape[:2] + (3,), dtype=np.uint16)
        counts = np.zeros(result.shape[:2], dtype=np.uint16)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            source_y = slice(max(0, -dy), result.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), result.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), result.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), result.shape[1] - max(0, -dx))
            valid = frontier[source_y, source_x]
            rgb_sum[target_y, target_x] += result[source_y, source_x, :3].astype(np.uint16) * valid[..., None]
            counts[target_y, target_x] += valid.astype(np.uint16)
        new_pixels = (counts > 0) & (~frontier) & (result[..., 3] == 0)
        result[new_pixels, :3] = (rgb_sum[new_pixels] // counts[new_pixels, None]).astype(np.uint8)
        frontier |= new_pixels
    return result


def _linear_to_srgb8(color: np.ndarray) -> np.ndarray:
    encoded = np.where(color <= 0.0031308, 12.92 * color, 1.055 * np.power(color, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def _dimension(value: int, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if not 1 <= result <= 4096:
        raise ValueError(f"{name} must be in [1, 4096]")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
