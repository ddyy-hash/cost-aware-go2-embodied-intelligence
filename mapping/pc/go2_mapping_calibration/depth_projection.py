"""Native depth projection with explicit frame and encoding contracts."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def image_to_depth_m(image: np.ndarray, encoding: str, scale_m: float) -> np.ndarray:
    if encoding == "16UC1":
        value = np.asarray(image, dtype=np.float64) * float(scale_m)
    elif encoding == "32FC1":
        value = np.asarray(image, dtype=np.float64)
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    if value.ndim != 2 or not np.all(np.isfinite(value[value > 0.0])):
        raise ValueError("depth image must be a finite two-dimensional array")
    if not math.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("depth scale must be finite and positive")
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def project_lidar_to_depth(
    points_lidar: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    depth_from_lidar: np.ndarray,
    *,
    depth_scale_m: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[int, int], ...]]:
    points = np.asarray(points_lidar, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    camera = np.asarray(camera_matrix, dtype=np.float64)
    transform = np.asarray(depth_from_lidar, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_lidar must be a finite N by 3 array")
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth_m must be a non-empty two-dimensional array")
    if camera.shape != (3, 3) or camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
        raise ValueError("camera_matrix is unusable")
    if transform.shape != (4, 4) or not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-9):
        raise ValueError("depth_from_lidar must be homogeneous")
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth scale must be finite and positive")
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    valid = transformed[:, 2] > 0.0
    transformed = transformed[valid]
    if len(transformed) == 0:
        return (), (), ()
    u = np.rint(camera[0, 0] * transformed[:, 0] / transformed[:, 2] + camera[0, 2]).astype(np.int64)
    v = np.rint(camera[1, 1] * transformed[:, 1] / transformed[:, 2] + camera[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    transformed, u, v = transformed[inside], u[inside], v[inside]
    if len(transformed) == 0:
        return (), (), ()
    # z-buffer: only the nearest LiDAR point is a valid surface candidate per pixel.
    order = np.argsort(transformed[:, 2], kind="stable")
    selected_pixels: set[int] = set()
    selected: list[int] = []
    width = depth.shape[1]
    for index in order:
        pixel = int(v[index] * width + u[index])
        if pixel not in selected_pixels:
            selected_pixels.add(pixel)
            selected.append(int(index))
    selected_array = np.asarray(selected, dtype=np.int64)
    transformed, u, v = transformed[selected_array], u[selected_array], v[selected_array]
    observed = depth[v, u]
    valid = np.isfinite(observed) & (observed > 0.0)
    observed = observed[valid]
    transformed = transformed[valid]
    pixels = tuple((int(row), int(column)) for row, column in zip(v[valid], u[valid]))
    return tuple(float(value) for value in observed), tuple(float(value) for value in transformed[:, 2]), pixels
