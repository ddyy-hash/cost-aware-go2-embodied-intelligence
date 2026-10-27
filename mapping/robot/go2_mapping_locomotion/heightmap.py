"""Local Livox PointCloud2 to PGTT 11x9 elevation grid conversion."""
from __future__ import annotations

import math
import struct
from typing import Any, Iterable, Sequence

import numpy as np


class HeightmapError(ValueError):
    pass


def transform_lidar_points(
    points_xyz: Sequence[Sequence[Any]],
    transform_id: str,
    *,
    source_frame_id: str | None = None,
) -> np.ndarray:
    """Apply the versioned LiDAR-to-body transform selected by source frame."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise HeightmapError("points must be a finite (N,3) array")
    if transform_id == "AUTO_GO2_LIDAR_FRAME_v1":
        source = (source_frame_id or "").strip()
        if source in {"utlidar_lidar", "base_link"}:
            return points.copy()
        if source in {"go2_lidar_raw", "utlidar_lidar_raw"}:
            return points * np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
        raise HeightmapError("unknown source frame for automatic sensor transform")
    if transform_id == "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2":
        return points * np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
    if transform_id == "IDENTITY_SENSOR_FRAME_v1":
        return points.copy()
    raise HeightmapError("unknown sensor frame transform")


def _field_map(fields: Iterable[Any]) -> dict[str, tuple[int, int]]:
    result = {}
    for field in fields:
        result[str(field.name)] = (int(field.offset), int(field.datatype))
    for name in ("x", "y", "z"):
        if name not in result or result[name][1] != 7:
            raise HeightmapError("point cloud requires float32 x/y/z fields")
    return result


def pointcloud_xyz(message: Any) -> np.ndarray:
    """Decode finite float32 XYZ values without sensor_msgs_py dependency."""
    width, height, step = int(message.width), int(message.height), int(message.point_step)
    if width <= 0 or height <= 0 or step <= 0:
        raise HeightmapError("point cloud dimensions are invalid")
    raw = bytes(message.data)
    if len(raw) < width * height * step:
        raise HeightmapError("point cloud payload is truncated")
    fields = _field_map(message.fields)
    points = np.empty((width * height, 3), dtype=np.float32)
    for index in range(width * height):
        base = index * step
        points[index] = [
            struct.unpack_from("<f", raw, base + fields[name][0])[0]
            for name in ("x", "y", "z")
        ]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise HeightmapError("point cloud has no finite points")
    return points


def build_pgtt_heightmap(
    points_xyz: Sequence[Sequence[Any]],
    *,
    shape: tuple[int, int] = (11, 9),
    spacing_m: float = 0.1,
    sensor_height_m: float = 0.6,
) -> np.ndarray:
    """Build a finite robot-centred height map from local XYZ points.

    X is forward and Y is lateral. Empty bins use the nearest observed bin,
    then the conservative ground value, so the policy never receives NaN.
    """
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise HeightmapError("points must be a finite (N,3) array")
    if len(shape) != 2 or shape != (11, 9) or spacing_m <= 0 or not math.isfinite(spacing_m):
        raise HeightmapError("PGTT heightmap must be 11x9 with positive spacing")
    local = points[
        (points[:, 0] >= -0.5)
        & (points[:, 0] <= 0.5)
        & (points[:, 1] >= -0.4)
        & (points[:, 1] <= 0.4)
        & (points[:, 2] >= -1.0)
        & (points[:, 2] <= 2.0)
    ]
    if len(local) == 0:
        raise HeightmapError("point cloud has no points in the PGTT local window")
    output = np.full(shape, np.nan, dtype=np.float32)
    x_origin = -(shape[0] - 1) * spacing_m / 2.0
    y_origin = -(shape[1] - 1) * spacing_m / 2.0
    columns: dict[tuple[int, int], list[float]] = {}
    for x, y, z in local:
        row = int(round((float(x) - x_origin) / spacing_m))
        col = int(round((float(y) - y_origin) / spacing_m))
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            columns.setdefault((row, col), []).append(float(z))
    for (row, col), values in columns.items():
        output[row, col] = float(np.median(values))
    ground = float(np.median(local[:, 2]))
    output = np.where(np.isfinite(output), output, ground)
    output = (output - float(sensor_height_m)).astype(np.float32)
    if not np.isfinite(output).all():
        raise HeightmapError("generated PGTT heightmap is not finite")
    return output


def build_stair_probe_profile(points_xyz: Sequence[Sequence[Any]]) -> np.ndarray:
    """Build a 2 m forward, 0.1 m resolution profile for early stair detection."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise HeightmapError("points must be a finite (N,3) array")
    local = points[(points[:, 0] >= -0.5) & (points[:, 0] <= 1.5) &
                   (points[:, 1] >= -0.4) & (points[:, 1] <= 0.4) &
                   (points[:, 2] >= -1.0) & (points[:, 2] <= 2.0)]
    if len(local) == 0:
        raise HeightmapError("point cloud has no points in the stair probe window")
    output = np.full((21, 9), np.nan, dtype=np.float32)
    columns: dict[tuple[int, int], list[float]] = {}
    for x, y, z in local:
        row = int(round((float(x) + 0.5) / 0.1))
        col = int(round((float(y) + 0.4) / 0.1))
        if 0 <= row < 21 and 0 <= col < 9:
            columns.setdefault((row, col), []).append(float(z))
    # LiDAR returns from a tread are often sparse compared with the ground
    # returns in the same bin.  A plain median therefore erases the tread.
    # Keep a high cluster only when it has enough support to be more than a
    # single-point outlier; row-wise medians below still reject isolated bins.
    ground = float(np.percentile(local[:, 2], 35.0))
    for key, values in columns.items():
        samples = np.asarray(values, dtype=np.float32)
        elevated = samples[samples >= ground + 0.045]
        minimum_support = max(2, int(np.ceil(samples.size * 0.08)))
        if elevated.size >= minimum_support:
            output[key] = float(np.median(elevated))
        else:
            output[key] = float(np.median(samples))
    return np.where(np.isfinite(output), output, ground).astype(np.float32) - 0.6


__all__ = ["HeightmapError", "build_pgtt_heightmap", "build_stair_probe_profile", "pointcloud_xyz", "transform_lidar_points"]
