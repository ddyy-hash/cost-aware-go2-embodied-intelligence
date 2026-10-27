"""Build a conservative Nav2 occupancy candidate from observed LIO points."""
from __future__ import annotations

from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .coarse_map import validate_coarse_map
from .nav2_map import FREE, OCCUPIED, UNKNOWN, _distance_to_segment, _point_in_polygon, _polygons, _lines


def build_observed_nav2_map(
    coarse_payload: dict[str, Any],
    points_xyz: np.ndarray,
    *,
    resolution: float = 0.05,
    padding: float = 1.0,
    wall_thickness: float = 0.12,
    floor_min_z: float = -0.25,
    floor_max_z: float = 0.35,
    obstacle_min_z: float = 0.12,
    max_cells: int = 4_000_000,
) -> dict[str, Any]:
    coarse = validate_coarse_map(coarse_payload)
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("observed points must be a finite Nx3 array")
    resolution = float(resolution)
    padding = float(padding)
    if not isfinite(resolution) or resolution <= 0 or not isfinite(padding) or padding < 0:
        raise ValueError("resolution and padding are invalid")
    spaces = _polygons(coarse, "rooms") + _polygons(coarse, "corridors")
    keep_outs = _polygons(coarse, "keep_out_zones")
    lines = _lines(coarse)
    xy = points[:, :2]
    map_points = [tuple(point) for polygon in spaces + keep_outs for point in polygon]
    map_points.extend(tuple(point) for line in lines for point in line)
    map_points.extend((float(x), float(y)) for x, y in xy)
    min_x = min(point[0] for point in map_points) - padding
    min_y = min(point[1] for point in map_points) - padding
    max_x = max(point[0] for point in map_points) + padding
    max_y = max(point[1] for point in map_points) + padding
    width = max(1, int(ceil((max_x - min_x) / resolution)))
    height = max(1, int(ceil((max_y - min_y) / resolution)))
    if width * height > max_cells:
        raise ValueError("observed Nav2 raster exceeds max_cells")
    grid = [[UNKNOWN for _ in range(width)] for _ in range(height)]
    for row in range(height):
        y = max_y - (row + 0.5) * resolution
        for column in range(width):
            x = min_x + (column + 0.5) * resolution
            cell = (x, y)
            if any(_point_in_polygon(cell, polygon) for polygon in spaces):
                grid[row][column] = FREE
            if any(_point_in_polygon(cell, polygon) for polygon in keep_outs):
                grid[row][column] = OCCUPIED
            if any(_distance_to_segment(cell, start, end) <= wall_thickness / 2 for line in lines for start, end in zip(line, line[1:])):
                grid[row][column] = OCCUPIED
    observed = points[(points[:, 2] >= floor_min_z) & (points[:, 2] <= floor_max_z)]
    obstacles = points[points[:, 2] >= obstacle_min_z]
    for point in obstacles:
        column = int((float(point[0]) - min_x) / resolution)
        row = int((max_y - float(point[1])) / resolution)
        if 0 <= row < height and 0 <= column < width:
            grid[row][column] = OCCUPIED
    metadata = {
        "schema_version": 1,
        "map_id": coarse["map_id"],
        "coordinate_frame": coarse["coordinate_frame"],
        "source_mode": "RECORDED_LIO_SESSION_ODOM_CLOUD",
        "provenance": "OBSERVED_LIDAR_PREVIEW",
        "geometry_authority": False,
        "runtime_authority": False,
        "source_revision": coarse["revision"],
        "resolution": resolution,
        "origin": [round(min_x, 6), round(min_y, 6), 0.0],
        "width": width,
        "height": height,
        "unknown_value": UNKNOWN,
        "free_value": FREE,
        "occupied_value": OCCUPIED,
        "observed_point_count": int(len(points)),
        "floor_point_count": int(len(observed)),
        "obstacle_point_count": int(len(obstacles)),
    }
    return {"metadata": metadata, "grid": grid, "width": width, "height": height}


__all__ = ["build_observed_nav2_map"]
