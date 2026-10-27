"""Convert an operator coarse-map prior into a conservative Nav2 occupancy map."""
from __future__ import annotations

from math import ceil, isfinite
import json
from pathlib import Path
import os
import tempfile
from typing import Any

from .coarse_map import validate_coarse_map


UNKNOWN = 205
FREE = 254
OCCUPIED = 0


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] point")
    return _number(value[0], f"{label}.x"), _number(value[1], f"{label}.y")


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        if (y0 > y) != (y1 > y):
            crossing_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < crossing_x:
                inside = not inside
    return inside


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
    projection = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_squared))
    closest = (sx + projection * dx, sy + projection * dy)
    return ((px - closest[0]) ** 2 + (py - closest[1]) ** 2) ** 0.5


def _polygons(coarse: dict[str, Any], collection: str) -> list[list[tuple[float, float]]]:
    result = []
    for index, entity in enumerate(coarse.get(collection, [])):
        points = entity.get("polygon") if isinstance(entity, dict) else None
        if points is None:
            points = entity.get("geometry") if isinstance(entity, dict) else None
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"{collection}[{index}] has no valid polygon")
        result.append([_point(point, f"{collection}[{index}]") for point in points])
    return result


def _lines(coarse: dict[str, Any]) -> list[list[tuple[float, float]]]:
    result = []
    for index, entity in enumerate(coarse.get("walls", [])):
        points = entity.get("geometry") if isinstance(entity, dict) else None
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"walls[{index}] has no valid line")
        result.append([_point(point, f"walls[{index}]") for point in points])
    return result


def build_nav2_occupancy_map(
    payload: dict[str, Any],
    *,
    resolution: float = 0.05,
    padding: float = 1.0,
    wall_thickness: float = 0.12,
    max_cells: int = 4_000_000,
) -> dict[str, Any]:
    coarse = validate_coarse_map(payload)
    if coarse.get("schema_version") != 2:
        raise ValueError("Nav2 export requires coarse map schema_version 2")
    resolution = _number(resolution, "resolution")
    padding = _number(padding, "padding")
    wall_thickness = _number(wall_thickness, "wall_thickness")
    if resolution <= 0 or padding < 0 or wall_thickness <= 0:
        raise ValueError("resolution, padding, and wall_thickness must be positive")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")

    spaces = _polygons(coarse, "rooms") + _polygons(coarse, "corridors")
    keep_outs = _polygons(coarse, "keep_out_zones")
    lines = _lines(coarse)
    points = [point for polygon in spaces + keep_outs for point in polygon]
    points.extend(point for line in lines for point in line)
    for index, entity in enumerate(coarse.get("doors", []) + coarse.get("features", []) + coarse.get("anchors", [])):
        position = entity.get("position") if isinstance(entity, dict) else None
        if position is not None:
            points.append(_point(position, f"feature[{index}].position"))
    if not points:
        raise ValueError("coarse map must contain a room, corridor, wall, or feature")

    min_x = min(point[0] for point in points) - padding
    min_y = min(point[1] for point in points) - padding
    max_x = max(point[0] for point in points) + padding
    max_y = max(point[1] for point in points) + padding
    width = max(1, int(ceil((max_x - min_x) / resolution)))
    height = max(1, int(ceil((max_y - min_y) / resolution)))
    if width * height > max_cells:
        raise ValueError(f"raster has {width * height} cells; max_cells is {max_cells}")

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
            if any(
                _distance_to_segment(cell, start, end) <= wall_thickness / 2
                for line in lines
                for start, end in zip(line, line[1:])
            ):
                grid[row][column] = OCCUPIED

    metadata = {
        "schema_version": 1,
        "map_id": coarse["map_id"],
        "coordinate_frame": coarse["coordinate_frame"],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "source_revision": coarse["revision"],
        "resolution": resolution,
        "origin": [round(min_x, 6), round(min_y, 6), 0.0],
        "width": width,
        "height": height,
        "unknown_value": UNKNOWN,
        "free_value": FREE,
        "occupied_value": OCCUPIED,
    }
    return {"metadata": metadata, "grid": grid, "width": width, "height": height}


def _write_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_nav2_map(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    resolution: float = 0.05,
    padding: float = 1.0,
    wall_thickness: float = 0.12,
    max_cells: int = 4_000_000,
) -> dict[str, Any]:
    result = build_nav2_occupancy_map(
        payload,
        resolution=resolution,
        padding=padding,
        wall_thickness=wall_thickness,
        max_cells=max_cells,
    )
    output = Path(output_dir).resolve(strict=False)
    pgm_path = output / "map.pgm"
    yaml_path = output / "map.yaml"
    metadata_path = output / "map.provenance.json"
    metadata = result["metadata"]
    pgm = f"P5\n{result['width']} {result['height']}\n255\n".encode("ascii")
    pgm += b"".join(bytes(row) for row in result["grid"])
    yaml = (
        "image: map.pgm\n"
        f"resolution: {metadata['resolution']:.9g}\n"
        f"origin: [{metadata['origin'][0]:.9g}, {metadata['origin'][1]:.9g}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
        "mode: trinary\n"
    ).encode("ascii")
    provenance = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _write_atomic(pgm_path, pgm)
    _write_atomic(yaml_path, yaml)
    _write_atomic(metadata_path, provenance)
    return {"pgm_path": pgm_path, "yaml_path": yaml_path, "metadata_path": metadata_path, **result["metadata"]}


__all__ = ["build_nav2_occupancy_map", "write_nav2_map"]
