"""Read-only frontier candidate planning for operator-assisted exploration."""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Mapping, Sequence


UNKNOWN = 205
FREE = 254
OCCUPIED = 0
_SUPPORTED_FRAMES = {"SESSION_ODOM_LOCAL", "BUILDING_LOCAL"}
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT_CONNECTED = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _grid_cells(grid: Sequence[Sequence[Any]]) -> tuple[list[list[int]], int, int]:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or not grid:
        raise ValueError("frontier grid must be a non-empty rectangular grid")
    if not isinstance(grid[0], Sequence) or isinstance(grid[0], (str, bytes)) or not grid[0]:
        raise ValueError("frontier grid must be a non-empty rectangular grid")
    width = len(grid[0])
    normalized: list[list[int]] = []
    for row in grid:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ValueError("frontier grid must be rectangular")
        values: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError("frontier grid cell must be an integer from 0 to 255")
            values.append(value)
        normalized.append(values)
    return normalized, width, len(normalized)


def _neighbor(row: int, column: int, height: int, width: int, delta: tuple[int, int]) -> tuple[int, int] | None:
    next_row = row + delta[0]
    next_column = column + delta[1]
    if 0 <= next_row < height and 0 <= next_column < width:
        return next_row, next_column
    return None


def _world_point(
    row: int,
    column: int,
    *,
    height: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> tuple[float, float]:
    # The exporter stores row zero at the high-y edge of the PGM image.
    return (
        origin_x + (column + 0.5) * resolution,
        origin_y + (height - row - 0.5) * resolution,
    )


def plan_frontier_candidates(
    grid: Sequence[Sequence[Any]],
    *,
    origin: Sequence[Any] = (0.0, 0.0),
    resolution: float = 0.05,
    robot_pose: Mapping[str, Any],
    frame: str = "SESSION_ODOM_LOCAL",
    min_cluster_size: int = 1,
    max_candidates: int = 32,
) -> list[dict[str, Any]]:
    """Return ranked unknown-boundary candidates without changing navigation state."""

    cells, width, height = _grid_cells(grid)
    if not isinstance(origin, Sequence) or isinstance(origin, (str, bytes)) or len(origin) != 2:
        raise ValueError("frontier origin must be an [x, y] point")
    origin_x = _finite_number(origin[0], "frontier origin.x")
    origin_y = _finite_number(origin[1], "frontier origin.y")
    resolution_value = _finite_number(resolution, "frontier resolution")
    if resolution_value <= 0.0:
        raise ValueError("frontier resolution must be positive")
    if frame not in _SUPPORTED_FRAMES:
        raise ValueError(f"unsupported frontier frame: {frame}")
    if not isinstance(robot_pose, Mapping):
        raise ValueError("frontier robot_pose must be an object")
    robot_x = _finite_number(robot_pose.get("x"), "frontier robot_pose.x")
    robot_y = _finite_number(robot_pose.get("y"), "frontier robot_pose.y")
    if isinstance(min_cluster_size, bool) or not isinstance(min_cluster_size, int) or min_cluster_size <= 0:
        raise ValueError("frontier min_cluster_size must be positive")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("frontier max_candidates must be positive")

    frontier_cells: set[tuple[int, int]] = set()
    for row in range(height):
        for column in range(width):
            if cells[row][column] != FREE:
                continue
            if any(
                (neighbor := _neighbor(row, column, height, width, delta)) is not None
                and cells[neighbor[0]][neighbor[1]] == UNKNOWN
                for delta in _CARDINAL
            ):
                frontier_cells.add((row, column))

    clusters: list[list[tuple[int, int]]] = []
    unvisited = set(frontier_cells)
    while unvisited:
        seed = unvisited.pop()
        cluster = [seed]
        queue = deque([seed])
        while queue:
            row, column = queue.popleft()
            for delta in _EIGHT_CONNECTED:
                neighbor = _neighbor(row, column, height, width, delta)
                if neighbor is not None and neighbor in unvisited:
                    unvisited.remove(neighbor)
                    cluster.append(neighbor)
                    queue.append(neighbor)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    candidates: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        points = [
            _world_point(
                row,
                column,
                height=height,
                origin_x=origin_x,
                origin_y=origin_y,
                resolution=resolution_value,
            )
            for row, column in cluster
        ]
        centroid_x = sum(point[0] for point in points) / len(points)
        centroid_y = sum(point[1] for point in points) / len(points)
        unknown_neighbors: set[tuple[int, int]] = set()
        for row, column in cluster:
            for delta in _EIGHT_CONNECTED:
                neighbor = _neighbor(row, column, height, width, delta)
                if neighbor is not None and cells[neighbor[0]][neighbor[1]] == UNKNOWN:
                    unknown_neighbors.add(neighbor)
        distance = math.hypot(centroid_x - robot_x, centroid_y - robot_y)
        information_gain = max(len(cluster), len(unknown_neighbors))
        score = information_gain / max(distance, resolution_value * 0.5)
        candidates.append(
            {
                "candidate_id": f"frontier-{index:03d}",
                "frame": frame,
                "position": {
                    "x": round(centroid_x, 6),
                    "y": round(centroid_y, 6),
                    "yaw": 0.0,
                    "frame": frame,
                },
                "size_cells": len(cluster),
                "information_gain_cells": information_gain,
                "distance_m": round(distance, 6),
                "score": round(score, 6),
                "read_only": True,
                "goal_available": False,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    return candidates[:max_candidates]


__all__ = ["plan_frontier_candidates"]
