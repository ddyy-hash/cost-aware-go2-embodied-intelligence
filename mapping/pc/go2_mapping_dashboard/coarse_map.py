"""Versioned coarse-map priors and room verification tasks.

Hand-authored walls and equipment remain navigation priors. Robot observations
are attached to tasks and may later promote a room-local surface generation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any


_ALLOWED_ROOM_KINDS = {"room", "corridor", "stair", "landing", "elevator_hall"}
_ALLOWED_FEATURE_KINDS = {"door", "equipment", "obstacle", "anchor", "inspection_target"}
_ALLOWED_TASK_KINDS = {"perimeter", "corners", "feature", "floor_coverage"}
_ALLOWED_TASK_MODES = {"MANUAL", "ASSISTED", "AUTONOMOUS"}
_ALLOWED_VERTICAL_CONNECTION_KINDS = {"stairs", "elevator"}
_ALLOWED_STAIR_DIRECTIONS = {"up", "down", "bidirectional"}
FURNITURE_CATALOG = (
    {"id": "sofa", "label": "沙发", "category": "seating", "asset_id": "local/sofa-v1", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/unit-sofa.gltf", "dimensions_m": [2.0, 0.85, 0.8], "resizable": True, "rotatable": True},
    {"id": "table", "label": "桌子", "category": "table", "asset_id": "cc0/poly-pizza/rAEBvfb1FT", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/cc0/table-small-quaternius.glb", "asset_sha256": "fd3c98a8293913697b3942d74897263d27352bd1ec4513aa3489f6e1470e59f9", "license": "CC0-1.0", "dimensions_m": [1.4, 0.8, 0.75], "resizable": True, "rotatable": True},
    {"id": "bed", "label": "床", "category": "sleep", "asset_id": "cc0/poly-pizza/sn8az3odMR", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/cc0/bed-single-kenney.glb", "asset_sha256": "23c5c32096840f3dd347b1c68029d2f50e2af468ccd38daa33b494f1943b3dc8", "license": "CC0-1.0", "dimensions_m": [2.0, 1.6, 0.55], "resizable": True, "rotatable": True},
    {"id": "chair", "label": "椅子", "category": "seating", "asset_id": "cc0/poly-pizza/iMNqRzPwwe", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/cc0/chair-quaternius.glb", "asset_sha256": "bdc6aeeb64524bbdccba2f079e6485e2a372aed1c428d976d5da67aadcf18b7a", "license": "CC0-1.0", "dimensions_m": [0.55, 0.55, 0.9], "resizable": True, "rotatable": True},
    {"id": "cabinet", "label": "柜子", "category": "storage", "asset_id": "cc0/poly-pizza/wOiMrnUuhe", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/cc0/cabinet-creativetrio.glb", "asset_sha256": "e5226312183a51c5027f6e6c2e46873c0b0a7b3c9b4ff334c02cb03954b1b944", "license": "CC0-1.0", "dimensions_m": [1.2, 0.45, 1.8], "resizable": True, "rotatable": True},
    {"id": "plant", "label": "绿植", "category": "decor", "asset_id": "cc0/poly-pizza/bfLOqIV5uP", "asset_source": "LOCAL_GLTF", "asset_url": "/static/assets/furniture/cc0/houseplant-quaternius.glb", "asset_sha256": "0de1daa8be12b908fb8d1bf6bd9261e2a50061783e2ebe204bfb434768ca7c53", "license": "CC0-1.0", "dimensions_m": [0.45, 0.45, 1.2], "resizable": True, "rotatable": True},
)
_FURNITURE_ASSET_IDS = {item["asset_id"] for item in FURNITURE_CATALOG} | {"procedural/sofa", "procedural/table", "procedural/bed", "procedural/chair", "procedural/cabinet", "procedural/plant"}
COARSE_MAP_TEMPLATES = (
    {
        "id": "single-room",
        "label": "单房间",
        "description": "6m x 4m 空房间，适合快速单房测试。",
        "rooms": [{"id": "room-1", "name": "功能区 1", "kind": "room", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]]}],
        "corridors": [],
        "doors": [{"id": "door-1", "label": "入口", "kind": "door", "position": [3, 0], "room_id": "room-1"}],
    },
    {
        "id": "office-floor",
        "label": "办公楼层",
        "description": "中央走廊与四个办公功能区。",
        "rooms": [
            {"id": "office-101", "name": "办公室 101", "kind": "room", "polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]},
            {"id": "office-102", "name": "办公室 102", "kind": "room", "polygon": [[5, 0], [10, 0], [10, 4], [5, 4]]},
            {"id": "office-103", "name": "办公室 103", "kind": "room", "polygon": [[0, 6], [5, 6], [5, 10], [0, 10]]},
            {"id": "office-104", "name": "办公室 104", "kind": "room", "polygon": [[5, 6], [10, 6], [10, 10], [5, 10]]},
        ],
        "corridors": [{"id": "corridor-1", "name": "中央走廊", "kind": "corridor", "polygon": [[0, 4], [10, 4], [10, 6], [0, 6]]}],
        "doors": [],
    },
    {
        "id": "teaching-floor",
        "label": "教学楼层",
        "description": "两间教室、公共走廊、楼梯间与平台。",
        "rooms": [
            {"id": "classroom-a", "name": "教室 A", "kind": "room", "polygon": [[0, 0], [7, 0], [7, 5], [0, 5]]},
            {"id": "classroom-b", "name": "教室 B", "kind": "room", "polygon": [[7, 0], [14, 0], [14, 5], [7, 5]]},
            {"id": "stairs-1", "name": "主楼梯", "kind": "stair", "polygon": [[10, 7], [14, 7], [14, 11], [10, 11]], "stair_profile": {"step_count": 16, "riser_height_m": 0.18, "tread_depth_m": 0.28, "width_m": 1.2, "direction": "bidirectional"}, "entry": [12, 7], "exit": [12, 11]},
            {"id": "landing-1", "name": "楼梯平台", "kind": "landing", "polygon": [[8, 7], [10, 7], [10, 11], [8, 11]]},
        ],
        "corridors": [{"id": "corridor-1", "name": "公共走廊", "kind": "corridor", "polygon": [[0, 5], [14, 5], [14, 7], [0, 7]]}],
        "doors": [],
    },
)
_TASK_DEFAULT_MODES = {
    "perimeter": "ASSISTED",
    "corners": "ASSISTED",
    "feature": "MANUAL",
    "floor_coverage": "ASSISTED",
}
_MISSING = object()


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _point(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] point")
    return [_number(value[0], f"{label}.x"), _number(value[1], f"{label}.y")]


def _stair_profile(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    step_count = value.get("step_count")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 1 or step_count > 500:
        raise ValueError(f"{label}.step_count must be an integer between 1 and 500")
    direction = value.get("direction", "bidirectional")
    if direction not in _ALLOWED_STAIR_DIRECTIONS:
        raise ValueError(f"{label}.direction is unsupported")
    profile = {
        "step_count": step_count,
        "riser_height_m": round(_number(value.get("riser_height_m"), f"{label}.riser_height_m"), 4),
        "tread_depth_m": round(_number(value.get("tread_depth_m"), f"{label}.tread_depth_m"), 4),
        "width_m": round(_number(value.get("width_m"), f"{label}.width_m"), 4),
        "direction": direction,
    }
    if any(profile[key] <= 0 for key in ("riser_height_m", "tread_depth_m", "width_m")):
        raise ValueError(f"{label} dimensions must be positive")
    if profile["riser_height_m"] > 0.5 or profile["tread_depth_m"] > 1.5 or profile["width_m"] > 10:
        raise ValueError(f"{label} dimensions exceed supported coarse-map bounds")
    return profile


def _optional_point(value: Any, label: str) -> list[float] | None:
    return None if value is None else _point(value, label)


def _polygon(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{label}.polygon must contain at least three points")
    points = [_point(item, f"{label}.polygon[{index}]") for index, item in enumerate(value)]
    area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2.0)
    if area <= 1e-6:
        raise ValueError(f"{label}.polygon must enclose a positive area")
    return points


def _polygon_area(points: list[list[float]]) -> float:
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2.0)


def _polygon_perimeter(points: list[list[float]]) -> float:
    return sum(math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        if min(y0, y1) <= y <= max(y0, y1) and min(x0, x1) <= x <= max(x0, x1):
            if abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) <= 1e-8:
                return True
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _axis_aligned_rectangle(points: list[list[float]]) -> tuple[float, float, float, float]:
    normalized = _polygon(points, "function zone")
    xs = sorted({point[0] for point in normalized})
    ys = sorted({point[1] for point in normalized})
    if len(normalized) != 4 or len(xs) != 2 or len(ys) != 2:
        raise ValueError("function zone must be an axis-aligned rectangle")
    expected = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])}
    if {tuple(point) for point in normalized} != expected:
        raise ValueError("function zone must be an axis-aligned rectangle")
    return xs[0], ys[0], xs[1], ys[1]


def split_room_polygon(points: list[list[float]], *, direction: str) -> tuple[list[list[float]], list[list[float]]]:
    """Split any valid polygon through the center of its bounding box."""

    polygon = _polygon(points, "function zone")
    if direction not in {"vertical", "horizontal"}:
        raise ValueError("split direction must be vertical or horizontal")
    axis = 0 if direction == "vertical" else 1
    middle = (min(point[axis] for point in polygon) + max(point[axis] for point in polygon)) / 2.0

    def clip(keep_less: bool) -> list[list[float]]:
        result: list[list[float]] = []
        for index, current in enumerate(polygon):
            previous = polygon[index - 1]
            current_value = current[axis] - middle
            previous_value = previous[axis] - middle
            current_inside = current_value <= 1e-9 if keep_less else current_value >= -1e-9
            previous_inside = previous_value <= 1e-9 if keep_less else previous_value >= -1e-9
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                ratio = previous_value / denominator if abs(denominator) > 1e-12 else 0.0
                result.append([previous[0] + ratio * (current[0] - previous[0]), previous[1] + ratio * (current[1] - previous[1])])
            if current_inside:
                result.append(list(current))
        deduped: list[list[float]] = []
        for point in result:
            if not deduped or not math.isclose(point[0], deduped[-1][0], abs_tol=1e-9) or not math.isclose(point[1], deduped[-1][1], abs_tol=1e-9):
                deduped.append(point)
        if len(deduped) > 1 and math.isclose(deduped[0][0], deduped[-1][0], abs_tol=1e-9) and math.isclose(deduped[0][1], deduped[-1][1], abs_tol=1e-9):
            deduped.pop()
        if len(deduped) < 3 or _polygon_area(deduped) <= 1e-6:
            raise ValueError("function zone cannot be split into two non-empty polygons")
        return deduped

    return clip(True), clip(False)


def merge_room_polygons(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    """Merge polygons sharing an exact complete edge, with rectangle fallback."""

    first_polygon = _polygon(first, "first function zone")
    second_polygon = _polygon(second, "second function zone")
    def same(a: list[float], b: list[float]) -> bool:
        return math.isclose(a[0], b[0], abs_tol=1e-8) and math.isclose(a[1], b[1], abs_tol=1e-8)
    shared: tuple[int, int] | None = None
    for i, start in enumerate(first_polygon):
        end = first_polygon[(i + 1) % len(first_polygon)]
        for j, other_start in enumerate(second_polygon):
            other_end = second_polygon[(j + 1) % len(second_polygon)]
            if same(start, other_end) and same(end, other_start):
                shared = (i, j)
                break
        if shared:
            break
    if shared is not None:
        i, j = shared
        edges = []
        for polygon in (first_polygon, second_polygon):
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                if (polygon is first_polygon and index == i) or (polygon is second_polygon and index == j):
                    continue
                edges.append((list(start), list(end)))
        merged = [edges[0][0]]
        current = edges[0][1]
        remaining = edges[1:]
        while remaining and not same(current, merged[0]):
            merged.append(current)
            match = next((index for index, edge in enumerate(remaining) if same(edge[0], current)), None)
            if match is None:
                break
            current = remaining.pop(match)[1]
        if same(current, merged[0]) and len(merged) >= 3 and _polygon_area(merged) > 1e-6:
            simplified: list[list[float]] = []
            for point in merged:
                simplified.append(point)
                while len(simplified) >= 3:
                    a, b, c = simplified[-3:]
                    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                    if abs(cross) > 1e-8:
                        break
                    simplified.pop(-2)
            if len(simplified) >= 3 and _polygon_area(simplified) > 1e-6:
                return simplified

    ax0, ay0, ax1, ay1 = _axis_aligned_rectangle(first_polygon)
    bx0, by0, bx1, by1 = _axis_aligned_rectangle(second_polygon)
    if math.isclose(ay0, by0) and math.isclose(ay1, by1) and (math.isclose(ax1, bx0) or math.isclose(bx1, ax0)):
        return [[min(ax0, bx0), ay0], [max(ax1, bx1), ay0], [max(ax1, bx1), ay1], [min(ax0, bx0), ay1]]
    if math.isclose(ax0, bx0) and math.isclose(ax1, bx1) and (math.isclose(ay1, by0) or math.isclose(by1, ay0)):
        return [[ax0, min(ay0, by0)], [ax1, min(ay0, by0)], [ax1, max(ay1, by1)], [ax0, max(ay1, by1)]]
    raise ValueError("function zones must share a complete edge")


def _default_task_mode(kind: str) -> str:
    return _TASK_DEFAULT_MODES.get(kind, "ASSISTED")


def _task_mode(value: Any, label: str) -> str:
    if not isinstance(value, str) or value not in _ALLOWED_TASK_MODES:
        raise ValueError(f"{label} is unsupported")
    return value


def _task_lease_id(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    lease_id = value.strip()
    if not lease_id:
        return None
    if len(lease_id) > 128:
        raise ValueError(f"{label} is invalid")
    return lease_id


def assess_corner_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Turn a short wall-corner replay into a conservative task decision."""

    if not isinstance(diagnostics, dict):
        raise ValueError("corner diagnostics must be an object")
    p95 = _number(diagnostics.get("point_to_trajectory_p95_m"), "point_to_trajectory_p95_m")
    plane = _number(diagnostics.get("plane_residual_p95_m"), "plane_residual_p95_m")
    vertical = int(diagnostics.get("vertical_support_count", 0))
    horizontal = int(diagnostics.get("horizontal_support_count", 0))
    if vertical < 20 or horizontal < 20:
        state = "QUARANTINED"
        reason = "INSUFFICIENT_WALL_OR_FLOOR_SUPPORT"
    elif p95 > 0.25:
        state = "QUARANTINED"
        reason = "POSE_RESIDUAL_TOO_LARGE"
    elif plane > 0.03:
        state = "QUARANTINED"
        reason = "CORNER_PLANE_RESIDUAL_TOO_LARGE"
    else:
        state = "VERIFIED"
        reason = None
    return {
        "state": state,
        "reason": reason,
        "point_to_trajectory_p95_m": round(p95, 4),
        "plane_residual_p95_m": round(plane, 4),
        "vertical_support_count": vertical,
        "horizontal_support_count": horizontal,
    }


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{label} is required")
    return value.strip()


def _normalise_v2_entity_list(value: Any, label: str, *, geometry: str = "position") -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, entity in enumerate(value):
        if not isinstance(entity, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        entity_id = _identifier(entity.get("id"), f"{label}[{index}].id")
        if entity_id in ids:
            raise ValueError(f"{label}[{index}].id must be unique")
        normalised = dict(entity)
        normalised["id"] = entity_id
        if geometry == "position":
            position = entity.get("position")
            if position is not None:
                normalised["position"] = _point(position, f"{label}[{index}].position")
        elif geometry == "polygon":
            polygon = entity.get("polygon", entity.get("geometry"))
            normalised["polygon"] = _polygon(polygon, f"{label}[{index}]")
        elif geometry == "line":
            line = entity.get("geometry", entity.get("line"))
            if not isinstance(line, list) or len(line) < 2:
                raise ValueError(f"{label}[{index}].geometry must contain at least two points")
            normalised["geometry"] = [_point(point, f"{label}[{index}].geometry[{point_index}]") for point_index, point in enumerate(line)]
        ids.add(entity_id)
        result.append(normalised)
    return result


def _normalise_v2_entity_metadata(
    entities: list[dict[str, Any]],
    label: str,
    *,
    coordinate_frame: str,
    revision: int,
    default_kind: str,
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for index, entity in enumerate(entities):
        item = dict(entity)
        entity_kind = item.get("kind", default_kind)
        if not isinstance(entity_kind, str) or not entity_kind.strip():
            raise ValueError(f"{label}[{index}].kind is required")
        entity_frame = item.get("coordinate_frame", coordinate_frame)
        if entity_frame != coordinate_frame:
            raise ValueError(f"{label}[{index}].coordinate_frame does not match coarse map")
        entity_provenance = item.get("provenance", "USER_PRIOR")
        if entity_provenance != "USER_PRIOR":
            raise ValueError(f"{label}[{index}].provenance must remain USER_PRIOR")
        entity_revision = item.get("revision", revision)
        if isinstance(entity_revision, bool) or not isinstance(entity_revision, int) or entity_revision <= 0:
            raise ValueError(f"{label}[{index}].revision must be a positive integer")
        confidence = _number(item.get("confidence", 0.35), f"{label}[{index}].confidence")
        item["kind"] = entity_kind.strip()
        item["name"] = str(item.get("name") or item.get("label") or item["id"])
        item["coordinate_frame"] = coordinate_frame
        item["provenance"] = "USER_PRIOR"
        item["confidence"] = round(max(0.0, min(1.0, confidence)), 3)
        item["revision"] = entity_revision
        normalised.append(item)
    return normalised


def _validate_coarse_map_v2(payload: dict[str, Any]) -> dict[str, Any]:
    map_id = _identifier(payload.get("map_id"), "coarse map map_id")
    frame = payload.get("coordinate_frame", "BUILDING_LOCAL")
    if frame not in {"SESSION_ODOM_LOCAL", "BUILDING_LOCAL"}:
        raise ValueError("coarse map coordinate_frame is unsupported")
    building_id = _identifier(payload.get("building_id", "building-1"), "coarse map building_id")
    floor_id = _identifier(payload.get("floor_id", map_id), "coarse map floor_id")
    provenance = payload.get("provenance", "USER_PRIOR")
    if provenance != "USER_PRIOR":
        raise ValueError("coarse map provenance must remain USER_PRIOR")
    if payload.get("geometry_authority", False) is not False:
        raise ValueError("coarse map geometry_authority must be false")
    revision = payload.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise ValueError("coarse map revision must be a positive integer")
    required_collections = (
        "rooms",
        "corridors",
        "walls",
        "doors",
        "features",
        "keep_out_zones",
        "anchors",
    )
    for collection in required_collections:
        if collection not in payload:
            raise ValueError(f"coarse map {collection} is required")

    rooms: list[dict[str, Any]] = []
    for index, room in enumerate(payload.get("rooms", [])):
        if not isinstance(room, dict):
            raise ValueError(f"rooms[{index}] must be an object")
        room_id = _identifier(room.get("id"), f"rooms[{index}].id")
        if room.get("coordinate_frame", frame) != frame:
            raise ValueError(f"rooms[{index}].coordinate_frame does not match coarse map")
        if room.get("provenance", "USER_PRIOR") != "USER_PRIOR":
            raise ValueError(f"rooms[{index}].provenance must remain USER_PRIOR")
        room_revision = room.get("revision", revision)
        if isinstance(room_revision, bool) or not isinstance(room_revision, int) or room_revision <= 0:
            raise ValueError(f"rooms[{index}].revision must be a positive integer")
        polygon = room.get("polygon", room.get("geometry"))
        normalised = {
            "id": room_id,
            "name": str(room.get("name") or room_id),
            "kind": room.get("kind", "room"),
            "polygon": _polygon(polygon, f"rooms[{index}]"),
            "coverage_percent": round(max(0.0, min(100.0, _number(room.get("coverage_percent", 0.0), f"rooms[{index}].coverage_percent"))), 2),
            "provenance": "USER_PRIOR",
            "confidence": round(max(0.0, min(1.0, _number(room.get("confidence", 0.35), f"rooms[{index}].confidence"))), 3),
            "coordinate_frame": frame,
            "revision": room_revision,
        }
        if normalised["kind"] not in _ALLOWED_ROOM_KINDS:
            raise ValueError(f"rooms[{index}].kind is unsupported")
        if normalised["kind"] == "stair":
            profile = _stair_profile(room.get("stair_profile"), f"rooms[{index}].stair_profile")
            if profile is not None:
                normalised["stair_profile"] = profile
            entry = _optional_point(room.get("entry"), f"rooms[{index}].entry")
            exit_point = _optional_point(room.get("exit"), f"rooms[{index}].exit")
            if entry is not None:
                normalised["entry"] = entry
            if exit_point is not None:
                normalised["exit"] = exit_point
        rooms.append(normalised)

    corridors = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("corridors"), "corridors", geometry="polygon"),
        "corridors", coordinate_frame=frame, revision=revision, default_kind="corridor",
    )
    for index, corridor in enumerate(corridors):
        coverage = _number(corridor.get("coverage_percent", 0.0), f"corridors[{index}].coverage_percent")
        corridor["coverage_percent"] = round(max(0.0, min(100.0, coverage)), 2)
    walls = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("walls"), "walls", geometry="line"),
        "walls", coordinate_frame=frame, revision=revision, default_kind="wall",
    )
    doors = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("doors"), "doors", geometry="position"),
        "doors", coordinate_frame=frame, revision=revision, default_kind="door",
    )
    features = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("features"), "features", geometry="position"),
        "features", coordinate_frame=frame, revision=revision, default_kind="inspection_target",
    )
    for index, feature in enumerate(features):
        if "locked" in feature and not isinstance(feature["locked"], bool):
            raise ValueError(f"features[{index}].locked must be boolean")
        if feature.get("kind") != "equipment" or "dimensions_m" not in feature:
            continue
        asset_id = feature.get("asset_id")
        if asset_id is not None and (not isinstance(asset_id, str) or asset_id not in _FURNITURE_ASSET_IDS):
            raise ValueError(f"features[{index}].asset_id is not in the furniture catalog")
        asset_source = feature.get("asset_source")
        if asset_source is not None and asset_source not in {"PROCEDURAL_FALLBACK", "LOCAL_GLTF", "LOCAL_OBJ"}:
            raise ValueError(f"features[{index}].asset_source is unsupported")
        catalog_item = next((item for item in FURNITURE_CATALOG if item["asset_id"] == asset_id), None)
        if catalog_item is not None:
            if asset_source is not None and asset_source != catalog_item["asset_source"]:
                raise ValueError(f"features[{index}].asset_source does not match the furniture catalog")
            if catalog_item.get("asset_url"):
                feature["asset_url"] = catalog_item["asset_url"]
            if catalog_item.get("asset_sha256"):
                feature["asset_sha256"] = catalog_item["asset_sha256"]
            if catalog_item.get("license"):
                feature["license"] = catalog_item["license"]
        dimensions = feature["dimensions_m"]
        if not isinstance(dimensions, list) or len(dimensions) != 3 or any(_number(value, f"features[{index}].dimensions_m") <= 0 for value in dimensions):
            raise ValueError(f"features[{index}].dimensions_m must contain three positive values")
        rotation = _number(feature.get("rotation_rad", 0.0), f"features[{index}].rotation_rad")
        feature["dimensions_m"] = [round(_number(value, f"features[{index}].dimensions_m"), 4) for value in dimensions]
        feature["rotation_rad"] = round(rotation, 6)
    keep_out_zones = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("keep_out_zones"), "keep_out_zones", geometry="polygon"),
        "keep_out_zones", coordinate_frame=frame, revision=revision, default_kind="keep_out_zone",
    )
    anchors = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("anchors"), "anchors", geometry="position"),
        "anchors", coordinate_frame=frame, revision=revision, default_kind="anchor",
    )
    spaces_by_id = {item["id"]: item for item in rooms}
    spaces_by_id.update({item["id"]: item for item in corridors})
    for label, entities in (("doors", doors), ("features", features), ("anchors", anchors)):
        for index, entity in enumerate(entities):
            owner_id = entity.get("room_id")
            if owner_id is None:
                continue
            owner = spaces_by_id.get(str(owner_id))
            if owner is None:
                raise ValueError(f"{label}[{index}].room_id does not reference a room or corridor")
            if not _point_in_polygon(entity["position"], owner["polygon"]):
                raise ValueError(f"{label}[{index}].position must lie inside its room_id polygon")
    carpets = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("carpets"), "carpets", geometry="polygon"),
        "carpets", coordinate_frame=frame, revision=revision, default_kind="carpet",
    )
    thresholds = _normalise_v2_entity_metadata(
        _normalise_v2_entity_list(payload.get("thresholds"), "thresholds", geometry="line"),
        "thresholds", coordinate_frame=frame, revision=revision, default_kind="threshold",
    )
    return {
        "schema_version": 2,
        "map_id": map_id,
        "coordinate_frame": frame,
        "building_id": building_id,
        "floor_id": floor_id,
        "rooms": rooms,
        "corridors": corridors,
        "walls": walls,
        "doors": doors,
        "features": features,
        "keep_out_zones": keep_out_zones,
        "anchors": anchors,
        "carpets": carpets,
        "thresholds": thresholds,
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": revision,
        "tasks": list(payload.get("tasks") or []),
        "updated_at": payload.get("updated_at"),
    }


def migrate_coarse_map_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy map prior without treating it as measured geometry."""

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("coarse map v1 payload is required")
    legacy = validate_coarse_map(payload)
    legacy_features = list(legacy.get("features") or [])
    doors = [dict(feature) for feature in legacy_features if feature.get("kind") == "door"]
    anchors = [dict(feature) for feature in legacy_features if feature.get("kind") == "anchor"]
    features = [dict(feature) for feature in legacy_features if feature.get("kind") not in {"door", "anchor"}]
    return {
        "schema_version": 2,
        "map_id": legacy["map_id"],
        "coordinate_frame": legacy.get("coordinate_frame", "SESSION_ODOM_LOCAL"),
        "building_id": "building-1",
        "floor_id": legacy["map_id"],
        "rooms": legacy.get("rooms", []),
        "corridors": [],
        "walls": [],
        "doors": doors,
        "features": features,
        "keep_out_zones": [dict(zone) for zone in legacy.get("keep_out_zones", [])],
        "anchors": anchors,
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 1,
        "tasks": legacy.get("tasks", []),
        "updated_at": legacy.get("updated_at"),
    }


def validate_coarse_map(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("coarse map must be an object")
    if payload.get("schema_version") == 2:
        return _validate_coarse_map_v2(payload)
    if payload.get("schema_version") != 1:
        raise ValueError("coarse map schema_version must be 1")
    map_id = payload.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip() or len(map_id) > 128:
        raise ValueError("coarse map map_id is required")
    frame = payload.get("coordinate_frame", "SESSION_ODOM_LOCAL")
    if frame not in {"SESSION_ODOM_LOCAL", "BUILDING_LOCAL"}:
        raise ValueError("coarse map coordinate_frame is unsupported")
    rooms = payload.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("coarse map rooms must be a list")
    room_ids: set[str] = set()
    normalized_rooms: list[dict[str, Any]] = []
    for index, room in enumerate(rooms):
        if not isinstance(room, dict):
            raise ValueError(f"rooms[{index}] must be an object")
        room_id = room.get("id")
        if not isinstance(room_id, str) or not room_id.strip() or room_id in room_ids:
            raise ValueError(f"rooms[{index}].id must be unique")
        kind = room.get("kind", "room")
        if kind not in _ALLOWED_ROOM_KINDS:
            raise ValueError(f"rooms[{index}].kind is unsupported")
        polygon = _polygon(room.get("polygon"), f"rooms[{index}]")
        coverage = _number(room.get("coverage_percent", 0.0), f"rooms[{index}].coverage_percent")
        if not 0 <= coverage <= 100:
            raise ValueError(f"rooms[{index}].coverage_percent must be between 0 and 100")
        room_ids.add(room_id)
        normalized_rooms.append({
            "id": room_id,
            "name": str(room.get("name") or room_id),
            "kind": kind,
            "polygon": polygon,
            "coverage_percent": round(coverage, 2),
            "provenance": str(room.get("provenance") or "USER_PRIOR"),
            "confidence": round(max(0.0, min(1.0, _number(room.get("confidence", 0.35), f"rooms[{index}].confidence"))), 3),
        })
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError("coarse map features must be a list")
    feature_ids: set[str] = set()
    normalized_features: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"features[{index}] must be an object")
        feature_id = feature.get("id")
        if not isinstance(feature_id, str) or not feature_id.strip() or feature_id in feature_ids:
            raise ValueError(f"features[{index}].id must be unique")
        kind = feature.get("kind", "inspection_target")
        if kind not in _ALLOWED_FEATURE_KINDS:
            raise ValueError(f"features[{index}].kind is unsupported")
        normalized_features.append({
            "id": feature_id,
            "kind": kind,
            "label": str(feature.get("label") or feature_id),
            "position": _point(feature.get("position"), f"features[{index}].position"),
            "room_id": feature.get("room_id"),
            "provenance": str(feature.get("provenance") or "USER_PRIOR"),
            "confidence": round(max(0.0, min(1.0, _number(feature.get("confidence", 0.35), f"features[{index}].confidence"))), 3),
        })
        feature_ids.add(feature_id)
    keep_out_zones = payload.get("keep_out_zones", [])
    if not isinstance(keep_out_zones, list):
        raise ValueError("coarse map keep_out_zones must be a list")
    keep_out_ids: set[str] = set()
    normalized_keep_out_zones: list[dict[str, Any]] = []
    for index, keep_out in enumerate(keep_out_zones):
        if not isinstance(keep_out, dict):
            raise ValueError(f"keep_out_zones[{index}] must be an object")
        keep_out_id = keep_out.get("id")
        if not isinstance(keep_out_id, str) or not keep_out_id.strip() or keep_out_id in keep_out_ids:
            raise ValueError(f"keep_out_zones[{index}].id must be unique")
        normalized_keep_out_zones.append({
            "id": keep_out_id,
            "kind": "keep_out_zone",
            "label": str(keep_out.get("label") or keep_out_id),
            "polygon": _polygon(keep_out.get("polygon"), f"keep_out_zones[{index}]"),
            "provenance": str(keep_out.get("provenance") or "USER_PRIOR"),
            "confidence": round(max(0.0, min(1.0, _number(keep_out.get("confidence", 0.35), f"keep_out_zones[{index}].confidence"))), 3),
        })
        keep_out_ids.add(keep_out_id)
    return {
        "schema_version": 1,
        "map_id": map_id,
        "coordinate_frame": frame,
        "rooms": normalized_rooms,
        "features": normalized_features,
        "keep_out_zones": normalized_keep_out_zones,
        "tasks": list(payload.get("tasks") or []),
        "updated_at": payload.get("updated_at"),
    }


def _room_for_feature(feature: dict[str, Any], rooms: list[dict[str, Any]]) -> str | None:
    if feature.get("room_id") in {room["id"] for room in rooms}:
        return feature["room_id"]
    if not rooms:
        return None
    px, py = feature["position"]
    return min(
        rooms,
        key=lambda room: min(math.dist([px, py], point) for point in room["polygon"]),
    )["id"]


def _task_spaces(coarse: dict[str, Any]) -> list[dict[str, Any]]:
    """Return room-like map entities that can own verification tasks."""

    spaces = [dict(room) for room in coarse.get("rooms", [])]
    for corridor in coarse.get("corridors", []) or []:
        if isinstance(corridor, dict):
            spaces.append(dict(corridor))
    return spaces


def _task_features(coarse: dict[str, Any]) -> list[dict[str, Any]]:
    features = [dict(feature) for feature in coarse.get("features", []) if isinstance(feature, dict)]
    for door in coarse.get("doors", []) or []:
        if not isinstance(door, dict):
            continue
        feature = dict(door)
        feature.setdefault("kind", "door")
        feature.setdefault("label", feature.get("id", "door"))
        features.append(feature)
    return features


def build_task_graph(payload: dict[str, Any]) -> dict[str, Any]:
    coarse = validate_coarse_map(payload)
    now = time.time()
    previous_tasks = {str(task.get("id")): task for task in coarse.get("tasks", []) if isinstance(task, dict)}
    tasks: list[dict[str, Any]] = []
    spaces = _task_spaces(coarse)
    task_features = _task_features(coarse)
    for room in spaces:
        area = _polygon_area(room["polygon"])
        perimeter = _polygon_perimeter(room["polygon"])
        base = {"room_id": room["id"], "status": "PENDING", "coverage_percent": room["coverage_percent"], "provenance": "TASK_PRIOR"}
        tasks.append({
            **base,
            "id": f"{room['id']}:perimeter",
            "kind": "perimeter",
            "title": f"沿墙扫描 · {room['name']}",
            "estimated_minutes": max(1, round(perimeter / 0.2 / 60)),
            "priority": 1,
        })
        tasks.append({
            **base,
            "id": f"{room['id']}:corners",
            "kind": "corners",
            "title": f"墙角验证 · {room['name']}",
            "estimated_minutes": max(1, len(room["polygon"])),
            "priority": 2,
        })
        for feature in task_features:
            if _room_for_feature(feature, spaces) != room["id"]:
                continue
            tasks.append({
                **base,
                "id": f"{room['id']}:feature:{feature['id']}",
                "kind": "feature",
                "feature_id": feature["id"],
                "title": f"验证 · {feature['label']}",
                "estimated_minutes": 1,
                "priority": 3,
            })
        tasks.append({
            **base,
            "id": f"{room['id']}:floor",
            "kind": "floor_coverage",
            "title": f"地面补扫 · {room['name']}",
            "estimated_minutes": max(1, round(area / 0.6)),
            "priority": 4,
        })
    for task in tasks:
        previous = previous_tasks.get(task["id"])
        task["mode"] = _default_task_mode(task["kind"])
        task["lease_id"] = None
        task["lease_state"] = "UNLEASED"
        task["revision"] = 1
        if previous:
            task["status"] = previous.get("status", task["status"])
            if previous.get("mode") in _ALLOWED_TASK_MODES:
                task["mode"] = previous["mode"]
            task["lease_id"] = _task_lease_id(previous.get("lease_id"), f"tasks[{task['id']}].lease_id")
            task["lease_state"] = "LEASED" if task["lease_id"] else "UNLEASED"
            previous_revision = previous.get("revision", 1)
            if isinstance(previous_revision, int) and previous_revision > 0:
                task["revision"] = previous_revision
            if "diagnostics" in previous:
                task["diagnostics"] = previous["diagnostics"]
            if "coverage_percent" in previous:
                task["coverage_percent"] = previous["coverage_percent"]
    coarse["tasks"] = tasks
    coarse["updated_at"] = now
    return coarse


def update_task_record(
    payload: dict[str, Any],
    task_id: str,
    *,
    status: Any = _MISSING,
    coverage_percent: Any = _MISSING,
    diagnostics: Any = _MISSING,
    mode: Any = _MISSING,
    lease_id: Any = _MISSING,
    expected_revision: Any = _MISSING,
) -> dict[str, Any]:
    coarse = build_task_graph(payload)
    tasks = list(coarse.get("tasks") or [])
    for task in tasks:
        if task.get("id") != task_id:
            continue
        revision = task.get("revision", 1)
        if isinstance(expected_revision, int) and expected_revision > 0 and revision != expected_revision:
            raise ValueError("task revision does not match expected_revision")
        changed = False
        if status is not _MISSING:
            if status not in {"PENDING", "ACTIVE", "DONE", "BLOCKED", "QUARANTINED", "READY", "PAUSED", "CANCELLED"}:
                raise ValueError("task status is unsupported")
            task["status"] = status
            changed = True
        if coverage_percent is not _MISSING:
            coverage = _number(coverage_percent, f"tasks[{task_id}].coverage_percent")
            if not 0 <= coverage <= 100:
                raise ValueError("coverage_percent must be between 0 and 100")
            task["coverage_percent"] = round(coverage, 2)
            changed = True
        if diagnostics is not _MISSING:
            task["diagnostics"] = (
                assess_corner_diagnostics(diagnostics)
                if task.get("kind") == "corners"
                else diagnostics
            )
            if task.get("kind") == "corners" and isinstance(task["diagnostics"], dict) and task["diagnostics"].get("state") == "QUARANTINED":
                task["status"] = "QUARANTINED"
            changed = True
        if mode is not _MISSING:
            normalized_mode = _task_mode(mode, f"tasks[{task_id}].mode")
            if task.get("status") == "ACTIVE" and normalized_mode != task.get("mode"):
                raise ValueError("task mode cannot change while task is active")
            if normalized_mode == "AUTONOMOUS" and not task.get("lease_id") and lease_id in {_MISSING, None, ""}:
                raise ValueError("AUTONOMOUS mode requires a lease_id")
            task["mode"] = normalized_mode
            changed = True
        if lease_id is not _MISSING:
            normalized_lease = _task_lease_id(lease_id, f"tasks[{task_id}].lease_id")
            task["lease_id"] = normalized_lease
            task["lease_state"] = "LEASED" if normalized_lease else "UNLEASED"
            changed = True
        if changed:
            task["revision"] = int(revision) + 1
        coarse["tasks"] = tasks
        coarse["updated_at"] = time.time()
        return coarse
    raise ValueError("coarse map task not found")


class CoarseMapStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    @property
    def _revision_root(self) -> Path:
        return self.path.with_suffix(".revisions")

    def _revision_path(self, revision: int) -> Path:
        return self._revision_root / f"{revision:08d}.json"

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{path.name}.", suffix=".partial", dir=path.parent
        )
        try:
            os.close(fd)
            temporary = Path(temp_name)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def _record_revision(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != 2:
            return
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
            raise ValueError("coarse map revision must be a positive integer")
        path = self._revision_path(revision)
        if not path.is_file():
            self._write_atomic(path, payload)

    def _load_canonical(self, map_id: str) -> dict[str, Any]:
        current = self.load()
        if current.get("map_id") != map_id:
            raise KeyError(map_id)
        if current.get("schema_version") == 1:
            current = self.save(migrate_coarse_map_v1_to_v2(current))
        if current.get("schema_version") != 2:
            raise ValueError("coarse map schema_version must be 2")
        self._record_revision(current)
        return current

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                return self.save({
                    "schema_version": 1,
                    "map_id": "go2-demo-floor",
                    "coordinate_frame": "SESSION_ODOM_LOCAL",
                    "rooms": [{"id": "room-1", "name": "待建房间", "kind": "room", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}],
                    "features": [],
                })
            return validate_coarse_map(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = build_task_graph(payload)
        with self._lock:
            self._write_atomic(self.path, normalized)
            self._record_revision(normalized)
        return normalized

    def load_canonical(self, map_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._load_canonical(_identifier(map_id, "map_id")))

    def replace_canonical(
        self,
        map_id: str,
        payload: dict[str, Any],
        *,
        expected_revision: Any,
    ) -> dict[str, Any]:
        with self._lock:
            canonical_map_id = _identifier(map_id, "map_id")
            current = self._load_canonical(canonical_map_id)
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision <= 0:
                raise ValueError("expected_revision must be a positive integer")
            if expected_revision != current["revision"]:
                raise RuntimeError("coarse map revision does not match expected_revision")
            replacement = validate_coarse_map(payload)
            if replacement.get("schema_version") != 2:
                raise ValueError("coarse map schema_version must be 2")
            if replacement.get("map_id") != canonical_map_id:
                raise ValueError("coarse map map_id does not match request path")
            if "tasks" not in payload:
                replacement["tasks"] = current.get("tasks", [])
            replacement["revision"] = current["revision"] + 1
            return deepcopy(self.save(replacement))

    def revision_history(self, map_id: str) -> list[dict[str, Any]]:
        with self._lock:
            canonical_map_id = _identifier(map_id, "map_id")
            self._load_canonical(canonical_map_id)
            revisions: list[dict[str, Any]] = []
            for path in sorted(self._revision_root.glob("*.json")):
                payload = validate_coarse_map(json.loads(path.read_text(encoding="utf-8")))
                if payload.get("map_id") == canonical_map_id:
                    revisions.append(payload)
            return deepcopy(revisions)

    def load_revision(self, map_id: str, revision: int) -> dict[str, Any]:
        with self._lock:
            canonical_map_id = _identifier(map_id, "map_id")
            self._load_canonical(canonical_map_id)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
                raise ValueError("revision must be a positive integer")
            path = self._revision_path(revision)
            if not path.is_file():
                raise KeyError(revision)
            payload = validate_coarse_map(json.loads(path.read_text(encoding="utf-8")))
            if payload.get("map_id") != canonical_map_id:
                raise KeyError(revision)
            return deepcopy(payload)


class CoarseBuildingStore:
    """Persist a lightweight building index over independent floor maps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    @staticmethod
    def _validate_building(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("coarse building must be an object")
        building_id = _identifier(payload.get("building_id"), "building_id")
        floors = payload.get("floors", [])
        if not isinstance(floors, list):
            raise ValueError("coarse building floors must be a list")
        normalized_floors: list[dict[str, Any]] = []
        floor_ids: set[str] = set()
        map_ids: set[str] = set()
        for index, floor in enumerate(floors):
            if not isinstance(floor, dict):
                raise ValueError(f"floors[{index}] must be an object")
            floor_id = _identifier(floor.get("floor_id"), f"floors[{index}].floor_id")
            map_id = _identifier(floor.get("map_id"), f"floors[{index}].map_id")
            if floor_id in floor_ids or map_id in map_ids:
                raise ValueError("coarse building floor and map IDs must be unique")
            revision = floor.get("revision", 1)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
                raise ValueError(f"floors[{index}].revision must be positive")
            floor_ids.add(floor_id)
            map_ids.add(map_id)
            normalized_floors.append({
                "map_id": map_id,
                "floor_id": floor_id,
                "label": str(floor.get("label") or floor_id),
                "revision": revision,
            })
        connections = payload.get("connections", [])
        if not isinstance(connections, list):
            raise ValueError("coarse building connections must be a list")
        normalized_connections: list[dict[str, Any]] = []
        connection_ids: set[str] = set()
        for index, connection in enumerate(connections):
            if not isinstance(connection, dict):
                raise ValueError(f"connections[{index}] must be an object")
            connection_id = _identifier(connection.get("id"), f"connections[{index}].id")
            from_floor = _identifier(connection.get("from_floor_id"), f"connections[{index}].from_floor_id")
            to_floor = _identifier(connection.get("to_floor_id"), f"connections[{index}].to_floor_id")
            if connection_id in connection_ids:
                raise ValueError("coarse building connection IDs must be unique")
            if from_floor == to_floor:
                raise ValueError(f"connections[{index}] must connect two different floors")
            # Keep legacy index updates tolerant of stale connections; a connection
            # becomes canonical only after both referenced floors are registered.
            if from_floor not in floor_ids or to_floor not in floor_ids:
                continue
            kind = str(connection.get("kind") or "stairs")
            if kind not in _ALLOWED_VERTICAL_CONNECTION_KINDS:
                raise ValueError(f"connections[{index}].kind is unsupported")
            from_anchor = _optional_point(connection.get("from_anchor"), f"connections[{index}].from_anchor")
            to_anchor = _optional_point(connection.get("to_anchor"), f"connections[{index}].to_anchor")
            if (from_anchor is None) != (to_anchor is None):
                raise ValueError(f"connections[{index}] must provide both anchors or neither")
            connection_ids.add(connection_id)
            normalized = {
                "id": connection_id,
                "kind": kind,
                "from_floor_id": from_floor,
                "to_floor_id": to_floor,
                "label": str(connection.get("label") or connection_id),
            }
            if from_anchor is not None:
                normalized["from_anchor"] = from_anchor
                normalized["to_anchor"] = to_anchor
            if kind == "stairs":
                profile = _stair_profile(connection.get("stair_profile"), f"connections[{index}].stair_profile")
                if profile is not None:
                    normalized["stair_profile"] = profile
            normalized_connections.append(normalized)
        return {
            "schema_version": 1,
            "building_id": building_id,
            "floors": normalized_floors,
            "connections": normalized_connections,
        }

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                return {"schema_version": 1, "building_id": "building-1", "floors": [], "connections": []}
            return self._validate_building(json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._validate_building(payload)
        CoarseMapStore._write_atomic(self.path, normalized)
        return deepcopy(normalized)

    def register_floor(self, floor_map: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
        coarse = validate_coarse_map(floor_map)
        building = self.load()
        if building["floors"] and coarse["building_id"] != building["building_id"]:
            raise ValueError("floor map building_id does not match coarse building")
        building["building_id"] = coarse["building_id"]
        floors = [item for item in building["floors"] if item["floor_id"] != coarse["floor_id"]]
        floors.append({"map_id": coarse["map_id"], "floor_id": coarse["floor_id"], "label": label or coarse["floor_id"], "revision": coarse["revision"]})
        building["floors"] = sorted(floors, key=lambda item: item["floor_id"])
        return self._save(building)

    def create_floor(self, building_id: str, floor_id: str, *, label: str | None = None) -> dict[str, Any]:
        building_id = _identifier(building_id, "building_id")
        floor_id = _identifier(floor_id, "floor_id")
        building = self.load()
        if building["floors"] and building["building_id"] != building_id:
            raise ValueError("building_id does not match coarse building")
        if any(item["floor_id"] == floor_id for item in building["floors"]):
            raise ValueError("floor_id is already registered")
        root = self.path.parent / "coarse-floors"
        floor_path = root / f"{floor_id}.json"
        if floor_path.exists():
            raise ValueError("floor map already exists")
        floor_map = validate_coarse_map({
            "schema_version": 2,
            "map_id": floor_id,
            "coordinate_frame": "BUILDING_LOCAL",
            "building_id": building_id,
            "floor_id": floor_id,
            "rooms": [],
            "corridors": [],
            "walls": [],
            "doors": [],
            "features": [],
            "keep_out_zones": [],
            "anchors": [],
            "carpets": [],
            "thresholds": [],
            "provenance": "USER_PRIOR",
            "geometry_authority": False,
            "revision": 1,
        })
        CoarseMapStore(floor_path).save(floor_map)
        self.register_floor(floor_map, label=label or floor_id)
        return deepcopy(floor_map)

    def update_connections(self, payload: dict[str, Any]) -> dict[str, Any]:
        building = self.load()
        building["connections"] = payload.get("connections", []) if isinstance(payload, dict) else []
        return self._save(building)
