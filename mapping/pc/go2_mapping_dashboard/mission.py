"""Mission compilation and task execution state for coarse-map priors."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any

from .coarse_map import build_task_graph, validate_coarse_map


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", re.ASCII)
_ALLOWED_TASK_MODES = {"MANUAL", "ASSISTED", "AUTONOMOUS"}
_ALLOWED_TASK_STATUSES = {"PENDING", "READY", "ACTIVE", "PAUSED", "DONE", "BLOCKED", "QUARANTINED", "CANCELLED"}
_TASK_STATUS_TRANSITIONS = {
    "PENDING": {"READY", "ACTIVE", "PAUSED", "BLOCKED", "QUARANTINED", "CANCELLED"},
    "READY": {"ACTIVE", "PAUSED", "BLOCKED", "QUARANTINED", "CANCELLED"},
    "ACTIVE": {"PAUSED", "DONE", "BLOCKED", "QUARANTINED", "CANCELLED"},
    "PAUSED": {"ACTIVE", "DONE", "BLOCKED", "QUARANTINED", "CANCELLED"},
    "BLOCKED": {"READY", "ACTIVE", "PAUSED", "QUARANTINED", "CANCELLED"},
    "DONE": set(),
    "QUARANTINED": set(),
    "CANCELLED": set(),
}
_MISSING = object()


def _identifier(value: Any, label: str) -> str:
    result = str(value)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ValueError(f"{label} is invalid")
    return result


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(fd, "wb", closefd=True) as target:
            target.write(_canonical_bytes(payload))
            target.flush()
            os_fileno = target.fileno()
            if os_fileno >= 0:
                import os

                os.fsync(os_fileno)
        path_temp = Path(temp_name)
        path_temp.replace(path)
        import os

        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        body = path.read_bytes()
        payload = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict) or body != _canonical_bytes(payload):
        raise ValueError(f"{label} is not canonical")
    return payload


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _point(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] point")
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{label} must be finite")
    return [x, y]


def _polygon_centroid(points: list[list[float]]) -> list[float]:
    area = 0.0
    cx = 0.0
    cy = 0.0
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-9:
        return [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)]
    return [cx / (6.0 * area), cy / (6.0 * area)]


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index in range(len(polygon)):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % len(polygon)]
        intersects = ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
        )
        if intersects:
            inside = not inside
    return inside


def _point_on_segment(point: list[float], start: list[float], end: list[float]) -> bool:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    cross = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
    if abs(cross) > 1e-8:
        return False
    return min(x0, x1) - 1e-8 <= px <= max(x0, x1) + 1e-8 and min(y0, y1) - 1e-8 <= py <= max(y0, y1) + 1e-8


def _point_in_or_on_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    return _point_in_polygon(point, polygon) or any(
        _point_on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _orientation(first: list[float], second: list[float], third: list[float]) -> int:
    cross = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
    if abs(cross) <= 1e-8:
        return 0
    return 1 if cross > 0 else -1


def _segments_intersect(first_start: list[float], first_end: list[float], second_start: list[float], second_end: list[float]) -> bool:
    first = _orientation(first_start, first_end, second_start)
    second = _orientation(first_start, first_end, second_end)
    third = _orientation(second_start, second_end, first_start)
    fourth = _orientation(second_start, second_end, first_end)
    if first != second and third != fourth:
        return True
    return (
        (first == 0 and _point_on_segment(second_start, first_start, first_end))
        or (second == 0 and _point_on_segment(second_end, first_start, first_end))
        or (third == 0 and _point_on_segment(first_start, second_start, second_end))
        or (fourth == 0 and _point_on_segment(first_end, second_start, second_end))
    )


def _route_segments(points: list[list[float]]) -> list[tuple[list[float], list[float]]]:
    return list(zip(points, points[1:]))


def _route_length(points: list[list[float]]) -> float:
    return sum(math.hypot(end[0] - start[0], end[1] - start[1]) for start, end in _route_segments(points))


def _route_leaves_polygon(points: list[list[float]], polygon: list[list[float]]) -> bool:
    if not polygon or any(not _point_in_or_on_polygon(point, polygon) for point in points):
        return True
    for start, end in _route_segments(points):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, math.ceil(distance / 0.1))
        for index in range(1, samples):
            fraction = index / samples
            point = [start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction]
            if not _point_in_or_on_polygon(point, polygon):
                return True
    return False


def _route_crosses_polygon(points: list[list[float]], polygon: list[list[float]]) -> bool:
    if any(_point_in_or_on_polygon(point, polygon) for point in points):
        return True
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    return any(
        _segments_intersect(start, end, edge_start, edge_end)
        for start, end in _route_segments(points)
        for edge_start, edge_end in edges
    )


def _route_points(task: dict[str, Any]) -> tuple[list[list[float]], str, str | None]:
    raw_waypoints = task.get("waypoints")
    if isinstance(raw_waypoints, list) and len(raw_waypoints) >= 2:
        try:
            return ([_point(point, "waypoint") for point in raw_waypoints], "WAYPOINTS", None)
        except ValueError:
            return [], "INVALID", "ROUTE_INVALID"
    raw_polygon = task.get("room_polygon")
    if isinstance(raw_polygon, list) and len(raw_polygon) >= 3:
        try:
            return ([_point(point, "room_polygon") for point in raw_polygon], "POLYGON_FALLBACK", None)
        except ValueError:
            return [], "INVALID", "ROUTE_INVALID"
    if isinstance(task.get("inspection_targets"), list) and task["inspection_targets"]:
        return [], "INSPECTION_TARGET_ONLY", None
    return [], "MISSING", "MISSING_ROUTE"


def audit_mission_route(task: dict[str, Any], coarse_map_payload: dict[str, Any]) -> dict[str, Any]:
    """Assess a Mission route against its current user-prior safety geometry.

    The report is deliberately advisory for valid paths: an operator still has
    to review a route before any downstream navigation workflow. Invalid or
    unsafe paths are marked ineligible and can be rejected before goal compile.
    """

    route, route_source, route_error = _route_points(task)
    source_map = validate_coarse_map(coarse_map_payload)
    findings: list[dict[str, str]] = []

    def finding(code: str, severity: str) -> None:
        if not any(item["code"] == code for item in findings):
            findings.append({"code": code, "severity": severity})

    if route_error:
        finding(route_error, "BLOCKER")
    if route_source == "POLYGON_FALLBACK":
        finding("POLYGON_FALLBACK", "REVIEW")
    if route_source == "INSPECTION_TARGET_ONLY":
        finding("INSPECTION_TARGET_ROUTE_REQUIRED", "REVIEW")

    if route:
        room_id = str(task.get("room_id") or "")
        zone = task.get("room_polygon")
        if not isinstance(zone, list) or len(zone) < 3:
            zone = next(
                (
                    space.get("polygon")
                    for collection in ("rooms", "corridors")
                    for space in source_map.get(collection, [])
                    if isinstance(space, dict) and str(space.get("id") or "") == room_id
                ),
                [],
            )
        try:
            normalized_zone = [_point(point, "task zone") for point in zone]
        except ValueError:
            normalized_zone = []
        if str(task.get("kind") or "") == "vertical_connection":
            finding("VERTICAL_CONNECTION_REVIEW_REQUIRED", "REVIEW")
        elif len(normalized_zone) < 3:
            finding("ROUTE_ZONE_UNAVAILABLE", "BLOCKER")
        elif _route_leaves_polygon(route, normalized_zone):
            finding("ROUTE_OUTSIDE_TASK_ZONE", "BLOCKER")

        for keep_out in source_map.get("keep_out_zones", []):
            if not isinstance(keep_out, dict) or not isinstance(keep_out.get("polygon"), list):
                continue
            try:
                polygon = [_point(point, "keep_out_zone") for point in keep_out["polygon"]]
            except ValueError:
                continue
            if len(polygon) >= 3 and _route_crosses_polygon(route, polygon):
                finding("KEEP_OUT_ZONE_CROSSING", "BLOCKER")

    has_blocker = any(item["severity"] == "BLOCKER" for item in findings)
    if has_blocker:
        state = "BLOCKED"
    elif route_source == "WAYPOINTS":
        state = "READY_FOR_HUMAN_REVIEW"
        finding("HUMAN_REVIEW_REQUIRED", "REVIEW")
    else:
        state = "REVIEW_REQUIRED"
    return {
        "schema_version": 1,
        "task_id": task.get("task_id") or task.get("id"),
        "route_source": route_source,
        "state": state,
        "navigation_eligible": state == "READY_FOR_HUMAN_REVIEW",
        "waypoint_count": len(route),
        "route_length_m": round(_route_length(route), 2) if len(route) >= 2 else None,
        "findings": findings,
    }


def _coverage_waypoints(polygon: list[list[float]]) -> list[list[float]]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
        return [_polygon_centroid(polygon)]
    candidates: list[list[float]] = []
    for row in range(3):
        y = min_y + (max_y - min_y) * (row + 0.5) / 3.0
        cols = range(3) if row % 2 == 0 else range(2, -1, -1)
        for col in cols:
            x = min_x + (max_x - min_x) * (col + 0.5) / 3.0
            point = [round(x, 4), round(y, 4)]
            if _point_in_polygon(point, polygon):
                candidates.append(point)
    if not candidates:
        candidates.append(_polygon_centroid(polygon))
    return candidates


def _normalise_status(status: Any) -> str:
    if not isinstance(status, str) or status not in _ALLOWED_TASK_STATUSES:
        raise ValueError("task status is unsupported")
    return status


def _normalise_mode(mode: Any) -> str:
    if not isinstance(mode, str) or mode not in _ALLOWED_TASK_MODES:
        raise ValueError("task mode is unsupported")
    return mode


def _normalise_lease_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("lease_id is invalid")
    lease_id = value.strip()
    if not lease_id:
        return None
    if len(lease_id) > 128:
        raise ValueError("lease_id is invalid")
    return lease_id


def _mission_state(tasks: list[dict[str, Any]]) -> str:
    statuses = {str(task.get("status") or "READY") for task in tasks}
    if not tasks:
        return "READY"
    if "QUARANTINED" in statuses:
        return "QUARANTINED"
    if "CANCELLED" in statuses:
        return "CANCELLED"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "ACTIVE" in statuses:
        return "ACTIVE"
    if "PAUSED" in statuses:
        return "PAUSED"
    if statuses == {"DONE"}:
        return "DONE"
    return "READY"


def _task_geometry(task: dict[str, Any], rooms_by_id: dict[str, dict[str, Any]], features_by_id: dict[str, dict[str, Any]]) -> tuple[list[list[float]], list[list[float]], float]:
    room = rooms_by_id.get(str(task.get("room_id")))
    polygon = [list(point) for point in (room.get("polygon") if room else [])]
    kind = str(task.get("kind") or "")
    if kind == "perimeter":
        return polygon, [], 100.0
    if kind == "corners":
        return [], polygon, 100.0
    if kind == "feature":
        feature = features_by_id.get(str(task.get("feature_id")))
        target = [_point(feature["position"], "feature.position")] if feature else []
        return [], target, 100.0
    if kind == "floor_coverage":
        return _coverage_waypoints(polygon), [], 100.0
    return [], [], 100.0


def _generated_mission_id() -> str:
    return time.strftime("mission-%Y%m%dT%H%M%SZ", time.gmtime())


def compile_mission(
    coarse_map_payload: dict[str, Any],
    *,
    mission_id: str | None = None,
    previous_mission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_map = validate_coarse_map(coarse_map_payload)
    coarse = build_task_graph(source_map)
    mission_id = _identifier(mission_id or _generated_mission_id(), "mission_id")
    rooms_by_id = {str(room["id"]): room for room in coarse.get("rooms", []) if isinstance(room, dict)}
    rooms_by_id.update(
        {
            str(corridor["id"]): corridor
            for corridor in coarse.get("corridors", [])
            if isinstance(corridor, dict) and corridor.get("id")
        }
    )
    features_by_id = {str(feature["id"]): feature for feature in coarse.get("features", []) if isinstance(feature, dict)}
    features_by_id.update(
        {
            str(door["id"]): {**door, "kind": "door", "label": door.get("label") or door["id"]}
            for door in coarse.get("doors", [])
            if isinstance(door, dict) and door.get("id")
        }
    )
    previous_tasks = {
        str(task.get("task_id")): task
        for task in (previous_mission or {}).get("tasks", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    tasks: list[dict[str, Any]] = []
    for coarse_task in coarse.get("tasks", []):
        if not isinstance(coarse_task, dict):
            continue
        waypoints, inspection_targets, coverage_target = _task_geometry(coarse_task, rooms_by_id, features_by_id)
        task = {
            "task_id": str(coarse_task["id"]),
            "room_id": coarse_task.get("room_id"),
            "room_name": rooms_by_id.get(str(coarse_task.get("room_id")), {}).get("name"),
            "room_kind": rooms_by_id.get(str(coarse_task.get("room_id")), {}).get("kind"),
            "room_polygon": deepcopy(rooms_by_id.get(str(coarse_task.get("room_id")), {}).get("polygon", [])),
            "kind": coarse_task.get("kind"),
            "title": coarse_task.get("title"),
            "priority": coarse_task.get("priority", 0),
            "estimated_minutes": int(coarse_task.get("estimated_minutes", 1)),
            "coverage_target_percent": coverage_target,
            "waypoints": waypoints,
            "inspection_targets": inspection_targets,
            "mode": _normalise_mode(coarse_task.get("mode", "ASSISTED")),
            "status": "READY" if str(coarse_task.get("status") or "PENDING") == "PENDING" else str(coarse_task.get("status")),
            "lease_id": _normalise_lease_id(coarse_task.get("lease_id")),
            "lease_state": "LEASED" if coarse_task.get("lease_id") else "UNLEASED",
            "revision": 1,
            "source_revision": int(coarse_task.get("revision", 1)),
            "diagnostics": coarse_task.get("diagnostics"),
        }
        previous = previous_tasks.get(task["task_id"])
        if previous:
            for key in ("status", "mode", "lease_id", "lease_state", "revision", "diagnostics", "coverage_target_percent"):
                if key in previous:
                    task[key] = previous[key]
        if task["mode"] == "AUTONOMOUS" and not task.get("lease_id"):
            raise ValueError(f"mission task {task['task_id']} requires a lease_id for AUTONOMOUS mode")
        tasks.append(task)
    mission = {
        "schema_version": 1,
        "mission_id": mission_id,
        "map_id": coarse.get("map_id"),
        "coordinate_frame": coarse.get("coordinate_frame"),
        "source_coarse_map_sha256": _hash_payload(source_map),
        "state": _mission_state(tasks),
        "tasks": tasks,
        "task_count": len(tasks),
        "ready_task_count": sum(1 for task in tasks if task["status"] == "READY"),
        "active_task_id": next((task["task_id"] for task in tasks if task["status"] == "ACTIVE"), None),
        "current_task_id": next((task["task_id"] for task in tasks if task["status"] in {"ACTIVE", "PAUSED", "READY"}), None),
        "created_at_unix": (previous_mission or {}).get("created_at_unix", time.time()),
        "updated_at_unix": time.time(),
    }
    return mission


def compile_building_mission(
    building_payload: dict[str, Any],
    floor_maps: dict[str, dict[str, Any]],
    *,
    mission_id: str | None = None,
) -> dict[str, Any]:
    """Compile independent floor tasks plus explicit vertical connections.

    Floor maps remain independent canonical artifacts; this aggregate mission
    only namespaces their tasks and adds a vertical segment when endpoint
    coordinates are available.
    """
    if not isinstance(building_payload, dict):
        raise ValueError("building payload must be an object")
    building_id = _identifier(building_payload.get("building_id"), "building_id")
    floors = building_payload.get("floors") or []
    if not isinstance(floors, list):
        raise ValueError("building floors must be a list")
    connections = building_payload.get("connections") or []
    tasks: list[dict[str, Any]] = []
    source_maps: list[dict[str, Any]] = []
    floor_ids: set[str] = set()
    for floor in floors:
        if not isinstance(floor, dict):
            continue
        floor_id = _identifier(floor.get("floor_id"), "floor_id")
        floor_ids.add(floor_id)
        source = floor_maps.get(str(floor.get("map_id")))
        if not isinstance(source, dict):
            continue
        compiled = compile_mission(source, mission_id=f"{building_id}-{floor_id}")
        source_maps.append(source)
        for task in compiled["tasks"]:
            item = deepcopy(task)
            item["task_id"] = f"{floor_id}:{item['task_id']}"
            item["floor_id"] = floor_id
            item["building_id"] = building_id
            tasks.append(item)
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            continue
        from_floor = str(connection.get("from_floor_id") or "")
        to_floor = str(connection.get("to_floor_id") or "")
        if from_floor not in floor_ids or to_floor not in floor_ids or from_floor == to_floor:
            continue
        task_id = f"vertical:{connection.get('id') or index}"
        from_anchor = connection.get("from_anchor")
        to_anchor = connection.get("to_anchor")
        has_endpoints = isinstance(from_anchor, list) and len(from_anchor) == 2 and isinstance(to_anchor, list) and len(to_anchor) == 2
        item = {
            "task_id": task_id,
            "kind": "vertical_connection",
            "title": str(connection.get("label") or connection.get("id") or "vertical connection"),
            "building_id": building_id,
            "floor_id": from_floor,
            "to_floor_id": to_floor,
            "connection_id": str(connection.get("id") or index),
            "connection_kind": str(connection.get("kind") or "stairs"),
            "waypoints": [deepcopy(from_anchor), deepcopy(to_anchor)] if has_endpoints else [],
            "inspection_targets": [],
            "mode": "ASSISTED",
            "status": "READY" if has_endpoints else "BLOCKED",
            "blocker": None if has_endpoints else "VERTICAL_CONNECTION_ENDPOINTS_REQUIRED",
            "lease_id": None,
            "lease_state": "UNLEASED",
            "revision": 1,
            "source_revision": 1,
            "priority": 0,
            "estimated_minutes": max(1, int((connection.get("stair_profile") or {}).get("step_count", 1) / 4)),
        }
        if isinstance(connection.get("stair_profile"), dict):
            item["stair_profile"] = deepcopy(connection["stair_profile"])
        tasks.append(item)
    mission_id = _identifier(mission_id or _generated_mission_id(), "mission_id")
    return {
        "schema_version": 1,
        "mission_id": mission_id,
        "building_id": building_id,
        "map_id": building_id,
        "coordinate_frame": "BUILDING_LOCAL",
        "source_floor_revisions": {str(item.get("floor_id")): int(item.get("revision", 1)) for item in floors if isinstance(item, dict)},
        "source_coarse_map_sha256": _hash_payload(source_maps),
        "state": _mission_state(tasks),
        "tasks": tasks,
        "task_count": len(tasks),
        "ready_task_count": sum(task["status"] == "READY" for task in tasks),
        "active_task_id": next((task["task_id"] for task in tasks if task["status"] == "ACTIVE"), None),
        "current_task_id": next((task["task_id"] for task in tasks if task["status"] in {"ACTIVE", "PAUSED", "READY"}), None),
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }


def update_mission_task(
    mission: dict[str, Any],
    task_id: str,
    *,
    status: Any = _MISSING,
    mode: Any = _MISSING,
    lease_id: Any = _MISSING,
    expected_revision: Any = _MISSING,
) -> dict[str, Any]:
    updated = deepcopy(mission)
    tasks = list(updated.get("tasks") or [])
    for task in tasks:
        if task.get("task_id") != task_id:
            continue
        revision = int(task.get("revision", 1))
        if isinstance(expected_revision, int) and expected_revision > 0 and revision != expected_revision:
            raise ValueError("task revision does not match expected_revision")
        current_status = str(task.get("status") or "READY")
        changed = False
        lease_value = task.get("lease_id")
        if lease_id is not _MISSING:
            lease_value = _normalise_lease_id(lease_id)
            if current_status == "ACTIVE" and task.get("mode") == "AUTONOMOUS" and lease_value is None:
                raise ValueError("AUTONOMOUS task cannot release its lease while active")
            task["lease_id"] = lease_value
            task["lease_state"] = "LEASED" if lease_value else "UNLEASED"
            changed = True
        if mode is not _MISSING:
            normalized_mode = _normalise_mode(mode)
            if current_status == "ACTIVE" and normalized_mode != task.get("mode"):
                raise ValueError("task mode cannot change while task is active")
            if normalized_mode == "AUTONOMOUS" and not lease_value:
                raise ValueError("AUTONOMOUS mode requires a lease_id")
            task["mode"] = normalized_mode
            changed = True
        if status is not _MISSING:
            normalized_status = _normalise_status(status)
            allowed = _TASK_STATUS_TRANSITIONS.get(current_status, set())
            if normalized_status != current_status and normalized_status not in allowed:
                raise ValueError("task status transition is unsupported")
            if normalized_status == "ACTIVE" and task.get("mode") == "AUTONOMOUS" and not task.get("lease_id"):
                raise ValueError("AUTONOMOUS mode requires a lease_id")
            task["status"] = normalized_status
            changed = True
        if changed:
            task["revision"] = revision + 1
        updated["tasks"] = tasks
        updated["state"] = _mission_state(tasks)
        updated["task_count"] = len(tasks)
        updated["ready_task_count"] = sum(1 for item in tasks if item["status"] == "READY")
        updated["active_task_id"] = next((item["task_id"] for item in tasks if item["status"] == "ACTIVE"), None)
        updated["current_task_id"] = next((item["task_id"] for item in tasks if item["status"] in {"ACTIVE", "PAUSED", "READY"}), None)
        updated["updated_at_unix"] = time.time()
        return updated
    raise ValueError("mission task not found")


class MissionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, mission_id: str) -> Path:
        return self.root / f"{_identifier(mission_id, 'mission_id')}.json"

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            missions = []
            for path in sorted(self.root.glob("*.json")):
                payload = _read_json(path, "mission")
                if payload.get("schema_version") != 1:
                    continue
                missions.append(
                    {
                        "mission_id": payload.get("mission_id"),
                        "map_id": payload.get("map_id"),
                        "state": payload.get("state"),
                        "task_count": payload.get("task_count", len(payload.get("tasks") or [])),
                        "updated_at_unix": payload.get("updated_at_unix"),
                    }
                )
            missions.sort(key=lambda mission: float(mission.get("updated_at_unix") or 0.0), reverse=True)
            return missions

    def load(self, mission_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(mission_id)
            if not path.is_file():
                raise KeyError(mission_id)
            payload = _read_json(path, "mission")
            if payload.get("schema_version") != 1:
                raise ValueError("mission schema version is unsupported")
            return payload

    def save(self, mission: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(mission)
        payload["schema_version"] = 1
        payload["mission_id"] = _identifier(payload["mission_id"], "mission_id")
        payload["updated_at_unix"] = time.time()
        with self._lock:
            _write_atomic(self._path(payload["mission_id"]), payload)
            return deepcopy(payload)

    def compile(self, coarse_map_payload: dict[str, Any], *, mission_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if mission_id is None:
                mission = compile_mission(coarse_map_payload)
            else:
                previous = None
                path = self._path(mission_id)
                if path.is_file():
                    previous = self.load(mission_id)
                mission = compile_mission(coarse_map_payload, mission_id=mission_id, previous_mission=previous)
            return self.save(mission)

    def update_task(self, mission_id: str, task_id: str, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            mission = self.load(mission_id)
            updated = update_mission_task(mission, task_id, **kwargs)
            return self.save(updated)
