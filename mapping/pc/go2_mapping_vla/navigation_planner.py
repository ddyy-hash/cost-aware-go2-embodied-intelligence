"""High-level MobileVLA intent to auditable room-level Nav2 waypoints."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .offline import MobileVLADecision
from mapping.pc.go2_mapping_dashboard.coarse_map import validate_coarse_map
from mapping.pc.go2_mapping_dashboard.mission import audit_mission_route


def _point(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] point")
    point = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label} must be finite")
    return point


def _centroid(polygon: list[list[float]]) -> list[float]:
    area = sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))
    if abs(area) < 1e-9:
        return [sum(point[0] for point in polygon) / len(polygon), sum(point[1] for point in polygon) / len(polygon)]
    cx = sum((polygon[i][0] + polygon[(i + 1) % len(polygon)][0]) * (polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]) for i in range(len(polygon)))
    cy = sum((polygon[i][1] + polygon[(i + 1) % len(polygon)][1]) * (polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]) for i in range(len(polygon)))
    return [cx / (3 * area), cy / (3 * area)]


def plan_mobilevla_navigation(
    decision: MobileVLADecision,
    *,
    coarse_map: Mapping[str, Any],
    current_position: list[float],
) -> dict[str, Any]:
    """Resolve a MobileVLA semantic decision into a safe room-level route."""
    if not isinstance(decision, MobileVLADecision):
        raise ValueError("decision is invalid")
    source_map = validate_coarse_map(dict(coarse_map))
    start = _point(current_position, "current_position")
    spaces = [item for item in (source_map.get("rooms", []) + source_map.get("corridors", [])) if isinstance(item, dict)]
    target = None
    if decision.intent != "stop_and_request_operator":
        target_text = str(decision.target or "")
        target = next((item for item in spaces if str(item.get("id")) == target_text or str(item.get("name")) == target_text), None)
        if target is None and decision.intent == "explore_frontiers":
            target = min(spaces, key=lambda item: float(item.get("coverage_percent", 0.0)), default=None)
        if target is None:
            raise ValueError(f"MobileVLA target room is unavailable: {target_text}")
        polygon = [[float(point[0]), float(point[1])] for point in target["polygon"]]
        goal = _centroid(polygon)
        waypoints = [start, goal]
        # Room-to-room transit starts outside the destination room. Audit the
        # destination arrival against its geofence while retaining the full
        # Nav2 transit route in the returned plan.
        audit_task = {"task_id": f"vla:{target['id']}", "kind": "floor_coverage", "room_id": target["id"], "room_polygon": polygon, "waypoints": [goal, goal]}
        audit = audit_mission_route(audit_task, source_map)
        audit["transit_start"] = start
        audit["transit_route_review"] = "DESTINATION_GEOFENCE_ONLY"
        if not audit.get("navigation_eligible"):
            raise ValueError("MobileVLA route failed safety audit")
        return {
            "schema_version": 1,
            "intent": decision.intent,
            "target": {"id": target["id"], "name": target.get("name"), "kind": target.get("kind", "room")},
            "waypoints": waypoints,
            "geofence": {"frame": source_map["coordinate_frame"], "polygon": polygon},
            "navigation_eligible": True,
            "operator_confirmation_required": True,
            "route_audit": audit,
            "vla": {"confidence": decision.confidence, "reasoning_summary": decision.reasoning_summary},
        }
    return {
        "schema_version": 1,
        "intent": decision.intent,
        "target": None,
        "waypoints": [],
        "geofence": {"frame": source_map["coordinate_frame"], "polygon": []},
        "navigation_eligible": False,
        "operator_confirmation_required": True,
        "route_audit": {"navigation_eligible": False, "findings": [{"code": "OPERATOR_STOP", "severity": "BLOCKER"}]},
    }


__all__ = ["plan_mobilevla_navigation"]
