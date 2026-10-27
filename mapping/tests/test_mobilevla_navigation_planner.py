import pytest

from mapping.pc.go2_mapping_vla.navigation_planner import plan_mobilevla_navigation
from mapping.pc.go2_mapping_vla.offline import parse_mobilevla_decision


MAP = {
    "schema_version": 2,
    "map_id": "demo",
    "coordinate_frame": "SESSION_ODOM_LOCAL",
    "building_id": "building-1",
    "floor_id": "floor-1",
    "rooms": [
        {"id": "room-a", "name": "客厅", "kind": "room", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]], "coverage_percent": 0},
        {"id": "room-b", "name": "卧室", "kind": "room", "polygon": [[5, 0], [8, 0], [8, 3], [5, 3]], "coverage_percent": 0},
    ],
    "corridors": [], "walls": [], "doors": [], "features": [],
    "keep_out_zones": [], "anchors": [], "carpets": [], "thresholds": [],
    "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
}


def test_mobilevla_room_intent_generates_room_constrained_waypoints():
    decision = parse_mobilevla_decision('{"intent":"go_to_room","target":"room-b","confidence":0.9,"constraints":{"require_operator_confirmation":true}}')
    plan = plan_mobilevla_navigation(decision, coarse_map=MAP, current_position=[1.0, 1.0])
    assert plan["target"]["id"] == "room-b"
    assert len(plan["waypoints"]) >= 2
    assert plan["geofence"]["frame"] == "SESSION_ODOM_LOCAL"
    assert plan["navigation_eligible"] is True
    assert plan["waypoints"][-1] == pytest.approx([6.5, 1.5])


def test_mobilevla_unknown_target_fails_closed():
    decision = parse_mobilevla_decision('{"intent":"go_to_room","target":"missing","confidence":0.9,"constraints":{"require_operator_confirmation":true}}')
    with pytest.raises(ValueError, match="target"):
        plan_mobilevla_navigation(decision, coarse_map=MAP, current_position=[1.0, 1.0])


def test_mobilevla_plan_never_contains_low_level_commands():
    decision = parse_mobilevla_decision('{"intent":"inspect_room","target":"room-a","confidence":0.9,"constraints":{"require_operator_confirmation":true}}')
    plan = plan_mobilevla_navigation(decision, coarse_map=MAP, current_position=[1.0, 1.0])
    assert "cmd_vel" not in plan
    assert "joint_targets" not in plan
    assert plan["operator_confirmation_required"] is True
