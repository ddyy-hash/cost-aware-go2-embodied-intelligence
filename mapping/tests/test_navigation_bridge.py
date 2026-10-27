from mapping.pc.go2_mapping_dashboard.coarse_map import build_task_graph
from mapping.pc.go2_mapping_dashboard.mission import compile_mission, update_mission_task
from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import (
    NavigationBridgeStore,
    compile_explicit_navigation_goal,
    compile_navigation_goal,
)


def _coarse_map():
    return {
        "schema_version": 1,
        "map_id": "demo",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [
            {
                "id": "room-1",
                "name": "Room 1",
                "kind": "room",
                "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "coverage_percent": 0,
            }
        ],
        "features": [
            {"id": "door-1", "kind": "door", "label": "Door", "position": [0, 1.5]},
        ],
    }


def test_navigation_bridge_compiles_a_goal_with_a_geofence_and_estimated_runtime(tmp_path):
    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-1")
    task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")
    goal = compile_navigation_goal(mission, task)

    assert goal["mission_id"] == "mission-1"
    assert goal["task_id"] == task["task_id"]
    assert goal["state"] == "NAV_READY"
    assert goal["state_reason"] == "OPERATOR_CONFIRMATION_REQUIRED"
    assert goal["waypoints"]
    assert goal["geofence"]["polygon"][0] == {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame": "SESSION_ODOM_LOCAL"}
    assert goal["geofence"]["polygon"][2] == {"x": 4.0, "y": 3.0, "yaw": 0.0, "frame": "SESSION_ODOM_LOCAL"}
    assert goal["max_duration_s"] >= 60
    assert goal["max_speed_mps"] > 0


def test_navigation_bridge_requires_a_lease_for_autonomous_goals(tmp_path):
    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-1")
    floor_task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")
    blocked = compile_explicit_navigation_goal(
        {
            "mission_id": "mission-1",
            "task_id": floor_task["task_id"],
            "task_kind": floor_task["kind"],
            "task_status": "READY",
            "task_revision": floor_task["revision"],
            "mode": "AUTONOMOUS",
            "waypoints": floor_task["waypoints"],
            "inspection_targets": floor_task["inspection_targets"],
            "geofence": {"frame": "SESSION_ODOM_LOCAL", "polygon": floor_task["room_polygon"]},
            "safety_revision": mission["source_coarse_map_sha256"],
            "estimated_minutes": floor_task["estimated_minutes"],
        }
    )
    assert blocked["state"] == "NAV_BLOCKED"
    assert blocked["state_reason"] == "LEASE_REQUIRED"

    mission = update_mission_task(
        mission,
        floor_task["task_id"],
        lease_id="lease-room-1",
        mode="AUTONOMOUS",
        expected_revision=floor_task["revision"],
    )
    floor_task = next(task for task in mission["tasks"] if task["task_id"] == floor_task["task_id"])
    ready = compile_navigation_goal(mission, floor_task)
    assert ready["state"] == "NAV_READY"
    assert ready["lease_id"] == "lease-room-1"


def test_navigation_bridge_state_machine_handles_pause_resume_cancel_and_safety(tmp_path):
    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-1")
    task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")
    store = NavigationBridgeStore(tmp_path / "nav")
    current = store.compile_from_mission(mission, task)

    active = store.activate()
    assert active["state"] == "NAV_ACTIVE"

    paused = store.pause()
    assert paused["state"] == "NAV_PAUSED"

    resumed = store.resume()
    assert resumed["state"] == "NAV_ACTIVE"

    stopped = store.safety_stop()
    assert stopped["state"] == "SAFETY_STOP"

    acknowledged = store.acknowledge_safety_stop()
    assert acknowledged["state"] == "MANUAL_OVERRIDE"

    cancelled = store.cancel()
    assert cancelled["state"] == "CANCELLED"
    assert store.status()["state"] == "CANCELLED"
