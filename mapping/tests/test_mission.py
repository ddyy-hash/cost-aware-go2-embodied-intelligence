from pathlib import Path

import pytest

from mapping.pc.go2_mapping_dashboard.coarse_map import build_task_graph


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


def test_mission_compilation_creates_task_geometry_and_defaults(tmp_path):
    from mapping.pc.go2_mapping_dashboard.mission import compile_mission

    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-1")

    assert mission["mission_id"] == "mission-1"
    assert mission["state"] == "READY"
    assert mission["tasks"][0]["kind"] == "perimeter"
    assert mission["tasks"][0]["waypoints"]
    assert mission["tasks"][1]["inspection_targets"]
    assert mission["tasks"][2]["mode"] == "MANUAL"
    assert mission["tasks"][0]["lease_state"] == "UNLEASED"


def test_mission_task_state_machine_enforces_revision_and_mode_leases(tmp_path):
    from mapping.pc.go2_mapping_dashboard.mission import compile_mission, update_mission_task

    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-1")
    task_id = mission["tasks"][0]["task_id"]
    revision = mission["tasks"][0]["revision"]

    mission = update_mission_task(mission, task_id, status="ACTIVE", expected_revision=revision)
    task = next(task for task in mission["tasks"] if task["task_id"] == task_id)
    assert task["status"] == "ACTIVE"
    with pytest.raises(ValueError, match="active"):
        update_mission_task(mission, task_id, mode="AUTONOMOUS", expected_revision=task["revision"])

    mission = compile_mission(build_task_graph(_coarse_map()), mission_id="mission-2")
    task_id = next(task["task_id"] for task in mission["tasks"] if task["kind"] == "floor_coverage")
    mission = update_mission_task(
        mission,
        task_id,
        mode="AUTONOMOUS",
        lease_id="lease-room-1",
        expected_revision=mission["tasks"][0]["revision"],
    )
    task = next(task for task in mission["tasks"] if task["task_id"] == task_id)
    assert task["mode"] == "AUTONOMOUS"
    assert task["lease_id"] == "lease-room-1"
    with pytest.raises(ValueError, match="lease"):
        update_mission_task(mission, task_id, mode="AUTONOMOUS", lease_id="", expected_revision=task["revision"])
def test_building_mission_namespaces_floor_tasks_and_blocks_unbound_vertical_links():
    from mapping.pc.go2_mapping_dashboard.mission import compile_building_mission

    def floor(map_id):
        return {"schema_version": 2, "map_id": map_id, "building_id": "building-a", "floor_id": map_id,
                "coordinate_frame": "BUILDING_LOCAL", "rooms": [{"id": "room-1", "name": map_id,
                "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}], "corridors": [], "walls": [], "doors": [],
                "features": [], "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR",
                "geometry_authority": False, "revision": 1}

    building = {"building_id": "building-a", "floors": [{"floor_id": "floor-01", "map_id": "floor-01", "revision": 1},
                {"floor_id": "floor-02", "map_id": "floor-02", "revision": 1}], "connections": [
                {"id": "stairs-1", "kind": "stairs", "from_floor_id": "floor-01", "to_floor_id": "floor-02",
                 "from_anchor": [1, 1], "to_anchor": [1, 1]},
                {"id": "lift-1", "kind": "elevator", "from_floor_id": "floor-01", "to_floor_id": "floor-02"},
            ]}
    mission = compile_building_mission(building, {"floor-01": floor("floor-01"), "floor-02": floor("floor-02")}, mission_id="building-mission")
    ids = [task["task_id"] for task in mission["tasks"]]
    assert "floor-01:room-1:perimeter" in ids
    assert "floor-02:room-1:perimeter" in ids
    vertical = {task["connection_id"]: task for task in mission["tasks"] if task["kind"] == "vertical_connection"}
    assert vertical["stairs-1"]["status"] == "READY"
    assert vertical["lift-1"]["status"] == "BLOCKED"
    assert vertical["lift-1"]["blocker"] == "VERTICAL_CONNECTION_ENDPOINTS_REQUIRED"
