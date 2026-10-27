from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def test_mission_api_compiles_current_coarse_map_and_lists_tasks(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))

    client = TestClient(app)
    response = client.post("/api/missions", json={"mission_id": "mission-1"})
    assert response.status_code == 200
    mission = response.json()
    assert mission["mission_id"] == "mission-1"
    assert mission["state"] == "READY"
    assert mission["tasks"][0]["waypoints"]
    assert mission["tasks"][0]["lease_state"] == "UNLEASED"

    listed = client.get("/api/missions")
    assert listed.status_code == 200
    assert listed.json()["missions"][0]["mission_id"] == "mission-1"

    tasks = client.get("/api/missions/mission-1/tasks")
    assert tasks.status_code == 200
    assert tasks.json()["tasks"][0]["task_id"] == mission["tasks"][0]["task_id"]

    recompilation = client.post("/api/missions/mission-1/compile", json={})
    assert recompilation.status_code == 200
    assert recompilation.json()["mission_id"] == "mission-1"


def test_mission_api_supports_task_state_mode_and_lease_transitions(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))

    client = TestClient(app)
    mission = client.post("/api/missions", json={"mission_id": "mission-1"}).json()
    perimeter_task = mission["tasks"][0]
    floor_task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")

    response = client.post(
        f"/api/missions/mission-1/tasks/{perimeter_task['task_id']}/activate",
        json={"expected_revision": perimeter_task["revision"]},
    )
    assert response.status_code == 200, response.text
    updated = next(task for task in response.json()["tasks"] if task["task_id"] == perimeter_task["task_id"])
    assert updated["status"] == "ACTIVE"

    response = client.post(
        f"/api/missions/mission-1/tasks/{perimeter_task['task_id']}/pause",
        json={"expected_revision": updated["revision"]},
    )
    assert response.status_code == 200
    paused = next(task for task in response.json()["tasks"] if task["task_id"] == perimeter_task["task_id"])
    assert paused["status"] == "PAUSED"

    response = client.post(
        f"/api/missions/mission-1/tasks/{perimeter_task['task_id']}/resume",
        json={"expected_revision": paused["revision"]},
    )
    assert response.status_code == 200
    resumed = next(task for task in response.json()["tasks"] if task["task_id"] == perimeter_task["task_id"])
    assert resumed["status"] == "ACTIVE"

    response = client.post(
        f"/api/missions/mission-1/tasks/{perimeter_task['task_id']}/complete",
        json={"expected_revision": resumed["revision"]},
    )
    assert response.status_code == 200
    completed = next(task for task in response.json()["tasks"] if task["task_id"] == perimeter_task["task_id"])
    assert completed["status"] == "DONE"

    response = client.post(
        f"/api/missions/mission-1/tasks/{floor_task['task_id']}/mode",
        json={"mode": "AUTONOMOUS", "lease_id": "lease-room-1", "expected_revision": floor_task["revision"]},
    )
    assert response.status_code == 200
    autonomous = next(task for task in response.json()["tasks"] if task["task_id"] == floor_task["task_id"])
    assert autonomous["mode"] == "AUTONOMOUS"
    assert autonomous["lease_id"] == "lease-room-1"

    response = client.post(
        f"/api/missions/mission-1/tasks/{floor_task['task_id']}/lease",
        json={"lease_id": "", "expected_revision": autonomous["revision"]},
    )
    assert response.status_code == 200
    released = next(task for task in response.json()["tasks"] if task["task_id"] == floor_task["task_id"])
    assert released["lease_state"] == "UNLEASED"

    response = client.post(
        f"/api/missions/mission-1/tasks/{floor_task['task_id']}/activate",
        json={"expected_revision": released["revision"]},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "AUTONOMOUS mode requires a lease_id"
