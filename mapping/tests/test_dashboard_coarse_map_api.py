from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def test_coarse_map_api_supports_map_and_task_lifecycle(tmp_path, monkeypatch):
    # The dashboard store is module-owned; swap it for an isolated temporary
    # file so this contract test never mutates the operator's map.
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    client = TestClient(app)
    response = client.get("/api/coarse-map")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rooms"]
    task_id = payload["tasks"][0]["id"]

    response = client.post(f"/api/coarse-map/tasks/{task_id}", json={"status": "ACTIVE"})
    assert response.status_code == 200
    assert next(task for task in response.json()["tasks"] if task["id"] == task_id)["status"] == "ACTIVE"

    response = client.post(f"/api/coarse-map/tasks/{task_id}", json={"status": "DONE", "coverage_percent": 87})
    assert response.status_code == 200
    assert next(task for task in response.json()["tasks"] if task["id"] == task_id)["coverage_percent"] == 87

    corner_id = next(task["id"] for task in response.json()["tasks"] if task["kind"] == "corners")
    response = client.post(
        f"/api/coarse-map/tasks/{corner_id}",
        json={
            "status": "ACTIVE",
            "diagnostics": {
                "point_to_trajectory_p95_m": 0.6,
                "plane_residual_p95_m": 0.01,
                "vertical_support_count": 100,
                "horizontal_support_count": 100,
            },
        },
    )
    assert response.status_code == 200
    corner = next(task for task in response.json()["tasks"] if task["id"] == corner_id)
    assert corner["status"] == "QUARANTINED"
    assert corner["diagnostics"]["reason"] == "POSE_RESIDUAL_TOO_LARGE"


def test_coarse_map_api_supports_task_mode_and_lease_transitions(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    client = TestClient(app)
    payload = client.get("/api/coarse-map").json()
    task_id = next(task["id"] for task in payload["tasks"] if task["kind"] == "floor_coverage")
    revision = next(task["revision"] for task in payload["tasks"] if task["id"] == task_id)

    response = client.post(
        f"/api/coarse-map/tasks/{task_id}/mode",
        json={"mode": "AUTONOMOUS", "lease_id": "lease-room-1", "expected_revision": revision},
    )
    assert response.status_code == 200
    task = next(task for task in response.json()["tasks"] if task["id"] == task_id)
    assert task["mode"] == "AUTONOMOUS"
    assert task["lease_id"] == "lease-room-1"
    assert task["lease_state"] == "LEASED"
    assert task["revision"] == revision + 1

    response = client.post(
        f"/api/coarse-map/tasks/{task_id}/lease",
        json={"lease_id": "", "expected_revision": task["revision"]},
    )
    assert response.status_code == 200
    task = next(task for task in response.json()["tasks"] if task["id"] == task_id)
    assert task["lease_id"] is None
    assert task["lease_state"] == "UNLEASED"
    assert task["revision"] == revision + 2


def test_coarse_map_api_rejects_mode_changes_while_active(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    client = TestClient(app)
    payload = client.get("/api/coarse-map").json()
    task_id = payload["tasks"][0]["id"]
    revision = payload["tasks"][0]["revision"]

    response = client.post(f"/api/coarse-map/tasks/{task_id}", json={"status": "ACTIVE"})
    assert response.status_code == 200
    active_revision = next(task["revision"] for task in response.json()["tasks"] if task["id"] == task_id)
    assert active_revision == revision + 1

    response = client.post(
        f"/api/coarse-map/tasks/{task_id}/mode",
        json={"mode": "MANUAL", "expected_revision": active_revision},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "task mode cannot change while task is active"


def test_coarse_map_api_requires_a_lease_for_autonomous_mode(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    client = TestClient(app)
    payload = client.get("/api/coarse-map").json()
    task_id = next(task["id"] for task in payload["tasks"] if task["kind"] == "floor_coverage")
    revision = next(task["revision"] for task in payload["tasks"] if task["id"] == task_id)

    response = client.post(
        f"/api/coarse-map/tasks/{task_id}/mode",
        json={"mode": "AUTONOMOUS", "expected_revision": revision},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "AUTONOMOUS mode requires a lease_id"


def test_mission_route_audit_endpoint_returns_navigation_gate(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    coarse_store = CoarseMapStore(tmp_path / "coarse-map.json")
    monkeypatch.setattr(server, "_coarse_map_store", coarse_store)
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    source = {
        "schema_version": 2,
        "map_id": "route-floor",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-1",
        "floor_id": "route-floor",
        "rooms": [{"id": "room-1", "name": "Room 1", "kind": "room", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]]}],
        "corridors": [], "walls": [], "doors": [], "features": [],
        "keep_out_zones": [],
        "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }
    client = TestClient(app)
    mission = client.post("/api/missions", json={"source_coarse_map": source}).json()
    task = next(item for item in mission["tasks"] if item["kind"] == "floor_coverage")

    response = client.post(
        f"/api/missions/{mission['mission_id']}/tasks/{task['task_id']}/route-audit",
        json={"source_coarse_map": source},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["task_id"] == task["task_id"]
    assert report["navigation_eligible"] is True
    assert report["state"] == "READY_FOR_HUMAN_REVIEW"


def test_mission_route_audit_endpoint_blocks_keep_out_route(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    source = {
        "schema_version": 2,
        "map_id": "route-floor",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-1",
        "floor_id": "route-floor",
        "rooms": [{"id": "room-1", "name": "Room 1", "kind": "room", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]]}],
        "corridors": [], "walls": [], "doors": [], "features": [],
        "keep_out_zones": [{"id": "keep-out-1", "polygon": [[2, 0.5], [4, 0.5], [4, 2.5], [2, 2.5]]}],
        "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
        "tasks": [{"id": "room-1:floor_coverage", "room_id": "room-1", "kind": "floor_coverage", "status": "PENDING"}],
    }
    client = TestClient(app)
    mission = client.post("/api/missions", json={"source_coarse_map": source}).json()
    task = next(item for item in mission["tasks"] if item["kind"] == "floor_coverage")
    task["waypoints"] = [[1, 1], [5, 1]]

    response = client.post(
        f"/api/missions/{mission['mission_id']}/tasks/{task['task_id']}/route-audit",
        json={"source_coarse_map": source, "task": task},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["navigation_eligible"] is False
    assert "KEEP_OUT_ZONE_CROSSING" in {finding["code"] for finding in report["findings"]}
