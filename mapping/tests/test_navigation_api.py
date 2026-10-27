from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def test_robot_capabilities_expose_evidence_bound_sensor_and_navigation_contract():
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/api/robot/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["measurement_authority"] is False
    assert payload["transport"]["production"] == "TAILSCALE_DIRECT"
    assert payload["transport"]["ssh_business_path"] is False
    assert payload["sensors"]["lidar"]["topic"] == "/go2/lio/lidar"
    assert payload["sensors"]["lio_pose"]["topic"] == "/go2/lio/pose"
    assert payload["sensors"]["imu"]["topic"] == "/dog_imu_raw"
    assert payload["sensors"]["color"]["topic"] == "/camera/color/image_raw"
    assert payload["sensors"]["depth"]["topic"] == "/camera/aligned_depth_to_color/image_raw"
    assert payload["sensors"]["tf"]["topic"] == "/tf"
    assert payload["navigation"]["nav2"]["verified"] is False
    assert payload["navigation"]["manual_takeover"]["verified"] is False
    assert payload["navigation"]["direct_unitree_gait"] is False


def test_navigation_api_compiles_and_activates_an_autonomous_mission_task(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    monkeypatch.setattr(server, "_navigation_bridge_store", NavigationBridgeStore(tmp_path / "navigation"))

    client = TestClient(app)
    mission = client.post("/api/missions", json={"mission_id": "mission-1"}).json()
    floor_task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")

    mission = client.post(
        f"/api/missions/mission-1/tasks/{floor_task['task_id']}/mode",
        json={"mode": "AUTONOMOUS", "lease_id": "lease-room-1", "expected_revision": floor_task["revision"]},
    ).json()
    floor_task = next(task for task in mission["tasks"] if task["task_id"] == floor_task["task_id"])

    response = client.post(
        "/api/robot/navigation/goal",
        json={"mission_id": "mission-1", "task_id": floor_task["task_id"]},
    )
    assert response.status_code == 200
    goal = response.json()
    assert goal["mission_id"] == "mission-1"
    assert goal["task_id"] == floor_task["task_id"]
    assert goal["state"] == "NAV_ACTIVE"
    assert goal["lease_id"] == "lease-room-1"

    response = client.post("/api/robot/navigation/pause")
    assert response.status_code == 200
    assert response.json()["state"] == "NAV_PAUSED"

    response = client.post("/api/robot/navigation/resume")
    assert response.status_code == 200
    assert response.json()["state"] == "NAV_ACTIVE"

    response = client.post("/api/robot/navigation/cancel")
    assert response.status_code == 200
    assert response.json()["state"] == "CANCELLED"


def test_navigation_api_rejects_autonomous_goal_without_a_lease(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    monkeypatch.setattr(server, "_navigation_bridge_store", NavigationBridgeStore(tmp_path / "navigation"))

    client = TestClient(app)
    response = client.post(
        "/api/robot/navigation/goal",
        json={
            "mode": "AUTONOMOUS",
            "task_status": "READY",
            "task_kind": "floor_coverage",
            "task_revision": 1,
            "waypoints": [[0.5, 0.5], [1.5, 1.0]],
            "geofence": {"frame": "SESSION_ODOM_LOCAL", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
            "estimated_minutes": 1,
        },
    )
    assert response.status_code == 200
    goal = response.json()
    assert goal["state"] == "NAV_BLOCKED"
    assert goal["state_reason"] == "LEASE_REQUIRED"


def test_navigation_api_rejects_mission_route_that_crosses_keep_out_zone(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore

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
    }
    coarse_store = CoarseMapStore(tmp_path / "coarse-map.json")
    coarse_store.save(source)
    monkeypatch.setattr(server, "_coarse_map_store", coarse_store)
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    navigation_store = NavigationBridgeStore(tmp_path / "navigation")
    monkeypatch.setattr(server, "_navigation_bridge_store", navigation_store)
    client = TestClient(app)
    mission = client.post("/api/missions", json={"mission_id": "mission-route-gate"}).json()
    task = next(item for item in mission["tasks"] if item["kind"] == "floor_coverage")
    task["waypoints"] = [[1, 1], [5, 1]]
    mission_path = tmp_path / "missions" / "mission-route-gate.json"
    mission_path.unlink()
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore as _MissionStore
    _MissionStore(tmp_path / "missions").save(mission)

    response = client.post("/api/robot/navigation/goal", json={"mission_id": "mission-route-gate", "task_id": task["task_id"]})

    assert response.status_code == 422
    assert response.json()["detail"]["state"] == "NAVIGATION_ROUTE_BLOCKED"
    assert response.json()["detail"]["route_audit"]["navigation_eligible"] is False
    assert navigation_store.status()["state"] == "NAV_READY"


def test_navigation_api_supports_safety_stop_and_acknowledgement(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    monkeypatch.setattr(server, "_navigation_bridge_store", NavigationBridgeStore(tmp_path / "navigation"))

    client = TestClient(app)
    mission = client.post("/api/missions", json={"mission_id": "mission-1"}).json()
    task = next(task for task in mission["tasks"] if task["kind"] == "floor_coverage")

    mission = client.post(
        f"/api/missions/mission-1/tasks/{task['task_id']}/mode",
        json={"mode": "ASSISTED", "expected_revision": task["revision"]},
    ).json()
    task = next(item for item in mission["tasks"] if item["task_id"] == task["task_id"])
    assert client.post("/api/robot/navigation/goal", json={"mission_id": "mission-1", "task_id": task["task_id"]}).json()["state"] == "NAV_READY"
    assert client.post("/api/robot/navigation/safety-stop", json={"reason": "OBSTACLE"}).json()["state"] == "SAFETY_STOP"
    assert client.post("/api/robot/navigation/acknowledge").json()["state"] == "MANUAL_OVERRIDE"


def test_robot_safety_acknowledge_alias_uses_the_same_fail_closed_transition(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    monkeypatch.setattr(server, "_navigation_bridge_store", NavigationBridgeStore(tmp_path / "navigation"))

    client = TestClient(app)
    blocked = client.post("/api/robot/safety/acknowledge")
    assert blocked.status_code == 422

    stopped = client.post("/api/robot/navigation/safety-stop", json={"reason": "OBSTACLE"})
    assert stopped.status_code == 200
    acknowledged = client.post("/api/robot/safety/acknowledge")
    assert acknowledged.status_code == 200
    assert acknowledged.json()["state"] == "MANUAL_OVERRIDE"
