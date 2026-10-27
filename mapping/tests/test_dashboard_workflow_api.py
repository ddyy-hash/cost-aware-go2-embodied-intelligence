from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def _v2_map():
    return {
        "schema_version": 2,
        "map_id": "floor-01",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-01",
        "floor_id": "floor-01",
        "rooms": [{"id": "room-1", "name": "Room 1", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}],
        "corridors": [],
        "walls": [],
        "doors": [],
        "features": [],
        "keep_out_zones": [],
        "anchors": [],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 1,
    }


def test_coverage_api_returns_task_and_room_progress(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    monkeypatch.setattr(server, "_coarse_map_store", CoarseMapStore(tmp_path / "coarse-map.json"))
    monkeypatch.setattr(server, "_mission_store", MissionStore(tmp_path / "missions"))
    client = TestClient(app)

    mission = client.post("/api/missions", json={"source_coarse_map": _v2_map()}).json()
    task = mission["tasks"][0]
    activated = client.post(
        f"/api/missions/{mission['mission_id']}/tasks/{task['task_id']}/activate",
        json={"expected_revision": task["revision"]},
    )
    assert activated.status_code == 200
    task = next(item for item in activated.json()["tasks"] if item["task_id"] == task["task_id"])
    client.post(
        f"/api/missions/{mission['mission_id']}/tasks/{task['task_id']}/complete",
        json={"expected_revision": task["revision"]},
    )

    response = client.get(f"/api/coverage/{mission['mission_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mission_id"] == mission["mission_id"]
    assert payload["task_count"] == len(mission["tasks"])
    assert payload["tasks"][0]["coverage_state"] == "VERIFIED"
    assert payload["rooms"][0]["room_id"] == "room-1"
    assert payload["rooms"][0]["coverage_percent"] == 33.33


def test_appearance_aliases_fail_closed_with_explicit_surface_blockers(monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    monkeypatch.setattr(server, "_active_derived_surface_preview", lambda: None)
    monkeypatch.setattr(server, "_legacy_surface_preview", None)
    monkeypatch.setattr(server.dense_mapper, "measured_surface_snapshot", lambda: None)
    client = TestClient(app)

    status = client.get("/api/appearance/measured-surface/status")
    assert status.status_code == 200
    assert status.json()["delivery_authorized"] is False
    assert "MEASURED_SURFACE_REQUIRED" in status.json()["blockers"]

    validation = client.post("/api/appearance/measured-surface/validate", json={})
    assert validation.status_code == 423
    assert "MEASURED_SURFACE_REQUIRED" in validation.json()["detail"]["blockers"]


def test_glb_export_api_returns_423_without_formal_appearance_authority(monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    client = TestClient(app)

    response = client.post("/api/exports/glb", json={"kind": "building"})

    assert response.status_code == 423
    detail = response.json()["detail"]
    assert detail["state"] == "FORMAL_EXPORT_BLOCKED"
    assert "BUILDING_APPEARANCE_UNAVAILABLE" in detail["blockers"]


def test_segment_quality_api_exposes_unknown_evidence_without_fabricating_acceptance(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    monkeypatch.setattr(server, "_capture_inbox_root", Path(tmp_path))
    monkeypatch.setattr(server, "_reconstruction_root", Path(tmp_path))
    monkeypatch.setattr(server, "robot_session", type("Session", (), {"snapshot": staticmethod(lambda: {"session_id": None})}))
    monkeypatch.setattr(server, "_automatic_reconstruction_status_path", None)
    monkeypatch.setattr(server, "_session_odom_local_status_path", None)
    client = TestClient(app)

    response = client.get("/api/reconstruction/segments")

    assert response.status_code == 200
    assert response.json()["segments"] == []

    missing = client.get("/api/reconstruction/segments/segment-20260812T083241Z/quality")
    assert missing.status_code == 404


def test_segment_replay_requires_complete_verified_closure(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    session_id = "segment-20260812T083241Z"
    source_root = tmp_path / "lidar-imu-rgbd" / session_id
    source_root.mkdir(parents=True)
    (source_root / f"{session_id}_0.db3.zstd").write_bytes(b"closed-part")
    monkeypatch.setattr(server, "_capture_inbox_root", tmp_path)
    monkeypatch.setattr(server, "_reconstruction_root", tmp_path / "reconstruction")
    client = TestClient(app)

    response = client.post(f"/api/reconstruction/segments/{session_id}/replay", json={})

    assert response.status_code == 423
    detail = response.json()["detail"]
    assert detail["state"] == "REPLAY_BLOCKED"
    assert "COMPLETE_CHECKSUM_MANIFEST_REQUIRED" in detail["blockers"]


def test_room_scan_start_uses_fixed_local_capture_contract_and_operator_token(monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    calls = []

    class FakeRoomScanService:
        def start(self, duration_seconds, operator_token):
            calls.append((duration_seconds, operator_token))
            return {"request_id": "request-1", "session_id": "segment-20260827T120000Z", "state": "RUNNING"}

    monkeypatch.setattr(server, "_room_scan_service", FakeRoomScanService())
    response = TestClient(app).post(
        "/api/room-scan/start",
        json={"duration_seconds": 600, "operator_token": "short-lived-token"},
    )

    assert response.status_code == 200
    assert calls == [(600, "short-lived-token")]
    assert response.json()["state"] == "RUNNING"
    assert response.json()["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert response.json()["mode"] == "full"
    assert response.json()["geometry_only_calibration"] is True


def test_room_scan_start_rejects_missing_or_invalid_operator_token(monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    class FakeRoomScanService:
        def start(self, duration_seconds, operator_token):
            raise AssertionError("invalid requests must not reach the service")

    monkeypatch.setattr(server, "_room_scan_service", FakeRoomScanService())
    client = TestClient(app)

    missing = client.post("/api/room-scan/start", json={"duration_seconds": 600})
    assert missing.status_code == 401

    invalid = client.post(
        "/api/room-scan/start",
        json={"duration_seconds": 59, "operator_token": "short-lived-token"},
    )
    assert invalid.status_code == 422
