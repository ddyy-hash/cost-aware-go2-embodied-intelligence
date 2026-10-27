from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def _profile(riser_height_m=0.16):
    return {
        "step_count": 5,
        "riser_height_m": riser_height_m,
        "tread_depth_m": 0.28,
        "width_m": 0.82,
        "pitch_deg": 29.7,
        "geometry_confidence": 0.94,
        "top_landing_supported": True,
        "bottom_landing_supported": True,
        "static_scene": True,
        "unobstructed": True,
        "non_hollow": True,
        "straight_flight": True,
    }


def test_stair_survey_api_returns_an_auditable_read_only_qualification(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.stair_survey import StairSurveyStore

    monkeypatch.setattr(server, "_stair_survey_store", StairSurveyStore(tmp_path / "stair-survey.json"))
    response = TestClient(app).post(
        "/api/stairs/survey",
        json={
            "source_identity": {"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "stair-1"},
            "observations": [
                {"timestamp_ns": 1, "profile": _profile(), "imu_fresh": True, "low_state_fresh": True},
                {"timestamp_ns": 2, "profile": _profile(), "imu_fresh": True, "low_state_fresh": True},
                {"timestamp_ns": 3, "profile": _profile(), "imu_fresh": True, "low_state_fresh": True},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "STAIR_QUALIFIED"
    assert payload["provenance"] == "OBSERVED_LIDAR_IMU_LOWSTATE"
    assert payload["measurement_authority"] is False
    assert payload["locomotion_authorized"] is False
    assert payload["motion_lease_required"] is True


def test_stair_survey_latest_artifact_is_retrievable_without_raw_point_clouds(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.stair_survey import StairSurveyStore

    monkeypatch.setattr(server, "_stair_survey_store", StairSurveyStore(tmp_path / "stair-survey.json"))
    client = TestClient(app)
    body = {
        "source_identity": {"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "stair-latest"},
        "observations": [
            {"timestamp_ns": index, "profile": _profile(), "imu_fresh": True, "low_state_fresh": True}
            for index in (1, 2, 3)
        ],
    }

    saved = client.post("/api/stairs/survey", json=body)
    latest = client.get("/api/stairs/survey/latest")

    assert saved.status_code == 200
    assert latest.status_code == 200
    assert latest.json()["state"] == "STAIR_QUALIFIED"
    assert latest.json()["source_identity"]["capture_id"] == "stair-latest"
    assert "points_xyz" not in latest.text
    assert latest.json()["locomotion_authorized"] is False


def test_stair_survey_api_rejects_non_object_payload_without_side_effects(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.stair_survey import StairSurveyStore

    monkeypatch.setattr(server, "_stair_survey_store", StairSurveyStore(tmp_path / "stair-survey.json"))
    response = TestClient(app).post("/api/stairs/survey", json={"observations": []})

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "STAIR_REJECTED"
    assert "SOURCE_IDENTITY_MISSING" in payload["blocking_reasons"]
    assert "INSUFFICIENT_REPEATED_OBSERVATIONS" in payload["blocking_reasons"]
    assert payload["locomotion_authorized"] is False


def test_stair_survey_api_extracts_offline_point_cloud_observations_before_qualification(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.stair_survey import StairSurveyStore

    monkeypatch.setattr(server, "_stair_survey_store", StairSurveyStore(tmp_path / "stair-survey.json"))
    points = []
    for step in range(5):
        start_x = 0.5 + step * 0.28
        for sample in range(12):
            x = start_x + sample * 0.02
            for lateral in range(9):
                points.append([x, -0.4 + lateral * 0.1, step * 0.16])
    for sample in range(20):
        x = sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.0])
        x = 1.9 + sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.64])
    evidence = {
        "top_landing_supported": True,
        "bottom_landing_supported": True,
        "static_scene": True,
        "unobstructed": True,
        "non_hollow": True,
        "straight_flight": True,
    }

    response = TestClient(app).post(
        "/api/stairs/survey",
        json={
            "source_identity": {"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "stair-points"},
            "observations": [
                {"timestamp_ns": index, "points_xyz": points, "imu_fresh": True, "low_state_fresh": True, **evidence}
                for index in (1, 2, 3)
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "STAIR_QUALIFIED"
    assert payload["observation_count"] == 3
    assert payload["profile"]["step_count"] >= 3
    assert payload["locomotion_authorized"] is False


def test_stair_survey_api_fails_closed_when_point_cloud_profile_extraction_fails(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.stair_survey import StairSurveyStore

    monkeypatch.setattr(server, "_stair_survey_store", StairSurveyStore(tmp_path / "stair-survey.json"))
    response = TestClient(app).post(
        "/api/stairs/survey",
        json={
            "source_identity": {"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "bad-stair"},
            "observations": [
                {"timestamp_ns": index, "points_xyz": [[0.0, 0.0, "not-a-number"]], "imu_fresh": True, "low_state_fresh": True}
                for index in (1, 2, 3)
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "STAIR_REJECTED"
    assert "POINT_CLOUD_PROFILE_EXTRACTION_FAILED" in payload["blocking_reasons"]
    assert payload["locomotion_authorized"] is False


def test_stair_replay_cli_reads_closed_captures_and_never_owns_motion():
    from pathlib import Path

    cli = (Path(__file__).resolve().parents[2] / "tools" / "replay_go2_stair_survey.py").read_text(encoding="utf-8")

    assert "CalibrationCapture" in cli
    assert "CalibrationTopicMap" in cli
    assert "topics=topic_map" in cli
    assert "transform_id" in cli
    assert "coordinate_frame" in cli
    assert "measurement_authority" in cli
    assert "artifact_sha256" in cli
    assert "replay_stair_capture" in cli
    assert "StairSurveyStore" in cli
    assert "db3.zstd" in cli
    assert "locomotion" not in cli.lower() or "locomotion_authorized" in cli
    assert "Unitree" not in cli
