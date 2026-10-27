import json

from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def test_vla_status_exposes_local_only_checkpoint_and_action_boundary():
    response = TestClient(app).get("/api/robot/vla/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["local_files_only"] is True
    assert payload["action_boundary"] == "ASSISTED_NAV2_INTENT_ONLY"
    assert payload["checkpoint_state"] == "READY"
    assert payload["preferred_variant"] == "rl"
    assert payload["local_checkpoint_path"].endswith("weight/rl")
    assert payload["variant_capabilities"]["rl"]["structured_navigation"] is True
    assert payload["variant_capabilities"]["sft"]["structured_navigation"] is False
    assert payload["adapter"]["state"] == "INFERENCE_BACKEND_UNAVAILABLE"


def test_vla_intent_endpoint_returns_non_activating_assisted_nav_payload():
    response = TestClient(app).post(
        "/api/robot/vla/intent",
        json={
            "response": json.dumps(
                {
                    "intent": "inspect_room",
                    "target": "room_a",
                    "confidence": 0.88,
                    "reasoning_summary": "Inspect the mapped room perimeter.",
                    "constraints": {"max_speed_mps": 0.12, "require_operator_confirmation": True},
                }
            ),
            "mission_id": "mission-1",
            "task_id": "task-room-a",
            "task_revision": 1,
            "waypoints": [[1.0, 1.0], [2.0, 1.0]],
            "geofence": {"frame": "SESSION_ODOM_LOCAL", "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"]["intent"] == "inspect_room"
    assert payload["navigation"]["mode"] == "ASSISTED"
    assert payload["navigation"]["task_status"] == "READY"
    assert payload["navigation"]["goal_available"] is False
    assert payload["navigation"]["operator_confirmation_required"] is True
    assert "cmd_vel" not in payload["navigation"]


def test_vla_intent_endpoint_rejects_low_level_commands():
    response = TestClient(app).post(
        "/api/robot/vla/intent",
        json={"response": '{"intent":"inspect_room","target":"room_a","confidence":0.9,"cmd_vel":{"x":1}}'},
    )

    assert response.status_code == 422
    assert "low-level action" in response.json()["detail"]


def test_vla_intent_endpoint_can_plan_waypoints_from_current_coarse_map():
    response = TestClient(app).post(
        "/api/robot/vla/intent",
        json={
            "response": json.dumps({
                "intent": "go_to_room",
                "target": "room-1",
                "confidence": 0.91,
                "constraints": {"require_operator_confirmation": True},
            }),
            "auto_plan": True,
            "current_position": [1.0, 1.0],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["planning"]["navigation_eligible"] is True
    assert len(payload["navigation"]["waypoints"]) >= 2
    assert payload["navigation"]["mode"] == "ASSISTED"
