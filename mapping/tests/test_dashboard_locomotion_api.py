from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def test_locomotion_status_is_truthful_until_robot_adapter_is_verified():
    response = TestClient(app).get("/api/robot/locomotion/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["actuation_authorized"] is False
    assert payload["active_backend"] is None
    backend_ids = {item["id"] for item in payload["backends"]}
    assert "pgtt-go2" in backend_ids
    assert "genesis-go2" not in backend_ids


def test_robot_capabilities_expose_locomotion_without_claiming_hardware_ready():
    payload = TestClient(app).get("/api/robot/capabilities").json()

    assert payload["locomotion"]["enabled"] is False
    assert payload["locomotion"]["actuation_authorized"] is False
    assert payload["locomotion"]["hardware_verified"] is False
