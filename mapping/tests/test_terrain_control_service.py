from __future__ import annotations

import json
from pathlib import Path
import os

from mapping.robot.go2_mapping_locomotion.terrain_control_service import (
    TerrainControlSupervisor,
)
from mapping.robot.go2_mapping_lio.go2_mapping_lio.session_control import (
    SessionControlApplication,
    SessionController,
)


def test_status_is_blocked_when_motion_helper_is_unavailable(tmp_path: Path):
    supervisor = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )
    (tmp_path / "terrain-router-status.json").write_text(
        json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}),
        encoding="utf-8",
    )

    result = supervisor.status()

    assert result["state"] == "ENHANCED_WALK_BLOCKED"
    assert result["reason"] == "VENDOR_SPORT_INTERFACE_UNAVAILABLE"
    assert result["transport_authorized"] is False


def test_arm_requires_field_confirmation_and_reports_native_contract(tmp_path: Path):
    router = tmp_path / "terrain-router-status.json"
    router.write_text(
        json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}),
        encoding="utf-8",
    )
    supervisor = TerrainControlSupervisor(
        status_path=router,
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )

    result = supervisor.arm("wrong")

    assert result["state"] == "NATIVE_READY"
    assert result["reason"] == "FIELD_CONFIRMATION_REQUIRED"
    assert supervisor.status()["transport_authorized"] is False


def test_stop_is_idempotent_and_never_authorizes_transport(tmp_path: Path):
    supervisor = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )

    first = supervisor.stop("operator-stop")
    second = supervisor.stop("operator-stop")

    assert first["state"] == "NATIVE_READY"
    assert first["reason"] == "operator-stop"
    assert second["sequence"] == first["sequence"] + 1
    assert second["transport_authorized"] is False


def test_terrain_facade_is_exposed_through_existing_authenticated_control(tmp_path: Path):
    terrain = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )
    app = SessionControlApplication(
        SessionController(tmp_path / "sessions"),
        "t" * 32,
        "192.0.2.10",
        lambda: {"ready": True},
        terrain,
    )
    headers = {"Authorization": "Bearer " + "t" * 32}

    status, payload = app.handle("GET", "/v1/terrain/status", headers, b"", "192.0.2.10")

    assert status == 200
    assert payload["state"] == "ENHANCED_WALK_BLOCKED"
    assert payload["vendor_sport_available"] is False


def test_high_level_request_is_bounded_and_fail_closed(tmp_path: Path):
    terrain = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )
    app = SessionControlApplication(
        SessionController(tmp_path / "sessions"),
        "t" * 32,
        "192.0.2.10",
        lambda: {"ready": True},
        terrain,
    )
    headers = {"Authorization": "Bearer " + "t" * 32}
    body = json.dumps({
        "task": "STAIR_ASCEND",
        "controller": "pgtt-go2",
        "lease_id": "lease-1",
        "max_speed_mps": 0.12,
    }).encode()

    status, payload = app.handle("POST", "/v1/terrain/request", headers, body, "192.0.2.10")

    assert status == 200
    assert payload["state"] == "ENHANCED_WALK_BLOCKED"
    assert payload["reason"] == "MOTION_LEASE_UNAVAILABLE"
    assert payload["request"]["task"] == "STAIR_ASCEND"


def test_high_level_request_persists_its_fail_closed_status_for_later_callers(tmp_path: Path):
    router = tmp_path / "terrain-router-status.json"
    adapter = tmp_path / "vendor-adapter"
    router.write_text(
        json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}),
        encoding="utf-8",
    )
    adapter.write_text("ready", encoding="utf-8")
    adapter.chmod(adapter.stat().st_mode | os.X_OK)
    terrain = TerrainControlSupervisor(
        status_path=router,
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
        vendor_adapter=adapter,
        executor=tmp_path / "missing-executor",
    )

    result = terrain.request({
        "task": "STAIR_ASCEND",
        "controller": "pgtt-go2",
        "lease_id": "lease-1",
        "max_speed_mps": 0.12,
    })

    assert result["state"] == "ENHANCED_WALK_BLOCKED"
    assert result["reason"] == "MOTION_LEASE_UNAVAILABLE"
    persisted = terrain.status()
    assert persisted["state"] == "ENHANCED_WALK_BLOCKED"
    assert persisted["reason"] == "MOTION_LEASE_UNAVAILABLE"
    request_file = tmp_path / "terrain-control-request.json"
    request = json.loads(request_file.read_text(encoding="utf-8"))
    assert request["task"] == "STAIR_ASCEND"
    assert request["controller"] == "pgtt-go2"
    assert request["max_speed_mps"] == 0.12


def test_high_level_request_rejects_low_level_fields(tmp_path: Path):
    terrain = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=tmp_path / "missing-helper",
    )
    app = SessionControlApplication(
        SessionController(tmp_path / "sessions"), "t" * 32, "192.0.2.10", lambda: {"ready": True}, terrain
    )
    headers = {"Authorization": "Bearer " + "t" * 32}
    body = json.dumps({
        "task": "STAIR_ASCEND", "controller": "pgtt-go2", "lease_id": "lease-1",
        "max_speed_mps": 0.12, "joint": [0.0],
    }).encode()

    status, payload = app.handle("POST", "/v1/terrain/request", headers, body, "192.0.2.10")

    assert status == 422
    assert payload["error"] == "TERRAIN_REQUEST_INVALID"


def test_status_normalizes_legacy_state_records_with_vendor_capability(tmp_path: Path):
    control = tmp_path / "terrain-control-status.json"
    control.write_text(json.dumps({"schema_version": 1, "sequence": 4, "state": "NATIVE_READY"}), encoding="utf-8")
    supervisor = TerrainControlSupervisor(
        status_path=tmp_path / "terrain-router-status.json",
        control_status_path=control,
        motion_helper=tmp_path / "missing-helper",
    )

    result = supervisor.status()

    assert result["sequence"] == 4
    assert result["vendor_sport_available"] is False
    assert result["motion_helper_available"] is False


def test_vendor_sdk_ready_is_distinct_from_motion_lease_ready(tmp_path: Path):
    helper = tmp_path / "motion-helper"
    adapter = tmp_path / "vendor-adapter"
    helper.write_text("ready", encoding="utf-8")
    adapter.write_text("ready", encoding="utf-8")
    helper.chmod(helper.stat().st_mode | os.X_OK)
    adapter.chmod(adapter.stat().st_mode | os.X_OK)
    router = tmp_path / "terrain-router-status.json"
    router.write_text(json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}), encoding="utf-8")
    supervisor = TerrainControlSupervisor(
        status_path=router,
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=helper,
        vendor_adapter=adapter,
    )

    status = supervisor.status()
    armed = supervisor.arm("GO2_PGTT_FIELD_TEST_CONFIRMED")

    assert status["vendor_sport_available"] is True
    assert status["motion_lease_available"] is False
    assert armed["state"] == "ENHANCED_WALK_BLOCKED"
    assert armed["reason"] == "EXECUTOR_UNAVAILABLE"


def test_arm_starts_native_executor_and_stop_ends_it(tmp_path: Path):
    router = tmp_path / "terrain-router-status.json"
    router.write_text(json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}), encoding="utf-8")
    helper = tmp_path / "motion-helper"
    adapter = tmp_path / "vendor-adapter"
    executor = Path("/bin/true")
    for path in (helper, adapter):
        path.write_text("ready", encoding="utf-8")
        path.chmod(path.stat().st_mode | os.X_OK)
    supervisor = TerrainControlSupervisor(
        status_path=router,
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=helper,
        vendor_adapter=adapter,
        executor=executor,
    )

    armed = supervisor.arm("GO2_PGTT_FIELD_TEST_CONFIRMED")
    stopped = supervisor.stop("test-stop")

    assert armed["state"] == "HANDOFF_PREP"
    assert armed["motion_lease_available"] is False
    assert stopped["state"] == "NATIVE_READY"
    assert stopped["motion_lease_available"] is False


def test_arm_remains_blocked_without_a_real_motion_helper(tmp_path: Path):
    router = tmp_path / "terrain-router-status.json"
    router.write_text(
        json.dumps({"state": "STAIR_UP", "executable": True, "policy_level": "10"}),
        encoding="utf-8",
    )
    helper = tmp_path / "motion-helper"
    helper.write_text("ready", encoding="utf-8")
    supervisor = TerrainControlSupervisor(
        status_path=router,
        control_status_path=tmp_path / "terrain-control-status.json",
        motion_helper=helper,
    )

    result = supervisor.arm("GO2_PGTT_FIELD_TEST_CONFIRMED")

    assert result["state"] == "ENHANCED_WALK_BLOCKED"
    assert result["reason"] == "VENDOR_SPORT_INTERFACE_UNAVAILABLE"
    assert result["transport_authorized"] is False
