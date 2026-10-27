import json

import pytest

from mapping.pc.go2_mapping_core.preflight import (
    CheckRecord,
    ExitCode,
    OperationalState,
    OperationResult,
    write_result_atomic,
)


def test_exit_codes_are_stable_and_complete():
    assert {member.value for member in ExitCode} == {0, 2, 3, 4, 5, 6, 7, 8, 9}
    assert ExitCode.SUCCESS.value == 0
    assert ExitCode.SENSOR_PREFLIGHT.value == 5
    assert ExitCode.QUALITY_GATE.value == 7


def test_success_result_requires_a_passing_postcondition():
    with pytest.raises(ValueError, match="postcondition"):
        OperationResult(
            command="preflight",
            state_before=OperationalState.CONNECTED,
            state_after=OperationalState.CAPTURE_READY,
            exit_code=ExitCode.SUCCESS,
            checks=(CheckRecord("required_topics", False, blocking=True, detail="missing IMU"),),
        )


def test_failure_result_requires_a_machine_readable_blocker():
    with pytest.raises(ValueError, match="blocker"):
        OperationResult(
            command="preflight",
            state_before=OperationalState.CONNECTED,
            state_after=OperationalState.PREFLIGHT_FAILED,
            exit_code=ExitCode.SENSOR_PREFLIGHT,
        )


def test_atomic_result_contains_identity_checks_and_diagnostics(tmp_path):
    target = tmp_path / "state" / "preflight.json"
    result = OperationResult(
        command="preflight",
        state_before=OperationalState.CONNECTED,
        state_after=OperationalState.CAPTURE_READY,
        exit_code=ExitCode.SUCCESS,
        checks=(CheckRecord("required_topics", True, detail="all live"),),
        artifacts=("sha256:abc",),
        blockers=(),
        log_paths=("logs/preflight.log",),
        release_hash="a" * 64,
        config_hash="b" * 64,
    )

    write_result_atomic(target, result)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["command"] == "preflight"
    assert payload["exit_code"] == 0
    assert payload["state_before"] == "CONNECTED"
    assert payload["state_after"] == "CAPTURE_READY"
    assert payload["checks"] == [
        {"name": "required_topics", "passed": True, "blocking": True, "detail": "all live"}
    ]
    assert payload["release_hash"] == "a" * 64
    assert payload["config_hash"] == "b" * 64
    assert not list(target.parent.glob("*.partial"))


def test_atomic_writer_replaces_an_existing_complete_result(tmp_path):
    target = tmp_path / "preflight.json"
    first = OperationResult.success("status", OperationalState.CONNECTED)
    second = OperationResult.failure(
        "preflight",
        OperationalState.CONNECTED,
        OperationalState.PREFLIGHT_FAILED,
        ExitCode.ENDPOINT_UNAVAILABLE,
        blocker="ROBOT_UNREACHABLE",
    )

    write_result_atomic(target, first)
    write_result_atomic(target, second)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["command"] == "preflight"
    assert payload["exit_code"] == 4
    assert payload["blockers"] == ["ROBOT_UNREACHABLE"]
