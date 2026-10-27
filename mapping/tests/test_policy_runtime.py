import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.policy_runtime import (
    GO2_JOINT_ORDER,
    Go2PolicyRuntime,
    PolicyRuntimeError,
)


REQUEST = {"task": "STAIR_ASCEND", "controller": "pgtt-go2", "lease_id": "lease-1", "max_speed_mps": 0.1}


def test_policy_runtime_requires_measured_initial_positions():
    runtime = Go2PolicyRuntime(lambda _obs: [0.0] * 12)
    with pytest.raises(PolicyRuntimeError, match="reset"):
        runtime.step(REQUEST, {}, tick=1, request_id="r1")


def test_policy_runtime_produces_unarmed_ordered_frame():
    runtime = Go2PolicyRuntime(lambda _obs: np.full(12, 0.05))
    runtime.reset([0.0] * 12)
    frame = runtime.step(REQUEST, {"heightmap": []}, tick=1, request_id="r1")
    assert frame.joint_order == GO2_JOINT_ORDER
    assert frame.positions_rad == (0.05,) * 12
    assert frame.transport_authorized is False


def test_policy_runtime_rejects_large_delta_and_watchdog_gap():
    runtime = Go2PolicyRuntime(lambda _obs: [0.3] * 12)
    runtime.reset([0.0] * 12)
    with pytest.raises(PolicyRuntimeError, match="delta"):
        runtime.step(REQUEST, {}, tick=1, request_id="r1")
    runtime.reset([0.0] * 12)
    with pytest.raises(PolicyRuntimeError, match="watchdog"):
        runtime.step(REQUEST, {}, tick=3, request_id="r1")


def test_policy_runtime_rejects_invalid_policy_shape():
    runtime = Go2PolicyRuntime(lambda _obs: [0.0] * 11)
    runtime.reset([0.0] * 12)
    with pytest.raises(PolicyRuntimeError, match="12"):
        runtime.step(REQUEST, {}, tick=1, request_id="r1")


def test_policy_runtime_returns_unarmed_hold_frame_for_safe_stop():
    runtime = Go2PolicyRuntime(lambda _obs: [0.05] * 12)
    runtime.reset([0.2] * 12)
    frame = runtime.safe_stop(request_id="stale-1")
    assert frame.positions_rad == (0.2,) * 12
    assert frame.source_controller == "watchdog-safe-stop"
    assert frame.transport_authorized is False
