from types import SimpleNamespace

import pytest

from mapping.robot.go2_mapping_locomotion.policy_runtime import JointTargetFrame
from mapping.robot.go2_mapping_locomotion.sdk2_adapter import (
    PGTT_TO_UNITREE_MOTOR_INDEX,
    SDK2AdapterConfig,
    SDK2AdapterError,
    UnitreeSDK2Go2Adapter,
)


def test_sdk2_adapter_is_disarmed_by_default():
    adapter = UnitreeSDK2Go2Adapter()
    assert adapter.config.armed is False
    assert adapter._publisher is None
    assert len(PGTT_TO_UNITREE_MOTOR_INDEX) == 12


def test_sdk2_adapter_rejects_armed_config():
    with pytest.raises(SDK2AdapterError, match="hardware gate"):
        SDK2AdapterConfig(armed=True)


def test_sdk2_adapter_reorders_measured_lowstate_into_policy_order():
    adapter = UnitreeSDK2Go2Adapter()
    adapter._latest_lowstate = SimpleNamespace(
        motor_state=[SimpleNamespace(q=float(index), dq=float(-index)) for index in range(20)]
    )
    positions, velocities = adapter.measured_policy_joint_state()
    assert positions == tuple(float(index) for index in PGTT_TO_UNITREE_MOTOR_INDEX)
    assert velocities == tuple(float(-index) for index in PGTT_TO_UNITREE_MOTOR_INDEX)


def test_sdk2_adapter_close_is_nonblocking_and_stops_state_updates():
    adapter = UnitreeSDK2Go2Adapter()
    adapter._latest_lowstate = object()
    adapter.close()
    assert adapter._closed is True
    assert adapter._publisher is None
