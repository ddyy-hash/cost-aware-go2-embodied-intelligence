import pytest

from mapping.robot.go2_mapping_locomotion.request_gate import (
    RequestGateError,
    validate_controller_request,
)


def test_robot_gate_normalizes_bounded_request():
    assert validate_controller_request(
        {
            "task": "STAIR_ASCEND",
            "controller": " pgtt-go2 ",
            "lease_id": " lease-1 ",
            "max_speed_mps": 0.12,
        }
    ) == {
        "task": "STAIR_ASCEND",
        "controller": "pgtt-go2",
        "lease_id": "lease-1",
        "max_speed_mps": 0.12,
    }


@pytest.mark.parametrize("field", ["joint", "torque", "motor", "footsteps", "gait"])
def test_robot_gate_rejects_low_level_model_output(field):
    payload = {
        "task": "STAIR_ASCEND",
        "controller": "pgtt-go2",
        "lease_id": "lease-1",
        "max_speed_mps": 0.1,
        field: [],
    }
    with pytest.raises(RequestGateError, match="low-level"):
        validate_controller_request(payload)


def test_robot_gate_rejects_speed_above_hardware_envelope():
    with pytest.raises(RequestGateError, match="exceeds"):
        validate_controller_request(
            {"task": "STAIR_ASCEND", "controller": "pgtt-go2", "lease_id": "lease-1", "max_speed_mps": 0.2}
        )
