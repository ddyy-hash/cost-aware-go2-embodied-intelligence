from types import SimpleNamespace

import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.readonly_pgtt_runner import (
    ReadonlyPGTTError,
    compose_go2_observation,
    gravity_orientation,
)


def state():
    return SimpleNamespace(
        imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[1.0, 2.0, 3.0]),
        motor_state=[SimpleNamespace(q=float(i), dq=float(-i)) for i in range(20)],
    )


def test_real_go2_observation_order_is_153_values():
    result = compose_go2_observation(
        lowstate=state(), heightmap=np.zeros((11, 9), np.float32), phase=np.zeros(8),
        last_action=np.zeros(12), command=np.zeros(3), default_position=np.zeros(12),
    )
    assert result.shape == (153,)
    assert result[:6].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, -1.0]


def test_gravity_orientation_rejects_invalid_quaternion():
    with pytest.raises(ReadonlyPGTTError, match="quaternion"):
        gravity_orientation([1.0, 2.0])


def test_pgtt_heightmap_is_normalized_like_upstream_deploy():
    result = compose_go2_observation(
        lowstate=state(), heightmap=np.full((11, 9), 0.4, np.float32), phase=np.zeros(8),
        last_action=np.zeros(12), command=np.zeros(3), default_position=np.zeros(12),
    )
    assert np.allclose(result[38:137], 0.0)
