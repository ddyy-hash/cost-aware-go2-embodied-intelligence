import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.action_contract import (
    ActionContractError,
    DEFAULT_GO2_POSE,
    TargetLimiter,
    pgtt_joint_targets,
)


def test_pgtt_action_is_offset_from_default_pose_with_verified_scale():
    action = np.linspace(-1, 1, 12, dtype=np.float32)
    assert np.allclose(pgtt_joint_targets(action), DEFAULT_GO2_POSE + action * 0.5)


def test_pgtt_action_rejects_invalid_shape_range_and_scale():
    with pytest.raises(ActionContractError, match="12"):
        pgtt_joint_targets([0] * 11)
    with pytest.raises(ActionContractError, match="tanh"):
        pgtt_joint_targets([1.1] * 12)
    with pytest.raises(ActionContractError, match="scale"):
        pgtt_joint_targets([0] * 12, action_scale=0.6)


def test_target_limiter_bounds_policy_and_policy_switch_steps():
    limiter = TargetLimiter(maximum_step_rad=0.02)
    limiter.reset(DEFAULT_GO2_POSE)
    first = limiter.apply(DEFAULT_GO2_POSE + 0.5)
    second = limiter.apply(DEFAULT_GO2_POSE - 0.5)
    assert np.max(np.abs(first - DEFAULT_GO2_POSE)) <= 0.020001
    assert np.max(np.abs(second - first)) <= 0.020001
