from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]


def test_genesis_stair_checkpoint_contract_and_inference():
    from mapping.pc.go2_mapping_locomotion.genesis_stair_policy import (
        DEFAULT_GENESIS_STAIR_CHECKPOINT,
        GenesisStairPolicy,
        build_genesis_stair_observation,
    )

    checkpoint = ROOT / DEFAULT_GENESIS_STAIR_CHECKPOINT
    assert checkpoint.is_file()
    policy = GenesisStairPolicy(checkpoint)
    assert policy.input_size == 49
    assert policy.action_size == 16
    observation = build_genesis_stair_observation(
        angular_velocity=np.zeros(3),
        gravity=np.asarray([0.0, 0.0, -1.0]),
        command=np.asarray([0.4, 0.0, 0.0]),
        joint_position_error=np.zeros(12),
        joint_velocity=np.zeros(12),
        applied_action=np.zeros(16),
    )
    action = policy(observation[None, :])
    assert action.shape == (1, 16)
    assert np.isfinite(action).all()


def test_genesis_stair_manifest_is_pinned_and_offline_only():
    import json

    manifest = json.loads((ROOT / "mapping/config/genesis_go2_stair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == "ac10f805b60bc3a8fa5b8acb561d1cc3b4b3a22ff12aea14a6d427bec6174cfd"
    assert manifest["input_size"] == 49
    assert manifest["action_size"] == 16
    assert manifest["num_pos_actions"] == 12
    assert manifest["num_stiffness_actions"] == 4
    assert manifest["executable"] is False
    assert manifest["transport_authorized"] is False
    assert manifest["runtime_enabled"] is False


def test_genesis_stair_torque_matches_trained_pls_contract():
    from mapping.pc.go2_mapping_locomotion.genesis_stair_policy import (
        GENESIS_STAIR_DEFAULT_JOINTS,
        compute_genesis_stair_torque,
    )

    action = np.asarray([1.0, -2.0, 0.0] * 4 + [1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    torque = compute_genesis_stair_torque(
        action=action,
        joint_positions=GENESIS_STAIR_DEFAULT_JOINTS,
        joint_velocities=np.zeros(12, dtype=np.float32),
    )

    np.testing.assert_allclose(
        torque,
        [15.0, -23.7, 0.0, 5.0, -10.0, 0.0, 12.5, -23.7, 0.0, 7.5, -15.0, 0.0],
        atol=1e-5,
    )
