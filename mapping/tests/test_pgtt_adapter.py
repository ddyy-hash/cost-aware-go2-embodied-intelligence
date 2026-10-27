import hashlib

import numpy as np
import pytest

from mapping.pc.go2_mapping_locomotion.pgtt_adapter import (
    PgttAdapterError,
    PgttObservationSpec,
    build_observation,
    proposal_from_policy,
    verify_checkpoint,
    load_pgtt_policy,
    compose_policy_input,
)
from mapping.pc.go2_mapping_locomotion.supervisor import TerrainObservation


SPEC = PgttObservationSpec((4, 5), 6)
TERRAIN = TerrainObservation("STAIR_UP", 0.95, 0.12, 0.3, 0.9, 22.0, 5, True, True)


def observation():
    return build_observation(
        heightmap=np.zeros((4, 5)),
        proprioception=[0] * 6,
        command=[0.1, 0.0, 0.0],
        spec=SPEC,
    )


def test_pgtt_observation_is_shape_and_finite_checked():
    assert observation().heightmap.shape == (4, 5)
    with pytest.raises(PgttAdapterError, match="shape"):
        build_observation(heightmap=[[0]], proprioception=[0] * 6, command=[0, 0, 0], spec=SPEC)


def test_pgtt_policy_input_has_the_official_153_value_order():
    obs = build_observation(heightmap=np.zeros((11, 9)), proprioception=[0] * 24, command=[0.1, 0.0, 0.0], spec=PgttObservationSpec((11, 9), 24))
    vector = compose_policy_input(
        obs,
        angular_velocity=[1, 2, 3],
        gravity_orientation=[4, 5, 6],
        phase=list(range(8)),
        gait_frequency=2.0,
        last_action=[0.0] * 12,
    )
    assert vector.shape == (153,)
    assert vector[:6].tolist() == [1, 2, 3, 4, 5, 6]
    assert vector[30:38].tolist() == list(range(8))
    assert np.allclose(vector[150:], [0.1, 0.0, 0.0])


def test_pgtt_proposal_is_high_level_and_bounded():
    result = proposal_from_policy(
        lambda _obs: {"task": "STAIR_ASCEND", "confidence": 0.9}, observation(), terrain=TERRAIN
    )
    assert result["controller"] == "pgtt-go2"
    assert result["executable"] is False
    assert result["requires_operator_confirmation"] is True


def test_pgtt_rejects_low_level_policy_output():
    with pytest.raises(PgttAdapterError, match="low-level"):
        proposal_from_policy(lambda _obs: {"task": "STAIR_ASCEND", "joint": [0] * 12}, observation(), terrain=TERRAIN)


def test_checkpoint_identity_is_verified(tmp_path):
    path = tmp_path / "policy.pt"
    path.write_bytes(b"policy")
    digest = hashlib.sha256(b"policy").hexdigest()
    assert verify_checkpoint(path, digest)["verified"] is True
    with pytest.raises(PgttAdapterError, match="identity"):
        verify_checkpoint(path, "0" * 64)


def test_real_go2_pgtt_checkpoint_loads_and_runs_offline():
    path = "models/pgtt-go2/policy_go2_pgtt_level03_run0"
    digest = "bc1175d241044400fd2d975642bbe44e1e19fcde0da9b159522ead01d4e39189"
    policy = load_pgtt_policy(path, digest)
    output = policy(np.zeros((1, 153), dtype=np.float32))
    assert tuple(output.shape) == (1, 12)
    assert np.isfinite(output.detach().numpy()).all()
    assert policy.pgtt_metadata["input_size"] == 153
