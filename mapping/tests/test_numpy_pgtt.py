import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.numpy_pgtt import NumpyPGTTError, NumpyPGTTPolicy


def test_numpy_pgtt_loads_exported_policy_and_runs(tmp_path):
    path = tmp_path / "policy.npz"
    arrays = {"mean": np.zeros(153, np.float32), "std": np.ones(153, np.float32)}
    for i, (left, right) in enumerate(zip((153, 512, 256, 128), (512, 256, 128, 24))):
        arrays[f"hidden_{i}_kernel"] = np.zeros((left, right), np.float32)
        arrays[f"hidden_{i}_bias"] = np.zeros(right, np.float32)
    np.savez(path, **arrays)
    import hashlib
    policy = NumpyPGTTPolicy(path, hashlib.sha256(path.read_bytes()).hexdigest())
    result = policy(np.zeros((1, 153), np.float32))
    assert result.shape == (1, 12)
    assert np.isfinite(result).all()


def test_numpy_pgtt_rejects_identity_mismatch(tmp_path):
    path = tmp_path / "policy.npz"
    path.write_bytes(b"bad")
    with pytest.raises(NumpyPGTTError, match="identity"):
        NumpyPGTTPolicy(path, "0" * 64)
