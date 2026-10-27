"""Torch-free PGTT Go2 policy inference for the Jetson runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class NumpyPGTTError(ValueError):
    pass


class NumpyPGTTPolicy:
    def __init__(self, path: str | Path, expected_sha256: str) -> None:
        self.path = Path(path)
        expected = expected_sha256.lower().strip()
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.is_file() else ""
        if actual != expected:
            raise NumpyPGTTError("PGTT NumPy weight identity mismatch")
        with np.load(self.path, allow_pickle=False) as bundle:
            self.mean = np.array(bundle["mean"], dtype=np.float32, copy=True)
            self.std = np.array(bundle["std"], dtype=np.float32, copy=True)
            self.weights = [np.array(bundle[f"hidden_{i}_kernel"], dtype=np.float32, copy=True) for i in range(4)]
            self.biases = [np.array(bundle[f"hidden_{i}_bias"], dtype=np.float32, copy=True) for i in range(4)]
        if self.mean.shape != (153,) or self.std.shape != (153,) or self.weights[-1].shape != (128, 24):
            raise NumpyPGTTError("PGTT NumPy bundle shape is incompatible with Go2")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] != 153 or not np.isfinite(values).all():
            raise NumpyPGTTError("PGTT observation must have shape (N, 153) and be finite")
        values = (values - self.mean) / self.std
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            values = values @ weight + bias
            values = values / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
        values = values @ self.weights[-1] + self.biases[-1]
        return np.tanh(values[..., :12]).astype(np.float32, copy=False)


__all__ = ["NumpyPGTTError", "NumpyPGTTPolicy"]
