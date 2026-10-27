"""PGTT-compatible observation and checkpoint boundary.

PGTT's upstream policy implementation remains an external dependency. This
adapter makes the project-side contract explicit without guessing the upstream
network shape or allowing an arbitrary policy output to reach the robot.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .supervisor import TerrainObservation


class PgttAdapterError(ValueError):
    """Raised when a PGTT input, checkpoint, or proposal is invalid."""


def load_pgtt_policy(path: str | Path, expected_sha256: str):
    """Load the upstream Go2 PGTT MLP from a verified local checkpoint."""
    checkpoint = verify_checkpoint(path, expected_sha256)
    try:
        import pickle
        import torch
        import torch.nn as nn
        with Path(path).open("rb") as source:
            payload = pickle.load(source)
    except (ImportError, OSError, pickle.UnpicklingError, AttributeError, EOFError) as exc:
        raise PgttAdapterError("PGTT checkpoint cannot be loaded in this environment") from exc
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise PgttAdapterError("PGTT checkpoint format is unsupported")
    statistics, network_payload, _ = payload
    try:
        mean = np.array(statistics.mean["state"], dtype=np.float32, copy=True)
        std = np.array(statistics.std["state"], dtype=np.float32, copy=True)
        parameters = network_payload["params"]
        layers = [parameters[name] for name in ("hidden_0", "hidden_1", "hidden_2", "hidden_3")]
        weights = [np.array(layer["kernel"], dtype=np.float32, copy=True) for layer in layers]
        biases = [np.array(layer["bias"], dtype=np.float32, copy=True) for layer in layers]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PgttAdapterError("PGTT checkpoint parameters are malformed") from exc
    if mean.shape != (153,) or std.shape != (153,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise PgttAdapterError("PGTT Go2 policy must have a finite 153-value normalization vector")
    if any(weight.ndim != 2 or bias.ndim != 1 for weight, bias in zip(weights, biases)):
        raise PgttAdapterError("PGTT MLP layer tensors are malformed")

    class _PgttMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean))
            self.register_buffer("std", torch.from_numpy(std))
            self.layers = nn.ModuleList(
                [nn.Linear(int(weight.shape[0]), int(weight.shape[1])) for weight in weights]
            )
            for layer, weight, bias in zip(self.layers, weights, biases):
                layer.weight.data.copy_(torch.from_numpy(weight.T))
                layer.bias.data.copy_(torch.from_numpy(bias))

        def forward(self, x):
            x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.mean.device)
            if x.ndim != 2 or x.shape[-1] != 153:
                raise ValueError("PGTT policy input must have shape (N, 153)")
            x = (x - self.mean) / self.std
            for layer in self.layers[:-1]:
                x = torch.nn.functional.silu(layer(x))
            return torch.tanh(self.layers[-1](x)[..., :12])

    model = _PgttMLP().eval()
    model.pgtt_metadata = {
        **checkpoint,
        "input_size": 153,
        "action_size": 12,
        "heightmap_shape": [11, 9],
        "control_frequency_hz": 50,
        "action_scale": 0.5,
    }
    return model


@dataclass(frozen=True)
class PgttObservationSpec:
    heightmap_shape: tuple[int, int]
    proprioception_size: int
    control_frequency_hz: int = 50

    def __post_init__(self) -> None:
        if len(self.heightmap_shape) != 2 or any(value <= 0 for value in self.heightmap_shape):
            raise PgttAdapterError("heightmap_shape must contain two positive dimensions")
        if self.proprioception_size <= 0 or self.control_frequency_hz <= 0:
            raise PgttAdapterError("observation sizes and control frequency must be positive")


@dataclass(frozen=True)
class PgttObservation:
    heightmap: np.ndarray
    proprioception: np.ndarray
    command: tuple[float, float, float]


def compose_policy_input(
    observation: PgttObservation,
    *,
    angular_velocity: Sequence[Any],
    gravity_orientation: Sequence[Any],
    phase: Sequence[Any],
    gait_frequency: float,
    last_action: Sequence[Any],
) -> np.ndarray:
    """Compose the exact 153-value Go2 PGTT observation order."""
    if not isinstance(observation, PgttObservation):
        raise PgttAdapterError("observation must be a PgttObservation")
    parts = []
    for values, size, label in (
        (angular_velocity, 3, "angular_velocity"),
        (gravity_orientation, 3, "gravity_orientation"),
        (observation.proprioception, 24, "proprioception"),
        (phase, 8, "phase"),
        (observation.heightmap.reshape(-1), 99, "heightmap"),
        (last_action, 12, "last_action"),
    ):
        try:
            array = np.asarray(values, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise PgttAdapterError(f"{label} is not numeric") from exc
        if array.size != size or not np.isfinite(array).all():
            raise PgttAdapterError(f"{label} must contain {size} finite values")
        parts.append(array)
    frequency = float(gait_frequency)
    if not np.isfinite(frequency):
        raise PgttAdapterError("gait_frequency must be finite")
    command = np.asarray(observation.command, dtype=np.float32)
    result = np.concatenate((*parts[:2], parts[2][:12], parts[2][12:], parts[3], parts[4], [frequency], parts[5], command))
    if result.shape != (153,) or not np.isfinite(result).all():
        raise PgttAdapterError("composed PGTT input is not a finite 153-value vector")
    return result


def build_observation(
    *,
    heightmap: Sequence[Sequence[Any]],
    proprioception: Sequence[Any],
    command: Sequence[Any],
    spec: PgttObservationSpec,
) -> PgttObservation:
    height = np.asarray(heightmap, dtype=np.float32)
    proprio = np.asarray(proprioception, dtype=np.float32).reshape(-1)
    try:
        command_array = np.asarray(command, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PgttAdapterError("command must contain three finite values") from exc
    if height.shape != spec.heightmap_shape:
        raise PgttAdapterError(f"heightmap shape {height.shape} does not match {spec.heightmap_shape}")
    if proprio.size != spec.proprioception_size:
        raise PgttAdapterError("proprioception size does not match the pinned policy contract")
    if command_array.size != 3 or not np.isfinite(command_array).all():
        raise PgttAdapterError("command must contain three finite values")
    if not np.isfinite(height).all() or not np.isfinite(proprio).all():
        raise PgttAdapterError("observation values must be finite")
    return PgttObservation(height, proprio, tuple(float(value) for value in command_array))


def verify_checkpoint(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    checkpoint = Path(path)
    digest = expected_sha256.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PgttAdapterError("expected_sha256 must be a lowercase SHA-256 digest")
    if not checkpoint.is_file():
        raise PgttAdapterError(f"PGTT checkpoint is unavailable: {checkpoint}")
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    actual = hasher.hexdigest()
    if actual != digest:
        raise PgttAdapterError("PGTT checkpoint identity does not match the pinned digest")
    return {"path": str(checkpoint), "sha256": actual, "verified": True, "local_files_only": True}


def proposal_from_policy(
    policy: Callable[[PgttObservation], Any],
    observation: PgttObservation,
    *,
    terrain: TerrainObservation,
    controller_id: str = "pgtt-go2",
) -> dict[str, Any]:
    """Convert a policy proposal to a supervisor input, never to actuation."""

    proposal = policy(observation)
    if not isinstance(proposal, Mapping):
        raise PgttAdapterError("PGTT policy must return a mapping proposal")
    forbidden = {"joint", "joints", "torque", "motor", "footsteps", "gait", "cmd_vel"}
    if forbidden.intersection(proposal):
        raise PgttAdapterError("PGTT adapter rejects low-level policy output")
    task = proposal.get("task")
    if task not in {"STAIR_ASCEND", "STAIR_DESCEND", "ROUGH_TERRAIN"}:
        raise PgttAdapterError("PGTT policy proposal has no supported terrain task")
    if not terrain.qualified:
        raise PgttAdapterError("terrain is not qualified for PGTT execution")
    return {
        "task": task,
        "controller": controller_id,
        "terrain_state": terrain.state,
        "confidence": float(proposal.get("confidence", 0.0)),
        "requires_operator_confirmation": True,
        "executable": False,
    }


__all__ = [
    "PgttAdapterError",
    "PgttObservation",
    "PgttObservationSpec",
    "build_observation",
    "proposal_from_policy",
    "verify_checkpoint",
    "load_pgtt_policy",
    "compose_policy_input",
]
