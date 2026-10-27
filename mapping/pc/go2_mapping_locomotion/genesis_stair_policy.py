"""Archived Genesis Go2 stair policy adapter for offline reproduction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GENESIS_STAIR_CHECKPOINT = "third_party/go2-sim2real-locomotion-rl/logs/go2-stairs/model_104000.pt"
GENESIS_STAIR_CHECKPOINT_SHA256 = "ac10f805b60bc3a8fa5b8acb561d1cc3b4b3a22ff12aea14a6d427bec6174cfd"
GENESIS_STAIR_DEFAULT_JOINTS = np.asarray(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)
GENESIS_STAIR_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
GENESIS_STAIR_TORQUE_LIMITS = np.asarray(
    [23.7, 23.7, 45.0] * 4,
    dtype=np.float32,
)


def build_genesis_stair_observation(
    *,
    angular_velocity: Any,
    gravity: Any,
    command: Any,
    joint_position_error: Any,
    joint_velocity: Any,
    applied_action: Any,
) -> np.ndarray:
    """Build the Genesis 49-value actor observation in its trained order."""
    parts = [
        np.asarray(angular_velocity, dtype=np.float32).reshape(3) * 0.25,
        np.asarray(gravity, dtype=np.float32).reshape(3),
        np.asarray(command, dtype=np.float32).reshape(3) * np.asarray([2.0, 2.0, 0.25], dtype=np.float32),
        np.asarray(joint_position_error, dtype=np.float32).reshape(12),
        np.asarray(joint_velocity, dtype=np.float32).reshape(12) * 0.05,
        np.asarray(applied_action, dtype=np.float32).reshape(16),
    ]
    observation = np.concatenate(parts).astype(np.float32)
    if observation.shape != (49,) or not np.isfinite(observation).all():
        raise ValueError("Genesis stair observation must be finite with shape (49,)")
    return observation


def compute_genesis_stair_torque(
    *,
    action: Any,
    joint_positions: Any,
    joint_velocities: Any,
) -> np.ndarray:
    """Reproduce the trained environment's PLS position-to-torque controller."""
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
    velocities = np.asarray(joint_velocities, dtype=np.float32).reshape(-1)
    if values.shape != (16,) or positions.shape != (12,) or velocities.shape != (12,):
        raise ValueError("Genesis stair torque inputs must have shapes (16,), (12,), and (12,)")
    if not np.isfinite(values).all() or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Genesis stair torque inputs must be finite")

    kp_per_leg = np.clip(40.0 + 20.0 * values[12:16], 10.0, 70.0)
    kp = np.repeat(kp_per_leg, 3).astype(np.float32)
    kd = (0.2 * np.sqrt(kp)).astype(np.float32)
    target = GENESIS_STAIR_DEFAULT_JOINTS + 0.25 * values[:12]
    torque = kp * (target - positions) - kd * velocities
    return np.clip(torque, -GENESIS_STAIR_TORQUE_LIMITS, GENESIS_STAIR_TORQUE_LIMITS).astype(np.float32)


class GenesisStairPolicy:
    """Load and run the Genesis actor without importing the Genesis simulator."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        actor_keys = [key for key in state if key.startswith("actor.") and key.endswith(".weight")]
        actor_keys.sort(key=lambda key: int(key.split(".")[1]))
        if len(actor_keys) != 4:
            raise ValueError("Genesis stair checkpoint must contain four actor layers")
        self._weights = [state[key].detach().to(device=device) for key in actor_keys]
        self._biases = [state[key.replace(".weight", ".bias")].detach().to(device=device) for key in actor_keys]
        self.input_size = int(self._weights[0].shape[1])
        self.action_size = int(self._weights[-1].shape[0])
        if self.input_size != 49 or self.action_size != 16:
            raise ValueError("unsupported Genesis stair actor contract")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self._torch
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_size or not np.isfinite(values).all():
            raise ValueError("Genesis stair policy expects finite shape (N, 49) input")
        output = torch.as_tensor(values, dtype=torch.float32, device=self._weights[0].device)
        for index, (weight, bias) in enumerate(zip(self._weights, self._biases)):
            output = torch.nn.functional.linear(output, weight, bias)
            if index < len(self._weights) - 1:
                output = torch.nn.functional.elu(output)
        return output.detach().cpu().numpy().astype(np.float32)


__all__ = [
    "DEFAULT_GENESIS_STAIR_CHECKPOINT",
    "GENESIS_STAIR_CHECKPOINT_SHA256",
    "GENESIS_STAIR_DEFAULT_JOINTS",
    "GENESIS_STAIR_JOINT_ORDER",
    "GENESIS_STAIR_TORQUE_LIMITS",
    "GenesisStairPolicy",
    "build_genesis_stair_observation",
    "compute_genesis_stair_torque",
]
