"""PGTT inference runner using real Go2 state, without lowcmd output."""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Optional, Union

import numpy as np

from .numpy_pgtt import NumpyPGTTPolicy
from .sdk2_adapter import PGTT_TO_UNITREE_MOTOR_INDEX


class ReadonlyPGTTError(ValueError):
    pass


def gravity_orientation(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ReadonlyPGTTError("IMU quaternion must contain four finite values")
    qw, qx, qy, qz = map(float, q)
    return np.asarray((-2 * (qz * qx - qw * qy), -2 * (qz * qy + qw * qx), -1 + 2 * (qx * qx + qy * qy)), dtype=np.float32)


def heightmap_from_pointcloud(message: Any) -> np.ndarray:
    """Read the official 11x9 row-major elevation PointCloud2 payload."""
    if int(message.width) != 9 or int(message.height) != 11:
        raise ReadonlyPGTTError("elevation heightmap must be 11x9")
    fields = {field.name: field for field in message.fields}
    if "z" not in fields or int(fields["z"].datatype) != 7:
        raise ReadonlyPGTTError("elevation heightmap requires float32 z field")
    offset = int(fields["z"].offset)
    step = int(message.point_step)
    raw = bytes(message.data)
    expected = int(message.width) * int(message.height) * step
    if step <= 0 or len(raw) < expected:
        raise ReadonlyPGTTError("elevation heightmap payload is truncated")
    values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, step)[:, offset:offset + 4].copy()
    result = values.view("<f4").reshape(11, 9)
    if not np.isfinite(result).all():
        raise ReadonlyPGTTError("elevation heightmap contains non-finite values")
    return result.astype(np.float32, copy=False)


def compose_go2_observation(
    *,
    lowstate: Any,
    heightmap: np.ndarray,
    phase: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    default_position: np.ndarray,
    gait_frequency: float = 2.0,
) -> np.ndarray:
    try:
        angular = np.asarray(lowstate.imu_state.gyroscope, dtype=np.float32).reshape(-1)
        gravity = gravity_orientation(lowstate.imu_state.quaternion)
        positions = np.asarray([lowstate.motor_state[i].q for i in PGTT_TO_UNITREE_MOTOR_INDEX], dtype=np.float32)
        velocities = np.asarray([lowstate.motor_state[i].dq for i in PGTT_TO_UNITREE_MOTOR_INDEX], dtype=np.float32)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ReadonlyPGTTError("Go2 lowstate layout is incompatible") from exc
    height = np.asarray(heightmap, dtype=np.float32)
    if height.shape != (11, 9):
        raise ReadonlyPGTTError("heightmap must have shape 11x9")
    phase = np.asarray(phase, dtype=np.float32).reshape(-1)
    last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
    command = np.asarray(command, dtype=np.float32).reshape(-1)
    default_position = np.asarray(default_position, dtype=np.float32).reshape(-1)
    if angular.size != 3 or phase.size != 8 or last_action.size != 12 or command.size != 3 or default_position.size != 12:
        raise ReadonlyPGTTError("PGTT state dimensions are invalid")
    # Match the upstream deploy_real.py preprocessing exactly: terrain height
    # is relative to the lowest observed cell and scaled before inference.
    normalized_height = (height.reshape(-1) - float(np.min(height))) * 1.5
    parts = [angular, gravity, positions - default_position, velocities, phase, normalized_height, [float(gait_frequency)], last_action, command]
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (153,) or not np.isfinite(result).all():
        raise ReadonlyPGTTError("composed Go2 PGTT observation is invalid")
    return result


def run_readonly_inference(policy_path: Union[str, Path], policy_sha256: str, lowstate: Any, heightmap: np.ndarray, *, default_position: np.ndarray, phase: Optional[np.ndarray] = None, last_action: Optional[np.ndarray] = None, command: Optional[np.ndarray] = None) -> np.ndarray:
    policy = NumpyPGTTPolicy(policy_path, policy_sha256)
    observation = compose_go2_observation(
        lowstate=lowstate,
        heightmap=heightmap,
        phase=np.zeros(8, np.float32) if phase is None else phase,
        last_action=np.zeros(12, np.float32) if last_action is None else last_action,
        command=np.zeros(3, np.float32) if command is None else command,
        default_position=default_position,
    )
    return policy(observation[None, :])[0]


__all__ = ["ReadonlyPGTTError", "compose_go2_observation", "gravity_orientation", "heightmap_from_pointcloud", "run_readonly_inference"]
