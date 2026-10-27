"""PGTT action-to-Go2 joint target conversion and transition limiting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_GO2_POSE = np.asarray([0.0, 0.9, -1.8] * 4, dtype=np.float32)


class ActionContractError(ValueError):
    pass


def pgtt_joint_targets(action: Any, *, action_scale: float = 0.5) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ActionContractError("PGTT action must contain 12 finite values")
    if np.max(np.abs(values)) > 1.0001:
        raise ActionContractError("PGTT action exceeds tanh range")
    if not 0.0 < action_scale <= 0.5:
        raise ActionContractError("PGTT action scale exceeds verified contract")
    return (DEFAULT_GO2_POSE + values * float(action_scale)).astype(np.float32)


@dataclass
class TargetLimiter:
    maximum_step_rad: float = 0.025

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_step_rad <= 0.05:
            raise ActionContractError("target limiter step is invalid")
        self._last: np.ndarray | None = None

    def reset(self, measured: Any) -> None:
        values = np.asarray(measured, dtype=np.float32).reshape(-1)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ActionContractError("measured target state must contain 12 finite values")
        self._last = values.copy()

    def apply(self, requested: Any) -> np.ndarray:
        if self._last is None:
            raise ActionContractError("target limiter must be reset")
        values = np.asarray(requested, dtype=np.float32).reshape(-1)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ActionContractError("requested targets must contain 12 finite values")
        delta = np.clip(values - self._last, -self.maximum_step_rad, self.maximum_step_rad)
        self._last = self._last + delta
        return self._last.copy()


__all__ = ["ActionContractError", "DEFAULT_GO2_POSE", "TargetLimiter", "pgtt_joint_targets"]
