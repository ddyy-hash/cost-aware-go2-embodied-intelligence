"""Go2 terrain-policy runtime contract, independent of the Unitree SDK.

The runtime validates a controller's numeric output and produces a bounded
joint-target frame for a later SDK transport. It never opens a network socket
or sends an actuator command itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .request_gate import RequestGateError, validate_controller_request


GO2_JOINT_ORDER = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)


class PolicyRuntimeError(ValueError):
    """Raised when a policy frame cannot be safely constructed."""


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    control_frequency_hz: int = 50
    max_joint_delta_rad: float = 0.20
    max_joint_position_rad: float = 3.14
    watchdog_periods: int = 2

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0 or self.watchdog_periods <= 0:
            raise PolicyRuntimeError("runtime frequencies must be positive")
        if not 0.0 < self.max_joint_delta_rad <= self.max_joint_position_rad:
            raise PolicyRuntimeError("joint delta limit is invalid")


@dataclass(frozen=True)
class JointTargetFrame:
    joint_order: tuple[str, ...]
    positions_rad: tuple[float, ...]
    source_controller: str
    request_id: str
    transport_authorized: bool = False


class Go2PolicyRuntime:
    """Run a pre-approved policy behind a high-level request gate."""

    def __init__(self, policy: Callable[[Mapping[str, Any]], Sequence[Any]], *, config: PolicyRuntimeConfig | None = None) -> None:
        self._policy = policy
        self._config = config or PolicyRuntimeConfig()
        self._last_positions: np.ndarray | None = None
        self._last_tick: int | None = None

    def reset(self, positions_rad: Sequence[Any], *, tick: int = 0) -> None:
        positions = self._positions(positions_rad, "initial positions")
        self._last_positions = positions
        self._last_tick = int(tick)

    def step(
        self,
        request: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        tick: int,
        request_id: str,
    ) -> JointTargetFrame:
        normalized = validate_controller_request(request)
        if not isinstance(observation, Mapping):
            raise PolicyRuntimeError("policy observation must be an object")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PolicyRuntimeError("request_id is required")
        if self._last_positions is None or self._last_tick is None:
            raise PolicyRuntimeError("runtime must be reset with measured positions")
        current_tick = int(tick)
        if current_tick <= self._last_tick:
            raise PolicyRuntimeError("policy tick must be strictly monotonic")
        if current_tick - self._last_tick > self._config.watchdog_periods:
            raise PolicyRuntimeError("policy watchdog expired")
        raw = self._policy(observation)
        action = self._positions(raw, "policy output")
        delta = action - self._last_positions
        if np.max(np.abs(delta)) > self._config.max_joint_delta_rad:
            raise PolicyRuntimeError("policy joint delta exceeds runtime limit")
        if np.max(np.abs(action)) > self._config.max_joint_position_rad:
            raise PolicyRuntimeError("policy joint position exceeds runtime limit")
        self._last_positions = action
        self._last_tick = current_tick
        return JointTargetFrame(
            joint_order=GO2_JOINT_ORDER,
            positions_rad=tuple(float(value) for value in action),
            source_controller=normalized["controller"],
            request_id=request_id.strip(),
            transport_authorized=False,
        )

    def safe_stop(self, *, request_id: str = "watchdog-stop") -> JointTargetFrame:
        """Return an unarmed hold-position frame after a stale-input fault."""
        if self._last_positions is None:
            raise PolicyRuntimeError("runtime must be reset before safe stop")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PolicyRuntimeError("request_id is required")
        return JointTargetFrame(
            joint_order=GO2_JOINT_ORDER,
            positions_rad=tuple(float(value) for value in self._last_positions),
            source_controller="watchdog-safe-stop",
            request_id=request_id.strip(),
            transport_authorized=False,
        )

    @staticmethod
    def _positions(values: Sequence[Any], label: str) -> np.ndarray:
        try:
            positions = np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise PolicyRuntimeError(f"{label} must contain 12 finite values") from exc
        if positions.size != len(GO2_JOINT_ORDER) or not np.isfinite(positions).all():
            raise PolicyRuntimeError(f"{label} must contain 12 finite values")
        return positions


__all__ = [
    "GO2_JOINT_ORDER",
    "Go2PolicyRuntime",
    "JointTargetFrame",
    "PolicyRuntimeConfig",
    "PolicyRuntimeError",
]
