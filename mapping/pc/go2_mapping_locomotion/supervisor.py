"""Check whether a complex-terrain task may be sent to a controller.

A READY result contains a bounded request, not a motor command. The Go2 adapter
must still check its watchdog, firmware, actuators, and emergency stop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


_TASKS = {"STAIR_ASCEND", "STAIR_DESCEND", "ROUGH_TERRAIN"}
_QUALIFIED_STATES = {"STAIR_UP", "STAIR_DOWN", "ROUGH_TERRAIN"}
_MAX_SPEED_MPS = 0.12


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MotionLease:
    lease_id: str
    owner: str

    def __post_init__(self) -> None:
        if not self.lease_id.strip() or not self.owner.strip():
            raise ValueError("motion lease requires lease_id and owner")


@dataclass(frozen=True)
class TerrainObservation:
    state: str
    confidence: float
    riser_height_m: float
    tread_depth_m: float
    width_m: float
    pitch_deg: float
    step_count: int
    top_landing_supported: bool
    bottom_landing_supported: bool

    def __post_init__(self) -> None:
        if self.state not in _QUALIFIED_STATES | {"UNSAFE"}:
            raise ValueError(f"unsupported terrain state: {self.state}")
        for name in ("confidence", "riser_height_m", "tread_depth_m", "width_m", "pitch_deg"):
            _finite(getattr(self, name), name)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("terrain confidence must be between 0 and 1")
        if isinstance(self.step_count, bool) or self.step_count < 0:
            raise ValueError("step_count must be a non-negative integer")

    @property
    def qualified(self) -> bool:
        return (
            self.state in _QUALIFIED_STATES
            and self.confidence >= 0.80
            and 0.05 <= self.riser_height_m <= 0.22
            and 0.15 <= self.tread_depth_m <= 0.55
            and 0.45 <= self.width_m <= 1.50
            and 5.0 <= self.pitch_deg <= 40.0
            and self.step_count >= 3
            and self.top_landing_supported
            and self.bottom_landing_supported
        )


@dataclass(frozen=True)
class LocomotionDecision:
    state: str
    command: dict[str, Any] | None
    blocking_reasons: tuple[str, ...]


class LocomotionSupervisor:
    """Authorize only bounded high-level requests for an approved backend."""

    def __init__(self, *, approved_controllers: set[str] | None = None) -> None:
        self._approved_controllers = set(approved_controllers or ())
        self._terrain: TerrainObservation | None = None
        self._operator_confirmed = False
        self._lease: MotionLease | None = None
        self._controller: str | None = None

    def update_terrain(self, terrain: TerrainObservation) -> None:
        if not isinstance(terrain, TerrainObservation):
            raise TypeError("terrain must be a TerrainObservation")
        self._terrain = terrain

    def confirm_operator(self) -> None:
        self._operator_confirmed = True

    def revoke_operator_confirmation(self) -> None:
        self._operator_confirmed = False

    def set_lease(self, lease: MotionLease | None) -> None:
        self._lease = lease

    def select_controller(self, controller_id: str) -> None:
        self._controller = controller_id.strip()

    def request(self, task: str) -> LocomotionDecision:
        if task not in _TASKS:
            raise ValueError(f"unsupported locomotion task: {task}")
        if not self._operator_confirmed:
            return LocomotionDecision("WAITING_OPERATOR", None, ("OPERATOR_CONFIRMATION_REQUIRED",))
        if self._terrain is None or not self._terrain.qualified:
            return LocomotionDecision("BLOCKED", None, ("TERRAIN_NOT_QUALIFIED",))
        if self._lease is None:
            return LocomotionDecision("BLOCKED", None, ("MOTION_LEASE_REQUIRED",))
        if self._controller not in self._approved_controllers:
            return LocomotionDecision("BLOCKED", None, ("NO_APPROVED_CONTROLLER",))
        return LocomotionDecision(
            "READY",
            {
                "task": task,
                "controller": self._controller,
                "max_speed_mps": _MAX_SPEED_MPS,
                "lease_id": self._lease.lease_id,
            },
            (),
        )


__all__ = [
    "LocomotionDecision",
    "LocomotionSupervisor",
    "MotionLease",
    "TerrainObservation",
]
