"""Validate terrain-controller requests before they reach the Go2 adapter.

The SDK adapter is loaded on the robot and remains responsible for its
watchdog and emergency-stop path.
"""
from __future__ import annotations

from typing import Any, Mapping


_TASKS = {"STAIR_ASCEND", "STAIR_DESCEND", "ROUGH_TERRAIN"}
_MAX_SPEED_MPS = 0.12


class RequestGateError(ValueError):
    """Raised when a controller request violates the runtime contract."""


def validate_controller_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RequestGateError("locomotion request must be an object")
    task = payload.get("task")
    controller = payload.get("controller")
    lease_id = payload.get("lease_id")
    if task not in _TASKS:
        raise RequestGateError("unsupported locomotion task")
    if not isinstance(controller, str) or not controller.strip():
        raise RequestGateError("controller is required")
    if not isinstance(lease_id, str) or not lease_id.strip():
        raise RequestGateError("motion lease is required")
    try:
        speed = float(payload.get("max_speed_mps"))
    except (TypeError, ValueError) as exc:
        raise RequestGateError("max_speed_mps must be finite") from exc
    if speed != speed or speed in {float("inf"), float("-inf")} or not 0.0 < speed <= _MAX_SPEED_MPS:
        raise RequestGateError("max_speed_mps exceeds the hardware gate")
    forbidden = {"joint", "joints", "torque", "motor", "motor_command", "footsteps", "gait"}
    if forbidden.intersection(payload):
        raise RequestGateError("low-level actuation fields are not accepted")
    return {
        "task": task,
        "controller": controller.strip(),
        "lease_id": lease_id.strip(),
        "max_speed_mps": speed,
    }


__all__ = ["RequestGateError", "validate_controller_request"]
