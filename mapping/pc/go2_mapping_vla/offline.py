"""Offline MobileVLA-R1 adapter with a high-level-only action boundary.

The upstream project emits continuous robot actions. This project deliberately
adapts only its semantic decision layer into an ASSISTED Nav2 task. A model
response can never contain or produce velocity, joint, gait, or motor output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", re.ASCII)
_ALLOWED_INTENTS = {
    "inspect_room",
    "inspect_feature",
    "explore_frontiers",
    "go_to_room",
    "stop_and_request_operator",
}
_LOW_LEVEL_FIELDS = {
    "cmd_vel",
    "velocity",
    "velocity_vector",
    "joint_targets",
    "joint_commands",
    "gait",
    "footsteps",
    "motor_commands",
    "torque_commands",
    "action_vector",
}
_FORWARD_RE = re.compile(r"\b(?:move|go)\s+forward\s+(?P<distance>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>cm|m)\b", re.IGNORECASE)
_TURN_RE = re.compile(r"\bturn\s+(?P<direction>left|right)\s+(?P<angle>[0-9]+(?:\.[0-9]+)?)\s*degrees?\b", re.IGNORECASE)


class MobileVLADecisionError(ValueError):
    """Raised when a model response cannot enter the navigation task layer."""


@dataclass(frozen=True)
class MobileVLADecision:
    intent: str
    target: str | None
    confidence: float
    reasoning_summary: str
    constraints: dict[str, Any]
    requires_operator_confirmation: bool = True
    allow_navigation: bool = False


@dataclass(frozen=True)
class MobileVLAActionProposal:
    """A model action description retained for audit, never an execution command."""

    action: str
    distance_m: float | None = None
    angle_deg: float | None = None
    executable: bool = False
    requires_operator_confirmation: bool = True


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MobileVLADecisionError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise MobileVLADecisionError(f"{label} must be finite")
    return result


def _decode_payload(raw: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        payload = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        if "<answer>" in text and "</answer>" in text:
            text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MobileVLADecisionError("MobileVLA response must be JSON") from exc
    else:
        raise MobileVLADecisionError("MobileVLA response must be an object or JSON")
    if not isinstance(payload, Mapping):
        raise MobileVLADecisionError("MobileVLA response must be a JSON object")
    return payload


def parse_mobilevla_decision(raw: str | Mapping[str, Any]) -> MobileVLADecision:
    """Parse a MobileVLA response into a safe semantic decision."""
    payload = _decode_payload(raw)
    low_level = sorted(_LOW_LEVEL_FIELDS.intersection(payload.keys()))
    if low_level:
        raise MobileVLADecisionError(f"low-level action fields are forbidden: {', '.join(low_level)}")
    intent = payload.get("intent")
    if not isinstance(intent, str) or intent not in _ALLOWED_INTENTS:
        raise MobileVLADecisionError("intent is unsupported")
    target = payload.get("target")
    if target is not None:
        if not isinstance(target, str) or _IDENTIFIER.fullmatch(target) is None:
            raise MobileVLADecisionError("target is invalid")
    if intent != "stop_and_request_operator" and target is None:
        raise MobileVLADecisionError("target is required for this intent")
    confidence = _finite_number(payload.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise MobileVLADecisionError("confidence must be between 0 and 1")
    reasoning = payload.get("reasoning_summary", "")
    if not isinstance(reasoning, str) or len(reasoning) > 2000:
        raise MobileVLADecisionError("reasoning_summary is invalid")
    constraints = payload.get("constraints", {})
    if not isinstance(constraints, Mapping):
        raise MobileVLADecisionError("constraints must be an object")
    normalised_constraints = dict(constraints)
    if "max_speed_mps" in normalised_constraints:
        speed = _finite_number(normalised_constraints["max_speed_mps"], "constraints.max_speed_mps")
        if not 0.0 <= speed <= 0.22:
            raise MobileVLADecisionError("constraints.max_speed_mps is outside the assisted limit")
        normalised_constraints["max_speed_mps"] = speed
    confirmation = normalised_constraints.get("require_operator_confirmation", True)
    if not isinstance(confirmation, bool) or not confirmation:
        raise MobileVLADecisionError("operator confirmation is mandatory")
    if payload.get("allow_navigation") is True:
        raise MobileVLADecisionError("MobileVLA cannot authorize navigation")
    return MobileVLADecision(
        intent=intent,
        target=target,
        confidence=confidence,
        reasoning_summary=reasoning,
        constraints=normalised_constraints,
        requires_operator_confirmation=True,
        allow_navigation=False,
    )


def classify_model_response(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    """Record whether a raw model response is safe structured semantics."""
    try:
        decision = parse_mobilevla_decision(raw)
    except MobileVLADecisionError as exc:
        return {
            "schema_version": 1,
            "state": "REJECTED",
            "reason": str(exc),
            "executable": False,
        }
    return {
        "schema_version": 1,
        "state": "ACCEPTED",
        "decision": asdict(decision),
        "executable": False,
    }


def parse_mobilevla_action_proposal(raw: str) -> MobileVLAActionProposal:
    """Parse known RL text actions without creating a robot command."""
    if not isinstance(raw, str) or not raw.strip():
        raise MobileVLADecisionError("action proposal must be non-empty text")
    forward = _FORWARD_RE.search(raw)
    if forward:
        distance = _finite_number(forward.group("distance"), "distance")
        if forward.group("unit").lower() == "cm":
            distance /= 100.0
        if not 0.0 < distance <= 2.0:
            raise MobileVLADecisionError("action proposal distance is outside the bounded range")
        return MobileVLAActionProposal(action="move_forward", distance_m=distance)
    turn = _TURN_RE.search(raw)
    if turn:
        angle = _finite_number(turn.group("angle"), "angle")
        if not 0.0 < angle <= 180.0:
            raise MobileVLADecisionError("action proposal angle is outside the bounded range")
        return MobileVLAActionProposal(action=f"turn_{turn.group('direction').lower()}", angle_deg=angle)
    raise MobileVLADecisionError("action proposal is unsupported or ambiguous")


def decision_to_navigation_payload(
    decision: MobileVLADecision,
    *,
    mission_id: str,
    task_id: str,
    task_revision: int,
    waypoints: list[list[float]],
    geofence: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate a semantic decision into a non-activating assisted goal."""
    if not isinstance(decision, MobileVLADecision):
        raise MobileVLADecisionError("decision is invalid")
    for value, label in ((mission_id, "mission_id"), (task_id, "task_id")):
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise MobileVLADecisionError(f"{label} is invalid")
    if isinstance(task_revision, bool) or not isinstance(task_revision, int) or task_revision <= 0:
        raise MobileVLADecisionError("task_revision must be positive")
    if not waypoints and decision.intent != "stop_and_request_operator":
        raise MobileVLADecisionError("waypoints are required for navigation intent")
    if not isinstance(geofence, Mapping):
        raise MobileVLADecisionError("geofence must be an object")
    speed = float(decision.constraints.get("max_speed_mps", 0.12))
    return {
        "schema_version": 1,
        "mission_id": mission_id,
        "task_id": task_id,
        "task_revision": task_revision,
        "task_kind": decision.intent,
        "task_status": "READY",
        "mode": "ASSISTED",
        "waypoints": waypoints,
        "geofence": dict(geofence),
        "max_speed_mps": min(speed, 0.22),
        "operator_confirmation_required": True,
        "goal_available": False,
        "vla": {
            "intent": decision.intent,
            "target": decision.target,
            "confidence": decision.confidence,
            "reasoning_summary": decision.reasoning_summary,
            "allow_navigation": False,
        },
    }


class OfflineVLAAdapter:
    """Load only a local checkpoint and delegate inference through a local callable."""

    def __init__(self, checkpoint_path: str | Path, infer: Callable[[str, Mapping[str, Any]], Any] | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._infer = infer

    def status(self) -> dict[str, Any]:
        if not self.checkpoint_path.is_dir():
            return {"schema_version": 1, "state": "CHECKPOINT_UNAVAILABLE", "local_files_only": True}
        if self._infer is None:
            return {"schema_version": 1, "state": "INFERENCE_BACKEND_UNAVAILABLE", "local_files_only": True}
        return {"schema_version": 1, "state": "READY", "local_files_only": True}

    def decide(self, instruction: str, *, observation: Mapping[str, Any]) -> MobileVLADecision:
        if not self.checkpoint_path.is_dir():
            raise MobileVLADecisionError("MobileVLA checkpoint is unavailable")
        if self._infer is None:
            raise MobileVLADecisionError("MobileVLA local inference checkpoint/backend is unavailable")
        if not isinstance(instruction, str) or not instruction.strip():
            raise MobileVLADecisionError("instruction is required")
        if not isinstance(observation, Mapping):
            raise MobileVLADecisionError("observation must be an object")
        return parse_mobilevla_decision(self._infer(instruction, observation))
