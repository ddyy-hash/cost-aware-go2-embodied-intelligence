"""Convert MobileVLA task labels into requests for the terrain supervisor.

Terrain classification and locomotion authorization stay separate. This
offline bridge has no transport or actuator integration.
"""
from __future__ import annotations

from dataclasses import dataclass

from mapping.pc.go2_mapping_vla.offline import MobileVLADecision

from .supervisor import LocomotionDecision, LocomotionSupervisor, TerrainObservation


_TASKS = {"STAIR_ASCEND", "STAIR_DESCEND", "ROUGH_TERRAIN"}
_TERRAIN_INTENTS = {"inspect_room", "inspect_feature", "explore_frontiers", "go_to_room"}


class VLAControlBridgeError(ValueError):
    """Raised when a semantic decision cannot enter terrain replay."""


@dataclass(frozen=True)
class VLAControlBridgeResult:
    model_intent: str
    terrain_task: str
    supervisor_decision: LocomotionDecision
    executable: bool = False


class MobileVLATerrainCoordinator:
    """Compose high-level VLA intent with an independently qualified terrain."""

    def __init__(self, supervisor: LocomotionSupervisor) -> None:
        if not isinstance(supervisor, LocomotionSupervisor):
            raise TypeError("supervisor must be a LocomotionSupervisor")
        self._supervisor = supervisor

    def request(
        self,
        decision: MobileVLADecision,
        *,
        terrain: TerrainObservation,
        terrain_task: str,
    ) -> VLAControlBridgeResult:
        if not isinstance(decision, MobileVLADecision):
            raise VLAControlBridgeError("MobileVLA decision is invalid")
        if decision.intent not in _TERRAIN_INTENTS:
            raise VLAControlBridgeError("MobileVLA intent cannot enter terrain replay")
        if not decision.requires_operator_confirmation or decision.allow_navigation:
            raise VLAControlBridgeError("MobileVLA decision lacks mandatory operator boundary")
        if terrain_task not in _TASKS:
            raise VLAControlBridgeError("terrain task is unsupported")
        if not isinstance(terrain, TerrainObservation):
            raise VLAControlBridgeError("terrain observation is invalid")

        # Terrain evidence is never inferred from language-model output.
        self._supervisor.update_terrain(terrain)
        supervisor_decision = self._supervisor.request(terrain_task)
        return VLAControlBridgeResult(
            model_intent=decision.intent,
            terrain_task=terrain_task,
            supervisor_decision=supervisor_decision,
        )


__all__ = [
    "MobileVLATerrainCoordinator",
    "VLAControlBridgeError",
    "VLAControlBridgeResult",
]
