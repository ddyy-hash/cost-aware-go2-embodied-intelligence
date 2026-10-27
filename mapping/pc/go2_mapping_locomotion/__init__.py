"""Safety-bounded locomotion contracts for Go2 terrain controllers."""

from .supervisor import (
    LocomotionDecision,
    LocomotionSupervisor,
    MotionLease,
    TerrainObservation,
)
from .pgtt_adapter import (
    PgttAdapterError,
    PgttObservation,
    PgttObservationSpec,
    build_observation,
    proposal_from_policy,
    verify_checkpoint,
    load_pgtt_policy,
    compose_policy_input,
)
from .vla_bridge import (
    MobileVLATerrainCoordinator,
    VLAControlBridgeError,
    VLAControlBridgeResult,
)

__all__ = [
    "LocomotionDecision",
    "LocomotionSupervisor",
    "MotionLease",
    "TerrainObservation",
    "PgttAdapterError",
    "PgttObservation",
    "PgttObservationSpec",
    "build_observation",
    "proposal_from_policy",
    "verify_checkpoint",
    "load_pgtt_policy",
    "compose_policy_input",
    "MobileVLATerrainCoordinator",
    "VLAControlBridgeError",
    "VLAControlBridgeResult",
]
