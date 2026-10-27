import pytest

from mapping.pc.go2_mapping_locomotion.supervisor import (
    LocomotionSupervisor,
    MotionLease,
    TerrainObservation,
)
from mapping.pc.go2_mapping_locomotion.vla_bridge import (
    MobileVLATerrainCoordinator,
    VLAControlBridgeError,
)
from mapping.pc.go2_mapping_vla.offline import parse_mobilevla_decision


def decision():
    return parse_mobilevla_decision(
        {
            "intent": "inspect_room",
            "target": "room_a",
            "confidence": 0.95,
            "constraints": {"require_operator_confirmation": True},
        }
    )


def qualified_stairs():
    return TerrainObservation("STAIR_UP", 0.95, 0.12, 0.30, 0.90, 22.0, 6, True, True)


def test_vla_terrain_bridge_keeps_model_and_terrain_authorities_separate():
    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    supervisor.confirm_operator()
    supervisor.set_lease(MotionLease("lease-1", "operator"))
    supervisor.select_controller("pgtt-go2")

    result = MobileVLATerrainCoordinator(supervisor).request(
        decision(), terrain=qualified_stairs(), terrain_task="STAIR_ASCEND"
    )

    assert result.model_intent == "inspect_room"
    assert result.terrain_task == "STAIR_ASCEND"
    assert result.supervisor_decision.state == "READY"
    assert result.executable is False
    assert "cmd_vel" not in (result.supervisor_decision.command or {})


def test_vla_bridge_stays_blocked_when_terrain_is_not_qualified():
    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    result = MobileVLATerrainCoordinator(supervisor).request(
        decision(),
        terrain=TerrainObservation("UNSAFE", 0.99, 0.35, 0.10, 0.30, 55.0, 4, False, False),
        terrain_task="STAIR_ASCEND",
    )

    assert result.supervisor_decision.state == "WAITING_OPERATOR"
    assert result.supervisor_decision.command is None


def test_vla_bridge_rejects_unsupported_or_stop_semantics():
    supervisor = LocomotionSupervisor()
    bridge = MobileVLATerrainCoordinator(supervisor)
    with pytest.raises(VLAControlBridgeError, match="unsupported"):
        bridge.request(decision(), terrain=qualified_stairs(), terrain_task="JUMP")

    stop = parse_mobilevla_decision(
        {"intent": "stop_and_request_operator", "confidence": 1.0, "constraints": {}}
    )
    with pytest.raises(VLAControlBridgeError, match="cannot enter"):
        bridge.request(stop, terrain=qualified_stairs(), terrain_task="STAIR_ASCEND")
