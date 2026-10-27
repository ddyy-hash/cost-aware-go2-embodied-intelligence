from __future__ import annotations

import pytest

from mapping.pc.go2_mapping_locomotion.supervisor import (
    LocomotionSupervisor,
    MotionLease,
    TerrainObservation,
)


def qualified_stairs() -> TerrainObservation:
    return TerrainObservation(
        state="STAIR_UP",
        confidence=0.92,
        riser_height_m=0.12,
        tread_depth_m=0.30,
        width_m=0.90,
        pitch_deg=22.0,
        step_count=6,
        top_landing_supported=True,
        bottom_landing_supported=True,
    )


def test_supervisor_is_fail_closed_until_operator_and_lease_are_present():
    supervisor = LocomotionSupervisor()
    supervisor.update_terrain(qualified_stairs())

    decision = supervisor.request("STAIR_ASCEND")

    assert decision.state == "WAITING_OPERATOR"
    assert decision.command is None
    assert decision.blocking_reasons == ("OPERATOR_CONFIRMATION_REQUIRED",)


def test_supervisor_requires_explicit_lease_and_approved_backend():
    supervisor = LocomotionSupervisor()
    supervisor.update_terrain(qualified_stairs())
    supervisor.confirm_operator()

    decision = supervisor.request("STAIR_ASCEND")
    assert decision.state == "BLOCKED"
    assert decision.blocking_reasons == ("MOTION_LEASE_REQUIRED",)

    supervisor.set_lease(MotionLease("lease-1", "operator"))
    decision = supervisor.request("STAIR_ASCEND")
    assert decision.state == "BLOCKED"
    assert decision.blocking_reasons == ("NO_APPROVED_CONTROLLER",)


def test_supervisor_emits_only_bounded_controller_request_after_all_gates():
    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    supervisor.update_terrain(qualified_stairs())
    supervisor.confirm_operator()
    supervisor.set_lease(MotionLease("lease-1", "operator"))
    supervisor.select_controller("pgtt-go2")

    decision = supervisor.request("STAIR_ASCEND")

    assert decision.state == "READY"
    assert decision.command == {
        "task": "STAIR_ASCEND",
        "controller": "pgtt-go2",
        "max_speed_mps": 0.12,
        "lease_id": "lease-1",
    }
    assert "joint" not in decision.command
    assert "torque" not in decision.command


def test_supervisor_rejects_unqualified_or_unsafe_terrain():
    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    supervisor.update_terrain(qualified_stairs())
    supervisor.confirm_operator()
    supervisor.set_lease(MotionLease("lease-1", "operator"))
    supervisor.select_controller("pgtt-go2")
    supervisor.update_terrain(
        TerrainObservation(
            state="UNSAFE",
            confidence=0.99,
            riser_height_m=0.35,
            tread_depth_m=0.10,
            width_m=0.30,
            pitch_deg=55.0,
            step_count=4,
            top_landing_supported=False,
            bottom_landing_supported=False,
        )
    )

    decision = supervisor.request("STAIR_ASCEND")

    assert decision.state == "BLOCKED"
    assert decision.command is None
    assert decision.blocking_reasons == ("TERRAIN_NOT_QUALIFIED",)


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unsupported locomotion task"):
        LocomotionSupervisor().request("JUMP")
