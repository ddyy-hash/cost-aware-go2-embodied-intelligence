import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.joint import (
    JointCapture,
    solve_joint_diagnostic,
)
from mapping.pc.go2_mapping_calibration.models import (
    CalibrationCandidate,
    MotionExcitation,
    PlaneObservation,
)


class _Solver:
    solver_version = "joint-test-solver"

    def __init__(self):
        self.calls = []

    def solve(self, camera, lidar, motion, **_kwargs):
        self.calls.append((camera, lidar, motion))
        transform = np.eye(4)
        transform[:3, 3] = (0.1, -0.2, 0.3)
        return CalibrationCandidate(
            "PROVISIONAL_AUTOCAL",
            transform,
            2.5,
            120.0,
            0.4,
            (),
            self.solver_version,
        )


def _viable_motion():
    return MotionExcitation(
        cumulative_rotation_deg=42.0,
        rotation_axes=2,
        translation_m=0.8,
        angular_speed_duration_s=1.2,
        lio_epoch_count=1,
        robust_translation_span_m=0.8,
        robust_rotation_span_deg=42.0,
    )


def _planes(start_time_s):
    return (
        PlaneObservation(start_time_s, np.asarray((1.0, 0.0, 0.0)), 1.0),
        PlaneObservation(start_time_s + 0.05, np.asarray((0.0, 1.0, 0.0)), 1.1),
    )


def test_joint_diagnostic_rebases_sessions_and_never_grants_calibration_authority():
    solver = _Solver()
    first = JointCapture(
        capture_id="first",
        artifact_sha256="a" * 64,
        camera_planes=_planes(100.0),
        lidar_planes=_planes(100.01),
        motion=_viable_motion(),
    )
    second = JointCapture(
        capture_id="second",
        artifact_sha256="b" * 64,
        camera_planes=_planes(20_000.0),
        lidar_planes=_planes(20_000.01),
        motion=_viable_motion(),
    )

    result = solve_joint_diagnostic((first, second), solver=solver)

    assert result.status == "JOINT_DIAGNOSTIC_PROVISIONAL"
    assert result.calibration_authority is False
    assert result.manifest_written is False
    assert result.source_artifact_sha256 == ("a" * 64, "b" * 64)
    assert result.camera_from_lidar[:3, 3] == pytest.approx((0.1, -0.2, 0.3))
    assert len(solver.calls) == 1
    camera, lidar, _motion = solver.calls[0]
    assert max(item.recorder_time_s for item in camera[:2]) < min(
        item.recorder_time_s for item in camera[2:]
    )
    assert max(item.recorder_time_s for item in lidar[:2]) < min(
        item.recorder_time_s for item in lidar[2:]
    )


def test_joint_diagnostic_does_not_add_motion_from_separate_captures():
    insufficient_motion = MotionExcitation(
        cumulative_rotation_deg=20.0,
        rotation_axes=1,
        translation_m=0.3,
        angular_speed_duration_s=0.5,
        lio_epoch_count=1,
        robust_translation_span_m=0.3,
        robust_rotation_span_deg=20.0,
    )
    first = JointCapture(
        capture_id="first",
        artifact_sha256="a" * 64,
        camera_planes=_planes(100.0),
        lidar_planes=_planes(100.01),
        motion=insufficient_motion,
    )
    second = JointCapture(
        capture_id="second",
        artifact_sha256="b" * 64,
        camera_planes=_planes(200.0),
        lidar_planes=_planes(200.01),
        motion=insufficient_motion,
    )

    result = solve_joint_diagnostic((first, second), solver=_Solver())

    assert result.status == "JOINT_DIAGNOSTIC_UNOBSERVABLE"
    assert result.camera_from_lidar is None
    assert "first:ROTATION_SPAN_BELOW_30_DEG" in result.blockers
    assert "second:TRANSLATION_SPAN_BELOW_0_5_M" in result.blockers


def test_operator_relaxed_joint_diagnostic_bypasses_quality_gates_but_not_authority():
    insufficient_motion = MotionExcitation(
        cumulative_rotation_deg=0.0,
        rotation_axes=0,
        translation_m=0.0,
        angular_speed_duration_s=0.0,
        lio_epoch_count=1,
        robust_translation_span_m=0.0,
        robust_rotation_span_deg=0.0,
    )
    first = JointCapture(
        capture_id="first",
        artifact_sha256="a" * 64,
        camera_planes=_planes(100.0),
        lidar_planes=_planes(100.01),
        motion=insufficient_motion,
        blockers=("CLOCK_FIT_FAILED:color:SPAN_BELOW_MINIMUM",),
    )
    second = JointCapture(
        capture_id="second",
        artifact_sha256="b" * 64,
        camera_planes=_planes(200.0),
        lidar_planes=_planes(200.01),
        motion=insufficient_motion,
    )
    solver = _Solver()

    result = solve_joint_diagnostic(
        (first, second), solver=solver, operator_relaxed=True
    )

    assert result.status == "JOINT_DIAGNOSTIC_PROVISIONAL"
    assert result.calibration_authority is False
    assert solver.calls
    assert any(
        item.startswith("OPERATOR_RELAXED_BYPASS:first:ROTATION_SPAN")
        for item in result.blockers
    )
