from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.estimation import (
    CalibrationEstimationError,
    solve_three_estimation_captures,
)
from mapping.pc.go2_mapping_calibration.models import CalibrationCandidate, MotionExcitation


@dataclass(frozen=True)
class _Capture:
    artifact_sha256: str
    camera_planes: tuple
    lidar_planes: tuple
    motion: MotionExcitation


class _Solver:
    solver_version = "unit-test-solver"

    def __init__(self, transforms, offsets):
        self.transforms = iter(transforms)
        self.offsets = iter(offsets)
        self.calls = []

    def solve(self, camera, lidar, motion, **_kwargs):
        self.calls.append((camera, lidar, motion))
        return CalibrationCandidate(
            "PROVISIONAL_AUTOCAL",
            next(self.transforms),
            next(self.offsets),
            100.0,
            0.2,
            (),
            self.solver_version,
        )


def _transform(x_mm, yaw_deg):
    value = np.eye(4)
    value[:3, :3], _ = cv2.Rodrigues(np.asarray([0.0, 0.0, np.radians(yaw_deg)]))
    value[0, 3] = x_mm / 1000.0
    return value


def test_three_capture_estimation_solves_each_independently_and_never_receives_holdout():
    captures = [
        _Capture(str(index) * 64, (f"camera-{index}",), (f"lidar-{index}",), MotionExcitation(40, 2, 0.8, 1.5))
        for index in (1, 2, 3)
    ]
    solver = _Solver(
        [_transform(-2, -0.1), _transform(0, 0), _transform(2, 0.1)],
        [14.0, 15.0, 16.0],
    )

    result = solve_three_estimation_captures(captures, solver=solver)

    assert len(solver.calls) == 3
    assert [call[0] for call in solver.calls] == [("camera-1",), ("camera-2",), ("camera-3",)]
    assert result.source_artifact_sha256 == tuple(item.artifact_sha256 for item in captures)
    assert result.translation_repeatability_mm == pytest.approx(4.0)
    assert result.rotation_repeatability_deg == pytest.approx(0.2)
    assert result.timing_repeatability_ms == pytest.approx(2.0)
    assert result.accepted is True


def test_three_capture_estimation_rejects_duplicate_artifacts_and_any_failed_solution():
    capture = _Capture("a" * 64, ("camera",), ("lidar",), MotionExcitation(40, 2, 0.8, 1.5))
    solver = _Solver([np.eye(4)] * 3, [0.0] * 3)
    with pytest.raises(CalibrationEstimationError, match="unique"):
        solve_three_estimation_captures([capture, capture, capture], solver=solver)

    distinct = [
        _Capture(chr(97 + index) * 64, ("camera",), ("lidar",), MotionExcitation(40, 2, 0.8, 1.5))
        for index in range(3)
    ]

    class _Failed(_Solver):
        def solve(self, camera, lidar, motion, **kwargs):
            result = super().solve(camera, lidar, motion, **kwargs)
            if len(self.calls) == 2:
                return CalibrationCandidate("UNOBSERVABLE", None, None, None, None, ("bad",), self.solver_version)
            return result

    with pytest.raises(CalibrationEstimationError, match="estimation_2"):
        solve_three_estimation_captures(distinct, solver=_Failed([np.eye(4)] * 3, [0.0] * 3))
