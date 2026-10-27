"""Calibration target, timebase, and sensor-frame utilities."""

from .target import (
    BoardDetection,
    BoardPlaneDetection,
    BoardPlaneDetector,
    BoardPose,
    BoardSpec,
    TargetDetector,
    TargetSpecError,
    go2_lab_board_spec,
    verify_source_template,
)
from .timebase import AffineClockEstimator, AffineClockFit, corrected_camera_time_ns
from .capture import CAPTURE_SLOTS, CalibrationCaptureState
from .models import CalibrationCandidate, MotionExcitation, PlaneObservation
from .solver import CalibrationSolver
from .axis_contract import diagnostic_camera_from_lidar_seed
from .joint import (
    JointCapture,
    JointDiagnosticResult,
    joint_capture_from_analysis,
    solve_joint_diagnostic,
)

__all__ = [
    "BoardDetection",
    "BoardPlaneDetection",
    "BoardPlaneDetector",
    "BoardPose",
    "BoardSpec",
    "AffineClockEstimator",
    "AffineClockFit",
    "CAPTURE_SLOTS",
    "CalibrationCaptureState",
    "CalibrationCandidate",
    "CalibrationSolver",
    "diagnostic_camera_from_lidar_seed",
    "JointCapture",
    "JointDiagnosticResult",
    "joint_capture_from_analysis",
    "solve_joint_diagnostic",
    "MotionExcitation",
    "PlaneObservation",
    "TargetDetector",
    "TargetSpecError",
    "go2_lab_board_spec",
    "verify_source_template",
    "corrected_camera_time_ns",
]
