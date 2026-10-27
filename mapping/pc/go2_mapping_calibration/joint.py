"""Produce provenance-bound diagnostics from multiple calibration captures.

The results assess observability but never write or authorize a measured
calibration manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Protocol, Sequence

import numpy as np

from .models import CalibrationCandidate, MotionExcitation, PlaneObservation
from .solver import CalibrationSolver


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIAGNOSTIC_PROVISIONAL = "JOINT_DIAGNOSTIC_PROVISIONAL"
_DIAGNOSTIC_UNOBSERVABLE = "JOINT_DIAGNOSTIC_UNOBSERVABLE"


class _AnalysisLike(Protocol):
    artifact_sha256: str
    camera_planes: Sequence[PlaneObservation]
    lidar_planes: Sequence[PlaneObservation]
    motion: MotionExcitation
    blockers: Sequence[str]


@dataclass(frozen=True)
class JointCapture:
    """One independently captured, provenance-bound diagnostic input."""

    capture_id: str
    artifact_sha256: str
    camera_planes: tuple[PlaneObservation, ...]
    lidar_planes: tuple[PlaneObservation, ...]
    motion: MotionExcitation
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.capture_id, str) or not self.capture_id.strip():
            raise ValueError("joint capture_id is required")
        digest = str(self.artifact_sha256).lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError("joint artifact_sha256 must be lowercase SHA-256")
        if not isinstance(self.motion, MotionExcitation):
            raise ValueError("joint capture motion must be MotionExcitation")
        camera = tuple(self.camera_planes)
        lidar = tuple(self.lidar_planes)
        if not all(isinstance(item, PlaneObservation) for item in (*camera, *lidar)):
            raise ValueError("joint capture planes must be PlaneObservation instances")
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "camera_planes", camera)
        object.__setattr__(self, "lidar_planes", lidar)
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))


@dataclass(frozen=True)
class JointDiagnosticResult:
    """A non-authoritative outcome from a provenance-bound joint replay."""

    status: str
    source_artifact_sha256: tuple[str, ...]
    camera_from_lidar: np.ndarray | None
    delta_camera_lidar_ms: float | None
    normalized_condition_number: float | None
    residual_p95: float | None
    solver_version: str
    blockers: tuple[str, ...]
    calibration_authority: bool = False
    manifest_written: bool = False

    def __post_init__(self) -> None:
        if self.status not in {_DIAGNOSTIC_PROVISIONAL, _DIAGNOSTIC_UNOBSERVABLE}:
            raise ValueError("joint diagnostic status is invalid")
        if self.calibration_authority or self.manifest_written:
            raise ValueError("joint diagnostics cannot grant calibration authority")
        hashes = tuple(str(item).lower() for item in self.source_artifact_sha256)
        if len(hashes) < 2 or any(not _SHA256.fullmatch(item) for item in hashes):
            raise ValueError("joint diagnostics require two SHA-256 source artifacts")
        object.__setattr__(self, "source_artifact_sha256", hashes)
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        if self.camera_from_lidar is not None:
            transform = np.asarray(self.camera_from_lidar, dtype=np.float64).reshape(4, 4)
            transform.setflags(write=False)
            object.__setattr__(self, "camera_from_lidar", transform)

    @property
    def observable(self) -> bool:
        return self.status == _DIAGNOSTIC_PROVISIONAL

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source_artifact_sha256": list(self.source_artifact_sha256),
            "camera_from_lidar": None
            if self.camera_from_lidar is None
            else self.camera_from_lidar.reshape(-1).tolist(),
            "delta_camera_lidar_ms": self.delta_camera_lidar_ms,
            "normalized_condition_number": self.normalized_condition_number,
            "residual_p95": self.residual_p95,
            "solver_version": self.solver_version,
            "blockers": list(self.blockers),
            "calibration_authority": False,
            "manifest_written": False,
        }


def joint_capture_from_analysis(capture_id: str, analysis: _AnalysisLike) -> JointCapture:
    """Adapt one read-only analyzer result without changing its evidence."""

    return JointCapture(
        capture_id=capture_id,
        artifact_sha256=analysis.artifact_sha256,
        camera_planes=tuple(analysis.camera_planes),
        lidar_planes=tuple(analysis.lidar_planes),
        motion=analysis.motion,
        blockers=tuple(analysis.blockers),
    )


def _capture_gate_blockers(capture: JointCapture) -> tuple[str, ...]:
    motion = capture.motion
    blockers: list[str] = []
    if not capture.camera_planes:
        blockers.append("CAMERA_BOARD_OBSERVATIONS_MISSING")
    if not capture.lidar_planes:
        blockers.append("LIDAR_BOARD_OBSERVATIONS_MISSING")
    if motion.lio_reset_detected:
        blockers.append("LIO_EPOCH_RESET_DETECTED")
    if motion.calibration_rotation_span_deg < 30.0:
        blockers.append("ROTATION_SPAN_BELOW_30_DEG")
    if motion.rotation_axes < 2:
        blockers.append("ROTATION_AXES_BELOW_TWO")
    if motion.calibration_translation_span_m < 0.5:
        blockers.append("TRANSLATION_SPAN_BELOW_0_5_M")
    if motion.angular_speed_duration_s < 1.0:
        blockers.append("ANGULAR_EXCITATION_BELOW_ONE_SECOND")
    blockers.extend(
        item
        for item in capture.blockers
        if item.startswith(("CLOCK_FIT_", "MISSING_REQUIRED_STREAM:", "INITIAL_FRAME_CHAIN_REQUIRED"))
    )
    return tuple(dict.fromkeys(blockers))


def _rebase_planes(
    camera: Iterable[PlaneObservation],
    lidar: Iterable[PlaneObservation],
    *,
    start_time_s: float,
) -> tuple[tuple[PlaneObservation, ...], tuple[PlaneObservation, ...], float]:
    source_camera = tuple(camera)
    source_lidar = tuple(lidar)
    source = (*source_camera, *source_lidar)
    if not source:
        raise ValueError("joint capture needs at least one plane")
    origin = min(item.recorder_time_s for item in source)

    def rebase(items: Sequence[PlaneObservation]) -> tuple[PlaneObservation, ...]:
        return tuple(
            PlaneObservation(
                recorder_time_s=start_time_s + item.recorder_time_s - origin,
                normal=item.normal,
                distance_m=item.distance_m,
                weight=item.weight,
            )
            for item in items
        )

    end_time_s = start_time_s + max(item.recorder_time_s for item in source) - origin
    return rebase(source_camera), rebase(source_lidar), end_time_s + 1.0


def _joint_motion(captures: Sequence[JointCapture]) -> MotionExcitation:
    """Use the weakest per-capture excitation; sessions are never concatenated."""

    motions = tuple(capture.motion for capture in captures)
    return MotionExcitation(
        cumulative_rotation_deg=min(item.calibration_rotation_span_deg for item in motions),
        rotation_axes=min(item.rotation_axes for item in motions),
        translation_m=min(item.calibration_translation_span_m for item in motions),
        angular_speed_duration_s=min(item.angular_speed_duration_s for item in motions),
        lio_epoch_count=1,
        robust_translation_span_m=min(item.calibration_translation_span_m for item in motions),
        robust_rotation_span_deg=min(item.calibration_rotation_span_deg for item in motions),
    )


def _result(
    captures: Sequence[JointCapture],
    *,
    candidate: CalibrationCandidate | None,
    blockers: Iterable[str],
) -> JointDiagnosticResult:
    unique_blockers = tuple(dict.fromkeys(str(item) for item in blockers))
    if candidate is not None and candidate.status == "PROVISIONAL_AUTOCAL" and candidate.camera_from_lidar is not None:
        return JointDiagnosticResult(
            status=_DIAGNOSTIC_PROVISIONAL,
            source_artifact_sha256=tuple(item.artifact_sha256 for item in captures),
            camera_from_lidar=candidate.camera_from_lidar,
            delta_camera_lidar_ms=candidate.delta_camera_lidar_ms,
            normalized_condition_number=candidate.normalized_condition_number,
            residual_p95=candidate.residual_p95,
            solver_version=candidate.solver_version,
            blockers=unique_blockers,
        )
    solver_version = candidate.solver_version if candidate is not None else CalibrationSolver().solver_version
    return JointDiagnosticResult(
        status=_DIAGNOSTIC_UNOBSERVABLE,
        source_artifact_sha256=tuple(item.artifact_sha256 for item in captures),
        camera_from_lidar=None,
        delta_camera_lidar_ms=None,
        normalized_condition_number=None if candidate is None else candidate.normalized_condition_number,
        residual_p95=None if candidate is None else candidate.residual_p95,
        solver_version=solver_version,
        blockers=unique_blockers if unique_blockers else ("JOINT_SOLVER_UNOBSERVABLE",),
    )


def solve_joint_diagnostic(
    captures: Sequence[JointCapture],
    *,
    solver: CalibrationSolver | None = None,
    operator_relaxed: bool = False,
) -> JointDiagnosticResult:
    """Solve a diagnostic-only joint replay without crossing capture boundaries."""

    inputs = tuple(captures)
    if len(inputs) < 2:
        raise ValueError("joint diagnostic requires at least two captures")
    capture_ids = tuple(item.capture_id for item in inputs)
    hashes = tuple(item.artifact_sha256 for item in inputs)
    blockers: list[str] = []
    if len(set(capture_ids)) != len(capture_ids):
        blockers.append("CAPTURE_IDS_NOT_UNIQUE")
    if len(set(hashes)) != len(hashes):
        blockers.append("CAPTURE_ARTIFACTS_NOT_UNIQUE")
    quality_blockers: list[str] = []
    for capture in inputs:
        quality_blockers.extend(
            f"{capture.capture_id}:{reason}" for reason in _capture_gate_blockers(capture)
        )
    if operator_relaxed:
        blockers.extend(
            f"OPERATOR_RELAXED_BYPASS:{reason}" for reason in quality_blockers
        )
    else:
        blockers.extend(quality_blockers)
    if blockers and not operator_relaxed:
        return _result(inputs, candidate=None, blockers=blockers)

    rebased_camera: list[PlaneObservation] = []
    rebased_lidar: list[PlaneObservation] = []
    next_start_s = 0.0
    for capture in inputs:
        camera, lidar, next_start_s = _rebase_planes(
            capture.camera_planes,
            capture.lidar_planes,
            start_time_s=next_start_s,
        )
        rebased_camera.extend(camera)
        rebased_lidar.extend(lidar)

    engine = solver or CalibrationSolver()
    candidate = engine.solve(
        tuple(rebased_camera),
        tuple(rebased_lidar),
        _joint_motion(inputs),
        allow_unobservable=operator_relaxed,
        maximum_interpolation_span_s=5.0 if operator_relaxed else 0.2,
    )
    return _result(
        inputs,
        candidate=candidate,
        blockers=(*blockers, *candidate.blockers),
    )


__all__ = [
    "JointCapture",
    "JointDiagnosticResult",
    "joint_capture_from_analysis",
    "solve_joint_diagnostic",
]
