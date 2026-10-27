"""Independent three-capture camera-LiDAR estimation and repeatability."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import Se3Consensus, se3_consensus
from .models import CalibrationCandidate
from .solver import CalibrationSolver


class CalibrationEstimationError(ValueError):
    pass


@dataclass(frozen=True)
class ThreeCaptureEstimation:
    accepted: bool
    source_artifact_sha256: tuple[str, ...]
    independent_candidates: tuple[CalibrationCandidate, ...]
    camera_from_lidar: np.ndarray
    delta_camera_lidar_ms: float
    translation_repeatability_mm: float
    rotation_repeatability_deg: float
    timing_repeatability_ms: float
    blockers: tuple[str, ...]
    solver_version: str

    def __post_init__(self) -> None:
        value = np.asarray(self.camera_from_lidar, dtype=np.float64).reshape(4, 4)
        value.setflags(write=False)
        object.__setattr__(self, "camera_from_lidar", value)


def _gates(path: Path | None = None) -> dict[str, Any]:
    source = path or Path(__file__).resolve().parents[2] / "config" / "quality_gates.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    return dict(payload["calibration"])


def solve_three_estimation_captures(
    captures: Sequence[Any],
    *,
    solver: CalibrationSolver | None = None,
    quality_gates_path: Path | None = None,
) -> ThreeCaptureEstimation:
    if len(captures) != 3:
        raise CalibrationEstimationError("exactly three estimation captures are required")
    digests = tuple(str(item.artifact_sha256) for item in captures)
    if len(set(digests)) != 3:
        raise CalibrationEstimationError("estimation capture artifacts must be unique")
    engine = solver or CalibrationSolver()
    candidates = []
    for index, capture in enumerate(captures, start=1):
        candidate = engine.solve(capture.camera_planes, capture.lidar_planes, capture.motion)
        if candidate.status != "PROVISIONAL_AUTOCAL" or candidate.camera_from_lidar is None or candidate.delta_camera_lidar_ms is None:
            detail = ",".join(candidate.blockers) if candidate.blockers else candidate.status
            raise CalibrationEstimationError(f"estimation_{index} failed: {detail}")
        candidates.append(candidate)
    consensus = se3_consensus([item.camera_from_lidar for item in candidates])
    offsets = np.asarray([item.delta_camera_lidar_ms for item in candidates], dtype=np.float64)
    timing_spread = float(np.max(offsets) - np.min(offsets))
    gates = _gates(quality_gates_path)
    blockers = []
    if consensus.translation_repeatability_mm > gates["lidar_camera_translation_repeatability_mm"]:
        blockers.append("LIDAR_CAMERA_TRANSLATION_REPEATABILITY_EXCEEDED")
    if consensus.rotation_repeatability_deg > gates["lidar_camera_rotation_repeatability_deg"]:
        blockers.append("LIDAR_CAMERA_ROTATION_REPEATABILITY_EXCEEDED")
    if timing_spread > gates["lidar_camera_timing_repeatability_ms"]:
        blockers.append("LIDAR_CAMERA_TIMING_REPEATABILITY_EXCEEDED")
    return ThreeCaptureEstimation(
        accepted=not blockers,
        source_artifact_sha256=digests,
        independent_candidates=tuple(candidates),
        camera_from_lidar=consensus.transform,
        delta_camera_lidar_ms=float(np.median(offsets)),
        translation_repeatability_mm=consensus.translation_repeatability_mm,
        rotation_repeatability_deg=consensus.rotation_repeatability_deg,
        timing_repeatability_ms=timing_spread,
        blockers=tuple(blockers),
        solver_version=engine.solver_version,
    )
