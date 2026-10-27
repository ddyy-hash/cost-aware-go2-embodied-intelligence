"""Typed observations and candidates for camera-LiDAR calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneObservation:
    recorder_time_s: float
    normal: np.ndarray
    distance_m: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(normal)) or norm <= 1e-12:
            raise ValueError("plane normal must be finite and nonzero")
        if not np.isfinite(self.recorder_time_s) or not np.isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("plane time and distance must be finite and valid")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("plane weight must be finite and positive")
        normal = normal / norm
        normal.setflags(write=False)
        object.__setattr__(self, "normal", normal)


@dataclass(frozen=True)
class MotionExcitation:
    cumulative_rotation_deg: float
    rotation_axes: int
    translation_m: float
    angular_speed_duration_s: float
    # The legacy cumulative fields are retained for diagnostics. Calibration
    # gates prefer the spans measured inside one continuous LIO epoch.
    lio_epoch_count: int = 1
    robust_translation_span_m: float | None = None
    robust_rotation_span_deg: float | None = None
    reset_reasons: tuple[str, ...] = ()

    @property
    def lio_reset_detected(self) -> bool:
        return self.lio_epoch_count > 1 or bool(self.reset_reasons)

    @property
    def calibration_translation_span_m(self) -> float:
        return self.translation_m if self.robust_translation_span_m is None else self.robust_translation_span_m

    @property
    def calibration_rotation_span_deg(self) -> float:
        return self.cumulative_rotation_deg if self.robust_rotation_span_deg is None else self.robust_rotation_span_deg


@dataclass(frozen=True)
class CalibrationCandidate:
    status: str
    camera_from_lidar: np.ndarray | None
    delta_camera_lidar_ms: float | None
    normalized_condition_number: float | None
    residual_p95: float | None
    blockers: tuple[str, ...]
    solver_version: str

    def __post_init__(self) -> None:
        if self.camera_from_lidar is not None:
            transform = np.asarray(self.camera_from_lidar, dtype=np.float64).reshape(4, 4)
            transform.setflags(write=False)
            object.__setattr__(self, "camera_from_lidar", transform)
