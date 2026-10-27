"""Deterministic robust sensor-header to recorder-clock mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import (
    ClockFitGates,
    canonical_sha256,
    load_reconstruction_config,
)


@dataclass(frozen=True)
class AffineClockFit:
    slope: float
    offset_ns: float
    sample_count: int
    span_s: float
    inlier_fraction: float
    median_absolute_residual_ms: float
    p95_residual_ms: float
    maximum_prediction_stddev_ms: float
    inlier_mask: tuple[bool, ...]
    accepted: bool
    reasons: tuple[str, ...]
    estimator_version: str
    sha256: str

    def map_header_ns(self, header_timestamp_ns: int | float) -> float:
        return self.slope * float(header_timestamp_ns) + self.offset_ns


class AffineClockEstimator:
    def __init__(
        self,
        gates: ClockFitGates | None = None,
        *,
        estimator_version: str = "affine-ransac-wls-v1",
        seed: int = 20260731,
    ) -> None:
        self.gates = gates or load_reconstruction_config().appearance.clock_fit
        self.estimator_version = estimator_version
        self.seed = seed

    def fit(self, header_timestamp_ns: np.ndarray, record_timestamp_ns: np.ndarray) -> AffineClockFit:
        header = np.asarray(header_timestamp_ns, dtype=np.float64).reshape(-1)
        record = np.asarray(record_timestamp_ns, dtype=np.float64).reshape(-1)
        if len(header) != len(record) or len(header) < 2:
            raise ValueError("clock fitting requires at least two paired timestamps")
        if not np.all(np.isfinite(header)) or not np.all(np.isfinite(record)):
            raise ValueError("clock timestamps must be finite")
        if np.any(np.diff(header) <= 0.0):
            raise ValueError("header timestamps must be strictly increasing")

        header_origin = float(header[0])
        record_origin = float(record[0])
        x = (header - header_origin) / 1e9
        y = (record - record_origin) / 1e9
        threshold_s = self.gates.maximum_p95_residual_ms / 1000.0
        inliers = self._ransac_inliers(x, y, threshold_s)
        slope, local_offset_s, covariance = self._least_squares(x[inliers], y[inliers])
        residual_s = y - (slope * x + local_offset_s)
        refined = np.abs(residual_s) <= threshold_s
        if np.count_nonzero(refined) >= 2:
            inliers = refined
            slope, local_offset_s, covariance = self._least_squares(x[inliers], y[inliers])
            residual_s = y - (slope * x + local_offset_s)

        inlier_residual_ms = np.abs(residual_s[inliers]) * 1000.0
        median_ms = float(np.median(inlier_residual_ms))
        p95_ms = float(np.percentile(inlier_residual_ms, 95))
        prediction_std_ms = self._maximum_prediction_stddev_ms(x[inliers], covariance)
        span_s = float(x[-1] - x[0])
        inlier_fraction = float(np.mean(inliers))
        slope_deviation_ppm = abs(slope - 1.0) * 1e6
        accepted = self.gates.accepts(
            sample_count=len(header),
            span_s=span_s,
            inlier_fraction=inlier_fraction,
            slope_deviation_ppm=slope_deviation_ppm,
            median_absolute_residual_ms=median_ms,
            p95_residual_ms=p95_ms,
            prediction_stddev_ms=prediction_std_ms,
        )
        reasons: list[str] = []
        if len(header) < self.gates.minimum_sample_count:
            reasons.append("SAMPLE_COUNT_BELOW_MINIMUM")
        if span_s < self.gates.minimum_span_s:
            reasons.append("SPAN_BELOW_MINIMUM")
        if inlier_fraction < self.gates.minimum_inlier_fraction:
            reasons.append("INLIER_FRACTION_BELOW_MINIMUM")
        if slope_deviation_ppm > self.gates.maximum_slope_deviation_ppm:
            reasons.append("SLOPE_DRIFT_EXCEEDED")
        if median_ms > self.gates.maximum_median_absolute_residual_ms:
            reasons.append("MEDIAN_RESIDUAL_EXCEEDED")
        if p95_ms > self.gates.maximum_p95_residual_ms:
            reasons.append("P95_RESIDUAL_EXCEEDED")
        if prediction_std_ms > self.gates.maximum_prediction_stddev_ms:
            reasons.append("PREDICTION_UNCERTAINTY_EXCEEDED")

        offset_ns = record_origin + local_offset_s * 1e9 - slope * header_origin
        identity = {
            "schema_version": 1,
            "estimator_version": self.estimator_version,
            "slope": slope,
            "offset_ns": offset_ns,
            "sample_count": len(header),
            "span_s": span_s,
            "inlier_fraction": inlier_fraction,
            "median_absolute_residual_ms": median_ms,
            "p95_residual_ms": p95_ms,
            "maximum_prediction_stddev_ms": prediction_std_ms,
            "inlier_indices": np.flatnonzero(inliers).tolist(),
        }
        return AffineClockFit(
            slope=slope,
            offset_ns=offset_ns,
            sample_count=len(header),
            span_s=span_s,
            inlier_fraction=inlier_fraction,
            median_absolute_residual_ms=median_ms,
            p95_residual_ms=p95_ms,
            maximum_prediction_stddev_ms=prediction_std_ms,
            inlier_mask=tuple(bool(value) for value in inliers),
            accepted=accepted,
            reasons=tuple(reasons),
            estimator_version=self.estimator_version,
            sha256=canonical_sha256(identity),
        )

    def _ransac_inliers(self, x: np.ndarray, y: np.ndarray, threshold_s: float) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        best = np.ones(len(x), dtype=bool)
        best_count = 0
        best_residual = math.inf
        candidate_pairs = [(0, len(x) - 1)]
        candidate_pairs.extend(
            tuple(int(value) for value in rng.choice(len(x), size=2, replace=False))
            for _ in range(min(512, len(x) * 4))
        )
        for left, right in candidate_pairs:
            if left == right or abs(x[right] - x[left]) <= 1e-12:
                continue
            slope = (y[right] - y[left]) / (x[right] - x[left])
            offset = y[left] - slope * x[left]
            residual = np.abs(y - (slope * x + offset))
            mask = residual <= threshold_s
            count = int(np.count_nonzero(mask))
            score = float(np.median(residual[mask])) if count else math.inf
            if count > best_count or (count == best_count and score < best_residual):
                best, best_count, best_residual = mask, count, score
        if best_count < 2:
            best = np.ones(len(x), dtype=bool)
        return best

    @staticmethod
    def _least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        design = np.column_stack([x, np.ones(len(x))])
        parameters, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ parameters
        degrees = max(len(x) - 2, 1)
        variance = float(np.dot(residual, residual) / degrees)
        covariance = variance * np.linalg.pinv(design.T @ design)
        return float(parameters[0]), float(parameters[1]), covariance

    @staticmethod
    def _maximum_prediction_stddev_ms(x: np.ndarray, covariance: np.ndarray) -> float:
        predictions = []
        for value in (float(np.min(x)), float(np.max(x))):
            vector = np.asarray([value, 1.0])
            predictions.append(math.sqrt(max(float(vector @ covariance @ vector), 0.0)) * 1000.0)
        return max(predictions)


def corrected_camera_time_ns(header_timestamp_ns: int | float, fit: AffineClockFit, delta_c_l_ns: int | float) -> float:
    """Positive camera-LiDAR offset shifts the mapped camera sample later."""

    return fit.map_header_ns(header_timestamp_ns) + float(delta_c_l_ns)
