import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration import (
    AffineClockEstimator,
    corrected_camera_time_ns,
)


def _pairs(*, slope=1.0, offset_ns=25_000_000.0, noise_ns=0.0, count=240, span_s=40.0):
    header = 1_700_000_000_000_000_000.0 + np.linspace(0.0, span_s * 1e9, count)
    rng = np.random.default_rng(7)
    record = slope * header + offset_ns + rng.normal(0.0, noise_ns, size=count)
    return header, record


def test_clock_fit_recovers_exact_affine_mapping_and_is_deterministic():
    header, record = _pairs()
    estimator = AffineClockEstimator()

    first = estimator.fit(header, record)
    second = estimator.fit(header, record)

    assert first.accepted is True
    assert first.slope == pytest.approx(1.0, abs=1e-12)
    assert first.map_header_ns(header[-1]) == pytest.approx(record[-1], abs=512.0)
    assert first.p95_residual_ms < 1e-6
    assert first.sha256 == second.sha256
    assert first.inlier_mask == second.inlier_mask


def test_clock_fit_rejects_excessive_drift():
    header, record = _pairs(slope=1.000051)

    fit = AffineClockEstimator().fit(header, record)

    assert fit.accepted is False
    assert "SLOPE_DRIFT_EXCEEDED" in fit.reasons


def test_clock_fit_rejects_too_few_samples_or_short_span():
    header, record = _pairs(count=199, span_s=29.0)

    fit = AffineClockEstimator().fit(header, record)

    assert fit.accepted is False
    assert "SAMPLE_COUNT_BELOW_MINIMUM" in fit.reasons
    assert "SPAN_BELOW_MINIMUM" in fit.reasons


def test_clock_fit_classifies_outliers_and_retains_accepted_inlier_fit():
    header, record = _pairs(noise_ns=200_000.0)
    record[::40] += 25_000_000.0

    fit = AffineClockEstimator().fit(header, record)

    assert fit.inlier_fraction == pytest.approx(234 / 240)
    assert fit.accepted is True
    assert fit.p95_residual_ms < 1.0


def test_positive_camera_offset_shifts_camera_later():
    header, record = _pairs()
    fit = AffineClockEstimator().fit(header, record)

    assert corrected_camera_time_ns(header[20], fit, 2_000_000) == pytest.approx(record[20] + 2_000_000, abs=512.0)
