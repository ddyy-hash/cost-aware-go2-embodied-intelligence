import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd import DepthValidator


HASH = "a" * 64


def _scene(depth_value=1.0):
    camera = np.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    pixels = [(u, v) for v in range(20) for u in range(20)]
    points = np.asarray([[u / 100.0, v / 100.0, 1.0] for u, v in pixels], dtype=np.float64)
    depth = np.full((20, 20), depth_value, dtype=np.float64)
    return points, depth, camera


def test_uncalibrated_depth_reports_quality_but_cannot_validate_lidar():
    validator = DepthValidator()
    depth = np.full((20, 20), 1.0)

    first = validator.camera_internal(depth, 1_000, aligned_to_color=False)
    second = validator.camera_internal(depth + 0.001, 2_000, aligned_to_color=False)

    assert first.valid_fraction == 1.0
    assert first.authorizes_texture is False
    assert "DEPTH_NOT_ALIGNED_TO_COLOR" in first.blockers
    assert second.temporal_stability_m == pytest.approx(0.001)


def test_calibrated_depth_agreement_accepts_a_dense_tile():
    points, depth, camera = _scene()

    report = DepthValidator().cross_sensor(
        points,
        depth,
        camera,
        np.eye(4),
        calibration_id="cal-1",
        calibration_sha256=HASH,
        measured_generation="pg-1",
        frame_sha256="b" * 64,
        timestamp_ns=1234,
    )

    assert report.correspondence_count == 400
    assert report.accepted is True
    assert report.authorizes_texture is True
    assert len(report.tiles) == 1
    assert report.tiles[0].p95_absolute_residual_m == 0.0
    assert len(report.correspondences) == 400
    first = report.correspondences[0]
    assert first.lidar_point_index == 0
    assert first.depth_pixel == (0, 0)
    assert first.color_pixel == (0, 0)
    assert first.point_camera_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert first.observed_metric_depth_m == pytest.approx(1.0)
    assert first.residual_class == "CONSENSUS"
    assert first.accepted_region_id == report.tiles[0].tile_id
    assert first.frame_sha256 == "b" * 64
    assert first.calibration_sha256 == HASH
    assert first.measured_generation == "pg-1"
    assert first.timestamp_ns == 1234


def test_depth_conflict_never_moves_measured_positions():
    points, depth, camera = _scene(depth_value=1.1)
    before = points.copy()

    report = DepthValidator().cross_sensor(
        points,
        depth,
        camera,
        np.eye(4),
        calibration_id="cal-1",
        calibration_sha256=HASH,
        measured_generation="pg-1",
    )

    assert report.accepted is False
    assert report.tiles[0].free_space_violation_fraction == 1.0
    assert np.array_equal(points, before)


def test_sparse_depth_overlap_fails_the_per_tile_correspondence_gate():
    points, depth, camera = _scene()

    report = DepthValidator().cross_sensor(
        points[:99],
        depth,
        camera,
        np.eye(4),
        calibration_id="cal-1",
        calibration_sha256=HASH,
        measured_generation="pg-1",
    )

    assert report.accepted is False
    assert report.tiles[0].correspondence_count == 99
