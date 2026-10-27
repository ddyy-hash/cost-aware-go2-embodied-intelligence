import math

import cv2
import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.analysis import (
    LIO_EPOCH_MAX_ROTATION_STEP_DEG,
    LIO_EPOCH_MAX_TRANSLATION_STEP_M,
    motion_excitation_from_poses,
    resolve_frame_transform,
    segment_lio_pose_epochs,
    se3_consensus,
)
from mapping.pc.go2_mapping_rgbd.decode import TransformRecord


def _transform(rotation_vector=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, :3], _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    value[:3, 3] = translation
    return value


def test_se3_consensus_uses_max_pairwise_repeatability_and_returns_proper_rigid_mean():
    transforms = [
        _transform((0.0, 0.0, math.radians(-0.1)), (-0.002, 0.0, 0.0)),
        _transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _transform((0.0, 0.0, math.radians(0.1)), (0.002, 0.0, 0.0)),
    ]

    result = se3_consensus(transforms)

    assert result.translation_repeatability_mm == pytest.approx(4.0)
    assert result.rotation_repeatability_deg == pytest.approx(0.2, abs=1e-8)
    assert result.transform[:3, 3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert np.allclose(result.transform[:3, :3] @ result.transform[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(result.transform[:3, :3]) == pytest.approx(1.0)


def test_tf_graph_resolves_target_from_source_in_both_directions():
    records = (
        TransformRecord(1, "camera_color_optical_frame", "camera_link", (0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0)),
        TransformRecord(1, "camera_link", "camera_depth_optical_frame", (0.02, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    depth_from_color = resolve_frame_transform(
        records, target_frame="camera_depth_optical_frame", source_frame="camera_color_optical_frame"
    )
    color_from_depth = resolve_frame_transform(
        records, target_frame="camera_color_optical_frame", source_frame="camera_depth_optical_frame"
    )

    assert depth_from_color @ color_from_depth == pytest.approx(np.eye(4), abs=1e-12)
    assert depth_from_color[:3, 3] == pytest.approx((-0.02, 0.0, -0.1))


def test_motion_excitation_measures_rotation_axes_translation_and_fast_rotation_duration():
    poses = []
    for index in range(101):
        time_s = index * 0.02
        rotation = (0.25 * math.sin(time_s * 2.0), 0.20 * math.sin(time_s * 3.0), 0.0)
        translation = (0.6 * index / 100.0, 0.05 * math.sin(time_s), 0.0)
        poses.append((int(time_s * 1e9), _transform(rotation, translation)))

    motion = motion_excitation_from_poses(poses)

    assert motion.cumulative_rotation_deg > 30.0
    assert motion.rotation_axes >= 2
    assert motion.translation_m >= 0.6
    assert motion.angular_speed_duration_s >= 1.0


def test_lio_reset_splits_epochs_and_is_not_counted_as_calibration_excitation():
    poses = []
    for index in range(4):
        poses.append(
            (
                index * 100_000_000,
                _transform(translation=(index * 0.01, 0.0, 0.0)),
            )
        )
    reset_index = 4
    poses.extend(
        (
            index * 100_000_000,
            _transform(
                rotation_vector=(0.0, 0.0, math.radians(45.0)),
                translation=(0.60 + (index - reset_index) * 0.01, 0.0, 0.0),
            ),
        )
        for index in range(reset_index, reset_index + 4)
    )

    epochs = segment_lio_pose_epochs(poses)

    assert LIO_EPOCH_MAX_TRANSLATION_STEP_M < 0.60
    assert LIO_EPOCH_MAX_ROTATION_STEP_DEG < 45.0
    assert len(epochs) == 2
    assert epochs[0].source_indices == (0, 1, 2, 3)
    assert epochs[1].source_indices == (4, 5, 6, 7)
    assert epochs[1].reset_reason is not None
    assert "TRANSLATION" in epochs[1].reset_reason
    assert "ROTATION" in epochs[1].reset_reason

    motion = motion_excitation_from_poses(poses)

    assert motion.lio_epoch_count == 2
    assert motion.robust_translation_span_m is not None
    assert motion.robust_rotation_span_deg is not None
    assert motion.robust_translation_span_m < 0.10
    assert motion.robust_rotation_span_deg < 1.0


def test_relaxed_lidar_interpolation_allows_a_bounded_representative_sample_gap():
    from mapping.pc.go2_mapping_calibration.models import PlaneObservation
    from mapping.pc.go2_mapping_calibration.solver import CalibrationSolver

    lidar = (
        PlaneObservation(0.0, np.asarray((0.0, 0.0, 1.0)), 1.0),
        PlaneObservation(1.0, np.asarray((0.0, 0.0, 1.0)), 1.2),
    )

    assert CalibrationSolver._interpolate(lidar, 0.5) is None
    normal, distance = CalibrationSolver._interpolate(
        lidar, 0.5, maximum_span_s=2.0
    )

    assert normal == pytest.approx((0.0, 0.0, 1.0))
    assert distance == pytest.approx(1.1)


def test_solver_residual_keeps_initial_lidar_pairs_when_offset_crosses_sample_boundary():
    """Least squares needs a fixed residual dimension across offset trials."""
    from mapping.pc.go2_mapping_calibration.models import PlaneObservation
    from mapping.pc.go2_mapping_calibration.solver import CalibrationSolver

    camera = (
        PlaneObservation(0.05, np.asarray((0.0, 0.0, 1.0)), 1.0),
        PlaneObservation(0.50, np.asarray((0.0, 0.0, 1.0)), 1.1),
        PlaneObservation(0.95, np.asarray((0.0, 0.0, 1.0)), 1.2),
    )
    lidar = (
        PlaneObservation(0.00, np.asarray((0.0, 0.0, 1.0)), 1.0),
        PlaneObservation(1.00, np.asarray((0.0, 0.0, 1.0)), 1.2),
        PlaneObservation(2.00, np.asarray((0.0, 0.0, 1.0)), 1.4),
    )
    solver = CalibrationSolver()

    initial = solver._residual(
        np.zeros(7), camera, lidar, maximum_interpolation_span_s=2.0
    )
    shifted = solver._residual(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1)),
        camera,
        lidar,
        maximum_interpolation_span_s=2.0,
    )

    assert len(initial) == 12
    assert len(shifted) == len(initial)
