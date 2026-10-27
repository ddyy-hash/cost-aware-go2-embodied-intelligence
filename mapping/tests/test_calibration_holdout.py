import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.capture_analysis import CameraBoardObservation
from mapping.pc.go2_mapping_calibration.holdout import (
    board_depth_correspondences,
    evaluate_holdout_metrics,
    select_independent_board_observations,
)
from mapping.pc.go2_mapping_calibration.target import BoardPose, go2_lab_board_spec


def _pose(count, *, residual=0.0, start=0.6, stop=3.9):
    predicted = np.linspace(start, stop, count)
    observed = predicted + residual
    return observed, predicted


def test_holdout_metrics_accept_exact_independent_pose_and_depth_boundaries():
    poses = [_pose(4000, residual=value) for value in (-0.015, 0.0, 0.015)]

    result = evaluate_holdout_metrics(
        poses,
        reprojection_residuals_px=np.asarray([0.2] * 100 + [1.0]),
        t_d_c_validated=True,
    )

    assert result.accepted is True
    assert result.accepted_board_poses == 3
    assert result.correspondences_per_pose == (4000, 4000, 4000)
    assert result.total_correspondences == 12000
    assert result.camera_reprojection_p95_px <= 1.0
    assert result.depth_p95_absolute_residual_m <= 0.06
    assert result.absolute_mean_signed_bias_m <= 0.015
    assert result.t_d_c_validated is True


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("two_poses", "HOLDOUT_BOARD_POSES_BELOW_MINIMUM"),
        ("few_per_pose", "HOLDOUT_CORRESPONDENCES_PER_POSE_BELOW_MINIMUM"),
        ("few_total", "HOLDOUT_TOTAL_CORRESPONDENCES_BELOW_MINIMUM"),
        ("reprojection", "HOLDOUT_REPROJECTION_P95_EXCEEDED"),
        ("depth_bias", "HOLDOUT_DEPTH_BIAS_EXCEEDED"),
        ("bridge", "HOLDOUT_T_D_C_NOT_VALIDATED"),
    ],
)
def test_holdout_metrics_fail_closed_for_every_promotion_class(mutation, reason):
    poses = [_pose(4000) for _ in range(3)]
    reprojection = np.asarray([0.2] * 100)
    bridge = True
    if mutation == "two_poses": poses = poses[:2]
    if mutation == "few_per_pose": poses[0] = _pose(499)
    if mutation == "few_total": poses = [_pose(3000) for _ in range(3)]
    if mutation == "reprojection": reprojection[-10:] = 1.1
    if mutation == "depth_bias": poses = [_pose(4000, residual=0.016) for _ in range(3)]
    if mutation == "bridge": bridge = False

    result = evaluate_holdout_metrics(poses, reprojection_residuals_px=reprojection, t_d_c_validated=bridge)

    assert result.accepted is False
    assert reason in result.blockers


def _observation(pose, timestamp_ns):
    return CameraBoardObservation.from_pose(
        pose,
        header_timestamp_ns=timestamp_ns,
        recorder_timestamp_ns=timestamp_ns,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )


def test_holdout_pose_selection_rejects_repeated_frames_and_keeps_spatially_distinct_poses():
    repeated = [
        _observation(
            BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
            index * 1_000_000_000,
        )
        for index in range(5)
    ]
    distinct = repeated + [
        _observation(
            BoardPose(rotation_vector=(0.0, 0.25, 0.0), translation_m=(0.15, 0.0, 1.2)),
            6_000_000_000,
        ),
        _observation(
            BoardPose(rotation_vector=(0.2, 0.0, 0.0), translation_m=(-0.15, 0.1, 1.4)),
            7_000_000_000,
        ),
    ]

    assert len(select_independent_board_observations(repeated)) == 1
    selected = select_independent_board_observations(distinct)
    assert len(selected) == 3
    assert [item.header_timestamp_ns for item in selected] == [0, 6_000_000_000, 7_000_000_000]


def test_native_depth_board_projection_uses_only_pixels_inside_physical_board():
    depth = np.ones((100, 100), dtype=np.float64)
    camera = np.asarray(((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)))
    depth_from_board = np.eye(4)
    depth_from_board[0, 3] = -0.12
    depth_from_board[1, 3] = -0.09
    depth_from_board[2, 3] = 1.0

    observed, predicted, pixels = board_depth_correspondences(
        depth,
        camera,
        depth_from_board,
        go2_lab_board_spec(),
    )

    assert len(observed) > 350
    assert len(observed) == len(predicted) == len(pixels)
    assert observed == pytest.approx(np.ones(len(observed)))
    assert predicted == pytest.approx(np.ones(len(predicted)))
    assert len(set(pixels)) == len(pixels)


def test_holdout_capture_requires_three_depth_and_lidar_validations():
    class _Inventory:
        runtime_manifest = type("Runtime", (), {"camera_serial": "d435"})()

    class _Analysis:
        inventory = _Inventory()
        color_camera_info = type("Info", (), {"k": np.asarray(((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)))})()
        depth_camera_info = color_camera_info
        camera_board_observations = (_observation(BoardPose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0),)
        depth_records = ()
        lidar_records = ()
        clock_fits = {}

    from mapping.pc.go2_mapping_calibration.holdout import evaluate_holdout_capture

    result = evaluate_holdout_capture(
        _Analysis(),
        t_d_c=np.eye(4),
        t_c_l=np.eye(4),
        expected_camera_serial="d435",
    )

    assert result.accepted is False
    assert "HOLDOUT_INDEPENDENT_BOARD_POSES_BELOW_MINIMUM" in result.blockers
    assert "HOLDOUT_CORRESPONDENCES_PER_POSE_BELOW_MINIMUM" in result.blockers
    assert "HOLDOUT_LIDAR_BOARD_POSES_BELOW_MINIMUM" in result.blockers
