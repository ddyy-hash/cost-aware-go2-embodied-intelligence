from pathlib import Path

import cv2
import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.target import (
    BoardSpec,
    BoardPlaneDetector,
    TargetDetector,
    TargetSpecError,
    go2_lab_board_spec,
    verify_source_template,
)
from mapping.pc.go2_mapping_calibration import CalibrationCaptureState
from mapping.pc.go2_mapping_calibration import (
    CalibrationSolver,
    MotionExcitation,
    PlaneObservation,
)


EXPECTED_PDF_SHA256 = "fe05d01ca5fdde303f712dad177f53c3ae75d3493ff7d9a783b010f892a2d743"


def test_measured_square_and_marker_sizes_are_the_scale_authority():
    spec = go2_lab_board_spec()

    assert spec.dictionary == "DICT_4X4_100"
    assert spec.squares_x == 12
    assert spec.squares_y == 9
    assert spec.marker_length_m == pytest.approx(0.0145)
    assert spec.square_length_m == pytest.approx(0.020)
    assert spec.pattern_width_m == pytest.approx(0.240)
    assert spec.pattern_height_m == pytest.approx(0.180)
    assert spec.source_template_sha256 == EXPECTED_PDF_SHA256
    assert spec.scale_authority == "measured_printed_marker_and_square"


def test_board_uses_the_smallest_dictionary_covering_all_printed_ids():
    spec = go2_lab_board_spec()
    board = spec.create_board()

    assert board.getIds().reshape(-1).tolist() == list(range(54))
    assert spec.expected_marker_ids == tuple(range(54))


def test_measured_dimensions_need_not_preserve_the_pdf_nominal_ratio():
    spec = BoardSpec(
        dictionary="DICT_4X4_100",
        squares_x=12,
        squares_y=9,
        marker_length_m=0.0145,
        square_length_m=0.020,
        template_marker_length_m=0.005,
        template_square_length_m=0.007,
        source_template_sha256=EXPECTED_PDF_SHA256,
        scale_authority="measured_printed_marker_and_square",
    )

    assert spec.square_length_m / spec.marker_length_m != pytest.approx(7.0 / 5.0)


def test_source_template_hash_is_verified_over_exact_pdf_bytes(tmp_path):
    source = tmp_path / "board.pdf"
    source.write_bytes(b"calibration-board")
    expected = "8a2ba40cffce0447a329b72c5a38600281b3ae619cc0aafc8df3dc2c2d520815"

    assert verify_source_template(source, expected) == expected
    with pytest.raises(TargetSpecError, match="SHA-256"):
        verify_source_template(source, "0" * 64)


def test_target_detector_recovers_a_rectified_full_board():
    spec = go2_lab_board_spec()
    image = spec.render(pixels_per_square=100, margin_px=32)

    detection = TargetDetector(spec).detect(image)

    assert detection.accepted is True
    assert detection.marker_ids == tuple(range(54))
    assert detection.decoded_fraction >= 0.99
    assert detection.charuco_fraction >= 0.95
    assert detection.reprojection_p95_px < 0.5
    assert detection.coverage_fraction > 0.70


def test_camera_pose_fit_rejects_fewer_than_six_charuco_correspondences():
    spec = go2_lab_board_spec()
    detector = TargetDetector(
        spec,
        camera_matrix=np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    charuco_ids = np.arange(5, dtype=np.int32).reshape(-1, 1)
    charuco_corners = np.asarray(
        [[120.0, 100.0], [140.0, 100.0], [160.0, 100.0], [180.0, 100.0], [120.0, 120.0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    residuals, pose = detector._fit_geometry(charuco_corners, charuco_ids)

    assert residuals == ()
    assert pose is None


def test_three_identical_prints_are_reported_as_separate_board_instances():
    spec = go2_lab_board_spec()
    board = spec.render(pixels_per_square=60, margin_px=20)
    gap = 100
    canvas = np.full((board.shape[0], board.shape[1] * 3 + gap * 2), 255, dtype=np.uint8)
    for index in range(3):
        x = index * (board.shape[1] + gap)
        canvas[:, x : x + board.shape[1]] = board

    detections = TargetDetector(spec).detect_all(canvas)

    assert len(detections) == 3
    assert all(result.decoded_fraction >= 0.99 for result in detections)
    assert all(result.marker_ids == tuple(range(54)) for result in detections)


def test_blank_frame_cannot_be_accepted_as_calibration_evidence():
    result = TargetDetector(go2_lab_board_spec()).detect(np.full((720, 1280), 255, dtype=np.uint8))

    assert result.accepted is False
    assert result.marker_ids == ()
    assert result.blocking_reasons


def test_actual_prepared_pdf_matches_the_locked_template_when_available():
    source = Path(
        "/mnt/c/Users/A/Documents/xwechat_files/"
        "wxid_5lhi0em3yhyn32_1630/msg/file/2026-07/"
        "calib.io_charuco_200x150_9x12_7_5_DICT_4X4.pdf"
    )
    if not source.exists():
        pytest.skip("operator-provided calibration PDF is not mounted")

    assert verify_source_template(source, EXPECTED_PDF_SHA256) == EXPECTED_PDF_SHA256


def test_capture_progress_keeps_four_slots_and_marks_live_detection_non_authoritative():
    spec = go2_lab_board_spec()
    image = spec.render(pixels_per_square=40, margin_px=16)
    height, width = image.shape
    camera_matrix = np.asarray([[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]])
    state = CalibrationCaptureState(TargetDetector(spec, camera_matrix=camera_matrix), (width, height))

    state.ingest_image(image, 10)
    state.select_slot("holdout")
    state.ingest_image(image, 20)
    snapshot = state.snapshot()

    assert set(snapshot["slots"]) == {"estimation_1", "estimation_2", "estimation_3", "holdout"}
    assert snapshot["slots"]["estimation_1"]["accepted_frames"] == 1
    assert snapshot["slots"]["holdout"]["accepted_frames"] == 1
    assert snapshot["preview_authoritative"] is False
    assert "BOARD_NORMAL_DIVERSITY_BELOW_THREE" in snapshot["blockers"]


def _lidar_plane(time_s):
    normal = np.asarray(
        [0.55 * np.sin(0.9 * time_s), 0.45 * np.cos(0.7 * time_s), 1.0],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    distance = 1.2 + 0.15 * np.sin(1.3 * time_s) + 0.05 * np.cos(0.4 * time_s)
    return normal, distance


def test_solver_recovers_provisional_transform_and_time_offset():
    truth_rotation_vector = np.asarray([0.03, -0.05, 0.02])
    truth_rotation, _ = cv2.Rodrigues(truth_rotation_vector)
    truth_translation = np.asarray([0.05, -0.02, 0.03])
    truth_offset_s = 0.015
    lidar = []
    for time_s in np.linspace(0.0, 4.0, 401):
        normal, distance = _lidar_plane(time_s)
        lidar.append(PlaneObservation(time_s, normal, distance))
    camera = []
    for time_s in np.linspace(0.2, 3.8, 160):
        normal_lidar, distance_lidar = _lidar_plane(time_s + truth_offset_s)
        normal_camera = truth_rotation @ normal_lidar
        distance_camera = distance_lidar + np.dot(normal_camera, truth_translation)
        camera.append(PlaneObservation(time_s, normal_camera, distance_camera))

    result = CalibrationSolver().solve(
        camera,
        lidar,
        MotionExcitation(45.0, 2, 0.8, 1.5),
    )

    assert result.status == "PROVISIONAL_AUTOCAL"
    assert np.linalg.norm(result.camera_from_lidar[:3, 3] - truth_translation) < 0.005
    rotation_error, _ = cv2.Rodrigues(result.camera_from_lidar[:3, :3] @ truth_rotation.T)
    assert np.degrees(np.linalg.norm(rotation_error)) < 0.2
    assert abs(result.delta_camera_lidar_ms - truth_offset_s * 1000.0) < 2.0


def test_solver_version_identifies_the_observability_checked_implementation():
    assert CalibrationSolver().solver_version == "plane-se3-time-softl1-v2"


def test_solver_rejects_static_coplanar_board_capture_as_unobservable():
    planes = [PlaneObservation(index * 0.05, np.asarray([0.0, 0.0, 1.0]), 1.0) for index in range(100)]

    result = CalibrationSolver().solve(
        planes,
        planes,
        MotionExcitation(0.0, 1, 0.0, 0.0),
    )

    assert result.status == "UNOBSERVABLE"
    assert "BOARD_NORMAL_DIVERSITY_BELOW_THREE" in result.blockers
    assert result.camera_from_lidar is None


def test_solver_rejects_motion_that_only_looks_large_across_lio_epochs():
    planes = []
    for index, angle_deg in enumerate((-35.0, -20.0, 0.0, 20.0, 35.0) * 5):
        angle = np.radians(angle_deg)
        normal = np.asarray([np.sin(angle), 0.2 * np.sin(2.0 * angle), np.cos(angle)])
        planes.append(PlaneObservation(index * 0.05, normal, 1.0 + index * 0.001))
    motion = MotionExcitation(
        cumulative_rotation_deg=181.0,
        rotation_axes=3,
        translation_m=1.6,
        angular_speed_duration_s=5.0,
        lio_epoch_count=2,
        robust_translation_span_m=0.12,
        robust_rotation_span_deg=7.0,
        reset_reasons=("LIO_EPOCH_RESET:TRANSLATION_STEP_EXCEEDED",),
    )

    result = CalibrationSolver().solve(planes, planes, motion)

    assert result.status == "UNOBSERVABLE"
    assert "LIO_EPOCH_RESET_DETECTED" in result.blockers
    assert "ROTATION_SPAN_BELOW_30_DEG" in result.blockers
    assert "TRANSLATION_SPAN_BELOW_0_5_M" in result.blockers
    assert result.camera_from_lidar is None


def test_solver_rejects_three_angular_buckets_when_normal_matrix_is_rank_two():
    planes = []
    for index, angle_deg in enumerate((-35.0, 0.0, 35.0) * 10):
        angle = np.radians(angle_deg)
        planes.append(
            PlaneObservation(index * 0.05, np.asarray([np.sin(angle), 0.0, np.cos(angle)]), 1.0)
        )

    assert CalibrationSolver._independent_normals([item.normal for item in planes]) >= 3
    result = CalibrationSolver().solve(planes, planes, MotionExcitation(45.0, 2, 0.8, 1.5))

    assert result.status == "UNOBSERVABLE"
    assert "CAMERA_BOARD_NORMAL_MATRIX_RANK_BELOW_THREE" in result.blockers
    assert "LIDAR_BOARD_NORMAL_MATRIX_RANK_BELOW_THREE" in result.blockers


def test_solver_rejects_full_rank_but_ill_conditioned_normal_matrix():
    planes = []
    pattern = ((-35.0, 1.0), (0.0, -1.0), (35.0, 1.0))
    for index, (angle_deg, y_sign) in enumerate(pattern * 10):
        angle = np.radians(angle_deg)
        planes.append(
            PlaneObservation(
                index * 0.05,
                np.asarray([np.sin(angle), y_sign * 1e-5, np.cos(angle)]),
                1.0,
            )
        )

    result = CalibrationSolver().solve(planes, planes, MotionExcitation(45.0, 2, 0.8, 1.5))

    assert result.status == "UNOBSERVABLE"
    assert "CAMERA_BOARD_NORMAL_MATRIX_CONDITION_EXCEEDED" in result.blockers
    assert "LIDAR_BOARD_NORMAL_MATRIX_CONDITION_EXCEEDED" in result.blockers


def test_lidar_plane_detector_accepts_bounded_board_support():
    spec = go2_lab_board_spec()
    points = np.asarray(
        [
            [1.0, y, z]
            for y in np.linspace(-spec.pattern_width_m / 2, spec.pattern_width_m / 2, 30)
            for z in np.linspace(-spec.pattern_height_m / 2, spec.pattern_height_m / 2, 24)
        ]
    )

    result = BoardPlaneDetector(spec).detect(
        points,
        predicted_center_lidar=np.asarray([1.0, 0.0, 0.0]),
        association_error_ms=2.0,
    )

    assert result.accepted is True
    assert result.finite_boundary_verified is True
    assert result.support_count == len(points)
    assert result.residual_p95_m < 1e-6


def test_lidar_plane_detector_rejects_ambiguous_room_wall():
    points = np.asarray(
        [[1.0, y, z] for y in np.linspace(-2.0, 2.0, 100) for z in np.linspace(-1.5, 1.5, 80)]
    )

    result = BoardPlaneDetector(go2_lab_board_spec()).detect(
        points,
        predicted_center_lidar=np.asarray([1.0, 0.0, 0.0]),
        association_error_ms=2.0,
    )

    assert result.accepted is False
    assert "PLANE_EXTENT_DOES_NOT_MATCH_BOARD" in result.blocking_reasons


def test_lidar_plane_detector_keeps_backing_plane_diagnostic_but_rejects_it_as_board():
    points = np.asarray(
        [[1.0, y, z] for y in np.linspace(-2.0, 2.0, 100) for z in np.linspace(-1.5, 1.5, 80)]
    )

    result = BoardPlaneDetector(
        go2_lab_board_spec(),
        allow_backing_plane=True,
    ).detect(
        points,
        predicted_center_lidar=np.asarray([1.0, 0.0, 0.0]),
        predicted_normal_lidar=np.asarray([1.0, 0.0, 0.0]),
        association_error_ms=2.0,
    )

    assert result.accepted is False
    assert result.finite_boundary_verified is False
    assert "FINITE_BOARD_BOUNDARY_NOT_VERIFIED" in result.blocking_reasons
    assert result.normal_lidar == pytest.approx((1.0, 0.0, 0.0), abs=1e-3)


def test_lidar_plane_detector_finds_board_behind_dominant_wrong_normal_wall():
    spec = go2_lab_board_spec()
    board = np.asarray(
        [
            [1.0, y, z]
            for y in np.linspace(-spec.pattern_width_m / 2, spec.pattern_width_m / 2, 30)
            for z in np.linspace(-spec.pattern_height_m / 2, spec.pattern_height_m / 2, 24)
        ]
    )
    wall = np.asarray(
        [
            [x, y, -0.24]
            for x in np.linspace(0.72, 1.28, 70)
            for y in np.linspace(-0.28, 0.28, 70)
        ]
    )

    result = BoardPlaneDetector(spec).detect(
        np.concatenate((wall, board), axis=0),
        predicted_center_lidar=np.asarray([1.0, 0.0, 0.0]),
        predicted_normal_lidar=np.asarray([1.0, 0.0, 0.0]),
        association_error_ms=2.0,
    )

    assert result.accepted is True
    assert result.normal_lidar == pytest.approx((1.0, 0.0, 0.0), abs=1e-3)
    assert result.extent_m == pytest.approx(
        tuple(sorted((spec.pattern_width_m, spec.pattern_height_m))),
        abs=0.015,
    )


def test_lidar_plane_detector_allows_bounded_bootstrap_center_offset():
    spec = go2_lab_board_spec()
    actual_center = np.asarray([1.0, 0.23, 0.0])
    points = np.asarray(
        [
            [actual_center[0], actual_center[1] + y, actual_center[2] + z]
            for y in np.linspace(-spec.pattern_width_m / 2, spec.pattern_width_m / 2, 30)
            for z in np.linspace(-spec.pattern_height_m / 2, spec.pattern_height_m / 2, 24)
        ]
    )

    result = BoardPlaneDetector(spec, maximum_center_error_m=0.30).detect(
        points,
        predicted_center_lidar=np.asarray([1.0, 0.0, 0.0]),
        predicted_normal_lidar=np.asarray([1.0, 0.0, 0.0]),
        association_error_ms=2.0,
    )

    assert result.accepted is True
    assert result.center_lidar == pytest.approx(actual_center, abs=0.01)


def test_lidar_plane_detector_prefers_best_normal_over_first_accepted_plane():
    spec = go2_lab_board_spec()
    expected_normal = np.asarray([1.0, 0.0, 0.0])
    true_center = np.asarray([1.0, 0.0, 0.0])
    true_board = np.asarray(
        [
            true_center + np.asarray([0.0, y, z])
            for y in np.linspace(-spec.pattern_width_m / 2, spec.pattern_width_m / 2, 20)
            for z in np.linspace(-spec.pattern_height_m / 2, spec.pattern_height_m / 2, 15)
        ]
    )
    angle = np.radians(20.0)
    decoy_normal = np.asarray([np.cos(angle), np.sin(angle), 0.0])
    decoy_axis = np.asarray([-np.sin(angle), np.cos(angle), 0.0])
    decoy_center = true_center + 0.10 * expected_normal
    decoy = np.asarray(
        [
            decoy_center + y * decoy_axis + z * np.asarray([0.0, 0.0, 1.0])
            for y in np.linspace(-spec.pattern_width_m / 2, spec.pattern_width_m / 2, 40)
            for z in np.linspace(-spec.pattern_height_m / 2, spec.pattern_height_m / 2, 30)
        ]
    )

    result = BoardPlaneDetector(spec).detect(
        np.concatenate((decoy, true_board), axis=0),
        predicted_center_lidar=true_center,
        predicted_normal_lidar=expected_normal,
        association_error_ms=2.0,
    )

    assert result.accepted is True
    assert result.normal_lidar == pytest.approx(expected_normal, abs=1e-3)
