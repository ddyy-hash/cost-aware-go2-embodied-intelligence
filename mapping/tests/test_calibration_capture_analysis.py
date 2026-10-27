import numpy as np
import pytest
import cv2
from pathlib import Path

from mapping.pc.go2_mapping_calibration.capture_analysis import (
    CameraBoardObservation,
    DepthFrameEvidence,
    LidarFrameEvidence,
    camera_info_identity,
    camera_plane_from_board_pose,
    aggregate_lidar_window,
    mapped_camera_plane,
    nearest_within,
)
from mapping.pc.go2_mapping_calibration.target import BoardPose, go2_lab_board_spec
from mapping.pc.go2_mapping_rgbd.decode import CameraInfoRecord, TransformRecord


def _camera_info(header_timestamp_ns, raw_hash):
    k = np.asarray([[300.0, 0.0, 200.0], [0.0, 301.0, 120.0], [0.0, 0.0, 1.0]])
    return CameraInfoRecord(
        header_timestamp_ns=header_timestamp_ns,
        frame_id="camera_color_optical_frame",
        width=424,
        height=240,
        distortion_model="plumb_bob",
        d=np.zeros(5),
        k=k,
        r=np.eye(3),
        p=np.column_stack([k, np.zeros(3)]),
        binning_x=0,
        binning_y=0,
        roi={"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
        message_sha256=raw_hash,
    )


def test_camera_info_identity_ignores_header_time_but_preserves_full_intrinsics():
    first = camera_info_identity(_camera_info(1, "a" * 64))
    second = camera_info_identity(_camera_info(2, "b" * 64))

    assert first.sha256 == second.sha256
    assert first.raw_message_sha256 != second.raw_message_sha256
    assert first.contract["k"] == [300.0, 0.0, 200.0, 0.0, 301.0, 120.0, 0.0, 0.0, 1.0]


def test_camera_board_pose_becomes_a_positive_normalized_plane_observation():
    pose = BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.1, -0.2, 1.5))

    plane = camera_plane_from_board_pose(pose, recorder_time_s=12.5, weight=4.0)

    assert plane.normal == pytest.approx((0.0, 0.0, 1.0))
    assert plane.distance_m == pytest.approx(1.5)
    assert plane.weight == pytest.approx(4.0)


def test_nearest_association_is_bounded_and_does_not_silently_use_distant_frame():
    samples = [(1.000, "a"), (1.020, "b"), (1.080, "c")]

    assert nearest_within(samples, 1.018, maximum_delta_s=0.005) == (1.020, "b")
    assert nearest_within(samples, 1.050, maximum_delta_s=0.005) is None


def test_lidar_window_motion_compensates_five_frames_into_reference_pose():
    lidar_samples = []
    pose_samples = []
    for timestamp_s, sensor_x in zip(np.linspace(0.8, 1.2, 5), np.linspace(-0.08, 0.08, 5)):
        world_from_lidar = np.eye(4)
        world_from_lidar[0, 3] = sensor_x
        pose_samples.append((timestamp_s, world_from_lidar))
        lidar_samples.append((timestamp_s, np.asarray([[2.0 - sensor_x, 0.0, 0.0]])))

    window = aggregate_lidar_window(
        lidar_samples,
        pose_samples,
        target_time_s=1.0,
        maximum_center_delta_s=0.01,
        window_radius_s=0.21,
        maximum_frames=5,
    )

    assert window is not None
    assert window.reference_time_s == pytest.approx(1.0)
    assert window.frame_count == 5
    assert window.points_xyz == pytest.approx(np.tile((2.0, 0.0, 0.0), (5, 1)))


def test_lidar_window_does_not_cross_a_tagged_lio_epoch_boundary():
    stationary_pose = np.eye(4)
    lidar_samples = [
        (0.80, np.asarray([[2.0, 0.0, 0.0]]), 0),
        (0.90, np.asarray([[2.0, 0.0, 0.0]]), 0),
        (1.00, np.asarray([[9.0, 0.0, 0.0]]), 1),
        (1.10, np.asarray([[9.0, 0.0, 0.0]]), 1),
    ]
    pose_samples = [
        (0.80, stationary_pose, 0),
        (0.90, stationary_pose, 0),
        (1.00, stationary_pose, 1),
        (1.10, stationary_pose, 1),
    ]

    window = aggregate_lidar_window(
        lidar_samples,
        pose_samples,
        target_time_s=0.90,
        maximum_center_delta_s=0.01,
        window_radius_s=0.25,
        maximum_frames=5,
    )

    assert window is not None
    assert window.frame_count == 2
    assert window.points_xyz == pytest.approx(
        np.asarray([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )


def test_lidar_plane_association_uses_aggregated_window_and_camera_normal():
    from mapping.pc.go2_mapping_calibration.analyzer import _associate_lidar_planes
    from mapping.pc.go2_mapping_calibration.target import BoardPlaneDetector, go2_lab_board_spec

    spec = go2_lab_board_spec()
    observation = CameraBoardObservation.from_pose(
        BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
        header_timestamp_ns=1_000_000_000,
        recorder_timestamp_ns=1_000_000_000,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )
    sparse_board = np.asarray(
        [
            [x, y, 1.0]
            for x in np.linspace(0.0, spec.pattern_width_m, 8)
            for y in np.linspace(0.0, spec.pattern_height_m, 5)
        ]
    )
    lidar_samples = [(timestamp_s, sparse_board) for timestamp_s in np.linspace(0.8, 1.2, 5)]
    pose_samples = [(timestamp_s, np.eye(4)) for timestamp_s in np.linspace(0.8, 1.2, 5)]

    planes = _associate_lidar_planes(
        (observation,),
        camera_fit=None,
        lidar_samples=lidar_samples,
        pose_samples=pose_samples,
        lidar_from_camera=np.eye(4),
        detector=BoardPlaneDetector(spec),
    )

    assert len(planes) == 1
    assert planes[0].normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-3)
    assert planes[0].distance_m == pytest.approx(1.0, abs=1e-3)


def test_lidar_plane_association_attempts_each_reference_frame_once():
    from mapping.pc.go2_mapping_calibration.analyzer import _associate_lidar_planes
    from mapping.pc.go2_mapping_calibration.target import BoardPlaneDetector, go2_lab_board_spec

    spec = go2_lab_board_spec()
    observation = CameraBoardObservation.from_pose(
        BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
        board=spec,
        header_timestamp_ns=1_000_000_000,
        recorder_timestamp_ns=1_000_000_000,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )

    class CountingDetector:
        def __init__(self):
            self.calls = 0
            self.real = BoardPlaneDetector(spec)

        def detect(self, *args, **kwargs):
            self.calls += 1
            return self.real.detect(*args, **kwargs)

    detector = CountingDetector()
    _associate_lidar_planes(
        (observation, observation),
        camera_fit=None,
        lidar_samples=[(1.0, np.asarray([[0.0, 0.0, 1.0]]))],
        pose_samples=[(1.0, np.eye(4))],
        lidar_from_camera=np.eye(4),
        detector=detector,
    )

    assert detector.calls == 1


def test_lidar_plane_association_reports_backing_wall_without_accepting_it():
    from mapping.pc.go2_mapping_calibration.analyzer import _associate_lidar_planes
    from mapping.pc.go2_mapping_calibration.target import BoardPlaneDetector, go2_lab_board_spec

    spec = go2_lab_board_spec()
    observation = CameraBoardObservation.from_pose(
        BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
        board=spec,
        header_timestamp_ns=1_000_000_000,
        recorder_timestamp_ns=1_000_000_000,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )
    wall = np.asarray(
        [[x, y, 1.0] for x in np.linspace(-1.0, 1.0, 80) for y in np.linspace(-1.0, 1.0, 80)]
    )
    rejection_reasons = []

    planes = _associate_lidar_planes(
        (observation,),
        camera_fit=None,
        lidar_samples=[(timestamp_s, wall) for timestamp_s in np.linspace(0.8, 1.2, 5)],
        pose_samples=[(timestamp_s, np.eye(4)) for timestamp_s in np.linspace(0.8, 1.2, 5)],
        lidar_from_camera=np.eye(4),
        detector=BoardPlaneDetector(
            spec,
            minimum_support=30,
            maximum_center_error_m=0.30,
            allow_backing_plane=True,
        ),
        rejection_reasons=rejection_reasons,
    )

    assert planes == []
    assert "FINITE_BOARD_BOUNDARY_NOT_VERIFIED" in rejection_reasons


def test_lidar_rejections_are_promoted_only_when_capture_lacks_accepted_planes():
    from mapping.pc.go2_mapping_calibration import analyzer

    blockers = getattr(analyzer, "_lidar_capture_blockers", None)
    assert blockers is not None
    reasons = (
        "FINITE_BOARD_BOUNDARY_NOT_VERIFIED",
        "PLANE_CENTER_MISMATCH",
        "FINITE_BOARD_BOUNDARY_NOT_VERIFIED",
    )

    assert blockers(20, reasons) == ()
    assert blockers(19, reasons) == (
        "FINITE_BOARD_BOUNDARY_NOT_VERIFIED",
        "PLANE_CENTER_MISMATCH",
        "INSUFFICIENT_LIDAR_BOARD_OBSERVATIONS",
    )


def test_operator_relaxed_lidar_association_retains_a_rejected_plane_for_diagnostic_math():
    from mapping.pc.go2_mapping_calibration.analyzer import _associate_lidar_planes
    from mapping.pc.go2_mapping_calibration.target import BoardPlaneDetection

    observation = CameraBoardObservation.from_pose(
        BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
        header_timestamp_ns=1_000_000_000,
        recorder_timestamp_ns=1_000_000_000,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )

    class RejectedBackingPlane:
        def detect(self, *_args, **_kwargs):
            return BoardPlaneDetection(
                accepted=False,
                normal_lidar=(0.0, 0.0, 1.0),
                distance_m=1.0,
                center_lidar=(0.0, 0.0, 1.0),
                extent_m=(1.0, 1.0),
                support_count=200,
                residual_p95_m=0.04,
                blocking_reasons=("FINITE_BOARD_BOUNDARY_NOT_VERIFIED",),
            )

    reasons = []
    planes = _associate_lidar_planes(
        (observation,),
        camera_fit=None,
        lidar_samples=[(1.0, np.asarray([[0.0, 0.0, 1.0]]))],
        pose_samples=[(1.0, np.eye(4))],
        lidar_from_camera=np.eye(4),
        detector=RejectedBackingPlane(),
        rejection_reasons=reasons,
        operator_relaxed=True,
    )

    assert len(planes) == 1
    assert planes[0].normal == pytest.approx((0.0, 0.0, 1.0))
    assert reasons == [
        "OPERATOR_RELAXED_LIDAR_PLANE:FINITE_BOARD_BOUNDARY_NOT_VERIFIED"
    ]


def test_lidar_association_cap_keeps_temporally_distributed_observations():
    from mapping.pc.go2_mapping_calibration.analyzer import _bounded_association_observations

    observations = tuple(range(11))

    assert _bounded_association_observations(observations, maximum=4) == (0, 3, 6, 10)
    assert _bounded_association_observations(observations, maximum=20) == observations

    with pytest.raises(ValueError, match="max_lidar_associations"):
        _bounded_association_observations(observations, maximum=0)


def test_operator_relaxed_association_uses_recorder_time_instead_of_failed_clock_fit():
    from mapping.pc.go2_mapping_calibration.analyzer import _association_clock_fit

    fit = object()

    assert _association_clock_fit(fit, operator_relaxed=False) is fit
    assert _association_clock_fit(fit, operator_relaxed=True) is None


def test_camera_board_observation_uses_physical_center_not_charuco_origin():
    pose = BoardPose(rotation_vector=(0.1, -0.2, 0.05), translation_m=(0.42, -0.31, 1.7))
    board = go2_lab_board_spec()

    observation = CameraBoardObservation.from_pose(
        pose,
        board=board,
        header_timestamp_ns=1_000_000_000,
        recorder_timestamp_ns=1_002_000_000,
        reprojection_p95_px=0.4,
        correspondence_count=88,
    )

    expected_rotation, _ = cv2.Rodrigues(np.asarray(pose.rotation_vector, dtype=np.float64))
    expected_center = expected_rotation @ np.asarray(
        [board.pattern_width_m / 2, board.pattern_height_m / 2, 0.0]
    ) + np.asarray(pose.translation_m)
    assert observation.center_camera == pytest.approx(expected_center)
    assert observation.plane.distance_m > 0.0
    assert observation.correspondence_count == 88
    assert observation.camera_from_board[:3, :3] == pytest.approx(expected_rotation)
    assert observation.camera_from_board[:3, 3] == pytest.approx(pose.translation_m)


def test_camera_board_observation_normalizes_array_like_transform():
    observation = CameraBoardObservation(
        header_timestamp_ns=1,
        recorder_timestamp_ns=2,
        center_camera=(0.0, 0.0, 1.0),
        plane=camera_plane_from_board_pose(
            BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)),
            recorder_time_s=2e-9,
        ),
        reprojection_p95_px=0.1,
        correspondence_count=12,
        camera_from_board=np.eye(4).tolist(),
    )

    assert isinstance(observation.camera_from_board, np.ndarray)
    assert observation.camera_from_board == pytest.approx(np.eye(4))
    assert not observation.camera_from_board.flags.writeable


def test_camera_plane_time_uses_affine_header_mapping_not_recorder_arrival():
    pose = BoardPose(rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0))
    observation = CameraBoardObservation.from_pose(
        pose,
        header_timestamp_ns=10_000_000_000,
        recorder_timestamp_ns=99_000_000_000,
        reprojection_p95_px=0.2,
        correspondence_count=80,
    )

    class _Fit:
        maximum_prediction_stddev_ms = 0.25

        @staticmethod
        def map_header_ns(value):
            return value + 2_000_000_000

    plane = mapped_camera_plane(observation, _Fit())

    assert plane.recorder_time_s == pytest.approx(12.0)
    assert plane.weight == pytest.approx(80.0)


def test_decoded_holdout_frames_are_copied_read_only_and_keep_both_clocks():
    depth_source = np.ones((2, 3), dtype=np.uint16)
    lidar_source = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))

    depth = DepthFrameEvidence(11, 10, "depth_optical", "16UC1", depth_source)
    lidar = LidarFrameEvidence(21, 20, "livox_frame", lidar_source)
    depth_source[0, 0] = 99
    lidar_source[0, 0] = 99.0

    assert depth.recorder_timestamp_ns == 11
    assert depth.header_timestamp_ns == 10
    assert depth.image[0, 0] == 1
    assert lidar.recorder_timestamp_ns == 21
    assert lidar.header_timestamp_ns == 20
    assert lidar.points_xyz[0, 0] == pytest.approx(1.0)
    assert not depth.image.flags.writeable
    assert not lidar.points_xyz.flags.writeable


def test_explicit_association_seed_takes_precedence_over_dynamic_tf_records():
    from inspect import signature

    from mapping.pc.go2_mapping_calibration import analyzer

    assert "initial_camera_from_lidar" in signature(analyzer.analyze_capture).parameters
    resolver = getattr(analyzer, "_camera_from_lidar_for_association", None)
    assert resolver is not None
    dynamic_records = (
        TransformRecord(1, "odom", "utlidar_lidar", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TransformRecord(2, "odom", "utlidar_lidar", (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    seed = np.eye(4)
    seed[:3, 3] = (0.1, -0.2, 0.3)

    actual = resolver(
        dynamic_records,
        target_frame="camera_color_optical_frame",
        source_frame="utlidar_lidar",
        initial_camera_from_lidar=seed,
    )

    assert actual == pytest.approx(seed)
    assert not actual.flags.writeable


def test_parallel_color_detection_bounds_workers_and_restores_global_order():
    from mapping.pc.go2_mapping_calibration.analyzer import (
        _ColorFrameDetection,
        _detect_default_color_parts,
    )
    from mapping.pc.go2_mapping_calibration.bag import CalibrationTopicMap

    requested_workers = []
    paths = tuple(Path(f"part-{index}.db3") for index in range(4))

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            requested_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, worker, tasks):
            return [worker(task) for task in tasks]

    def worker(task):
        return tuple(
            _ColorFrameDetection(
                recorder_timestamp_ns=10 - part_index,
                header_timestamp_ns=100 - part_index,
                part_index=part_index,
                message_index=0,
                detection=None,
                failure=None,
            )
            for part_index, _path in task.part_paths
        )

    results = _detect_default_color_parts(
        paths,
        CalibrationTopicMap.defaults(),
        camera_matrix=np.eye(3),
        distortion=np.zeros(5),
        workers=20,
        executor_factory=ImmediateExecutor,
        worker=worker,
    )

    assert requested_workers == [4]
    assert [item.recorder_timestamp_ns for item in results] == [7, 8, 9, 10]


def test_capture_analysis_uses_parallel_default_color_detection_budget(monkeypatch):
    from types import SimpleNamespace

    from mapping.pc.go2_mapping_calibration import analyzer
    from mapping.pc.go2_mapping_calibration.analyzer import _ColorFrameDetection
    from mapping.pc.go2_mapping_calibration.bag import CalibrationTopicMap
    from mapping.pc.go2_mapping_calibration.target import BoardDetection

    color_info = SimpleNamespace(
        k=np.eye(3), d=np.zeros(5), frame_id="camera_color_optical_frame"
    )
    detection = BoardDetection(
        accepted=True,
        marker_ids=(0,),
        charuco_ids=(0, 1, 2, 3, 4, 5),
        decoded_fraction=1.0,
        charuco_fraction=1.0,
        coverage_fraction=1.0,
        blur_variance=1.0,
        reprojection_p95_px=0.2,
        reprojection_residuals_px=(0.2,),
        pose=BoardPose(
            rotation_vector=(0.0, 0.0, 0.0), translation_m=(0.0, 0.0, 1.0)
        ),
        blocking_reasons=(),
    )
    requested_workers = []

    class FakeCapture:
        def __init__(self, _path, _topics, *, materialization_workers):
            self._paths = (Path("part-0.db3"), Path("part-1.db3"))
            self.topics = CalibrationTopicMap.defaults()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def inventory():
            return SimpleNamespace(missing_streams=())

        @staticmethod
        def iter_messages(_stream):
            return iter(())

    def info(_bag, stream, _blockers):
        if stream == "color_camera_info":
            return color_info, {"frame_id": color_info.frame_id}, [], []
        return None, None, [], []

    def detect_parts(_paths, _topics, **kwargs):
        requested_workers.append(kwargs["workers"])
        return (
            _ColorFrameDetection(
                recorder_timestamp_ns=10,
                header_timestamp_ns=9,
                part_index=0,
                message_index=0,
                detection=detection,
                failure=None,
            ),
        )

    monkeypatch.setattr(analyzer, "CalibrationCapture", FakeCapture)
    monkeypatch.setattr(analyzer, "_info", info)
    monkeypatch.setattr(analyzer, "_detect_default_color_parts", detect_parts)

    result = analyzer.analyze_capture("capture", target_detection_workers=5)

    assert requested_workers == [5]
    assert result.board_detections == 1
    assert result.accepted_board_detections == 1
