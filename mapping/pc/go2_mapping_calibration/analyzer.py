"""Extract auditable calibration observations from one verified ROS 2 bag."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from mapping.pc.go2_mapping_rgbd.decode import CameraInfoRecord, ImageRecord, TransformRecord, decode_camera_info, decode_image, decode_tf_message

from .analysis import (
    CalibrationAnalysisError,
    motion_excitation_from_poses,
    resolve_frame_transform,
    segment_lio_pose_epochs,
)
from .axis_contract import diagnostic_camera_from_lidar_seed
from .bag import CalibrationBag, CalibrationCapture, CalibrationTopicMap, CaptureInventory
from .capture_analysis import CameraBoardObservation, aggregate_lidar_window, camera_info_identity, mapped_camera_plane
from .cdr import CalibrationCdrError, decode_imu, decode_pointcloud, decode_pose
from .models import MotionExcitation, PlaneObservation
from .target import BoardDetection, BoardPlaneDetector, TargetDetector, go2_lab_board_spec
from .timebase import AffineClockEstimator, AffineClockFit


def _errors() -> tuple[type[BaseException], ...]:
    return (ValueError, struct.error, CalibrationCdrError)


@dataclass(frozen=True)
class CaptureAnalysis:
    inventory: CaptureInventory
    clock_fits: dict[str, AffineClockFit]
    color_camera_info: CameraInfoRecord | None
    depth_camera_info: CameraInfoRecord | None
    color_camera_identity: dict[str, Any] | None
    depth_camera_identity: dict[str, Any] | None
    tf_records: tuple[TransformRecord, ...]
    tf_static_records: tuple[TransformRecord, ...]
    camera_planes: tuple[PlaneObservation, ...]
    camera_board_observations: tuple[CameraBoardObservation, ...]
    lidar_planes: tuple[PlaneObservation, ...]
    pose_records: tuple[tuple[int, np.ndarray], ...]
    depth_records: tuple[tuple[int, ImageRecord], ...]
    lidar_records: tuple[tuple[int, int, str, np.ndarray], ...]
    motion: MotionExcitation
    board_detections: int
    accepted_board_detections: int
    accepted_lidar_planes: int
    blockers: tuple[str, ...]
    lidar_rejection_reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.blockers

    @property
    def artifact_sha256(self) -> str:
        return self.inventory.artifact_sha256


@dataclass(frozen=True)
class _ColorFrameDetection:
    recorder_timestamp_ns: int
    header_timestamp_ns: int | None
    part_index: int
    message_index: int
    detection: BoardDetection | None
    failure: str | None


@dataclass(frozen=True)
class _ColorPartTask:
    part_paths: tuple[tuple[int, Path], ...]
    topics: CalibrationTopicMap
    camera_matrix: np.ndarray
    distortion: np.ndarray


def _detect_color_part_task(task: _ColorPartTask) -> tuple[_ColorFrameDetection, ...]:
    """Detect the locked board in an independent read-only bag-part group."""

    cv2.setNumThreads(1)
    detector = TargetDetector(
        go2_lab_board_spec(),
        camera_matrix=task.camera_matrix,
        distortion=task.distortion,
    )
    results: list[_ColorFrameDetection] = []
    for part_index, path in task.part_paths:
        with CalibrationBag(path, task.topics) as bag:
            for message_index, (record_time, payload) in enumerate(
                bag.iter_messages("color")
            ):
                try:
                    image = decode_image(payload)
                except _errors() as exc:
                    results.append(
                        _ColorFrameDetection(
                            recorder_timestamp_ns=record_time,
                            header_timestamp_ns=None,
                            part_index=part_index,
                            message_index=message_index,
                            detection=None,
                            failure=f"COLOR_DECODE_FAILED:{exc}",
                        )
                    )
                    continue
                try:
                    detection = detector.detect(image.image)
                except _errors() as exc:
                    results.append(
                        _ColorFrameDetection(
                            recorder_timestamp_ns=record_time,
                            header_timestamp_ns=image.header_timestamp_ns,
                            part_index=part_index,
                            message_index=message_index,
                            detection=None,
                            failure=f"COLOR_TARGET_DETECT_FAILED:{exc}",
                        )
                    )
                    continue
                results.append(
                    _ColorFrameDetection(
                        recorder_timestamp_ns=record_time,
                        header_timestamp_ns=image.header_timestamp_ns,
                        part_index=part_index,
                        message_index=message_index,
                        detection=detection,
                        failure=None,
                    )
                )
    return tuple(results)


def _detect_default_color_parts(
    paths: Sequence[Path],
    topics: CalibrationTopicMap,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    workers: int,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    worker: Callable[[_ColorPartTask], tuple[_ColorFrameDetection, ...]] = _detect_color_part_task,
) -> tuple[_ColorFrameDetection, ...]:
    """Distribute independent recorded RGB parts without changing frame order."""

    if int(workers) <= 0:
        raise ValueError("target_detection_workers must be positive")
    if not paths:
        return ()
    worker_count = min(int(workers), len(paths))
    groups: list[list[tuple[int, Path]]] = [[] for _ in range(worker_count)]
    for part_index, path in enumerate(paths):
        groups[part_index % worker_count].append((part_index, path))
    tasks = tuple(
        _ColorPartTask(
            part_paths=tuple(group),
            topics=topics,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            distortion=np.asarray(distortion, dtype=np.float64),
        )
        for group in groups
    )
    with executor_factory(max_workers=worker_count) as executor:
        batches = tuple(executor.map(worker, tasks))
    return tuple(
        sorted(
            (item for batch in batches for item in batch),
            key=lambda item: (
                item.recorder_timestamp_ns,
                item.part_index,
                item.message_index,
            ),
        )
    )


def _fit(headers: list[int], arrivals: list[int], estimator: AffineClockEstimator) -> AffineClockFit | None:
    if len(headers) < 2:
        return None
    order = np.argsort(np.asarray(headers, dtype=np.int64), kind="stable")
    h = np.asarray(headers, dtype=np.float64)[order]
    r = np.asarray(arrivals, dtype=np.float64)[order]
    unique, indexes = np.unique(h, return_index=True)
    if len(unique) != len(h):
        h, r = unique, r[indexes]
    try:
        return estimator.fit(h, r)
    except (ValueError, np.linalg.LinAlgError):
        return None


def _info(bag: CalibrationCapture, stream: str, blockers: list[str]):
    records = []
    headers, arrivals = [], []
    for record_time, payload in bag.iter_messages(stream):
        try:
            value = decode_camera_info(payload)
        except _errors() as exc:
            blockers.append(f"CAMERA_INFO_DECODE_FAILED:{stream}:{exc}")
            continue
        records.append(value)
        headers.append(value.header_timestamp_ns)
        arrivals.append(record_time)
    if not records:
        blockers.append(f"CAMERA_INFO_NO_VALID_RECORD:{stream}")
        return None, None, headers, arrivals
    identity = camera_info_identity(records[0])
    if any(camera_info_identity(item).sha256 != identity.sha256 for item in records[1:]):
        blockers.append(f"CAMERA_INFO_CHANGED_DURING_CAPTURE:{stream}")
    return records[0], {"sha256": identity.sha256, "raw_message_sha256": identity.raw_message_sha256, "frame_id": records[0].frame_id, "contract": identity.contract}, headers, arrivals


def _camera_from_lidar_for_association(
    records: tuple[TransformRecord, ...] | list[TransformRecord],
    *,
    target_frame: str,
    source_frame: str,
    initial_camera_from_lidar: np.ndarray | None,
) -> np.ndarray:
    if initial_camera_from_lidar is None:
        return resolve_frame_transform(records, target_frame=target_frame, source_frame=source_frame)
    value = np.asarray(initial_camera_from_lidar, dtype=np.float64)
    rotation = value[:3, :3] if value.shape == (4, 4) else np.empty((0, 0))
    if (
        value.shape != (4, 4)
        or not np.all(np.isfinite(value))
        or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise CalibrationAnalysisError("initial_camera_from_lidar must be a finite rigid transform")
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _associate_lidar_planes(
    observations: tuple[CameraBoardObservation, ...] | list[CameraBoardObservation],
    *,
    camera_fit: AffineClockFit | None,
    lidar_samples: list[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
    pose_samples: list[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
    lidar_from_camera: np.ndarray,
    detector: BoardPlaneDetector,
    rejection_reasons: list[str] | None = None,
    operator_relaxed: bool = False,
) -> list[PlaneObservation]:
    result: list[PlaneObservation] = []
    used_reference_times: set[float] = set()
    rotation = np.asarray(lidar_from_camera[:3, :3], dtype=np.float64)
    translation = np.asarray(lidar_from_camera[:3, 3], dtype=np.float64)
    for observation in observations:
        camera_time = (
            camera_fit.map_header_ns(observation.header_timestamp_ns) / 1e9
            if camera_fit is not None
            else observation.recorder_timestamp_ns / 1e9
        )
        window = aggregate_lidar_window(
            lidar_samples,
            pose_samples,
            target_time_s=camera_time,
        )
        if window is None or window.reference_time_s in used_reference_times:
            continue
        used_reference_times.add(window.reference_time_s)
        camera_center = np.asarray(observation.center_camera, dtype=np.float64)
        predicted_center = rotation @ camera_center + translation
        predicted_normal = rotation @ np.asarray(observation.plane.normal, dtype=np.float64)
        detection = detector.detect(
            window.points_xyz,
            predicted_center_lidar=predicted_center,
            predicted_normal_lidar=predicted_normal,
            association_error_ms=20.0,
        )
        has_plane = (
            detection.normal_lidar is not None and detection.distance_m is not None
        )
        if not detection.accepted and operator_relaxed and has_plane:
            if rejection_reasons is not None:
                rejection_reasons.extend(
                    f"OPERATOR_RELAXED_LIDAR_PLANE:{reason}"
                    for reason in detection.blocking_reasons
                )
        elif not detection.accepted or not has_plane:
            if rejection_reasons is not None:
                rejection_reasons.extend(detection.blocking_reasons)
            continue
        result.append(
            PlaneObservation(
                window.reference_time_s,
                detection.normal_lidar,
                detection.distance_m,
                weight=float(detection.support_count),
            )
        )
    return result


def _lidar_capture_blockers(
    accepted_lidar: int,
    rejection_reasons: tuple[str, ...] | list[str],
    *,
    minimum_observations: int = 20,
) -> tuple[str, ...]:
    """Promote per-frame LiDAR diagnostics only when the count gate fails."""

    if accepted_lidar >= minimum_observations:
        return ()
    return tuple(
        dict.fromkeys(
            (*rejection_reasons, "INSUFFICIENT_LIDAR_BOARD_OBSERVATIONS")
        )
    )


def _bounded_association_observations(
    observations: Sequence[Any], *, maximum: int | None
) -> tuple[Any, ...]:
    """Keep a deterministic temporal spread for expensive LiDAR plane fits."""

    values = tuple(observations)
    if maximum is None or len(values) <= int(maximum):
        return values
    if int(maximum) <= 0:
        raise ValueError("max_lidar_associations must be positive")
    indexes = np.linspace(0, len(values) - 1, int(maximum), dtype=np.int64)
    return tuple(values[int(index)] for index in indexes)


def _association_clock_fit(
    fit: AffineClockFit | None, *, operator_relaxed: bool
) -> AffineClockFit | None:
    """Avoid applying a known-failed clock mapping in forced diagnostics."""

    return None if operator_relaxed else fit


def analyze_capture(
    path: str,
    *,
    topics=None,
    detector: TargetDetector | None = None,
    allow_incomplete: bool = False,
    initial_camera_from_lidar: np.ndarray | None = None,
    materialization_workers: int = 1,
    target_detection_workers: int = 1,
    operator_relaxed_lidar_planes: bool = False,
    max_lidar_associations: int | None = None,
) -> CaptureAnalysis:
    if int(target_detection_workers) <= 0:
        raise ValueError("target_detection_workers must be positive")
    if max_lidar_associations is not None and int(max_lidar_associations) <= 0:
        raise ValueError("max_lidar_associations must be positive")
    blockers: list[str] = []
    with CalibrationCapture(
        path, topics, materialization_workers=materialization_workers
    ) as bag:
        inventory = bag.inventory()
        if inventory.missing_streams and not allow_incomplete:
            blockers.extend(f"MISSING_REQUIRED_STREAM:{item}" for item in inventory.missing_streams)
        color_info, color_identity, _color_info_h, _color_info_r = _info(bag, "color_camera_info", blockers)
        depth_info, depth_identity, _depth_info_h, _depth_info_r = _info(bag, "depth_camera_info", blockers)
        use_parallel_default_color_detection = (
            detector is None
            and color_info is not None
            and int(target_detection_workers) > 1
        )
        target_detector = detector
        if (
            target_detector is None
            and color_info is not None
            and not use_parallel_default_color_detection
        ):
            target_detector = TargetDetector(go2_lab_board_spec(), camera_matrix=color_info.k, distortion=color_info.d)
        camera_planes, camera_board_observations, camera_count, camera_accepted = [], [], 0, 0
        color_times, color_headers = [], []
        if use_parallel_default_color_detection:
            parallel_frames = _detect_default_color_parts(
                bag._paths,
                bag.topics,
                camera_matrix=color_info.k,
                distortion=color_info.d,
                workers=int(target_detection_workers),
            )
            for frame in parallel_frames:
                if frame.failure is not None:
                    blockers.append(frame.failure)
                if frame.header_timestamp_ns is None:
                    continue
                color_times.append(frame.recorder_timestamp_ns)
                color_headers.append(frame.header_timestamp_ns)
                camera_count += 1
                detection = frame.detection
                if not detection or not detection.accepted or detection.pose is None:
                    continue
                camera_accepted += 1
                observation = CameraBoardObservation.from_pose(
                    detection.pose,
                    board=go2_lab_board_spec(),
                    header_timestamp_ns=frame.header_timestamp_ns,
                    recorder_timestamp_ns=frame.recorder_timestamp_ns,
                    reprojection_p95_px=float(detection.reprojection_p95_px),
                    correspondence_count=len(detection.charuco_ids),
                    reprojection_residuals_px=detection.reprojection_residuals_px,
                )
                camera_board_observations.append(observation)
                camera_planes.append(observation.plane)
        else:
            for record_time, payload in bag.iter_messages("color"):
                try:
                    image = decode_image(payload)
                except _errors() as exc:
                    blockers.append(f"COLOR_DECODE_FAILED:{exc}")
                    continue
                color_times.append(record_time); color_headers.append(image.header_timestamp_ns); camera_count += 1
                if target_detector is None:
                    continue
                try:
                    detection = target_detector.detect(image.image)
                except _errors() as exc:
                    blockers.append(f"COLOR_TARGET_DETECT_FAILED:{exc}")
                    continue
                if detection.accepted and detection.pose is not None:
                    camera_accepted += 1
                    observation = CameraBoardObservation.from_pose(
                        detection.pose,
                        board=target_detector.spec,
                        header_timestamp_ns=image.header_timestamp_ns,
                        recorder_timestamp_ns=record_time,
                        reprojection_p95_px=float(detection.reprojection_p95_px),
                        correspondence_count=len(detection.charuco_ids),
                        reprojection_residuals_px=detection.reprojection_residuals_px,
                    )
                    camera_board_observations.append(observation)
                    camera_planes.append(observation.plane)
        depth_times, depth_headers, depth_records = [], [], []
        for record_time, payload in bag.iter_messages("depth"):
            try:
                depth_image = decode_image(payload)
            except _errors() as exc:
                blockers.append(f"DEPTH_DECODE_FAILED:{exc}")
                continue
            depth_times.append(record_time)
            depth_headers.append(depth_image.header_timestamp_ns)
            depth_records.append((record_time, depth_image))
        tf_records: list[TransformRecord] = []
        tf_static_records: list[TransformRecord] = []
        for stream in ("tf", "tf_static"):
            for _, payload in bag.iter_messages(stream):
                try:
                    decoded = decode_tf_message(payload)
                    tf_records.extend(decoded)
                    if stream == "tf_static":
                        tf_static_records.extend(decoded)
                except _errors() as exc:
                    blockers.append(f"TF_DECODE_FAILED:{stream}:{exc}")
        pose_records, pose_h, pose_r = [], [], []
        for record_time, payload in bag.iter_messages("lio_pose"):
            try:
                value = decode_pose(payload)
            except _errors() as exc:
                blockers.append(f"LIO_POSE_DECODE_FAILED:{exc}")
                continue
            pose_records.append((record_time, value.pose)); pose_h.append(value.header_timestamp_ns); pose_r.append(record_time)
        lidar_frames, lidar_h, lidar_r = [], [], []
        for record_time, payload in bag.iter_messages("lidar"):
            try:
                value = decode_pointcloud(payload)
            except _errors() as exc:
                blockers.append(f"LIDAR_DECODE_FAILED:{exc}")
                continue
            lidar_frames.append((record_time, value.header_timestamp_ns, value.frame_id, value.points_xyz)); lidar_h.append(value.header_timestamp_ns); lidar_r.append(record_time)
        imu_h, imu_r = [], []
        for record_time, payload in bag.iter_messages("imu"):
            try:
                value = decode_imu(payload)
            except _errors() as exc:
                blockers.append(f"IMU_DECODE_FAILED:{exc}"); continue
            imu_h.append(value.header_timestamp_ns); imu_r.append(record_time)
        estimator = AffineClockEstimator()
        streams = {"color": (color_headers, color_times), "depth": (depth_headers, depth_times), "lio_pose": (pose_h, pose_r), "lidar": (lidar_h, lidar_r), "imu": (imu_h, imu_r)}
        fits: dict[str, AffineClockFit] = {}
        for stream, (headers, arrivals) in streams.items():
            fit = _fit(headers, arrivals, estimator)
            if fit is None:
                blockers.append(f"CLOCK_FIT_INSUFFICIENT:{stream}")
            else:
                fits[stream] = fit
                blockers.extend(f"CLOCK_FIT_FAILED:{stream}:{reason}" for reason in fit.reasons)
        motion = motion_excitation_from_poses(pose_records)
        if not pose_records:
            blockers.append("LIO_POSE_NO_VALID_RECORD")
        if motion.lio_reset_detected:
            blockers.append("LIO_EPOCH_RESET_DETECTED")
        for name, value, limit in (
            ("ROTATION_SPAN_BELOW_30_DEG", motion.calibration_rotation_span_deg, 30.0),
            ("TRANSLATION_SPAN_BELOW_0_5_M", motion.calibration_translation_span_m, 0.5),
            ("ANGULAR_EXCITATION_BELOW_ONE_SECOND", motion.angular_speed_duration_s, 1.0),
        ):
            if value < limit: blockers.append(name)
        if motion.rotation_axes < 2: blockers.append("ROTATION_AXES_BELOW_TWO")
        association_observations = _bounded_association_observations(
            camera_board_observations,
            maximum=max_lidar_associations,
        )
        camera_fit = _association_clock_fit(
            fits.get("color"), operator_relaxed=operator_relaxed_lidar_planes
        )
        lidar_fit = _association_clock_fit(
            fits.get("lidar"), operator_relaxed=operator_relaxed_lidar_planes
        )
        if camera_fit is not None:
            camera_planes = [
                mapped_camera_plane(item, camera_fit)
                for item in association_observations
            ]
        else:
            camera_planes = [item.plane for item in association_observations]
        lidar_planes, accepted_lidar = [], 0
        if tf_records and color_info is not None and lidar_frames:
            try:
                lidar_frame = lidar_frames[0][2]
                camera_from_lidar = _camera_from_lidar_for_association(
                    tf_records,
                    target_frame=color_info.frame_id,
                    source_frame=lidar_frame,
                    initial_camera_from_lidar=initial_camera_from_lidar,
                )
                plane_detector = BoardPlaneDetector(
                    go2_lab_board_spec(),
                    minimum_support=3 if operator_relaxed_lidar_planes else 30,
                    distance_threshold_m=0.050 if operator_relaxed_lidar_planes else 0.012,
                    maximum_center_error_m=5.0 if operator_relaxed_lidar_planes else 0.30,
                    maximum_normal_error_deg=180.0 if operator_relaxed_lidar_planes else 25.0,
                    allow_backing_plane=True,
                )
                lidar_from_camera = np.linalg.inv(camera_from_lidar)
                pose_fit = _association_clock_fit(
                    fits.get("lio_pose"),
                    operator_relaxed=operator_relaxed_lidar_planes,
                )
                pose_epochs = segment_lio_pose_epochs(pose_records)
                pose_frames_s = []
                for epoch_id, epoch in enumerate(pose_epochs):
                    for source_index, (record_time, pose) in zip(epoch.source_indices, epoch.poses):
                        header = pose_h[source_index]
                        mapped_time = (
                            pose_fit.map_header_ns(header) / 1e9
                            if pose_fit is not None
                            else record_time / 1e9
                        )
                        pose_frames_s.append((mapped_time, pose, epoch_id))
                lidar_rejection_reasons: list[str] = []
                lidar_frames_s = [
                    (lidar_fit.map_header_ns(header) / 1e9 if lidar_fit is not None else timestamp / 1e9, points)
                    for timestamp, header, _, points in lidar_frames
                ]
                lidar_planes = _associate_lidar_planes(
                    association_observations,
                    camera_fit=camera_fit,
                    lidar_samples=lidar_frames_s,
                    pose_samples=pose_frames_s,
                    lidar_from_camera=lidar_from_camera,
                    detector=plane_detector,
                    rejection_reasons=lidar_rejection_reasons,
                    operator_relaxed=operator_relaxed_lidar_planes,
                )
                accepted_lidar = len(lidar_planes)
            except (CalibrationAnalysisError, np.linalg.LinAlgError, StopIteration) as exc:
                blockers.append(f"INITIAL_FRAME_CHAIN_REQUIRED:{exc}")
                lidar_rejection_reasons = []
        else:
            blockers.append("INITIAL_FRAME_CHAIN_REQUIRED")
            lidar_rejection_reasons = []
        if camera_accepted < 20: blockers.append("INSUFFICIENT_CAMERA_BOARD_OBSERVATIONS")
        blockers.extend(_lidar_capture_blockers(accepted_lidar, lidar_rejection_reasons))
        return CaptureAnalysis(
            inventory,
            fits,
            color_info,
            depth_info,
            color_identity,
            depth_identity,
            tuple(tf_records),
            tuple(tf_static_records),
            tuple(camera_planes),
            tuple(association_observations),
            tuple(lidar_planes),
            tuple(pose_records),
            tuple(depth_records),
            tuple(lidar_frames),
            motion,
            camera_count,
            camera_accepted,
            accepted_lidar,
            tuple(dict.fromkeys(blockers)),
            tuple(dict.fromkeys(lidar_rejection_reasons)),
        )
