"""Untouched holdout metrics for calibration promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config
from .capture_analysis import CameraBoardObservation
from .depth_projection import image_to_depth_m
from .target import BoardPlaneDetector, BoardSpec, go2_lab_board_spec
from .capture_analysis import nearest_within
from .analysis import CalibrationAnalysisError, resolve_frame_transform


@dataclass(frozen=True)
class HoldoutMetrics:
    accepted: bool
    accepted_board_poses: int
    correspondences_per_pose: tuple[int, ...]
    total_correspondences: int
    validated_range_min_m: float | None
    validated_range_max_m: float | None
    camera_reprojection_median_px: float | None
    camera_reprojection_p95_px: float | None
    depth_median_absolute_residual_m: float | None
    depth_p95_absolute_residual_m: float | None
    absolute_mean_signed_bias_m: float | None
    occlusion_conflict_fraction: float | None
    free_space_violation_fraction: float | None
    t_d_c_validated: bool
    blockers: tuple[str, ...]

    def manifest_payload(self, artifact_sha256: str) -> dict[str, object]:
        if not self.accepted:
            raise ValueError("blocked holdout metrics cannot enter a calibration manifest")
        return {
            "artifact_sha256": artifact_sha256,
            "accepted_board_poses": self.accepted_board_poses,
            "correspondences_per_pose": list(self.correspondences_per_pose),
            "total_correspondences": self.total_correspondences,
            "validated_range_min_m": self.validated_range_min_m,
            "validated_range_max_m": self.validated_range_max_m,
            "camera_reprojection_median_px": self.camera_reprojection_median_px,
            "camera_reprojection_p95_px": self.camera_reprojection_p95_px,
            "depth_median_absolute_residual_m": self.depth_median_absolute_residual_m,
            "depth_p95_absolute_residual_m": self.depth_p95_absolute_residual_m,
            "absolute_mean_signed_bias_m": self.absolute_mean_signed_bias_m,
            "occlusion_conflict_fraction": self.occlusion_conflict_fraction,
            "free_space_violation_fraction": self.free_space_violation_fraction,
            "t_d_c_validated": self.t_d_c_validated,
        }


def select_independent_board_observations(
    observations: Sequence[CameraBoardObservation],
    *,
    minimum_translation_m: float = 0.05,
    minimum_rotation_deg: float = 10.0,
) -> tuple[CameraBoardObservation, ...]:
    """Greedily retain board views with real spatial excitation.

    Adjacent frames of a stationary board are not independent evidence. A new
    view is retained only when it differs from every retained view by either a
    meaningful translation or rotation. The first observation is retained to
    make the result deterministic.
    """
    if minimum_translation_m < 0.0 or minimum_rotation_deg < 0.0:
        raise ValueError("independence thresholds cannot be negative")
    ordered = sorted(observations, key=lambda item: item.header_timestamp_ns)
    selected: list[CameraBoardObservation] = []
    for candidate in ordered:
        if not selected:
            selected.append(candidate)
            continue
        candidate_transform = np.asarray(candidate.camera_from_board, dtype=np.float64)
        candidate_rotation = candidate_transform[:3, :3]
        candidate_translation = candidate_transform[:3, 3]
        independent = True
        for previous in selected:
            previous_transform = np.asarray(previous.camera_from_board, dtype=np.float64)
            translation_delta = float(np.linalg.norm(candidate_translation - previous_transform[:3, 3]))
            relative = candidate_rotation @ previous_transform[:3, :3].T
            cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            rotation_delta = float(np.degrees(np.arccos(cosine)))
            if translation_delta < minimum_translation_m and rotation_delta < minimum_rotation_deg:
                independent = False
                break
        if independent:
            selected.append(candidate)
    return tuple(selected)


def board_depth_correspondences(
    depth_image: np.ndarray,
    camera_matrix: np.ndarray,
    depth_from_board: np.ndarray,
    board: BoardSpec,
    *,
    depth_encoding: str = "32FC1",
    depth_scale_m: float = 0.001,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[int, int], ...]]:
    """Project native-depth pixels onto a finite ChArUco board rectangle.

    ``depth_from_board`` is the measured transform from board coordinates to
    the native depth optical frame.  Rays that do not hit the physical board
    are excluded; no RGB-aligned or interpolated depth is accepted here.
    """
    depth = image_to_depth_m(depth_image, depth_encoding, depth_scale_m)
    camera = np.asarray(camera_matrix, dtype=np.float64)
    transform = np.asarray(depth_from_board, dtype=np.float64)
    if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
        raise ValueError("camera_matrix must be a finite 3 by 3 matrix")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("depth_from_board must be a finite 4 by 4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("depth_from_board must be homogeneous")
    if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    board_from_depth = np.linalg.inv(transform)
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    rays = np.stack(
        ((cols - camera[0, 2]) / camera[0, 0], (rows - camera[1, 2]) / camera[1, 1], np.ones_like(rows)),
        axis=-1,
    ).reshape(-1, 3)
    rays_board = rays @ board_from_depth[:3, :3].T
    origins_board = np.broadcast_to(board_from_depth[:3, 3], rays_board.shape)
    denominator = rays_board[:, 2]
    valid = np.abs(denominator) > 1e-12
    scale = -origins_board[:, 2] / denominator
    valid &= scale > 0.0
    board_points = origins_board + scale[:, None] * rays_board
    valid &= board_points[:, 0] >= 0.0
    valid &= board_points[:, 0] <= board.pattern_width_m
    valid &= board_points[:, 1] >= 0.0
    valid &= board_points[:, 1] <= board.pattern_height_m
    valid &= np.isfinite(scale)
    if not np.any(valid):
        return (), (), ()
    flat_depth = depth.reshape(-1)
    valid_indices = np.flatnonzero(valid)
    observed = flat_depth[valid_indices]
    predicted = (transform[:3, :3] @ (board_points[valid_indices].T) + transform[:3, 3:4])[2]
    usable = np.isfinite(observed) & (observed > 0.0) & np.isfinite(predicted) & (predicted > 0.0)
    valid_indices = valid_indices[usable]
    observed = observed[usable]
    predicted = predicted[usable]
    pixels = tuple((int(index // width), int(index % width)) for index in valid_indices)
    return tuple(float(value) for value in observed), tuple(float(value) for value in predicted), pixels


def evaluate_holdout_metrics(
    pose_depth_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    reprojection_residuals_px: np.ndarray,
    t_d_c_validated: bool,
    additional_blockers: Sequence[str] = (),
) -> HoldoutMetrics:
    config = load_reconstruction_config().appearance
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    blockers: list[str] = list(additional_blockers)
    for index, (observed, predicted) in enumerate(pose_depth_pairs):
        observed_array = np.asarray(observed, dtype=np.float64).reshape(-1)
        predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if len(observed_array) != len(predicted_array):
            raise ValueError(f"holdout pose {index} observed/predicted lengths differ")
        valid = np.isfinite(observed_array) & np.isfinite(predicted_array)
        valid &= observed_array >= config.depth_validation.minimum_range_m
        valid &= observed_array <= config.depth_validation.maximum_range_m
        valid &= predicted_array > 0.0
        pairs.append((observed_array[valid], predicted_array[valid]))
    counts = tuple(len(observed) for observed, _ in pairs)
    total = sum(counts)
    if len(pairs) < config.holdout.minimum_board_poses:
        blockers.append("HOLDOUT_BOARD_POSES_BELOW_MINIMUM")
    if not counts or min(counts) < config.holdout.minimum_correspondences_per_pose:
        blockers.append("HOLDOUT_CORRESPONDENCES_PER_POSE_BELOW_MINIMUM")
    if total < config.holdout.minimum_total_correspondences:
        blockers.append("HOLDOUT_TOTAL_CORRESPONDENCES_BELOW_MINIMUM")
    reprojection = np.asarray(reprojection_residuals_px, dtype=np.float64).reshape(-1)
    reprojection = reprojection[np.isfinite(reprojection) & (reprojection >= 0.0)]
    reprojection_median = None if len(reprojection) == 0 else float(np.median(reprojection))
    reprojection_p95 = None if len(reprojection) == 0 else float(np.percentile(reprojection, 95))
    if reprojection_median is None:
        blockers.append("HOLDOUT_REPROJECTION_MISSING")
    else:
        if reprojection_median > config.holdout.maximum_reprojection_median_px:
            blockers.append("HOLDOUT_REPROJECTION_MEDIAN_EXCEEDED")
        if reprojection_p95 > config.holdout.maximum_reprojection_p95_px:
            blockers.append("HOLDOUT_REPROJECTION_P95_EXCEEDED")
    observed_all = np.concatenate([item[0] for item in pairs]) if total else np.asarray([], dtype=np.float64)
    predicted_all = np.concatenate([item[1] for item in pairs]) if total else np.asarray([], dtype=np.float64)
    residual = observed_all - predicted_all
    absolute = np.abs(residual)
    minimum = None if total == 0 else float(np.min(observed_all))
    maximum = None if total == 0 else float(np.max(observed_all))
    median = None if total == 0 else float(np.median(absolute))
    p95 = None if total == 0 else float(np.percentile(absolute, 95))
    bias = None if total == 0 else abs(float(np.mean(residual)))
    threshold = np.maximum(0.030, 0.015 * predicted_all) if total else np.asarray([])
    occlusion = None if total == 0 else float(np.mean(residual < -threshold))
    free_space = None if total == 0 else float(np.mean(residual > threshold))
    gates = config.depth_validation
    if median is not None:
        if median > gates.maximum_median_absolute_residual_m:
            blockers.append("HOLDOUT_DEPTH_MEDIAN_EXCEEDED")
        if p95 > gates.maximum_p95_absolute_residual_m:
            blockers.append("HOLDOUT_DEPTH_P95_EXCEEDED")
        if bias > gates.maximum_absolute_mean_signed_bias_m:
            blockers.append("HOLDOUT_DEPTH_BIAS_EXCEEDED")
        if occlusion > gates.maximum_occlusion_conflict_fraction:
            blockers.append("HOLDOUT_OCCLUSION_CONFLICT_EXCEEDED")
        if free_space > gates.maximum_free_space_violation_fraction:
            blockers.append("HOLDOUT_FREE_SPACE_VIOLATION_EXCEEDED")
    if not t_d_c_validated:
        blockers.append("HOLDOUT_T_D_C_NOT_VALIDATED")
    return HoldoutMetrics(
        accepted=not blockers,
        accepted_board_poses=len(pairs),
        correspondences_per_pose=counts,
        total_correspondences=total,
        validated_range_min_m=minimum,
        validated_range_max_m=maximum,
        camera_reprojection_median_px=reprojection_median,
        camera_reprojection_p95_px=reprojection_p95,
        depth_median_absolute_residual_m=median,
        depth_p95_absolute_residual_m=p95,
        absolute_mean_signed_bias_m=bias,
        occlusion_conflict_fraction=occlusion,
        free_space_violation_fraction=free_space,
        t_d_c_validated=t_d_c_validated,
        blockers=tuple(blockers),
    )


def evaluate_holdout_capture(
    analysis: Any,
    *,
    t_d_c: np.ndarray,
    t_c_l: np.ndarray,
    expected_camera_serial: str,
    board: BoardSpec | None = None,
    maximum_depth_time_delta_s: float = 0.100,
    maximum_lidar_time_delta_s: float = 0.100,
    maximum_lidar_plane_residual_m: float = 0.010,
    camera_lidar_delta_ms: float = 0.0,
) -> HoldoutMetrics:
    """Evaluate one excluded capture without changing any estimated parameter.

    The supplied transforms are frozen outputs from the three estimation
    captures. This function only consumes the holdout's decoded evidence and
    therefore cannot leak holdout observations into the solver.
    """
    blockers: list[str] = []
    board = board or go2_lab_board_spec()
    t_d_c = _rigid_transform(t_d_c, "t_d_c")
    t_c_l = _rigid_transform(t_c_l, "t_c_l")
    runtime = getattr(getattr(analysis, "inventory", None), "runtime_manifest", None)
    serial_validated = bool(runtime is not None and runtime.camera_serial == expected_camera_serial)
    if runtime is None:
        blockers.append("HOLDOUT_RUNTIME_MANIFEST_MISSING")
    elif runtime.camera_serial != expected_camera_serial:
        blockers.append("HOLDOUT_CAMERA_SERIAL_MISMATCH")
    color_info = getattr(analysis, "color_camera_info", None)
    depth_info = getattr(analysis, "depth_camera_info", None)
    if color_info is None or depth_info is None:
        blockers.append("HOLDOUT_CAMERA_INFO_MISSING")
    depth_scale_m = None if runtime is None else getattr(runtime, "depth_scale_m", None)
    if depth_scale_m is None and any(image.encoding == "16UC1" for _, image in getattr(analysis, "depth_records", ())):
        blockers.append("HOLDOUT_DEPTH_SCALE_UNVERIFIED")
    elif depth_scale_m is None:
        depth_scale_m = 1.0
    static_t_d_c_validated = False
    if (
        color_info is not None
        and depth_info is not None
        and hasattr(color_info, "frame_id")
        and hasattr(depth_info, "frame_id")
    ):
        try:
            static_t_d_c = resolve_frame_transform(
                getattr(analysis, "tf_static_records", ()),
                target_frame=depth_info.frame_id,
                source_frame=color_info.frame_id,
            )
            if not np.allclose(static_t_d_c, t_d_c, atol=1e-5):
                blockers.append("HOLDOUT_T_D_C_DOES_NOT_MATCH_STATIC_TF")
            else:
                static_t_d_c_validated = True
        except (CalibrationAnalysisError, ValueError) as exc:
            blockers.append(f"HOLDOUT_T_D_C_STATIC_CHAIN_REQUIRED:{exc}")
    observations = select_independent_board_observations(getattr(analysis, "camera_board_observations", ()))
    color_fit = getattr(analysis, "clock_fits", {}).get("color")
    depth_fit = getattr(analysis, "clock_fits", {}).get("depth")
    depth_records = getattr(analysis, "depth_records", ())
    depth_samples: list[tuple[float, Any]] = []
    for recorder_ns, image in depth_records:
        timestamp_s = (depth_fit.map_header_ns(image.header_timestamp_ns) / 1e9) if depth_fit is not None else recorder_ns / 1e9
        depth_samples.append((timestamp_s, image))
    pose_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    reprojection_values: list[float] = []
    lidar_blockers: list[str] = []
    validated_lidar_poses = 0
    used_depth_times: set[float] = set()
    used_lidar_times: set[float] = set()
    lidar_samples: list[tuple[float, tuple[int, int, str, np.ndarray]]] = []
    lidar_fit = getattr(analysis, "clock_fits", {}).get("lidar")
    for record in getattr(analysis, "lidar_records", ()):
        recorder_ns, header_ns, frame_id, points = record
        timestamp_s = (lidar_fit.map_header_ns(header_ns) / 1e9) if lidar_fit is not None else recorder_ns / 1e9
        lidar_samples.append((timestamp_s, record))
    detector = BoardPlaneDetector(board, distance_threshold_m=maximum_lidar_plane_residual_m)
    for observation in observations:
        camera_time = (color_fit.map_header_ns(observation.header_timestamp_ns) / 1e9) if color_fit is not None else observation.recorder_timestamp_ns / 1e9
        depth_match = nearest_within(depth_samples, camera_time, maximum_delta_s=maximum_depth_time_delta_s)
        if depth_match is None:
            continue
        depth_time, depth_image = depth_match
        if depth_time in used_depth_times or depth_scale_m is None:
            continue
        try:
            observed, predicted, _ = board_depth_correspondences(
                depth_image.image,
                np.asarray(depth_info.k, dtype=np.float64),
                t_d_c @ np.asarray(observation.camera_from_board, dtype=np.float64),
                board,
                depth_encoding=depth_image.encoding,
                depth_scale_m=float(depth_scale_m),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if observed:
            pose_pairs.append((np.asarray(observed), np.asarray(predicted)))
            used_depth_times.add(depth_time)
            if observation.reprojection_residuals_px:
                reprojection_values.extend(observation.reprojection_residuals_px)
            else:
                blockers.append("HOLDOUT_RAW_REPROJECTION_RESIDUALS_REQUIRED")
        lidar_match = nearest_within(
            lidar_samples,
            camera_time + float(camera_lidar_delta_ms) / 1000.0,
            maximum_delta_s=maximum_lidar_time_delta_s,
        )
        if lidar_match is None:
            continue
        lidar_time, lidar_record = lidar_match
        if lidar_time in used_lidar_times:
            continue
        _, _, _, points = lidar_record
        board_center_lidar = np.linalg.inv(t_c_l)[:3, :3] @ np.asarray(observation.center_camera) + np.linalg.inv(t_c_l)[:3, 3]
        target_lidar_time = camera_time + float(camera_lidar_delta_ms) / 1000.0
        result = detector.detect(
            points,
            predicted_center_lidar=board_center_lidar,
            association_error_ms=abs(lidar_time - target_lidar_time) * 1000.0,
        )
        if not result.accepted or result.normal_lidar is None or result.distance_m is None:
            lidar_blockers.append("HOLDOUT_LIDAR_BOARD_NOT_VALIDATED")
            continue
        normal_camera = np.asarray(observation.plane.normal, dtype=np.float64)
        predicted_normal_lidar = t_c_l[:3, :3].T @ normal_camera
        predicted_normal_lidar /= max(float(np.linalg.norm(predicted_normal_lidar)), 1e-12)
        predicted_distance_lidar = float(observation.plane.distance_m - np.dot(normal_camera, t_c_l[:3, 3]))
        if np.dot(predicted_normal_lidar, result.normal_lidar) < 0.0:
            predicted_normal_lidar = -predicted_normal_lidar
            predicted_distance_lidar = -predicted_distance_lidar
        if abs(predicted_distance_lidar - result.distance_m) > maximum_lidar_plane_residual_m:
            lidar_blockers.append("HOLDOUT_LIDAR_PLANE_RESIDUAL_EXCEEDED")
        else:
            validated_lidar_poses += 1
            used_lidar_times.add(lidar_time)
    if not observations:
        blockers.append("HOLDOUT_INDEPENDENT_BOARD_POSES_MISSING")
    if len(observations) < 3:
        blockers.append("HOLDOUT_INDEPENDENT_BOARD_POSES_BELOW_MINIMUM")
    if validated_lidar_poses < 3:
        blockers.append("HOLDOUT_LIDAR_BOARD_POSES_BELOW_MINIMUM")
    if lidar_samples and not lidar_blockers and len(pose_pairs) >= 3:
        pass
    blockers.extend(lidar_blockers)
    t_d_c_validated = serial_validated and static_t_d_c_validated and depth_scale_m is not None
    if not t_d_c_validated:
        blockers.append("HOLDOUT_T_D_C_NOT_VALIDATED")
    return evaluate_holdout_metrics(
        pose_pairs,
        reprojection_residuals_px=np.asarray(reprojection_values, dtype=np.float64),
        t_d_c_validated=t_d_c_validated,
        additional_blockers=tuple(dict.fromkeys(blockers)),
    )


def _rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be a finite homogeneous 4 by 4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{name} rotation must be proper orthonormal")
    return matrix.copy()
