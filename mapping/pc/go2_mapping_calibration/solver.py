"""Robust provisional SE(3) and camera-LiDAR time-offset solver."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from .models import CalibrationCandidate, MotionExcitation, PlaneObservation


NORMAL_MATRIX_CONDITION_LIMIT = 1.0e4
NORMAL_MATRIX_RANK_TOLERANCE = 1.0e-6


class CalibrationSolver:
    def __init__(self, solver_version: str = "plane-se3-time-softl1-v2") -> None:
        self.solver_version = solver_version

    def solve(
        self,
        camera_planes: list[PlaneObservation] | tuple[PlaneObservation, ...],
        lidar_planes: list[PlaneObservation] | tuple[PlaneObservation, ...],
        motion: MotionExcitation,
        *,
        initial_camera_from_lidar: np.ndarray | None = None,
        initial_delta_ms: float = 0.0,
        allow_unobservable: bool = False,
        maximum_interpolation_span_s: float = 0.2,
    ) -> CalibrationCandidate:
        if maximum_interpolation_span_s <= 0.0:
            raise ValueError("maximum_interpolation_span_s must be positive")
        blockers = self._observability(camera_planes, lidar_planes, motion)
        if blockers and not allow_unobservable:
            return CalibrationCandidate("UNOBSERVABLE", None, None, None, None, tuple(blockers), self.solver_version)
        camera = tuple(sorted(camera_planes, key=lambda item: item.recorder_time_s))
        lidar = tuple(sorted(lidar_planes, key=lambda item: item.recorder_time_s))
        initial = np.eye(4) if initial_camera_from_lidar is None else np.asarray(initial_camera_from_lidar, dtype=np.float64)
        if initial.shape != (4, 4):
            raise ValueError("initial_camera_from_lidar must be 4 by 4")
        rotation, _ = cv2.Rodrigues(initial[:3, :3])
        parameters = np.concatenate([rotation.reshape(3), initial[:3, 3], [initial_delta_ms / 1000.0]])
        interpolation_pairs = self._fixed_interpolation_pairs(
            camera,
            lidar,
            anchor_delta_s=float(parameters[6]),
            maximum_span_s=maximum_interpolation_span_s,
        )
        result = least_squares(
            lambda value: self._residual(
                value,
                camera,
                lidar,
                maximum_interpolation_span_s=maximum_interpolation_span_s,
                interpolation_pairs=interpolation_pairs,
            ),
            parameters,
            bounds=(np.asarray([-math.pi] * 3 + [-2.0] * 3 + [-0.1]), np.asarray([math.pi] * 3 + [2.0] * 3 + [0.1])),
            x_scale=np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02]),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=3000,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return CalibrationCandidate("SOLVER_FAILED", None, None, None, None, (str(result.message),), self.solver_version)
        parameter_scale = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02])
        singular = np.linalg.svd(result.jac @ np.diag(parameter_scale), compute_uv=False)
        condition = math.inf if singular[-1] <= 1e-12 else float(singular[0] / singular[-1])
        residual = np.abs(
            self._residual(
                result.x,
                camera,
                lidar,
                maximum_interpolation_span_s=maximum_interpolation_span_s,
                interpolation_pairs=interpolation_pairs,
            )
        )
        residual_p95 = float(np.percentile(residual, 95))
        if condition > 1e8 and not allow_unobservable:
            return CalibrationCandidate(
                "UNOBSERVABLE",
                None,
                None,
                condition,
                residual_p95,
                ("NORMALIZED_JACOBIAN_CONDITION_EXCEEDED",),
                self.solver_version,
            )
        rotation_matrix, _ = cv2.Rodrigues(result.x[:3])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = result.x[3:6]
        return CalibrationCandidate(
            "PROVISIONAL_AUTOCAL",
            transform,
            float(result.x[6] * 1000.0),
            condition,
            residual_p95,
            (),
            self.solver_version,
        )

    @staticmethod
    def _observability(
        camera: list[PlaneObservation] | tuple[PlaneObservation, ...],
        lidar: list[PlaneObservation] | tuple[PlaneObservation, ...],
        motion: MotionExcitation,
    ) -> list[str]:
        blockers = []
        if len(camera) < 20 or len(lidar) < 20:
            blockers.append("INSUFFICIENT_PLANE_OBSERVATIONS")
        normals = CalibrationSolver._independent_normals([item.normal for item in camera])
        if normals < 3:
            blockers.append("BOARD_NORMAL_DIVERSITY_BELOW_THREE")
        camera_normal_rank, camera_normal_condition = CalibrationSolver._normal_matrix_observability(
            [item.normal for item in camera]
        )
        lidar_normal_rank, lidar_normal_condition = CalibrationSolver._normal_matrix_observability(
            [item.normal for item in lidar]
        )
        if camera_normal_rank < 3:
            blockers.append("CAMERA_BOARD_NORMAL_MATRIX_RANK_BELOW_THREE")
        elif camera_normal_condition > NORMAL_MATRIX_CONDITION_LIMIT:
            blockers.append("CAMERA_BOARD_NORMAL_MATRIX_CONDITION_EXCEEDED")
        if lidar_normal_rank < 3:
            blockers.append("LIDAR_BOARD_NORMAL_MATRIX_RANK_BELOW_THREE")
        elif lidar_normal_condition > NORMAL_MATRIX_CONDITION_LIMIT:
            blockers.append("LIDAR_BOARD_NORMAL_MATRIX_CONDITION_EXCEEDED")
        if motion.lio_reset_detected:
            blockers.append("LIO_EPOCH_RESET_DETECTED")
        if motion.calibration_rotation_span_deg < 30.0:
            blockers.append("ROTATION_SPAN_BELOW_30_DEG")
        if motion.rotation_axes < 2:
            blockers.append("ROTATION_AXES_BELOW_TWO")
        if motion.calibration_translation_span_m < 0.5:
            blockers.append("TRANSLATION_SPAN_BELOW_0_5_M")
        if motion.angular_speed_duration_s < 1.0:
            blockers.append("ANGULAR_EXCITATION_BELOW_ONE_SECOND")
        if camera and lidar:
            earliest = max(min(item.recorder_time_s for item in camera) - 0.1, min(item.recorder_time_s for item in lidar))
            latest = min(max(item.recorder_time_s for item in camera) + 0.1, max(item.recorder_time_s for item in lidar))
            if latest <= earliest:
                blockers.append("CAMERA_LIDAR_TIME_RANGES_DO_NOT_OVERLAP")
        return blockers

    @staticmethod
    def _independent_normals(normals: list[np.ndarray]) -> int:
        selected: list[np.ndarray] = []
        for normal in normals:
            if all(
                math.degrees(math.acos(float(np.clip(abs(np.dot(normal, previous)), -1.0, 1.0)))) >= 15.0
                for previous in selected
            ):
                selected.append(normal)
        return len(selected)

    @staticmethod
    def _normal_matrix_observability(normals: list[np.ndarray]) -> tuple[int, float]:
        if not normals:
            return 0, math.inf
        matrix = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(matrix) < 3 or not np.all(np.isfinite(matrix)):
            return min(len(matrix), 2), math.inf
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = max(float(singular[0]) * NORMAL_MATRIX_RANK_TOLERANCE, 1.0e-12)
        rank = int(np.count_nonzero(singular > tolerance))
        condition = math.inf if singular[-1] <= tolerance else float(singular[0] / singular[-1])
        return rank, condition

    def _residual(
        self,
        parameters: np.ndarray,
        camera: tuple[PlaneObservation, ...],
        lidar: tuple[PlaneObservation, ...],
        *,
        maximum_interpolation_span_s: float = 0.2,
        interpolation_pairs: tuple[
            tuple[PlaneObservation, PlaneObservation, PlaneObservation], ...
        ] | None = None,
    ) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(parameters[:3])
        translation = parameters[3:6]
        delta_s = float(parameters[6])
        pairs = interpolation_pairs
        if pairs is None:
            pairs = self._fixed_interpolation_pairs(
                camera,
                lidar,
                anchor_delta_s=0.0,
                maximum_span_s=maximum_interpolation_span_s,
            )
        residuals = []
        for observation, left, right in pairs:
            span = right.recorder_time_s - left.recorder_time_s
            alpha = (observation.recorder_time_s + delta_s - left.recorder_time_s) / span
            right_normal = right.normal if np.dot(left.normal, right.normal) >= 0.0 else -right.normal
            normal_lidar = (1.0 - alpha) * left.normal + alpha * right_normal
            normal_lidar /= max(float(np.linalg.norm(normal_lidar)), 1e-12)
            distance_lidar = (1.0 - alpha) * left.distance_m + alpha * right.distance_m
            predicted_normal = rotation @ normal_lidar
            if np.dot(predicted_normal, observation.normal) < 0.0:
                predicted_normal = -predicted_normal
                distance_lidar = -distance_lidar
            predicted_distance = distance_lidar + float(np.dot(predicted_normal, translation))
            weight = math.sqrt(observation.weight)
            residuals.extend(((predicted_normal - observation.normal) * (weight / math.radians(1.0))).tolist())
            residuals.append((predicted_distance - observation.distance_m) * (weight / 0.01))
        return np.asarray(residuals if residuals else [1e6] * 7, dtype=np.float64)

    @staticmethod
    def _fixed_interpolation_pairs(
        camera: tuple[PlaneObservation, ...],
        lidar: tuple[PlaneObservation, ...],
        *,
        anchor_delta_s: float,
        maximum_span_s: float,
    ) -> tuple[tuple[PlaneObservation, PlaneObservation, PlaneObservation], ...]:
        """Select time-neighbor pairs once so least-squares sees a stable shape."""
        times = np.asarray([item.recorder_time_s for item in lidar], dtype=np.float64)
        pairs: list[tuple[PlaneObservation, PlaneObservation, PlaneObservation]] = []
        for observation in camera:
            timestamp_s = observation.recorder_time_s + anchor_delta_s
            index = int(np.searchsorted(times, timestamp_s))
            if index == 0 or index >= len(lidar):
                continue
            left, right = lidar[index - 1], lidar[index]
            span = right.recorder_time_s - left.recorder_time_s
            if span <= 0.0 or span > maximum_span_s:
                continue
            pairs.append((observation, left, right))
        return tuple(pairs)

    @staticmethod
    def _interpolate(
        lidar: tuple[PlaneObservation, ...],
        timestamp_s: float,
        *,
        maximum_span_s: float = 0.2,
    ) -> tuple[np.ndarray, float] | None:
        times = np.asarray([item.recorder_time_s for item in lidar])
        index = int(np.searchsorted(times, timestamp_s))
        if index == 0 or index >= len(lidar):
            return None
        left, right = lidar[index - 1], lidar[index]
        span = right.recorder_time_s - left.recorder_time_s
        if span <= 0.0 or span > maximum_span_s:
            return None
        alpha = (timestamp_s - left.recorder_time_s) / span
        right_normal = right.normal if np.dot(left.normal, right.normal) >= 0.0 else -right.normal
        normal = (1.0 - alpha) * left.normal + alpha * right_normal
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        distance = (1.0 - alpha) * left.distance_m + alpha * right.distance_m
        return normal, float(distance)
