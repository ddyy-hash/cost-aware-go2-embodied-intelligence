"""Fail-closed ChArUco target specification and image detection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}


class TargetSpecError(ValueError):
    """Raised when target geometry or provenance cannot be trusted."""


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetSpecError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TargetSpecError(f"{name} must be a finite positive number")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TargetSpecError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    marker_length_m: float
    square_length_m: float
    template_marker_length_m: float
    template_square_length_m: float
    source_template_sha256: str
    scale_authority: str

    def __post_init__(self) -> None:
        if self.dictionary not in _DICTIONARIES:
            raise TargetSpecError(f"unsupported dictionary: {self.dictionary}")
        squares_x = _positive_int(self.squares_x, "squares_x")
        squares_y = _positive_int(self.squares_y, "squares_y")
        if squares_x < 3 or squares_y < 3:
            raise TargetSpecError("a ChArUco board requires at least 3 by 3 squares")

        marker = _positive_number(self.marker_length_m, "marker_length_m")
        square = _positive_number(self.square_length_m, "square_length_m")
        template_marker = _positive_number(self.template_marker_length_m, "template_marker_length_m")
        template_square = _positive_number(self.template_square_length_m, "template_square_length_m")
        if marker >= square:
            raise TargetSpecError("marker_length_m must be smaller than square_length_m")
        if template_marker >= template_square:
            raise TargetSpecError("template marker must be smaller than template square")
        if not isinstance(self.source_template_sha256, str) or not _SHA256_RE.fullmatch(
            self.source_template_sha256
        ):
            raise TargetSpecError("source_template_sha256 must be exactly 64 hexadecimal characters")
        if self.scale_authority != "measured_printed_marker_and_square":
            raise TargetSpecError("scale_authority must identify both measured printed dimensions")

    @property
    def pattern_width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def pattern_height_m(self) -> float:
        return self.squares_y * self.square_length_m

    @property
    def expected_marker_ids(self) -> tuple[int, ...]:
        return tuple(range((self.squares_x * self.squares_y) // 2))

    @property
    def expected_charuco_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def create_board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[self.dictionary])
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )
        ids = tuple(int(value) for value in board.getIds().reshape(-1))
        if ids != self.expected_marker_ids:
            raise TargetSpecError("OpenCV board IDs do not match the locked printed target")
        return board

    def render(self, pixels_per_square: int = 100, margin_px: int = 24) -> np.ndarray:
        pixels_per_square = _positive_int(pixels_per_square, "pixels_per_square")
        if isinstance(margin_px, bool) or not isinstance(margin_px, int) or margin_px < 0:
            raise TargetSpecError("margin_px must be a non-negative integer")
        size = (self.squares_x * pixels_per_square, self.squares_y * pixels_per_square)
        image = self.create_board().generateImage(size, marginSize=0, borderBits=1)
        if margin_px:
            image = cv2.copyMakeBorder(
                image,
                margin_px,
                margin_px,
                margin_px,
                margin_px,
                cv2.BORDER_CONSTANT,
                value=255,
            )
        return image


def go2_lab_board_spec() -> BoardSpec:
    """Return the measured 2026-07 Go2 lab board geometry."""

    return BoardSpec(
        dictionary="DICT_4X4_100",
        squares_x=12,
        squares_y=9,
        marker_length_m=0.0145,
        square_length_m=0.020,
        template_marker_length_m=0.005,
        template_square_length_m=0.007,
        source_template_sha256="fe05d01ca5fdde303f712dad177f53c3ae75d3493ff7d9a783b010f892a2d743",
        scale_authority="measured_printed_marker_and_square",
    )


def verify_source_template(path: str | Path, expected_sha256: str) -> str:
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise TargetSpecError("expected source-template SHA-256 must be exactly 64 hexadecimal characters")
    source = Path(path)
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise TargetSpecError(f"cannot read source template: {source}") from exc
    if digest.lower() != expected_sha256.lower():
        raise TargetSpecError(f"source-template SHA-256 mismatch: expected {expected_sha256}, observed {digest}")
    return digest


@dataclass(frozen=True)
class BoardPose:
    rotation_vector: tuple[float, float, float]
    translation_m: tuple[float, float, float]


@dataclass(frozen=True)
class BoardDetection:
    accepted: bool
    marker_ids: tuple[int, ...]
    charuco_ids: tuple[int, ...]
    decoded_fraction: float
    charuco_fraction: float
    coverage_fraction: float
    blur_variance: float
    reprojection_p95_px: float | None
    reprojection_residuals_px: tuple[float, ...]
    pose: BoardPose | None
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BoardPlaneDetection:
    accepted: bool
    normal_lidar: tuple[float, float, float] | None
    distance_m: float | None
    center_lidar: tuple[float, float, float] | None
    extent_m: tuple[float, float] | None
    support_count: int
    residual_p95_m: float | None
    blocking_reasons: tuple[str, ...]
    finite_boundary_verified: bool = False


class BoardPlaneDetector:
    def __init__(
        self,
        spec: BoardSpec,
        *,
        distance_threshold_m: float = 0.012,
        minimum_support: int = 100,
        maximum_association_error_ms: float = 20.0,
        maximum_normal_error_deg: float = 25.0,
        maximum_plane_candidates: int = 8,
        maximum_center_error_m: float = 0.12,
        allow_backing_plane: bool = False,
    ) -> None:
        self.spec = spec
        self.distance_threshold_m = distance_threshold_m
        self.minimum_support = minimum_support
        self.maximum_association_error_ms = maximum_association_error_ms
        self.maximum_normal_error_deg = maximum_normal_error_deg
        self.maximum_plane_candidates = maximum_plane_candidates
        self.maximum_center_error_m = maximum_center_error_m
        self.allow_backing_plane = allow_backing_plane

    def detect(
        self,
        points_lidar: np.ndarray,
        *,
        predicted_center_lidar: np.ndarray | None,
        predicted_normal_lidar: np.ndarray | None = None,
        association_error_ms: float | None,
    ) -> BoardPlaneDetection:
        points = np.asarray(points_lidar, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise TargetSpecError("LiDAR target candidates must be an N by 3 array")
        points = points[np.all(np.isfinite(points), axis=1)]
        blockers = []
        if predicted_center_lidar is None:
            blockers.append("LIO_PREDICTED_BOARD_CENTER_REQUIRED")
            candidate = points
        else:
            predicted = np.asarray(predicted_center_lidar, dtype=np.float64).reshape(3)
            half_diagonal = 0.5 * math.hypot(self.spec.pattern_width_m, self.spec.pattern_height_m)
            radius = max(
                max(self.spec.pattern_width_m, self.spec.pattern_height_m) * 1.25,
                self.maximum_center_error_m + half_diagonal,
            )
            candidate = points[np.linalg.norm(points - predicted, axis=1) <= radius]
        if association_error_ms is None or association_error_ms > self.maximum_association_error_ms:
            blockers.append("TEMPORAL_ASSOCIATION_EXCEEDED")
        if len(candidate) < self.minimum_support:
            blockers.append("INSUFFICIENT_LIDAR_BOARD_SUPPORT")
            return BoardPlaneDetection(False, None, None, None, None, len(candidate), None, tuple(blockers))

        predicted_normal = None
        if predicted_normal_lidar is not None:
            predicted_normal = np.asarray(predicted_normal_lidar, dtype=np.float64).reshape(3)
            normal_length = float(np.linalg.norm(predicted_normal))
            if not np.all(np.isfinite(predicted_normal)) or normal_length <= 1e-12:
                raise TargetSpecError("predicted LiDAR board normal must be finite and nonzero")
            predicted_normal = predicted_normal / normal_length

        import open3d as o3d

        o3d.utility.random.seed(20260731)
        remaining = candidate
        detections: list[BoardPlaneDetection] = []
        for _ in range(self.maximum_plane_candidates):
            if len(remaining) < self.minimum_support:
                break
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
            model, inlier_indices = cloud.segment_plane(
                distance_threshold=self.distance_threshold_m,
                ransac_n=3,
                num_iterations=1000,
            )
            indexes = np.asarray(inlier_indices, dtype=np.int64)
            inliers = remaining[indexes]
            if len(inliers) < self.minimum_support:
                break
            detection = self._evaluate_plane(
                model,
                inliers,
                predicted_center_lidar=predicted_center_lidar,
                predicted_normal_lidar=predicted_normal,
                initial_blockers=blockers,
            )
            detections.append(detection)
            keep = np.ones(len(remaining), dtype=bool)
            keep[indexes] = False
            remaining = remaining[keep]
        if not detections:
            blockers.append("INSUFFICIENT_LIDAR_BOARD_SUPPORT")
            return BoardPlaneDetection(False, None, None, None, None, len(remaining), None, tuple(blockers))
        accepted = [item for item in detections if item.accepted]
        if accepted:
            return min(
                accepted,
                key=lambda item: self._candidate_score(
                    item,
                    predicted_center_lidar=predicted_center_lidar,
                    predicted_normal_lidar=predicted_normal,
                ),
            )
        if predicted_normal is None:
            return detections[0]
        return min(
            detections,
            key=lambda item: (
                "PLANE_NORMAL_MISMATCH" in item.blocking_reasons,
                len(item.blocking_reasons),
                -item.support_count,
            ),
        )

    def _candidate_score(
        self,
        detection: BoardPlaneDetection,
        *,
        predicted_center_lidar: np.ndarray | None,
        predicted_normal_lidar: np.ndarray | None,
    ) -> tuple[float, float, float, float, int]:
        normal_error = 0.0
        if predicted_normal_lidar is not None and detection.normal_lidar is not None:
            cosine = float(
                np.clip(
                    abs(np.dot(detection.normal_lidar, predicted_normal_lidar)),
                    -1.0,
                    1.0,
                )
            )
            normal_error = math.degrees(math.acos(cosine))
        expected_extent = np.sort((self.spec.pattern_width_m, self.spec.pattern_height_m))
        observed_extent = np.asarray(detection.extent_m, dtype=np.float64)
        extent_error = float(np.linalg.norm((observed_extent - expected_extent) / expected_extent))
        center_error = 0.0
        if predicted_center_lidar is not None and detection.center_lidar is not None:
            center_error = float(
                np.linalg.norm(np.asarray(detection.center_lidar) - np.asarray(predicted_center_lidar))
            )
        return (
            normal_error,
            extent_error,
            center_error,
            float(detection.residual_p95_m or 0.0),
            -detection.support_count,
        )

    def _evaluate_plane(
        self,
        model: np.ndarray,
        inliers: np.ndarray,
        *,
        predicted_center_lidar: np.ndarray | None,
        predicted_normal_lidar: np.ndarray | None,
        initial_blockers: Iterable[str],
    ) -> BoardPlaneDetection:
        blockers = list(initial_blockers)
        normal = np.asarray(model[:3], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        distance = -float(model[3]) / max(float(np.linalg.norm(model[:3])), 1e-12)
        if distance < 0.0:
            normal = -normal
            distance = -distance
        center = np.median(inliers, axis=0)
        residual = np.abs(inliers @ normal - distance)
        residual_p95 = float(np.percentile(residual, 95))
        centered = inliers - center
        basis = np.linalg.svd(centered, full_matrices=False)[2][:2]
        planar = centered @ basis.T
        extent = np.percentile(planar, 99, axis=0) - np.percentile(planar, 1, axis=0)
        observed_extent = tuple(sorted(float(value) for value in extent))
        expected_extent = tuple(sorted((self.spec.pattern_width_m, self.spec.pattern_height_m)))
        extent_too_small = any(
            observed < expected * 0.75
            for observed, expected in zip(observed_extent, expected_extent)
        )
        extent_too_large = any(
            observed > expected * 1.25
            for observed, expected in zip(observed_extent, expected_extent)
        )
        finite_boundary_verified = not extent_too_small and not extent_too_large
        if extent_too_small or (extent_too_large and not self.allow_backing_plane):
            blockers.append("PLANE_EXTENT_DOES_NOT_MATCH_BOARD")
        elif extent_too_large:
            # A backing wall may provide a useful normal and range diagnostic,
            # but cropping that wall around the predicted center does not prove
            # the calibration board's four finite edges.
            blockers.append("FINITE_BOARD_BOUNDARY_NOT_VERIFIED")
        if residual_p95 > self.distance_threshold_m:
            blockers.append("PLANE_RESIDUAL_EXCEEDED")
        if predicted_normal_lidar is not None:
            cosine = float(np.clip(abs(np.dot(normal, predicted_normal_lidar)), -1.0, 1.0))
            if math.degrees(math.acos(cosine)) > self.maximum_normal_error_deg:
                blockers.append("PLANE_NORMAL_MISMATCH")
        if (
            predicted_center_lidar is not None
            and np.linalg.norm(center - np.asarray(predicted_center_lidar)) > self.maximum_center_error_m
        ):
            blockers.append("PLANE_CENTER_MISMATCH")
        return BoardPlaneDetection(
            accepted=not blockers,
            normal_lidar=tuple(float(value) for value in normal),
            distance_m=distance,
            center_lidar=tuple(float(value) for value in center),
            extent_m=observed_extent,
            support_count=len(inliers),
            residual_p95_m=residual_p95,
            blocking_reasons=tuple(blockers),
            finite_boundary_verified=finite_boundary_verified,
        )


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    elif array.ndim != 2:
        raise TargetSpecError("target image must be grayscale, BGR, or BGRA")
    if array.size == 0:
        raise TargetSpecError("target image cannot be empty")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise TargetSpecError("target image must contain finite numeric pixels")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _marker_edge_px(corners: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return float(np.median(edges))


def _spatial_components(corners: list[np.ndarray]) -> list[list[int]]:
    if not corners:
        return []
    centers = np.asarray([np.asarray(item).reshape(4, 2).mean(axis=0) for item in corners])
    edges = np.asarray([max(_marker_edge_px(item), 1.0) for item in corners])
    remaining = set(range(len(corners)))
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            linked = [
                candidate
                for candidate in remaining
                if np.linalg.norm(centers[current] - centers[candidate])
                <= 3.25 * max(edges[current], edges[candidate])
            ]
            for candidate in linked:
                remaining.remove(candidate)
                component.append(candidate)
                frontier.append(candidate)
        components.append(component)
    return components


class TargetDetector:
    def __init__(
        self,
        spec: BoardSpec,
        camera_matrix: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        *,
        minimum_decoded_fraction: float = 0.5,
        minimum_charuco_fraction: float = 0.5,
        maximum_reprojection_p95_px: float = 1.5,
    ) -> None:
        self.spec = spec
        self.board = spec.create_board()
        self.dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[spec.dictionary])
        self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.camera_matrix = self._camera_matrix(camera_matrix)
        self.distortion = None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1)
        self.minimum_decoded_fraction = float(minimum_decoded_fraction)
        self.minimum_charuco_fraction = float(minimum_charuco_fraction)
        self.maximum_reprojection_p95_px = float(maximum_reprojection_p95_px)

    @staticmethod
    def _camera_matrix(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise TargetSpecError("camera_matrix must be a finite 3 by 3 matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or not math.isclose(matrix[2, 2], 1.0):
            raise TargetSpecError("camera_matrix must contain usable pinhole intrinsics")
        return matrix

    def detect(self, image: np.ndarray) -> BoardDetection:
        detections = self.detect_all(image)
        if detections:
            return detections[0]
        return BoardDetection(
            accepted=False,
            marker_ids=(),
            charuco_ids=(),
            decoded_fraction=0.0,
            charuco_fraction=0.0,
            coverage_fraction=0.0,
            blur_variance=float(cv2.Laplacian(_as_gray_u8(image), cv2.CV_64F).var()),
            reprojection_p95_px=None,
            reprojection_residuals_px=(),
            pose=None,
            blocking_reasons=("no target markers detected",),
        )

    def detect_all(self, image: np.ndarray) -> tuple[BoardDetection, ...]:
        gray = _as_gray_u8(image)
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        if marker_ids is None or not marker_corners:
            return ()
        ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
        components = _spatial_components(list(marker_corners))
        detections = [self._detect_component(gray, marker_corners, ids, component) for component in components]
        detections = [item for item in detections if item.marker_ids]
        return tuple(
            sorted(
                detections,
                key=lambda item: (item.accepted, item.decoded_fraction, item.charuco_fraction),
                reverse=True,
            )
        )

    def _detect_component(
        self,
        gray: np.ndarray,
        marker_corners: Iterable[np.ndarray],
        marker_ids: np.ndarray,
        indexes: list[int],
    ) -> BoardDetection:
        all_corners = list(marker_corners)
        component_corners = [all_corners[index] for index in indexes]
        component_ids = marker_ids[indexes]
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
            gray,
            markerCorners=component_corners,
            markerIds=component_ids,
        )

        observed_markers = tuple(
            sorted(set(int(value) for value in component_ids.reshape(-1)) & set(self.spec.expected_marker_ids))
        )
        observed_charuco = (
            ()
            if charuco_ids is None
            else tuple(sorted(set(int(value) for value in np.asarray(charuco_ids).reshape(-1))))
        )
        decoded_fraction = len(observed_markers) / len(self.spec.expected_marker_ids)
        charuco_fraction = len(observed_charuco) / self.spec.expected_charuco_corners
        points = np.concatenate([np.asarray(item).reshape(4, 2) for item in component_corners], axis=0)
        hull = cv2.convexHull(points.astype(np.float32))
        coverage = float(cv2.contourArea(hull) / (gray.shape[0] * gray.shape[1]))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        reprojection_residuals, pose = self._fit_geometry(charuco_corners, charuco_ids)
        reprojection = None if not reprojection_residuals else float(np.percentile(reprojection_residuals, 95))

        blockers: list[str] = []
        if decoded_fraction < self.minimum_decoded_fraction:
            blockers.append("insufficient decoded marker coverage")
        if charuco_fraction < self.minimum_charuco_fraction:
            blockers.append("insufficient ChArUco corner coverage")
        if reprojection is None:
            blockers.append("insufficient correspondences for reprojection validation")
        elif reprojection > self.maximum_reprojection_p95_px:
            blockers.append("reprojection P95 exceeds the capture threshold")
        return BoardDetection(
            accepted=not blockers,
            marker_ids=observed_markers,
            charuco_ids=observed_charuco,
            decoded_fraction=decoded_fraction,
            charuco_fraction=charuco_fraction,
            coverage_fraction=coverage,
            blur_variance=blur,
            reprojection_p95_px=reprojection,
            reprojection_residuals_px=reprojection_residuals,
            pose=pose,
            blocking_reasons=tuple(blockers),
        )

    def _fit_geometry(
        self,
        charuco_corners: np.ndarray | None,
        charuco_ids: np.ndarray | None,
    ) -> tuple[tuple[float, ...], BoardPose | None]:
        minimum_correspondences = 6 if self.camera_matrix is not None else 4
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < minimum_correspondences
        ):
            return (), None
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        object_points = np.asarray(self.board.getChessboardCorners(), dtype=np.float64)[ids]

        pose = None
        if self.camera_matrix is not None:
            success, rotation, translation = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return (), None
            projected, _ = cv2.projectPoints(
                object_points,
                rotation,
                translation,
                self.camera_matrix,
                self.distortion,
            )
            residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            pose = BoardPose(
                rotation_vector=tuple(float(value) for value in rotation.reshape(3)),
                translation_m=tuple(float(value) for value in translation.reshape(3)),
            )
        else:
            homography, _ = cv2.findHomography(object_points[:, :2], image_points, method=0)
            if homography is None:
                return (), None
            projected = cv2.perspectiveTransform(
                object_points[:, :2].reshape(-1, 1, 2).astype(np.float64), homography
            ).reshape(-1, 2)
            residuals = np.linalg.norm(projected - image_points, axis=1)
        return tuple(float(value) for value in residuals), pose
