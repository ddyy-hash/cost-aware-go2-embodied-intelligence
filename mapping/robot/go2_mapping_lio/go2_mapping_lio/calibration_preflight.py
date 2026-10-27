#!/usr/bin/env python3
"""Read one local RGB frame and raw MD360 header progress for calibration preflight."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from .lio_clock_alignment import HeaderProgressGate
except ImportError:  # Direct execution on the robot keeps the module directory on sys.path.
    from lio_clock_alignment import HeaderProgressGate


class PreflightError(ValueError):
    """Raised when a local preflight input cannot be interpreted safely."""


@dataclass(frozen=True)
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    marker_length_m: float
    square_length_m: float

    @property
    def expected_marker_count(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def expected_marker_ids(self) -> tuple[int, ...]:
        return tuple(range(self.expected_marker_count))

    @property
    def expected_charuco_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def create_board(self):
        aruco = _aruco()
        board = aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            aruco.getPredefinedDictionary(aruco.DICT_4X4_100),
        )
        ids = tuple(int(value) for value in board.getIds().reshape(-1))
        if ids != self.expected_marker_ids:
            raise PreflightError("LOCKED_BOARD_ID_MISMATCH")
        return board


def go2_lab_board_spec() -> BoardSpec:
    """The measured, printed Go2 lab ChArUco target used by the PC solver."""

    return BoardSpec(
        dictionary="DICT_4X4_100",
        squares_x=12,
        squares_y=9,
        marker_length_m=0.0145,
        square_length_m=0.0200,
    )


def _aruco():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise PreflightError("OPENCV_ARUCO_UNAVAILABLE")
    for name in ("ArucoDetector", "CharucoDetector", "CharucoBoard"):
        if not hasattr(aruco, name):
            raise PreflightError("OPENCV_ARUCO_UNAVAILABLE")
    return aruco


def _gray_u8(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    else:
        raise PreflightError("UNSUPPORTED_CAMERA_IMAGE")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if gray.size == 0:
        raise PreflightError("EMPTY_CAMERA_IMAGE")
    return gray


def _camera_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise PreflightError("INVALID_CAMERA_INTRINSICS")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or not math.isclose(matrix[2, 2], 1.0):
        raise PreflightError("INVALID_CAMERA_INTRINSICS")
    return matrix


def _spatial_components(corners: Iterable[np.ndarray]) -> list[list[int]]:
    marker_corners = [np.asarray(item, dtype=np.float64).reshape(4, 2) for item in corners]
    if not marker_corners:
        return []
    centers = [np.mean(item, axis=0) for item in marker_corners]
    edges = [
        max(float(np.linalg.norm(item[index] - item[(index + 1) % 4])) for index in range(4))
        for item in marker_corners
    ]
    remaining = set(range(len(marker_corners)))
    components: list[list[int]] = []
    while remaining:
        root = remaining.pop()
        component = [root]
        frontier = [root]
        while frontier:
            current = frontier.pop()
            linked = [
                candidate
                for candidate in remaining
                if float(np.linalg.norm(centers[current] - centers[candidate]))
                <= 3.25 * max(edges[current], edges[candidate])
            ]
            for candidate in linked:
                remaining.remove(candidate)
                component.append(candidate)
                frontier.append(candidate)
        components.append(component)
    return components


def _empty_report(spec: BoardSpec, reason: str, *, blur_variance: float) -> dict[str, Any]:
    return {
        "accepted": False,
        "dictionary": spec.dictionary,
        "markers": 0,
        "expected_markers": spec.expected_marker_count,
        "charuco_corners": 0,
        "expected_charuco_corners": spec.expected_charuco_corners,
        "decoded_fraction": 0.0,
        "charuco_fraction": 0.0,
        "coverage_fraction": 0.0,
        "blur_variance": blur_variance,
        "reprojection_p95_px": None,
        "pose_camera": None,
        "blocking_reasons": [reason],
    }


def _component_report(
    gray: np.ndarray,
    *,
    spec: BoardSpec,
    board,
    charuco_detector,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    indexes: list[int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    component_corners = [marker_corners[index] for index in indexes]
    component_ids = marker_ids[indexes]
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
        gray,
        markerCorners=component_corners,
        markerIds=component_ids,
    )
    expected_marker_ids = set(spec.expected_marker_ids)
    observed_markers = tuple(
        sorted(set(int(value) for value in component_ids.reshape(-1)) & expected_marker_ids)
    )
    observed_charuco = (
        ()
        if charuco_ids is None
        else tuple(sorted(set(int(value) for value in np.asarray(charuco_ids).reshape(-1))))
    )
    decoded_fraction = len(observed_markers) / spec.expected_marker_count
    charuco_fraction = len(observed_charuco) / spec.expected_charuco_corners
    points = np.concatenate([item.reshape(4, 2) for item in component_corners], axis=0)
    coverage_fraction = float(
        cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / (gray.shape[0] * gray.shape[1])
    )
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    pose_camera = None
    reprojection_p95_px = None
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 6:
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids]
        image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        solved, rotation, translation = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if solved:
            projected, _ = cv2.projectPoints(
                object_points,
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            reprojection_p95_px = float(np.percentile(residuals, 95))
            pose_camera = {
                "rotation_vector": [float(value) for value in rotation.reshape(-1)],
                "translation_m": [float(value) for value in translation.reshape(-1)],
            }

    blockers: list[str] = []
    if decoded_fraction < 0.5:
        blockers.append("INSUFFICIENT_DECODED_MARKER_COVERAGE")
    if charuco_fraction < 0.5:
        blockers.append("INSUFFICIENT_CHARUCO_CORNER_COVERAGE")
    if reprojection_p95_px is None:
        blockers.append("INSUFFICIENT_REPROJECTION_CORRESPONDENCES")
    elif reprojection_p95_px > 1.5:
        blockers.append("REPROJECTION_P95_EXCEEDS_CAPTURE_THRESHOLD")
    return {
        "accepted": not blockers,
        "dictionary": spec.dictionary,
        "markers": len(observed_markers),
        "expected_markers": spec.expected_marker_count,
        "charuco_corners": len(observed_charuco),
        "expected_charuco_corners": spec.expected_charuco_corners,
        "decoded_fraction": decoded_fraction,
        "charuco_fraction": charuco_fraction,
        "coverage_fraction": coverage_fraction,
        "blur_variance": blur_variance,
        "reprojection_p95_px": reprojection_p95_px,
        "pose_camera": pose_camera,
        "blocking_reasons": blockers,
    }


def detect_charuco_frame(
    image: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    """Apply the locked ChArUco camera eligibility gate to one RGB frame."""

    spec = go2_lab_board_spec()
    gray = _gray_u8(image)
    board = spec.create_board()
    aruco = _aruco()
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_100),
        aruco.DetectorParameters(),
    )
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if marker_ids is None or not marker_corners:
        return _empty_report(spec, "NO_TARGET_MARKERS", blur_variance=blur_variance)
    matrix = _camera_matrix(camera_matrix)
    distortion_values = np.asarray(distortion, dtype=np.float64).reshape(-1)
    candidates = [
        _component_report(
            gray,
            spec=spec,
            board=board,
            charuco_detector=aruco.CharucoDetector(board),
            marker_corners=list(marker_corners),
            marker_ids=np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1),
            indexes=indexes,
            camera_matrix=matrix,
            distortion=distortion_values,
        )
        for indexes in _spatial_components(marker_corners)
    ]
    if not candidates:
        return _empty_report(spec, "NO_TARGET_MARKERS", blur_variance=blur_variance)
    return max(
        candidates,
        key=lambda report: (
            bool(report["accepted"]),
            float(report["decoded_fraction"]),
            float(report["charuco_fraction"]),
        ),
    )


def preflight_result(*, camera_report: dict[str, Any], lidar_header_advancing: bool) -> dict[str, Any]:
    """State the narrow local readiness result without starting any capture work."""

    camera_accepted = bool(camera_report.get("accepted"))
    frame_eligible = camera_accepted and bool(lidar_header_advancing)
    if frame_eligible:
        status = "READY_FOR_DIAGNOSTIC_CALIBRATION_CAPTURE"
    elif not camera_accepted:
        status = "BLOCKED_CAMERA_TARGET"
    else:
        status = "BLOCKED_RAW_MD360_HEADER_PROGRESS"
    return {
        "schema_version": 1,
        "status": status,
        "frame_eligible": frame_eligible,
        "camera_target_accepted": camera_accepted,
        "raw_md360_header_advancing": bool(lidar_header_advancing),
        "recording_started": False,
        "upload_started": False,
        "calibration_solved": False,
        "camera": camera_report,
        "limitations": [
            "SINGLE_FRAME_PRECHECK_ONLY",
            "FINITE_LIDAR_BOARD_PLANE_NOT_VERIFIED",
            "CALIBRATION_NOT_SOLVED",
        ],
    }


def _image_to_bgr(message: object) -> np.ndarray:
    encoding = str(getattr(message, "encoding", "")).lower()
    height = int(getattr(message, "height", 0))
    width = int(getattr(message, "width", 0))
    step = int(getattr(message, "step", 0))
    raw = np.frombuffer(getattr(message, "data", b""), dtype=np.uint8)
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
    if height <= 0 or width <= 0 or channels is None or step < width * channels:
        raise PreflightError("UNSUPPORTED_CAMERA_IMAGE")
    if raw.size < height * step:
        raise PreflightError("TRUNCATED_CAMERA_IMAGE")
    rows = raw[: height * step].reshape(height, step)[:, : width * channels]
    if channels == 1:
        return cv2.cvtColor(rows.reshape(height, width), cv2.COLOR_GRAY2BGR)
    image = rows.reshape(height, width, channels)
    return image if encoding == "bgr8" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _header_timestamp_ns(message: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    sec = getattr(stamp, "sec", None) if stamp is not None else None
    nanosec = getattr(stamp, "nanosec", None) if stamp is not None else None
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        return 0
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        return 0
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        return 0
    return sec * 1_000_000_000 + nanosec


def _blocked_camera_report(reason: str) -> dict[str, Any]:
    return _empty_report(go2_lab_board_spec(), reason, blur_variance=0.0)


def _positive_seconds(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return value


def _positive_milliseconds(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("milliseconds must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-color-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--timeout", type=_positive_seconds, default=8.0)
    parser.add_argument("--minimum-header-progress-ms", type=_positive_milliseconds, default=1.0)
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    except ImportError as exc:
        raise SystemExit("ROS_2_PYTHON_UNAVAILABLE: %s" % exc) from exc

    class PreflightNode(Node):
        def __init__(self) -> None:
            super().__init__("go2_mapping_calibration_preflight")
            self.camera_image: object | None = None
            self.camera_info: object | None = None
            self.header_gate = HeaderProgressGate(minimum_progress_ms=args.minimum_header_progress_ms)
            self.lidar_header_advancing = False
            self.create_subscription(Image, args.camera_color_topic, self._camera, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, args.lidar_topic, self._lidar, qos_profile_sensor_data)

        def _camera(self, message: object) -> None:
            self.camera_image = message

        def _camera_info(self, message: object) -> None:
            self.camera_info = message

        def _lidar(self, message: object) -> None:
            if self.header_gate.observe(_header_timestamp_ns(message)):
                self.lidar_header_advancing = True

    rclpy.init(args=None)
    node = PreflightNode()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and (
            node.camera_image is None or node.camera_info is None or not node.lidar_header_advancing
        ):
            rclpy.spin_once(node, timeout_sec=min(0.2, deadline - time.monotonic()))
        if node.camera_image is None:
            camera_report = _blocked_camera_report("CAMERA_FRAME_UNAVAILABLE")
        elif node.camera_info is None:
            camera_report = _blocked_camera_report("CAMERA_INFO_UNAVAILABLE")
        else:
            try:
                camera_report = detect_charuco_frame(
                    _image_to_bgr(node.camera_image),
                    camera_matrix=np.asarray(getattr(node.camera_info, "k", ()), dtype=np.float64).reshape(3, 3),
                    distortion=np.asarray(getattr(node.camera_info, "d", ()), dtype=np.float64),
                )
            except (PreflightError, ValueError) as exc:
                camera_report = _blocked_camera_report(str(exc))
        result = preflight_result(
            camera_report=camera_report,
            lidar_header_advancing=node.lidar_header_advancing,
        )
        result["camera_header_timestamp_ns"] = (
            _header_timestamp_ns(node.camera_image) if node.camera_image is not None else None
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0 if result["frame_eligible"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
