"""Detail-preserving helpers shared by the ROS producer and PC-side tests."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence
import zlib

import numpy as np


IDENTITY_SENSOR_FRAME_V1 = "IDENTITY_SENSOR_FRAME_v1"
PROVISIONAL_GO2_UTLIDAR_PITCH180_V2 = "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2"
_SENSOR_FRAME_TRANSFORM_IDS = {
    IDENTITY_SENSOR_FRAME_V1,
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
}


def normalize_sensor_frame_transform_id(transform_id: str | None) -> str:
    """Keep the robot wire contract limited to transforms understood by the PC."""
    key = IDENTITY_SENSOR_FRAME_V1 if transform_id is None else str(transform_id).strip()
    if key not in _SENSOR_FRAME_TRANSFORM_IDS:
        raise ValueError(f"unknown sensor frame transform: {key}")
    return key


@dataclass(frozen=True)
class DetailCloud:
    points_xyz: np.ndarray
    source_points: int
    occupied_cells: int


@dataclass(frozen=True)
class EncodedMessage:
    payload: dict[str, Any]
    json_text: str
    message_bytes: int


class PayloadLimitExceeded(ValueError):
    def __init__(self, message_bytes: int, payload_limit_bytes: int) -> None:
        super().__init__(
            f"complete WebSocket message is {message_bytes} bytes; limit is {payload_limit_bytes} bytes"
        )
        self.message_bytes = int(message_bytes)
        self.payload_limit_bytes = int(payload_limit_bytes)


def adaptive_detail_hz(
    *,
    target_hz: float,
    minimum_hz: float,
    queued_frames: int,
    queue_capacity: int,
) -> float:
    """Reduce capture cadence only when the delivery queue accumulates work."""
    target = float(target_hz)
    minimum = float(minimum_hz)
    if not math.isfinite(target) or not math.isfinite(minimum) or not 0.0 < minimum <= target:
        raise ValueError("detail cadence must satisfy 0 < minimum_hz <= target_hz")
    if isinstance(queued_frames, bool) or int(queued_frames) != queued_frames or queued_frames < 0:
        raise ValueError("queued_frames must be a non-negative integer")
    if isinstance(queue_capacity, bool) or int(queue_capacity) != queue_capacity or queue_capacity <= 0:
        raise ValueError("queue_capacity must be a positive integer")
    pressure = min(1.0, int(queued_frames) / int(queue_capacity))
    return target - (target - minimum) * pressure


def deduplicate_surface_cells(points_xyz: np.ndarray, cell_m: float = 0.005) -> DetailCloud:
    """Retain one centroid-nearest return per occupied sensor-frame cell."""
    if not math.isfinite(cell_m) or cell_m <= 0.0:
        raise ValueError("cell_m must be finite and positive")
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    if len(points) == 0:
        return DetailCloud(np.empty((0, 3), dtype="<f4"), 0, 0)

    points64 = points.astype(np.float64, copy=False)
    cells = np.floor(points64 / cell_m).astype(np.int64)
    centres = (cells.astype(np.float64) + 0.5) * cell_m
    distance2 = np.einsum("ij,ij->i", points64 - centres, points64 - centres)
    source_order = np.arange(len(points), dtype=np.int64)
    order = np.lexsort((source_order, distance2, cells[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    first_in_cell = np.ones(len(order), dtype=bool)
    first_in_cell[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)
    selected = order[first_in_cell]
    detail_points = np.ascontiguousarray(points[selected], dtype="<f4")
    return DetailCloud(detail_points, int(len(points)), int(len(detail_points)))


def _identifier(value: str, name: str) -> str:
    result = str(value).strip()
    if not result or any(ord(character) < 0x20 for character in result):
        raise ValueError(f"{name} must be a non-empty printable string")
    return result


def _sha256(value: str, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _encoded_message(payload: Mapping[str, Any]) -> EncodedMessage:
    body = json.dumps(dict(payload), separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return EncodedMessage(dict(payload), body, len(body.encode("utf-8")))


def build_live_keyframe_message(
    *,
    building_id: str,
    session_id: str,
    map_id: str,
    producer_instance_id: str,
    keyframe_sequence: int,
    timestamp_ns: int,
    calibration_id: str,
    source_config_sha256: str,
    points_xyz: np.ndarray,
    pose_map_from_lidar: np.ndarray,
    imu_quaternion_wxyz: Sequence[float],
    point_cloud_source: str,
    motion_compensated: bool,
    points_in_map_frame: bool,
    sensor_frame_transform_id: str | None = None,
    payload_limit_bytes: int = 4 * 1024 * 1024,
    camera: Optional[Mapping[str, Any]] = None,
    registration_residual_m: Optional[float] = None,
    calibrated_lio: bool = False,
    previous_ack_rtt_ms: Optional[float] = None,
) -> EncodedMessage:
    if isinstance(keyframe_sequence, bool) or int(keyframe_sequence) < 0:
        raise ValueError("keyframe_sequence must be a non-negative integer")
    if isinstance(timestamp_ns, bool) or int(timestamp_ns) <= 0:
        raise ValueError("timestamp_ns must be a positive integer")
    if isinstance(payload_limit_bytes, bool) or int(payload_limit_bytes) <= 0:
        raise ValueError("payload_limit_bytes must be a positive integer")
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_xyz must be a finite (N, 3) array")
    exact_points = np.ascontiguousarray(points, dtype="<f4")
    point_bytes = exact_points.tobytes(order="C")
    pose = np.asarray(pose_map_from_lidar, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose_map_from_lidar must be a finite 4x4 matrix")
    imu = tuple(float(value) for value in imu_quaternion_wxyz)
    if len(imu) != 4 or not all(math.isfinite(value) for value in imu):
        raise ValueError("imu_quaternion_wxyz must contain four finite values")
    transform_field_supplied = sensor_frame_transform_id is not None
    sensor_frame_transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)

    payload: dict[str, Any] = {
        "message_type": "live_keyframe",
        "protocol_version": 3,
        "building_id": _identifier(building_id, "building_id"),
        "session_id": _identifier(session_id, "session_id"),
        "map_id": _identifier(map_id, "map_id"),
        "producer_instance_id": _identifier(producer_instance_id, "producer_instance_id"),
        "keyframe_sequence": int(keyframe_sequence),
        "timestamp_ns": int(timestamp_ns),
        "calibration_id": _identifier(calibration_id, "calibration_id"),
        "source_config_sha256": _sha256(source_config_sha256, "source_config_sha256"),
        "point_cloud_source": _identifier(point_cloud_source, "point_cloud_source"),
        "point_encoding": "f32-le-xyz-zlib-base64",
        "point_count": int(len(exact_points)),
        "point_bytes_sha256": hashlib.sha256(point_bytes).hexdigest(),
        "points_f32_zlib_b64": base64.b64encode(zlib.compress(point_bytes, level=1)).decode("ascii"),
        "pose_map_from_lidar": pose.ravel().tolist(),
        "imu_quaternion_wxyz": list(imu),
        "motion_compensated": bool(motion_compensated),
        "points_in_map_frame": bool(points_in_map_frame),
        "registration_residual_m": registration_residual_m,
        "calibrated_lio": bool(calibrated_lio),
        "previous_ack_rtt_ms": previous_ack_rtt_ms,
        "source_clipped": False,
    }
    if transform_field_supplied:
        payload["sensor_frame_transform_id"] = sensor_frame_transform_id
    if camera is not None:
        payload["camera"] = dict(camera)
    message = _encoded_message(payload)
    if message.message_bytes > int(payload_limit_bytes):
        raise PayloadLimitExceeded(message.message_bytes, int(payload_limit_bytes))
    return message


def build_surface_source_status_message(
    *,
    building_id: str,
    session_id: str,
    map_id: str,
    producer_instance_id: str,
    status_sequence: int,
    timestamp_ns: int,
    source_state: str,
    source_clipped: bool,
    source_points_before_dedup: int,
    source_points_after_dedup: int,
    compressed_point_bytes: int,
    websocket_message_bytes: int,
    accepted_detail_hz: float,
    source_config_sha256: str,
    blocker: Optional[str],
) -> EncodedMessage:
    allowed_states = {"SOURCE_READY", "DETAIL_SOURCE_CLIPPED", "DETAIL_PAYLOAD_LIMIT_EXCEEDED"}
    if source_state not in allowed_states:
        raise ValueError("source_state is invalid")
    if isinstance(status_sequence, bool) or int(status_sequence) < 0:
        raise ValueError("status_sequence must be a non-negative integer")
    if isinstance(timestamp_ns, bool) or int(timestamp_ns) <= 0:
        raise ValueError("timestamp_ns must be a positive integer")
    counts = (
        source_points_before_dedup,
        source_points_after_dedup,
        compressed_point_bytes,
        websocket_message_bytes,
    )
    if any(isinstance(value, bool) or int(value) < 0 for value in counts):
        raise ValueError("source status counters must be non-negative integers")
    if not math.isfinite(accepted_detail_hz) or accepted_detail_hz < 0.0:
        raise ValueError("accepted_detail_hz must be finite and non-negative")
    payload = {
        "message_type": "surface_source_status",
        "protocol_version": 3,
        "building_id": _identifier(building_id, "building_id"),
        "session_id": _identifier(session_id, "session_id"),
        "map_id": _identifier(map_id, "map_id"),
        "producer_instance_id": _identifier(producer_instance_id, "producer_instance_id"),
        "status_sequence": int(status_sequence),
        "timestamp_ns": int(timestamp_ns),
        "source_state": source_state,
        "source_clipped": bool(source_clipped),
        "source_points_before_dedup": int(source_points_before_dedup),
        "source_points_after_dedup": int(source_points_after_dedup),
        "compressed_point_bytes": int(compressed_point_bytes),
        "websocket_message_bytes": int(websocket_message_bytes),
        "accepted_detail_hz": float(accepted_detail_hz),
        "source_config_sha256": _sha256(source_config_sha256, "source_config_sha256"),
        "blocker": None if blocker is None else str(blocker),
    }
    return _encoded_message(payload)
