"""Typed protocol-v3 validation for detailed live surface input."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Optional
import zlib

import numpy as np

from .sensor_frame import normalize_sensor_frame_transform_id


PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_POINT_BYTES = 64 * 1024 * 1024
MAX_TEXTURE_COLOR_BYTES = 2 * 1024 * 1024
MAX_TEXTURE_DEPTH_BYTES = 32 * 1024 * 1024
_ACK_DISPOSITIONS = {"QUEUED_NEW", "DUPLICATE_QUEUED", "DUPLICATE_INTEGRATED", "REJECTED"}
_SOURCE_STATES = {"SOURCE_READY", "DETAIL_SOURCE_CLIPPED", "DETAIL_PAYLOAD_LIMIT_EXCEEDED"}


class LiveProtocolError(ValueError):
    def __init__(self, message: str, error_code: str = "INVALID_MESSAGE") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class StreamKey:
    building_id: str
    session_id: str
    map_id: str
    producer_instance_id: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.building_id, self.session_id, self.map_id, self.producer_instance_id)


@dataclass(frozen=True)
class FrameIdentity:
    building_id: str
    session_id: str
    map_id: str
    producer_instance_id: str
    sequence: int

    def stream_key(self) -> StreamKey:
        return StreamKey(self.building_id, self.session_id, self.map_id, self.producer_instance_id)

    def as_tuple(self) -> tuple[str, str, str, str, int]:
        return (*self.stream_key().as_tuple(), self.sequence)


@dataclass(frozen=True)
class LiveKeyframeMessage:
    identity: FrameIdentity
    timestamp_ns: int
    calibration_id: str
    source_config_sha256: str
    point_bytes_sha256: str
    points_xyz: np.ndarray
    pose_map_from_lidar: np.ndarray
    imu_quaternion_wxyz: tuple[float, float, float, float]
    points_in_map_frame: bool
    sensor_frame_transform_id: str
    message_bytes: int
    canonical_json: str
    payload: dict[str, Any]

    def preview_payload(self) -> dict[str, Any]:
        result = dict(self.payload)
        result["sequence"] = self.identity.sequence
        return result


@dataclass(frozen=True)
class SurfaceStatusIdentity:
    building_id: str
    session_id: str
    map_id: str
    producer_instance_id: str

    def stream_key(self) -> StreamKey:
        return StreamKey(self.building_id, self.session_id, self.map_id, self.producer_instance_id)


@dataclass(frozen=True)
class SurfaceSourceStatus:
    identity: SurfaceStatusIdentity
    status_sequence: int
    timestamp_ns: int
    source_state: str
    source_clipped: bool
    source_config_sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CameraPreviewMessage:
    identity: SurfaceStatusIdentity
    calibration_id: str
    camera_sequence: int
    camera: dict[str, Any]
    message_bytes: int


@dataclass(frozen=True)
class TextureFrameIdentity:
    building_id: str
    session_id: str
    map_id: str
    producer_instance_id: str
    producer_epoch: int
    sequence: int

    def stream_key(self) -> tuple[str, str, str, str, int]:
        return (
            self.building_id,
            self.session_id,
            self.map_id,
            self.producer_instance_id,
            self.producer_epoch,
        )

    def as_tuple(self) -> tuple[str, str, str, str, int, int]:
        return (*self.stream_key(), self.sequence)


@dataclass(frozen=True)
class TextureKeyframeMessage:
    identity: TextureFrameIdentity
    timestamp_ns: int
    calibration_id: str
    source_config_sha256: str
    content_sha256: str
    color_jpeg: bytes
    source: dict[str, Any]
    source_dimensions: tuple[int, int]
    intrinsics: dict[str, Any]
    lio_pose_timestamp_ns: int
    pose_skew_ns: int
    pose_map_from_lidar: np.ndarray
    quality: dict[str, Any]
    depth_u16_bytes: Optional[bytes]
    depth_compressed_bytes: Optional[bytes]
    message_bytes: int
    canonical_json: str
    payload: dict[str, Any]


def _canonical(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LiveProtocolError("message is not finite canonical JSON") from exc


def _identifier(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise LiveProtocolError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > 200 or any(ord(character) < 0x20 for character in value):
        raise LiveProtocolError(f"{name} must be a non-empty printable identifier")
    return value


def _integer(payload: Mapping[str, Any], name: str, *, positive: bool = False) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveProtocolError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise LiveProtocolError(f"{name} must be {qualifier}")
    return int(value)


def _signed_integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveProtocolError(f"{name} must be an integer")
    return int(value)


def _finite_number(
    payload: Mapping[str, Any],
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LiveProtocolError(f"{name} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise LiveProtocolError(f"{name} is below its minimum")
    if maximum is not None and number > maximum:
        raise LiveProtocolError(f"{name} exceeds its maximum")
    return number


def _sha256(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LiveProtocolError(f"{name} must be a lowercase SHA-256")
    return value


def _require_header(payload: Mapping[str, Any], message_type: str) -> None:
    if payload.get("message_type") != message_type:
        raise LiveProtocolError(f"message_type must be {message_type}", "UNSUPPORTED_MESSAGE_TYPE")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise LiveProtocolError("protocol_version must be 3", "UNSUPPORTED_PROTOCOL_VERSION")


def _decode_points(payload: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    if payload.get("point_encoding") != "f32-le-xyz-zlib-base64":
        raise LiveProtocolError("point_encoding is unsupported")
    point_count = _integer(payload, "point_count")
    expected_bytes = point_count * 3 * np.dtype("<f4").itemsize
    if expected_bytes > MAX_POINT_BYTES:
        raise LiveProtocolError("decoded point payload exceeds the safety limit", "POINT_PAYLOAD_TOO_LARGE")
    encoded = payload.get("points_f32_zlib_b64")
    if not isinstance(encoded, str):
        raise LiveProtocolError("points_f32_zlib_b64 must be a base64 string")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        point_bytes = decoder.decompress(compressed, expected_bytes + 1)
        if len(point_bytes) > expected_bytes or decoder.unconsumed_tail:
            raise LiveProtocolError("decoded point byte count exceeds point_count")
        point_bytes += decoder.flush()
    except (ValueError, TypeError, zlib.error) as exc:
        raise LiveProtocolError("points_f32_zlib_b64 is not valid lossless point data") from exc
    if not decoder.eof or decoder.unused_data or len(point_bytes) != expected_bytes:
        raise LiveProtocolError("decoded point byte count does not match point_count")
    expected_hash = _sha256(payload, "point_bytes_sha256")
    if hashlib.sha256(point_bytes).hexdigest() != expected_hash:
        raise LiveProtocolError("point checksum does not match exact float32 bytes", "POINT_CHECKSUM_MISMATCH")
    points = np.frombuffer(point_bytes, dtype="<f4").reshape((-1, 3)).copy()
    if not np.isfinite(points).all():
        raise LiveProtocolError("point array contains non-finite values")
    return points, expected_hash


def validate_live_keyframe(
    payload: Mapping[str, Any],
    *,
    message_bytes: Optional[int] = None,
    message_limit_bytes: int = MAX_MESSAGE_BYTES,
) -> LiveKeyframeMessage:
    if not isinstance(payload, Mapping):
        raise LiveProtocolError("live keyframe must be a JSON object")
    _require_header(payload, "live_keyframe")
    canonical = _canonical(payload)
    actual_message_bytes = len(canonical.encode("utf-8")) if message_bytes is None else int(message_bytes)
    if actual_message_bytes <= 0 or actual_message_bytes > int(message_limit_bytes):
        raise LiveProtocolError("complete live keyframe exceeds the message limit", "MESSAGE_LIMIT_EXCEEDED")
    identity = FrameIdentity(
        _identifier(payload, "building_id"),
        _identifier(payload, "session_id"),
        _identifier(payload, "map_id"),
        _identifier(payload, "producer_instance_id"),
        _integer(payload, "keyframe_sequence"),
    )
    timestamp_ns = _integer(payload, "timestamp_ns", positive=True)
    calibration_id = _identifier(payload, "calibration_id")
    source_config_sha256 = _sha256(payload, "source_config_sha256")
    try:
        sensor_frame_transform_id = normalize_sensor_frame_transform_id(payload.get("sensor_frame_transform_id"))
    except (TypeError, ValueError) as exc:
        raise LiveProtocolError(f"sensor frame transform is invalid: {exc}", "UNKNOWN_SENSOR_FRAME_TRANSFORM") from exc
    points_in_map_frame = payload.get("points_in_map_frame", False)
    if not isinstance(points_in_map_frame, bool):
        raise LiveProtocolError("points_in_map_frame must be boolean")
    if payload.get("source_clipped") is not False:
        raise LiveProtocolError("typed detailed keyframes must declare source_clipped=false", "DETAIL_SOURCE_CLIPPED")
    points, point_hash = _decode_points(payload)
    try:
        pose = np.asarray(payload.get("pose_map_from_lidar"), dtype=np.float64).reshape((4, 4))
        imu = tuple(float(value) for value in payload.get("imu_quaternion_wxyz", ()))
    except (TypeError, ValueError) as exc:
        raise LiveProtocolError("pose or IMU orientation has an invalid shape") from exc
    if not np.isfinite(pose).all() or len(imu) != 4 or not all(math.isfinite(value) for value in imu):
        raise LiveProtocolError("pose and IMU orientation must be finite")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise LiveProtocolError("pose_map_from_lidar must be a homogeneous transform")
    return LiveKeyframeMessage(
        identity=identity,
        timestamp_ns=timestamp_ns,
        calibration_id=calibration_id,
        source_config_sha256=source_config_sha256,
        point_bytes_sha256=point_hash,
        points_xyz=points,
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=imu,
        points_in_map_frame=points_in_map_frame,
        sensor_frame_transform_id=sensor_frame_transform_id,
        message_bytes=actual_message_bytes,
        canonical_json=canonical,
        payload=dict(payload),
    )


def validate_surface_source_status(payload: Mapping[str, Any]) -> SurfaceSourceStatus:
    if not isinstance(payload, Mapping):
        raise LiveProtocolError("surface source status must be a JSON object")
    _require_header(payload, "surface_source_status")
    identity = SurfaceStatusIdentity(
        _identifier(payload, "building_id"),
        _identifier(payload, "session_id"),
        _identifier(payload, "map_id"),
        _identifier(payload, "producer_instance_id"),
    )
    status_sequence = _integer(payload, "status_sequence")
    timestamp_ns = _integer(payload, "timestamp_ns", positive=True)
    source_state = payload.get("source_state")
    if source_state not in _SOURCE_STATES:
        raise LiveProtocolError("source_state is invalid")
    source_clipped = payload.get("source_clipped")
    if not isinstance(source_clipped, bool):
        raise LiveProtocolError("source_clipped must be boolean")
    for name in (
        "source_points_before_dedup",
        "source_points_after_dedup",
        "compressed_point_bytes",
        "websocket_message_bytes",
    ):
        _integer(payload, name)
    accepted_hz = payload.get("accepted_detail_hz")
    if isinstance(accepted_hz, bool) or not isinstance(accepted_hz, (int, float)) or not math.isfinite(accepted_hz) or accepted_hz < 0:
        raise LiveProtocolError("accepted_detail_hz must be finite and non-negative")
    return SurfaceSourceStatus(
        identity=identity,
        status_sequence=status_sequence,
        timestamp_ns=timestamp_ns,
        source_state=str(source_state),
        source_clipped=source_clipped,
        source_config_sha256=_sha256(payload, "source_config_sha256"),
        payload=dict(payload),
    )


def validate_camera_preview(
    payload: Mapping[str, Any],
    *,
    message_bytes: Optional[int] = None,
    message_limit_bytes: int = MAX_MESSAGE_BYTES,
) -> CameraPreviewMessage:
    if not isinstance(payload, Mapping):
        raise LiveProtocolError("camera preview must be a JSON object")
    _require_header(payload, "camera_preview")
    canonical = _canonical(payload)
    actual_message_bytes = len(canonical.encode("utf-8")) if message_bytes is None else int(message_bytes)
    if actual_message_bytes <= 0 or actual_message_bytes > int(message_limit_bytes):
        raise LiveProtocolError("camera preview exceeds the message limit", "MESSAGE_LIMIT_EXCEEDED")
    identity = SurfaceStatusIdentity(
        _identifier(payload, "building_id"),
        _identifier(payload, "session_id"),
        _identifier(payload, "map_id"),
        _identifier(payload, "producer_instance_id"),
    )
    camera = payload.get("camera")
    if not isinstance(camera, Mapping):
        raise LiveProtocolError("camera preview payload is required")
    return CameraPreviewMessage(
        identity=identity,
        calibration_id=_identifier(payload, "calibration_id"),
        camera_sequence=_integer(camera, "sequence"),
        camera=dict(camera),
        message_bytes=actual_message_bytes,
    )


def validate_texture_keyframe(
    payload: Mapping[str, Any],
    *,
    message_bytes: Optional[int] = None,
    message_limit_bytes: int = MAX_MESSAGE_BYTES,
) -> TextureKeyframeMessage:
    """Validate one durable RGB texture observation without changing camera ownership."""
    if not isinstance(payload, Mapping):
        raise LiveProtocolError("texture keyframe must be a JSON object")
    _require_header(payload, "texture_keyframe")
    canonical = _canonical(payload)
    actual_message_bytes = len(canonical.encode("utf-8")) if message_bytes is None else int(message_bytes)
    if actual_message_bytes <= 0 or actual_message_bytes > int(message_limit_bytes):
        raise LiveProtocolError("complete texture keyframe exceeds the message limit", "MESSAGE_LIMIT_EXCEEDED")

    identity = TextureFrameIdentity(
        _identifier(payload, "building_id"),
        _identifier(payload, "session_id"),
        _identifier(payload, "map_id"),
        _identifier(payload, "producer_instance_id"),
        _integer(payload, "producer_epoch"),
        _integer(payload, "texture_sequence"),
    )
    timestamp_ns = _integer(payload, "timestamp_ns", positive=True)
    calibration_id = _identifier(payload, "calibration_id")
    source_config_sha256 = _sha256(payload, "source_config_sha256")
    content_sha256 = _sha256(payload, "content_sha256")

    source_value = payload.get("source")
    if not isinstance(source_value, Mapping):
        raise LiveProtocolError("texture source identity is required")
    source = dict(source_value)
    _identifier(source, "topic")
    _identifier(source, "frame_id")
    if source.get("encoding") not in {"bgr8", "rgb8"}:
        raise LiveProtocolError("texture source encoding is unsupported")
    source_width = _integer(source, "width", positive=True)
    source_height = _integer(source, "height", positive=True)
    if source_width > 4096 or source_height > 4096:
        raise LiveProtocolError("texture source dimensions exceed the safety limit")

    color_value = payload.get("color")
    if not isinstance(color_value, Mapping) or color_value.get("encoding") != "jpeg-base64":
        raise LiveProtocolError("texture color encoding must be jpeg-base64")
    color_width = _integer(color_value, "width", positive=True)
    color_height = _integer(color_value, "height", positive=True)
    if (color_width, color_height) != (source_width, source_height):
        raise LiveProtocolError("texture source dimensions must match the transported color frame")
    encoded_color = color_value.get("jpeg_b64")
    if not isinstance(encoded_color, str):
        raise LiveProtocolError("texture jpeg_b64 must be a base64 string")
    try:
        color_jpeg = base64.b64decode(encoded_color, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise LiveProtocolError("texture jpeg_b64 is invalid") from exc
    if not 4 <= len(color_jpeg) <= MAX_TEXTURE_COLOR_BYTES:
        raise LiveProtocolError("texture JPEG exceeds the content safety limit", "TEXTURE_PAYLOAD_TOO_LARGE")
    if not color_jpeg.startswith(b"\xff\xd8") or not color_jpeg.endswith(b"\xff\xd9"):
        raise LiveProtocolError("texture content is not a complete JPEG")
    if hashlib.sha256(color_jpeg).hexdigest() != content_sha256:
        raise LiveProtocolError("texture content checksum does not match JPEG bytes", "TEXTURE_CHECKSUM_MISMATCH")

    intrinsics_value = payload.get("intrinsics")
    if not isinstance(intrinsics_value, Mapping):
        raise LiveProtocolError("texture camera intrinsics are required")
    intrinsics = dict(intrinsics_value)
    intrinsics_width = _integer(intrinsics, "width", positive=True)
    intrinsics_height = _integer(intrinsics, "height", positive=True)
    if (intrinsics_width, intrinsics_height) != (color_width, color_height):
        raise LiveProtocolError("texture intrinsics dimensions must match the color frame")
    for name in ("fx", "fy"):
        _finite_number(intrinsics, name, minimum=1e-6)
    _finite_number(intrinsics, "cx", minimum=0.0, maximum=float(color_width))
    _finite_number(intrinsics, "cy", minimum=0.0, maximum=float(color_height))

    lio_pose_timestamp_ns = _integer(payload, "lio_pose_timestamp_ns", positive=True)
    pose_skew_ns = _signed_integer(payload, "pose_skew_ns")
    if payload.get("pose_association_clock") not in {"ros_header", "subscriber_monotonic"}:
        raise LiveProtocolError("texture pose_association_clock is invalid")
    if abs(pose_skew_ns) > 250_000_000:
        raise LiveProtocolError("texture LIO pose association exceeds 250 ms", "TEXTURE_POSE_SKEW_EXCEEDED")
    try:
        pose = np.asarray(payload.get("pose_map_from_lidar"), dtype=np.float64).reshape((4, 4))
    except (TypeError, ValueError) as exc:
        raise LiveProtocolError("texture LIO pose has an invalid shape") from exc
    if not np.isfinite(pose).all() or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise LiveProtocolError("texture LIO pose must be a finite homogeneous transform")

    quality_value = payload.get("quality")
    if not isinstance(quality_value, Mapping):
        raise LiveProtocolError("texture quality evidence is required")
    quality = dict(quality_value)
    if quality.get("selection_accepted") is not True:
        raise LiveProtocolError("texture quality selection must be accepted")
    _finite_number(quality, "sharpness_laplacian", minimum=0.0)
    _finite_number(quality, "mean_luma", minimum=0.0, maximum=255.0)
    _finite_number(quality, "clipped_fraction", minimum=0.0, maximum=1.0)
    _finite_number(quality, "viewpoint_baseline_m", minimum=0.0)
    _identifier(quality, "independent_view_id")

    depth_u16_bytes: Optional[bytes] = None
    depth_compressed_bytes: Optional[bytes] = None
    depth_value = payload.get("depth")
    if depth_value is not None:
        if not isinstance(depth_value, Mapping) or depth_value.get("encoding") != "u16-le-zlib-base64":
            raise LiveProtocolError("texture depth encoding must be u16-le-zlib-base64")
        depth_width = _integer(depth_value, "width", positive=True)
        depth_height = _integer(depth_value, "height", positive=True)
        if (depth_width, depth_height) != (color_width, color_height):
            raise LiveProtocolError("aligned texture depth dimensions must match the color frame")
        if depth_value.get("aligned_to_color") is not True:
            raise LiveProtocolError("texture depth must be aligned to color")
        _finite_number(depth_value, "scale_m", minimum=1e-9, maximum=0.1)
        expected_depth_bytes = depth_width * depth_height * 2
        if expected_depth_bytes > MAX_TEXTURE_DEPTH_BYTES:
            raise LiveProtocolError("texture depth exceeds the decoded safety limit", "TEXTURE_PAYLOAD_TOO_LARGE")
        encoded_depth = depth_value.get("data_b64")
        if not isinstance(encoded_depth, str):
            raise LiveProtocolError("texture depth data_b64 must be a base64 string")
        try:
            depth_compressed_bytes = base64.b64decode(encoded_depth, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise LiveProtocolError("texture depth data_b64 is invalid") from exc
        if hashlib.sha256(depth_compressed_bytes).hexdigest() != _sha256(
            depth_value, "compressed_bytes_sha256"
        ):
            raise LiveProtocolError("texture depth checksum does not match compressed bytes", "TEXTURE_CHECKSUM_MISMATCH")
        try:
            decoder = zlib.decompressobj()
            depth_u16_bytes = decoder.decompress(depth_compressed_bytes, expected_depth_bytes + 1)
            if len(depth_u16_bytes) > expected_depth_bytes or decoder.unconsumed_tail:
                raise LiveProtocolError("texture depth expands beyond its declared dimensions")
            depth_u16_bytes += decoder.flush()
        except zlib.error as exc:
            raise LiveProtocolError("texture depth is not valid zlib data") from exc
        if not decoder.eof or decoder.unused_data or len(depth_u16_bytes) != expected_depth_bytes:
            raise LiveProtocolError("texture depth byte count does not match its dimensions")

    return TextureKeyframeMessage(
        identity=identity,
        timestamp_ns=timestamp_ns,
        calibration_id=calibration_id,
        source_config_sha256=source_config_sha256,
        content_sha256=content_sha256,
        color_jpeg=color_jpeg,
        source=source,
        source_dimensions=(source_width, source_height),
        intrinsics=intrinsics,
        lio_pose_timestamp_ns=lio_pose_timestamp_ns,
        pose_skew_ns=pose_skew_ns,
        pose_map_from_lidar=pose,
        quality=quality,
        depth_u16_bytes=depth_u16_bytes,
        depth_compressed_bytes=depth_compressed_bytes,
        message_bytes=actual_message_bytes,
        canonical_json=canonical,
        payload=dict(payload),
    )


def build_live_keyframe_ack(
    frame: LiveKeyframeMessage,
    *,
    accepted: bool,
    disposition: str,
    wal_record_sha256: Optional[str],
    next_sequence: int,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    if disposition not in _ACK_DISPOSITIONS:
        raise ValueError("disposition is invalid")
    if accepted and disposition == "REJECTED" or not accepted and disposition != "REJECTED":
        raise ValueError("accepted and disposition disagree")
    if wal_record_sha256 is not None and (
        len(wal_record_sha256) != 64 or any(character not in "0123456789abcdef" for character in wal_record_sha256)
    ):
        raise ValueError("wal_record_sha256 must be a lowercase SHA-256")
    identity = frame.identity
    return {
        "message_type": "live_keyframe_ack",
        "protocol_version": PROTOCOL_VERSION,
        "building_id": identity.building_id,
        "session_id": identity.session_id,
        "map_id": identity.map_id,
        "producer_instance_id": identity.producer_instance_id,
        "keyframe_sequence": identity.sequence,
        "sensor_frame_transform_id": frame.sensor_frame_transform_id,
        "accepted": bool(accepted),
        "ack_stage": "QUEUED" if accepted and disposition != "DUPLICATE_INTEGRATED" else None,
        "disposition": disposition,
        "wal_record_sha256": wal_record_sha256,
        "next_sequence": int(next_sequence),
        "error_code": error_code,
    }


def build_texture_keyframe_ack(
    frame: TextureKeyframeMessage,
    *,
    accepted: bool,
    disposition: str,
    wal_record_sha256: Optional[str],
    next_sequence: int,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    if disposition not in _ACK_DISPOSITIONS:
        raise ValueError("disposition is invalid")
    if accepted and disposition == "REJECTED" or not accepted and disposition != "REJECTED":
        raise ValueError("accepted and disposition disagree")
    if wal_record_sha256 is not None and (
        len(wal_record_sha256) != 64
        or any(character not in "0123456789abcdef" for character in wal_record_sha256)
    ):
        raise ValueError("wal_record_sha256 must be a lowercase SHA-256")
    identity = frame.identity
    return {
        "message_type": "texture_keyframe_ack",
        "protocol_version": PROTOCOL_VERSION,
        "building_id": identity.building_id,
        "session_id": identity.session_id,
        "map_id": identity.map_id,
        "producer_instance_id": identity.producer_instance_id,
        "producer_epoch": identity.producer_epoch,
        "texture_sequence": identity.sequence,
        "content_sha256": frame.content_sha256,
        "accepted": bool(accepted),
        "ack_stage": "QUEUED" if accepted and disposition != "DUPLICATE_INTEGRATED" else None,
        "disposition": disposition,
        "wal_record_sha256": wal_record_sha256,
        "next_sequence": int(next_sequence),
        "error_code": error_code,
    }


def build_surface_source_status_ack(
    status: SurfaceSourceStatus,
    *,
    accepted: bool,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    identity = status.identity
    return {
        "message_type": "surface_source_status_ack",
        "protocol_version": PROTOCOL_VERSION,
        "building_id": identity.building_id,
        "session_id": identity.session_id,
        "map_id": identity.map_id,
        "producer_instance_id": identity.producer_instance_id,
        "status_sequence": status.status_sequence,
        "accepted": bool(accepted),
        "source_config_sha256": status.source_config_sha256,
        "error_code": error_code,
    }


def build_camera_preview_ack(
    preview: CameraPreviewMessage,
    *,
    accepted: bool,
    error_code: Optional[str] = None,
) -> dict[str, Any]:
    identity = preview.identity
    return {
        "message_type": "camera_preview_ack",
        "protocol_version": PROTOCOL_VERSION,
        "building_id": identity.building_id,
        "session_id": identity.session_id,
        "map_id": identity.map_id,
        "producer_instance_id": identity.producer_instance_id,
        "camera_sequence": preview.camera_sequence,
        "accepted": bool(accepted),
        "error_code": error_code,
    }
