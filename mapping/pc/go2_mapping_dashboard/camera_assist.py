"""Keep the latest validated RGB-D frame for Dashboard measurement previews.

Full-resolution data remains in the durable ROS bag queue.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping
import zlib


class CameraAssistError(ValueError):
    pass


_MAX_JPEG_BYTES = 320 * 1024
_MAX_DEPTH_BYTES = 320 * 240 * 2
_MAX_DIMENSION = 640
_MAX_FRAME_AGE_S = 5.0


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise CameraAssistError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraAssistError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise CameraAssistError(f"{name} is outside supported bounds")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CameraAssistError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraAssistError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise CameraAssistError(f"{name} must be a finite number")
    return result


def _decoded(value: Any, name: str, maximum: int) -> bytes:
    if not isinstance(value, str):
        raise CameraAssistError(f"{name} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise CameraAssistError(f"{name} is not valid base64") from exc
    if not decoded or len(decoded) > maximum:
        raise CameraAssistError(f"{name} exceeds the preview budget")
    return decoded


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    timestamp_ns: int
    producer_instance_id: str
    calibration_id: str
    color_jpeg_b64: str
    color_width: int
    color_height: int
    depth_u16_le_b64: str | None
    depth_width: int | None
    depth_height: int | None
    depth_scale_m: float | None
    depth_aligned_to_color: bool
    intrinsics: dict[str, float]
    depth_intrinsics: dict[str, float] | None
    received_at: float


class CameraAssistState:
    """Stores one validated, low-rate frame; the latest frame replaces stale data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: CameraFrame | None = None
        self._arrivals: deque[float] = deque(maxlen=120)

    @staticmethod
    def _is_fresh(frame: CameraFrame, now: float) -> bool:
        return now - frame.received_at <= _MAX_FRAME_AGE_S

    def ingest(self, payload: Mapping[str, Any], calibration_id: str, *, now: float | None = None) -> None:
        if not isinstance(payload, Mapping):
            raise CameraAssistError("camera payload must be an object")
        sequence = _integer(payload.get("sequence"), "camera.sequence", 0, 2_147_483_647)
        timestamp_ns = _integer(payload.get("timestamp_ns"), "camera.timestamp_ns", 1, 9_223_372_036_854_775_807)
        producer_instance_id = payload.get("producer_instance_id", "legacy")
        if not isinstance(producer_instance_id, str) or not producer_instance_id or len(producer_instance_id) > 128:
            raise CameraAssistError("camera.producer_instance_id is invalid")
        color = payload.get("color")
        if not isinstance(color, Mapping):
            raise CameraAssistError("camera.color is required")
        color_width = _integer(color.get("width"), "camera.color.width", 1, _MAX_DIMENSION)
        color_height = _integer(color.get("height"), "camera.color.height", 1, _MAX_DIMENSION)
        jpeg = _decoded(color.get("jpeg_b64"), "camera.color.jpeg_b64", _MAX_JPEG_BYTES)
        if jpeg[:2] != b"\xff\xd8" or jpeg[-2:] != b"\xff\xd9":
            raise CameraAssistError("camera.color.jpeg_b64 is not a JPEG")
        intrinsics_raw = payload.get("intrinsics")
        if not isinstance(intrinsics_raw, Mapping):
            raise CameraAssistError("camera intrinsics are required")
        intrinsics = {name: _number(intrinsics_raw.get(name), f"camera.intrinsics.{name}") for name in ("fx", "fy", "cx", "cy")}
        if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
            raise CameraAssistError("camera focal lengths must be positive")

        depth = payload.get("depth")
        depth_b64 = None
        depth_width = None
        depth_height = None
        depth_scale_m = None
        aligned = False
        depth_intrinsics = None
        if depth is not None:
            if not isinstance(depth, Mapping):
                raise CameraAssistError("camera.depth must be an object")
            depth_width = _integer(depth.get("width"), "camera.depth.width", 1, _MAX_DIMENSION)
            depth_height = _integer(depth.get("height"), "camera.depth.height", 1, _MAX_DIMENSION)
            compressed_depth = depth.get("u16_le_zlib_b64")
            if compressed_depth is None:
                decoded_depth = _decoded(depth.get("u16_le_b64"), "camera.depth.u16_le_b64", _MAX_DEPTH_BYTES)
            else:
                encoded_depth = _decoded(compressed_depth, "camera.depth.u16_le_zlib_b64", _MAX_DEPTH_BYTES)
                try:
                    decoded_depth = zlib.decompress(encoded_depth)
                except zlib.error as exc:
                    raise CameraAssistError("camera.depth.u16_le_zlib_b64 is not valid zlib data") from exc
                if not decoded_depth or len(decoded_depth) > _MAX_DEPTH_BYTES:
                    raise CameraAssistError("camera.depth.u16_le_zlib_b64 exceeds the preview budget")
            if len(decoded_depth) != depth_width * depth_height * 2:
                raise CameraAssistError("camera.depth byte count does not match its dimensions")
            depth_scale_m = _number(depth.get("scale_m"), "camera.depth.scale_m")
            if not 0 < depth_scale_m <= 0.1:
                raise CameraAssistError("camera.depth.scale_m is outside supported bounds")
            aligned = depth.get("aligned_to_color") is True
            if aligned and (depth_width != color_width or depth_height != color_height):
                raise CameraAssistError("aligned depth must share the preview color dimensions")
            depth_intrinsics_raw = depth.get("intrinsics", intrinsics_raw if aligned else None)
            if not isinstance(depth_intrinsics_raw, Mapping):
                raise CameraAssistError("camera.depth.intrinsics are required for unaligned depth")
            depth_intrinsics = {
                name: _number(depth_intrinsics_raw.get(name), f"camera.depth.intrinsics.{name}")
                for name in ("fx", "fy", "cx", "cy")
            }
            if depth_intrinsics["fx"] <= 0 or depth_intrinsics["fy"] <= 0:
                raise CameraAssistError("camera depth focal lengths must be positive")
            depth_b64 = base64.b64encode(decoded_depth).decode("ascii")

        received_at = time.monotonic() if now is None else float(now)
        frame = CameraFrame(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            producer_instance_id=producer_instance_id,
            calibration_id=calibration_id,
            color_jpeg_b64=base64.b64encode(jpeg).decode("ascii"),
            color_width=color_width,
            color_height=color_height,
            depth_u16_le_b64=depth_b64,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_scale_m=depth_scale_m,
            depth_aligned_to_color=aligned,
            intrinsics=intrinsics,
            depth_intrinsics=depth_intrinsics,
            received_at=received_at,
        )
        with self._condition:
            # A new producer epoch may begin with a reset RealSense clock.
            # Only reject a timestamp rollback from the same producer process.
            if (
                self._frame is not None
                and frame.producer_instance_id == self._frame.producer_instance_id
                and frame.timestamp_ns < self._frame.timestamp_ns
            ):
                return
            self._frame = frame
            self._arrivals.append(received_at)
            self._condition.notify_all()

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._frame
            while self._arrivals and current - self._arrivals[0] > 5.0:
                self._arrivals.popleft()
            arrivals = tuple(self._arrivals)
        if frame is None:
            return {
                "state": "AWAITING_CAMERA_KEYFRAME",
                "sequence": -1,
                "distance_measurement_ready": False,
                "preview_fps": 0.0,
            }
        age_ms = int(max(0.0, current - frame.received_at) * 1000.0)
        if not self._is_fresh(frame, current):
            return {
                "state": "CAMERA_PREVIEW_STALE",
                "sequence": -1,
                "age_ms": age_ms,
                "distance_measurement_ready": False,
                "preview_fps": 0.0,
                "depth_aligned_to_color": False,
            }
        preview_fps = 0.0
        if len(arrivals) >= 2 and arrivals[-1] > arrivals[0]:
            preview_fps = round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2)
        distance_ready = frame.depth_u16_le_b64 is not None and frame.depth_intrinsics is not None
        return {
            "state": "LIVE_ALIGNED_RGBD" if distance_ready and frame.depth_aligned_to_color else "LIVE_RGB_DEPTH_UNALIGNED" if distance_ready else "LIVE_COLOR_ONLY",
            "sequence": frame.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "producer_instance_id": frame.producer_instance_id,
            "calibration_id": frame.calibration_id,
            "age_ms": age_ms,
            "preview_fps": preview_fps,
            "color_width": frame.color_width,
            "color_height": frame.color_height,
            "depth_aligned_to_color": frame.depth_aligned_to_color,
            "distance_measurement_ready": distance_ready,
            "measurement_view": "color" if frame.depth_aligned_to_color else "depth" if distance_ready else None,
        }

    def frame(self, *, now: float | None = None) -> dict[str, Any] | None:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._frame
        if frame is None or not self._is_fresh(frame, current):
            return None
        return {
            "sequence": frame.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "producer_instance_id": frame.producer_instance_id,
            "calibration_id": frame.calibration_id,
            "color": {"jpeg_b64": frame.color_jpeg_b64, "width": frame.color_width, "height": frame.color_height},
            "depth": None if frame.depth_u16_le_b64 is None else {
                "u16_le_b64": frame.depth_u16_le_b64,
                "width": frame.depth_width,
                "height": frame.depth_height,
                "scale_m": frame.depth_scale_m,
                "aligned_to_color": frame.depth_aligned_to_color,
                "intrinsics": dict(frame.depth_intrinsics or {}),
            },
            "intrinsics": dict(frame.intrinsics),
        }

    def wait_for_color_frame(
        self, after_identity: str | None, *, timeout: float = 1.0
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                frame = self._frame
                current = time.monotonic()
                identity = None if frame is None else f"{frame.producer_instance_id}:{frame.sequence}"
                if frame is not None and self._is_fresh(frame, current) and identity != after_identity:
                    return {
                        "identity": identity,
                        "jpeg": base64.b64decode(frame.color_jpeg_b64),
                        "width": frame.color_width,
                        "height": frame.color_height,
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
