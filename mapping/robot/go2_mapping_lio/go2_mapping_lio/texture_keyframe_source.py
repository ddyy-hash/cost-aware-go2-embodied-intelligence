"""Texture-frame selection and protocol-v3 encoding for the subscriber-only producer."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Optional
import zlib

import numpy as np

try:
    from .detail_surface_source import EncodedMessage, PayloadLimitExceeded
except ImportError:  # Direct execution from the deployed release directory.
    from detail_surface_source import EncodedMessage, PayloadLimitExceeded


@dataclass(frozen=True)
class TextureSelection:
    accepted: bool
    blocker: Optional[str]
    sharpness_laplacian: float
    mean_luma: float
    clipped_fraction: float
    viewpoint_baseline_m: float
    viewpoint_rotation_deg: float
    independent_view_id: str

    def quality_payload(self) -> dict[str, Any]:
        return {
            "selection_accepted": self.accepted,
            "sharpness_laplacian": self.sharpness_laplacian,
            "mean_luma": self.mean_luma,
            "clipped_fraction": self.clipped_fraction,
            "viewpoint_baseline_m": self.viewpoint_baseline_m,
            "viewpoint_rotation_deg": self.viewpoint_rotation_deg,
            "independent_view_id": self.independent_view_id,
        }


class TextureQualitySelector:
    """Admit focused, normally exposed frames with a real camera baseline."""

    def __init__(
        self,
        *,
        minimum_sharpness: float = 20.0,
        minimum_luma: float = 20.0,
        maximum_luma: float = 235.0,
        maximum_clipped_fraction: float = 0.1,
        minimum_translation_m: float = 0.08,
        minimum_rotation_deg: float = 5.0,
    ) -> None:
        values = (
            minimum_sharpness,
            minimum_luma,
            maximum_luma,
            maximum_clipped_fraction,
            minimum_translation_m,
            minimum_rotation_deg,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("texture quality thresholds must be finite")
        if minimum_sharpness < 0 or not 0 <= minimum_luma < maximum_luma <= 255:
            raise ValueError("texture focus or exposure thresholds are invalid")
        if not 0 <= maximum_clipped_fraction <= 1:
            raise ValueError("maximum_clipped_fraction must be in [0, 1]")
        if minimum_translation_m < 0 or minimum_rotation_deg < 0:
            raise ValueError("texture baseline thresholds must be non-negative")
        self.minimum_sharpness = float(minimum_sharpness)
        self.minimum_luma = float(minimum_luma)
        self.maximum_luma = float(maximum_luma)
        self.maximum_clipped_fraction = float(maximum_clipped_fraction)
        self.minimum_translation_m = float(minimum_translation_m)
        self.minimum_rotation_deg = float(minimum_rotation_deg)
        self._last_accepted_pose: Optional[np.ndarray] = None

    @staticmethod
    def _image_metrics(image_bgr: np.ndarray) -> tuple[float, float, float]:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("image_bgr must have shape (H, W, 3)")
        if not np.issubdtype(image.dtype, np.integer) or image.min() < 0 or image.max() > 255:
            raise ValueError("image_bgr must contain uint8-range pixels")
        values = image.astype(np.float32, copy=False)
        gray = values[:, :, 0] * 0.114 + values[:, :, 1] * 0.587 + values[:, :, 2] * 0.299
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            sharpness = 0.0
        else:
            centre = gray[1:-1, 1:-1]
            laplacian = (
                gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
                - 4.0 * centre
            )
            sharpness = float(np.var(laplacian, dtype=np.float64))
        mean_luma = float(np.mean(gray, dtype=np.float64))
        clipped_fraction = float(np.mean((gray <= 2.0) | (gray >= 253.0)))
        return sharpness, mean_luma, clipped_fraction

    @staticmethod
    def _baseline(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        translation = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
        relative_rotation = previous[:3, :3].T @ current[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        rotation_deg = math.degrees(math.acos(cosine))
        return translation, rotation_deg

    def evaluate(
        self,
        image_bgr: np.ndarray,
        pose_map_from_lidar: np.ndarray,
        *,
        independent_view_id: str,
    ) -> TextureSelection:
        pose = np.asarray(pose_map_from_lidar, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose_map_from_lidar must be a finite 4x4 matrix")
        if not independent_view_id:
            raise ValueError("independent_view_id is required")
        sharpness, mean_luma, clipped_fraction = self._image_metrics(image_bgr)
        baseline_m, rotation_deg = (0.0, 0.0)
        if self._last_accepted_pose is not None:
            baseline_m, rotation_deg = self._baseline(self._last_accepted_pose, pose)

        blocker: Optional[str] = None
        if not self.minimum_luma <= mean_luma <= self.maximum_luma:
            blocker = "EXPOSURE_OUT_OF_RANGE"
        elif clipped_fraction > self.maximum_clipped_fraction:
            blocker = "CLIPPED_EXPOSURE"
        elif sharpness < self.minimum_sharpness:
            blocker = "INSUFFICIENT_SHARPNESS"
        elif self._last_accepted_pose is not None and (
            baseline_m < self.minimum_translation_m and rotation_deg < self.minimum_rotation_deg
        ):
            blocker = "INSUFFICIENT_VIEWPOINT_BASELINE"

        accepted = blocker is None
        if accepted:
            self._last_accepted_pose = pose.copy()
        return TextureSelection(
            accepted=accepted,
            blocker=blocker,
            sharpness_laplacian=round(sharpness, 6),
            mean_luma=round(mean_luma, 6),
            clipped_fraction=round(clipped_fraction, 8),
            viewpoint_baseline_m=round(baseline_m, 6),
            viewpoint_rotation_deg=round(rotation_deg, 6),
            independent_view_id=str(independent_view_id),
        )


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


def build_texture_keyframe_message(
    *,
    building_id: str,
    session_id: str,
    map_id: str,
    producer_instance_id: str,
    producer_epoch: int,
    texture_sequence: int,
    timestamp_ns: int,
    calibration_id: str,
    source_config_sha256: str,
    source_topic: str,
    source_frame_id: str,
    source_encoding: str,
    source_width: int,
    source_height: int,
    color_jpeg: bytes,
    intrinsics: Mapping[str, Any],
    lio_pose_timestamp_ns: int,
    pose_skew_ns: int,
    pose_association_clock: str,
    pose_map_from_lidar: np.ndarray,
    quality: Mapping[str, Any],
    aligned_depth_u16: Optional[np.ndarray] = None,
    depth_scale_m: float = 0.001,
    payload_limit_bytes: int = 4 * 1024 * 1024,
) -> EncodedMessage:
    for name, value in (("producer_epoch", producer_epoch), ("texture_sequence", texture_sequence)):
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(timestamp_ns, bool) or int(timestamp_ns) <= 0:
        raise ValueError("timestamp_ns must be a positive integer")
    if isinstance(lio_pose_timestamp_ns, bool) or int(lio_pose_timestamp_ns) <= 0:
        raise ValueError("lio_pose_timestamp_ns must be a positive integer")
    if pose_association_clock not in {"ros_header", "subscriber_monotonic"}:
        raise ValueError("pose_association_clock is invalid")
    width, height = int(source_width), int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    jpeg = bytes(color_jpeg)
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("color_jpeg must be a complete JPEG")
    pose = np.asarray(pose_map_from_lidar, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose_map_from_lidar must be a finite 4x4 matrix")
    camera_intrinsics = {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "width": width,
        "height": height,
    }
    quality_payload = dict(quality)
    if quality_payload.get("selection_accepted") is not True:
        raise ValueError("only quality-selected texture frames may be encoded")
    payload: dict[str, Any] = {
        "message_type": "texture_keyframe",
        "protocol_version": 3,
        "building_id": _identifier(building_id, "building_id"),
        "session_id": _identifier(session_id, "session_id"),
        "map_id": _identifier(map_id, "map_id"),
        "producer_instance_id": _identifier(producer_instance_id, "producer_instance_id"),
        "producer_epoch": int(producer_epoch),
        "texture_sequence": int(texture_sequence),
        "timestamp_ns": int(timestamp_ns),
        "calibration_id": _identifier(calibration_id, "calibration_id"),
        "source_config_sha256": _sha256(source_config_sha256, "source_config_sha256"),
        "content_sha256": hashlib.sha256(jpeg).hexdigest(),
        "source": {
            "topic": _identifier(source_topic, "source_topic"),
            "frame_id": _identifier(source_frame_id, "source_frame_id"),
            "encoding": source_encoding,
            "width": width,
            "height": height,
        },
        "color": {
            "encoding": "jpeg-base64",
            "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
            "width": width,
            "height": height,
        },
        "intrinsics": camera_intrinsics,
        "lio_pose_timestamp_ns": int(lio_pose_timestamp_ns),
        "pose_skew_ns": int(pose_skew_ns),
        "pose_association_clock": pose_association_clock,
        "pose_map_from_lidar": pose.tolist(),
        "quality": quality_payload,
    }
    if aligned_depth_u16 is not None:
        depth = np.asarray(aligned_depth_u16)
        if depth.shape != (height, width):
            raise ValueError("aligned depth dimensions must match the color frame")
        exact_depth = np.ascontiguousarray(depth, dtype="<u2").tobytes()
        compressed_depth = zlib.compress(exact_depth, level=1)
        payload["depth"] = {
            "encoding": "u16-le-zlib-base64",
            "data_b64": base64.b64encode(compressed_depth).decode("ascii"),
            "compressed_bytes_sha256": hashlib.sha256(compressed_depth).hexdigest(),
            "width": width,
            "height": height,
            "scale_m": float(depth_scale_m),
            "aligned_to_color": True,
        }
    json_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    message = EncodedMessage(payload, json_text, len(json_text.encode("utf-8")))
    if message.message_bytes > int(payload_limit_bytes):
        raise PayloadLimitExceeded(message.message_bytes, int(payload_limit_bytes))
    return message
