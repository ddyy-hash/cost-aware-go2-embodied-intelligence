"""Versioned LiDAR sensor-frame transforms for live and offline fusion.

Recorded data stays in native sensor coordinates. Consumers apply a registered
transform whose ID is part of the source and derived identities. The current
Go2 transform contains rotation only; mounting translation is not accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping

import numpy as np


IDENTITY_SENSOR_FRAME_V1 = "IDENTITY_SENSOR_FRAME_v1"
PROVISIONAL_GO2_UTLIDAR_PITCH180_V2 = "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2"


@dataclass(frozen=True)
class SensorFrameTransform:
    id: str
    rotation: np.ndarray
    translation: np.ndarray
    provisional: bool
    description: str

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("sensor frame transform must contain a 3x3 rotation and XYZ translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("sensor frame transform must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
            raise ValueError("sensor frame transform rotation must be orthonormal")
        if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-9):
            raise ValueError("sensor frame transform rotation must have determinant +1")
        if not str(self.id).strip() or not str(self.description).strip():
            raise ValueError("sensor frame transform ID and description are required")
        rotation = np.array(rotation, copy=True)
        translation = np.array(translation, copy=True)
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)


_TRANSFORMS: Mapping[str, SensorFrameTransform] = {
    IDENTITY_SENSOR_FRAME_V1: SensorFrameTransform(
        IDENTITY_SENSOR_FRAME_V1,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        False,
        "Native LiDAR coordinates; retained for legacy and already-rectified sources.",
    ),
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2: SensorFrameTransform(
        PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        np.diag((-1.0, 1.0, -1.0)),
        np.zeros(3, dtype=np.float64),
        True,
        "Provisional Go2 utlidar half-turn used to place observed ground below the sensor.",
    ),
}


def get_sensor_frame_transform(transform_id: str | None) -> SensorFrameTransform:
    """Return a registered transform, defaulting missing legacy metadata to identity."""

    key = IDENTITY_SENSOR_FRAME_V1 if transform_id is None else str(transform_id).strip()
    try:
        return _TRANSFORMS[key]
    except KeyError as exc:
        raise ValueError(f"unknown sensor frame transform: {key}") from exc


def normalize_sensor_frame_transform_id(transform_id: str | None) -> str:
    return get_sensor_frame_transform(transform_id).id


def transform_points(points: np.ndarray, transform_id: str | None) -> np.ndarray:
    """Transform row-major native XYZ points into the canonical mapping frame."""

    transform = get_sensor_frame_transform(transform_id)
    source = np.asarray(points)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("sensor points must be an N by 3 array")
    if not np.isfinite(source).all():
        raise ValueError("sensor points must be finite")
    if not len(source):
        return np.ascontiguousarray(source, dtype=np.float64)
    if transform.id == IDENTITY_SENSOR_FRAME_V1:
        return np.array(source, copy=True)
    # CUDA is opt-in because device transfer is slower for tiny clouds. The
    # CPU path remains the deterministic fallback when CUDA is unavailable.
    if os.environ.get("GO2_MAPPING_USE_CUDA", "0") == "1":
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda")
                values = torch.as_tensor(source, dtype=torch.float32, device=device)
                rotation = torch.as_tensor(np.array(transform.rotation.T, copy=True), dtype=torch.float32, device=device)
                translation = torch.as_tensor(np.array(transform.translation, copy=True), dtype=torch.float32, device=device)
                accelerated = values @ rotation + translation
                return np.ascontiguousarray(accelerated.cpu().numpy(), dtype=np.float64)
        except (ImportError, RuntimeError):
            pass
    return np.ascontiguousarray(
        source.astype(np.float64, copy=False) @ transform.rotation.T + transform.translation,
        dtype=np.float64,
    )


def bind_pose_generation(pose_generation: str, transform_id: str | None) -> str:
    """Bind a non-identity point-frame transform to a content-addressed pose generation."""

    if not isinstance(pose_generation, str) or not pose_generation:
        raise ValueError("pose_generation must be a non-empty string")
    transform = get_sensor_frame_transform(transform_id)
    if transform.id == IDENTITY_SENSOR_FRAME_V1:
        return pose_generation
    payload: dict[str, Any] = {
        "schema_version": 1,
        "base_pose_generation": pose_generation,
        "sensor_frame_transform_id": transform.id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def transform_id_from_payload(payload: Mapping[str, Any] | None) -> str:
    """Read the optional wire field while keeping pre-transform payloads readable."""

    if payload is None:
        return IDENTITY_SENSOR_FRAME_V1
    return normalize_sensor_frame_transform_id(payload.get("sensor_frame_transform_id"))
