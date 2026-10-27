"""Immutable LiDAR observations with raw-artifact provenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ObservationError(ValueError):
    """Raised when an observation frame violates its wire contract."""


def _readonly_array(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ObservationError(f"{field} must be a finite array with shape {shape}")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LidarObservationFrame:
    artifact_sha256: str
    frame_id: str
    record_timestamp_ns: int
    sensor_origin_xyz: np.ndarray
    points_xyz: np.ndarray
    pose_map_from_lidar: np.ndarray
    pose_generation: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_sha256, str) or not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ObservationError("artifact_sha256 must be exactly 64 ASCII hexadecimal characters")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ObservationError("frame_id is required")
        if isinstance(self.record_timestamp_ns, bool) or not isinstance(self.record_timestamp_ns, int) or self.record_timestamp_ns < 0:
            raise ObservationError("record_timestamp_ns must be a non-negative integer")
        if not isinstance(self.pose_generation, str) or not self.pose_generation:
            raise ObservationError("pose_generation is required")
        origin = _readonly_array(self.sensor_origin_xyz, (3,), "sensor_origin_xyz")
        points = np.asarray(self.points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ObservationError("points_xyz must be an N by 3 array")
        points = np.array(points, dtype=np.float64, copy=True)
        points.setflags(write=False)
        pose = _readonly_array(self.pose_map_from_lidar, (4, 4), "pose_map_from_lidar")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ObservationError("pose_map_from_lidar must be homogeneous")
        rotation = pose[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
        ):
            raise ObservationError("pose_map_from_lidar must be rigid")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        object.__setattr__(self, "sensor_origin_xyz", origin)
        object.__setattr__(self, "points_xyz", points)
        object.__setattr__(self, "pose_map_from_lidar", pose)


@dataclass(frozen=True)
class RejectedObservation:
    index: int
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class FilteredFrame:
    source_count: int
    accepted_indices: np.ndarray
    rejected: tuple[RejectedObservation, ...]
    artifact_sha256: str
    frame_id: str
    record_timestamp_ns: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.accepted_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) and (np.any(indices < 0) or np.any(indices >= self.source_count)):
            raise ObservationError("accepted_indices are outside the source frame")
        indices = np.unique(indices)
        indices.setflags(write=False)
        if not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ObservationError("filtered artifact hash is malformed")
        object.__setattr__(self, "accepted_indices", indices)
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
