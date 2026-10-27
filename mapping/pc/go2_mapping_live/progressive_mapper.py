"""Incremental LiDAR preview map with explicit survey-quality gates."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sensor_frame import (
    IDENTITY_SENSOR_FRAME_V1,
    normalize_sensor_frame_transform_id,
    transform_points,
)


class KeyframeError(ValueError):
    pass


@dataclass(frozen=True)
class Keyframe:
    map_id: str
    sequence: int
    timestamp_ns: int
    calibration_id: str
    points_xyz: np.ndarray
    imu_quaternion_wxyz: Tuple[float, float, float, float]
    pose_map_from_lidar: Optional[np.ndarray] = None
    points_in_map_frame: bool = False
    sensor_frame_transform_id: str = IDENTITY_SENSOR_FRAME_V1

    def validate(self) -> None:
        if not self.map_id or self.sequence < 0 or self.timestamp_ns <= 0 or not self.calibration_id:
            raise KeyframeError("keyframe identity is incomplete")
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3 or len(self.points_xyz) < 100:
            raise KeyframeError("keyframe must contain at least 100 XYZ points")
        if not np.isfinite(self.points_xyz).all():
            raise KeyframeError("keyframe contains non-finite points")
        if not np.isfinite(self.imu_quaternion_wxyz).all():
            raise KeyframeError("keyframe contains an invalid IMU quaternion")
        try:
            normalize_sensor_frame_transform_id(self.sensor_frame_transform_id)
        except (TypeError, ValueError) as exc:
            raise KeyframeError(f"sensor frame transform is invalid: {exc}") from exc
        if self.pose_map_from_lidar is not None:
            if self.pose_map_from_lidar.shape != (4, 4) or not np.isfinite(self.pose_map_from_lidar).all():
                raise KeyframeError("LIO pose must be a finite 4x4 matrix")


class ProgressiveMapper:
    """Maintains a bounded voxel submap; never labels preview as survey-grade."""

    def __init__(self, voxel_m: float = 0.05, preview_point_budget: int = 12000) -> None:
        if voxel_m <= 0 or preview_point_budget < 100:
            raise ValueError("invalid map budget")
        self.voxel_m = voxel_m
        self.preview_point_budget = preview_point_budget
        self._voxels: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._map_id: Optional[str] = None
        self._calibration_id: Optional[str] = None
        self._last_sequence = -1
        self._last_keyframe_identity: Optional[Tuple[str, int, int, str, str]] = None
        self._sensor_frame_transform_id = IDENTITY_SENSOR_FRAME_V1
        self._trajectory: List[Tuple[float, float, float]] = []
        self._last_pose_map_from_lidar: Optional[np.ndarray] = None
        self._heading_xy: Tuple[float, float] = (1.0, 0.0)
        self._residual_m: Optional[float] = None
        self._quality = "AWAITING_KEYFRAME"
        self._source_points = 0
        self._frames = 0
        self._first_timestamp_ns: Optional[int] = None
        self._last_timestamp_ns: Optional[int] = None
        self._last_update_monotonic: Optional[float] = None
        self._last_trajectory_position: Optional[np.ndarray] = None
        self._trajectory_length_m = 0.0
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None

    def integrate(self, keyframe: Keyframe, residual_m: Optional[float], calibrated_lio: bool) -> bool:
        keyframe.validate()
        transform_id = normalize_sensor_frame_transform_id(keyframe.sensor_frame_transform_id)
        if keyframe.sequence <= self._last_sequence:
            return False
        if residual_m is not None and (not np.isfinite(residual_m) or residual_m < 0):
            raise KeyframeError("invalid registration residual")
        if keyframe.pose_map_from_lidar is None:
            raise KeyframeError("keyframe has no verified map-from-lidar pose")
        if residual_m is not None and residual_m > 0.20:
            raise KeyframeError("registration residual exceeds preview safety limit")
        if self._map_id is not None and keyframe.map_id != self._map_id:
            raise KeyframeError("keyframe map_id does not match the active live map")
        if self._frames and transform_id != self._sensor_frame_transform_id:
            raise KeyframeError("sensor frame transform cannot change within an active live map")
        if calibrated_lio:
            if keyframe.calibration_id.upper().startswith("PROVISIONAL_") or "UNVERIFIED" in keyframe.calibration_id.upper():
                raise KeyframeError("unverified calibration cannot promote a preview map")
            if residual_m is None or residual_m > 0.05:
                raise KeyframeError("refined LIO requires a measured residual at or below 5 cm")

        pose = keyframe.pose_map_from_lidar
        points = keyframe.points_xyz.astype(np.float64, copy=False)
        # Registered LIO clouds already live in the map frame.  Transforming
        # them a second time produces the thick, duplicated geometry seen in
        # a raw preview map.  The pose remains authoritative for trajectory.
        if keyframe.points_in_map_frame:
            world = points
        else:
            rectified = transform_points(points, transform_id)
            world = rectified @ pose[:3, :3].T + pose[:3, 3]
        indices = np.floor(world / self.voxel_m).astype(np.int64)
        for index, point in zip(indices, world):
            self._voxels[tuple(index)] = point.astype(np.float32)
        self._map_id = keyframe.map_id
        self._calibration_id = keyframe.calibration_id
        self._sensor_frame_transform_id = transform_id
        self._last_sequence = keyframe.sequence
        self._last_keyframe_identity = (
            keyframe.map_id,
            keyframe.sequence,
            keyframe.timestamp_ns,
            keyframe.calibration_id,
            transform_id,
        )
        self._trajectory.append(tuple(float(value) for value in pose[:3, 3]))
        self._trajectory = self._trajectory[-2000:]
        self._last_pose_map_from_lidar = pose.copy()
        self._source_points += len(points)
        self._frames += 1
        self._first_timestamp_ns = keyframe.timestamp_ns if self._first_timestamp_ns is None else self._first_timestamp_ns
        self._last_timestamp_ns = keyframe.timestamp_ns
        self._last_update_monotonic = time.monotonic()
        position = np.asarray(pose[:3, 3], dtype=np.float64)
        if self._last_trajectory_position is not None:
            self._trajectory_length_m += float(np.linalg.norm(position - self._last_trajectory_position))
        self._last_trajectory_position = np.array(position, copy=True)
        self._bounds_min = np.min(world, axis=0) if self._bounds_min is None else np.minimum(self._bounds_min, np.min(world, axis=0))
        self._bounds_max = np.max(world, axis=0) if self._bounds_max is None else np.maximum(self._bounds_max, np.max(world, axis=0))
        heading_xy = pose[:2, 0]
        heading_norm = float(np.linalg.norm(heading_xy))
        if heading_norm > 1e-9:
            self._heading_xy = (float(heading_xy[0] / heading_norm), float(heading_xy[1] / heading_norm))
        self._residual_m = residual_m
        self._quality = "REFINED" if calibrated_lio else "PREVIEW_UNCALIBRATED"
        return True

    def room_scan_snapshot(self, now: Optional[float] = None) -> dict:
        """Return truthful global live-scan state without exposing point data."""
        if self._last_update_monotonic is None:
            age_ms = None
            state = "WAITING_FOR_LIVE_DATA"
        else:
            age_ms = max(0, int(round(((time.monotonic() if now is None else now) - self._last_update_monotonic) * 1000.0)))
            state = "SCANNING" if age_ms <= 15_000 else "STALE"
        bounds_min = None if self._bounds_min is None else self._bounds_min.astype(float).tolist()
        bounds_max = None if self._bounds_max is None else self._bounds_max.astype(float).tolist()
        extent = (
            [0.0, 0.0, 0.0]
            if bounds_min is None or bounds_max is None
            else [round(float(value), 3) for value in (self._bounds_max - self._bounds_min)]
        )
        return {
            "state": state,
            "live": state == "SCANNING",
            "map_id": self._map_id,
            "calibration_id": self._calibration_id,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
            "frames": self._frames,
            "source_points": self._source_points,
            "voxel_points": len(self._voxels),
            "voxel_size_m": self.voxel_m,
            "first_timestamp_ns": self._first_timestamp_ns,
            "last_sequence": self._last_sequence,
            "last_timestamp_ns": self._last_timestamp_ns,
            "last_frame_age_ms": age_ms,
            "bounds_min_m": bounds_min,
            "bounds_max_m": bounds_max,
            "extent_m": extent,
            "trajectory_length_m": round(self._trajectory_length_m, 3),
        }

    def is_current_duplicate(self, keyframe: Keyframe) -> bool:
        """Allow a lost acknowledgement to be replayed without a second fusion."""
        return self._last_keyframe_identity == (
            keyframe.map_id,
            keyframe.sequence,
            keyframe.timestamp_ns,
            keyframe.calibration_id,
            normalize_sensor_frame_transform_id(keyframe.sensor_frame_transform_id),
        )

    def snapshot(self) -> dict:
        voxels = list(self._voxels.values())
        stride = max(1, (len(voxels) + self.preview_point_budget - 1) // self.preview_point_budget)
        preview = [point.tolist() for point in voxels[::stride]]
        position = self._trajectory[-1] if self._trajectory else (0.0, 0.0, 0.0)
        return {
            "quality": self._quality,
            "map_id": self._map_id,
            "calibration_id": self._calibration_id,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
            "last_sequence": self._last_sequence,
            "points": len(self._voxels),
            "residual_cm": None if self._residual_m is None else round(self._residual_m * 100.0, 2),
            "position": list(position),
            "heading_xy": list(self._heading_xy),
            "trajectory": [list(pose) for pose in self._trajectory[-300:]],
            "pose_map_from_lidar": None if self._last_pose_map_from_lidar is None else self._last_pose_map_from_lidar.ravel().tolist(),
            "preview_points": preview,
            "room_scan": self.room_scan_snapshot(),
        }

    def telemetry_snapshot(self) -> dict:
        """Return the 15 Hz dashboard state without materializing map points.

        Full snapshots use a lower rate to preserve the whole-map view without
        slowing pose and camera refreshes as the map grows.
        """
        position = self._trajectory[-1] if self._trajectory else (0.0, 0.0, 0.0)
        return {
            "quality": self._quality,
            "map_id": self._map_id,
            "calibration_id": self._calibration_id,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
            "last_sequence": self._last_sequence,
            "points": len(self._voxels),
            "residual_cm": None if self._residual_m is None else round(self._residual_m * 100.0, 2),
            "position": list(position),
            "heading_xy": list(self._heading_xy),
            "trajectory": [list(pose) for pose in self._trajectory[-300:]],
            "pose_map_from_lidar": None if self._last_pose_map_from_lidar is None else self._last_pose_map_from_lidar.ravel().tolist(),
            "room_scan": self.room_scan_snapshot(),
        }

    def acknowledgement(self) -> dict:
        """Return constant-size live-ingest state without materializing points."""
        return {
            "points": len(self._voxels),
            "quality": self._quality,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
        }

    def next_sequence(self) -> int:
        """Return the next unused sequence for a producer recovering after restart."""
        return self._last_sequence + 1
