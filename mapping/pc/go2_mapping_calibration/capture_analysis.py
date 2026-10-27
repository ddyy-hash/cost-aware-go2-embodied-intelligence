"""Calibration capture identities and bounded association utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import cv2
import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_rgbd.decode import CameraInfoRecord

from .analysis import segment_lio_pose_epochs
from .models import PlaneObservation
from .target import BoardPose, BoardSpec, go2_lab_board_spec
from .timebase import AffineClockFit


@dataclass(frozen=True)
class CameraInfoIdentity:
    sha256: str
    raw_message_sha256: str
    contract: dict[str, object]


@dataclass(frozen=True)
class DepthFrameEvidence:
    recorder_timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    encoding: str
    image: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.image)
        if value.ndim != 2 or value.size == 0:
            raise ValueError("depth evidence must be a non-empty two-dimensional image")
        value = np.array(value, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "image", value)


@dataclass(frozen=True)
class LidarFrameEvidence:
    recorder_timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    points_xyz: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.points_xyz, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
            raise ValueError("LiDAR evidence must be a finite N by 3 point array")
        value = np.array(value, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "points_xyz", value)


@dataclass(frozen=True)
class AggregatedLidarWindow:
    reference_time_s: float
    association_error_ms: float
    frame_count: int
    points_xyz: np.ndarray
    lio_epoch_id: int = 0

    def __post_init__(self) -> None:
        value = np.asarray(self.points_xyz, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
            raise ValueError("aggregated LiDAR points must be a finite N by 3 array")
        value = np.array(value, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "points_xyz", value)
        if int(self.lio_epoch_id) < 0:
            raise ValueError("aggregated LiDAR LIO epoch ID cannot be negative")
        object.__setattr__(self, "lio_epoch_id", int(self.lio_epoch_id))


@dataclass(frozen=True)
class CameraBoardObservation:
    header_timestamp_ns: int
    recorder_timestamp_ns: int
    center_camera: tuple[float, float, float]
    plane: PlaneObservation
    reprojection_p95_px: float
    correspondence_count: int
    camera_from_board: np.ndarray
    reprojection_residuals_px: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        value = np.asarray(self.camera_from_board, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(value)) or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("camera_from_board must be a finite homogeneous transform")
        if not np.allclose(value[:3, :3] @ value[:3, :3].T, np.eye(3), atol=1e-5):
            raise ValueError("camera_from_board rotation must be orthonormal")
        value = np.array(value, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "camera_from_board", value)

    @classmethod
    def from_pose(
        cls,
        pose: BoardPose,
        *,
        board: BoardSpec | None = None,
        header_timestamp_ns: int,
        recorder_timestamp_ns: int,
        reprojection_p95_px: float,
        correspondence_count: int,
        reprojection_residuals_px: Sequence[float] = (),
    ) -> "CameraBoardObservation":
        if correspondence_count <= 0:
            raise ValueError("board correspondence_count must be positive")
        if reprojection_p95_px < 0.0:
            raise ValueError("board reprojection residual cannot be negative")
        board = board or go2_lab_board_spec()
        camera_from_board = _pose_transform(pose)
        center_board = np.asarray(
            (board.pattern_width_m / 2.0, board.pattern_height_m / 2.0, 0.0),
            dtype=np.float64,
        )
        center_camera = camera_from_board[:3, :3] @ center_board + camera_from_board[:3, 3]
        plane = camera_plane_from_board_pose(
            pose,
            recorder_time_s=recorder_timestamp_ns / 1e9,
            weight=float(correspondence_count),
        )
        return cls(
            header_timestamp_ns=int(header_timestamp_ns),
            recorder_timestamp_ns=int(recorder_timestamp_ns),
            center_camera=tuple(float(value) for value in center_camera),
            plane=plane,
            reprojection_p95_px=float(reprojection_p95_px),
            correspondence_count=int(correspondence_count),
            camera_from_board=camera_from_board,
            reprojection_residuals_px=tuple(float(value) for value in reprojection_residuals_px),
        )


def _pose_transform(pose: BoardPose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(pose.translation_m, dtype=np.float64).reshape(3)
    return transform


def camera_info_identity(record: CameraInfoRecord) -> CameraInfoIdentity:
    """Hash the full calibration contract without its changing ROS header."""

    contract = {
        "width": int(record.width),
        "height": int(record.height),
        "distortion_model": record.distortion_model,
        "k": np.asarray(record.k, dtype=np.float64).reshape(-1).tolist(),
        "d": np.asarray(record.d, dtype=np.float64).reshape(-1).tolist(),
        "r": np.asarray(record.r, dtype=np.float64).reshape(-1).tolist(),
        "p": np.asarray(record.p, dtype=np.float64).reshape(-1).tolist(),
        "rectified": not np.allclose(record.r, np.eye(3), atol=1e-12),
    }
    return CameraInfoIdentity(
        sha256=canonical_sha256(contract),
        raw_message_sha256=record.message_sha256,
        contract=contract,
    )


def camera_plane_from_board_pose(
    pose: BoardPose, *, recorder_time_s: float, weight: float = 1.0
) -> PlaneObservation:
    rotation, _ = cv2.Rodrigues(np.asarray(pose.rotation_vector, dtype=np.float64))
    translation = np.asarray(pose.translation_m, dtype=np.float64)
    normal = rotation @ np.asarray((0.0, 0.0, 1.0))
    distance = float(np.dot(normal, translation))
    if distance < 0.0:
        normal = -normal
        distance = -distance
    return PlaneObservation(recorder_time_s, normal, distance, weight)


T = TypeVar("T")


def nearest_within(
    samples: Sequence[tuple[float, T]], timestamp_s: float, *, maximum_delta_s: float
) -> tuple[float, T] | None:
    if maximum_delta_s < 0.0:
        raise ValueError("maximum_delta_s cannot be negative")
    if not samples:
        return None
    best = min(samples, key=lambda value: abs(value[0] - timestamp_s))
    return best if abs(best[0] - timestamp_s) <= maximum_delta_s else None


def _tag_pose_samples(
    samples: Sequence[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
) -> tuple[tuple[float, np.ndarray, int], ...]:
    if not samples:
        return ()
    tagged: list[tuple[float, np.ndarray, int]] = []
    if all(len(item) == 3 for item in samples):
        for timestamp_s, pose, epoch_id in samples:
            if int(epoch_id) < 0:
                raise ValueError("LIO epoch ID cannot be negative")
            tagged.append((float(timestamp_s), np.asarray(pose, dtype=np.float64), int(epoch_id)))
    elif all(len(item) == 2 for item in samples):
        plain = [(float(timestamp_s), np.asarray(pose, dtype=np.float64)) for timestamp_s, pose in samples]
        epochs = segment_lio_pose_epochs(
            [(int(round(timestamp_s * 1e9)), pose) for timestamp_s, pose in plain]
        )
        for epoch_id, epoch in enumerate(epochs):
            for source_index in epoch.source_indices:
                timestamp_s, pose = plain[source_index]
                tagged.append((timestamp_s, pose, epoch_id))
    else:
        raise ValueError("LIO pose samples must either all include epoch IDs or all omit them")
    return tuple(sorted(tagged, key=lambda item: item[0]))


def _pose_at(
    samples: Sequence[tuple[float, np.ndarray, int]],
    timestamp_s: float,
    *,
    maximum_delta_s: float,
    epoch_id: int,
) -> np.ndarray | None:
    same_epoch = [item for item in samples if item[2] == epoch_id]
    if not same_epoch:
        return None
    ordered = sorted(same_epoch, key=lambda item: item[0])
    times = np.asarray([item[0] for item in ordered], dtype=np.float64)
    index = int(np.searchsorted(times, timestamp_s))
    if index == 0 or index == len(ordered):
        nearest_time, nearest_pose, _ = ordered[0 if index == 0 else -1]
        return None if abs(nearest_time - timestamp_s) > maximum_delta_s else np.asarray(nearest_pose, dtype=np.float64)
    left_time, left_pose, _ = ordered[index - 1]
    right_time, right_pose, _ = ordered[index]
    if min(timestamp_s - left_time, right_time - timestamp_s) > maximum_delta_s:
        return None
    span = right_time - left_time
    if span <= 0.0:
        return None
    alpha = float((timestamp_s - left_time) / span)
    left = np.asarray(left_pose, dtype=np.float64)
    right = np.asarray(right_pose, dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("LIO poses must be 4 by 4 transforms")
    relative_rotation = left[:3, :3].T @ right[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(relative_rotation)
    incremental_rotation, _ = cv2.Rodrigues(rotation_vector * alpha)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = left[:3, :3] @ incremental_rotation
    result[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return result


def _tag_lidar_samples(
    samples: Sequence[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
    pose_samples: Sequence[tuple[float, np.ndarray, int]],
    *,
    maximum_pose_delta_s: float,
) -> tuple[tuple[float, np.ndarray, int], ...]:
    tagged: list[tuple[float, np.ndarray, int]] = []
    for item in samples:
        if len(item) == 3:
            timestamp_s, points, epoch_id = item
            if int(epoch_id) < 0:
                raise ValueError("LiDAR LIO epoch ID cannot be negative")
            tagged.append((float(timestamp_s), np.asarray(points), int(epoch_id)))
            continue
        if len(item) != 2:
            raise ValueError("LiDAR samples must contain time, points, and optional epoch ID")
        timestamp_s, points = item
        nearest_pose = nearest_within(
            pose_samples,
            float(timestamp_s),
            maximum_delta_s=maximum_pose_delta_s,
        )
        if nearest_pose is None:
            continue
        tagged.append((float(timestamp_s), np.asarray(points), int(nearest_pose[2])))
    return tuple(sorted(tagged, key=lambda value: value[0]))


def aggregate_lidar_window(
    lidar_samples: Sequence[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
    pose_samples: Sequence[tuple[float, np.ndarray] | tuple[float, np.ndarray, int]],
    *,
    target_time_s: float,
    maximum_center_delta_s: float = 0.100,
    window_radius_s: float = 0.150,
    maximum_frames: int = 5,
    maximum_pose_delta_s: float = 0.150,
) -> AggregatedLidarWindow | None:
    """Motion-compensate a bounded LiDAR window into its center-frame pose."""

    if maximum_center_delta_s < 0.0 or window_radius_s < 0.0 or maximum_pose_delta_s < 0.0:
        raise ValueError("LiDAR aggregation time bounds cannot be negative")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    tagged_poses = _tag_pose_samples(pose_samples)
    tagged_lidar = _tag_lidar_samples(
        lidar_samples,
        tagged_poses,
        maximum_pose_delta_s=maximum_pose_delta_s,
    )
    center = nearest_within(tagged_lidar, target_time_s, maximum_delta_s=maximum_center_delta_s)
    if center is None:
        return None
    reference_time_s, _, reference_epoch_id = center
    reference_pose = _pose_at(
        tagged_poses,
        reference_time_s,
        maximum_delta_s=maximum_pose_delta_s,
        epoch_id=reference_epoch_id,
    )
    if reference_pose is None:
        return None
    candidates = [
        item
        for item in tagged_lidar
        if item[2] == reference_epoch_id and abs(item[0] - reference_time_s) <= window_radius_s
    ]
    candidates = sorted(candidates, key=lambda item: (abs(item[0] - reference_time_s), item[0]))[
        :maximum_frames
    ]
    reference_from_world = np.linalg.inv(reference_pose)
    transformed: list[np.ndarray] = []
    for frame_time_s, points, _ in sorted(candidates, key=lambda item: item[0]):
        world_from_frame = _pose_at(
            tagged_poses,
            frame_time_s,
            maximum_delta_s=maximum_pose_delta_s,
            epoch_id=reference_epoch_id,
        )
        if world_from_frame is None:
            continue
        frame = np.asarray(points, dtype=np.float64)
        if frame.ndim != 2 or frame.shape[1] != 3:
            raise ValueError("LiDAR frame points must be an N by 3 array")
        reference_from_frame = reference_from_world @ world_from_frame
        transformed.append(
            frame @ reference_from_frame[:3, :3].T + reference_from_frame[:3, 3]
        )
    if not transformed:
        return None
    return AggregatedLidarWindow(
        reference_time_s=float(reference_time_s),
        association_error_ms=abs(float(reference_time_s) - float(target_time_s)) * 1000.0,
        frame_count=len(transformed),
        points_xyz=np.concatenate(transformed, axis=0),
        lio_epoch_id=reference_epoch_id,
    )


def mapped_camera_plane(observation: CameraBoardObservation, fit: AffineClockFit) -> PlaneObservation:
    """Map a camera header timestamp into the recorder reference clock."""

    return PlaneObservation(
        recorder_time_s=fit.map_header_ns(observation.header_timestamp_ns) / 1e9,
        normal=observation.plane.normal,
        distance_m=observation.plane.distance_m,
        weight=observation.plane.weight,
    )
