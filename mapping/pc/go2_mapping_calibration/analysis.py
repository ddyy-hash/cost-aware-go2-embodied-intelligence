"""SE(3), TF-chain, and motion-analysis primitives for calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from mapping.pc.go2_mapping_rgbd.decode import TransformRecord

from .models import MotionExcitation


class CalibrationAnalysisError(ValueError):
    """Raised when a transform or motion record is not physically usable."""


# These thresholds account for normal motion between poses in the 5 Hz LIO
# stream. A larger step starts a new world epoch and is excluded from operator
# excitation measurements.
LIO_EPOCH_MAX_TRANSLATION_STEP_M = 0.08
LIO_EPOCH_MAX_ROTATION_STEP_DEG = 8.0


@dataclass(frozen=True)
class LioPoseEpoch:
    """A contiguous sequence of LIO poses sharing one local world epoch."""

    poses: tuple[tuple[int, np.ndarray], ...]
    source_indices: tuple[int, ...]
    reset_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.source_indices) or not self.poses:
            raise ValueError("LIO pose epoch must contain matching non-empty poses and indices")
        values: list[tuple[int, np.ndarray]] = []
        for timestamp, pose in self.poses:
            values.append((int(timestamp), _rigid(pose, "LIO pose")))
        indexes = tuple(int(value) for value in self.source_indices)
        if any(value < 0 for value in indexes) or len(set(indexes)) != len(indexes):
            raise ValueError("LIO epoch source indices must be unique and non-negative")
        object.__setattr__(self, "poses", tuple(values))
        object.__setattr__(self, "source_indices", indexes)


def _pose_step(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
    relative_rotation = left[:3, :3].T @ right[:3, :3]
    vector, _ = cv2.Rodrigues(relative_rotation)
    rotation_deg = math.degrees(float(np.linalg.norm(vector)))
    return translation, rotation_deg


def segment_lio_pose_epochs(
    poses: Sequence[tuple[int, np.ndarray]],
    *,
    maximum_translation_step_m: float = LIO_EPOCH_MAX_TRANSLATION_STEP_M,
    maximum_rotation_step_deg: float = LIO_EPOCH_MAX_ROTATION_STEP_DEG,
) -> tuple[LioPoseEpoch, ...]:
    """Split poses at discontinuities that indicate LIO reinitialization.

    Input order is not trusted. Returned epochs are time ordered, while each
    epoch retains the original source indices for joining synchronized streams.
    A reset reason is attached to the epoch after the discontinuity.
    """

    if maximum_translation_step_m <= 0.0 or maximum_rotation_step_deg <= 0.0:
        raise ValueError("LIO epoch step limits must be positive")
    indexed = [
        (index, int(timestamp), _rigid(pose, f"LIO pose[{index}]"))
        for index, (timestamp, pose) in enumerate(poses)
    ]
    if not indexed:
        return ()
    indexed.sort(key=lambda item: (item[1], item[0]))
    epochs: list[LioPoseEpoch] = []
    current: list[tuple[int, int, np.ndarray]] = [indexed[0]]
    pending_reason: str | None = None
    previous = indexed[0]
    for item in indexed[1:]:
        _, timestamp, pose = item
        _, previous_timestamp, previous_pose = previous
        reasons: list[str] = []
        if timestamp <= previous_timestamp:
            reasons.append("TIMESTAMP_NON_MONOTONIC")
        else:
            translation_step, rotation_step_deg = _pose_step(previous_pose, pose)
            if translation_step > maximum_translation_step_m:
                reasons.append(
                    f"TRANSLATION_STEP_EXCEEDED:{translation_step:.6f}M"
                )
            if rotation_step_deg > maximum_rotation_step_deg:
                reasons.append(f"ROTATION_STEP_EXCEEDED:{rotation_step_deg:.6f}DEG")
        if reasons:
            epochs.append(
                LioPoseEpoch(
                    tuple((stamp, value) for _, stamp, value in current),
                    tuple(source_index for source_index, _, _ in current),
                    pending_reason,
                )
            )
            current = [item]
            pending_reason = "LIO_EPOCH_RESET:" + ",".join(reasons)
        else:
            current.append(item)
        previous = item
    epochs.append(
        LioPoseEpoch(
            tuple((stamp, value) for _, stamp, value in current),
            tuple(source_index for source_index, _, _ in current),
            pending_reason,
        )
    )
    return tuple(epochs)


def tag_lio_pose_epochs(
    poses: Sequence[tuple[int, np.ndarray]],
) -> tuple[tuple[int, np.ndarray, int], ...]:
    """Return sorted ``(timestamp, pose, epoch_id)`` samples for association."""

    epochs = segment_lio_pose_epochs(poses)
    tagged: list[tuple[int, np.ndarray, int]] = []
    for epoch_id, epoch in enumerate(epochs):
        tagged.extend((timestamp, pose, epoch_id) for timestamp, pose in epoch.poses)
    return tuple(tagged)


def _robust_translation_span(poses: Sequence[tuple[int, np.ndarray]]) -> float:
    values = np.asarray([pose[:3, 3] for _, pose in poses], dtype=np.float64)
    if len(values) < 2:
        return 0.0
    if len(values) < 5:
        low, high = np.min(values, axis=0), np.max(values, axis=0)
    else:
        low, high = np.quantile(values, (0.05, 0.95), axis=0)
    return float(np.linalg.norm(high - low))


def _robust_rotation_span(poses: Sequence[tuple[int, np.ndarray]]) -> float:
    if len(poses) < 2:
        return 0.0
    reference = poses[0][1][:3, :3]
    angles: list[float] = []
    for _, pose in poses:
        _, rotation_deg = _pose_step(
            np.block(
                [
                    [reference, np.zeros((3, 1))],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ),
            pose,
        )
        angles.append(rotation_deg)
    return float(np.percentile(np.asarray(angles, dtype=np.float64), 95))


@dataclass(frozen=True)
class Se3Consensus:
    transform: np.ndarray
    translation_repeatability_mm: float
    rotation_repeatability_deg: float

    def __post_init__(self) -> None:
        value = _rigid(self.transform, "consensus transform")
        value.setflags(write=False)
        object.__setattr__(self, "transform", value)


def _rigid(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise CalibrationAnalysisError(f"{name} must be a finite 4 by 4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise CalibrationAnalysisError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise CalibrationAnalysisError(f"{name} rotation must be proper orthonormal")
    return np.array(matrix, copy=True)


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left @ right.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def se3_consensus(transforms: Sequence[np.ndarray]) -> Se3Consensus:
    """Return a projected rigid mean and conservative maximum pairwise spread."""

    if len(transforms) < 2:
        raise CalibrationAnalysisError("SE(3) consensus requires at least two independent transforms")
    values = [_rigid(value, f"transform[{index}]") for index, value in enumerate(transforms)]
    translation_spread = 0.0
    rotation_spread = 0.0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            translation_spread = max(
                translation_spread,
                float(np.linalg.norm(values[left][:3, 3] - values[right][:3, 3])) * 1000.0,
            )
            rotation_spread = max(
                rotation_spread,
                _rotation_distance_deg(values[left][:3, :3], values[right][:3, :3]),
            )
    average_rotation = np.mean([value[:3, :3] for value in values], axis=0)
    left_singular, _, right_singular = np.linalg.svd(average_rotation)
    projected = left_singular @ right_singular
    if np.linalg.det(projected) < 0.0:
        left_singular[:, -1] *= -1.0
        projected = left_singular @ right_singular
    transform = np.eye(4)
    transform[:3, :3] = projected
    transform[:3, 3] = np.mean([value[:3, 3] for value in values], axis=0)
    return Se3Consensus(transform, translation_spread, rotation_spread)


def _transform_matrix(record: TransformRecord) -> np.ndarray:
    x, y, z, w = record.rotation_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise CalibrationAnalysisError("TF quaternion is not normalized")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    result = np.eye(4)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    result[:3, 3] = record.translation_xyz
    return result


def resolve_frame_transform(
    records: Iterable[TransformRecord], *, target_frame: str, source_frame: str
) -> np.ndarray:
    """Resolve ``T_target_source`` from ROS TF parent/child transforms."""

    if not target_frame or not source_frame:
        raise CalibrationAnalysisError("source and target frame IDs are required")
    if target_frame == source_frame:
        return np.eye(4)
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    directed_pairs: dict[tuple[str, str], np.ndarray] = {}
    for record in records:
        parent_from_child = _transform_matrix(record)
        pair = (record.parent_frame_id, record.child_frame_id)
        previous = directed_pairs.get(pair)
        if previous is not None and not np.allclose(previous, parent_from_child, atol=1e-5):
            raise CalibrationAnalysisError(f"conflicting TF records for {pair[0]} <- {pair[1]}")
        directed_pairs[pair] = parent_from_child
    for (parent, child), parent_from_child in directed_pairs.items():
        # At a node representing coordinates in `child`, the edge to `parent`
        # converts those coordinates with T_parent_child.
        graph.setdefault(child, []).append((parent, parent_from_child))
        graph.setdefault(parent, []).append((child, np.linalg.inv(parent_from_child)))
    queue = deque([(source_frame, np.eye(4))])
    visited = {source_frame}
    while queue:
        frame, frame_from_source = queue.popleft()
        for neighbor, neighbor_from_frame in graph.get(frame, ()):
            if neighbor in visited:
                continue
            neighbor_from_source = neighbor_from_frame @ frame_from_source
            if neighbor == target_frame:
                return neighbor_from_source
            visited.add(neighbor)
            queue.append((neighbor, neighbor_from_source))
    raise CalibrationAnalysisError(f"no TF chain from {source_frame} to {target_frame}")


def motion_excitation_from_poses(
    poses: Sequence[tuple[int, np.ndarray]], *, fast_angular_speed_deg_s: float = 10.0
) -> MotionExcitation:
    """Measure excitation without counting motion across an LIO reset."""

    if len(poses) < 2:
        return MotionExcitation(0.0, 0, 0.0, 0.0)
    epochs = segment_lio_pose_epochs(poses)
    cumulative_rotation = 0.0
    cumulative_translation = 0.0
    fast_duration = 0.0
    rotation_vectors: list[np.ndarray] = []
    robust_translation_span = 0.0
    robust_rotation_span = 0.0
    for epoch in epochs:
        ordered = epoch.poses
        robust_translation_span = max(
            robust_translation_span, _robust_translation_span(ordered)
        )
        robust_rotation_span = max(robust_rotation_span, _robust_rotation_span(ordered))
        for (left_time, left), (right_time, right) in zip(ordered, ordered[1:]):
            duration_s = (right_time - left_time) / 1e9
            if duration_s <= 0.0:
                continue
            relative_rotation = left[:3, :3].T @ right[:3, :3]
            vector, _ = cv2.Rodrigues(relative_rotation)
            vector = vector.reshape(3)
            angle_deg = math.degrees(float(np.linalg.norm(vector)))
            cumulative_rotation += angle_deg
            cumulative_translation += float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
            if angle_deg / duration_s >= fast_angular_speed_deg_s:
                fast_duration += duration_s
            if np.linalg.norm(vector) > math.radians(0.01):
                rotation_vectors.append(vector)
    axes = 0
    if rotation_vectors:
        singular = np.linalg.svd(np.vstack(rotation_vectors), compute_uv=False)
        threshold = max(float(singular[0]) * 0.05, math.radians(0.01))
        axes = int(np.count_nonzero(singular >= threshold))
    return MotionExcitation(
        cumulative_rotation_deg=cumulative_rotation,
        rotation_axes=axes,
        translation_m=cumulative_translation,
        angular_speed_duration_s=fast_duration,
        lio_epoch_count=len(epochs),
        robust_translation_span_m=robust_translation_span,
        robust_rotation_span_deg=robust_rotation_span,
        reset_reasons=tuple(
            epoch.reset_reason for epoch in epochs[1:] if epoch.reset_reason is not None
        ),
    )
