"""Fuse dense LiDAR from checksum-verified ROS 2 SQLite bag parts.

LiDAR frames are paired with recorded LIO poses. RGB-D remains source data until
measured extrinsics and time offset pass calibration gates.
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
import zstandard

from mapping.pc.go2_mapping_appearance.identity import measured_surface_identity
from mapping.pc.go2_mapping_fusion import (
    FilteredFrame,
    FrameFilter,
    LidarObservationFrame,
    MeasuredSurfaceBuilder,
    MeasuredVoxelMap,
    VoxelState,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import is_sha256, load_reconstruction_config
from mapping.pc.go2_mapping_core.pose_graph import PoseGraph, PoseGraphSolution
from .sensor_frame import (
    IDENTITY_SENSOR_FRAME_V1,
    bind_pose_generation,
    normalize_sensor_frame_transform_id,
    transform_points,
)


RAW_LIDAR_WITH_RECORDED_LIO = "RAW_LIDAR_WITH_RECORDED_LIO"
RECORDED_LIO_REGISTERED_CLOUD = "RECORDED_LIO_REGISTERED_CLOUD"
RECORDED_LIO_SESSION_ODOM_CLOUD = "RECORDED_LIO_SESSION_ODOM_CLOUD"
_RECORDED_GLOBAL_CLOUD_TOPIC = "/go2/mapping/map_cloud"
_RECORDED_GLOBAL_CLOUD_FRAME = "map"
_RECORDED_SESSION_ODOM_CLOUD_TOPIC = "/lio_sam_ros2/mapping/cloud_registered"
_RECORDED_SESSION_ODOM_CLOUD_FRAME = "odom"
_RECORDED_LIO_CLOUD_SOURCE_MODES = frozenset(
    (RECORDED_LIO_REGISTERED_CLOUD, RECORDED_LIO_SESSION_ODOM_CLOUD)
)


def _batch_world_and_voxels(
    points: np.ndarray,
    poses: np.ndarray,
    voxel_m: float,
    *,
    use_cuda: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a point batch and compute voxel keys on CPU or CUDA."""

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    matrices = np.asarray(poses, dtype=np.float32).reshape(-1, 4, 4)
    if len(values) != len(matrices):
        raise ValueError("points and poses must have the same row count")
    if voxel_m <= 0 or not np.isfinite(voxel_m):
        raise ValueError("voxel_m must be finite and positive")
    if use_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda")
                points_t = torch.as_tensor(np.array(values, copy=True), device=device)
                poses_t = torch.as_tensor(np.array(matrices, copy=True), device=device)
                world_t = torch.bmm(points_t.unsqueeze(1), poses_t[:, :3, :3].transpose(1, 2)).squeeze(1)
                world_t = world_t + poses_t[:, :3, 3]
                indices_t = torch.floor(world_t / float(voxel_m)).to(torch.int64)
                return world_t.cpu().numpy().astype(np.float32), indices_t.cpu().numpy()
        except (ImportError, RuntimeError):
            pass
    world = np.einsum("nij,nj->ni", matrices[:, :3, :3], values) + matrices[:, :3, 3]
    return world.astype(np.float32, copy=False), np.floor(world / float(voxel_m)).astype(np.int64)


def _recorded_cloud_frame(source_mode: str) -> str | None:
    if source_mode == RECORDED_LIO_REGISTERED_CLOUD:
        return _RECORDED_GLOBAL_CLOUD_FRAME
    if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD:
        return _RECORDED_SESSION_ODOM_CLOUD_FRAME
    return None


def _coordinate_scope(source_mode: str) -> str | None:
    if source_mode == RECORDED_LIO_REGISTERED_CLOUD:
        return "GLOBAL_MAP"
    if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD:
        return "SESSION_ODOM_LOCAL"
    return None


class CdrDecodeError(ValueError):
    pass


class _CdrReader:
    """Small little-endian CDR reader for the standard ROS messages we ingest."""

    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4 or payload[:2] not in (b"\x00\x01", b"\x00\x03"):
            raise CdrDecodeError("only little-endian CDR payloads are supported")
        self._payload = memoryview(payload)[4:]
        self._offset = 0

    def _align(self, alignment: int) -> None:
        self._offset += (-self._offset) % alignment
        if self._offset > len(self._payload):
            raise CdrDecodeError("CDR alignment exceeds payload")

    def _take(self, count: int, alignment: int = 1) -> memoryview:
        self._align(alignment)
        end = self._offset + count
        if end > len(self._payload):
            raise CdrDecodeError("truncated CDR payload")
        value = self._payload[self._offset : end]
        self._offset = end
        return value

    def u8(self) -> int:
        return int(self._take(1)[0])

    def boolean(self) -> bool:
        return bool(self.u8())

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4, 4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4, 4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._take(8, 8))[0]

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        raw = bytes(self._take(length))
        if raw[-1:] != b"\0":
            raise CdrDecodeError("CDR string is missing its terminator")
        try:
            return raw[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CdrDecodeError("CDR string is not UTF-8") from exc

    def bytes(self, count: int) -> bytes:
        return bytes(self._take(count))


def _read_header_with_frame(reader: _CdrReader) -> Tuple[int, str]:
    seconds = reader.i32()
    nanoseconds = reader.u32()
    frame_id = reader.string()
    if not frame_id:
        raise CdrDecodeError("message header frame_id is required")
    return seconds * 1_000_000_000 + nanoseconds, frame_id


def _read_header(reader: _CdrReader) -> int:
    timestamp_ns, _ = _read_header_with_frame(reader)
    return timestamp_ns


def _decode_pose(payload: bytes) -> Tuple[int, np.ndarray]:
    reader = _CdrReader(payload)
    timestamp_ns = _read_header(reader)
    position = (reader.f64(), reader.f64(), reader.f64())
    x, y, z, w = (reader.f64(), reader.f64(), reader.f64(), reader.f64())
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 0.99 or norm > 1.01:
        raise CdrDecodeError("pose quaternion is not normalized")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = position
    return timestamp_ns, matrix


def _decode_pointcloud_with_frame(payload: bytes) -> Tuple[int, str, np.ndarray]:
    reader = _CdrReader(payload)
    timestamp_ns, frame_id = _read_header_with_frame(reader)
    height = reader.u32()
    width = reader.u32()
    field_count = reader.u32()
    if height == 0 or width == 0 or field_count > 128:
        raise CdrDecodeError("invalid PointCloud2 dimensions")
    fields = {}
    for _ in range(field_count):
        name = reader.string()
        offset = reader.u32()
        datatype = reader.u8()
        count = reader.u32()
        fields[name] = (offset, datatype, count)
    is_bigendian = reader.boolean()
    point_step = reader.u32()
    row_step = reader.u32()
    raw = reader.bytes(reader.u32())
    reader.boolean()  # is_dense; finite filtering below is authoritative.
    if point_step <= 0 or row_step < point_step * width or len(raw) < row_step * height:
        raise CdrDecodeError("PointCloud2 payload layout is invalid")
    required = [fields.get(name) for name in ("x", "y", "z")]
    if any(field is None for field in required):
        raise CdrDecodeError("PointCloud2 has no XYZ fields")
    if any(field[1:] != (7, 1) for field in required):
        raise CdrDecodeError("only float32 XYZ PointCloud2 fields are supported")
    if any(field[0] + 4 > point_step for field in required):
        raise CdrDecodeError("PointCloud2 XYZ field lies outside point_step")
    dtype = ">f4" if is_bigendian else "<f4"
    columns: List[np.ndarray] = []
    for offset, _, _ in required:
        rows = [
            np.ndarray((width,), dtype=dtype, buffer=raw, offset=row * row_step + offset, strides=(point_step,))
            for row in range(height)
        ]
        columns.append(np.concatenate(rows))
    points = np.column_stack(columns)
    return timestamp_ns, frame_id, points[np.isfinite(points).all(axis=1)]


def _decode_pointcloud(payload: bytes) -> Tuple[int, np.ndarray]:
    timestamp_ns, _, points = _decode_pointcloud_with_frame(payload)
    return timestamp_ns, points


@dataclass(frozen=True)
class IngestResult:
    duplicate: bool
    integrated_frames: int
    integrated_points: int
    skipped_pose_frames: int
    decode_errors: int

    def as_dict(self) -> dict:
        return {
            "duplicate": self.duplicate,
            "integrated_frames": self.integrated_frames,
            "integrated_points": self.integrated_points,
            "skipped_pose_frames": self.skipped_pose_frames,
            "decode_errors": self.decode_errors,
        }


@dataclass(frozen=True)
class _RawFrame:
    node_id: str
    artifact_sha256: str
    source_artifact_path: Path
    cloud_index: int
    timestamp_ns: int
    initial_pose_map_from_lidar: np.ndarray
    source_kind: str = "registered"

    def __post_init__(self) -> None:
        pose = np.array(self.initial_pose_map_from_lidar, dtype=np.float64, copy=True)
        pose.setflags(write=False)
        object.__setattr__(self, "source_artifact_path", Path(self.source_artifact_path).resolve())
        object.__setattr__(self, "initial_pose_map_from_lidar", pose)
        if self.source_kind not in {"registered", "native_lidar"}:
            raise ValueError("raw frame source_kind is unsupported")


class HighFidelityMapper:
    """Bounded dense LiDAR submap with an externally supplied quality gate."""

    def __init__(
        self,
        voxel_m: float = 0.015,
        display_point_budget: int = 100_000,
        max_pose_delta_ns: int = 100_000_000,
        lidar_topics: Sequence[str] = ("/go2/lio/lidar", "/livox/lidar"),
        pose_topics: Sequence[str] = ("/go2/lio/pose",),
        source_mode: str = RAW_LIDAR_WITH_RECORDED_LIO,
        quality_snapshot: Callable[[], dict[str, Any]] | None = None,
        pose_generation: str = "live-lio-record-v1",
        rgbd_refiner: Any | None = None,
        sensor_frame_transform_id: str = IDENTITY_SENSOR_FRAME_V1,
        reintegration_work_root: str | Path | None = None,
        prewarm_surface_after_ingest: bool = True,
        defer_pose_graph_optimization: bool = False,
    ) -> None:
        if not math.isfinite(voxel_m) or voxel_m <= 0:
            raise ValueError("voxel_m must be finite and positive")
        if display_point_budget < 100:
            raise ValueError("display_point_budget must be at least 100")
        if max_pose_delta_ns <= 0:
            raise ValueError("max_pose_delta_ns must be positive")
        if not isinstance(prewarm_surface_after_ingest, bool):
            raise ValueError("prewarm_surface_after_ingest must be boolean")
        if not isinstance(defer_pose_graph_optimization, bool):
            raise ValueError("defer_pose_graph_optimization must be boolean")
        if defer_pose_graph_optimization and rgbd_refiner is not None:
            raise ValueError("deferred pose graph optimization does not support incremental RGB-D refinement")
        if source_mode not in {RAW_LIDAR_WITH_RECORDED_LIO, *_RECORDED_LIO_CLOUD_SOURCE_MODES}:
            raise ValueError(f"unknown mapping source mode: {source_mode}")
        self.voxel_m = voxel_m
        self.display_point_budget = display_point_budget
        self.max_pose_delta_ns = max_pose_delta_ns
        self._mapping_source_mode = source_mode
        self.lidar_topics = (
            frozenset((_RECORDED_GLOBAL_CLOUD_TOPIC,))
            if source_mode == RECORDED_LIO_REGISTERED_CLOUD
            else frozenset((_RECORDED_SESSION_ODOM_CLOUD_TOPIC, "/go2/lio/lidar"))
            if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD
            else frozenset(lidar_topics)
        )
        self.pose_topics = (
            frozenset(pose_topics)
            if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD
            else frozenset()
            if source_mode == RECORDED_LIO_REGISTERED_CLOUD
            else frozenset(pose_topics)
        )
        self._quality_snapshot = quality_snapshot
        self._rgbd_refiner = rgbd_refiner
        self._prewarm_surface_after_ingest = prewarm_surface_after_ingest
        self._defer_pose_graph_optimization = defer_pose_graph_optimization
        self._deferred_batch_pending = False
        self._sensor_frame_transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)
        self._reintegration_work_root = (
            None if reintegration_work_root is None else Path(reintegration_work_root).resolve(strict=False)
        )
        self._pose_generation = bind_pose_generation(pose_generation, self._sensor_frame_transform_id)
        reconstruction = load_reconstruction_config()
        self._frame_filter = FrameFilter(reconstruction.fusion.filtering)
        self._measured_map = MeasuredVoxelMap(reconstruction.fusion, voxel_size_m=voxel_m)
        self._surface_builder = MeasuredSurfaceBuilder(reconstruction.fusion.surfacing)
        self._voxels: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._artifact_digests = set()
        self._artifacts = 0
        self._frames = 0
        self._source_points = 0
        self._skipped_pose_frames = 0
        self._decode_errors = 0
        self._lock = threading.RLock()
        self._pose_graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
        self._raw_frames: list[_RawFrame] = []
        self._pose_history: list[Tuple[int, np.ndarray]] = []
        self._loop_clouds: dict[str, np.ndarray] = {}
        self._filtered_indices: dict[str, np.ndarray] = {}
        self._last_pose_graph_solution: PoseGraphSolution | None = None
        self._pose_graph_optimization = "NOT_RUN"
        self._accepted_loop_closures = 0
        self._last_pose_graph_error: str | None = None
        self._last_rgbd_error: str | None = None
        self._last_surface_error: str | None = None
        self._reconstruction_phase = "IDLE"
        self._surface_cache_lock = threading.Lock()
        self._surface_cache_key: tuple[object, ...] | None = None
        self._surface_cache_snapshot: dict | None = None
        self._registered_observed_points: list[np.ndarray] = []
        self._registered_observed_ids: list[str] = []
        self._recorded_global_cloud_frame_ids: set[str] = set()
        self._recorded_cloud_frame_contract_violations: set[str] = set()

    def ingest_sqlite_bag(self, path: Path, *, artifact_sha256: str | None = None, source_artifact_path: Path | None = None) -> dict:
        path = Path(path)
        digest = self._sha256(path) if artifact_sha256 is None else str(artifact_sha256).lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        source_path = path if source_artifact_path is None else Path(source_artifact_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"published source artifact is unavailable: {source_path}")
        if self._sha256(source_path) != digest:
            raise ValueError("published source artifact SHA-256 does not match artifact_sha256")
        with self._lock:
            if digest in self._artifact_digests:
                return IngestResult(True, 0, 0, 0, 0).as_dict()
            self._reconstruction_phase = "DECODING_ARTIFACT"
        poses: List[Tuple[int, np.ndarray]] = []
        clouds: List[Tuple[int, str, np.ndarray, str]] = []
        decode_errors = 0
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
            topics = {
                row[0]: (row[1], row[2])
                for row in database.execute("SELECT id, name, type FROM topics")
            }
            for topic_id, timestamp, payload in database.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp, id"
            ):
                topic = topics.get(topic_id)
                if topic is None:
                    continue
                name, type_name = topic
                try:
                    if name in self.pose_topics and type_name == "geometry_msgs/msg/PoseWithCovarianceStamped":
                        _, pose = _decode_pose(payload)
                        # The MID-360 and LIO headers can use different clock
                        # domains. rosbag's record timestamp is the shared
                        # Recorder arrival time is the shared clock domain for
                        # this capture. The sensor header clocks are allowed to
                        # differ, so do not pair native streams by raw header
                        # timestamp without a measured clock fit.
                        poses.append((int(timestamp), pose))
                    elif name in self.lidar_topics and type_name == "sensor_msgs/msg/PointCloud2":
                        header_timestamp_ns, frame_id, points = _decode_pointcloud_with_frame(payload)
                        expected_frame = _recorded_cloud_frame(self._mapping_source_mode)
                        native_lidar = self._mapping_source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD and name == "/go2/lio/lidar"
                        if expected_frame is not None and frame_id != expected_frame and not native_lidar:
                            self._recorded_cloud_frame_contract_violations.add(frame_id)
                        if self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO or native_lidar:
                            points = transform_points(points, self._sensor_frame_transform_id)
                        clouds.append((int(timestamp), frame_id, points, "native_lidar" if native_lidar else "registered"))
                except (CdrDecodeError, ValueError, struct.error):
                    if (
                        name in self.lidar_topics
                        and type_name == "sensor_msgs/msg/PointCloud2"
                        and self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES
                    ):
                        self._recorded_cloud_frame_contract_violations.add("<unreadable-or-empty>")
                    decode_errors += 1
        poses.sort(key=lambda value: value[0])
        if poses:
            known = {timestamp for timestamp, _ in self._pose_history}
            self._pose_history.extend(value for value in poses if value[0] not in known)
            self._pose_history.sort(key=lambda value: value[0])
        # Native Go2 LiDAR is the authoritative raw geometry for local room
        # scans when present. The registered cloud is retained as a fallback
        # for older captures, but mixing both representations would double
        # count the same returns and create the observed fragmented wedge.
        if self._mapping_source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD and any(item[3] == "native_lidar" for item in clouds):
            clouds = [item for item in clouds if item[3] == "native_lidar"]
        pose_candidates = self._pose_history if self._pose_history else poses
        pose_times = [value[0] for value in pose_candidates]
        integrated_frames = 0
        integrated_points = 0
        skipped_pose_frames = 0
        new_frames: list[_RawFrame] = []
        new_frame_points: dict[str, np.ndarray] = {}
        for cloud_index, (timestamp_ns, frame_id, points, source_kind) in enumerate(clouds):
            pose = (
                np.eye(4, dtype=np.float64)
                if self._mapping_source_mode == RECORDED_LIO_REGISTERED_CLOUD or (self._mapping_source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD and source_kind == "registered")
                else self._nearest_pose(timestamp_ns, pose_times, pose_candidates)
            )
            if pose is None:
                skipped_pose_frames += 1
                continue
            node_id = f"{digest[:16]}-{timestamp_ns:020d}-{cloud_index:06d}"
            if node_id in self._loop_clouds:
                continue
            if self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO:
                self._pose_graph.add_keyframe(node_id, pose, "LIDAR_MEASURED_GEOMETRY", f"{digest}:{timestamp_ns}")
            previous = new_frames[-1] if new_frames else (self._raw_frames[-1] if self._raw_frames else None)
            if previous is not None and self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO:
                relative = np.linalg.inv(previous.initial_pose_map_from_lidar) @ pose
                self._pose_graph.add_lio_constraint(
                    previous.node_id,
                    node_id,
                    relative,
                    translation_sigma_m=0.03,
                    rotation_sigma_deg=0.5,
                )
            raw_frame = _RawFrame(
                node_id,
                digest,
                source_path,
                cloud_index,
                timestamp_ns,
                np.array(pose, copy=True),
                source_kind,
            )
            new_frames.append(raw_frame)
            new_frame_points[node_id] = points
            if self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES:
                self._recorded_global_cloud_frame_ids.add(expected_frame if source_kind == "native_lidar" else frame_id)
                point_keys = np.floor(points / self.voxel_m).astype(np.int64)
                _, first_indices = np.unique(point_keys, axis=0, return_index=True)
                first_indices.sort()
                self._registered_observed_points.extend(points[first_indices])
                self._registered_observed_ids.extend(
                    f"{node_id}:{int(index)}" for index in first_indices
                )
            integrated_frames += 1
            integrated_points += len(points)
        if not new_frames:
            # Recorder rotation can produce a startup or boundary part before
            # its first pose arrives. Keep that immutable artifact accounted
            # for, but never ask an empty pose graph to optimize.
            with self._lock:
                self._artifact_digests.add(digest)
                self._artifacts += 1
                self._skipped_pose_frames += skipped_pose_frames
                self._decode_errors += decode_errors
                self._deferred_batch_pending = self._defer_pose_graph_optimization and bool(
                    self._recorded_cloud_frame_contract_violations
                )
                self._reconstruction_phase = (
                    "DEFERRED_BATCH_PENDING" if self._deferred_batch_pending else "IDLE"
                )
            return IngestResult(False, 0, 0, skipped_pose_frames, decode_errors).as_dict()
        self._raw_frames.extend(new_frames)
        for frame in new_frames:
            self._loop_clouds[frame.node_id] = self._sparse_cloud(new_frame_points[frame.node_id])
        if self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES:
            with self._lock:
                self._artifact_digests.add(digest)
                self._artifacts += 1
                self._frames += integrated_frames
                self._source_points += integrated_points
                self._skipped_pose_frames += skipped_pose_frames
                self._decode_errors += decode_errors
                self._deferred_batch_pending = self._defer_pose_graph_optimization
                self._reconstruction_phase = (
                    "DEFERRED_BATCH_PENDING" if self._defer_pose_graph_optimization else "PUBLISHING_PREVIEW_AND_MEASURED_EVIDENCE"
                )
            if not self._defer_pose_graph_optimization:
                self._integrate_recorded_lio_batch()
                with self._lock:
                    self._reconstruction_phase = "IDLE"
                if self._prewarm_surface_after_ingest:
                    self._prewarm_measured_surface()
            return IngestResult(
                False,
                integrated_frames,
                integrated_points,
                skipped_pose_frames,
                decode_errors,
            ).as_dict()
        with self._lock:
            self._reconstruction_phase = "VERIFYING_LOOP_CANDIDATES"
        loop_added = self._detect_and_add_loops(new_frames)
        with self._lock:
            self._artifact_digests.add(digest)
            self._artifacts += 1
            self._frames += integrated_frames
            self._source_points += integrated_points
            self._skipped_pose_frames += skipped_pose_frames
            self._decode_errors += decode_errors
        if self._defer_pose_graph_optimization:
            with self._lock:
                self._deferred_batch_pending = True
                self._reconstruction_phase = "DEFERRED_BATCH_PENDING"
            return IngestResult(
                False,
                integrated_frames,
                integrated_points,
                skipped_pose_frames,
                decode_errors,
            ).as_dict()
        if loop_added:
            try:
                with self._lock:
                    self._reconstruction_phase = "OPTIMIZING_AND_REINTEGRATING"
                self._optimize_and_reintegrate()
            except Exception as exc:
                # Raw frames and graph evidence remain durable, but an invalid
                # derived generation must never replace the active map.
                self._last_pose_graph_error = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                self._reconstruction_phase = "PUBLISHING_PREVIEW_AND_MEASURED_EVIDENCE"
            self._publish_unadjusted_solution(new_frames, new_frame_points)
        if not loop_added or self._last_pose_graph_error is None:
            with self._lock:
                self._reconstruction_phase = "VALIDATING_RGBD_AND_TEXTURE"
            self._process_rgbd_refinement(path, digest, new_frames, new_frame_points)
        with self._lock:
            self._reconstruction_phase = "IDLE"
        if self._prewarm_surface_after_ingest:
            self._prewarm_measured_surface()
        return IngestResult(False, integrated_frames, integrated_points, skipped_pose_frames, decode_errors).as_dict()

    def finalize_deferred_batch(self, *, global_optimization: bool = True) -> bool:
        """Finalize one deferred offline batch from recorded or optimized poses."""

        if not self._defer_pose_graph_optimization:
            raise ValueError("mapper was not configured for deferred pose graph optimization")
        if not isinstance(global_optimization, bool):
            raise ValueError("global_optimization must be boolean")
        with self._lock:
            if not self._deferred_batch_pending:
                self._reconstruction_phase = "IDLE"
                return False
            self._reconstruction_phase = "OPTIMIZING_AND_REINTEGRATING"
        try:
            if global_optimization and self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO:
                self._optimize_and_reintegrate()
            else:
                self._integrate_recorded_lio_batch()
        except Exception as exc:
            with self._lock:
                self._last_pose_graph_error = f"{type(exc).__name__}: {exc}"
                self._reconstruction_phase = "IDLE"
            raise
        with self._lock:
            self._deferred_batch_pending = False
            self._reconstruction_phase = "PUBLISHING_PREVIEW_AND_MEASURED_EVIDENCE"
        if self._prewarm_surface_after_ingest:
            self._prewarm_measured_surface()
        with self._lock:
            self._reconstruction_phase = "IDLE"
        return True

    def _integrate_recorded_lio_batch(self) -> None:
        """Fuse a closed batch directly from immutable recorded LIO poses.

        This remains strictly observed-only. It is an explicit scale fallback
        for a provisional preview when global loop closure optimization is not
        yet tractable for the captured keyframe count.
        """

        if self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES:
            observed_frames = sorted(self._recorded_global_cloud_frame_ids)
            expected_frame = _recorded_cloud_frame(self._mapping_source_mode)
            assert expected_frame is not None
            violations = sorted(self._recorded_cloud_frame_contract_violations)
            if observed_frames != [expected_frame] or violations:
                rendered = ", ".join(repr(value) for value in sorted({*observed_frames, *violations})) or "none"
                raise ValueError(
                    "recorded LIO "
                    + ("expected global" if self._mapping_source_mode == RECORDED_LIO_REGISTERED_CLOUD else "local session replay expected")
                    + f" frame '{expected_frame}'; observed {rendered}"
                )
        if not self._raw_frames:
            return
        points_by_node = self._reload_frame_points(self._raw_frames)
        candidate_map = MeasuredVoxelMap(load_reconstruction_config().fusion, voxel_size_m=self.voxel_m)
        all_world: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        batch_points: list[np.ndarray] = []
        batch_poses: list[np.ndarray] = []
        for frame in self._raw_frames:
            points = points_by_node[frame.node_id]
            pose = frame.initial_pose_map_from_lidar
            observation = LidarObservationFrame(
                artifact_sha256=frame.artifact_sha256,
                frame_id=_recorded_cloud_frame(self._mapping_source_mode) or "mid360",
                record_timestamp_ns=frame.timestamp_ns,
                sensor_origin_xyz=pose[:3, 3],
                points_xyz=points,
                pose_map_from_lidar=pose,
                pose_generation=self._pose_generation,
            )
            filtered = self._filtered_frame(frame, observation)
            candidate_map.integrate(observation, filtered)
            batch_points.append(points.astype(np.float32, copy=False))
            batch_poses.append(np.repeat(pose[None, :, :], len(points), axis=0))
        if batch_points:
            world, indices = _batch_world_and_voxels(
                np.concatenate(batch_points, axis=0),
                np.concatenate(batch_poses, axis=0),
                self.voxel_m,
                use_cuda=os.environ.get("GO2_MAPPING_USE_CUDA", "0") == "1",
            )
            all_indices.append(indices)
            all_world.append(world)
        if all_indices:
            indices = np.concatenate(all_indices, axis=0)
            world = np.concatenate(all_world, axis=0)
            _, reversed_positions = np.unique(indices[::-1], axis=0, return_index=True)
            latest_positions = len(indices) - 1 - reversed_positions
            candidate_voxels = {
                tuple(index): point
                for index, point in zip(indices[latest_positions], world[latest_positions])
            }
        else:
            candidate_voxels = {}
        pose_payload = {
            "mode": self._mapping_source_mode,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
            "poses": {
                frame.node_id: {
                    "pose": frame.initial_pose_map_from_lidar.reshape(-1).tolist(),
                    "source_kind": frame.source_kind,
                }
                for frame in self._raw_frames
            },
        }
        pose_generation = hashlib.sha256(
            json.dumps(pose_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()
        readonly_poses = {}
        for frame in self._raw_frames:
            pose = np.array(frame.initial_pose_map_from_lidar, copy=True)
            pose.setflags(write=False)
            readonly_poses[frame.node_id] = pose
        solution = PoseGraphSolution(
            poses_map_from_keyframe=MappingProxyType(readonly_poses),
            pose_generation=pose_generation,
            initial_cost=0.0,
            final_cost=0.0,
            accepted_loop_closures=0,
            iterations=0,
        )
        with self._lock:
            self._measured_map = candidate_map
            self._voxels = candidate_voxels
            self._last_pose_graph_solution = solution
            self._pose_graph_optimization = (
                self._mapping_source_mode
                if self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES
                else "RECORDED_LIO_POSES_ONLY"
            )
            self._pose_generation = bind_pose_generation(pose_generation, self._sensor_frame_transform_id)
            self._accepted_loop_closures = 0
            self._last_pose_graph_error = None

    def _process_rgbd_refinement(
        self,
        path: Path,
        artifact_sha256: str,
        frames: Sequence[_RawFrame],
        points_by_node: dict[str, np.ndarray],
    ) -> None:
        if self._rgbd_refiner is None:
            return
        lidar_frames = []
        optimized_poses = (
            self._last_pose_graph_solution.poses_map_from_keyframe
            if self._last_pose_graph_solution is not None
            else {}
        )
        for frame in frames:
            pose = optimized_poses.get(frame.node_id, frame.initial_pose_map_from_lidar)
            lidar_frames.append({
                "timestamp_ns": frame.timestamp_ns,
                "points_xyz": points_by_node[frame.node_id],
                "pose_map_from_lidar": pose,
            })
        try:
            self._rgbd_refiner.process_sqlite_bag(
                path,
                artifact_sha256=artifact_sha256,
                lidar_frames=lidar_frames,
                measured_generation=self._pose_generation,
            )
        except Exception as exc:
            self._last_rgbd_error = f"{type(exc).__name__}: {exc}"
        else:
            self._last_rgbd_error = None

    def refresh_derived_artifact(
        self,
        path: Path,
        *,
        artifact_sha256: str,
        source_artifact_path: Path | None = None,
    ) -> bool:
        if self._rgbd_refiner is None or not self._rgbd_refiner.needs_processing(
            artifact_sha256,
            self._pose_generation,
        ):
            return False
        frames = [frame for frame in self._raw_frames if frame.artifact_sha256 == artifact_sha256]
        if not frames:
            return False
        points = self._reload_frame_points(frames)
        self._process_rgbd_refinement(Path(path), artifact_sha256, frames, points)
        return self._last_rgbd_error is None

    @staticmethod
    def _sparse_cloud(points: np.ndarray, maximum: int = 800) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if len(points) <= maximum:
            sparse = np.array(points, copy=True)
            sparse.setflags(write=False)
            return sparse
        stride = max(1, math.ceil(len(points) / maximum))
        sparse = np.array(points[::stride], copy=True)
        sparse.setflags(write=False)
        return sparse

    def _detect_and_add_loops(self, new_frames: Sequence[_RawFrame]) -> int:
        added = 0
        attempts = 0
        if not new_frames:
            return 0
        first_new_index = len(self._raw_frames) - len(new_frames)
        for offset, frame in enumerate(new_frames):
            frame_index = first_new_index + offset
            # Bound loop search so a large artifact cannot
            # block publication of the realtime preview. The final frame is
            # always considered, while intermediate frames are sampled.
            if frame_index < 10 or (frame_index % 5 and offset != len(new_frames) - 1):
                continue
            candidates = [
                previous
                for previous in self._raw_frames[: max(0, frame_index - 9)]
                if np.linalg.norm(previous.initial_pose_map_from_lidar[:3, 3] - frame.initial_pose_map_from_lidar[:3, 3]) < 1.5
            ]
            candidates.sort(
                key=lambda previous: float(
                    np.linalg.norm(previous.initial_pose_map_from_lidar[:3, 3] - frame.initial_pose_map_from_lidar[:3, 3])
                )
            )
            for previous in candidates[:3]:
                if attempts >= 3:
                    return added
                attempts += 1
                measurement, fitness, rmse = self._icp_relative(
                    self._loop_clouds[frame.node_id],
                    self._loop_clouds[previous.node_id],
                    np.linalg.inv(previous.initial_pose_map_from_lidar) @ frame.initial_pose_map_from_lidar,
                )
                if fitness < 0.50 or rmse > 0.03:
                    continue
                if self._pose_graph.add_loop_constraint(
                    previous.node_id,
                    frame.node_id,
                    measurement,
                    translation_sigma_m=0.01,
                    rotation_sigma_deg=0.25,
                ):
                    added += 1
                    break
        return added

    @staticmethod
    def _icp_relative(source: np.ndarray, target: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, float, float]:
        if len(source) < 20 or len(target) < 20:
            return initial, 0.0, math.inf
        tree = cKDTree(target)
        transform = np.array(initial, dtype=np.float64, copy=True)
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        for _ in range(15):
            transformed = source @ transform[:3, :3].T + transform[:3, 3]
            distances, indices = tree.query(transformed, k=1, distance_upper_bound=0.15)
            valid = np.isfinite(distances) & (distances < 0.15)
            if np.count_nonzero(valid) < 20:
                break
            delta = HighFidelityMapper._best_fit_transform(transformed[valid], target[indices[valid]])
            transform = delta @ transform
            if np.max(np.abs(delta - np.eye(4))) < 1e-6:
                break
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        distances, _ = tree.query(transformed, k=1, distance_upper_bound=0.15)
        valid = np.isfinite(distances) & (distances < 0.15)
        if not np.any(valid):
            return transform, 0.0, math.inf
        return transform, float(np.mean(valid)), float(np.sqrt(np.mean(distances[valid] ** 2)))

    @staticmethod
    def _best_fit_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_center - rotation @ source_center
        return transform

    def _integrate_initial_frames(
        self,
        frames: Sequence[_RawFrame],
        points_by_node: dict[str, np.ndarray],
        solution: PoseGraphSolution | None = None,
    ) -> None:
        # Publish the vectorized provisional voxel layer first. Evidence-state
            # accumulation runs at a lower rate without blocking
        # the operator's coarse 3D view.
        for frame in frames:
            points = points_by_node[frame.node_id]
            pose = (
                frame.initial_pose_map_from_lidar
                if solution is None
                else solution.poses_map_from_keyframe[frame.node_id]
            )
            world = points @ pose[:3, :3].T + pose[:3, 3]
            self._upsert_voxels(world)
        for frame in frames:
            points = points_by_node[frame.node_id]
            pose = (
                frame.initial_pose_map_from_lidar
                if solution is None
                else solution.poses_map_from_keyframe[frame.node_id]
            )
            observation = LidarObservationFrame(
                artifact_sha256=frame.artifact_sha256,
                frame_id="mid360",
                record_timestamp_ns=frame.timestamp_ns,
                sensor_origin_xyz=pose[:3, 3],
                points_xyz=points,
                pose_map_from_lidar=pose,
                pose_generation=self._pose_generation,
            )
            filtered = self._filtered_frame(frame, observation)
            self._measured_map.integrate(observation, filtered)

    def _publish_unadjusted_solution(
        self,
        frames: Sequence[_RawFrame],
        points_by_node: dict[str, np.ndarray],
    ) -> None:
        solution = self._pose_graph.optimize()
        for frame in self._raw_frames:
            if not np.allclose(
                solution.poses_map_from_keyframe[frame.node_id],
                frame.initial_pose_map_from_lidar,
                rtol=0.0,
                atol=1e-9,
            ):
                self._optimize_and_reintegrate()
                return
        with self._lock:
            self._last_pose_graph_solution = solution
            self._pose_graph_optimization = "GLOBAL_CONSTRAINT_OPTIMIZATION"
            self._pose_generation = bind_pose_generation(
                solution.pose_generation,
                self._sensor_frame_transform_id,
            )
            self._accepted_loop_closures = solution.accepted_loop_closures
            self._last_pose_graph_error = None
        self._integrate_initial_frames(frames, points_by_node, solution)

    def _filtered_frame(self, frame: _RawFrame, observation: LidarObservationFrame) -> FilteredFrame:
        accepted_indices = self._filtered_indices.get(frame.node_id)
        if accepted_indices is None:
            filtered = self._frame_filter.classify(observation)
            accepted_indices = np.array(filtered.accepted_indices, dtype=np.int64, copy=True)
            accepted_indices.setflags(write=False)
            self._filtered_indices[frame.node_id] = accepted_indices
            return filtered
        return FilteredFrame(
            source_count=len(observation.points_xyz),
            accepted_indices=accepted_indices,
            rejected=(),
            artifact_sha256=observation.artifact_sha256,
            frame_id=observation.frame_id,
            record_timestamp_ns=observation.record_timestamp_ns,
        )

    def _optimize_and_reintegrate(self) -> None:
        solution = self._pose_graph.optimize()
        points_by_node = self._reload_frame_points(self._raw_frames)
        candidate_map = MeasuredVoxelMap(load_reconstruction_config().fusion, voxel_size_m=self.voxel_m)
        candidate_voxels: Dict[Tuple[int, int, int], np.ndarray] = {}
        for frame in self._raw_frames:
            points = points_by_node[frame.node_id]
            pose = solution.poses_map_from_keyframe[frame.node_id]
            observation = LidarObservationFrame(
                artifact_sha256=frame.artifact_sha256,
                frame_id="mid360",
                record_timestamp_ns=frame.timestamp_ns,
                sensor_origin_xyz=pose[:3, 3],
                points_xyz=points,
                pose_map_from_lidar=pose,
                pose_generation=bind_pose_generation(
                    solution.pose_generation,
                    self._sensor_frame_transform_id,
                ),
            )
            filtered = self._filtered_frame(frame, observation)
            candidate_map.integrate(observation, filtered)
            world = points @ pose[:3, :3].T + pose[:3, 3]
            indices = np.floor(world / self.voxel_m).astype(np.int64)
            _, reversed_positions = np.unique(indices[::-1], axis=0, return_index=True)
            latest_positions = len(indices) - 1 - reversed_positions
            candidate_voxels.update({
                tuple(index): point
                for index, point in zip(indices[latest_positions], world[latest_positions].astype(np.float32))
            })
        with self._lock:
            self._measured_map = candidate_map
            self._voxels = candidate_voxels
            self._last_pose_graph_solution = solution
            self._pose_generation = bind_pose_generation(
                solution.pose_generation,
                self._sensor_frame_transform_id,
            )
            self._accepted_loop_closures = solution.accepted_loop_closures
            self._last_pose_graph_error = None

    def _reload_frame_points(self, frames: Sequence[_RawFrame]) -> dict[str, np.ndarray]:
        grouped: dict[tuple[Path, str], list[_RawFrame]] = {}
        for frame in frames:
            grouped.setdefault((frame.source_artifact_path, frame.artifact_sha256), []).append(frame)
        loaded: dict[str, np.ndarray] = {}
        for (artifact_path, expected_digest), artifact_frames in grouped.items():
            if not artifact_path.is_file():
                raise FileNotFoundError(f"raw artifact required for reintegration is unavailable: {artifact_path}")
            if self._sha256(artifact_path) != expected_digest:
                raise ValueError(f"raw artifact SHA-256 changed before reintegration: {artifact_path}")
            with self._sqlite_artifact_path(
                artifact_path,
                work_root=self._reintegration_work_root,
            ) as sqlite_path:
                clouds = self._read_lidar_clouds(sqlite_path)
            for frame in artifact_frames:
                if frame.cloud_index >= len(clouds):
                    raise ValueError(f"raw artifact no longer contains cloud index {frame.cloud_index}")
                timestamp_ns, points, source_kind = clouds[frame.cloud_index]
                if timestamp_ns != frame.timestamp_ns:
                    raise ValueError("raw artifact cloud timestamp changed before reintegration")
                loaded[frame.node_id] = (
                    transform_points(points, self._sensor_frame_transform_id)
                    if self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO or source_kind == "native_lidar"
                    else points
                )
        if len(loaded) != len(frames):
            raise ValueError("not every raw frame was recovered for reintegration")
        return loaded

    def _read_lidar_clouds(self, path: Path) -> list[tuple[int, np.ndarray, str]]:
        clouds: list[tuple[int, np.ndarray, str]] = []
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
            topics = {
                row[0]: (row[1], row[2])
                for row in database.execute("SELECT id, name, type FROM topics")
            }
            for topic_id, timestamp, payload in database.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp, id"
            ):
                topic = topics.get(topic_id)
                if topic is None:
                    continue
                name, type_name = topic
                if name not in self.lidar_topics or type_name != "sensor_msgs/msg/PointCloud2":
                    continue
                try:
                    header_timestamp_ns, frame_id, points = _decode_pointcloud_with_frame(payload)
                except (CdrDecodeError, ValueError, struct.error) as exc:
                    raise ValueError("raw LiDAR frame cannot be decoded during reintegration") from exc
                native_lidar = self._mapping_source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD and name == "/go2/lio/lidar"
                if (
                    self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES
                    and frame_id != _recorded_cloud_frame(self._mapping_source_mode)
                    and not native_lidar
                ):
                    expected_frame = _recorded_cloud_frame(self._mapping_source_mode)
                    assert expected_frame is not None
                    raise ValueError(
                        "recorded LIO "
                        + ("global replay" if self._mapping_source_mode == RECORDED_LIO_REGISTERED_CLOUD else "local session replay")
                        + f" expected frame '{expected_frame}'; observed {frame_id!r}"
                    )
                if self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO or native_lidar:
                    points = transform_points(points, self._sensor_frame_transform_id)
                clouds.append((int(timestamp), points, "native_lidar" if native_lidar else "registered"))
        if self._mapping_source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD and any(item[2] == "native_lidar" for item in clouds):
            clouds = [item for item in clouds if item[2] == "native_lidar"]
        return clouds

    @staticmethod
    @contextmanager
    def _sqlite_artifact_path(path: Path, *, work_root: Path | None = None):
        if path.suffix != ".zstd":
            yield path
            return
        root = None if work_root is None else Path(work_root)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="go2-reintegration-",
            suffix=".db3",
            dir=None if root is None else str(root),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination, path.open("rb") as source:
                zstandard.ZstdDecompressor().copy_stream(source, destination)
            yield temporary_path
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _upsert_voxels(self, world: np.ndarray) -> None:
        indices = np.floor(world / self.voxel_m).astype(np.int64)
        # Reversing first gives the same last-observation-wins behavior as the
        # former point-by-point update, while reducing Python dict work to one
        # assignment per occupied voxel in this bag part.
        _, reversed_positions = np.unique(indices[::-1], axis=0, return_index=True)
        latest_positions = len(indices) - 1 - reversed_positions
        updates = {
            tuple(index): point
            for index, point in zip(indices[latest_positions], world[latest_positions].astype(np.float32))
        }
        with self._lock:
            self._voxels.update(updates)

    def snapshot(self) -> dict:
        with self._lock:
            voxels = list(self._voxels.values())
            frames = self._frames
            artifacts = self._artifacts
            source_points = self._source_points
            skipped_pose_frames = self._skipped_pose_frames
            decode_errors = self._decode_errors
            artifact_digests = set(self._artifact_digests)
            pose_generation = self._pose_generation
            accepted_loop_closures = self._accepted_loop_closures
            pose_graph_optimization = self._pose_graph_optimization
            last_pose_graph_error = self._last_pose_graph_error
            last_rgbd_error = self._last_rgbd_error
            last_surface_error = self._last_surface_error
            source_artifacts = len({frame.source_artifact_path for frame in self._raw_frames})
            recorded_global_cloud_frame_ids = sorted(self._recorded_global_cloud_frame_ids)
            measured_vertex_ids = tuple(f"{key[0]}:{key[1]}:{key[2]}" for key in self._voxels)
            reconstruction_phase = self._reconstruction_phase
        stride = max(1, math.ceil(len(voxels) / self.display_point_budget))
        quality_evidence = self._quality_snapshot() if self._quality_snapshot is not None else None
        coordinate_scope = _coordinate_scope(self._mapping_source_mode)
        quality = "DENSE_LIDAR_PREVIEW_UNCALIBRATED"
        if frames and quality_evidence is not None:
            quality_evidence = dict(quality_evidence)
            required = set(quality_evidence.get("required_artifact_sha256", ()))
            artifacts_materialized = bool(required) and required.issubset(artifact_digests)
            quality_evidence["map_artifacts_materialized"] = artifacts_materialized
            assessed_level = quality_evidence.get("level")
            if assessed_level in {"GEOMETRY_VALIDATED_CM", "BIM_REGISTERED_VALIDATED"} and not artifacts_materialized:
                quality_evidence["assessment_level"] = assessed_level
                quality_evidence["level"] = "CALIBRATED_DENSE"
                quality_evidence["blocking_reasons"] = list(quality_evidence.get("blocking_reasons", ())) + [
                    "MAP_ARTIFACTS_NOT_FULLY_INGESTED"
                ]
                eligible = dict(quality_evidence.get("eligible_for", {}))
                eligible["centimetre_geometry"] = False
                eligible["bim_coordinate_delivery"] = False
                quality_evidence["eligible_for"] = eligible
            quality = str(quality_evidence.get("dense_quality", quality))
        measurement = self._measured_map.snapshot()
        measured_points = self._measured_map.measured_points()
        measured_stride = max(1, math.ceil(len(measured_points) / self.display_point_budget))
        rgbd_refinement = (
            {
                "state": "RGBD_REFINER_DISABLED",
                "depth_correspondence_count": 0,
                "texture": {"surface_coverage": 0.0, "delivery_authorized": False},
                "texture_authorization_blocker": "RGBD_REFINER_DISABLED",
                "blockers": ["RGBD_REFINER_DISABLED"],
                "depth_anything": {
                    "state": "DEPTH_ANYTHING_DISABLED",
                    "augmentation_reference": None,
                    "blockers": ["RGBD_REFINER_DISABLED"],
                },
            }
            if self._rgbd_refiner is None
            else self._rgbd_refiner.snapshot(measured_vertex_ids)
        )
        return {
            "quality": quality if frames else "AWAITING_VERIFIED_LIDAR_ARTIFACT",
            "points": len(voxels),
            "artifacts": artifacts,
            "frames": frames,
            "source_points": source_points,
            "skipped_pose_frames": skipped_pose_frames,
            "decode_errors": decode_errors,
            "voxel_m": self.voxel_m,
            "preview_points": [point.tolist() for point in voxels[::stride]],
            "measured_preview_points": measured_points[::measured_stride].tolist(),
            "candidate_points": measurement["candidate_points"],
            "measured_points": measurement["measured_points"],
            "unstable_points": measurement["unstable_points"],
            "rejected_points": measurement["rejected_points"],
            "pose_generation": pose_generation,
            "optimized_pose_generation": pose_generation,
            "sensor_frame_transform_id": self._sensor_frame_transform_id,
            "mapping_source_mode": self._mapping_source_mode,
            "recorded_coordinate_frame_ids": recorded_global_cloud_frame_ids,
            "coordinate_scope": coordinate_scope,
            "coordinate_frame": _recorded_cloud_frame(self._mapping_source_mode),
            "sensor_frame_transform_applied": self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO,
            "accepted_loop_closures": accepted_loop_closures,
            "pose_graph_optimization": pose_graph_optimization,
            "resident_full_lidar_frames": 0,
            "raw_source_artifacts": source_artifacts,
            "reconstruction_phase": reconstruction_phase,
            "last_pose_graph_error": last_pose_graph_error,
            "rgbd_refinement": rgbd_refinement,
            "depth_anything": rgbd_refinement["depth_anything"],
            "last_rgbd_error": last_rgbd_error,
            "last_surface_error": last_surface_error,
            "pose_graph": None if self._last_pose_graph_solution is None else {
                "initial_cost": self._last_pose_graph_solution.initial_cost,
                "final_cost": self._last_pose_graph_solution.final_cost,
                "iterations": self._last_pose_graph_solution.iterations,
            },
            "quality_evidence": quality_evidence,
        }

    def measured_surface_snapshot(self) -> dict | None:
        with self._lock:
            measured_map = self._measured_map
            artifact_sha256 = sorted(self._artifact_digests)
            pose_generation = self._pose_generation
            reconstruction_phase = self._reconstruction_phase
            registered_points = np.asarray(self._registered_observed_points, dtype=np.float64).reshape(-1, 3)
            registered_ids = tuple(self._registered_observed_ids)
            recorded_global_cloud_frame_ids = sorted(self._recorded_global_cloud_frame_ids)
            recorded_frame_contract_violations = sorted(self._recorded_cloud_frame_contract_violations)
        cache_key = (
            id(measured_map),
            pose_generation,
            self._sensor_frame_transform_id,
            tuple(artifact_sha256),
            reconstruction_phase,
            len(registered_points),
        )
        with self._surface_cache_lock:
            if reconstruction_phase != "IDLE" and self._surface_cache_snapshot is not None:
                return self._copy_surface_snapshot(self._surface_cache_snapshot)
            if cache_key == self._surface_cache_key and self._surface_cache_snapshot is not None:
                return self._copy_surface_snapshot(self._surface_cache_snapshot)
            evidence = measured_map.evidence(VoxelState.MEASURED)
            if self._mapping_source_mode in _RECORDED_LIO_CLOUD_SOURCE_MODES:
                expected_frame = _recorded_cloud_frame(self._mapping_source_mode)
                assert expected_frame is not None
                if recorded_global_cloud_frame_ids != [expected_frame] or recorded_frame_contract_violations:
                    self._surface_cache_key = cache_key
                    self._surface_cache_snapshot = None
                    return None
                if not len(registered_points):
                    self._surface_cache_key = cache_key
                    self._surface_cache_snapshot = None
                    return None
                surface = self._surface_builder.build_observed_positions(
                    registered_points,
                    source_vertex_ids=registered_ids,
                    source_layer="LIO_REGISTERED_LIDAR_OBSERVED",
                    point_authority="LIO_REGISTERED_LIDAR_OBSERVED",
                )
            elif not evidence:
                self._surface_cache_key = cache_key
                self._surface_cache_snapshot = None
                return None
            else:
                surface = self._surface_builder.build(evidence)
            appearance_blocker = None
            geometry_revision_id = None
            if not artifact_sha256:
                appearance_blocker = "SOURCE_ARTIFACT_IDENTITY_MISSING"
            elif not is_sha256(pose_generation) or pose_generation != pose_generation.lower():
                appearance_blocker = "POSE_GENERATION_NOT_CONTENT_ADDRESSED"
            else:
                identity = measured_surface_identity(
                    positions=surface.positions,
                    normals=surface.normals,
                    triangles=surface.triangles,
                    pose_generation=pose_generation,
                    artifact_sha256=artifact_sha256,
                    fusion_config_sha256=load_reconstruction_config().fusion_sha256,
                )
                geometry_revision_id = identity.revision_id
            snapshot = {
                "pose_generation": pose_generation,
                "reconstruction_phase": reconstruction_phase,
                "artifact_sha256": artifact_sha256,
                "voxel_size_m": self.voxel_m,
                "measured_surface_geometry_revision_id": geometry_revision_id,
                "appearance_blocker": appearance_blocker,
                **surface.bounded_snapshot(),
                "sensor_frame_transform_id": self._sensor_frame_transform_id,
                "mapping_source_mode": self._mapping_source_mode,
                "recorded_coordinate_frame_ids": recorded_global_cloud_frame_ids,
                "coordinate_scope": _coordinate_scope(self._mapping_source_mode),
                "coordinate_frame": _recorded_cloud_frame(self._mapping_source_mode),
                "sensor_frame_transform_applied": self._mapping_source_mode == RAW_LIDAR_WITH_RECORDED_LIO,
                "geometry_authority": False,
                "quality": "PREVIEW_UNCALIBRATED",
            }
            self._surface_cache_key = cache_key
            self._surface_cache_snapshot = snapshot
            return self._copy_surface_snapshot(snapshot)

    def _prewarm_measured_surface(self) -> None:
        try:
            self.measured_surface_snapshot()
        except Exception as exc:
            with self._lock:
                self._last_surface_error = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                self._last_surface_error = None

    @staticmethod
    def _copy_surface_snapshot(snapshot: dict) -> dict:
        copied = dict(snapshot)
        for key in ("artifact_sha256", "source_vertex_ids", "positions", "normals", "triangles"):
            copied[key] = list(snapshot[key])
        return copied

    def _nearest_pose(
        self,
        timestamp_ns: int,
        pose_times: Sequence[int],
        poses: Sequence[Tuple[int, np.ndarray]],
    ) -> np.ndarray | None:
        if not poses:
            return None
        index = bisect.bisect_left(pose_times, timestamp_ns)
        candidates = []
        if index < len(poses):
            candidates.append(poses[index])
        if index:
            candidates.append(poses[index - 1])
        closest = min(candidates, key=lambda value: abs(value[0] - timestamp_ns))
        return closest[1] if abs(closest[0] - timestamp_ns) <= self.max_pose_delta_ns else None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
