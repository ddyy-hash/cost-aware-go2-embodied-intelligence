"""Calibrated-only RGB-D validation and texture evidence from verified bags."""

from __future__ import annotations

import bisect
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import (
    CalibrationManifestV2,
    ConfigError,
    canonical_sha256,
    is_sha256,
    parse_calibration_manifest,
)

from .decode import RgbdDecodeError, decode_camera_info, decode_image
from .depth_validation import DepthValidator
from .depth_anything_worker import DepthAnythingWorkItem, DepthAnythingWorker
from .texture import ColorAccumulator, ColorObservation, VertexColor, srgb_to_linear


@dataclass(frozen=True)
class AppearanceEvidence:
    ready: bool
    vertex_colors: tuple[VertexColor, ...]
    source_frame_sha256: tuple[str, ...]
    minimum_independent_accepted_views: int
    physical_depth_validation_revision: str | None
    calibration_id: str | None
    calibration_sha256: str | None
    measured_generation: str | None
    blockers: tuple[str, ...]


class ArtifactRgbdRefiner:
    """Builds validation/appearance evidence without modifying LiDAR geometry."""

    def __init__(
        self,
        calibration_manifest_path: str | Path | None,
        *,
        depth_scale_m: float | None = None,
        voxel_size_m: float = 0.015,
        max_pair_delta_ns: int = 100_000_000,
        color_topics: Sequence[str] = ("/camera/color/image_raw",),
        depth_topics: Sequence[str] = ("/camera/depth/image_rect_raw",),
        color_info_topics: Sequence[str] = ("/camera/color/camera_info",),
        depth_info_topics: Sequence[str] = ("/camera/depth/camera_info",),
        depth_anything_worker: DepthAnythingWorker | None = None,
    ) -> None:
        if depth_scale_m is not None and (not math.isfinite(depth_scale_m) or not 0.0 < depth_scale_m <= 0.1):
            raise ValueError("depth_scale_m must be in (0, 0.1]")
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if max_pair_delta_ns <= 0:
            raise ValueError("max_pair_delta_ns must be positive")
        self.calibration_manifest_path = None if calibration_manifest_path is None else Path(calibration_manifest_path)
        self.depth_scale_m = depth_scale_m
        self.voxel_size_m = float(voxel_size_m)
        self.max_pair_delta_ns = int(max_pair_delta_ns)
        self.color_topics = frozenset(color_topics)
        self.depth_topics = frozenset(depth_topics)
        self.color_info_topics = frozenset(color_info_topics)
        self.depth_info_topics = frozenset(depth_info_topics)
        self._depth_anything_worker = depth_anything_worker
        self._validator = DepthValidator()
        self._calibration_signature: tuple[int, int] | None = None
        self._calibration: CalibrationManifestV2 | None = None
        self._calibration_sha256: str | None = None
        self._color_accumulator: ColorAccumulator | None = None
        self._blockers: list[str] = []
        self._depth_frames = 0
        self._accepted_depth_frames = 0
        self._rejected_depth_frames = 0
        self._correspondences = 0
        self._tile_medians: list[tuple[float, int]] = []
        self._tile_p95: list[float] = []
        self._accepted_depth_reports: list[dict[str, Any]] = []
        self._accepted_color_observations = 0
        self._rejected_color_observations = 0
        self._observed_vertices: set[str] = set()
        self._processed_artifacts: set[tuple[str, str, str]] = set()
        self._active_identity: tuple[str, str] | None = None
        self._lock = threading.RLock()

    def process_sqlite_bag(
        self,
        path: Path,
        *,
        artifact_sha256: str,
        lidar_frames: Sequence[Mapping[str, Any]],
        measured_generation: str,
    ) -> None:
        calibration = self._load_calibration()
        if calibration is not None and self._calibration_sha256 is not None:
            self._activate_identity(self._calibration_sha256, measured_generation, calibration.calibration_id)
        processing_key = (
            artifact_sha256,
            "" if self._calibration_sha256 is None else self._calibration_sha256,
            measured_generation,
        )
        if calibration is not None and processing_key in self._processed_artifacts:
            return
        colors: list[tuple[int, Any]] = []
        depths: list[tuple[int, Any]] = []
        color_info = []
        depth_info = []
        decode_blockers: list[str] = []
        with sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True) as database:
            topics = {row[0]: (row[1], row[2]) for row in database.execute("SELECT id, name, type FROM topics")}
            for topic_id, timestamp, payload in database.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp, id"
            ):
                topic = topics.get(topic_id)
                if topic is None:
                    continue
                name, type_name = topic
                try:
                    if type_name == "sensor_msgs/msg/Image" and name in self.color_topics:
                        colors.append((int(timestamp), decode_image(payload)))
                    elif type_name == "sensor_msgs/msg/Image" and name in self.depth_topics:
                        depths.append((int(timestamp), decode_image(payload)))
                    elif type_name == "sensor_msgs/msg/CameraInfo" and name in self.color_info_topics:
                        color_info.append(decode_camera_info(payload))
                    elif type_name == "sensor_msgs/msg/CameraInfo" and name in self.depth_info_topics:
                        depth_info.append(decode_camera_info(payload))
                except (RgbdDecodeError, ValueError) as exc:
                    decode_blockers.append(f"RGBD_DECODE_FAILED:{name}:{type(exc).__name__}")

        with self._lock:
            self._blockers.extend(decode_blockers)
            self._depth_frames += len(depths)
        for _, record in depths:
            try:
                depth_m = self._depth_metres(record)
                internal = self._validator.camera_internal(depth_m, record.header_timestamp_ns, aligned_to_color=False)
                with self._lock:
                    self._blockers.extend(
                        blocker for blocker in internal.blockers
                        if not (calibration is not None and blocker == "DEPTH_NOT_ALIGNED_TO_COLOR")
                    )
            except ValueError as exc:
                with self._lock:
                    self._blockers.append(str(exc))

        if calibration is None:
            return
        if not self._camera_info_matches(calibration, color_info, depth_info):
            with self._lock:
                self._blockers.append("CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING")
                self._processed_artifacts.add(processing_key)
            return
        if not lidar_frames or not depths:
            with self._lock:
                self._blockers.append("RGBD_OR_LIDAR_FRAME_MISSING")
                self._processed_artifacts.add(processing_key)
            return

        lidar_by_time = sorted((int(frame["timestamp_ns"]), frame) for frame in lidar_frames)
        lidar_times = [item[0] for item in lidar_by_time]
        color_by_time = sorted(colors, key=lambda item: item[0])
        color_times = [item[0] for item in color_by_time]
        depth_from_lidar = self._matrix(calibration.frame_chain.t_d_c.row_major) @ self._matrix(
            calibration.frame_chain.t_c_l.row_major
        )
        color_from_lidar = self._matrix(calibration.frame_chain.t_c_l.row_major)
        depth_camera = np.asarray(calibration.frame_chain.depth_camera_info.k, dtype=np.float64).reshape(3, 3)
        color_camera = np.asarray(calibration.frame_chain.color_camera_info.k, dtype=np.float64).reshape(3, 3)

        for record_time, depth_record in depths:
            lidar = self._nearest(record_time, lidar_times, lidar_by_time)
            if lidar is None:
                with self._lock:
                    self._rejected_depth_frames += 1
                    self._blockers.append("DEPTH_LIDAR_TIME_ASSOCIATION_FAILED")
                continue
            try:
                depth_m = self._depth_metres(depth_record)
                report = self._validator.cross_sensor(
                    np.asarray(lidar["points_xyz"], dtype=np.float64),
                    depth_m,
                    depth_camera,
                    depth_from_lidar,
                    calibration_id=calibration.calibration_id,
                    calibration_sha256=str(self._calibration_sha256),
                    measured_generation=measured_generation,
                    frame_sha256=depth_record.message_sha256,
                    timestamp_ns=record_time,
                    color_camera_matrix=color_camera,
                    color_from_lidar=color_from_lidar,
                )
            except ValueError as exc:
                with self._lock:
                    self._rejected_depth_frames += 1
                    self._blockers.append(f"DEPTH_VALIDATION_FAILED:{exc}")
                continue
            with self._lock:
                self._correspondences += report.correspondence_count
                report_core = self._depth_report_core(
                    report,
                    artifact_sha256=artifact_sha256,
                    depth_frame_sha256=depth_record.message_sha256,
                    record_timestamp_ns=record_time,
                )
                for tile in report.tiles:
                    self._tile_medians.append((tile.median_absolute_residual_m, tile.correspondence_count))
                    self._tile_p95.append(tile.p95_absolute_residual_m)
                if report.accepted:
                    self._accepted_depth_frames += 1
                    self._accepted_depth_reports.append(report_core)
                else:
                    self._rejected_depth_frames += 1
                    self._blockers.extend(report.blockers)
            color = self._nearest(record_time, color_times, color_by_time)
            if report.accepted and color is not None:
                self._queue_depth_anything_from_report(
                    report,
                    report_core,
                    artifact_sha256=artifact_sha256,
                    depth_record=depth_record,
                    color_record=color,
                    measured_generation=measured_generation,
                )
            if not report.authorizes_texture:
                continue
            if color is None:
                with self._lock:
                    self._blockers.append("COLOR_DEPTH_TIME_ASSOCIATION_FAILED")
                continue
            self._accumulate_color(
                lidar,
                depth_m,
                color,
                depth_from_lidar,
                color_from_lidar,
                depth_camera,
                color_camera,
            )
        with self._lock:
            self._processed_artifacts.add(processing_key)

    def needs_processing(self, artifact_sha256: str, measured_generation: str) -> bool:
        calibration = self._load_calibration()
        if calibration is None or self._calibration_sha256 is None:
            return False
        self._activate_identity(self._calibration_sha256, measured_generation, calibration.calibration_id)
        return (artifact_sha256, self._calibration_sha256, measured_generation) not in self._processed_artifacts

    def _queue_depth_anything_from_report(
        self,
        report: Any,
        report_core: Mapping[str, Any],
        *,
        artifact_sha256: str,
        depth_record: Any,
        color_record: Any,
        measured_generation: str,
    ) -> None:
        worker = self._depth_anything_worker
        model_revision = None if worker is None else getattr(worker, "model_revision_id", None)
        config_sha256 = None if worker is None else getattr(worker, "assistance_config_sha256", None)
        calibration_sha256 = self._calibration_sha256
        if (
            worker is None
            or not report.accepted
            or not report.correspondences
            or not is_sha256(measured_generation)
            or not is_sha256(artifact_sha256)
            or not is_sha256(calibration_sha256)
            or not is_sha256(model_revision)
            or not is_sha256(config_sha256)
        ):
            return
        rgb = self._rgb_image(color_record)
        if rgb is None:
            return
        from .depth_anything_worker import DepthAnythingWorkItem

        validation_revision = canonical_sha256({
            "schema_version": 1,
            "calibration_sha256": calibration_sha256,
            "measured_generation": measured_generation,
            "reports": [dict(report_core)],
        })
        item = DepthAnythingWorkItem(
            frame_sha256=depth_record.message_sha256,
            source_generation_id=measured_generation,
            source_revision_id=artifact_sha256,
            calibration_sha256=calibration_sha256,
            physical_depth_validation_revision=validation_revision,
            model_revision_id=model_revision,
            assistance_config_sha256=config_sha256,
            rectified_rgb=rgb,
            anchors=tuple(report.correspondences),
        )
        try:
            worker.submit(item)
        except (TypeError, ValueError) as exc:
            with self._lock:
                self._blockers.append(f"DEPTH_ANYTHING_QUEUE_REJECTED:{type(exc).__name__}")

    @staticmethod
    def _rgb_image(record: Any) -> np.ndarray | None:
        rgb = np.asarray(record.image)
        if record.encoding == "bgr8":
            rgb = rgb[..., ::-1]
        elif record.encoding == "mono8":
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        elif record.encoding != "rgb8":
            return None
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            return None
        return np.ascontiguousarray(rgb)

    def _activate_identity(self, calibration_sha256: str, generation: str, calibration_id: str) -> None:
        identity = (calibration_sha256, generation)
        with self._lock:
            if self._active_identity == identity:
                return
            self._active_identity = identity
            self._validator = DepthValidator()
            self._color_accumulator = ColorAccumulator(calibration_id)
            self._blockers = []
            self._depth_frames = 0
            self._accepted_depth_frames = 0
            self._rejected_depth_frames = 0
            self._correspondences = 0
            self._tile_medians = []
            self._tile_p95 = []
            self._accepted_depth_reports = []
            self._accepted_color_observations = 0
            self._rejected_color_observations = 0
            self._observed_vertices = set()

    def snapshot(self, measured_vertex_ids: Sequence[str] = ()) -> dict[str, Any]:
        with self._lock:
            calibration = self._calibration
            accumulator = self._color_accumulator
            weighted = list(self._tile_medians)
            p95_values = list(self._tile_p95)
            blockers = tuple(dict.fromkeys(self._blockers))
            observed_vertices = set(self._observed_vertices)
            result = {
                "state": "CALIBRATED_RGBD_ACTIVE" if calibration is not None else "RGBD_CALIBRATION_BLOCKED",
                "calibration_id": None if calibration is None else calibration.calibration_id,
                "calibration_sha256": self._calibration_sha256,
                "processed_artifacts": len({key[0] for key in self._processed_artifacts}),
                "depth_frames": self._depth_frames,
                "accepted_depth_frames": self._accepted_depth_frames,
                "rejected_depth_frames": self._rejected_depth_frames,
                "depth_correspondence_count": self._correspondences,
                "depth_residual_median_m": self._weighted_median(weighted),
                "depth_residual_p95_m": None if not p95_values else max(p95_values),
                "accepted_color_observations": self._accepted_color_observations,
                "rejected_color_observations": self._rejected_color_observations,
                "blockers": list(blockers),
            }
        vertex_ids = tuple(measured_vertex_ids) if measured_vertex_ids else tuple(sorted(observed_vertices))
        delivery = (
            {"measured_vertex_count": len(vertex_ids), "textured_vertex_count": 0, "surface_coverage": 0.0,
             "delivery_authorized": False, "calibration_id": None}
            if accumulator is None
            else accumulator.delivery_report(vertex_ids)
        )
        result["texture"] = delivery
        result["texture_authorization_blocker"] = None if delivery["delivery_authorized"] else (
            "TEXTURE_COVERAGE_OR_VIEW_DIVERSITY_GATE_FAILED" if calibration is not None else "MEASURED_CALIBRATION_REQUIRED"
        )
        result["depth_anything"] = self.assistance_snapshot()
        return result

    def queue_depth_anything(self, item: DepthAnythingWorkItem) -> bool:
        """Queue only evidence aligned to this refiner's accepted physical identity."""

        worker = self._depth_anything_worker
        if worker is None:
            return False
        with self._lock:
            active_identity = self._active_identity
            accepted_depth_frames = self._accepted_depth_frames
        if (
            active_identity is None
            or active_identity != (item.calibration_sha256, item.source_generation_id)
            or accepted_depth_frames <= 0
            or not item.anchors
        ):
            return False
        return worker.submit(item)

    def assistance_snapshot(self) -> dict[str, Any]:
        worker = self._depth_anything_worker
        if worker is not None:
            return worker.snapshot()
        return {
            "state": "DEPTH_ANYTHING_DISABLED",
            "queue_capacity": 1,
            "pending_frames": 0,
            "submitted_frames": 0,
            "processed_frames": 0,
            "failed_frames": 0,
            "coalesced_frames": 0,
            "evidence_drops": 0,
            "idempotent_replays": 0,
            "identity_invalidations": 0,
            "source_generation_id": None,
            "model_revision_id": None,
            "assistance_config_sha256": None,
            "last_inference_ms": None,
            "fit_accepted": False,
            "assistance_coverage": 0.0,
            "augmentation_reference": None,
            "last_error": None,
            "blockers": ["DEPTH_ANYTHING_PROVIDER_UNAVAILABLE"],
        }

    def field_acceptance_snapshot(self) -> dict[str, Any]:
        """Return copy-safe physical and optional-assistance evidence only."""

        with self._lock:
            reports = copy.deepcopy(self._accepted_depth_reports)
            rejected_depth_frames = self._rejected_depth_frames
        return {
            "accepted_depth_reports": reports,
            "rejected_depth_frames": rejected_depth_frames,
            "depth_anything": self.assistance_snapshot(),
        }

    def appearance_evidence(self, measured_vertex_ids: Sequence[str]) -> AppearanceEvidence:
        vertex_ids = tuple(measured_vertex_ids)
        with self._lock:
            calibration = self._calibration
            calibration_sha256 = self._calibration_sha256
            accumulator = self._color_accumulator
            accepted_depth_frames = self._accepted_depth_frames
            accepted_color_observations = self._accepted_color_observations
            reports = tuple(dict(report) for report in self._accepted_depth_reports)
            recorded_blockers = tuple(dict.fromkeys(self._blockers))
            active_identity = self._active_identity

        blockers: list[str] = []
        if calibration is None or calibration_sha256 is None:
            blockers.append("MEASURED_CALIBRATION_REQUIRED")
        if "VERIFIED_DEPTH_SCALE_REQUIRED_FOR_16UC1" in recorded_blockers:
            blockers.append("VERIFIED_DEPTH_SCALE_REQUIRED_FOR_16UC1")
        if accepted_depth_frames <= 0:
            blockers.append("ACCEPTED_PHYSICAL_DEPTH_REQUIRED")
        if accepted_color_observations <= 0:
            blockers.append("ACCEPTED_COLOR_OBSERVATION_REQUIRED")
        if not vertex_ids:
            blockers.append("MEASURED_SURFACE_VERTEX_IDS_REQUIRED")

        colors = () if accumulator is None else accumulator.vertices(vertex_ids)
        matching = tuple(color for color in colors if color.accepted_count > 0)
        if vertex_ids and not matching:
            blockers.append("NO_MATCHING_COLOR_VERTICES")
        if not reports:
            validation_revision = None
        else:
            validation_revision = canonical_sha256({
                "schema_version": 1,
                "calibration_sha256": calibration_sha256,
                "measured_generation": None if active_identity is None else active_identity[1],
                "reports": sorted(reports, key=canonical_sha256),
            })
        if validation_revision is None:
            blockers.append("PHYSICAL_DEPTH_VALIDATION_REVISION_REQUIRED")

        blockers = list(dict.fromkeys(blockers))
        if blockers:
            return AppearanceEvidence(
                ready=False,
                vertex_colors=(),
                source_frame_sha256=(),
                minimum_independent_accepted_views=0,
                physical_depth_validation_revision=validation_revision,
                calibration_id=None if calibration is None else calibration.calibration_id,
                calibration_sha256=calibration_sha256,
                measured_generation=None if active_identity is None else active_identity[1],
                blockers=tuple(blockers),
            )

        source_hashes = tuple(sorted({
            frame_hash
            for color in matching
            for frame_hash in color.source_frame_sha256
        }))
        return AppearanceEvidence(
            ready=True,
            vertex_colors=colors,
            source_frame_sha256=source_hashes,
            minimum_independent_accepted_views=min(color.independent_views for color in matching),
            physical_depth_validation_revision=validation_revision,
            calibration_id=calibration.calibration_id,
            calibration_sha256=calibration_sha256,
            measured_generation=active_identity[1],
            blockers=(),
        )

    @staticmethod
    def _depth_report_core(
        report: Any,
        *,
        artifact_sha256: str,
        depth_frame_sha256: str,
        record_timestamp_ns: int,
    ) -> dict[str, Any]:
        return {
            "artifact_sha256": artifact_sha256,
            "depth_frame_sha256": depth_frame_sha256,
            "record_timestamp_ns": int(record_timestamp_ns),
            "calibration_id": report.calibration_id,
            "calibration_sha256": report.calibration_sha256,
            "measured_generation": report.measured_generation,
            "correspondence_count": report.correspondence_count,
            "tiles": [
                {
                    "tile_id": tile.tile_id,
                    "correspondence_count": tile.correspondence_count,
                    "median_absolute_residual_m": tile.median_absolute_residual_m,
                    "p95_absolute_residual_m": tile.p95_absolute_residual_m,
                    "mean_signed_residual_m": tile.mean_signed_residual_m,
                    "occlusion_conflict_fraction": tile.occlusion_conflict_fraction,
                    "free_space_violation_fraction": tile.free_space_violation_fraction,
                    "minimum_observed_range_m": tile.minimum_observed_range_m,
                    "maximum_observed_range_m": tile.maximum_observed_range_m,
                }
                for tile in sorted(report.tiles, key=lambda value: value.tile_id)
            ],
        }

    def _load_calibration(self) -> CalibrationManifestV2 | None:
        path = self.calibration_manifest_path
        if path is None or not path.is_file():
            with self._lock:
                self._blockers.append("MEASURED_CALIBRATION_MANIFEST_MISSING")
            return None
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._calibration_signature:
            return self._calibration
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse_calibration_manifest(payload)
            if parsed.status != "measured":
                raise ConfigError("calibration.status must be measured")
            calibration_hash = canonical_sha256(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ConfigError) as exc:
            with self._lock:
                self._calibration = None
                self._calibration_sha256 = None
                self._calibration_signature = signature
                self._color_accumulator = None
                self._blockers.append(f"MEASURED_CALIBRATION_INVALID:{exc}")
            return None
        with self._lock:
            if self._calibration is None or self._calibration.calibration_id != parsed.calibration_id:
                self._color_accumulator = ColorAccumulator(parsed.calibration_id)
            self._calibration = parsed
            self._calibration_sha256 = calibration_hash
            self._calibration_signature = signature
        return parsed

    def _camera_info_matches(self, calibration: CalibrationManifestV2, colors: Sequence[Any], depths: Sequence[Any]) -> bool:
        return bool(colors and depths) and all(
            self._camera_info_sha256(record) == expected
            for records, expected in (
                (colors, calibration.frame_chain.color_camera_info_sha256),
                (depths, calibration.frame_chain.depth_camera_info_sha256),
            )
            for record in records
        )

    @staticmethod
    def _camera_info_sha256(record: Any) -> str:
        return canonical_sha256({
            "width": int(record.width),
            "height": int(record.height),
            "distortion_model": record.distortion_model,
            "k": np.asarray(record.k, dtype=np.float64).reshape(-1).tolist(),
            "d": np.asarray(record.d, dtype=np.float64).reshape(-1).tolist(),
            "r": np.asarray(record.r, dtype=np.float64).reshape(-1).tolist(),
            "p": np.asarray(record.p, dtype=np.float64).reshape(-1).tolist(),
            "rectified": not np.allclose(record.r, np.eye(3), atol=1e-12),
        })

    def _depth_metres(self, record: Any) -> np.ndarray:
        if record.encoding == "16UC1":
            if self.depth_scale_m is None:
                raise ValueError("VERIFIED_DEPTH_SCALE_REQUIRED_FOR_16UC1")
            return np.asarray(record.image, dtype=np.float64) * self.depth_scale_m
        if record.encoding == "32FC1":
            return np.asarray(record.image, dtype=np.float64)
        raise ValueError(f"unsupported depth encoding: {record.encoding}")

    def _accumulate_color(
        self,
        lidar: Mapping[str, Any],
        depth_m: np.ndarray,
        color_record: Any,
        depth_from_lidar: np.ndarray,
        color_from_lidar: np.ndarray,
        depth_camera: np.ndarray,
        color_camera: np.ndarray,
    ) -> None:
        accumulator = self._color_accumulator
        calibration = self._calibration
        if accumulator is None or calibration is None:
            return
        rgb = np.asarray(color_record.image)
        if color_record.encoding == "bgr8":
            rgb = rgb[..., ::-1]
        elif color_record.encoding == "mono8":
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        elif color_record.encoding != "rgb8":
            with self._lock:
                self._blockers.append(f"UNSUPPORTED_COLOR_ENCODING:{color_record.encoding}")
            return
        points = np.asarray(lidar["points_xyz"], dtype=np.float64)
        pose = np.asarray(lidar["pose_map_from_lidar"], dtype=np.float64)
        depth_points = points @ depth_from_lidar[:3, :3].T + depth_from_lidar[:3, 3]
        color_points = points @ color_from_lidar[:3, :3].T + color_from_lidar[:3, 3]
        valid = (depth_points[:, 2] > 0.0) & (color_points[:, 2] > 0.0)
        indices = np.flatnonzero(valid)
        if not len(indices):
            return
        du, dv = self._project(depth_points[indices], depth_camera)
        cu, cv = self._project(color_points[indices], color_camera)
        inside = (
            (du >= 0) & (du < depth_m.shape[1]) & (dv >= 0) & (dv < depth_m.shape[0])
            & (cu >= 0) & (cu < rgb.shape[1]) & (cv >= 0) & (cv < rgb.shape[0])
        )
        indices, du, dv, cu, cv = indices[inside], du[inside], dv[inside], cu[inside], cv[inside]
        world = points[indices] @ pose[:3, :3].T + pose[:3, 3]
        origin = pose[:3, 3]
        sharpness = self._sharpness(rgb)
        for index, point_world, depth_u, depth_v, color_u, color_v in zip(indices, world, du, dv, cu, cv):
            observed = float(depth_m[depth_v, depth_u])
            predicted = float(depth_points[index, 2])
            residual = observed - predicted
            threshold = max(0.030, 0.015 * predicted)
            key = tuple(np.floor(point_world / self.voxel_size_m).astype(np.int64).tolist())
            vertex_id = f"{key[0]}:{key[1]}:{key[2]}"
            color = srgb_to_linear(np.asarray(rgb[color_v, color_u], dtype=np.float64) / 255.0)
            accepted = accumulator.observe(vertex_id, ColorObservation(
                linear_rgb=tuple(float(value) for value in color),
                depth_residual_m=residual,
                range_m=predicted,
                incidence_cosine=1.0,
                sharpness=sharpness,
                occluded=residual < -threshold,
                view_origin_xyz=tuple(float(value) for value in origin),
                view_direction_xyz=tuple(float(value) for value in point_world - origin),
                frame_sha256=color_record.message_sha256,
                calibration_id=calibration.calibration_id,
            ))
            with self._lock:
                self._observed_vertices.add(vertex_id)
                if accepted:
                    self._accepted_color_observations += 1
                else:
                    self._rejected_color_observations += 1

    def _nearest(self, timestamp_ns: int, times: Sequence[int], values: Sequence[tuple[int, Any]]) -> Any | None:
        if not values:
            return None
        index = bisect.bisect_left(times, timestamp_ns)
        candidates = []
        if index < len(values):
            candidates.append(values[index])
        if index:
            candidates.append(values[index - 1])
        closest = min(candidates, key=lambda item: abs(item[0] - timestamp_ns))
        return closest[1] if abs(closest[0] - timestamp_ns) <= self.max_pair_delta_ns else None

    @staticmethod
    def _matrix(row_major: Sequence[float]) -> np.ndarray:
        return np.asarray(row_major, dtype=np.float64).reshape(4, 4)

    @staticmethod
    def _project(points: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.rint(camera[0, 0] * points[:, 0] / points[:, 2] + camera[0, 2]).astype(np.int64)
        v = np.rint(camera[1, 1] * points[:, 1] / points[:, 2] + camera[1, 2]).astype(np.int64)
        return u, v

    @staticmethod
    def _sharpness(rgb: np.ndarray) -> float:
        gray = np.mean(np.asarray(rgb, dtype=np.float64), axis=2)
        gradients = []
        if gray.shape[0] > 1:
            gradients.append(np.mean(np.abs(np.diff(gray, axis=0))))
        if gray.shape[1] > 1:
            gradients.append(np.mean(np.abs(np.diff(gray, axis=1))))
        return max(0.05, min(1.0, (float(np.mean(gradients)) if gradients else 0.0) / 32.0))

    @staticmethod
    def _weighted_median(values: Sequence[tuple[float, int]]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        threshold = sum(weight for _, weight in ordered) / 2.0
        running = 0
        for value, weight in ordered:
            running += weight
            if running >= threshold:
                return value
        return ordered[-1][0]
