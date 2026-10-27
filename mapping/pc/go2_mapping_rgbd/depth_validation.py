"""Camera-internal and calibrated LiDAR/depth validation metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import DepthValidationGates, load_reconstruction_config


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CameraInternalReport:
    timestamp_ns: int
    valid_fraction: float
    missing_fraction: float
    saturated_fraction: float
    planar_noise_m: float | None
    temporal_stability_m: float | None
    timestamp_monotonic: bool
    aligned_to_color: bool
    authorizes_texture: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class DepthTileReport:
    tile_id: str
    correspondence_count: int
    median_absolute_residual_m: float
    p95_absolute_residual_m: float
    mean_signed_residual_m: float
    occlusion_conflict_fraction: float
    free_space_violation_fraction: float
    minimum_observed_range_m: float
    maximum_observed_range_m: float
    accepted: bool


@dataclass(frozen=True)
class AcceptedDepthCorrespondence:
    lidar_point_index: int
    depth_pixel: tuple[int, int]
    color_pixel: tuple[int, int]
    point_camera_xyz: tuple[float, float, float]
    observed_metric_depth_m: float
    residual_class: str
    accepted_region_id: str
    frame_sha256: str
    calibration_sha256: str
    measured_generation: str
    timestamp_ns: int


@dataclass(frozen=True)
class CrossSensorDepthReport:
    calibration_id: str
    calibration_sha256: str
    measured_generation: str
    correspondence_count: int
    tiles: tuple[DepthTileReport, ...]
    accepted: bool
    authorizes_texture: bool
    blockers: tuple[str, ...]
    correspondences: tuple[AcceptedDepthCorrespondence, ...] = ()


class DepthValidator:
    def __init__(self, gates: DepthValidationGates | None = None) -> None:
        self.gates = gates or load_reconstruction_config().appearance.depth_validation
        self._previous_depth: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None

    def camera_internal(
        self,
        depth_m: np.ndarray,
        timestamp_ns: int,
        *,
        aligned_to_color: bool,
    ) -> CameraInternalReport:
        depth = self._depth(depth_m)
        finite = np.isfinite(depth)
        valid = finite & (depth >= self.gates.minimum_range_m) & (depth <= self.gates.maximum_range_m)
        valid_fraction = float(np.mean(valid))
        missing_fraction = float(np.mean(~finite | (depth <= 0.0)))
        saturated_fraction = float(np.mean(finite & (depth > self.gates.maximum_range_m)))
        tile_noise = []
        for y in range(0, depth.shape[0], 8):
            for x in range(0, depth.shape[1], 8):
                tile = depth[y : y + 8, x : x + 8]
                mask = valid[y : y + 8, x : x + 8]
                if np.count_nonzero(mask) >= max(4, mask.size // 2):
                    tile_noise.append(float(np.std(tile[mask])))
        planar_noise = None if not tile_noise else float(np.median(tile_noise))
        temporal = None
        if self._previous_depth is not None and self._previous_depth.shape == depth.shape:
            common = valid & np.isfinite(self._previous_depth)
            common &= (self._previous_depth >= self.gates.minimum_range_m) & (
                self._previous_depth <= self.gates.maximum_range_m
            )
            if np.any(common):
                temporal = float(np.median(np.abs(depth[common] - self._previous_depth[common])))
        monotonic = self._previous_timestamp_ns is None or timestamp_ns > self._previous_timestamp_ns
        blockers = []
        if not np.any(valid):
            blockers.append("NO_VALID_DEPTH_IN_CALIBRATED_RANGE")
        if not monotonic:
            blockers.append("DEPTH_TIMESTAMP_NOT_MONOTONIC")
        if not aligned_to_color:
            blockers.append("DEPTH_NOT_ALIGNED_TO_COLOR")
        self._previous_depth = np.array(depth, copy=True)
        self._previous_timestamp_ns = timestamp_ns
        return CameraInternalReport(
            timestamp_ns=timestamp_ns,
            valid_fraction=valid_fraction,
            missing_fraction=missing_fraction,
            saturated_fraction=saturated_fraction,
            planar_noise_m=planar_noise,
            temporal_stability_m=temporal,
            timestamp_monotonic=monotonic,
            aligned_to_color=aligned_to_color,
            authorizes_texture=False,
            blockers=tuple(blockers),
        )

    def cross_sensor(
        self,
        measured_positions_lidar: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
        depth_from_lidar: np.ndarray,
        *,
        calibration_id: str,
        calibration_sha256: str,
        measured_generation: str,
        frame_sha256: str | None = None,
        timestamp_ns: int | None = None,
        color_camera_matrix: np.ndarray | None = None,
        color_from_lidar: np.ndarray | None = None,
    ) -> CrossSensorDepthReport:
        points = np.asarray(measured_positions_lidar, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("measured_positions_lidar must be a finite N by 3 array")
        depth = self._depth(depth_m)
        camera = np.asarray(camera_matrix, dtype=np.float64)
        transform = np.asarray(depth_from_lidar, dtype=np.float64)
        if camera.shape != (3, 3) or camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
            raise ValueError("camera_matrix is unusable")
        if transform.shape != (4, 4) or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("depth_from_lidar must be homogeneous")
        if not calibration_id or not measured_generation or not _SHA256_RE.fullmatch(calibration_sha256):
            raise ValueError("calibration and measured-generation identities are required")
        if frame_sha256 is not None and not _SHA256_RE.fullmatch(frame_sha256):
            raise ValueError("frame_sha256 must be a SHA-256 digest")
        if timestamp_ns is not None and (isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0):
            raise ValueError("timestamp_ns must be a non-negative integer")
        if (color_camera_matrix is None) != (color_from_lidar is None):
            raise ValueError("color camera and transform must be provided together")
        color_camera = None
        color_transform = None
        if color_camera_matrix is not None and color_from_lidar is not None:
            color_camera = np.asarray(color_camera_matrix, dtype=np.float64)
            color_transform = np.asarray(color_from_lidar, dtype=np.float64)
            if color_camera.shape != (3, 3) or color_camera[0, 0] <= 0.0 or color_camera[1, 1] <= 0.0:
                raise ValueError("color_camera_matrix is unusable")
            if color_transform.shape != (4, 4) or not np.allclose(color_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                raise ValueError("color_from_lidar must be homogeneous")

        point_indices = np.arange(len(points), dtype=np.int64)
        depth_points = points @ transform[:3, :3].T + transform[:3, 3]
        positive = depth_points[:, 2] > 0.0
        point_indices = point_indices[positive]
        lidar_points = points[positive]
        depth_points = depth_points[positive]
        if len(depth_points) == 0:
            return self._empty(calibration_id, calibration_sha256, measured_generation, "NO_POINTS_IN_DEPTH_FOV")
        u = np.rint(camera[0, 0] * depth_points[:, 0] / depth_points[:, 2] + camera[0, 2]).astype(np.int64)
        v = np.rint(camera[1, 1] * depth_points[:, 1] / depth_points[:, 2] + camera[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        depth_points, lidar_points, point_indices, u, v = (
            depth_points[inside], lidar_points[inside], point_indices[inside], u[inside], v[inside]
        )
        if len(depth_points) == 0:
            return self._empty(calibration_id, calibration_sha256, measured_generation, "NO_PROJECTED_OVERLAP")

        order = np.argsort(depth_points[:, 2], kind="stable")
        pixel = v[order] * depth.shape[1] + u[order]
        _, first = np.unique(pixel, return_index=True)
        selected = order[first]
        depth_points, lidar_points, point_indices, u, v = (
            depth_points[selected], lidar_points[selected], point_indices[selected], u[selected], v[selected]
        )
        observed = depth[v, u]
        valid = np.isfinite(observed) & (observed >= self.gates.minimum_range_m) & (
            observed <= self.gates.maximum_range_m
        )
        depth_points, lidar_points, point_indices, u, v, observed = (
            depth_points[valid], lidar_points[valid], point_indices[valid], u[valid], v[valid], observed[valid]
        )
        if len(observed) == 0:
            return self._empty(calibration_id, calibration_sha256, measured_generation, "NO_VALID_DEPTH_OVERLAP")

        residual = observed - depth_points[:, 2]
        tile_size = load_reconstruction_config().fusion.metric_domain.depth_tile_size_m
        tile_keys = np.floor(depth_points / tile_size).astype(np.int64)
        reports = []
        for key in sorted(set(map(tuple, tile_keys.tolist()))):
            mask = np.all(tile_keys == key, axis=1)
            reports.append(self._tile(key, observed[mask], depth_points[mask, 2], residual[mask]))
        blockers = []
        rejected = [item.tile_id for item in reports if not item.accepted]
        if rejected:
            blockers.append("DEPTH_TILE_GATE_FAILED:" + ",".join(rejected[:20]))
        accepted = bool(reports) and not blockers
        report_by_tile = {item.tile_id: item for item in reports}
        color_u = np.array(u, copy=True)
        color_v = np.array(v, copy=True)
        color_inside = np.ones(len(depth_points), dtype=bool)
        if color_camera is not None and color_transform is not None:
            color_points = lidar_points @ color_transform[:3, :3].T + color_transform[:3, 3]
            color_positive = color_points[:, 2] > 0.0
            safe_z = np.where(color_positive, color_points[:, 2], 1.0)
            color_u = np.rint(color_camera[0, 0] * color_points[:, 0] / safe_z + color_camera[0, 2]).astype(np.int64)
            color_v = np.rint(color_camera[1, 1] * color_points[:, 1] / safe_z + color_camera[1, 2]).astype(np.int64)
            color_inside = color_positive & (color_u >= 0) & (color_v >= 0)
        correspondences = []
        if frame_sha256 is not None:
            thresholds = np.maximum(0.030, 0.015 * depth_points[:, 2])
            for index in range(len(depth_points)):
                tile_id = f"{tile_keys[index, 0]}:{tile_keys[index, 1]}:{tile_keys[index, 2]}"
                tile = report_by_tile[tile_id]
                if not tile.accepted or not color_inside[index] or abs(residual[index]) > thresholds[index]:
                    continue
                correspondences.append(
                    AcceptedDepthCorrespondence(
                        lidar_point_index=int(point_indices[index]),
                        depth_pixel=(int(u[index]), int(v[index])),
                        color_pixel=(int(color_u[index]), int(color_v[index])),
                        point_camera_xyz=tuple(float(value) for value in depth_points[index]),
                        observed_metric_depth_m=float(observed[index]),
                        residual_class="CONSENSUS",
                        accepted_region_id=tile_id,
                        frame_sha256=frame_sha256.lower(),
                        calibration_sha256=calibration_sha256.lower(),
                        measured_generation=measured_generation,
                        timestamp_ns=0 if timestamp_ns is None else timestamp_ns,
                    )
                )
        return CrossSensorDepthReport(
            calibration_id=calibration_id,
            calibration_sha256=calibration_sha256.lower(),
            measured_generation=measured_generation,
            correspondence_count=len(observed),
            tiles=tuple(reports),
            accepted=accepted,
            authorizes_texture=accepted,
            blockers=tuple(blockers),
            correspondences=tuple(correspondences),
        )

    def _tile(
        self,
        key: tuple[int, int, int],
        observed: np.ndarray,
        predicted: np.ndarray,
        residual: np.ndarray,
    ) -> DepthTileReport:
        absolute = np.abs(residual)
        threshold = np.maximum(0.030, 0.015 * predicted)
        occlusion = float(np.mean(residual < -threshold))
        free_space = float(np.mean(residual > threshold))
        median = float(np.median(absolute))
        p95 = float(np.percentile(absolute, 95))
        bias = float(np.mean(residual))
        minimum = float(np.min(observed))
        maximum = float(np.max(observed))
        accepted = self.gates.accepts(
            correspondence_count=len(residual),
            median_absolute_residual_m=median,
            p95_absolute_residual_m=p95,
            absolute_mean_signed_bias_m=abs(bias),
            occlusion_conflict_fraction=occlusion,
            free_space_violation_fraction=free_space,
            minimum_observed_range_m=minimum,
            maximum_observed_range_m=maximum,
        )
        return DepthTileReport(
            tile_id=f"{key[0]}:{key[1]}:{key[2]}",
            correspondence_count=len(residual),
            median_absolute_residual_m=median,
            p95_absolute_residual_m=p95,
            mean_signed_residual_m=bias,
            occlusion_conflict_fraction=occlusion,
            free_space_violation_fraction=free_space,
            minimum_observed_range_m=minimum,
            maximum_observed_range_m=maximum,
            accepted=accepted,
        )

    @staticmethod
    def _depth(value: np.ndarray) -> np.ndarray:
        depth = np.asarray(value, dtype=np.float64)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth_m must be a non-empty two-dimensional array")
        return depth

    @staticmethod
    def _empty(calibration_id: str, calibration_sha256: str, generation: str, reason: str) -> CrossSensorDepthReport:
        return CrossSensorDepthReport(
            calibration_id,
            calibration_sha256.lower(),
            generation,
            0,
            (),
            False,
            False,
            (reason,),
        )
