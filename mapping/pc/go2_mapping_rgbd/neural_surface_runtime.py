"""Live RGB-D measured surfaces and physically gated DA3 multi-view hole filling."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import threading
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from mapping.pc.go2_mapping_live.live_protocol import TextureKeyframeMessage
from mapping.pc.go2_mapping_live.live_surface_mesher import TriangleProvenance
from mapping.pc.go2_mapping_live.live_tiles import DecodedSurfaceTile, decode_surface_tile

from .depth_anything import MultiViewDepthFrame
from .multiview_surface import (
    Da3MultiViewWorker,
    Da3ViewObservation,
    Da3ViewWindow,
    Da3WindowAssembler,
    NeuralSurfaceCandidate,
    NeuralSurfaceDecision,
    NeuralSurfaceGate,
)


@dataclass(frozen=True)
class SurfaceTriangleBatch:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    triangle_provenance: np.ndarray
    triangle_uncertainty_m: np.ndarray
    source_revision_id: str


def _surface_from_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_from_world: np.ndarray,
    *,
    provenance: TriangleProvenance,
    source_revision_id: str,
    maximum_triangles: int = 250_000,
    reject_boundaries: bool,
) -> tuple[SurfaceTriangleBatch, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    camera_transform = np.asarray(camera_from_world, dtype=np.float64)
    if depth.ndim != 2 or camera_matrix.shape != (3, 3) or camera_transform.shape != (4, 4):
        raise ValueError("depth surface camera inputs have invalid shapes")
    height, width = depth.shape
    if height < 2 or width < 2:
        return _empty_surface(source_revision_id), np.empty(0, dtype=bool)
    native_triangles = 2 * (height - 1) * (width - 1)
    stride = max(1, int(math.ceil(math.sqrt(native_triangles / max(1, maximum_triangles)))))
    rows = np.arange(0, height, stride, dtype=np.int64)
    columns = np.arange(0, width, stride, dtype=np.int64)
    if len(rows) < 2 or len(columns) < 2:
        return _empty_surface(source_revision_id), np.empty(0, dtype=bool)
    grid_u, grid_v = np.meshgrid(columns, rows)
    sampled_depth = depth[grid_v, grid_u]
    valid_vertices = np.isfinite(sampled_depth) & (sampled_depth >= 0.10) & (sampled_depth <= 20.0)
    safe_depth = np.where(valid_vertices, sampled_depth, 1.0)
    camera_points = np.stack((
        (grid_u - camera_matrix[0, 2]) * safe_depth / camera_matrix[0, 0],
        (grid_v - camera_matrix[1, 2]) * safe_depth / camera_matrix[1, 1],
        safe_depth,
        np.ones_like(safe_depth),
    ), axis=-1).reshape(-1, 4)
    world_from_camera = np.linalg.inv(camera_transform)
    world_positions = (world_from_camera @ camera_points.T).T[:, :3]
    grid_width = len(columns)
    upper_left = np.arange((len(rows) - 1) * (grid_width - 1), dtype=np.int64)
    upper_left += upper_left // (grid_width - 1)
    raw_triangles = np.concatenate((
        np.column_stack((upper_left, upper_left + grid_width, upper_left + 1)),
        np.column_stack((upper_left + 1, upper_left + grid_width, upper_left + grid_width + 1)),
    )).astype(np.int64, copy=False)
    flat_valid = valid_vertices.reshape(-1)
    triangle_valid = np.all(flat_valid[raw_triangles], axis=1)
    triangle_depth = safe_depth.reshape(-1)[raw_triangles]
    depth_span = np.ptp(triangle_depth, axis=1)
    depth_centre = np.mean(triangle_depth, axis=1)
    boundary = depth_span > np.maximum(0.025, depth_centre * 0.02)
    triangle_valid &= ~boundary if reject_boundaries else True
    triangles = raw_triangles[triangle_valid]
    boundary = boundary[triangle_valid]
    if not len(triangles):
        return _empty_surface(source_revision_id), np.empty(0, dtype=bool)
    triangle_positions = world_positions[triangles]
    edge_a = triangle_positions[:, 1] - triangle_positions[:, 0]
    edge_b = triangle_positions[:, 2] - triangle_positions[:, 0]
    face_normals = np.cross(edge_a, edge_b)
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    nondegenerate = normal_lengths > 1e-8
    triangles = triangles[nondegenerate]
    boundary = boundary[nondegenerate]
    triangle_positions = triangle_positions[nondegenerate]
    face_normals = face_normals[nondegenerate] / normal_lengths[nondegenerate, None]
    camera_position = world_from_camera[:3, 3]
    centres = triangle_positions.mean(axis=1)
    reverse = np.einsum("ij,ij->i", face_normals, camera_position[None, :] - centres) < 0
    if np.any(reverse):
        triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
        face_normals[reverse] *= -1
    used = np.unique(triangles)
    remap = np.full(len(world_positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    local_triangles = remap[triangles].astype("<u4")
    positions = np.asarray(world_positions[used], dtype="<f4")
    normals = np.zeros_like(positions, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, local_triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    triangle_positions = positions[local_triangles].astype(np.float64)
    edge_lengths = np.stack((
        np.linalg.norm(triangle_positions[:, 1] - triangle_positions[:, 0], axis=1),
        np.linalg.norm(triangle_positions[:, 2] - triangle_positions[:, 1], axis=1),
        np.linalg.norm(triangle_positions[:, 0] - triangle_positions[:, 2], axis=1),
    ), axis=1)
    scales = np.maximum(np.max(edge_lengths, axis=1), 0.005).astype("<f4")
    mean_depth = depth_centre[triangle_valid][nondegenerate]
    uncertainty = np.maximum(0.005, mean_depth * 0.005).astype("<f4")
    batch = SurfaceTriangleBatch(
        positions=positions,
        normals=np.asarray(normals, dtype="<f4"),
        triangles=local_triangles,
        triangle_scale_m=scales,
        triangle_provenance=np.full(len(local_triangles), int(provenance), dtype="u1"),
        triangle_uncertainty_m=uncertainty,
        source_revision_id=str(source_revision_id),
    )
    return batch, boundary


def _empty_surface(source_revision_id: str) -> SurfaceTriangleBatch:
    return SurfaceTriangleBatch(
        np.empty((0, 3), dtype="<f4"),
        np.empty((0, 3), dtype="<f4"),
        np.empty((0, 3), dtype="<u4"),
        np.empty(0, dtype="<f4"),
        np.empty(0, dtype="u1"),
        np.empty(0, dtype="<f4"),
        str(source_revision_id),
    )


def build_measured_rgbd_surface(observation: Da3ViewObservation) -> SurfaceTriangleBatch:
    if observation.aligned_depth_m is None:
        return _empty_surface(observation.frame_sha256)
    surface, _ = _surface_from_depth(
        observation.aligned_depth_m,
        observation.intrinsics,
        observation.extrinsics_world_to_camera,
        provenance=TriangleProvenance.MEASURED_RGBD,
        source_revision_id=observation.frame_sha256,
        reject_boundaries=True,
    )
    return surface


class Da3SurfaceCandidateBuilder:
    gate = NeuralSurfaceGate()

    def build(
        self,
        window: Da3ViewWindow,
        prediction: MultiViewDepthFrame,
        measured_tiles: Sequence[Any],
    ) -> NeuralSurfaceCandidate:
        if tuple(view.frame_sha256 for view in window.views) != prediction.frame_sha256s:
            raise ValueError("DA3 prediction frame identity does not match its view window")
        reference = window.views[window.reference_index]
        neural_surface, reference_boundaries = _surface_from_depth(
            prediction.depth_m[window.reference_index],
            reference.intrinsics,
            reference.extrinsics_world_to_camera,
            provenance=TriangleProvenance.NEURAL_MULTIVIEW,
            source_revision_id=window.window_sha256,
            reject_boundaries=False,
        )
        triangles = neural_surface.triangles
        count = len(triangles)
        if count == 0:
            return self._empty_candidate(window.window_sha256)
        measured_points, measured_triangles = self._measured_geometry(measured_tiles)
        physical_maps = self._physical_depth_maps(window, measured_points, measured_triangles)
        global_metrics, physical_frames, physical_views = self._physical_holdout(
            prediction, physical_maps, window
        )
        centres = neural_surface.positions[triangles].mean(axis=1).astype(np.float64)
        face_normals = neural_surface.normals[triangles].mean(axis=1).astype(np.float64)
        normal_lengths = np.linalg.norm(face_normals, axis=1)
        face_normals[normal_lengths > 0] /= normal_lengths[normal_lengths > 0, None]
        residuals = np.full((count, len(window.views)), np.nan, dtype=np.float64)
        support = np.zeros((count, len(window.views)), dtype=bool)
        free_space = np.zeros(count, dtype=bool)
        occlusion = np.zeros(count, dtype=bool)
        measured_conflict = np.zeros(count, dtype=bool)
        observed_front = np.zeros(count, dtype=bool)
        for view_index, view in enumerate(window.views):
            projected = self._project(centres, view.extrinsics_world_to_camera, view.intrinsics)
            valid, pixel_x, pixel_y, camera_depth = projected
            sampled = np.full(count, np.nan, dtype=np.float64)
            sampled[valid] = prediction.depth_m[view_index, pixel_y[valid], pixel_x[valid]]
            residuals[valid, view_index] = sampled[valid] - camera_depth[valid]
            support[:, view_index] = valid & (np.abs(residuals[:, view_index]) <= 0.060)
            physical = physical_maps[view_index]
            physical_depth = np.full(count, np.nan, dtype=np.float64)
            physical_depth[valid] = physical[pixel_y[valid], pixel_x[valid]]
            physical_valid = valid & np.isfinite(physical_depth)
            delta = camera_depth - physical_depth
            free_space |= physical_valid & (delta < -0.030)
            occlusion |= physical_valid & (delta > 0.030)
            measured_conflict |= physical_valid & (np.abs(delta) <= 0.020)
            camera_position = np.linalg.inv(view.extrinsics_world_to_camera)[:3, 3]
            observed_front |= valid & (
                np.einsum("ij,ij->i", face_normals, camera_position[None, :] - centres) > 0
            )
        if len(measured_points):
            distances, _ = cKDTree(measured_points).query(centres, k=1)
            measured_conflict |= distances <= 0.018
        support_count = np.count_nonzero(support, axis=1).astype("u1")
        independent_count = np.zeros(count, dtype="u1")
        for independent_view_id in {view.independent_view_id for view in window.views}:
            columns = [
                index for index, view in enumerate(window.views)
                if view.independent_view_id == independent_view_id
            ]
            independent_count += np.any(support[:, columns], axis=1).astype("u1")
        support_count = np.minimum(support_count, physical_frames).astype("u1")
        independent_count = np.minimum(independent_count, physical_views).astype("u1")
        median, p95, bias, uncertainty = self._triangle_residual_metrics(residuals)
        median = np.maximum(median, global_metrics[0])
        p95 = np.maximum(p95, global_metrics[1])
        bias = np.maximum(bias, global_metrics[2])
        uncertainty = np.maximum(uncertainty, global_metrics[0])
        reference_camera = np.linalg.inv(reference.extrinsics_world_to_camera)[:3, 3]
        near = np.linalg.norm(centres - reference_camera[None, :], axis=1) <= 2.0
        near_error = float(global_metrics[1] * 1000.0) if np.any(near) else None
        return NeuralSurfaceCandidate(
            positions=neural_surface.positions,
            normals=neural_surface.normals,
            triangles=triangles,
            triangle_scale_m=neural_surface.triangle_scale_m,
            supporting_frame_count=support_count,
            independent_view_count=independent_count,
            holdout_median_residual_m=median.astype("<f4"),
            holdout_p95_residual_m=p95.astype("<f4"),
            holdout_absolute_bias_m=bias.astype("<f4"),
            uncertainty_m=uncertainty.astype("<f4"),
            observed_front=observed_front,
            free_space_conflict=free_space,
            occlusion_conflict=occlusion,
            depth_boundary_conflict=reference_boundaries,
            measured_surface_conflict=measured_conflict,
            source_window_sha256=window.window_sha256,
            near_object_size_holdout_error_mm=near_error,
        )

    @staticmethod
    def _measured_geometry(measured_tiles: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        for tile in measured_tiles:
            positions = np.asarray(tile.positions, dtype=np.float64)
            faces = np.asarray(tile.triangles, dtype=np.int64)
            provenance = np.asarray(tile.triangle_provenance, dtype=np.uint8)
            measured_faces = faces[provenance != int(TriangleProvenance.NEURAL_MULTIVIEW)]
            if len(measured_faces):
                geometry = positions[measured_faces]
                points.extend((geometry.reshape(-1, 3), geometry.mean(axis=1)))
                triangles.append(geometry)
        return (
            np.concatenate(points) if points else np.empty((0, 3), dtype=np.float64),
            np.concatenate(triangles) if triangles else np.empty((0, 3, 3), dtype=np.float64),
        )

    @classmethod
    def _physical_depth_maps(
        cls,
        window: Da3ViewWindow,
        measured_points: np.ndarray,
        measured_triangles: np.ndarray,
    ) -> list[np.ndarray]:
        del measured_triangles
        maps: list[np.ndarray] = []
        for view in window.views:
            height, width = view.rectified_rgb.shape[:2]
            physical = np.full((height, width), np.nan, dtype=np.float64)
            if view.aligned_depth_m is not None:
                aligned = np.asarray(view.aligned_depth_m, dtype=np.float64)
                physical[np.isfinite(aligned)] = aligned[np.isfinite(aligned)]
            if len(measured_points):
                valid, pixel_x, pixel_y, depth = cls._project(
                    measured_points, view.extrinsics_world_to_camera, view.intrinsics
                )
                flat = np.full(height * width, np.inf, dtype=np.float64)
                indices = pixel_y[valid] * width + pixel_x[valid]
                np.minimum.at(flat, indices, depth[valid])
                projected = flat.reshape(height, width)
                projected[~np.isfinite(projected)] = np.nan
                combine = np.isfinite(projected) & (
                    ~np.isfinite(physical) | (projected < physical)
                )
                physical[combine] = projected[combine]
            dilated = minimum_filter(
                np.where(np.isfinite(physical), physical, np.inf), size=3, mode="constant", cval=np.inf
            )
            dilated[~np.isfinite(dilated)] = np.nan
            maps.append(dilated)
        return maps

    @staticmethod
    def _physical_holdout(
        prediction: MultiViewDepthFrame,
        physical_maps: Sequence[np.ndarray],
        window: Da3ViewWindow,
    ) -> tuple[tuple[float, float, float], int, int]:
        residual_parts: list[np.ndarray] = []
        evidence_view_ids: set[str] = set()
        frame_count = 0
        for index, physical in enumerate(physical_maps):
            valid = np.isfinite(physical)
            if not np.any(valid):
                continue
            residual_parts.append(prediction.depth_m[index][valid].astype(np.float64) - physical[valid])
            frame_count += 1
            evidence_view_ids.add(window.views[index].independent_view_id)
        if not residual_parts:
            return (1_000_000.0, 1_000_000.0, 1_000_000.0), 0, 0
        residual = np.concatenate(residual_parts)
        absolute = np.abs(residual)
        return (
            float(np.median(absolute)),
            float(np.percentile(absolute, 95.0)),
            float(abs(np.median(residual))),
        ), frame_count, len(evidence_view_ids)

    @staticmethod
    def _triangle_residual_metrics(residuals: np.ndarray) -> tuple[np.ndarray, ...]:
        count = len(residuals)
        median = np.full(count, 1_000_000.0)
        p95 = np.full(count, 1_000_000.0)
        bias = np.full(count, 1_000_000.0)
        uncertainty = np.full(count, 1_000_000.0)
        for index, values in enumerate(residuals):
            valid = values[np.isfinite(values)]
            if not len(valid):
                continue
            absolute = np.abs(valid)
            median[index] = np.median(absolute)
            p95[index] = np.percentile(absolute, 95.0)
            bias[index] = abs(np.median(valid))
            uncertainty[index] = math.sqrt(float(np.mean(np.square(valid))))
        return median, p95, bias, uncertainty

    @staticmethod
    def _project(
        points: np.ndarray,
        camera_from_world: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        transform = np.asarray(camera_from_world, dtype=np.float64)
        camera = (transform @ np.column_stack((values, np.ones(len(values)))).T).T[:, :3]
        depth = camera[:, 2]
        safe = np.where(depth > 1e-6, depth, 1.0)
        matrix = np.asarray(intrinsics, dtype=np.float64)
        x = np.rint(matrix[0, 0] * camera[:, 0] / safe + matrix[0, 2]).astype(np.int64)
        y = np.rint(matrix[1, 1] * camera[:, 1] / safe + matrix[1, 2]).astype(np.int64)
        width = max(1, int(round(matrix[0, 2] * 2 + 1)))
        height = max(1, int(round(matrix[1, 2] * 2 + 1)))
        valid = (depth > 0.05) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return valid, x, y, depth

    @staticmethod
    def _empty_candidate(window_sha256: str) -> NeuralSurfaceCandidate:
        empty_float = np.empty(0, dtype="<f4")
        empty_bool = np.empty(0, dtype=bool)
        return NeuralSurfaceCandidate(
            np.empty((0, 3), dtype="<f4"), np.empty((0, 3), dtype="<f4"),
            np.empty((0, 3), dtype="<u4"), empty_float,
            np.empty(0, dtype="u1"), np.empty(0, dtype="u1"),
            empty_float, empty_float, empty_float, empty_float,
            empty_bool, empty_bool, empty_bool, empty_bool, empty_bool,
            window_sha256,
        )


class LiveDa3SurfaceRuntime:
    """Converts the durable texture stream into measured RGB-D and gated DA3 tiles."""

    def __init__(
        self,
        provider: Any,
        surface_pipeline: Any,
        *,
        on_publish: Optional[Callable[[str], None]] = None,
        candidate_builder: Optional[Da3SurfaceCandidateBuilder] = None,
    ) -> None:
        self._pipeline = surface_pipeline
        self._on_publish = on_publish
        self._candidate_builder = candidate_builder or Da3SurfaceCandidateBuilder()
        self._assemblers: dict[tuple[str, str, str], Da3WindowAssembler] = {}
        self._lock = threading.RLock()
        self._worker = None if provider is None else Da3MultiViewWorker(provider, self._process_window)
        self._seen: set[tuple[str, str, int, int]] = set()
        self._accepted_observations = 0
        self._rgbd_published_triangles = 0
        self._published_neural_triangles = 0
        self._rejected_neural_triangles = 0
        self._last_metrics: Optional[dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._blockers: list[str] = []
        if provider is None:
            self._blockers.append("DA3_PROVIDER_UNAVAILABLE")

    def ingest(
        self,
        frame: TextureKeyframeMessage,
        *,
        submap_id: str,
        camera_from_lidar: Optional[np.ndarray],
        calibration_authority: bool,
    ) -> bool:
        try:
            return self._ingest(
                frame,
                submap_id=submap_id,
                camera_from_lidar=camera_from_lidar,
                calibration_authority=calibration_authority,
            )
        except Exception as exc:
            self.record_error(exc, blocker="DA3_SURFACE_INGEST_FAILED")
            raise

    def _ingest(
        self,
        frame: TextureKeyframeMessage,
        *,
        submap_id: str,
        camera_from_lidar: Optional[np.ndarray],
        calibration_authority: bool,
    ) -> bool:
        if not calibration_authority or camera_from_lidar is None:
            self._block("DA3_CALIBRATION_AUTHORITY_REQUIRED")
            return False
        identity = frame.identity
        seen_key = (
            submap_id,
            identity.producer_instance_id,
            identity.producer_epoch,
            identity.sequence,
        )
        with self._lock:
            if seen_key in self._seen:
                return False
        camera_from_map = np.asarray(camera_from_lidar, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(
            np.asarray(frame.pose_map_from_lidar, dtype=np.float64).reshape(4, 4)
        )
        try:
            with Image.open(BytesIO(frame.color_jpeg)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError(f"texture JPEG cannot be decoded for DA3: {exc}") from exc
        if rgb.shape[:2] != (frame.source_dimensions[1], frame.source_dimensions[0]):
            raise ValueError("decoded texture dimensions do not match the durable source identity")
        observation_sha256 = hashlib.sha256(json.dumps({
            "schema_version": 1,
            "content_sha256": frame.content_sha256,
            "submap_id": submap_id,
            "producer_instance_id": identity.producer_instance_id,
            "producer_epoch": identity.producer_epoch,
            "texture_sequence": identity.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "pose_map_from_lidar": frame.pose_map_from_lidar.reshape(-1).tolist(),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        intrinsics = frame.intrinsics
        camera_matrix = np.asarray([
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        aligned_depth = None
        if frame.depth_u16_bytes is not None:
            depth_payload = frame.payload["depth"]
            depth_u16 = np.frombuffer(frame.depth_u16_bytes, dtype="<u2").reshape(rgb.shape[:2])
            aligned_depth = depth_u16.astype(np.float32) * float(depth_payload["scale_m"])
            aligned_depth[depth_u16 == 0] = np.nan
        observation = Da3ViewObservation(
            frame_sha256=observation_sha256,
            independent_view_id=str(frame.quality["independent_view_id"]),
            timestamp_ns=frame.timestamp_ns,
            rectified_rgb=rgb,
            intrinsics=camera_matrix,
            extrinsics_world_to_camera=camera_from_map.astype(np.float32),
            aligned_depth_m=aligned_depth,
            building_id=identity.building_id,
            session_id=identity.session_id,
            submap_id=submap_id,
        )
        measured = build_measured_rgbd_surface(observation)
        if len(measured.triangles):
            result = self._pipeline.publish_measured_rgbd_surface(
                identity.building_id, submap_id, measured
            )
            published = int(result.get("triangle_count", 0))
            with self._lock:
                self._rgbd_published_triangles += published
            if published > 0:
                self._notify_publish(identity.building_id)
        key = (identity.building_id, identity.session_id, submap_id)
        with self._lock:
            self._seen.add(seen_key)
            self._accepted_observations += 1
            if self._worker is None:
                return True
            assembler = self._assemblers.setdefault(key, Da3WindowAssembler(window_size=5, overlap=2))
            window = assembler.submit(observation)
        if window is not None:
            self._worker.submit(identity.session_id, window)
        return True

    def _process_window(self, window: Da3ViewWindow, prediction: MultiViewDepthFrame) -> None:
        first = window.views[0]
        if first.building_id is None or first.session_id is None or first.submap_id is None:
            raise ValueError("DA3 live view context is incomplete")
        if any(
            (view.building_id, view.session_id, view.submap_id)
            != (first.building_id, first.session_id, first.submap_id)
            for view in window.views
        ):
            raise ValueError("DA3 live window crosses a Building, session, or submap boundary")
        evidence = self._pipeline.neural_surface_evidence(first.building_id, first.submap_id)
        measured_tiles = [decode_surface_tile(item["body"]) for item in evidence.get("tiles", ())]
        if not measured_tiles:
            self._block("DA3_MEASURED_SURFACE_EVIDENCE_REQUIRED")
            return
        candidate = self._candidate_builder.build(window, prediction, measured_tiles)
        decision = self._candidate_builder.gate.evaluate(candidate)
        with self._lock:
            self._last_metrics = dict(decision.metrics)
            self._rejected_neural_triangles += int(decision.metrics["rejected_triangle_count"])
            for blocker in decision.blockers:
                self._block(blocker)
        if not len(decision.accepted_triangle_indices):
            return
        result = self._pipeline.publish_neural_surface(
            first.building_id,
            first.submap_id,
            candidate,
            decision,
            expected_source_generation_id=str(evidence["source_generation_id"]),
        )
        published = int(result.get("triangle_count", 0))
        with self._lock:
            self._published_neural_triangles += published
        if published > 0:
            self._notify_publish(first.building_id)

    def run_once(self) -> bool:
        return False if self._worker is None else self._worker.run_once()

    def start(self) -> None:
        if self._worker is not None:
            self._worker.start()

    def close(self, timeout_s: float = 2.0) -> None:
        if self._worker is not None:
            self._worker.close(timeout_s)

    def snapshot(self) -> dict[str, Any]:
        worker = self._worker.snapshot() if self._worker is not None else {
            "state": "DA3_SURFACE_UNAVAILABLE",
            "queue_capacity": 0,
            "pending_windows": 0,
            "active_session_id": None,
            "disabled_session_id": None,
            "submitted_windows": 0,
            "processed_windows": 0,
            "failed_windows": 0,
            "coalesced_windows": 0,
            "last_inference_ms": None,
            "last_error": None,
            "blockers": ["DA3_PROVIDER_UNAVAILABLE"],
        }
        with self._lock:
            blockers = list(dict.fromkeys([*worker["blockers"], *self._blockers]))
            last_error = self._last_error or worker["last_error"]
            state = worker["state"]
            if self._worker is not None and self._last_error is not None:
                state = "DA3_SURFACE_DEGRADED"
            return {
                **worker,
                "state": state,
                "blockers": blockers,
                "last_error": last_error,
                "accepted_observations": self._accepted_observations,
                "active_view_assemblers": len(self._assemblers),
                "rgbd_published_triangles": self._rgbd_published_triangles,
                "published_neural_triangles": self._published_neural_triangles,
                "rejected_neural_triangles": self._rejected_neural_triangles,
                "last_surface_metrics": None if self._last_metrics is None else dict(self._last_metrics),
                "surface_blockers": list(dict.fromkeys(self._blockers)),
            }

    def record_error(self, error: BaseException | str, *, blocker: str) -> None:
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        with self._lock:
            self._last_error = message
            if blocker not in self._blockers:
                self._blockers.append(blocker)

    def _notify_publish(self, building_id: str) -> None:
        if self._on_publish is None:
            return
        try:
            self._on_publish(building_id)
        except Exception as exc:
            self.record_error(exc, blocker="DA3_PUBLISH_CALLBACK_FAILED")

    def _block(self, blocker: str) -> None:
        with self._lock:
            if blocker not in self._blockers:
                self._blockers.append(blocker)
