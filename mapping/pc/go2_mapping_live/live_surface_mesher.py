"""Scale-separated observed-surface meshing with conservative face gates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .live_surface import LiveSurfaceMap, SurfelSnapshot, TileId


class TriangleProvenance(IntEnum):
    MEASURED_LIDAR = 1
    MEASURED_RGBD = 2
    NEURAL_MULTIVIEW = 3


TRIANGLE_PROVENANCE_NAMES = {
    int(TriangleProvenance.MEASURED_LIDAR): "MEASURED_LIDAR",
    int(TriangleProvenance.MEASURED_RGBD): "MEASURED_RGBD",
    int(TriangleProvenance.NEURAL_MULTIVIEW): "NEURAL_MULTIVIEW",
}


@dataclass(frozen=True)
class MeshTileRevision:
    tile_id: TileId
    revision_id: str
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    unresolved_surfel_count: int
    triangle_provenance: Optional[np.ndarray] = None
    triangle_uncertainty_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MeshingResult:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    triangle_provenance: np.ndarray
    triangle_uncertainty_m: np.ndarray
    tile_revisions: dict[TileId, MeshTileRevision]
    scale_sample_counts: dict[float, int]
    unresolved_surfel_count: int

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


@dataclass(frozen=True)
class _FineMetric:
    surfel: SurfelSnapshot
    normal: np.ndarray
    curvature: float
    confidence: float
    neighbors_15: int
    neighbors_30: int
    neighbors_40: int
    normal_spread_deg: float


@dataclass(frozen=True)
class _MeshSamples:
    scale_m: float
    positions: np.ndarray
    normals: np.ndarray
    source_indices: tuple[tuple[int, ...], ...]
    exclusion_mask: frozenset[tuple[int, int, int]]


class LiveSurfaceMesher:
    def __init__(self, *, halo_m: float = 0.05) -> None:
        if not math.isfinite(halo_m) or halo_m < 0:
            raise ValueError("halo_m must be finite and non-negative")
        self.halo_m = float(halo_m)

    def build(self, surface: LiveSurfaceMap, *, tiles: Optional[set[TileId]] = None) -> MeshingResult:
        if tiles is None:
            surfels = surface.surfels()
        else:
            if not tiles:
                return self._empty_result()
            surfels = surface.surfels_for_tiles(tiles, halo_m=self.halo_m)
        if not surfels:
            return self._empty_result()
        metrics = self._fine_metrics(surfels)
        samples, unresolved = self._assign_scales(metrics)
        positions_parts: list[np.ndarray] = []
        normals_parts: list[np.ndarray] = []
        triangles_parts: list[np.ndarray] = []
        triangle_scale_parts: list[np.ndarray] = []
        offset = 0
        for sample_set in samples:
            triangles = self._triangulate(sample_set)
            if len(triangles) == 0:
                mesh_positions, mesh_normals, triangles = self._tangent_patches(
                    sample_set
                )
            else:
                mesh_positions = sample_set.positions
                mesh_normals = sample_set.normals
            if len(triangles) == 0:
                continue
            positions_parts.append(mesh_positions.astype("<f4", copy=False))
            normals_parts.append(mesh_normals.astype("<f4", copy=False))
            triangles_parts.append(triangles.astype("<u4", copy=False) + offset)
            triangle_scale_parts.append(np.full(len(triangles), sample_set.scale_m, dtype="<f4"))
            offset += len(mesh_positions)
        if positions_parts:
            positions = np.concatenate(positions_parts)
            normals = np.concatenate(normals_parts)
            triangles = np.concatenate(triangles_parts)
            triangle_scales = np.concatenate(triangle_scale_parts)
        else:
            positions = np.empty((0, 3), dtype="<f4")
            normals = np.empty((0, 3), dtype="<f4")
            triangles = np.empty((0, 3), dtype="<u4")
            triangle_scales = np.empty((0,), dtype="<f4")
        triangle_provenance = np.full(
            len(triangles), int(TriangleProvenance.MEASURED_LIDAR), dtype="u1"
        )
        triangle_uncertainty = np.asarray(triangle_scales, dtype="<f4")
        tile_revisions = self._tile_revisions(
            positions,
            normals,
            triangles,
            triangle_scales,
            triangle_provenance,
            triangle_uncertainty,
            surface.tile_m,
            unresolved,
            tiles,
        )
        return MeshingResult(
            positions=positions,
            normals=normals,
            triangles=triangles,
            triangle_scale_m=triangle_scales,
            triangle_provenance=triangle_provenance,
            triangle_uncertainty_m=triangle_uncertainty,
            tile_revisions=tile_revisions,
            scale_sample_counts={sample.scale_m: len(sample.positions) for sample in samples},
            unresolved_surfel_count=unresolved,
        )

    @staticmethod
    def _empty_result() -> MeshingResult:
        return MeshingResult(
            positions=np.empty((0, 3), dtype="<f4"),
            normals=np.empty((0, 3), dtype="<f4"),
            triangles=np.empty((0, 3), dtype="<u4"),
            triangle_scale_m=np.empty((0,), dtype="<f4"),
            triangle_provenance=np.empty((0,), dtype="u1"),
            triangle_uncertainty_m=np.empty((0,), dtype="<f4"),
            tile_revisions={},
            scale_sample_counts={0.005: 0, 0.01: 0, 0.02: 0},
            unresolved_surfel_count=0,
        )

    @staticmethod
    def _fine_metrics(surfels: tuple[SurfelSnapshot, ...]) -> tuple[_FineMetric, ...]:
        positions = np.asarray([surfel.centroid for surfel in surfels], dtype=np.float64)
        tree = cKDTree(positions)
        neighborhoods_15 = tree.query_ball_point(positions, r=0.0150001)
        neighborhoods_30 = tree.query_ball_point(positions, r=0.0300001)
        neighborhoods_40 = tree.query_ball_point(positions, r=0.0400001)
        normals = np.zeros_like(positions)
        curvatures = np.full(len(surfels), math.inf, dtype=np.float64)
        confidences = np.zeros(len(surfels), dtype=np.float64)
        for index, neighbors in enumerate(neighborhoods_40):
            local_indices = [item for item in neighbors if item != index]
            if len(local_indices) < 3:
                continue
            local = positions[np.asarray([index, *local_indices], dtype=np.int64)]
            centred = local - local.mean(axis=0)
            covariance = centred.T @ centred / max(len(local) - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            total = max(float(eigenvalues.sum()), 1e-15)
            normal = eigenvectors[:, 0]
            view = np.asarray(surfels[index].mean_view_direction, dtype=np.float64)
            if float(np.dot(normal, view)) > 0:
                normal = -normal
            normals[index] = normal / max(float(np.linalg.norm(normal)), 1e-15)
            curvatures[index] = float(eigenvalues[0] / total)
            confidences[index] = float(1.0 - eigenvalues[0] / total)
        results: list[_FineMetric] = []
        for index, surfel in enumerate(surfels):
            indices_15 = [item for item in neighborhoods_15[index] if item != index]
            indices_30 = [item for item in neighborhoods_30[index] if item != index]
            indices_40 = [item for item in neighborhoods_40[index] if item != index]
            spread = 180.0
            valid_normals = [normals[item] for item in indices_40 if np.linalg.norm(normals[item]) > 0]
            if np.linalg.norm(normals[index]) > 0 and valid_normals:
                spread = max(
                    math.degrees(
                        math.acos(float(np.clip(abs(np.dot(normals[index], neighbor)), -1.0, 1.0)))
                    )
                    for neighbor in valid_normals
                )
            results.append(
                _FineMetric(
                    surfel=surfel,
                    normal=normals[index],
                    curvature=float(curvatures[index]),
                    confidence=float(confidences[index]),
                    neighbors_15=len(indices_15),
                    neighbors_30=len(indices_30),
                    neighbors_40=len(indices_40),
                    normal_spread_deg=spread,
                )
            )
        return tuple(results)

    @staticmethod
    def _eligible_20(metric: _FineMetric) -> bool:
        return (
            metric.surfel.independent_frame_support >= 3
            and metric.neighbors_40 >= 12
            and metric.confidence >= 0.90
            and metric.curvature <= 0.01
            and metric.normal_spread_deg <= 7.5
        )

    @staticmethod
    def _eligible_10(metric: _FineMetric) -> bool:
        return (
            metric.surfel.independent_frame_support >= 2
            and metric.neighbors_30 >= 8
            and metric.confidence >= 0.75
            and metric.curvature <= 0.08
        )

    @staticmethod
    def _eligible_5(metric: _FineMetric) -> bool:
        return (
            metric.surfel.independent_frame_support >= 2
            and metric.neighbors_15 >= 6
            and metric.confidence >= 0.65
            and metric.curvature <= 0.20
        )

    def _assign_scales(self, metrics: tuple[_FineMetric, ...]) -> tuple[tuple[_MeshSamples, ...], int]:
        positions = np.asarray([metric.surfel.centroid for metric in metrics], dtype=np.float64)
        remaining = set(range(len(metrics)))
        samples: list[_MeshSamples] = []

        groups_20 = self._groups(positions, remaining, 0.02)
        eligible_20 = [indices for indices in groups_20.values() if all(self._eligible_20(metrics[index]) for index in indices)]
        consumed_20 = {index for group in eligible_20 for index in group}
        remaining.difference_update(consumed_20)
        samples.append(self._aggregate(0.02, eligible_20, metrics, positions, self._expanded_mask(positions, remaining, 0.02)))

        groups_10 = self._groups(positions, remaining, 0.01)
        eligible_10 = [indices for indices in groups_10.values() if all(self._eligible_10(metrics[index]) for index in indices)]
        consumed_10 = {index for group in eligible_10 for index in group}
        remaining.difference_update(consumed_10)
        samples.append(self._aggregate(0.01, eligible_10, metrics, positions, self._expanded_mask(positions, remaining, 0.01)))

        fine_indices = sorted(index for index in remaining if self._eligible_5(metrics[index]))
        unresolved = len(remaining) - len(fine_indices)
        fine_groups = [(index,) for index in fine_indices]
        samples.append(self._aggregate(0.005, fine_groups, metrics, positions, frozenset()))
        return tuple(reversed(samples)), unresolved

    @staticmethod
    def _groups(positions: np.ndarray, indices: set[int], scale_m: float) -> dict[tuple[int, int, int], tuple[int, ...]]:
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for index in sorted(indices):
            key = tuple(int(value) for value in np.floor(positions[index] / scale_m).astype(np.int64))
            grouped.setdefault(key, []).append(index)
        return {key: tuple(values) for key, values in sorted(grouped.items())}

    @staticmethod
    def _expanded_mask(positions: np.ndarray, indices: set[int], scale_m: float) -> frozenset[tuple[int, int, int]]:
        mask: set[tuple[int, int, int]] = set()
        for index in sorted(indices):
            base = np.floor(positions[index] / scale_m).astype(np.int64)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        mask.add((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
        return frozenset(mask)

    @staticmethod
    def _aggregate(
        scale_m: float,
        groups: list[tuple[int, ...]],
        metrics: tuple[_FineMetric, ...],
        positions: np.ndarray,
        exclusion_mask: frozenset[tuple[int, int, int]],
    ) -> _MeshSamples:
        aggregated_positions: list[np.ndarray] = []
        aggregated_normals: list[np.ndarray] = []
        sources: list[tuple[int, ...]] = []
        for group in sorted(groups, key=lambda item: tuple(metrics[index].surfel.key for index in item)):
            counts = np.asarray([metrics[index].surfel.observation_count for index in group], dtype=np.float64)
            position = np.average(positions[np.asarray(group)], axis=0, weights=counts)
            normal = np.average(np.asarray([metrics[index].normal for index in group]), axis=0, weights=counts)
            length = float(np.linalg.norm(normal))
            if length <= 1e-12:
                continue
            aggregated_positions.append(position)
            aggregated_normals.append(normal / length)
            sources.append(tuple(group))
        return _MeshSamples(
            scale_m=scale_m,
            positions=np.asarray(aggregated_positions, dtype=np.float64).reshape(-1, 3),
            normals=np.asarray(aggregated_normals, dtype=np.float64).reshape(-1, 3),
            source_indices=tuple(sources),
            exclusion_mask=exclusion_mask,
        )

    @staticmethod
    def _triangulate(samples: _MeshSamples) -> np.ndarray:
        if len(samples.positions) < 3:
            return np.empty((0, 3), dtype="<u4")
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(samples.positions)
        cloud.normals = o3d.utility.Vector3dVector(samples.normals)
        radii = o3d.utility.DoubleVector([samples.scale_m * value for value in (1.1, 1.8, 2.7)])
        candidate = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
        triangles = np.asarray(candidate.triangles, dtype=np.int64).reshape(-1, 3)
        accepted: list[np.ndarray] = []
        maximum_edge = samples.scale_m * 3.0 + 1e-9
        for triangle in triangles:
            if len(set(int(value) for value in triangle)) != 3:
                continue
            vertices = samples.positions[triangle]
            edges = vertices[[1, 2, 0]] - vertices[[0, 1, 2]]
            if float(np.linalg.norm(edges, axis=1).max()) > maximum_edge:
                continue
            face = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            area2 = float(np.linalg.norm(face))
            if not math.isfinite(area2) or area2 <= samples.scale_m * samples.scale_m * 1e-4:
                continue
            face /= area2
            normal = samples.normals[triangle].mean(axis=0)
            if abs(float(np.dot(face, normal))) < math.cos(math.radians(75.0)):
                continue
            if samples.exclusion_mask and LiveSurfaceMesher._intersects_mask(vertices, samples.scale_m, samples.exclusion_mask):
                continue
            if float(np.dot(face, normal)) < 0:
                triangle = triangle[[0, 2, 1]]
            accepted.append(triangle)
        return np.asarray(accepted, dtype="<u4").reshape(-1, 3)

    @staticmethod
    def _tangent_patches(
        samples: _MeshSamples,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(samples.positions) == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype="<u4"),
            )
        radius = samples.scale_m * 0.45
        angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
        axes = np.eye(3, dtype=np.float64)
        positions: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        triangles: list[tuple[int, int, int]] = []
        for centre, normal in zip(samples.positions, samples.normals):
            reference = axes[int(np.argmin(np.abs(normal)))]
            tangent_u = np.cross(normal, reference)
            tangent_u /= max(float(np.linalg.norm(tangent_u)), 1e-15)
            tangent_v = np.cross(normal, tangent_u)
            start = len(positions)
            positions.extend(
                centre
                + radius
                * (math.cos(angle) * tangent_u + math.sin(angle) * tangent_v)
                for angle in angles
            )
            normals.extend((normal, normal, normal))
            triangles.append((start, start + 1, start + 2))
        return (
            np.asarray(positions, dtype=np.float64).reshape(-1, 3),
            np.asarray(normals, dtype=np.float64).reshape(-1, 3),
            np.asarray(triangles, dtype="<u4").reshape(-1, 3),
        )

    @staticmethod
    def _intersects_mask(
        vertices: np.ndarray,
        scale_m: float,
        mask: frozenset[tuple[int, int, int]],
    ) -> bool:
        probes = [vertices.mean(axis=0)]
        for start, end in ((0, 1), (1, 2), (2, 0)):
            probes.extend(vertices[start] * (1.0 - fraction) + vertices[end] * fraction for fraction in (0.25, 0.5, 0.75))
        return any(tuple(int(value) for value in np.floor(point / scale_m).astype(np.int64)) in mask for point in probes)

    @staticmethod
    def _tile_revisions(
        positions: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        triangle_scales: np.ndarray,
        triangle_provenance: np.ndarray,
        triangle_uncertainty: np.ndarray,
        tile_m: float,
        unresolved: int,
        requested_tiles: Optional[set[TileId]],
    ) -> dict[TileId, MeshTileRevision]:
        grouped: dict[TileId, list[int]] = {}
        for index, triangle in enumerate(triangles):
            centre = positions[triangle].mean(axis=0)
            tile_key = np.floor(centre / tile_m).astype(np.int64)
            tile = TileId(*(int(value) for value in tile_key))
            if requested_tiles is None or tile in requested_tiles:
                grouped.setdefault(tile, []).append(index)
        revisions: dict[TileId, MeshTileRevision] = {}
        for tile, triangle_indices in sorted(grouped.items()):
            selected_triangles = triangles[np.asarray(triangle_indices, dtype=np.int64)]
            used = np.unique(selected_triangles)
            remap = {int(old): new for new, old in enumerate(used)}
            local_triangles = np.asarray(
                [[remap[int(index)] for index in triangle] for triangle in selected_triangles], dtype="<u4"
            )
            local_positions = np.asarray(positions[used], dtype="<f4")
            local_normals = np.asarray(normals[used], dtype="<f4")
            local_scales = np.asarray(triangle_scales[np.asarray(triangle_indices)], dtype="<f4")
            local_provenance = np.asarray(
                triangle_provenance[np.asarray(triangle_indices)], dtype="u1"
            )
            local_uncertainty = np.asarray(
                triangle_uncertainty[np.asarray(triangle_indices)], dtype="<f4"
            )
            digest = hashlib.sha256()
            digest.update(json.dumps(tile.as_tuple(), separators=(",", ":")).encode("ascii"))
            digest.update(local_positions.tobytes())
            digest.update(local_normals.tobytes())
            digest.update(local_triangles.tobytes())
            digest.update(local_scales.tobytes())
            digest.update(local_provenance.tobytes())
            digest.update(local_uncertainty.tobytes())
            revision_id = digest.hexdigest()
            revisions[tile] = MeshTileRevision(
                tile_id=tile,
                revision_id=revision_id,
                positions=local_positions,
                normals=local_normals,
                triangles=local_triangles,
                triangle_scale_m=local_scales,
                unresolved_surfel_count=unresolved,
                triangle_provenance=local_provenance,
                triangle_uncertainty_m=local_uncertainty,
            )
        return revisions
