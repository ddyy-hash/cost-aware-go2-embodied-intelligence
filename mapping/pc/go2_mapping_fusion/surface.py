"""Deterministic display surfaces derived only from measured LiDAR voxels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import SurfacingGates, load_reconstruction_config

from .voxel_map import VoxelEvidence, VoxelState
from .room_surface_filter import filter_room_surface_points


@dataclass(frozen=True)
class MeasuredSurface:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    source_vertex_ids: tuple[str, ...] = ()
    source_layer: str = "LIDAR_MEASURED"
    geometry_authority: bool = False
    point_authority: str = "LIDAR_MEASURED"
    source_states: tuple[str, ...] = (VoxelState.MEASURED.value,)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(self.triangles, dtype=np.int32).reshape(-1, 3)
        if len(normals) != len(positions):
            raise ValueError("surface needs one normal per position")
        source_vertex_ids = tuple(self.source_vertex_ids)
        if source_vertex_ids and len(source_vertex_ids) != len(positions):
            raise ValueError("surface needs one source vertex ID per position")
        if any(not isinstance(value, str) or not value for value in source_vertex_ids):
            raise ValueError("surface source vertex IDs must be non-empty strings")
        if len(triangles) and (np.min(triangles) < 0 or np.max(triangles) >= len(positions)):
            raise ValueError("surface triangle index lies outside positions")
        for value in (positions, normals, triangles):
            value.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "source_vertex_ids", source_vertex_ids)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    def bounded_snapshot(self) -> dict:
        return {
            "source_layer": self.source_layer,
            "geometry_authority": self.geometry_authority,
            "point_authority": self.point_authority,
            "source_vertex_ids": list(self.source_vertex_ids),
            "positions": self.positions.astype(np.float32).reshape(-1).tolist(),
            "normals": self.normals.astype(np.float32).reshape(-1).tolist(),
            "triangles": self.triangles.reshape(-1).tolist(),
        }


class MeasuredSurfaceBuilder:
    def __init__(self, gates: SurfacingGates | None = None) -> None:
        self.gates = gates or load_reconstruction_config().fusion.surfacing

    def build(self, voxels: list[VoxelEvidence] | tuple[VoxelEvidence, ...]) -> MeasuredSurface:
        measured = sorted((item for item in voxels if item.state is VoxelState.MEASURED), key=lambda item: item.key)
        if len(measured) > self.gates.display_point_budget:
            stride = int(np.ceil(len(measured) / self.gates.display_point_budget))
            measured = measured[::stride]
        positions = np.asarray([item.position for item in measured], dtype=np.float64).reshape(-1, 3)
        source_vertex_ids = tuple(f"{item.key[0]}:{item.key[1]}:{item.key[2]}" for item in measured)
        return self.build_observed_positions(positions, source_vertex_ids=source_vertex_ids)

    def build_observed_positions(
        self,
        positions: np.ndarray,
        *,
        source_vertex_ids: tuple[str, ...],
        source_layer: str = "LIDAR_MEASURED",
        point_authority: str = "LIDAR_MEASURED",
    ) -> MeasuredSurface:
        """Mesh explicit observed coordinates without inventing missing evidence.

        This is used for an already LIO-registered point-cloud topic.  The
        caller supplies stable source identities and the exact authority label;
        the mesher only connects local observed neighborhoods.
        """

        values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        identifiers = tuple(source_vertex_ids)
        if len(identifiers) != len(values):
            raise ValueError("observed positions need one stable source ID per point")
        if len(values) > self.gates.display_point_budget:
            stride = int(np.ceil(len(values) / self.gates.display_point_budget))
            values = values[::stride]
            identifiers = identifiers[::stride]
        filtered_values, keep = filter_room_surface_points(values)
        if len(keep):
            values = filtered_values
            identifiers = tuple(identifiers[int(index)] for index in keep)
        normals, triangles = self._mesh(values)
        if len(triangles) > self.gates.display_triangle_budget:
            stride = int(np.ceil(len(triangles) / self.gates.display_triangle_budget))
            triangles = triangles[::stride]
        return MeasuredSurface(
            values,
            normals,
            triangles,
            identifiers,
            source_layer=source_layer,
            geometry_authority=False,
            point_authority=point_authority,
        )

    def _mesh(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(positions) == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int32),
            )
        if len(positions) < 3:
            return (
                np.tile([0.0, 0.0, 1.0], (len(positions), 1)),
                np.empty((0, 3), dtype=np.int32),
            )

        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(positions)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.gates.voxel_size_m * 3.1,
                max_nn=30,
            )
        )
        try:
            cloud.orient_normals_consistent_tangent_plane(min(20, len(positions) - 1))
        except RuntimeError:
            cloud.orient_normals_towards_camera_location(positions.mean(axis=0))
        normals = np.asarray(cloud.normals, dtype=np.float64).reshape(-1, 3)
        candidate = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(
                [self.gates.voxel_size_m * value for value in (1.1, 1.8, 2.7)]
            ),
        )
        triangles = np.asarray(candidate.triangles, dtype=np.int32).reshape(-1, 3)
        maximum_edge = self.gates.voxel_size_m * 2.2 + 1e-9
        minimum_area2 = self.gates.voxel_size_m * self.gates.voxel_size_m * 1e-4
        minimum_alignment = math.cos(math.radians(75.0))
        keep: list[np.ndarray] = []
        for triangle in triangles:
            points = positions[triangle]
            edges = points[[1, 2, 0]] - points[[0, 1, 2]]
            if float(np.linalg.norm(edges, axis=1).max()) > maximum_edge:
                continue
            face_normal = np.cross(points[1] - points[0], points[2] - points[0])
            area2 = float(np.linalg.norm(face_normal))
            if not math.isfinite(area2) or area2 <= minimum_area2:
                continue
            face_normal /= area2
            edge_lengths = np.linalg.norm(edges, axis=1)
            if float(edge_lengths.max() / max(edge_lengths.min(), 1e-9)) > 4.0:
                continue
            vertex_normal = normals[triangle].mean(axis=0)
            if abs(float(np.dot(face_normal, vertex_normal))) < minimum_alignment:
                continue
            if float(np.dot(face_normal, vertex_normal)) < 0.0:
                triangle = triangle[[0, 2, 1]]
            keep.append(triangle)
        return normals, np.asarray(keep, dtype=np.int32).reshape(-1, 3)
