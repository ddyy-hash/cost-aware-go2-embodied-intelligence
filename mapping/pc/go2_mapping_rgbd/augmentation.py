"""Separately labeled bounded RGB-D hole augmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import AugmentationGates, load_reconstruction_config


@dataclass(frozen=True)
class AugmentedLayer:
    name: str
    positions: np.ndarray
    uncertainty_m: np.ndarray
    geometry_authority: bool
    source: str
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint32))
    source_vertex_ids: tuple[str, ...] = ()
    source_triangle_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        uncertainty = np.asarray(self.uncertainty_m, dtype=np.float64).reshape(-1)
        if len(positions) != len(uncertainty):
            raise ValueError("augmentation needs one uncertainty per position")
        normals = np.asarray(self.normals, dtype=np.float64)
        if normals.size == 0:
            normals = np.zeros_like(positions)
        normals = normals.reshape(-1, 3)
        if normals.shape != positions.shape or not np.all(np.isfinite(normals)):
            raise ValueError("augmentation normals must be finite and match positions")
        triangles = np.asarray(self.triangles)
        if triangles.size == 0:
            triangles = np.empty((0, 3), dtype=np.uint32)
        if triangles.ndim != 2 or triangles.shape[1:] != (3,) or triangles.dtype.kind not in "iu":
            raise ValueError("augmentation triangles must be an N by 3 integer array")
        if np.any(triangles < 0) or (triangles.size and int(np.max(triangles)) >= len(positions)):
            raise ValueError("augmentation triangles must reference existing positions")
        triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        source_vertex_ids = self.source_vertex_ids or tuple(
            f"compat-vertex-{index}" for index in range(len(positions))
        )
        source_triangle_ids = self.source_triangle_ids or tuple(
            f"compat-triangle-{index}" for index in range(len(triangles))
        )
        if len(source_vertex_ids) != len(positions):
            raise ValueError("augmentation needs one source vertex ID per position")
        if len(source_triangle_ids) != len(triangles):
            raise ValueError("augmentation needs one source triangle ID per triangle")
        if any(not isinstance(value, str) or not value for value in source_vertex_ids + source_triangle_ids):
            raise ValueError("augmentation source IDs must be non-empty strings")
        positions.setflags(write=False)
        uncertainty.setflags(write=False)
        normals.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "uncertainty_m", uncertainty)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "source_vertex_ids", tuple(source_vertex_ids))
        object.__setattr__(self, "source_triangle_ids", tuple(source_triangle_ids))


@dataclass(frozen=True)
class BoundedHoleComponent:
    pixel_ids: tuple[tuple[int, int], ...]
    metric_positions: np.ndarray
    uncertainty_m: np.ndarray
    support_frame_ids: tuple[str, ...]
    support_view_ids: tuple[str, ...]
    maximum_boundary_distance_m: float
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        positions = np.ascontiguousarray(self.metric_positions, dtype=np.float64).reshape(-1, 3)
        uncertainty = np.ascontiguousarray(self.uncertainty_m, dtype=np.float64).reshape(-1)
        if len(self.pixel_ids) != len(positions) or len(positions) != len(uncertainty):
            raise ValueError("bounded component arrays must align with pixel IDs")
        positions.setflags(write=False)
        uncertainty.setflags(write=False)
        object.__setattr__(self, "metric_positions", positions)
        object.__setattr__(self, "uncertainty_m", uncertainty)
        object.__setattr__(self, "support_frame_ids", tuple(self.support_frame_ids))
        object.__setattr__(self, "support_view_ids", tuple(self.support_view_ids))
        object.__setattr__(self, "blockers", tuple(dict.fromkeys(self.blockers)))

    @property
    def accepted(self) -> bool:
        return not self.blockers


def bounded_hole_components(
    candidate_mask: np.ndarray,
    *,
    accepted_boundary_mask: np.ndarray,
    metric_positions: np.ndarray,
    uncertainty_m: np.ndarray,
    boundary_distance_m: np.ndarray,
    support_frame_ids: Mapping[tuple[int, int], Sequence[str]],
    support_view_ids: Mapping[tuple[int, int], Sequence[str]],
    tile_edge_mask: np.ndarray | None = None,
    mesh_edge_mask: np.ndarray | None = None,
    occlusion_edge_mask: np.ndarray | None = None,
    depth_discontinuity_edge_mask: np.ndarray | None = None,
    invalid_projection_mask: np.ndarray | None = None,
    missing_data_mask: np.ndarray | None = None,
    invalid_projection_or_missing_data_mask: np.ndarray | None = None,
    gates: AugmentationGates | None = None,
) -> tuple[BoundedHoleComponent, ...]:
    """Validate each four-connected candidate component without inventing surfaces."""

    candidate = _mask(candidate_mask, "candidate_mask")
    shape = candidate.shape
    boundary = _mask(accepted_boundary_mask, "accepted_boundary_mask", shape)
    positions = np.asarray(metric_positions, dtype=np.float64)
    uncertainty = np.asarray(uncertainty_m, dtype=np.float64)
    distance = np.asarray(boundary_distance_m, dtype=np.float64)
    if positions.shape != shape + (3,):
        raise ValueError("metric_positions must have shape H by W by 3")
    if uncertainty.shape != shape or distance.shape != shape:
        raise ValueError("uncertainty and boundary distance must match candidate_mask")

    forbidden_masks = tuple(
        _mask(value, name, shape)
        for name, value in (
            ("tile_edge_mask", tile_edge_mask),
            ("mesh_edge_mask", mesh_edge_mask),
            ("occlusion_edge_mask", occlusion_edge_mask),
            ("depth_discontinuity_edge_mask", depth_discontinuity_edge_mask),
            ("invalid_projection_mask", invalid_projection_mask),
            ("missing_data_mask", missing_data_mask),
            ("invalid_projection_or_missing_data_mask", invalid_projection_or_missing_data_mask),
        )
        if value is not None
    )
    active_gates = gates or load_reconstruction_config().appearance.augmentation
    height, width = shape
    seen = np.zeros(shape, dtype=bool)
    results: list[BoundedHoleComponent] = []

    for start_y, start_x in np.argwhere(candidate):
        if seen[start_y, start_x]:
            continue
        queue = deque([(int(start_x), int(start_y))])
        seen[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            pixels.append((x, y))
            for nx, ny in _neighbors(x, y):
                if 0 <= nx < width and 0 <= ny < height and candidate[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        pixels.sort(key=lambda value: (value[1], value[0]))
        pixel_set = set(pixels)
        boundary_pixels: set[tuple[int, int]] = set()
        touches_image_edge = False
        for x, y in pixels:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                touches_image_edge = True
            for nx, ny in _neighbors(x, y):
                if not (0 <= nx < width and 0 <= ny < height):
                    touches_image_edge = True
                elif (nx, ny) not in pixel_set:
                    boundary_pixels.add((nx, ny))

        blockers: list[str] = []
        if touches_image_edge:
            blockers.append("IMAGE_EDGE_CONTACT")
        if not boundary_pixels or any(not boundary[y, x] for x, y in boundary_pixels):
            blockers.append("COMPLETE_ACCEPTED_BOUNDARY_REQUIRED")
        checked_pixels = pixel_set | boundary_pixels
        if any(mask[y, x] for mask in forbidden_masks for x, y in checked_pixels):
            blockers.append("FORBIDDEN_BOUNDARY_EDGE")

        component_positions = np.asarray([positions[y, x] for x, y in pixels], dtype=np.float64)
        component_uncertainty = np.asarray([uncertainty[y, x] for x, y in pixels], dtype=np.float64)
        component_distance = np.asarray([distance[y, x] for x, y in pixels], dtype=np.float64)
        if not np.all(np.isfinite(component_positions)):
            blockers.append("FINITE_METRIC_POSITION_REQUIRED")
        if (
            not np.all(np.isfinite(component_uncertainty))
            or np.any(component_uncertainty < 0.0)
            or np.any(component_uncertainty > active_gates.maximum_depth_stddev_m)
        ):
            blockers.append("UNCERTAINTY_GATE_FAILED")
        if (
            not np.all(np.isfinite(component_distance))
            or np.any(component_distance < 0.0)
            or np.any(component_distance > active_gates.maximum_hole_radius_m)
        ):
            blockers.append("BOUNDARY_DISTANCE_GATE_FAILED")

        frames_by_pixel = [tuple(dict.fromkeys(support_frame_ids.get(pixel, ()))) for pixel in pixels]
        views_by_pixel = [tuple(dict.fromkeys(support_view_ids.get(pixel, ()))) for pixel in pixels]
        if any(len(values) < active_gates.minimum_depth_frames for values in frames_by_pixel):
            blockers.append("PER_POINT_FRAME_SUPPORT_REQUIRED")
        if any(len(values) < active_gates.minimum_independent_views for values in views_by_pixel):
            blockers.append("PER_POINT_VIEW_SUPPORT_REQUIRED")
        all_frames = tuple(sorted({value for values in frames_by_pixel for value in values}))
        all_views = tuple(sorted({value for values in views_by_pixel for value in values}))
        max_distance = float(np.max(component_distance)) if len(component_distance) else float("inf")
        results.append(
            BoundedHoleComponent(
                pixel_ids=tuple(pixels),
                metric_positions=component_positions,
                uncertainty_m=component_uncertainty,
                support_frame_ids=all_frames,
                support_view_ids=all_views,
                maximum_boundary_distance_m=max_distance,
                blockers=tuple(blockers),
            )
        )
    return tuple(results)


def _mask(value: np.ndarray, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or (shape is not None and result.shape != shape):
        expected = "a two-dimensional array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must have {expected}")
    return np.asarray(result, dtype=bool)


def _neighbors(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))


class AugmentationBuilder:
    def __init__(self, gates: AugmentationGates | None = None) -> None:
        self.gates = gates or load_reconstruction_config().appearance.augmentation

    def fill(
        self,
        positions: np.ndarray,
        *,
        quality_level: str,
        depth_region_accepted: bool,
        depth_frame_count: int,
        independent_views: int,
        depth_stddev_m: float,
        completely_bounded: bool,
        maximum_boundary_distance_m: float,
    ) -> AugmentedLayer:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("augmentation positions must be a finite N by 3 array")
        accepted = self.gates.accepts(
            quality_level=quality_level,
            depth_region_accepted=depth_region_accepted,
            depth_frame_count=depth_frame_count,
            independent_views=independent_views,
            depth_stddev_m=depth_stddev_m,
            completely_bounded=completely_bounded,
            hole_radius_m=maximum_boundary_distance_m,
        )
        if not accepted:
            raise PermissionError("RGB-D hole does not satisfy augmentation authority gates")
        uncertainty = np.full(len(points), float(depth_stddev_m), dtype=np.float64)
        return AugmentedLayer(
            name="RGBD_AUGMENTED",
            positions=points,
            uncertainty_m=uncertainty,
            geometry_authority=False,
            source="CALIBRATED_DEPTH_VALIDATED",
        )
