"""Adaptive texture density, deterministic chart packing, and V1 LOD policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DensityDecision:
    accepted: bool
    mm_per_texel: float | None
    geometric_detail_mm: float
    source_pixel_footprint_mm: float
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ChartPlacement:
    chart_id: int
    atlas_index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PackedAtlases:
    atlas_sizes: tuple[tuple[int, int], ...]
    placements: tuple[ChartPlacement, ...]
    padding_pixels: int
    maximum_dimension: int


@dataclass(frozen=True)
class LodPolicy:
    name: str
    order: int
    mm_per_texel: float
    minimum_distance_m: float
    maximum_distance_m: float | None
    source_triangle_ids: tuple[int, ...]

    def core(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "mm_per_texel": self.mm_per_texel,
            "minimum_distance_m": self.minimum_distance_m,
            "maximum_distance_m": self.maximum_distance_m,
            "source_triangle_ids": list(self.source_triangle_ids),
        }


def select_adaptive_density(
    *,
    geometric_detail_m: float,
    source_pixel_footprint_m: float,
) -> DensityDecision:
    detail = _positive_finite(geometric_detail_m, "geometric_detail_m") * 1000.0
    source = _positive_finite(source_pixel_footprint_m, "source_pixel_footprint_m") * 1000.0
    if source > 5.0 + 1e-12:
        return DensityDecision(False, None, detail, source, ("SOURCE_SAMPLING_TOO_COARSE",))
    selected = min(5.0, max(3.0, detail, source))
    return DensityDecision(True, float(selected), detail, source, ())


def pack_charts_v1(
    chart_sizes_px: Mapping[int, tuple[int, int]],
    *,
    padding_pixels: int,
    maximum_dimension: int = 4096,
) -> PackedAtlases:
    padding = _nonnegative_integer(padding_pixels, "padding_pixels")
    maximum = _positive_integer(maximum_dimension, "maximum_dimension")
    if maximum > 4096:
        raise ValueError("maximum_dimension cannot exceed 4096")
    charts: list[tuple[int, int, int]] = []
    for chart_id, size in chart_sizes_px.items():
        identifier = _nonnegative_integer(chart_id, "chart_id")
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("chart size must be a width/height tuple")
        width = _positive_integer(size[0], "chart width")
        height = _positive_integer(size[1], "chart height")
        if width + 2 * padding > maximum or height + 2 * padding > maximum:
            raise ValueError("chart cannot fit in an atlas with configured padding")
        charts.append((identifier, width, height))
    if len({value[0] for value in charts}) != len(charts):
        raise ValueError("chart IDs must be unique")
    charts.sort(key=lambda value: (-(value[1] * value[2]), -max(value[1], value[2]), value[0]))

    atlases: list[dict] = []
    placements: list[ChartPlacement] = []
    for chart_id, width, height in charts:
        outer_width = width + 2 * padding
        outer_height = height + 2 * padding
        placed = False
        for atlas_index, atlas in enumerate(atlases):
            for shelf in atlas["shelves"]:
                if outer_height <= shelf["height"] and shelf["x"] + outer_width <= maximum:
                    x = shelf["x"]
                    y = shelf["y"]
                    shelf["x"] += outer_width
                    atlas["used_width"] = max(atlas["used_width"], shelf["x"])
                    placements.append(ChartPlacement(chart_id, atlas_index, x + padding, y + padding, width, height))
                    placed = True
                    break
            if placed:
                break
            next_y = atlas["used_height"]
            if next_y + outer_height <= maximum:
                atlas["shelves"].append({"y": next_y, "height": outer_height, "x": outer_width})
                atlas["used_height"] += outer_height
                atlas["used_width"] = max(atlas["used_width"], outer_width)
                placements.append(ChartPlacement(chart_id, atlas_index, padding, next_y + padding, width, height))
                placed = True
                break
        if not placed:
            atlas_index = len(atlases)
            atlases.append(
                {
                    "shelves": [{"y": 0, "height": outer_height, "x": outer_width}],
                    "used_width": outer_width,
                    "used_height": outer_height,
                }
            )
            placements.append(ChartPlacement(chart_id, atlas_index, padding, padding, width, height))
    atlas_sizes = tuple((int(value["used_width"]), int(value["used_height"])) for value in atlases)
    return PackedAtlases(
        atlas_sizes=atlas_sizes,
        placements=tuple(sorted(placements, key=lambda value: value.chart_id)),
        padding_pixels=padding,
        maximum_dimension=maximum,
    )


def build_lod_policies(
    source_triangle_ids: Sequence[int],
    *,
    near_mm_per_texel: float,
    medium_distance_m: float = 3.0,
    far_distance_m: float = 8.0,
) -> tuple[LodPolicy, ...]:
    near_density = _finite_range(near_mm_per_texel, 3.0, 5.0, "near_mm_per_texel")
    medium_distance = _positive_finite(medium_distance_m, "medium_distance_m")
    far_distance = _positive_finite(far_distance_m, "far_distance_m")
    if far_distance <= medium_distance:
        raise ValueError("far_distance_m must be greater than medium_distance_m")
    source_ids = tuple(sorted(source_triangle_ids))
    if not source_ids or len(set(source_ids)) != len(source_ids) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in source_ids
    ):
        raise ValueError("source_triangle_ids must contain unique non-negative integers")
    medium_ids = source_ids[::2]
    far_ids = medium_ids[::2]
    medium_density = min(5.0, near_density + 1.0)
    return (
        LodPolicy("near", 0, near_density, 0.0, medium_distance, source_ids),
        LodPolicy("medium", 1, medium_density, medium_distance, far_distance, medium_ids),
        LodPolicy("far", 2, 5.0, far_distance, None, far_ids),
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_range(value: float, minimum: float, maximum: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: int, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result
