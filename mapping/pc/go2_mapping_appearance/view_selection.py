"""Deterministic validated color-view selection in linear RGB."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


@dataclass(frozen=True)
class TexelObservation:
    source_frame_sha256: str
    pixel_x: int
    pixel_y: int
    view_id: str
    linear_rgb: tuple[float, float, float]
    in_frustum: bool
    orientation_cosine: float
    physical_depth_accepted: bool
    consensus_depth_accepted: bool
    occluded: bool
    depth_residual_m: float
    maximum_depth_residual_m: float
    sharpness: float
    exposure: float

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.source_frame_sha256) is None:
            raise ValueError("source_frame_sha256 must be a lowercase SHA-256")
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError("view_id must be non-empty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.pixel_x, self.pixel_y)):
            raise ValueError("pixel coordinates must be non-negative integers")
        color = tuple(float(value) for value in self.linear_rgb)
        if len(color) != 3 or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in color):
            raise ValueError("linear_rgb must contain three finite normalized channels")
        object.__setattr__(self, "linear_rgb", color)


@dataclass(frozen=True)
class ValidatedTexel:
    accepted: bool
    linear_rgb: tuple[float, float, float] | None
    source_frame_sha256: tuple[str, ...]
    independent_view_ids: tuple[str, ...]
    blockers: tuple[str, ...]


def select_validated_texel(
    observations: Sequence[TexelObservation],
    *,
    minimum_independent_views: int = 2,
    maximum_linear_rgb_channel_stddev: float = 0.1,
    minimum_orientation_cosine: float = 1e-6,
    minimum_sharpness: float = 0.05,
    minimum_exposure: float = 0.05,
    maximum_exposure: float = 0.95,
) -> ValidatedTexel:
    if minimum_independent_views < 1:
        raise ValueError("minimum_independent_views must be positive")
    if not math.isfinite(maximum_linear_rgb_channel_stddev) or maximum_linear_rgb_channel_stddev < 0.0:
        raise ValueError("maximum color standard deviation must be finite and non-negative")
    valid = [value for value in observations if _valid(value, minimum_orientation_cosine, minimum_sharpness, minimum_exposure, maximum_exposure)]
    best_by_view: dict[str, TexelObservation] = {}
    for observation in valid:
        previous = best_by_view.get(observation.view_id)
        if previous is None or _score(observation) < _score(previous):
            best_by_view[observation.view_id] = observation
    selected = tuple(best_by_view[key] for key in sorted(best_by_view))
    frames = tuple(sorted(value.source_frame_sha256 for value in selected))
    views = tuple(sorted(value.view_id for value in selected))
    blockers: list[str] = []
    if len(selected) < minimum_independent_views:
        blockers.append("INSUFFICIENT_VALIDATED_VIEWS")
        return ValidatedTexel(False, None, frames, views, tuple(blockers))
    colors = np.asarray([value.linear_rgb for value in selected], dtype=np.float64)
    if np.any(np.std(colors, axis=0) > maximum_linear_rgb_channel_stddev + 1e-12):
        blockers.append("INCONSISTENT_VIEW_COLOR")
        return ValidatedTexel(False, None, frames, views, tuple(blockers))
    mean = tuple(float(value) for value in np.mean(colors, axis=0))
    return ValidatedTexel(True, mean, frames, views, ())


def _valid(
    value: TexelObservation,
    minimum_orientation_cosine: float,
    minimum_sharpness: float,
    minimum_exposure: float,
    maximum_exposure: float,
) -> bool:
    numbers = (
        value.orientation_cosine,
        value.depth_residual_m,
        value.maximum_depth_residual_m,
        value.sharpness,
        value.exposure,
    )
    return bool(
        isinstance(value, TexelObservation)
        and all(math.isfinite(number) for number in numbers)
        and value.in_frustum is True
        and value.orientation_cosine >= minimum_orientation_cosine
        and (value.physical_depth_accepted is True or value.consensus_depth_accepted is True)
        and value.occluded is False
        and value.maximum_depth_residual_m >= 0.0
        and abs(value.depth_residual_m) <= value.maximum_depth_residual_m
        and value.sharpness >= minimum_sharpness
        and minimum_exposure <= value.exposure <= maximum_exposure
    )


def _score(value: TexelObservation) -> tuple:
    return (
        -value.orientation_cosine,
        abs(value.depth_residual_m),
        -value.sharpness,
        abs(value.exposure - 0.5),
        value.source_frame_sha256,
        value.pixel_y,
        value.pixel_x,
    )
