"""Occlusion-aware, multi-view linear-RGB texture accumulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True)
class ColorObservation:
    linear_rgb: tuple[float, float, float]
    depth_residual_m: float
    range_m: float
    incidence_cosine: float
    sharpness: float
    occluded: bool
    view_origin_xyz: tuple[float, float, float]
    view_direction_xyz: tuple[float, float, float]
    frame_sha256: str
    calibration_id: str


@dataclass(frozen=True)
class VertexColor:
    vertex_id: str
    mean_linear_rgb: tuple[float, float, float]
    channel_stddev: tuple[float, float, float]
    accepted_count: int
    rejected_count: int
    independent_views: int
    source_frame_sha256: tuple[str, ...]
    accepted_for_texture: bool


@dataclass
class _ColorState:
    weight: float
    mean: np.ndarray
    m2: np.ndarray
    accepted_count: int
    rejected_count: int
    origins: list[np.ndarray]
    directions: list[np.ndarray]
    sources: set[str]


class ColorAccumulator:
    def __init__(self, calibration_id: str) -> None:
        if not calibration_id:
            raise ValueError("calibration_id is required")
        config = load_reconstruction_config()
        self.gates = config.appearance.texture
        self.view_gates = config.fusion.independent_view
        self.calibration_id = calibration_id
        self._vertices: dict[str, _ColorState] = {}

    def observe(self, vertex_id: str, observation: ColorObservation) -> bool:
        if not vertex_id:
            raise ValueError("vertex_id is required")
        state = self._vertices.setdefault(
            vertex_id,
            _ColorState(0.0, np.zeros(3), np.zeros(3), 0, 0, [], [], set()),
        )
        color = np.asarray(observation.linear_rgb, dtype=np.float64)
        valid = (
            observation.calibration_id == self.calibration_id
            and _SHA256_RE.fullmatch(observation.frame_sha256) is not None
            and color.shape == (3,)
            and np.all(np.isfinite(color))
            and np.all((0.0 <= color) & (color <= 1.0))
            and not observation.occluded
            and self.gates.accepts_sample(
                depth_residual_m=observation.depth_residual_m,
                range_m=observation.range_m,
            )
            and math.isfinite(observation.incidence_cosine)
            and observation.incidence_cosine > 0.0
            and math.isfinite(observation.sharpness)
            and observation.sharpness > 0.0
        )
        origin = np.asarray(observation.view_origin_xyz, dtype=np.float64)
        direction = np.asarray(observation.view_direction_xyz, dtype=np.float64)
        if origin.shape != (3,) or direction.shape != (3,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            valid = False
        direction_norm = float(np.linalg.norm(direction)) if direction.shape == (3,) else 0.0
        if direction_norm <= 1e-12:
            valid = False
        if not valid:
            state.rejected_count += 1
            return False

        direction = direction / direction_norm
        weight = max(float(observation.incidence_cosine), 0.05) * max(float(observation.sharpness), 0.05)
        weight /= max(float(observation.range_m) ** 2, 0.25)
        new_weight = state.weight + weight
        delta = color - state.mean
        new_mean = state.mean + delta * (weight / new_weight)
        state.m2 += weight * delta * (color - new_mean)
        state.mean = new_mean
        state.weight = new_weight
        state.accepted_count += 1
        state.origins.append(origin)
        state.directions.append(direction)
        state.sources.add(observation.frame_sha256.lower())
        return True

    def vertex(self, vertex_id: str) -> VertexColor:
        state = self._vertices.get(vertex_id)
        if state is None:
            return VertexColor(vertex_id, (0.0, 0.0, 0.0), (math.inf,) * 3, 0, 0, 0, (), False)
        variance = state.m2 / max(state.weight, 1e-12)
        stddev = np.sqrt(np.maximum(variance, 0.0))
        independent = self._independent_views(state.origins, state.directions)
        accepted = self.gates.accepts_vertex(
            independent_views=independent,
            channel_stddev=float(np.max(stddev)),
        )
        return VertexColor(
            vertex_id=vertex_id,
            mean_linear_rgb=tuple(float(value) for value in state.mean),
            channel_stddev=tuple(float(value) for value in stddev),
            accepted_count=state.accepted_count,
            rejected_count=state.rejected_count,
            independent_views=independent,
            source_frame_sha256=tuple(sorted(state.sources)),
            accepted_for_texture=accepted,
        )

    def vertices(self, vertex_ids: list[str] | tuple[str, ...]) -> tuple[VertexColor, ...]:
        """Return immutable values aligned to the requested geometry order."""

        return tuple(self.vertex(vertex_id) for vertex_id in tuple(vertex_ids))

    def delivery_report(self, measured_vertex_ids: list[str] | tuple[str, ...]) -> dict:
        total = len(measured_vertex_ids)
        accepted = sum(self.vertex(vertex_id).accepted_for_texture for vertex_id in measured_vertex_ids)
        coverage = accepted / total if total else 0.0
        return {
            "measured_vertex_count": total,
            "textured_vertex_count": accepted,
            "surface_coverage": coverage,
            "delivery_authorized": self.gates.delivery_authorized(surface_coverage=coverage),
            "calibration_id": self.calibration_id,
        }

    def _independent_views(self, origins: list[np.ndarray], directions: list[np.ndarray]) -> int:
        selected: list[int] = []
        for index, (origin, direction) in enumerate(zip(origins, directions)):
            if not selected:
                selected.append(index)
                continue
            independent = True
            for previous in selected:
                baseline = float(np.linalg.norm(origin - origins[previous]))
                cosine = float(np.clip(np.dot(direction, directions[previous]), -1.0, 1.0))
                separation = math.degrees(math.acos(cosine))
                if not self.view_gates.accepts(
                    origin_baseline_m=baseline,
                    direction_separation_deg=separation,
                ):
                    independent = False
                    break
            if independent:
                selected.append(index)
        return len(selected)
