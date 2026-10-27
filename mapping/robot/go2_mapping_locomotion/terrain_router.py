"""Automatic terrain routing from a PGTT heightmap and operator velocity intent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class TerrainRouterError(ValueError):
    pass


@dataclass(frozen=True)
class TerrainRoute:
    state: str
    task: str
    policy_level: str
    confidence: float
    relief_m: float
    maximum_step_m: float
    executable: bool
    blocker: str | None = None


def _policy_level(relief_m: float) -> str:
    if relief_m <= 0.06:
        return "03"
    if relief_m <= 0.10:
        return "07"
    if relief_m <= 0.14:
        return "10"
    if relief_m <= 0.18:
        return "13"
    if relief_m <= 0.22:
        return "17"
    return "20"


def classify_heightmap(heightmap: Any, *, forward_intent: float) -> TerrainRoute:
    grid = np.asarray(heightmap, dtype=np.float32)
    if grid.shape != (11, 9) or not np.isfinite(grid).all():
        raise TerrainRouterError("heightmap must be finite with shape 11x9")
    intent = float(forward_intent)
    if not np.isfinite(intent) or abs(intent) > 1.0:
        raise TerrainRouterError("forward intent must be finite in [-1,1]")
    if abs(intent) < 0.08:
        return TerrainRoute("IDLE", "HOLD", "03", 1.0, 0.0, 0.0, False, "DIRECTION_ZERO")

    directed = grid if intent > 0 else grid[::-1]
    rows = np.median(directed, axis=1)
    # Compare the immediate support region with the furthest observable region.
    near = float(np.median(rows[3:6]))
    far = float(np.median(rows[8:11]))
    signed_relief = far - near
    relief = abs(signed_relief)
    deltas = np.diff(rows)
    directed_steps = deltas if signed_relief >= 0 else -deltas
    positive_steps = directed_steps[directed_steps > 0.045]
    maximum_step = float(positive_steps.max()) if positive_steps.size else 0.0
    lateral_mad = float(np.median(np.abs(directed - np.median(directed, axis=1, keepdims=True))))
    roughness = float(np.percentile(np.abs(deltas), 80)) + lateral_mad

    stair = relief >= 0.10 and positive_steps.size >= 2 and maximum_step <= 0.24
    if stair:
        ascending = signed_relief > 0
        state = "STAIR_UP" if ascending else "STAIR_DOWN"
        task = "STAIR_ASCEND" if ascending else "STAIR_DESCEND"
        confidence = min(0.99, 0.55 + relief * 1.5 + positive_steps.size * 0.06)
        executable = maximum_step <= 0.12
        blocker = None if executable else "STAIR_STEP_EXCEEDS_INITIAL_HARDWARE_GATE"
        return TerrainRoute(state, task, _policy_level(max(relief, maximum_step)), confidence, relief, maximum_step, executable, blocker)
    if relief >= 0.08 or roughness >= 0.075:
        confidence = min(0.95, 0.5 + max(relief, roughness) * 2.0)
        return TerrainRoute("ROUGH_TERRAIN", "ROUGH_TERRAIN", _policy_level(max(relief, roughness)), confidence, relief, maximum_step, True)
    return TerrainRoute("FLAT", "ROUGH_TERRAIN", "03", max(0.55, 1.0 - roughness * 5.0), relief, maximum_step, True)


def classify_stair_profile(profile: Any, *, forward_intent: float) -> TerrainRoute:
    """Classify an extended 21x9 forward profile without changing policy scale."""
    grid = np.asarray(profile, dtype=np.float32)
    if grid.shape != (21, 9) or not np.isfinite(grid).all():
        raise TerrainRouterError("stair profile must be finite with shape 21x9")
    intent = float(forward_intent)
    if not np.isfinite(intent) or abs(intent) < 0.08 or abs(intent) > 1.0:
        return TerrainRoute("IDLE", "HOLD", "03", 1.0, 0.0, 0.0, False, "DIRECTION_ZERO")
    directed = grid if intent > 0 else grid[::-1]
    rows = np.median(directed, axis=1)
    near = float(np.median(rows[2:6]))
    far = float(np.median(rows[-4:]))
    signed_relief = far - near
    relief = abs(signed_relief)
    directed_steps = np.diff(rows) if signed_relief >= 0 else -np.diff(rows)
    positive_steps = directed_steps[directed_steps > 0.045]
    maximum_step = float(positive_steps.max()) if positive_steps.size else 0.0
    if relief >= 0.10 and positive_steps.size >= 2 and maximum_step <= 0.24:
        ascending = signed_relief > 0
        return TerrainRoute("STAIR_UP" if ascending else "STAIR_DOWN", "STAIR_ASCEND" if ascending else "STAIR_DESCEND", _policy_level(max(relief, maximum_step)), min(0.99, 0.55 + relief * 1.5 + positive_steps.size * 0.03), relief, maximum_step, maximum_step <= 0.12, None if maximum_step <= 0.12 else "STAIR_STEP_EXCEEDS_INITIAL_HARDWARE_GATE")
    return TerrainRoute("FLAT", "ROUGH_TERRAIN", "03", 0.7, relief, maximum_step, True)


class StableTerrainRouter:
    """Require consecutive frames before changing the active terrain route."""

    def __init__(self, *, enter_frames: int = 8, exit_frames: int = 12, minimum_hold_frames: int = 25) -> None:
        if enter_frames <= 0 or exit_frames <= 0 or minimum_hold_frames < 0:
            raise TerrainRouterError("router confirmation windows must be positive")
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.minimum_hold_frames = minimum_hold_frames
        self.active: TerrainRoute | None = None
        self._candidate: TerrainRoute | None = None
        self._count = 0
        self._held_frames = 0

    def update(self, heightmap: Any, *, forward_intent: float) -> TerrainRoute:
        route = classify_heightmap(heightmap, forward_intent=forward_intent)
        if self.active is None:
            self.active = route
            self._held_frames = 1
            return route
        if route.state == "IDLE":
            self.active = route
            self._candidate = None
            self._count = 0
            self._held_frames = 0
            return route
        if route.state == self.active.state:
            self.active = route
            self._held_frames += 1
            self._candidate = None
            self._count = 0
            return route
        if self._held_frames < self.minimum_hold_frames and route.state != "IDLE":
            self._held_frames += 1
            return self.active
        if self._candidate is None or self._candidate.state != route.state:
            self._candidate = route
            self._count = 1
        else:
            self._candidate = route
            self._count += 1
        required = self.exit_frames if route.state in {"FLAT", "IDLE"} else self.enter_frames
        if self._count >= required:
            self.active = route
            self._held_frames = 0
            self._candidate = None
            self._count = 0
        return self.active


__all__ = ["StableTerrainRouter", "TerrainRoute", "TerrainRouterError", "classify_heightmap", "classify_stair_profile"]
