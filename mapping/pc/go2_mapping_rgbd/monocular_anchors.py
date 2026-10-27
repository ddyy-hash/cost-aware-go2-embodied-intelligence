"""Exact consensus anchors for deterministic monocular metric validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .depth_anything import RelativeDepthFrame
from .depth_validation import AcceptedDepthCorrespondence


_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


@dataclass(frozen=True)
class ConsensusAnchor:
    pixel_x: int
    pixel_y: int
    lidar_point_index: int
    model_inverse_depth: float
    metric_depth_m: float
    point_camera_xyz: tuple[float, float, float]
    frame_sha256: str
    accepted_region_id: str


@dataclass(frozen=True)
class AnchorBuildResult:
    anchors: tuple[ConsensusAnchor, ...]
    calibration_sha256: str
    measured_generation: str
    blockers: tuple[str, ...]


class ConsensusAnchorBuilder:
    def build(
        self,
        relative_depth: RelativeDepthFrame,
        correspondences: Sequence[AcceptedDepthCorrespondence],
        *,
        calibration_sha256: str,
        measured_generation: str,
        rectified_rgb: np.ndarray,
        dynamic_mask: np.ndarray | None = None,
        occlusion_mask: np.ndarray | None = None,
        free_space_mask: np.ndarray | None = None,
        depth_discontinuity_mask: np.ndarray | None = None,
    ) -> AnchorBuildResult:
        if not isinstance(calibration_sha256, str) or _SHA256.fullmatch(calibration_sha256) is None:
            raise ValueError("calibration_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(measured_generation, str) or not measured_generation:
            raise ValueError("measured_generation must be non-empty")
        rgb = np.asarray(rectified_rgb)
        shape = relative_depth.inverse_depth.shape
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("rectified_rgb must be uint8 and aligned to the model grid")
        masks = [
            self._mask(dynamic_mask, shape, "dynamic_mask"),
            self._mask(occlusion_mask, shape, "occlusion_mask"),
            self._mask(free_space_mask, shape, "free_space_mask"),
            self._mask(depth_discontinuity_mask, shape, "depth_discontinuity_mask"),
        ]
        ordered = sorted(
            tuple(correspondences),
            key=lambda value: (value.lidar_point_index, value.color_pixel[1], value.color_pixel[0]),
        )
        blockers = []
        if any(item.calibration_sha256 != calibration_sha256 for item in ordered):
            blockers.append("CALIBRATION_IDENTITY_MISMATCH")
        if any(item.measured_generation != measured_generation for item in ordered):
            blockers.append("MEASURED_GENERATION_MISMATCH")
        if any(item.frame_sha256 != relative_depth.frame_sha256 for item in ordered):
            blockers.append("MODEL_FRAME_IDENTITY_MISMATCH")
        if blockers:
            return AnchorBuildResult((), calibration_sha256, measured_generation, tuple(blockers))

        anchors = []
        height, width = shape
        for item in ordered:
            x, y = item.color_pixel
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if any(mask[y, x] for mask in masks):
                continue
            color = rgb[y, x]
            if np.any(color == 0) or np.any(color == 255):
                continue
            model_value = float(relative_depth.inverse_depth[y, x])
            if not np.isfinite(model_value) or model_value <= 0.0:
                continue
            if item.residual_class != "CONSENSUS" or not item.accepted_region_id:
                continue
            anchors.append(
                ConsensusAnchor(
                    pixel_x=x,
                    pixel_y=y,
                    lidar_point_index=item.lidar_point_index,
                    model_inverse_depth=model_value,
                    metric_depth_m=item.observed_metric_depth_m,
                    point_camera_xyz=item.point_camera_xyz,
                    frame_sha256=item.frame_sha256,
                    accepted_region_id=item.accepted_region_id,
                )
            )
        if not anchors:
            blockers.append("NO_ACCEPTED_CONSENSUS_ANCHORS")
        return AnchorBuildResult(tuple(anchors), calibration_sha256, measured_generation, tuple(blockers))

    @staticmethod
    def _mask(value: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
        if value is None:
            return np.zeros(shape, dtype=bool)
        mask = np.asarray(value)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean model-grid mask")
        return mask
