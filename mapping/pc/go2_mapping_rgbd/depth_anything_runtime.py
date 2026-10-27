"""Production wiring for local-only, physically anchored Depth Anything assistance."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config

from .depth_anything import (
    DA3_SOURCE_REVISION,
    DepthAnything3Provider,
    DepthAnythingError,
    RelativeDepthFrame,
    TransformersDepthAnythingProvider,
)
from .depth_anything_worker import (
    DepthAnythingWorkItem,
    DepthAnythingWorker,
    DepthAnythingWorkResult,
)
from .monocular_anchors import ConsensusAnchorBuilder
from .monocular_validation import (
    AnchorSeedIdentity,
    bootstrap_stability,
    compute_metric_depth_uncertainty,
    fit_metric_depth,
    stratified_holdout,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ZERO_SHA256 = "0" * 64


class DepthAnythingProcessor:
    """Runs deterministic metric validation before any augmentation can publish."""

    def __init__(self, augmentation_store: Any) -> None:
        self._augmentation_store = augmentation_store
        self._anchor_builder = ConsensusAnchorBuilder()

    def __call__(
        self,
        item: DepthAnythingWorkItem,
        relative: RelativeDepthFrame,
    ) -> DepthAnythingWorkResult:
        height, width = relative.inverse_depth.shape
        anchor_result = self._anchor_builder.build(
            relative,
            item.anchors,
            calibration_sha256=item.calibration_sha256,
            measured_generation=item.source_generation_id,
            rectified_rgb=item.rectified_rgb,
        )
        seed = AnchorSeedIdentity(
            item.model_revision_id,
            item.frame_sha256,
            item.assistance_config_sha256,
            width,
            height,
        )
        partition = stratified_holdout(anchor_result.anchors, seed_identity=seed)
        fit = fit_metric_depth(partition, model_revision_id=item.model_revision_id)
        bootstrap = bootstrap_stability(
            partition,
            fit,
            model_revision_id=item.model_revision_id,
            seed_identity=seed,
        )

        physical = np.full((height, width), np.nan, dtype=np.float64)
        distance = np.zeros((height, width), dtype=np.float64)
        boundary = np.zeros((height, width), dtype=bool)
        for anchor in anchor_result.anchors:
            physical[anchor.pixel_y, anchor.pixel_x] = anchor.metric_depth_m
            boundary[anchor.pixel_y, anchor.pixel_x] = True
        uncertainty = compute_metric_depth_uncertainty(
            relative.inverse_depth,
            fit,
            bootstrap,
            physical_depth_m=physical,
            anchor_distance_m=distance,
            boundary_accepted=boundary,
        )
        valid_uncertainty = uncertainty.sigma_total_m[uncertainty.valid]
        maximum_uncertainty = (
            None if valid_uncertainty.size == 0 else float(np.max(valid_uncertainty))
        )
        total_cells = load_reconstruction_config().appearance.depth_anything
        metrics = {
            "surviving_anchor_count": len(anchor_result.anchors),
            "occupied_cells": len(partition.occupied_cells),
            "total_cells": total_cells.grid_columns * total_cells.grid_rows,
            "quadrants": [value in partition.quadrants_in_fit for value in range(4)],
            "held_out_median_m": fit.holdout_median_absolute_residual_m,
            "held_out_p95_m": fit.holdout_p95_absolute_residual_m,
            "held_out_signed_bias_m": fit.holdout_absolute_bias_m,
            "scale": fit.scale,
            "relative_scale_stddev": bootstrap.scale_stddev_fraction,
            "maximum_uncertainty_m": maximum_uncertainty,
            "fit_revision_id": fit.revision_id,
            "bootstrap_replicates": bootstrap.replicates,
            "model_confidence_p10": None,
            "model_confidence_median": None,
        }
        if relative.model_confidence is not None:
            metrics["model_confidence_p10"] = float(np.percentile(relative.model_confidence, 10.0))
            metrics["model_confidence_median"] = float(np.median(relative.model_confidence))
        blockers = tuple(dict.fromkeys((
            *anchor_result.blockers,
            *partition.blockers,
            *fit.blockers,
            *bootstrap.blockers,
            "MULTI_FRAME_BOUNDARY_EVIDENCE_REQUIRED",
        )))
        coverage = float(np.count_nonzero(uncertainty.valid) / uncertainty.valid.size)
        return DepthAnythingWorkResult(
            accepted=False,
            fit_accepted=fit.accepted and bootstrap.accepted and uncertainty.accepted,
            assistance_coverage=coverage,
            blockers=blockers,
            augmentation_reference=None,
            validation_metrics=metrics,
        )


def create_depth_anything_worker_from_environment(
    environment: Mapping[str, str] | None,
    *,
    augmentation_store: Any,
    provider_factory: Callable[..., Any] | None = None,
    assistance_config_sha256: str | None = None,
) -> DepthAnythingWorker | None:
    """Create the production worker only from explicit verified local inputs."""

    values = os.environ if environment is None else environment
    if values.get("GO2_MAPPING_DEPTH_ANYTHING_ENABLED") != "1":
        return None
    weights_path = values.get("GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_PATH", "")
    weights_sha256 = values.get("GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256", "")
    if (
        not weights_path
        or _SHA256.fullmatch(weights_sha256) is None
        or weights_sha256 == _ZERO_SHA256
        or not Path(weights_path).exists()
    ):
        return None
    gates = load_reconstruction_config().appearance.depth_anything
    backend = values.get("GO2_MAPPING_DEPTH_ANYTHING_BACKEND", "transformers")
    if backend not in {"transformers", "da3"}:
        return None
    if provider_factory is None:
        provider_factory = DepthAnything3Provider if backend == "da3" else TransformersDepthAnythingProvider
    model_name = values.get("GO2_MAPPING_DEPTH_ANYTHING_MODEL_NAME", "")
    if not model_name:
        model_name = "DA3-BASE" if backend == "da3" else gates.model_name
    preprocessing_revision = (
        "depth-anything-3-preprocess-v1" if backend == "da3" else gates.preprocessing_revision
    )
    input_size = (504, 504) if backend == "da3" else gates.input_size
    try:
        provider_arguments = {
            "weights_path": weights_path,
            "weights_sha256": weights_sha256,
            "model_name": model_name,
            "preprocessing_revision": preprocessing_revision,
            "input_size": input_size,
            "device": values.get("GO2_MAPPING_DEPTH_ANYTHING_DEVICE", "cpu"),
            "precision": values.get("GO2_MAPPING_DEPTH_ANYTHING_PRECISION", "float32"),
        }
        if backend == "da3":
            source_revision = values.get("GO2_MAPPING_DA3_SOURCE_REVISION", DA3_SOURCE_REVISION)
            if source_revision != DA3_SOURCE_REVISION:
                return None
            provider_arguments.update({
                "source_revision": source_revision,
                "gpu_memory_budget_gib": float(values.get("GO2_MAPPING_DA3_GPU_MEMORY_BUDGET_GIB", "9.5")),
                "inference_timeout_s": float(values.get("GO2_MAPPING_DA3_INFERENCE_TIMEOUT_S", "30")),
            })
        provider = provider_factory(
            **provider_arguments,
        )
    except (DepthAnythingError, OSError, TypeError, ValueError):
        return None
    worker = DepthAnythingWorker(
        provider,
        DepthAnythingProcessor(augmentation_store),
        assistance_config_sha256=assistance_config_sha256,
    )
    return worker
