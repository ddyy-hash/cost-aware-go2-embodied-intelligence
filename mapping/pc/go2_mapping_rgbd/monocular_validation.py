"""Deterministic spatial validation for metric monocular assistance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import threading
from typing import Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import (
    canonical_json_bytes,
    canonical_sha256,
    load_reconstruction_config,
)

from .monocular_anchors import ConsensusAnchor


_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


def _sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class AnchorSeedIdentity:
    model_revision_id: str
    frame_sha256: str
    appearance_config_sha256: str
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_revision_id", _sha256(self.model_revision_id, "model_revision_id"))
        object.__setattr__(self, "frame_sha256", _sha256(self.frame_sha256, "frame_sha256"))
        object.__setattr__(
            self,
            "appearance_config_sha256",
            _sha256(self.appearance_config_sha256, "appearance_config_sha256"),
        )
        for name in ("image_width", "image_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def seed_payload(self) -> dict[str, object]:
        return {
            "appearance_config_sha256": self.appearance_config_sha256,
            "domain": "holdout-v1",
            "frame_sha256": self.frame_sha256,
            "model_revision_id": self.model_revision_id,
        }


@dataclass(frozen=True)
class AnchorPartition:
    fit: tuple[ConsensusAnchor, ...]
    holdout: tuple[ConsensusAnchor, ...]
    occupied_cells: tuple[int, ...]
    quadrants_in_fit: tuple[int, ...]
    seed64: int
    accepted: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class MetricDepthFit:
    scale: float | None
    bias: float | None
    iterations: int
    inlier_ids: tuple[int, ...]
    fit_residual_scale: float | None
    holdout_median_absolute_residual_m: float | None
    holdout_p95_absolute_residual_m: float | None
    holdout_absolute_bias_m: float | None
    accepted: bool
    blockers: tuple[str, ...]
    revision_id: str


@dataclass(frozen=True)
class BootstrapStability:
    parameter_samples: np.ndarray
    seed64: int
    replicates: int
    scale_stddev_fraction: float | None
    accepted: bool
    blockers: tuple[str, ...]
    first_cell_samples: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        samples = np.asarray(self.parameter_samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("parameter_samples must be N by 2")
        samples = np.array(samples, dtype=np.float64, order="C", copy=True)
        samples.setflags(write=False)
        object.__setattr__(self, "parameter_samples", samples)


_BOOTSTRAP_CACHE: dict[str, BootstrapStability] = {}
_BOOTSTRAP_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class MetricDepthUncertainty:
    metric_depth_m: np.ndarray
    sigma_fit_m: np.ndarray
    sigma_residual_m: np.ndarray
    sigma_local_m: np.ndarray
    sigma_physical_m: np.ndarray
    sigma_anchor_m: np.ndarray
    sigma_total_m: np.ndarray
    valid: np.ndarray
    accepted: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = np.asarray(self.metric_depth_m).shape
        for name in (
            "metric_depth_m",
            "sigma_fit_m",
            "sigma_residual_m",
            "sigma_local_m",
            "sigma_physical_m",
            "sigma_anchor_m",
            "sigma_total_m",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} shape does not match metric depth")
            value = np.array(value, dtype=np.float64, order="C", copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        valid = np.asarray(self.valid, dtype=bool)
        if valid.shape != shape:
            raise ValueError("valid shape does not match metric depth")
        valid = np.array(valid, dtype=bool, order="C", copy=True)
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)


def stratified_holdout(
    anchors: Sequence[ConsensusAnchor],
    *,
    seed_identity: AnchorSeedIdentity,
) -> AnchorPartition:
    gates = load_reconstruction_config().appearance.depth_anything
    digest = hashlib.sha256(canonical_json_bytes(seed_identity.seed_payload)).digest()
    seed64 = int.from_bytes(digest[:8], "little", signed=False)
    ordered = tuple(
        sorted(
            tuple(anchors),
            key=lambda value: (value.pixel_y, value.pixel_x, value.lidar_point_index, value.frame_sha256),
        )
    )
    blockers = []
    if len(ordered) < gates.minimum_anchor_count:
        blockers.append("INSUFFICIENT_CONSENSUS_ANCHORS")

    cells: dict[int, list[ConsensusAnchor]] = {}
    for anchor in ordered:
        if (
            anchor.pixel_x < 0
            or anchor.pixel_y < 0
            or anchor.pixel_x >= seed_identity.image_width
            or anchor.pixel_y >= seed_identity.image_height
        ):
            blockers.append("ANCHOR_OUTSIDE_MODEL_GRID")
            continue
        column = min(gates.grid_columns - 1, anchor.pixel_x * gates.grid_columns // seed_identity.image_width)
        row = min(gates.grid_rows - 1, anchor.pixel_y * gates.grid_rows // seed_identity.image_height)
        cell = row * gates.grid_columns + column
        cells.setdefault(cell, []).append(anchor)
    occupied = tuple(sorted(cells))
    if len(occupied) < (gates.grid_columns * gates.grid_rows) // 2:
        blockers.append("INSUFFICIENT_ANCHOR_GRID_COVERAGE")
    quadrants = {
        (cell // gates.grid_columns >= gates.grid_rows // 2) * 2
        + (cell % gates.grid_columns >= gates.grid_columns // 2)
        for cell in occupied
    }
    if quadrants != {0, 1, 2, 3}:
        blockers.append("ANCHOR_QUADRANT_COVERAGE_REQUIRED")
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        return AnchorPartition((), (), occupied, (), seed64, False, tuple(blockers))

    anchor_count = len(ordered)
    holdout_size = max(gates.minimum_holdout_count, math.ceil(gates.holdout_fraction * anchor_count))
    if holdout_size >= anchor_count:
        return AnchorPartition((), (), occupied, (), seed64, False, ("HOLDOUT_EXHAUSTS_FITTING_SET",))
    allocation: dict[int, int] = {}
    remainders = []
    allocated = 0
    for cell in occupied:
        exact = len(cells[cell]) * holdout_size / anchor_count
        base = math.floor(exact)
        allocation[cell] = base
        allocated += base
        remainders.append((exact - base, cell))
    remaining = holdout_size - allocated
    for _remainder, cell in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocation[cell] += 1

    generator = np.random.Generator(np.random.PCG64(seed64))
    holdout = []
    fit = []
    for cell in occupied:
        records = cells[cell]
        permutation = generator.permutation(len(records))
        selected = set(int(index) for index in permutation[: allocation[cell]])
        holdout.extend(records[index] for index in permutation[: allocation[cell]])
        fit.extend(record for index, record in enumerate(records) if index not in selected)
    fit_quadrants = set()
    for anchor in fit:
        column = min(gates.grid_columns - 1, anchor.pixel_x * gates.grid_columns // seed_identity.image_width)
        row = min(gates.grid_rows - 1, anchor.pixel_y * gates.grid_rows // seed_identity.image_height)
        fit_quadrants.add((row >= gates.grid_rows // 2) * 2 + (column >= gates.grid_columns // 2))
    if fit_quadrants != {0, 1, 2, 3}:
        return AnchorPartition((), (), occupied, tuple(sorted(fit_quadrants)), seed64, False, ("FITTING_QUADRANT_COVERAGE_REQUIRED",))
    return AnchorPartition(
        fit=tuple(fit),
        holdout=tuple(holdout),
        occupied_cells=occupied,
        quadrants_in_fit=tuple(sorted(fit_quadrants)),
        seed64=seed64,
        accepted=True,
        blockers=(),
    )


def _huber_irls(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, int]:
    gates = load_reconstruction_config().appearance.depth_anything
    design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    if len(x) < 3 or np.linalg.matrix_rank(design) < 2:
        return None, 0
    weights = np.ones(len(x), dtype=np.float64)
    parameters = np.zeros(2, dtype=np.float64)
    for iteration in range(1, gates.maximum_irls_iterations + 1):
        root = np.sqrt(weights)
        weighted_design = design * root[:, None]
        weighted_target = y * root
        if np.linalg.matrix_rank(weighted_design) < 2:
            return None, iteration
        try:
            updated, _residuals, _rank, _singular = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        except np.linalg.LinAlgError:
            return None, iteration
        if not np.all(np.isfinite(updated)):
            return None, iteration
        denominator = max(float(np.linalg.norm(parameters)), np.finfo(np.float64).eps)
        relative_change = float(np.linalg.norm(updated - parameters)) / denominator
        parameters = updated
        residual = y - design @ parameters
        absolute = np.abs(residual)
        weights = np.ones(len(x), dtype=np.float64)
        outliers = absolute > gates.huber_delta_inverse_m
        weights[outliers] = gates.huber_delta_inverse_m / absolute[outliers]
        if relative_change <= gates.parameter_relative_tolerance:
            return parameters, iteration
    return parameters, gates.maximum_irls_iterations


def _fit_revision(
    model_revision_id: str,
    *,
    scale: float | None,
    bias: float | None,
    inlier_ids: tuple[int, ...],
    blockers: tuple[str, ...],
    holdout_metrics: tuple[float | None, float | None, float | None],
) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "model_revision_id": model_revision_id,
            "appearance_config_sha256": load_reconstruction_config().appearance_sha256,
            "scale": scale,
            "bias": bias,
            "inlier_ids": list(inlier_ids),
            "holdout_median_absolute_residual_m": holdout_metrics[0],
            "holdout_p95_absolute_residual_m": holdout_metrics[1],
            "holdout_absolute_bias_m": holdout_metrics[2],
            "blockers": list(blockers),
            "algorithm": "huber-affine-inverse-depth-v1",
        }
    )


def _rejected_metric_fit(model_revision_id: str, blockers: Sequence[str]) -> MetricDepthFit:
    unique = tuple(dict.fromkeys(blockers))
    return MetricDepthFit(
        scale=None,
        bias=None,
        iterations=0,
        inlier_ids=(),
        fit_residual_scale=None,
        holdout_median_absolute_residual_m=None,
        holdout_p95_absolute_residual_m=None,
        holdout_absolute_bias_m=None,
        accepted=False,
        blockers=unique,
        revision_id=_fit_revision(
            model_revision_id,
            scale=None,
            bias=None,
            inlier_ids=(),
            blockers=unique,
            holdout_metrics=(None, None, None),
        ),
    )


def fit_metric_depth(partition: AnchorPartition, *, model_revision_id: str) -> MetricDepthFit:
    model_revision = _sha256(model_revision_id, "model_revision_id")
    if not partition.accepted:
        return _rejected_metric_fit(model_revision, partition.blockers or ("ANCHOR_PARTITION_REJECTED",))
    if len(partition.fit) < 3 or not partition.holdout:
        return _rejected_metric_fit(model_revision, ("INSUFFICIENT_FIT_OR_HOLDOUT_ANCHORS",))
    fit_model = np.asarray([item.model_inverse_depth for item in partition.fit], dtype=np.float64)
    fit_depth = np.asarray([item.metric_depth_m for item in partition.fit], dtype=np.float64)
    holdout_model = np.asarray([item.model_inverse_depth for item in partition.holdout], dtype=np.float64)
    holdout_depth = np.asarray([item.metric_depth_m for item in partition.holdout], dtype=np.float64)
    if (
        not np.all(np.isfinite(fit_model))
        or not np.all(np.isfinite(fit_depth))
        or not np.all(np.isfinite(holdout_model))
        or not np.all(np.isfinite(holdout_depth))
        or np.any(fit_model <= 0.0)
        or np.any(holdout_model <= 0.0)
        or np.any(fit_depth <= 0.0)
        or np.any(holdout_depth <= 0.0)
    ):
        return _rejected_metric_fit(model_revision, ("NON_FINITE_ANCHOR_VALUES",))
    fit_inverse = 1.0 / fit_depth
    parameters, iterations = _huber_irls(fit_model, fit_inverse)
    if parameters is None:
        return _rejected_metric_fit(model_revision, ("SINGULAR_METRIC_FIT",))
    residual = fit_inverse - (parameters[0] * fit_model + parameters[1])
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    threshold = max(3.0 * 1.4826 * mad, np.finfo(np.float64).eps * 64.0)
    inlier_mask = np.abs(residual - center) <= threshold
    if np.count_nonzero(inlier_mask) < 3:
        return _rejected_metric_fit(model_revision, ("INSUFFICIENT_POST_MAD_INLIERS",))
    refit, refit_iterations = _huber_irls(fit_model[inlier_mask], fit_inverse[inlier_mask])
    if refit is None:
        return _rejected_metric_fit(model_revision, ("SINGULAR_METRIC_REFIT",))
    scale, bias = (float(refit[0]), float(refit[1]))
    inlier_ids = tuple(
        item.lidar_point_index for item, accepted in zip(partition.fit, inlier_mask, strict=True) if accepted
    )
    final_residual = fit_inverse[inlier_mask] - (scale * fit_model[inlier_mask] + bias)
    residual_center = float(np.median(final_residual))
    residual_scale = float(1.4826 * np.median(np.abs(final_residual - residual_center)))
    blockers = []
    if not math.isfinite(scale) or scale <= 0.0:
        blockers.append("NON_POSITIVE_SCALE")
    predicted_inverse = scale * holdout_model + bias
    if not np.all(np.isfinite(predicted_inverse)) or np.any(predicted_inverse <= 0.0):
        blockers.append("NON_POSITIVE_HOLDOUT_INVERSE_DEPTH")
        metrics = (None, None, None)
    else:
        holdout_residual = 1.0 / predicted_inverse - holdout_depth
        absolute = np.abs(holdout_residual)
        median = float(np.median(absolute))
        p95 = float(np.percentile(absolute, 95))
        absolute_bias = abs(float(np.mean(holdout_residual)))
        metrics = (median, p95, absolute_bias)
        gates = load_reconstruction_config().appearance.depth_anything
        if (
            median > gates.maximum_holdout_median_residual_m
            or p95 > gates.maximum_holdout_p95_residual_m
            or absolute_bias > gates.maximum_holdout_absolute_bias_m
        ):
            blockers.append("HOLDOUT_RESIDUAL_GATE_FAILED")
    unique_blockers = tuple(dict.fromkeys(blockers))
    return MetricDepthFit(
        scale=scale,
        bias=bias,
        iterations=refit_iterations,
        inlier_ids=inlier_ids,
        fit_residual_scale=residual_scale,
        holdout_median_absolute_residual_m=metrics[0],
        holdout_p95_absolute_residual_m=metrics[1],
        holdout_absolute_bias_m=metrics[2],
        accepted=not unique_blockers,
        blockers=unique_blockers,
        revision_id=_fit_revision(
            model_revision,
            scale=scale,
            bias=bias,
            inlier_ids=inlier_ids,
            blockers=unique_blockers,
            holdout_metrics=metrics,
        ),
    )


def _post_outlier_parameters(records: Sequence[ConsensusAnchor]) -> np.ndarray | None:
    model = np.asarray([item.model_inverse_depth for item in records], dtype=np.float64)
    depth = np.asarray([item.metric_depth_m for item in records], dtype=np.float64)
    if (
        len(model) < 3
        or not np.all(np.isfinite(model))
        or not np.all(np.isfinite(depth))
        or np.any(model <= 0.0)
        or np.any(depth <= 0.0)
    ):
        return None
    target = 1.0 / depth
    parameters, _iterations = _huber_irls(model, target)
    if parameters is None:
        return None
    residual = target - (parameters[0] * model + parameters[1])
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    threshold = max(3.0 * 1.4826 * mad, np.finfo(np.float64).eps * 64.0)
    mask = np.abs(residual - center) <= threshold
    if np.count_nonzero(mask) < 3:
        return None
    refit, _iterations = _huber_irls(model[mask], target[mask])
    return refit


def bootstrap_stability(
    partition: AnchorPartition,
    fit: MetricDepthFit,
    *,
    model_revision_id: str,
    seed_identity: AnchorSeedIdentity,
) -> BootstrapStability:
    model_revision = _sha256(model_revision_id, "model_revision_id")
    gates = load_reconstruction_config().appearance.depth_anything
    frame_hashes = sorted({item.frame_sha256 for item in partition.fit})
    seed_payload = {
        "appearance_config_sha256": seed_identity.appearance_config_sha256,
        "domain": "bootstrap-v1",
        "frame_sha256": frame_hashes,
        "model_revision_id": model_revision,
    }
    digest = hashlib.sha256(canonical_json_bytes(seed_payload)).digest()
    seed64 = int.from_bytes(digest[:8], "little", signed=False)
    cache_key = canonical_sha256(
        {
            "schema_version": 1,
            "fit_revision_id": fit.revision_id,
            "seed_payload": seed_payload,
            "anchors": [
                {
                    "pixel_x": item.pixel_x,
                    "pixel_y": item.pixel_y,
                    "lidar_point_index": item.lidar_point_index,
                    "model_inverse_depth": item.model_inverse_depth,
                    "metric_depth_m": item.metric_depth_m,
                    "frame_sha256": item.frame_sha256,
                    "accepted_region_id": item.accepted_region_id,
                }
                for item in partition.fit
            ],
        }
    )
    with _BOOTSTRAP_CACHE_LOCK:
        cached = _BOOTSTRAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not partition.accepted or not fit.accepted or fit.scale is None or fit.scale <= 0.0:
        blockers = tuple(dict.fromkeys((*partition.blockers, *fit.blockers, "ACCEPTED_METRIC_FIT_REQUIRED")))
        result = BootstrapStability(np.empty((0, 2)), seed64, 0, None, False, blockers)
        with _BOOTSTRAP_CACHE_LOCK:
            _BOOTSTRAP_CACHE[cache_key] = result
        return result

    cells: dict[int, list[ConsensusAnchor]] = {}
    for anchor in partition.fit:
        column = min(gates.grid_columns - 1, anchor.pixel_x * gates.grid_columns // seed_identity.image_width)
        row = min(gates.grid_rows - 1, anchor.pixel_y * gates.grid_rows // seed_identity.image_height)
        cells.setdefault(row * gates.grid_columns + column, []).append(anchor)
    generator = np.random.Generator(np.random.PCG64(seed64))
    samples = np.empty((gates.bootstrap_replicates, 2), dtype=np.float64)
    first_cell_samples = []
    first_cell = min(cells)
    for replicate in range(gates.bootstrap_replicates):
        records = []
        for cell in sorted(cells):
            source = cells[cell]
            selected = generator.integers(0, len(source), size=len(source), dtype=np.int64)
            if cell == first_cell and replicate < 3:
                first_cell_samples.append(tuple(int(index) for index in selected))
            records.extend(source[int(index)] for index in selected)
        parameters = _post_outlier_parameters(records)
        if parameters is None or not np.all(np.isfinite(parameters)) or parameters[0] <= 0.0:
            result = BootstrapStability(
                samples[:replicate],
                seed64,
                replicate,
                None,
                False,
                ("BOOTSTRAP_METRIC_FIT_FAILED",),
                tuple(first_cell_samples),
            )
            with _BOOTSTRAP_CACHE_LOCK:
                _BOOTSTRAP_CACHE[cache_key] = result
            return result
        samples[replicate] = parameters
    relative_stddev = float(np.std(samples[:, 0], ddof=1) / fit.scale)
    blockers = () if relative_stddev <= gates.maximum_scale_stddev_fraction else ("BOOTSTRAP_SCALE_STABILITY_FAILED",)
    result = BootstrapStability(
        samples,
        seed64,
        gates.bootstrap_replicates,
        relative_stddev,
        not blockers,
        blockers,
        tuple(first_cell_samples),
    )
    with _BOOTSTRAP_CACHE_LOCK:
        _BOOTSTRAP_CACHE[cache_key] = result
    return result


def _local_mad(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = values.shape
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            y0, y1 = max(0, y - 1), min(height, y + 2)
            x0, x1 = max(0, x - 1), min(width, x + 2)
            region = values[y0:y1, x0:x1]
            mask = valid[y0:y1, x0:x1]
            if np.any(mask):
                selected = region[mask]
                center = float(np.median(selected))
                result[y, x] = 1.4826 * float(np.median(np.abs(selected - center)))
    return result


def compute_metric_depth_uncertainty(
    model_inverse_depth: np.ndarray,
    fit: MetricDepthFit,
    bootstrap: BootstrapStability,
    *,
    physical_depth_m: np.ndarray | None,
    anchor_distance_m: np.ndarray,
    boundary_accepted: np.ndarray,
) -> MetricDepthUncertainty:
    model = np.asarray(model_inverse_depth, dtype=np.float64)
    if model.ndim != 2 or model.size == 0:
        raise ValueError("model_inverse_depth must be a non-empty H by W array")
    distance = np.asarray(anchor_distance_m, dtype=np.float64)
    boundary = np.asarray(boundary_accepted, dtype=bool)
    if distance.shape != model.shape or boundary.shape != model.shape:
        raise ValueError("anchor distance and boundary masks must match model depth")
    physical = np.full(model.shape, np.nan, dtype=np.float64) if physical_depth_m is None else np.asarray(physical_depth_m, dtype=np.float64)
    if physical.shape != model.shape:
        raise ValueError("physical_depth_m must match model depth")
    if not fit.accepted or fit.scale is None or fit.bias is None or not bootstrap.accepted:
        nan = np.full(model.shape, np.nan, dtype=np.float64)
        blockers = tuple(dict.fromkeys((*fit.blockers, *bootstrap.blockers, "ACCEPTED_FIT_AND_BOOTSTRAP_REQUIRED")))
        return MetricDepthUncertainty(nan, nan, nan, nan, nan, nan, nan, np.zeros(model.shape, dtype=bool), False, blockers)

    predicted_inverse = fit.scale * model + fit.bias
    model_valid = np.isfinite(model) & (model > 0.0) & np.isfinite(predicted_inverse) & (predicted_inverse > 0.0)
    metric = np.full(model.shape, np.nan, dtype=np.float64)
    metric[model_valid] = 1.0 / predicted_inverse[model_valid]
    sigma_fit = np.full(model.shape, np.nan, dtype=np.float64)
    flat_model = model.reshape(-1)
    flat_sigma = sigma_fit.reshape(-1)
    for start in range(0, flat_model.size, 4096):
        stop = min(flat_model.size, start + 4096)
        inverse_samples = (
            bootstrap.parameter_samples[:, 0, None] * flat_model[None, start:stop]
            + bootstrap.parameter_samples[:, 1, None]
        )
        sample_valid = np.all(np.isfinite(inverse_samples) & (inverse_samples > 0.0), axis=0)
        chunk = np.full(stop - start, np.nan, dtype=np.float64)
        if np.any(sample_valid):
            chunk[sample_valid] = np.std(1.0 / inverse_samples[:, sample_valid], axis=0, ddof=1)
        flat_sigma[start:stop] = chunk
    residual_scale = 0.0 if fit.fit_residual_scale is None else fit.fit_residual_scale
    sigma_residual = np.full(model.shape, np.nan, dtype=np.float64)
    sigma_residual[model_valid] = residual_scale / np.square(predicted_inverse[model_valid])
    sigma_local = _local_mad(metric, model_valid)
    sigma_physical = np.zeros(model.shape, dtype=np.float64)
    physical_valid = np.isfinite(physical) & (physical > 0.0) & model_valid
    sigma_physical[physical_valid] = np.abs(metric[physical_valid] - physical[physical_valid])
    sigma_anchor = 0.10 * distance
    total = np.sqrt(
        np.square(sigma_fit)
        + np.square(sigma_residual)
        + np.square(sigma_local)
        + np.square(sigma_physical)
        + np.square(sigma_anchor)
    )
    gates = load_reconstruction_config().appearance.depth_anything
    finite_components = (
        model_valid
        & np.isfinite(distance)
        & (distance >= 0.0)
        & np.isfinite(total)
    )
    valid = (
        finite_components
        & boundary
        & (distance <= gates.maximum_boundary_distance_m)
        & (total <= gates.maximum_uncertainty_m)
    )
    blockers = []
    if np.any(finite_components & ~boundary):
        blockers.append("BOUNDARY_SUPPORT_REQUIRED")
    if np.any(finite_components & (distance > gates.maximum_boundary_distance_m)):
        blockers.append("BOUNDARY_DISTANCE_GATE_FAILED")
    if np.any(finite_components & (total > gates.maximum_uncertainty_m)):
        blockers.append("UNCERTAINTY_GATE_FAILED")
    if np.any(~finite_components):
        blockers.append("NON_FINITE_UNCERTAINTY_COMPONENT")
    return MetricDepthUncertainty(
        metric,
        sigma_fit,
        sigma_residual,
        sigma_local,
        sigma_physical,
        sigma_anchor,
        total,
        valid,
        bool(np.any(valid)),
        tuple(blockers),
    )
