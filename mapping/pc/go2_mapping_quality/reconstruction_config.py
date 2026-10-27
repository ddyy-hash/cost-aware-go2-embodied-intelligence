"""Strict, versioned reconstruction configuration and derived identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
DEFAULT_FUSION_CONFIG_PATH = CONFIG_ROOT / "fusion_gates.json"
DEFAULT_APPEARANCE_CONFIG_PATH = CONFIG_ROOT / "appearance_gates.json"


class ConfigError(ValueError):
    """Raised when versioned reconstruction input fails closed validation."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: root must be an object")
    return value


def sha256_file(path: str | Path) -> str:
    """Hash JSON by canonical content so formatting cannot change identity."""
    return canonical_sha256(_read_json(path))


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ConfigError(f"{path} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{path} unknown fields: {', '.join(sorted(unknown))}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigError(f"{path} must be {operator} {minimum:g}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{path} must be <= {maximum:g}")
    return result


def _positive(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0, minimum_inclusive=False)


def _nonnegative(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0)


def _probability(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0, maximum=1.0)


def _runtime_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _runtime_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _runtime_nonnegative(value: Any) -> float | None:
    result = _runtime_number(value)
    return result if result is not None and result >= 0.0 else None


def _runtime_probability(value: Any) -> float | None:
    result = _runtime_number(value)
    return result if result is not None and 0.0 <= result <= 1.0 else None


_SHA256_RE = re.compile(r"[0-9A-Fa-f]{64}", re.ASCII)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _hash(value: Any, path: str) -> str:
    if not is_sha256(value):
        raise ConfigError(f"{path} must be exactly 64 ASCII hexadecimal characters")
    return value.lower()


@dataclass(frozen=True)
class FilteringGates:
    minimum_range_m: float
    maximum_range_m: float
    radius_m: float
    radius_min_neighbors: int
    statistical_neighbor_count: int
    statistical_stddev_multiplier: float

    def accepts_range(self, *, range_m: float) -> bool:
        distance = _runtime_nonnegative(range_m)
        return bool(
            distance is not None
            and self.minimum_range_m <= distance <= self.maximum_range_m
        )

    def within_radius(self, *, distance_m: float) -> bool:
        distance = _runtime_nonnegative(distance_m)
        return bool(distance is not None and distance <= self.radius_m)

    def has_radius_support(self, *, neighbor_count: int) -> bool:
        count = _runtime_integer(neighbor_count)
        return bool(count is not None and count >= self.radius_min_neighbors)

    def has_statistical_support(self, *, neighbor_count: int) -> bool:
        count = _runtime_integer(neighbor_count)
        return bool(count is not None and count >= self.statistical_neighbor_count)

    def accepts_statistical_distance(self, *, distance: float, mean: float, stddev: float) -> bool:
        observed = _runtime_number(distance)
        center = _runtime_number(mean)
        spread = _runtime_nonnegative(stddev)
        return bool(
            observed is not None
            and center is not None
            and spread is not None
            and abs(observed - center) <= self.statistical_stddev_multiplier * spread
        )


@dataclass(frozen=True)
class MeasuredPromotionGates:
    minimum_observations: int
    minimum_independent_views: int
    minimum_temporal_support_s: float
    maximum_principal_stddev_m: float
    maximum_point_to_plane_p95_m: float
    minimum_static_probability: float

    def accepts(
        self,
        *,
        observation_count: int,
        independent_views: int,
        temporal_support_s: float,
        principal_stddev_m: float,
        point_to_plane_p95_m: float,
        static_probability: float,
    ) -> bool:
        observation_count_value = _runtime_integer(observation_count)
        independent_views_value = _runtime_integer(independent_views)
        values = (
            _runtime_nonnegative(temporal_support_s),
            _runtime_nonnegative(principal_stddev_m),
            _runtime_nonnegative(point_to_plane_p95_m),
            _runtime_probability(static_probability),
        )
        if observation_count_value is None or independent_views_value is None or any(value is None for value in values):
            return False
        temporal, principal, residual, static = values
        return bool(
            observation_count_value >= self.minimum_observations
            and independent_views_value >= self.minimum_independent_views
            and temporal >= self.minimum_temporal_support_s
            and principal <= self.maximum_principal_stddev_m
            and residual <= self.maximum_point_to_plane_p95_m
            and static >= self.minimum_static_probability
        )


@dataclass(frozen=True)
class IndependentViewGates:
    minimum_origin_baseline_m: float
    minimum_direction_separation_deg: float

    def accepts(self, *, origin_baseline_m: float, direction_separation_deg: float) -> bool:
        baseline = _runtime_nonnegative(origin_baseline_m)
        separation = _runtime_nonnegative(direction_separation_deg)
        return bool(
            baseline is not None
            and separation is not None
            and separation <= 180.0
            and (
                baseline >= self.minimum_origin_baseline_m
                or separation >= self.minimum_direction_separation_deg
            )
        )


@dataclass(frozen=True)
class DynamicEvidenceGates:
    minimum_votes: int
    unstable_free_space_fraction: float
    rejected_unstable_generations: int

    def is_unstable(self, *, vote_count: int, free_space_fraction: float) -> bool:
        votes = _runtime_integer(vote_count)
        fraction = _runtime_probability(free_space_fraction)
        return bool(
            votes is not None
            and fraction is not None
            and votes >= self.minimum_votes
            and fraction > self.unstable_free_space_fraction
        )

    def is_rejected(self, *, unstable_generations: int) -> bool:
        generations = _runtime_integer(unstable_generations)
        return bool(generations is not None and generations >= self.rejected_unstable_generations)


@dataclass(frozen=True)
class StabilityGates:
    minimum_measured_coverage: float
    maximum_generation_rms_change_m: float
    settled_generation_count: int

    def accepts(
        self,
        measured_coverage: float,
        generation_rms_change_m: float,
        settled_generations: int,
    ) -> bool:
        coverage = _runtime_probability(measured_coverage)
        change = _runtime_nonnegative(generation_rms_change_m)
        generations = _runtime_integer(settled_generations)
        return bool(
            coverage is not None
            and change is not None
            and generations is not None
            and coverage >= self.minimum_measured_coverage
            and change <= self.maximum_generation_rms_change_m
            and generations >= self.settled_generation_count
        )


@dataclass(frozen=True)
class MetricDomainGates:
    eligible_minimum_observations: int
    depth_tile_size_m: float

    def is_eligible(
        self,
        *,
        finite_post_filter_observations: int,
        tile_size_m: float,
    ) -> bool:
        observations = _runtime_integer(finite_post_filter_observations)
        tile_size = _runtime_nonnegative(tile_size_m)
        return bool(
            observations is not None
            and observations >= self.eligible_minimum_observations
            and tile_size is not None
            and math.isclose(tile_size, self.depth_tile_size_m, rel_tol=0.0, abs_tol=1e-12)
        )


@dataclass(frozen=True)
class SurfacingGates:
    voxel_size_m: float
    display_point_budget: int
    display_triangle_budget: int

    def accepts_output(self, *, voxel_size_m: float, point_count: int, triangle_count: int) -> bool:
        voxel_size = _runtime_nonnegative(voxel_size_m)
        points = _runtime_integer(point_count)
        triangles = _runtime_integer(triangle_count)
        return bool(
            voxel_size is not None
            and math.isclose(voxel_size, self.voxel_size_m, rel_tol=0.0, abs_tol=1e-12)
            and points is not None
            and 0 <= points <= self.display_point_budget
            and triangles is not None
            and 0 <= triangles <= self.display_triangle_budget
        )


@dataclass(frozen=True)
class FusionGates:
    schema_version: int
    algorithm_version: str
    filtering: FilteringGates
    measured_promotion: MeasuredPromotionGates
    independent_view: IndependentViewGates
    dynamic_evidence: DynamicEvidenceGates
    stability: StabilityGates
    metric_domain: MetricDomainGates
    surfacing: SurfacingGates


@dataclass(frozen=True)
class DepthValidationGates:
    minimum_range_m: float
    maximum_range_m: float
    minimum_correspondences_per_tile: int
    maximum_median_absolute_residual_m: float
    maximum_p95_absolute_residual_m: float
    maximum_absolute_mean_signed_bias_m: float
    maximum_occlusion_conflict_fraction: float
    maximum_free_space_violation_fraction: float

    def accepts(
        self,
        *,
        correspondence_count: int,
        median_absolute_residual_m: float,
        p95_absolute_residual_m: float,
        absolute_mean_signed_bias_m: float,
        occlusion_conflict_fraction: float,
        free_space_violation_fraction: float,
        minimum_observed_range_m: float,
        maximum_observed_range_m: float,
    ) -> bool:
        count = _runtime_integer(correspondence_count)
        values = (
            _runtime_nonnegative(median_absolute_residual_m),
            _runtime_nonnegative(p95_absolute_residual_m),
            _runtime_nonnegative(absolute_mean_signed_bias_m),
            _runtime_probability(occlusion_conflict_fraction),
            _runtime_probability(free_space_violation_fraction),
            _runtime_nonnegative(minimum_observed_range_m),
            _runtime_nonnegative(maximum_observed_range_m),
        )
        if count is None or any(value is None for value in values):
            return False
        median, p95, bias, conflict, violation, minimum_range, maximum_range = values
        return bool(
            count >= self.minimum_correspondences_per_tile
            and median <= self.maximum_median_absolute_residual_m
            and p95 <= self.maximum_p95_absolute_residual_m
            and median <= p95
            and bias <= self.maximum_absolute_mean_signed_bias_m
            and conflict <= self.maximum_occlusion_conflict_fraction
            and violation <= self.maximum_free_space_violation_fraction
            and minimum_range >= self.minimum_range_m
            and maximum_range <= self.maximum_range_m
            and minimum_range <= maximum_range
        )


@dataclass(frozen=True)
class ClockFitGates:
    minimum_sample_count: int
    minimum_span_s: float
    minimum_inlier_fraction: float
    maximum_slope_deviation_ppm: float
    maximum_median_absolute_residual_ms: float
    maximum_p95_residual_ms: float
    maximum_prediction_stddev_ms: float

    def accepts(
        self,
        *,
        sample_count: int,
        span_s: float,
        inlier_fraction: float,
        slope_deviation_ppm: float,
        median_absolute_residual_ms: float,
        p95_residual_ms: float,
        prediction_stddev_ms: float,
    ) -> bool:
        count = _runtime_integer(sample_count)
        values = (
            _runtime_nonnegative(span_s),
            _runtime_probability(inlier_fraction),
            _runtime_nonnegative(slope_deviation_ppm),
            _runtime_nonnegative(median_absolute_residual_ms),
            _runtime_nonnegative(p95_residual_ms),
            _runtime_nonnegative(prediction_stddev_ms),
        )
        if count is None or any(value is None for value in values):
            return False
        span, inliers, slope, median, p95, prediction = values
        return bool(
            count >= self.minimum_sample_count
            and span >= self.minimum_span_s
            and inliers >= self.minimum_inlier_fraction
            and slope <= self.maximum_slope_deviation_ppm + 1e-9
            and median <= self.maximum_median_absolute_residual_ms
            and p95 <= self.maximum_p95_residual_ms
            and prediction <= self.maximum_prediction_stddev_ms
            and median <= p95
        )


@dataclass(frozen=True)
class HoldoutGates:
    minimum_board_poses: int
    minimum_correspondences_per_pose: int
    minimum_total_correspondences: int
    maximum_reprojection_median_px: float
    maximum_reprojection_p95_px: float

    def accepts(
        self,
        *,
        board_pose_count: int,
        minimum_correspondences_in_any_pose: int,
        total_correspondences: int,
        reprojection_median_px: float,
        reprojection_p95_px: float,
    ) -> bool:
        poses = _runtime_integer(board_pose_count)
        per_pose = _runtime_integer(minimum_correspondences_in_any_pose)
        total = _runtime_integer(total_correspondences)
        median = _runtime_nonnegative(reprojection_median_px)
        p95 = _runtime_nonnegative(reprojection_p95_px)
        return bool(
            poses is not None
            and per_pose is not None
            and total is not None
            and median is not None
            and p95 is not None
            and poses >= self.minimum_board_poses
            and per_pose >= self.minimum_correspondences_per_pose
            and total >= self.minimum_total_correspondences
            and median <= self.maximum_reprojection_median_px
            and p95 <= self.maximum_reprojection_p95_px
            and median <= p95
        )


@dataclass(frozen=True)
class TextureGates:
    maximum_depth_residual_floor_m: float
    maximum_depth_residual_range_fraction: float
    minimum_independent_views: int
    maximum_linear_rgb_channel_stddev: float
    minimum_surface_coverage: float

    def accepts_sample(self, *, depth_residual_m: float, range_m: float) -> bool:
        residual = _runtime_number(depth_residual_m)
        distance = _runtime_nonnegative(range_m)
        if residual is None or distance is None:
            return False
        limit = max(
            self.maximum_depth_residual_floor_m,
            self.maximum_depth_residual_range_fraction * distance,
        )
        return abs(residual) <= limit

    def accepts_vertex(self, *, independent_views: int, channel_stddev: float) -> bool:
        views = _runtime_integer(independent_views)
        deviation = _runtime_nonnegative(channel_stddev)
        return bool(
            views is not None
            and deviation is not None
            and views >= self.minimum_independent_views
            and deviation <= self.maximum_linear_rgb_channel_stddev
        )

    def delivery_authorized(self, *, surface_coverage: float) -> bool:
        coverage = _runtime_probability(surface_coverage)
        return bool(coverage is not None and coverage >= self.minimum_surface_coverage)


_QUALITY_RANK = {
    "PREVIEW_UNCALIBRATED": 0,
    "RAW_CAPTURE_VALID": 1,
    "CALIBRATED_DENSE": 2,
    "GEOMETRY_VALIDATED_CM": 3,
    "BIM_REGISTERED_VALIDATED": 4,
}


@dataclass(frozen=True)
class AugmentationGates:
    minimum_quality_level: str
    minimum_depth_frames: int
    minimum_independent_views: int
    maximum_depth_stddev_m: float
    maximum_hole_radius_m: float

    def accepts(
        self,
        *,
        quality_level: str,
        depth_region_accepted: bool,
        depth_frame_count: int,
        independent_views: int,
        depth_stddev_m: float,
        completely_bounded: bool,
        hole_radius_m: float,
    ) -> bool:
        frames = _runtime_integer(depth_frame_count)
        views = _runtime_integer(independent_views)
        deviation = _runtime_nonnegative(depth_stddev_m)
        radius = _runtime_nonnegative(hole_radius_m)
        return bool(
            isinstance(quality_level, str)
            and _QUALITY_RANK.get(quality_level, -1) >= _QUALITY_RANK[self.minimum_quality_level]
            and depth_region_accepted is True
            and frames is not None
            and frames >= self.minimum_depth_frames
            and views is not None
            and views >= self.minimum_independent_views
            and deviation is not None
            and deviation <= self.maximum_depth_stddev_m
            and completely_bounded is True
            and radius is not None
            and radius <= self.maximum_hole_radius_m
        )


@dataclass(frozen=True)
class DepthAnythingGates:
    schema_version: int
    minimum_anchor_count: int
    grid_columns: int
    grid_rows: int
    holdout_fraction: float
    minimum_holdout_count: int
    huber_delta_inverse_m: float
    maximum_irls_iterations: int
    parameter_relative_tolerance: float
    maximum_holdout_median_residual_m: float
    maximum_holdout_p95_residual_m: float
    maximum_holdout_absolute_bias_m: float
    bootstrap_replicates: int
    maximum_scale_stddev_fraction: float
    maximum_uncertainty_m: float
    maximum_boundary_distance_m: float
    model_name: str
    weights_sha256: str
    preprocessing_revision: str
    input_size: tuple[int, int]
    precision: str


@dataclass(frozen=True)
class AppearanceGates:
    schema_version: int
    algorithm_version: str
    clock_fit: ClockFitGates
    depth_validation: DepthValidationGates
    holdout: HoldoutGates
    texture: TextureGates
    augmentation: AugmentationGates
    depth_anything: DepthAnythingGates


@dataclass(frozen=True)
class DerivedIdentity:
    artifact_hashes: tuple[str, ...]
    pose_generation: str
    calibration_hash: str
    roi_hash: str
    mask_hash: str
    fusion_config_sha256: str
    appearance_config_sha256: str
    algorithm_version: str
    cache_key: str


@dataclass(frozen=True)
class ReconstructionConfig:
    fusion: FusionGates
    appearance: AppearanceGates
    fusion_sha256: str
    appearance_sha256: str

    def derived_identity(
        self,
        *,
        artifact_hashes: Sequence[str],
        pose_generation: str,
        calibration_hash: str,
        roi_hash: str,
        mask_hash: str,
        algorithm_version: str,
    ) -> DerivedIdentity:
        if isinstance(artifact_hashes, (str, bytes)) or not artifact_hashes:
            raise ConfigError("artifact_hashes must contain at least one SHA-256 digest")
        artifacts = tuple(sorted({_hash(value, "artifact_hashes") for value in artifact_hashes}))
        pose = _string(pose_generation, "pose_generation")
        calibration = _hash(calibration_hash, "calibration_hash")
        roi = _hash(roi_hash, "roi_hash")
        mask = _hash(mask_hash, "mask_hash")
        algorithm = _string(algorithm_version, "algorithm_version")
        payload = {
            "identity_schema_version": 1,
            "artifact_hashes": artifacts,
            "pose_generation": pose,
            "calibration_hash": calibration,
            "roi_hash": roi,
            "mask_hash": mask,
            "fusion_config_sha256": self.fusion_sha256,
            "appearance_config_sha256": self.appearance_sha256,
            "algorithm_version": algorithm,
        }
        return DerivedIdentity(
            artifact_hashes=artifacts,
            pose_generation=pose,
            calibration_hash=calibration,
            roi_hash=roi,
            mask_hash=mask,
            fusion_config_sha256=self.fusion_sha256,
            appearance_config_sha256=self.appearance_sha256,
            algorithm_version=algorithm,
            cache_key=canonical_sha256(payload),
        )


def _parse_fusion(payload: Mapping[str, Any]) -> FusionGates:
    expected = {
        "schema_version",
        "algorithm_version",
        "filtering",
        "measured_promotion",
        "independent_view",
        "dynamic_evidence",
        "stability",
        "metric_domain",
        "surfacing",
    }
    _keys(payload, expected, "fusion")
    schema_version = _integer(payload["schema_version"], "fusion.schema_version")
    if schema_version != 2:
        raise ConfigError(f"fusion.schema_version unsupported: {schema_version}")

    filtering = _object(payload["filtering"], "fusion.filtering")
    filtering_fields = {
        "minimum_range_m",
        "maximum_range_m",
        "radius_m",
        "radius_min_neighbors",
        "statistical_neighbor_count",
        "statistical_stddev_multiplier",
    }
    _keys(filtering, filtering_fields, "fusion.filtering")
    minimum_range = _nonnegative(filtering["minimum_range_m"], "filtering.minimum_range_m")
    maximum_range = _positive(filtering["maximum_range_m"], "filtering.maximum_range_m")
    if minimum_range >= maximum_range:
        raise ConfigError("filtering.minimum_range_m must be less than maximum_range_m")

    measured = _object(payload["measured_promotion"], "fusion.measured_promotion")
    measured_fields = {
        "minimum_observations",
        "minimum_independent_views",
        "minimum_temporal_support_s",
        "maximum_principal_stddev_m",
        "maximum_point_to_plane_p95_m",
        "minimum_static_probability",
    }
    _keys(measured, measured_fields, "fusion.measured_promotion")
    minimum_observations = _integer(measured["minimum_observations"], "measured_promotion.minimum_observations")
    minimum_measured_views = _integer(measured["minimum_independent_views"], "measured_promotion.minimum_independent_views")
    if minimum_measured_views > minimum_observations:
        raise ConfigError("measured_promotion.minimum_independent_views cannot exceed minimum_observations")

    view = _object(payload["independent_view"], "fusion.independent_view")
    view_fields = {"minimum_origin_baseline_m", "minimum_direction_separation_deg"}
    _keys(view, view_fields, "fusion.independent_view")

    dynamic = _object(payload["dynamic_evidence"], "fusion.dynamic_evidence")
    dynamic_fields = {"minimum_votes", "unstable_free_space_fraction", "rejected_unstable_generations"}
    _keys(dynamic, dynamic_fields, "fusion.dynamic_evidence")

    stability = _object(payload["stability"], "fusion.stability")
    stability_fields = {"minimum_measured_coverage", "maximum_generation_rms_change_m", "settled_generation_count"}
    _keys(stability, stability_fields, "fusion.stability")

    metric = _object(payload["metric_domain"], "fusion.metric_domain")
    metric_fields = {"eligible_minimum_observations", "depth_tile_size_m"}
    _keys(metric, metric_fields, "fusion.metric_domain")

    surfacing = _object(payload["surfacing"], "fusion.surfacing")
    surfacing_fields = {"voxel_size_m", "display_point_budget", "display_triangle_budget"}
    _keys(surfacing, surfacing_fields, "fusion.surfacing")

    return FusionGates(
        schema_version=schema_version,
        algorithm_version=_string(payload["algorithm_version"], "fusion.algorithm_version"),
        filtering=FilteringGates(
            minimum_range_m=minimum_range,
            maximum_range_m=maximum_range,
            radius_m=_positive(filtering["radius_m"], "filtering.radius_m"),
            radius_min_neighbors=_integer(filtering["radius_min_neighbors"], "filtering.radius_min_neighbors"),
            statistical_neighbor_count=_integer(filtering["statistical_neighbor_count"], "filtering.statistical_neighbor_count"),
            statistical_stddev_multiplier=_positive(filtering["statistical_stddev_multiplier"], "filtering.statistical_stddev_multiplier"),
        ),
        measured_promotion=MeasuredPromotionGates(
            minimum_observations=minimum_observations,
            minimum_independent_views=minimum_measured_views,
            minimum_temporal_support_s=_nonnegative(measured["minimum_temporal_support_s"], "measured_promotion.minimum_temporal_support_s"),
            maximum_principal_stddev_m=_nonnegative(measured["maximum_principal_stddev_m"], "measured_promotion.maximum_principal_stddev_m"),
            maximum_point_to_plane_p95_m=_nonnegative(measured["maximum_point_to_plane_p95_m"], "measured_promotion.maximum_point_to_plane_p95_m"),
            minimum_static_probability=_probability(measured["minimum_static_probability"], "measured_promotion.minimum_static_probability"),
        ),
        independent_view=IndependentViewGates(
            minimum_origin_baseline_m=_positive(view["minimum_origin_baseline_m"], "independent_view.minimum_origin_baseline_m"),
            minimum_direction_separation_deg=_number(
                view["minimum_direction_separation_deg"],
                "independent_view.minimum_direction_separation_deg",
                minimum=0.0,
                maximum=180.0,
                minimum_inclusive=False,
            ),
        ),
        dynamic_evidence=DynamicEvidenceGates(
            minimum_votes=_integer(dynamic["minimum_votes"], "dynamic_evidence.minimum_votes"),
            unstable_free_space_fraction=_probability(dynamic["unstable_free_space_fraction"], "dynamic_evidence.unstable_free_space_fraction"),
            rejected_unstable_generations=_integer(dynamic["rejected_unstable_generations"], "dynamic_evidence.rejected_unstable_generations"),
        ),
        stability=StabilityGates(
            minimum_measured_coverage=_probability(stability["minimum_measured_coverage"], "stability.minimum_measured_coverage"),
            maximum_generation_rms_change_m=_nonnegative(stability["maximum_generation_rms_change_m"], "stability.maximum_generation_rms_change_m"),
            settled_generation_count=_integer(stability["settled_generation_count"], "stability.settled_generation_count"),
        ),
        metric_domain=MetricDomainGates(
            eligible_minimum_observations=_integer(metric["eligible_minimum_observations"], "metric_domain.eligible_minimum_observations"),
            depth_tile_size_m=_positive(metric["depth_tile_size_m"], "metric_domain.depth_tile_size_m"),
        ),
        surfacing=SurfacingGates(
            voxel_size_m=_positive(surfacing["voxel_size_m"], "surfacing.voxel_size_m"),
            display_point_budget=_integer(surfacing["display_point_budget"], "surfacing.display_point_budget"),
            display_triangle_budget=_integer(surfacing["display_triangle_budget"], "surfacing.display_triangle_budget"),
        ),
    )


def _parse_appearance(payload: Mapping[str, Any]) -> AppearanceGates:
    expected = {
        "schema_version",
        "algorithm_version",
        "clock_fit",
        "depth_validation",
        "holdout",
        "texture",
        "augmentation",
        "depth_anything",
    }
    _keys(payload, expected, "appearance")
    schema_version = _integer(payload["schema_version"], "appearance.schema_version")
    if schema_version != 2:
        raise ConfigError(f"appearance.schema_version unsupported: {schema_version}")

    clock = _object(payload["clock_fit"], "appearance.clock_fit")
    clock_fields = {
        "minimum_sample_count",
        "minimum_span_s",
        "minimum_inlier_fraction",
        "maximum_slope_deviation_ppm",
        "maximum_median_absolute_residual_ms",
        "maximum_p95_residual_ms",
        "maximum_prediction_stddev_ms",
    }
    _keys(clock, clock_fields, "appearance.clock_fit")
    clock_median = _nonnegative(clock["maximum_median_absolute_residual_ms"], "clock_fit.maximum_median_absolute_residual_ms")
    clock_p95 = _nonnegative(clock["maximum_p95_residual_ms"], "clock_fit.maximum_p95_residual_ms")
    if clock_median > clock_p95:
        raise ConfigError("clock_fit.maximum_median_absolute_residual_ms must be <= maximum_p95_residual_ms")

    depth = _object(payload["depth_validation"], "appearance.depth_validation")
    depth_fields = {
        "minimum_range_m",
        "maximum_range_m",
        "minimum_correspondences_per_tile",
        "maximum_median_absolute_residual_m",
        "maximum_p95_absolute_residual_m",
        "maximum_absolute_mean_signed_bias_m",
        "maximum_occlusion_conflict_fraction",
        "maximum_free_space_violation_fraction",
    }
    _keys(depth, depth_fields, "appearance.depth_validation")
    minimum_range = _nonnegative(depth["minimum_range_m"], "depth_validation.minimum_range_m")
    maximum_range = _positive(depth["maximum_range_m"], "depth_validation.maximum_range_m")
    if minimum_range >= maximum_range:
        raise ConfigError("depth_validation.minimum_range_m must be less than maximum_range_m")
    median_residual = _nonnegative(depth["maximum_median_absolute_residual_m"], "depth_validation.maximum_median_absolute_residual_m")
    p95_residual = _nonnegative(depth["maximum_p95_absolute_residual_m"], "depth_validation.maximum_p95_absolute_residual_m")
    if median_residual > p95_residual:
        raise ConfigError("depth_validation.maximum_median_absolute_residual_m must be <= maximum_p95_absolute_residual_m")

    holdout = _object(payload["holdout"], "appearance.holdout")
    holdout_fields = {
        "minimum_board_poses",
        "minimum_correspondences_per_pose",
        "minimum_total_correspondences",
        "maximum_reprojection_median_px",
        "maximum_reprojection_p95_px",
    }
    _keys(holdout, holdout_fields, "appearance.holdout")
    minimum_board_poses = _integer(holdout["minimum_board_poses"], "holdout.minimum_board_poses")
    minimum_per_pose = _integer(holdout["minimum_correspondences_per_pose"], "holdout.minimum_correspondences_per_pose")
    minimum_total = _integer(holdout["minimum_total_correspondences"], "holdout.minimum_total_correspondences")
    if minimum_total < minimum_board_poses * minimum_per_pose:
        raise ConfigError("holdout.minimum_total_correspondences cannot be less than the required per-pose total")
    reprojection_median = _nonnegative(holdout["maximum_reprojection_median_px"], "holdout.maximum_reprojection_median_px")
    reprojection_p95 = _nonnegative(holdout["maximum_reprojection_p95_px"], "holdout.maximum_reprojection_p95_px")
    if reprojection_median > reprojection_p95:
        raise ConfigError("holdout.maximum_reprojection_median_px must be <= maximum_reprojection_p95_px")

    texture = _object(payload["texture"], "appearance.texture")
    texture_fields = {
        "maximum_depth_residual_floor_m",
        "maximum_depth_residual_range_fraction",
        "minimum_independent_views",
        "maximum_linear_rgb_channel_stddev",
        "minimum_surface_coverage",
    }
    _keys(texture, texture_fields, "appearance.texture")

    augmentation = _object(payload["augmentation"], "appearance.augmentation")
    augmentation_fields = {
        "minimum_quality_level",
        "minimum_depth_frames",
        "minimum_independent_views",
        "maximum_depth_stddev_m",
        "maximum_hole_radius_m",
    }
    _keys(augmentation, augmentation_fields, "appearance.augmentation")
    minimum_quality = _string(augmentation["minimum_quality_level"], "augmentation.minimum_quality_level")
    if minimum_quality not in _QUALITY_RANK:
        raise ConfigError("augmentation.minimum_quality_level is unknown")
    minimum_depth_frames = _integer(augmentation["minimum_depth_frames"], "augmentation.minimum_depth_frames")
    minimum_augmentation_views = _integer(augmentation["minimum_independent_views"], "augmentation.minimum_independent_views")
    if minimum_augmentation_views > minimum_depth_frames:
        raise ConfigError("augmentation.minimum_independent_views cannot exceed minimum_depth_frames")

    depth_anything = _object(payload["depth_anything"], "appearance.depth_anything")
    depth_anything_fields = {
        "schema_version",
        "minimum_anchor_count",
        "grid_columns",
        "grid_rows",
        "holdout_fraction",
        "minimum_holdout_count",
        "huber_delta_inverse_m",
        "maximum_irls_iterations",
        "parameter_relative_tolerance",
        "maximum_holdout_median_residual_m",
        "maximum_holdout_p95_residual_m",
        "maximum_holdout_absolute_bias_m",
        "bootstrap_replicates",
        "maximum_scale_stddev_fraction",
        "maximum_uncertainty_m",
        "maximum_boundary_distance_m",
        "model_name",
        "weights_sha256",
        "preprocessing_revision",
        "input_size",
        "precision",
    }
    _keys(depth_anything, depth_anything_fields, "appearance.depth_anything")
    depth_schema = _integer(depth_anything["schema_version"], "depth_anything.schema_version")
    if depth_schema != 1:
        raise ConfigError(f"depth_anything.schema_version unsupported: {depth_schema}")
    minimum_anchors = _integer(depth_anything["minimum_anchor_count"], "depth_anything.minimum_anchor_count")
    grid_columns = _integer(depth_anything["grid_columns"], "depth_anything.grid_columns")
    grid_rows = _integer(depth_anything["grid_rows"], "depth_anything.grid_rows")
    holdout_fraction = _number(depth_anything["holdout_fraction"], "depth_anything.holdout_fraction", minimum=0.0, maximum=1.0, minimum_inclusive=False)
    if holdout_fraction >= 1.0:
        raise ConfigError("depth_anything.holdout_fraction must be < 1")
    minimum_holdout = _integer(depth_anything["minimum_holdout_count"], "depth_anything.minimum_holdout_count")
    huber_delta = _positive(depth_anything["huber_delta_inverse_m"], "depth_anything.huber_delta_inverse_m")
    max_iterations = _integer(depth_anything["maximum_irls_iterations"], "depth_anything.maximum_irls_iterations")
    parameter_tolerance = _positive(depth_anything["parameter_relative_tolerance"], "depth_anything.parameter_relative_tolerance")
    holdout_median = _nonnegative(depth_anything["maximum_holdout_median_residual_m"], "depth_anything.maximum_holdout_median_residual_m")
    holdout_p95 = _nonnegative(depth_anything["maximum_holdout_p95_residual_m"], "depth_anything.maximum_holdout_p95_residual_m")
    holdout_bias = _nonnegative(depth_anything["maximum_holdout_absolute_bias_m"], "depth_anything.maximum_holdout_absolute_bias_m")
    if holdout_median > holdout_p95:
        raise ConfigError("depth_anything.maximum_holdout_median_residual_m must be <= maximum_holdout_p95_residual_m")
    bootstrap_replicates = _integer(depth_anything["bootstrap_replicates"], "depth_anything.bootstrap_replicates")
    scale_stability = _probability(depth_anything["maximum_scale_stddev_fraction"], "depth_anything.maximum_scale_stddev_fraction")
    uncertainty = _positive(depth_anything["maximum_uncertainty_m"], "depth_anything.maximum_uncertainty_m")
    boundary_distance = _positive(depth_anything["maximum_boundary_distance_m"], "depth_anything.maximum_boundary_distance_m")
    input_size = depth_anything["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise ConfigError("depth_anything.input_size must contain width and height")
    input_width = _integer(input_size[0], "depth_anything.input_size[0]")
    input_height = _integer(input_size[1], "depth_anything.input_size[1]")
    precision = _string(depth_anything["precision"], "depth_anything.precision")
    if precision not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("depth_anything.precision is unsupported")

    return AppearanceGates(
        schema_version=schema_version,
        algorithm_version=_string(payload["algorithm_version"], "appearance.algorithm_version"),
        clock_fit=ClockFitGates(
            minimum_sample_count=_integer(clock["minimum_sample_count"], "clock_fit.minimum_sample_count"),
            minimum_span_s=_positive(clock["minimum_span_s"], "clock_fit.minimum_span_s"),
            minimum_inlier_fraction=_probability(clock["minimum_inlier_fraction"], "clock_fit.minimum_inlier_fraction"),
            maximum_slope_deviation_ppm=_nonnegative(clock["maximum_slope_deviation_ppm"], "clock_fit.maximum_slope_deviation_ppm"),
            maximum_median_absolute_residual_ms=clock_median,
            maximum_p95_residual_ms=clock_p95,
            maximum_prediction_stddev_ms=_nonnegative(clock["maximum_prediction_stddev_ms"], "clock_fit.maximum_prediction_stddev_ms"),
        ),
        depth_validation=DepthValidationGates(
            minimum_range_m=minimum_range,
            maximum_range_m=maximum_range,
            minimum_correspondences_per_tile=_integer(depth["minimum_correspondences_per_tile"], "depth_validation.minimum_correspondences_per_tile"),
            maximum_median_absolute_residual_m=median_residual,
            maximum_p95_absolute_residual_m=p95_residual,
            maximum_absolute_mean_signed_bias_m=_nonnegative(depth["maximum_absolute_mean_signed_bias_m"], "depth_validation.maximum_absolute_mean_signed_bias_m"),
            maximum_occlusion_conflict_fraction=_probability(depth["maximum_occlusion_conflict_fraction"], "depth_validation.maximum_occlusion_conflict_fraction"),
            maximum_free_space_violation_fraction=_probability(depth["maximum_free_space_violation_fraction"], "depth_validation.maximum_free_space_violation_fraction"),
        ),
        holdout=HoldoutGates(
            minimum_board_poses=minimum_board_poses,
            minimum_correspondences_per_pose=minimum_per_pose,
            minimum_total_correspondences=minimum_total,
            maximum_reprojection_median_px=reprojection_median,
            maximum_reprojection_p95_px=reprojection_p95,
        ),
        texture=TextureGates(
            maximum_depth_residual_floor_m=_positive(texture["maximum_depth_residual_floor_m"], "texture.maximum_depth_residual_floor_m"),
            maximum_depth_residual_range_fraction=_positive(texture["maximum_depth_residual_range_fraction"], "texture.maximum_depth_residual_range_fraction"),
            minimum_independent_views=_integer(texture["minimum_independent_views"], "texture.minimum_independent_views"),
            maximum_linear_rgb_channel_stddev=_nonnegative(texture["maximum_linear_rgb_channel_stddev"], "texture.maximum_linear_rgb_channel_stddev"),
            minimum_surface_coverage=_probability(texture["minimum_surface_coverage"], "texture.minimum_surface_coverage"),
        ),
        augmentation=AugmentationGates(
            minimum_quality_level=minimum_quality,
            minimum_depth_frames=minimum_depth_frames,
            minimum_independent_views=minimum_augmentation_views,
            maximum_depth_stddev_m=_nonnegative(augmentation["maximum_depth_stddev_m"], "augmentation.maximum_depth_stddev_m"),
            maximum_hole_radius_m=_positive(augmentation["maximum_hole_radius_m"], "augmentation.maximum_hole_radius_m"),
        ),
        depth_anything=DepthAnythingGates(
            schema_version=depth_schema,
            minimum_anchor_count=minimum_anchors,
            grid_columns=grid_columns,
            grid_rows=grid_rows,
            holdout_fraction=holdout_fraction,
            minimum_holdout_count=minimum_holdout,
            huber_delta_inverse_m=huber_delta,
            maximum_irls_iterations=max_iterations,
            parameter_relative_tolerance=parameter_tolerance,
            maximum_holdout_median_residual_m=holdout_median,
            maximum_holdout_p95_residual_m=holdout_p95,
            maximum_holdout_absolute_bias_m=holdout_bias,
            bootstrap_replicates=bootstrap_replicates,
            maximum_scale_stddev_fraction=scale_stability,
            maximum_uncertainty_m=uncertainty,
            maximum_boundary_distance_m=boundary_distance,
            model_name=_string(depth_anything["model_name"], "depth_anything.model_name"),
            weights_sha256=_hash(depth_anything["weights_sha256"], "depth_anything.weights_sha256"),
            preprocessing_revision=_string(depth_anything["preprocessing_revision"], "depth_anything.preprocessing_revision"),
            input_size=(input_width, input_height),
            precision=precision,
        ),
    )


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _numeric_sequence(
    value: Any,
    path: str,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be an array")
    if length is not None and len(value) != length:
        raise ConfigError(f"{path} must contain exactly {length} numbers")
    return tuple(_number(component, f"{path}[{index}]") for index, component in enumerate(value))


def _rigid_row_major(value: Any, path: str) -> tuple[float, ...]:
    matrix = _numeric_sequence(value, path, length=16)
    if any(abs(matrix[index] - target) > 1e-6 for index, target in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0))):
        raise ConfigError(f"{path} must have a homogeneous final row")
    _validate_rotation_3x3(
        matrix[0:3] + matrix[4:7] + matrix[8:11],
        f"{path} rotation",
    )
    return matrix


def _validate_rotation_3x3(rotation: Sequence[float], path: str) -> None:
    rows = (rotation[0:3], rotation[3:6], rotation[6:9])
    for row in rows:
        if abs(sum(component * component for component in row) - 1.0) > 1e-3:
            raise ConfigError(f"{path} rotation must be orthonormal")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if abs(sum(rows[left][index] * rows[right][index] for index in range(3))) > 1e-3:
            raise ConfigError(f"{path} rotation must be orthonormal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ConfigError(f"{path} rotation must have determinant one")


@dataclass(frozen=True)
class CameraInfoContract:
    width: int
    height: int
    distortion_model: str
    k: tuple[float, ...]
    d: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]
    rectified: bool


@dataclass(frozen=True)
class RigidTransformContract:
    sha256: str
    row_major: tuple[float, ...]


@dataclass(frozen=True)
class FrameChain:
    lidar_frame_id: str
    color_optical_frame_id: str
    depth_optical_frame_id: str
    device_serial: str
    color_camera_info_sha256: str
    color_camera_info: CameraInfoContract
    depth_camera_info_sha256: str
    depth_camera_info: CameraInfoContract
    t_c_l: RigidTransformContract
    t_d_c: RigidTransformContract


@dataclass(frozen=True)
class BoardTarget:
    family: str
    dictionary: str
    rows: int
    columns: int
    marker_length_m: float
    square_length_m: float
    pattern_width_m: float
    pattern_height_m: float
    source_template_pdf_sha256: str
    source_pdf_nominal_marker_length_m: float
    source_pdf_nominal_square_length_m: float
    dimension_authority: str
    geometry_sha256: str


@dataclass(frozen=True)
class ClockFitEvidence:
    slope: float
    offset_ns: float
    sample_count: int
    span_s: float
    inlier_fraction: float
    median_absolute_residual_ms: float
    p95_residual_ms: float
    maximum_prediction_stddev_ms: float
    estimator_version: str
    sha256: str


@dataclass(frozen=True)
class CalibrationRepeatability:
    lidar_imu_translation_mm: float
    lidar_imu_rotation_deg: float
    lidar_camera_translation_mm: float
    lidar_camera_rotation_deg: float


@dataclass(frozen=True)
class TimingRepeatability:
    lidar_imu_repeatability_ms: float
    lidar_camera_repeatability_ms: float


@dataclass(frozen=True)
class LidarImuEvidenceContract:
    source_capture_sha256: tuple[str, ...]
    result_artifact_sha256: tuple[str, ...]
    transform_sha256: tuple[str, ...]
    raw_lidar_topic: str
    raw_imu_topic: str
    lidar_frame_id: str
    imu_frame_id: str
    solver_name: str
    solver_version: str
    solver_implementation_sha256: str
    uses_vendor_extrinsic: bool


@dataclass(frozen=True)
class HoldoutEvidence:
    artifact_sha256: str
    accepted_board_poses: int
    correspondences_per_pose: tuple[int, ...]
    total_correspondences: int
    validated_range_min_m: float
    validated_range_max_m: float
    camera_reprojection_median_px: float
    camera_reprojection_p95_px: float
    depth_median_absolute_residual_m: float
    depth_p95_absolute_residual_m: float
    absolute_mean_signed_bias_m: float
    occlusion_conflict_fraction: float
    free_space_violation_fraction: float
    t_d_c_validated: bool


@dataclass(frozen=True)
class CalibrationManifestV2:
    status: str
    calibration_id: str
    robot_serial: str
    estimation_artifact_sha256: tuple[str, ...]
    holdout_artifact_sha256: str
    target: BoardTarget
    solver_version: str
    solver_implementation_sha256: str
    frame_chain: FrameChain
    clock_fits: Mapping[str, ClockFitEvidence]
    lidar_imu_evidence: LidarImuEvidenceContract
    repeatability: CalibrationRepeatability
    timing: TimingRepeatability
    holdout_evidence: HoldoutEvidence


@dataclass(frozen=True)
class RawCaptureEvidence:
    sha256_verified: bool
    artifact_count: int
    artifact_sha256: tuple[str, ...]
    required_streams: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionIdentityEvidence:
    pose_generation: str
    fusion_config_sha256: str
    appearance_config_sha256: str
    roi_sha256: str
    mask_sha256: str
    algorithm_version: str


@dataclass(frozen=True)
class DerivedProductStageContract:
    owner: str
    cache_key: str
    invalidates_on: tuple[str, ...]


@dataclass(frozen=True)
class GeometryValidationEvidence:
    independent_measurement_count: int
    independent_measurements_used_for_calibration: bool
    p95_dimension_error_mm: float
    maximum_dimension_error_mm: float
    accepted_loop_closures: int
    loop_point_to_plane_p95_mm: float


@dataclass(frozen=True)
class BimRegistrationEvidence:
    independent_structural_groups: int
    holdout_structural_groups: int
    holdout_p95_residual_mm: float
    rigid_se3_only: bool
    design_bim_hash: str


@dataclass(frozen=True)
class SessionEvidenceV2:
    calibration_id: str
    raw_capture: RawCaptureEvidence
    reconstruction_identity: ReconstructionIdentityEvidence
    derived_product_contract: Mapping[str, DerivedProductStageContract]
    geometry_validation: GeometryValidationEvidence
    bim_registration: BimRegistrationEvidence


def _parse_camera_info(value: Any, path: str) -> CameraInfoContract:
    payload = _object(value, path)
    expected = {"width", "height", "distortion_model", "k", "d", "r", "p", "rectified"}
    _keys(payload, expected, path)
    width = _integer(payload["width"], f"{path}.width")
    height = _integer(payload["height"], f"{path}.height")
    k = _numeric_sequence(payload["k"], f"{path}.k", length=9)
    r = _numeric_sequence(payload["r"], f"{path}.r", length=9)
    p = _numeric_sequence(payload["p"], f"{path}.p", length=12)
    if k[0] <= 0.0 or k[4] <= 0.0:
        raise ConfigError(f"{path}.k must contain positive fx and fy")
    if abs(k[6]) > 1e-9 or abs(k[7]) > 1e-9 or abs(k[8] - 1.0) > 1e-9:
        raise ConfigError(f"{path}.k must have a usable homogeneous camera row")
    if not (0.0 <= k[2] < width and 0.0 <= k[5] < height):
        raise ConfigError(f"{path}.k principal point must lie inside the image")
    if p[0] <= 0.0 or p[5] <= 0.0:
        raise ConfigError(f"{path}.p must contain positive fx and fy")
    if abs(p[8]) > 1e-9 or abs(p[9]) > 1e-9 or abs(p[10] - 1.0) > 1e-9 or abs(p[11]) > 1e-9:
        raise ConfigError(f"{path}.p must have a usable homogeneous projection row")
    if not (0.0 <= p[2] < width and 0.0 <= p[6] < height):
        raise ConfigError(f"{path}.p principal point must lie inside the image")
    _validate_rotation_3x3(r, f"{path}.r")
    return CameraInfoContract(
        width=width,
        height=height,
        distortion_model=_string(payload["distortion_model"], f"{path}.distortion_model"),
        k=k,
        d=_numeric_sequence(payload["d"], f"{path}.d"),
        r=r,
        p=p,
        rectified=_boolean(payload["rectified"], f"{path}.rectified"),
    )


def _parse_transform(value: Any, path: str, *, expected_serial: str | None = None) -> RigidTransformContract:
    payload = _object(value, path)
    expected = {"sha256", "row_major"}
    if expected_serial is not None:
        expected.add("device_serial")
    _keys(payload, expected, path)
    if expected_serial is not None and payload["device_serial"] != expected_serial:
        raise ConfigError(f"{path}.device_serial must match frame_chain.device_serial")
    row_major = _rigid_row_major(payload["row_major"], f"{path}.row_major")
    digest = _hash(payload["sha256"], f"{path}.sha256")
    identity: dict[str, Any] = {"row_major": list(row_major)}
    if expected_serial is not None:
        identity["device_serial"] = expected_serial
    if digest != canonical_sha256(identity):
        raise ConfigError(f"{path}.sha256 must hash the canonical transform identity")
    return RigidTransformContract(sha256=digest, row_major=row_major)


def _parse_clock_fit(value: Any, path: str, gates: ClockFitGates) -> ClockFitEvidence:
    payload = _object(value, path)
    expected = {
        "slope",
        "offset_ns",
        "sample_count",
        "span_s",
        "inlier_fraction",
        "median_absolute_residual_ms",
        "p95_residual_ms",
        "maximum_prediction_stddev_ms",
        "estimator_version",
        "sha256",
    }
    _keys(payload, expected, path)
    evidence = ClockFitEvidence(
        slope=_number(payload["slope"], f"{path}.slope"),
        offset_ns=_number(payload["offset_ns"], f"{path}.offset_ns"),
        sample_count=_integer(payload["sample_count"], f"{path}.sample_count"),
        span_s=_nonnegative(payload["span_s"], f"{path}.span_s"),
        inlier_fraction=_probability(payload["inlier_fraction"], f"{path}.inlier_fraction"),
        median_absolute_residual_ms=_nonnegative(payload["median_absolute_residual_ms"], f"{path}.median_absolute_residual_ms"),
        p95_residual_ms=_nonnegative(payload["p95_residual_ms"], f"{path}.p95_residual_ms"),
        maximum_prediction_stddev_ms=_nonnegative(payload["maximum_prediction_stddev_ms"], f"{path}.maximum_prediction_stddev_ms"),
        estimator_version=_string(payload["estimator_version"], f"{path}.estimator_version"),
        sha256=_hash(payload["sha256"], f"{path}.sha256"),
    )
    if evidence.median_absolute_residual_ms > evidence.p95_residual_ms:
        raise ConfigError(f"{path}.median_absolute_residual_ms must be <= p95_residual_ms")
    if not gates.accepts(
        sample_count=evidence.sample_count,
        span_s=evidence.span_s,
        inlier_fraction=evidence.inlier_fraction,
        slope_deviation_ppm=abs(evidence.slope - 1.0) * 1_000_000.0,
        median_absolute_residual_ms=evidence.median_absolute_residual_ms,
        p95_residual_ms=evidence.p95_residual_ms,
        prediction_stddev_ms=evidence.maximum_prediction_stddev_ms,
    ):
        raise ConfigError(f"{path} failed configured clock-fit gates")
    return evidence


def _three_unique_hashes(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{path} must contain exactly three hashes")
    hashes = tuple(_hash(item, path) for item in value)
    if len(set(hashes)) != 3:
        raise ConfigError(f"{path} must contain three unique artifacts")
    return hashes


def _parse_lidar_imu_evidence(value: Any, *, lidar_frame_id: str) -> LidarImuEvidenceContract:
    path = "calibration.lidar_imu_evidence"
    payload = _object(value, path)
    fields = {
        "source_capture_sha256",
        "result_artifact_sha256",
        "transform_sha256",
        "raw_lidar_topic",
        "raw_imu_topic",
        "lidar_frame_id",
        "imu_frame_id",
        "solver",
        "uses_vendor_extrinsic",
    }
    _keys(payload, fields, path)
    raw_lidar_topic = _string(payload["raw_lidar_topic"], f"{path}.raw_lidar_topic")
    raw_imu_topic = _string(payload["raw_imu_topic"], f"{path}.raw_imu_topic")
    if not raw_lidar_topic.startswith("/") or not raw_imu_topic.startswith("/"):
        raise ConfigError(f"{path} raw sensor topics must be absolute ROS names")
    evidence_lidar_frame = _string(payload["lidar_frame_id"], f"{path}.lidar_frame_id")
    imu_frame = _string(payload["imu_frame_id"], f"{path}.imu_frame_id")
    if evidence_lidar_frame != lidar_frame_id:
        raise ConfigError(f"{path}.lidar_frame_id must match frame_chain.lidar_frame_id")
    if imu_frame == evidence_lidar_frame:
        raise ConfigError(f"{path}.imu_frame_id must be distinct from lidar_frame_id")
    if _boolean(payload["uses_vendor_extrinsic"], f"{path}.uses_vendor_extrinsic"):
        raise ConfigError(f"{path} cannot use a vendor/provisional extrinsic as independent evidence")
    solver = _object(payload["solver"], f"{path}.solver")
    _keys(solver, {"name", "version", "implementation_sha256"}, f"{path}.solver")
    return LidarImuEvidenceContract(
        source_capture_sha256=_three_unique_hashes(payload["source_capture_sha256"], f"{path}.source_capture_sha256"),
        result_artifact_sha256=_three_unique_hashes(payload["result_artifact_sha256"], f"{path}.result_artifact_sha256"),
        transform_sha256=_three_unique_hashes(payload["transform_sha256"], f"{path}.transform_sha256"),
        raw_lidar_topic=raw_lidar_topic,
        raw_imu_topic=raw_imu_topic,
        lidar_frame_id=evidence_lidar_frame,
        imu_frame_id=imu_frame,
        solver_name=_string(solver["name"], f"{path}.solver.name"),
        solver_version=_string(solver["version"], f"{path}.solver.version"),
        solver_implementation_sha256=_hash(solver["implementation_sha256"], f"{path}.solver.implementation_sha256"),
        uses_vendor_extrinsic=False,
    )


def parse_calibration_manifest(
    value: Mapping[str, Any],
    appearance: AppearanceGates | None = None,
) -> CalibrationManifestV2:
    """Parse a complete schema-v2 production calibration manifest."""
    payload = _object(value, "calibration")
    expected = {
        "schema_version",
        "status",
        "calibration_id",
        "robot",
        "capture_sets",
        "target",
        "solver",
        "frame_chain",
        "clock_fits",
        "lidar_imu_evidence",
        "repeatability",
        "timing",
        "holdout_evidence",
    }
    _keys(payload, expected, "calibration")
    schema_version = _integer(payload["schema_version"], "calibration.schema_version")
    if schema_version != 2:
        raise ConfigError(f"calibration.schema_version unsupported: {schema_version}")
    if appearance is None:
        appearance = load_reconstruction_config().appearance

    status = _string(payload["status"], "calibration.status")
    if status not in {"provisional", "PROVISIONAL_AUTOCAL", "measured"}:
        raise ConfigError("calibration.status is unknown")

    robot = _object(payload["robot"], "calibration.robot")
    _keys(robot, {"model", "serial"}, "calibration.robot")
    if robot["model"] != "go2_edu":
        raise ConfigError("calibration.robot.model must be go2_edu")
    robot_serial = _string(robot["serial"], "calibration.robot.serial")

    captures = _object(payload["capture_sets"], "calibration.capture_sets")
    _keys(captures, {"estimation_artifact_sha256", "holdout_artifact_sha256"}, "calibration.capture_sets")
    estimation_values = captures["estimation_artifact_sha256"]
    if not isinstance(estimation_values, list) or len(estimation_values) != 3:
        raise ConfigError("capture_sets.estimation_artifact_sha256 must contain exactly three hashes")
    estimation_hashes = tuple(_hash(item, "capture_sets.estimation_artifact_sha256") for item in estimation_values)
    if len(set(estimation_hashes)) != 3:
        raise ConfigError("capture_sets.estimation_artifact_sha256 must be independent unique captures")
    holdout_hash = _hash(captures["holdout_artifact_sha256"], "capture_sets.holdout_artifact_sha256")
    if holdout_hash in estimation_hashes:
        raise ConfigError("capture_sets.holdout_artifact_sha256 must be excluded from estimation")

    target = _object(payload["target"], "calibration.target")
    target_fields = {
        "family",
        "dictionary",
        "rows",
        "columns",
        "marker_length_m",
        "square_length_m",
        "pattern_width_m",
        "pattern_height_m",
        "source_template_pdf_sha256",
        "source_pdf_nominal_marker_length_m",
        "source_pdf_nominal_square_length_m",
        "dimension_authority",
        "geometry_sha256",
    }
    _keys(target, target_fields, "calibration.target")
    family = _string(target["family"], "target.family")
    dictionary = _string(target["dictionary"], "target.dictionary")
    rows = _integer(target["rows"], "target.rows")
    columns = _integer(target["columns"], "target.columns")
    if family != "charuco":
        raise ConfigError("target.family must be charuco")
    if dictionary != "DICT_4X4_100":
        raise ConfigError("target.dictionary must be DICT_4X4_100 for the locked Go2 target")
    if rows != 9 or columns != 12:
        raise ConfigError("target rows/columns must be 9 by 12 for the locked Go2 target")
    marker_length = _positive(target["marker_length_m"], "target.marker_length_m")
    square_length = _positive(target["square_length_m"], "target.square_length_m")
    pattern_width = _positive(target["pattern_width_m"], "target.pattern_width_m")
    pattern_height = _positive(target["pattern_height_m"], "target.pattern_height_m")
    source_pdf_hash = _hash(target["source_template_pdf_sha256"], "target.source_template_pdf_sha256")
    nominal_marker = _positive(target["source_pdf_nominal_marker_length_m"], "target.source_pdf_nominal_marker_length_m")
    nominal_square = _positive(target["source_pdf_nominal_square_length_m"], "target.source_pdf_nominal_square_length_m")
    dimension_authority = _string(target["dimension_authority"], "target.dimension_authority")
    if dimension_authority != "physical_print":
        raise ConfigError("target.dimension_authority must be physical_print")
    if marker_length >= square_length:
        raise ConfigError("target.marker_length_m must be less than square_length_m")
    if not math.isclose(marker_length, 0.0145, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError("target.marker_length_m must match the measured 14.5 mm Go2 lab marker")
    if not math.isclose(square_length, 0.020, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError("target.square_length_m must match the measured 20.0 mm Go2 lab square")
    if not math.isclose(pattern_width, columns * square_length, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError("target.pattern_width_m must equal columns * square_length_m")
    if not math.isclose(pattern_height, rows * square_length, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError("target.pattern_height_m must equal rows * square_length_m")
    geometry_payload = {
        "family": family,
        "dictionary": dictionary,
        "rows": rows,
        "columns": columns,
        "marker_length_m": marker_length,
        "square_length_m": square_length,
        "pattern_width_m": pattern_width,
        "pattern_height_m": pattern_height,
        "source_template_pdf_sha256": source_pdf_hash,
        "source_pdf_nominal_marker_length_m": nominal_marker,
        "source_pdf_nominal_square_length_m": nominal_square,
        "dimension_authority": dimension_authority,
    }
    geometry_hash = _hash(target["geometry_sha256"], "target.geometry_sha256")
    if geometry_hash != canonical_sha256(geometry_payload):
        raise ConfigError("target.geometry_sha256 must hash the canonical measured board geometry")
    board = BoardTarget(
        family=family,
        dictionary=dictionary,
        rows=rows,
        columns=columns,
        marker_length_m=marker_length,
        square_length_m=square_length,
        pattern_width_m=pattern_width,
        pattern_height_m=pattern_height,
        source_template_pdf_sha256=source_pdf_hash,
        source_pdf_nominal_marker_length_m=nominal_marker,
        source_pdf_nominal_square_length_m=nominal_square,
        dimension_authority=dimension_authority,
        geometry_sha256=geometry_hash,
    )

    solver = _object(payload["solver"], "calibration.solver")
    _keys(solver, {"version", "implementation_sha256"}, "calibration.solver")
    solver_version = _string(solver["version"], "solver.version")
    solver_hash = _hash(solver["implementation_sha256"], "solver.implementation_sha256")

    frame = _object(payload["frame_chain"], "calibration.frame_chain")
    frame_fields = {
        "lidar_frame_id",
        "color_optical_frame_id",
        "depth_optical_frame_id",
        "device_serial",
        "color_camera_info_sha256",
        "color_camera_info",
        "depth_camera_info_sha256",
        "depth_camera_info",
        "t_c_l",
        "t_d_c",
    }
    _keys(frame, frame_fields, "calibration.frame_chain")
    color_frame = _string(frame["color_optical_frame_id"], "frame_chain.color_optical_frame_id")
    depth_frame = _string(frame["depth_optical_frame_id"], "frame_chain.depth_optical_frame_id")
    if color_frame == depth_frame:
        raise ConfigError("color and depth optical frame IDs must be distinct")
    device_serial = _string(frame["device_serial"], "frame_chain.device_serial")
    frame_chain = FrameChain(
        lidar_frame_id=_string(frame["lidar_frame_id"], "frame_chain.lidar_frame_id"),
        color_optical_frame_id=color_frame,
        depth_optical_frame_id=depth_frame,
        device_serial=device_serial,
        color_camera_info_sha256=_hash(frame["color_camera_info_sha256"], "frame_chain.color_camera_info_sha256"),
        color_camera_info=_parse_camera_info(frame["color_camera_info"], "frame_chain.color_camera_info"),
        depth_camera_info_sha256=_hash(frame["depth_camera_info_sha256"], "frame_chain.depth_camera_info_sha256"),
        depth_camera_info=_parse_camera_info(frame["depth_camera_info"], "frame_chain.depth_camera_info"),
        t_c_l=_parse_transform(frame["t_c_l"], "frame_chain.t_c_l"),
        t_d_c=_parse_transform(frame["t_d_c"], "frame_chain.t_d_c", expected_serial=device_serial),
    )

    fits_payload = _object(payload["clock_fits"], "calibration.clock_fits")
    _keys(fits_payload, {"lidar", "color", "depth"}, "calibration.clock_fits")
    clock_fits = {
        stream: _parse_clock_fit(fits_payload[stream], f"clock_fits.{stream}", appearance.clock_fit)
        for stream in ("lidar", "color", "depth")
    }
    lidar_imu_evidence = _parse_lidar_imu_evidence(
        payload["lidar_imu_evidence"],
        lidar_frame_id=frame_chain.lidar_frame_id,
    )

    repeatability_payload = _object(payload["repeatability"], "calibration.repeatability")
    repeatability_fields = {
        "lidar_imu_translation_mm",
        "lidar_imu_rotation_deg",
        "lidar_camera_translation_mm",
        "lidar_camera_rotation_deg",
    }
    _keys(repeatability_payload, repeatability_fields, "calibration.repeatability")
    repeatability = CalibrationRepeatability(
        lidar_imu_translation_mm=_nonnegative(repeatability_payload["lidar_imu_translation_mm"], "repeatability.lidar_imu_translation_mm"),
        lidar_imu_rotation_deg=_nonnegative(repeatability_payload["lidar_imu_rotation_deg"], "repeatability.lidar_imu_rotation_deg"),
        lidar_camera_translation_mm=_nonnegative(repeatability_payload["lidar_camera_translation_mm"], "repeatability.lidar_camera_translation_mm"),
        lidar_camera_rotation_deg=_nonnegative(repeatability_payload["lidar_camera_rotation_deg"], "repeatability.lidar_camera_rotation_deg"),
    )

    timing_payload = _object(payload["timing"], "calibration.timing")
    timing_fields = {"lidar_imu_repeatability_ms", "lidar_camera_repeatability_ms"}
    _keys(timing_payload, timing_fields, "calibration.timing")
    timing = TimingRepeatability(
        lidar_imu_repeatability_ms=_nonnegative(timing_payload["lidar_imu_repeatability_ms"], "timing.lidar_imu_repeatability_ms"),
        lidar_camera_repeatability_ms=_nonnegative(timing_payload["lidar_camera_repeatability_ms"], "timing.lidar_camera_repeatability_ms"),
    )

    holdout_payload = _object(payload["holdout_evidence"], "calibration.holdout_evidence")
    holdout_fields = {
        "artifact_sha256",
        "accepted_board_poses",
        "correspondences_per_pose",
        "total_correspondences",
        "validated_range_min_m",
        "validated_range_max_m",
        "camera_reprojection_median_px",
        "camera_reprojection_p95_px",
        "depth_median_absolute_residual_m",
        "depth_p95_absolute_residual_m",
        "absolute_mean_signed_bias_m",
        "occlusion_conflict_fraction",
        "free_space_violation_fraction",
        "t_d_c_validated",
    }
    _keys(holdout_payload, holdout_fields, "calibration.holdout_evidence")
    evidence_hash = _hash(holdout_payload["artifact_sha256"], "holdout_evidence.artifact_sha256")
    if evidence_hash != holdout_hash:
        raise ConfigError("holdout_evidence.artifact_sha256 must match the excluded holdout")
    board_poses = _integer(holdout_payload["accepted_board_poses"], "holdout_evidence.accepted_board_poses")
    correspondence_values = holdout_payload["correspondences_per_pose"]
    if not isinstance(correspondence_values, list) or len(correspondence_values) != board_poses:
        raise ConfigError("holdout_evidence.correspondences_per_pose must contain one count per accepted pose")
    correspondences = tuple(
        _integer(item, f"holdout_evidence.correspondences_per_pose[{index}]")
        for index, item in enumerate(correspondence_values)
    )
    total_correspondences = _integer(holdout_payload["total_correspondences"], "holdout_evidence.total_correspondences")
    if sum(correspondences) != total_correspondences:
        raise ConfigError("holdout_evidence.total_correspondences must equal the per-pose sum")
    t_d_c_validated = _boolean(holdout_payload["t_d_c_validated"], "holdout_evidence.t_d_c_validated")
    if not t_d_c_validated:
        raise ConfigError("holdout_evidence.t_d_c_validated must be true")
    holdout = HoldoutEvidence(
        artifact_sha256=evidence_hash,
        accepted_board_poses=board_poses,
        correspondences_per_pose=correspondences,
        total_correspondences=total_correspondences,
        validated_range_min_m=_nonnegative(holdout_payload["validated_range_min_m"], "holdout_evidence.validated_range_min_m"),
        validated_range_max_m=_positive(holdout_payload["validated_range_max_m"], "holdout_evidence.validated_range_max_m"),
        camera_reprojection_median_px=_nonnegative(holdout_payload["camera_reprojection_median_px"], "holdout_evidence.camera_reprojection_median_px"),
        camera_reprojection_p95_px=_nonnegative(holdout_payload["camera_reprojection_p95_px"], "holdout_evidence.camera_reprojection_p95_px"),
        depth_median_absolute_residual_m=_nonnegative(holdout_payload["depth_median_absolute_residual_m"], "holdout_evidence.depth_median_absolute_residual_m"),
        depth_p95_absolute_residual_m=_nonnegative(holdout_payload["depth_p95_absolute_residual_m"], "holdout_evidence.depth_p95_absolute_residual_m"),
        absolute_mean_signed_bias_m=_nonnegative(holdout_payload["absolute_mean_signed_bias_m"], "holdout_evidence.absolute_mean_signed_bias_m"),
        occlusion_conflict_fraction=_probability(holdout_payload["occlusion_conflict_fraction"], "holdout_evidence.occlusion_conflict_fraction"),
        free_space_violation_fraction=_probability(holdout_payload["free_space_violation_fraction"], "holdout_evidence.free_space_violation_fraction"),
        t_d_c_validated=t_d_c_validated,
    )
    if not appearance.holdout.accepts(
        board_pose_count=holdout.accepted_board_poses,
        minimum_correspondences_in_any_pose=min(holdout.correspondences_per_pose),
        total_correspondences=holdout.total_correspondences,
        reprojection_median_px=holdout.camera_reprojection_median_px,
        reprojection_p95_px=holdout.camera_reprojection_p95_px,
    ):
        raise ConfigError("holdout_evidence failed configured pose/correspondence/reprojection gates")
    if not appearance.depth_validation.accepts(
        correspondence_count=holdout.total_correspondences,
        median_absolute_residual_m=holdout.depth_median_absolute_residual_m,
        p95_absolute_residual_m=holdout.depth_p95_absolute_residual_m,
        absolute_mean_signed_bias_m=holdout.absolute_mean_signed_bias_m,
        occlusion_conflict_fraction=holdout.occlusion_conflict_fraction,
        free_space_violation_fraction=holdout.free_space_violation_fraction,
        minimum_observed_range_m=holdout.validated_range_min_m,
        maximum_observed_range_m=holdout.validated_range_max_m,
    ):
        raise ConfigError("holdout_evidence failed configured depth gates")

    return CalibrationManifestV2(
        status=status,
        calibration_id=_string(payload["calibration_id"], "calibration.calibration_id"),
        robot_serial=robot_serial,
        estimation_artifact_sha256=estimation_hashes,
        holdout_artifact_sha256=holdout_hash,
        target=board,
        solver_version=solver_version,
        solver_implementation_sha256=solver_hash,
        frame_chain=frame_chain,
        clock_fits=clock_fits,
        lidar_imu_evidence=lidar_imu_evidence,
        repeatability=repeatability,
        timing=timing,
        holdout_evidence=holdout,
    )


_DERIVED_PRODUCT_CONTRACT = {
    "measured_fusion": (
        "LIDAR_MEASURED",
        {"raw_artifact", "pose_generation", "fusion_config", "roi", "mask", "algorithm"},
    ),
    "calibration": (
        "CAMERA_LIDAR_CALIBRATION",
        {"capture", "target", "camera_info", "tf", "solver", "clock_fit"},
    ),
    "depth_validation": (
        "DEPTH_VALIDATION",
        {"measured_generation", "calibration", "appearance_config"},
    ),
    "texture": (
        "CALIBRATED_TEXTURE",
        {"measured_generation", "depth_validation", "calibration", "appearance_config", "roi", "mask", "algorithm"},
    ),
    "augmentation": (
        "RGBD_AUGMENTED",
        {"measured_generation", "depth_validation", "calibration", "appearance_config", "roi", "mask", "algorithm"},
    ),
    "evidence": ("VALIDATING", {"referenced_stage_output"}),
}


def _string_sequence(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be an array")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def parse_session_evidence(value: Mapping[str, Any]) -> SessionEvidenceV2:
    """Parse complete schema-v2 session evidence without legacy fallthrough."""
    payload = _object(value, "session")
    expected = {
        "schema_version",
        "calibration_id",
        "raw_capture",
        "reconstruction_identity",
        "derived_product_contract",
        "geometry_validation",
        "bim_registration",
    }
    _keys(payload, expected, "session")
    schema_version = _integer(payload["schema_version"], "session.schema_version")
    if schema_version != 2:
        raise ConfigError(f"session.schema_version unsupported: {schema_version}")

    raw_payload = _object(payload["raw_capture"], "session.raw_capture")
    raw_fields = {"sha256_verified", "artifact_count", "artifact_sha256", "required_streams"}
    _keys(raw_payload, raw_fields, "session.raw_capture")
    artifact_count = _integer(raw_payload["artifact_count"], "raw_capture.artifact_count", minimum=0)
    artifact_values = raw_payload["artifact_sha256"]
    if not isinstance(artifact_values, list) or len(artifact_values) != artifact_count:
        raise ConfigError("raw_capture.artifact_sha256 must match artifact_count")
    artifact_hashes = tuple(
        _hash(item, f"raw_capture.artifact_sha256[{index}]")
        for index, item in enumerate(artifact_values)
    )
    if len(artifact_hashes) != len(set(artifact_hashes)):
        raise ConfigError("raw_capture.artifact_sha256 must not contain duplicates")
    raw_capture = RawCaptureEvidence(
        sha256_verified=_boolean(raw_payload["sha256_verified"], "raw_capture.sha256_verified"),
        artifact_count=artifact_count,
        artifact_sha256=artifact_hashes,
        required_streams=_string_sequence(raw_payload["required_streams"], "raw_capture.required_streams"),
    )

    identity_payload = _object(payload["reconstruction_identity"], "session.reconstruction_identity")
    identity_fields = {
        "pose_generation",
        "fusion_config_sha256",
        "appearance_config_sha256",
        "roi_sha256",
        "mask_sha256",
        "algorithm_version",
    }
    _keys(identity_payload, identity_fields, "session.reconstruction_identity")
    reconstruction_identity = ReconstructionIdentityEvidence(
        pose_generation=_string(identity_payload["pose_generation"], "reconstruction_identity.pose_generation"),
        fusion_config_sha256=_hash(identity_payload["fusion_config_sha256"], "reconstruction_identity.fusion_config_sha256"),
        appearance_config_sha256=_hash(identity_payload["appearance_config_sha256"], "reconstruction_identity.appearance_config_sha256"),
        roi_sha256=_hash(identity_payload["roi_sha256"], "reconstruction_identity.roi_sha256"),
        mask_sha256=_hash(identity_payload["mask_sha256"], "reconstruction_identity.mask_sha256"),
        algorithm_version=_string(identity_payload["algorithm_version"], "reconstruction_identity.algorithm_version"),
    )

    contracts_payload = _object(payload["derived_product_contract"], "session.derived_product_contract")
    _keys(contracts_payload, set(_DERIVED_PRODUCT_CONTRACT), "session.derived_product_contract")
    contracts: dict[str, DerivedProductStageContract] = {}
    for stage, (expected_owner, expected_invalidations) in _DERIVED_PRODUCT_CONTRACT.items():
        contract_payload = _object(contracts_payload[stage], f"derived_product_contract.{stage}")
        _keys(contract_payload, {"owner", "cache_key", "invalidates_on"}, f"derived_product_contract.{stage}")
        owner = _string(contract_payload["owner"], f"derived_product_contract.{stage}.owner")
        if owner != expected_owner:
            raise ConfigError(f"derived_product_contract.{stage}.owner must be {expected_owner}")
        invalidations = _string_sequence(
            contract_payload["invalidates_on"],
            f"derived_product_contract.{stage}.invalidates_on",
        )
        if set(invalidations) != expected_invalidations:
            raise ConfigError(f"derived_product_contract.{stage}.invalidates_on is incomplete")
        contracts[stage] = DerivedProductStageContract(
            owner=owner,
            cache_key=_hash(contract_payload["cache_key"], f"derived_product_contract.{stage}.cache_key"),
            invalidates_on=invalidations,
        )

    geometry_payload = _object(payload["geometry_validation"], "session.geometry_validation")
    geometry_fields = {
        "independent_measurement_count",
        "independent_measurements_used_for_calibration",
        "p95_dimension_error_mm",
        "maximum_dimension_error_mm",
        "accepted_loop_closures",
        "loop_point_to_plane_p95_mm",
    }
    _keys(geometry_payload, geometry_fields, "session.geometry_validation")
    geometry = GeometryValidationEvidence(
        independent_measurement_count=_integer(geometry_payload["independent_measurement_count"], "geometry_validation.independent_measurement_count", minimum=0),
        independent_measurements_used_for_calibration=_boolean(geometry_payload["independent_measurements_used_for_calibration"], "geometry_validation.independent_measurements_used_for_calibration"),
        p95_dimension_error_mm=_nonnegative(geometry_payload["p95_dimension_error_mm"], "geometry_validation.p95_dimension_error_mm"),
        maximum_dimension_error_mm=_nonnegative(geometry_payload["maximum_dimension_error_mm"], "geometry_validation.maximum_dimension_error_mm"),
        accepted_loop_closures=_integer(geometry_payload["accepted_loop_closures"], "geometry_validation.accepted_loop_closures", minimum=0),
        loop_point_to_plane_p95_mm=_nonnegative(geometry_payload["loop_point_to_plane_p95_mm"], "geometry_validation.loop_point_to_plane_p95_mm"),
    )
    if geometry.p95_dimension_error_mm > geometry.maximum_dimension_error_mm:
        raise ConfigError("geometry_validation.p95_dimension_error_mm must be <= maximum_dimension_error_mm")

    bim_payload = _object(payload["bim_registration"], "session.bim_registration")
    bim_fields = {
        "independent_structural_groups",
        "holdout_structural_groups",
        "holdout_p95_residual_mm",
        "rigid_se3_only",
        "design_bim_hash",
    }
    _keys(bim_payload, bim_fields, "session.bim_registration")
    bim = BimRegistrationEvidence(
        independent_structural_groups=_integer(bim_payload["independent_structural_groups"], "bim_registration.independent_structural_groups", minimum=0),
        holdout_structural_groups=_integer(bim_payload["holdout_structural_groups"], "bim_registration.holdout_structural_groups", minimum=0),
        holdout_p95_residual_mm=_nonnegative(bim_payload["holdout_p95_residual_mm"], "bim_registration.holdout_p95_residual_mm"),
        rigid_se3_only=_boolean(bim_payload["rigid_se3_only"], "bim_registration.rigid_se3_only"),
        design_bim_hash=_hash(bim_payload["design_bim_hash"], "bim_registration.design_bim_hash"),
    )
    if bim.holdout_structural_groups > bim.independent_structural_groups:
        raise ConfigError("bim_registration.holdout_structural_groups cannot exceed independent_structural_groups")

    return SessionEvidenceV2(
        calibration_id=_string(payload["calibration_id"], "session.calibration_id"),
        raw_capture=raw_capture,
        reconstruction_identity=reconstruction_identity,
        derived_product_contract=contracts,
        geometry_validation=geometry,
        bim_registration=bim,
    )


def load_reconstruction_config(
    fusion_path: str | Path = DEFAULT_FUSION_CONFIG_PATH,
    appearance_path: str | Path = DEFAULT_APPEARANCE_CONFIG_PATH,
) -> ReconstructionConfig:
    fusion_payload = _read_json(fusion_path)
    appearance_payload = _read_json(appearance_path)
    return ReconstructionConfig(
        fusion=_parse_fusion(fusion_payload),
        appearance=_parse_appearance(appearance_payload),
        fusion_sha256=canonical_sha256(fusion_payload),
        appearance_sha256=canonical_sha256(appearance_payload),
    )
