import json
from pathlib import Path

import pytest

from mapping.pc.go2_mapping_quality.reconstruction_config import (
    ConfigError,
    canonical_sha256,
    load_reconstruction_config,
    sha256_file,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
FUSION = CONFIG_ROOT / "fusion_gates.json"
APPEARANCE = CONFIG_ROOT / "appearance_gates.json"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _write_json(path, value, *, indent=None):
    path.write_text(json.dumps(value, indent=indent), encoding="utf-8")
    return path


def _payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(config, **overrides):
    values = {
        "artifact_hashes": (HASH_A, HASH_B),
        "pose_generation": "pg-1",
        "calibration_hash": HASH_C,
        "roi_hash": HASH_D,
        "mask_hash": "e" * 64,
        "algorithm_version": "fusion-v1",
    }
    values.update(overrides)
    return config.derived_identity(**values)


def test_schema_v2_loads_typed_gates_and_hashes_canonical_json(tmp_path):
    config = load_reconstruction_config(FUSION, APPEARANCE)
    identity = _identity(config)

    reformatted_fusion = _write_json(
        tmp_path / "fusion.json",
        dict(reversed(list(_payload(FUSION).items()))),
        indent=4,
    )

    assert config.fusion.schema_version == 2
    assert config.appearance.schema_version == 2
    assert config.fusion_sha256 == sha256_file(FUSION)
    assert config.appearance_sha256 == sha256_file(APPEARANCE)
    assert sha256_file(reformatted_fusion) == config.fusion_sha256
    assert canonical_sha256(_payload(FUSION)) == config.fusion_sha256
    assert identity.fusion_config_sha256 == config.fusion_sha256
    assert identity.appearance_config_sha256 == config.appearance_sha256
    assert len(identity.cache_key) == 64


@pytest.mark.parametrize(
    "override",
    [
        {"artifact_hashes": (HASH_A, "f" * 64)},
        {"pose_generation": "pg-2"},
        {"calibration_hash": "f" * 64},
        {"roi_hash": "f" * 64},
        {"mask_hash": "f" * 64},
        {"algorithm_version": "fusion-v2"},
    ],
)
def test_derived_identity_cache_key_changes_for_every_input_domain(override):
    config = load_reconstruction_config(FUSION, APPEARANCE)

    assert _identity(config, **override).cache_key != _identity(config).cache_key


def test_derived_identity_rejects_missing_or_malformed_domains():
    config = load_reconstruction_config(FUSION, APPEARANCE)

    with pytest.raises(ConfigError, match="artifact_hashes"):
        _identity(config, artifact_hashes=())
    with pytest.raises(ConfigError, match="calibration_hash"):
        _identity(config, calibration_hash="not-a-hash")
    with pytest.raises(ConfigError, match="pose_generation"):
        _identity(config, pose_generation="")


@pytest.mark.parametrize("signed_hash", ["+" + "a" * 63, "-" + "a" * 63])
def test_derived_identity_rejects_signed_64_character_hashes(signed_hash):
    config = load_reconstruction_config(FUSION, APPEARANCE)

    with pytest.raises(ConfigError, match="calibration_hash"):
        _identity(config, calibration_hash=signed_hash)


def test_loader_rejects_unknown_versions_and_fields(tmp_path):
    fusion = _payload(FUSION)
    fusion["schema_version"] = 99
    bad_version = _write_json(tmp_path / "bad-version.json", fusion)

    with pytest.raises(ConfigError, match="schema_version"):
        load_reconstruction_config(bad_version, APPEARANCE)

    fusion = _payload(FUSION)
    fusion["unexpected"] = 1
    unknown_field = _write_json(tmp_path / "unknown-field.json", fusion)

    with pytest.raises(ConfigError, match="unexpected"):
        load_reconstruction_config(unknown_field, APPEARANCE)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("filtering", "minimum_range_m", True),
        ("filtering", "maximum_range_m", float("nan")),
        ("measured_promotion", "minimum_static_probability", 1.01),
        ("independent_view", "minimum_direction_separation_deg", 181.0),
        ("dynamic_evidence", "unstable_free_space_fraction", -0.01),
        ("stability", "minimum_measured_coverage", False),
        ("metric_domain", "depth_tile_size_m", 0.0),
        ("surfacing", "display_point_budget", 0),
    ],
)
def test_fusion_loader_rejects_boolean_nonfinite_and_invalid_thresholds(
    tmp_path, section, field, value
):
    fusion = _payload(FUSION)
    fusion[section][field] = value
    path = _write_json(tmp_path / "fusion.json", fusion)

    with pytest.raises(ConfigError, match=field):
        load_reconstruction_config(path, APPEARANCE)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("depth_validation", "minimum_range_m", True),
        ("depth_validation", "maximum_p95_absolute_residual_m", float("inf")),
        ("depth_validation", "maximum_occlusion_conflict_fraction", 1.01),
        ("holdout", "minimum_board_poses", 0),
        ("holdout", "maximum_reprojection_p95_px", -0.01),
        ("texture", "maximum_linear_rgb_channel_stddev", False),
        ("texture", "minimum_surface_coverage", -0.01),
        ("augmentation", "maximum_hole_radius_m", 0.0),
    ],
)
def test_appearance_loader_rejects_boolean_nonfinite_and_invalid_thresholds(
    tmp_path, section, field, value
):
    appearance = _payload(APPEARANCE)
    appearance[section][field] = value
    path = _write_json(tmp_path / "appearance.json", appearance)

    with pytest.raises(ConfigError, match=field):
        load_reconstruction_config(FUSION, path)


def test_loader_rejects_inconsistent_ranges_and_percentiles(tmp_path):
    fusion = _payload(FUSION)
    fusion["filtering"]["minimum_range_m"] = fusion["filtering"]["maximum_range_m"]
    bad_fusion = _write_json(tmp_path / "fusion.json", fusion)

    appearance = _payload(APPEARANCE)
    appearance["depth_validation"]["maximum_median_absolute_residual_m"] = 0.07
    appearance["depth_validation"]["maximum_p95_absolute_residual_m"] = 0.06
    bad_appearance = _write_json(tmp_path / "appearance.json", appearance)

    with pytest.raises(ConfigError, match="minimum_range_m"):
        load_reconstruction_config(bad_fusion, APPEARANCE)
    with pytest.raises(ConfigError, match="maximum_median_absolute_residual_m"):
        load_reconstruction_config(FUSION, bad_appearance)


def test_loader_rejects_impossible_cross_field_thresholds(tmp_path):
    fusion = _payload(FUSION)
    fusion["measured_promotion"]["minimum_independent_views"] = 4
    bad_fusion = _write_json(tmp_path / "fusion.json", fusion)

    appearance = _payload(APPEARANCE)
    appearance["augmentation"]["minimum_independent_views"] = 4
    bad_appearance = _write_json(tmp_path / "appearance.json", appearance)

    with pytest.raises(ConfigError, match="minimum_independent_views"):
        load_reconstruction_config(bad_fusion, APPEARANCE)
    with pytest.raises(ConfigError, match="minimum_independent_views"):
        load_reconstruction_config(FUSION, bad_appearance)


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("observation_count", 2),
        ("independent_views", 1),
        ("temporal_support_s", 0.499999),
        ("principal_stddev_m", 0.020001),
        ("point_to_plane_p95_m", 0.030001),
        ("static_probability", 0.799999),
    ],
)
def test_every_measured_promotion_threshold_has_closed_boundary(field, failing_value):
    gates = load_reconstruction_config(FUSION, APPEARANCE).fusion.measured_promotion
    exact = {
        "observation_count": 3,
        "independent_views": 2,
        "temporal_support_s": 0.5,
        "principal_stddev_m": 0.02,
        "point_to_plane_p95_m": 0.03,
        "static_probability": 0.80,
    }

    assert gates.accepts(**exact) is True
    assert gates.accepts(**{**exact, field: failing_value}) is False


def test_independent_view_thresholds_are_inclusive_and_or_combined():
    gates = load_reconstruction_config(FUSION, APPEARANCE).fusion.independent_view

    assert gates.accepts(origin_baseline_m=0.20, direction_separation_deg=0.0)
    assert gates.accepts(origin_baseline_m=0.0, direction_separation_deg=10.0)
    assert not gates.accepts(origin_baseline_m=0.199999, direction_separation_deg=9.999999)


def test_filtering_radius_neighborhood_and_statistical_boundaries():
    gates = load_reconstruction_config(FUSION, APPEARANCE).fusion.filtering

    assert gates.accepts_range(range_m=0.3)
    assert gates.accepts_range(range_m=15.0)
    assert not gates.accepts_range(range_m=0.299999)
    assert not gates.accepts_range(range_m=15.000001)
    assert gates.within_radius(distance_m=0.05)
    assert not gates.within_radius(distance_m=0.050001)
    assert gates.has_radius_support(neighbor_count=2)
    assert not gates.has_radius_support(neighbor_count=1)
    assert gates.has_statistical_support(neighbor_count=16)
    assert not gates.has_statistical_support(neighbor_count=15)
    assert gates.accepts_statistical_distance(distance=1.15, mean=1.0, stddev=0.1)
    assert not gates.accepts_statistical_distance(distance=1.150001, mean=1.0, stddev=0.1)


def test_surfacing_voxel_point_and_triangle_boundaries():
    gates = load_reconstruction_config(FUSION, APPEARANCE).fusion.surfacing

    assert gates.accepts_output(
        voxel_size_m=0.015,
        point_count=200_000,
        triangle_count=300_000,
    )
    assert not gates.accepts_output(voxel_size_m=0.015001, point_count=200_000, triangle_count=300_000)
    assert not gates.accepts_output(voxel_size_m=0.015, point_count=200_001, triangle_count=300_000)
    assert not gates.accepts_output(voxel_size_m=0.015, point_count=200_000, triangle_count=300_001)


def test_dynamic_and_stability_thresholds_have_closed_boundaries():
    fusion = load_reconstruction_config(FUSION, APPEARANCE).fusion

    assert fusion.dynamic_evidence.is_unstable(vote_count=5, free_space_fraction=0.200001)
    assert not fusion.dynamic_evidence.is_unstable(vote_count=4, free_space_fraction=1.0)
    assert not fusion.dynamic_evidence.is_unstable(vote_count=5, free_space_fraction=0.20)
    assert fusion.dynamic_evidence.is_rejected(unstable_generations=3)
    assert not fusion.dynamic_evidence.is_rejected(unstable_generations=2)
    assert fusion.metric_domain.is_eligible(
        finite_post_filter_observations=2,
        tile_size_m=0.5,
    )
    assert not fusion.metric_domain.is_eligible(
        finite_post_filter_observations=1,
        tile_size_m=0.5,
    )
    assert not fusion.metric_domain.is_eligible(
        finite_post_filter_observations=2,
        tile_size_m=0.500001,
    )
    assert fusion.stability.accepts(
        measured_coverage=0.80,
        generation_rms_change_m=0.005,
        settled_generations=2,
    )
    assert not fusion.stability.accepts(0.799999, 0.005, 2)
    assert not fusion.stability.accepts(0.80, 0.005001, 2)
    assert not fusion.stability.accepts(0.80, 0.005, 1)


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("correspondence_count", 99),
        ("median_absolute_residual_m", 0.025001),
        ("p95_absolute_residual_m", 0.060001),
        ("absolute_mean_signed_bias_m", 0.015001),
        ("occlusion_conflict_fraction", 0.020001),
        ("free_space_violation_fraction", 0.010001),
        ("minimum_observed_range_m", 0.499999),
        ("maximum_observed_range_m", 4.000001),
    ],
)
def test_every_depth_region_threshold_has_closed_boundary(field, failing_value):
    gates = load_reconstruction_config(FUSION, APPEARANCE).appearance.depth_validation
    exact = {
        "correspondence_count": 100,
        "median_absolute_residual_m": 0.025,
        "p95_absolute_residual_m": 0.060,
        "absolute_mean_signed_bias_m": 0.015,
        "occlusion_conflict_fraction": 0.02,
        "free_space_violation_fraction": 0.01,
        "minimum_observed_range_m": 0.5,
        "maximum_observed_range_m": 4.0,
    }

    assert gates.accepts(**exact)
    assert not gates.accepts(**{**exact, field: failing_value})


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("board_pose_count", 2),
        ("minimum_correspondences_in_any_pose", 499),
        ("total_correspondences", 9_999),
        ("reprojection_median_px", 0.500001),
        ("reprojection_p95_px", 1.000001),
    ],
)
def test_every_holdout_threshold_has_closed_boundary(field, failing_value):
    gates = load_reconstruction_config(FUSION, APPEARANCE).appearance.holdout
    exact = {
        "board_pose_count": 3,
        "minimum_correspondences_in_any_pose": 500,
        "total_correspondences": 10_000,
        "reprojection_median_px": 0.5,
        "reprojection_p95_px": 1.0,
    }

    assert gates.accepts(**exact)
    assert not gates.accepts(**{**exact, field: failing_value})


def test_every_texture_threshold_has_closed_boundary():
    gates = load_reconstruction_config(FUSION, APPEARANCE).appearance.texture

    assert gates.accepts_sample(depth_residual_m=0.03, range_m=1.0)
    assert not gates.accepts_sample(depth_residual_m=0.030001, range_m=1.0)
    assert gates.accepts_sample(depth_residual_m=0.06, range_m=4.0)
    assert not gates.accepts_sample(depth_residual_m=0.060001, range_m=4.0)
    assert gates.accepts_vertex(independent_views=2, channel_stddev=0.10)
    assert not gates.accepts_vertex(independent_views=1, channel_stddev=0.10)
    assert not gates.accepts_vertex(independent_views=2, channel_stddev=0.100001)
    assert gates.delivery_authorized(surface_coverage=0.80)
    assert not gates.delivery_authorized(surface_coverage=0.799999)


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("depth_frame_count", 2),
        ("independent_views", 1),
        ("depth_stddev_m", 0.030001),
        ("hole_radius_m", 0.150001),
    ],
)
def test_every_augmentation_threshold_has_closed_boundary(field, failing_value):
    gates = load_reconstruction_config(FUSION, APPEARANCE).appearance.augmentation
    exact = {
        "quality_level": "CALIBRATED_DENSE",
        "depth_region_accepted": True,
        "depth_frame_count": 3,
        "independent_views": 2,
        "depth_stddev_m": 0.03,
        "completely_bounded": True,
        "hole_radius_m": 0.15,
    }

    assert gates.accepts(**exact)
    assert not gates.accepts(**{**exact, field: failing_value})
    assert not gates.accepts(**{**exact, "quality_level": "RAW_CAPTURE_VALID"})
    assert not gates.accepts(**{**exact, "depth_region_accepted": False})
    assert not gates.accepts(**{**exact, "completely_bounded": False})


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("sample_count", 199),
        ("span_s", 29.999999),
        ("inlier_fraction", 0.949999),
        ("slope_deviation_ppm", 50.000001),
        ("median_absolute_residual_ms", 1.000001),
        ("p95_residual_ms", 3.000001),
        ("prediction_stddev_ms", 1.000001),
    ],
)
def test_every_clock_fit_threshold_has_closed_boundary(field, failing_value):
    gates = load_reconstruction_config(FUSION, APPEARANCE).appearance.clock_fit
    exact = {
        "sample_count": 200,
        "span_s": 30.0,
        "inlier_fraction": 0.95,
        "slope_deviation_ppm": 50.0,
        "median_absolute_residual_ms": 1.0,
        "p95_residual_ms": 3.0,
        "prediction_stddev_ms": 1.0,
    }

    assert gates.accepts(**exact)
    assert not gates.accepts(**{**exact, field: failing_value})


def test_schema_v2_examples_document_frame_chain_evidence_and_stage_ownership():
    calibration = _payload(CONFIG_ROOT / "calibration_manifest.example.json")
    session = _payload(CONFIG_ROOT / "session_evidence.example.json")

    assert calibration["schema_version"] == 2
    assert set(calibration["capture_sets"]) == {
        "estimation_artifact_sha256",
        "holdout_artifact_sha256",
    }
    assert {
        "color_optical_frame_id",
        "depth_optical_frame_id",
        "color_camera_info",
        "color_camera_info_sha256",
        "depth_camera_info",
        "depth_camera_info_sha256",
        "t_c_l",
        "t_d_c",
    } <= set(calibration["frame_chain"])
    assert {
        "marker_length_m",
        "square_length_m",
        "pattern_width_m",
        "pattern_height_m",
        "source_template_pdf_sha256",
        "source_pdf_nominal_marker_length_m",
        "source_pdf_nominal_square_length_m",
        "dimension_authority",
        "geometry_sha256",
    } <= set(calibration["target"])
    assert calibration["target"]["dictionary"] == "DICT_4X4_100"
    assert calibration["target"]["columns"] == 12
    assert calibration["target"]["rows"] == 9
    assert calibration["target"]["marker_length_m"] == 0.0145
    assert calibration["target"]["square_length_m"] == 0.020
    assert calibration["target"]["pattern_width_m"] == 0.240
    assert calibration["target"]["pattern_height_m"] == 0.180
    assert calibration["target"]["source_template_pdf_sha256"].lower() == (
        "fe05d01ca5fdde303f712dad177f53c3ae75d3493ff7d9a783b010f892a2d743"
    )
    assert calibration["clock_fits"]
    assert calibration["holdout_evidence"]
    assert session["schema_version"] == 2
    assert set(session["derived_product_contract"]) == {
        "measured_fusion",
        "calibration",
        "depth_validation",
        "texture",
        "augmentation",
        "evidence",
    }


def test_gate_predicates_reject_invalid_runtime_measurements():
    config = load_reconstruction_config(FUSION, APPEARANCE)
    measured = config.fusion.measured_promotion
    depth = config.appearance.depth_validation
    clock = config.appearance.clock_fit
    texture = config.appearance.texture
    augmentation = config.appearance.augmentation

    assert not measured.accepts(
        observation_count=3,
        independent_views=2,
        temporal_support_s=0.5,
        principal_stddev_m=-0.001,
        point_to_plane_p95_m=0.03,
        static_probability=0.8,
    )
    assert not measured.accepts(
        observation_count=3,
        independent_views=2,
        temporal_support_s=0.5,
        principal_stddev_m=0.02,
        point_to_plane_p95_m=0.03,
        static_probability=1.01,
    )
    assert not config.fusion.stability.accepts(1.01, 0.005, 2)
    assert not depth.accepts(
        correspondence_count=100,
        median_absolute_residual_m=-0.001,
        p95_absolute_residual_m=0.06,
        absolute_mean_signed_bias_m=0.015,
        occlusion_conflict_fraction=0.02,
        free_space_violation_fraction=0.01,
        minimum_observed_range_m=0.5,
        maximum_observed_range_m=4.0,
    )
    assert not depth.accepts(
        correspondence_count=100,
        median_absolute_residual_m=0.02,
        p95_absolute_residual_m=0.01,
        absolute_mean_signed_bias_m=0.015,
        occlusion_conflict_fraction=0.02,
        free_space_violation_fraction=0.01,
        minimum_observed_range_m=0.5,
        maximum_observed_range_m=4.0,
    )
    assert not clock.accepts(
        sample_count=200,
        span_s=30.0,
        inlier_fraction=0.95,
        slope_deviation_ppm=0.0,
        median_absolute_residual_ms=float("nan"),
        p95_residual_ms=3.0,
        prediction_stddev_ms=1.0,
    )
    assert not texture.accepts_vertex(independent_views=2, channel_stddev=-0.001)
    assert not augmentation.accepts(
        quality_level="CALIBRATED_DENSE",
        depth_region_accepted=True,
        depth_frame_count=3,
        independent_views=2,
        depth_stddev_m=-0.001,
        completely_bounded=True,
        hole_radius_m=0.15,
    )
