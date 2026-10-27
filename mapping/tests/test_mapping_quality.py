import json
from pathlib import Path

import pytest

from mapping.pc.go2_mapping_live.high_fidelity_mapper import HighFidelityMapper
from mapping.pc.go2_mapping_quality import MappingQualityGate, QualityLevel, assess_quality, meets_requirement
from mapping.pc.go2_mapping_quality.reconstruction_config import (
    ConfigError,
    canonical_sha256,
    load_reconstruction_config,
    parse_calibration_manifest,
    parse_session_evidence,
)


HASH = "a" * 64
SECOND_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
FIFTH_HASH = "e" * 64
BOARD_PDF_HASH = "fe05d01ca5fdde303f712dad177f53c3ae75d3493ff7d9a783b010f892a2d743"
BOARD_GEOMETRY_HASH = "311eaf2f593c6cc6bb2d4eec228974723df8bbd29acc211aee787d485edadd1a"
IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _calibration():
    return {
        "schema_version": 1,
        "status": "measured",
        "calibration_id": "go2edu-lab-cal-001",
        "robot": {"model": "go2_edu", "serial": "unit-test"},
        "source_captures": [HASH, SECOND_HASH, THIRD_HASH],
        "camera_intrinsics": {"camera_info_sha256": HASH},
        "transforms": {
            "lidar_to_imu": {"sha256": HASH},
            "lidar_to_camera": {
                "sha256": SECOND_HASH,
                "camera_from_lidar_row_major": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
            },
            "camera_to_base": {"sha256": THIRD_HASH},
        },
        "repeatability": {
            "lidar_imu_translation_mm": 2.0,
            "lidar_imu_rotation_deg": 0.1,
            "lidar_camera_translation_mm": 3.0,
            "lidar_camera_rotation_deg": 0.15,
        },
        "timing": {"lidar_imu_repeatability_ms": 0.5, "lidar_camera_repeatability_ms": 1.0},
        "camera_reprojection": {"median_px": 0.25, "p95_px": 0.75},
    }


def _session():
    return {
        "schema_version": 1,
        "calibration_id": "go2edu-lab-cal-001",
        "raw_capture": {
            "sha256_verified": True,
            "artifact_count": 2,
            "artifact_sha256": [HASH, SECOND_HASH],
            "required_streams": ["lidar", "imu", "lio_pose", "rgb", "depth", "camera_info", "tf", "tf_static"],
        },
        "geometry_validation": {
            "independent_measurement_count": 12,
            "independent_measurements_used_for_calibration": False,
            "p95_dimension_error_mm": 18.0,
            "maximum_dimension_error_mm": 28.0,
            "accepted_loop_closures": 2,
            "loop_point_to_plane_p95_mm": 20.0,
        },
        "bim_registration": {
            "independent_structural_groups": 3,
            "holdout_structural_groups": 1,
            "holdout_p95_residual_mm": 70.0,
            "rigid_se3_only": True,
            "design_bim_hash": THIRD_HASH,
        },
    }


def _derived_product_contract():
    return {
        "measured_fusion": {
            "owner": "LIDAR_MEASURED",
            "cache_key": HASH,
            "invalidates_on": ["raw_artifact", "pose_generation", "fusion_config", "roi", "mask", "algorithm"],
        },
        "calibration": {
            "owner": "CAMERA_LIDAR_CALIBRATION",
            "cache_key": SECOND_HASH,
            "invalidates_on": ["capture", "target", "camera_info", "tf", "solver", "clock_fit"],
        },
        "depth_validation": {
            "owner": "DEPTH_VALIDATION",
            "cache_key": THIRD_HASH,
            "invalidates_on": ["measured_generation", "calibration", "appearance_config"],
        },
        "texture": {
            "owner": "CALIBRATED_TEXTURE",
            "cache_key": FOURTH_HASH,
            "invalidates_on": [
                "measured_generation", "depth_validation", "calibration", "appearance_config",
                "roi", "mask", "algorithm",
            ],
        },
        "augmentation": {
            "owner": "RGBD_AUGMENTED",
            "cache_key": FIFTH_HASH,
            "invalidates_on": [
                "measured_generation", "depth_validation", "calibration", "appearance_config",
                "roi", "mask", "algorithm",
            ],
        },
        "evidence": {
            "owner": "VALIDATING",
            "cache_key": "f" * 64,
            "invalidates_on": ["referenced_stage_output"],
        },
    }


def _session_v2():
    session = _session()
    session["schema_version"] = 2
    session["calibration_id"] = "go2edu-lab-cal-v2-001"
    session["reconstruction_identity"] = {
        "pose_generation": "pg-2",
        "fusion_config_sha256": HASH,
        "appearance_config_sha256": SECOND_HASH,
        "roi_sha256": THIRD_HASH,
        "mask_sha256": FOURTH_HASH,
        "algorithm_version": "reconstruction-v2",
    }
    session["derived_product_contract"] = _derived_product_contract()
    return session


def _camera_info(width, height):
    return {
        "width": width,
        "height": height,
        "distortion_model": "plumb_bob",
        "k": [600.0, 0.0, width / 2, 0.0, 600.0, height / 2, 0.0, 0.0, 1.0],
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [600.0, 0.0, width / 2, 0.0, 0.0, 600.0, height / 2, 0.0, 0.0, 0.0, 1.0, 0.0],
        "rectified": False,
    }


def _clock_fit(sha256):
    return {
        "slope": 1.00005,
        "offset_ns": 1000.0,
        "sample_count": 200,
        "span_s": 30.0,
        "inlier_fraction": 0.95,
        "median_absolute_residual_ms": 1.0,
        "p95_residual_ms": 3.0,
        "maximum_prediction_stddev_ms": 1.0,
        "estimator_version": "affine-ransac-wls-v1",
        "sha256": sha256,
    }


def _lidar_imu_evidence():
    return {
        "source_capture_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "result_artifact_sha256": ["4" * 64, "5" * 64, "6" * 64],
        "transform_sha256": ["7" * 64, "8" * 64, "9" * 64],
        "raw_lidar_topic": "/livox/lidar",
        "raw_imu_topic": "/livox/imu",
        "lidar_frame_id": "livox_frame",
        "imu_frame_id": "imu_link",
        "solver": {"name": "li-init", "version": "2024.1", "implementation_sha256": FIFTH_HASH},
        "uses_vendor_extrinsic": False,
    }


def _calibration_v2():
    payload = {
        "schema_version": 2,
        "status": "measured",
        "calibration_id": "go2edu-lab-cal-v2-001",
        "robot": {"model": "go2_edu", "serial": "robot-unit-test"},
        "capture_sets": {
            "estimation_artifact_sha256": [HASH, SECOND_HASH, THIRD_HASH],
            "holdout_artifact_sha256": FOURTH_HASH,
        },
        "target": {
            "family": "charuco",
            "dictionary": "DICT_4X4_100",
            "rows": 9,
            "columns": 12,
            "marker_length_m": 0.0145,
            "square_length_m": 0.020,
            "pattern_width_m": 0.240,
            "pattern_height_m": 0.180,
            "source_template_pdf_sha256": BOARD_PDF_HASH,
            "source_pdf_nominal_marker_length_m": 0.005,
            "source_pdf_nominal_square_length_m": 0.007,
            "dimension_authority": "physical_print",
            "geometry_sha256": BOARD_GEOMETRY_HASH,
        },
        "solver": {
            "version": "camera-lidar-se3-time-v1",
            "implementation_sha256": HASH,
        },
        "frame_chain": {
            "lidar_frame_id": "livox_frame",
            "color_optical_frame_id": "camera_color_optical_frame",
            "depth_optical_frame_id": "camera_depth_optical_frame",
            "device_serial": "realsense-unit-test",
            "color_camera_info_sha256": HASH,
            "color_camera_info": _camera_info(1280, 720),
            "depth_camera_info_sha256": SECOND_HASH,
            "depth_camera_info": _camera_info(848, 480),
            "t_c_l": {"sha256": "", "row_major": IDENTITY.copy()},
            "t_d_c": {
                "sha256": "",
                "row_major": IDENTITY.copy(),
                "device_serial": "realsense-unit-test",
            },
        },
        "clock_fits": {
            "lidar": _clock_fit(HASH),
            "color": _clock_fit(SECOND_HASH),
            "depth": _clock_fit(THIRD_HASH),
        },
        "lidar_imu_evidence": _lidar_imu_evidence(),
        "repeatability": {
            "lidar_imu_translation_mm": 2.0,
            "lidar_imu_rotation_deg": 0.1,
            "lidar_camera_translation_mm": 5.0,
            "lidar_camera_rotation_deg": 0.2,
        },
        "timing": {
            "lidar_imu_repeatability_ms": 0.5,
            "lidar_camera_repeatability_ms": 2.0,
        },
        "holdout_evidence": {
            "artifact_sha256": FOURTH_HASH,
            "accepted_board_poses": 3,
            "correspondences_per_pose": [500, 500, 9000],
            "total_correspondences": 10000,
            "validated_range_min_m": 0.5,
            "validated_range_max_m": 4.0,
            "camera_reprojection_median_px": 0.5,
            "camera_reprojection_p95_px": 1.0,
            "depth_median_absolute_residual_m": 0.025,
            "depth_p95_absolute_residual_m": 0.06,
            "absolute_mean_signed_bias_m": 0.015,
            "occlusion_conflict_fraction": 0.02,
            "free_space_violation_fraction": 0.01,
            "t_d_c_validated": True,
        },
    }
    payload["frame_chain"]["t_c_l"]["sha256"] = canonical_sha256(
        {"row_major": payload["frame_chain"]["t_c_l"]["row_major"]}
    )
    payload["frame_chain"]["t_d_c"]["sha256"] = canonical_sha256(
        {
            "row_major": payload["frame_chain"]["t_d_c"]["row_major"],
            "device_serial": payload["frame_chain"]["t_d_c"]["device_serial"],
        }
    )
    return payload


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_missing_or_provisional_calibration_never_promotes_preview(tmp_path):
    missing = assess_quality()
    provisional = _calibration()
    provisional["status"] = "provisional"
    path = _write_json(tmp_path / "provisional.json", provisional)
    assessed = assess_quality(path)

    assert missing.level is QualityLevel.PREVIEW_UNCALIBRATED
    assert assessed.level is QualityLevel.PREVIEW_UNCALIBRATED
    assert "CALIBRATION_NOT_MEASURED" in assessed.blocking_reasons
    assert assessed.dense_quality == "DENSE_LIDAR_PREVIEW_UNCALIBRATED"


def test_geometry_only_runtime_accepts_measured_v2_manifest_without_appearance(monkeypatch, tmp_path):
    payload = json.loads(Path("mapping/config/calibration_manifest.json").read_text(encoding="utf-8"))
    path = _write_json(tmp_path / "geometry-only.json", payload)
    monkeypatch.setenv("GO2_MAPPING_GEOMETRY_ONLY_CALIBRATION", "1")
    assessed = assess_quality(path)
    assert assessed.level is QualityLevel.CALIBRATED_DENSE
    assert assessed.measurement_authority is True
    assert assessed.runtime_authority == "MEASURED"
    assert assessed.appearance_authorized is False


def test_verified_raw_capture_is_reported_without_a_calibration_manifest(tmp_path):
    evidence = _write_json(tmp_path / "session.json", _session())

    assessed = assess_quality(session_path=evidence)

    assert assessed.level is QualityLevel.PREVIEW_UNCALIBRATED
    assert assessed.raw_capture_verified is True
    assert meets_requirement(assessed, QualityLevel.RAW_CAPTURE_VALID)
    assert "CALIBRATION_MISSING" in assessed.blocking_reasons


def test_lidar_camera_projection_requires_a_measured_rigid_matrix(tmp_path):
    calibration = _calibration()
    calibration["transforms"]["lidar_to_camera"]["camera_from_lidar_row_major"][-1] = 0.0
    path = _write_json(tmp_path / "bad-transform.json", calibration)

    assessed = assess_quality(path)

    assert assessed.level is QualityLevel.PREVIEW_UNCALIBRATED
    assert "TRANSFORM_MATRIX_NOT_HOMOGENEOUS:lidar_to_camera" in assessed.blocking_reasons


def test_calibrated_capture_requires_held_out_geometry_before_cm_promotion(tmp_path):
    calibration = _write_json(tmp_path / "calibration.json", _calibration())
    session = _session()
    session["geometry_validation"]["p95_dimension_error_mm"] = 25.0
    evidence = _write_json(tmp_path / "session.json", session)

    calibrated_only = assess_quality(calibration)
    raw_valid = assess_quality(calibration, evidence)

    assert calibrated_only.level is QualityLevel.CALIBRATED_DENSE
    assert raw_valid.level is QualityLevel.CALIBRATED_DENSE
    assert raw_valid.snapshot()["raw_capture_verified"] is True
    assert meets_requirement(raw_valid, QualityLevel.RAW_CAPTURE_VALID)
    assert not meets_requirement(calibrated_only, QualityLevel.RAW_CAPTURE_VALID)
    assert any(reason.startswith("METRIC_EXCEEDS:p95_dimension_error_mm") for reason in raw_valid.blocking_reasons)


def test_full_evidence_reaches_bim_validated_only_with_rigid_holdout_registration(tmp_path):
    calibration = _write_json(tmp_path / "calibration.json", _calibration())
    session = _write_json(tmp_path / "session.json", _session())
    gate = MappingQualityGate(calibration, session)

    report = gate.snapshot()

    assert report["level"] == QualityLevel.BIM_REGISTERED_VALIDATED.value
    assert report["eligible_for"]["centimetre_geometry"] is True
    assert report["eligible_for"]["bim_coordinate_delivery"] is True
    assert report["dense_quality"] == "DENSE_LIDAR_CALIBRATED"
    assert report["required_artifact_sha256"] == [HASH, SECOND_HASH]
    assert report["camera_from_lidar_row_major"] == _calibration()["transforms"]["lidar_to_camera"]["camera_from_lidar_row_major"]


def test_schema_v1_remains_readable_but_never_authorizes_rgbd_or_texture(tmp_path):
    calibration = _write_json(tmp_path / "calibration-v1.json", _calibration())
    session = _write_json(tmp_path / "session-v1.json", _session())

    report = assess_quality(calibration, session).snapshot()

    assert report["level"] == QualityLevel.BIM_REGISTERED_VALIDATED.value
    assert report["eligible_for"]["rgbd_geometry_augmentation"] is False
    assert report["eligible_for"]["texture"] is False
    assert report["eligible_for"]["rgbd_or_neural_geometry_fusion"] is False


def test_schema_v2_typed_manifest_authorizes_rgbd_and_texture_at_exact_boundaries(tmp_path):
    payload = _calibration_v2()
    typed = parse_calibration_manifest(payload, load_reconstruction_config().appearance)
    calibration = _write_json(tmp_path / "calibration-v2.json", payload)
    session_payload = _session_v2()
    session = _write_json(tmp_path / "session-v2.json", session_payload)

    report = assess_quality(calibration, session).snapshot()

    assert typed.frame_chain.color_optical_frame_id == "camera_color_optical_frame"
    assert typed.frame_chain.depth_optical_frame_id == "camera_depth_optical_frame"
    assert typed.holdout_evidence.t_d_c_validated is True
    assert report["level"] == QualityLevel.BIM_REGISTERED_VALIDATED.value
    assert report["eligible_for"]["rgbd_geometry_augmentation"] is True
    assert report["eligible_for"]["texture"] is True
    assert report["camera_from_lidar_row_major"] == IDENTITY


def test_schema_v2_transform_hash_must_bind_the_exact_rigid_matrix():
    payload = _calibration_v2()
    payload["frame_chain"]["t_c_l"]["sha256"] = canonical_sha256(
        {"row_major": payload["frame_chain"]["t_c_l"]["row_major"]}
    )
    payload["frame_chain"]["t_d_c"]["sha256"] = canonical_sha256(
        {
            "row_major": payload["frame_chain"]["t_d_c"]["row_major"],
            "device_serial": payload["frame_chain"]["t_d_c"]["device_serial"],
        }
    )
    parse_calibration_manifest(payload, load_reconstruction_config().appearance)

    payload["frame_chain"]["t_c_l"]["row_major"][3] = 0.01
    with pytest.raises(ConfigError, match="hash the canonical transform"):
        parse_calibration_manifest(payload, load_reconstruction_config().appearance)


def test_schema_v2_session_evidence_is_typed_and_rejects_legacy_relabel():
    typed = parse_session_evidence(_session_v2())
    legacy_relabel = _session()
    legacy_relabel["schema_version"] = 2

    assert typed.reconstruction_identity.pose_generation == "pg-2"
    assert typed.derived_product_contract["measured_fusion"].owner == "LIDAR_MEASURED"
    with pytest.raises(ConfigError, match="reconstruction_identity"):
        parse_session_evidence(legacy_relabel)


@pytest.mark.parametrize("session_mutation", [
    "missing",
    "legacy",
    "malformed_v2",
    "mismatched_calibration",
])
def test_appearance_authorization_requires_valid_matching_schema_v2_session(tmp_path, session_mutation):
    calibration = _write_json(tmp_path / "calibration.json", _calibration_v2())
    session_payload = _session_v2()
    if session_mutation == "legacy":
        session_payload = _session()
    elif session_mutation == "malformed_v2":
        session_payload["derived_product_contract"]["texture"]["cache_key"] = "bad"
    elif session_mutation == "mismatched_calibration":
        session_payload["calibration_id"] = "different-calibration"
    session = None if session_mutation == "missing" else _write_json(tmp_path / "session.json", session_payload)

    report = assess_quality(calibration, session).snapshot()

    assert report["level"] == QualityLevel.CALIBRATED_DENSE.value or report["level"] == QualityLevel.BIM_REGISTERED_VALIDATED.value
    assert report["eligible_for"]["texture"] is False
    assert report["eligible_for"]["rgbd_geometry_augmentation"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value["reconstruction_identity"].update({"unknown": True}),
        lambda value: value["derived_product_contract"]["texture"].update({"unknown": True}),
    ],
)
def test_schema_v2_session_evidence_rejects_unknown_fields(mutation):
    session = _session_v2()
    mutation(session)

    with pytest.raises(ConfigError, match="unknown"):
        parse_session_evidence(session)


def test_calibration_and_session_identity_hashes_are_canonical_json(tmp_path):
    calibration_payload = _calibration_v2()
    session_payload = _session_v2()
    compact_calibration = _write_json(tmp_path / "calibration-compact.json", calibration_payload)
    compact_session = _write_json(tmp_path / "session-compact.json", session_payload)
    pretty_calibration = tmp_path / "calibration-pretty.json"
    pretty_session = tmp_path / "session-pretty.json"
    pretty_calibration.write_text(json.dumps(calibration_payload, indent=4), encoding="utf-8")
    pretty_session.write_text(json.dumps(session_payload, indent=4), encoding="utf-8")

    compact = assess_quality(compact_calibration, compact_session).snapshot()
    pretty = assess_quality(pretty_calibration, pretty_session).snapshot()

    assert compact["calibration_sha256"] == pretty["calibration_sha256"]
    assert compact["session_evidence_sha256"] == pretty["session_evidence_sha256"]


def _replace_nested(payload, path, value):
    parent = payload
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("schema_version",), 3, "SCHEMA_VERSION"),
        (("capture_sets", "estimation_artifact_sha256"), [HASH, SECOND_HASH], "estimation_artifact_sha256"),
        (("frame_chain", "depth_optical_frame_id"), "camera_color_optical_frame", "optical frame IDs"),
        (("frame_chain", "color_camera_info_sha256"), "not-a-hash", "color_camera_info_sha256"),
        (("frame_chain", "t_c_l", "row_major"), IDENTITY[:-1], "t_c_l.row_major"),
        (("frame_chain", "t_d_c", "device_serial"), "other-device", "device_serial"),
        (("clock_fits", "color", "p95_residual_ms"), 3.000001, "clock_fits.color"),
        (("clock_fits", "depth", "sample_count"), True, "sample_count"),
        (("holdout_evidence", "total_correspondences"), 9999, "holdout_evidence"),
        (("holdout_evidence", "depth_p95_absolute_residual_m"), 0.060001, "holdout_evidence"),
        (("holdout_evidence", "t_d_c_validated"), False, "t_d_c_validated"),
        (("target", "marker_length_m"), True, "marker_length_m"),
    ],
)
def test_schema_v2_invalid_frame_chain_clock_target_or_holdout_fails_closed(
    tmp_path, path, value, reason
):
    payload = _calibration_v2()
    _replace_nested(payload, path, value)
    calibration = _write_json(tmp_path / "invalid-v2.json", payload)

    report = assess_quality(calibration).snapshot()

    assert report["level"] == QualityLevel.PREVIEW_UNCALIBRATED.value
    assert report["eligible_for"]["rgbd_geometry_augmentation"] is False
    assert report["eligible_for"]["texture"] is False
    assert any(reason in blocker for blocker in report["blocking_reasons"])


def test_schema_v2_board_physical_fields_are_required(tmp_path):
    payload = _calibration_v2()
    del payload["target"]["pattern_height_m"]
    calibration = _write_json(tmp_path / "missing-board-dimension.json", payload)

    report = assess_quality(calibration).snapshot()

    assert report["level"] == QualityLevel.PREVIEW_UNCALIBRATED.value
    assert any("pattern_height_m" in reason for reason in report["blocking_reasons"])


def test_board_rejects_marker_length_mistaken_for_square_length():
    payload = _calibration_v2()
    payload["target"]["marker_length_m"] = 0.020

    with pytest.raises(ConfigError, match="marker_length_m"):
        parse_calibration_manifest(payload, load_reconstruction_config().appearance)


def test_old_28_20_mm_board_manifest_is_incompatible():
    payload = _calibration_v2()
    payload["target"].update(
        marker_length_m=0.020,
        square_length_m=0.028,
        pattern_width_m=0.336,
        pattern_height_m=0.252,
    )

    with pytest.raises(ConfigError, match="measured 14.5 mm"):
        parse_calibration_manifest(payload, load_reconstruction_config().appearance)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("family", "aruco", "target.family"),
        ("dictionary", "DICT_4X4_250", "target.dictionary"),
        ("rows", 1, "target rows/columns"),
        ("columns", 1, "target rows/columns"),
    ],
)
def test_schema_v2_manifest_rejects_non_locked_target_contract(field, value, reason):
    payload = _calibration_v2()
    payload["target"][field] = value

    with pytest.raises(ConfigError, match=reason):
        parse_calibration_manifest(payload, load_reconstruction_config().appearance)


@pytest.mark.parametrize("signed_hash", ["+" + "a" * 63, "-" + "a" * 63])
def test_signed_64_character_hashes_fail_closed_in_every_evidence_schema(tmp_path, signed_hash):
    legacy = _calibration()
    legacy["source_captures"][0] = signed_hash
    legacy_path = _write_json(tmp_path / "legacy.json", legacy)

    calibration_v2 = _calibration_v2()
    calibration_v2["capture_sets"]["estimation_artifact_sha256"][0] = signed_hash
    calibration_v2_path = _write_json(tmp_path / "calibration-v2.json", calibration_v2)

    session_v2 = _session_v2()
    session_v2["reconstruction_identity"]["fusion_config_sha256"] = signed_hash

    assert assess_quality(legacy_path).level is QualityLevel.PREVIEW_UNCALIBRATED
    assert assess_quality(calibration_v2_path).level is QualityLevel.PREVIEW_UNCALIBRATED
    with pytest.raises(ConfigError, match="fusion_config_sha256"):
        parse_session_evidence(session_v2)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("frame_chain", "color_camera_info", "k", 0), 0.0, "color_camera_info.k"),
        (("frame_chain", "color_camera_info", "k", 4), -1.0, "color_camera_info.k"),
        (("frame_chain", "color_camera_info", "k", 8), 0.0, "color_camera_info.k"),
        (("frame_chain", "depth_camera_info", "p", 0), 0.0, "depth_camera_info.p"),
        (("frame_chain", "depth_camera_info", "p", 5), -1.0, "depth_camera_info.p"),
        (("frame_chain", "depth_camera_info", "p", 10), 0.0, "depth_camera_info.p"),
        (("frame_chain", "color_camera_info", "r"), [1.0] * 9, "color_camera_info.r"),
    ],
)
def test_camera_info_rejects_degenerate_intrinsics_projection_and_rectification(path, value, reason):
    payload = _calibration_v2()
    _replace_nested(payload, path, value)

    with pytest.raises(ConfigError, match=reason):
        parse_calibration_manifest(payload, load_reconstruction_config().appearance)


def test_dense_mapper_uses_gate_but_remains_preview_by_default():
    mapper = HighFidelityMapper(display_point_budget=100)
    gated = HighFidelityMapper(display_point_budget=100, quality_snapshot=lambda: {"dense_quality": "DENSE_LIDAR_CALIBRATED"})

    assert mapper.snapshot()["quality"] == "AWAITING_VERIFIED_LIDAR_ARTIFACT"
    assert gated.snapshot()["quality"] == "AWAITING_VERIFIED_LIDAR_ARTIFACT"


def test_dense_mapper_does_not_apply_cm_quality_until_all_evidence_artifacts_are_materialized():
    mapper = HighFidelityMapper(
        display_point_budget=100,
        quality_snapshot=lambda: {
            "level": "GEOMETRY_VALIDATED_CM",
            "dense_quality": "DENSE_LIDAR_CALIBRATED",
            "required_artifact_sha256": [HASH, SECOND_HASH],
            "blocking_reasons": [],
            "eligible_for": {"centimetre_geometry": True, "bim_coordinate_delivery": False},
        },
    )
    mapper._frames = 1
    mapper._artifact_digests.add(HASH)

    state = mapper.snapshot()

    assert state["quality"] == "DENSE_LIDAR_CALIBRATED"
    assert state["quality_evidence"]["level"] == "CALIBRATED_DENSE"
    assert state["quality_evidence"]["assessment_level"] == "GEOMETRY_VALIDATED_CM"
    assert state["quality_evidence"]["map_artifacts_materialized"] is False
