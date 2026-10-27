from __future__ import annotations

import copy
import hashlib
import re

import pytest
from pathlib import Path

from mapping.pc.go2_mapping_appearance.field_acceptance import (
    evaluate_textured_field_acceptance,
)


SHA = "a" * 64


def _valid_report() -> dict:
    provenance = b'{"schema_version":1}'
    return {
        "schema_version": 1,
        "evidence_kind": "PHYSICAL_GO2_CAPTURE",
        "transport": "TAILSCALE_DIRECT",
        "synthetic": False,
        "surface": {
            "coverage_ppm": 800000,
            "independent_color_views": 2,
            "ingest_errors": 0,
            "pose_errors": 0,
            "rgbd_errors": 0,
            "surface_errors": 0,
        },
        "physical_depth": {
            "correspondence_count": 100,
            "median_m": 0.025,
            "p95_m": 0.060,
            "signed_bias_m": -0.015,
            "occlusion_fraction": 0.02,
            "free_space_fraction": 0.01,
        },
        "depth_anything": {
            "surviving_anchor_count": 256,
            "occupied_cells": 24,
            "total_cells": 48,
            "quadrants": [True, True, True, True],
            "held_out_median_m": 0.025,
            "held_out_p95_m": 0.060,
            "held_out_signed_bias_m": 0.015,
            "scale": 1.0,
            "relative_scale_stddev": 0.02,
            "maximum_uncertainty_m": 0.030,
        },
        "identity": {
            "restart_performed": True,
            "pre_restart": {
                "geometry_revision_id": SHA,
                "tile_appearance_revision_id": "b" * 64,
                "building_appearance_revision_id": "c" * 64,
                "placement_revision_id": "d" * 64,
                "source_generation_id": "e" * 64,
            },
            "post_restart": {
                "geometry_revision_id": SHA,
                "tile_appearance_revision_id": "b" * 64,
                "building_appearance_revision_id": "c" * 64,
                "placement_revision_id": "d" * 64,
                "source_generation_id": "e" * 64,
            },
        },
        "exports": {
            "local": {
                "pygltflib_version": "1.16.5",
                "trimesh_version": "4.7.1",
                "pygltflib_loaded": True,
                "trimesh_loaded": True,
                "threejs_loaded": True,
                "metres_verified": True,
                "coordinate_transform_verified": True,
                "provenance_bytes": provenance,
                "provenance_sha256": hashlib.sha256(provenance).hexdigest(),
            },
            "building": {
                "pygltflib_version": "1.16.5",
                "trimesh_version": "4.7.1",
                "pygltflib_loaded": True,
                "trimesh_loaded": True,
                "threejs_loaded": True,
                "metres_verified": True,
                "coordinate_transform_verified": True,
                "provenance_bytes": provenance,
                "provenance_sha256": hashlib.sha256(provenance).hexdigest(),
            },
        },
    }


def _set(report: dict, path: str, value) -> None:
    target = report
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_exact_physical_report_is_accepted_at_inclusive_boundaries():
    result = evaluate_textured_field_acceptance(_valid_report())
    assert result.accepted is True
    assert result.state == "LIVE_TEXTURED_MAPPING_ACCEPTED"
    assert result.blockers == ()


@pytest.mark.parametrize(
    ("path", "value", "blocker"),
    [
        ("surface.coverage_ppm", 799999, "SURFACE_COVERAGE_BELOW_THRESHOLD"),
        ("surface.independent_color_views", 1, "INDEPENDENT_COLOR_VIEWS_BELOW_THRESHOLD"),
        ("physical_depth.correspondence_count", 99, "PHYSICAL_DEPTH_CORRESPONDENCE_COUNT"),
        ("physical_depth.median_m", 0.025001, "PHYSICAL_DEPTH_MEDIAN"),
        ("physical_depth.p95_m", 0.060001, "PHYSICAL_DEPTH_P95"),
        ("physical_depth.signed_bias_m", -0.015001, "PHYSICAL_DEPTH_BIAS"),
        ("physical_depth.occlusion_fraction", 0.020001, "PHYSICAL_DEPTH_OCCLUSION"),
        ("physical_depth.free_space_fraction", 0.010001, "PHYSICAL_DEPTH_FREE_SPACE"),
        ("depth_anything.surviving_anchor_count", 255, "DEPTH_ANYTHING_ANCHORS"),
        ("depth_anything.occupied_cells", 23, "DEPTH_ANYTHING_GRID_COVERAGE"),
        ("depth_anything.held_out_median_m", 0.025001, "DEPTH_ANYTHING_MEDIAN"),
        ("depth_anything.held_out_p95_m", 0.060001, "DEPTH_ANYTHING_P95"),
        ("depth_anything.held_out_signed_bias_m", 0.015001, "DEPTH_ANYTHING_BIAS"),
        ("depth_anything.scale", 0.0, "DEPTH_ANYTHING_SCALE"),
        ("depth_anything.relative_scale_stddev", 0.020001, "DEPTH_ANYTHING_SCALE_STABILITY"),
        ("depth_anything.maximum_uncertainty_m", 0.030001, "DEPTH_ANYTHING_UNCERTAINTY"),
    ],
)
def test_numeric_boundaries_fail_closed(path: str, value, blocker: str):
    report = _valid_report()
    _set(report, path, value)
    result = evaluate_textured_field_acceptance(report)
    assert result.accepted is False
    assert blocker in result.blockers


@pytest.mark.parametrize("error_name", ["ingest_errors", "pose_errors", "rgbd_errors", "surface_errors"])
def test_any_pipeline_error_fails_closed(error_name: str):
    report = _valid_report()
    report["surface"][error_name] = 1
    assert "PIPELINE_ERRORS_PRESENT" in evaluate_textured_field_acceptance(report).blockers


def test_all_quadrants_are_required():
    report = _valid_report()
    report["depth_anything"]["quadrants"][2] = False
    assert "DEPTH_ANYTHING_QUADRANTS" in evaluate_textured_field_acceptance(report).blockers


def test_restart_identity_must_be_exact_and_lowercase_sha256():
    changed = _valid_report()
    changed["identity"]["post_restart"]["placement_revision_id"] = "f" * 64
    assert "RESTART_IDENTITY_CHANGED" in evaluate_textured_field_acceptance(changed).blockers

    malformed = _valid_report()
    malformed["identity"]["pre_restart"]["geometry_revision_id"] = "A" * 64
    assert "REVISION_IDENTITY_INVALID" in evaluate_textured_field_acceptance(malformed).blockers


def test_restart_must_have_actually_been_performed():
    report = _valid_report()
    report["identity"]["restart_performed"] = False
    assert "OWNED_DASHBOARD_RESTART_REQUIRED" in evaluate_textured_field_acceptance(report).blockers


@pytest.mark.parametrize(
    ("path", "value", "blocker"),
    [
        ("transport", "SSH", "TRANSPORT_NOT_TAILSCALE_DIRECT"),
        ("synthetic", True, "SYNTHETIC_EVIDENCE_REJECTED"),
        ("evidence_kind", "SYNTHETIC_TEST_ONLY", "PHYSICAL_CAPTURE_REQUIRED"),
        ("exports.local.pygltflib_version", "1.16.4", "LOCAL_GLTF_PARSER_VERSION"),
        ("exports.building.trimesh_loaded", False, "BUILDING_TRIMESH_LOAD_FAILED"),
        ("exports.local.threejs_loaded", False, "LOCAL_THREEJS_LOAD_FAILED"),
        ("exports.building.metres_verified", False, "BUILDING_METRE_UNITS_INVALID"),
        ("exports.local.coordinate_transform_verified", False, "LOCAL_COORDINATE_TRANSFORM_INVALID"),
        ("exports.building.provenance_sha256", "f" * 64, "BUILDING_PROVENANCE_DIGEST_MISMATCH"),
    ],
)
def test_transport_synthetic_export_and_provenance_failures_are_rejected(path, value, blocker):
    report = _valid_report()
    _set(report, path, value)
    assert blocker in evaluate_textured_field_acceptance(report).blockers


def test_evaluator_does_not_mutate_input():
    report = _valid_report()
    original = copy.deepcopy(report)
    evaluate_textured_field_acceptance(report)
    assert report == original


def test_operator_verifier_is_exact_fail_closed_and_writes_canonical_acceptance():
    root = Path(__file__).resolve().parents[2]
    tool = (root / "tools" / "verify_go2_textured_mapping_acceptance.ps1").read_text(encoding="utf-8")
    operator = (
        root / "mapping" / "pc" / "go2_mapping_appearance" / "field_acceptance_operator.py"
    ).read_text(encoding="utf-8")
    implementation = tool + operator
    package = (root / "mapping" / "pc" / "go2_mapping_appearance" / "__init__.py").read_text(encoding="utf-8")

    assert "ContextPath" in tool
    assert "OutputDirectory" in tool
    assert "evaluate_textured_field_acceptance" in implementation
    assert "acceptance.json" in implementation
    assert "LIVE_TEXTURED_MAPPING_ACCEPTED" in tool
    assert "synthetic" in tool.lower()
    assert "TAILSCALE_DIRECT" in tool
    assert re.search(r"\bssh(?:\.exe)?\b", tool, re.IGNORECASE) is None
    assert "/api/reconstruction/field-acceptance-evidence" in implementation
    assert "/api/reconstruction/export-textured-tile" in implementation
    assert "/api/reconstruction/export-textured-building" in implementation
    assert "pygltflib" in implementation
    assert "trimesh" in implementation
    assert "Restart-OwnedDashboard" in tool
    assert "field_acceptance_operator" in tool
    assert "THREEJS_LOADED" in tool
    assert '"FieldAcceptanceResult"' in package
    assert '"evaluate_textured_field_acceptance"' in package


def test_tailnet_capture_context_is_directly_consumable_by_field_verifier():
    root = Path(__file__).resolve().parents[2]
    session = (root / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")

    assert (
        "evidence_kind = if ($UncalibratedCapture) { 'PHYSICAL_GO2_CALIBRATION_CAPTURE' } "
        "else { 'PHYSICAL_GO2_CAPTURE' }"
    ) in session
    assert "dashboard_url =" in session
    assert "owned_dashboard_pid =" in session
    assert "calibration_manifest_path =" in session
    assert "session_evidence_path =" in session
