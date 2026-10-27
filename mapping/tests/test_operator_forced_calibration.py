"""Contracts for an explicitly operator-approved Go2 runtime calibration override."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mapping.pc.go2_mapping_quality import MappingQualityGate, QualityLevel, assess_quality


CALIBRATION_ID = "PROVISIONAL_AUTOCAL_PENDING_v1"
AXIS_ONLY_CAMERA_FROM_LIDAR = [
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]
NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR = [
    0.008844417981016149,
    -0.9565635934374991,
    0.2913895467934907,
    -1.9999999999964886,
    0.9185651134466347,
    0.12292952584157568,
    0.3756680237059531,
    0.9718851062983953,
    -0.3951707335182455,
    0.2643377070837686,
    0.879753185833588,
    -0.06849787999466213,
    0.0,
    0.0,
    0.0,
    1.0,
]
SOURCE_ARTIFACTS = [
    "f847838bdb007c9b9df61a2fa129fad793d58c9608979ca083f1370c516df041",
    "68fd633f8b4593bd8011c9758ed27315cddf7f0f1f0b357506070d587185b16c",
]
PROVISIONAL_FIT_ARTIFACTS = SOURCE_ARTIFACTS + [
    "53d20a23464aa3470d34317303958798f645a871f31d7e4a7e53c5edf6b213c0",
]
HOLDOUT_ARTIFACT_SHA256 = "8ca166e97ea0a2a6ed98f2d1934e7f4f5be9824c4f3d8a4824e7a405907c0b5f"
FIT_CAPTURE_IDS = ["estimation_1", "estimation_2", "estimation_3"]
HOLDOUT_CAPTURE_IDS = ["holdout"]
ROOT = Path(__file__).resolve().parents[2]
CREATE_OVERRIDE_TOOL = ROOT / "tools" / "create_go2_operator_forced_calibration_override.py"
PC_STACK = ROOT / "tools" / "start_go2_pc_stack.ps1"
TAILNET_SESSION = ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1"
COMPATIBILITY_SESSION = ROOT / "tools" / "start_go2_mapping_session.ps1"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _valid_override_paths(tmp_path):
    calibration = _write_json(
        tmp_path / "capture-manifest.json",
        {
            "schema_version": 1,
            "status": "provisional",
            "purpose": "calibration_capture",
            "calibration_id": CALIBRATION_ID,
            "geometry_authority": False,
            "appearance_authority": False,
            "acceptance_authority": False,
        },
    )
    diagnostic = _write_json(
        tmp_path / "joint-diagnostic.json",
        {
            "schema_version": 1,
            "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
            "calibration_authority": False,
            "manifest_written": False,
            "captures": [{"artifact_sha256": item} for item in SOURCE_ARTIFACTS],
            "joint": {
                "status": "JOINT_DIAGNOSTIC_UNOBSERVABLE",
                "calibration_authority": False,
                "manifest_written": False,
                "camera_from_lidar": None,
                "source_artifact_sha256": SOURCE_ARTIFACTS,
            },
        },
    )
    diagnostic_sha256 = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    override = _write_json(
        tmp_path / "formal-operator-override.json",
        {
            "schema_version": 1,
            "kind": "GO2_OPERATOR_FORCED_CALIBRATION_OVERRIDE",
            "status": "FORMAL_OPERATOR_FORCED_RUNTIME",
            "calibration_id": CALIBRATION_ID,
            "formal_runtime_override": True,
            "operator_acknowledged": True,
            "measurement_authority": False,
            "runtime_appearance_authority": True,
            "camera_from_lidar": {
                "axis_contract_id": "DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_v1",
                "row_major": AXIS_ONLY_CAMERA_FROM_LIDAR,
                "translation_m": [0.0, 0.0, 0.0],
                "translation_measured": False,
            },
            "source_joint_diagnostic": {
                "relative_path": "joint-diagnostic.json",
                "sha256": diagnostic_sha256,
                "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
                "status": "JOINT_DIAGNOSTIC_UNOBSERVABLE",
                "calibration_authority": False,
                "manifest_written": False,
                "source_artifact_sha256": SOURCE_ARTIFACTS,
            },
        },
    )
    return calibration, diagnostic, override


def _valid_provisional_candidate_paths(tmp_path):
    calibration = _write_json(
        tmp_path / "capture-manifest.json",
        {
            "schema_version": 1,
            "status": "provisional",
            "purpose": "calibration_capture",
            "calibration_id": CALIBRATION_ID,
            "geometry_authority": False,
            "appearance_authority": False,
            "acceptance_authority": False,
        },
    )
    captures = [
        {
            "capture_id": capture_id,
            "artifact_sha256": artifact_sha256,
        }
        for capture_id, artifact_sha256 in zip(
            FIT_CAPTURE_IDS + HOLDOUT_CAPTURE_IDS,
            PROVISIONAL_FIT_ARTIFACTS + [HOLDOUT_ARTIFACT_SHA256],
            strict=True,
        )
    ]
    diagnostic = _write_json(
        tmp_path / "joint-diagnostic-provisional.json",
        {
            "schema_version": 1,
            "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
            "calibration_authority": False,
            "manifest_written": False,
            "operator_relaxed": True,
            "provisional": True,
            "captures": captures,
            "fit_capture_ids": FIT_CAPTURE_IDS,
            "holdout_capture_ids": HOLDOUT_CAPTURE_IDS,
            "holdout_diagnostics": [
                {
                    "capture_id": "holdout",
                    "status": "OPERATOR_RELAXED_HOLDOUT_DIAGNOSTIC",
                    "calibration_authority": False,
                    "matched_plane_pairs": 118,
                    "normalized_residual_p95": 207.6318630779048,
                }
            ],
            "joint": {
                "status": "JOINT_DIAGNOSTIC_PROVISIONAL",
                "calibration_authority": False,
                "manifest_written": False,
                "camera_from_lidar": NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR,
                "delta_camera_lidar_ms": 95.79664090417468,
                "normalized_condition_number": 530.3585011490599,
                "residual_p95": 281.0074050936914,
                "source_artifact_sha256": PROVISIONAL_FIT_ARTIFACTS,
                "blockers": [
                    "OPERATOR_RELAXED_BYPASS:estimation_1:ROTATION_SPAN_BELOW_30_DEG",
                    "OPERATOR_RELAXED_BYPASS:estimation_2:LIO_EPOCH_RESET_DETECTED",
                ],
            },
        },
    )
    diagnostic_sha256 = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    override = _write_json(
        tmp_path / "formal-operator-candidate-override.json",
        {
            "schema_version": 1,
            "kind": "GO2_OPERATOR_FORCED_CALIBRATION_OVERRIDE",
            "status": "FORMAL_OPERATOR_FORCED_RUNTIME",
            "calibration_id": CALIBRATION_ID,
            "formal_runtime_override": True,
            "operator_acknowledged": True,
            "measurement_authority": False,
            "runtime_appearance_authority": True,
            "runtime_override_kind": "NUMERICAL_CANDIDATE",
            "candidate_usable_by_operator_decision": True,
            "quality_limits_bypassed": [
                "CLOCK_FIT_FAILED",
                "LOW_MOTION_EXCITATION",
                "BOUND_HIT_TRANSLATION_X",
            ],
            "camera_from_lidar": {
                "source": "SOURCE_JOINT_DIAGNOSTIC_CAMERA_FROM_LIDAR",
                "row_major": NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR,
                "translation_m": [-1.9999999999964886, 0.9718851062983953, -0.06849787999466213],
                "translation_measured": False,
                "delta_camera_lidar_ms": 95.79664090417468,
            },
            "source_joint_diagnostic": {
                "relative_path": diagnostic.name,
                "sha256": diagnostic_sha256,
                "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
                "status": "JOINT_DIAGNOSTIC_PROVISIONAL",
                "calibration_authority": False,
                "manifest_written": False,
                "source_artifact_sha256": PROVISIONAL_FIT_ARTIFACTS,
                "fit_capture_ids": FIT_CAPTURE_IDS,
                "holdout_capture_ids": HOLDOUT_CAPTURE_IDS,
            },
        },
    )
    return calibration, diagnostic, override


def test_explicit_operator_override_enables_formal_runtime_projection_without_claiming_measurement(tmp_path):
    calibration, _, override = _valid_override_paths(tmp_path)

    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=override,
        allow_operator_forced_override=True,
    ).snapshot()

    assert report["level"] == QualityLevel.CALIBRATED_DENSE.value
    assert report["calibration_id"] == CALIBRATION_ID
    assert report["camera_from_lidar_row_major"] == AXIS_ONLY_CAMERA_FROM_LIDAR
    assert report["formal_runtime_override"] is True
    assert report["measurement_authority"] is False
    assert report["runtime_authority"] == "OPERATOR_FORCED_FORMAL"
    assert report["eligible_for"]["dense_lidar_calibrated"] is True
    assert report["eligible_for"]["rgbd_or_neural_geometry_fusion"] is True
    assert report["eligible_for"]["bim_coordinate_delivery"] is False


def test_operator_override_remains_preview_until_explicitly_enabled(tmp_path):
    calibration, _, override = _valid_override_paths(tmp_path)

    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=override,
    ).snapshot()

    assert report["level"] == QualityLevel.PREVIEW_UNCALIBRATED.value
    assert report["formal_runtime_override"] is False
    assert "OPERATOR_FORCED_OVERRIDE_DISABLED" in report["blocking_reasons"]


def test_operator_override_rejects_a_nonzero_or_modified_axis_transform(tmp_path):
    calibration, _, override = _valid_override_paths(tmp_path)
    payload = json.loads(override.read_text(encoding="utf-8"))
    payload["camera_from_lidar"]["row_major"][3] = 0.01
    _write_json(override, payload)

    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=override,
        allow_operator_forced_override=True,
    ).snapshot()

    assert report["level"] == QualityLevel.PREVIEW_UNCALIBRATED.value
    assert any("OVERRIDE_TRANSFORM_NOT_AXIS_ONLY_ZERO_TRANSLATION" in item for item in report["blocking_reasons"])


def test_quality_gate_revokes_operator_override_when_its_source_diagnostic_changes(tmp_path):
    calibration, diagnostic, override = _valid_override_paths(tmp_path)
    gate = MappingQualityGate(
        calibration_path=calibration,
        operator_override_path=override,
        allow_operator_forced_override=True,
    )

    assert gate.snapshot()["level"] == QualityLevel.CALIBRATED_DENSE.value
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    payload["joint"]["status"] = "JOINT_DIAGNOSTIC_REPLACED"
    _write_json(diagnostic, payload)

    refreshed = gate.snapshot()

    assert refreshed["level"] == QualityLevel.PREVIEW_UNCALIBRATED.value
    assert any("OPERATOR_FORCED_OVERRIDE_INVALID" in reason for reason in refreshed["blocking_reasons"])


def test_override_tool_generates_an_enabled_runtime_contract_from_joint_diagnostic(tmp_path):
    calibration, diagnostic, _ = _valid_override_paths(tmp_path)
    output = tmp_path / "formal-operator-override-generated.json"

    result = subprocess.run(
        [
            sys.executable,
            str(CREATE_OVERRIDE_TOOL),
            "--joint-diagnostic",
            str(diagnostic),
            "--output",
            str(output),
            "--calibration-id",
            CALIBRATION_ID,
            "--operator-acknowledged",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["operator_acknowledged"] is True
    assert generated["measurement_authority"] is False
    assert generated["source_joint_diagnostic"]["relative_path"] == diagnostic.name
    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=output,
        allow_operator_forced_override=True,
    ).snapshot()
    assert report["runtime_authority"] == "OPERATOR_FORCED_FORMAL"


def test_numerical_candidate_override_enables_runtime_projection_without_claiming_measurement(tmp_path):
    calibration, _, override = _valid_provisional_candidate_paths(tmp_path)

    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=override,
        allow_operator_forced_override=True,
    ).snapshot()

    assert report["level"] == QualityLevel.CALIBRATED_DENSE.value
    assert report["camera_from_lidar_row_major"] == NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR
    assert report["formal_runtime_override"] is True
    assert report["measurement_authority"] is False
    assert report["runtime_authority"] == "OPERATOR_FORCED_FORMAL"


def test_override_tool_generates_a_numerical_candidate_contract_from_provisional_joint_diagnostic(tmp_path):
    calibration, diagnostic, _ = _valid_provisional_candidate_paths(tmp_path)
    output = tmp_path / "formal-operator-candidate-override-generated.json"

    result = subprocess.run(
        [
            sys.executable,
            str(CREATE_OVERRIDE_TOOL),
            "--joint-diagnostic",
            str(diagnostic),
            "--output",
            str(output),
            "--calibration-id",
            CALIBRATION_ID,
            "--operator-acknowledged",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["runtime_override_kind"] == "NUMERICAL_CANDIDATE"
    assert generated["candidate_usable_by_operator_decision"] is True
    assert generated["camera_from_lidar"]["row_major"] == NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR
    report = assess_quality(
        calibration_path=calibration,
        operator_override_path=output,
        allow_operator_forced_override=True,
    ).snapshot()
    assert report["camera_from_lidar_row_major"] == NUMERICAL_CANDIDATE_CAMERA_FROM_LIDAR


def test_tailnet_launchers_wire_only_an_explicit_operator_override_to_the_dashboard_runtime():
    pc_stack = PC_STACK.read_text(encoding="utf-8")
    tailnet_session = TAILNET_SESSION.read_text(encoding="utf-8")
    compatibility_session = COMPATIBILITY_SESSION.read_text(encoding="utf-8")

    assert "[string]$OperatorForcedCalibrationOverride" in pc_stack
    assert "[switch]$AllowOperatorForcedCalibrationOverride" in pc_stack
    assert "operator_forced_calibration_override" in pc_stack
    assert "allow_operator_forced_calibration" in pc_stack
    assert "$linuxOperatorForcedCalibrationOverride" in pc_stack
    assert "GO2_MAPPING_OPERATOR_FORCED_CALIBRATION_OVERRIDE=$linuxOperatorForcedCalibrationOverride" in pc_stack
    assert "GO2_MAPPING_ALLOW_OPERATOR_FORCED_CALIBRATION=$([int][bool]$AllowOperatorForcedCalibrationOverride)" in pc_stack

    for script in (tailnet_session, compatibility_session):
        assert "[string]$OperatorForcedCalibrationOverride" in script
        assert "OPERATOR_FORCED_FORMAL_RUNTIME" in script
        assert "formal_runtime_override" in script
        assert "measurement_authority" in script
    assert "OperatorForcedCalibrationOverride = $OperatorForcedCalibrationOverride" in tailnet_session
    assert "AllowOperatorForcedCalibrationOverride = $true" in tailnet_session
    assert "/api/runtime/calibration-status" in tailnet_session
    assert "OPERATOR_FORCED_RUNTIME_NOT_ACTIVE" in tailnet_session


def test_pc_stack_reuses_dashboard_only_when_the_requested_operator_override_is_active():
    pc_stack = PC_STACK.read_text(encoding="utf-8")

    assert "$operatorOverrideReady" in pc_stack
    assert "/api/runtime/calibration-status" in pc_stack
    assert "$runtimeCalibration.formal_runtime_override -eq $true" in pc_stack
    assert "$runtimeCalibration.measurement_authority -eq $false" in pc_stack
    assert "$runtimeCalibration.runtime_authority -eq 'OPERATOR_FORCED_FORMAL'" in pc_stack
    assert "$runtimeCalibration.runtime_projection_authorized -eq $true" in pc_stack
    assert "@($runtimeCalibration.camera_from_lidar_row_major).Count -eq 16" in pc_stack
