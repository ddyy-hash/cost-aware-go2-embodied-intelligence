"""Strict, evidence-based promotion for Go2 mapping quality labels.

The mapper may always produce a preview.  It may not silently promote that
preview to a calibrated, centimetre-grade, or BIM-aligned result.  Promotion
requires versioned calibration evidence and independent session validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .reconstruction_config import (
    ConfigError,
    canonical_sha256,
    is_sha256,
    load_reconstruction_config,
    parse_calibration_manifest,
    parse_session_evidence,
)
from .operator_override import (
    OperatorForcedCalibrationOverrideError,
    load_operator_forced_calibration_override,
    operator_override_source_diagnostic_path,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "quality_gates.json"


class QualityLevel(str, Enum):
    PREVIEW_UNCALIBRATED = "PREVIEW_UNCALIBRATED"
    RAW_CAPTURE_VALID = "RAW_CAPTURE_VALID"
    CALIBRATED_DENSE = "CALIBRATED_DENSE"
    GEOMETRY_VALIDATED_CM = "GEOMETRY_VALIDATED_CM"
    BIM_REGISTERED_VALIDATED = "BIM_REGISTERED_VALIDATED"


_RANK = {level: rank for rank, level in enumerate(QualityLevel)}


def _load_json(path: str | Path | None, label: str) -> tuple[dict[str, Any] | None, list[str], str | None]:
    if not path:
        return None, [f"{label}_MISSING"], None
    source = Path(path)
    if not source.is_file():
        return None, [f"{label}_NOT_FOUND:{source}"], None
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"{label}_INVALID_JSON:{exc}"], None
    if not isinstance(value, dict):
        return None, [f"{label}_ROOT_MUST_BE_OBJECT"], None
    try:
        identity_hash = canonical_sha256(value)
    except ConfigError as exc:
        return None, [f"{label}_INVALID_JSON:{exc}"], None
    return value, [], identity_hash


def _number(value: Any, field: str, reasons: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        reasons.append(f"METRIC_INVALID:{field}")
        return None
    return float(value)


def _sha256(value: Any) -> bool:
    return is_sha256(value)


def _rigid_matrix(value: Any, field: str, reasons: list[str]) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != 16:
        reasons.append(f"TRANSFORM_MATRIX_INVALID:{field}")
        return None
    try:
        matrix = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        reasons.append(f"TRANSFORM_MATRIX_INVALID:{field}")
        return None
    if not all(math.isfinite(component) for component in matrix):
        reasons.append(f"TRANSFORM_MATRIX_INVALID:{field}")
        return None
    if any(abs(matrix[index] - target) > 1e-6 for index, target in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0))):
        reasons.append(f"TRANSFORM_MATRIX_NOT_HOMOGENEOUS:{field}")
        return None
    rotation = (
        matrix[0:3],
        matrix[4:7],
        matrix[8:11],
    )
    for row in rotation:
        if abs(sum(component * component for component in row) - 1.0) > 1e-3:
            reasons.append(f"TRANSFORM_MATRIX_NOT_RIGID:{field}")
            return None
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if abs(sum(rotation[left][index] * rotation[right][index] for index in range(3))) > 1e-3:
            reasons.append(f"TRANSFORM_MATRIX_NOT_RIGID:{field}")
            return None
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        reasons.append(f"TRANSFORM_MATRIX_NOT_RIGID:{field}")
        return None
    return matrix


def _at_most(payload: Mapping[str, Any], field: str, threshold: float, reasons: list[str]) -> bool:
    value = _number(payload.get(field), field, reasons)
    if value is None:
        return False
    if value > threshold:
        reasons.append(f"METRIC_EXCEEDS:{field}={value:g}>{threshold:g}")
        return False
    return True


def _at_least(payload: Mapping[str, Any], field: str, threshold: float, reasons: list[str]) -> bool:
    value = _number(payload.get(field), field, reasons)
    if value is None:
        return False
    if value < threshold:
        reasons.append(f"METRIC_BELOW:{field}={value:g}<{threshold:g}")
        return False
    return True


def _object(payload: Mapping[str, Any], field: str, reasons: list[str]) -> Mapping[str, Any] | None:
    value = payload.get(field)
    if not isinstance(value, dict):
        reasons.append(f"OBJECT_MISSING:{field}")
        return None
    return value


@dataclass(frozen=True)
class QualityAssessment:
    level: QualityLevel
    calibration_id: str | None
    calibration_hash: str | None
    session_hash: str | None
    blocking_reasons: tuple[str, ...]
    required_artifact_sha256: tuple[str, ...] = ()
    raw_capture_verified: bool = False
    camera_from_lidar_row_major: tuple[float, ...] = ()
    appearance_authorized: bool = False
    operator_forced_runtime_override: bool = False
    calibration_manifest_hash: str | None = None
    operator_forced_override_hash: str | None = None

    @property
    def measurement_authority(self) -> bool:
        return (
            not self.operator_forced_runtime_override
            and _RANK[self.level] >= _RANK[QualityLevel.CALIBRATED_DENSE]
        )

    @property
    def runtime_authority(self) -> str:
        if self.operator_forced_runtime_override:
            return "OPERATOR_FORCED_FORMAL"
        if self.measurement_authority:
            return "MEASURED"
        return "NONE"

    @property
    def dense_quality(self) -> str:
        if self.level in {
            QualityLevel.CALIBRATED_DENSE,
            QualityLevel.GEOMETRY_VALIDATED_CM,
            QualityLevel.BIM_REGISTERED_VALIDATED,
        }:
            return "DENSE_LIDAR_CALIBRATED"
        return "DENSE_LIDAR_PREVIEW_UNCALIBRATED"

    def snapshot(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "dense_quality": self.dense_quality,
            "calibration_id": self.calibration_id,
            "calibration_sha256": self.calibration_hash,
            "calibration_manifest_sha256": self.calibration_manifest_hash,
            "operator_forced_calibration_override_sha256": self.operator_forced_override_hash,
            "session_evidence_sha256": self.session_hash,
            "blocking_reasons": list(self.blocking_reasons),
            "required_artifact_sha256": list(self.required_artifact_sha256),
            "raw_capture_verified": self.raw_capture_verified,
            "camera_from_lidar_row_major": list(self.camera_from_lidar_row_major),
            "formal_runtime_override": self.operator_forced_runtime_override,
            "measurement_authority": self.measurement_authority,
            "runtime_authority": self.runtime_authority,
            "eligible_for": {
                "dense_lidar_calibrated": _RANK[self.level] >= _RANK[QualityLevel.CALIBRATED_DENSE],
                "centimetre_geometry": _RANK[self.level] >= _RANK[QualityLevel.GEOMETRY_VALIDATED_CM],
                "bim_coordinate_delivery": _RANK[self.level] >= _RANK[QualityLevel.BIM_REGISTERED_VALIDATED],
                "rgbd_geometry_augmentation": self.appearance_authorized,
                "texture": self.appearance_authorized,
                "neural_geometry": self.appearance_authorized,
                "rgbd_or_neural_geometry_fusion": self.appearance_authorized,
                "operator_forced_runtime_projection": self.operator_forced_runtime_override,
            },
        }


def _calibration_v1_passes(
    calibration: Mapping[str, Any], config: Mapping[str, Any], reasons: list[str]
) -> tuple[bool, tuple[float, ...]]:
    if calibration.get("schema_version") != 1:
        reasons.append("CALIBRATION_SCHEMA_VERSION_UNSUPPORTED")
    if calibration.get("status") != "measured":
        reasons.append("CALIBRATION_NOT_MEASURED")
    if not isinstance(calibration.get("calibration_id"), str) or not calibration["calibration_id"].strip():
        reasons.append("CALIBRATION_ID_MISSING")
    robot = _object(calibration, "robot", reasons)
    if robot is not None and robot.get("model") != "go2_edu":
        reasons.append("ROBOT_MODEL_MUST_BE_GO2_EDU")
    captures = calibration.get("source_captures")
    calibration_gates = config["calibration"]
    if not isinstance(captures, list) or len(captures) < calibration_gates["minimum_source_captures"]:
        reasons.append("CALIBRATION_SOURCE_CAPTURES_INSUFFICIENT")
    elif any(not _sha256(item) for item in captures):
        reasons.append("CALIBRATION_SOURCE_CAPTURE_HASH_INVALID")

    intrinsics = _object(calibration, "camera_intrinsics", reasons)
    if intrinsics is not None and not _sha256(intrinsics.get("camera_info_sha256")):
        reasons.append("CAMERA_INFO_HASH_MISSING")
    camera_from_lidar: tuple[float, ...] = ()
    transforms = _object(calibration, "transforms", reasons)
    if transforms is not None:
        for transform_name in ("lidar_to_imu", "lidar_to_camera", "camera_to_base"):
            transform = transforms.get(transform_name)
            if not isinstance(transform, dict) or not _sha256(transform.get("sha256")):
                reasons.append(f"TRANSFORM_MISSING:{transform_name}")
            elif transform_name == "lidar_to_camera":
                camera_from_lidar = _rigid_matrix(transform.get("camera_from_lidar_row_major"), transform_name, reasons) or ()

    repeatability = _object(calibration, "repeatability", reasons)
    timing = _object(calibration, "timing", reasons)
    reprojection = _object(calibration, "camera_reprojection", reasons)
    if repeatability is not None:
        _at_most(repeatability, "lidar_imu_translation_mm", calibration_gates["lidar_imu_translation_repeatability_mm"], reasons)
        _at_most(repeatability, "lidar_imu_rotation_deg", calibration_gates["lidar_imu_rotation_repeatability_deg"], reasons)
        _at_most(repeatability, "lidar_camera_translation_mm", calibration_gates["lidar_camera_translation_repeatability_mm"], reasons)
        _at_most(repeatability, "lidar_camera_rotation_deg", calibration_gates["lidar_camera_rotation_repeatability_deg"], reasons)
    if timing is not None:
        _at_most(timing, "lidar_imu_repeatability_ms", calibration_gates["lidar_imu_timing_repeatability_ms"], reasons)
        _at_most(timing, "lidar_camera_repeatability_ms", calibration_gates["lidar_camera_timing_repeatability_ms"], reasons)
    if reprojection is not None:
        _at_most(reprojection, "median_px", calibration_gates["camera_reprojection_median_px"], reasons)
        _at_most(reprojection, "p95_px", calibration_gates["camera_reprojection_p95_px"], reasons)
    return not reasons, camera_from_lidar


def _calibration_passes(
    calibration: Mapping[str, Any], config: Mapping[str, Any], reasons: list[str]
) -> tuple[bool, tuple[float, ...], bool]:
    schema_version = calibration.get("schema_version")
    if schema_version == 1:
        accepted, camera_from_lidar = _calibration_v1_passes(calibration, config, reasons)
        return accepted, camera_from_lidar, False
    if schema_version != 2:
        reasons.append("CALIBRATION_SCHEMA_VERSION_UNSUPPORTED")
        return False, (), False
    try:
        appearance = load_reconstruction_config().appearance
        manifest = parse_calibration_manifest(calibration, appearance)
    except ConfigError as exc:
        reasons.append(f"CALIBRATION_V2_INVALID:{exc}")
        return False, (), False
    if manifest.status != "measured":
        reasons.append("CALIBRATION_NOT_MEASURED")
    gates = config["calibration"]
    repeatability = {
        "lidar_imu_translation_mm": manifest.repeatability.lidar_imu_translation_mm,
        "lidar_imu_rotation_deg": manifest.repeatability.lidar_imu_rotation_deg,
        "lidar_camera_translation_mm": manifest.repeatability.lidar_camera_translation_mm,
        "lidar_camera_rotation_deg": manifest.repeatability.lidar_camera_rotation_deg,
    }
    timing = {
        "lidar_imu_repeatability_ms": manifest.timing.lidar_imu_repeatability_ms,
        "lidar_camera_repeatability_ms": manifest.timing.lidar_camera_repeatability_ms,
    }
    _at_most(repeatability, "lidar_imu_translation_mm", gates["lidar_imu_translation_repeatability_mm"], reasons)
    _at_most(repeatability, "lidar_imu_rotation_deg", gates["lidar_imu_rotation_repeatability_deg"], reasons)
    _at_most(repeatability, "lidar_camera_translation_mm", gates["lidar_camera_translation_repeatability_mm"], reasons)
    _at_most(repeatability, "lidar_camera_rotation_deg", gates["lidar_camera_rotation_repeatability_deg"], reasons)
    _at_most(timing, "lidar_imu_repeatability_ms", gates["lidar_imu_timing_repeatability_ms"], reasons)
    _at_most(timing, "lidar_camera_repeatability_ms", gates["lidar_camera_timing_repeatability_ms"], reasons)
    return not reasons, manifest.frame_chain.t_c_l.row_major, not reasons


def _geometry_only_calibration_passes(calibration: Mapping[str, Any], reasons: list[str]) -> tuple[bool, tuple[float, ...]]:
    """Accept the measured geometry contract when appearance evidence is absent.

    Model delivery uses the LiDAR geometry and camera projection only. Texture
    and depth-validation evidence remain separate and cannot be inferred here.
    This path is enabled explicitly by the owned runtime launcher.
    """
    if calibration.get("schema_version") != 2 or calibration.get("status") != "measured":
        reasons.append("GEOMETRY_ONLY_CALIBRATION_NOT_MEASURED")
        return False, ()
    frame = calibration.get("frame_chain")
    transform = frame.get("t_c_l") if isinstance(frame, Mapping) else None
    values = transform.get("row_major") if isinstance(transform, Mapping) else None
    if not isinstance(values, list) or len(values) != 16:
        reasons.append("GEOMETRY_ONLY_CAMERA_FROM_LIDAR_MISSING")
        return False, ()
    try:
        matrix = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        reasons.append("GEOMETRY_ONLY_CAMERA_FROM_LIDAR_INVALID")
        return False, ()
    if not all(math.isfinite(value) for value in matrix):
        reasons.append("GEOMETRY_ONLY_CAMERA_FROM_LIDAR_NONFINITE")
        return False, ()
    captures = calibration.get("capture_sets")
    if not isinstance(captures, Mapping) or not isinstance(captures.get("estimation_artifact_sha256"), list):
        reasons.append("GEOMETRY_ONLY_CAPTURE_IDENTITY_MISSING")
        return False, ()
    return True, matrix


def _raw_capture_passes(session: Mapping[str, Any], reasons: list[str]) -> bool:
    if session.get("schema_version") == 2:
        try:
            parse_session_evidence(session)
        except ConfigError as exc:
            reasons.append(f"SESSION_V2_INVALID:{exc}")
            return False
    elif session.get("schema_version") != 1:
        reasons.append("SESSION_SCHEMA_VERSION_UNSUPPORTED")
    raw = _object(session, "raw_capture", reasons)
    if raw is not None:
        if raw.get("sha256_verified") is not True:
            reasons.append("RAW_CAPTURE_NOT_SHA256_VERIFIED")
        _at_least(raw, "artifact_count", 1, reasons)
        artifact_hashes = raw.get("artifact_sha256")
        artifact_count = raw.get("artifact_count")
        if (
            not isinstance(artifact_hashes, list)
            or not isinstance(artifact_count, int)
            or len(artifact_hashes) != artifact_count
            or any(not _sha256(value) for value in artifact_hashes)
        ):
            reasons.append("RAW_CAPTURE_ARTIFACT_HASHES_INVALID")
        streams = raw.get("required_streams")
        required = {"lidar", "imu", "lio_pose", "rgb", "depth", "camera_info", "tf", "tf_static"}
        if not isinstance(streams, list) or not required.issubset(set(streams)):
            reasons.append("RAW_CAPTURE_REQUIRED_STREAMS_MISSING")
    return not reasons


def _session_matches_calibration(session: Mapping[str, Any], calibration_id: str, reasons: list[str]) -> bool:
    if session.get("calibration_id") != calibration_id:
        reasons.append("SESSION_CALIBRATION_ID_MISMATCH")
    return not reasons


def _required_artifact_hashes(session: Mapping[str, Any]) -> tuple[str, ...]:
    raw = session.get("raw_capture")
    if not isinstance(raw, dict):
        return ()
    hashes = raw.get("artifact_sha256")
    if not isinstance(hashes, list) or any(not _sha256(value) for value in hashes):
        return ()
    return tuple(hashes)


def _geometry_passes(session: Mapping[str, Any], config: Mapping[str, Any], reasons: list[str]) -> bool:
    geometry = _object(session, "geometry_validation", reasons)
    if geometry is None:
        return False
    gates = config["geometry"]
    _at_least(geometry, "independent_measurement_count", gates["minimum_independent_measurements"], reasons)
    _at_most(geometry, "p95_dimension_error_mm", gates["p95_dimension_error_mm"], reasons)
    _at_most(geometry, "maximum_dimension_error_mm", gates["maximum_dimension_error_mm"], reasons)
    _at_least(geometry, "accepted_loop_closures", gates["minimum_accepted_loop_closures"], reasons)
    _at_most(geometry, "loop_point_to_plane_p95_mm", gates["loop_point_to_plane_p95_mm"], reasons)
    if geometry.get("independent_measurements_used_for_calibration") is not False:
        reasons.append("INDEPENDENT_MEASUREMENTS_NOT_HELD_OUT")
    return not reasons


def _bim_passes(session: Mapping[str, Any], config: Mapping[str, Any], reasons: list[str]) -> bool:
    bim = _object(session, "bim_registration", reasons)
    if bim is None:
        return False
    gates = config["bim"]
    _at_least(bim, "independent_structural_groups", gates["minimum_independent_structural_groups"], reasons)
    _at_least(bim, "holdout_structural_groups", gates["minimum_holdout_structural_groups"], reasons)
    _at_most(bim, "holdout_p95_residual_mm", gates["registration_p95_residual_mm"], reasons)
    if bim.get("rigid_se3_only") is not True:
        reasons.append("BIM_REGISTRATION_NOT_RIGID_SE3")
    if not _sha256(bim.get("design_bim_hash")):
        reasons.append("BIM_SOURCE_HASH_MISSING")
    return not reasons


def _operator_forced_runtime_context_passes(
    calibration: Mapping[str, Any],
    override_calibration_id: str,
    reasons: list[str],
) -> bool:
    if calibration.get("calibration_id") != override_calibration_id:
        reasons.append("OPERATOR_FORCED_OVERRIDE_CALIBRATION_ID_MISMATCH")
    if calibration.get("schema_version") != 1:
        reasons.append("OPERATOR_FORCED_OVERRIDE_CAPTURE_CONTEXT_SCHEMA_INVALID")
    if calibration.get("status") != "provisional":
        reasons.append("OPERATOR_FORCED_OVERRIDE_CAPTURE_CONTEXT_STATUS_INVALID")
    if calibration.get("purpose") != "calibration_capture":
        reasons.append("OPERATOR_FORCED_OVERRIDE_CAPTURE_CONTEXT_PURPOSE_INVALID")
    if calibration.get("acceptance_authority") is not False:
        reasons.append("OPERATOR_FORCED_OVERRIDE_CAPTURE_CONTEXT_AUTHORITY_INVALID")
    return not reasons


def assess_quality(
    calibration_path: str | Path | None = None,
    session_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    operator_override_path: str | Path | None = None,
    allow_operator_forced_override: bool = False,
) -> QualityAssessment:
    """Assess all supplied evidence, returning the highest defensible level."""
    config, config_reasons, _ = _load_json(config_path, "QUALITY_CONFIG")
    calibration, calibration_reasons, calibration_hash = _load_json(calibration_path, "CALIBRATION")
    session, session_reasons, session_hash = _load_json(session_path, "SESSION_EVIDENCE")
    operator_override = None
    operator_override_reasons: list[str] = []
    if operator_override_path:
        if not allow_operator_forced_override:
            operator_override_reasons.append("OPERATOR_FORCED_OVERRIDE_DISABLED")
        else:
            try:
                operator_override = load_operator_forced_calibration_override(operator_override_path)
            except OperatorForcedCalibrationOverrideError as exc:
                operator_override_reasons.append(f"OPERATOR_FORCED_OVERRIDE_INVALID:{exc}")
    elif allow_operator_forced_override:
        operator_override_reasons.append("OPERATOR_FORCED_OVERRIDE_MISSING")
    calibration_id = calibration.get("calibration_id") if calibration else None
    reasons = config_reasons + calibration_reasons + operator_override_reasons
    if config is None:
        return QualityAssessment(QualityLevel.PREVIEW_UNCALIBRATED, calibration_id, calibration_hash, session_hash, tuple(reasons))
    raw_blockers: list[str] = []
    raw_capture = session is not None and _raw_capture_passes(session, raw_blockers)
    artifact_hashes = _required_artifact_hashes(session) if raw_capture and session is not None else ()
    session_v2_valid = False
    if session is not None and session.get("schema_version") == 2:
        try:
            parse_session_evidence(session)
            session_v2_valid = True
        except ConfigError as exc:
            session_reasons.append(f"SESSION_SCHEMA_INVALID:{exc}")
    if operator_override is not None:
        override_context_reasons: list[str] = []
        if calibration is None or not _operator_forced_runtime_context_passes(
            calibration,
            operator_override.calibration_id,
            override_context_reasons,
        ):
            return QualityAssessment(
                QualityLevel.PREVIEW_UNCALIBRATED,
                operator_override.calibration_id,
                operator_override.override_sha256,
                session_hash,
                tuple(reasons + override_context_reasons + session_reasons + raw_blockers),
                artifact_hashes,
                raw_capture,
                (),
                False,
                True,
                calibration_hash,
                operator_override.override_sha256,
            )
        return QualityAssessment(
            QualityLevel.CALIBRATED_DENSE,
            operator_override.calibration_id,
            operator_override.override_sha256,
            session_hash,
            tuple(reasons + session_reasons),
            artifact_hashes,
            raw_capture,
            operator_override.camera_from_lidar_row_major,
            True,
            True,
            calibration_hash,
            operator_override.override_sha256,
        )
    if calibration is None:
        return QualityAssessment(
            QualityLevel.PREVIEW_UNCALIBRATED,
            None,
            calibration_hash,
            session_hash,
            tuple(reasons + session_reasons + raw_blockers),
            artifact_hashes,
            raw_capture,
        )

    calibration_blockers: list[str] = []
    calibrated, camera_from_lidar, appearance_authorized = _calibration_passes(
        calibration, config, calibration_blockers
    )
    if not calibrated and os.environ.get("GO2_MAPPING_GEOMETRY_ONLY_CALIBRATION") == "1":
        calibration_blockers.clear()
        calibrated, camera_from_lidar = _geometry_only_calibration_passes(calibration, calibration_blockers)
        appearance_authorized = False
    if not calibrated:
        return QualityAssessment(
            QualityLevel.PREVIEW_UNCALIBRATED,
            calibration_id,
            calibration_hash,
            session_hash,
            tuple(reasons + calibration_blockers),
            artifact_hashes,
            raw_capture,
            camera_from_lidar,
        )

    if session is None:
        return QualityAssessment(
            QualityLevel.CALIBRATED_DENSE,
            calibration_id,
            calibration_hash,
            session_hash,
            tuple(session_reasons),
            (),
            False,
            camera_from_lidar,
            False,
        )

    calibration_session_blockers: list[str] = []
    calibration_session_matches = _session_matches_calibration(session, str(calibration_id), calibration_session_blockers)
    if not raw_capture or not calibration_session_matches:
        return QualityAssessment(
            QualityLevel.CALIBRATED_DENSE,
            calibration_id,
            calibration_hash,
            session_hash,
            tuple(session_reasons + raw_blockers + calibration_session_blockers),
            artifact_hashes,
            raw_capture,
            camera_from_lidar,
            False,
        )

    # Appearance products need the typed schema-v2 contract, verified raw data,
    # and a calibration identity match. Calibration alone never unlocks them.
    appearance_authorized = appearance_authorized and session_v2_valid

    geometry_blockers: list[str] = []
    geometry = _geometry_passes(session, config, geometry_blockers)
    if not geometry:
        return QualityAssessment(
            QualityLevel.CALIBRATED_DENSE,
            calibration_id,
            calibration_hash,
            session_hash,
            tuple(geometry_blockers),
            artifact_hashes,
            True,
            camera_from_lidar,
            appearance_authorized,
        )

    bim_blockers: list[str] = []
    bim = _bim_passes(session, config, bim_blockers)
    if not bim:
        return QualityAssessment(
            QualityLevel.GEOMETRY_VALIDATED_CM,
            calibration_id,
            calibration_hash,
            session_hash,
            tuple(bim_blockers),
            artifact_hashes,
            True,
            camera_from_lidar,
            appearance_authorized,
        )
    return QualityAssessment(
        QualityLevel.BIM_REGISTERED_VALIDATED,
        calibration_id,
        calibration_hash,
        session_hash,
        (),
        artifact_hashes,
        True,
        camera_from_lidar,
        appearance_authorized,
    )


def meets_requirement(assessment: QualityAssessment, required: QualityLevel | str) -> bool:
    """Test a requested delivery condition without treating raw capture as a rank."""
    required_level = QualityLevel(required)
    if required_level is QualityLevel.RAW_CAPTURE_VALID:
        return assessment.raw_capture_verified
    return _RANK[assessment.level] >= _RANK[required_level]


class MappingQualityGate:
    """Reload JSON evidence only when it changes, for dashboard-safe snapshots."""

    def __init__(
        self,
        calibration_path: str | Path | None = None,
        session_path: str | Path | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        operator_override_path: str | Path | None = None,
        allow_operator_forced_override: bool = False,
    ) -> None:
        self.calibration_path = str(calibration_path) if calibration_path else None
        self.session_path = str(session_path) if session_path else None
        self.config_path = str(config_path)
        self.operator_override_path = str(operator_override_path) if operator_override_path else None
        self.allow_operator_forced_override = bool(allow_operator_forced_override)
        self._signature: tuple[tuple[str | None, int | None, int | None], ...] | None = None
        self._assessment: QualityAssessment | None = None

    @classmethod
    def from_environment(cls) -> "MappingQualityGate":
        return cls(
            calibration_path=os.environ.get("GO2_MAPPING_CALIBRATION_MANIFEST"),
            session_path=os.environ.get("GO2_MAPPING_SESSION_EVIDENCE"),
            config_path=os.environ.get("GO2_MAPPING_QUALITY_GATES", str(DEFAULT_CONFIG_PATH)),
            operator_override_path=os.environ.get("GO2_MAPPING_OPERATOR_FORCED_CALIBRATION_OVERRIDE"),
            allow_operator_forced_override=os.environ.get("GO2_MAPPING_ALLOW_OPERATOR_FORCED_CALIBRATION") == "1",
        )

    def snapshot(self) -> dict[str, Any]:
        operator_override_source = operator_override_source_diagnostic_path(
            self.operator_override_path
        )
        signature = tuple(
            self._file_signature(path)
            for path in (
                self.config_path,
                self.calibration_path,
                self.session_path,
                self.operator_override_path,
                operator_override_source,
            )
        )
        if self._assessment is None or signature != self._signature:
            self._assessment = assess_quality(
                self.calibration_path,
                self.session_path,
                self.config_path,
                self.operator_override_path,
                self.allow_operator_forced_override,
            )
            self._signature = signature
        return self._assessment.snapshot()

    @staticmethod
    def _file_signature(path: str | None) -> tuple[str | None, int | None, int | None]:
        if not path:
            return None, None, None
        try:
            stat = Path(path).stat()
        except OSError:
            return path, None, None
        return path, stat.st_mtime_ns, stat.st_size
