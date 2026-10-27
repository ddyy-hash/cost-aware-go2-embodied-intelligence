"""Contracts for an auditable operator-forced calibration override.

The override is separate from a measured manifest. It can activate an axis-only
diagnostic seed for an approved session, but it is not solver evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from mapping.pc.go2_mapping_calibration.axis_contract import (
    DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_ID,
    DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1,
)

from .reconstruction_config import canonical_sha256, is_sha256


OPERATOR_FORCED_CALIBRATION_OVERRIDE_KIND = "GO2_OPERATOR_FORCED_CALIBRATION_OVERRIDE"
OPERATOR_FORCED_CALIBRATION_OVERRIDE_STATUS = "FORMAL_OPERATOR_FORCED_RUNTIME"
OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND = "AXIS_ONLY_SEED"
OPERATOR_FORCED_NUMERICAL_CANDIDATE_OVERRIDE_KIND = "NUMERICAL_CANDIDATE"
_SOURCE_STATUS_UNOBSERVABLE = "JOINT_DIAGNOSTIC_UNOBSERVABLE"
_SOURCE_STATUS_PROVISIONAL = "JOINT_DIAGNOSTIC_PROVISIONAL"
_SOURCE_CAMERA_FROM_LIDAR = "SOURCE_JOINT_DIAGNOSTIC_CAMERA_FROM_LIDAR"
_CALIBRATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_AXIS_ONLY_ROW_MAJOR = tuple(
    float(value)
    for row in DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1
    for value in (*row, 0)
) + (0.0, 0.0, 0.0, 1.0)


class OperatorForcedCalibrationOverrideError(ValueError):
    """Raised when an override cannot prove its declared non-measured source."""


@dataclass(frozen=True)
class OperatorForcedCalibrationOverride:
    calibration_id: str
    override_sha256: str
    camera_from_lidar_row_major: tuple[float, ...]
    source_joint_diagnostic_sha256: str
    source_artifact_sha256: tuple[str, ...]
    runtime_override_kind: str = OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND
    source_joint_status: str = _SOURCE_STATUS_UNOBSERVABLE
    delta_camera_lidar_ms: float | None = None
    fit_capture_ids: tuple[str, ...] = ()
    holdout_capture_ids: tuple[str, ...] = ()
    quality_limits_bypassed: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ValidatedSourceDiagnostic:
    sha256: str
    status: str
    source_artifact_sha256: tuple[str, ...]
    fit_capture_ids: tuple[str, ...] = ()
    holdout_capture_ids: tuple[str, ...] = ()
    camera_from_lidar_row_major: tuple[float, ...] | None = None
    delta_camera_lidar_ms: float | None = None


def _error(code: str) -> None:
    raise OperatorForcedCalibrationOverrideError(code)


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorForcedCalibrationOverrideError(f"{label}_UNREADABLE:{exc}") from exc
    if not isinstance(payload, dict):
        _error(f"{label}_ROOT_MUST_BE_OBJECT")
    return payload, raw


def _required_string(payload: Mapping[str, Any], field: str, code: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        _error(f"{code}:{field}")
    return value


def _required_bool(payload: Mapping[str, Any], field: str, expected: bool, code: str) -> None:
    if payload.get(field) is not expected:
        _error(f"{code}:{field}")


def _sha256_values(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not is_sha256(item) or item != item.lower() for item in value
    ):
        _error(code)
    return tuple(value)


def _string_values(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        _error(code)
    return tuple(value)


def _optional_string_values(value: Any, code: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _error(code)
    return tuple(value)


def _finite_float(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OperatorForcedCalibrationOverrideError(code) from exc
    if not math.isfinite(result):
        _error(code)
    return result


def _rigid_transform_row_major(value: Any, code: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 16:
        _error(code)
    try:
        transform = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OperatorForcedCalibrationOverrideError(code) from exc
    if not all(math.isfinite(item) for item in transform):
        _error(code)
    if any(abs(transform[index]) > 1e-9 for index in (12, 13, 14)) or abs(transform[15] - 1.0) > 1e-9:
        _error(code)
    return transform


def _same_float_lists(left: tuple[float, ...], right: tuple[float, ...], *, atol: float = 1e-9) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= atol for a, b in zip(left, right, strict=True))


def _axis_only_transform(payload: Mapping[str, Any]) -> tuple[float, ...]:
    if payload.get("axis_contract_id") != DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_ID:
        _error("OVERRIDE_AXIS_CONTRACT_INVALID")
    transform = _rigid_transform_row_major(payload.get("row_major"), "OVERRIDE_TRANSFORM_INVALID")
    if transform != _AXIS_ONLY_ROW_MAJOR:
        _error("OVERRIDE_TRANSFORM_NOT_AXIS_ONLY_ZERO_TRANSLATION")
    if payload.get("translation_m") != [0.0, 0.0, 0.0] or payload.get("translation_measured") is not False:
        _error("OVERRIDE_TRANSLATION_MUST_BE_UNMEASURED_ZERO")
    return transform


def _candidate_transform(payload: Mapping[str, Any]) -> tuple[tuple[float, ...], float]:
    if payload.get("source") != _SOURCE_CAMERA_FROM_LIDAR:
        _error("OVERRIDE_TRANSFORM_SOURCE_INVALID")
    transform = _rigid_transform_row_major(payload.get("row_major"), "OVERRIDE_TRANSFORM_INVALID")
    translation = payload.get("translation_m")
    if not isinstance(translation, list) or len(translation) != 3:
        _error("OVERRIDE_TRANSLATION_INVALID")
    expected_translation = (transform[3], transform[7], transform[11])
    actual_translation = tuple(_finite_float(item, "OVERRIDE_TRANSLATION_INVALID") for item in translation)
    if not _same_float_lists(actual_translation, expected_translation):
        _error("OVERRIDE_TRANSLATION_DOES_NOT_MATCH_ROW_MAJOR")
    if payload.get("translation_measured") is not False:
        _error("OVERRIDE_TRANSLATION_AUTHORITY_INVALID")
    delta_camera_lidar_ms = _finite_float(
        payload.get("delta_camera_lidar_ms"),
        "OVERRIDE_DELTA_CAMERA_LIDAR_MS_INVALID",
    )
    return transform, delta_camera_lidar_ms


def _source_path(override_path: Path, source: Mapping[str, Any]) -> Path:
    relative_path = _required_string(source, "relative_path", "OVERRIDE_SOURCE_PATH_INVALID")
    relative = Path(relative_path)
    if relative.is_absolute() or ":" in relative_path or ".." in relative.parts:
        _error("OVERRIDE_SOURCE_PATH_INVALID:relative_path")
    return override_path.parent / relative


def operator_override_source_diagnostic_path(path: str | Path | None) -> Path | None:
    """Return the declared source path for cache invalidation without trusting it."""

    if not path:
        return None
    try:
        override_path = Path(path)
        override, _ = _load_object(override_path, "OPERATOR_FORCED_OVERRIDE")
        source = override.get("source_joint_diagnostic")
        if not isinstance(source, dict):
            return None
        return _source_path(override_path, source)
    except OperatorForcedCalibrationOverrideError:
        return None


def _validate_joint_diagnostic(
    override_path: Path,
    source: Mapping[str, Any],
) -> _ValidatedSourceDiagnostic:
    source_path = _source_path(override_path, source)
    diagnostic, raw = _load_object(source_path, "OVERRIDE_SOURCE_DIAGNOSTIC")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    declared_sha256 = _required_string(source, "sha256", "OVERRIDE_SOURCE_SHA256_INVALID")
    if not is_sha256(declared_sha256) or declared_sha256 != actual_sha256:
        _error("OVERRIDE_SOURCE_SHA256_MISMATCH")
    joint = diagnostic.get("joint")
    if not isinstance(joint, dict):
        _error("OVERRIDE_SOURCE_DIAGNOSTIC_JOINT_MISSING")
    expected = {
        "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
        "calibration_authority": False,
        "manifest_written": False,
    }
    if diagnostic.get("analysis_kind") != expected["analysis_kind"]:
        _error("OVERRIDE_SOURCE_ANALYSIS_KIND_INVALID")
    if diagnostic.get("calibration_authority") is not False or diagnostic.get("manifest_written") is not False:
        _error("OVERRIDE_SOURCE_MEASUREMENT_AUTHORITY_INVALID")
    declared_status = _required_string(source, "status", "OVERRIDE_SOURCE_STATUS_INVALID")
    if joint.get("status") != declared_status:
        _error("OVERRIDE_SOURCE_STATUS_INVALID")
    if joint.get("calibration_authority") is not False or joint.get("manifest_written") is not False:
        _error("OVERRIDE_SOURCE_JOINT_AUTHORITY_INVALID")
    for field, expected_value in expected.items():
        if source.get(field) != expected_value:
            _error(f"OVERRIDE_SOURCE_DECLARATION_MISMATCH:{field}")
    captures = diagnostic.get("captures")
    if not isinstance(captures, list) or not captures:
        _error("OVERRIDE_SOURCE_CAPTURES_INVALID")

    if declared_status == _SOURCE_STATUS_UNOBSERVABLE:
        if joint.get("camera_from_lidar") is not None:
            _error("OVERRIDE_SOURCE_MUST_NOT_CONTAIN_MEASURED_TRANSFORM")
        artifacts = _sha256_values(joint.get("source_artifact_sha256"), "OVERRIDE_SOURCE_ARTIFACTS_INVALID")
        if _sha256_values(source.get("source_artifact_sha256"), "OVERRIDE_DECLARED_ARTIFACTS_INVALID") != artifacts:
            _error("OVERRIDE_SOURCE_ARTIFACTS_MISMATCH")
        if tuple(item.get("artifact_sha256") if isinstance(item, dict) else None for item in captures) != artifacts:
            _error("OVERRIDE_SOURCE_CAPTURE_ARTIFACTS_MISMATCH")
        return _ValidatedSourceDiagnostic(
            sha256=actual_sha256,
            status=declared_status,
            source_artifact_sha256=artifacts,
        )

    if declared_status != _SOURCE_STATUS_PROVISIONAL:
        _error("OVERRIDE_SOURCE_STATUS_INVALID")
    if diagnostic.get("provisional") is not True:
        _error("OVERRIDE_SOURCE_PROVISIONAL_FLAG_INVALID")
    artifacts = _sha256_values(joint.get("source_artifact_sha256"), "OVERRIDE_SOURCE_ARTIFACTS_INVALID")
    if _sha256_values(source.get("source_artifact_sha256"), "OVERRIDE_DECLARED_ARTIFACTS_INVALID") != artifacts:
        _error("OVERRIDE_SOURCE_ARTIFACTS_MISMATCH")

    capture_artifacts: dict[str, str] = {}
    for item in captures:
        if not isinstance(item, dict):
            _error("OVERRIDE_SOURCE_CAPTURES_INVALID")
        capture_id = item.get("capture_id")
        artifact_sha256 = item.get("artifact_sha256")
        if not isinstance(capture_id, str) or not capture_id or not is_sha256(artifact_sha256):
            _error("OVERRIDE_SOURCE_CAPTURES_INVALID")
        capture_artifacts[capture_id] = artifact_sha256

    fit_capture_ids = _string_values(diagnostic.get("fit_capture_ids"), "OVERRIDE_SOURCE_FIT_CAPTURE_IDS_INVALID")
    if _string_values(source.get("fit_capture_ids"), "OVERRIDE_DECLARED_FIT_CAPTURE_IDS_INVALID") != fit_capture_ids:
        _error("OVERRIDE_SOURCE_FIT_CAPTURE_IDS_MISMATCH")
    if tuple(capture_artifacts.get(capture_id) for capture_id in fit_capture_ids) != artifacts:
        _error("OVERRIDE_SOURCE_CAPTURE_ARTIFACTS_MISMATCH")

    holdout_capture_ids = _optional_string_values(
        diagnostic.get("holdout_capture_ids"),
        "OVERRIDE_SOURCE_HOLDOUT_CAPTURE_IDS_INVALID",
    )
    if _optional_string_values(
        source.get("holdout_capture_ids"),
        "OVERRIDE_DECLARED_HOLDOUT_CAPTURE_IDS_INVALID",
    ) != holdout_capture_ids:
        _error("OVERRIDE_SOURCE_HOLDOUT_CAPTURE_IDS_MISMATCH")
    holdout_diagnostics = diagnostic.get("holdout_diagnostics")
    if holdout_capture_ids:
        if not isinstance(holdout_diagnostics, list) or not holdout_diagnostics:
            _error("OVERRIDE_SOURCE_HOLDOUT_DIAGNOSTICS_INVALID")
        diagnostics_by_capture_id = {
            item.get("capture_id"): item
            for item in holdout_diagnostics
            if isinstance(item, dict) and isinstance(item.get("capture_id"), str)
        }
        for capture_id in holdout_capture_ids:
            holdout_entry = diagnostics_by_capture_id.get(capture_id)
            if not isinstance(holdout_entry, dict):
                _error("OVERRIDE_SOURCE_HOLDOUT_DIAGNOSTICS_INVALID")
            if holdout_entry.get("calibration_authority") is not False:
                _error("OVERRIDE_SOURCE_HOLDOUT_AUTHORITY_INVALID")

    camera_from_lidar = _rigid_transform_row_major(
        joint.get("camera_from_lidar"),
        "OVERRIDE_SOURCE_JOINT_TRANSFORM_INVALID",
    )
    delta_camera_lidar_ms = _finite_float(
        joint.get("delta_camera_lidar_ms"),
        "OVERRIDE_SOURCE_JOINT_DELTA_CAMERA_LIDAR_MS_INVALID",
    )
    return _ValidatedSourceDiagnostic(
        sha256=actual_sha256,
        status=declared_status,
        source_artifact_sha256=artifacts,
        fit_capture_ids=fit_capture_ids,
        holdout_capture_ids=holdout_capture_ids,
        camera_from_lidar_row_major=camera_from_lidar,
        delta_camera_lidar_ms=delta_camera_lidar_ms,
    )


def load_operator_forced_calibration_override(
    path: str | Path,
) -> OperatorForcedCalibrationOverride:
    """Load the only allowed operator-forced runtime override form."""

    override_path = Path(path)
    override, _ = _load_object(override_path, "OPERATOR_FORCED_OVERRIDE")
    if override.get("schema_version") != 1:
        _error("OVERRIDE_SCHEMA_VERSION_UNSUPPORTED")
    if override.get("kind") != OPERATOR_FORCED_CALIBRATION_OVERRIDE_KIND:
        _error("OVERRIDE_KIND_INVALID")
    if override.get("status") != OPERATOR_FORCED_CALIBRATION_OVERRIDE_STATUS:
        _error("OVERRIDE_STATUS_INVALID")
    _required_bool(override, "formal_runtime_override", True, "OVERRIDE_ACKNOWLEDGEMENT_INVALID")
    _required_bool(override, "operator_acknowledged", True, "OVERRIDE_ACKNOWLEDGEMENT_INVALID")
    _required_bool(override, "measurement_authority", False, "OVERRIDE_MEASUREMENT_AUTHORITY_INVALID")
    _required_bool(override, "runtime_appearance_authority", True, "OVERRIDE_RUNTIME_APPEARANCE_INVALID")
    calibration_id = _required_string(override, "calibration_id", "OVERRIDE_CALIBRATION_ID_INVALID")
    if _CALIBRATION_ID_RE.fullmatch(calibration_id) is None:
        _error("OVERRIDE_CALIBRATION_ID_INVALID")
    source = override.get("source_joint_diagnostic")
    if not isinstance(source, dict):
        _error("OVERRIDE_SOURCE_DIAGNOSTIC_MISSING")
    source_diagnostic = _validate_joint_diagnostic(override_path, source)
    camera = override.get("camera_from_lidar")
    if not isinstance(camera, dict):
        _error("OVERRIDE_TRANSFORM_MISSING")
    runtime_override_kind = override.get("runtime_override_kind")
    quality_limits_bypassed: tuple[str, ...] = ()
    delta_camera_lidar_ms: float | None = None
    if source_diagnostic.status == _SOURCE_STATUS_UNOBSERVABLE:
        if runtime_override_kind not in (None, OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND):
            _error("OVERRIDE_RUNTIME_KIND_INVALID")
        transform = _axis_only_transform(camera)
        runtime_override_kind = OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND
    else:
        if runtime_override_kind != OPERATOR_FORCED_NUMERICAL_CANDIDATE_OVERRIDE_KIND:
            _error("OVERRIDE_RUNTIME_KIND_INVALID")
        _required_bool(
            override,
            "candidate_usable_by_operator_decision",
            True,
            "OVERRIDE_CANDIDATE_ACKNOWLEDGEMENT_INVALID",
        )
        quality_limits_bypassed = _string_values(
            override.get("quality_limits_bypassed"),
            "OVERRIDE_QUALITY_LIMITS_INVALID",
        )
        transform, delta_camera_lidar_ms = _candidate_transform(camera)
        if source_diagnostic.camera_from_lidar_row_major is None or not _same_float_lists(
            transform,
            source_diagnostic.camera_from_lidar_row_major,
        ):
            _error("OVERRIDE_TRANSFORM_DOES_NOT_MATCH_SOURCE_DIAGNOSTIC")
        if source_diagnostic.delta_camera_lidar_ms is None or abs(
            delta_camera_lidar_ms - source_diagnostic.delta_camera_lidar_ms
        ) > 1e-9:
            _error("OVERRIDE_DELTA_DOES_NOT_MATCH_SOURCE_DIAGNOSTIC")
    return OperatorForcedCalibrationOverride(
        calibration_id=calibration_id,
        override_sha256=canonical_sha256(override),
        camera_from_lidar_row_major=transform,
        source_joint_diagnostic_sha256=source_diagnostic.sha256,
        source_artifact_sha256=source_diagnostic.source_artifact_sha256,
        runtime_override_kind=runtime_override_kind,
        source_joint_status=source_diagnostic.status,
        delta_camera_lidar_ms=delta_camera_lidar_ms,
        fit_capture_ids=source_diagnostic.fit_capture_ids,
        holdout_capture_ids=source_diagnostic.holdout_capture_ids,
        quality_limits_bypassed=quality_limits_bypassed,
    )
