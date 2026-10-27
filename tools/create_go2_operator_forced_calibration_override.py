#!/usr/bin/env python3
"""Create an auditable formal operator-forced Go2 runtime override."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_calibration.axis_contract import (  # noqa: E402
    DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_ID,
    DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1,
)
from mapping.pc.go2_mapping_quality.operator_override import (  # noqa: E402
    OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND,
    OPERATOR_FORCED_CALIBRATION_OVERRIDE_KIND,
    OPERATOR_FORCED_CALIBRATION_OVERRIDE_STATUS,
    OPERATOR_FORCED_NUMERICAL_CANDIDATE_OVERRIDE_KIND,
    OperatorForcedCalibrationOverrideError,
    load_operator_forced_calibration_override,
)


def _axis_only_row_major() -> list[float]:
    return [
        float(value)
        for row in DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1
        for value in (*row, 0)
    ] + [0.0, 0.0, 0.0, 1.0]


def _load_joint_diagnostic(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"joint diagnostic is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("joint"), dict):
        raise ValueError("joint diagnostic must contain an object-valued joint result")
    return payload, raw


def _build_payload(
    diagnostic_path: Path,
    diagnostic: dict,
    diagnostic_raw: bytes,
    calibration_id: str,
) -> dict:
    joint = diagnostic["joint"]
    try:
        joint_status = joint["status"]
    except KeyError as exc:
        raise ValueError(f"joint diagnostic is missing {exc.args[0]}") from exc
    try:
        source = {
            "relative_path": diagnostic_path.name,
            "sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
            "analysis_kind": diagnostic["analysis_kind"],
            "status": joint_status,
            "calibration_authority": diagnostic["calibration_authority"],
            "manifest_written": diagnostic["manifest_written"],
            "source_artifact_sha256": joint["source_artifact_sha256"],
        }
    except KeyError as exc:
        raise ValueError(f"joint diagnostic is missing {exc.args[0]}") from exc

    payload = {
        "schema_version": 1,
        "kind": OPERATOR_FORCED_CALIBRATION_OVERRIDE_KIND,
        "status": OPERATOR_FORCED_CALIBRATION_OVERRIDE_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "calibration_id": calibration_id,
        "formal_runtime_override": True,
        "operator_acknowledged": True,
        "operator_acknowledgement": "EXPLICIT_COMMAND_LINE_ACKNOWLEDGEMENT",
        "measurement_authority": False,
        "runtime_appearance_authority": True,
        "source_joint_diagnostic": source,
    }
    if joint_status == "JOINT_DIAGNOSTIC_UNOBSERVABLE":
        payload["runtime_override_kind"] = OPERATOR_FORCED_AXIS_ONLY_OVERRIDE_KIND
        payload["camera_from_lidar"] = {
            "axis_contract_id": DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_ID,
            "row_major": _axis_only_row_major(),
            "translation_m": [0.0, 0.0, 0.0],
            "translation_measured": False,
        }
        return payload

    if joint_status != "JOINT_DIAGNOSTIC_PROVISIONAL":
        raise ValueError(f"joint diagnostic status is unsupported for operator override: {joint_status}")

    camera_from_lidar = joint.get("camera_from_lidar")
    if not isinstance(camera_from_lidar, list) or len(camera_from_lidar) != 16:
        raise ValueError("joint diagnostic provisional result must contain a 16-value camera_from_lidar")
    try:
        translation = [float(camera_from_lidar[3]), float(camera_from_lidar[7]), float(camera_from_lidar[11])]
        delta_camera_lidar_ms = float(joint["delta_camera_lidar_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("joint diagnostic provisional result must contain finite delta_camera_lidar_ms") from exc
    fit_capture_ids = diagnostic.get("fit_capture_ids")
    if not isinstance(fit_capture_ids, list) or not fit_capture_ids or any(
        not isinstance(item, str) or not item for item in fit_capture_ids
    ):
        raise ValueError("joint diagnostic provisional result must declare non-empty fit_capture_ids")
    holdout_capture_ids = diagnostic.get("holdout_capture_ids")
    if holdout_capture_ids is not None and (
        not isinstance(holdout_capture_ids, list)
        or any(not isinstance(item, str) or not item for item in holdout_capture_ids)
    ):
        raise ValueError("joint diagnostic provisional result has invalid holdout_capture_ids")
    source["fit_capture_ids"] = fit_capture_ids
    if holdout_capture_ids:
        source["holdout_capture_ids"] = holdout_capture_ids
    payload["runtime_override_kind"] = OPERATOR_FORCED_NUMERICAL_CANDIDATE_OVERRIDE_KIND
    payload["candidate_usable_by_operator_decision"] = True
    payload["quality_limits_bypassed"] = list(joint.get("blockers") or ["SOURCE_DIAGNOSTIC_PROVISIONAL"])
    payload["camera_from_lidar"] = {
        "source": "SOURCE_JOINT_DIAGNOSTIC_CAMERA_FROM_LIDAR",
        "row_major": [float(value) for value in camera_from_lidar],
        "translation_m": translation,
        "translation_measured": False,
        "delta_camera_lidar_ms": delta_camera_lidar_ms,
    }
    return payload


def _atomic_write_verified(output: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_operator_forced_calibration_override(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-id",
        default="PROVISIONAL_AUTOCAL_PENDING_v1",
    )
    parser.add_argument("--operator-acknowledged", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if not args.operator_acknowledged:
        parser.error("--operator-acknowledged is required to create this formal runtime override")

    diagnostic_path = args.joint_diagnostic.resolve()
    output = args.output.resolve()
    if diagnostic_path.parent != output.parent:
        parser.error("--output must be in the same directory as --joint-diagnostic")
    if output.exists() and not args.replace:
        parser.error(f"output already exists: {output}; use --replace to replace it")
    try:
        diagnostic, diagnostic_raw = _load_joint_diagnostic(diagnostic_path)
        payload = _build_payload(diagnostic_path, diagnostic, diagnostic_raw, args.calibration_id)
        _atomic_write_verified(output, payload)
    except (OSError, ValueError, OperatorForcedCalibrationOverrideError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "calibration_id": args.calibration_id,
                "measurement_authority": False,
                "output": str(output),
                "runtime_authority": "OPERATOR_FORCED_FORMAL",
                "runtime_override_kind": payload.get("runtime_override_kind"),
                "source_joint_status": payload["source_joint_diagnostic"]["status"],
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
