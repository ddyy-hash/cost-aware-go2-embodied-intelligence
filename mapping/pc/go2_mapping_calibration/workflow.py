"""Fail-closed orchestration for four-capture calibration evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from .bag import BagEvidenceError, CalibrationCapture, CalibrationTopicMap
from .analyzer import CaptureAnalysis, analyze_capture
from .estimation import CalibrationEstimationError, solve_three_estimation_captures
from .analysis import CalibrationAnalysisError, resolve_frame_transform
from .axis_contract import diagnostic_camera_from_lidar_seed
from .holdout import evaluate_holdout_capture
from .lidar_imu import LidarImuEvidenceError, load_lidar_imu_evidence
from .manifest import build_calibration_manifest
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


CALIBRATION_BLOCKED_EXIT_CODE = 7


@dataclass(frozen=True)
class CalibrationWorkflowResult:
    accepted: bool
    exit_code: int
    blockers: tuple[str, ...]
    diagnostics_path: Path
    manifest_path: Path | None
    estimation: object | None = None


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clock_fit_payload(fit: object) -> dict[str, object]:
    fields = (
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
    )
    return {field: getattr(fit, field) for field in fields}


def _solver_implementation_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "solver.py",
        "estimation.py",
        "workflow.py",
        "analyzer.py",
        "analysis.py",
        "capture_analysis.py",
        "target.py",
        "axis_contract.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _holdout_report(metrics: object) -> dict[str, object]:
    return {
        "accepted": bool(getattr(metrics, "accepted")),
        "accepted_board_poses": int(getattr(metrics, "accepted_board_poses")),
        "correspondences_per_pose": list(getattr(metrics, "correspondences_per_pose")),
        "total_correspondences": int(getattr(metrics, "total_correspondences")),
        "validated_range_min_m": getattr(metrics, "validated_range_min_m"),
        "validated_range_max_m": getattr(metrics, "validated_range_max_m"),
        "camera_reprojection_median_px": getattr(metrics, "camera_reprojection_median_px"),
        "camera_reprojection_p95_px": getattr(metrics, "camera_reprojection_p95_px"),
        "depth_median_absolute_residual_m": getattr(metrics, "depth_median_absolute_residual_m"),
        "depth_p95_absolute_residual_m": getattr(metrics, "depth_p95_absolute_residual_m"),
        "absolute_mean_signed_bias_m": getattr(metrics, "absolute_mean_signed_bias_m"),
        "occlusion_conflict_fraction": getattr(metrics, "occlusion_conflict_fraction"),
        "free_space_violation_fraction": getattr(metrics, "free_space_violation_fraction"),
        "t_d_c_validated": bool(getattr(metrics, "t_d_c_validated")),
        "blockers": list(getattr(metrics, "blockers")),
    }


def run_calibration_workflow(
    *,
    estimation_paths: Sequence[str | Path],
    holdout_path: str | Path,
    lidar_imu_evidence_path: str | Path | None = None,
    output_manifest: str | Path,
    output_diagnostics: str | Path,
    robot_serial: str,
    camera_serial: str,
    topics: CalibrationTopicMap | None = None,
) -> CalibrationWorkflowResult:
    manifest_path = Path(output_manifest)
    diagnostics_path = Path(output_diagnostics)
    blockers: list[str] = []
    capture_reports: list[dict] = []
    if len(estimation_paths) != 3:
        blockers.append("THREE_ESTIMATION_CAPTURES_REQUIRED")
    paths = [Path(value) for value in estimation_paths] + [Path(holdout_path)]
    digests: list[str] = []
    for index, path in enumerate(paths):
        slot = f"estimation_{index + 1}" if index < len(estimation_paths) else "holdout"
        try:
            with CalibrationCapture(path, topics) as bag:
                inventory = bag.inventory()
                digests.append(inventory.artifact_sha256)
                capture_reports.append({"slot": slot, **inventory.as_dict()})
                blockers.extend(
                    f"MISSING_REQUIRED_STREAM:{slot}:{stream}"
                    for stream in inventory.missing_streams
                )
                blockers.extend(
                    f"MISSING_CAPTURE_EVIDENCE:{slot}:{evidence}"
                    for evidence in inventory.missing_evidence
                )
                runtime = inventory.runtime_manifest
                if runtime is not None:
                    if runtime.robot_serial != robot_serial:
                        blockers.append(f"ROBOT_SERIAL_MISMATCH:{slot}")
                    if runtime.camera_serial != camera_serial:
                        blockers.append(f"CAMERA_SERIAL_MISMATCH:{slot}")
                    if not runtime.robot_factory_serial_verified:
                        blockers.append(f"ROBOT_FACTORY_SERIAL_NOT_VERIFIED:{slot}")
        except BagEvidenceError as exc:
            blockers.append(f"CAPTURE_UNREADABLE:{slot}:{exc}")
            capture_reports.append({"slot": slot, "error": str(exc)})
    if len(digests) != len(set(digests)):
        blockers.append("CAPTURE_ARTIFACTS_NOT_UNIQUE")
    if not robot_serial.strip():
        blockers.append("ROBOT_SERIAL_REQUIRED")
    if not camera_serial.strip():
        blockers.append("CAMERA_SERIAL_REQUIRED")

    analyses: list[CaptureAnalysis] = []
    if not blockers:
        for index, path in enumerate(paths):
            slot = f"estimation_{index + 1}" if index < len(estimation_paths) else "holdout"
            try:
                analysis = analyze_capture(
                    str(path),
                    topics=topics,
                    initial_camera_from_lidar=diagnostic_camera_from_lidar_seed(),
                )
                analyses.append(analysis)
                capture_reports[index]["analysis"] = {
                    "accepted": analysis.accepted,
                    "blockers": list(analysis.blockers),
                    "clock_fits": {stream: fit.__dict__ for stream, fit in analysis.clock_fits.items()},
                    "board_detections": analysis.board_detections,
                    "accepted_board_detections": analysis.accepted_board_detections,
                    "accepted_lidar_planes": analysis.accepted_lidar_planes,
                    "lidar_rejection_reasons": list(
                        getattr(analysis, "lidar_rejection_reasons", ())
                    ),
                    "motion": {
                        "cumulative_rotation_deg": analysis.motion.cumulative_rotation_deg,
                        "rotation_axes": analysis.motion.rotation_axes,
                        "translation_m": analysis.motion.translation_m,
                        "angular_speed_duration_s": analysis.motion.angular_speed_duration_s,
                        "lio_epoch_count": int(getattr(analysis.motion, "lio_epoch_count", 1)),
                        "robust_translation_span_m": getattr(
                            analysis.motion, "robust_translation_span_m", None
                        ),
                        "robust_rotation_span_deg": getattr(
                            analysis.motion, "robust_rotation_span_deg", None
                        ),
                        "reset_reasons": list(getattr(analysis.motion, "reset_reasons", ())),
                    },
                }
                blockers.extend(f"{slot}:{reason}" for reason in analysis.blockers)
            except (BagEvidenceError, ValueError) as exc:
                blockers.append(f"CAPTURE_ANALYSIS_FAILED:{slot}:{exc}")
    lidar_imu = None
    if lidar_imu_evidence_path is None:
        blockers.append("LIDAR_IMU_INDEPENDENT_EVIDENCE_PENDING")
    else:
        try:
            lidar_imu = load_lidar_imu_evidence(
                lidar_imu_evidence_path,
                expected_robot_serial=robot_serial,
            )
            blockers.extend(lidar_imu.blockers)
            if tuple(run.source_capture_sha256 for run in lidar_imu.runs) != tuple(digests[:3]):
                blockers.append("LIDAR_IMU_SOURCE_CAPTURES_DO_NOT_MATCH_ESTIMATION_CAPTURES")
        except LidarImuEvidenceError as exc:
            blockers.append(f"LIDAR_IMU_EVIDENCE_INVALID:{exc}")

    if analyses:
        color_hashes = {
            analysis.color_camera_identity["sha256"]
            for analysis in analyses
            if analysis.color_camera_identity is not None
        }
        depth_hashes = {
            analysis.depth_camera_identity["sha256"]
            for analysis in analyses
            if analysis.depth_camera_identity is not None
        }
        if len(color_hashes) != 1 or any(analysis.color_camera_identity is None for analysis in analyses):
            blockers.append("COLOR_CAMERA_INFO_NOT_IDENTICAL_ACROSS_CAPTURES")
        if len(depth_hashes) != 1 or any(analysis.depth_camera_identity is None for analysis in analyses):
            blockers.append("DEPTH_CAMERA_INFO_NOT_IDENTICAL_ACROSS_CAPTURES")
        lidar_frames = {
            frame_id
            for analysis in analyses
            for _, _, frame_id, _ in analysis.lidar_records
            if frame_id
        }
        if len(lidar_frames) != 1:
            blockers.append("LIDAR_FRAME_ID_NOT_IDENTICAL_ACROSS_CAPTURES")
        elif lidar_imu is not None and next(iter(lidar_frames)) != lidar_imu.lidar_frame_id:
            blockers.append("LIDAR_IMU_FRAME_DOES_NOT_MATCH_CAPTURE")
    estimation = None
    holdout_metrics = None
    t_d_c = None
    if not blockers and len(analyses) == 4:
        try:
            estimation = solve_three_estimation_captures(analyses[:3])
            blockers.extend(estimation.blockers)
            holdout_analysis = analyses[3]
            try:
                static_t_d_c = []
                for analysis in analyses:
                    if analysis.color_camera_info is None or analysis.depth_camera_info is None:
                        raise CalibrationAnalysisError("CameraInfo is incomplete")
                    static_t_d_c.append(resolve_frame_transform(
                        analysis.tf_static_records,
                        target_frame=analysis.depth_camera_info.frame_id,
                        source_frame=analysis.color_camera_info.frame_id,
                    ))
                t_d_c = np.asarray(static_t_d_c[0], dtype=np.float64)
                if any(not np.allclose(t_d_c, item, atol=1e-9) for item in static_t_d_c[1:]):
                    raise CalibrationAnalysisError("T_D_C differs across captures")
            except (CalibrationAnalysisError, ValueError) as exc:
                t_d_c = np.eye(4)
                blockers.append(f"HOLDOUT_T_D_C_STATIC_CHAIN_REQUIRED:{exc}")
            holdout_metrics = evaluate_holdout_capture(
                holdout_analysis,
                t_d_c=t_d_c,
                t_c_l=estimation.camera_from_lidar,
                expected_camera_serial=camera_serial,
                camera_lidar_delta_ms=estimation.delta_camera_lidar_ms,
            )
            blockers.extend(holdout_metrics.blockers)
            capture_reports[3]["holdout_evidence"] = _holdout_report(holdout_metrics)
        except CalibrationEstimationError as exc:
            blockers.append(f"ESTIMATION_FAILED:{exc}")
    elif not blockers:
        blockers.append("CALIBRATION_ANALYSIS_INCOMPLETE")
    blockers = list(dict.fromkeys(blockers))

    if (
        not blockers
        and estimation is not None
        and holdout_metrics is not None
        and lidar_imu is not None
        and t_d_c is not None
    ):
        try:
            holdout_analysis = analyses[3]
            color_identity = holdout_analysis.color_camera_identity
            depth_identity = holdout_analysis.depth_camera_identity
            if color_identity is None or depth_identity is None:
                raise ValueError("CameraInfo identity is incomplete")
            required_fits = {
                stream: _clock_fit_payload(holdout_analysis.clock_fits[stream])
                for stream in ("lidar", "color", "depth")
            }
            t_c_l_row = estimation.camera_from_lidar.reshape(-1).tolist()
            t_d_c_row = np.asarray(t_d_c, dtype=np.float64).reshape(-1).tolist()
            manifest = build_calibration_manifest(
                robot_serial=robot_serial,
                camera_serial=camera_serial,
                estimation_hashes=digests[:3],
                holdout_hash=digests[3],
                solver_version=estimation.solver_version,
                solver_implementation_sha256=_solver_implementation_sha256(),
                t_c_l=t_c_l_row,
                t_d_c=t_d_c_row,
                t_c_l_sha256=canonical_sha256({"row_major": t_c_l_row}),
                t_d_c_sha256=canonical_sha256({"row_major": t_d_c_row, "device_serial": camera_serial}),
                color_camera_info_sha256=color_identity["sha256"],
                depth_camera_info_sha256=depth_identity["sha256"],
                color_camera_info=color_identity["contract"],
                depth_camera_info=depth_identity["contract"],
                lidar_frame_id=lidar_imu.lidar_frame_id,
                color_frame_id=holdout_analysis.color_camera_info.frame_id,
                depth_frame_id=holdout_analysis.depth_camera_info.frame_id,
                clock_fits=required_fits,
                lidar_imu_evidence=lidar_imu.manifest_evidence(),
                repeatability={
                    **lidar_imu.manifest_repeatability(),
                    "lidar_camera_translation_mm": estimation.translation_repeatability_mm,
                    "lidar_camera_rotation_deg": estimation.rotation_repeatability_deg,
                },
                timing={
                    "lidar_imu_repeatability_ms": lidar_imu.manifest_timing(),
                    "lidar_camera_repeatability_ms": estimation.timing_repeatability_ms,
                },
                holdout=holdout_metrics.manifest_payload(digests[3]),
            )
            _atomic_json(manifest_path, manifest)
            report = {
                "schema_version": 1,
                "accepted": True,
                "exit_code": 0,
                "quality_level": "CALIBRATED_DENSE",
                "robot_serial": robot_serial,
                "camera_serial": camera_serial,
                "captures": capture_reports,
                "lidar_imu_evidence": lidar_imu.manifest_evidence(),
                "blockers": [],
                "manifest_written": True,
                "manifest_path": str(manifest_path),
            }
            _atomic_json(diagnostics_path, report)
            return CalibrationWorkflowResult(
                accepted=True,
                exit_code=0,
                blockers=(),
                diagnostics_path=diagnostics_path,
                manifest_path=manifest_path,
                estimation=estimation,
            )
        except (KeyError, OSError, ValueError) as exc:
            blockers.append(f"MANIFEST_BUILD_FAILED:{exc}")

    manifest_path.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "accepted": False,
        "exit_code": CALIBRATION_BLOCKED_EXIT_CODE,
        "quality_level": "PREVIEW_UNCALIBRATED",
        "robot_serial": robot_serial,
        "camera_serial": camera_serial,
        "captures": capture_reports,
        "blockers": blockers,
        "manifest_written": False,
    }
    _atomic_json(diagnostics_path, report)
    return CalibrationWorkflowResult(
        accepted=False,
        exit_code=CALIBRATION_BLOCKED_EXIT_CODE,
        blockers=tuple(blockers),
        diagnostics_path=diagnostics_path,
        manifest_path=None,
        estimation=estimation,
    )
