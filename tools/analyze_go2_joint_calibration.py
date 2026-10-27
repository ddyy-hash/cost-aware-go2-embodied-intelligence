"""Write one non-authoritative joint calibration diagnostic from raw captures."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import cv2
import numpy as np

from mapping.pc.go2_mapping_calibration.analyzer import analyze_capture
from mapping.pc.go2_mapping_calibration.axis_contract import (
    diagnostic_camera_from_lidar_seed,
)
from mapping.pc.go2_mapping_calibration.joint import (
    joint_capture_from_analysis,
    solve_joint_diagnostic,
)
from mapping.pc.go2_mapping_calibration.solver import CalibrationSolver
from mapping.pc.go2_mapping_calibration.staging import stage_capture_for_diagnostic


def _capture_payload(capture_id: str, analysis) -> dict[str, object]:
    return {
        "capture_id": capture_id,
        "artifact_sha256": analysis.artifact_sha256,
        "artifact_bytes": analysis.inventory.artifact_bytes,
        "part_count": analysis.inventory.part_count,
        "duration_s": analysis.inventory.duration_s,
        "message_counts": dict(analysis.inventory.message_counts),
        "accepted_camera_board_observations": analysis.accepted_board_detections,
        "accepted_lidar_board_observations": analysis.accepted_lidar_planes,
        "lio_epoch_count": analysis.motion.lio_epoch_count,
        "robust_rotation_span_deg": analysis.motion.calibration_rotation_span_deg,
        "robust_translation_span_m": analysis.motion.calibration_translation_span_m,
        "angular_excitation_duration_s": analysis.motion.angular_speed_duration_s,
        "factory_serial_verified": (
            None
            if analysis.inventory.runtime_manifest is None
            else analysis.inventory.runtime_manifest.robot_factory_serial_verified
        ),
        "blockers": list(analysis.blockers),
        "lidar_rejection_reasons": list(analysis.lidar_rejection_reasons),
    }


def _holdout_diagnostic(capture, joint) -> dict[str, object]:
    """Score a named holdout without letting it change the diagnostic fit."""
    transform = joint.camera_from_lidar
    if transform is None or joint.delta_camera_lidar_ms is None:
        return {
            "capture_id": capture.capture_id,
            "status": "NOT_EVALUATED_NO_PROVISIONAL_TRANSFORM",
            "calibration_authority": False,
        }
    rotation_vector, _ = cv2.Rodrigues(np.asarray(transform, dtype=np.float64)[:3, :3])
    parameters = np.concatenate(
        (
            rotation_vector.reshape(3),
            np.asarray(transform, dtype=np.float64)[:3, 3],
            [float(joint.delta_camera_lidar_ms) / 1000.0],
        )
    )
    residuals = CalibrationSolver()._residual(
        parameters,
        capture.camera_planes,
        capture.lidar_planes,
        maximum_interpolation_span_s=5.0,
    )
    fallback = residuals.shape == (7,) and bool(np.all(residuals == 1.0e6))
    return {
        "capture_id": capture.capture_id,
        "status": "OPERATOR_RELAXED_HOLDOUT_DIAGNOSTIC",
        "matched_plane_pairs": 0 if fallback else int(len(residuals) // 4),
        "normalized_residual_p95": None
        if fallback
        else float(np.percentile(np.abs(residuals), 95)),
        "calibration_authority": False,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _analyze_capture_task(
    task: tuple[str, Path],
    opencv_threads: int,
    allow_incomplete: bool = False,
    operator_relaxed: bool = False,
    max_lidar_associations: int | None = None,
) -> tuple[dict[str, object], object]:
    """Analyze one independent capture in a bounded worker process."""

    capture_id, path = task
    cv2.setNumThreads(1)
    analysis = analyze_capture(
        str(path),
        initial_camera_from_lidar=diagnostic_camera_from_lidar_seed(),
        materialization_workers=1,
        target_detection_workers=max(1, int(opencv_threads)),
        allow_incomplete=allow_incomplete,
        operator_relaxed_lidar_planes=operator_relaxed,
        max_lidar_associations=max_lidar_associations,
    )
    return _capture_payload(capture_id, analysis), joint_capture_from_analysis(
        capture_id, analysis
    )


def _analyze_captures(
    tasks: Iterable[tuple[str, Path]],
    *,
    workers: int,
    cpu_count: int | None = None,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    worker: Callable[[tuple[str, Path], int], object] = _analyze_capture_task,
    allow_incomplete: bool = False,
    operator_relaxed: bool = False,
    max_lidar_associations: int | None = None,
) -> list[object]:
    """Run independent captures concurrently without oversubscribing OpenCV."""

    ordered_tasks = tuple(tasks)
    if not ordered_tasks:
        return []
    available_cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    bounded_workers = min(max(1, int(workers)), available_cpus, len(ordered_tasks))
    threads_per_worker = max(1, available_cpus // bounded_workers)
    with executor_factory(max_workers=bounded_workers) as executor:
        return list(
            executor.map(
                worker,
                ordered_tasks,
                repeat(threads_per_worker, len(ordered_tasks)),
                repeat(bool(allow_incomplete), len(ordered_tasks)),
                repeat(bool(operator_relaxed), len(ordered_tasks)),
                repeat(max_lidar_associations, len(ordered_tasks)),
            )
        )


def _stage_capture_tasks(
    tasks: Iterable[tuple[str, Path]],
    *,
    staging_root: Path,
    workers: int,
    stager: Callable[..., object] = stage_capture_for_diagnostic,
) -> tuple[list[tuple[str, Path]], dict[str, object]]:
    """Build complete, read-only uncompressed inputs for one diagnostic run."""

    staged_tasks: list[tuple[str, Path]] = []
    captures: list[dict[str, object]] = []
    for capture_id, source_path in tasks:
        staged = stager(source_path, staging_root, workers=workers)
        staged_tasks.append((capture_id, staged.staged_path))
        captures.append(
            {
                "capture_id": capture_id,
                "source_path": str(staged.source_path),
                "staged_path": str(staged.staged_path),
                "source_parts_sha256": staged.source_parts_sha256,
                "source_part_count": len(staged.parts),
                "manifest_path": str(staged.manifest_path),
            }
        )
    return staged_tasks, {
        "kind": "READ_ONLY_UNCOMPRESSED_DIAGNOSTIC_STAGING",
        "staging_root": str(staging_root),
        "captures": captures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="independent capture processes (default: up to four)",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        help="new disposable directory for read-only uncompressed diagnostic inputs",
    )
    parser.add_argument(
        "--staging-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="bounded bag materialization processes when --staging-root is set",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="diagnostic-only: retain usable window evidence despite non-consumed streams",
    )
    parser.add_argument(
        "--operator-relaxed",
        action="store_true",
        help="operator-approved diagnostic bypass for LiDAR plane and observability gates",
    )
    parser.add_argument(
        "--max-lidar-associations",
        type=int,
        help="maximum evenly distributed RGB board observations per capture for LiDAR association",
    )
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.staging_workers <= 0:
        parser.error("--staging-workers must be positive")
    if args.max_lidar_associations is not None and args.max_lidar_associations <= 0:
        parser.error("--max-lidar-associations must be positive")
    tasks: list[tuple[str, Path]] = []
    for value in args.capture:
        capture_id, separator, path = value.partition("=")
        if not separator or not capture_id or not path:
            parser.error("--capture must use ID=PATH")
        tasks.append((capture_id, Path(path)))

    staging_report = None
    analysis_tasks = tasks
    if args.staging_root is not None:
        analysis_tasks, staging_report = _stage_capture_tasks(
            tasks,
            staging_root=args.staging_root,
            workers=args.staging_workers,
        )
    reports_and_captures = _analyze_captures(
        analysis_tasks,
        workers=args.workers,
        allow_incomplete=args.allow_incomplete,
        operator_relaxed=args.operator_relaxed,
        max_lidar_associations=args.max_lidar_associations,
    )
    capture_reports = [item[0] for item in reports_and_captures]
    captures = [item[1] for item in reports_and_captures]
    fit_pairs = [
        (report, capture)
        for report, capture in zip(capture_reports, captures)
        if report["capture_id"] != "holdout"
    ]
    holdout_capture_ids = [
        str(report["capture_id"])
        for report in capture_reports
        if report["capture_id"] == "holdout"
    ]

    joint = solve_joint_diagnostic(
        [capture for _, capture in fit_pairs], operator_relaxed=args.operator_relaxed
    )
    report = {
        "schema_version": 1,
        "analysis_kind": "JOINT_DIAGNOSTIC_ONLY",
        "evidence_scope": "OPERATOR_ACCEPTED_PARTIAL",
        "provisional": True,
        "allow_incomplete": bool(args.allow_incomplete),
        "operator_relaxed": bool(args.operator_relaxed),
        "max_lidar_associations": args.max_lidar_associations,
        "calibration_authority": False,
        "manifest_written": False,
        "captures": capture_reports,
        "fit_capture_ids": [str(report["capture_id"]) for report, _ in fit_pairs],
        "holdout_capture_ids": holdout_capture_ids,
        "holdout_diagnostics": [
            _holdout_diagnostic(capture, joint)
            for report, capture in zip(capture_reports, captures)
            if report["capture_id"] == "holdout"
        ],
        "input_staging": staging_report,
        "joint": joint.as_dict(),
    }
    _atomic_json(args.output, report)
    print(
        f"JOINT_CALIBRATION_DIAGNOSTIC status={joint.status} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
