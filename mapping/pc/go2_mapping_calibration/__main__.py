"""Command line entrypoint for fail-closed Go2 calibration evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import CALIBRATION_BLOCKED_EXIT_CODE, run_calibration_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a verified Go2 camera-LiDAR calibration manifest")
    parser.add_argument("--estimation", dest="estimation", action="append", required=True, type=Path, metavar="BAG_OR_DIR", help="repeat exactly three times")
    parser.add_argument("--holdout", required=True, type=Path, metavar="BAG_OR_DIR")
    parser.add_argument("--lidar-imu-evidence", required=True, type=Path, metavar="JSON")
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-diagnostics", required=True, type=Path)
    parser.add_argument("--robot-serial", required=True)
    parser.add_argument("--camera-serial", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_calibration_workflow(
        estimation_paths=args.estimation,
        holdout_path=args.holdout,
        lidar_imu_evidence_path=args.lidar_imu_evidence,
        output_manifest=args.output_manifest,
        output_diagnostics=args.output_diagnostics,
        robot_serial=args.robot_serial,
        camera_serial=args.camera_serial,
    )
    print(f"CALIBRATION accepted={str(result.accepted).lower()} exit_code={result.exit_code} diagnostics={result.diagnostics_path}")
    if result.blockers:
        print("BLOCKERS " + " | ".join(result.blockers))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
