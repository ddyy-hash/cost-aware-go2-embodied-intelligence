#!/usr/bin/env python3
"""Run one persistent, session-scoped verified LiDAR reconstruction worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Direct execution makes ``tools/`` the first import path.  Add the project
# root explicitly so this supported CLI works outside ``python -m`` as well.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from mapping.pc.go2_mapping_reconstruction.session_worker import SessionReconstructionWorker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--inbox-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--sensor-frame-transform-id", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-artifacts-per-pass", type=int, default=1)
    parser.add_argument("--capture-label", default="lidar-imu-rgbd")
    parser.add_argument(
        "--finalize-verified-session",
        action="store_true",
        help="run one full replay only after the final manifest and all lifecycle records verify",
    )
    parser.add_argument(
        "--recorded-lio-poses-only",
        action="store_true",
        help="finalize the verified session using recorded LIO poses without global loop optimization",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--source-mode", choices=("RAW_LIDAR_WITH_RECORDED_LIO", "RECORDED_LIO_REGISTERED_CLOUD", "RECORDED_LIO_SESSION_ODOM_CLOUD"), default="RAW_LIDAR_WITH_RECORDED_LIO")
    parser.add_argument("--incremental-publish-interval-artifacts", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    if args.max_artifacts_per_pass < 1:
        raise SystemExit("--max-artifacts-per-pass must be at least 1")
    def report_progress(event: dict[str, object]) -> None:
        print(json.dumps({"state": "RECONSTRUCTION_PROGRESS", "session_id": args.session_id, **event}, ensure_ascii=True, sort_keys=True), flush=True)

    worker = SessionReconstructionWorker(
        session_id=args.session_id,
        inbox_root=args.inbox_root,
        output_root=args.output_root,
        status_path=args.status_path,
        sensor_frame_transform_id=args.sensor_frame_transform_id,
        capture_label=args.capture_label,
        max_artifacts_per_pass=args.max_artifacts_per_pass,
        final_global_optimization=not args.recorded_lio_poses_only,
        source_mode=args.source_mode,
        incremental_publish_interval_artifacts=args.incremental_publish_interval_artifacts,
        progress_callback=report_progress,
    )
    if args.finalize_verified_session:
        try:
            result = worker.finalize_verified_session()
            print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
            return 0
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "state": "AUTOMATIC_RECONSTRUCTION_ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "session_id": args.session_id,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 1
    while True:
        try:
            result = worker.process_once()
            print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "state": "AUTOMATIC_RECONSTRUCTION_ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "session_id": args.session_id,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
