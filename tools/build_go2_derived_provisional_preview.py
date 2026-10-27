#!/usr/bin/env python3
"""Build a read-only, uncalibrated RGB-D color preview for the active LiDAR surface.

This inspection command replays only the raw source artifacts that are already
bound to the active formal LiDAR mesh.  It writes a separate content-addressed
vertex-color sidecar, never a final texture, GLB, map pointer, or mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_appearance.derived_provisional_workflow import (
    build_active_provisional_surface_preview,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=ROOT / "mapping-data" / "pc-stack" / "automatic-reconstruction-current.json",
        help="immutable active automatic-reconstruction pointer to validate",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "reconstruction" / "derived",
        help="formal derived-surface root containing the active mesh generation",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=ROOT / "mapping-data" / "inbox" / "lidar-imu-rgbd",
        help="immutable RGB-D/LiDAR capture root; the active session subdirectory is selected automatically",
    )
    parser.add_argument(
        "--operator-override-path",
        type=Path,
        default=(
            ROOT
            / "mapping-data"
            / "calibration-work"
            / "20260810"
            / "formal-operator-runtime-override-20260810T131705Z.json"
        ),
        help="formal operator-forced runtime projection only; it is not measurement calibration",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT / "output" / "reconstruction" / "derived-provisional-appearance",
        help="separate content-addressed preview store; never a formal surface or appearance store",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="optional WSL-local temporary directory for one decompressed ROS SQLite part at a time",
    )
    parser.add_argument(
        "--max-pair-delta-ms",
        type=float,
        default=100.0,
        help="maximum recorded timestamp association distance for LiDAR/LIO/RGB (default: 100 ms)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_pair_delta_ms <= 0.0:
        raise SystemExit("--max-pair-delta-ms must be positive")
    max_pair_delta_ns = round(args.max_pair_delta_ms * 1_000_000.0)
    if max_pair_delta_ns <= 0:
        raise SystemExit("--max-pair-delta-ms is too small")
    try:
        result = build_active_provisional_surface_preview(
            status_path=args.status_path,
            output_root=args.output_root,
            capture_root=args.capture_root,
            operator_override_path=args.operator_override_path,
            preview_root=args.preview_root,
            max_pair_delta_ns=max_pair_delta_ns,
            work_root=args.work_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "state": "PROVISIONAL_VERTEX_COLOR_PREVIEW_ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "delivery_authorized": False,
                    "measurement_authority": False,
                    "final_store_eligible": False,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
