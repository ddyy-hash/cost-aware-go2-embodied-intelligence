#!/usr/bin/env python3
"""Derive an isolated provisional surface from immutable Go2 capture parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_live.sensor_frame import (  # noqa: E402
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
)
from mapping.pc.go2_mapping_reconstruction.offline import (  # noqa: E402
    PREVIEW_UNCALIBRATED,
    RAW_LIDAR_WITH_RECORDED_LIO,
    RECORDED_LIO_REGISTERED_CLOUD,
    RECORDED_LIO_SESSION_ODOM_CLOUD,
    derive_capture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay closed .db3.zstd parts into an independent, observed-surface "
            f"generation. Output remains {PREVIEW_UNCALIBRATED}."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="capture directory containing closed .db3.zstd parts",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="independent root for the new derived generation",
    )
    parser.add_argument(
        "--sensor-frame-transform-id",
        default=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        help="registered LiDAR sensor-frame transform ID",
    )
    parser.add_argument(
        "--source-mode",
        choices=(
            RAW_LIDAR_WITH_RECORDED_LIO,
            RECORDED_LIO_REGISTERED_CLOUD,
            RECORDED_LIO_SESSION_ODOM_CLOUD,
        ),
        default=RAW_LIDAR_WITH_RECORDED_LIO,
        help=(
            "raw LiDAR plus recorded LIO poses, or the recorded LIO-SAM "
            "project-owned map-frame PointCloud2 stream whose header frame is map, "
            "or one explicit registered-cloud odom local validation session"
        ),
    )
    parser.add_argument(
        "--lidar-topic",
        action="append",
        default=None,
        help="raw LiDAR topic for RAW_LIDAR_WITH_RECORDED_LIO (repeatable)",
    )
    parser.add_argument("--voxel-m", type=float, default=0.02, help="fusion voxel size in metres")
    parser.add_argument(
        "--display-point-budget",
        type=int,
        default=100_000,
        help="maximum bounded display points retained by the mapper",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="optional positive prefix count for a smoke replay",
    )
    parser.add_argument(
        "--require-surface",
        action="store_true",
        help="return failure when the selected parts produce no measured surface",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = derive_capture(
            args.input_dir,
            args.output_root,
            sensor_frame_transform_id=args.sensor_frame_transform_id,
            source_mode=args.source_mode,
            lidar_topics=tuple(args.lidar_topic or ("/go2/lio/lidar", "/livox/lidar")),
            voxel_m=args.voxel_m,
            display_point_budget=args.display_point_budget,
            max_parts=args.max_parts,
        )
        if args.require_surface and result.surface_json_path is None:
            raise ValueError("selected capture parts produced no measured surface")
    except (OSError, ValueError) as exc:
        print(f"derive_go2_reconstruction: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.manifest["status"],
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
                "measured_surface": None if result.surface_json_path is None else str(result.surface_json_path),
                "ply": None if result.ply_path is None else str(result.ply_path),
                "metadata": None if result.metadata_path is None else str(result.metadata_path),
                "derived_cache_key": result.manifest["derived_identity"]["cache_key"],
                "mapping_source_mode": result.manifest["mapping_source_mode"],
                "source_layer": result.manifest["source_layer"],
                "coordinate_scope": result.manifest["coordinate_scope"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
