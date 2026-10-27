#!/usr/bin/env python3
"""Replay a closed Go2 calibration capture into the read-only stair qualifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_calibration.bag import CalibrationCapture, CalibrationTopicMap  # noqa: E402
from mapping.pc.go2_mapping_dashboard.stair_survey import (  # noqa: E402
    StairSurveyStore,
    replay_stair_capture,
)


def _json_file(path: Path, label: str):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, help="closed .db3/.db3.zstd file or capture directory")
    parser.add_argument("--transform", required=True, type=Path, help="JSON containing lidar_to_gravity as a 4x4 matrix")
    parser.add_argument("--output", required=True, type=Path, help="independent stair report JSON output path")
    parser.add_argument("--capture-id", required=True, help="operator-assigned source capture identity")
    parser.add_argument("--frame-id", default="base_link", help="gravity-aligned profile frame identity")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--low-state-timestamps", type=Path, help="JSON array of recorded low-state timestamp nanoseconds")
    parser.add_argument("--evidence", type=Path, help="JSON object of static/landing/obstruction evidence flags")
    parser.add_argument("--maximum-sensor-gap-ns", type=int, default=100_000_000)
    parser.add_argument("--max-frames", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        transform_payload = _json_file(args.transform, "transform")
        matrix = transform_payload.get("lidar_to_gravity") if isinstance(transform_payload, dict) else None
        if matrix is None:
            raise ValueError("transform must contain lidar_to_gravity")
        if not isinstance(transform_payload, dict):
            raise ValueError("transform must be an object")
        transform_identity = {
            "transform_id": transform_payload.get("transform_id"),
            "coordinate_frame": transform_payload.get("coordinate_frame"),
            "measurement_authority": transform_payload.get("measurement_authority"),
        }
        low_states = ()
        if args.low_state_timestamps:
            low_states = _json_file(args.low_state_timestamps, "low-state timestamps")
            if not isinstance(low_states, list):
                raise ValueError("low-state timestamps must be a JSON array")
        evidence = {}
        if args.evidence:
            evidence = _json_file(args.evidence, "stair evidence")
            if not isinstance(evidence, dict):
                raise ValueError("stair evidence must be a JSON object")
        source_identity = {
            "lidar_topic": args.lidar_topic,
            "frame_id": args.frame_id,
            "capture_id": args.capture_id,
        }
        defaults = CalibrationTopicMap.defaults()
        topic_map = CalibrationTopicMap(
            lidar=args.lidar_topic,
            imu=defaults.imu,
            lio_pose=defaults.lio_pose,
            color=defaults.color,
            depth=defaults.depth,
            color_camera_info=defaults.color_camera_info,
            depth_camera_info=defaults.depth_camera_info,
            tf=defaults.tf,
            tf_static=defaults.tf_static,
        )
        with CalibrationCapture(args.capture, topics=topic_map) as capture:
            inventory = capture.inventory()
            source_identity["artifact_sha256"] = inventory.artifact_sha256
            report = replay_stair_capture(
                capture,
                source_identity=source_identity,
                lidar_to_gravity=matrix,
                transform_identity=transform_identity,
                low_state_timestamps_ns=low_states,
                evidence=evidence,
                maximum_sensor_gap_ns=args.maximum_sensor_gap_ns,
                max_frames=args.max_frames,
            )
        output = StairSurveyStore(args.output)
        output.save(report)
    except (OSError, TypeError, ValueError) as exc:
        print(f"replay_go2_stair_survey: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"state": report["state"], "output": str(args.output.resolve()), "replay": report["replay"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
