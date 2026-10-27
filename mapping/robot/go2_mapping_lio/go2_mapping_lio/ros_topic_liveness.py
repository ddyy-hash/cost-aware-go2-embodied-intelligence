#!/usr/bin/env python3
"""Verify that a ROS 2 topic delivers a sample with an explicit QoS profile."""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

from lio_clock_alignment import HeaderProgressGate


def positive_seconds(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return value


def positive_milliseconds(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("milliseconds must be positive")
    return value


def header_timestamp_ns(message: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    sec = getattr(stamp, "sec", None) if stamp is not None else None
    nanosec = getattr(stamp, "nanosec", None) if stamp is not None else None
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        return 0
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        return 0
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        return 0
    return sec * 1_000_000_000 + nanosec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--message-type", required=True, help="ROS type, e.g. sensor_msgs/msg/PointCloud2")
    parser.add_argument("--timeout", type=positive_seconds, default=6.0)
    parser.add_argument("--require-frame-id", default="")
    parser.add_argument("--minimum-header-progress-ms", type=positive_milliseconds)
    parser.add_argument(
        "--require-nonempty-data",
        action="store_true",
        help="require a non-empty payload; Image messages must also have valid dimensions and row data",
    )
    args = parser.parse_args()

    try:
        message_class = get_message(args.message_type)
    except (AttributeError, ModuleNotFoundError, ValueError) as exc:
        parser.error("invalid ROS message type %r: %s" % (args.message_type, exc))

    rclpy.init(args=None)
    node = rclpy.create_node("go2_mapping_topic_liveness_probe")
    samples = 0
    progress_gate = (
        HeaderProgressGate(minimum_progress_ms=args.minimum_header_progress_ms)
        if args.minimum_header_progress_ms is not None
        else None
    )

    def on_message(message: object) -> None:
        nonlocal samples
        if args.require_frame_id:
            header = getattr(message, "header", None)
            frame_id = getattr(header, "frame_id", "") if header is not None else ""
            if frame_id != args.require_frame_id:
                return
        if progress_gate is not None and not progress_gate.observe(header_timestamp_ns(message)):
            return
        if args.require_nonempty_data:
            data = getattr(message, "data", None)
            try:
                data_length = len(data) if data is not None else 0
            except TypeError:
                data_length = 0
            if data_length <= 0:
                return
            if args.message_type == "sensor_msgs/msg/Image":
                width = getattr(message, "width", 0)
                height = getattr(message, "height", 0)
                step = getattr(message, "step", 0)
                if not all(isinstance(value, int) and value > 0 for value in (width, height, step)):
                    return
                if data_length < step * height:
                    return
        samples += 1

    subscription = node.create_subscription(
        message_class,
        args.topic,
        on_message,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
    )
    deadline = time.monotonic() + args.timeout
    try:
        while samples == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.2, max(0.0, deadline - time.monotonic())))
    finally:
        subscription.destroy()
        node.destroy_node()
        rclpy.shutdown()

    if samples:
        suffix = " frame_id=%s" % args.require_frame_id if args.require_frame_id else ""
        if args.minimum_header_progress_ms is not None:
            print(
                "LIVE_ADVANCING_HEADER topic=%s type=%s samples=%d minimum_progress_ms=%.3f%s"
                % (args.topic, args.message_type, samples, args.minimum_header_progress_ms, suffix)
            )
            return 0
        marker = "LIVE_PAYLOAD" if args.require_nonempty_data else "LIVE_SAMPLE"
        print("%s topic=%s type=%s samples=%d%s" % (marker, args.topic, args.message_type, samples, suffix))
        return 0
    suffix = " required_frame_id=%s" % args.require_frame_id if args.require_frame_id else ""
    if args.minimum_header_progress_ms is not None:
        print(
            "NO_ADVANCING_HEADER topic=%s type=%s timeout=%.1f minimum_progress_ms=%.3f%s"
            % (args.topic, args.message_type, args.timeout, args.minimum_header_progress_ms, suffix)
        )
        return 1
    print("NO_LIVE_SAMPLE topic=%s type=%s timeout=%.1f%s" % (args.topic, args.message_type, args.timeout, suffix))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
