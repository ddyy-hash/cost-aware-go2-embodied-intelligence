"""Record caller-selected ROS 2 topics into sealed MCAP segments.

The recorder does not start cameras, publish motion messages, or change ROS
parameters. Sealed files are handed to :mod:`go2_mapping_capture.retention`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .retention import RetentionPolicy, SegmentStore


GIB = 1024 * 1024 * 1024


def build_rosbag_command(output_uri: Path, topics: Sequence[str]) -> List[str]:
    """Return the only recording invocation allowed by the mapping profile."""
    if not topics:
        raise ValueError("at least one explicit ROS topic is required")
    if any(not topic.startswith("/") for topic in topics):
        raise ValueError("ROS topics must be absolute names")
    return [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--compression-mode",
        "file",
        "--compression-format",
        "zstd",
        "--output",
        str(output_uri),
        *topics,
    ]


def record_one_segment(
    store: SegmentStore,
    segment_id: str,
    topics: Sequence[str],
    duration_seconds: int,
) -> str:
    """Record one bounded segment and atomically commit its MCAP payload.

    A nonzero recorder result, missing output, or multiple MCAP files is an
    error. The unfinished ROS bag output stays under ``staging`` and is later
    quarantined by ``recover_interrupted_writes``.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    store.initialize()
    store.recover_interrupted_writes()
    store.enforce_retention()

    output_uri = store.staging_dir / f"{segment_id}.rosbag.partial"
    command = build_rosbag_command(output_uri, topics)
    process = subprocess.Popen(command)
    try:
        process.wait(timeout=duration_seconds)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
    if process.returncode != 0:
        raise RuntimeError(f"ros2 bag record failed with exit code {process.returncode}")

    mcap_files = sorted(output_uri.rglob("*.mcap"))
    if len(mcap_files) != 1:
        raise RuntimeError(f"expected exactly one MCAP file in {output_uri}, found {len(mcap_files)}")
    staged_payload = store.staging_dir / f"{segment_id}.mcap.partial"
    _copy_with_fsync(mcap_files[0], staged_payload)
    shutil.rmtree(output_uri)
    store.commit_file(segment_id, staged_payload, acknowledged=False)
    store.enforce_retention()
    return segment_id


def default_policy() -> RetentionPolicy:
    return RetentionPolicy(
        raw_limit_bytes=28 * GIB,
        spool_limit_bytes=8 * GIB,
        recovery_headroom_bytes=4 * GIB,
        max_segment_bytes=2 * GIB,
    )


def _copy_with_fsync(source: Path, target: Path) -> None:
    with source.open("rb") as input_file, target.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one acknowledgement-safe Go2 mapping MCAP segment")
    parser.add_argument("--root", type=Path, default=Path("/home/unitree/go2-mapping-data"))
    parser.add_argument("--segment-id", default=str(time.time_ns()))
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--topic", action="append", dest="topics", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    store = SegmentStore(args.root, default_policy())
    record_one_segment(store, args.segment_id, args.topics, args.duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
