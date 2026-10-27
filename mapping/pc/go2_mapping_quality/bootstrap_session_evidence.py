"""Build the immutable raw-capture portion of a session evidence document."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

import zstandard


REQUIRED_TOPICS = {
    "lidar": {"/livox/lidar"},
    "imu": {"/dog_imu_raw"},
    "lio_pose": {"/go2/lio/pose"},
    "rgb": {"/camera/color/image_raw"},
    "depth": {"/camera/depth/image_rect_raw"},
    "camera_info": {"/camera/color/camera_info", "/camera/depth/camera_info"},
    "tf": {"/tf"},
    "tf_static": {"/tf_static"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _topics_in_bag(path: Path) -> set[str]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
            return {str(row[0]) for row in database.execute("SELECT name FROM topics")}
    except sqlite3.Error as exc:
        raise ValueError(f"cannot read verified ROS bag {path}: {exc}") from exc


def _verified_bags(root: Path) -> Iterable[Path]:
    for path in sorted([*root.rglob("*.db3"), *root.rglob("*.db3.zstd")]):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not any(part.startswith(".") for part in relative.parts):
            yield path


@contextmanager
def _sqlite_bag_path(path: Path) -> Iterable[Path]:
    if path.suffix != ".zstd":
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="go2-evidence-") as temporary_directory:
        sqlite_bag = Path(temporary_directory) / "artifact.db3"
        with path.open("rb") as source, sqlite_bag.open("wb") as destination:
            zstandard.ZstdDecompressor().copy_stream(source, destination)
        yield sqlite_bag


def build_evidence(inbox_root: Path, calibration_id: str) -> dict:
    inbox_root = Path(inbox_root)
    if not calibration_id.strip():
        raise ValueError("calibration_id must not be empty")
    if not inbox_root.is_dir():
        raise ValueError(f"verified inbox does not exist: {inbox_root}")
    bags = list(_verified_bags(inbox_root))
    if not bags:
        raise ValueError(f"no verified .db3 artifacts found under {inbox_root}")
    observed_topics: set[str] = set()
    artifacts = []
    for bag in bags:
        with _sqlite_bag_path(bag) as sqlite_bag:
            observed_topics.update(_topics_in_bag(sqlite_bag))
        artifacts.append({
            "relative_path": bag.relative_to(inbox_root).as_posix(),
            "sha256": _sha256(bag),
            "bytes": bag.stat().st_size,
        })
    streams = sorted(
        stream for stream, topic_names in REQUIRED_TOPICS.items() if topic_names.issubset(observed_topics)
    )
    return {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "raw_capture": {
            "sha256_verified": True,
            "artifact_count": len(artifacts),
            "artifact_sha256": [artifact["sha256"] for artifact in artifacts],
            "artifact_paths": artifacts,
            "required_streams": streams,
            "observed_topics": sorted(observed_topics),
        },
        "geometry_validation": {
            "independent_measurement_count": 0,
            "independent_measurements_used_for_calibration": False,
            "p95_dimension_error_mm": None,
            "maximum_dimension_error_mm": None,
            "accepted_loop_closures": 0,
            "loop_point_to_plane_p95_mm": None,
        },
        "bim_registration": {
            "independent_structural_groups": 0,
            "holdout_structural_groups": 0,
            "holdout_p95_residual_mm": None,
            "rigid_se3_only": True,
            "design_bim_hash": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Go2 session evidence from verified ROS bag artifacts")
    parser.add_argument("--inbox-root", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_evidence(args.inbox_root, args.calibration_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
