import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROBOT_PACKAGE = ROOT / "mapping" / "robot" / "go2_mapping_capture"
sys.path.insert(0, str(ROBOT_PACKAGE))

from go2_mapping_capture.retention import (  # noqa: E402
    CapacityError,
    IntegrityError,
    RetentionPolicy,
    SegmentStore,
)
from go2_mapping_capture.recording import build_rosbag_command  # noqa: E402


def make_store(tmp_path, raw_limit=10, spool_limit=10):
    policy = RetentionPolicy(
        raw_limit_bytes=raw_limit,
        spool_limit_bytes=spool_limit,
        recovery_headroom_bytes=0,
        max_segment_bytes=10,
    )
    store = SegmentStore(tmp_path, policy)
    store.initialize()
    return store


def test_retention_never_deletes_unacknowledged_segments_when_raw_budget_is_exhausted(tmp_path):
    store = make_store(tmp_path)
    first = store.commit_bytes("000001", b"a" * 6, acknowledged=False)
    second = store.commit_bytes("000002", b"b" * 5, acknowledged=False)

    with pytest.raises(CapacityError, match="unacknowledged"):
        store.enforce_retention()

    assert store.payload_path(first).exists()
    assert store.payload_path(second).exists()


def test_retention_evicts_oldest_acknowledged_segment_first(tmp_path):
    store = make_store(tmp_path)
    oldest = store.commit_bytes("000001", b"a" * 4, acknowledged=True)
    newest = store.commit_bytes("000002", b"b" * 4, acknowledged=True)
    protected = store.commit_bytes("000003", b"c" * 6, acknowledged=False)

    result = store.enforce_retention()

    assert result.removed_segment_ids == (oldest,)
    assert not store.payload_path(oldest).exists()
    assert store.payload_path(newest).exists()
    assert store.payload_path(protected).exists()


def test_recovery_quarantines_interrupted_writes_without_exposing_a_final_segment(tmp_path):
    store = make_store(tmp_path)
    partial = store.staging_dir / "000001.mcap.partial"
    partial.write_bytes(b"incomplete")

    recovered = store.recover_interrupted_writes()

    assert recovered == ("000001.mcap.partial",)
    assert not partial.exists()
    assert (store.recovery_dir / "000001.mcap.partial").exists()
    assert list(store.raw_dir.glob("*.manifest.json")) == []


def test_checksum_mismatch_is_quarantined_and_never_silently_evicted(tmp_path):
    store = make_store(tmp_path, raw_limit=5)
    corrupt = store.commit_bytes("000001", b"a" * 4, acknowledged=True)
    store.payload_path(corrupt).write_bytes(b"corrupted")
    store.commit_bytes("000002", b"b" * 4, acknowledged=True)

    with pytest.raises(IntegrityError, match=corrupt):
        store.enforce_retention()

    assert not store.payload_path(corrupt).exists()
    assert (store.quarantine_dir / f"{corrupt}.mcap").exists()


def test_rosbag_command_requires_mcap_file_compression_and_explicit_topics(tmp_path):
    command = build_rosbag_command(
        output_uri=tmp_path / "staging" / "000001",
        topics=("/livox/lidar", "/dog_imu_raw"),
    )

    assert command == [
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
        str(tmp_path / "staging" / "000001"),
        "/livox/lidar",
        "/dog_imu_raw",
    ]
