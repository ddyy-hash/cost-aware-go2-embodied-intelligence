import hashlib
from pathlib import Path

import numpy as np

from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe, validate_surface_source_status
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    build_live_keyframe_message,
    build_surface_source_status_message,
)


SOURCE_HASH = hashlib.sha256(b"detail-source-config").hexdigest()


def _frame(sequence: int, *, timestamp_ns: int | None = None, producer: str = "producer-a"):
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id=producer,
        keyframe_sequence=sequence,
        timestamp_ns=timestamp_ns or (10_000 + sequence),
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray([[sequence, 0.0, 1.0]], dtype="<f4"),
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
    )
    return validate_live_keyframe(encoded.payload, message_bytes=encoded.message_bytes)


def test_wal_durably_admits_new_and_exact_duplicate_but_rejects_conflict_and_gap(tmp_path):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)

    first = wal.admit(_frame(20))
    duplicate = wal.admit(_frame(20))
    conflict = wal.admit(_frame(20, timestamp_ns=99_999))
    gap = wal.admit(_frame(22))
    second = wal.admit(_frame(21))

    assert first.disposition == "QUEUED_NEW"
    assert duplicate.disposition == "DUPLICATE_QUEUED"
    assert duplicate.record_sha256 == first.record_sha256
    assert conflict.disposition == "REJECTED"
    assert conflict.error_code == "IDENTITY_CONFLICT"
    assert gap.disposition == "REJECTED"
    assert gap.error_code == "SEQUENCE_GAP"
    assert gap.next_sequence == 21
    assert second.disposition == "QUEUED_NEW"
    assert [record.frame.identity.sequence for record in wal.pending()] == [20, 21]


def test_wal_restart_replays_only_records_not_covered_by_integrated_checkpoint(tmp_path):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    frame_40 = _frame(40)
    frame_41 = _frame(41)
    wal.admit(frame_40)
    wal.admit(frame_41)
    wal.commit_integrated(frame_40, checkpoint_id="surface-generation-1")

    restarted = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)

    assert [record.frame.identity.sequence for record in restarted.pending()] == [41]
    integrated_duplicate = restarted.admit(frame_40)
    assert integrated_duplicate.disposition == "DUPLICATE_INTEGRATED"
    assert integrated_duplicate.next_sequence == 42
    restarted.reclaim_integrated(frame_40.identity.stream_key())
    assert len(list((tmp_path / "records").glob("*.json"))) == 1


def test_wal_rejects_before_write_when_capacity_cannot_reserve_record(tmp_path):
    frame = _frame(1)
    wal = LiveKeyframeWal(tmp_path, max_bytes=32)

    admission = wal.admit(frame)

    assert admission.disposition == "REJECTED"
    assert admission.error_code == "WAL_CAPACITY_EXCEEDED"
    assert not list((tmp_path / "records").glob("*.json"))


def test_source_status_high_water_is_persisted_independently(tmp_path):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    encoded = build_surface_source_status_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        status_sequence=8,
        timestamp_ns=55_000,
        source_state="SOURCE_READY",
        source_clipped=False,
        source_points_before_dedup=200,
        source_points_after_dedup=180,
        compressed_point_bytes=800,
        websocket_message_bytes=1200,
        accepted_detail_hz=5.0,
        source_config_sha256=SOURCE_HASH,
        blocker=None,
    )
    status = validate_surface_source_status(encoded.payload)

    assert wal.accept_status(status) is True
    assert wal.accept_status(status) is False
    restarted = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    assert restarted.latest_status(status.identity.stream_key()).status_sequence == 8


def test_wal_restart_recovers_latest_status_without_in_memory_stream_key(tmp_path):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)

    def _status(*, session_id: str, producer: str, timestamp_ns: int, state: str, clipped: bool):
        encoded = build_surface_source_status_message(
            building_id="building-a",
            session_id=session_id,
            map_id=f"map-{session_id}",
            producer_instance_id=producer,
            status_sequence=1,
            timestamp_ns=timestamp_ns,
            source_state=state,
            source_clipped=clipped,
            source_points_before_dedup=200,
            source_points_after_dedup=100 if clipped else 180,
            compressed_point_bytes=800,
            websocket_message_bytes=1200,
            accepted_detail_hz=5.0,
            source_config_sha256=SOURCE_HASH,
            blocker="DETAIL_SOURCE_CLIPPED" if clipped else None,
        )
        return validate_surface_source_status(encoded.payload)

    older = _status(session_id="session-a", producer="producer-a", timestamp_ns=55_000, state="SOURCE_READY", clipped=False)
    newer = _status(session_id="session-b", producer="producer-b", timestamp_ns=56_000, state="DETAIL_SOURCE_CLIPPED", clipped=True)
    wal.accept_status(older)
    wal.accept_status(newer)

    restarted = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    recovered = restarted.latest_status_any()

    assert recovered.identity.stream_key() == newer.identity.stream_key()
    assert recovered.source_state == "DETAIL_SOURCE_CLIPPED"


def test_wal_pending_uses_startup_index_instead_of_reparsing_payloads(tmp_path, monkeypatch):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    wal.admit(_frame(20))
    wal.admit(_frame(21))
    real_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.parent == wal.records_root:
            raise AssertionError("pending WAL records were reparsed from disk")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert [record.frame.identity.sequence for record in wal.pending()] == [20, 21]


def test_wal_batch_commit_persists_one_cursor_update(tmp_path, monkeypatch):
    wal = LiveKeyframeWal(tmp_path, max_bytes=1_000_000)
    frames = [_frame(sequence) for sequence in (20, 21, 22)]
    for frame in frames:
        wal.admit(frame)
    saves = 0
    real_save_state = wal._save_state

    def counted_save_state():
        nonlocal saves
        saves += 1
        real_save_state()

    monkeypatch.setattr(wal, "_save_state", counted_save_state)

    wal.commit_integrated_batch(frames, checkpoint_id="surface-generation-1")

    assert saves == 1
    assert wal.pending() == []
