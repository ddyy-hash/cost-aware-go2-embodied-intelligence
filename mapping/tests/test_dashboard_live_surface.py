import asyncio
import hashlib
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_live.live_pipeline import LiveSurfacePipeline
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    build_live_keyframe_message,
    build_surface_source_status_message,
)


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()


@pytest.fixture(autouse=True)
def _reset_dashboard_surface_state(monkeypatch):
    monkeypatch.setattr(server, "_live_ingest_last_error", None)
    monkeypatch.setattr(server, "_live_source_status_stream", None)
    monkeypatch.setattr(server, "_surface_mesh_pending", set())
    monkeypatch.setattr(server, "_surface_mesh_deferred", set())
    monkeypatch.setattr(server, "_surface_mesh_retry_after", {})
    monkeypatch.setattr(server, "_surface_mesh_retry_delay_s", 0.5)
    monkeypatch.setattr(server, "_live_wal", None)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    yield


def _message(sequence: int = 1):
    return build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=10_000 + sequence,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]], dtype="<f4"),
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=True,
    )


def _source_status(*, timestamp_ns: int, source_state: str = "SOURCE_READY", source_clipped: bool = False, blocker=None):
    return build_surface_source_status_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        status_sequence=1,
        timestamp_ns=timestamp_ns,
        source_state=source_state,
        source_clipped=source_clipped,
        source_points_before_dedup=100,
        source_points_after_dedup=90,
        compressed_point_bytes=400,
        websocket_message_bytes=700,
        accepted_detail_hz=5.0,
        source_config_sha256=SOURCE_HASH,
        blocker=blocker,
    )


def test_dashboard_worker_checkpoints_typed_wal_and_exposes_count_only_building_state(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)
    monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=4))
    monkeypatch.setattr(server, "_typed_inflight", set())
    monkeypatch.setattr(server, "_typed_processed", set())
    encoded = _message(20)

    acknowledgement = server._dispatch_live_message(encoded.payload, message_bytes=encoded.message_bytes)
    record = server._live_ingest_queue.get_nowait()
    update = server._process_typed_record(record, force_checkpoint=True)
    state = server.dashboard_snapshot()
    patch = server.dashboard_telemetry_patch()

    assert acknowledgement["ack_stage"] == "QUEUED"
    assert update.checkpoint_generation_id is not None
    assert not wal.pending()
    assert state["building_scan"]["surfel_count"] == 2
    assert state["building_scan"]["geometry_authority"] is False
    assert state["building_scan"]["quality"] == "PREVIEW_UNCALIBRATED"
    assert patch["snapshot"]["building_scan"]["surfel_count"] == 2
    assert "tiles" not in patch["snapshot"]["building_scan"]


def test_typed_wal_replay_survives_a_legacy_preview_cursor_already_past_the_frame(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    encoded = _message(71)
    frame = server.validate_live_keyframe(encoded.payload, message_bytes=encoded.message_bytes)
    assert wal.admit(frame).accepted is True
    record = wal.pending()[0]
    identity = frame.identity.as_tuple()

    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)
    monkeypatch.setattr(server, "_typed_inflight", {identity})
    monkeypatch.setattr(server, "_typed_processed", set())
    monkeypatch.setattr(server, "_schedule_pending_typed_records", lambda: None)
    monkeypatch.setattr(
        server,
        "ingest_live_keyframe",
        lambda _payload: {
            "accepted": False,
            "sequence": 71,
            "points": 100,
            "quality": "PREVIEW_UNCALIBRATED",
            "next_sequence": 72,
        },
    )

    async def _run():
        queue = asyncio.Queue(maxsize=1)
        monkeypatch.setattr(server, "_live_ingest_queue", queue)
        await queue.put(record)
        worker = asyncio.create_task(server._live_ingest_worker())
        try:
            await asyncio.wait_for(queue.join(), timeout=0.5)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(_run())

    assert identity in server._typed_processed
    assert identity not in server._typed_inflight
    assert wal.pending()
    pipeline.flush_due(force=True)
    assert not wal.pending()
    assert server._live_ingest_last_error is None


def test_surface_manifest_is_small_and_tile_geometry_uses_separate_immutable_endpoint(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)

    manifest = server.building_surface_manifest()

    assert manifest["building_scan"]["tile_count"] == 0
    assert manifest["building_scan"]["source_status_age_s"] is None
    assert manifest["building_scan"]["source_blocker"] == "SOURCE_STATUS_STALE"
    assert manifest["tiles"] == []


def test_local_surface_manifest_does_not_wait_for_pipeline_snapshot(monkeypatch):
    active_tile = {"revision_id": "a" * 64, "submap_id": "submap-a"}
    historical_tile = {"revision_id": "b" * 64, "submap_id": "submap-old"}
    stored_tiles = [active_tile, historical_tile]

    class _TileStore:
        def manifest(self, *, scope):
            assert scope == "local"
            return {"schema_version": 3, "scope": scope, "tiles": stored_tiles}

    class _Pipeline:
        tile_store = _TileStore()

        def manifest(self, *, scope):
            raise AssertionError("local manifest must not call pipeline.manifest")

        def snapshot(self):
            raise AssertionError("local manifest must not call pipeline.snapshot")

    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline())
    monkeypatch.setattr(
        server,
        "_building_scan_summary",
        {
            "state": "BUILDING",
            "active_submap_id": "submap-a",
            "tile_count": 2,
            "triangle_count": 2,
        },
    )

    manifest = server.building_surface_manifest("local")

    assert manifest["scope"] == "local"
    assert manifest["tiles"] == [active_tile]
    assert manifest["building_scan"]["tile_count"] == 2


def test_building_scan_projects_a_fresh_durable_source_status_into_snapshot_and_telemetry(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    now_ns = time.time_ns()
    status = _source_status(timestamp_ns=now_ns - 1_500_000_000)
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    building_scan = server.dashboard_snapshot()["building_scan"]
    telemetry_scan = server.dashboard_telemetry_patch()["snapshot"]["building_scan"]

    assert 1.4 < building_scan["source_status_age_s"] < 1.7
    assert building_scan["source_state"] == "SOURCE_READY"
    assert building_scan["source_clipped"] is False
    assert building_scan["source_blocker"] is None
    assert building_scan["state"] == "WAITING_FOR_SURFACE"
    assert building_scan["quality"] == "PREVIEW_UNCALIBRATED"
    assert building_scan["geometry_authority"] is False
    assert telemetry_scan["source_state"] == "SOURCE_READY"
    assert telemetry_scan["source_blocker"] is None


def test_building_scan_marks_status_older_than_three_seconds_stale_at_strict_boundary(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    status = _source_status(timestamp_ns=1_000_000_000)
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)
    monkeypatch.setattr(server.time, "time_ns", lambda: 4_000_000_001)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_status_age_s"] > 3.0
    assert building_scan["source_state"] == "SOURCE_READY"
    assert building_scan["source_blocker"] == "SOURCE_STATUS_STALE"
    assert building_scan["state"] == "SOURCE_STATUS_STALE"
    assert building_scan["delivery_authorized"] is False
    assert building_scan["quality"] == "PREVIEW_UNCALIBRATED"
    assert building_scan["geometry_authority"] is False


def test_building_scan_keeps_status_fresh_at_exactly_three_seconds(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    status = _source_status(timestamp_ns=1_000_000_000)
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)
    monkeypatch.setattr(server.time, "time_ns", lambda: 4_000_000_000)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_status_age_s"] == 3.0
    assert building_scan["source_blocker"] is None
    assert building_scan["state"] == "WAITING_FOR_SURFACE"


def test_building_scan_recovers_latest_durable_status_after_process_restart(monkeypatch, tmp_path):
    wal_root = tmp_path / "wal"
    wal = LiveKeyframeWal(wal_root, max_bytes=1_000_000)
    status = _source_status(timestamp_ns=1_000_000_000)
    wal.accept_status(server.validate_surface_source_status(status.payload))
    restarted = LiveKeyframeWal(wal_root, max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", restarted)
    monkeypatch.setattr(server, "_live_source_status_stream", None)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    monkeypatch.setattr(server.time, "time_ns", lambda: 1_000_000_000)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_status_age_s"] == 0.0
    assert building_scan["source_state"] == "SOURCE_READY"
    assert building_scan["source_blocker"] is None


def test_source_payload_blockers_are_exposed_and_keep_detailed_readiness_closed(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    status = _source_status(
        timestamp_ns=time.time_ns(),
        source_state="DETAIL_PAYLOAD_LIMIT_EXCEEDED",
        blocker="DETAIL_PAYLOAD_LIMIT_EXCEEDED",
    )
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_state"] == "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
    assert building_scan["source_clipped"] is False
    assert building_scan["source_blocker"] == "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
    assert building_scan["state"] == "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
    assert building_scan["delivery_authorized"] is False
    assert building_scan["quality"] == "PREVIEW_UNCALIBRATED"
    assert building_scan["geometry_authority"] is False


def test_clipped_source_status_is_a_detailed_readiness_blocker(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    status = _source_status(
        timestamp_ns=time.time_ns(),
        source_state="DETAIL_SOURCE_CLIPPED",
        source_clipped=True,
        blocker="DETAIL_SOURCE_CLIPPED",
    )
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_state"] == "DETAIL_SOURCE_CLIPPED"
    assert building_scan["source_clipped"] is True
    assert building_scan["source_blocker"] == "DETAIL_SOURCE_CLIPPED"
    assert building_scan["state"] == "DETAIL_SOURCE_CLIPPED"
    assert building_scan["delivery_authorized"] is False


def test_source_blocker_precedence_prefers_capacity_over_stale_source(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    monkeypatch.setattr(server, "_live_ingest_last_error", "SURFACE_CAPACITY_BLOCKED")
    status = _source_status(timestamp_ns=time.time_ns() - 4_000_000_000)
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_blocker"] == "SURFACE_CAPACITY_BLOCKED"
    assert building_scan["state"] == "SURFACE_CAPACITY_BLOCKED"


def test_pc_pipeline_error_has_precedence_over_stale_source_status(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    monkeypatch.setattr(server, "_live_ingest_last_error", "RuntimeError: checkpoint failed")
    status = _source_status(timestamp_ns=time.time_ns() - 4_000_000_000)
    server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    building_scan = server.dashboard_snapshot()["building_scan"]

    assert building_scan["source_blocker"] == "PC_PIPELINE_ERROR"
    assert building_scan["state"] == "FAULTED"
    assert building_scan["pipeline_error"] == "RuntimeError: checkpoint failed"
    assert building_scan["source_status_age_s"] > 3.0
    assert building_scan["quality"] == "PREVIEW_UNCALIBRATED"
    assert building_scan["geometry_authority"] is False


def test_surface_mesh_queue_retries_deferred_targets_after_single_slot_frees(monkeypatch):
    queue = asyncio.Queue(maxsize=1)
    calls = []

    class _Pipeline:
        def mesh(self, building_id, submap_id):
            calls.append((building_id, submap_id))

    monkeypatch.setattr(server, "_surface_mesh_queue", queue)
    monkeypatch.setattr(server, "_surface_mesh_pending", set())
    monkeypatch.setattr(server, "_surface_mesh_deferred", set())
    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline())
    server._queue_surface_mesh("building-a", "submap-a")
    server._queue_surface_mesh("building-b", "submap-b")

    async def _run_worker_once():
        worker = asyncio.create_task(server._surface_mesh_worker())
        try:
            await asyncio.wait_for(queue.join(), timeout=1.0)
            await asyncio.sleep(0)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(_run_worker_once())

    assert calls == [("building-a", "submap-a"), ("building-b", "submap-b")]
    assert server._surface_mesh_pending == set()
    assert server._surface_mesh_deferred == set()


def test_surface_mesh_queue_rebuilds_same_target_when_it_becomes_dirty_while_pending(monkeypatch):
    import threading

    queue = asyncio.Queue(maxsize=1)
    calls = []
    started = threading.Event()
    release = threading.Event()

    class _Pipeline:
        def mesh(self, building_id, submap_id):
            calls.append((building_id, submap_id))
            if len(calls) == 1:
                started.set()
                assert release.wait(timeout=1.0)

    monkeypatch.setattr(server, "_surface_mesh_queue", queue)
    monkeypatch.setattr(server, "_surface_mesh_pending", set())
    monkeypatch.setattr(server, "_surface_mesh_deferred", set())
    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline())
    server._queue_surface_mesh("building-a", "submap-a")

    async def _run():
        worker = asyncio.create_task(server._surface_mesh_worker())
        try:
            assert await asyncio.to_thread(started.wait, 1.0)
            server._queue_surface_mesh("building-a", "submap-a")
            release.set()
            await asyncio.wait_for(queue.join(), timeout=1.0)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(_run())

    assert calls == [("building-a", "submap-a"), ("building-a", "submap-a")]


def test_surface_mesh_worker_retries_target_after_transient_failure(monkeypatch):
    queue = asyncio.Queue(maxsize=1)
    calls = []

    class _Pipeline:
        def mesh(self, building_id, submap_id):
            calls.append((building_id, submap_id))
            if len(calls) == 1:
                raise OSError("temporary tile publish failure")

    monkeypatch.setattr(server, "_surface_mesh_queue", queue)
    monkeypatch.setattr(server, "_surface_mesh_pending", set())
    monkeypatch.setattr(server, "_surface_mesh_deferred", set())
    monkeypatch.setattr(server, "_surface_mesh_retry_after", {})
    monkeypatch.setattr(server, "_surface_mesh_retry_delay_s", 0.0)
    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline())
    monkeypatch.setattr(server, "_live_ingest_last_error", None)
    server._queue_surface_mesh("building-a", "submap-a")

    async def _run():
        worker = asyncio.create_task(server._surface_mesh_worker())
        try:
            await asyncio.wait_for(queue.join(), timeout=1.0)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(_run())

    assert calls == [("building-a", "submap-a"), ("building-a", "submap-a")]
    assert server._surface_mesh_deferred == set()
    assert server._live_ingest_last_error is None
