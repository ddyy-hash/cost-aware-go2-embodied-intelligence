import asyncio
import threading

import pytest

from mapping.pc.go2_mapping_dashboard import server


@pytest.mark.parametrize("state", ["STARTING", "FINALIZING", "STALE", "UNKNOWN"])
def test_calibration_detection_rejects_preview_outside_active_raw_capture(monkeypatch, state):
    monkeypatch.setattr(
        server.robot_session,
        "snapshot",
        lambda: {"state": state, "capture_active": False, "session_id": "segment-test"},
    )
    monkeypatch.setattr(
        server.camera_assist,
        "frame",
        lambda: pytest.fail("inactive raw capture must be rejected before reading a preview frame"),
    )

    with pytest.raises(server.HTTPException) as raised:
        server.calibration_detect_latest()

    assert raised.value.status_code == 409
    assert raised.value.detail == "RAW_CALIBRATION_CAPTURE_NOT_RECORDING"


def test_calibration_detection_accepts_input_only_during_active_raw_capture(monkeypatch):
    frame = {"identity": "captured-frame"}
    monkeypatch.setattr(
        server.robot_session,
        "snapshot",
        lambda: {"state": "RECORDING", "capture_active": True, "session_id": "segment-test"},
    )
    monkeypatch.setattr(server.camera_assist, "frame", lambda: frame)
    monkeypatch.setattr(
        server.calibration_capture,
        "ingest_camera_payload",
        lambda payload: {"accepted": payload is frame},
    )

    assert server.calibration_detect_latest() == {"accepted": True}


def test_dashboard_index_is_not_cached_across_ui_deployments():
    response = server.index()
    index = (
        server.ROOT / "static" / "index.html"
    ).read_text(encoding="utf-8")

    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert 'href="/static/style.css?v=coarse-map-editor-v8"' in index
    assert 'src="/static/app.js?v=room-direct-rgb-v11"' in index
    assert 'src="/static/coarse-map.js?v=coarse-map-editor-v8"' in index


def test_readiness_is_lightweight_and_excludes_surface_point_arrays(monkeypatch):
    readiness = getattr(server, "readiness", None)
    assert callable(readiness), "dashboard must expose a lightweight readiness endpoint"
    monkeypatch.setattr(
        server.mapper,
        "room_scan_snapshot",
        lambda: {"state": "STALE", "live": False, "frames": 4},
    )
    monkeypatch.setattr(
        server.robot_session,
        "snapshot",
        lambda: {"state": "STOPPED", "capture_active": False, "session_id": None},
    )
    monkeypatch.setattr(
        server,
        "_building_scan_snapshot",
        lambda: {"state": "BUILDING", "surfel_count": 12, "tile_count": 2},
    )

    payload = readiness()

    assert payload["ready"] is True
    assert payload["dashboard_protocol"] == "rgbd-producer-epoch-v2"
    assert payload["room_scan"]["frames"] == 4
    assert payload["building_scan"]["surfel_count"] == 12
    assert payload["session"]["capture_active"] is False
    assert "live_map" not in payload
    assert "dense_map" not in payload


def test_live_readiness_never_waits_for_mapping_state(monkeypatch):
    endpoint = getattr(server, "live_readiness", None)
    assert callable(endpoint), "dashboard must expose a lock-free live transport readiness endpoint"
    monkeypatch.setattr(
        server,
        "_building_scan_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("live readiness must not read building state")),
    )
    monkeypatch.setattr(
        server.mapper,
        "room_scan_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("live readiness must not read room state")),
    )

    payload = asyncio.run(endpoint())

    assert payload == {
        "ready": True,
        "live_protocol": 3,
        "transport": "tailnet-websocket",
        "texture_keyframes": {
            "endpoint": "/ws/live/texture",
            "message_type": "texture_keyframe",
            "ack_type": "texture_keyframe_ack",
            "independent_wal": True,
            "max_message_bytes": 4 * 1024 * 1024,
        },
    }


def test_stack_readiness_is_lock_free_and_advertises_startup_capabilities(monkeypatch):
    endpoint = getattr(server, "stack_readiness", None)
    assert callable(endpoint), "dashboard must expose a lock-free PC stack readiness endpoint"
    monkeypatch.setattr(
        server,
        "_building_scan_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("stack readiness must not read building state")),
    )
    monkeypatch.setattr(
        server.mapper,
        "room_scan_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("stack readiness must not read room state")),
    )
    monkeypatch.setattr(
        server.robot_session,
        "snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("stack readiness must not read session state")),
    )

    payload = asyncio.run(endpoint())

    assert payload == {
        "ready": True,
        "dashboard_protocol": "rgbd-producer-epoch-v2",
        "live_protocol": 3,
        "automatic_reconstruction_pointer_protocol": "automatic-reconstruction-pointer-v1",
        "runtime_config_sha256": server._runtime_config_sha256,
        "capabilities": {
            "building_render_manifest": True,
            "camera_mjpeg": True,
            "field_acceptance_evidence": True,
            "textured_tile_export": True,
            "textured_building_export": True,
        },
    }


def test_runtime_calibration_status_is_lightweight_and_reports_operator_override(monkeypatch):
    endpoint = getattr(server, "runtime_calibration_status", None)
    assert callable(endpoint), "dashboard must expose lightweight runtime calibration status"
    monkeypatch.setattr(
        server.quality_gate,
        "snapshot",
        lambda: {
            "level": "CALIBRATED_DENSE",
            "calibration_id": "PROVISIONAL_AUTOCAL_PENDING_v1",
            "formal_runtime_override": True,
            "measurement_authority": False,
            "runtime_authority": "OPERATOR_FORCED_FORMAL",
            "camera_from_lidar_row_major": list(range(16)),
            "eligible_for": {"rgbd_or_neural_geometry_fusion": True},
        },
    )
    monkeypatch.setattr(
        server,
        "_dashboard_snapshot_uncached",
        lambda: (_ for _ in ()).throw(AssertionError("runtime status must not build a dashboard snapshot")),
    )

    payload = endpoint()

    assert payload == {
        "calibration_id": "PROVISIONAL_AUTOCAL_PENDING_v1",
        "formal_runtime_override": True,
        "measurement_authority": False,
        "runtime_authority": "OPERATOR_FORCED_FORMAL",
        "camera_from_lidar_row_major": list(range(16)),
        "runtime_projection_authorized": True,
    }


def test_active_building_id_uses_cached_summary_without_pipeline_snapshot(monkeypatch):
    class _Pipeline:
        def snapshot(self):
            raise AssertionError("active building lookup must not scan tile state")

    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline())
    monkeypatch.setattr(server, "_building_scan_summary", {"building_id": "building-a"})

    assert server._active_building_id() == "building-a"


def test_realtime_maintenance_does_not_run_high_fidelity_artifact_ingest_in_process(monkeypatch):
    class StopLoop(Exception):
        pass

    class Ingestor:
        def __init__(self):
            self.calls = 0

        def ingest_once(self):
            self.calls += 1

    async def stop_after_one_cycle(_delay):
        raise StopLoop

    ingestor = Ingestor()
    maintenance_calls = []
    offloaded = []
    scheduler = lambda: maintenance_calls.append(True)

    async def run_offloaded(function, *args, **kwargs):
        offloaded.append(function)
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "artifact_ingestor", ingestor)
    monkeypatch.setattr(server, "_in_process_artifact_ingest_enabled", False, raising=False)
    monkeypatch.setattr(server, "_refresh_tile_appearance_scheduler", scheduler)
    monkeypatch.setattr(server.asyncio, "to_thread", run_offloaded)
    monkeypatch.setattr(server.asyncio, "sleep", stop_after_one_cycle)

    with pytest.raises(StopLoop):
        asyncio.run(server._progressive_ingest_loop())

    assert ingestor.calls == 0
    assert maintenance_calls == [True]
    assert offloaded == [scheduler]


def test_lifespan_starts_maintenance_without_artifact_ingestor_and_closes_pipeline(
    monkeypatch,
):
    async def exercise():
        maintenance_started = asyncio.Event()
        queued_meshes = []

        async def maintenance_loop():
            maintenance_started.set()
            await asyncio.Event().wait()

        class Pipeline:
            def __init__(self, *args, **kwargs):
                self.tile_store = object()
                self.closed = False
                self.restored = False
                self.recovered = False

            def restore_persisted_runtimes(self, **kwargs):
                self.restored = True
                self.restore_kwargs = kwargs
                return 0

            def recover_pending(self):
                self.recovered = True
                return ()

            def active_targets(self):
                return (("building-a", "submap-a"),)

            def close(self):
                self.closed = True

        pipeline = Pipeline()
        class AssistanceWorker:
            def __init__(self):
                self.started = False
                self.closed = False

            def start(self):
                self.started = True

            def close(self):
                self.closed = True

        assistance = AssistanceWorker()
        monkeypatch.setattr(server, "artifact_ingestor", None)
        monkeypatch.setattr(server, "_depth_anything_worker", assistance)
        monkeypatch.setattr(server, "_progressive_ingest_loop", maintenance_loop)
        monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)
        monkeypatch.setattr(server, "_live_wal", object())
        monkeypatch.setattr(server, "_schedule_pending_typed_records", lambda: None)
        monkeypatch.setattr(server, "_queue_surface_mesh", lambda *target: queued_meshes.append(target))

        async with server._lifespan(server.app):
            await asyncio.wait_for(maintenance_started.wait(), timeout=1)

        assert pipeline.closed is True
        assert pipeline.restored is True
        assert pipeline.recovered is True
        assert pipeline.restore_kwargs == {"latest_only": True}
        assert server._live_surface_pipeline is None
        assert assistance.started is True
        assert assistance.closed is True
        assert queued_meshes == [("building-a", "submap-a")]

    asyncio.run(exercise())


def test_dashboard_can_explicitly_disable_the_surface_mesh_process_pool_for_local_preview(monkeypatch):
    monkeypatch.setenv("GO2_MAPPING_DISABLE_SURFACE_MESH_EXECUTOR", "1")

    def process_pool_must_not_start(**_kwargs):
        raise AssertionError("local preview must not create a process pool")

    monkeypatch.setattr(server, "ProcessPoolExecutor", process_pool_must_not_start)

    assert server._create_surface_mesh_executor() is None


def test_lifespan_serves_control_plane_before_surface_recovery_finishes(monkeypatch):
    async def exercise():
        restore_started = threading.Event()
        release_restore = threading.Event()
        entered = asyncio.Event()
        leave = asyncio.Event()

        async def idle_loop():
            await asyncio.Event().wait()

        class Pipeline:
            tile_store = object()

            def restore_persisted_runtimes(self, *, latest_only=False):
                assert latest_only is True
                restore_started.set()
                assert release_restore.wait(timeout=2)
                return 1

            def recover_pending(self):
                return ()

            def active_targets(self):
                return ()

            def close(self):
                return None

        monkeypatch.setattr(server, "artifact_ingestor", None)
        monkeypatch.setattr(server, "_depth_anything_worker", None)
        monkeypatch.setattr(server, "_progressive_ingest_loop", idle_loop)
        monkeypatch.setattr(server, "_texture_ingest_loop", idle_loop)
        monkeypatch.setattr(server, "_snapshot_refresh_loop", idle_loop)
        monkeypatch.setattr(server, "_live_ingest_worker", idle_loop)
        monkeypatch.setattr(server, "_surface_checkpoint_loop", idle_loop)
        monkeypatch.setattr(server, "_surface_mesh_worker", idle_loop)
        monkeypatch.setattr(server, "_live_surface_pipeline", Pipeline())
        monkeypatch.setattr(server, "_live_wal", object())
        monkeypatch.setattr(server, "_texture_wal", object())
        monkeypatch.setattr(server, "_schedule_pending_typed_records", lambda: None)
        monkeypatch.setattr(server, "_queue_surface_mesh", lambda *_target: None)

        async def run_lifespan():
            async with server._lifespan(server.app):
                entered.set()
                await leave.wait()

        task = asyncio.create_task(run_lifespan())
        assert await asyncio.to_thread(restore_started.wait, 1)
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.2)
        finally:
            release_restore.set()
            leave.set()
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(exercise())


def test_lifespan_starts_one_shared_full_snapshot_refresher(monkeypatch):
    async def exercise():
        refresher_started = asyncio.Event()
        refresher_instances = 0

        async def snapshot_refresher():
            nonlocal refresher_instances
            refresher_instances += 1
            refresher_started.set()
            await asyncio.Event().wait()

        async def maintenance_loop():
            await asyncio.Event().wait()

        class Pipeline:
            tile_store = object()

            def restore_persisted_runtimes(self, *, latest_only=False):
                assert latest_only is True
                return 0

            def recover_pending(self):
                return ()

            def active_targets(self):
                return ()

            def close(self):
                return None

        monkeypatch.setattr(server, "artifact_ingestor", None)
        monkeypatch.setattr(server, "_depth_anything_worker", None)
        monkeypatch.setattr(server, "_progressive_ingest_loop", maintenance_loop)
        monkeypatch.setattr(server, "_snapshot_refresh_loop", snapshot_refresher)
        monkeypatch.setattr(server, "_live_surface_pipeline", Pipeline())
        monkeypatch.setattr(server, "_live_wal", object())
        monkeypatch.setattr(server, "_schedule_pending_typed_records", lambda: None)

        async with server._lifespan(server.app):
            await asyncio.wait_for(refresher_started.wait(), timeout=1)
            assert refresher_instances == 1

        assert server._snapshot_refresh_task is None

    asyncio.run(exercise())
