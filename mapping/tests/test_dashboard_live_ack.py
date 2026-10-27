import asyncio
import base64
import contextlib
import hashlib
import json
import threading
import zlib

import numpy as np
import pytest

pytest.importorskip("fastapi")
from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_dashboard.camera_assist import CameraAssistState
from mapping.pc.go2_mapping_dashboard.live_transport import LiveTransportMetrics
from mapping.pc.go2_mapping_live.progressive_mapper import ProgressiveMapper
from mapping.pc.go2_mapping_live.live_pipeline import LiveSurfacePipeline
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    build_live_keyframe_message,
    build_surface_source_status_message,
)


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    async def json(self):
        return self.payload


def _typed_keyframe(sequence: int = 0):
    return build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=10_000 + sequence,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=hashlib.sha256(b"source-config").hexdigest(),
        points_xyz=np.asarray([[0.1, 0.2, 0.3]], dtype="<f4"),
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
    )


def test_typed_live_ack_is_returned_only_after_a_durable_wal_record(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", LiveSurfacePipeline(tmp_path / "buildings", wal=wal))
    monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=2))
    monkeypatch.setattr(server, "_typed_inflight", set())
    encoded = _typed_keyframe(50)

    acknowledgement = server._dispatch_live_message(encoded.payload, message_bytes=encoded.message_bytes)

    assert acknowledgement["message_type"] == "live_keyframe_ack"
    assert acknowledgement["accepted"] is True
    assert acknowledgement["ack_stage"] == "QUEUED"
    assert acknowledgement["disposition"] == "QUEUED_NEW"
    assert acknowledgement["keyframe_sequence"] == 50
    records = list((tmp_path / "wal" / "records").glob("*.json"))
    assert len(records) == 1
    assert records[0].stat().st_size > 0


def test_typed_live_admission_does_not_block_the_asyncio_event_loop(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class BlockingPipeline:
        def admit(self, _frame):
            started.set()
            assert release.wait(timeout=2)
            return type(
                "Admission",
                (),
                {
                    "accepted": True,
                    "disposition": "QUEUED_NEW",
                    "record_sha256": hashlib.sha256(b"record").hexdigest(),
                    "next_sequence": 1,
                    "error_code": None,
                },
            )()

    async def exercise():
        monkeypatch.setattr(server, "_live_wal", object())
        monkeypatch.setattr(server, "_live_surface_pipeline", BlockingPipeline())
        monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=2))
        monkeypatch.setattr(server, "_typed_inflight", set())
        monkeypatch.setattr(server, "_schedule_pending_typed_records", lambda: None)
        encoded = _typed_keyframe(0)
        task = asyncio.create_task(
            server._dispatch_live_message_async(encoded.payload, message_bytes=encoded.message_bytes)
        )
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        release.set()
        acknowledgement = await asyncio.wait_for(task, timeout=1)
        assert acknowledgement["accepted"] is True

    asyncio.run(exercise())


def test_typed_live_capacity_failure_never_claims_queued(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=32)
    monkeypatch.setattr(server, "_live_wal", wal)
    monkeypatch.setattr(server, "_live_surface_pipeline", LiveSurfacePipeline(tmp_path / "buildings", wal=wal))
    monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=2))
    encoded = _typed_keyframe(1)

    acknowledgement = server._dispatch_live_message(encoded.payload, message_bytes=encoded.message_bytes)

    assert acknowledgement["accepted"] is False
    assert acknowledgement["ack_stage"] is None
    assert acknowledgement["error_code"] == "WAL_CAPACITY_EXCEEDED"
    assert not list((tmp_path / "wal" / "records").glob("*.json"))


def test_typed_source_status_uses_independent_persisted_ack(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_live_wal", LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000))
    status = build_surface_source_status_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        status_sequence=3,
        timestamp_ns=20_000,
        source_state="SOURCE_READY",
        source_clipped=False,
        source_points_before_dedup=100,
        source_points_after_dedup=90,
        compressed_point_bytes=400,
        websocket_message_bytes=700,
        accepted_detail_hz=5.0,
        source_config_sha256=hashlib.sha256(b"source-config").hexdigest(),
        blocker=None,
    )

    acknowledgement = server._dispatch_live_message(status.payload, message_bytes=status.message_bytes)

    assert acknowledgement["message_type"] == "surface_source_status_ack"
    assert acknowledgement["status_sequence"] == 3
    assert acknowledgement["accepted"] is True
    assert "keyframe_sequence" not in acknowledgement


def test_live_keyframe_response_is_a_small_ack_not_a_map_snapshot(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)

    def snapshot_must_not_be_called():
        raise AssertionError("live keyframe acknowledgements must not build a preview map")

    monkeypatch.setattr(mapper, "snapshot", snapshot_must_not_be_called)
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-live-map",
        "sequence": 7,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }

    response = asyncio.run(server.live_keyframe(JsonRequest(payload)))

    assert response == {
        "accepted": True,
        "sequence": 7,
        "points": 1,
        "quality": "PREVIEW_UNCALIBRATED",
    }


def test_websocket_ingest_uses_the_same_small_keyframe_ack(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-ws-map",
        "sequence": 3,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }

    response = server.ingest_live_keyframe(payload)

    assert response == {
        "accepted": True,
        "sequence": 3,
        "points": 1,
        "quality": "PREVIEW_UNCALIBRATED",
    }


def test_replayed_current_keyframe_is_acknowledged_without_duplicate_integration(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-live-map",
        "sequence": 250,
        "timestamp_ns": 9_000,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }

    assert server.ingest_live_keyframe(payload)["accepted"] is True
    replay = server.ingest_live_keyframe(payload)

    assert replay == {
        "accepted": True,
        "sequence": 250,
        "points": 1,
        "quality": "PREVIEW_UNCALIBRATED",
        "duplicate": True,
    }
    assert mapper.snapshot()["last_sequence"] == 250
    assert mapper.snapshot()["points"] == 1


def test_snapshot_merges_live_transport_measurements_and_live_pose(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    monkeypatch.setattr(server, "live_transport", LiveTransportMetrics())
    points = np.zeros((100, 3), dtype="<f4")
    pose = np.eye(4)
    pose[:3, 3] = (1.25, -2.5, 0.4)
    payload = {
        "map_id": "test-live-map",
        "sequence": 1,
        "timestamp_ns": 9_000,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": pose.ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
        "previous_ack_rtt_ms": 84.5,
    }

    assert server.ingest_live_keyframe(payload)["accepted"] is True
    state = server.snapshot()

    assert state["transport"]["app_goodput_mbps"] > 0
    assert state["transport"]["keyframe_hz"] > 0
    assert state["transport"]["ack_rtt_ms"] == 84.5
    assert state["transport"]["last_frame_age_ms"] is not None
    assert state["mapping"]["points"] == 1
    assert state["robot"]["position"] == [1.25, -2.5, 0.4]


def test_stale_producer_sequence_receives_the_next_safe_sequence(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-live-map",
        "sequence": 250,
        "timestamp_ns": 9_000,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }
    assert server.ingest_live_keyframe(payload)["accepted"] is True
    stale = {**payload, "sequence": 1, "timestamp_ns": 10_000}

    response = server.ingest_live_keyframe(stale)

    assert response == {
        "accepted": False,
        "sequence": 1,
        "points": 1,
        "quality": "PREVIEW_UNCALIBRATED",
        "next_sequence": 251,
    }


def test_dashboard_websocket_uses_the_same_live_transport_snapshot_as_the_http_api(monkeypatch):
    class StopAfterFirstMessage(Exception):
        pass

    class OneMessageSocket:
        def __init__(self):
            self.state = None

        async def accept(self):
            return None

        async def send_json(self, state):
            self.state = state
            raise StopAfterFirstMessage

    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    monkeypatch.setattr(server, "live_transport", LiveTransportMetrics())
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-live-map",
        "sequence": 1,
        "timestamp_ns": 9_000,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
        "previous_ack_rtt_ms": 84.5,
    }
    assert server.ingest_live_keyframe(payload)["accepted"] is True
    server.dashboard_snapshot_json(force=True)
    websocket = OneMessageSocket()

    with pytest.raises(StopAfterFirstMessage):
        asyncio.run(server.live(websocket))

    assert websocket.state["transport"]["app_goodput_mbps"] > 0
    assert websocket.state["transport"]["ack_rtt_ms"] == 84.5
    assert websocket.state["mapping"]["points"] == 1


def test_protocol_2_websocket_telemetry_is_not_blocked_by_a_slow_full_snapshot(monkeypatch):
    release_snapshot = threading.Event()

    def slow_snapshot_json():
        assert release_snapshot.wait(timeout=2)
        return json.dumps({"full_snapshot": True})

    class TelemetrySocket:
        query_params = {"protocol": "2"}

        def __init__(self):
            self.telemetry = asyncio.Event()

        async def accept(self):
            return None

        async def send_text(self, payload):
            message = json.loads(payload)
            if message.get("message_type") == "telemetry":
                self.telemetry.set()
                raise server.WebSocketDisconnect()

    async def exercise():
        websocket = TelemetrySocket()
        task = asyncio.create_task(server.live(websocket))
        try:
            await asyncio.wait_for(websocket.telemetry.wait(), timeout=0.2)
        finally:
            release_snapshot.set()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    monkeypatch.setattr(server, "dashboard_snapshot_json", slow_snapshot_json)
    asyncio.run(exercise())


def test_protocol_2_clients_do_not_build_full_snapshots_per_connection(monkeypatch):
    snapshot_builds = 0

    def counted_snapshot_json():
        nonlocal snapshot_builds
        snapshot_builds += 1
        return json.dumps({"full_snapshot": snapshot_builds})

    class OneTelemetrySocket:
        query_params = {"protocol": "2"}

        async def accept(self):
            return None

        async def send_text(self, payload):
            if json.loads(payload).get("message_type") == "telemetry":
                raise server.WebSocketDisconnect()

    async def exercise():
        await asyncio.gather(server.live(OneTelemetrySocket()), server.live(OneTelemetrySocket()))

    monkeypatch.setattr(server, "dashboard_snapshot_json", counted_snapshot_json)
    asyncio.run(exercise())

    assert snapshot_builds == 0


def test_protocol_2_websocket_sends_a_refreshed_shared_full_snapshot(monkeypatch):
    initial_generation = server._publish_snapshot_cache(json.dumps({"map_revision": 1}))

    class RefreshSocket:
        query_params = {"protocol": "2"}

        def __init__(self):
            self.full_revisions = []

        async def accept(self):
            return None

        async def send_text(self, payload):
            message = json.loads(payload)
            if "map_revision" in message:
                self.full_revisions.append(message["map_revision"])
            if message.get("message_type") == "telemetry" and len(self.full_revisions) == 1:
                generation = server._publish_snapshot_cache(json.dumps({"map_revision": 2}))
                assert generation > initial_generation
            elif self.full_revisions == [1, 2]:
                raise server.WebSocketDisconnect()

    websocket = RefreshSocket()
    monkeypatch.setattr(server, "_dashboard_telemetry_hz", 1000.0)

    asyncio.run(server.live(websocket))

    assert websocket.full_revisions == [1, 2]


def test_protocol_3_websocket_is_point_free_and_emits_pose_trajectory_and_tile_revision(monkeypatch):
    monkeypatch.setattr(server.mapper, "telemetry_snapshot", lambda: {
        "position": [1.0, 2.0, 0.4],
        "heading_xy": [0.0, 1.0],
        "trajectory": [[0.0, 0.0, 0.4], [1.0, 2.0, 0.4]],
    })
    monkeypatch.setattr(server.robot_session, "snapshot", lambda: {
        "session_id": "session-a",
        "state": "RECORDING",
        "capture_active": True,
    })
    monkeypatch.setattr(server, "_building_scan_telemetry_snapshot", lambda: {
        "building_id": "building-a",
        "state": "MAPPING",
        "quality": "PREVIEW_UNCALIBRATED",
        "active_floor_id": "floor-01",
        "active_submap_id": "submap-a",
    })
    monkeypatch.setattr(
        server,
        "_active_building_id",
        lambda: (_ for _ in ()).throw(
            AssertionError("protocol 3 must use its cached Building snapshot")
        ),
    )

    class Published:
        revision_id = "a" * 64

    monkeypatch.setattr(server._render_manifest_store, "current", lambda _building_id: Published())

    class Protocol3Socket:
        query_params = {"protocol": "3"}

        def __init__(self):
            self.messages = []

        async def accept(self):
            pass

        async def send_text(self, payload):
            self.messages.append(json.loads(payload))
            if len(self.messages) == 3:
                raise server.WebSocketDisconnect()

    websocket = Protocol3Socket()
    asyncio.run(server.live(websocket))

    assert [message["message_type"] for message in websocket.messages] == [
        "robot_state", "trajectory_reset", "tile_revision",
    ]
    assert websocket.messages[0]["robot"]["position"] == [1.0, 2.0, 0.4]
    assert websocket.messages[0]["building_quality"] == "PREVIEW_UNCALIBRATED"
    assert websocket.messages[0]["active_floor_id"] == "floor-01"
    assert websocket.messages[0]["active_submap_id"] == "submap-a"
    assert websocket.messages[1]["points"] == [[0.0, 0.0, 0.4], [1.0, 2.0, 0.4]]
    assert websocket.messages[2]["scene_revision_id"] == "a" * 64
    encoded = json.dumps(websocket.messages)
    assert "preview_points" not in encoded
    assert "measured_preview_points" not in encoded
    assert "dense_map" not in encoded
    assert "camera" not in encoded


def test_camera_assist_uses_a_separate_latest_frame_endpoint_without_expanding_ack(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    monkeypatch.setattr(server, "camera_assist", CameraAssistState())
    points = np.zeros((100, 3), dtype="<f4")
    depth = base64.b64encode((1000).to_bytes(2, "little") * 4).decode("ascii")
    payload = {
        "map_id": "test-camera-map",
        "sequence": 1,
        "timestamp_ns": 9_000,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
        "camera": {
            "sequence": 1,
            "timestamp_ns": 9_000,
            "color": {"jpeg_b64": base64.b64encode(b"\xff\xd8test\xff\xd9").decode("ascii"), "width": 2, "height": 2},
            "depth": {"u16_le_b64": depth, "width": 2, "height": 2, "scale_m": 0.001, "aligned_to_color": True},
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        },
    }

    acknowledgement = server.ingest_live_keyframe(payload)
    state = server.snapshot()
    frame = server.camera_frame()

    assert acknowledgement == {"accepted": True, "sequence": 1, "points": 1, "quality": "PREVIEW_UNCALIBRATED"}
    assert state["camera"]["distance_measurement_ready"] is True
    assert state["camera"]["texture_authorized"] is False
    assert frame["sequence"] == 1


def test_camera_preview_message_updates_camera_without_waiting_for_a_lidar_keyframe(monkeypatch):
    monkeypatch.setattr(server, "camera_assist", CameraAssistState())
    payload = {
        "message_type": "camera_preview",
        "protocol_version": 3,
        "building_id": "building-camera",
        "session_id": "segment-camera",
        "map_id": "map-camera",
        "producer_instance_id": "producer-camera",
        "calibration_id": "PROVISIONAL_TEST",
        "camera": {
            "sequence": 11,
            "timestamp_ns": 11_000,
            "producer_instance_id": "camera-epoch",
            "color": {
                "jpeg_b64": base64.b64encode(b"\xff\xd8camera\xff\xd9").decode("ascii"),
                "width": 2,
                "height": 2,
            },
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        },
    }

    acknowledgement = server._dispatch_live_message(payload)

    assert acknowledgement == {
        "message_type": "camera_preview_ack",
        "protocol_version": 3,
        "building_id": "building-camera",
        "session_id": "segment-camera",
        "map_id": "map-camera",
        "producer_instance_id": "producer-camera",
        "camera_sequence": 11,
        "accepted": True,
        "error_code": None,
    }
    assert server.camera_frame()["sequence"] == 11


def test_camera_mjpeg_stream_emits_a_latest_only_unbuffered_part(monkeypatch):
    state = CameraAssistState()
    state.ingest(
        {
            "sequence": 12,
            "timestamp_ns": 12_000,
            "producer_instance_id": "camera-stream",
            "color": {
                "jpeg_b64": base64.b64encode(b"\xff\xd8stream\xff\xd9").decode("ascii"),
                "width": 2,
                "height": 2,
            },
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        },
        "PROVISIONAL_TEST",
    )
    monkeypatch.setattr(server, "camera_assist", state)

    part = next(server._camera_mjpeg_parts(wait_timeout=0))
    response = server.camera_stream()

    assert b"Content-Type: image/jpeg\r\n" in part
    assert b"X-Go2-Camera-Identity: camera-stream:12\r\n" in part
    assert part.endswith(b"\xff\xd8stream\xff\xd9\r\n")
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    assert response.headers["x-accel-buffering"] == "no"


def test_camera_status_exposes_fresh_metadata_without_copying_frame_payload(monkeypatch):
    state = CameraAssistState()
    state.ingest(
        {
            "sequence": 13,
            "timestamp_ns": 13_000,
            "producer_instance_id": "camera-status",
            "color": {
                "jpeg_b64": base64.b64encode(b"\xff\xd8status\xff\xd9").decode("ascii"),
                "width": 2,
                "height": 2,
            },
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        },
        "PROVISIONAL_TEST",
    )
    monkeypatch.setattr(server, "camera_assist", state)
    monkeypatch.setattr(
        server.live_transport,
        "snapshot",
        lambda: {
            "camera_deferred_by_bandwidth": False,
            "camera_depth_deferred_by_bandwidth": True,
        },
    )

    status = server.camera_status()

    assert status["state"] == "LIVE_COLOR_ONLY"
    assert status["sequence"] == 13
    assert status["color_width"] == 2
    assert status["color_height"] == 2
    assert status["deferred_by_bandwidth"] is False
    assert status["depth_deferred_by_bandwidth"] is True
    encoded = json.dumps(status)
    assert "jpeg_b64" not in encoded
    assert "u16_le_b64" not in encoded


def test_dashboard_telemetry_patch_keeps_the_global_map_out_of_the_15_hz_path(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.zeros((100, 3), dtype="<f4")
    payload = {
        "map_id": "test-live-map",
        "sequence": 4,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }
    assert server.ingest_live_keyframe(payload)["accepted"] is True

    patch = server.dashboard_telemetry_patch()

    assert patch["message_type"] == "telemetry"
    assert patch["snapshot"]["live_map"]["last_sequence"] == 4
    assert "preview_points" not in patch["snapshot"]["live_map"]
    assert "camera" in patch["snapshot"]


def test_dashboard_telemetry_patch_reuses_the_cached_building_summary(monkeypatch):
    expected = {"state": "BUILDING", "surfel_count": 123, "pipeline_error": None}
    monkeypatch.setattr(server, "_building_scan_telemetry_snapshot", lambda: expected)
    monkeypatch.setattr(
        server,
        "_building_scan_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("15 Hz telemetry must not read the surface pipeline")),
    )

    patch = server.dashboard_telemetry_patch()

    assert patch["snapshot"]["building_scan"] == expected


def test_live_admission_applies_backpressure_without_evicting_an_older_frame(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=1))
    monkeypatch.setattr(server, "_live_ingest_next_sequence", 0)
    monkeypatch.setattr(server, "_live_ingest_last_admitted", {})
    payload = {
        "map_id": "test-live-map",
        "sequence": 0,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
    }

    assert server._queue_live_keyframe(payload)["accepted"] is True
    blocked = server._queue_live_keyframe({**payload, "sequence": 1, "timestamp_ns": 2})

    assert blocked["accepted"] is False
    assert blocked["retry_after_ms"] == 100
    queued = server._live_ingest_queue.get_nowait()
    assert queued["sequence"] == 0
    server._live_ingest_queue.task_done()
    assert server._queue_live_keyframe({**payload, "sequence": 1, "timestamp_ns": 2})["accepted"] is True


def test_live_keyframe_losslessly_decodes_compressed_float32_points(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.arange(300, dtype="<f4").reshape((-1, 3))
    payload = {
        "map_id": "test-live-map",
        "sequence": 1,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
        "points_f32_zlib_b64": base64.b64encode(zlib.compress(points.tobytes(), level=1)).decode("ascii"),
        "pose_map_from_lidar": np.eye(4).ravel().tolist(),
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None,
        "calibrated_lio": False,
    }

    assert server.ingest_live_keyframe(payload)["accepted"] is True
    assert mapper.snapshot()["points"] == len(points)


def test_registered_map_frame_points_are_not_transformed_by_the_robot_pose(monkeypatch):
    mapper = ProgressiveMapper(voxel_m=0.01)
    monkeypatch.setattr(server, "mapper", mapper)
    points = np.asarray([[1.0, 2.0, 3.0]] * 100, dtype="<f4")
    pose = np.eye(4)
    pose[:3, 3] = [20.0, 30.0, 40.0]
    payload = {
        "map_id": "test-live-map", "sequence": 1, "timestamp_ns": 1, "calibration_id": "PROVISIONAL_TEST",
        "points_f32_b64": base64.b64encode(points.tobytes()).decode("ascii"),
        "pose_map_from_lidar": pose.ravel().tolist(), "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "registration_residual_m": None, "calibrated_lio": False, "points_in_map_frame": True,
    }
    assert server.ingest_live_keyframe(payload)["accepted"] is True
    assert mapper.snapshot()["preview_points"] == [[1.0, 2.0, 3.0]]


def test_empty_dashboard_accepts_the_producer_sequence_after_a_dashboard_restart(monkeypatch):
    mapper = ProgressiveMapper()
    monkeypatch.setattr(server, "mapper", mapper)
    monkeypatch.setattr(server, "_live_ingest_queue", asyncio.Queue(maxsize=2))
    monkeypatch.setattr(server, "_live_ingest_next_sequence", None)
    monkeypatch.setattr(server, "_live_ingest_last_admitted", {})

    acknowledgement = server._queue_live_keyframe({
        "map_id": "test-live-map",
        "sequence": 107,
        "timestamp_ns": 1,
        "calibration_id": "PROVISIONAL_TEST",
    })

    assert acknowledgement["accepted"] is True
    assert acknowledgement["sequence"] == 107
