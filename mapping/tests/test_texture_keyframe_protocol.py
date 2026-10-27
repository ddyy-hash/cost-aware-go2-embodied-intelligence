import base64
import asyncio
import hashlib
import json
import zlib
from types import SimpleNamespace

import pytest

from mapping.pc.go2_mapping_live.live_protocol import (
    LiveProtocolError,
    build_texture_keyframe_ack,
    validate_texture_keyframe,
)
from mapping.pc.go2_mapping_live import texture_wal as texture_wal_module
from mapping.pc.go2_mapping_live.texture_wal import TextureKeyframeWal
from mapping.pc.go2_mapping_dashboard import server


def test_texture_ingest_loop_batches_backlog_and_refreshes_each_building_once(monkeypatch):
    records = [
        SimpleNamespace(
            frame=SimpleNamespace(
                identity=SimpleNamespace(building_id="building-a"),
            ),
            sequence=index,
        )
        for index in range(10)
    ]

    class BackloggedWal:
        def pending(self):
            return records

    async def exercise():
        processed = []
        refreshed = []
        refresh_complete = asyncio.Event()
        loop = asyncio.get_running_loop()

        def process(record):
            processed.append(record.sequence)

        def refresh(building_id):
            refreshed.append(building_id)
            loop.call_soon_threadsafe(refresh_complete.set)

        monkeypatch.setattr(server, "_texture_wal", BackloggedWal())
        monkeypatch.setattr(server, "_texture_ingest_batch_size", 3)
        monkeypatch.setattr(server, "_texture_render_refresh_interval_s", 60.0)
        monkeypatch.setattr(server, "_texture_render_next_refresh", 0.0)
        monkeypatch.setattr(server, "_texture_render_pending_buildings", set())
        monkeypatch.setattr(server, "_process_texture_record", process)
        monkeypatch.setattr(server, "_refresh_render_manifest", refresh)

        task = asyncio.create_task(server._texture_ingest_loop())
        try:
            await asyncio.wait_for(refresh_complete.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert processed == [0, 1, 2]
        assert refreshed == ["building-a"]

    asyncio.run(exercise())


def texture_payload(
    sequence: int = 7,
    *,
    producer_epoch: int = 11,
    color_bytes: bytes | None = None,
) -> dict:
    color = color_bytes or (b"\xff\xd8" + b"go2-texture-frame" * 32 + b"\xff\xd9")
    return {
        "message_type": "texture_keyframe",
        "protocol_version": 3,
        "building_id": "lab-building",
        "session_id": "scan-session",
        "map_id": "go2-live",
        "producer_instance_id": "texture-producer-a",
        "producer_epoch": producer_epoch,
        "texture_sequence": sequence,
        "timestamp_ns": 1_800_000_000 + sequence,
        "calibration_id": "calibration-a",
        "source_config_sha256": "a" * 64,
        "content_sha256": hashlib.sha256(color).hexdigest(),
        "source": {
            "topic": "/camera/color/image_raw",
            "frame_id": "camera_color_optical_frame",
            "encoding": "bgr8",
            "width": 424,
            "height": 240,
        },
        "color": {
            "encoding": "jpeg-base64",
            "jpeg_b64": base64.b64encode(color).decode("ascii"),
            "width": 424,
            "height": 240,
        },
        "intrinsics": {
            "fx": 305.2,
            "fy": 304.9,
            "cx": 211.5,
            "cy": 119.5,
            "width": 424,
            "height": 240,
        },
        "lio_pose_timestamp_ns": 1_800_000_000 + sequence,
        "pose_skew_ns": 0,
        "pose_association_clock": "subscriber_monotonic",
        "pose_map_from_lidar": [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "quality": {
            "selection_accepted": True,
            "sharpness_laplacian": 84.5,
            "mean_luma": 112.0,
            "clipped_fraction": 0.01,
            "viewpoint_baseline_m": 0.18,
            "independent_view_id": f"view-{sequence}",
        },
    }


def test_texture_keyframe_validates_exact_content_source_pose_and_optional_depth():
    payload = texture_payload()
    depth_compressed = zlib.compress((b"\x20\x03" * (424 * 240)), level=1)
    payload["depth"] = {
        "encoding": "u16-le-zlib-base64",
        "data_b64": base64.b64encode(depth_compressed).decode("ascii"),
        "compressed_bytes_sha256": hashlib.sha256(depth_compressed).hexdigest(),
        "width": 424,
        "height": 240,
        "scale_m": 0.001,
        "aligned_to_color": True,
    }

    frame = validate_texture_keyframe(payload)

    assert frame.identity.producer_epoch == 11
    assert frame.identity.sequence == 7
    assert frame.color_jpeg.startswith(b"\xff\xd8")
    assert frame.source_dimensions == (424, 240)
    assert frame.intrinsics["width"] == 424
    assert frame.depth_u16_bytes == b"\x20\x03" * (424 * 240)
    assert frame.pose_map_from_lidar.shape == (4, 4)
    acknowledgement = build_texture_keyframe_ack(
        frame,
        accepted=True,
        disposition="QUEUED_NEW",
        wal_record_sha256="b" * 64,
        next_sequence=8,
    )
    assert acknowledgement["message_type"] == "texture_keyframe_ack"
    assert acknowledgement["producer_epoch"] == 11
    assert acknowledgement["texture_sequence"] == 7
    assert acknowledgement["ack_stage"] == "QUEUED"


def test_texture_keyframe_rejects_content_tampering_bad_source_identity_and_unselected_frames():
    checksum = texture_payload()
    checksum["content_sha256"] = "0" * 64
    with pytest.raises(LiveProtocolError, match="content checksum") as error:
        validate_texture_keyframe(checksum)
    assert error.value.error_code == "TEXTURE_CHECKSUM_MISMATCH"

    dimensions = texture_payload()
    dimensions["source"]["width"] = 640
    with pytest.raises(LiveProtocolError, match="source dimensions"):
        validate_texture_keyframe(dimensions)

    rejected_quality = texture_payload()
    rejected_quality["quality"]["selection_accepted"] = False
    with pytest.raises(LiveProtocolError, match="quality selection"):
        validate_texture_keyframe(rejected_quality)


def test_texture_wal_is_content_addressed_ordered_idempotent_and_epoch_isolated(tmp_path):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)
    frame_7 = validate_texture_keyframe(texture_payload(7))
    frame_8 = validate_texture_keyframe(texture_payload(8, color_bytes=b"\xff\xd8next\xff\xd9"))

    first = wal.admit(frame_7)
    duplicate = wal.admit(frame_7)
    gap = wal.admit(validate_texture_keyframe(texture_payload(9, color_bytes=b"\xff\xd8gap\xff\xd9")))
    second = wal.admit(frame_8)

    assert first.disposition == "QUEUED_NEW"
    assert duplicate.disposition == "DUPLICATE_QUEUED"
    assert gap.error_code == "SEQUENCE_GAP"
    assert second.disposition == "QUEUED_NEW"
    assert (wal.blobs_root / frame_7.content_sha256[:2] / f"{frame_7.content_sha256}.jpg").read_bytes() == frame_7.color_jpeg
    assert [record.frame.identity.sequence for record in wal.pending()] == [7, 8]

    wal.commit_integrated(frame_7, checkpoint_id="provisional-texture-1")
    restarted = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)
    assert [record.frame.identity.sequence for record in restarted.pending()] == [8]

    new_epoch = validate_texture_keyframe(texture_payload(0, producer_epoch=12))
    assert restarted.admit(new_epoch).disposition == "QUEUED_NEW"
    assert sorted(record.frame.identity.sequence for record in restarted.pending()) == [0, 8]


def test_texture_wal_records_include_integrated_and_pending_in_full_stream_order(tmp_path):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)
    frame_7 = validate_texture_keyframe(texture_payload(7))
    frame_8 = validate_texture_keyframe(
        texture_payload(8, color_bytes=b"\xff\xd8next\xff\xd9")
    )
    next_epoch = validate_texture_keyframe(texture_payload(0, producer_epoch=12))
    assert wal.admit(next_epoch).accepted is True
    assert wal.admit(frame_7).accepted is True
    assert wal.admit(frame_8).accepted is True
    wal.commit_integrated(frame_7, checkpoint_id="provisional-texture-7")

    records = wal.records()
    record_keys = [
        (*record.frame.identity.stream_key(), record.frame.identity.sequence)
        for record in records
    ]

    assert record_keys == sorted(record_keys)
    assert {(record.frame.identity.producer_epoch, record.frame.identity.sequence) for record in records} == {
        (11, 7),
        (11, 8),
        (12, 0),
    }
    assert {
        (record.frame.identity.producer_epoch, record.frame.identity.sequence)
        for record in wal.pending()
    } == {(11, 8), (12, 0)}

    corrupt_path = next(record.path for record in records if record.frame.identity.sequence == 8)
    corrupt_path.write_text("{not-json", encoding="utf-8")
    assert all(record.path != corrupt_path for record in wal.records())
    assert all(record.path != corrupt_path for record in wal.pending())


def test_texture_wal_reclaim_fsyncs_saved_record_parents_and_never_deletes_blobs(
    monkeypatch,
    tmp_path,
):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)
    frame = validate_texture_keyframe(texture_payload(7))
    assert wal.admit(frame).accepted is True
    record = wal.pending()[0]
    color_blob = wal.blobs_root / frame.content_sha256[:2] / f"{frame.content_sha256}.jpg"
    wal.commit_integrated(frame, checkpoint_id="provisional-texture-7")
    fsynced_directories = []
    monkeypatch.setattr(
        texture_wal_module,
        "_fsync_directory",
        fsynced_directories.append,
    )

    wal.reclaim_integrated(frame.identity.stream_key())

    assert record.path.exists() is False
    assert fsynced_directories == [record.path.parent]
    assert color_blob.read_bytes() == frame.color_jpeg


def test_texture_wal_rejects_conflicts_and_capacity_before_acknowledgement(tmp_path):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=1)
    frame = validate_texture_keyframe(texture_payload())
    admission = wal.admit(frame)

    assert admission.accepted is False
    assert admission.error_code == "WAL_CAPACITY_EXCEEDED"
    assert not list(wal.records_root.rglob("*.json"))
    assert not list(wal.blobs_root.rglob("*.jpg"))


def test_texture_dispatch_acks_only_after_its_independent_wal_admission(monkeypatch, tmp_path):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)

    class LiveWalMustNotBeUsed:
        def admit(self, _frame):
            raise AssertionError("texture admission must not touch the LiDAR WAL")

    monkeypatch.setattr(server, "_texture_wal", wal)
    monkeypatch.setattr(server, "_live_wal", LiveWalMustNotBeUsed())
    payload = texture_payload(21)
    acknowledgement = server._dispatch_texture_message(
        payload,
        message_bytes=len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    )

    assert acknowledgement["accepted"] is True
    assert acknowledgement["ack_stage"] == "QUEUED"
    assert acknowledgement["texture_sequence"] == 21
    assert acknowledgement["wal_record_sha256"]
    assert [record.frame.identity.sequence for record in wal.pending()] == [21]


def test_texture_websocket_and_readiness_advertise_an_independent_tailnet_lane(monkeypatch, tmp_path):
    monkeypatch.delenv("GO2_MAPPING_LIVE_TOKEN", raising=False)
    monkeypatch.setattr(
        server,
        "_texture_wal",
        TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000),
    )

    class OneTextureFrameSocket:
        headers = {}

        def __init__(self):
            self.accepted = False
            self.sent = []
            self._received = False

        async def accept(self):
            self.accepted = True

        async def receive_text(self):
            if self._received:
                raise server.WebSocketDisconnect()
            self._received = True
            return json.dumps(texture_payload(0), separators=(",", ":"))

        async def send_json(self, payload):
            self.sent.append(payload)

    socket = OneTextureFrameSocket()
    asyncio.run(server.live_texture_websocket(socket))
    readiness = asyncio.run(server.live_readiness())

    assert socket.accepted is True
    assert socket.sent[0]["message_type"] == "texture_keyframe_ack"
    assert readiness["transport"] == "tailnet-websocket"
    assert readiness["texture_keyframes"] == {
        "endpoint": "/ws/live/texture",
        "message_type": "texture_keyframe",
        "ack_type": "texture_keyframe_ack",
        "independent_wal": True,
        "max_message_bytes": 4 * 1024 * 1024,
    }
    assert "ssh" not in json.dumps(readiness).lower()
