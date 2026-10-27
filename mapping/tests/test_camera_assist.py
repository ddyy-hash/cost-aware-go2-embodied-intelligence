import base64
import time
import zlib

import pytest

from mapping.pc.go2_mapping_dashboard.camera_assist import CameraAssistError, CameraAssistState


def _payload():
    depth = (1000).to_bytes(2, "little") * 4
    return {
        "sequence": 4,
        "timestamp_ns": 2_000_000_000,
        "color": {"jpeg_b64": base64.b64encode(b"\xff\xd8test\xff\xd9").decode("ascii"), "width": 2, "height": 2},
        "depth": {
            "u16_le_b64": base64.b64encode(depth).decode("ascii"),
            "width": 2,
            "height": 2,
            "scale_m": 0.001,
            "aligned_to_color": True,
        },
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
    }


def test_camera_assist_retains_only_valid_aligned_rgbd_and_returns_it_outside_snapshots():
    state = CameraAssistState()
    state.ingest(_payload(), "cal-001")

    snapshot = state.snapshot()
    frame = state.frame()

    assert snapshot["state"] == "LIVE_ALIGNED_RGBD"
    assert snapshot["distance_measurement_ready"] is True
    assert frame["sequence"] == 4
    assert frame["depth"]["aligned_to_color"] is True
    assert frame["color"]["jpeg_b64"] == _payload()["color"]["jpeg_b64"]


def test_camera_assist_rejects_unaligned_depth_dimension_mismatch():
    state = CameraAssistState()
    payload = _payload()
    payload["depth"]["width"] = 1
    payload["depth"]["u16_le_b64"] = base64.b64encode((1000).to_bytes(2, "little") * 2).decode("ascii")

    with pytest.raises(CameraAssistError, match="aligned depth"):
        state.ingest(payload, "cal-001")


def test_camera_assist_allows_unaligned_depth_only_for_depth_camera_measurement():
    state = CameraAssistState()
    payload = _payload()
    payload["depth"]["aligned_to_color"] = False
    payload["depth"]["intrinsics"] = {"fx": 90.0, "fy": 91.0, "cx": 1.0, "cy": 1.0}

    state.ingest(payload, "cal-001")
    snapshot = state.snapshot()
    frame = state.frame()

    assert snapshot["state"] == "LIVE_RGB_DEPTH_UNALIGNED"
    assert snapshot["distance_measurement_ready"] is True
    assert snapshot["measurement_view"] == "depth"
    assert frame["depth"]["intrinsics"]["fx"] == 90.0


def test_camera_assist_losslessly_restores_compressed_depth():
    state = CameraAssistState()
    payload = _payload()
    raw_depth = base64.b64decode(payload["depth"].pop("u16_le_b64"))
    payload["depth"]["u16_le_zlib_b64"] = base64.b64encode(zlib.compress(raw_depth, level=1)).decode("ascii")

    state.ingest(payload, "cal-001")

    assert base64.b64decode(state.frame()["depth"]["u16_le_b64"]) == raw_depth


def test_camera_assist_accepts_a_newer_camera_timestamp_after_producer_sequence_restart():
    state = CameraAssistState()
    first = _payload()
    first["sequence"] = 20
    first["timestamp_ns"] = 1_000
    restarted = _payload()
    restarted["sequence"] = 1
    restarted["timestamp_ns"] = 2_000

    state.ingest(first, "cal-001")
    state.ingest(restarted, "cal-001")

    assert state.frame()["sequence"] == 1
    assert state.frame()["timestamp_ns"] == 2_000


def test_camera_assist_accepts_a_new_producer_epoch_when_the_camera_clock_restarts():
    state = CameraAssistState()
    first = _payload()
    first["producer_instance_id"] = "producer-a"
    first["timestamp_ns"] = 20_000
    restarted = _payload()
    restarted["producer_instance_id"] = "producer-b"
    restarted["sequence"] = 1
    restarted["timestamp_ns"] = 1_000

    state.ingest(first, "cal-001")
    state.ingest(restarted, "cal-001")

    assert state.frame()["producer_instance_id"] == "producer-b"
    assert state.frame()["timestamp_ns"] == 1_000


def test_camera_assist_rejects_a_rolled_back_timestamp_within_one_producer_epoch():
    state = CameraAssistState()
    first = _payload()
    first["producer_instance_id"] = "producer-a"
    first["timestamp_ns"] = 20_000
    stale = _payload()
    stale["producer_instance_id"] = "producer-a"
    stale["timestamp_ns"] = 1_000

    state.ingest(first, "cal-001")
    state.ingest(stale, "cal-001")

    assert state.frame()["timestamp_ns"] == 20_000


def test_camera_assist_reports_independent_preview_frame_rate():
    state = CameraAssistState()
    for sequence in range(1, 7):
        payload = _payload()
        payload["sequence"] = sequence
        payload["timestamp_ns"] = sequence * 200_000_000
        state.ingest(payload, "cal-001", now=100.0 + sequence * 0.2)

    snapshot = state.snapshot(now=101.2)

    assert snapshot["preview_fps"] == pytest.approx(5.0)
    assert snapshot["age_ms"] == 0


def test_camera_assist_does_not_serve_an_expired_frame_as_current_preview():
    state = CameraAssistState()
    state.ingest(_payload(), "cal-001", now=time.monotonic() - 6.0)

    snapshot = state.snapshot()

    assert snapshot["state"] == "CAMERA_PREVIEW_STALE"
    assert snapshot["sequence"] == -1
    assert state.frame() is None
    assert state.wait_for_color_frame(None, timeout=0) is None


def test_camera_color_stream_returns_only_a_new_latest_frame_without_batching():
    state = CameraAssistState()
    state.ingest(_payload(), "cal-001")

    first = state.wait_for_color_frame(None, timeout=0)

    assert first is not None
    assert first["identity"] == "legacy:4"
    assert first["jpeg"] == b"\xff\xd8test\xff\xd9"
    assert state.wait_for_color_frame(first["identity"], timeout=0) is None

    newer = _payload()
    newer["sequence"] = 5
    newer["timestamp_ns"] = 2_200_000_000
    newer["color"]["jpeg_b64"] = base64.b64encode(b"\xff\xd8new\xff\xd9").decode("ascii")
    state.ingest(newer, "cal-001")

    latest = state.wait_for_color_frame(first["identity"], timeout=0)
    assert latest is not None
    assert latest["identity"] == "legacy:5"
    assert latest["jpeg"] == b"\xff\xd8new\xff\xd9"
