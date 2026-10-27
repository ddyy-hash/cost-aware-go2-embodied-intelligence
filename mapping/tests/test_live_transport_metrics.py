import math

from mapping.pc.go2_mapping_dashboard.live_transport import LiveTransportMetrics


def test_transport_metrics_report_measured_ingress_rate_ack_rtt_and_freshness():
    metrics = LiveTransportMetrics(window_seconds=5.0)

    metrics.record(payload_bytes=1_000_000, acknowledgement_rtt_ms=None, received_at=10.0)
    metrics.record(payload_bytes=1_000_000, acknowledgement_rtt_ms=84.5, received_at=11.0)
    state = metrics.snapshot(now=11.0)

    assert math.isclose(state["app_goodput_mbps"], 16.0)
    assert state["keyframe_hz"] == 2.0
    assert state["ack_rtt_ms"] == 84.5
    assert state["last_frame_age_ms"] == 0
    assert state["duplicate_frames"] == 0


def test_transport_metrics_leave_unknown_values_unset_before_the_first_frame():
    state = LiveTransportMetrics().snapshot(now=10.0)

    assert state["app_goodput_mbps"] is None
    assert state["keyframe_hz"] is None
    assert state["ack_rtt_ms"] is None
    assert state["last_frame_age_ms"] is None


def test_transport_metrics_keep_last_frame_age_after_the_rate_window_expires():
    metrics = LiveTransportMetrics(window_seconds=5.0)
    metrics.record(payload_bytes=1_000, acknowledgement_rtt_ms=20.0, received_at=10.0)

    state = metrics.snapshot(now=18.0)

    assert state["app_goodput_mbps"] is None
    assert state["keyframe_hz"] is None
    assert state["last_frame_age_ms"] == 8000


def test_transport_metrics_exposes_depth_only_deferral_without_claiming_camera_failure():
    metrics = LiveTransportMetrics()

    metrics.record(
        payload_bytes=1_000,
        acknowledgement_rtt_ms=None,
        camera_depth_deferred_by_bandwidth=True,
    )

    state = metrics.snapshot()
    assert state["camera_deferred_by_bandwidth"] is False
    assert state["camera_depth_deferred_by_bandwidth"] is True
