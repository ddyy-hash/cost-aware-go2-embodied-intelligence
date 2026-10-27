import json
import http.client
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from mapping.pc.ssh_tunnel_receiver import Receiver
from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
from mapping.robot.go2_mapping_lio.go2_mapping_lio import progressive_artifact_uploader as uploader
from mapping.robot.go2_mapping_lio.go2_mapping_lio.progressive_artifact_uploader import upload_artifact


def test_uploader_uses_derp_tolerant_backpressure_limits():
    connection = uploader._connection(urlsplit("http://127.0.0.1:19000"))

    assert uploader._SEND_BLOCK_BYTES <= 16 * 1024
    assert connection.timeout >= 300


def test_capacity_sample_excludes_intentional_token_bucket_sleep():
    capacity_elapsed = uploader._capacity_elapsed_s(wall_elapsed_s=21.0, throttle_sleep_s=20.0)
    observed_mbps = 256 * 1024 * 8.0 / capacity_elapsed / 1_000_000.0

    assert capacity_elapsed == 1.0
    assert observed_mbps > 2.0


def test_capacity_probe_is_due_immediately_then_only_after_its_interval():
    assert uploader._capacity_probe_due(last_probe_at=None, now=100.0, interval_s=30.0) is True
    assert uploader._capacity_probe_due(last_probe_at=100.0, now=129.999, interval_s=30.0) is False
    assert uploader._capacity_probe_due(last_probe_at=100.0, now=130.0, interval_s=30.0) is True


def test_parallel_ack_controller_uses_a_shared_goodput_window():
    source = uploader.upload_artifact.__code__.co_names

    assert "VerifiedAckGoodputWindow" in source


def test_dynamic_uploader_floor_preserves_preview_when_previous_state_collapsed(tmp_path, monkeypatch):
    state = tmp_path / "bandwidth.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "updated_at": 100.0,
                "route_id": "route:test",
                "total_mbps": 0.1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(uploader.time, "time", lambda: 100.0)
    store = uploader.BandwidthAllocationStore(state, max_age_s=30.0)
    previous = store.read(expected_route_id="route:test")
    assert previous is not None and previous.total_mbps == pytest.approx(0.1)
    minimum = float("4.0")
    controller = uploader.AdaptiveBandwidthController(
        ceiling_mbps=120.0,
        floor_mbps=minimum,
        initial_total_mbps=max(previous.total_mbps, minimum),
    )
    assert controller.budget.total_mbps == pytest.approx(4.0)
    assert controller.budget.preview_mbps == pytest.approx(0.0)


def test_artifact_uploader_resumes_idempotently_and_seals_a_multichunk_file(tmp_path):
    previous_root = Receiver.root
    previous_status_path = getattr(Receiver, "status_path", None)
    Receiver.root = tmp_path / "inbox"
    Receiver.status_path = Receiver.root / ".progressive-status.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "source.db3"
    source.write_bytes(b"0123456789" * 733)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        first = upload_artifact(
            endpoint,
            session_id="session-100",
            source=source,
            relative_path="lidar/segment-100.db3",
            chunk_size_bytes=1024,
            rate_mbps=100.0,
        )
        second = upload_artifact(
            endpoint,
            session_id="session-100",
            source=source,
            relative_path="lidar/segment-100.db3",
            chunk_size_bytes=1024,
            rate_mbps=100.0,
        )

        assert first["duplicate"] is False
        assert second["duplicate"] is True
        assert first["chunk_count"] == 8
        assert (Receiver.root / "lidar" / "segment-100.db3").read_bytes() == source.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
        Receiver.status_path = previous_status_path


def test_artifact_uploader_publishes_measured_dynamic_bandwidth_for_preview(tmp_path):
    previous_root = Receiver.root
    previous_status_path = getattr(Receiver, "status_path", None)
    Receiver.root = tmp_path / "inbox"
    Receiver.status_path = Receiver.root / ".progressive-status.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "source.db3.zstd"
    source.write_bytes(b"adaptive-bandwidth" * 4096)
    bandwidth_state = tmp_path / "runtime" / "bandwidth-allocation.json"
    preview_demand_state = tmp_path / "runtime" / "preview-demand.json"
    uploader.PreviewDemandStore(preview_demand_state).publish(
        required_mbps=1.25,
        payload_bytes=78_125,
        cadence_hz=2.0,
    )
    try:
        result = upload_artifact(
            f"http://127.0.0.1:{server.server_port}",
            session_id="session-adaptive",
            source=source,
            relative_path="lidar/segment-adaptive.db3.zstd",
            chunk_size_bytes=16 * 1024,
            rate_mbps=100.0,
            bandwidth_state_path=bandwidth_state,
            preview_demand_state_path=preview_demand_state,
            total_ceiling_mbps=120.0,
            route_id="wifi:lab|tailscale:direct",
            parallel_workers=3,
        )

        assert result["bandwidth"]["dynamic"] is True
        assert result["bandwidth"]["preview_policy"] == "realtime_preview_priority_high_fidelity_remaining_capacity"
        assert result["bandwidth"]["preview_mbps"] == pytest.approx(1.25)
        assert result["bandwidth"]["high_fidelity_mbps"] == pytest.approx(
            result["bandwidth"]["total_mbps"] - 1.25
        )
        assert result["bandwidth"]["route_id"] == "wifi:lab|tailscale:direct"
        assert result["parallel_workers"] == 3
        assert result["capacity_probe"]["unthrottled"] is True
        assert bandwidth_state.is_file()
        receiver_status = json.loads(Receiver.status_path.read_text(encoding="utf-8"))["high_fidelity"]
        assert receiver_status["observed_capacity_mbps"] > 0
        assert receiver_status["total_budget_mbps"] > 0
        total_budget = receiver_status["total_budget_mbps"]
        expected_preview = 1.25
        assert receiver_status["preview_budget_mbps"] == pytest.approx(expected_preview, abs=1e-6)
        assert receiver_status["high_fidelity_budget_mbps"] == pytest.approx(
            total_budget - expected_preview, abs=1e-6
        )
        lifecycle = ArtifactLifecycleStore(Receiver.root)
        artifact_id = uploader._sha256(source)
        verified = lifecycle.status("session-adaptive", artifact_id)
        assert verified["state"] == "SHA256_VERIFIED"
        assert verified["release_authorized"] is False
        target = Receiver.root / "lidar" / "segment-adaptive.db3.zstd"
        lifecycle.mark_reconstructed(target, result={"integrated_frames": 1, "integrated_points": 1})
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", f"/progressive/artifacts/session-adaptive/{artifact_id}/status")
        status_response = connection.getresponse()
        status = json.loads(status_response.read())
        connection.close()
        assert status_response.status == 200
        assert status["release_authorized"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
        Receiver.status_path = previous_status_path


def test_receiver_rejects_clients_outside_the_explicit_tailnet_allowlist(tmp_path):
    previous_root = Receiver.root
    previous_allowed = Receiver.allowed_clients
    Receiver.root = tmp_path / "inbox"
    Receiver.allowed_clients = frozenset({"192.0.2.11"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        connection.close()
        assert response.status == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
        Receiver.allowed_clients = previous_allowed
