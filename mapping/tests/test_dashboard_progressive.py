import json
from pathlib import Path

from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_dashboard.progressive_transfer import ProgressiveTransferTelemetry


def test_dashboard_diagnostics_fallback_is_single_flight_and_websocket_first():
    source = (
        Path(__file__).parents[1]
        / "pc"
        / "go2_mapping_dashboard"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "snapshotRequestInFlight" in source
    assert "socketIsFresh" in source
    assert "liveSocketConnected = true" in source
    assert "if (diagnosticsDrawer.open) refreshSnapshot(true);" in source
    assert "if (!diagnosticsDrawer.open) return;" in source
    assert 'fetch("/api/diagnostics", {cache:\'no-store\'})' in source
    assert "setInterval(refreshSnapshot" not in source


def test_dashboard_uses_a_15_hz_point_free_telemetry_path_between_global_snapshots():
    source = (
        Path(__file__).parents[1]
        / "pc"
        / "go2_mapping_dashboard"
        / "server.py"
    ).read_text(encoding="utf-8")
    app = (
        Path(__file__).parents[1]
        / "pc"
        / "go2_mapping_dashboard"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'GO2_MAPPING_DASHBOARD_TELEMETRY_HZ", "15"' in source
    assert "mapper.telemetry_snapshot()" in source
    assert "await asyncio.sleep(1.0 / _dashboard_telemetry_hz)" in source
    assert "Telemetry is intentionally point-free at 15 Hz" in app
    assert "measured_preview_points" in app
    assert "unstable/single-view voxels make a wall look like a solid fog" in app
    assert "measuredCoverageReady" in app


def test_dashboard_prioritizes_a_fresh_global_live_room_scan_over_stale_dense_artifacts():
    app = (
        Path(__file__).parents[1]
        / "pc"
        / "go2_mapping_dashboard"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "const roomScan = snapshot.room_scan || live.room_scan || {};" in app
    assert "const liveRoomMapReady = liveRenderEnabled && liveFresh && Number(live.points || 0) > 0;" in app
    assert "const activeMap = liveRoomMapReady ? live : hasDenseMap ? dense : live;" in app
    assert "LIVE ROOM SCAN" in app


def test_dashboard_snapshot_exposes_dense_lidar_separately_from_the_low_bandwidth_preview(monkeypatch):
    class DenseMapper:
        def snapshot(self):
            return {
                "quality": "DENSE_LIDAR_PREVIEW_UNCALIBRATED",
                "points": 42_000,
                "artifacts": 4,
                "frames": 80,
                "preview_points": [[0.0, 0.0, 0.0]],
            }

    class Ingestor:
        def snapshot(self):
            return {"integrated_artifacts": 4, "integrated_points": 500_000, "errors": 0}

    monkeypatch.setattr(server, "dense_mapper", DenseMapper())
    monkeypatch.setattr(server, "artifact_ingestor", Ingestor())

    snapshot = server.dashboard_snapshot()

    assert snapshot["dense_map"]["points"] == 42_000
    assert snapshot["progressive_ingest"]["integrated_points"] == 500_000
    assert snapshot["mapping"]["dense_points"] == 42_000
    assert snapshot["mapping"]["points"] == 42_000


def test_measured_surface_status_is_no_store_and_avoids_expected_423(monkeypatch):
    class DenseMapper:
        def __init__(self):
            self.surface = None

        def measured_surface_snapshot(self):
            return self.surface

    mapper = DenseMapper()
    monkeypatch.setattr(server, "dense_mapper", mapper)

    unavailable = server.measured_surface_status()
    assert unavailable.status_code == 200
    assert unavailable.headers["cache-control"] == "no-store"
    assert json.loads(unavailable.body) == {
        "schema_version": 1,
        "available": False,
        "blocker": "LIDAR_MEASURED_SURFACE_UNAVAILABLE",
    }

    mapper.surface = {"schema_version": 1}
    available = server.measured_surface_status()
    assert json.loads(available.body) == {
        "schema_version": 1,
        "available": True,
        "blocker": None,
    }


def test_progressive_transfer_telemetry_loads_only_receiver_written_high_fidelity_stats(tmp_path):
    path = tmp_path / "progressive-transfer-status.json"
    path.write_text(
        json.dumps(
            {
                "high_fidelity": {
                    "state": "CHECKSUM_VERIFIED",
                    "app_goodput_mbps": 92.4,
                    "verified_artifacts": 8,
                    "verified_bytes": 123456,
                    "updated_at_ms": 1_000_000,
                    "total_budget_mbps": 40.0,
                    "preview_budget_mbps": 2.0,
                    "high_fidelity_budget_mbps": 38.0,
                    "observed_capacity_mbps": 44.0,
                }
            }
        ),
        encoding="utf-8",
    )

    state = ProgressiveTransferTelemetry(path, max_age_s=30.0).snapshot(now_ms=1_005_000)

    assert state["state"] == "CHECKSUM_VERIFIED"
    assert state["app_goodput_mbps"] == 92.4
    assert state["verified_artifacts"] == 8
    assert state["active"] is True
    assert state["total_budget_mbps"] == 40.0
    assert state["preview_budget_mbps"] == 2.0
    assert state["high_fidelity_budget_mbps"] == 38.0
    assert state["observed_capacity_mbps"] == 44.0


def test_progressive_transfer_clears_stale_rates_but_preserves_durable_counts(tmp_path):
    path = tmp_path / "progressive-transfer-status.json"
    path.write_text(
        json.dumps({"high_fidelity": {"state": "RECEIVING_CHUNKS", "app_goodput_mbps": 8.389,
                    "accepted_chunks": 9, "verified_artifacts": 2, "updated_at_ms": 1_000_000}}),
        encoding="utf-8",
    )

    state = ProgressiveTransferTelemetry(path, max_age_s=10.0).snapshot(now_ms=1_020_001)

    assert state["active"] is False
    assert state["state"] == "STALE"
    assert state["app_goodput_mbps"] is None
    assert state["accepted_chunks"] == 9
    assert state["verified_artifacts"] == 2
