from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapping.pc.go2_mapping_dashboard.room_scan import RoomScanError, RoomScanService


class _Response:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def _service(tmp_path: Path, opener, *, expires_at: int = 2_000) -> RoomScanService:
    operator = tmp_path / "operator.json"
    operator.write_text(json.dumps({"schema_version": 1, "token": "operator-token", "expires_at_unix": expires_at}), encoding="utf-8")
    control = tmp_path / "session-control-token"
    control.write_text("a" * 32, encoding="ascii")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"calibration_id": "PROVISIONAL_AUTOCAL_PENDING_v1"}), encoding="utf-8")
    return RoomScanService(
        operator_token_path=operator,
        session_control_token_path=control,
        calibration_manifest_path=calibration,
        readiness=lambda: {"ready": True},
        runtime_calibration=lambda: {
            "formal_runtime_override": True,
            "measurement_authority": False,
            "camera_from_lidar_row_major": [1.0] * 16,
        },
        session_snapshot=lambda: {"capture_active": False, "state": "UNKNOWN"},
        opener=opener,
        clock=lambda: 1_000,
    )


def test_room_scan_service_posts_only_fixed_local_capture_contract(tmp_path):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/health"):
            return _Response(200, {"progressive_protocol": "sha256-chunk-v1", "capacity_probe": True})
        if request.full_url.endswith("/v1/preflight"):
            return _Response(200, {"ready": True})
        if request.full_url.endswith("/v1/sessions"):
            return _Response(201, {"state": "RUNNING"})
        raise AssertionError(request.full_url)

    result = _service(tmp_path, opener).start(600, "operator-token")

    payload = json.loads(requests[-1].data.decode("utf-8"))
    assert payload["mode"] == "full"
    assert payload["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert payload["duration_seconds"] == 600
    assert payload["pc_tailnet_ip"] == "192.0.2.10"
    assert result["geometry_only_calibration"] is True
    assert result["measurement_authority"] is False


def test_room_scan_service_accepts_measured_geometry_runtime(tmp_path):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/health"):
            return _Response(200, {"progressive_protocol": "sha256-chunk-v1", "capacity_probe": True})
        if request.full_url.endswith("/v1/preflight"):
            return _Response(200, {"ready": True})
        if request.full_url.endswith("/v1/sessions"):
            return _Response(201, {"state": "RUNNING"})
        raise AssertionError(request.full_url)

    service = _service(tmp_path, opener)
    service.runtime_calibration = lambda: {
        "formal_runtime_override": False,
        "measurement_authority": True,
        "runtime_authority": "MEASURED",
        "camera_from_lidar_row_major": [1.0] * 16,
    }

    result = service.start(600, "operator-token")

    assert result["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert result["geometry_only_calibration"] is True


def test_room_scan_service_rejects_expired_operator_token_before_transport(tmp_path):
    def opener(*_args, **_kwargs):
        raise AssertionError("expired token must reject before a network request")

    service = _service(tmp_path, opener, expires_at=999)

    with pytest.raises(RoomScanError, match="expired"):
        service.start(600, "operator-token")


def test_room_scan_service_reads_power_shell_utf8_token_and_controls_exact_request(tmp_path):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/v1/sessions/request-1"):
            return _Response(200, {"request_id": "request-1", "state": "RUNNING"})
        if request.full_url.endswith("/v1/sessions/request-1/stop"):
            return _Response(200, {"request_id": "request-1", "state": "STOPPING"})
        raise AssertionError(request.full_url)

    operator = tmp_path / "operator.json"
    operator.write_bytes(b"\xef\xbb\xbf" + json.dumps({"token": "operator-token", "expires_at_unix": 2_000}).encode("utf-8"))
    service = _service(tmp_path, opener)
    service.active_request_id = "request-1"

    assert service.status("request-1", "operator-token")["state"] == "RUNNING"
    stopped = service.stop("request-1", "operator-token")

    assert stopped["state"] == "STOPPING"
    assert service.active_request_id is None
    assert requests[-1].full_url.endswith("/v1/sessions/request-1/stop")


def test_room_scan_capabilities_hide_missing_or_expired_operator_tokens(tmp_path):
    service = _service(tmp_path, lambda *_args, **_kwargs: None, expires_at=999)

    capabilities = service.capabilities()

    assert capabilities["enabled"] is False
    assert capabilities["coordinate_scope"] == "SESSION_ODOM_LOCAL"


def test_room_scan_default_transport_bypasses_ambient_proxies(tmp_path, monkeypatch):
    built = []

    class _Opener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("transport is not used in this constructor test")

    def build_opener(handler):
        built.append(handler)
        return _Opener()

    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    service = _service(tmp_path, lambda *_args, **_kwargs: None)
    service.opener = None
    service = RoomScanService(
        operator_token_path=None,
        session_control_token_path=None,
        calibration_manifest_path=None,
        readiness=lambda: {"ready": True},
        runtime_calibration=lambda: {},
        session_snapshot=lambda: {},
    )

    assert len(built) == 1
    assert isinstance(built[0], __import__("urllib.request", fromlist=["ProxyHandler"]).ProxyHandler)
    assert service.opener is not None


def test_room_scan_allocates_a_session_id_that_does_not_reuse_existing_capture(tmp_path):
    existing = tmp_path / "mapping-data" / "inbox" / "lidar-imu-rgbd" / "segment-19700101T000000Z"
    existing.mkdir(parents=True)
    calibration = tmp_path / "mapping" / "config" / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text(json.dumps({"calibration_id": "measured"}), encoding="utf-8")
    service = RoomScanService(
        operator_token_path=None,
        session_control_token_path=None,
        calibration_manifest_path=calibration,
        readiness=lambda: {"ready": True},
        runtime_calibration=lambda: {},
        session_snapshot=lambda: {},
        clock=lambda: 0,
    )

    assert service._new_session_id() == "segment-19700101T000001Z"
    assert service._new_session_id() != "segment-19700101T000001Z"
