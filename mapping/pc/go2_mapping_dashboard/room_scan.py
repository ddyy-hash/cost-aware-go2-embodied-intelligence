"""Authenticated direct Tailnet control for the local single-room capture."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid


class RoomScanError(RuntimeError):
    status_code = 503


class RoomScanUnauthorized(RoomScanError):
    status_code = 401


class RoomScanInvalid(RoomScanError):
    status_code = 422


class RoomScanConflict(RoomScanError):
    status_code = 409


class RoomScanService:
    def __init__(
        self,
        *,
        operator_token_path: str | Path | None,
        session_control_token_path: str | Path | None,
        calibration_manifest_path: str | Path | None,
        robot_tailnet_ip: str = "192.0.2.11",
        pc_tailnet_ip: str = "192.0.2.10",
        control_port: int = 8788,
        readiness: Callable[[], dict[str, Any]],
        runtime_calibration: Callable[[], dict[str, Any]],
        session_snapshot: Callable[[], dict[str, Any]],
        receiver_health_url: str = "http://127.0.0.1:8000/health",
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.operator_token_path = self._path(operator_token_path)
        self.session_control_token_path = self._path(session_control_token_path)
        self.calibration_manifest_path = self._path(calibration_manifest_path)
        self.robot_tailnet_ip = robot_tailnet_ip
        self.pc_tailnet_ip = pc_tailnet_ip
        self.control_port = int(control_port)
        self.readiness = readiness
        self.runtime_calibration = runtime_calibration
        self.session_snapshot = session_snapshot
        self.receiver_health_url = receiver_health_url
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self.clock = clock
        self.active_request_id: str | None = None
        self._issued_session_ids: set[str] = set()

    def capabilities(self) -> dict[str, Any]:
        enabled = False
        if self.operator_token_path and self.operator_token_path.is_file():
            try:
                credential = json.loads(self.operator_token_path.read_text(encoding="utf-8-sig"))
                enabled = (
                    isinstance(credential, dict)
                    and isinstance(credential.get("token"), str)
                    and bool(credential.get("token"))
                    and isinstance(credential.get("expires_at_unix"), (int, float))
                    and not isinstance(credential.get("expires_at_unix"), bool)
                    and float(credential["expires_at_unix"]) > self.clock()
                )
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                enabled = False
        return {
            "enabled": enabled,
            "coordinate_scope": "SESSION_ODOM_LOCAL",
            "mode": "full",
            "geometry_only_calibration": True,
            "read_only": True,
        }

    @staticmethod
    def _path(value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        return Path(value).resolve(strict=False)

    @staticmethod
    def _json_bytes(payload: dict[str, Any]) -> bytes:
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    def _read_json_file(self, path: Path | None, label: str) -> dict[str, Any]:
        if path is None or not path.is_file():
            raise RoomScanError(f"{label} is not configured")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RoomScanError(f"{label} is unreadable") from exc
        if not isinstance(payload, dict):
            raise RoomScanError(f"{label} is invalid")
        return payload

    def _authorize(self, supplied: str) -> None:
        credential = self._read_json_file(self.operator_token_path, "room scan operator token")
        expected = credential.get("token")
        expires_at = credential.get("expires_at_unix")
        if not isinstance(expected, str) or not expected:
            raise RoomScanError("room scan operator token is invalid")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)) or float(expires_at) <= self.clock():
            raise RoomScanError("room scan operator token has expired")
        if not isinstance(supplied, str) or not hmac.compare_digest(expected, supplied):
            raise RoomScanUnauthorized("room scan operator token is invalid")

    def _control_token(self) -> str:
        path = self.session_control_token_path
        if path is None or not path.is_file():
            raise RoomScanError("session-control token is not configured")
        try:
            token = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise RoomScanError("session-control token is unreadable") from exc
        if not 32 <= len(token) <= 512:
            raise RoomScanError("session-control token is invalid")
        return token

    def _http_json(self, method: str, url: str, *, token: str | None = None, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        body = None if payload is None else self._json_bytes(payload)
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with self.opener(request, timeout=15) as response:
                raw = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
        except (OSError, urllib.error.URLError) as exc:
            raise RoomScanError(f"room scan transport failed: {exc}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RoomScanError(f"room scan endpoint returned invalid JSON: {url}") from exc
        if not isinstance(decoded, dict):
            raise RoomScanError("room scan endpoint returned an invalid payload")
        return status, decoded

    def _assert_local_ready(self) -> None:
        local = self.readiness()
        if not isinstance(local, dict) or local.get("ready") is not True:
            raise RoomScanError("Dashboard readiness is incomplete")
        receiver_status, receiver = self._http_json("GET", self.receiver_health_url)
        if receiver_status != 200 or receiver.get("progressive_protocol") != "sha256-chunk-v1" or receiver.get("capacity_probe") is not True:
            raise RoomScanError("verified high-fidelity receiver is not ready")
        runtime = self.runtime_calibration()
        measured_geometry = (
            runtime.get("measurement_authority") is True
            and runtime.get("formal_runtime_override") is False
            and runtime.get("runtime_authority") == "MEASURED"
        )
        provisional_override = (
            runtime.get("measurement_authority") is False
            and runtime.get("formal_runtime_override") is True
        )
        if not measured_geometry and not provisional_override:
            raise RoomScanError("runtime calibration is neither measured geometry nor an explicit provisional override")
        if len(runtime.get("camera_from_lidar_row_major") or []) != 16:
            raise RoomScanError("runtime calibration transform is unavailable")
        session = self.session_snapshot()
        if session.get("capture_active") is True or session.get("state") in {"STARTING", "RECORDING", "FINALIZING"}:
            raise RoomScanConflict("another mapping session is already active")

    def _robot_url(self, path: str) -> str:
        return f"http://{self.robot_tailnet_ip}:{self.control_port}{path}"

    def _new_session_id(self) -> str:
        """Choose a timestamp-shaped ID that cannot reuse a local capture."""
        roots: list[Path] = []
        if self.calibration_manifest_path is not None:
            workspace = self.calibration_manifest_path.parents[2]
            roots.extend(
                (
                    workspace / "mapping-data" / "inbox" / "lidar-imu-rgbd",
                    workspace / "mapping-data" / "sessions",
                )
            )
        base = int(self.clock())
        for offset in range(0, 7200):
            candidate = "segment-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(base + offset))
            if candidate in self._issued_session_ids:
                continue
            if any((root / candidate).exists() for root in roots):
                continue
            self._issued_session_ids.add(candidate)
            return candidate
        raise RoomScanError("unable to allocate a unique room scan session id")

    def start(self, duration_seconds: int, operator_token: str) -> dict[str, Any]:
        self._authorize(operator_token)
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int) or not 60 <= duration_seconds <= 3600:
            raise RoomScanInvalid("duration_seconds must be between 60 and 3600")
        if self.active_request_id:
            try:
                status = self.status(self.active_request_id, operator_token)
            except RoomScanError:
                status = None
            if status and status.get("state") in {"RUNNING", "STOPPING", "FINALIZING"}:
                raise RoomScanConflict("another room scan is already active")
            self.active_request_id = None
        self._assert_local_ready()
        token = self._control_token()
        try:
            calibration = self._read_json_file(self.calibration_manifest_path, "calibration manifest")
            calibration_id = calibration["calibration_id"]
        except KeyError as exc:
            raise RoomScanError("calibration manifest has no calibration_id") from exc
        try:
            calibration_sha256 = hashlib.sha256(self.calibration_manifest_path.read_bytes()).hexdigest()  # type: ignore[union-attr]
        except OSError as exc:
            raise RoomScanError("calibration manifest is unreadable") from exc
        preflight_status, preflight = self._http_json("GET", self._robot_url("/v1/preflight"), token=token)
        if preflight_status != 200 or preflight.get("ready") is not True:
            raise RoomScanError(f"robot Tailnet preflight is not ready: {preflight.get('report') or preflight}")
        request_id = str(uuid.uuid4())
        session_id = self._new_session_id()
        request_payload = {
            "request_id": request_id,
            "mode": "full",
            "duration_seconds": duration_seconds,
            "session_id": session_id,
            "calibration_id": calibration_id,
            "calibration_sha256": calibration_sha256,
            "pc_tailnet_ip": self.pc_tailnet_ip,
            "coordinate_scope": "SESSION_ODOM_LOCAL",
        }
        status, started = self._http_json("POST", self._robot_url("/v1/sessions"), token=token, payload=request_payload)
        if status not in {200, 201} or started.get("state") != "RUNNING":
            if status == 409:
                raise RoomScanConflict(started.get("detail") or "robot session control rejected the request")
            raise RoomScanError(started.get("detail") or "robot session control did not start the request")
        self.active_request_id = request_id
        return {
            **started,
            "request_id": request_id,
            "session_id": session_id,
            "mode": "full",
            "coordinate_scope": "SESSION_ODOM_LOCAL",
            "geometry_only_calibration": True,
            "measurement_authority": False,
            "read_only": True,
        }

    def status(self, request_id: str, operator_token: str) -> dict[str, Any]:
        self._authorize(operator_token)
        token = self._control_token()
        status, payload = self._http_json("GET", self._robot_url(f"/v1/sessions/{request_id}"), token=token)
        if status != 200:
            raise RoomScanError(payload.get("detail") or "room scan status is unavailable")
        return payload

    def stop(self, request_id: str, operator_token: str) -> dict[str, Any]:
        self._authorize(operator_token)
        token = self._control_token()
        status, payload = self._http_json("POST", self._robot_url(f"/v1/sessions/{request_id}/stop"), token=token, payload={})
        if status != 200:
            raise RoomScanError(payload.get("detail") or "room scan stop was rejected")
        if request_id == self.active_request_id:
            self.active_request_id = None
        return payload
