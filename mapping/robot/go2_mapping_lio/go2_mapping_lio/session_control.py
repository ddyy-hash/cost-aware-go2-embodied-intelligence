"""Authenticated Tailscale-only control service for Go2 mapping sessions."""

import argparse
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import uuid

from mapping.robot.go2_mapping_locomotion.terrain_control_service import TerrainControlSupervisor


MAX_REQUEST_BYTES = 4096
SCAN_SCRIPT = Path("/home/unitree/go2_mapping/current/tools/start_go2_robot_scan.sh")
PREFLIGHT_SCRIPT = Path("/home/unitree/go2_mapping/current/tools/verify_go2_mapping_robot.sh")
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", re.ASCII)
_TASK_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", re.ASCII)
_SESSION_ID = re.compile(r"^segment-[0-9]{8}T[0-9]{6}Z$", re.ASCII)
_CALIBRATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_SESSION_PHASES = {"STARTING", "RECORDING", "FINALIZING"}
_ACTIVE_STATES = {"RUNNING", "STOPPING", "FINALIZING"}
_START_OUTPUT = re.compile(
    r"GO2_ROBOT_SCAN_STARTED pid=([1-9][0-9]*) session=(segment-[0-9]{8}T[0-9]{6}Z)"
)
_REQUEST_PATH = re.compile(
    r"^/v1/sessions/([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(/stop)?$",
    re.ASCII,
)


class SessionConflictError(RuntimeError):
    pass


class SessionOwnershipError(RuntimeError):
    pass


class SessionNotFoundError(KeyError):
    pass


def require_tailnet_ipv4(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Tailnet address must be an IPv4 string")
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("Tailnet address is invalid") from exc
    if not isinstance(address, ipaddress.IPv4Address) or address not in _TAILNET:
        raise ValueError("Tailnet address must be inside 100.64.0.0/10")
    if str(address) != value:
        raise ValueError("Tailnet address must be canonical")
    return value


def _request_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("request_id must be a canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("request_id must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("request_id must be a canonical UUIDv4")
    return value


@dataclass(frozen=True)
class SessionRequest:
    request_id: str
    mode: str
    duration_seconds: int
    session_id: str
    calibration_id: str
    calibration_sha256: str
    pc_tailnet_ip: str
    coordinate_scope: str
    mission_id: Optional[str] = None
    task_id: Optional[str] = None
    task_revision: Optional[int] = None
    lease_id: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SessionRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("session request must be a JSON object")
        required = {
            "request_id",
            "mode",
            "duration_seconds",
            "session_id",
            "calibration_id",
            "calibration_sha256",
            "pc_tailnet_ip",
            "coordinate_scope",
        }
        optional = {"mission_id", "task_id", "task_revision", "lease_id"}
        if not required.issubset(set(payload)) or any(key not in required | optional for key in payload):
            raise ValueError("session request fields are incomplete or unknown")
        mode = payload["mode"]
        if mode not in {"baseline", "full"}:
            raise ValueError("mode must be baseline or full")
        coordinate_scope = payload["coordinate_scope"]
        if coordinate_scope not in {"GLOBAL_MAP", "SESSION_ODOM_LOCAL"}:
            raise ValueError("coordinate_scope is invalid")
        if coordinate_scope == "SESSION_ODOM_LOCAL" and mode != "full":
            raise ValueError("SESSION_ODOM_LOCAL requires full mode")
        duration = payload["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 86400:
            raise ValueError("duration_seconds must be an integer in [1, 86400]")
        session_id = payload["session_id"]
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("session_id must be canonical")
        calibration_id = payload["calibration_id"]
        if not isinstance(calibration_id, str) or _CALIBRATION_ID.fullmatch(calibration_id) is None:
            raise ValueError("calibration_id is invalid")
        calibration_sha256 = payload["calibration_sha256"]
        if not isinstance(calibration_sha256, str) or _SHA256.fullmatch(calibration_sha256) is None:
            raise ValueError("calibration_sha256 must be lowercase SHA-256")
        mission_id = payload.get("mission_id")
        if mission_id is not None and _IDENTIFIER.fullmatch(str(mission_id)) is None:
            raise ValueError("mission_id is invalid")
        task_id = payload.get("task_id")
        if task_id is not None and _TASK_IDENTIFIER.fullmatch(str(task_id)) is None:
            raise ValueError("task_id is invalid")
        task_revision = payload.get("task_revision")
        if task_revision is not None:
            if isinstance(task_revision, bool) or not isinstance(task_revision, int) or task_revision <= 0:
                raise ValueError("task_revision must be a positive integer")
        lease_id = payload.get("lease_id")
        if lease_id is not None:
            if not isinstance(lease_id, str) or _IDENTIFIER.fullmatch(lease_id) is None:
                raise ValueError("lease_id is invalid")
        return cls(
            _request_id(payload["request_id"]),
            mode,
            duration,
            session_id,
            calibration_id,
            calibration_sha256,
            require_tailnet_ipv4(payload["pc_tailnet_ip"]),
            coordinate_scope,
            None if mission_id is None else str(mission_id),
            None if task_id is None else str(task_id),
            task_revision,
            None if lease_id is None else str(lease_id),
        )

    def canonical_payload(self) -> Dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "mode": self.mode,
            "duration_seconds": self.duration_seconds,
            "session_id": self.session_id,
            "calibration_id": self.calibration_id,
            "calibration_sha256": self.calibration_sha256,
            "pc_tailnet_ip": self.pc_tailnet_ip,
            "coordinate_scope": self.coordinate_scope,
        }
        if self.mission_id is not None:
            payload["mission_id"] = self.mission_id
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.task_revision is not None:
            payload["task_revision"] = self.task_revision
        if self.lease_id is not None:
            payload["lease_id"] = self.lease_id
        return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _request_digest(request: SessionRequest) -> str:
    return hashlib.sha256(_canonical_bytes(request.canonical_payload())).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.partial" % (path.name, uuid.uuid4().hex))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(temporary), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(_canonical_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_record(path: Path) -> Dict[str, Any]:
    body = path.read_bytes()
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("session record is unreadable") from exc
    if not isinstance(payload, dict) or body != _canonical_bytes(payload):
        raise ValueError("session record is not canonical")
    if payload.get("schema_version") != 1:
        raise ValueError("session record schema is invalid")
    return payload


def _read_session_result(path: Path, expected_session_id: str) -> Dict[str, Any]:
    body = path.read_bytes()
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("session result is unreadable") from exc
    if not isinstance(payload, dict) or body != _canonical_bytes(payload):
        raise ValueError("session result is not canonical")
    if set(payload) != {"schema_version", "session_id", "state", "exit_code"}:
        raise ValueError("session result fields are invalid")
    if payload["schema_version"] != 1 or payload["session_id"] != expected_session_id:
        raise ValueError("session result identity is invalid")
    exit_code = payload["exit_code"]
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or not 0 <= exit_code <= 255:
        raise ValueError("session result exit code is invalid")
    expected_state = "COMPLETED" if exit_code == 0 else "FAILED"
    if payload["state"] != expected_state:
        raise ValueError("session result state conflicts with exit code")
    return payload


def owned_process_status(pid: int, scan_script: Path = SCAN_SCRIPT) -> str:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "FOREIGN"
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        command = cmdline_path.read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return "MISSING"
    except OSError:
        return "FOREIGN"
    decoded = [value.decode("utf-8", "replace") for value in command if value]
    if str(scan_script) in decoded and "supervise" in decoded:
        return "OWNED"
    return "FOREIGN"


def launch_scan(request: SessionRequest, scan_script: Path = SCAN_SCRIPT) -> int:
    environment = dict(os.environ)
    control_runtime = Path(
        environment.get(
            "GO2_MAPPING_SESSION_CONTROL_RUNTIME",
            "/home/unitree/go2_mapping/runtime/session-control",
        )
    )
    environment.update(
        {
            "GO2_MAPPING_CAPTURE_SESSION_ID": request.session_id,
            "GO2_MAPPING_CALIBRATION_ID": request.calibration_id,
            "GO2_MAPPING_CALIBRATION_SHA256": request.calibration_sha256,
            "GO2_MAPPING_PC_TAILNET_IP": request.pc_tailnet_ip,
            "GO2_MAPPING_HIGH_FIDELITY_URL": "http://%s:8000" % request.pc_tailnet_ip,
            "GO2_MAPPING_LIVE_ENDPOINT": "ws://%s:8765/ws/live/keyframe" % request.pc_tailnet_ip,
            "GO2_MAPPING_LIVE_HEALTH_URL": "http://%s:8765/api/live/readiness" % request.pc_tailnet_ip,
            "GO2_MAPPING_SESSION_STOP_FILE": str(
                control_runtime / "stop-requests" / (request.session_id + ".json")
            ),
            "GO2_MAPPING_COORDINATE_SCOPE": request.coordinate_scope,
        }
    )
    if request.mission_id is not None:
        environment["GO2_MAPPING_MISSION_ID"] = request.mission_id
    if request.task_id is not None:
        environment["GO2_MAPPING_TASK_ID"] = request.task_id
    if request.task_revision is not None:
        environment["GO2_MAPPING_TASK_REVISION"] = str(request.task_revision)
    if request.lease_id is not None:
        environment["GO2_MAPPING_TASK_LEASE_ID"] = request.lease_id
    argv = [
        "/bin/bash",
        str(scan_script),
        "start",
        str(request.duration_seconds),
        request.mode,
    ]
    result = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError("robot scan launcher failed: %s" % result.stderr.strip()[:1024])
    match = _START_OUTPUT.search(result.stdout)
    if match is None:
        raise RuntimeError("robot scan launcher did not return an owned PID")
    pid = int(match.group(1))
    if owned_process_status(pid, scan_script) != "OWNED":
        raise SessionOwnershipError("robot scan PID ownership verification failed")
    return pid


def stop_owned_scan(pid: int, scan_script: Path = SCAN_SCRIPT) -> None:
    if owned_process_status(pid, scan_script) != "OWNED":
        raise SessionOwnershipError("robot scan PID is not owned by session control")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        state = owned_process_status(pid, scan_script)
        if state == "MISSING":
            return
        if state != "OWNED":
            raise SessionOwnershipError("robot scan PID identity changed during stop")
        time.sleep(0.1)
    if owned_process_status(pid, scan_script) != "OWNED":
        raise SessionOwnershipError("robot scan PID identity changed before forced stop")
    os.kill(pid, signal.SIGKILL)


class SessionController:
    def __init__(
        self,
        runtime_dir: Path,
        launcher: Callable[[SessionRequest], int] = launch_scan,
        process_status: Callable[[int], str] = owned_process_status,
        stopper: Callable[[int], None] = stop_owned_scan,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.requests_dir = self.runtime_dir / "requests"
        self.results_dir = self.runtime_dir / "results"
        self.phases_dir = self.runtime_dir / "phases"
        self.stop_requests_dir = self.runtime_dir / "stop-requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._launcher = launcher
        self._process_status = process_status
        self._stopper = stopper
        self._clock = clock
        self._lock = threading.RLock()

    def _path(self, request_id: str) -> Path:
        return self.requests_dir / ("%s.json" % _request_id(request_id))

    def _load(self, request_id: str) -> Dict[str, Any]:
        path = self._path(request_id)
        if not path.is_file():
            raise SessionNotFoundError(request_id)
        return _read_record(path)

    def _save(self, record: Dict[str, Any]) -> Dict[str, Any]:
        _write_atomic(self._path(record["request_id"]), record)
        return dict(record)

    def _phase(self, session_id: str) -> Optional[str]:
        path = self.phases_dir / (session_id + ".state")
        try:
            phase = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise SessionOwnershipError("session phase is unreadable") from exc
        if phase not in _SESSION_PHASES:
            raise SessionOwnershipError("session phase is invalid")
        return phase

    def _stop_path(self, session_id: str) -> Path:
        return self.stop_requests_dir / (session_id + ".json")

    def _refresh(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if record["state"] not in _ACTIVE_STATES:
            return record
        state = self._process_status(record["pid"])
        if state == "OWNED":
            phase = self._phase(record["request"]["session_id"])
            if phase is None:
                phase = str(record.get("phase", "STARTING"))
            refreshed = dict(record)
            refreshed["phase"] = phase
            if phase == "FINALIZING":
                refreshed["state"] = "FINALIZING"
            if refreshed != record:
                refreshed["updated_at_unix"] = self._clock()
                self._save(refreshed)
            return refreshed
        if state == "FOREIGN":
            raise SessionOwnershipError("persisted PID no longer belongs to this session")
        record = dict(record)
        session_id = record["request"]["session_id"]
        result_path = self.results_dir / ("%s.json" % session_id)
        try:
            result = _read_session_result(result_path, session_id)
        except FileNotFoundError:
            record["state"] = "FAILED"
            record["failure_reason"] = "SESSION_RESULT_MISSING"
        except (OSError, ValueError):
            record["state"] = "FAILED"
            record["failure_reason"] = "SESSION_RESULT_INVALID"
        else:
            record["state"] = result["state"]
            record["exit_code"] = result["exit_code"]
        record["updated_at_unix"] = self._clock()
        self._save(record)
        return record

    def start(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request = SessionRequest.from_payload(payload)
        digest = _request_digest(request)
        with self._lock:
            path = self._path(request.request_id)
            if path.is_file():
                existing = self._load(request.request_id)
                if existing["request_sha256"] != digest:
                    raise SessionConflictError("request ID replayed with different content")
                return dict(self._refresh(existing))
            for candidate in sorted(self.requests_dir.glob("*.json")):
                active = self._refresh(_read_record(candidate))
                if active["state"] in _ACTIVE_STATES:
                    raise SessionConflictError("another mapping session is already running")
            pid = self._launcher(request)
            now = self._clock()
            record = {
                "schema_version": 1,
                "request_id": request.request_id,
                "request_sha256": digest,
                "request": request.canonical_payload(),
                "state": "RUNNING",
                "phase": "STARTING",
                "stop_requested": False,
                "pid": pid,
                "created_at_unix": now,
                "updated_at_unix": now,
            }
            return self._save(record)

    def status(self, request_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._refresh(self._load(request_id)))

    def stop(self, request_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._refresh(self._load(request_id))
            if record["state"] == "FINALIZING":
                raise SessionConflictError("session finalization cannot be interrupted")
            if record["state"] == "STOPPING":
                return dict(record)
            if record["state"] != "RUNNING":
                return dict(record)
            if self._process_status(record["pid"]) != "OWNED":
                raise SessionOwnershipError("persisted PID is not owned; refusing to signal it")
            stop_payload = {
                "schema_version": 1,
                "request_id": record["request_id"],
                "session_id": record["request"]["session_id"],
            }
            _write_atomic(self._stop_path(record["request"]["session_id"]), stop_payload)
            record = dict(record)
            record["state"] = "STOPPING"
            record["stop_requested"] = True
            record["updated_at_unix"] = self._clock()
            return self._save(record)


class SessionControlApplication:
    def __init__(
        self,
        controller: SessionController,
        token: str,
        allowed_peer: str,
        preflight: Callable[[], Mapping[str, Any]],
        terrain_control: Optional[TerrainControlSupervisor] = None,
    ) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("session-control token must not be empty")
        self.controller = controller
        self.token = token
        self.allowed_peer = require_tailnet_ipv4(allowed_peer)
        self.preflight = preflight
        self.terrain_control = terrain_control

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return ""

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        peer: str,
    ) -> Tuple[int, Dict[str, Any]]:
        authorization = self._header(headers, "Authorization")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied, self.token):
            return 401, {"error": "AUTHENTICATION_REQUIRED"}
        if peer != self.allowed_peer:
            return 403, {"error": "TAILNET_PEER_REJECTED"}
        if len(body) > MAX_REQUEST_BYTES:
            return 413, {"error": "REQUEST_BODY_TOO_LARGE"}
        try:
            if method == "GET" and path == "/v1/preflight":
                return 200, dict(self.preflight())
            if method == "POST" and path == "/v1/sessions":
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    return 400, {"error": "INVALID_JSON"}
                if not isinstance(payload, dict):
                    return 422, {"error": "SESSION_REQUEST_INVALID"}
                return 201, self.controller.start(payload)
            if self.terrain_control is not None and method == "GET" and path == "/v1/terrain/status":
                return 200, self.terrain_control.status()
            if self.terrain_control is not None and method == "POST" and path == "/v1/terrain/arm":
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    return 400, {"error": "INVALID_JSON"}
                if not isinstance(payload, dict):
                    return 422, {"error": "TERRAIN_ARM_INVALID"}
                return 200, self.terrain_control.arm(payload.get("confirmation"))
            if self.terrain_control is not None and method == "POST" and path == "/v1/terrain/request":
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    return 400, {"error": "INVALID_JSON"}
                if not isinstance(payload, dict):
                    return 422, {"error": "TERRAIN_REQUEST_INVALID"}
                try:
                    return 200, self.terrain_control.request(payload)
                except ValueError as exc:
                    return 422, {"error": "TERRAIN_REQUEST_INVALID", "detail": str(exc)}
            if self.terrain_control is not None and method == "POST" and path == "/v1/terrain/stop":
                reason = "OPERATOR_STOP"
                if body:
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError):
                        return 400, {"error": "INVALID_JSON"}
                    if isinstance(payload, dict) and payload.get("reason") is not None:
                        reason = str(payload["reason"])
                return 200, self.terrain_control.stop(reason)
            match = _REQUEST_PATH.fullmatch(path)
            if match is None:
                return 404, {"error": "NOT_FOUND"}
            request_id, stop_suffix = match.groups()
            if method == "GET" and stop_suffix is None:
                return 200, self.controller.status(request_id)
            if method == "POST" and stop_suffix == "/stop":
                return 200, self.controller.stop(request_id)
            return 405, {"error": "METHOD_NOT_ALLOWED"}
        except ValueError as exc:
            return 422, {"error": "SESSION_REQUEST_INVALID", "detail": str(exc)}
        except SessionNotFoundError:
            return 404, {"error": "SESSION_NOT_FOUND"}
        except SessionConflictError as exc:
            return 409, {"error": "SESSION_CONFLICT", "detail": str(exc)}
        except SessionOwnershipError as exc:
            return 409, {"error": "PID_OWNERSHIP_REJECTED", "detail": str(exc)}
        except RuntimeError as exc:
            return 503, {"error": "SESSION_CONTROL_FAILED", "detail": str(exc)}


class SessionControlHandler(BaseHTTPRequestHandler):
    server_version = "Go2SessionControl/1"

    def _dispatch(self) -> None:
        try:
            length_text = self.headers.get("Content-Length", "0")
            length = int(length_text)
        except ValueError:
            self._respond(400, {"error": "INVALID_CONTENT_LENGTH"})
            return
        if length < 0:
            self._respond(400, {"error": "INVALID_CONTENT_LENGTH"})
            return
        if length > MAX_REQUEST_BYTES:
            self._respond(413, {"error": "REQUEST_BODY_TOO_LARGE"})
            return
        if self.headers.get("Transfer-Encoding"):
            self._respond(400, {"error": "TRANSFER_ENCODING_REJECTED"})
            return
        body = self.rfile.read(length) if length else b""
        application = self.server.application  # type: ignore[attr-defined]
        status, payload = application.handle(
            self.command,
            self.path,
            dict(self.headers.items()),
            body,
            self.client_address[0],
        )
        self._respond(status, payload)

    def _respond(self, status: int, payload: Mapping[str, Any]) -> None:
        body = _canonical_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def log_message(self, _format: str, *args: Any) -> None:
        return


def run_preflight(preflight_script: Path = PREFLIGHT_SCRIPT) -> Dict[str, Any]:
    environment = dict(os.environ)
    environment["GO2_MAPPING_REPORT_DIR"] = str(
        Path(os.environ.get("GO2_MAPPING_SESSION_CONTROL_RUNTIME", "/home/unitree/go2_mapping/runtime/session-control"))
        / "preflight-reports"
    )
    result = subprocess.run(
        ["/bin/bash", str(preflight_script)],
        shell=False,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=environment,
    )
    report = (result.stdout + result.stderr)[-65536:]
    return {
        "schema_version": 1,
        "ready": result.returncode == 0,
        "exit_code": result.returncode,
        "report": report,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--allow-peer", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("/home/unitree/go2_mapping/runtime/session-control"),
    )
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)
    bind = require_tailnet_ipv4(args.bind)
    peer = require_tailnet_ipv4(args.allow_peer)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        token = args.token_file.read_text(encoding="ascii").strip()
    except OSError as exc:
        parser.error("session-control token is unreadable: %s" % exc)
    if len(token) < 32 or len(token) > 512:
        parser.error("session-control token must contain 32-512 ASCII characters")
    controller = SessionController(args.runtime_dir)
    terrain_control = TerrainControlSupervisor(
        status_path=args.runtime_dir.parent / "terrain-router-status.json",
        control_status_path=args.runtime_dir.parent / "terrain-control-status.json",
        motion_helper=args.runtime_dir.parent / "bin" / "go2_motion_mode_helper",
        vendor_adapter=args.runtime_dir.parent / "bin" / "go2_vendor_sport_adapter",
        executor=args.runtime_dir.parent / "bin" / "go2_native_terrain_executor",
    )
    application = SessionControlApplication(controller, token, peer, run_preflight, terrain_control)
    server = ThreadingHTTPServer((bind, args.port), SessionControlHandler)
    server.application = application  # type: ignore[attr-defined]
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
