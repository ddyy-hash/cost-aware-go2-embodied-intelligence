import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "mapping"
    / "robot"
    / "go2_mapping_lio"
    / "go2_mapping_lio"
    / "session_control.py"
)
TAILNET_CLIENT = ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1"
COMPATIBILITY_CLIENT = ROOT / "tools" / "start_go2_mapping_session.ps1"
INSTALLER = ROOT / "tools" / "install_go2_tailnet_session_control.sh"
SERVICE_UNIT = ROOT / "tools" / "go2-mapping-session-control.service"


def test_session_control_user_unit_avoids_kernel_capability_sandbox_in_user_systemd():
    unit = SERVICE_UNIT.read_text(encoding="utf-8")
    assert "NoNewPrivileges=true" in unit
    assert "ProtectSystem=strict" in unit
    for directive in (
        "ProtectKernelTunables",
        "ProtectKernelModules",
        "ProtectControlGroups",
        "LockPersonality",
        "RestrictSUIDSGID",
    ):
        assert f"{directive}=" not in unit


def test_installer_persists_the_unitree_user_manager_before_starting_control():
    installer = INSTALLER.read_text(encoding="utf-8")

    assert "service_user=$(id -un)" in installer
    assert '[[ $service_user == unitree ]]' in installer
    linger_command = 'loginctl enable-linger "$service_user"'
    linger_check = 'loginctl show-user "$service_user" -p Linger --value'
    service_start = "systemctl --user enable --now go2-mapping-session-control.service"
    assert linger_command in installer
    assert "sudo -n loginctl" not in installer
    assert linger_check in installer
    assert installer.index(linger_command) < installer.index(service_start)
    assert installer.index(linger_check) < installer.index(service_start)
    assert "systemctl --user restart go2-mapping-session-control.service" in installer
    assert "session-control local authentication boundary returned HTTP" in installer
    assert '[[ $http_status == 401 ]]' in installer
    assert "for attempt in $(seq 1 100)" in installer
    assert "sleep 0.1" in installer
    assert '-H "Authorization: Bearer $token"' not in installer
    assert "--robot-serial" in installer
    assert "GO2_MAPPING_ROBOT_SERIAL" in installer
    assert "existing_robot_serial" in installer


def test_deployment_verifies_authenticated_preflight_from_the_allowed_pc_peer():
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert 'Invoke-DirectRestMethod -Uri "http://${RobotTailnetIp}:${SessionControlPort}/v1/preflight"' in deploy
    assert "$request.Proxy = $null" in deploy
    assert "Authorization = \"Bearer $sessionControlToken\"" in deploy
    assert "authenticated robot preflight failed after deployment" in deploy
    assert "RobotFactorySerial" in deploy
    assert "--robot-serial '$RobotFactorySerial'" in deploy


def _module():
    assert MODULE_PATH.is_file(), "the robot Tailnet session-control service is required"
    spec = importlib.util.spec_from_file_location("go2_session_control_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(**changes):
    payload = {
        "request_id": "2f3e9d92-0fa8-4df2-9ae8-b8e77c809f31",
        "mode": "full",
        "duration_seconds": 300,
        "session_id": "segment-20260805T120000Z",
        "calibration_id": "go2-calibration-v1",
        "calibration_sha256": hashlib.sha256(b"calibration").hexdigest(),
        "pc_tailnet_ip": "192.0.2.10",
        "coordinate_scope": "GLOBAL_MAP",
    }
    payload.update(changes)
    return payload


def test_request_contract_rejects_unknown_noncanonical_and_shell_values():
    module = _module()
    request = module.SessionRequest.from_payload(_payload())
    assert request.request_id == _payload()["request_id"]
    assert request.canonical_payload() == _payload()

    invalid = (
        _payload(unknown=True),
        _payload(request_id=_payload()["request_id"].upper()),
        _payload(request_id="not-a-uuid"),
        _payload(mode="diagnostic"),
        _payload(coordinate_scope="LOCAL"),
        _payload(mode="baseline", coordinate_scope="SESSION_ODOM_LOCAL"),
        _payload(duration_seconds=0),
        _payload(duration_seconds=86401),
        _payload(session_id="segment-20260805T120000Z; reboot"),
        _payload(calibration_id="calibration$(id)"),
        _payload(calibration_sha256="A" * 64),
        _payload(pc_tailnet_ip="192.168.1.20"),
    )
    for candidate in invalid:
        with pytest.raises(ValueError):
            module.SessionRequest.from_payload(candidate)


def test_request_contract_accepts_optional_mission_task_and_lease_identity():
    module = _module()
    payload = _payload(
        mission_id="mission-20260825T120000Z",
        task_id="room-1:corners",
        task_revision=3,
        lease_id="lease-room-1",
    )
    request = module.SessionRequest.from_payload(payload)

    assert request.mission_id == "mission-20260825T120000Z"
    assert request.task_id == "room-1:corners"
    assert request.task_revision == 3
    assert request.lease_id == "lease-room-1"
    assert request.canonical_payload() == payload


def test_tailnet_addresses_are_strict_cgnat_ipv4_only():
    module = _module()
    assert module.require_tailnet_ipv4("100.64.0.1") == "100.64.0.1"
    assert module.require_tailnet_ipv4("100.127.255.254") == "100.127.255.254"
    for value in ("100.63.255.255", "100.128.0.1", "127.0.0.1", "::1", "host.example"):
        with pytest.raises(ValueError):
            module.require_tailnet_ipv4(value)


def test_constant_time_bearer_and_allowlisted_peer_are_both_required(monkeypatch, tmp_path):
    module = _module()
    comparisons = []

    def compare(left, right):
        comparisons.append((left, right))
        return left == right

    monkeypatch.setattr(module.hmac, "compare_digest", compare)
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4101,
        process_status=lambda _pid: "OWNED",
        stopper=lambda _pid: None,
    )
    app = module.SessionControlApplication(
        controller,
        token="correct-secret",
        allowed_peer="192.0.2.10",
        preflight=lambda: {"ready": True},
    )
    body = json.dumps(_payload()).encode("utf-8")

    assert app.handle("POST", "/v1/sessions", {}, body, "192.0.2.10")[0] == 401
    assert app.handle(
        "POST",
        "/v1/sessions",
        {"Authorization": "Bearer wrong-secret"},
        body,
        "192.0.2.10",
    )[0] == 401
    assert app.handle(
        "POST",
        "/v1/sessions",
        {"Authorization": "Bearer correct-secret"},
        body,
        "100.121.140.95",
    )[0] == 403
    assert comparisons == [
        ("", "correct-secret"),
        ("wrong-secret", "correct-secret"),
        ("correct-secret", "correct-secret"),
    ]


def test_start_is_idempotent_and_conflicting_replay_is_rejected(tmp_path):
    module = _module()
    launched = []
    states = {}

    def launch(request):
        launched.append(request)
        states[4201] = "OWNED"
        return 4201

    controller = module.SessionController(
        tmp_path,
        launcher=launch,
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: states.__setitem__(pid, "MISSING"),
    )

    first = controller.start(_payload())
    repeated = controller.start(dict(reversed(tuple(_payload().items()))))
    assert first == repeated
    assert first["state"] == "RUNNING"
    assert first["pid"] == 4201
    assert len(launched) == 1

    with pytest.raises(module.SessionConflictError, match="different content"):
        controller.start(_payload(duration_seconds=301))


def test_concurrent_different_requests_allow_only_one_active_scan(tmp_path):
    module = _module()
    entered = threading.Event()
    release = threading.Event()
    launch_count = 0
    states = {}

    def launch(_request):
        nonlocal launch_count
        launch_count += 1
        entered.set()
        assert release.wait(3.0)
        states[4301] = "OWNED"
        return 4301

    controller = module.SessionController(
        tmp_path,
        launcher=launch,
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: states.__setitem__(pid, "MISSING"),
    )
    results = []
    errors = []

    def start(payload):
        try:
            results.append(controller.start(payload))
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=start, args=(_payload(),))
    second = threading.Thread(
        target=start,
        args=(_payload(request_id="3809ac6a-ae68-46e3-b136-8b5b0ff7d2ea"),),
    )
    first.start()
    assert entered.wait(3.0)
    second.start()
    release.set()
    first.join(3.0)
    second.join(3.0)

    assert launch_count == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], module.SessionConflictError)


def test_restart_status_and_stop_use_only_persisted_owned_pid(tmp_path):
    module = _module()
    states = {4401: "OWNED"}
    stopped = []
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4401,
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: (stopped.append(pid), states.__setitem__(pid, "MISSING")),
    )
    started = controller.start(_payload())

    restarted = module.SessionController(
        tmp_path,
        launcher=lambda _request: pytest.fail("restart must not relaunch an existing request"),
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: (stopped.append(pid), states.__setitem__(pid, "MISSING")),
    )
    assert restarted.status(started["request_id"])["state"] == "RUNNING"
    stopped_once = restarted.stop(started["request_id"])
    stopped_twice = restarted.stop(started["request_id"])
    assert stopped_once == stopped_twice
    assert stopped_once["state"] == "STOPPING"
    assert stopped_once["stop_requested"] is True
    assert stopped == []

    record_path = tmp_path / "requests" / f"{started['request_id']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    assert record_path.read_text(encoding="utf-8") == expected
    assert not list(tmp_path.rglob("*.partial"))


def test_recording_stop_is_a_durable_graceful_request_not_a_supervisor_signal(tmp_path):
    module = _module()
    states = {4402: "OWNED"}
    signalled = []
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4402,
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: signalled.append(pid),
    )
    started = controller.start(_payload())
    phase_path = tmp_path / "phases" / f"{started['request']['session_id']}.state"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text("RECORDING\n", encoding="ascii")

    stopping = controller.stop(started["request_id"])

    assert stopping["state"] == "STOPPING"
    assert stopping["phase"] == "RECORDING"
    assert stopping["stop_requested"] is True
    assert signalled == []
    marker = tmp_path / "stop-requests" / f"{started['request']['session_id']}.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload == {
        "request_id": started["request_id"],
        "schema_version": 1,
        "session_id": started["request"]["session_id"],
    }
    assert marker.read_text(encoding="utf-8") == (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )


def test_finalizing_capture_refuses_stop_and_remains_an_active_conflict(tmp_path):
    module = _module()
    states = {4403: "OWNED"}
    signalled = []
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4403,
        process_status=lambda pid: states.get(pid, "MISSING"),
        stopper=lambda pid: signalled.append(pid),
    )
    started = controller.start(_payload())
    phase_path = tmp_path / "phases" / f"{started['request']['session_id']}.state"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text("FINALIZING\n", encoding="ascii")

    status = controller.status(started["request_id"])

    assert status["state"] == "FINALIZING"
    assert status["phase"] == "FINALIZING"
    with pytest.raises(module.SessionConflictError, match="finalization"):
        controller.stop(started["request_id"])
    with pytest.raises(module.SessionConflictError, match="another mapping session"):
        controller.start(
            _payload(
                request_id="3809ac6a-ae68-46e3-b136-8b5b0ff7d2ea",
                session_id="segment-20260805T120001Z",
            )
        )
    assert signalled == []
    assert not (tmp_path / "stop-requests").exists()


def _write_session_result(runtime_dir, session_id, payload):
    result_dir = runtime_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{session_id}.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_missing_owned_supervisor_result_fails_closed_as_failed(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4701,
        process_status=lambda _pid: "MISSING",
    )
    started = controller.start(_payload())

    refreshed = controller.status(started["request_id"])

    assert refreshed["state"] == "FAILED"
    assert refreshed["failure_reason"] == "SESSION_RESULT_MISSING"


def test_zero_exit_supervisor_result_is_completed(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4702,
        process_status=lambda _pid: "MISSING",
    )
    started = controller.start(_payload())
    _write_session_result(
        tmp_path,
        started["request"]["session_id"],
        {"schema_version": 1, "session_id": started["request"]["session_id"], "state": "COMPLETED", "exit_code": 0},
    )

    refreshed = controller.status(started["request_id"])
    assert refreshed["state"] == "COMPLETED"
    assert refreshed["exit_code"] == 0


def test_nonzero_exit_supervisor_result_is_failed(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4703,
        process_status=lambda _pid: "MISSING",
    )
    started = controller.start(_payload())
    _write_session_result(
        tmp_path,
        started["request"]["session_id"],
        {"schema_version": 1, "session_id": started["request"]["session_id"], "state": "FAILED", "exit_code": 7},
    )

    assert controller.status(started["request_id"])["state"] == "FAILED"
    assert controller.status(started["request_id"])["exit_code"] == 7


def test_mismatched_session_result_fails_closed(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4704,
        process_status=lambda _pid: "MISSING",
    )
    started = controller.start(_payload())
    _write_session_result(
        tmp_path,
        started["request"]["session_id"],
        {"schema_version": 1, "session_id": "segment-20260805T120001Z", "state": "COMPLETED", "exit_code": 0},
    )

    refreshed = controller.status(started["request_id"])

    assert refreshed["state"] == "FAILED"
    assert refreshed["failure_reason"] == "SESSION_RESULT_INVALID"


def test_malformed_session_result_fails_closed(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4705,
        process_status=lambda _pid: "MISSING",
    )
    started = controller.start(_payload())
    result_path = tmp_path / "results" / f"{started['request']['session_id']}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{not-json}\n", encoding="utf-8")

    refreshed = controller.status(started["request_id"])

    assert refreshed["state"] == "FAILED"
    assert refreshed["failure_reason"] == "SESSION_RESULT_INVALID"


def test_foreign_pid_is_never_signalled(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4501,
        process_status=lambda _pid: "FOREIGN",
        stopper=lambda _pid: pytest.fail("a foreign PID must never be signalled"),
    )
    started = controller.start(_payload())
    with pytest.raises(module.SessionOwnershipError):
        controller.stop(started["request_id"])


def test_http_contract_rejects_large_invalid_and_unknown_requests(tmp_path):
    module = _module()
    controller = module.SessionController(
        tmp_path,
        launcher=lambda _request: 4601,
        process_status=lambda _pid: "OWNED",
        stopper=lambda _pid: None,
    )
    app = module.SessionControlApplication(
        controller,
        token="secret",
        allowed_peer="192.0.2.10",
        preflight=lambda: {"ready": True, "storage_ready": True},
    )
    headers = {"Authorization": "Bearer secret"}

    assert app.handle("POST", "/v1/sessions", headers, b"x" * 4097, "192.0.2.10")[0] == 413
    assert app.handle("POST", "/v1/sessions", headers, b"not-json", "192.0.2.10")[0] == 400
    assert app.handle("POST", "/v1/sessions", headers, b"[]", "192.0.2.10")[0] == 422
    assert app.handle("GET", "/v1/preflight", headers, b"", "192.0.2.10") == (
        200,
        {"ready": True, "storage_ready": True},
    )
    assert app.handle("GET", "/unknown", headers, b"", "192.0.2.10")[0] == 404


def test_fixed_launcher_uses_no_shell_and_validates_the_owned_supervisor(monkeypatch):
    module = _module()
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GO2_ROBOT_SCAN_STARTED pid=4701 "
                "session=segment-20260805T120000Z duration_s=300 mode=full\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "owned_process_status", lambda _pid, _script: "OWNED")
    request = module.SessionRequest.from_payload(_payload())
    pid = module.launch_scan(request)

    assert pid == 4701
    argv, options = calls[0]
    assert argv == [
        "/bin/bash",
        "/home/unitree/go2_mapping/current/tools/start_go2_robot_scan.sh",
        "start",
        "300",
        "full",
    ]
    assert options["shell"] is False
    assert options["timeout"] <= 30
    assert options["env"]["GO2_MAPPING_CAPTURE_SESSION_ID"] == _payload()["session_id"]
    assert options["env"]["GO2_MAPPING_CALIBRATION_ID"] == _payload()["calibration_id"]
    assert options["env"]["GO2_MAPPING_CALIBRATION_SHA256"] == _payload()["calibration_sha256"]
    assert options["env"]["GO2_MAPPING_LIVE_HEALTH_URL"] == "http://192.0.2.10:8765/api/live/readiness"
    assert options["env"]["GO2_MAPPING_SESSION_STOP_FILE"] == (
        "/home/unitree/go2_mapping/runtime/session-control/stop-requests/"
        "segment-20260805T120000Z.json"
    )
    assert options["env"]["GO2_MAPPING_COORDINATE_SCOPE"] == "GLOBAL_MAP"


def test_launcher_exports_mission_task_and_lease_identity_into_the_robot_environment(monkeypatch, tmp_path):
    module = _module()
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GO2_ROBOT_SCAN_STARTED pid=4701 "
                "session=segment-20260805T120000Z duration_s=300 mode=full\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "owned_process_status", lambda _pid, _script: "OWNED")
    request = module.SessionRequest.from_payload(
        _payload(
            mission_id="mission-20260825T120000Z",
            task_id="room-1:corners",
            task_revision=3,
            lease_id="lease-room-1",
        )
    )

    assert module.launch_scan(request) == 4701
    env = calls[0][1]["env"]
    assert env["GO2_MAPPING_MISSION_ID"] == "mission-20260825T120000Z"
    assert env["GO2_MAPPING_TASK_ID"] == "room-1:corners"
    assert env["GO2_MAPPING_TASK_REVISION"] == "3"
    assert env["GO2_MAPPING_TASK_LEASE_ID"] == "lease-room-1"


def test_tailnet_request_allows_only_an_explicit_full_odom_local_validation_scope(monkeypatch):
    module = _module()
    request = module.SessionRequest.from_payload(_payload(coordinate_scope="SESSION_ODOM_LOCAL"))
    assert request.coordinate_scope == "SESSION_ODOM_LOCAL"
    assert request.canonical_payload()["coordinate_scope"] == "SESSION_ODOM_LOCAL"


def test_installer_service_and_pc_clients_enforce_the_transport_boundary():
    for path in (INSTALLER, SERVICE_UNIT, TAILNET_CLIENT, COMPATIBILITY_CLIENT):
        assert path.is_file(), f"missing Task 9 file: {path}"

    installer = INSTALLER.read_text(encoding="utf-8")
    unit = SERVICE_UNIT.read_text(encoding="utf-8")
    client = TAILNET_CLIENT.read_text(encoding="utf-8")
    compatibility = COMPATIBILITY_CLIENT.read_text(encoding="utf-8")
    combined_capture = f"{client}\n{compatibility}".lower()

    assert "tailscale ip -4" in installer
    assert "systemctl --user" in installer
    assert "session-control.env" in installer
    assert "EnvironmentFile=" in unit
    assert "session_control.py" in unit
    assert "--bind" in unit and "--allow-peer" in unit
    assert "invoke-directrestmethod" in client.lower()
    assert "[System.Net.WebRequest]::Create($Uri)" in client
    assert "$request.Proxy = $null" in client
    assert "/v1/preflight" in client
    assert "/v1/sessions" in client
    assert "Authorization" in client and "Bearer" in client
    assert "start_go2_pc_stack.ps1" in client
    assert "start_go2_mapping_session_tailnet.ps1" in compatibility
    for forbidden in ("ssh.exe", "& ssh ", "scp ", "rsync", "proxycommand", "-r 127.0.0.1"):
        assert forbidden not in combined_capture

    service_text = MODULE_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("cmd_vel", "sport", "wirelesscontroller", "realsense2_camera", "ros2 launch"):
        assert forbidden not in service_text
