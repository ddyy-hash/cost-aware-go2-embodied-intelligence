import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PC = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")
SESSION = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")
ROBOT = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")
SESSION_RUNNER = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")
CAPTURE = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
DEPLOY = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
PREPARE = (ROOT / "tools" / "prepare_go2_mapping_lab.ps1").read_text(encoding="utf-8")
BOOTSTRAP_MANIFEST = ROOT / "mapping" / "config" / "calibration_capture_manifest.json"
LIO_PROFILE = (ROOT / "mapping" / "config" / "lio_preview_mid360.yaml").read_text(encoding="utf-8")
CALIBRATION_START_PATH = ROOT / "tools" / "start_go2_calibration.ps1"
POWERSHELL = (shutil.which("powershell.exe") or shutil.which("pwsh.exe")) if os.name == "nt" else None
CALIBRATION_TEST_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
CALIBRATION_TEST_SESSION_ID = "segment-20260809T000000Z"


def _ps_literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_calibration_harness(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    if not POWERSHELL:
        import pytest

        pytest.skip("PowerShell is unavailable")
    tmp_path.mkdir(parents=True, exist_ok=True)
    harness = tmp_path / "calibration-harness.ps1"
    launch_dir = tmp_path / "launch"
    identity_override = """
function New-CalibrationLaunchIdentity {
    New-Item -ItemType Directory -Force -Path __LAUNCH_DIR__ | Out-Null
    [pscustomobject]@{
        request_id = '__REQUEST_ID__'
        session_id = '__SESSION_ID__'
        context_path = Join-Path __LAUNCH_DIR__ 'capture-context.json'
    }
}
"""
    identity_override = identity_override.replace("__LAUNCH_DIR__", _ps_literal(launch_dir))
    identity_override = identity_override.replace("__REQUEST_ID__", CALIBRATION_TEST_REQUEST_ID)
    identity_override = identity_override.replace("__SESSION_ID__", CALIBRATION_TEST_SESSION_ID)
    context_writer = r"""
function Write-TestCalibrationContext {
    param(
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$AcceptanceContextPath
    )
    @{
        schema_version = 1
        state = 'RUNNING'
        request_id = $RequestId
        session_id = $SessionId
        mode = 'full'
        duration_seconds = $DurationSeconds
        capture_purpose = 'calibration_capture'
        calibration_id = 'PROVISIONAL_AUTOCAL_PENDING_v1'
        calibration_status = 'provisional'
        acceptance_authority = $false
        pc_tailnet_ip = '192.0.2.10'
        robot_tailnet_ip = '192.0.2.11'
    } | ConvertTo-Json | Set-Content -LiteralPath $AcceptanceContextPath -Encoding UTF8
}
"""
    harness.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". {_ps_literal(CALIBRATION_START_PATH)}\n"
        + identity_override
        + context_writer
        + body
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_pc_one_click_starts_tailnet_receiver_and_direct_preview_backend():
    assert "ssh_tunnel_receiver.py" in PC
    assert "mapping/pc/start_dashboard_wsl.sh" in PC
    assert "GO2_MAPPING_LIFECYCLE_ROOT" in PC
    assert "tailscale.exe" in PC
    assert "$tailnetIpProcess.StartInfo.Arguments = 'ip -4'" in PC
    assert "--allow-client" in PC
    assert "127.0.0.1:19000:127.0.0.1:8000" not in PC
    assert "GO2_MAPPING_LIVE_ENDPOINT" in PC
    assert "ws://${tailnetIp}:8765/ws/live/keyframe" in PC
    assert "Start-OwnedReverseTunnel" not in PC
    assert "preview-tunnel" not in PC
    assert "127.0.0.1:19001:127.0.0.1:8765" not in PC
    assert 'http://127.0.0.1:$dashboardInternalPort' in PC
    assert "GO2_PC_STACK_LOCAL_READY_DIRECT_PREVIEW_PENDING" not in PC
    assert "GO2_PC_STACK_READY" in PC
    assert "run_go2_mapping_session.sh" not in PC
    assert "PreferredDerpRegion = 0" in PC
    assert "force-prefer-derp" in PC
    assert "Direct path remains preferred" in PC
    assert "official regions remain fallback" in PC
    assert "HighFidelityTransport = 'direct'" in PC
    assert "manage_autodl_derp.ps1" in PC
    assert "$HighFidelityTransport -eq 'autodl'" in PC
    assert "dashboard-wsl.pid" in PC
    assert "dashboard-linux.pid" in PC
    assert "GO2_MAPPING_DASHBOARD_PID_FILE" in PC
    assert "stop-owned" in PC
    assert "verify-owned" in PC
    assert "Windows command line is unavailable" in PC
    assert "dashboard Linux ownership does not match this worktree" in PC
    assert "$dashboardInternalPort = 8767" in PC
    assert "127.0.0.1:$dashboardInternalPort" in PC
    assert "GO2_MAPPING_DASHBOARD_PORT=$dashboardInternalPort" in PC
    assert 'Get-CimInstance Win32_Process -Filter "ProcessId=$ownedDashboardPid"' in PC
    assert '$ownedDashboard.CommandLine -notlike "*--cd $linuxWorkspace*"' in PC
    assert "$ownedDashboard.CommandLine -notlike '*mapping/pc/start_dashboard_wsl.sh*'" in PC
    assert "owned Go2 dashboard" in PC
    assert PC.index("stop-owned") < PC.index("Start-Process -FilePath wsl.exe")
    assert "/api/stack/readiness" in PC
    assert "$readiness.capabilities.building_render_manifest -eq $true" in PC
    assert "$readiness.capabilities.camera_mjpeg -eq $true" in PC
    assert "$readiness.capabilities.field_acceptance_evidence -eq $true" in PC
    assert "$readiness.capabilities.textured_tile_export -eq $true" in PC
    assert "$readiness.capabilities.textured_building_export -eq $true" in PC


def test_pc_stack_reports_tailscale_local_api_denial_without_a_null_method_failure():
    startup = PC[PC.index("$ErrorActionPreference = 'Stop'"):PC.index("if ($Node -notmatch")]

    assert "[System.Diagnostics.Process]::new()" in startup
    assert "RedirectStandardError = $true" in startup
    assert "$tailnetIpResult.ExitCode -ne 0" in startup
    assert "$tailnetIpOutput" in startup
    assert "Tailscale local API access was denied" in startup
    assert "normal interactive desktop administrator token" in startup
    assert "$tailnetIp = (& tailscale.exe ip -4 | Select-Object -First 1).Trim()" not in startup


def test_elevated_dashboard_maintenance_is_probe_first_and_has_no_direct_process_kill():
    maintenance = (
        ROOT
        / ".autonomous"
        / "official-dashboard-surface-publication"
        / "elevated_dashboard_maintenance.ps1"
    ).read_text(encoding="utf-8")

    assert "[ValidateSet('probe', 'restart')]" in maintenance
    assert "if ($Action -eq 'restart')" in maintenance
    assert "$lines.Add(($_ | Out-String))" in maintenance
    assert "start_go2_pc_stack.ps1" in maintenance
    assert "Stop-Process" not in maintenance
    assert "ssh" not in maintenance.lower()


def test_pc_stack_publishes_the_wsl_dashboard_on_the_tailnet():
    assert "tailscale.exe serve --bg --yes --tcp=8765" in PC
    assert 'tcp://127.0.0.1:$dashboardInternalPort' in PC
    assert "Tailnet dashboard publication failed" in PC
    assert "$serveStatus" in PC
    assert "Tailnet dashboard publication does not expose the expected TCP route" in PC


def test_pc_dashboard_restarts_only_when_runtime_configuration_changes():
    assert "$dashboardConfigSha256" in PC
    assert "GO2_MAPPING_RUNTIME_CONFIG_SHA256=$dashboardConfigSha256" in PC
    assert "$readiness.runtime_config_sha256 -eq $dashboardConfigSha256" in PC
    for field in (
        "calibration_manifest",
        "session_evidence",
        "depth_anything_weights",
        "depth_anything_weights_sha256",
        "depth_anything_device",
        "depth_anything_precision",
        "depth_anything_backend",
        "depth_anything_model_name",
    ):
        assert field in PC
    assert "GO2_MAPPING_DEPTH_ANYTHING_DEVICE=$DepthAnythingDevice" in PC
    assert "GO2_MAPPING_DEPTH_ANYTHING_PRECISION=$DepthAnythingPrecision" in PC
    assert "GO2_MAPPING_DEPTH_ANYTHING_BACKEND=$DepthAnythingBackend" in PC

    server = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "server.py").read_text(encoding="utf-8")
    assert '"runtime_config_sha256": _runtime_config_sha256' in server


def test_pc_dashboard_reuse_requires_the_automatic_reconstruction_pointer_protocol():
    protocol = "automatic-reconstruction-pointer-v1"

    assert protocol in PC
    dashboard_check = PC[PC.index("function Test-Dashboard"):PC.index("$status = [ordered]")]
    assert (
        "$readiness.automatic_reconstruction_pointer_protocol -eq "
        "$automaticReconstructionPointerProtocol"
    ) in dashboard_check


def test_pc_stack_can_bind_one_verified_derived_surface_to_the_official_dashboard():
    assert "[string]$DerivedSurfaceJson" in PC
    assert "Resolve-Path -LiteralPath $DerivedSurfaceJson" in PC
    assert "derived_surface_json = [string]$DerivedSurfaceJson" in PC
    assert "$linuxDerivedSurface = Convert-ToWslPath $DerivedSurfaceJson" in PC
    assert "GO2_MAPPING_DERIVED_SURFACE_JSON=$linuxDerivedSurface" in PC
    assert "/api/reconstruction/measured-surface/status" in PC
    assert "$derivedStatus.source_mode -eq 'DERIVED_READ_ONLY'" in PC


def test_full_tailnet_sessions_activate_one_scoped_automatic_high_fidelity_reconstruction():
    worker = ROOT / "tools" / "run_go2_automatic_reconstruction.py"
    worker_launcher = ROOT / "mapping" / "pc" / "start_session_reconstruction_wsl.sh"

    assert worker.is_file(), "full sessions need an owned automatic reconstruction worker"
    assert worker_launcher.is_file(), "full sessions need a WSL-owned reconstruction launcher"
    launcher = worker_launcher.read_text(encoding="utf-8").lower()
    assert "run_go2_automatic_reconstruction.py" in launcher
    assert "--session-id" in launcher
    assert "--inbox-root" in launcher
    assert "--output-root" in launcher
    assert "--status-path" in launcher
    assert "--sensor-frame-transform-id" in launcher
    assert "verify-owned" in launcher
    for forbidden in ("ssh", "scp", "rsync", "proxycommand", "remotecommand"):
        assert forbidden not in launcher
    assert "[string]$IngestSessionId" in PC
    assert "[switch]$AllowDashboardRestart" in PC
    assert "GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_STATUS_PATH" in PC
    assert "GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_ROOT" in PC
    assert "GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID" in PC
    assert "run_go2_automatic_reconstruction.py" in PC
    assert "mapping/pc/start_session_reconstruction_wsl.sh" in PC
    assert "GO2_MAPPING_RECONSTRUCTION_PID_FILE" in PC
    assert "high_fidelity_batch_reconstruction" in PC
    assert "$linuxInboxRoot = Convert-ToWslPath $inboxRoot" in PC
    worker_block = PC[PC.index("if ($IngestSessionId)"):PC.index("& tailscale.exe serve")]
    assert "$linuxInboxRoot, $linuxAutomaticReconstructionRoot" in worker_block
    assert "$linuxIngest, $linuxAutomaticReconstructionRoot" not in worker_block
    assert "$workerReady = $false" in worker_block
    assert "verify-owned" in worker_block
    assert "$worker.HasExited" in worker_block
    assert "Automatic reconstruction worker did not become ready" in worker_block
    assert "AllowDashboardRestart" in PC[PC.index("if (-not (Test-Dashboard))"):PC.index("if ($IngestSessionId)")]
    for forbidden in ("& ssh.exe", "& ssh ", "scp ", "rsync", "proxycommand", "remotecommand"):
        assert forbidden not in PC.lower()
    tailnet_full_block = SESSION[SESSION.index("if ($Mode -eq 'full' -and $CoordinateScope -eq 'GLOBAL_MAP')"):SESSION.index("& (Join-Path $PSScriptRoot 'start_go2_pc_stack.ps1') @pcStackArgs")]
    assert "$pcStackArgs.IngestSessionId = $SessionId" in tailnet_full_block
    assert "$pcStackArgs.IngestSessionId = $SessionId" not in SESSION[:SESSION.index("if ($Mode -eq 'full' -and $CoordinateScope -eq 'GLOBAL_MAP')")]
    assert SESSION.index("$SessionId = if ($SessionId)") < SESSION.index(
        "& (Join-Path $PSScriptRoot 'start_go2_pc_stack.ps1') @pcStackArgs"
    )


def test_explicit_local_odom_tailnet_session_does_not_start_the_global_automatic_worker():
    assert "[ValidateSet('GLOBAL_MAP', 'SESSION_ODOM_LOCAL')]" in SESSION
    assert "coordinate_scope = $CoordinateScope" in SESSION
    assert "$CoordinateScope -eq 'GLOBAL_MAP'" in SESSION
    assert "[string]$MissionId" in SESSION
    assert "[string]$TaskId" in SESSION
    assert "[string]$LeaseId" in SESSION
    assert "request.mission_id = $MissionId" in SESSION
    assert "request.task_id = $TaskId" in SESSION
    assert "if ($PSBoundParameters.ContainsKey('TaskRevision')) { $request.task_revision = $TaskRevision }" in SESSION
    assert "request.lease_id = $LeaseId" in SESSION


def test_pc_stack_configures_one_separate_wsl_local_odom_preview_pointer():
    assert "$sessionOdomLocalRoot = Join-Path $workspaceRoot 'output\\reconstruction\\session-odom-local'" in PC
    assert "$sessionOdomLocalStatusPath = Join-Path $runtimeRoot 'session-odom-local-current.json'" in PC
    assert "session_odom_local_status_path = $sessionOdomLocalStatusPath" in PC
    assert "session_odom_local_root = $sessionOdomLocalRoot" in PC
    assert "GO2_MAPPING_SESSION_ODOM_LOCAL_STATUS_PATH=$linuxSessionOdomLocalStatusPath" in PC
    assert "GO2_MAPPING_SESSION_ODOM_LOCAL_ROOT=$linuxSessionOdomLocalRoot" in PC


def test_session_uses_loopback_readiness_but_keeps_robot_endpoints_on_tailnet():
    assert "$localLiveHealthUrl = 'http://127.0.0.1:8767/api/stack/readiness'" in SESSION
    assert "Invoke-DirectRestMethod -Method Get -Uri $localLiveHealthUrl" in SESSION
    assert '$liveHealthUrl = "http://${pcTailnetIp}:8765/api/stack/readiness"' in SESSION
    assert '$receiverHealthUrl = "http://${pcTailnetIp}:8000/health"' in SESSION
    assert SESSION.index("start_go2_pc_stack.ps1") < SESSION.index(
        "Invoke-DirectRestMethod -Method Get -Uri $localLiveHealthUrl"
    )


def test_compatibility_entrypoint_propagates_mission_task_and_lease_arguments():
    compatibility = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    assert "[string]$MissionId" in compatibility
    assert "[string]$TaskId" in compatibility
    assert "[string]$LeaseId" in compatibility
    assert "MissionId = $MissionId" in compatibility
    assert "TaskId = $TaskId" in compatibility
    assert "TaskRevision = $TaskRevision" in compatibility
    assert "LeaseId = $LeaseId" in compatibility


def test_robot_live_preflight_uses_lightweight_readiness_instead_of_full_snapshot():
    preview = (ROOT / "tools" / "start_go2_live_preview_stream.sh").read_text(encoding="utf-8")

    expected = "http://${pc_tailnet_ip}:8765/api/live/readiness"
    assert f"live_health_url=${{GO2_MAPPING_LIVE_HEALTH_URL:-{expected}}}" in ROBOT
    assert f"health_url=${{GO2_MAPPING_LIVE_HEALTH_URL:-{expected}}}" in preview
    for script in (ROBOT, preview):
        assert "GO2_MAPPING_LIVE_HEALTH_URL:-http://${pc_tailnet_ip}:8765/api/snapshot" not in script
        assert "GO2_MAPPING_LIVE_HEALTH_URL:-http://${pc_tailnet_ip}:8765/api/readiness" not in script


def test_tailnet_http_bypasses_proxies_with_windows_powershell_compatible_null():
    for script in (PC, SESSION, PREPARE):
        assert "New-Object System.Net.WebProxy" not in script
        assert "Proxy = $null" in script or "-Proxy $null" in script


def test_pc_dashboard_loopback_readiness_bypasses_the_ambient_proxy():
    uri = "http://127.0.0.1:$dashboardInternalPort/api/stack/readiness"
    request_lines = [line for line in PC.splitlines() if uri in line]
    assert request_lines, f"PC readiness does not request {uri}"
    assert "-Proxy $null" in request_lines[0]
    test_dashboard = PC[PC.index("function Test-Dashboard"):PC.index("$status = [ordered]")]
    assert "/api/readiness" not in test_dashboard
    assert "/api/building/surface/manifest" not in test_dashboard
    assert "/openapi.json" not in test_dashboard


def test_tailnet_bound_receiver_health_and_port_checks_use_the_actual_listener():
    for script in (PC,):
        assert '$receiverHost = $tailnetIp' in script
        assert '$receiverHealthUri = "http://${receiverHost}:8000/health"' in script
        assert "Invoke-RestMethod -Uri $receiverHealthUri" in script
        assert "$health.capacity_probe -eq $true" in script
        assert "Test-NetConnection -ComputerName $receiverHost -Port 8000" in script
        # The dashboard process may use the local receiver endpoint internally;
        # the operator-facing health probe above must remain Tailnet-bound.
        assert '$receiverHealthUri = "http://${receiverHost}:8000/health"' in script


def test_dashboard_room_scan_uses_the_existing_tailnet_receiver_health_route():
    assert 'GO2_MAPPING_RECEIVER_HEALTH_URL=http://127.0.0.1:8000/health' in PC
    assert "'--bind', $tailnetIp" in PC
    assert "'--bind', '127.0.0.1'" in PC
    assert "'--allow-client', '127.0.0.1'" in PC
    assert "'--allow-client', '::1'" in PC
    assert "room_scan_runtime_revision = 3" in PC
    assert "function Test-ReceiverLoopback" in PC
    assert "Test-ReceiverLoopback" in PC


def test_robot_one_click_uses_dynamic_queues_and_current_unaligned_depth_truth():
    assert "GO2_MAPPING_PREVIEW_SHARE" not in ROBOT
    assert "GO2_MAPPING_PREVIEW_MIN_MBPS" not in ROBOT
    assert "GO2_MAPPING_PREVIEW_MAX_SHARE" not in ROBOT
    assert "preview_policy=realtime_priority" in ROBOT
    assert "high_fidelity=remaining_capacity" in ROBOT
    assert 'progressive_upload=1' in ROBOT
    assert 'export GO2_MAPPING_PROGRESSIVE_UPLOAD="$progressive_upload"' in ROBOT
    assert "GO2_MAPPING_CAMERA_MAX_WIDTH=${GO2_MAPPING_CAMERA_MAX_WIDTH:-640}" in ROBOT
    assert "GO2_MAPPING_CAMERA_MAX_HEIGHT=${GO2_MAPPING_CAMERA_MAX_HEIGHT:-480}" in ROBOT
    assert "GO2_MAPPING_CAMERA_DEPTH_MAX_WIDTH=${GO2_MAPPING_CAMERA_DEPTH_MAX_WIDTH:-320}" in ROBOT
    assert "GO2_MAPPING_CAMERA_DEPTH_MAX_HEIGHT=${GO2_MAPPING_CAMERA_DEPTH_MAX_HEIGHT:-240}" in ROBOT
    assert "GO2_MAPPING_CAMERA_DEPTH_ALIGNED=0" in ROBOT
    assert "/camera/depth/image_rect_raw" in ROBOT
    assert "run_go2_mapping_session.sh" in ROBOT
    assert "flock -n" in ROBOT
    assert 'nohup bash "$release_root/tools/start_go2_robot_scan.sh" supervise' in ROBOT
    assert 'bash "$runner" "$duration_s" "$mode" &' in ROBOT
    assert '9>&- >>"$log_file"' in ROBOT
    assert "http://192.0.2.10:8000" in ROBOT
    assert "pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}" in ROBOT
    assert "ws://${pc_tailnet_ip}:8765/ws/live/keyframe" in ROBOT
    assert 'export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"' in ROBOT
    assert 'export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"' in ROBOT
    assert "127.0.0.1:19001" not in ROBOT


def test_robot_scan_preserves_the_tailnet_control_session_identity():
    assert 'requested_session_id=${GO2_MAPPING_CAPTURE_SESSION_ID:-}' in ROBOT
    assert 'session_id="$requested_session_id"' in ROBOT
    assert 'export GO2_MAPPING_CAPTURE_SESSION_ID="$session_id"' in ROBOT


def test_robot_session_runner_launches_navigation_bridge_for_mission_goals():
    assert "start_go2_nav_bridge.sh" in SESSION_RUNNER
    assert "GO2_MAPPING_MISSION_ID" in SESSION_RUNNER
    assert "GO2_MAPPING_TASK_ID" in SESSION_RUNNER
    assert "GO2_MAPPING_TASK_LEASE_ID" in SESSION_RUNNER
    assert "navigation_executor.py" in SESSION_RUNNER


def test_robot_entrypoint_never_controls_motion_or_launches_realsense():
    lowered = ROBOT.lower()
    assert "sport" not in lowered
    assert "wirelesscontroller" not in lowered
    assert "realsense2_camera" not in lowered
    assert "cmd_vel" not in lowered
    assert "start_go2_robot_scan.sh" in DEPLOY


def test_robot_status_is_successful_when_no_scan_has_started(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "start_go2_robot_scan.sh"), "status"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GO2_MAPPING_RUNTIME_DIR": str(tmp_path)},
    )

    assert result.returncode == 0
    assert "GO2_ROBOT_SCAN_STOPPED" in result.stdout


def test_robot_supervisor_atomically_persists_truthful_exit_result(tmp_path):
    release_tools = tmp_path / "release" / "tools"
    release_tools.mkdir(parents=True)
    supervisor = release_tools / "start_go2_robot_scan.sh"
    shutil.copy2(ROOT / "tools" / "start_go2_robot_scan.sh", supervisor)
    runner = release_tools / "run_go2_mapping_session.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    session_id = "segment-20260806T060000Z"

    result = subprocess.run(
        ["bash", str(supervisor), "supervise", "1", "full", session_id],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GO2_MAPPING_RUNTIME_DIR": str(runtime),
            "GO2_MAPPING_SESSION_ENDPOINT": "http://127.0.0.1:9/api/robot/session",
        },
        timeout=15,
    )

    assert result.returncode == 7
    marker = runtime / "session-control" / "results" / f"{session_id}.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "session_id": session_id,
        "state": "FAILED",
        "exit_code": 7,
    }
    assert marker.read_text(encoding="utf-8") == (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    assert not list(marker.parent.glob("*.partial"))


def test_pc_offline_export_is_hash_verified_and_never_requires_robot_transport():
    path = ROOT / "tools" / "export_go2_reconstruction.ps1"
    assert path.is_file()
    script = path.read_text(encoding="utf-8")

    assert "/api/reconstruction/publish-measured" in script
    assert "Invoke-WebRequest" in script
    assert "Get-FileHash" in script
    assert ".partial" in script
    assert "delivery_authorized" in script
    assert "[a-z0-9_-]*" in script
    assert "tailscale" not in script.lower()
    assert "ssh.exe" not in script.lower()


def test_session_capture_control_delegates_to_authenticated_tailnet_http_only():
    compatibility = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    tailnet = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")
    combined = f"{compatibility}\n{tailnet}".lower()

    assert "start_go2_mapping_session_tailnet.ps1" in compatibility
    assert "invoke-directrestmethod" in tailnet.lower()
    assert "invoke-directrestmethod" in tailnet.lower()
    assert "/v1/sessions" in tailnet
    assert "/v1/preflight" in tailnet
    assert "acceptancecontextpath" in tailnet.lower()
    for forbidden in ("ssh.exe", "& ssh ", "scp ", "rsync", "proxycommand", "remotecommand"):
        assert forbidden not in combined


def test_tailnet_control_accepts_a_healthy_authenticated_relay_endpoint_without_ping_gate():
    assert "tailscale.exe ping" not in SESSION
    assert 'Invoke-DirectRestMethod -Method Get -Uri "$controlBase/v1/preflight" -Headers $headers' in SESSION


def test_robot_startup_health_probes_are_bounded_under_tailnet_loss():
    assert 'curl --connect-timeout 3 --max-time 6 --fail --silent --show-error "${health_auth[@]}" "$high_fidelity_url/health"' in ROBOT
    assert 'curl --connect-timeout 3 --max-time 6 --fail --silent --show-error "$live_health_url"' in ROBOT


def test_robot_heartbeat_distinguishes_recording_from_upload_finalization():
    assert "GO2_MAPPING_SESSION_PHASE_FILE" in ROBOT
    assert "printf 'STARTING\\n'" in ROBOT
    assert 'post_session_state "$heartbeat_state" "$session_id"' in ROBOT
    assert "heartbeat_state == STARTING || $heartbeat_state == RECORDING || $heartbeat_state == FINALIZING" in ROBOT
    assert "publish_session_phase RECORDING" in CAPTURE
    assert "publish_session_phase FINALIZING" in CAPTURE
    assert CAPTURE.index("publish_session_phase RECORDING") > CAPTURE.index("recorder_pid=$!")
    assert CAPTURE.index("publish_session_phase FINALIZING") < CAPTURE.index(
        'stream_go2_high_fidelity_queue.sh" --finalize'
    )


def test_robot_capture_consumes_the_session_control_stop_marker():
    assert "GO2_MAPPING_SESSION_STOP_FILE" in CAPTURE
    assert '[[ -n $session_stop_file && -f $session_stop_file ]]' in CAPTURE
    assert "graceful stop requested by session control" in CAPTURE


def test_lio_preview_rejects_the_observed_ten_edge_underconstrained_frame():
    assert "edgeFeatureMinValidNum: 20" in LIO_PROFILE


def test_wait_for_completion_retries_only_idempotent_status_gets():
    assert "function Invoke-SessionStatusWithRetry" in SESSION
    assert SESSION.count("Invoke-SessionStatusWithRetry -Uri $statusUri -Headers $headers") == 2
    assert "$started = Invoke-DirectRestMethod -Method Post" in SESSION
    assert "Invoke-SessionStatusWithRetry -Method Post" not in SESSION
    assert "status retry exhausted" in SESSION


def test_wait_for_completion_allows_bounded_derp_upload_finalization():
    compatibility = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")

    assert "[int]$CompletionTimeoutSeconds = 7200" in SESSION
    assert "AddSeconds($DurationSeconds + $CompletionTimeoutSeconds)" in SESSION
    assert "AddSeconds($DurationSeconds + 180)" not in SESSION
    assert "[int]$CompletionTimeoutSeconds = 7200" in compatibility
    assert "CompletionTimeoutSeconds = $CompletionTimeoutSeconds" in compatibility


def test_uncalibrated_capture_is_explicit_provisional_and_never_acceptance_authority():
    compatibility = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    assert BOOTSTRAP_MANIFEST.is_file()
    manifest = json.loads(BOOTSTRAP_MANIFEST.read_text(encoding="utf-8"))

    assert manifest == {
        "schema_version": 1,
        "status": "provisional",
        "purpose": "calibration_capture",
        "calibration_id": "PROVISIONAL_AUTOCAL_PENDING_v1",
        "geometry_authority": False,
        "appearance_authority": False,
        "acceptance_authority": False,
    }
    assert "[switch]$UncalibratedCapture" in SESSION
    assert "[switch]$UncalibratedCapture" in compatibility
    assert "calibration_capture_manifest.json" in SESSION
    assert "PHYSICAL_GO2_CALIBRATION_CAPTURE" in SESSION
    assert "acceptance_authority = -not $UncalibratedCapture" in SESSION
    assert "UncalibratedCapture requires full mode" in SESSION
    assert "UncalibratedCapture cannot use Depth Anything weights" in SESSION
    assert "UncalibratedCapture = $true" in compatibility


def test_calibration_startup_is_provisional_tailnet_only_and_never_captures_a_slot():
    assert CALIBRATION_START_PATH.is_file()
    script = CALIBRATION_START_PATH.read_text(encoding="utf-8")
    lowered = script.lower()

    assert "start_go2_mapping_session.ps1" in script
    assert "UncalibratedCapture = $true" in script
    assert "Mode = 'full'" in script
    assert "192.0.2.11" in script
    assert "192.0.2.10:8765" in script
    assert "depth_aligned_to_color" not in script

    for forbidden in (
        "ssh.exe",
        "scp ",
        "/api/calibration/detect-latest",
        "/api/calibration/slot/",
        "cmd_vel",
        "wirelesscontroller",
        "sportclient",
    ):
        assert forbidden not in lowered


def test_calibration_startup_explicitly_uses_local_odom_capture_scope():
    script = CALIBRATION_START_PATH.read_text(encoding="utf-8")
    start_block = script[
        script.index("if ($Action -eq 'start')") : script.index("$output = & $script:SessionEntrypoint")
    ]

    assert "$arguments.CoordinateScope = 'SESSION_ODOM_LOCAL'" in start_block
    assert "$arguments.CoordinateScope = 'GLOBAL_MAP'" not in start_block


def test_calibration_startup_parses_as_windows_powershell():
    if not POWERSHELL:
        import pytest

        pytest.skip("PowerShell is unavailable")

    path_literal = "'" + str(CALIBRATION_START_PATH).replace("'", "''") + "'"
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"{path_literal},"
        "[ref]$tokens,[ref]$errors) | Out-Null; "
        "if($errors.Count){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_calibration_startup_rejects_exhausted_pre_start_budget(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:sessionCalls = 0
$script:dashboardCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    if ($script:clockCalls -eq 1) {
        return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z')
    }
    return [DateTimeOffset]::Parse('2026-08-09T00:00:10Z')
}
function Invoke-CalibrationSessionEntrypoint {
    $script:sessionCalls += 1
    throw 'delegated start must not be called'
}
function Get-FormalDashboardJson {
    $script:dashboardCalls += 1
    throw 'Dashboard must not be called'
}
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "session_calls=$script:sessionCalls dashboard_calls=$script:dashboardCalls"
    exit 1
}
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "CALIBRATION_READINESS_TIMEOUT" in output
    assert "phase=pre_start" in output
    assert f"request={CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"session={CALIBRATION_TEST_SESSION_ID}" in output
    assert "session_calls=0 dashboard_calls=0" in output


def test_calibration_startup_reports_delegated_start_failure(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:sessionCalls = 0
$script:dashboardCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds($script:clockCalls - 1)
}
function Invoke-CalibrationSessionEntrypoint {
    $script:sessionCalls += 1
    throw 'synthetic delegated failure'
}
function Get-FormalDashboardJson {
    $script:dashboardCalls += 1
    throw 'Dashboard must not be called'
}
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "session_calls=$script:sessionCalls dashboard_calls=$script:dashboardCalls"
    exit 1
}
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SESSION_START_FAILED" in output
    assert f"request={CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"session={CALIBRATION_TEST_SESSION_ID}" in output
    assert "session_calls=1 dashboard_calls=0" in output


def test_calibration_startup_rejects_mismatched_context(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:sessionCalls = 0
$script:dashboardCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds($script:clockCalls - 1)
}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    $script:sessionCalls += 1
    @{
        schema_version = 1
        state = 'RUNNING'
        request_id = '00000000-0000-4000-8000-000000000099'
        session_id = $SessionId
        mode = 'full'
        duration_seconds = $DurationSeconds
        capture_purpose = 'calibration_capture'
        calibration_id = 'PROVISIONAL_AUTOCAL_PENDING_v1'
        calibration_status = 'provisional'
        acceptance_authority = $false
        pc_tailnet_ip = '192.0.2.10'
        robot_tailnet_ip = '192.0.2.11'
    } | ConvertTo-Json | Set-Content -LiteralPath $AcceptanceContextPath -Encoding UTF8
}
function Get-FormalDashboardJson {
    $script:dashboardCalls += 1
    throw 'Dashboard must not be called'
}
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "session_calls=$script:sessionCalls dashboard_calls=$script:dashboardCalls"
    exit 1
}
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ACCEPTANCE_CONTEXT_MISMATCH" in output
    assert f"request={CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"session={CALIBRATION_TEST_SESSION_ID}" in output
    assert "session_calls=1 dashboard_calls=0" in output
    assert not (ROOT / "mapping-data" / "calibration-startup" / CALIBRATION_TEST_SESSION_ID).exists()


def test_calibration_startup_readiness_contract_is_read_only():
    script = CALIBRATION_START_PATH.read_text(encoding="utf-8")

    assert "/api/stack/readiness" in script
    assert "/api/camera/status" in script
    assert "/api/calibration/progress" in script
    assert "READY_FOR_CALIBRATION_CAPTURE" in script
    assert "acceptance_authority" in script


def test_calibration_startup_waits_for_the_recorder_phase_before_ready(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:cameraSequence = 39
$script:statusCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds([Math]::Min($script:clockCalls - 1, 4))
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    $script:statusCalls += 1
    $phase = if ($script:statusCalls -eq 1) { 'STARTING' } else { 'RECORDING' }
    return (@{
        request_id = $RequestId
        state = 'RUNNING'
        phase = $phase
        request = @{ session_id = $SessionId }
    } | ConvertTo-Json -Depth 4)
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{
            ready = $true
            live_protocol = 3
            capabilities = [pscustomobject]@{ camera_mjpeg = $true }
        }
    }
    if ($Path -eq '/api/camera/status') {
        $script:cameraSequence += 1
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            sequence = $script:cameraSequence
            age_ms = 50
            preview_fps = 5.0
            distance_measurement_ready = $true
        }
    }
    if ($Path -eq '/api/calibration/progress') {
        return [pscustomobject]@{ active_slot = 'holdout' }
    }
    throw "unexpected path $Path"
}
Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
Write-Output "status_calls=$script:statusCalls"
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "READY_FOR_CALIBRATION_CAPTURE" in output
    assert "status_calls=2" in output


def test_calibration_startup_reports_ready_for_advancing_rgbd(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:cameraSequence = 39
$script:statusCalls = 0
$script:browserCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds([Math]::Min($script:clockCalls - 1, 2))
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    $script:statusCalls += 1
    return (@{
        request_id = $RequestId
        state = 'RUNNING'
        phase = 'RECORDING'
        request = @{ session_id = $SessionId }
    } | ConvertTo-Json -Depth 4)
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{
            ready = $true
            live_protocol = 3
            capabilities = [pscustomobject]@{ camera_mjpeg = $true }
        }
    }
    if ($Path -eq '/api/camera/status') {
        $script:cameraSequence += 1
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            sequence = $script:cameraSequence
            age_ms = 80
            preview_fps = 5.2
            distance_measurement_ready = $true
        }
    }
    if ($Path -eq '/api/calibration/progress') {
        return [pscustomobject]@{ active_slot = 'estimation_2' }
    }
    throw "unexpected path $Path"
}
function Open-FormalDashboard {
    $script:browserCalls += 1
    throw 'browser must not open'
}
Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
Write-Output "status_calls=$script:statusCalls browser_calls=$script:browserCalls"
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "READY_FOR_CALIBRATION_CAPTURE" in output
    assert f"request={CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"session={CALIBRATION_TEST_SESSION_ID}" in output
    assert "slot=estimation_2" in output
    assert "acceptance_authority=false" in output
    assert "camera_sequence=41" in output
    assert "status_calls=1 browser_calls=0" in output


def test_calibration_startup_rejects_camera_payload_missing_required_numeric_field(tmp_path):
    for missing_field in ("sequence", "age_ms"):
        camera_fields = {
            "sequence": "sequence = 40",
            "age_ms": "age_ms = 80",
        }
        camera_fields.pop(missing_field)
        result = _run_calibration_harness(
            tmp_path / missing_field,
            r"""
$script:clockCalls = 0
$script:statusCalls = 0
$script:browserCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds($script:clockCalls - 1)
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    $script:statusCalls += 1
    throw 'status must not be called'
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{
            ready = $true
            live_protocol = 3
            capabilities = [pscustomobject]@{ camera_mjpeg = $true }
        }
    }
    if ($Path -eq '/api/camera/status') {
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            __CAMERA_FIELDS__
            preview_fps = 5.2
            distance_measurement_ready = $true
        }
    }
    return [pscustomobject]@{ active_slot = 'estimation_1' }
}
function Open-FormalDashboard { $script:browserCalls += 1 }
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10 -NoOpenDashboard
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "status_calls=$script:statusCalls browser_calls=$script:browserCalls"
    exit 1
}
""".replace("__CAMERA_FIELDS__", "\n            ".join(camera_fields.values())),
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "CALIBRATION_READINESS_TIMEOUT" in output
        assert "READY_FOR_CALIBRATION_CAPTURE" not in output
        assert "status_calls=0 browser_calls=0" in output


def test_calibration_startup_bounds_dashboard_body_reads_and_rejects_mutating_paths():
    script = CALIBRATION_START_PATH.read_text(encoding="utf-8")

    assert "$request.Timeout = 5000" in script
    assert "$request.ReadWriteTimeout = 5000" in script
    assert "$request.Proxy = $null" in script
    assert "$reader.Dispose()" in script
    assert "$response.Dispose()" in script
    assert "/api/calibration/detect-latest" not in script
    assert "/api/calibration/slot/" not in script


def test_calibration_startup_timeout_after_session_start_preserves_diagnostics(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:startCalls = 0
$script:statusCalls = 0
$script:dashboardCalls = 0
$script:browserCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    if ($script:clockCalls -eq 1) { return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z') }
    if ($script:clockCalls -eq 2) { return [DateTimeOffset]::Parse('2026-08-09T00:00:01Z') }
    return [DateTimeOffset]::Parse('2026-08-09T00:00:10Z')
}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        $script:startCalls += 1
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    $script:statusCalls += 1
    throw 'status must not be called'
}
function Get-FormalDashboardJson { $script:dashboardCalls += 1; throw 'Dashboard must not be called' }
function Open-FormalDashboard { $script:browserCalls += 1 }
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "start_calls=$script:startCalls status_calls=$script:statusCalls dashboard_calls=$script:dashboardCalls browser_calls=$script:browserCalls"
    exit 1
}
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "CALIBRATION_READINESS_TIMEOUT" in output
    assert f"request={CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"session={CALIBRATION_TEST_SESSION_ID}" in output
    assert f"status_command=.\\tools\\start_go2_mapping_session.ps1 -Action status -RequestId {CALIBRATION_TEST_REQUEST_ID}" in output
    assert f"stop_command=.\\tools\\start_go2_mapping_session.ps1 -Action stop -RequestId {CALIBRATION_TEST_REQUEST_ID}" in output
    assert "start_calls=1 status_calls=0 dashboard_calls=0 browser_calls=0" in output


def test_calibration_startup_rejects_ended_session_before_browser(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:cameraSequence = 9
$script:browserCalls = 0
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds([Math]::Min($script:clockCalls - 1, 2))
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    return (@{
        request_id = $RequestId
        state = 'COMPLETED'
        request = @{ session_id = $SessionId }
    } | ConvertTo-Json -Depth 4)
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{ ready = $true; live_protocol = 3; capabilities = [pscustomobject]@{ camera_mjpeg = $true } }
    }
    if ($Path -eq '/api/camera/status') {
        $script:cameraSequence += 1
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            sequence = $script:cameraSequence
            age_ms = 50
            preview_fps = 5.0
            distance_measurement_ready = $true
        }
    }
    return [pscustomobject]@{ active_slot = 'estimation_1' }
}
function Open-FormalDashboard { $script:browserCalls += 1 }
try {
    Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10
    exit 90
} catch {
    Write-Output $_.Exception.Message
    Write-Output "browser_calls=$script:browserCalls"
    exit 1
}
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SESSION_NOT_RUNNING" in output
    assert "state=COMPLETED" in output
    assert "READY_FOR_CALIBRATION_CAPTURE" not in output
    assert "browser_calls=0" in output


def test_calibration_startup_opens_formal_dashboard_only_after_running_status(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:cameraSequence = 19
$script:callOrder = New-Object System.Collections.Generic.List[string]
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds([Math]::Min($script:clockCalls - 1, 2))
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        $script:callOrder.Add('start')
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    $script:callOrder.Add('status')
    return (@{ request_id = $RequestId; state = 'RUNNING'; phase = 'RECORDING'; request = @{ session_id = $SessionId } } | ConvertTo-Json -Depth 4)
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{ ready = $true; live_protocol = 3; capabilities = [pscustomobject]@{ camera_mjpeg = $true } }
    }
    if ($Path -eq '/api/camera/status') {
        $script:cameraSequence += 1
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            sequence = $script:cameraSequence
            age_ms = 40
            preview_fps = 5.4
            distance_measurement_ready = $true
        }
    }
    return [pscustomobject]@{ active_slot = 'holdout' }
}
function Open-FormalDashboard {
    param([string]$Url)
    if ($Url -cne 'http://192.0.2.10:8765/?view=map') { throw "wrong URL $Url" }
    $script:callOrder.Add('browser')
}
Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10
Write-Output "call_order=$($script:callOrder -join ',')"
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "READY_FOR_CALIBRATION_CAPTURE" in output
    assert "slot=holdout" in output
    assert "call_order=start,status,browser" in output


def test_calibration_startup_keeps_readiness_when_browser_launch_fails(tmp_path):
    result = _run_calibration_harness(
        tmp_path,
        r"""
$script:clockCalls = 0
$script:cameraSequence = 29
function Get-CalibrationUtcNow {
    $script:clockCalls += 1
    return [DateTimeOffset]::Parse('2026-08-09T00:00:00Z').AddSeconds([Math]::Min($script:clockCalls - 1, 2))
}
function Wait-CalibrationPoll {}
function Invoke-CalibrationSessionEntrypoint {
    param(
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )
    if ($Action -eq 'start') {
        Write-TestCalibrationContext $RequestId $SessionId $DurationSeconds $AcceptanceContextPath
        return
    }
    return (@{ request_id = $RequestId; state = 'RUNNING'; phase = 'RECORDING'; request = @{ session_id = $SessionId } } | ConvertTo-Json -Depth 4)
}
function Get-FormalDashboardJson {
    param([string]$Path)
    if ($Path -eq '/api/stack/readiness') {
        return [pscustomobject]@{ ready = $true; live_protocol = 3; capabilities = [pscustomobject]@{ camera_mjpeg = $true } }
    }
    if ($Path -eq '/api/camera/status') {
        $script:cameraSequence += 1
        return [pscustomobject]@{
            state = 'LIVE_RGB_DEPTH_UNALIGNED'
            sequence = $script:cameraSequence
            age_ms = 30
            preview_fps = 5.0
            distance_measurement_ready = $true
        }
    }
    return [pscustomobject]@{ active_slot = 'estimation_3' }
}
function Open-FormalDashboard { throw 'synthetic browser failure' }
Invoke-Go2CalibrationStartup -DurationSeconds 120 -LiveReadinessTimeoutSeconds 10
""",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Dashboard could not be opened automatically" in output
    assert "http://192.0.2.10:8765/?view=map" in output
    assert "READY_FOR_CALIBRATION_CAPTURE" in output
