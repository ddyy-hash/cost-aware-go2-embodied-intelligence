from pathlib import Path
import os
import re
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_optional_depth_scale_timeout_cannot_abort_runtime_manifest():
    writer = (ROOT / "tools" / "write_go2_capture_runtime_manifest.sh").read_text(encoding="utf-8")
    assert re.search(
        r"candidate=\$\(timeout 3s ros2 param get [^\n]+\\\n"
        r"\s+\| awk [^\n]+\|\| true\)",
        writer,
    )


def test_full_capture_rotates_immutable_bag_parts_and_streams_only_closed_parts():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
    watcher = (ROOT / "tools" / "stream_go2_high_fidelity_queue.sh").read_text(encoding="utf-8")

    assert "GO2_MAPPING_ROTATE_MAX_BYTES" in capture
    assert "--max-bag-size" in capture
    assert "--max-bag-duration" not in capture
    assert "stream_go2_high_fidelity_queue.sh" in capture
    assert "--finalize" in capture
    assert "-name '*.db3.zstd'" in watcher
    assert "highest indexed bag remains open" not in watcher
    assert "progressive_artifact_uploader.py" in watcher
    assert "GO2_MAPPING_UPLOAD_RETRY_ATTEMPTS" in watcher
    assert "high-fidelity queue: upload attempt" in watcher
    assert "GO2_MAPPING_HIGH_FIDELITY_RATE_MBPS" in watcher
    assert "GO2_MAPPING_BANDWIDTH_STATE" in watcher
    assert "--bandwidth-state" in watcher
    assert "--total-ceiling-mbps" in watcher
    assert "--preview-share" not in watcher
    assert "--route-id" in watcher
    assert "GO2_MAPPING_UPLOAD_CHUNK_KIB:-8192" in watcher
    assert "GO2_MAPPING_UPLOAD_WORKERS:-4" in watcher
    assert "GO2_MAPPING_CAPACITY_PROBE_INTERVAL_SECONDS:-30" in watcher
    assert "--parallel-workers" in watcher
    assert "--capacity-probe-interval" in watcher
    assert "--await-reconstruction" not in watcher
    assert "GO2_MAPPING_DELETE_RECONSTRUCTED" in watcher
    assert "RECONSTRUCTION_RELEASE_AUTHORIZED" in watcher
    assert 'rm -f -- "$file"' in watcher
    assert 'deprecated_marker="$segment/DEPRECATED.json"' in watcher
    assert "refusing deprecated segment" in watcher


def test_pc_stack_prefers_tailnet_direct_high_fidelity_without_an_ssh_data_tunnel():
    pc = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")
    robot = (ROOT / "tools" / "stream_go2_high_fidelity_queue.sh").read_text(encoding="utf-8")
    uploader = (ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "progressive_artifact_uploader.py").read_text(encoding="utf-8")

    assert "$tailnetIpProcess.StartInfo.Arguments = 'ip -4'" in pc
    assert "--bind" in pc
    assert "--allow-client" in pc
    assert "GO2_MAPPING_HIGH_FIDELITY_URL" in robot
    assert "http://192.0.2.10:8000" in robot
    assert "GO2_MAPPING_HIGH_FIDELITY_URL" in uploader
    assert "http://192.0.2.10:8000" in uploader
    assert "127.0.0.1:19000" not in uploader
    assert "127.0.0.1:19000:127.0.0.1:8000" not in pc


def test_progressive_capture_defaults_to_eight_mib_micro_segments_everywhere():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")
    robot = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")
    uplink = (ROOT / "mapping" / "config" / "uplink.yaml").read_text(encoding="utf-8")

    assert "GO2_MAPPING_ROTATE_MAX_BYTES:-8388608" in capture
    assert "GO2_MAPPING_ROTATE_MAX_BYTES:-8388608" in runner
    assert "GO2_MAPPING_ROTATE_MAX_BYTES:-8388608" in robot
    assert "rotate_max_bytes: 8388608" in uplink


def test_live_preview_publishes_demand_without_reading_high_fidelity_allocation_state():
    preview = (ROOT / "tools" / "start_go2_live_preview_stream.sh").read_text(encoding="utf-8")
    watcher = (ROOT / "tools" / "stream_go2_high_fidelity_queue.sh").read_text(encoding="utf-8")
    robot = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")
    producer = (
        ROOT
        / "mapping"
        / "robot"
        / "go2_mapping_lio"
        / "go2_mapping_lio"
        / "live_keyframe_producer.py"
    ).read_text(encoding="utf-8")

    assert "GO2_MAPPING_BANDWIDTH_STATE" not in preview
    assert "--bandwidth-state" not in preview
    assert "BandwidthAllocationStore" not in producer
    assert "--route-id" not in preview
    assert "--route-id" not in producer
    assert "bandwidth-allocation.json" in watcher
    assert "GO2_MAPPING_BOOTSTRAP_BANDWIDTH_MBPS:-4.0" in preview
    assert "GO2_MAPPING_PREVIEW_SHARE" not in preview
    assert "GO2_MAPPING_PREVIEW_MIN_MBPS" not in preview
    assert "GO2_MAPPING_PREVIEW_MAX_SHARE" not in preview
    assert "GO2_MAPPING_ROUTE_ID" in watcher
    assert "--route-id" in watcher
    assert "--bootstrap-total-mbps" in preview
    assert "GO2_MAPPING_DETAIL_SURFACE_CELL_M:-0.005" in preview
    assert "GO2_MAPPING_DETAIL_PAYLOAD_LIMIT_BYTES:-4194304" in preview
    assert "GO2_MAPPING_DETAIL_TARGET_HZ:-5" in preview
    assert "GO2_MAPPING_LIVE_MAX_POINTS" not in preview
    assert "GO2_MAPPING_PREVIEW_MAX_HZ" not in robot
    assert robot.count("export GO2_MAPPING_CAMERA_MAX_HZ=") == 1
    assert "GO2_MAPPING_PREVIEW_DEMAND_CADENCE_HZ" not in robot
    assert "GO2_MAPPING_PREVIEW_DEMAND_CADENCE_HZ" not in preview
    assert "--preview-demand-cadence-hz" not in producer
    assert "_publish_preview_demand(payload, cadence_hz=detail_hz)" in producer
    assert 'payload["_detail_cadence_hz"] = detail_hz' in producer
    assert "--uplink-budget-mbps" not in preview
    assert "--uplink-budget-mbps" not in producer
    assert "GO2_MAPPING_BUILDING_ID" in preview
    assert "GO2_MAPPING_CAPTURE_SESSION_ID" in preview
    assert "--building-id" in preview
    assert "--session-id" in preview
    assert "GO2_MAPPING_LIVE_DELIVERY_WINDOW:-20" in preview
    assert "GO2_MAPPING_LIVE_PREVIEW_QUEUE_SIZE:-64" in preview


def test_session_runner_enables_the_independent_high_fidelity_queue_for_full_mapping():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")

    assert "GO2_MAPPING_PROGRESSIVE_UPLOAD" in runner
    assert "GO2_MAPPING_ROTATE_MAX_BYTES" in runner
    assert "GO2_MAPPING_UPLINK_BUDGET_MBPS" in runner
    assert "GO2_MAPPING_PREVIEW_SHARE" not in runner
    assert "preview_policy=realtime_priority" in runner
    assert "high_fidelity=remaining_capacity" in runner
    assert "high-fidelity receiver unavailable; recording continues locally" in runner
    assert "mapping session refused: Tailnet high-fidelity receiver is unavailable" not in runner
    assert 'DEPRECATED.json' in runner
    assert "mapping session refused to publish deprecated segment" in runner


def test_capture_keeps_successful_raw_recording_when_upload_is_temporarily_unavailable():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert "upload_pending=0" in capture
    assert "upload_pending=1" in capture
    assert "high-fidelity upload pending; raw artifacts remain local" in capture
    assert "upload_pending=%s" in capture


def test_capture_treats_duration_sigint_exit_130_as_a_normal_close():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert 'if [[ $status -eq 124 || $status -eq 130 ]]; then' in capture


def test_session_runner_cleans_only_lio_and_live_processes_started_by_this_session():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")

    assert "lio_started_by_session=0" in runner
    assert "live_started_by_session=0" in runner
    assert "cleanup_session_owned_processes" in runner
    assert 'if [[ $live_started_by_session == 1 ]]' in runner
    assert 'if [[ $lio_started_by_session == 1 ]]' in runner
    assert "live_keyframe_producer.py" in runner
    assert "start_go2_lio_preview.sh" in runner
    assert "pkill" not in runner


def test_session_runner_replaces_a_live_producer_from_another_capture_session():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")

    assert 'pid_matches_command "$live_pid_file" "--session-id $GO2_MAPPING_CAPTURE_SESSION_ID"' in runner
    assert 'stale_live_pid=$(<"$live_pid_file")' in runner
    assert re.search(
        r'stop_session_owned_process "\$stale_live_pid" "\$live_pid_file"\s+\\?\s*'
        r'"live_keyframe_producer\.py"',
        runner,
    )


@pytest.mark.skipif(os.name == "nt", reason="requires Linux /proc process identity")
def test_session_runner_rejects_a_live_unrelated_process_from_a_stale_pid_file(tmp_path):
    bash = shutil.which("bash")
    assert bash is not None
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^pid_is_alive\(\) \{\n.*?^\}\n\n"
        r"pid_matches_command\(\) \{\n.*?^\}\n",
        runner,
    )
    assert match is not None, "runner must validate both PID liveness and command identity"

    unrelated = subprocess.Popen(["sleep", "30"])
    expected = subprocess.Popen(
        [bash, "-c", "exec -a live_keyframe_producer.py sleep 30"]
    )
    try:
        pid_file = tmp_path / "live-preview.pid"
        pid_file.write_text(f"{unrelated.pid}\n", encoding="ascii")
        probe = tmp_path / "probe.sh"
        probe.write_text(
            "set -Eeuo pipefail\n"
            + match.group(0)
            + f"pid_matches_command {pid_file} live_keyframe_producer.py\n",
            encoding="utf-8",
        )
        result = subprocess.run([bash, str(probe)], check=False)
        assert result.returncode != 0

        pid_file.write_text(f"{expected.pid}\n", encoding="ascii")
        result = subprocess.run([bash, str(probe)], check=False)
        assert result.returncode == 0
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
        expected.terminate()
        expected.wait(timeout=5)


def test_robot_scan_supervisor_reports_structured_capture_heartbeats():
    robot = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")

    assert "supervise" in robot
    assert "/api/robot/session" in robot
    assert "heartbeat_interval_s=2" in robot
    assert 'post_session_state "$heartbeat_state" "$session_id"' in robot
    assert "heartbeat_state == RECORDING || $heartbeat_state == FINALIZING" in robot
    assert "post_session_state STOPPED" in robot


def test_robot_scan_supervisor_terminates_its_isolated_runner_process_group():
    robot = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")

    assert 'setsid bash "$runner" "$duration_s" "$mode" &' in robot
    assert 'kill -TERM -- "-$runner_pid"' in robot


def test_camera_preview_has_independent_five_hz_sampling_and_delivery_threads():
    producer = (
        ROOT
        / "mapping"
        / "robot"
        / "go2_mapping_lio"
        / "go2_mapping_lio"
        / "live_keyframe_producer.py"
    ).read_text(encoding="utf-8")

    assert "target=self._camera_sampling_loop" in producer
    assert "def _camera_sampling_loop" in producer
    assert "period_s = 1.0 / self.args.camera_max_hz" in producer
    assert "next_sample = max(next_sample + period_s, time.monotonic())" in producer
    assert "next_sample = max(next_sample + period_s, time.monotonic() + period_s)" not in producer
    assert "self.create_timer(1.0 / args.camera_max_hz, self._queue_camera_preview)" not in producer
    assert '"message_type": "camera_preview"' in producer
    assert 'if message_type == "camera_preview":' in producer
    assert "def _queue_camera_preview" in producer
    assert "receive_timeout_s = self.args.camera_ack_timeout_s" in producer
    assert "self._camera_socket = None" in producer
    assert "self._camera_delivery_queue: queue.Queue[dict] = queue.Queue(maxsize=1)" in producer
    assert "target=self._camera_delivery_loop" in producer
    assert "def _camera_delivery_loop" in producer
    assert "if color_timestamp_ns <= 0:" in producer
    assert "color_timestamp_ns = time.time_ns()" in producer
    assert "def _enqueue_latest_camera" in producer
    assert 'parser.add_argument("--camera-delivery-window", type=int, default=1)' in producer
    assert "while len(camera_pending) < self.args.camera_delivery_window:" in producer
    assert "camera_pending.clear()" in producer
    assert 'parser.add_argument("--camera-ack-timeout-s", type=float, default=2.0)' in producer
    assert "time.monotonic() - camera_pending[0][1] > self.args.camera_ack_timeout_s" in producer
    assert "camera acknowledgement deadline exceeded" in producer
    camera_callback = producer.split("def _queue_camera_preview", 1)[1].split("def _cloud_xyz", 1)[0]
    assert "self._enqueue_latest_camera(payload)" in camera_callback
    assert "self._enqueue_latest(payload)" not in camera_callback
    camera_payload = producer.split("def _camera_payload", 1)[1].split("def on_cloud", 1)[0]
    assert "now - self.last_camera_attempt" not in camera_payload
    cloud_callback = producer.split("def on_cloud", 1)[1].split("def build_parser", 1)[0]
    assert "camera = self._camera_payload" not in cloud_callback
    assert "camera=None" in cloud_callback


def test_texture_keyframes_use_a_separate_two_hz_ordered_socket_and_never_the_preview_queue():
    producer = (
        ROOT
        / "mapping"
        / "robot"
        / "go2_mapping_lio"
        / "go2_mapping_lio"
        / "live_keyframe_producer.py"
    ).read_text(encoding="utf-8")

    assert "target=self._texture_sampling_loop" in producer
    assert "target=self._texture_delivery_loop" in producer
    assert "self._texture_socket = None" in producer
    assert "self._texture_delivery_queue: queue.Queue[dict]" in producer
    assert "period_s = 1.0 / self.args.texture_max_hz" in producer
    assert 'default=2.0' in producer.split('parser.add_argument("--texture-max-hz"', 1)[1].split("\n", 1)[0]
    assert '"ws://192.0.2.10:8765/ws/live/texture"' in producer
    assert "build_texture_keyframe_message(" in producer
    texture_source = (
        ROOT
        / "mapping"
        / "robot"
        / "go2_mapping_lio"
        / "go2_mapping_lio"
        / "texture_keyframe_source.py"
    ).read_text(encoding="utf-8")
    assert '"message_type": "texture_keyframe"' in texture_source
    assert '"texture_keyframe_ack"' in producer
    texture_callback = producer.split("def _queue_texture_keyframe", 1)[1].split("def _cloud_xyz", 1)[0]
    assert "self._texture_delivery_queue.put_nowait" in texture_callback
    assert "self._enqueue_latest_camera" not in texture_callback
    assert "self._enqueue_latest(payload)" not in texture_callback
    launcher = (ROOT / "tools" / "start_go2_live_preview_stream.sh").read_text(encoding="utf-8")
    assert "GO2_MAPPING_TEXTURE_KEYFRAMES" in launcher
    assert "GO2_MAPPING_TEXTURE_ENDPOINT" in launcher
    assert "--texture-enabled" in launcher
    assert "--texture-endpoint" in launcher


def test_robot_session_cleanup_checks_process_existence_outside_bash_arithmetic():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")

    assert "for ((attempt = 0; attempt < 100; attempt++)); do" in runner
    assert "[[ -e /proc/$pid ]] || break" in runner
    assert "attempt < 100 && -e /proc/$pid" not in runner


def test_pc_session_passes_the_verified_inbox_and_high_fidelity_receiver_status_to_the_dashboard():
    session = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert "GO2_MAPPING_INGEST_ROOT" in session
    assert "GO2_MAPPING_TRANSFER_STATUS" in session
    assert "GO2_MAPPING_LIFECYCLE_ROOT" in session
    assert "--status-path" in session
    assert "GO2_MAPPING_PROGRESSIVE_UPLOAD" in capture
    assert "progressive_protocol" in session
    assert "Stop-Process" in session


def test_robot_release_contains_the_progressive_high_fidelity_queue_runner():
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "stream_go2_high_fidelity_queue.sh" in deploy


def test_pc_stack_defaults_to_direct_tailscale_without_an_ssh_business_path():
    session = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    stack = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")

    assert "start_go2_mapping_session_tailnet.ps1" in session
    assert "transport = 'TAILSCALE_DIRECT'" in session
    assert "[string]$HighFidelityTransport = 'direct'" in stack
    assert "[bool]$UseTailscaleProxy = $false" in stack
    assert "ProxyCommand=tailscale.exe nc %h %p" not in session


def test_full_session_scopes_robot_capture_and_dense_fusion_to_one_segment():
    session = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    stack = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")
    tailnet = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert "[string]$SessionId" in session
    assert "[string]$SessionId" in tailnet
    assert "session_id = $SessionId" in tailnet
    assert "GO2_MAPPING_INGEST_ROOT" in stack
    assert "GO2_MAPPING_CAPTURE_SESSION_ID" in capture
    assert "capture session id must match" in capture


def _watcher_environment(tmp_path, uploader):
    environment = os.environ.copy()
    environment.update(
        {
            "GO2_MAPPING_UPLOADER": str(uploader),
            "GO2_MAPPING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "GO2_MAPPING_HIGH_FIDELITY_RATE_MBPS": "10",
            "GO2_MAPPING_UPLINK_BUDGET_MBPS": "20",
            "GO2_MAPPING_UPLOAD_RETRY_ATTEMPTS": "1",
            "GO2_MAPPING_RECONSTRUCTION_WAIT_SECONDS": "1",
            "GO2_MAPPING_DELETE_RECONSTRUCTED": "1",
        }
    )
    return environment


def test_robot_deletes_only_a_reconstruction_authorized_closed_artifact(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120000Z"
    segment.mkdir(parents=True)
    artifact = segment / "segment-20260731T120000Z_0.db3.zstd"
    artifact.write_bytes(b"closed-verified-artifact")
    uploader = tmp_path / "authorized-uploader.sh"
    uploader.write_text(
        "#!/usr/bin/env python3\nprint('{\"state\": \"PC_RECONSTRUCTION_INGESTED\", \"release_authorized\": true}')\n",
        encoding="utf-8",
    )
    uploader.chmod(0o755)

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), "--finalize", str(segment), str(root)],
        env=_watcher_environment(tmp_path, uploader),
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert not artifact.exists()
    assert "RECONSTRUCTION_RELEASE_AUTHORIZED" in result.stdout
    audit = tmp_path / "runtime" / "high-fidelity-upload-state" / f"{segment.name}.released.tsv"
    assert artifact.name in audit.read_text(encoding="utf-8")


def test_robot_upload_does_not_wait_for_reconstruction_and_retains_unreleased_artifact(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120001Z"
    segment.mkdir(parents=True)
    artifact = segment / "segment-20260731T120001Z_0.db3.zstd"
    artifact.write_bytes(b"closed-but-not-reconstructed")
    uploader = tmp_path / "unverified-uploader.sh"
    arguments = tmp_path / "uploader-arguments.txt"
    uploader.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['UPLOADER_ARGUMENTS']).write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
        "print('{\"state\": \"SHA256_VERIFIED\", \"release_authorized\": false}')\n",
        encoding="utf-8",
    )
    uploader.chmod(0o755)
    environment = _watcher_environment(tmp_path, uploader)
    environment["UPLOADER_ARGUMENTS"] = str(arguments)

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), "--finalize", str(segment), str(root)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert artifact.exists()
    assert "RECONSTRUCTION_RELEASE_PENDING" in result.stdout
    assert "--await-reconstruction" not in arguments.read_text(encoding="utf-8")


def test_live_watcher_keeps_pending_artifacts_after_a_temporary_upload_failure(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120002Z"
    segment.mkdir(parents=True)
    artifact = segment / "segment-20260731T120002Z_0.db3.zstd"
    artifact.write_bytes(b"retain-until-receiver-recovers")
    uploader = tmp_path / "temporarily-unavailable-uploader.sh"
    uploader.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    uploader.chmod(0o755)
    environment = _watcher_environment(tmp_path, uploader)
    environment["GO2_MAPPING_PROGRESSIVE_POLL_SECONDS"] = "0.1"

    process = subprocess.Popen(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), str(segment), str(root)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.35)
        assert process.poll() is None, "live watcher exited after a recoverable upload failure"
        assert artifact.exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_live_watcher_priority_defers_pre_target_closed_bag_parts(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120002Z"
    segment.mkdir(parents=True)
    early = segment / "segment-20260731T120002Z_0.db3.zstd"
    target = segment / "segment-20260731T120002Z_470.db3.zstd"
    early.write_bytes(b"early-closed-artifact")
    target.write_bytes(b"target-window-artifact")
    uploaded = tmp_path / "uploaded-paths.txt"
    uploader = tmp_path / "recording-uploader.py"
    uploader.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "with pathlib.Path(os.environ['UPLOADED_PATHS']).open('a', encoding='utf-8') as output:\n"
        "    output.write(sys.argv[2] + '\\n')\n"
        "print('{\"state\": \"SHA256_VERIFIED\", \"release_authorized\": false}')\n",
        encoding="utf-8",
    )
    uploader.chmod(0o755)
    environment = _watcher_environment(tmp_path, uploader)
    environment.update(
        {
            "GO2_MAPPING_PRIORITY_PART_MIN_INDEX": "470",
            "GO2_MAPPING_PROGRESSIVE_POLL_SECONDS": "0.1",
            "UPLOADED_PATHS": str(uploaded),
        }
    )

    process = subprocess.Popen(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), str(segment), str(root)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not uploaded.is_file():
            time.sleep(0.05)
        assert uploaded.is_file(), "priority watcher did not upload the target-range part"
        time.sleep(0.25)
    finally:
        process.terminate()
        process.wait(timeout=5)

    paths = uploaded.read_text(encoding="utf-8").splitlines()
    assert paths
    assert set(paths) == {f"{segment.name}/{target.name}"}


def test_finalize_skips_verified_bag_when_raw_deletion_is_disabled(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120004Z"
    segment.mkdir(parents=True)
    artifact = segment / "segment-20260731T120004Z_0.db3.zstd"
    artifact.write_bytes(b"already-verified-immutable-artifact")
    invoked = tmp_path / "uploader-invoked.txt"
    uploader = tmp_path / "recording-uploader.py"
    uploader.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib\n"
        "pathlib.Path(os.environ['UPLOADER_INVOKED']).write_text('called', encoding='ascii')\n"
        "print('{\\\"state\\\": \\\"SHA256_VERIFIED\\\", \\\"release_authorized\\\": false}')\n",
        encoding="utf-8",
    )
    uploader.chmod(0o755)
    environment = _watcher_environment(tmp_path, uploader)
    environment.update({"GO2_MAPPING_DELETE_RECONSTRUCTED": "0", "UPLOADER_INVOKED": str(invoked)})
    state = tmp_path / "runtime" / "high-fidelity-upload-state" / f"{segment.name}.uploaded"
    state.parent.mkdir(parents=True)
    state.write_text(f"{artifact.relative_to(root).as_posix()}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), "--finalize", str(segment), str(root)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert artifact.exists()
    assert not invoked.exists(), "verified retained bag must not be uploaded again"


def test_terminating_live_watcher_also_terminates_its_active_uploader(tmp_path):
    root = tmp_path / "robot-data"
    segment = root / "segment-20260731T120003Z"
    segment.mkdir(parents=True)
    (segment / "segment-20260731T120003Z_0.db3.zstd").write_bytes(b"closed-artifact")
    uploader_pid = tmp_path / "uploader.pid"
    uploader = tmp_path / "blocking-uploader.py"
    uploader.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['UPLOADER_PID']).write_text(str(os.getpid()), encoding='ascii')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    uploader.chmod(0o755)
    environment = _watcher_environment(tmp_path, uploader)
    environment["UPLOADER_PID"] = str(uploader_pid)
    environment["GO2_MAPPING_PROGRESSIVE_POLL_SECONDS"] = "0.1"

    process = subprocess.Popen(
        ["bash", str(ROOT / "tools" / "stream_go2_high_fidelity_queue.sh"), str(segment), str(root)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not uploader_pid.is_file():
            time.sleep(0.05)
        assert uploader_pid.is_file(), "watcher did not start the test uploader"
        child_pid = int(uploader_pid.read_text(encoding="ascii"))
        assert Path(f"/proc/{child_pid}").exists()
        process.terminate()
        process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
            time.sleep(0.05)
        assert not Path(f"/proc/{child_pid}").exists(), "uploader survived its watcher"
        assert "unbound variable" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            os.kill(child_pid, 9)
