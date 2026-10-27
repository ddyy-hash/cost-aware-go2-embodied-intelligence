import re
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
GITATTRIBUTES = ROOT / ".gitattributes"
PREFLIGHT = ROOT / "tools" / "verify_go2_mapping_robot.sh"
DEPLOY = ROOT / "tools" / "deploy_go2_mapping_robot.ps1"
WSL_INSTALL = ROOT / "tools" / "install_wsl_mapping.ps1"
LAB_PREPARE = ROOT / "tools" / "prepare_go2_mapping_lab.ps1"
PC_ENVIRONMENT = ROOT / "mapping" / "pc" / "environment.yml"
LIVENESS_PROBE = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "ros_topic_liveness.py"
WIRED_DEV_ACCESS = ROOT / "tools" / "enable_go2_wired_dev_access.ps1"
WIFI_RECONNECT = ROOT / "tools" / "install_go2_wifi_reconnect.sh"
ROBOT_MANIFEST = ROOT / "mapping" / "config" / "robot_runtime_manifest.json"
MODEL_MANIFEST = ROOT / "mapping" / "config" / "pc_model_manifest.json"
ASSET_PREPARE = ROOT / "tools" / "prepare_go2_mapping_assets.ps1"
RELEASE_VERIFY = ROOT / "tools" / "verify_go2_mapping_release.sh"
RELEASE_BUNDLE = ROOT / "tools" / "prepare_go2_mapping_robot_bundle.ps1"
DEPLOY_LATEST = ROOT / "tools" / "deploy_latest_go2_mapping_robot.ps1"
ROBOT_WHEELHOUSE_MANIFEST = ROOT / "mapping" / "config" / "robot_wheelhouse_manifest.json"
ROBOT_RUNTIME_INSTALL = ROOT / "tools" / "install_go2_mapping_python_runtime.sh"
ROUTER_INSTALL = ROOT / "tools" / "install_go2_pgtt_terrain_router.sh"

from mapping.pc.go2_mapping_assets import (  # noqa: E402
    build_model_manifest,
    verify_model_manifest,
)


def test_posix_shell_sources_are_pinned_to_lf_for_windows_checkouts():
    assert GITATTRIBUTES.is_file(), ".gitattributes must protect POSIX shell entrypoints"
    attributes = GITATTRIBUTES.read_text(encoding="utf-8").splitlines()
    assert "*.sh text eol=lf" in attributes

    scripts = (
        sorted((ROOT / "tools").glob("*.sh"))
        + sorted((ROOT / "mapping").rglob("*.sh"))
        + sorted((ROOT / "ros2_ws" / "src").rglob("*.sh"))
    )
    crlf_paths = [path.relative_to(ROOT).as_posix() for path in scripts if b"\r\n" in path.read_bytes()]
    assert not crlf_paths, f"POSIX shell sources contain CRLF: {crlf_paths}"


def test_preflight_is_explicitly_read_only_and_never_uses_sudo():
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert "Read-only preflight" in text
    assert "\nsudo " not in text
    assert "systemctl" not in text
    assert "ros2 launch" not in text
    assert "go2_person_follow|realsense2_camera" in text
    assert "detect_active_ros_domain" in text
    assert "/proc/$pid/environ" in text


def test_live_preview_producer_reports_its_input_boundary_health():
    producer = (ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "live_keyframe_producer.py").read_text(
        encoding="utf-8"
    )

    assert "PREVIEW_INPUT_HEALTH" in producer
    assert "_input_counts" in producer


def test_preflight_enforces_mapping_storage_headroom_and_reports_failures():
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert "-ge 45" in text
    assert "need at least 45GiB" in text
    assert "exit 2" in text
    assert "SUMMARY" in text


def test_deploy_uses_hashed_atomic_user_space_releases_only():
    text = DEPLOY.read_text(encoding="utf-8")

    assert "Get-FileHash -Algorithm SHA256" in text
    assert "/home/unitree/go2_mapping/releases" in text
    assert "ln -sfn" in text
    assert "/unitree/module" not in text
    assert "sudo" not in text
    assert "install_go2_wifi_reconnect.sh" in text
    assert "install_go2_live_websocket_dependency.sh" in text
    assert "start_go2_nav_bridge.sh" in text
    assert "ProxyCommand=tailscale.exe nc %h %p" in text


def test_deploy_verifies_each_file_and_preflights_before_atomic_activation():
    text = DEPLOY.read_text(encoding="utf-8")

    assert "RELEASE.SHA256SUMS" in text
    assert "sha256sum -c RELEASE.SHA256SUMS" in text
    assert "WriteAllText" in text
    assert '"`n"' in text
    assert "bash -n" in text
    assert "current.next" in text
    assert "mv -Tf" in text
    assert "readlink -f /home/unitree/go2_mapping/current" in text
    assert "/home/unitree/go2_mapping/previous" in text
    verify_index = text.index("sha256sum -c RELEASE.SHA256SUMS")
    activate_index = text.index("mv -Tf")
    assert verify_index < activate_index


def test_deploy_rejects_empty_or_invalid_session_control_before_activation():
    text = DEPLOY.read_text(encoding="utf-8")
    entrypoint = "mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py"

    assert "requiredRobotPythonEntrypoints" in text
    assert ".Length -le 0" in text
    assert f"test -s {entrypoint}" in text
    assert f"python3 -m py_compile {entrypoint}" in text
    assert text.index(f"test -s {entrypoint}") < text.index("mv -Tf")
    assert text.index(f"python3 -m py_compile {entrypoint}") < text.index("mv -Tf")
    for nav_entrypoint in (
        "mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_bridge.py",
        "mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_executor.py",
    ):
        assert f"test -s {nav_entrypoint}" in text
        assert f"python3 -m py_compile {nav_entrypoint}" in text
        assert text.index(f"test -s {nav_entrypoint}") < text.index("mv -Tf")
        assert text.index(f"python3 -m py_compile {nav_entrypoint}") < text.index("mv -Tf")
    deploy_contract = text.replace("\\", "/")
    assert "mapping/robot/go2_mapping_nav_bridge/launch/go2_nav2_bringup.launch.py" in deploy_contract
    assert "mapping/robot/go2_mapping_nav_bridge/config/nav2_params.yaml" in deploy_contract
    assert "python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/launch/go2_nav2_bringup.launch.py" in deploy_contract
    assert "test -s tools/start_go2_nav_bridge.sh" in text
    assert "bash -n tools/start_go2_nav_bridge.sh" in text
    assert "test -s tools/verify_go2_nav2_runtime.sh" in text
    assert "bash -n tools/verify_go2_nav2_runtime.sh" in text
    liveness = "mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
    assert f"test -s {liveness}" in text
    assert f"python3 -m py_compile {liveness}" in text
    assert text.index(f"python3 -m py_compile {liveness}") < text.index("mv -Tf")
    nav2_params = "mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/nav2_params.py"
    assert f"test -s {nav2_params}" in text
    assert f"python3 -m py_compile {nav2_params}" in text
    assert text.index(f"python3 -m py_compile {nav2_params}") < text.index("mv -Tf")


def test_deploy_keeps_nav2_probe_as_a_separate_robot_release_item():
    text = DEPLOY.read_text(encoding="utf-8")

    assert "'diagnose_go2_lidar_network.sh',\n    'verify_go2_nav2_runtime.sh'" in text


def test_prepared_robot_bundle_contains_the_capture_runner():
    text = (ROOT / "tools" / "prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")

    assert "'run_go2_mapping_session.sh'" in text


def test_lio_autostart_unit_is_observation_only_and_uses_local_odom():
    unit = (ROOT / "tools" / "go2-mapping-lio-preview.service").read_text(encoding="utf-8")
    prepare = (ROOT / "tools" / "prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "start_go2_lio_preview.sh" in unit
    assert "ROS_DOMAIN_ID=42" in unit
    assert "GO2_MAPPING_LIO_TARGET_FRAME=odom" in unit
    assert "lowcmd" not in unit.lower()
    assert "sport" not in unit.lower()
    assert "go2-mapping-lio-preview.service" in prepare
    assert "go2-mapping-lio-preview.service" in deploy
    assert "systemctl --user enable --now go2-mapping-lio-preview.service" in deploy
    assert "systemctl --user enable --now go2-mapping-lowstate-imu.service" in deploy
    assert "systemctl --user enable --now go2-mapping-realsense-sensor.service" in deploy
    assert "find '$remoteRelease.incomplete/tools' -type f -name '*.sh' -exec chmod 0755 {} +" in deploy


def test_deploy_validates_navigation_ros_package_metadata_before_activation():
    text = DEPLOY.read_text(encoding="utf-8")
    package_files = (
        "mapping/robot/go2_mapping_nav_bridge/package.xml",
        "mapping/robot/go2_mapping_nav_bridge/setup.py",
        "mapping/robot/go2_mapping_nav_bridge/setup.cfg",
        "mapping/robot/go2_mapping_nav_bridge/resource/go2_mapping_nav_bridge",
        "mapping/robot/go2_mapping_nav_bridge/launch/go2_mapping_nav_bridge.launch.py",
    )

    for package_file in package_files:
        assert f"test -s {package_file}" in text
        assert text.index(f"test -s {package_file}") < text.index("mv -Tf")

    launch_file = package_files[-1]
    assert f"python3 -m py_compile {launch_file}" in text
    assert text.index(f"python3 -m py_compile {launch_file}") < text.index("mv -Tf")


def test_deploy_preflight_bypasses_ambient_http_proxies():
    text = DEPLOY.read_text(encoding="utf-8")

    assert "function Invoke-DirectRestMethod" in text
    assert "$request.Proxy = $null" in text
    assert "$preflight = Invoke-DirectRestMethod" in text


def test_tailnet_business_clients_disable_proxies_on_the_http_request_object():
    clients = (
        ROOT / "tools" / "prepare_go2_mapping_lab.ps1",
        ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1",
    )

    for path in clients:
        text = path.read_text(encoding="utf-8")
        assert "[System.Net.WebRequest]::Create($Uri)" in text, path
        assert "$request.Proxy = $null" in text, path


def test_wsl_installer_requires_administrator_and_never_restarts_automatically():
    text = WSL_INSTALL.read_text(encoding="utf-8")

    assert "#Requires -RunAsAdministrator" in text
    assert "Microsoft-Windows-Subsystem-Linux" in text
    assert "VirtualMachinePlatform" in text
    assert "/NoRestart" in text
    assert "@(0, 1641, 3010)" in text
    assert "Restart-Computer" not in text


def test_mapping_profiles_encode_the_required_safety_and_quality_limits():
    config_dir = ROOT / "mapping" / "config"
    capture = yaml.safe_load((config_dir / "robot_capture.yaml").read_text(encoding="utf-8"))
    uplink = yaml.safe_load((config_dir / "uplink.yaml").read_text(encoding="utf-8"))
    bim = yaml.safe_load((config_dir / "bim_registration.yaml").read_text(encoding="utf-8"))

    assert capture["storage"]["capacity_gib"] == 40
    assert capture["camera_mode"]["follow_active"] == "subscriber_only"
    assert uplink["transport"]["protocol"] == "tailnet_direct_http"
    assert uplink["transport"]["encryption"] == "tailscale_wireguard"
    assert uplink["transport"]["lan_only"] is False
    assert uplink["transport"]["control_plane"] == "tailscale"
    assert uplink["transport"]["receiver"]["allowed_robot"] == "192.0.2.11"
    assert uplink["transport"]["relay"]["preferred_region_id"] == 0
    assert uplink["transport"]["relay"]["selection"] == "direct_first_measured_fallback"
    assert uplink["transport"]["relay"]["official_regions_fallback"] is True
    assert uplink["transport"]["relay"]["stun_data_plane"] is False
    assert uplink["dual_queue"]["preview"]["allocation"] == "realtime_priority_measured_demand"
    assert uplink["dual_queue"]["high_fidelity"]["allocation"] == "remaining_measured_capacity"
    assert bim["coordinate_policy"]["geometry_authority"] == "as_built_map"
    assert bim["acceptance"]["minimum_independent_structural_groups"] == 3


def test_quality_gates_are_versioned_and_session_launcher_passes_only_optional_evidence():
    config_dir = ROOT / "mapping" / "config"
    quality = __import__("json").loads((config_dir / "quality_gates.json").read_text(encoding="utf-8"))
    session = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")

    assert quality["schema_version"] == 1
    assert quality["calibration"]["minimum_source_captures"] == 3
    assert quality["geometry"]["p95_dimension_error_mm"] == 20.0
    assert "[string]$CalibrationManifest" in session
    assert "[string]$SessionEvidence" in session
    assert "CalibrationManifest" in session
    assert "SessionEvidence" in session


def test_neural_environment_does_not_mix_conda_opencv_with_nerfstudio_pip_dependencies():
    environment = yaml.safe_load(PC_ENVIRONMENT.read_text(encoding="utf-8"))
    conda_dependencies = environment["dependencies"]
    pip_dependencies = next(item["pip"] for item in conda_dependencies if isinstance(item, dict))

    assert not any(str(item).startswith("opencv") for item in conda_dependencies)
    assert "nerfstudio==1.1.5" in pip_dependencies


def test_lab_prepare_deploys_then_runs_only_read_only_robot_preflight():
    text = LAB_PREPARE.read_text(encoding="utf-8")

    assert "deploy_go2_mapping_robot.ps1" not in text
    assert "verify_go2_mapping_robot.sh" not in text
    assert "Invoke-DirectRestMethod" in text
    assert "[System.Net.WebRequest]::Create($Uri)" in text
    assert "/v1/preflight" in text
    assert "ros2 bag record" not in text
    assert "ros2 launch" not in text
    assert "go2_person_follow" not in text


def test_full_capture_requires_live_lidar_and_imu_samples_not_only_discovery():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
    liveness = LIVENESS_PROBE.read_text(encoding="utf-8")

    assert "has_live_sample" in capture
    assert 'has_live_sample "$lidar_topic" sensor_msgs/msg/PointCloud2' in capture
    assert 'has_live_sample /dog_imu_raw' in capture
    assert "ros_topic_liveness.py" in capture
    assert 'lidar_topic=${GO2_MAPPING_CAPTURE_LIDAR_TOPIC:-/go2/lio/lidar}' in capture
    assert 'has_live_sample "$lidar_topic" sensor_msgs/msg/PointCloud2' in capture
    assert 'has_advancing_header_sample "$lidar_topic" sensor_msgs/msg/PointCloud2' in capture
    assert "--minimum-header-progress-ms" in liveness
    assert "LIVE_ADVANCING_HEADER" in liveness
    assert "has_live_sample /dog_imu_raw sensor_msgs/msg/Imu" in capture
    assert not re.search(r"^(?!\s*#).*\bros2\s+topic\s+hz\b", capture, re.MULTILINE)
    assert "--kill-after=180s" in capture
    assert "--kill-after=20s" not in capture
    assert 'topics=("${camera_topics[@]}" /tf)' in capture
    assert 'lio_registered_cloud_topic=${GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC:-/lio_sam_ros2/mapping/cloud_registered}' in capture
    assert 'lio_mapping_odometry_topic=${GO2_MAPPING_LIO_MAPPING_ODOMETRY_TOPIC:-/lio_sam_ros2/mapping/odometry}' in capture
    assert 'topics+=("$lio_pose_topic" "$lio_registered_cloud_topic" "$lio_mapping_odometry_topic" "$lio_global_map_topic")' in capture
    assert 'explicit global ``map``' in capture
    assert "optional aligned depth topic is unavailable" in capture
    assert "GO2_MAPPING_PUBLISHER_DISCOVERY_ATTEMPTS:-3" in capture
    assert "attempt < publisher_discovery_attempts" in capture


def test_progressive_full_capture_requires_a_live_global_lio_map_and_records_it_for_replay():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert 'lio_global_map_topic=${GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC:-/go2/mapping/map_cloud}' in capture
    assert 'has_publisher "$lio_global_map_topic"' in capture
    assert 'has_live_sample "$lio_global_map_topic" sensor_msgs/msg/PointCloud2 8 map' in capture
    assert "--require-frame-id" in capture
    assert "header.frame_id=map" in capture
    assert 'topics+=("$lio_pose_topic" "$lio_registered_cloud_topic" "$lio_mapping_odometry_topic" "$lio_global_map_topic")' in capture


def test_full_capture_has_an_explicit_local_odom_validation_branch_without_relaxing_global_map_liveness():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
    robot = (ROOT / "tools" / "start_go2_robot_scan.sh").read_text(encoding="utf-8")
    manifest = (ROOT / "tools" / "write_go2_capture_runtime_manifest.sh").read_text(encoding="utf-8")

    assert "coordinate_scope=${GO2_MAPPING_COORDINATE_SCOPE:-GLOBAL_MAP}" in capture
    assert "SESSION_ODOM_LOCAL" in capture
    assert 'has_live_sample "$lio_registered_cloud_topic" sensor_msgs/msg/PointCloud2 8 odom' in capture
    assert 'has_live_sample "$lio_global_map_topic" sensor_msgs/msg/PointCloud2 8 map' in capture
    assert 'if [[ $coordinate_scope == SESSION_ODOM_LOCAL && $progressive_upload != 1 ]]; then' in capture
    assert 'SESSION_ODOM_LOCAL requires progressive high-fidelity upload' in capture
    assert 'export GO2_MAPPING_COORDINATE_SCOPE="$coordinate_scope"' in robot
    assert 'coordinate_scope = os.environ.get("GO2_MAPPING_COORDINATE_SCOPE", "GLOBAL_MAP")' in manifest


def test_lio_preview_publishes_a_transform_verified_map_frame_cloud_for_full_replay():
    lio_starter = (ROOT / "tools" / "start_go2_lio_preview.sh").read_text(encoding="utf-8")
    publisher = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "map_frame_cloud_publisher.py"
    conversion = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "map_frame_cloud.py"

    assert publisher.is_file()
    assert conversion.is_file()
    assert "GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC" in lio_starter
    assert "GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC" in lio_starter
    assert "GO2_MAPPING_LIO_TARGET_FRAME" in lio_starter
    assert 'target_frame=${GO2_MAPPING_LIO_TARGET_FRAME:-odom}' in lio_starter
    assert '--map-frame "$target_frame"' in lio_starter
    lio_params = (ROOT / "mapping" / "config" / "lio_preview_mid360.yaml").read_text(encoding="utf-8")
    assert 'mapFrame: "odom"' in lio_params
    assert "/go2/mapping/map_cloud" in lio_starter
    assert "map_frame_cloud_publisher.py" in lio_starter
    assert 'header.frame_id = self._map_frame' in publisher.read_text(encoding="utf-8")
    assert "lookup_transform" in publisher.read_text(encoding="utf-8")
    assert "refused local registered cloud without verified map transform" in publisher.read_text(encoding="utf-8")


def test_robot_deployment_validates_the_map_frame_cloud_publisher_before_release_activation():
    deploy = DEPLOY.read_text(encoding="utf-8")
    entrypoint = "mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud_publisher.py"

    assert entrypoint in deploy
    assert f"test -s {entrypoint}" in deploy
    assert f"python3 -m py_compile {entrypoint}" in deploy
    assert deploy.index(f"python3 -m py_compile {entrypoint}") < deploy.index("mv -Tf")


def test_full_capture_has_a_bounded_raw_recording_limit_and_capacity_monitor():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert "GO2_MAPPING_CAPTURE_MAX_GIB:-40" in capture
    assert "GO2_MAPPING_RECOVERY_HEADROOM_GIB:-4" in capture
    assert "monitor_capture_size" in capture
    assert "capture exceeds the configured" in capture


def test_full_capture_writes_a_serial_and_release_bound_runtime_sidecar_before_checksums():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")
    writer_path = ROOT / "tools" / "write_go2_capture_runtime_manifest.sh"
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    watcher = (ROOT / "tools" / "stream_go2_high_fidelity_queue.sh").read_text(encoding="utf-8")

    assert writer_path.is_file()
    writer = writer_path.read_text(encoding="utf-8")
    assert "factory_serial_verified" in writer
    assert "rs-enumerate-devices" in writer
    assert "os.replace" in writer
    assert "write_go2_capture_runtime_manifest.sh" in capture
    assert capture.index("write_go2_capture_runtime_manifest.sh") < capture.index('find "$segment" -type f')
    assert "write_go2_capture_runtime_manifest.sh" in deploy
    assert 'runtime="$segment/capture.runtime.json"' in watcher


def test_full_capture_normalizes_only_a_verified_tailnet_stop_marker_closure():
    capture = (ROOT / "tools" / "start_go2_mapping_capture.sh").read_text(encoding="utf-8")

    assert "intentional_stop_requested=0" in capture
    assert "if [[ -n $stop_file && -f $stop_file && -f $intentional_stop_marker ]]; then" in capture
    assert "intentional_stop_requested=1" in capture
    assert "if [[ $intentional_stop_requested -eq 1 && $status -eq 2 ]]; then" in capture
    assert "status=0" in capture
    assert capture.index("intentional_stop_requested=1") < capture.index(
        "if [[ $intentional_stop_requested -eq 1 && $status -eq 2 ]]; then"
    )


def test_ros_liveness_probe_uses_explicit_reliable_qos_and_requires_a_callback():
    text = LIVENESS_PROBE.read_text(encoding="utf-8")

    assert "get_message" in text
    assert "ReliabilityPolicy.RELIABLE" in text
    assert "create_subscription" in text
    assert "LIVE_SAMPLE" in text
    assert "NO_LIVE_SAMPLE" in text


def test_liveness_probe_can_require_nonempty_image_payloads():
    text = LIVENESS_PROBE.read_text(encoding="utf-8")

    assert "--require-nonempty-data" in text
    assert "data_length < step * height" in text
    assert "LIVE_PAYLOAD" in text


def test_nav2_runtime_probe_requires_nonempty_color_and_depth_payloads():
    text = (ROOT / "tools" / "verify_go2_nav2_runtime.sh").read_text(encoding="utf-8")

    assert "--require-nonempty-data" in text
    assert 'check_liveness sensor_color "$nav2_color_topic" sensor_msgs/msg/Image' in text
    assert 'check_liveness sensor_depth "$nav2_depth_topic" sensor_msgs/msg/Image' in text


def test_mapping_ros_units_share_the_camera_and_lowstate_domain():
    pointcloud = (ROOT / "tools" / "go2-mapping-pointcloud-bridge.service").read_text(encoding="utf-8")
    heightmap = (ROOT / "tools" / "go2-mapping-pgtt-heightmap.service").read_text(encoding="utf-8")
    nav = (ROOT / "tools" / "go2-mapping-nav-bridge.service").read_text(encoding="utf-8")

    assert "Environment=ROS_DOMAIN_ID=42" in pointcloud
    assert "Environment=ROS_DOMAIN_ID=42" in heightmap
    assert "Environment=ROS_DOMAIN_ID=42" in nav
    assert "Environment=RMW_IMPLEMENTATION=rmw_fastrtps_cpp" in nav


def test_native_lowstate_imu_bridge_is_read_only_and_defaults_to_an_isolated_topic():
    root = ROOT / "mapping" / "robot" / "go2_mapping_native_lowstate"
    source = (root / "native_lowstate_imu_socket_server.cpp").read_text(encoding="utf-8")
    start = (ROOT / "tools" / "start_go2_lowstate_imu_bridge.sh").read_text(encoding="utf-8")
    canonical_start = (ROOT / "tools" / "start_go2_unitree_lowstate_imu_bridge.sh").read_text(encoding="utf-8")

    assert 'kNativeLowStateTopic[] = "rt/lowstate"' in source
    assert "ChannelSubscriber" in source
    assert "SportClient" not in source
    assert "ChannelPublisher" not in source
    assert "rclcpp" not in source
    assert "start_go2_unitree_lowstate_imu_bridge.sh" in start
    assert "GO2_MAPPING_LOWSTATE_IMU_TOPIC" in canonical_start


def test_pgtt_terrain_router_service_is_read_only_and_packaged():
    runner = (ROOT / "tools" / "start_go2_pgtt_terrain_router.sh").read_text(encoding="utf-8")
    service = (ROOT / "tools" / "go2-mapping-pgtt-terrain-router.service").read_text(encoding="utf-8")
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    prepare = (ROOT / "tools" / "prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")

    assert "terrain_router_monitor" in runner
    assert "lowcmd" not in runner.lower()
    assert "ChannelPublisher" not in runner
    assert "Restart=on-failure" in service
    assert "go2-mapping-pgtt-terrain-router.service" in deploy
    assert "install_go2_pgtt_terrain_router.sh" in deploy
    assert "go2-mapping-pgtt-terrain-router.service" in prepare


def test_native_pointcloud_bridge_uses_the_observed_unitree_pointcloud_dds_type():
    root = ROOT / "mapping" / "robot" / "go2_mapping_lowstate_imu_bridge"
    source = (root / "src" / "go2_native_pointcloud_bridge.cpp").read_text(encoding="utf-8")
    start = (ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh").read_text(encoding="utf-8")
    native = (
        ROOT / "mapping" / "robot" / "go2_mapping_native_pointcloud" / "native_pointcloud_socket_server.cpp"
    ).read_text(encoding="utf-8")

    assert 'kDefaultNativePointCloudTopic[] = "rt/utlidar/cloud"' in source
    assert 'kDefaultPointCloudTopic[] = "rt/utlidar/cloud"' in native
    assert "sensor_msgs::msg::dds_::PointCloud2_" in source
    assert "ChannelSubscriber" in source
    assert "ChannelPublisher" not in source
    assert '"/go2/lidar/points_raw"' in source
    assert "Dropping malformed native point cloud" in source
    assert "GO2_MAPPING_NATIVE_LIDAR_TOPIC" in start
    assert 'GO2_MAPPING_NATIVE_LIDAR_TOPIC:-rt/utlidar/cloud' in start


def test_bridge_builder_uses_the_isolated_native_reader_and_ros_publisher():
    builder = (ROOT / "tools" / "build_go2_lowstate_imu_bridge.sh").read_text(encoding="utf-8")

    assert "build_go2_native_lowstate_imu_reader.sh" in builder
    assert "native_lowstate_imu_ros.py" in builder
    assert "python3 -m py_compile" in builder
    assert "colcon" not in builder


def test_unitree_cyclonedds_headers_are_preferred_over_ros_fallback_headers():
    cmake = (ROOT / "mapping" / "robot" / "go2_mapping_lowstate_imu_bridge" / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "target_include_directories(go2_native_pointcloud_bridge BEFORE PRIVATE" in cmake
    assert "target_include_directories(go2_lowstate_imu_bridge BEFORE PRIVATE" in cmake
    assert "target_compile_options(go2_native_pointcloud_bridge BEFORE PRIVATE -I/usr/local/include)" in cmake
    assert "target_compile_options(go2_lowstate_imu_bridge BEFORE PRIVATE -I/usr/local/include)" in cmake


def test_lidar_diagnostic_tests_l2_and_a_live_ros_sample_without_mutating_services():
    diagnostic = (ROOT / "tools" / "diagnose_go2_lidar_network.sh").read_text(encoding="utf-8")

    assert "no_arp_neighbor" in diagnostic
    assert "controller_managed_stream" in diagnostic
    assert "ros_topic_liveness.py" in diagnostic
    assert "ros2 topic hz" not in diagnostic
    assert "tcpdump" in diagnostic
    assert "systemctl" not in diagnostic
    assert "ros2 launch" not in diagnostic


def test_native_pointcloud_path_is_split_between_unitree_dds_and_ros_runtime():
    native = ROOT / "mapping" / "robot" / "go2_mapping_native_pointcloud" / "native_pointcloud_socket_server.cpp"
    publisher = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "native_pointcloud_ros.py"

    assert native.is_file(), "native Unitree DDS reader is required"
    assert publisher.is_file(), "ROS-only PointCloud2 publisher is required"
    native_text = native.read_text(encoding="utf-8")
    publisher_text = publisher.read_text(encoding="utf-8")
    assert "ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_>" in native_text
    assert "rclcpp" not in native_text
    assert "rclpy" in publisher_text
    assert "PointCloud2" in publisher_text
    assert "G2PC" in native_text
    assert "native_pointcloud_protocol" in publisher_text


def test_native_pointcloud_starter_launches_ros_receiver_before_sdk_reader():
    starter = (ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh").read_text(encoding="utf-8")

    assert "native_pointcloud_ros.py" in starter
    assert "go2_native_pointcloud_socket_server" in starter
    assert "/livox/lidar" in starter
    assert "livox_custom_ros.py" in starter
    assert "GO2_MAPPING_LIO_LIVOX_TOPIC" in starter
    assert "source /unitree/module/graph_pid_ws/install/setup.bash" in starter
    assert "wait_for_socket" in starter


def test_navigation_bridge_launcher_is_release_owned_and_ros_aware():
    launcher = (ROOT / "tools" / "start_go2_nav_bridge.sh").read_text(encoding="utf-8")

    assert "navigation_executor" in launcher
    assert "GO2_MAPPING_NAVIGATION_ROOT" in launcher
    assert "GO2_MAPPING_NAVIGATION_POLL_SECONDS" in launcher
    assert "GO2_MAPPING_NAVIGATION_TIMEOUT_SECONDS" in launcher
    assert "source /opt/ros/foxy/setup.bash" in launcher
    assert "source /unitree/module/graph_pid_ws/install/setup.bash" in launcher
    assert "python3 -m mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_executor" in launcher


def test_nav2_runtime_probe_is_read_only_and_checks_actions_tf_and_costmaps():
    probe = ROOT / "tools" / "verify_go2_nav2_runtime.sh"
    assert probe.is_file()
    text = probe.read_text(encoding="utf-8")
    assert "Read-only" in text
    assert "ros2 pkg prefix" in text
    assert "nav2_bringup" in text
    assert "nav2_msgs" in text
    assert "navigate_to_pose" in text
    assert "follow_waypoints" in text
    assert "tf" in text.lower()
    assert "costmap" in text.lower()
    assert "cmd_vel" in text
    assert "ros2 topic info" in text
    assert "ros2 lifecycle get" in text
    assert "tf2_echo" in text
    assert "ros_topic_liveness.py" in text
    assert "sensor_msgs/msg/PointCloud2" in text
    assert "sensor_msgs/msg/Imu" in text
    assert "map odom" in text
    assert "odom base_link" in text
    assert "Action servers:" in text
    assert "ros2 topic pub" not in text
    assert "ros2 action send_goal" not in text
    assert "ros2 launch" not in text
    assert "--require-nav2" in text


def test_robot_session_runner_records_then_uploads_only_finalized_artifacts():
    path = ROOT / "tools" / "run_go2_mapping_session.sh"
    assert path.is_file(), "robot session runner is required"
    runner = path.read_text(encoding="utf-8")

    assert "start_go2_mapping_capture.sh" in runner
    assert "start_go2_unitree_lowstate_imu_bridge.sh" in runner
    assert "wait_for_process" in runner
    assert "lowstate-imu-bridge.session.log" in runner
    assert "upload_go2_segment_via_tunnel.sh" in runner
    assert "find \"$segment\" -type f" in runner
    assert "CAPTURE segment=" in runner
    assert "mode must be full or baseline" in runner
    uploader = (ROOT / "tools" / "upload_go2_segment_via_tunnel.sh").read_text(encoding="utf-8")
    assert "GO2_MAPPING_HIGH_FIDELITY_URL" in uploader
    assert "http://192.0.2.10:8000" in uploader
    assert "GO2_MAPPING_TUNNEL_URL" not in uploader
    assert "127.0.0.1:19000" not in uploader
    assert "reverse SSH tunnel" not in uploader


def test_live_preview_is_explicitly_gated_by_lio_pose_and_calibration_identity():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")
    starter = (ROOT / "tools" / "start_go2_live_preview_stream.sh").read_text(encoding="utf-8")
    lio = ROOT / "tools" / "start_go2_lio_preview.sh"
    adapter = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "lio_pose_adapter.py"
    producer = ROOT / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "live_keyframe_producer.py"

    assert "GO2_MAPPING_LIVE_ENABLED" in runner
    assert "wait_for_lio_pose" in runner
    assert "start_go2_lio_preview.sh" in runner
    assert "start_go2_live_preview_stream.sh" in runner
    assert "wait_for_livox_lio_input" in runner
    assert "GO2_MAPPING_CALIBRATION_ID" in starter
    assert "GO2_MAPPING_LIVE_HEALTH_URL" in starter
    assert "GO2_MAPPING_LIVE_LIDAR_MIN_RANGE_M" in starter
    assert "camera assist using unaligned depth fallback" in starter
    assert "GO2_MAPPING_CAMERA_DEPTH_INFO_TOPIC" in starter
    assert "websocket-client" in starter
    assert "pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}" in starter
    assert "endpoint=${GO2_MAPPING_LIVE_ENDPOINT:-ws://${pc_tailnet_ip}:8765/ws/live/keyframe}" in starter
    assert 'export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"' in starter
    assert 'export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"' in starter
    assert "ros_topic_liveness.py" in starter
    assert "geometry_msgs/msg/PoseWithCovarianceStamped" in starter
    assert "timeout 5s ros2 topic info" not in starter
    assert lio.is_file()
    assert "lio_sam_ros2_mapOptmization" in lio.read_text(encoding="utf-8")
    assert "timeout 5s ros2 topic info" in lio.read_text(encoding="utf-8")
    assert "kill -TERM" in lio.read_text(encoding="utf-8")
    assert 'ros2 run lio_sam_ros2' not in lio.read_text(encoding="utf-8")
    assert "PREVIEW_UNCALIBRATED" in adapter.read_text(encoding="utf-8")
    assert "--preview-max-hz" not in producer.read_text(encoding="utf-8")
    assert "--preview-max-hz" not in starter
    assert "GO2_MAPPING_PREVIEW_MAX_HZ" not in starter
    assert '"GO2_MAPPING_LIVE_ENDPOINT"' in producer.read_text(encoding="utf-8")
    assert "ws://192.0.2.10:8765/ws/live/keyframe" in producer.read_text(encoding="utf-8")
    assert "127.0.0.1:19001" not in producer.read_text(encoding="utf-8")
    assert "adaptive_detail_hz(" in producer.read_text(encoding="utf-8")
    assert "1.0 / detail_hz" in producer.read_text(encoding="utf-8")
    assert "--detail-minimum-hz" in producer.read_text(encoding="utf-8")
    assert "--delivery-window" in producer.read_text(encoding="utf-8")
    assert "_delivery_loop" in producer.read_text(encoding="utf-8")
    assert "_enqueue_latest" in producer.read_text(encoding="utf-8")
    assert "_preview_range_filter" in producer.read_text(encoding="utf-8")
    assert "GO2_MAPPING_LIVE_DELIVERY_WINDOW:-20" in starter
    assert "GO2_MAPPING_LIVE_PREVIEW_QUEUE_SIZE:-64" in starter
    assert 'GO2_MAPPING_LIVE_LIDAR_TOPIC:-/lio_sam_ros2/deskew/cloud_deskewed' in starter
    assert 'GO2_MAPPING_LIVE_POINTS_IN_MAP_FRAME:-0' in starter
    assert 'no live deskewed LiDAR cloud' in starter
    assert "--pose-cloud-max-skew-ms" in starter
    assert "sensor_msgs/msg/PointCloud2" in starter
    assert "range_points = xyz - pose[:3, 3] if self.args.points_in_map_frame else xyz" in producer.read_text(encoding="utf-8")
    assert "args.delivery_window <= 32" in producer.read_text(encoding="utf-8")
    assert "args.preview_queue_size <= 128" in producer.read_text(encoding="utf-8")


def test_pc_session_uses_the_dashboard_tailnet_endpoint_for_live_keyframes():
    session = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")

    assert "127.0.0.1:19001:127.0.0.1:8765" not in session
    assert "go2-preview-tunnel" not in session
    assert "ExitOnForwardFailure=yes" not in session
    assert "start_go2_pc_stack.ps1" in session
    assert "TAILSCALE_DIRECT" in session


def test_robot_session_runner_keeps_pc_tailnet_requests_outside_ambient_proxies():
    runner = (ROOT / "tools" / "run_go2_mapping_session.sh").read_text(encoding="utf-8")

    assert "pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}" in runner
    assert 'export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"' in runner
    assert 'export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"' in runner


def test_dashboard_uses_a_caller_owned_wsl_process_instead_of_a_vanishing_nohup_child():
    starter = (ROOT / "mapping" / "pc" / "start_dashboard_wsl.sh").read_text(encoding="utf-8")

    assert "exec \"${MAMBA_ROOT_PREFIX}/bin/micromamba\"" in starter
    assert "nohup" not in starter
    assert "GO2_MAPPING_DASHBOARD_PID_FILE" in starter
    assert "stop-owned" in starter
    assert "verify-owned" in starter
    assert "/proc/${owned_pid}/cwd" in starter
    assert "kill -TERM -- \"-${owned_pid}\"" in starter
    assert "dashboard_port=${3:-8767}" in starter
    assert "--port \"${dashboard_port}\"" in starter
    assert "pkill" not in starter
    assert "--host 0.0.0.0" in starter


def test_wired_development_access_is_explicitly_temporary_and_does_not_reconfigure_runtime_transport():
    text = WIRED_DEV_ACCESS.read_text(encoding="utf-8")

    assert "#Requires -RunAsAdministrator" in text
    assert "PolicyStore ActiveStore" in text
    assert "DefaultGateway" not in text
    assert "Tailscale routes" in text
    assert "Remove-NetIPAddress" in text


def test_wifi_reconnect_installer_rescans_saved_wifi_profiles_every_twenty_seconds():
    text = WIFI_RECONNECT.read_text(encoding="utf-8")

    assert "connection.autoconnect-retries 0" in text
    assert "802-11-wireless.powersave 2" in text
    assert "OnUnitActiveSec=20s" in text
    assert "interval_s=20" in text
    assert "nmcli connection up id" in text
    assert 'nmcli device wifi rescan ifname "$INTERFACE"' in text
    assert "nmcli device connect" in text
    assert "systemctl restart NetworkManager" not in text
    assert "go2_person_follow" not in text
    assert "realsense" not in text.lower()


def test_large_segment_upload_disables_http_expect_continue_for_the_minimal_receiver():
    uploader = (ROOT / "tools" / "upload_go2_segment_via_tunnel.sh").read_text(encoding="utf-8")

    assert "-H 'Expect:'" in uploader


def test_pc_session_entrypoint_owns_tailnet_receiver_direct_preview_and_remote_run():
    path = ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1"
    assert path.is_file(), "PC session entrypoint is required"
    session = path.read_text(encoding="utf-8")

    assert "Invoke-DirectRestMethod" in session
    assert "[System.Net.WebRequest]::Create($Uri)" in session
    assert "$request.Proxy = $null" in session
    assert "/v1/sessions" in session
    assert "TAILSCALE_DIRECT" in session
    assert "ssh.exe" not in session
    assert "run_go2_mapping_session.sh" not in session


def test_pc_session_exposes_the_bounded_10_to_40_gib_raw_capture_limit():
    session = (ROOT / "tools" / "start_go2_mapping_session_tailnet.ps1").read_text(encoding="utf-8")

    assert "[ValidateRange(1, 86400)]" in session
    assert "DurationSeconds" in session
    assert "/v1/sessions" in session


def test_task9_deployment_installs_control_service_only_in_maintenance_phase():
    deploy = DEPLOY.read_text(encoding="utf-8")
    prepare = LAB_PREPARE.read_text(encoding="utf-8")

    assert "install_go2_tailnet_session_control.sh" in deploy
    assert "go2-mapping-session-control.service" in deploy
    assert "SessionControlTokenFile" in deploy
    assert "AllowedPcTailnetIp" in deploy
    assert "RobotTailnetIp" in deploy
    assert "RobotFactorySerial" in deploy
    assert "GO2_MAPPING_ROBOT_SERIAL" in (ROOT / "tools" / "install_go2_tailnet_session_control.sh").read_text(encoding="utf-8")
    assert "RandomNumberGenerator]::Create" in deploy
    assert "GetBytes($tokenBytes)" in deploy
    assert "scp" in deploy.lower()
    assert "ssh" in deploy.lower()

    assert "deploy_go2_mapping_robot.ps1" not in prepare
    assert "verify_go2_mapping_robot.sh" not in prepare
    assert "Invoke-DirectRestMethod" in prepare
    assert "[System.Net.WebRequest]::Create($Uri)" in prepare
    assert "/v1/preflight" in prepare
    assert "CalibrationManifest" in prepare
    assert "SessionEvidence" in prepare
    assert "DepthAnythingWeights" in prepare
    assert "DepthAnythingWeightsSha256File" in prepare
    assert "ssh.exe" not in prepare.lower()
    assert "& ssh " not in prepare.lower()
    assert "scp " not in prepare.lower()
    assert "rsync " not in prepare.lower()


def test_navigation_bridge_deployment_installs_the_persistent_user_service():
    deploy = DEPLOY.read_text(encoding="utf-8")
    nav_service = ROOT / "tools" / "go2-mapping-nav-bridge.service"
    nav_installer = ROOT / "tools" / "install_go2_nav_bridge.sh"

    assert "start_go2_nav_bridge.sh" in deploy
    assert "go2-mapping-nav-bridge.service" in deploy
    assert "install_go2_nav_bridge.sh" in deploy
    assert nav_service.is_file()
    assert nav_installer.is_file()
    service_text = nav_service.read_text(encoding="utf-8")
    installer_text = nav_installer.read_text(encoding="utf-8")
    assert "ExecStart=/bin/bash %h/go2_mapping/current/tools/start_go2_nav_bridge.sh" in service_text
    assert "GO2_MAPPING_NAVIGATION_ROOT" in service_text
    assert "go2-mapping-nav-bridge.service" in installer_text
    assert "systemctl --user enable --now go2-mapping-nav-bridge.service" in installer_text


def test_navigation_bridge_installer_builds_a_user_ros_overlay_before_starting_services():
    installer = (ROOT / "tools" / "install_go2_nav_bridge.sh").read_text(encoding="utf-8")

    assert "colcon build" in installer
    assert "--merge-install" in installer
    assert "GO2_MAPPING_NAV2_WS" in installer
    assert "go2_mapping_nav_bridge" in installer
    assert installer.index("colcon build") < installer.index("systemctl --user enable --now")


def test_robot_deployment_installs_the_navigation_overlay_after_atomic_release_activation():
    deploy = DEPLOY.read_text(encoding="utf-8")
    activation = "mv -Tf /home/unitree/go2_mapping/current.next /home/unitree/go2_mapping/current"
    install = "bash /home/unitree/go2_mapping/current/tools/install_go2_nav_bridge.sh"

    assert install in deploy
    assert deploy.index(activation) < deploy.index(install)


def test_model_manifest_uses_the_provider_tree_digest_and_verifies_local_files(tmp_path):
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"depth_anything"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "preprocessor_config.json").write_text('{}\n', encoding="utf-8")

    manifest = build_model_manifest(model, model_id="test-depth", revision="rev-1")

    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "test-depth"
    assert manifest["source_revision"] == "rev-1"
    assert manifest["file_count"] == 3
    assert verify_model_manifest(model, manifest) == manifest["directory_sha256"]

    (model / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_model_manifest(model, manifest)


def test_checked_in_asset_manifests_describe_the_ready_pc_and_robot_paths():
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    robot_manifest = json.loads(ROBOT_MANIFEST.read_text(encoding="utf-8"))

    assert model_manifest["model_id"] == "Depth-Anything-V2-Small-hf"
    assert model_manifest["relative_path"] == "models/depth-anything-v2-small-hf-5426e4f"
    assert model_manifest["directory_sha256"] == "37a91a613d005c92f71e6b0caf2790c112ab06a7794fd2fa275521be14510137"
    assert model_manifest["file_count"] == 3
    assert {item["path"] for item in model_manifest["files"]} == {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    assert robot_manifest["schema_version"] == 1
    assert "go2_mapping_nav_bridge" in robot_manifest["ros_packages"]
    assert "livox_ros_driver2" in robot_manifest["ros_packages"]
    assert "yaml" in robot_manifest["python_modules"]
    assert "mapping/robot/environment.yml" in robot_manifest["release_files"]
    assert any("wheelhouse-aarch64-cp38/PyYAML-6.0.2" in path for path in robot_manifest["release_files"])
    assert "mapping/config/pgtt_go2_policy_catalog.json" in robot_manifest["release_files"]
    assert {
        "mapping/robot/go2_mapping_native_lowstate/native_lowstate_imu_socket_server.cpp",
        "mapping/robot/go2_mapping_lio/go2_mapping_lio/native_lowstate_imu_protocol.py",
        "mapping/robot/go2_mapping_lio/go2_mapping_lio/native_lowstate_imu_ros.py",
        "mapping/robot/go2_mapping_lowstate_imu_bridge/package.xml",
        "tools/build_go2_lowstate_imu_bridge.sh",
        "tools/start_go2_unitree_lowstate_imu_bridge.sh",
    }.issubset(set(robot_manifest["release_files"]))
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert "build_go2_lowstate_imu_bridge.sh" in deploy
    assert "go2-mapping-lowstate-imu.service" in deploy
    lowstate_build = (ROOT / "tools" / "build_go2_lowstate_imu_bridge.sh").read_text(encoding="utf-8")
    assert "build_go2_native_lowstate_imu_reader.sh" in lowstate_build
    assert "colcon build" not in lowstate_build
    assert {
        f"mapping/robot/pgtt-go2/policy_level{level}_run0.npz"
        for level in ("03", "07", "10", "13", "17", "20")
    }.issubset(set(robot_manifest["release_files"]))


def test_release_contains_asset_verifier_and_offline_preparation_contract():
    prepare = ASSET_PREPARE.read_text(encoding="utf-8")
    verifier = RELEASE_VERIFY.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")

    assert "pc_model_manifest.json" in prepare
    assert "Get-FileHash" in prepare
    assert "GO2_MAPPING_ASSET_MANIFEST" in verifier
    assert "python3 -m" not in verifier
    assert "verify_go2_mapping_release.sh" in deploy
    assert "robot_runtime_manifest.json" in deploy


def test_pc_stack_defaults_to_the_verified_local_model_manifest():
    stack = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")

    assert "depth-anything-v2-small-hf-5426e4f" in stack
    assert "37a91a613d005c92f71e6b0caf2790c112ab06a7794fd2fa275521be14510137" in stack
    assert "pc_model_manifest.json" in stack
    assert "local_files_only" in stack or "prepare_go2_mapping_assets.ps1" in stack


def test_first_robot_release_allows_its_overlay_before_build_then_rechecks_after_install():
    verifier = RELEASE_VERIFY.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")

    assert "GO2_MAPPING_SKIP_LOCAL_BRIDGE" in verifier
    assert "GO2_MAPPING_SKIP_LOCAL_BRIDGE=1" in deploy
    assert "GO2_MAPPING_SKIP_LOCAL_BRIDGE=1" in deploy
    assert deploy.index("GO2_MAPPING_SKIP_LOCAL_BRIDGE=1") < deploy.index("mv '$remoteRelease.incomplete'")
    assert deploy.index("bash /home/unitree/go2_mapping/current/tools/install_go2_nav_bridge.sh") < deploy.rindex(
        "bash tools/verify_go2_mapping_release.sh"
    )


def test_offline_robot_bundle_builder_contains_only_robot_runtime_and_checksum_contract():
    bundle = RELEASE_BUNDLE.read_text(encoding="utf-8")

    assert "robot_runtime_manifest.json" in bundle
    assert "RELEASE.SHA256SUMS" in bundle
    assert "mapping\\pc" not in bundle
    assert "models" not in bundle
    assert "tar -czf" in bundle
    assert "GO2_MAPPING_ROBOT_BUNDLE_READY" in bundle


def test_robot_wheelhouse_is_pinned_for_go2_python38_aarch64_and_installed_offline():
    manifest = json.loads(ROBOT_WHEELHOUSE_MANIFEST.read_text(encoding="utf-8"))
    installer = ROBOT_RUNTIME_INSTALL.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")
    bundle = RELEASE_BUNDLE.read_text(encoding="utf-8")

    assert manifest["schema_version"] == 1
    assert manifest["target"] == "cp38-manylinux2014_aarch64"
    assert {item["distribution"] for item in manifest["wheels"]} == {
        "PyYAML",
        "zstandard",
        "mcap",
        "psutil",
        "websocket-client",
    }
    assert "--no-index" in installer
    assert "python3 -m pip install" in installer
    assert "sha256" in installer
    assert "robot_wheelhouse_manifest.json" in bundle
    assert "wheelhouse-aarch64-cp38" in deploy
    assert "install_go2_mapping_python_runtime.sh" in deploy
    assert "wheelhouse-aarch64-cp38" in bundle


def test_latest_deployer_selects_a_local_hashed_bundle_and_delegates_to_atomic_deployer():
    wrapper = DEPLOY_LATEST.read_text(encoding="utf-8")

    assert "go2-mapping-robot-*.tar.gz" in wrapper
    assert "Sort-Object LastWriteTime -Descending" in wrapper
    assert "Get-FileHash -Algorithm SHA256" in wrapper
    assert "deploy_go2_mapping_robot.ps1" in wrapper
    assert "-PreparedArchive" in wrapper
    assert "-UseTailscaleProxy" in wrapper


def test_router_install_restarts_an_already_running_process_after_release_update():
    installer = ROUTER_INSTALL.read_text(encoding="utf-8")

    assert "systemctl --user restart go2-mapping-pgtt-terrain-router.service" in installer


def test_deployment_can_reuse_a_prepared_robot_bundle_without_deleting_it():
    deploy = DEPLOY.read_text(encoding="utf-8")

    assert "[string]$PreparedArchive" in deploy
    assert "Resolve-Path -LiteralPath $PreparedArchive" in deploy
    assert "usePreparedArchive" in deploy
    assert "cleanupArchive" in deploy
    assert "if (-not $usePreparedArchive)" in deploy
    assert "if ($cleanupArchive -and $archiveFile)" in deploy


def test_motion_mode_helper_build_skips_when_robot_sdk_is_not_compatible():
    helper_build = (ROOT / "tools" / "build_go2_motion_mode_helper.sh").read_text(encoding="utf-8")

    assert "motion_switcher_client.hpp" in helper_build
    assert "GO2_MOTION_MODE_HELPER_SKIPPED" in helper_build
    assert "nm -C" in helper_build
    assert "exit 0" in helper_build
    assert "grep -q 'MotionSwitcherClient::'" not in helper_build


def test_terrain_control_facade_is_packed_into_existing_authenticated_service():
    service = (ROOT / "mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py").read_text(encoding="utf-8")
    facade = (ROOT / "mapping/robot/go2_mapping_locomotion/terrain_control_service.py").read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")

    assert "TerrainControlSupervisor" in service
    assert "/v1/terrain/status" in service
    assert "/v1/terrain/arm" in service
    assert "/v1/terrain/stop" in service
    assert "terrain-control-audit.jsonl" in facade
    assert "foreach ($relativePath in @('robot', 'config'))" in deploy
    assert "query_go2_motion_mode.py" in deploy
    unit = (ROOT / "tools/go2-mapping-session-control.service").read_text(encoding="utf-8")
    assert "Environment=PYTHONPATH=%h/go2_mapping/current" in unit
