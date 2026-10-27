from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_native_executor_uses_vendor_lease_and_high_level_sport_only():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")
    build = (ROOT / "tools/build_go2_native_terrain_executor.sh").read_text(encoding="utf-8")

    assert "SportClient client(true)" in source
    assert "WaitLeaseApplied" in source
    assert "GetSpeedLevel" in source
    assert "SwitchGait" in source
    assert "StopMove" in source
    assert "Move" in source
    assert "rt/lowcmd" not in source
    assert "ChannelPublisher" not in source
    assert "GetSpeedLevel" in source
    assert "GO2_PGTT_FIELD_TEST_CONFIRMED" in source
    assert "GO2_NATIVE_TERRAIN_EXECUTOR_BLOCKED" in source
    assert "motor.lost()" in source
    assert "FAULT_LOCKOUT" in source
    assert "std::memcpy" in source
    assert "GO2_NATIVE_TERRAIN_EXECUTOR_READY" in build


def test_native_executor_is_not_a_boot_unit():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    assert "build_go2_native_terrain_executor.sh" in prepare
    assert "build_go2_native_terrain_executor.sh" in deploy
    assert "go2-mapping-native-terrain-executor.service" not in prepare
    assert "go2-mapping-native-terrain-executor.service" not in deploy


def test_native_executor_releases_its_sport_lease_when_route_is_lost():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    # A retained lease would leave Unitree native walking unavailable after a
    # normal deadman release. The executor must publish its transition and
    # leave the process; this old SDK's destructor is ABI-unsafe, so the
    # vendor's short lease TTL is allowed to expire after StopMove.
    assert 'active_state = had_motion ? "RETURN_NATIVE" : "NATIVE_READY";' in source
    assert 'active_state = "FAULT_LOCKOUT"' in source
    assert 'stop_requested.store(true)' in source
    assert 'std::_Exit(0);' in source
    assert 'lease TTL' in source
    assert 'DDS discovery can take a few cycles' in source
    assert 'for (int i = 0; i < 50; ++i)' in source
    assert '--request-path' in source
    assert 'STAIR_ASCEND' in source
    assert 'bool had_motion = false;' in source
    assert 'std::string blocker;' in source
    assert '"MOTOR_HEALTH"' in source
    assert '"REMOTE_A_SOFTWARE_STOP"' in source
    assert '\\"blocker\\"' in source


def test_native_executor_uses_official_go2_gaits_for_stairs_and_rough_terrain():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    assert 'if (route.state == "STAIR_UP") return 3;' in source
    assert 'if (route.state == "STAIR_DOWN") return 4;' in source
    assert 'if (route.state == "ROUGH_TERRAIN") return 1;' in source
    assert 'const int gait = gait_for_route(route);' in source


def test_native_executor_accepts_successful_speed_read_when_legacy_firmware_leaves_level_unset():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    assert 'client.GetSpeedLevel(level) == 0 ? 0 : 1' in source
    assert '&& level >= 0' not in source


def test_native_executor_settles_after_gait_switch_before_first_move():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    assert "gait_ready_at" in source
    assert "GAIT_SWITCH_SETTLING" in source
    assert "std::chrono::milliseconds(100)" in source
    assert "std::chrono::steady_clock::now() >= gait_ready_at" in source


def test_native_executor_requires_fresh_heightmap_before_any_motion():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    assert "heightmap_age_ms" in source
    assert "kMaxHeightmapAgeMs" in source
    assert "route.heightmap_age_ms <= kMaxHeightmapAgeMs" in source


def test_native_executor_does_not_use_l1_as_motion_gate():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp").read_text(encoding="utf-8")

    assert "REMOTE_DEADMAN_RELEASED" not in source
    assert "const bool deadman" not in source
    assert "else if (!deadman)" not in source
    assert "const float ly = remote_float(state.wireless_remote(), 20);" in source


def test_terrain_router_monitor_uses_remote_direction_without_l1_gate():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/terrain_router_monitor.py").read_text(encoding="utf-8")

    assert "intent = float(args.simulate_forward) if args.simulate_forward is not None else remote.ly" in source
    assert "remote.ly if remote.deadman else 0.0" not in source


def test_native_terrain_sources_and_build_tools_are_required_by_release_verifier():
    manifest = (ROOT / "mapping/config/robot_runtime_manifest.json").read_text(encoding="utf-8")
    verifier = (ROOT / "tools/verify_go2_mapping_release.sh").read_text(encoding="utf-8")

    for path in (
        "mapping/robot/go2_mapping_locomotion/terrain_control_service.py",
        "mapping/robot/go2_mapping_locomotion/vendor_sport_adapter.py",
        "mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp",
        "tools/build_go2_native_terrain_executor.sh",
    ):
        assert path in manifest
        assert path.rsplit("/", 1)[-1] in verifier
