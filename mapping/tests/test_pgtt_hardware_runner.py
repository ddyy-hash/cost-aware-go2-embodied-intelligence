from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hardware_runner_is_explicit_and_not_a_boot_service():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/hardware_runner.py").read_text(encoding="utf-8")
    assert "--arm-hardware" in source
    assert "--confirmation-file" in source
    assert "DIRECTION_ONLY" in source
    assert "REMOTE_DEADMAN_RELEASED" not in source
    assert "HEIGHTMAP_STALE" in source
    assert "POSTURE_GATE" in source
    assert "go2_motion_mode_helper" in source
    assert '_mode(args.motion_helper, "release")' in source
    assert '_mode(args.motion_helper, "select")' in source
    assert "ChannelPublisher" in source
    assert not list((ROOT / "tools").glob("*pgtt*hardware*.service"))


def test_hardware_runner_is_packaged_but_not_started_by_deploy():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "tools/run_go2_pgtt_hardware.sh").read_text(encoding="utf-8")
    assert "run_go2_pgtt_hardware.sh" in prepare
    assert "run_go2_pgtt_hardware.sh" in deploy
    assert "systemctl" not in launcher
    assert "--arm-hardware" in launcher
    assert "--confirmation-file" in launcher
