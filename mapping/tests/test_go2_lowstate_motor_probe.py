from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_lowstate_motor_probe_is_read_only_and_uses_pgtt_order():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_lowstate_motor_probe.cpp").read_text(encoding="utf-8")
    build = (ROOT / "tools/build_go2_lowstate_motor_probe.sh").read_text(encoding="utf-8")

    assert "rt/lowstate" in source
    assert "ChannelSubscriber" in source
    assert "motor_state" in source
    assert "PGTT_TO_UNITREE" in source
    assert "ChannelPublisher" not in source
    assert "rt/lowcmd" not in source
    assert "-lunitree_go2_sdk" in build


def test_lowstate_motor_probe_is_packaged_as_a_diagnostic_only():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "build_go2_lowstate_motor_probe.sh" in prepare
    assert "build_go2_lowstate_motor_probe.sh" in deploy
