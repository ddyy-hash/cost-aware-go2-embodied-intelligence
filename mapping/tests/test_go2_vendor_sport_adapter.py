from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vendor_adapter_binds_real_go2_sport_api_and_guards_mutations():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_sport_adapter.cpp").read_text(encoding="utf-8")
    build = (ROOT / "tools/build_go2_vendor_sport_adapter.sh").read_text(encoding="utf-8")

    assert "unitree/robot/go2/sport/sport_client.hpp" in source
    assert "unitree::robot::SportClient" in source
    assert "GetSpeedLevel" in source
    assert "SwitchGait" in source
    assert "StopMove" in source
    assert "Move" in source
    assert "I_UNDERSTAND_GO2_HIGH_LEVEL_MOTION" in source
    assert "-lunitree_go2_sdk" in build
    assert "GO2_VENDOR_SPORT_ADAPTER_READY" in build


def test_vendor_adapter_is_packaged_without_boot_autostart():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "build_go2_vendor_sport_adapter.sh" in prepare
    assert "build_go2_vendor_sport_adapter.sh" in deploy
    assert "go2_vendor_sport_adapter" in deploy
