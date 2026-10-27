from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vendor_probe_uses_installed_go2_sport_sdk_and_only_read_api():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_sport_probe.cpp").read_text(encoding="utf-8")
    build = (ROOT / "tools/build_go2_vendor_sport_probe.sh").read_text(encoding="utf-8")

    assert "unitree/robot/go2/sport/sport_client.hpp" in source
    assert "unitree::robot::SportClient" in source
    assert "GetSpeedLevel" in source
    assert "SwitchGait" not in source
    assert "Move(" not in source
    assert "StandUp" not in source
    assert "-lunitree_go2_sdk" in build
    assert "GO2_VENDOR_SPORT_PROBE_READY" in build


def test_vendor_probe_is_packaged_and_not_a_boot_service():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    query = (ROOT / "tools/query_go2_vendor_sport.py").read_text(encoding="utf-8")

    assert "build_go2_vendor_sport_probe.sh" in prepare
    assert "build_go2_vendor_sport_probe.sh" in deploy
    assert "query_go2_vendor_sport.py" in prepare
    assert "query_go2_vendor_sport.py" in deploy
    assert "SwitchGait" not in query
    assert "systemctl" not in query
