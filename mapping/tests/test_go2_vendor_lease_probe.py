from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vendor_lease_probe_only_reads_lease_and_speed():
    source = (ROOT / "mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_lease_probe.cpp").read_text(encoding="utf-8")
    build = (ROOT / "tools/build_go2_vendor_lease_probe.sh").read_text(encoding="utf-8")

    assert "SportClient client(true)" in source
    assert "WaitLeaseApplied" in source
    assert "GetSpeedLevel" in source
    assert "Move(" not in source
    assert "SwitchGait" not in source
    assert "StandUp" not in source
    assert "-lunitree_go2_sdk" in build


def test_vendor_lease_probe_is_packaged_without_autostart_unit():
    prepare = (ROOT / "tools/prepare_go2_mapping_robot_bundle.ps1").read_text(encoding="utf-8")
    deploy = (ROOT / "tools/deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "build_go2_vendor_lease_probe.sh" in prepare
    assert "build_go2_vendor_lease_probe.sh" in deploy
    assert "go2-mapping-vendor-lease" not in prepare
    assert "go2-mapping-vendor-lease" not in deploy
