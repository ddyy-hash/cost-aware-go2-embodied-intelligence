from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sensor_owner_is_separate_from_tracking_and_motion():
    script = (ROOT / "tools" / "start_go2_realsense_sensor.sh").read_text(encoding="utf-8")
    service = (ROOT / "tools" / "go2-mapping-realsense-sensor.service").read_text(encoding="utf-8")
    installer = (ROOT / "tools" / "install_go2_realsense_sensor.sh").read_text(encoding="utf-8")
    assert "navigation" in script.lower()
    assert "cmd_vel" not in script
    assert "go2_mapping_nav_bridge" not in script
    assert "realsense2_camera" in script
    assert "rgb_camera.profile:=\"${GO2_MAPPING_CAMERA_COLOR_PROFILE:-424x240x15}\"" in script
    assert "depth_module.profile:=\"${GO2_MAPPING_CAMERA_DEPTH_PROFILE:-424x240x15}\"" in script
    assert "WantedBy=default.target" in service
    assert "start_go2_realsense_sensor.sh" in service
    assert "disable --now" in installer
