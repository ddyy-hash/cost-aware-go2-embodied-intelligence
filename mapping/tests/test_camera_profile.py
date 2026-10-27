import json
from pathlib import Path

import pytest


PROFILE = Path(__file__).parents[1] / "config" / "go2edu_realsense_d435i_camera_profile.json"


def test_robot_camera_profile_records_the_observed_runtime_calibration():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))

    assert profile["device"] == {
        "model": "Intel RealSense D435I",
        "serial": "REALSENSE-DEVICE-REDACTED",
        "firmware": "5.15.1.55",
    }
    assert profile["color_camera_info"]["width"] == 424
    assert profile["color_camera_info"]["height"] == 240
    assert profile["color_camera_info"]["k"][:3] == pytest.approx([301.9424743652344, 0.0, 218.02423095703125])
    assert profile["color_camera_info"]["k"][4:6] == pytest.approx([301.6651916503906, 126.14513397216797])
    assert profile["depth_camera_info"]["k"] == pytest.approx(
        [213.42172241210938, 0.0, 210.29110717773438, 0.0, 213.42172241210938, 119.36937713623047, 0.0, 0.0, 1.0]
    )
    assert profile["runtime"]["depth_to_color_extrinsics"]["required_before_rgbd_lidar_fusion"] is True
