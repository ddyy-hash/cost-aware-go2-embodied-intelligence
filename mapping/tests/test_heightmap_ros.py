from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_heightmap_ros_entrypoint_is_opt_in_and_non_actuating():
    script = (ROOT / "tools" / "start_go2_pgtt_heightmap.sh").read_text(encoding="utf-8")
    node = (ROOT / "mapping" / "robot" / "go2_mapping_locomotion" / "heightmap_ros.py").read_text(encoding="utf-8")
    assert "/livox/lidar" in script
    assert "/rt/elevation_heightmap" in script
    assert "lowcmd" not in script.lower()
    assert "cmd_vel" not in script
    assert "create_publisher(PointCloud2" in node
    assert "rclpy.spin" in node
    assert "ReliabilityPolicy.RELIABLE" in node
    assert "DurabilityPolicy.VOLATILE" in node
    assert "import numpy as np" in node
    assert "GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID" in script
    assert "transform_lidar_points" in node
    assert "build_stair_probe_profile" in node
    assert "GO2_MAPPING_STAIR_PROBE_FILE" in script
    assert "--output-file" in script and "--stair-probe-file" in script
    assert "source_frame_id=message.header.frame_id" in node
