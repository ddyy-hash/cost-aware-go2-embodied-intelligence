import sys
import struct
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROBOT_PACKAGE = ROOT / "mapping" / "robot" / "go2_mapping_lio"
sys.path.insert(0, str(ROBOT_PACKAGE))


def test_native_pointcloud_protocol_module_declares_an_explicit_wire_version():
    module_path = ROBOT_PACKAGE / "go2_mapping_lio" / "native_pointcloud_protocol.py"

    assert module_path.is_file()


def test_decoder_preserves_pointcloud_metadata_and_bytes():
    from go2_mapping_lio.native_pointcloud_protocol import decode_frame_body

    field_name = b"x"
    fixed = struct.pack(
        "<4sHHiIIIIIHBH",
        b"G2PC",
        1,
        0,
        42,
        123,
        1,
        2,
        16,
        32,
        1,
        0b10,
        len(b"lidar"),
    )
    field = struct.pack("<H", len(field_name)) + field_name + struct.pack("<IBI", 0, 7, 1)
    body = fixed + b"lidar" + field + bytes(range(32))

    frame = decode_frame_body(body)

    assert frame.stamp_sec == 42
    assert frame.stamp_nanosec == 123
    assert frame.height == 1
    assert frame.width == 2
    assert frame.frame_id == "lidar"
    assert frame.is_dense is True
    assert frame.fields[0].name == "x"
    assert frame.fields[0].datatype == 7
    assert frame.data == bytes(range(32))


def test_decoder_rejects_a_frame_with_mismatched_point_data_length():
    from go2_mapping_lio.native_pointcloud_protocol import ProtocolError, decode_frame_body

    fixed = struct.pack("<4sHHiIIIIIHBH", b"G2PC", 1, 0, 0, 0, 1, 1, 16, 16, 0, 0, 0)
    with pytest.raises(ProtocolError, match="data length"):
        decode_frame_body(fixed + b"too-short")


def test_decoder_rejects_a_row_that_cannot_hold_its_declared_points():
    from go2_mapping_lio.native_pointcloud_protocol import ProtocolError, decode_frame_body

    fixed = struct.pack("<4sHHiIIIIIHBH", b"G2PC", 1, 0, 0, 0, 1, 2, 16, 16, 0, 0, 0)
    with pytest.raises(ProtocolError, match="row_step"):
        decode_frame_body(fixed + bytes(16))


def test_lio_clock_aligner_preserves_source_cadence_and_rejects_arrival_jitter():
    from go2_mapping_lio.lio_clock_alignment import LioCloudClockAligner

    aligner = LioCloudClockAligner(guard_ms=160.0, max_offset_step_ms=250.0, window_size=5)
    source_start_ns = 1_000_000_000_000
    host_offset_ns = 8_000_000_000_000

    assert aligner.align(source_start_ns, source_start_ns + host_offset_ns) == (
        source_start_ns + host_offset_ns - 160_000_000
    )
    assert aligner.align(source_start_ns + 64_500_000, source_start_ns + 64_500_000 + host_offset_ns) == (
        source_start_ns + 64_500_000 + host_offset_ns - 160_000_000
    )

    # A delayed socket callback must not drag the LiDAR timestamp forward.
    assert aligner.align(source_start_ns + 129_000_000, source_start_ns + 129_000_000 + host_offset_ns + 900_000_000) == (
        source_start_ns + 129_000_000 + host_offset_ns - 160_000_000
    )
    assert aligner.align(0, 99_000_000_000) == 98_840_000_000


def test_header_progress_gate_requires_a_second_advancing_source_timestamp():
    from go2_mapping_lio import lio_clock_alignment

    gate_type = getattr(lio_clock_alignment, "HeaderProgressGate", None)

    assert gate_type is not None, "raw sensor preflight must require advancing source headers"
    gate = gate_type(minimum_progress_ms=20.0)
    first = 1_700_000_000_000_000_000

    assert gate.observe(first) is False
    assert gate.observe(first) is False
    assert gate.observe(first + 19_999_999) is False
    assert gate.observe(first + 20_000_000) is True


def test_livox_custom_conversion_maps_unitree_pointcloud_fields_without_losing_scan_time():
    from go2_mapping_lio.livox_custom_conversion import PointFieldSpec, convert_unitree_pointcloud

    fields = (
        PointFieldSpec("x", 0, 7, 1),
        PointFieldSpec("y", 4, 7, 1),
        PointFieldSpec("z", 8, 7, 1),
        PointFieldSpec("intensity", 16, 7, 1),
        PointFieldSpec("ring", 20, 4, 1),
        PointFieldSpec("time", 24, 7, 1),
    )
    raw = bytearray(64)
    struct.pack_into("<fff", raw, 0, 1.0, 2.0, 3.0)
    struct.pack_into("<fH2xf", raw, 16, 12.7, 1, 0.000355504)
    struct.pack_into("<fff", raw, 32, -1.0, -2.0, -3.0)
    struct.pack_into("<fH2xf", raw, 48, 300.0, 3, 0.062518142)

    converted = convert_unitree_pointcloud(bytes(raw), point_step=32, point_count=2, fields=fields)

    assert [(point.x, point.y, point.z) for point in converted] == [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]
    assert [point.reflectivity for point in converted] == [13, 255]
    assert [point.line for point in converted] == [1, 3]
    assert [point.offset_time for point in converted] == [355_504, 62_518_142]


def test_unitree_reader_and_ros_publisher_are_separate_processes():
    reader = ROOT / "mapping" / "robot" / "go2_mapping_native_pointcloud" / "native_pointcloud_socket_server.cpp"
    publisher = ROBOT_PACKAGE / "go2_mapping_lio" / "native_pointcloud_ros.py"
    starter = ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh"

    assert reader.is_file()
    assert publisher.is_file()
    assert starter.is_file()


def test_bridge_starter_keeps_cleanup_trap_active_after_the_ros_publisher_exits():
    starter = (ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh").read_text(encoding="utf-8")

    assert "trap cleanup EXIT INT TERM" in starter
    assert "exec python3" not in starter


def test_bridge_uses_one_stable_runtime_directory_for_both_build_and_launch():
    starter = (ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh").read_text(encoding="utf-8")

    assert 'runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}' in starter


def test_bridge_runs_the_unitree_reader_with_the_unitree_dds_runtime_first():
    starter = (ROOT / "tools" / "start_go2_unitree_pointcloud_bridge.sh").read_text(encoding="utf-8")

    assert 'LD_LIBRARY_PATH=/usr/local/lib' in starter


def test_native_lowstate_imu_protocol_preserves_values_and_rejects_nonfinite_data():
    from go2_mapping_lio.native_lowstate_imu_protocol import ProtocolError, decode_frame_body

    body = struct.pack(
        "<4sHHiIH10f",
        b"G2IM",
        1,
        1,
        42,
        123,
        len(b"go2_imu"),
        1.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
        1.1,
        1.2,
        1.3,
    ) + b"go2_imu"

    frame = decode_frame_body(body)

    assert frame.stamp_sec == 42
    assert frame.stamp_nanosec == 123
    assert frame.frame_id == "go2_imu"
    assert frame.orientation_valid is True
    assert frame.angular_velocity == pytest.approx((0.1, 0.2, 0.3))
    assert frame.linear_acceleration == pytest.approx((1.1, 1.2, 1.3))

    invalid = bytearray(body)
    invalid[18:22] = struct.pack("<f", float("nan"))
    with pytest.raises(ProtocolError, match="non-finite"):
        decode_frame_body(bytes(invalid))


def test_native_lowstate_imu_path_is_split_between_unitree_dds_and_ros_runtime():
    reader = ROOT / "mapping" / "robot" / "go2_mapping_native_lowstate" / "native_lowstate_imu_socket_server.cpp"
    publisher = ROBOT_PACKAGE / "go2_mapping_lio" / "native_lowstate_imu_ros.py"
    starter = ROOT / "tools" / "start_go2_unitree_lowstate_imu_bridge.sh"

    assert reader.is_file(), "native Unitree LowState reader is required"
    assert publisher.is_file(), "ROS-only IMU publisher is required"
    assert starter.is_file(), "isolated IMU starter is required"
    reader_text = reader.read_text(encoding="utf-8")
    publisher_text = publisher.read_text(encoding="utf-8")
    starter_text = starter.read_text(encoding="utf-8")
    assert "ChannelSubscriber<unitree_go::msg::dds_::LowState_>" in reader_text
    assert "rclcpp" not in reader_text
    assert "ChannelPublisher" not in reader_text
    assert "G2IM" in reader_text
    assert "rclpy" in publisher_text
    assert "Imu" in publisher_text
    assert "native_lowstate_imu_protocol" in publisher_text
    assert 'LD_LIBRARY_PATH=/usr/local/lib' in starter_text
    assert 'GO2_MAPPING_LOWSTATE_IMU_TOPIC:-/dog_imu_raw' in starter_text
