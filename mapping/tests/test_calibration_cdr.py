import struct

import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.cdr import (
    CalibrationCdrError,
    decode_imu,
    decode_pointcloud,
    decode_pose,
)


class _Writer:
    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")
        self.origin = 4

    def align(self, size):
        self.data.extend(b"\0" * ((-(len(self.data) - self.origin)) % size))

    def pack(self, fmt, value, alignment):
        self.align(alignment)
        self.data.extend(struct.pack(fmt, value))

    def u8(self, value):
        self.pack("<B", value, 1)

    def u32(self, value):
        self.pack("<I", value, 4)

    def i32(self, value):
        self.pack("<i", value, 4)

    def f32(self, value):
        self.pack("<f", value, 4)

    def f64(self, value):
        self.pack("<d", value, 8)

    def string(self, value):
        raw = value.encode() + b"\0"
        self.u32(len(raw))
        self.data.extend(raw)

    def header(self, timestamp_ns, frame_id):
        self.i32(timestamp_ns // 1_000_000_000)
        self.u32(timestamp_ns % 1_000_000_000)
        self.string(frame_id)


def _pose_payload(timestamp_ns=2_000_000_003):
    writer = _Writer()
    writer.header(timestamp_ns, "lidar_odom")
    for value in (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0):
        writer.f64(value)
    for _ in range(36):
        writer.f64(0.0)
    return bytes(writer.data)


def _imu_payload(timestamp_ns=2_000_000_003, quaternion=(0.0, 0.0, 0.0, 1.0)):
    writer = _Writer()
    writer.header(timestamp_ns, "go2_imu")
    for value in quaternion:
        writer.f64(value)
    for _ in range(9):
        writer.f64(0.0)
    for value in (0.1, 0.2, 0.3):
        writer.f64(value)
    for _ in range(9):
        writer.f64(0.0)
    for value in (1.1, 1.2, 1.3):
        writer.f64(value)
    for _ in range(9):
        writer.f64(0.0)
    return bytes(writer.data)


def _pointcloud_payload(timestamp_ns=2_000_000_003):
    writer = _Writer()
    writer.header(timestamp_ns, "livox_frame")
    writer.u32(1)
    writer.u32(2)
    writer.u32(3)
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        writer.string(name)
        writer.u32(offset)
        writer.u8(7)
        writer.u32(1)
    writer.u8(0)
    writer.u32(16)
    writer.u32(32)
    raw = struct.pack("<ffff", 1.0, 2.0, 3.0, 7.0) + struct.pack("<ffff", 4.0, 5.0, 6.0, 8.0)
    writer.u32(len(raw))
    writer.data.extend(raw)
    writer.u8(1)
    return bytes(writer.data)


def test_calibration_cdr_preserves_header_frame_and_numeric_sensor_values():
    pose = decode_pose(_pose_payload())
    imu = decode_imu(_imu_payload())
    cloud = decode_pointcloud(_pointcloud_payload())

    assert pose.header_timestamp_ns == 2_000_000_003
    assert pose.frame_id == "lidar_odom"
    assert pose.pose[:3, 3] == pytest.approx((1.0, 2.0, 3.0))
    assert imu.angular_velocity == pytest.approx((0.1, 0.2, 0.3))
    assert imu.linear_acceleration == pytest.approx((1.1, 1.2, 1.3))
    assert cloud.frame_id == "livox_frame"
    assert np.array_equal(cloud.points_xyz, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert cloud.points_xyz.flags.writeable is False


def test_calibration_cdr_rejects_unnormalized_pose_and_imu_quaternions():
    bad_pose = bytearray(_pose_payload())
    # Replace the pose's w after the variable-length header.
    header_end = 4 + 4 + 4 + 4 + len("lidar_odom") + 1
    header_end += (-((header_end - 4))) % 8
    bad_pose[header_end + 6 * 8 : header_end + 7 * 8] = struct.pack("<d", 0.0)
    with pytest.raises(CalibrationCdrError, match="quaternion"):
        decode_pose(bytes(bad_pose))
    with pytest.raises(CalibrationCdrError, match="quaternion"):
        decode_imu(_imu_payload(quaternion=(0.0, 0.0, 0.0, 0.0)))
