"""Strict CDR decoders for calibration-critical LiDAR, IMU, and LIO data."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np


class CalibrationCdrError(ValueError):
    pass


class _Reader:
    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4 or payload[:2] not in (b"\x00\x01", b"\x00\x03"):
            raise CalibrationCdrError("only little-endian CDR encapsulation is supported")
        self.payload = memoryview(payload)[4:]
        self.offset = 0

    def take(self, count: int, alignment: int = 1) -> memoryview:
        self.offset += (-self.offset) % alignment
        end = self.offset + count
        if count < 0 or end > len(self.payload):
            raise CalibrationCdrError("truncated CDR payload")
        result = self.payload[self.offset:end]
        self.offset = end
        return result

    def u8(self) -> int:
        return int(self.take(1)[0])

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4, 4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4, 4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8, 8))[0]

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        raw = bytes(self.take(length))
        if raw[-1:] != b"\0":
            raise CalibrationCdrError("CDR string has no terminator")
        try:
            return raw[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CalibrationCdrError("CDR frame ID is not UTF-8") from exc


def _header(reader: _Reader) -> tuple[int, str]:
    seconds = reader.i32()
    nanoseconds = reader.u32()
    if nanoseconds >= 1_000_000_000:
        raise CalibrationCdrError("header nanoseconds are outside [0, 1e9)")
    frame_id = reader.string()
    if not frame_id:
        raise CalibrationCdrError("sensor frame ID is required")
    return seconds * 1_000_000_000 + nanoseconds, frame_id


def _quaternion(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not all(math.isfinite(value) for value in values):
        raise CalibrationCdrError("quaternion contains non-finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise CalibrationCdrError("quaternion is not normalized")
    return tuple(value / norm for value in values)


def _rotation_matrix(values: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = _quaternion(values)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PoseRecord:
    header_timestamp_ns: int
    frame_id: str
    pose: np.ndarray


@dataclass(frozen=True)
class ImuRecord:
    header_timestamp_ns: int
    frame_id: str
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]


@dataclass(frozen=True)
class PointCloudRecord:
    header_timestamp_ns: int
    frame_id: str
    points_xyz: np.ndarray


def decode_pose(payload: bytes) -> PoseRecord:
    reader = _Reader(payload)
    timestamp, frame_id = _header(reader)
    position = tuple(reader.f64() for _ in range(3))
    quaternion = tuple(reader.f64() for _ in range(4))
    if not all(math.isfinite(value) for value in position):
        raise CalibrationCdrError("pose translation contains non-finite values")
    pose = np.eye(4)
    pose[:3, :3] = _rotation_matrix(quaternion)
    pose[:3, 3] = position
    pose.setflags(write=False)
    return PoseRecord(timestamp, frame_id, pose)


def decode_imu(payload: bytes) -> ImuRecord:
    reader = _Reader(payload)
    timestamp, frame_id = _header(reader)
    orientation = _quaternion(tuple(reader.f64() for _ in range(4)))
    for _ in range(9):
        reader.f64()
    angular = tuple(reader.f64() for _ in range(3))
    for _ in range(9):
        reader.f64()
    acceleration = tuple(reader.f64() for _ in range(3))
    for _ in range(9):
        reader.f64()
    if not all(math.isfinite(value) for value in angular + acceleration):
        raise CalibrationCdrError("IMU contains non-finite values")
    return ImuRecord(timestamp, frame_id, orientation, angular, acceleration)


def decode_pointcloud(payload: bytes) -> PointCloudRecord:
    reader = _Reader(payload)
    timestamp, frame_id = _header(reader)
    height = reader.u32()
    width = reader.u32()
    field_count = reader.u32()
    if height <= 0 or width <= 0 or field_count > 128:
        raise CalibrationCdrError("PointCloud2 dimensions are invalid")
    fields: dict[str, tuple[int, int, int]] = {}
    for _ in range(field_count):
        name = reader.string()
        offset = reader.u32()
        datatype = reader.u8()
        count = reader.u32()
        fields[name] = (offset, datatype, count)
    big_endian = reader.u8()
    if big_endian not in (0, 1):
        raise CalibrationCdrError("PointCloud2 endian flag is invalid")
    point_step = reader.u32()
    row_step = reader.u32()
    raw = bytes(reader.take(reader.u32()))
    dense = reader.u8()
    if dense not in (0, 1):
        raise CalibrationCdrError("PointCloud2 dense flag is invalid")
    if point_step <= 0 or row_step < point_step * width or len(raw) != row_step * height:
        raise CalibrationCdrError("PointCloud2 byte layout is inconsistent")
    required = [fields.get(axis) for axis in ("x", "y", "z")]
    if any(item is None for item in required):
        raise CalibrationCdrError("PointCloud2 has no XYZ fields")
    if any(item[1:] != (7, 1) or item[0] + 4 > point_step for item in required):
        raise CalibrationCdrError("PointCloud2 XYZ fields must be scalar float32 values")
    dtype = ">f4" if big_endian else "<f4"
    columns = []
    for offset, _, _ in required:
        rows = [
            np.ndarray(
                (width,), dtype=dtype, buffer=raw, offset=row * row_step + offset, strides=(point_step,)
            )
            for row in range(height)
        ]
        columns.append(np.concatenate(rows))
    points = np.column_stack(columns)
    points = np.asarray(points[np.all(np.isfinite(points), axis=1)], dtype=np.float64)
    points.setflags(write=False)
    return PointCloudRecord(timestamp, frame_id, points)
