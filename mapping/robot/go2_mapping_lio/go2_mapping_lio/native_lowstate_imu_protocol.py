"""Versioned wire protocol for an isolated Unitree LowState IMU reader."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Tuple


MAGIC = b"G2IM"
VERSION = 1
ORIENTATION_VALID = 0x01
MAX_FRAME_ID_BYTES = 512
FIXED_HEADER = struct.Struct("<4sHHiIH10f")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class NativeImuFrame:
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    orientation_valid: bool
    orientation: Tuple[float, float, float, float]
    angular_velocity: Tuple[float, float, float]
    linear_acceleration: Tuple[float, float, float]


def decode_frame_body(body: bytes) -> NativeImuFrame:
    if len(body) < FIXED_HEADER.size:
        raise ProtocolError("IMU frame body is shorter than fixed header")
    (
        magic,
        version,
        flags,
        stamp_sec,
        stamp_nanosec,
        frame_id_length,
        orientation_w,
        orientation_x,
        orientation_y,
        orientation_z,
        angular_velocity_x,
        angular_velocity_y,
        angular_velocity_z,
        linear_acceleration_x,
        linear_acceleration_y,
        linear_acceleration_z,
    ) = FIXED_HEADER.unpack_from(body)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("unsupported native IMU protocol")
    if flags & ~ORIENTATION_VALID:
        raise ProtocolError("native IMU flags contain unsupported bits")
    if frame_id_length > MAX_FRAME_ID_BYTES or len(body) != FIXED_HEADER.size + frame_id_length:
        raise ProtocolError("native IMU frame ID length does not match body")
    values = (
        orientation_w,
        orientation_x,
        orientation_y,
        orientation_z,
        angular_velocity_x,
        angular_velocity_y,
        angular_velocity_z,
        linear_acceleration_x,
        linear_acceleration_y,
        linear_acceleration_z,
    )
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("native IMU contains non-finite values")
    try:
        frame_id = body[FIXED_HEADER.size:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("native IMU frame ID is not UTF-8") from exc
    return NativeImuFrame(
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        frame_id=frame_id,
        orientation_valid=bool(flags & ORIENTATION_VALID),
        orientation=(orientation_w, orientation_x, orientation_y, orientation_z),
        angular_velocity=(angular_velocity_x, angular_velocity_y, angular_velocity_z),
        linear_acceleration=(linear_acceleration_x, linear_acceleration_y, linear_acceleration_z),
    )
