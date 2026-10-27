"""Strict ROS 2 CDR decoders for verified RGB-D calibration streams."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np


class RgbdDecodeError(ValueError):
    pass


class _Reader:
    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4 or payload[:2] not in (b"\x00\x01", b"\x00\x03"):
            raise RgbdDecodeError("only little-endian CDR encapsulation is supported")
        self.payload = memoryview(payload)[4:]
        self.offset = 0

    def _take(self, count: int, alignment: int = 1) -> memoryview:
        self.offset += (-self.offset) % alignment
        end = self.offset + count
        if count < 0 or end > len(self.payload):
            raise RgbdDecodeError("truncated CDR payload")
        result = self.payload[self.offset:end]
        self.offset = end
        return result

    def u8(self) -> int:
        return int(self._take(1)[0])

    def boolean(self) -> bool:
        value = self.u8()
        if value not in (0, 1):
            raise RgbdDecodeError("CDR boolean is not 0 or 1")
        return bool(value)

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4, 4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4, 4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._take(8, 8))[0]

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        raw = bytes(self._take(length))
        if raw[-1:] != b"\0":
            raise RgbdDecodeError("CDR string has no NUL terminator")
        try:
            return raw[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RgbdDecodeError("CDR string is not UTF-8") from exc

    def raw(self, count: int) -> bytes:
        return bytes(self._take(count))

    def finished(self) -> None:
        if any(value != 0 for value in self.payload[self.offset:]):
            raise RgbdDecodeError("unexpected non-padding bytes after message")


def _header(reader: _Reader) -> tuple[int, str]:
    seconds = reader.i32()
    nanoseconds = reader.u32()
    if nanoseconds >= 1_000_000_000:
        raise RgbdDecodeError("header nanoseconds are outside [0, 1e9)")
    return seconds * 1_000_000_000 + nanoseconds, reader.string()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ImageRecord:
    header_timestamp_ns: int
    frame_id: str
    width: int
    height: int
    encoding: str
    step: int
    is_bigendian: bool
    image: np.ndarray
    message_sha256: str


@dataclass(frozen=True)
class CameraInfoRecord:
    header_timestamp_ns: int
    frame_id: str
    width: int
    height: int
    distortion_model: str
    d: np.ndarray
    k: np.ndarray
    r: np.ndarray
    p: np.ndarray
    binning_x: int
    binning_y: int
    roi: dict[str, int | bool]
    message_sha256: str


@dataclass(frozen=True)
class TransformRecord:
    header_timestamp_ns: int
    parent_frame_id: str
    child_frame_id: str
    translation_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


_ENCODINGS = {
    "rgb8": (np.dtype("u1"), 3),
    "bgr8": (np.dtype("u1"), 3),
    "mono8": (np.dtype("u1"), 1),
    "16UC1": (np.dtype("u2"), 1),
    "32FC1": (np.dtype("f4"), 1),
}


def decode_image(payload: bytes) -> ImageRecord:
    reader = _Reader(payload)
    timestamp_ns, frame_id = _header(reader)
    height = reader.u32()
    width = reader.u32()
    encoding = reader.string()
    is_bigendian = reader.boolean()
    step = reader.u32()
    length = reader.u32()
    raw = reader.raw(length)
    reader.finished()
    if not frame_id or width <= 0 or height <= 0:
        raise RgbdDecodeError("image header and dimensions are required")
    if encoding not in _ENCODINGS:
        raise RgbdDecodeError(f"unsupported image encoding: {encoding}")
    base_dtype, channels = _ENCODINGS[encoding]
    bytes_per_pixel = base_dtype.itemsize * channels
    if step < width * bytes_per_pixel or length != step * height:
        raise RgbdDecodeError("image byte count or row step is inconsistent")
    dtype = base_dtype.newbyteorder(">" if is_bigendian and base_dtype.itemsize > 1 else "<")
    rows = []
    for row in range(height):
        visible = raw[row * step : row * step + width * bytes_per_pixel]
        rows.append(np.frombuffer(visible, dtype=dtype, count=width * channels))
    image = np.stack(rows)
    if channels > 1:
        image = image.reshape(height, width, channels)
    else:
        image = image.reshape(height, width)
    if is_bigendian and base_dtype.itemsize > 1:
        image = image.astype(base_dtype.newbyteorder("="), copy=True)
    else:
        image = np.array(image, copy=True)
    return ImageRecord(
        header_timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        is_bigendian=is_bigendian,
        image=_readonly(image),
        message_sha256=hashlib.sha256(payload).hexdigest(),
    )


def decode_camera_info(payload: bytes) -> CameraInfoRecord:
    reader = _Reader(payload)
    timestamp_ns, frame_id = _header(reader)
    height = reader.u32()
    width = reader.u32()
    distortion_model = reader.string()
    d = np.asarray([reader.f64() for _ in range(reader.u32())], dtype=np.float64)
    k = np.asarray([reader.f64() for _ in range(9)], dtype=np.float64).reshape(3, 3)
    r = np.asarray([reader.f64() for _ in range(9)], dtype=np.float64).reshape(3, 3)
    p = np.asarray([reader.f64() for _ in range(12)], dtype=np.float64).reshape(3, 4)
    binning_x = reader.u32()
    binning_y = reader.u32()
    roi = {
        "x_offset": reader.u32(),
        "y_offset": reader.u32(),
        "height": reader.u32(),
        "width": reader.u32(),
        "do_rectify": reader.boolean(),
    }
    reader.finished()
    if not frame_id or width <= 0 or height <= 0 or not distortion_model:
        raise RgbdDecodeError("CameraInfo identity and dimensions are required")
    if not all(np.all(np.isfinite(value)) for value in (d, k, r, p)):
        raise RgbdDecodeError("CameraInfo contains non-finite values")
    if k[0, 0] <= 0.0 or k[1, 1] <= 0.0 or not math.isclose(k[2, 2], 1.0):
        raise RgbdDecodeError("CameraInfo intrinsics are unusable")
    if p[0, 0] <= 0.0 or p[1, 1] <= 0.0 or not math.isclose(p[2, 2], 1.0):
        raise RgbdDecodeError("CameraInfo projection is unusable")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-5):
        raise RgbdDecodeError("CameraInfo rectification is not a proper rotation")
    return CameraInfoRecord(
        header_timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        width=width,
        height=height,
        distortion_model=distortion_model,
        d=_readonly(d),
        k=_readonly(k),
        r=_readonly(r),
        p=_readonly(p),
        binning_x=binning_x,
        binning_y=binning_y,
        roi=roi,
        message_sha256=hashlib.sha256(payload).hexdigest(),
    )


def decode_tf_message(payload: bytes) -> tuple[TransformRecord, ...]:
    reader = _Reader(payload)
    records: list[TransformRecord] = []
    count = reader.u32()
    if count > 10_000:
        raise RgbdDecodeError("TF message transform count is unreasonable")
    for _ in range(count):
        timestamp_ns, parent = _header(reader)
        child = reader.string()
        translation = tuple(reader.f64() for _ in range(3))
        rotation = tuple(reader.f64() for _ in range(4))
        if not parent or not child or parent == child:
            raise RgbdDecodeError("TF frame IDs are missing or cyclic")
        if not all(math.isfinite(value) for value in translation + rotation):
            raise RgbdDecodeError("TF transform contains non-finite values")
        norm = math.sqrt(sum(value * value for value in rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise RgbdDecodeError("TF quaternion is not normalized")
        records.append(TransformRecord(timestamp_ns, parent, child, translation, rotation))
    reader.finished()
    return tuple(records)
