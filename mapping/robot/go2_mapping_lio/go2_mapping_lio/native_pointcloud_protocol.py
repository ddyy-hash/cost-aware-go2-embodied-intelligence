"""Versioned wire protocol shared by the Unitree reader and ROS publisher."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List


MAGIC = b"G2PC"
VERSION = 1
MAX_FIELDS = 32
MAX_NAME_BYTES = 128
MAX_FRAME_ID_BYTES = 512
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
FIXED_HEADER = struct.Struct("<4sHHiIIIIIHBH")
FIELD_TAIL = struct.Struct("<IBI")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class NativePointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class NativePointCloudFrame:
    stamp_sec: int
    stamp_nanosec: int
    height: int
    width: int
    point_step: int
    row_step: int
    is_bigendian: bool
    is_dense: bool
    frame_id: str
    fields: List[NativePointField]
    data: bytes


def decode_frame_body(body: bytes) -> NativePointCloudFrame:
    if len(body) < FIXED_HEADER.size:
        raise ProtocolError("frame body is shorter than fixed header")
    (
        magic,
        version,
        _reserved,
        stamp_sec,
        stamp_nanosec,
        height,
        width,
        point_step,
        row_step,
        field_count,
        flags,
        frame_id_length,
    ) = FIXED_HEADER.unpack_from(body)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("unsupported native point-cloud protocol")
    if not height or not width or not point_step:
        raise ProtocolError("invalid point-cloud dimensions")
    if row_step < width * point_step:
        raise ProtocolError("row_step cannot hold declared point-cloud width and point_step")
    if field_count > MAX_FIELDS or frame_id_length > MAX_FRAME_ID_BYTES:
        raise ProtocolError("point-cloud metadata exceeds protocol limits")
    expected_data_length = row_step * height
    if expected_data_length > MAX_PAYLOAD_BYTES:
        raise ProtocolError("point-cloud data length exceeds protocol limit")

    offset = FIXED_HEADER.size
    frame_end = offset + frame_id_length
    if frame_end > len(body):
        raise ProtocolError("frame ID exceeds body length")
    try:
        frame_id = body[offset:frame_end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("frame ID is not UTF-8") from exc
    offset = frame_end
    fields: List[NativePointField] = []
    for _ in range(field_count):
        if offset + 2 > len(body):
            raise ProtocolError("point field name length exceeds body length")
        name_length = struct.unpack_from("<H", body, offset)[0]
        offset += 2
        if name_length > MAX_NAME_BYTES or offset + name_length + FIELD_TAIL.size > len(body):
            raise ProtocolError("point field metadata exceeds body length")
        try:
            name = body[offset:offset + name_length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("point field name is not UTF-8") from exc
        offset += name_length
        field_offset, datatype, count = FIELD_TAIL.unpack_from(body, offset)
        offset += FIELD_TAIL.size
        fields.append(NativePointField(name=name, offset=field_offset, datatype=datatype, count=count))
    if len(body) - offset != expected_data_length:
        raise ProtocolError("point-cloud data length does not match row_step * height")
    return NativePointCloudFrame(
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        height=height,
        width=width,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=bool(flags & 0x01),
        is_dense=bool(flags & 0x02),
        frame_id=frame_id,
        fields=fields,
        data=body[offset:],
    )
