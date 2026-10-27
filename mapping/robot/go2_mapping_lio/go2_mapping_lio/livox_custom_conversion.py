"""Pure conversion of the observed Unitree PointCloud2 layout to Livox points."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


FLOAT32 = 7
UINT8 = 2
UINT16 = 4
UINT32_MAX = 4_294_967_295


class ConversionError(ValueError):
    """The PointCloud2 layout cannot be represented by Livox CustomMsg."""


@dataclass(frozen=True)
class PointFieldSpec:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class LivoxCustomPointData:
    offset_time: int
    x: float
    y: float
    z: float
    reflectivity: int
    tag: int
    line: int


def _field_map(fields: Iterable[PointFieldSpec], point_step: int) -> dict:
    result = {}
    for field in fields:
        if field.count != 1 or field.offset < 0 or field.offset + 4 > point_step:
            raise ConversionError("invalid PointCloud2 field %r" % (field.name,))
        result[field.name] = field
    required = ("x", "y", "z", "intensity", "ring", "time")
    missing = [name for name in required if name not in result]
    if missing:
        raise ConversionError("missing PointCloud2 fields: %s" % ", ".join(missing))
    for name in ("x", "y", "z", "intensity", "time"):
        if result[name].datatype != FLOAT32:
            raise ConversionError("PointCloud2 field %s must be float32" % name)
    if result["ring"].datatype not in (UINT8, UINT16):
        raise ConversionError("PointCloud2 field ring must be uint8 or uint16")
    return result


def _float32(data: bytes, start: int) -> float:
    return struct.unpack_from("<f", data, start)[0]


def _ring(data: bytes, start: int, datatype: int) -> int:
    if datatype == UINT8:
        return data[start]
    return struct.unpack_from("<H", data, start)[0]


def convert_unitree_pointcloud(
    data: bytes, point_step: int, point_count: int, fields: Sequence[PointFieldSpec]
) -> Tuple[LivoxCustomPointData, ...]:
    """Convert one Unitree scan while retaining its per-point time offsets."""
    if point_step <= 0 or point_count <= 0:
        raise ConversionError("PointCloud2 must contain at least one point")
    if len(data) < point_step * point_count:
        raise ConversionError("PointCloud2 data is shorter than its declared point count")
    field_map = _field_map(fields, point_step)
    converted = []
    for index in range(point_count):
        base = index * point_step
        x = _float32(data, base + field_map["x"].offset)
        y = _float32(data, base + field_map["y"].offset)
        z = _float32(data, base + field_map["z"].offset)
        intensity = _float32(data, base + field_map["intensity"].offset)
        seconds = _float32(data, base + field_map["time"].offset)
        line = _ring(data, base + field_map["ring"].offset, field_map["ring"].datatype)
        if not all(math.isfinite(value) for value in (x, y, z, intensity, seconds)):
            raise ConversionError("PointCloud2 contains non-finite point values")
        if seconds < 0 or seconds * 1_000_000_000 > UINT32_MAX:
            raise ConversionError("PointCloud2 time is outside Livox CustomPoint offset_time range")
        if line > 255:
            raise ConversionError("PointCloud2 ring is outside Livox CustomPoint line range")
        converted.append(
            LivoxCustomPointData(
                offset_time=int(round(seconds * 1_000_000_000)),
                x=x,
                y=y,
                z=z,
                reflectivity=max(0, min(255, int(round(intensity)))),
                tag=0,
                line=line,
            )
        )
    return tuple(converted)
