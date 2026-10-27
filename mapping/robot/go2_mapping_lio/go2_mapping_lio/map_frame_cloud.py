"""Convert verified local LIO PointCloud2 data into the LIO ``map`` frame."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence


FLOAT32 = 7


class PointCloudTransformError(ValueError):
    """A PointCloud2 payload cannot safely be transformed into the map frame."""


@dataclass(frozen=True)
class PointFieldSpec:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class RigidTransform:
    rotation: tuple[float, float, float, float, float, float, float, float, float]
    translation: tuple[float, float, float]

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(
            rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            translation=(0.0, 0.0, 0.0),
        )

    def matrix(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not all(math.isfinite(value) for value in (*self.rotation, *self.translation)):
            raise PointCloudTransformError("transform contains non-finite values")
        rotation = (
            self.rotation[:3],
            self.rotation[3:6],
            self.rotation[6:9],
        )
        for row_index in range(3):
            for column_index in range(3):
                dot = sum(rotation[row][row_index] * rotation[row][column_index] for row in range(3))
                expected = 1.0 if row_index == column_index else 0.0
                if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=1e-6):
                    raise PointCloudTransformError("transform must be a proper rigid transform")
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise PointCloudTransformError("transform must be a proper rigid transform")
        return rotation, self.translation


def _xyz_fields(fields: Iterable[PointFieldSpec], point_step: int) -> dict[str, PointFieldSpec]:
    result = {field.name: field for field in fields}
    missing = [name for name in ("x", "y", "z") if name not in result]
    if missing:
        raise PointCloudTransformError("missing PointCloud2 fields: %s" % ", ".join(missing))
    for name in ("x", "y", "z"):
        field = result[name]
        if field.count != 1 or field.offset < 0 or field.offset + 4 > point_step:
            raise PointCloudTransformError("field %s has invalid layout" % name)
        if field.datatype != FLOAT32:
            raise PointCloudTransformError("field %s must be float32" % name)
    return result


def transform_pointcloud_to_map(
    data: bytes,
    *,
    point_step: int,
    point_count: int,
    fields: Sequence[PointFieldSpec],
    map_from_odom: RigidTransform,
) -> bytes:
    """Return the input point records with only finite XYZ values transformed.

    The caller is responsible for obtaining a timestamp-matched ``map <- odom``
    transform. This function preserves each point's non-XYZ fields byte for byte.
    """

    if point_step <= 0 or point_count <= 0:
        raise PointCloudTransformError("PointCloud2 must contain at least one point")
    required_bytes = point_step * point_count
    if len(data) < required_bytes:
        raise PointCloudTransformError("PointCloud2 data is shorter than its declared point count")
    field_map = _xyz_fields(fields, point_step)
    rotation, translation = map_from_odom.matrix()
    transformed = bytearray(data)
    for index in range(point_count):
        base = index * point_step
        xyz = (
            struct.unpack_from("<f", data, base + field_map["x"].offset)[0],
            struct.unpack_from("<f", data, base + field_map["y"].offset)[0],
            struct.unpack_from("<f", data, base + field_map["z"].offset)[0],
        )
        if not all(math.isfinite(value) for value in xyz):
            raise PointCloudTransformError("PointCloud2 contains non-finite XYZ values")
        map_xyz = tuple(
            sum(rotation[row][column] * xyz[column] for column in range(3)) + translation[row]
            for row in range(3)
        )
        for name, value in zip(("x", "y", "z"), map_xyz, strict=True):
            struct.pack_into("<f", transformed, base + field_map[name].offset, float(value))
    return bytes(transformed)
