import math
import struct

import pytest

from mapping.robot.go2_mapping_lio.go2_mapping_lio.map_frame_cloud import (
    FLOAT32,
    PointCloudTransformError,
    PointFieldSpec,
    RigidTransform,
    transform_pointcloud_to_map,
)


def _fields():
    return (
        PointFieldSpec("x", 0, FLOAT32, 1),
        PointFieldSpec("y", 4, FLOAT32, 1),
        PointFieldSpec("z", 8, FLOAT32, 1),
        PointFieldSpec("intensity", 12, FLOAT32, 1),
    )


def _cloud(points):
    return b"".join(struct.pack("<ffff", *point) for point in points)


def test_map_transform_changes_only_xyz_and_preserves_other_point_fields():
    source = _cloud(((1.0, 0.0, 2.0, 13.5), (0.0, -2.0, 1.0, 21.25)))
    transform = RigidTransform(
        rotation=(
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        translation=(10.0, 20.0, -3.0),
    )

    transformed = transform_pointcloud_to_map(
        source,
        point_step=16,
        point_count=2,
        fields=_fields(),
        map_from_odom=transform,
    )

    assert struct.unpack_from("<ffff", transformed, 0) == pytest.approx((10.0, 21.0, -1.0, 13.5))
    assert struct.unpack_from("<ffff", transformed, 16) == pytest.approx((12.0, 20.0, -2.0, 21.25))
    assert transformed[12:16] == source[12:16]
    assert transformed[28:32] == source[28:32]


def test_map_transform_rejects_non_rigid_or_non_finite_coordinates():
    invalid_transform = RigidTransform(
        rotation=(
            2.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        translation=(0.0, 0.0, 0.0),
    )
    with pytest.raises(PointCloudTransformError, match="proper rigid transform"):
        transform_pointcloud_to_map(
            _cloud(((0.0, 0.0, 0.0, 1.0),)),
            point_step=16,
            point_count=1,
            fields=_fields(),
            map_from_odom=invalid_transform,
        )

    with pytest.raises(PointCloudTransformError, match="non-finite"):
        transform_pointcloud_to_map(
            _cloud(((math.nan, 0.0, 0.0, 1.0),)),
            point_step=16,
            point_count=1,
            fields=_fields(),
            map_from_odom=RigidTransform.identity(),
        )


def test_map_transform_rejects_missing_or_incompatible_xyz_fields():
    with pytest.raises(PointCloudTransformError, match="missing PointCloud2 fields: z"):
        transform_pointcloud_to_map(
            _cloud(((0.0, 0.0, 0.0, 1.0),)),
            point_step=16,
            point_count=1,
            fields=_fields()[:-2],
            map_from_odom=RigidTransform.identity(),
        )

    unsupported_x = (PointFieldSpec("x", 0, 2, 1), *_fields()[1:])
    with pytest.raises(PointCloudTransformError, match="field x must be float32"):
        transform_pointcloud_to_map(
            _cloud(((0.0, 0.0, 0.0, 1.0),)),
            point_step=16,
            point_count=1,
            fields=unsupported_x,
            map_from_odom=RigidTransform.identity(),
        )
