import numpy as np
import pytest

from mapping.pc.go2_mapping_live.sensor_frame import (
    IDENTITY_SENSOR_FRAME_V1,
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    bind_pose_generation,
    get_sensor_frame_transform,
    transform_points,
)


def test_provisional_go2_transform_is_a_proper_rotation_and_points_downward():
    transform = get_sensor_frame_transform(PROVISIONAL_GO2_UTLIDAR_PITCH180_V2)

    assert np.allclose(transform.rotation @ transform.rotation.T, np.eye(3))
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0)
    assert np.allclose(
        transform_points(np.asarray([[0.0, 0.0, 0.4]]), transform.id),
        [[0.0, 0.0, -0.4]],
    )
    assert transform.translation.tolist() == [0.0, 0.0, 0.0]
    assert transform.provisional is True


def test_identity_transform_preserves_points_and_unknown_ids_fail_closed():
    points = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32)

    assert np.array_equal(transform_points(points, IDENTITY_SENSOR_FRAME_V1), points)
    with pytest.raises(ValueError, match="sensor frame transform"):
        get_sensor_frame_transform("UNDECLARED_TRANSFORM")


def test_non_identity_transform_is_bound_into_content_addressed_pose_generation():
    base = "a" * 64

    assert bind_pose_generation(base, IDENTITY_SENSOR_FRAME_V1) == base
    bound = bind_pose_generation(base, PROVISIONAL_GO2_UTLIDAR_PITCH180_V2)

    assert len(bound) == 64
    assert bound != base
    assert bind_pose_generation(base, PROVISIONAL_GO2_UTLIDAR_PITCH180_V2) == bound
