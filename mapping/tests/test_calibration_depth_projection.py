import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.depth_projection import (
    image_to_depth_m,
    project_lidar_to_depth,
)


def test_depth_projection_uses_depth_frame_transform_and_z_buffer():
    depth = image_to_depth_m(np.full((8, 8), 2_000, dtype=np.uint16), "16UC1", 0.001)
    lidar = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.2, 0.0, 2.0]])
    camera = np.asarray([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]])
    depth_from_lidar = np.eye(4)
    depth_from_lidar[0, 3] = 0.1

    observed, predicted, pixels = project_lidar_to_depth(
        lidar,
        depth,
        camera,
        depth_from_lidar,
        depth_scale_m=0.001,
    )

    assert len(observed) == 2
    assert observed == pytest.approx((2.0, 2.0))
    assert predicted == pytest.approx((1.0, 2.0))
    assert len(set(pixels)) == 2


def test_depth_image_conversion_rejects_unknown_encoding_and_scales_losslessly():
    assert np.array_equal(image_to_depth_m(np.asarray([[1000]], dtype=np.uint16), "16UC1", 0.001), [[1.0]])
    assert np.array_equal(image_to_depth_m(np.asarray([[1.5]], dtype=np.float32), "32FC1", 1.0), [[1.5]])
    with pytest.raises(ValueError, match="encoding"):
        image_to_depth_m(np.ones((2, 2), dtype=np.uint8), "mono8", 1.0)
