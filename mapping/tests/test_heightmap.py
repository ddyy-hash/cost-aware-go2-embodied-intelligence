import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.heightmap import HeightmapError, build_pgtt_heightmap, build_stair_probe_profile, transform_lidar_points


def test_build_pgtt_heightmap_is_finite_and_stable():
    points = np.asarray([[x, y, 0.6 + 0.02 * x] for x in np.linspace(-0.5, 0.5, 21) for y in np.linspace(-0.4, 0.4, 17)])
    result = build_pgtt_heightmap(points)
    assert result.shape == (11, 9)
    assert np.isfinite(result).all()
    assert float(result[5, 4]) == pytest.approx(0.0, abs=0.01)


def test_heightmap_rejects_empty_local_window():
    with pytest.raises(HeightmapError, match="local window"):
        build_pgtt_heightmap([[10.0, 10.0, 0.0]])


def test_transform_lidar_points_applies_versioned_pitch_half_turn():
    points = np.asarray([[0.3, -0.2, 0.7], [-0.1, 0.4, 0.5]], dtype=np.float32)
    transformed = transform_lidar_points(points, "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2")
    np.testing.assert_allclose(transformed, [[-0.3, -0.2, -0.7], [0.1, 0.4, -0.5]])


def test_transform_lidar_points_rejects_unknown_transform():
    with pytest.raises(HeightmapError, match="transform"):
        transform_lidar_points([[0.0, 0.0, 0.0]], "unknown")


def test_transform_lidar_points_auto_uses_vendor_aligned_utlidar_frame():
    points = np.asarray([[0.3, -0.2, 0.7]], dtype=np.float32)
    transformed = transform_lidar_points(points, "AUTO_GO2_LIDAR_FRAME_v1", source_frame_id="utlidar_lidar")
    np.testing.assert_allclose(transformed, points)


def test_transform_lidar_points_auto_applies_half_turn_to_raw_fallback_frame():
    points = np.asarray([[0.3, -0.2, 0.7]], dtype=np.float32)
    transformed = transform_lidar_points(points, "AUTO_GO2_LIDAR_FRAME_v1", source_frame_id="go2_lidar_raw")
    np.testing.assert_allclose(transformed, [[-0.3, -0.2, -0.7]])


def test_transform_lidar_points_auto_rejects_unknown_source_frame():
    with pytest.raises(HeightmapError, match="source frame"):
        transform_lidar_points([[0.0, 0.0, 0.0]], "AUTO_GO2_LIDAR_FRAME_v1", source_frame_id="other")


def test_build_stair_probe_profile_covers_two_metres_without_changing_pgtt_grid():
    points = np.asarray([[x, y, 0.6 + (0.12 * int(max(0, x) // 0.3))] for x in np.linspace(-0.5, 1.5, 81) for y in np.linspace(-0.4, 0.4, 9)])
    profile = build_stair_probe_profile(points)
    assert profile.shape == (21, 9)
    assert np.isfinite(profile).all()
    assert float(profile[-1].mean() - profile[5].mean()) > 0.2


def test_build_stair_probe_profile_preserves_sparse_upper_treads_over_ground_returns():
    points = []
    for x in np.linspace(-0.5, 1.5, 81):
        tread_height = 0.6 + 0.12 * int(max(0.0, float(x)) // 0.3)
        points.extend([[x, y, 0.6] for y in np.linspace(-0.4, 0.4, 20)])
        points.extend([[x, y, tread_height] for y in np.linspace(-0.2, 0.2, 4)])
    profile = build_stair_probe_profile(points)
    assert float(profile[-1].mean() - profile[5].mean()) > 0.2


def test_build_stair_probe_profile_rejects_single_high_return_as_noise():
    points = [[x, y, 0.6] for x in np.linspace(-0.5, 1.5, 81) for y in np.linspace(-0.4, 0.4, 20)]
    points.append([1.2, 0.0, 1.4])
    profile = build_stair_probe_profile(points)
    assert float(profile.max()) < 0.05
