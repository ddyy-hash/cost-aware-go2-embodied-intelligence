import hashlib

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe
from mapping.pc.go2_mapping_live.live_surface import LiveSurfaceMap, TileId
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import build_live_keyframe_message
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()


def _frame(
    sequence: int,
    points: np.ndarray,
    *,
    origin=(0.0, 0.0, 0.0),
    calibration_id="PREVIEW_UNCALIBRATED",
    sensor_frame_transform_id=None,
    points_in_map_frame=True,
):
    pose = np.eye(4)
    pose[:3, 3] = origin
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=1_000_000 + sequence,
        calibration_id=calibration_id,
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray(points, dtype="<f4"),
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=points_in_map_frame,
        sensor_frame_transform_id=sensor_frame_transform_id,
    )
    return validate_live_keyframe(encoded.payload)


def test_surface_uses_online_aggregates_and_saturates_independent_witnesses_at_four():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    points = np.asarray([[0.1010, 0.1010, 0.1010], [0.1020, 0.1020, 0.1020]], dtype="<f4")

    surface.integrate(_frame(1, points, origin=(0.0, 0.0, 0.0)))
    surface.integrate(_frame(2, points[:1], origin=(0.0, 0.0, 0.0)))
    for sequence, x in enumerate((0.06, 0.12, 0.18, 0.24, 0.30), start=3):
        surface.integrate(_frame(sequence, points[:1], origin=(x, 0.0, 0.0)))

    surfel = surface.surfel((20, 20, 20))

    assert surfel.observation_count == 7
    assert surfel.independent_frame_support == 4
    assert len(surfel.witnesses) == 4
    assert [witness.frame_identity.sequence for witness in surfel.witnesses] == [1, 3, 4, 5]
    assert surfel.centroid.dtype == np.float64
    assert surfel.covariance_accumulator.shape == (3, 3)
    expected_centroid = (np.asarray([0.102, 0.102, 0.102]) + 6 * np.asarray([0.101, 0.101, 0.101])) / 7
    assert np.allclose(surfel.centroid, expected_centroid, atol=1e-6)


def test_surface_marks_owner_tiles_dirty_and_reads_a_halo_without_duplicate_ownership():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    points = np.asarray([[0.249, 0.01, 0.01], [0.251, 0.01, 0.01]], dtype="<f4")

    result = surface.integrate(_frame(1, points))
    left = TileId(0, 0, 0)
    right = TileId(1, 0, 0)

    assert set(result.dirty_tiles) == {left, right}
    assert {surfel.key for surfel in surface.surfels_for_tile(left, halo_m=0.05)} == {(49, 1, 1), (50, 1, 1)}
    assert {surfel.key for surfel in surface.owned_surfels(left)} == {(49, 1, 1)}
    assert {surfel.key for surfel in surface.owned_surfels(right)} == {(50, 1, 1)}
    assert surface.tile_revision_seed(left) == surface.tile_revision_seed(left)


def test_surface_ranks_dirty_tiles_by_supported_surfel_count():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    left = TileId(0, 0, 0)
    right = TileId(1, 0, 0)
    surface.integrate(_frame(1, [[0.10, 0.01, 0.01], [0.30, 0.01, 0.01]], origin=(0.0, 0.0, 0.0)))
    surface.integrate(_frame(2, [[0.30, 0.01, 0.01]], origin=(0.06, 0.0, 0.0)))

    assert surface.rank_tiles_by_support({left, right}) == (right, left)


def test_surface_associates_adjacent_cells_under_realistic_lidar_jitter():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    first = np.asarray(
        [
            [0.1025, 0.1025, 0.1025],
            [0.1225, 0.1025, 0.1025],
            [0.1425, 0.1025, 0.1025],
            [0.1625, 0.1025, 0.1025],
        ],
        dtype="<f4",
    )
    second = first + np.asarray([0.004, 0.0, 0.0], dtype="<f4")

    surface.integrate(_frame(1, first, origin=(0.0, 0.0, 0.0)))
    surface.integrate(_frame(2, second, origin=(0.10, 0.0, 0.0)))

    surfels = surface.surfels()
    assert len(surfels) == len(first)
    assert all(surfel.observation_count == 2 for surfel in surfels)
    assert all(surfel.independent_frame_support == 2 for surfel in surfels)


def test_surface_binds_first_calibration_and_rejects_change():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    points = np.asarray([[0.101, 0.101, 0.101]], dtype="<f4")

    surface.integrate(_frame(1, points, calibration_id="cal-a"))

    with pytest.raises(ValueError, match="calibration"):
        surface.integrate(_frame(2, points, calibration_id="cal-b"))


def test_surface_rectifies_native_points_and_reports_the_transform_identity():
    surface = LiveSurfaceMap(
        map_id="map-a",
        source_config_sha256=SOURCE_HASH,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )
    frame = _frame(
        1,
        np.asarray([[0.0, 0.0, 0.5]], dtype="<f4"),
        points_in_map_frame=False,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )

    surface.integrate(frame)

    assert surface.snapshot()["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert surface.surfel((0, 0, -100)).centroid[2] == pytest.approx(-0.5)


def test_surface_rejects_a_frame_that_changes_the_active_transform():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    points = np.asarray([[0.1, 0.1, 0.1]], dtype="<f4")
    surface.integrate(_frame(1, points))

    with pytest.raises(ValueError, match="sensor frame transform"):
        surface.integrate(
            _frame(
                2,
                points,
                points_in_map_frame=False,
                sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
            )
        )
