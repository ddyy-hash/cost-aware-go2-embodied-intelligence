from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe
from mapping.pc.go2_mapping_live import live_surface
from mapping.pc.go2_mapping_live.live_surface import LiveSurfaceMap
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import build_live_keyframe_message
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()


def _frame(
    sequence: int,
    origin_x: float,
    session_id: str = "session-a",
    calibration_id: str = "PREVIEW_UNCALIBRATED",
    sensor_frame_transform_id=None,
):
    pose = np.eye(4)
    pose[0, 3] = origin_x
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id=session_id,
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=5_000 + sequence,
        calibration_id=calibration_id,
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray([[0.101, 0.101, 0.101], [0.201, 0.101, 0.101]], dtype="<f4"),
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=True,
        sensor_frame_transform_id=sensor_frame_transform_id,
    )
    return validate_live_keyframe(encoded.payload)


def test_checkpoint_restores_complete_surfel_state_and_contiguous_cursor(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    first = _frame(10, 0.0)
    second = _frame(11, 0.06)
    surface.integrate(first)
    surface.integrate(second)

    checkpoint = surface.checkpoint(tmp_path)
    restored = LiveSurfaceMap.load_checkpoint(
        tmp_path,
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )

    assert len(checkpoint.generation_id) == 64
    assert len(checkpoint.pose_revision_id) == 64
    assert checkpoint.pose_revision_id != checkpoint.generation_id
    manifest = json.loads((checkpoint.generation_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4
    assert not (checkpoint.generation_path / "witnesses.json").exists()
    assert manifest["pose_revision_id"] == checkpoint.pose_revision_id
    assert (tmp_path / "CURRENT").read_text(encoding="ascii").strip() == checkpoint.generation_id
    assert restored.last_integrated_identity == second.identity
    assert restored.snapshot() == surface.snapshot()
    assert restored.pose_revision_id == checkpoint.pose_revision_id
    before = surface.surfel((20, 20, 20))
    after = restored.surfel((20, 20, 20))
    assert np.array_equal(after.centroid, before.centroid)
    assert np.array_equal(after.covariance_accumulator, before.covariance_accumulator)
    assert after.witnesses == before.witnesses

    third = _frame(12, 0.12)
    surface.integrate(third)
    restored.integrate(third)
    continued_before = surface.surfel((20, 20, 20))
    continued_after = restored.surfel((20, 20, 20))
    assert np.array_equal(continued_after.centroid, continued_before.centroid)
    assert np.allclose(continued_after.mean_view_direction, continued_before.mean_view_direction, atol=1e-7)


def test_checkpoint_manifest_binds_and_restores_sensor_frame_transform_identity(tmp_path):
    surface = LiveSurfaceMap(
        map_id="map-a",
        source_config_sha256=SOURCE_HASH,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )
    surface.integrate(
        _frame(10, 0.0, sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2)
    )
    checkpoint = surface.checkpoint(tmp_path)
    manifest = json.loads((checkpoint.generation_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    restored = LiveSurfaceMap.load_checkpoint(
        tmp_path,
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )
    assert restored.snapshot()["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def test_checkpoint_loader_decompresses_each_npz_array_once(tmp_path, monkeypatch):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(10, 0.0))
    surface.checkpoint(tmp_path)
    real_load = np.load
    accesses = Counter()

    class CountingArchive:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            self.archive.__enter__()
            return self

        def __exit__(self, *args):
            return self.archive.__exit__(*args)

        def __getitem__(self, name):
            accesses[name] += 1
            return self.archive[name]

    monkeypatch.setattr(live_surface.np, "load", lambda *args, **kwargs: CountingArchive(real_load(*args, **kwargs)))

    restored = LiveSurfaceMap.load_checkpoint(
        tmp_path,
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )

    assert restored.snapshot()["surfel_count"] == 2
    assert accesses
    assert set(accesses.values()) == {1}


def test_checkpoint_rejects_hash_mismatch_and_ignores_unpublished_partial_generation(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, 0.0))
    surface.checkpoint(tmp_path)
    partial = tmp_path / "generations" / ".partial-crash"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}", encoding="utf-8")

    restored = LiveSurfaceMap.load_checkpoint(
        tmp_path,
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )
    assert restored.snapshot()["surfel_count"] == 2

    with pytest.raises(ValueError, match="configuration"):
        LiveSurfaceMap.load_checkpoint(
            tmp_path,
            expected_map_id="map-a",
            expected_source_config_sha256="0" * 64,
        )


def test_checkpoint_does_not_reuse_a_corrupt_existing_content_addressed_generation(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, 0.0))
    first = surface.checkpoint(tmp_path)
    arrays = first.generation_path / "surfels.npz"
    arrays.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="existing checkpoint"):
        surface.checkpoint(tmp_path)


def test_checkpoint_provenance_is_required_when_loading_as_a_bound_target(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, 0.0))
    checkpoint = surface.checkpoint(
        tmp_path,
        building_id="building-a",
        session_id="session-a",
        floor_id="floor-01",
        submap_id="submap-a",
        calibration_id="PREVIEW_UNCALIBRATED",
    )

    restored = LiveSurfaceMap.load_checkpoint(
        tmp_path,
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
        expected_building_id="building-a",
        expected_session_id="session-a",
        expected_floor_id="floor-01",
        expected_submap_id="submap-a",
        expected_calibration_id="PREVIEW_UNCALIBRATED",
    )
    assert restored.last_integrated_identity == surface.last_integrated_identity
    with pytest.raises(ValueError, match="session provenance"):
        LiveSurfaceMap.load_checkpoint(
            tmp_path,
            expected_map_id="map-a",
            expected_source_config_sha256=SOURCE_HASH,
            expected_building_id="building-a",
            expected_session_id="session-other",
            expected_floor_id="floor-01",
            expected_submap_id="submap-a",
            expected_calibration_id="PREVIEW_UNCALIBRATED",
        )


def test_surface_rejects_cross_session_mutation(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, 0.0, session_id="session-a"))

    with pytest.raises(ValueError, match="session"):
        surface.integrate(_frame(2, 0.0, session_id="session-b"))


def test_bound_checkpoint_requires_exact_calibration(tmp_path):
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, 0.0, calibration_id="cal-a"))
    surface.checkpoint(
        tmp_path,
        building_id="building-a",
        session_id="session-a",
        floor_id="floor-01",
        submap_id="submap-a",
        calibration_id="cal-a",
    )

    with pytest.raises(ValueError, match="calibration"):
        LiveSurfaceMap.load_checkpoint(
            tmp_path,
            expected_map_id="map-a",
            expected_source_config_sha256=SOURCE_HASH,
            expected_building_id="building-a",
            expected_session_id="session-a",
            expected_floor_id="floor-01",
            expected_submap_id="submap-a",
            expected_calibration_id="cal-b",
        )


def test_pose_revision_retains_frames_older_than_bounded_dedupe_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveSurfaceMap, "JOURNAL_LIMIT", 4)
    left = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    right = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    left.integrate(_frame(0, 0.0))
    right.integrate(_frame(0, 0.01))
    for sequence in range(1, 5):
        frame = _frame(sequence, 0.02)
        left.integrate(frame)
        right.integrate(frame)

    left_checkpoint = left.checkpoint(tmp_path / "left")
    right_checkpoint = right.checkpoint(tmp_path / "right")

    assert left_checkpoint.pose_revision_id != right_checkpoint.pose_revision_id
    left_restored = LiveSurfaceMap.load_checkpoint(
        tmp_path / "left",
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )
    right_restored = LiveSurfaceMap.load_checkpoint(
        tmp_path / "right",
        expected_map_id="map-a",
        expected_source_config_sha256=SOURCE_HASH,
    )
    next_frame = _frame(5, 0.03)
    left_restored.integrate(next_frame)
    right_restored.integrate(next_frame)

    assert left_restored.pose_revision_id != right_restored.pose_revision_id
