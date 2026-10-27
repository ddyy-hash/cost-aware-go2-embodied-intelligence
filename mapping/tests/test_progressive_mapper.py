import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mapping" / "pc"))

from go2_mapping_live.progressive_mapper import Keyframe, KeyframeError, ProgressiveMapper
from go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def frame(sequence=1, pose=None):
    return Keyframe(
        map_id="indoor-a", sequence=sequence, timestamp_ns=1_785_253_276_000_000_000 + sequence,
        calibration_id="cal-v1", points_xyz=np.array([[0, 0, 0], [1, 0, 0]] * 60, dtype=np.float32),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0), pose_map_from_lidar=np.eye(4) if pose is None else pose,
    )


def test_preview_map_integrates_only_verified_pose_and_never_claims_survey_grade():
    mapper = ProgressiveMapper(voxel_m=0.05)

    assert mapper.integrate(frame(), residual_m=0.03, calibrated_lio=False) is True
    state = mapper.snapshot()
    assert state["quality"] == "PREVIEW_UNCALIBRATED"
    assert state["points"] >= 2
    assert state["residual_cm"] == 3.0
    assert mapper.integrate(frame(), residual_m=0.03, calibrated_lio=False) is False


def test_mapper_rejects_unposed_or_high_residual_keyframes():
    mapper = ProgressiveMapper()
    with pytest.raises(KeyframeError, match="no verified"):
        mapper.integrate(frame(pose=None)._replace() if False else Keyframe(**{**frame().__dict__, "pose_map_from_lidar": None}), 0.01, False)
    with pytest.raises(KeyframeError, match="residual"):
        mapper.integrate(frame(), 0.21, False)


def test_mapper_refuses_map_mixing_and_unverified_promotion():
    mapper = ProgressiveMapper()
    assert mapper.integrate(frame(), 0.03, False)

    with pytest.raises(KeyframeError, match="map_id"):
        mapper.integrate(Keyframe(**{**frame(2).__dict__, "map_id": "indoor-b"}), 0.03, False)
    with pytest.raises(KeyframeError, match="unverified"):
        mapper.integrate(
            Keyframe(**{**frame(2).__dict__, "calibration_id": "PROVISIONAL_MID360_UNVERIFIED"}),
            0.03,
            True,
        )


def test_mapper_exposes_lidar_heading_for_the_live_3d_robot_marker():
    mapper = ProgressiveMapper()
    pose = np.eye(4)
    pose[:3, :3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    assert mapper.integrate(frame(pose=pose), residual_m=None, calibrated_lio=False)

    assert mapper.snapshot()["heading_xy"] == [0.0, 1.0]


def test_live_room_scan_accumulates_global_extent_and_reports_staleness():
    mapper = ProgressiveMapper(voxel_m=0.02, preview_point_budget=20_000)
    first = frame(sequence=1)
    second_pose = np.eye(4)
    second_pose[:3, 3] = [2.0, 0.0, 0.0]
    second = frame(sequence=2, pose=second_pose)

    assert mapper.integrate(first, residual_m=0.03, calibrated_lio=False)
    assert mapper.integrate(second, residual_m=0.03, calibrated_lio=False)

    state = mapper.snapshot()
    scan = state["room_scan"]
    assert scan["state"] == "SCANNING"
    assert scan["frames"] == 2
    assert scan["source_points"] == 240
    assert scan["voxel_size_m"] == 0.02
    assert scan["first_timestamp_ns"] == first.timestamp_ns
    assert scan["last_timestamp_ns"] == second.timestamp_ns
    assert scan["extent_m"][0] >= 3.0
    assert scan["trajectory_length_m"] == pytest.approx(2.0)

    stale = mapper.room_scan_snapshot(now=mapper._last_update_monotonic + 16.0)
    assert stale["state"] == "STALE"
    assert stale["live"] is False


def test_live_room_scan_display_points_never_exceed_the_configured_budget():
    mapper = ProgressiveMapper(voxel_m=0.01, preview_point_budget=100)
    points = np.column_stack(
        (
            np.arange(199, dtype=np.float32) * 0.02,
            np.zeros(199, dtype=np.float32),
            np.zeros(199, dtype=np.float32),
        )
    )
    keyframe = Keyframe(**{**frame().__dict__, "points_xyz": points})

    assert mapper.integrate(keyframe, residual_m=0.03, calibrated_lio=False)

    assert mapper.snapshot()["points"] == 199
    assert len(mapper.snapshot()["preview_points"]) <= 100


def test_mapper_rectifies_native_sensor_points_before_voxelizing_them():
    mapper = ProgressiveMapper(voxel_m=0.01, preview_point_budget=100)
    source = frame(1)
    corrected = Keyframe(
        **{
            **source.__dict__,
            "points_xyz": np.asarray([[0.0, 0.0, 0.5]] * 120, dtype=np.float32),
            "points_in_map_frame": False,
            "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        }
    )

    assert mapper.integrate(corrected, 0.01, False) is True
    state = mapper.snapshot()
    assert any(point[2] == pytest.approx(-0.5) for point in state["preview_points"])
    assert state["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert state["room_scan"]["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert mapper.telemetry_snapshot()["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def test_mapper_rejects_unknown_or_mixed_sensor_frame_transforms():
    mapper = ProgressiveMapper()
    first = Keyframe(
        **{
            **frame(1).__dict__,
            "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        }
    )
    assert mapper.integrate(first, 0.01, False) is True

    unknown = Keyframe(**{**frame(2).__dict__, "sensor_frame_transform_id": "UNKNOWN_FRAME"})
    with pytest.raises(KeyframeError, match="sensor frame transform"):
        mapper.integrate(unknown, 0.01, False)

    legacy_identity = frame(2)
    with pytest.raises(KeyframeError, match="sensor frame transform"):
        mapper.integrate(legacy_identity, 0.01, False)
