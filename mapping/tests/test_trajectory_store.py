import hashlib

import numpy as np
import pytest

from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe
from mapping.pc.go2_mapping_live.trajectory_store import TrajectoryStore
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import build_live_keyframe_message


def _frame(sequence: int, *, x: float = 0.0):
    pose = np.eye(4)
    pose[0, 3] = x
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="go2-live",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=1_000_000_000 + sequence,
        calibration_id="calibration-a",
        source_config_sha256="a" * 64,
        points_xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lidar",
        motion_compensated=True,
        points_in_map_frame=False,
    )
    return validate_live_keyframe(encoded.payload)


def test_trajectory_store_is_idempotent_conflict_checked_and_filterable(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectories.sqlite3")

    assert store.append(_frame(0, x=0.0), floor_id="floor-01", submap_id="submap-a") is True
    assert store.append(_frame(0, x=0.0), floor_id="floor-01", submap_id="submap-a") is False
    assert store.append(_frame(1, x=0.5), floor_id="floor-02", submap_id="submap-b") is True
    with pytest.raises(ValueError, match="conflict"):
        store.append(_frame(0, x=2.0), floor_id="floor-01", submap_id="submap-a")

    all_paths = store.query(building_id="building-a")
    floor_one = store.query(building_id="building-a", floor_id="floor-01", session_id="session-a")

    assert all_paths["schema_version"] == 1
    assert len(all_paths["trajectories"]) == 2
    assert floor_one["filters"] == {"floor_id": "floor-01", "session_id": "session-a"}
    assert floor_one["trajectories"][0]["samples"][0]["position_m"] == [0.0, 0.0, 0.0]
    assert len(floor_one["revision_id"]) == 64
    assert floor_one["revision_id"] == hashlib.sha256(
        TrajectoryStore.canonical_revision_payload(floor_one).encode("utf-8")
    ).hexdigest()


def test_trajectory_store_resolves_texture_pose_to_exact_nearest_submap(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectories.sqlite3")
    store.append(_frame(0), floor_id="floor-01", submap_id="submap-a")
    store.append(_frame(200_000_000), floor_id="floor-02", submap_id="submap-b")

    first = store.resolve_submap_near(
        building_id="building-a",
        session_id="session-a",
        map_id="go2-live",
        producer_instance_id="producer-a",
        timestamp_ns=1_020_000_000,
        maximum_skew_ns=30_000_000,
    )
    second = store.resolve_submap_near(
        building_id="building-a",
        session_id="session-a",
        map_id="go2-live",
        producer_instance_id="producer-a",
        timestamp_ns=1_180_000_000,
        maximum_skew_ns=30_000_000,
    )

    assert first == {
        "submap_id": "submap-a", "floor_id": "floor-01", "sequence": 0,
        "timestamp_ns": 1_000_000_000, "skew_ns": 20_000_000,
    }
    assert second["submap_id"] == "submap-b"
    with pytest.raises(ValueError, match="maximum skew"):
        store.resolve_submap_near(
            building_id="building-a",
            session_id="session-a",
            map_id="go2-live",
            producer_instance_id="producer-a",
            timestamp_ns=1_100_000_000,
            maximum_skew_ns=30_000_000,
        )


def test_building_trajectory_api_uses_explicit_filters(monkeypatch, tmp_path):
    store = TrajectoryStore(tmp_path / "trajectories.sqlite3")
    store.append(_frame(0), floor_id="floor-01", submap_id="submap-a")
    monkeypatch.setattr(server, "_trajectory_store", store)

    response = server.building_trajectories(
        building_id="building-a",
        floor_id="floor-01",
        session_id="session-a",
    )

    assert response["building_id"] == "building-a"
    assert response["filters"]["floor_id"] == "floor-01"
    assert response["filters"]["session_id"] == "session-a"
    assert response["trajectories"][0]["session_id"] == "session-a"
