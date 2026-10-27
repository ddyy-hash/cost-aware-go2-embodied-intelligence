import sqlite3
import struct
import hashlib

import numpy as np
import pytest
import zstandard
import pytest

from mapping.pc.go2_mapping_appearance.identity import measured_surface_identity
from mapping.pc.go2_mapping_live.high_fidelity_mapper import (
    HighFidelityMapper,
    RECORDED_LIO_SESSION_ODOM_CLOUD,
)
from mapping.pc.go2_mapping_live.high_fidelity_mapper import _batch_world_and_voxels


def test_batch_world_and_voxels_has_cpu_gpu_equivalent_contract():
    points = np.asarray([[0.01, 0.02, 0.03], [0.11, 0.02, 0.03]], dtype=np.float32)
    poses = np.asarray([np.eye(4), np.eye(4)], dtype=np.float64)
    cpu_world, cpu_indices = _batch_world_and_voxels(points, poses, 0.1, use_cuda=False)
    assert cpu_world.shape == (2, 3)
    assert np.array_equal(cpu_indices, np.asarray([[0, 0, 0], [1, 0, 0]]))
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gpu_world, gpu_indices = _batch_world_and_voxels(points, poses, 0.1, use_cuda=True)
    assert np.allclose(gpu_world, cpu_world, atol=1e-5)
    assert np.array_equal(gpu_indices, cpu_indices)
from mapping.pc.go2_mapping_live.progressive_ingest import VerifiedArtifactIngestor
from mapping.pc.go2_mapping_live.sensor_frame import (
    IDENTITY_SENSOR_FRAME_V1,
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
)
from mapping.pc.go2_mapping_fusion import VoxelEvidence, VoxelState
from mapping.pc.go2_mapping_core.pose_graph import PoseGraphSolution
from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config


def _align(buffer, alignment):
    buffer.extend(b"\0" * ((-len(buffer)) % alignment))


def _u32(buffer, value):
    _align(buffer, 4)
    buffer.extend(struct.pack("<I", value))


def _i32(buffer, value):
    _align(buffer, 4)
    buffer.extend(struct.pack("<i", value))


def _f64(buffer, value):
    _align(buffer, 8)
    buffer.extend(struct.pack("<d", value))


def _string(buffer, value):
    encoded = value.encode("utf-8") + b"\0"
    _u32(buffer, len(encoded))
    buffer.extend(encoded)


def _header(buffer, timestamp_ns, frame_id="livox_frame"):
    _i32(buffer, timestamp_ns // 1_000_000_000)
    _u32(buffer, timestamp_ns % 1_000_000_000)
    _string(buffer, frame_id)


def _pointcloud_cdr(timestamp_ns, points, *, frame_id="livox_frame"):
    payload = bytearray(b"\0\1\0\0")
    body = bytearray()
    _header(body, timestamp_ns, frame_id)
    _u32(body, 1)
    _u32(body, len(points))
    _u32(body, 3)
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        _string(body, name)
        _u32(body, offset)
        body.extend(b"\x07")
        _u32(body, 1)
    body.extend(b"\0")
    _u32(body, 12)
    _u32(body, 12 * len(points))
    raw_points = b"".join(struct.pack("<fff", *point) for point in points)
    _u32(body, len(raw_points))
    body.extend(raw_points)
    body.extend(b"\x01")
    payload.extend(body)
    return bytes(payload)


def _pose_cdr(timestamp_ns, translation):
    payload = bytearray(b"\0\1\0\0")
    body = bytearray()
    _header(body, timestamp_ns)
    for value in (*translation, 0.0, 0.0, 0.0, 1.0):
        _f64(body, value)
    for _ in range(36):
        _f64(body, 0.0)
    payload.extend(body)
    return bytes(payload)


def _make_bag(path, *, lidar_header_timestamp=None, pose_header_timestamp=None):
    timestamp = 12_345_000_000
    lidar_header_timestamp = timestamp if lidar_header_timestamp is None else lidar_header_timestamp
    pose_header_timestamp = timestamp if pose_header_timestamp is None else pose_header_timestamp
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.execute("INSERT INTO topics VALUES (1, '/livox/lidar', 'sensor_msgs/msg/PointCloud2')")
    database.execute("INSERT INTO topics VALUES (2, '/go2/lio/pose', 'geometry_msgs/msg/PoseWithCovarianceStamped')")
    database.execute(
        "INSERT INTO messages VALUES (1, 2, ?, ?)",
        (timestamp, _pose_cdr(pose_header_timestamp, (1.0, 2.0, 0.5))),
    )
    database.execute(
        "INSERT INTO messages VALUES (2, 1, ?, ?)",
        (
            timestamp,
            _pointcloud_cdr(lidar_header_timestamp, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))),
        ),
    )
    database.commit()
    database.close()


def _make_multi_frame_bag(path, poses, clouds, *, start_index=0):
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.execute("INSERT INTO topics VALUES (1, '/livox/lidar', 'sensor_msgs/msg/PointCloud2')")
    database.execute("INSERT INTO topics VALUES (2, '/go2/lio/pose', 'geometry_msgs/msg/PoseWithCovarianceStamped')")
    message_id = 1
    for offset, (pose, cloud) in enumerate(zip(poses, clouds, strict=True)):
        frame_index = start_index + offset
        timestamp = 20_000_000_000 + frame_index * 100_000_000
        database.execute(
            "INSERT INTO messages VALUES (?, 2, ?, ?)",
            (message_id, timestamp, _pose_cdr(timestamp, pose)),
        )
        message_id += 1
        database.execute(
            "INSERT INTO messages VALUES (?, 1, ?, ?)",
            (message_id, timestamp, _pointcloud_cdr(timestamp, cloud)),
        )
        message_id += 1
    database.commit()
    database.close()


def _make_lidar_only_bag(path):
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.execute("INSERT INTO topics VALUES (1, '/livox/lidar', 'sensor_msgs/msg/PointCloud2')")
    database.execute(
        "INSERT INTO messages VALUES (1, 1, ?, ?)",
        (12_345_000_000, _pointcloud_cdr(12_345_000_000, ((0.0, 0.0, 0.0),))),
    )
    database.commit()
    database.close()




def _make_registered_cloud_bag(path, clouds, *, frame_id="map"):
    """Write global-map LIO-SAM clouds without a pose topic.

    The production recording supplies these on
    ``/go2/mapping/map_cloud`` only after the project publisher has placed
    them in its global ``map`` frame. A replay must not apply a sensor-frame
    correction or require a second pose to use this evidence.
    """

    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.execute(
        "INSERT INTO topics VALUES (1, '/go2/mapping/map_cloud', 'sensor_msgs/msg/PointCloud2')"
    )
    for index, points in enumerate(clouds, 1):
        timestamp = 30_000_000_000 + index * 100_000_000
        database.execute(
            "INSERT INTO messages VALUES (?, 1, ?, ?)",
            (index, timestamp, _pointcloud_cdr(timestamp, points, frame_id=frame_id)),
        )
    database.commit()
    database.close()


def _make_session_odom_cloud_bag(path, clouds, *, frame_ids=("odom",)):
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.execute(
        "INSERT INTO topics VALUES (1, '/lio_sam_ros2/mapping/cloud_registered', 'sensor_msgs/msg/PointCloud2')"
    )
    for index, points in enumerate(clouds, 1):
        timestamp = 40_000_000_000 + index * 100_000_000
        database.execute(
            "INSERT INTO messages VALUES (?, 1, ?, ?)",
            (index, timestamp, _pointcloud_cdr(timestamp, points, frame_id=frame_ids[(index - 1) % len(frame_ids)])),
        )
    database.commit()
    database.close()


def _synthetic_closed_loop():
    true_poses = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (3.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    drift = np.column_stack((np.linspace(0.0, 0.10, len(true_poses)), np.zeros((len(true_poses), 2))))
    measured_poses = true_poses + drift
    rng = np.random.default_rng(20260802)
    world_landmarks = rng.uniform((1.0, 0.8, 0.3), (2.2, 1.8, 1.4), size=(96, 3))
    clouds = [world_landmarks - pose for pose in true_poses]
    final_marker = np.asarray((5.0, 4.0, 2.0), dtype=np.float64)
    clouds[-1] = np.vstack((clouds[-1], final_marker))
    return measured_poses, clouds, final_marker


def _sorted_points(snapshot):
    return np.asarray(sorted(snapshot["preview_points"]), dtype=np.float64)


def test_high_fidelity_mapper_fuses_full_lidar_frames_only_after_matching_lio_pose(tmp_path):
    bag = tmp_path / "segment-000.db3"
    _make_bag(bag)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100)

    first = mapper.ingest_sqlite_bag(bag)
    duplicate = mapper.ingest_sqlite_bag(bag)
    state = mapper.snapshot()

    assert first["integrated_frames"] == 1
    assert first["integrated_points"] == 3
    assert duplicate["duplicate"] is True
    assert state["points"] == 3
    assert state["artifacts"] == 1
    assert state["quality"] == "DENSE_LIDAR_PREVIEW_UNCALIBRATED"
    assert np.allclose(sorted(state["preview_points"])[0], [1.0, 2.0, 0.5])


def test_high_fidelity_mapper_pairs_lidar_and_lio_by_the_shared_rosbag_record_time(tmp_path):
    bag = tmp_path / "segment-clock-domains.db3"
    _make_bag(
        bag,
        lidar_header_timestamp=12_345_000_000 - 1_555_000_000_000,
        pose_header_timestamp=12_345_000_000 - 900_000_000,
    )
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100)

    result = mapper.ingest_sqlite_bag(bag)

    assert result["integrated_frames"] == 1
    assert result["skipped_pose_frames"] == 0
    assert mapper.snapshot()["points"] == 3


def test_high_fidelity_mapper_records_a_part_without_a_matching_pose_and_continues(tmp_path):
    bag = tmp_path / "startup-without-pose.db3"
    _make_lidar_only_bag(bag)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100)

    result = mapper.ingest_sqlite_bag(bag)
    state = mapper.snapshot()

    assert result == {
        "duplicate": False,
        "integrated_frames": 0,
        "integrated_points": 0,
        "skipped_pose_frames": 1,
        "decode_errors": 0,
    }
    assert state["artifacts"] == 1
    assert state["frames"] == 0
    assert state["skipped_pose_frames"] == 1
    assert mapper.measured_surface_snapshot() is None


def test_high_fidelity_mapper_replays_recorded_global_lio_clouds_in_their_map_frame(tmp_path):
    """A room-scale registered cloud must not be reprojected as raw LiDAR."""

    bag = tmp_path / "global-map-room.db3"
    room_patch = np.asarray(
        (
            (3.00, -2.00, -1.00),
            (3.03, -2.00, -1.00),
            (3.00, -1.97, -1.00),
        ),
        dtype=np.float64,
    )
    _make_registered_cloud_bag(bag, [room_patch, room_patch, room_patch])
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        source_mode="RECORDED_LIO_REGISTERED_CLOUD",
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )

    result = mapper.ingest_sqlite_bag(bag)
    mapper.finalize_deferred_batch()
    state = mapper.snapshot()
    surface = mapper.measured_surface_snapshot()

    assert result["integrated_frames"] == 3
    assert result["skipped_pose_frames"] == 0
    assert state["mapping_source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert state["pose_graph_optimization"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert surface is not None
    assert surface["mapping_source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert surface["source_layer"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert surface["point_authority"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert surface["sensor_frame_transform_applied"] is False
    positions = np.asarray(surface["positions"], dtype=np.float64).reshape(-1, 3)
    assert any(np.allclose(point, room_patch[0]) for point in positions)


def test_high_fidelity_mapper_rejects_local_odom_registered_clouds_as_a_global_replay_source(tmp_path):
    """A LIO local-map cloud must not be relabelled as a global room map."""

    bag = tmp_path / "registered-local-odom.db3"
    _make_registered_cloud_bag(
        bag,
        [np.asarray(((0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (0.0, 0.03, 0.0)))],
        frame_id="odom",
    )
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        source_mode="RECORDED_LIO_REGISTERED_CLOUD",
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )

    mapper.ingest_sqlite_bag(bag)

    with pytest.raises(ValueError, match="expected global frame 'map'.*observed 'odom'"):
        mapper.finalize_deferred_batch()

    assert mapper.measured_surface_snapshot() is None


@pytest.mark.parametrize("frame_ids", [("map",), ("",), ("odom", "map")])
def test_high_fidelity_mapper_rejects_non_odom_clouds_for_the_local_session_source(tmp_path, frame_ids):
    bag = tmp_path / "session-odom-invalid.db3"
    cloud = np.asarray(((0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (0.0, 0.03, 0.0)))
    _make_session_odom_cloud_bag(bag, [cloud, cloud], frame_ids=frame_ids)
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        source_mode=RECORDED_LIO_SESSION_ODOM_CLOUD,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )

    mapper.ingest_sqlite_bag(bag)
    if frame_ids == ("",):
        assert mapper._recorded_cloud_frame_contract_violations == {"<unreadable-or-empty>"}

    with pytest.raises(ValueError, match="local session replay expected frame 'odom'"):
        mapper.finalize_deferred_batch(global_optimization=False)
    assert mapper.measured_surface_snapshot() is None


def test_high_fidelity_mapper_builds_an_observed_local_surface_only_from_registered_odom_clouds(tmp_path):
    bag = tmp_path / "session-odom-wall-floor.db3"
    wall_floor = np.asarray(
        ((0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (0.0, 0.03, 0.0), (0.03, 0.03, 0.0)),
        dtype=np.float64,
    )
    _make_session_odom_cloud_bag(bag, [wall_floor, wall_floor])
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        source_mode=RECORDED_LIO_SESSION_ODOM_CLOUD,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )

    result = mapper.ingest_sqlite_bag(bag)
    mapper.finalize_deferred_batch(global_optimization=False)
    state = mapper.snapshot()
    surface = mapper.measured_surface_snapshot()

    assert result["integrated_frames"] == 2
    assert state["mapping_source_mode"] == RECORDED_LIO_SESSION_ODOM_CLOUD
    assert state["recorded_coordinate_frame_ids"] == ["odom"]
    assert state["sensor_frame_transform_applied"] is False
    assert surface is not None
    assert surface["mapping_source_mode"] == RECORDED_LIO_SESSION_ODOM_CLOUD
    assert surface["recorded_coordinate_frame_ids"] == ["odom"]
    assert surface["sensor_frame_transform_applied"] is False


def test_high_fidelity_mapper_applies_the_versioned_sensor_frame_transform_to_every_preview_point(tmp_path):
    bag = tmp_path / "sensor-frame.db3"
    _make_bag(bag)
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )

    result = mapper.ingest_sqlite_bag(bag)
    state = mapper.snapshot()

    assert result["integrated_frames"] == 1
    assert state["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert any(np.allclose(point, [1.0, 3.0, -0.5]) for point in state["preview_points"])
    assert state["pose_generation"] != "live-lio-record-v1"


def test_high_fidelity_mapper_can_defer_surface_prewarm_for_offline_batch_ingest(tmp_path, monkeypatch):
    bag = tmp_path / "batch-part.db3"
    _make_bag(bag)
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        prewarm_surface_after_ingest=False,
    )
    prewarm_calls = []
    monkeypatch.setattr(mapper, "_prewarm_measured_surface", lambda: prewarm_calls.append(True))

    mapper.ingest_sqlite_bag(bag)

    assert prewarm_calls == []


def test_high_fidelity_mapper_defers_pose_optimization_until_offline_batch_finalize(tmp_path, monkeypatch):
    first_bag = tmp_path / "batch-0.db3"
    second_bag = tmp_path / "batch-1.db3"
    poses = np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
    cloud = np.asarray(((0.5, 0.0, 0.0), (0.5, 0.1, 0.0), (0.5, 0.0, 0.1)))
    _make_multi_frame_bag(first_bag, poses[:1], [cloud], start_index=0)
    _make_multi_frame_bag(second_bag, poses[1:], [cloud], start_index=1)
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )
    optimize_calls = 0
    original_optimize = mapper._pose_graph.optimize

    def counted_optimize():
        nonlocal optimize_calls
        optimize_calls += 1
        return original_optimize()

    monkeypatch.setattr(mapper._pose_graph, "optimize", counted_optimize)

    mapper.ingest_sqlite_bag(first_bag)
    mapper.ingest_sqlite_bag(second_bag)

    assert optimize_calls == 0
    assert mapper.snapshot()["reconstruction_phase"] == "DEFERRED_BATCH_PENDING"
    assert mapper.snapshot()["points"] == 0

    mapper.finalize_deferred_batch()
    finalized = mapper.snapshot()

    assert optimize_calls == 1
    assert finalized["reconstruction_phase"] == "IDLE"
    assert finalized["frames"] == 2
    assert finalized["points"] > 0
    assert len(finalized["pose_generation"]) == 64


def test_high_fidelity_mapper_can_finalize_a_large_batch_from_recorded_lio_poses(tmp_path, monkeypatch):
    """Observed-only preview fallback must not wait on an unbounded global solve."""

    bag = tmp_path / "large-batch.db3"
    poses = np.asarray(tuple((index * 0.1, 0.0, 0.0) for index in range(7)))
    world_patch = np.asarray(
        [
            (1.0, x, y)
            for x in np.linspace(-0.04, 0.04, 5)
            for y in np.linspace(-0.04, 0.04, 5)
        ],
        dtype=np.float64,
    )
    clouds = [world_patch - pose for pose in poses]
    _make_multi_frame_bag(bag, poses, clouds)
    mapper = HighFidelityMapper(
        voxel_m=0.01,
        display_point_budget=100,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )
    mapper.ingest_sqlite_bag(bag)
    monkeypatch.setattr(mapper._pose_graph, "optimize", lambda: (_ for _ in ()).throw(AssertionError("not used")))

    mapper.finalize_deferred_batch(global_optimization=False)

    state = mapper.snapshot()
    surface = mapper.measured_surface_snapshot()
    assert state["reconstruction_phase"] == "IDLE"
    assert state["pose_graph_optimization"] == "RECORDED_LIO_POSES_ONLY"
    assert state["accepted_loop_closures"] == 0
    assert surface is not None
    assert surface["source_layer"] == "LIDAR_MEASURED"
    assert surface["geometry_authority"] is False


def test_rgbd_refinement_uses_initial_pose_when_prior_solution_does_not_include_new_frame(tmp_path):
    class RecordingRefiner:
        def __init__(self):
            self.lidar_frames = None

        def process_sqlite_bag(self, path, *, artifact_sha256, lidar_frames, measured_generation):
            self.lidar_frames = tuple(lidar_frames)

    bag = tmp_path / "new-frame-after-optimization.db3"
    _make_bag(bag)
    refiner = RecordingRefiner()
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100, rgbd_refiner=refiner)
    mapper._last_pose_graph_solution = PoseGraphSolution({}, "pg-prior", 1.0, 0.5, 1, 2)

    result = mapper.ingest_sqlite_bag(bag)

    assert result["integrated_frames"] == 1
    assert refiner.lidar_frames is not None
    assert refiner.lidar_frames[0]["pose_map_from_lidar"] == pytest.approx(
        mapper._raw_frames[-1].initial_pose_map_from_lidar
    )


def test_real_icp_closure_optimizes_generation_and_reintegrates_raw_lidar_in_node_order(tmp_path):
    poses, clouds, final_marker = _synthetic_closed_loop()
    bag = tmp_path / "synthetic-loop.db3"
    _make_multi_frame_bag(bag, poses, clouds)
    mapper = HighFidelityMapper(voxel_m=0.005, display_point_budget=5_000, pose_generation="pg-live")

    mapper.ingest_sqlite_bag(bag)

    state = mapper.snapshot()
    frames = mapper._raw_frames
    lio_edges = [edge for edge in mapper._pose_graph.edges if edge.kind == "lio"]
    loop_edges = [edge for edge in mapper._pose_graph.edges if edge.kind == "verified_loop"]
    assert [(edge.source, edge.target) for edge in lio_edges] == [
        (frames[index].node_id, frames[index + 1].node_id) for index in range(10)
    ]
    assert [(edge.source, edge.target) for edge in loop_edges] == [(frames[0].node_id, frames[10].node_id)]
    assert state["accepted_loop_closures"] == 1
    assert state["optimized_pose_generation"] != "pg-live"
    assert state["pose_generation"] == state["optimized_pose_generation"]

    optimized_pose = mapper._last_pose_graph_solution.poses_map_from_keyframe[frames[-1].node_id]
    optimized_marker = final_marker @ optimized_pose[:3, :3].T + optimized_pose[:3, 3]
    unoptimized_marker = final_marker + poses[-1]
    published = _sorted_points(state)
    assert np.min(np.linalg.norm(published - optimized_marker, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(published - unoptimized_marker, axis=1)) > 0.02


def test_measured_surface_snapshot_binds_source_artifact_and_idle_generation(tmp_path, monkeypatch):
    bag = tmp_path / "surface-source.db3"
    _make_bag(bag)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100, pose_generation="pg-surface")
    mapper.ingest_sqlite_bag(bag)
    evidence = tuple(
        VoxelEvidence(
            key=(index, 0, 0),
            position=np.asarray(position, dtype=np.float64),
            observation_count=3,
            independent_views=2,
            temporal_support_s=1.0,
            principal_stddev_m=0.001,
            point_to_plane_p95_m=0.001,
            static_probability=1.0,
            state=VoxelState.MEASURED,
            pose_generation="pg-surface",
            source_artifact_sha256=(hashlib.sha256(bag.read_bytes()).hexdigest(),),
        )
        for index, position in enumerate(((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)))
    )
    evidence_calls = 0

    def measured_evidence(state):
        nonlocal evidence_calls
        evidence_calls += 1
        assert state is VoxelState.MEASURED
        return evidence

    mapper._pose_generation = "pg-surface"
    mapper._surface_cache_key = None
    monkeypatch.setattr(mapper._measured_map, "evidence", measured_evidence)

    surface = mapper.measured_surface_snapshot()
    repeated = mapper.measured_surface_snapshot()

    assert surface is not None
    assert repeated is not None
    assert surface["artifact_sha256"] == [hashlib.sha256(bag.read_bytes()).hexdigest()]
    assert surface["reconstruction_phase"] == "IDLE"
    assert surface["sensor_frame_transform_id"] == IDENTITY_SENSOR_FRAME_V1
    assert surface["geometry_authority"] is False
    assert surface["quality"] == "PREVIEW_UNCALIBRATED"
    assert surface["source_vertex_ids"] == ["0:0:0", "1:0:0", "2:0:0"]
    assert surface["measured_surface_geometry_revision_id"] is None
    assert surface["appearance_blocker"] == "POSE_GENERATION_NOT_CONTENT_ADDRESSED"
    assert repeated["positions"] == surface["positions"]
    assert evidence_calls == 1

    mapper._pose_generation = "d" * 64
    mapper._surface_cache_key = None
    canonical = mapper.measured_surface_snapshot()
    assert canonical is not None
    expected = measured_surface_identity(
        positions=np.asarray(canonical["positions"]).reshape(-1, 3),
        normals=np.asarray(canonical["normals"]).reshape(-1, 3),
        triangles=np.asarray(canonical["triangles"]).reshape(-1, 3),
        pose_generation="d" * 64,
        artifact_sha256=canonical["artifact_sha256"],
        fusion_config_sha256=load_reconstruction_config().fusion_sha256,
    )
    assert canonical["measured_surface_geometry_revision_id"] == expected.revision_id
    assert canonical["appearance_blocker"] is None


def test_closed_artifact_publishes_content_addressed_pose_without_a_loop(tmp_path):
    bag = tmp_path / "canonical-open-trajectory.db3"
    _make_bag(bag)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100, pose_generation="live-lio-record-v1")

    mapper.ingest_sqlite_bag(bag)

    generation = mapper.snapshot()["optimized_pose_generation"]
    assert len(generation) == 64
    assert set(generation) <= set("0123456789abcdef")
    assert mapper._last_pose_graph_solution is not None


def test_completed_artifact_prewarms_surface_and_optimization_serves_last_complete_snapshot(tmp_path, monkeypatch):
    bag = tmp_path / "prewarmed-surface.db3"
    _make_bag(bag)
    digest = hashlib.sha256(bag.read_bytes()).hexdigest()
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=100, pose_generation="pg-prewarm")
    evidence = tuple(
        VoxelEvidence(
            key=(index, 0, 0),
            position=np.asarray(position, dtype=np.float64),
            observation_count=3,
            independent_views=2,
            temporal_support_s=1.0,
            principal_stddev_m=0.001,
            point_to_plane_p95_m=0.001,
            static_probability=1.0,
            state=VoxelState.MEASURED,
            pose_generation="pg-prewarm",
            source_artifact_sha256=(digest,),
        )
        for index, position in enumerate(((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)))
    )
    monkeypatch.setattr(mapper._measured_map, "evidence", lambda state: evidence)

    mapper.ingest_sqlite_bag(bag)

    assert mapper._surface_cache_snapshot is not None
    completed = mapper.measured_surface_snapshot()
    mapper._reconstruction_phase = "OPTIMIZING_AND_REINTEGRATING"
    mapper._artifact_digests.add("b" * 64)
    monkeypatch.setattr(
        mapper._measured_map,
        "evidence",
        lambda state: (_ for _ in ()).throw(AssertionError("optimization must serve the last complete surface")),
    )

    during_optimization = mapper.measured_surface_snapshot()

    assert during_optimization == completed
    assert during_optimization["reconstruction_phase"] == "IDLE"
    assert during_optimization["artifact_sha256"] == [digest]


@pytest.mark.parametrize(
    ("fitness", "rmse"),
    ((0.49, 0.01), (0.90, 0.031)),
    ids=("fitness-below-gate", "rmse-above-gate"),
)
def test_icp_quality_gate_does_not_publish_a_pose_generation(tmp_path, monkeypatch, fitness, rmse):
    poses, clouds, _ = _synthetic_closed_loop()
    bag = tmp_path / f"rejected-{fitness}-{rmse}.db3"
    _make_multi_frame_bag(bag, poses, clouds)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    monkeypatch.setattr(
        HighFidelityMapper,
        "_icp_relative",
        staticmethod(lambda source, target, initial: (np.array(initial, copy=True), fitness, rmse)),
    )

    mapper.ingest_sqlite_bag(bag)

    state = mapper.snapshot()
    assert len(state["optimized_pose_generation"]) == 64
    assert state["accepted_loop_closures"] == 0
    assert state["pose_graph"] is not None


def test_icp_correction_gate_does_not_publish_a_pose_generation(tmp_path, monkeypatch):
    poses, clouds, _ = _synthetic_closed_loop()
    bag = tmp_path / "correction-rejected.db3"
    _make_multi_frame_bag(bag, poses, clouds)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    excessive_correction = np.eye(4)
    excessive_correction[0, 3] = 0.50
    monkeypatch.setattr(
        HighFidelityMapper,
        "_icp_relative",
        staticmethod(lambda source, target, initial: (excessive_correction, 0.90, 0.01)),
    )

    mapper.ingest_sqlite_bag(bag)

    state = mapper.snapshot()
    assert len(state["optimized_pose_generation"]) == 64
    assert state["accepted_loop_closures"] == 0
    assert state["pose_graph"] is not None


def test_optimizer_failure_keeps_previous_complete_map_and_generation(tmp_path, monkeypatch):
    poses, clouds, _ = _synthetic_closed_loop()
    initial_bag = tmp_path / "open-trajectory.db3"
    closing_bag = tmp_path / "closing-frame.db3"
    _make_multi_frame_bag(initial_bag, poses[:10], clouds[:10])
    _make_multi_frame_bag(closing_bag, poses[10:], clouds[10:], start_index=10)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    mapper.ingest_sqlite_bag(initial_bag)
    previous = mapper.snapshot()
    monkeypatch.setattr(mapper._pose_graph, "optimize", lambda: (_ for _ in ()).throw(RuntimeError("no convergence")))

    result = mapper.ingest_sqlite_bag(closing_bag)

    failed = mapper.snapshot()
    assert result["integrated_frames"] == 1
    assert failed["optimized_pose_generation"] == previous["optimized_pose_generation"]
    assert failed["accepted_loop_closures"] == previous["accepted_loop_closures"]
    assert failed["points"] == previous["points"]
    assert _sorted_points(failed) == pytest.approx(_sorted_points(previous))
    assert len(mapper._raw_frames) == 11


def test_pose_reintegration_reuses_frame_local_filter_results(tmp_path, monkeypatch):
    poses, clouds, _ = _synthetic_closed_loop()
    initial_bag = tmp_path / "filtered-open-trajectory.db3"
    closing_bag = tmp_path / "filtered-closing-frame.db3"
    _make_multi_frame_bag(initial_bag, poses[:10], clouds[:10])
    _make_multi_frame_bag(closing_bag, poses[10:], clouds[10:], start_index=10)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    mapper.ingest_sqlite_bag(initial_bag)
    original_classify = mapper._frame_filter.classify
    classify_calls = 0

    def counted_classify(frame):
        nonlocal classify_calls
        classify_calls += 1
        return original_classify(frame)

    monkeypatch.setattr(mapper._frame_filter, "classify", counted_classify)

    mapper.ingest_sqlite_bag(closing_bag)

    assert mapper.snapshot()["accepted_loop_closures"] == 1
    assert classify_calls == 1


def test_raw_lidar_frames_are_disk_backed_and_sha_verified(tmp_path):
    poses, clouds, _ = _synthetic_closed_loop()
    bag = tmp_path / "immutable-raw-lidar.db3"
    _make_multi_frame_bag(bag, poses, clouds)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000)
    mapper.ingest_sqlite_bag(bag)
    frame = mapper._raw_frames[0]

    assert not hasattr(frame, "points_xyz")
    assert frame.source_artifact_path == bag.resolve()
    assert frame.cloud_index == 0
    assert mapper.snapshot()["resident_full_lidar_frames"] == 0
    assert mapper.snapshot()["raw_source_artifacts"] == 1
    assert mapper._reload_frame_points((frame,))[frame.node_id] == pytest.approx(clouds[0])

    bag.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        mapper._reload_frame_points((frame,))


def test_pose_reintegration_replays_published_zstd_artifact(tmp_path):
    poses, clouds, _ = _synthetic_closed_loop()
    source_bag = tmp_path / "source-loop.db3"
    published = tmp_path / "source-loop.db3.zstd"
    _make_multi_frame_bag(source_bag, poses, clouds)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")

    mapper.ingest_sqlite_bag(source_bag, artifact_sha256=digest, source_artifact_path=published)

    state = mapper.snapshot()
    assert state["accepted_loop_closures"] == 1
    assert state["optimized_pose_generation"] != "pg-live"
    assert state["last_pose_graph_error"] is None
    assert {frame.source_artifact_path for frame in mapper._raw_frames} == {published.resolve()}


def test_missing_raw_artifact_blocks_new_generation_and_preserves_previous_map(tmp_path):
    poses, clouds, _ = _synthetic_closed_loop()
    initial_bag = tmp_path / "open-trajectory.db3"
    closing_bag = tmp_path / "closing-frame.db3"
    _make_multi_frame_bag(initial_bag, poses[:10], clouds[:10])
    _make_multi_frame_bag(closing_bag, poses[10:], clouds[10:], start_index=10)
    mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    mapper.ingest_sqlite_bag(initial_bag)
    previous = mapper.snapshot()
    initial_bag.unlink()

    mapper.ingest_sqlite_bag(closing_bag)

    failed = mapper.snapshot()
    assert failed["optimized_pose_generation"] == previous["optimized_pose_generation"]
    assert failed["points"] == previous["points"]
    assert _sorted_points(failed) == pytest.approx(_sorted_points(previous))
    assert "FileNotFoundError" in failed["last_pose_graph_error"]


def test_restart_recovers_graph_and_map_by_replaying_verified_published_artifacts(tmp_path):
    poses, clouds, _ = _synthetic_closed_loop()
    root = tmp_path / "inbox"
    source_bag = tmp_path / "source-loop.db3"
    published = root / "lidar" / "source-loop.db3.zstd"
    _make_multi_frame_bag(source_bag, poses, clouds)
    published.parent.mkdir(parents=True)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)

    first_mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    assert VerifiedArtifactIngestor(root, first_mapper).ingest_once()["ingested_artifacts"] == 1
    first = first_mapper.snapshot()

    recovered_mapper = HighFidelityMapper(voxel_m=0.01, display_point_budget=5_000, pose_generation="pg-live")
    assert VerifiedArtifactIngestor(root, recovered_mapper).ingest_once()["ingested_artifacts"] == 1
    recovered = recovered_mapper.snapshot()

    assert recovered["optimized_pose_generation"] == first["optimized_pose_generation"]
    assert recovered["accepted_loop_closures"] == first["accepted_loop_closures"] == 1
    assert _sorted_points(recovered) == pytest.approx(_sorted_points(first))
    assert recovered["resident_full_lidar_frames"] == 0
