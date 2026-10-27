from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import replace

import numpy as np
import pytest

from mapping.pc.go2_mapping_fusion import FrameFilter, LidarObservationFrame, MeasuredVoxelMap, VoxelState
from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config


HASH = "a" * 64


def _frame(points):
    return LidarObservationFrame(
        artifact_sha256=HASH,
        frame_id="mid360",
        record_timestamp_ns=1_000_000_000,
        sensor_origin_xyz=np.zeros(3),
        points_xyz=np.asarray(points, dtype=np.float64),
        pose_map_from_lidar=np.eye(4),
        pose_generation="pg-1",
    )


def _cluster():
    return np.asarray(
        [[1.0 + x * 0.01, y * 0.01, z * 0.01] for x in range(5) for y in range(5) for z in range(2)],
        dtype=np.float64,
    )


def _frame_at(origin, world_point, timestamp_ns):
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(origin, dtype=np.float64)
    return LidarObservationFrame(
        artifact_sha256=HASH,
        frame_id="mid360",
        record_timestamp_ns=timestamp_ns,
        sensor_origin_xyz=np.asarray(origin, dtype=np.float64),
        points_xyz=np.asarray([np.asarray(world_point) - np.asarray(origin)], dtype=np.float64),
        pose_map_from_lidar=pose,
        pose_generation="pg-1",
    )


def test_frame_filter_classifies_without_mutating_raw_observations():
    points = np.vstack([_cluster(), [[2.0, 2.0, 2.0]]])
    frame = _frame(points)
    original = frame.points_xyz.copy()
    gates = load_reconstruction_config().fusion.filtering

    result = FrameFilter(gates).classify(frame)

    assert result.source_count == len(points)
    assert result.artifact_sha256 == HASH
    assert len(result.accepted_indices) == len(points) - 1
    assert any(item.reason == "RADIUS_OUTLIER" for item in result.rejected)
    assert np.array_equal(frame.points_xyz, original)
    assert frame.points_xyz.flags.writeable is False


def test_frame_filter_preserves_provenance_and_applies_self_mask():
    points = _cluster()
    frame = _frame(points)
    gates = load_reconstruction_config().fusion.filtering

    result = FrameFilter(gates, self_mask=lambda values: values[:, 0] < 1.015).classify(frame)

    assert result.artifact_sha256 == frame.artifact_sha256
    assert result.frame_id == frame.frame_id
    assert result.record_timestamp_ns == frame.record_timestamp_ns
    assert all(item.reason == "SELF_MASK" for item in result.rejected if item.index < 3)


@pytest.mark.parametrize("value", [0.099999, 30.000001])
def test_observation_frame_range_boundaries_are_fail_closed(value):
    frame = _frame([[value, 0.0, 0.0]])
    gates = load_reconstruction_config().fusion.filtering

    result = FrameFilter(gates).classify(frame)

    assert len(result.accepted_indices) == 0
    assert result.rejected[0].reason == "RANGE_OUT_OF_BOUNDS"


def test_nonfinite_raw_observation_is_retained_and_classified():
    frame = _frame([[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]])
    result = FrameFilter(load_reconstruction_config().fusion.filtering).classify(frame)

    assert result.source_count == 2
    assert [item.reason for item in result.rejected] == ["NONFINITE_XYZ", "NONFINITE_XYZ"]


def test_observation_frame_rejects_non_rigid_pose_and_malformed_hash():
    with pytest.raises(ValueError, match="artifact_sha256"):
        LidarObservationFrame(
            artifact_sha256="not-a-hash",
            frame_id="mid360",
            record_timestamp_ns=1,
            sensor_origin_xyz=np.zeros(3),
            points_xyz=np.ones((1, 3)),
            pose_map_from_lidar=np.eye(4),
            pose_generation="pg-1",
        )


def test_voxel_requires_independent_persistent_observations_before_measured():
    grid = MeasuredVoxelMap()
    grid.integrate(_frame_at((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0))
    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.CANDIDATE
    grid.integrate(_frame_at((0.25, 0.0, 0.0), (1.001, 0.0, 0.0), 600_000_000))
    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.CANDIDATE
    grid.integrate(_frame_at((0.5, 0.0, 0.0), (0.999, 0.0, 0.0), 1_200_000_000))

    voxel = grid.voxel((1.0, 0.0, 0.0))
    assert voxel.state is VoxelState.MEASURED
    assert voxel.observation_count == 3
    assert voxel.independent_views >= 2
    assert voxel.temporal_support_s == pytest.approx(1.2)


def test_candidate_defers_full_metrics_until_minimum_observation_count(monkeypatch):
    grid = MeasuredVoxelMap()
    original_metrics = grid._metrics
    calls = 0

    def counted_metrics(evidence):
        nonlocal calls
        calls += 1
        return original_metrics(evidence)

    monkeypatch.setattr(grid, "_metrics", counted_metrics)

    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 600_000_000),
        ((0.5, 0.0, 0.0), 1_200_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))

    assert calls == 1
    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.MEASURED


def test_measured_promotion_ignores_diagnostic_metrics_not_in_gate_contract(monkeypatch):
    class Gate:
        minimum_observations = 3
        minimum_temporal_support_s = 0.0

        def accepts(
            self,
            *,
            observation_count,
            independent_views,
            temporal_support_s,
            principal_stddev_m,
            point_to_plane_p95_m,
            static_probability,
        ):
            return observation_count == 3 and independent_views == 3

    config = load_reconstruction_config().fusion
    grid = MeasuredVoxelMap(gates=replace(config, measured_promotion=Gate()))
    monkeypatch.setattr(
        grid,
        "_metrics",
        lambda evidence: {
            "observation_count": 3,
            "independent_views": 3,
            "temporal_support_s": 1.0,
            "principal_stddev_m": 0.01,
            "point_to_plane_p95_m": 0.01,
            "static_probability": 1.0,
            "incidence_cosine": 0.8,
        },
    )

    for index in range(3):
        grid.integrate(_frame_at((0.25 * index, 0.0, 0.0), (1.0, 0.0, 0.0), index * 1_000_000_000))

    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.MEASURED


def test_candidate_defers_full_metrics_until_minimum_temporal_support(monkeypatch):
    grid = MeasuredVoxelMap()
    original_metrics = grid._metrics
    calls = 0

    def counted_metrics(evidence):
        nonlocal calls
        calls += 1
        return original_metrics(evidence)

    monkeypatch.setattr(grid, "_metrics", counted_metrics)

    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 100_000_000),
        ((0.5, 0.0, 0.0), 200_000_000),
        ((0.75, 0.0, 0.0), 600_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))

    assert calls == 1
    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.MEASURED


def test_large_frame_releases_map_lock_between_voxel_updates():
    class InterleavingLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._integrator_thread_id = None
            self._integrator_acquisitions = 0
            self.first_observation_released = threading.Event()
            self.continue_integration = threading.Event()

        def __enter__(self):
            thread_id = threading.get_ident()
            if self._integrator_thread_id == thread_id and self._integrator_acquisitions >= 2:
                assert self.continue_integration.wait(timeout=2.0)
            self._lock.acquire()
            if self._integrator_thread_id is None and thread_id != threading.main_thread().ident:
                self._integrator_thread_id = thread_id
            if self._integrator_thread_id == thread_id:
                self._integrator_acquisitions += 1
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            thread_id = threading.get_ident()
            acquisition = self._integrator_acquisitions
            self._lock.release()
            if self._integrator_thread_id == thread_id and acquisition == 2:
                self.first_observation_released.set()

    grid = MeasuredVoxelMap(voxel_size_m=0.01)
    lock = InterleavingLock()
    grid._lock = lock

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(grid.integrate, _frame([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        assert lock.first_observation_released.wait(timeout=2.0)
        midway = grid.snapshot()
        lock.continue_integration.set()
        assert future.result(timeout=2.0) == 2

    assert midway["voxel_count"] == 1
    assert grid.snapshot()["voxel_count"] == 2


def test_measured_voxel_defers_metric_recomputation_until_snapshot(monkeypatch):
    grid = MeasuredVoxelMap()
    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 600_000_000),
        ((0.5, 0.0, 0.0), 1_200_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))
    assert grid.voxel((1.0, 0.0, 0.0)).state is VoxelState.MEASURED
    original_metrics = grid._metrics
    calls = 0

    def counted_metrics(evidence):
        nonlocal calls
        calls += 1
        return original_metrics(evidence)

    monkeypatch.setattr(grid, "_metrics", counted_metrics)

    grid.integrate(_frame_at((0.75, 0.0, 0.0), (1.0, 0.0, 0.0), 1_800_000_000))

    assert calls == 0
    evidence = grid.voxel((1.0, 0.0, 0.0))
    assert calls == 1
    assert evidence.state is VoxelState.MEASURED
    assert evidence.observation_count == 4
    assert evidence.temporal_support_s == pytest.approx(1.8)


def test_depth_cannot_move_measured_lidar_geometry():
    grid = MeasuredVoxelMap()
    for origin, timestamp in [((0.0, 0.0, 0.0), 0), ((0.25, 0.0, 0.0), 600_000_000), ((0.5, 0.0, 0.0), 1_200_000_000)]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))
    evidence = grid.voxel((1.0, 0.0, 0.0))

    with pytest.raises(PermissionError):
        grid.update_measured_position_from_depth(evidence.key, (1.1, 0.0, 0.0))


def test_evidence_snapshot_tolerates_a_new_voxel_arriving_during_materialization(monkeypatch):
    grid = MeasuredVoxelMap()
    grid.integrate(_frame([[1.0, 0.0, 0.0]]))
    snapshot_started = threading.Event()
    continue_snapshot = threading.Event()
    original_snapshot = grid._snapshot

    def paused_snapshot(evidence, *args, **kwargs):
        snapshot_started.set()
        assert continue_snapshot.wait(timeout=2.0)
        return original_snapshot(evidence, *args, **kwargs)

    monkeypatch.setattr(grid, "_snapshot", paused_snapshot)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(grid.evidence)
        assert snapshot_started.wait(timeout=2.0)
        grid.integrate(_frame([[2.0, 0.0, 0.0]]))
        continue_snapshot.set()
        evidence = future.result(timeout=2.0)

    assert len(evidence) == 1


def test_state_filtered_evidence_materializes_only_matching_voxels(monkeypatch):
    grid = MeasuredVoxelMap()
    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 600_000_000),
        ((0.5, 0.0, 0.0), 1_200_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))
    grid.integrate(_frame([[2.0, 0.0, 0.0]]))
    original_snapshot = grid._snapshot
    materialized_states = []

    def counted_snapshot(evidence, *args, **kwargs):
        materialized_states.append(evidence.state)
        return original_snapshot(evidence, *args, **kwargs)

    monkeypatch.setattr(grid, "_snapshot", counted_snapshot)

    evidence = grid.evidence(VoxelState.MEASURED)

    assert len(evidence) == 1
    assert materialized_states == [VoxelState.MEASURED]


def test_measured_preview_points_do_not_recompute_quality_metrics(monkeypatch):
    grid = MeasuredVoxelMap()
    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 600_000_000),
        ((0.5, 0.0, 0.0), 1_200_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))

    monkeypatch.setattr(grid, "_metrics", lambda _evidence: pytest.fail("preview recomputed quality metrics"))

    points = grid.measured_points()

    assert points.shape == (1, 3)
    assert points[0] == pytest.approx((1.0, 0.0, 0.0))


def test_evidence_snapshot_reuses_metrics_computed_during_integration(monkeypatch):
    grid = MeasuredVoxelMap()
    for origin, timestamp in [
        ((0.0, 0.0, 0.0), 0),
        ((0.25, 0.0, 0.0), 600_000_000),
        ((0.5, 0.0, 0.0), 1_200_000_000),
    ]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))

    monkeypatch.setattr(grid, "_metrics", lambda _evidence: pytest.fail("quality metrics recomputed"))

    evidence = grid.evidence(VoxelState.MEASURED)

    assert len(evidence) == 1


def test_free_space_conflicts_require_three_unstable_generations_before_rejection():
    grid = MeasuredVoxelMap()
    for origin, timestamp in [((0.0, 0.0, 0.0), 0), ((0.25, 0.0, 0.0), 600_000_000), ((0.5, 0.0, 0.0), 1_200_000_000)]:
        grid.integrate(_frame_at(origin, (1.0, 0.0, 0.0), timestamp))
    key = grid.voxel((1.0, 0.0, 0.0)).key

    grid.observe_free_space(key, "pg-1")
    grid.observe_free_space(key, "pg-1")
    assert grid.voxel(key).state is VoxelState.UNSTABLE
    grid.observe_free_space(key, "pg-2")
    assert grid.voxel(key).state is VoxelState.UNSTABLE
    grid.observe_free_space(key, "pg-3")
    assert grid.voxel(key).state is VoxelState.REJECTED
    non_rigid = np.eye(4)
    non_rigid[0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid"):
        LidarObservationFrame(
            artifact_sha256=HASH,
            frame_id="mid360",
            record_timestamp_ns=1,
            sensor_origin_xyz=np.zeros(3),
            points_xyz=np.ones((1, 3)),
            pose_map_from_lidar=non_rigid,
            pose_generation="pg-1",
        )
