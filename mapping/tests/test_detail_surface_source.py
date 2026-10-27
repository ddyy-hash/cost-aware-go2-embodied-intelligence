import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    PayloadLimitExceeded,
    adaptive_detail_hz,
    build_live_keyframe_message,
    build_surface_source_status_message,
    deduplicate_surface_cells,
)


def _occupied_cells(points: np.ndarray, cell_m: float = 0.005) -> set[tuple[int, int, int]]:
    cells = np.floor(points.astype(np.float64) / cell_m).astype(np.int64)
    return {tuple(int(value) for value in row) for row in cells}


def test_detail_cadence_degrades_from_target_to_minimum_with_queue_pressure():
    assert adaptive_detail_hz(target_hz=5.0, minimum_hz=1.0, queued_frames=0, queue_capacity=64) == 5.0
    assert adaptive_detail_hz(target_hz=5.0, minimum_hz=1.0, queued_frames=32, queue_capacity=64) == 3.0
    assert adaptive_detail_hz(target_hz=5.0, minimum_hz=1.0, queued_frames=64, queue_capacity=64) == 1.0
    assert adaptive_detail_hz(target_hz=5.0, minimum_hz=1.0, queued_frames=80, queue_capacity=64) == 1.0


@pytest.mark.parametrize(
    ("target_hz", "minimum_hz", "queued_frames", "queue_capacity"),
    [
        (0.0, 0.0, 0, 1),
        (1.0, 2.0, 0, 1),
        (5.0, 1.0, -1, 1),
        (5.0, 1.0, 0, 0),
    ],
)
def test_detail_cadence_rejects_invalid_pressure_inputs(target_hz, minimum_hz, queued_frames, queue_capacity):
    with pytest.raises(ValueError):
        adaptive_detail_hz(
            target_hz=target_hz,
            minimum_hz=minimum_hz,
            queued_frames=queued_frames,
            queue_capacity=queue_capacity,
        )


def test_detail_source_keeps_one_centroid_nearest_point_per_signed_surface_cell():
    points = np.asarray(
        [
            [0.0001, 0.0001, 0.0001],
            [0.0024, 0.0024, 0.0024],
            [0.0060, 0.0010, 0.0010],
            [-0.0010, -0.0010, -0.0010],
        ],
        dtype=np.float32,
    )

    detail = deduplicate_surface_cells(points, cell_m=0.005)

    assert detail.source_points == 4
    assert detail.occupied_cells == 3
    assert detail.points_xyz.dtype == np.dtype("<f4")
    assert _occupied_cells(detail.points_xyz) == _occupied_cells(points)
    assert any(np.allclose(point, [0.0024, 0.0024, 0.0024]) for point in detail.points_xyz)


def test_detail_source_preserves_every_stool_and_handle_cell_above_legacy_point_cap():
    floor_x, floor_y = np.meshgrid(
        np.arange(60, dtype=np.float32) * 0.01,
        np.arange(40, dtype=np.float32) * 0.01,
        indexing="xy",
    )
    floor = np.column_stack((floor_x.ravel(), floor_y.ravel(), np.zeros(floor_x.size, dtype=np.float32)))
    leg_z = np.arange(70, dtype=np.float32) * 0.005
    legs = np.concatenate(
        [
            np.column_stack((np.full_like(leg_z, x), np.full_like(leg_z, y), leg_z))
            for x, y in ((0.10, 0.10), (0.10, 0.30), (0.50, 0.10), (0.50, 0.30))
        ]
    )
    angle = np.linspace(-np.pi / 2, np.pi / 2, 80, dtype=np.float32)
    handle = np.column_stack(
        (
            np.full_like(angle, 0.59),
            0.20 + 0.035 * np.cos(angle),
            0.85 + 0.035 * np.sin(angle),
        )
    )
    points = np.concatenate((floor, legs, handle)).astype(np.float32)

    detail = deduplicate_surface_cells(points, cell_m=0.005)

    expected_cells = _occupied_cells(points)
    assert len(expected_cells) > 1200
    assert detail.occupied_cells == len(expected_cells)
    assert _occupied_cells(detail.points_xyz) == expected_cells
    assert _occupied_cells(handle).issubset(_occupied_cells(detail.points_xyz))
    assert _occupied_cells(legs).issubset(_occupied_cells(detail.points_xyz))


def test_detail_source_rejects_non_finite_points_and_invalid_cell_size():
    with pytest.raises(ValueError, match="finite"):
        deduplicate_surface_cells(np.asarray([[0.0, np.nan, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="cell_m"):
        deduplicate_surface_cells(np.zeros((1, 3), dtype=np.float32), cell_m=0.0)


def test_typed_keyframe_hashes_exact_points_and_measures_the_complete_json_message():
    points = np.asarray([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype="<f4")
    message = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=12,
        timestamp_ns=123_000,
        calibration_id="PREVIEW_TEST",
        source_config_sha256="a" * 64,
        points_xyz=points,
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
        payload_limit_bytes=4096,
    )

    decoded = json.loads(message.json_text)
    exact_bytes = points.tobytes(order="C")
    assert decoded["message_type"] == "live_keyframe"
    assert decoded["protocol_version"] == 3
    assert decoded["building_id"] == "building-a"
    assert decoded["session_id"] == "session-a"
    assert decoded["producer_instance_id"] == "producer-a"
    assert decoded["keyframe_sequence"] == 12
    assert decoded["point_encoding"] == "f32-le-xyz-zlib-base64"
    assert decoded["point_count"] == 2
    assert decoded["point_bytes_sha256"] == hashlib.sha256(exact_bytes).hexdigest()
    assert decoded["source_clipped"] is False
    assert message.message_bytes == len(message.json_text.encode("utf-8"))


def test_typed_keyframe_emits_the_explicit_sensor_frame_transform_identity():
    message = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=12,
        timestamp_ns=123_000,
        calibration_id="PREVIEW_TEST",
        source_config_sha256="a" * 64,
        points_xyz=np.asarray([[0.1, 0.2, 0.3]], dtype="<f4"),
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )

    assert json.loads(message.json_text)["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def test_typed_keyframe_rejects_when_the_complete_message_exceeds_the_limit():
    points = np.arange(300, dtype="<f4").reshape((-1, 3))

    with pytest.raises(PayloadLimitExceeded) as failure:
        build_live_keyframe_message(
            building_id="building-a",
            session_id="session-a",
            map_id="map-a",
            producer_instance_id="producer-a",
            keyframe_sequence=1,
            timestamp_ns=1,
            calibration_id="PREVIEW_TEST",
            source_config_sha256="a" * 64,
            points_xyz=points,
            pose_map_from_lidar=np.eye(4),
            imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            point_cloud_source="/lio/deskewed",
            motion_compensated=True,
            points_in_map_frame=False,
            payload_limit_bytes=256,
        )

    assert failure.value.message_bytes > 256
    assert failure.value.payload_limit_bytes == 256


def test_source_status_has_an_independent_typed_sequence_and_detail_counters():
    message = build_surface_source_status_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        status_sequence=8,
        timestamp_ns=123_000,
        source_state="SOURCE_READY",
        source_clipped=False,
        source_points_before_dedup=2000,
        source_points_after_dedup=1700,
        compressed_point_bytes=12000,
        websocket_message_bytes=17000,
        accepted_detail_hz=5.0,
        source_config_sha256="b" * 64,
        blocker=None,
    )

    decoded = json.loads(message.json_text)
    assert decoded["message_type"] == "surface_source_status"
    assert decoded["status_sequence"] == 8
    assert "keyframe_sequence" not in decoded
    assert decoded["source_points_after_dedup"] == 1700
    assert message.message_bytes == len(message.json_text.encode("utf-8"))


def test_actual_ros_producer_uses_detail_cells_without_global_stride():
    root = Path(__file__).resolve().parents[2]
    producer = (
        root / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "live_keyframe_producer.py"
    ).read_text(encoding="utf-8")

    assert "deduplicate_surface_cells(" in producer
    assert "self.args.surface_cell_m" in producer
    assert "xyz = xyz[::stride]" not in producer
    assert 'parser.add_argument("--surface-cell-m"' in producer
    assert 'parser.add_argument("--detail-payload-limit-bytes"' in producer
    assert "build_live_keyframe_message(" in producer
    assert "build_surface_source_status_message(" in producer
    assert '"live_keyframe_ack"' in producer
    assert '"surface_source_status_ack"' in producer
    assert 'parser.add_argument("--building-id"' in producer
    assert 'parser.add_argument("--session-id"' in producer
