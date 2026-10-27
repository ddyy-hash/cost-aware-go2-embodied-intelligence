import hashlib

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_protocol import (
    LiveProtocolError,
    build_live_keyframe_ack,
    build_surface_source_status_ack,
    validate_live_keyframe,
    validate_surface_source_status,
)
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    IDENTITY_SENSOR_FRAME_V1,
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    build_live_keyframe_message,
    build_surface_source_status_message,
)


SOURCE_HASH = hashlib.sha256(b"detail-source-config").hexdigest()


def _keyframe(sequence: int = 7):
    points = np.asarray([[0.1, 0.2, 0.3], [1.0, -2.0, 0.5]], dtype="<f4")
    return build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=123_000,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=points,
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
    )


def test_typed_keyframe_validates_exact_points_and_complete_identity():
    encoded = _keyframe()

    frame = validate_live_keyframe(encoded.payload, message_bytes=encoded.message_bytes)

    assert frame.identity.as_tuple() == ("building-a", "session-a", "map-a", "producer-a", 7)
    assert np.array_equal(frame.points_xyz, np.asarray([[0.1, 0.2, 0.3], [1.0, -2.0, 0.5]], dtype="<f4"))
    assert frame.point_bytes_sha256 == encoded.payload["point_bytes_sha256"]
    assert frame.message_bytes == encoded.message_bytes
    assert frame.sensor_frame_transform_id == IDENTITY_SENSOR_FRAME_V1


def test_keyframe_carries_a_registered_sensor_frame_and_legacy_missing_field_is_identity():
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=8,
        timestamp_ns=123_001,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray([[0.1, 0.2, 0.3]], dtype="<f4"),
        pose_map_from_lidar=np.eye(4),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=False,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )
    frame = validate_live_keyframe(encoded.payload)
    assert frame.sensor_frame_transform_id == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2

    legacy = dict(_keyframe().payload)
    legacy.pop("sensor_frame_transform_id", None)
    assert validate_live_keyframe(legacy).sensor_frame_transform_id == IDENTITY_SENSOR_FRAME_V1

    with pytest.raises(LiveProtocolError, match="sensor frame transform"):
        validate_live_keyframe({**encoded.payload, "sensor_frame_transform_id": "UNKNOWN_FRAME"})


def test_typed_keyframe_rejects_checksum_and_complete_message_size_mismatch():
    encoded = _keyframe()
    bad_hash = {**encoded.payload, "point_bytes_sha256": "0" * 64}

    with pytest.raises(LiveProtocolError, match="point checksum"):
        validate_live_keyframe(bad_hash)
    with pytest.raises(LiveProtocolError, match="message limit"):
        validate_live_keyframe(encoded.payload, message_bytes=4 * 1024 * 1024 + 1)


def test_keyframe_and_status_acknowledgements_are_independently_correlated():
    frame = validate_live_keyframe(_keyframe(9).payload)
    status_message = build_surface_source_status_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        status_sequence=31,
        timestamp_ns=124_000,
        source_state="SOURCE_READY",
        source_clipped=False,
        source_points_before_dedup=100,
        source_points_after_dedup=80,
        compressed_point_bytes=500,
        websocket_message_bytes=900,
        accepted_detail_hz=5.0,
        source_config_sha256=SOURCE_HASH,
        blocker=None,
    )
    status = validate_surface_source_status(status_message.payload)

    keyframe_ack = build_live_keyframe_ack(
        frame,
        accepted=True,
        disposition="QUEUED_NEW",
        wal_record_sha256="a" * 64,
        next_sequence=10,
    )
    status_ack = build_surface_source_status_ack(status, accepted=True)

    assert keyframe_ack["message_type"] == "live_keyframe_ack"
    assert keyframe_ack["keyframe_sequence"] == 9
    assert keyframe_ack["ack_stage"] == "QUEUED"
    assert keyframe_ack["disposition"] == "QUEUED_NEW"
    assert status_ack["message_type"] == "surface_source_status_ack"
    assert status_ack["status_sequence"] == 31
    assert "keyframe_sequence" not in status_ack
    assert "status_sequence" not in keyframe_ack
