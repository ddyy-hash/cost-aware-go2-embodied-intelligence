import hashlib
import sqlite3
import struct

import numpy as np
import pytest
import zstandard

from mapping.pc.go2_mapping_appearance.derived_provisional_builder import (
    DerivedSurfaceBinding,
    ProvisionalSourceArtifact,
    RuntimeProjection,
    _decode_rows,
    build_provisional_projection_evidence,
)
from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
from mapping.tests.test_high_fidelity_mapper import _pointcloud_cdr, _pose_cdr
from mapping.tests.test_rgbd_artifact_refinement import (
    _camera_contract,
    _camera_info_payload,
    _image_payload,
)


SURFACE_SHA256 = "a" * 64
GEOMETRY_REVISION_ID = "b" * 64
OVERRIDE_SHA256 = "c" * 64


def _write_source_part(path):
    raw = path.with_suffix("")
    color = np.zeros((3, 3, 3), dtype=np.uint8)
    color[1, 1] = [29, 71, 113]
    _, contract = _camera_contract("camera_color_optical_frame")
    contract = dict(contract)
    contract["width"] = 3
    contract["height"] = 3
    contract["k"] = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    contract["p"] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    database = sqlite3.connect(raw)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    database.executemany(
        "INSERT INTO topics VALUES (?, ?, ?)",
        (
            (1, "/livox/lidar", "sensor_msgs/msg/PointCloud2"),
            (2, "/go2/lio/pose", "geometry_msgs/msg/PoseWithCovarianceStamped"),
            (3, "/camera/color/image_raw", "sensor_msgs/msg/Image"),
            (4, "/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
        ),
    )
    timestamp = 1_000_000_000
    database.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (
            (1, 2, timestamp, _pose_cdr(timestamp, (0.0, 0.0, 0.0))),
            (2, 1, timestamp, _pointcloud_cdr(timestamp, ((0.0, 0.0, 1.0),))),
            (3, 4, timestamp, _camera_info_payload("camera_color_optical_frame", contract)),
            (4, 3, timestamp, _image_payload("camera_color_optical_frame", "rgb8", color)),
        ),
    )
    database.commit()
    database.close()
    path.write_bytes(zstandard.ZstdCompressor().compress(raw.read_bytes()))
    raw.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(source_artifact):
    return DerivedSurfaceBinding(
        session_id="segment-20260812T083241Z",
        source_surface_sha256=SURFACE_SHA256,
        measured_surface_geometry_revision_id=GEOMETRY_REVISION_ID,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        voxel_size_m=1.0,
        source_vertex_ids=("0:0:-1",),
        source_artifacts=(source_artifact,),
    )


def _projection():
    return RuntimeProjection(
        calibration_id="PROVISIONAL_AUTOCAL_PENDING_v1",
        operator_override_sha256=OVERRIDE_SHA256,
        camera_from_native_lidar=np.eye(4),
        runtime_projection_authorized=True,
        measurement_authority=False,
    )


def test_builder_replays_native_lidar_rgb_and_lio_into_active_surface_ids(tmp_path):
    source_root = tmp_path / "segment-20260812T083241Z"
    source_root.mkdir()
    part = source_root / "segment-20260812T083241Z_0.db3.zstd"
    digest = _write_source_part(part)
    artifact = ProvisionalSourceArtifact(part.name, digest, part.stat().st_size)

    evidence = build_provisional_projection_evidence(
        binding=_binding(artifact),
        source_root=source_root,
        projection=_projection(),
    )

    assert evidence.source_artifact_sha256 == (digest,)
    assert evidence.rgb.tolist() == [[29, 71, 113]]
    assert evidence.validity.tolist() == [True]
    assert evidence.lidar_frame_count == 1
    assert evidence.color_frame_count == 1
    assert evidence.paired_frame_count == 1
    assert evidence.projected_point_count == 1
    assert evidence.matched_surface_point_count == 1
    assert len(evidence.color_camera_info_sha256) == 64
    assert len(evidence.color_source_frame_sha256) == 1


def test_builder_rejects_source_artifact_when_its_bound_byte_count_changes(tmp_path):
    source_root = tmp_path / "segment-20260812T083241Z"
    source_root.mkdir()
    part = source_root / "segment-20260812T083241Z_0.db3.zstd"
    digest = _write_source_part(part)
    artifact = ProvisionalSourceArtifact(part.name, digest, part.stat().st_size)
    part.write_bytes(part.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="byte count changed"):
        build_provisional_projection_evidence(
            binding=_binding(artifact),
            source_root=source_root,
            projection=_projection(),
        )


def test_builder_rejects_same_size_source_artifact_content_tampering(tmp_path):
    source_root = tmp_path / "segment-20260812T083241Z"
    source_root.mkdir()
    part = source_root / "segment-20260812T083241Z_0.db3.zstd"
    digest = _write_source_part(part)
    artifact = ProvisionalSourceArtifact(part.name, digest, part.stat().st_size)
    changed = bytearray(part.read_bytes())
    changed[-1] ^= 0x01
    part.write_bytes(changed)

    with pytest.raises(ValueError, match="source artifact SHA-256"):
        build_provisional_projection_evidence(
            binding=_binding(artifact),
            source_root=source_root,
            projection=_projection(),
        )


def test_builder_reports_invalid_cdr_as_a_preview_decode_error():
    with pytest.raises(ValueError, match="cannot decode /go2/lio/pose"):
        _decode_rows(
            [
                (
                    "/go2/lio/pose",
                    "geometry_msgs/msg/PoseWithCovarianceStamped",
                    1_000_000_000,
                    b"invalid-cdr",
                )
            ]
        )
