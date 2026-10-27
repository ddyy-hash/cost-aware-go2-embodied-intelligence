import hashlib
import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.derived_provisional_preview import (
    PROVISIONAL_VERTEX_COLOR_STATE,
    ProvisionalProjectionAccumulator,
    ProvisionalProjectionEvidence,
    load_provisional_surface_preview,
    parse_provisional_vertex_color_v1,
    publish_provisional_surface_preview,
)
from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


SURFACE_SHA256 = "a" * 64
GEOMETRY_REVISION_ID = "b" * 64
OVERRIDE_SHA256 = "c" * 64
ARTIFACT_SHA256 = "d" * 64
COLOR_FRAME_SHA256 = "e" * 64
CAMERA_INFO_SHA256 = "f" * 64


def _evidence(*, rgb=None, validity=None, geometry_revision_id=GEOMETRY_REVISION_ID):
    source_vertex_ids = ("0:0:-1", "0:1:-1")
    return ProvisionalProjectionEvidence(
        session_id="segment-20260812T083241Z",
        source_surface_sha256=SURFACE_SHA256,
        measured_surface_geometry_revision_id=geometry_revision_id,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        voxel_size_m=1.0,
        source_artifact_sha256=(ARTIFACT_SHA256,),
        source_vertex_ids=source_vertex_ids,
        rgb=np.asarray(
            [[12, 34, 56], [0, 0, 0]] if rgb is None else rgb,
            dtype=np.uint8,
        ),
        validity=np.asarray([True, False] if validity is None else validity, dtype=np.bool_),
        calibration_id="PROVISIONAL_AUTOCAL_PENDING_v1",
        operator_override_sha256=OVERRIDE_SHA256,
        runtime_projection_authorized=True,
        measurement_authority=False,
        color_camera_info_sha256=CAMERA_INFO_SHA256,
        color_source_frame_sha256=(COLOR_FRAME_SHA256,),
        lidar_frame_count=3,
        color_frame_count=2,
        paired_frame_count=2,
        projected_point_count=7,
        matched_surface_point_count=4,
    )


def test_provisional_preview_is_geometry_bound_and_never_authorizes_texture(tmp_path):
    published = publish_provisional_surface_preview(tmp_path, _evidence())

    manifest = published.manifest
    assert manifest["state"] == PROVISIONAL_VERTEX_COLOR_STATE
    assert manifest["delivery_authorized"] is False
    assert manifest["final_store_eligible"] is False
    assert manifest["measurement_authority"] is False
    assert manifest["runtime_projection_authorized"] is True
    assert manifest["source_surface_sha256"] == SURFACE_SHA256
    assert manifest["measured_surface_geometry_revision_id"] == GEOMETRY_REVISION_ID
    assert manifest["colored_vertex_count"] == 1
    assert published.manifest_path.is_file()
    assert published.vertex_color_path.is_file()

    loaded = load_provisional_surface_preview(
        published.manifest_path,
        expected_surface_sha256=SURFACE_SHA256,
        expected_geometry_revision_id=GEOMETRY_REVISION_ID,
    )
    parsed = parse_provisional_vertex_color_v1(
        loaded.vertex_color_path.read_bytes(),
        expected_geometry_revision_id=GEOMETRY_REVISION_ID,
        expected_preview_revision_id=loaded.preview_revision_id,
        expected_vertex_count=2,
    )
    assert parsed.validity.tolist() == [255, 0]
    assert parsed.rgb.tolist() == [[12, 34, 56], [0, 0, 0]]

    tampered = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    tampered["delivery_authorized"] = True
    published.manifest_path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authority"):
        load_provisional_surface_preview(published.manifest_path)


def test_provisional_preview_loader_rejects_another_surface_geometry(tmp_path):
    published = publish_provisional_surface_preview(tmp_path, _evidence())

    with pytest.raises(ValueError, match="geometry revision"):
        load_provisional_surface_preview(
            published.manifest_path,
            expected_surface_sha256=SURFACE_SHA256,
            expected_geometry_revision_id="9" * 64,
        )


def test_projector_uses_native_lidar_for_camera_and_transformed_lidar_for_mesh_ids():
    accumulator = ProvisionalProjectionAccumulator(
        source_vertex_ids=("0:0:-1",),
        voxel_size_m=1.0,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )
    color = np.zeros((3, 3, 3), dtype=np.uint8)
    color[1, 1] = [31, 67, 101]
    camera = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)

    matched = accumulator.observe(
        points_native_lidar=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        pose_map_from_lidar=pose,
        color_image_rgb=color,
        color_camera_matrix=camera,
        camera_from_native_lidar=np.eye(4, dtype=np.float64),
        color_frame_sha256=COLOR_FRAME_SHA256,
    )

    rgb, validity, frame_hashes = accumulator.finalize()
    assert matched == 1
    assert rgb.tolist() == [[31, 67, 101]]
    assert validity.tolist() == [True]
    assert frame_hashes == (COLOR_FRAME_SHA256,)
