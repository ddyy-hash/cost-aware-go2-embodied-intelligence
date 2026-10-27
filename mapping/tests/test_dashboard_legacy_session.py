import hashlib
import json
import struct

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from mapping.pc.go2_mapping_dashboard import server


PREVIEW_QUALITY = "PREVIEW_UNCALIBRATED"
SOURCE_MODE = "LEGACY_SESSION_READ_ONLY"


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_binary_surface(path):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by Open3D\n"
        "element vertex 3\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "element face 1\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertices = b"".join(
        struct.pack("<6d3B", *record)
        for record in (
            (1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 255, 0, 0),
            (4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0, 255, 0),
            (7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0, 0, 255),
        )
    )
    face = struct.pack("<B3I", 3, 0, 1, 2)
    path.write_bytes(header + vertices + face)


def _legacy_session(tmp_path):
    root = tmp_path / "20260724_214444_157"
    reconstruction = root / "reconstruction"
    fusion = reconstruction / "linux_fusion"
    fusion.mkdir(parents=True)
    _write_json(
        root / "metadata.json",
        {
            "format_version": 1,
            "status": "complete",
            "start_time": "2026-07-24T21:44:44.157+0800",
            "stop_time": "2026-07-24T22:01:12.822+0800",
            "counts": {"lidar": 15194, "motion": 293287, "rgb": 14817, "depth": 14817},
        },
    )
    _write_json(
        reconstruction / "session_audit.json",
        {"calibration": {"body_to_depth_available": False}},
    )
    _write_json(
        reconstruction / "trajectory_meta.json",
        {"T_body_camera": "identity_uncalibrated", "n_frames": 14817},
    )
    _write_json(
        reconstruction / "lidar_world_meta.json",
        {"floor_height_offset_m": 3.0, "n_lidar_frames": 15194},
    )
    _write_json(
        fusion / "extrinsic_calibration.json",
        {
            "source": "nominal_axis_mapping",
            "metrics": {"accepted": False},
            "unresolved": "T_body_lidar is assumed identity because it was not recorded",
        },
    )
    surface_path = reconstruction / "reconstruction.ply"
    _write_binary_surface(surface_path)
    return root, surface_path


def test_legacy_session_loader_exposes_verified_uncalibrated_read_only_surface(tmp_path):
    root, surface_path = _legacy_session(tmp_path)

    preview = server.load_legacy_session_preview(root)

    assert preview.session_root == root.resolve()
    assert preview.surface_path == surface_path.resolve()
    assert preview.status == {
        "schema_version": 1,
        "available": True,
        "blocker": None,
        "source_mode": SOURCE_MODE,
        "read_only": True,
        "quality": PREVIEW_QUALITY,
        "geometry_authority": False,
        "hidden_surface_policy": "LEGACY_RECONSTRUCTION_UNVERIFIED",
        "sensor_frame_transform_id": "LEGACY_BODY_LIDAR_IDENTITY_UNCALIBRATED",
        "session_id": root.name,
        "surface_encoding": "PLY_BINARY_LITTLE_ENDIAN_OPEN3D_V1",
        "surface_sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        "vertex_count": 3,
        "triangle_count": 1,
        "has_vertex_colors": True,
        "body_to_depth_calibrated": False,
        "body_to_lidar_calibrated": False,
        "floor_height_authority": False,
        "floor_height_offset_m": 3.0,
    }


def test_legacy_session_loader_rejects_non_triangular_or_trailing_ply_data(tmp_path):
    root, surface_path = _legacy_session(tmp_path)
    surface_path.write_bytes(surface_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="byte count"):
        server.load_legacy_session_preview(root)


def test_dashboard_serves_legacy_session_without_reading_or_writing_live_mapper(tmp_path, monkeypatch):
    root, surface_path = _legacy_session(tmp_path)
    preview = server.load_legacy_session_preview(root)

    class LiveMapperMustNotBeRead:
        def measured_surface_snapshot(self):
            raise AssertionError("legacy preview must not read the live mapper")

    monkeypatch.setattr(server, "_derived_surface_preview", None)
    monkeypatch.setattr(server, "_legacy_surface_preview", preview)
    monkeypatch.setattr(server, "dense_mapper", LiveMapperMustNotBeRead())

    response = server.measured_surface()
    assert isinstance(response, FileResponse)
    assert response.path == str(surface_path.resolve())
    assert response.media_type == "application/vnd.go2.legacy-surface-ply"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-sha256"] == preview.status["surface_sha256"]
    assert json.loads(server.measured_surface_status().body) == preview.status
    assert server.building_render_manifest().status_code == 204
    assert server.measured_surface_appearance()["blockers"] == ["LEGACY_SESSION_READ_ONLY"]
    with pytest.raises(HTTPException) as blocked:
        server.publish_measured_reconstruction()
    assert blocked.value.status_code == 405
