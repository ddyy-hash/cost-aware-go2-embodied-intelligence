import hashlib
import json

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from mapping.pc.go2_mapping_appearance import measured_surface_identity
from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.sensor_frame import get_sensor_frame_transform
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore


TRANSFORM_ID = "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2"
PREVIEW_QUALITY = "PREVIEW_UNCALIBRATED"


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _derived_generation(
    tmp_path,
    *,
    source_layer="LIDAR_MEASURED",
    point_authority="LIDAR_MEASURED",
    mapping_source_mode="RAW_LIDAR_WITH_RECORDED_LIO",
    sensor_frame_transform_applied=True,
    legacy_manifest=True,
):
    source_hashes = ["a" * 64, "e" * 64]
    pose_generation = "b" * 64
    fusion_config_sha256 = "c" * 64
    source_artifacts = [
        {
            "relative_path": "segment_0.db3.zstd",
            "sha256": source_hashes[0],
            "bytes": 123,
        },
        {
            "relative_path": "segment_1.db3.zstd",
            "sha256": source_hashes[1],
            "bytes": 456,
        },
    ]
    identity_core = {
        "identity_schema_version": 1,
        "algorithm_version": "go2-offline-observed-surface-v1",
        "source_artifacts": source_artifacts,
        "source_artifact_sha256": source_hashes,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "mapping_source_mode": mapping_source_mode,
        "fusion_config_sha256": fusion_config_sha256,
        "pose_generation": pose_generation,
    }
    is_session_odom_local = mapping_source_mode == "RECORDED_LIO_SESSION_ODOM_CLOUD"
    if is_session_odom_local:
        identity_core["coordinate_scope"] = "SESSION_ODOM_LOCAL"
    identity = {**identity_core, "cache_key": canonical_sha256(identity_core)}
    generation = tmp_path / identity["cache_key"]
    artifacts = generation / "artifacts"
    artifacts.mkdir(parents=True)

    positions = np.asarray(
        [[0.0, 0.0, -0.5], [1.0, 0.0, -0.5], [0.0, 0.0, 0.5]],
        dtype=np.float64,
    )
    normals = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    source_vertex_ids = ("0:0:-1", "1:0:-1", "0:0:1")
    geometry_identity = measured_surface_identity(
        positions=positions,
        normals=normals,
        triangles=triangles,
        pose_generation=pose_generation,
        artifact_sha256=source_hashes,
        fusion_config_sha256=fusion_config_sha256,
    )
    surface = MeasuredSurface(
        positions,
        normals,
        triangles,
        source_vertex_ids,
        source_layer=source_layer,
        point_authority=point_authority,
    )
    published = LayerArtifactStore(artifacts).publish(
        surface,
        identity,
        annotations={
            "quality": PREVIEW_QUALITY,
            "status": PREVIEW_QUALITY,
            "sensor_frame_transform_id": TRANSFORM_ID,
            "mapping_source_mode": mapping_source_mode,
            "sensor_frame_transform_applied": sensor_frame_transform_applied,
            "coordinate_scope": "SESSION_ODOM_LOCAL" if is_session_odom_local else None,
            "coordinate_frame": "odom" if is_session_odom_local else None,
            "room_or_building": False if is_session_odom_local else None,
            "texture_authorized": False if is_session_odom_local else None,
            "volume_filled": False,
            "watertight": False,
            "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        },
    )
    surface_payload = {
        **surface.bounded_snapshot(),
        # Mapper snapshots sort the immutable artifact set independently from
        # the capture part order retained by the derivation identity.
        "artifact_sha256": list(reversed(source_hashes)),
        "pose_generation": pose_generation,
        "measured_surface_geometry_revision_id": geometry_identity.revision_id,
        "reconstruction_phase": "IDLE",
        "sensor_frame_transform_id": TRANSFORM_ID,
        "mapping_source_mode": mapping_source_mode,
        "sensor_frame_transform_applied": sensor_frame_transform_applied,
        "coordinate_scope": "SESSION_ODOM_LOCAL" if is_session_odom_local else None,
        "coordinate_frame": "odom" if is_session_odom_local else None,
        "room_or_building": False if is_session_odom_local else None,
        "texture_authorized": False if is_session_odom_local else None,
        "quality": PREVIEW_QUALITY,
        "status": PREVIEW_QUALITY,
        "volume_filled": False,
        "watertight": False,
        "voxel_size_m": 0.02,
    }
    surface_path = generation / "measured-surface.json"
    _write_json(surface_path, surface_payload)

    transform = get_sensor_frame_transform(TRANSFORM_ID)
    manifest = {
        "schema_version": 1,
        "status": PREVIEW_QUALITY,
        "quality": PREVIEW_QUALITY,
        "geometry_authority": False,
        "point_authority": point_authority,
        "volume_filled": False,
        "watertight": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "sensor_frame_transform_id": TRANSFORM_ID,
        "mapping_source_mode": mapping_source_mode,
        "sensor_frame_transform_applied": sensor_frame_transform_applied,
        "coordinate_scope": "SESSION_ODOM_LOCAL" if is_session_odom_local else None,
        "coordinate_frame": "odom" if is_session_odom_local else None,
        "room_or_building": False if is_session_odom_local else None,
        "texture_authorized": False if is_session_odom_local else None,
        "sensor_frame_transform": {
            "id": transform.id,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
            "provisional": True,
            "description": transform.description,
        },
        "source_root_name": "segment",
        "source_artifacts": source_artifacts,
        "derived_identity": identity,
        "surface_available": True,
        "surface": {
            "vertex_count": 3,
            "triangle_count": 1,
            "source_vertex_count": 3,
            "geometry_revision_id": geometry_identity.revision_id,
            "diagnostics": {"ground_below_trajectory": True},
        },
        "artifact": {
            "ply_name": published.ply_path.name,
            "metadata_name": published.metadata_path.name,
            "ply_sha256": published.ply_sha256,
        },
    }
    if not legacy_manifest:
        manifest["source_layer"] = source_layer
    _write_json(generation / "manifest.json", manifest)
    return surface_path, published.ply_path, identity


def _automatic_reconstruction_pointer(tmp_path):
    output_root = tmp_path / "automatic-derived"
    surface_path, _ply_path, _identity = _derived_generation(output_root)
    preview = server.load_derived_surface_preview(surface_path)
    worker_status = {
        "schema_version": 1,
        "session_id": "segment-20260812T083241Z",
        "state": "INCREMENTAL_SURFACE_AVAILABLE",
        "mode": "high_fidelity_incremental_reconstruction",
        "finalized": False,
        "available": True,
        "blocker": None,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "surface_json_path": str(surface_path),
        "surface_sha256": preview.status["surface_sha256"],
        "derived_cache_key": preview.status["derived_cache_key"],
        "ply_sha256": preview.status["ply_sha256"],
        "metadata_sha256": preview.status["metadata_sha256"],
        "vertex_count": preview.status["vertex_count"],
        "triangle_count": preview.status["triangle_count"],
        "quality": PREVIEW_QUALITY,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "source_layer": "LIDAR_MEASURED",
        "mapping_source_mode": "RAW_LIDAR_WITH_RECORDED_LIO",
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
    }
    pointer_path = tmp_path / "runtime" / "automatic-reconstruction-current.json"
    pointer_path.parent.mkdir(parents=True)
    _write_json(pointer_path, worker_status)
    return output_root, pointer_path, surface_path, preview, worker_status


def test_derived_surface_loader_verifies_and_exposes_content_addressed_preview(tmp_path):
    surface_path, ply_path, identity = _derived_generation(tmp_path)

    preview = server.load_derived_surface_preview(surface_path)

    assert preview.surface_path == surface_path.resolve()
    assert preview.status == {
        "schema_version": 1,
        "available": True,
        "blocker": None,
        "source_mode": "DERIVED_READ_ONLY",
        "read_only": True,
        "quality": PREVIEW_QUALITY,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "source_layer": "LIDAR_MEASURED",
        "mapping_source_mode": "RAW_LIDAR_WITH_RECORDED_LIO",
        "sensor_frame_transform_applied": True,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "derived_cache_key": identity["cache_key"],
        "surface_sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        "ply_sha256": hashlib.sha256(ply_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(ply_path.with_suffix(".json").read_bytes()).hexdigest(),
        "vertex_count": 3,
        "triangle_count": 1,
    }


def test_derived_surface_loader_accepts_a_legacy_raw_manifest_without_source_layer(tmp_path):
    """A verified pre-source-layer raw generation must remain readable."""

    surface_path, _ply_path, _identity = _derived_generation(tmp_path, legacy_manifest=True)

    preview = server.load_derived_surface_preview(surface_path)

    assert preview.status["source_layer"] == "LIDAR_MEASURED"
    assert preview.status["point_authority"] == "LIDAR_MEASURED"
    assert preview.status["mapping_source_mode"] == "RAW_LIDAR_WITH_RECORDED_LIO"


def test_derived_surface_loader_requires_explicit_registered_cloud_provenance(tmp_path):
    """The legacy fallback cannot relabel an incomplete registered-cloud artifact."""

    surface_path, _ply_path, _identity = _derived_generation(
        tmp_path,
        source_layer="LIO_REGISTERED_LIDAR_OBSERVED",
        point_authority="LIO_REGISTERED_LIDAR_OBSERVED",
        mapping_source_mode="RECORDED_LIO_REGISTERED_CLOUD",
        sensor_frame_transform_applied=False,
        legacy_manifest=True,
    )

    with pytest.raises(ValueError, match="allowed LiDAR observation layer"):
        server.load_derived_surface_preview(surface_path)


def test_derived_surface_loader_accepts_explicit_registered_cloud_provenance(tmp_path):
    surface_path, _ply_path, _identity = _derived_generation(
        tmp_path,
        source_layer="LIO_REGISTERED_LIDAR_OBSERVED",
        point_authority="LIO_REGISTERED_LIDAR_OBSERVED",
        mapping_source_mode="RECORDED_LIO_REGISTERED_CLOUD",
        sensor_frame_transform_applied=False,
        legacy_manifest=False,
    )

    preview = server.load_derived_surface_preview(surface_path)

    assert preview.status["source_layer"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert preview.status["point_authority"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert preview.status["mapping_source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert preview.status["sensor_frame_transform_applied"] is False


def test_automatic_pointer_rejects_a_declared_local_odom_scope_before_loading_any_surface(tmp_path):
    output_root, pointer_path, _surface_path, _preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    worker_status.update({
        "source_layer": "LIO_REGISTERED_LIDAR_OBSERVED",
        "point_authority": "LIO_REGISTERED_LIDAR_OBSERVED",
        "mapping_source_mode": "RECORDED_LIO_SESSION_ODOM_CLOUD",
        "sensor_frame_transform_applied": False,
        "coordinate_scope": "SESSION_ODOM_LOCAL",
        "coordinate_frame": "odom",
        "room_or_building": False,
        "texture_authorized": False,
    })
    _write_json(pointer_path, worker_status)

    with pytest.raises(ValueError, match="cannot load SESSION_ODOM_LOCAL"):
        server.load_automatic_reconstruction_preview(pointer_path, output_root=output_root)


def _session_odom_local_pointer(tmp_path):
    output_root = tmp_path / "session-odom-local"
    session_id = "segment-20260814T030405Z"
    surface_path, _ply_path, _identity = _derived_generation(
        output_root / session_id,
        source_layer="LIO_REGISTERED_LIDAR_OBSERVED",
        point_authority="LIO_REGISTERED_LIDAR_OBSERVED",
        mapping_source_mode="RECORDED_LIO_SESSION_ODOM_CLOUD",
        sensor_frame_transform_applied=False,
        legacy_manifest=False,
    )
    preview = server.load_derived_surface_preview(surface_path)
    pointer_path = tmp_path / "runtime" / "session-odom-local-current.json"
    pointer_path.parent.mkdir(parents=True)
    pointer = {
        "schema_version": 1,
        "session_id": session_id,
        "state": "SESSION_ODOM_LOCAL_SURFACE_AVAILABLE",
        "mode": "session_odom_local_observed_surface",
        "available": True,
        "blocker": None,
        "read_only": True,
        "quality": PREVIEW_QUALITY,
        "geometry_authority": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "mapping_source_mode": "RECORDED_LIO_SESSION_ODOM_CLOUD",
        "coordinate_scope": "SESSION_ODOM_LOCAL",
        "coordinate_frame": "odom",
        "room_or_building": False,
        "texture_authorized": False,
        "sensor_frame_transform_applied": False,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "source_layer": "LIO_REGISTERED_LIDAR_OBSERVED",
        "point_authority": "LIO_REGISTERED_LIDAR_OBSERVED",
        "surface_json_relative_path": surface_path.relative_to(output_root).as_posix(),
        "surface_sha256": preview.status["surface_sha256"],
        "derived_cache_key": preview.status["derived_cache_key"],
        "ply_sha256": preview.status["ply_sha256"],
        "metadata_sha256": preview.status["metadata_sha256"],
        "vertex_count": preview.status["vertex_count"],
        "triangle_count": preview.status["triangle_count"],
    }
    _write_json(pointer_path, pointer)
    return output_root, pointer_path, surface_path, preview, pointer


def test_session_odom_local_pointer_loads_only_a_verified_local_odom_surface(tmp_path):
    output_root, pointer_path, surface_path, _preview, pointer = _session_odom_local_pointer(tmp_path)

    loaded, status = server.load_session_odom_local_preview(pointer_path, output_root=output_root)

    assert loaded.surface_path == surface_path.resolve()
    assert status == pointer
    assert loaded.status["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert loaded.status["coordinate_frame"] == "odom"
    assert loaded.status["room_or_building"] is False
    assert loaded.status["texture_authorized"] is False


def test_session_odom_local_publisher_writes_an_atomic_separate_local_pointer(tmp_path):
    output_root, _pointer_path, surface_path, _preview, _pointer = _session_odom_local_pointer(tmp_path)
    status_path = tmp_path / "runtime" / "session-odom-local-current.json"

    published = server.publish_session_odom_local_preview(
        surface_path,
        output_root=output_root,
        status_path=status_path,
    )
    loaded, status = server.load_session_odom_local_preview(status_path, output_root=output_root)

    assert loaded.surface_path == surface_path.resolve()
    assert status == published
    assert published["surface_json_relative_path"] == surface_path.relative_to(output_root).as_posix()
    assert published["session_id"] == "segment-20260814T030405Z"


def test_dashboard_serves_session_odom_local_only_from_its_separate_read_only_api(tmp_path, monkeypatch):
    output_root, pointer_path, surface_path, _preview, _pointer = _session_odom_local_pointer(tmp_path)
    monkeypatch.setattr(server, "_session_odom_local_status_path", str(pointer_path))
    monkeypatch.setattr(server, "_session_odom_local_root", output_root.resolve())
    monkeypatch.setattr(server, "_session_odom_local_preview", None)
    monkeypatch.setattr(server, "_session_odom_local_status", None)

    local_response = server.session_odom_local_surface()
    local_status = json.loads(server.session_odom_local_surface_status().body)

    assert isinstance(local_response, FileResponse)
    assert local_response.path == str(surface_path.resolve())
    assert local_status["source_mode"] == "SESSION_ODOM_LOCAL_READ_ONLY"
    assert local_status["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert local_status["room_or_building"] is False
    assert local_status["texture_authorized"] is False
    automatic_status = json.loads(server.measured_surface_status().body)
    assert automatic_status.get("source_mode") != "SESSION_ODOM_LOCAL_READ_ONLY"


def test_derived_surface_loader_rejects_changed_ply(tmp_path):
    surface_path, ply_path, _identity = _derived_generation(tmp_path)
    ply_path.write_bytes(ply_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="PLY checksum"):
        server.load_derived_surface_preview(surface_path)


def test_derived_surface_loader_rejects_identity_or_transform_mismatch(tmp_path):
    surface_path, _ply_path, _identity = _derived_generation(tmp_path)
    manifest_path = surface_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derived_identity"]["pose_generation"] = "d" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="identity cache key"):
        server.load_derived_surface_preview(surface_path)


def test_dashboard_serves_derived_surface_read_only_without_touching_live_mapper(tmp_path, monkeypatch):
    surface_path, _ply_path, _identity = _derived_generation(tmp_path)
    preview = server.load_derived_surface_preview(surface_path)

    class LiveMapperMustNotBeRead:
        def measured_surface_snapshot(self):
            raise AssertionError("derived preview must not read the live mapper")

    monkeypatch.setattr(server, "_derived_surface_preview", preview)
    monkeypatch.setattr(server, "dense_mapper", LiveMapperMustNotBeRead())

    response = server.measured_surface()
    assert isinstance(response, FileResponse)
    assert response.path == str(surface_path.resolve())
    assert response.headers["cache-control"] == "no-store"
    status = json.loads(server.measured_surface_status().body)
    assert status == preview.status
    empty_manifest = server.building_render_manifest()
    assert empty_manifest.status_code == 204
    assert empty_manifest.headers["cache-control"] == "no-store"
    assert server.measured_surface_appearance()["blockers"] == ["DERIVED_SURFACE_READ_ONLY"]
    with pytest.raises(HTTPException) as blocked:
        server.publish_measured_reconstruction()
    assert blocked.value.status_code == 405


def test_dashboard_reloads_automatic_pointer_and_keeps_last_valid_surface_across_transient_damage(tmp_path, monkeypatch):
    output_root, pointer_path, surface_path, preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    monkeypatch.setattr(server, "_derived_surface_preview", None)
    monkeypatch.setattr(server, "_automatic_reconstruction_status_path", str(pointer_path))
    monkeypatch.setattr(server, "_automatic_reconstruction_root", output_root.resolve())
    monkeypatch.setattr(server, "_automatic_reconstruction_preview", None)
    monkeypatch.setattr(server, "_automatic_reconstruction_status", None)

    response = server.measured_surface()
    first_status = json.loads(server.measured_surface_status().body)

    assert isinstance(response, FileResponse)
    assert response.path == str(surface_path.resolve())
    assert first_status == {
        **preview.status,
        "automatic_reconstruction_status": worker_status,
    }

    pointer_path.write_text("{not valid JSON", encoding="utf-8")

    retained_response = server.measured_surface()
    retained_status = json.loads(server.measured_surface_status().body)

    assert isinstance(retained_response, FileResponse)
    assert retained_response.path == str(surface_path.resolve())
    assert retained_status == first_status


def test_automatic_reconstruction_pointer_rejects_a_surface_path_outside_the_output_root(tmp_path):
    _output_root, pointer_path, _surface_path, _preview, _worker_status = _automatic_reconstruction_pointer(tmp_path)

    with pytest.raises(ValueError, match="escapes its output root"):
        server.load_automatic_reconstruction_preview(
            pointer_path,
            output_root=tmp_path / "different-output-root",
        )


def test_automatic_reconstruction_pointer_uses_output_root_relative_surface_path_across_hosts(tmp_path):
    output_root, pointer_path, surface_path, _preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    worker_status["surface_json_path"] = r"C:\stale-host\output\reconstruction\derived\unavailable\measured-surface.json"
    worker_status["surface_json_relative_path"] = surface_path.relative_to(output_root).as_posix()
    _write_json(pointer_path, worker_status)

    preview, pointer = server.load_automatic_reconstruction_preview(
        pointer_path,
        output_root=output_root,
    )

    assert preview.surface_path == surface_path.resolve()
    assert pointer["surface_json_relative_path"] == surface_path.relative_to(output_root).as_posix()


def test_automatic_reconstruction_pointer_rejects_an_escaping_relative_surface_path(tmp_path):
    output_root, pointer_path, _surface_path, _preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    worker_status["surface_json_relative_path"] = "../outside/measured-surface.json"
    _write_json(pointer_path, worker_status)

    with pytest.raises(ValueError, match="relative surface path"):
        server.load_automatic_reconstruction_preview(pointer_path, output_root=output_root)


def test_automatic_reconstruction_pointer_rejects_a_noncanonical_session_id(tmp_path):
    output_root, pointer_path, _surface_path, _preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    worker_status["session_id"] = "segment-not-a-capture-timestamp"
    _write_json(pointer_path, worker_status)

    with pytest.raises(ValueError, match="session is invalid"):
        server.load_automatic_reconstruction_preview(pointer_path, output_root=output_root)
