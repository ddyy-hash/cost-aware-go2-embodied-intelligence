import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance import measured_surface_identity
from mapping.pc.go2_mapping_dashboard import static_surface_compatibility
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.sensor_frame import get_sensor_frame_transform
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore


TRANSFORM_ID = "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2"
WORK_ROOT = Path(__file__).resolve().parent / "fixtures" / "static-compatibility-work"


@pytest.fixture
def compatibility_workspace():
    """Use a normal workspace ACL rather than Windows tempfile's 0700 ACL."""

    path = WORK_ROOT / uuid.uuid4().hex
    path.mkdir(mode=0o755)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _automatic_reconstruction_pointer(tmp_path):
    source_hashes = ["a" * 64, "e" * 64]
    pose_generation = "b" * 64
    fusion_config_sha256 = "c" * 64
    source_artifacts = [
        {"relative_path": "segment_0.db3.zstd", "sha256": source_hashes[0], "bytes": 123},
        {"relative_path": "segment_1.db3.zstd", "sha256": source_hashes[1], "bytes": 456},
    ]
    identity_core = {
        "identity_schema_version": 1,
        "algorithm_version": "go2-offline-observed-surface-v1",
        "source_artifacts": source_artifacts,
        "source_artifact_sha256": source_hashes,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "fusion_config_sha256": fusion_config_sha256,
        "pose_generation": pose_generation,
    }
    identity = {**identity_core, "cache_key": canonical_sha256(identity_core)}
    output_root = tmp_path / "automatic-derived"
    generation = output_root / identity["cache_key"]
    artifacts = generation / "artifacts"
    artifacts.mkdir(parents=True)
    positions = np.asarray([[0.0, 0.0, -0.5], [1.0, 0.0, -0.5], [0.0, 0.0, 0.5]], dtype=np.float64)
    normals = np.asarray([[0.0, -1.0, 0.0]] * 3, dtype=np.float64)
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    source_vertex_ids = ("0:0:-1", "1:0:-1", "0:0:1")
    surface = MeasuredSurface(positions, normals, triangles, source_vertex_ids)
    published = LayerArtifactStore(artifacts).publish(
        surface,
        identity,
        annotations={
            "quality": "PREVIEW_UNCALIBRATED",
            "status": "PREVIEW_UNCALIBRATED",
            "sensor_frame_transform_id": TRANSFORM_ID,
            "volume_filled": False,
            "watertight": False,
            "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        },
    )
    geometry_identity = measured_surface_identity(
        positions=positions,
        normals=normals,
        triangles=triangles,
        pose_generation=pose_generation,
        artifact_sha256=source_hashes,
        fusion_config_sha256=fusion_config_sha256,
    )
    source_surface = generation / "measured-surface.json"
    _write_json(source_surface, {
        **surface.bounded_snapshot(),
        "artifact_sha256": list(reversed(source_hashes)),
        "pose_generation": pose_generation,
        "measured_surface_geometry_revision_id": geometry_identity.revision_id,
        "reconstruction_phase": "IDLE",
        "sensor_frame_transform_id": TRANSFORM_ID,
        "quality": "PREVIEW_UNCALIBRATED",
        "status": "PREVIEW_UNCALIBRATED",
        "volume_filled": False,
        "watertight": False,
        "voxel_size_m": 0.02,
    })
    transform = get_sensor_frame_transform(TRANSFORM_ID)
    _write_json(generation / "manifest.json", {
        "schema_version": 1,
        "status": "PREVIEW_UNCALIBRATED",
        "quality": "PREVIEW_UNCALIBRATED",
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "volume_filled": False,
        "watertight": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "sensor_frame_transform_id": TRANSFORM_ID,
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
        },
        "artifact": {
            "ply_name": published.ply_path.name,
            "metadata_name": published.metadata_path.name,
            "ply_sha256": published.ply_sha256,
        },
    })
    pointer_path = tmp_path / "runtime" / "automatic-reconstruction-current.json"
    pointer_path.parent.mkdir()
    pointer = {
        "schema_version": 1,
        "session_id": "segment-20260812T083241Z",
        "state": "INCREMENTAL_SURFACE_AVAILABLE",
        "mode": "high_fidelity_incremental_reconstruction",
        "finalized": False,
        "available": True,
        "blocker": None,
        "sensor_frame_transform_id": TRANSFORM_ID,
        "surface_json_path": str(source_surface),
        "surface_sha256": hashlib.sha256(source_surface.read_bytes()).hexdigest(),
        "derived_cache_key": identity["cache_key"],
        "ply_sha256": published.ply_sha256,
        "metadata_sha256": hashlib.sha256(published.metadata_path.read_bytes()).hexdigest(),
        "vertex_count": 3,
        "triangle_count": 1,
        "quality": "PREVIEW_UNCALIBRATED",
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
    }
    _write_json(pointer_path, pointer)
    class Preview:
        status = {
            "surface_sha256": pointer["surface_sha256"],
            "vertex_count": 3,
            "triangle_count": 1,
        }
    return output_root, pointer_path, source_surface, Preview(), pointer


def test_publish_verified_automatic_surface_as_content_addressed_static_mirror(compatibility_workspace):
    tmp_path = compatibility_workspace
    output_root, pointer_path, source_surface, preview, worker_status = _automatic_reconstruction_pointer(tmp_path)
    worker_status.update(
        {
            "state": "FINAL_SURFACE_AVAILABLE",
            "mode": "high_fidelity_batch_reconstruction",
            "finalized": True,
            "final_manifest_state": "VERIFIED",
        }
    )
    pointer_path.write_text(json.dumps(worker_status), encoding="utf-8")
    static_root = tmp_path / "dashboard-static"

    published = static_surface_compatibility.publish_verified_surface(
        status_path=pointer_path,
        output_root=output_root,
        static_root=static_root,
    )

    manifest_path = static_root / "compatibility" / "measured-surface-current.json"
    mirror_path = static_root / published["surface_relative_url"].removeprefix("/static/")
    assert manifest_path.is_file()
    assert mirror_path.is_file()
    assert mirror_path.read_bytes() == source_surface.read_bytes()
    assert published == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert published["schema_version"] == 1
    assert published["source_mode"] == "DERIVED_READ_ONLY"
    assert published["read_only"] is True
    assert published["point_authority"] == "LIDAR_MEASURED"
    assert published["geometry_authority"] is False
    assert published["hidden_surface_policy"] == "OBSERVED_POINTS_ONLY"
    assert published["quality"] == "PREVIEW_UNCALIBRATED"
    assert published["surface_sha256"] == preview.status["surface_sha256"]
    assert published["surface_sha256"] == hashlib.sha256(mirror_path.read_bytes()).hexdigest()
    assert published["vertex_count"] == preview.status["vertex_count"]
    assert published["triangle_count"] == preview.status["triangle_count"]
    assert published["session_id"] == worker_status["session_id"]


def test_static_publication_rejects_a_pointer_path_outside_the_verified_output_root(compatibility_workspace):
    tmp_path = compatibility_workspace
    output_root, pointer_path, _source_surface, _preview, _worker_status = _automatic_reconstruction_pointer(tmp_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["surface_json_relative_path"] = "../outside/measured-surface.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="relative surface path"):
        static_surface_compatibility.publish_verified_surface(
            status_path=pointer_path,
            output_root=output_root,
            static_root=tmp_path / "dashboard-static",
        )


def test_static_publication_does_not_publish_after_surface_bytes_change(compatibility_workspace):
    tmp_path = compatibility_workspace
    output_root, pointer_path, source_surface, _preview, _worker_status = _automatic_reconstruction_pointer(tmp_path)
    source_surface.write_bytes(source_surface.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="surface"):
        static_surface_compatibility.publish_verified_surface(
            status_path=pointer_path,
            output_root=output_root,
            static_root=tmp_path / "dashboard-static",
        )


def test_static_compatibility_manifest_is_written_atomically(compatibility_workspace, monkeypatch):
    tmp_path = compatibility_workspace
    output_root, pointer_path, _source_surface, _preview, _worker_status = _automatic_reconstruction_pointer(tmp_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer.update(
        {
            "state": "FINAL_SURFACE_AVAILABLE",
            "mode": "high_fidelity_batch_reconstruction",
            "finalized": True,
            "final_manifest_state": "VERIFIED",
        }
    )
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    static_root = tmp_path / "dashboard-static"
    replaced = []
    original_replace = static_surface_compatibility.os.replace

    def record_replace(source, destination):
        replaced.append((Path(source), Path(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(static_surface_compatibility.os, "replace", record_replace)

    static_surface_compatibility.publish_verified_surface(
        status_path=pointer_path,
        output_root=output_root,
        static_root=static_root,
    )

    assert any(destination.name == "measured-surface-current.json" for _source, destination in replaced)
    assert all(source.suffix == ".tmp" for source, _destination in replaced)


def test_static_publication_rejects_an_incremental_pointer(compatibility_workspace):
    output_root, pointer_path, _source_surface, _preview, _worker_status = _automatic_reconstruction_pointer(
        compatibility_workspace
    )

    with pytest.raises(ValueError, match="final verified automatic surface"):
        static_surface_compatibility.publish_verified_surface(
            status_path=pointer_path,
            output_root=output_root,
            static_root=compatibility_workspace / "dashboard-static",
        )
