from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.derived_provisional_preview import (
    ProvisionalProjectionEvidence,
    publish_provisional_surface_preview,
)
from mapping.pc.go2_mapping_dashboard import provisional_surface_preview
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2, get_sensor_frame_transform
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_appearance import measured_surface_identity
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore


WORK_ROOT = Path(__file__).resolve().parent / "fixtures" / "provisional-surface-preview-work"
SESSION_ID = "segment-20260812T083241Z"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def preview_workspace():
    path = WORK_ROOT / uuid.uuid4().hex
    path.mkdir(mode=0o755, parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _formal_surface(tmp_path: Path):
    source_hash = "a" * 64
    pose_generation = "b" * 64
    fusion_config_sha256 = "c" * 64
    artifact = {"relative_path": "segment_0.db3.zstd", "sha256": source_hash, "bytes": 123}
    identity_core = {
        "identity_schema_version": 1,
        "algorithm_version": "go2-offline-observed-surface-v1",
        "source_artifacts": [artifact],
        "source_artifact_sha256": [source_hash],
        "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        "fusion_config_sha256": fusion_config_sha256,
        "pose_generation": pose_generation,
    }
    identity = {**identity_core, "cache_key": canonical_sha256(identity_core)}
    output_root = tmp_path / "formal-derived"
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
            "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
            "volume_filled": False,
            "watertight": False,
            "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        },
    )
    geometry = measured_surface_identity(
        positions=positions,
        normals=normals,
        triangles=triangles,
        pose_generation=pose_generation,
        artifact_sha256=[source_hash],
        fusion_config_sha256=fusion_config_sha256,
    )
    surface_path = generation / "measured-surface.json"
    _write_json(surface_path, {
        **surface.bounded_snapshot(),
        "artifact_sha256": [source_hash],
        "pose_generation": pose_generation,
        "measured_surface_geometry_revision_id": geometry.revision_id,
        "reconstruction_phase": "IDLE",
        "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        "quality": "PREVIEW_UNCALIBRATED",
        "status": "PREVIEW_UNCALIBRATED",
        "volume_filled": False,
        "watertight": False,
        "voxel_size_m": 0.02,
    })
    transform = get_sensor_frame_transform(PROVISIONAL_GO2_UTLIDAR_PITCH180_V2)
    _write_json(generation / "manifest.json", {
        "schema_version": 1,
        "status": "PREVIEW_UNCALIBRATED",
        "quality": "PREVIEW_UNCALIBRATED",
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "volume_filled": False,
        "watertight": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        "sensor_frame_transform": {
            "id": transform.id,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
            "provisional": True,
            "description": transform.description,
        },
        "source_root_name": "segment",
        "source_artifacts": [artifact],
        "derived_identity": identity,
        "surface_available": True,
        "surface": {"vertex_count": 3, "triangle_count": 1, "source_vertex_count": 3, "geometry_revision_id": geometry.revision_id},
        "artifact": {
            "ply_name": published.ply_path.name,
            "metadata_name": published.metadata_path.name,
            "ply_sha256": published.ply_sha256,
        },
    })
    pointer_path = tmp_path / "runtime" / "automatic-reconstruction-current.json"
    pointer_path.parent.mkdir()
    _write_json(pointer_path, {
        "schema_version": 1,
        "session_id": SESSION_ID,
        "state": "FINAL_SURFACE_AVAILABLE",
        "mode": "high_fidelity_batch_reconstruction",
        "finalized": True,
        "final_manifest_state": "VERIFIED",
        "available": True,
        "blocker": None,
        "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        "surface_json_path": str(surface_path),
        "surface_sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        "derived_cache_key": identity["cache_key"],
        "ply_sha256": published.ply_sha256,
        "metadata_sha256": hashlib.sha256(published.metadata_path.read_bytes()).hexdigest(),
        "vertex_count": 3,
        "triangle_count": 1,
        "quality": "PREVIEW_UNCALIBRATED",
        "geometry_authority": False,
        "source_layer": "LIDAR_MEASURED",
        "point_authority": "LIDAR_MEASURED",
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
    })
    return output_root, pointer_path, source_vertex_ids, geometry.revision_id


def _preview(
    preview_root: Path,
    *,
    surface_sha256: str,
    geometry_revision_id: str,
    source_vertex_ids: tuple[str, ...],
    source_artifact_sha256: tuple[str, ...] = ("a" * 64,),
):
    return publish_provisional_surface_preview(
        preview_root,
        ProvisionalProjectionEvidence(
            session_id=SESSION_ID,
            source_surface_sha256=surface_sha256,
            measured_surface_geometry_revision_id=geometry_revision_id,
            sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
            voxel_size_m=0.02,
            source_artifact_sha256=source_artifact_sha256,
            source_vertex_ids=source_vertex_ids,
            rgb=np.asarray([[17, 34, 51], [0, 0, 0], [68, 85, 102]], dtype=np.uint8),
            validity=np.asarray([True, False, True], dtype=np.bool_),
            calibration_id="PROVISIONAL_AUTOCAL_PENDING_v1",
            operator_override_sha256="d" * 64,
            runtime_projection_authorized=True,
            measurement_authority=False,
            color_camera_info_sha256="e" * 64,
            color_source_frame_sha256=("f" * 64,),
            lidar_frame_count=2,
            color_frame_count=2,
            paired_frame_count=2,
            projected_point_count=3,
            matched_surface_point_count=2,
        ),
    )


def test_static_publication_binds_preview_to_current_read_only_surface_and_preserves_pointer(preview_workspace):
    output_root, pointer_path, source_vertex_ids, geometry = _formal_surface(preview_workspace)
    surface_sha = json.loads(pointer_path.read_text(encoding="utf-8"))["surface_sha256"]
    preview = _preview(preview_workspace / "derived-provisional-appearance", surface_sha256=surface_sha, geometry_revision_id=geometry, source_vertex_ids=source_vertex_ids)
    pointer_before = pointer_path.read_bytes()
    static_root = preview_workspace / "dashboard-static"

    published = provisional_surface_preview.publish_current_provisional_surface_preview(
        status_path=pointer_path,
        output_root=output_root,
        preview_manifest_path=preview.manifest_path,
        static_root=static_root,
    )

    manifest_path = static_root / "provisional-surface-preview" / "current.json"
    sidecar_path = static_root / published["vertex_color_relative_url"].removeprefix("/static/")
    copied_preview_manifest = static_root / published["preview_manifest_relative_url"].removeprefix("/static/")
    assert pointer_path.read_bytes() == pointer_before
    assert published == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert published["delivery_authorized"] is False
    assert published["measurement_authority"] is False
    assert published["final_store_eligible"] is False
    assert published["source_surface_sha256"] == surface_sha
    assert published["measured_surface_geometry_revision_id"] == geometry
    assert published["vertex_count"] == 3
    assert published["colored_vertex_count"] == 2
    assert sidecar_path.read_bytes() == preview.vertex_color_path.read_bytes()
    assert copied_preview_manifest.read_bytes() == preview.manifest_path.read_bytes()
    assert published["vertex_color_sha256"] == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def test_static_publication_rejects_stale_preview_before_writing_current_manifest(preview_workspace):
    output_root, pointer_path, source_vertex_ids, geometry = _formal_surface(preview_workspace)
    preview = _preview(
        preview_workspace / "derived-provisional-appearance",
        surface_sha256="9" * 64,
        geometry_revision_id=geometry,
        source_vertex_ids=source_vertex_ids,
    )
    static_root = preview_workspace / "dashboard-static"

    with pytest.raises(ValueError, match="source surface does not match"):
        provisional_surface_preview.publish_current_provisional_surface_preview(
            status_path=pointer_path,
            output_root=output_root,
            preview_manifest_path=preview.manifest_path,
            static_root=static_root,
        )

    assert not (static_root / "provisional-surface-preview" / "current.json").exists()


def test_static_publication_rejects_preview_with_a_different_source_artifact_inventory(preview_workspace):
    output_root, pointer_path, source_vertex_ids, geometry = _formal_surface(preview_workspace)
    surface_sha = json.loads(pointer_path.read_text(encoding="utf-8"))["surface_sha256"]
    preview = _preview(
        preview_workspace / "derived-provisional-appearance",
        surface_sha256=surface_sha,
        geometry_revision_id=geometry,
        source_vertex_ids=source_vertex_ids,
        source_artifact_sha256=("9" * 64,),
    )
    static_root = preview_workspace / "dashboard-static"

    with pytest.raises(ValueError, match="source artifact inventory"):
        provisional_surface_preview.publish_current_provisional_surface_preview(
            status_path=pointer_path,
            output_root=output_root,
            preview_manifest_path=preview.manifest_path,
            static_root=static_root,
        )

    assert not (static_root / "provisional-surface-preview" / "current.json").exists()


def test_static_preview_page_is_an_explicitly_uncalibrated_non_delivery_inspector():
    static_root = Path(__file__).resolve().parents[1] / "pc" / "go2_mapping_dashboard" / "static"
    page = (static_root / "provisional-surface-preview.html").read_text(encoding="utf-8")
    app = (static_root / "provisional-surface-preview.js").read_text(encoding="utf-8")

    assert "PROVISIONAL / UNCALIBRATED RGB-D COLOR PREVIEW" in page
    assert "NOT FINAL TEXTURE" in page
    assert "NOT MEASUREMENT AUTHORITY" in page
    for fragment in (
        "/api/reconstruction/measured-surface/status",
        "/api/reconstruction/measured-surface",
        "/static/provisional-surface-preview/current.json",
        "G2PVCOL1",
        "delivery_authorized !== false",
        "measurement_authority !== false",
        "final_store_eligible !== false",
        "source_surface_sha256",
        "measured_surface_geometry_revision_id",
        "vertex_count",
        "neutral",
    ):
        assert fragment in app
    assert "export-textured" not in app
    assert ".glb" not in app


def test_static_preview_page_reads_lidar_authority_from_the_hashed_surface_payload_not_the_status_summary():
    static_root = Path(__file__).resolve().parents[1] / "pc" / "go2_mapping_dashboard" / "static"
    app = (static_root / "provisional-surface-preview.js").read_text(encoding="utf-8")
    status_start = app.index("function validateSurfaceStatus(")
    status_end = app.index("function validateSurfacePayload(")
    status_validator = app[status_start:status_end]
    payload_end = app.index("async function loadCurrentSurface(")
    payload_validator = app[status_end:payload_end]

    assert "status.point_authority" not in status_validator
    assert 'surface.point_authority === "LIDAR_MEASURED"' in payload_validator
    assert 'surface.source_layer === "LIDAR_MEASURED"' in payload_validator


def test_static_preview_mobile_layout_separates_surface_label_tools_and_verification_status():
    static_root = Path(__file__).resolve().parents[1] / "pc" / "go2_mapping_dashboard" / "static"
    css = (static_root / "provisional-surface-preview.css").read_text(encoding="utf-8")
    mobile_start = css.index("@media (max-width: 639px)")
    mobile_end = css.index("@media (min-width: 768px)")
    mobile = css[mobile_start:mobile_end]

    assert ".stage-label { max-width: calc(100% - 1.5rem); }" in mobile
    assert ".view-tools { top: 3.625rem; }" in mobile
    assert ".surface-status { top: 6.5rem; }" in mobile


def test_static_preview_publication_tool_runs_directly_from_the_workspace_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "publish_go2_provisional_surface_preview.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--preview-manifest-path" in result.stdout
    assert "--static-root" in result.stdout
