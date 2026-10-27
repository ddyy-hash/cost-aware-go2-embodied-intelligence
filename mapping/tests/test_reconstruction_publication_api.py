import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from mapping.pc.go2_mapping_appearance import measured_surface_identity
from mapping.pc.go2_mapping_appearance.historical_store import HistoricalAppearanceStore
from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore
from mapping.pc.go2_mapping_rgbd.artifact_refinement import AppearanceEvidence
from mapping.pc.go2_mapping_rgbd.texture import VertexColor
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, load_reconstruction_config


def _surface_snapshot(*, phase="IDLE"):
    return {
        "source_layer": "LIDAR_MEASURED",
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "pose_generation": "pg-7",
        "reconstruction_phase": phase,
        "artifact_sha256": ["a" * 64, "b" * 64],
        "voxel_size_m": 0.015,
        "positions": [0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.015, 0.0],
        "normals": [0.0, 0.0, 1.0] * 3,
        "triangles": [0, 1, 2],
    }


class _Mapper:
    def __init__(self, surface):
        self.surface = surface

    def measured_surface_snapshot(self):
        return self.surface

    def snapshot(self):
        return {
            "quality": "DENSE_LIDAR_PREVIEW_UNCALIBRATED",
            "reconstruction_phase": self.surface.get("reconstruction_phase") if self.surface else "IDLE",
            "pose_generation": self.surface.get("pose_generation") if self.surface else "pg-7",
            "artifacts": len(self.surface.get("artifact_sha256", ())) if self.surface else 0,
            "quality_evidence": {
                "level": "PREVIEW_UNCALIBRATED",
                "calibration_sha256": None,
                "blocking_reasons": ["CALIBRATION_MISSING"],
                "eligible_for": {"centimetre_geometry": False},
            },
        }


def test_publish_measured_surface_is_hash_bound_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "dense_mapper", _Mapper(_surface_snapshot()))
    monkeypatch.setattr(server, "reconstruction_store", LayerArtifactStore(tmp_path))

    result = server.publish_measured_reconstruction()
    metadata = json.loads((tmp_path / result["metadata_name"]).read_text(encoding="utf-8"))

    assert result["quality_label"] == "PREVIEW_UNCALIBRATED"
    assert result["delivery_authorized"] is False
    assert result["ply_url"].endswith(result["ply_name"])
    assert metadata["derived_identity"]["artifact_hashes"] == ["a" * 64, "b" * 64]
    assert metadata["derived_identity"]["pose_generation"] == "pg-7"
    assert metadata["quality_label"] == "PREVIEW_UNCALIBRATED"
    assert metadata["quality_blockers"] == ["CALIBRATION_MISSING"]
    assert metadata["delivery_authorized"] is False
    assert metadata["point_authority"] == "LIDAR_MEASURED"


def test_publish_rejects_a_surface_while_reconstruction_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "dense_mapper", _Mapper(_surface_snapshot(phase="REINTEGRATING_OPTIMIZED_GENERATION")))
    monkeypatch.setattr(server, "reconstruction_store", LayerArtifactStore(tmp_path))

    with pytest.raises(HTTPException) as error:
        server.publish_measured_reconstruction()

    assert error.value.status_code == 409
    assert not list(tmp_path.iterdir())


def test_publish_rejects_missing_measured_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "dense_mapper", _Mapper(None))
    monkeypatch.setattr(server, "reconstruction_store", LayerArtifactStore(tmp_path))

    with pytest.raises(HTTPException) as error:
        server.publish_measured_reconstruction()

    assert error.value.status_code == 423


def test_published_artifacts_are_listed_and_resolved_without_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "dense_mapper", _Mapper(_surface_snapshot()))
    monkeypatch.setattr(server, "reconstruction_store", LayerArtifactStore(tmp_path))
    published = server.publish_measured_reconstruction()

    listing = server.reconstruction_artifacts()
    response = server.reconstruction_artifact(published["ply_name"])

    assert [item["metadata_name"] for item in listing["artifacts"]] == [published["metadata_name"]]
    assert Path(response.path).resolve() == (tmp_path / published["ply_name"]).resolve()
    with pytest.raises(HTTPException) as error:
        server.reconstruction_artifact("../outside.ply")
    assert error.value.status_code == 404


def test_measured_appearance_endpoint_is_blocked_without_content_addressed_geometry(monkeypatch):
    monkeypatch.setattr(server, "dense_mapper", _Mapper(_surface_snapshot()))

    manifest = server.measured_surface_appearance()

    assert manifest["state"] == "TEXTURE_NOT_AUTHORIZED"
    assert "POSE_GENERATION_NOT_CONTENT_ADDRESSED" in manifest["blockers"]


def test_measured_appearance_endpoint_publishes_exact_matching_sidecar(tmp_path, monkeypatch):
    surface = _surface_snapshot()
    surface["pose_generation"] = "b" * 64
    surface["source_vertex_ids"] = ["0:0:0", "1:0:0", "0:1:0"]
    identity = measured_surface_identity(
        positions=np.asarray(surface["positions"]).reshape(-1, 3),
        normals=np.asarray(surface["normals"]).reshape(-1, 3),
        triangles=np.asarray(surface["triangles"]).reshape(-1, 3),
        pose_generation=surface["pose_generation"],
        artifact_sha256=surface["artifact_sha256"],
        fusion_config_sha256=load_reconstruction_config().fusion_sha256,
    )
    surface["measured_surface_geometry_revision_id"] = identity.revision_id
    surface["appearance_blocker"] = None
    colors = tuple(
        VertexColor(vertex_id, (0.2, 0.4, 0.6), (0.0, 0.0, 0.0), 2, 0, 2, ("f" * 64,), True)
        for vertex_id in surface["source_vertex_ids"]
    )

    class Refiner:
        def appearance_evidence(self, vertex_ids):
            assert tuple(vertex_ids) == tuple(surface["source_vertex_ids"])
            return AppearanceEvidence(True, colors, ("f" * 64,), 2, "d" * 64, "cal-1", "c" * 64, "b" * 64, ())

    monkeypatch.setattr(server, "dense_mapper", _Mapper(surface))
    monkeypatch.setattr(server, "rgbd_refiner", Refiner())
    monkeypatch.setattr(server, "appearance_store", HistoricalAppearanceStore(tmp_path))

    manifest = server.measured_surface_appearance()
    response = server.reconstruction_vertex_color(manifest["appearance_revision_id"])

    assert manifest["delivery_authorized"] is True
    assert manifest["measured_surface_geometry_revision_id"] == identity.revision_id
    assert manifest["manifest_sha256"]
    assert json.loads(manifest["manifest_canonical_json"])["surface_coverage"] == 1.0
    assert canonical_sha256(json.loads(manifest["manifest_canonical_json"])) == manifest["manifest_sha256"]
    assert str(response.path).endswith("vertex-color.bin")
