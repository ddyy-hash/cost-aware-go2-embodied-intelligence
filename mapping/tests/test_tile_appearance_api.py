import hashlib
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
from fastapi import HTTPException

from mapping.pc.go2_mapping_appearance.atlas_codec import (
    encode_render_indices_v1,
    encode_render_uvs_v1,
    encode_render_vertices_v1,
)
from mapping.pc.go2_mapping_appearance.tile_identity import (
    ArtifactRef,
    PrimitiveRecord,
    build_lod_record,
    tile_appearance_revision,
)
from mapping.pc.go2_mapping_appearance.tile_store import TileAppearanceStore, TileLodPayload
from mapping.pc.go2_mapping_appearance.tile_evidence import ValidatedTileAppearanceProcessor
from mapping.pc.go2_mapping_appearance.tile_vertex_color import (
    TileVertexColorStore,
    build_tile_vertex_color_sidecar,
)
from mapping.pc.go2_mapping_dashboard import server
from mapping.pc.go2_mapping_rgbd.texture import VertexColor


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


class _TileStore:
    def load(self, revision_id):
        raise AssertionError(f"unexpected tile load: {revision_id}")


class _RefinerSnapshot:
    def __init__(self, calibration_sha256):
        self.calibration_sha256 = calibration_sha256

    def snapshot(self):
        return {"calibration_sha256": self.calibration_sha256}


def test_tile_atlas_processor_is_disabled_when_feature_flag_is_off(monkeypatch):
    monkeypatch.setattr(server, "_tile_atlas_enabled", False)
    monkeypatch.setattr(server, "rgbd_refiner", _RefinerSnapshot(HASHES[3]))

    assert server._tile_appearance_processor(_TileStore()) is None


def test_tile_atlas_processor_requires_current_measured_calibration(monkeypatch):
    monkeypatch.setattr(server, "_tile_atlas_enabled", True)
    monkeypatch.setattr(server, "rgbd_refiner", _RefinerSnapshot(None))

    assert server._tile_appearance_processor(_TileStore()) is None


def test_tile_atlas_processor_is_created_only_with_explicit_enable_and_calibration(monkeypatch):
    tile_store = _TileStore()
    refiner = _RefinerSnapshot(HASHES[3])
    monkeypatch.setattr(server, "_tile_atlas_enabled", True)
    monkeypatch.setattr(server, "rgbd_refiner", refiner)

    processor = server._tile_appearance_processor(tile_store)

    assert isinstance(processor, ValidatedTileAppearanceProcessor)
    assert processor.tile_store is tile_store
    assert processor.rgbd_refiner is refiner
    assert processor.publication_scope == "LOCAL"


def test_runtime_enables_tile_atlas_after_calibration_arrives(monkeypatch):
    class Pipeline:
        tile_store = _TileStore()

        def __init__(self):
            self.processor = None
            self.refreshes = 0

        def appearance_snapshot(self):
            return {"enabled": self.processor is not None}

        def configure_appearance_processor(self, processor):
            self.processor = processor

        def refresh_appearance_evidence(self):
            self.refreshes += 1

    pipeline = Pipeline()
    refiner = _RefinerSnapshot(None)
    monkeypatch.setattr(server, "_tile_atlas_enabled", True)
    monkeypatch.setattr(server, "rgbd_refiner", refiner)
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)

    assert server._refresh_tile_appearance_scheduler() is False
    assert pipeline.processor is None
    refiner.calibration_sha256 = HASHES[3]
    assert server._refresh_tile_appearance_scheduler() is True
    assert isinstance(pipeline.processor, ValidatedTileAppearanceProcessor)
    assert pipeline.refreshes == 1


def _png():
    image = Image.new("RGBA", (8, 8), (20, 40, 60, 255))
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _publish(tmp_path, monkeypatch, *, building_delivery_eligible=True):
    revision = tile_appearance_revision(
        tile_revision_id=HASHES[0],
        geometry_revision_id=HASHES[1],
        source_generation_id=HASHES[2],
        calibration_id="cal-1",
        calibration_sha256=HASHES[3],
        physical_depth_validation_revision=HASHES[4],
        pose_revision_id=HASHES[5],
        model_revision_id=None,
        assistance_config_sha256=None,
        source_frame_sha256=(HASHES[6], HASHES[7]),
        appearance_config_sha256=HASHES[8],
        charting={"implementation": "xatlas", "version": "0.0.11", "options": {}},
        atlas_density_policy={"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        padding_policy={"pixels": 2, "mip_levels": 1},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=None,
        contains_augmented_primitives=False,
        measured_surface_coverage_ppm=800000,
        minimum_independent_view_count=2,
    )
    positions = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
    artifacts = {
        "render-vertices.bin": encode_render_vertices_v1(
            HASHES[0], revision.appearance_revision_id, positions, np.asarray([[0, 0, 1]] * 3), [0, 1, 2]
        ),
        "render-indices.bin": encode_render_indices_v1(
            HASHES[0], revision.appearance_revision_id, np.asarray([[0, 1, 2]], dtype=np.uint32), [0]
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            HASHES[0], revision.appearance_revision_id, np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]), [0, 0, 0], [1, 1, 1]
        ),
        "atlas-0.png": _png(),
    }
    references = tuple(ArtifactRef(name, hashlib.sha256(body).hexdigest(), len(body)) for name, body in artifacts.items())
    record = build_lod_record(
        tile_revision_id=HASHES[0],
        appearance_revision_id=revision.appearance_revision_id,
        policy={"name": "near", "order": 0, "mm_per_texel": 4},
        vertex_count=3,
        triangle_count=1,
        atlas_count=1,
        artifacts=references,
        primitives=(PrimitiveRecord(0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,)),),
        rgbd_augmented_revision_id=None,
    )
    appearance_store = TileAppearanceStore(tmp_path)
    published = appearance_store.publish(
        revision,
        (TileLodPayload(record, artifacts),),
        building_delivery_eligible=building_delivery_eligible,
    )
    colors = tuple(
        VertexColor(f"{HASHES[0]}:{index}", (0.2, 0.4, 0.6), (0.01, 0.01, 0.01), 2, 0, 2, (HASHES[6], HASHES[7]), True)
        for index in range(3)
    )
    preview_store = TileVertexColorStore(tmp_path)
    preview_store.publish(build_tile_vertex_color_sidecar(revision=revision, vertex_colors=colors))
    monkeypatch.setattr(server, "tile_appearance_store", appearance_store)
    monkeypatch.setattr(server, "tile_vertex_color_store", preview_store)
    return revision, record, published


def test_current_and_exact_manifest_routes_have_distinct_resolution_and_cache_semantics(tmp_path, monkeypatch):
    revision, _, published = _publish(tmp_path, monkeypatch)

    current = server.building_tile_appearance_current(
        HASHES[0], expected_source_generation_id=HASHES[2]
    )
    exact = server.building_tile_appearance_manifest(HASHES[0], revision.appearance_revision_id)
    (published.root_path.parent / "CURRENT.json").unlink()
    exact_without_current = server.building_tile_appearance_manifest(
        HASHES[0], revision.appearance_revision_id
    )

    assert current.media_type == exact.media_type == "application/json"
    assert current.headers["cache-control"] == "no-store"
    assert exact.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert exact.body == exact_without_current.body == published.manifest_path.read_bytes()


def test_exact_vertex_color_and_lod_artifacts_are_verified_and_immutable(tmp_path, monkeypatch):
    revision, record, _ = _publish(tmp_path, monkeypatch)

    vertex = server.building_tile_vertex_color(HASHES[0], revision.appearance_revision_id)
    vertices = server.building_tile_appearance_artifact(
        HASHES[0], revision.appearance_revision_id, record.lod_id, "render-vertices.bin"
    )
    atlas = server.building_tile_appearance_artifact(
        HASHES[0], revision.appearance_revision_id, record.lod_id, "atlas-0.png"
    )

    assert vertex.media_type == vertices.media_type == "application/octet-stream"
    assert atlas.media_type == "image/png"
    for response in (vertex, vertices, atlas):
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert len(response.headers["etag"].strip('"')) == 64


def test_appearance_routes_enforce_status_and_allowlist_contract(tmp_path, monkeypatch):
    revision, record, _ = _publish(tmp_path, monkeypatch)

    cases = [
        (lambda: server.building_tile_appearance_current("BAD"), 422),
        (lambda: server.building_tile_appearance_current(HASHES[0], expected_source_generation_id="BAD"), 422),
        (lambda: server.building_tile_appearance_current(HASHES[0], expected_source_generation_id=HASHES[8]), 409),
        (lambda: server.building_tile_appearance_manifest(HASHES[0], "BAD"), 422),
        (lambda: server.building_tile_appearance_manifest(HASHES[0], "9" * 64), 404),
        (lambda: server.building_tile_appearance_artifact(HASHES[0], revision.appearance_revision_id, "BAD", "render-uvs.bin"), 422),
        (lambda: server.building_tile_appearance_artifact(HASHES[0], revision.appearance_revision_id, record.lod_id, "atlas-01.png"), 422),
        (lambda: server.building_tile_appearance_artifact(HASHES[0], revision.appearance_revision_id, record.lod_id, "secret.txt"), 404),
        (lambda: server.building_tile_appearance_artifact(HASHES[0], revision.appearance_revision_id, "9" * 64, "render-indices.bin"), 404),
    ]
    for invoke, expected in cases:
        with pytest.raises(HTTPException) as error:
            invoke()
        assert error.value.status_code == expected


def test_artifact_route_rechecks_hash_after_resolver_to_close_read_race(tmp_path, monkeypatch):
    revision, record, published = _publish(tmp_path, monkeypatch)
    original_resolve = server.tile_appearance_store.resolve

    def corrupt_after_resolve(tile, appearance):
        resolved = original_resolve(tile, appearance)
        target = resolved.lod_paths[record.lod_id] / "render-indices.bin"
        target.write_bytes(target.read_bytes() + b"corrupt")
        return resolved

    monkeypatch.setattr(server.tile_appearance_store, "resolve", corrupt_after_resolve)

    with pytest.raises(HTTPException) as error:
        server.building_tile_appearance_artifact(
            HASHES[0],
            revision.appearance_revision_id,
            record.lod_id,
            "render-indices.bin",
        )
    assert error.value.status_code == 404


def test_building_routes_reject_local_or_detached_appearance(tmp_path, monkeypatch):
    revision, record, _ = _publish(
        tmp_path,
        monkeypatch,
        building_delivery_eligible=False,
    )

    calls = (
        lambda: server.building_tile_appearance_current(HASHES[0]),
        lambda: server.building_tile_appearance_manifest(
            HASHES[0], revision.appearance_revision_id
        ),
        lambda: server.building_tile_vertex_color(
            HASHES[0], revision.appearance_revision_id
        ),
        lambda: server.building_tile_appearance_artifact(
            HASHES[0],
            revision.appearance_revision_id,
            record.lod_id,
            "atlas-0.png",
        ),
    )
    for invoke in calls:
        with pytest.raises(HTTPException) as error:
            invoke()
        assert error.value.status_code == 404
