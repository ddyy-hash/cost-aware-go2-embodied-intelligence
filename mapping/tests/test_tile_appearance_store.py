import hashlib
from io import BytesIO
import json
import threading

import numpy as np
from PIL import Image
import pytest

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
from mapping.pc.go2_mapping_appearance.tile_store import (
    TileAppearanceStore,
    TileLodPayload,
)


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


def _png(color=(20, 40, 60, 255)):
    image = Image.new("RGBA", (8, 8), color)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _generation(*, coverage=800000, views=2, png=None):
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
        padding_policy={"pixels": 8, "mip_levels": 4},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=None,
        contains_augmented_primitives=False,
        measured_surface_coverage_ppm=coverage,
        minimum_independent_view_count=views,
    )
    positions = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
    normals = np.asarray([[0, 0, 1]] * 3, dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.uint32)
    source_ids = np.asarray([0, 1, 2], dtype=np.uint32)
    source_triangles = np.asarray([0], dtype=np.uint32)
    uvs = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    atlas_indices = np.zeros(3, dtype=np.uint16)
    flags = np.ones(3, dtype=np.uint16)
    common = {
        "render-vertices.bin": encode_render_vertices_v1(
            HASHES[0], revision.appearance_revision_id, positions, normals, source_ids
        ),
        "render-indices.bin": encode_render_indices_v1(
            HASHES[0], revision.appearance_revision_id, triangles, source_triangles
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            HASHES[0], revision.appearance_revision_id, uvs, atlas_indices, flags
        ),
        "atlas-0.png": _png() if png is None else png,
    }
    payloads = []
    for order, name in enumerate(("near", "far")):
        artifacts = tuple(
            ArtifactRef(key, hashlib.sha256(value).hexdigest(), len(value))
            for key, value in common.items()
        )
        primitive = PrimitiveRecord(
            0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,)
        )
        lod = build_lod_record(
            tile_revision_id=HASHES[0],
            appearance_revision_id=revision.appearance_revision_id,
            policy={"name": name, "order": order, "mm_per_texel": 3 + 2 * order},
            vertex_count=3,
            triangle_count=1,
            atlas_count=1,
            artifacts=artifacts,
            primitives=(primitive,),
            rgbd_augmented_revision_id=None,
        )
        payloads.append(TileLodPayload(lod, common))
    return revision, tuple(payloads)


def test_store_publishes_all_lods_reuses_and_reloads_exact_generation(tmp_path):
    revision, lods = _generation()
    store = TileAppearanceStore(tmp_path)

    first = store.publish(revision, lods)
    repeated = store.publish(revision, lods)
    restarted = TileAppearanceStore(tmp_path).resolve(HASHES[0], revision.appearance_revision_id)
    current = TileAppearanceStore(tmp_path).current(HASHES[0], expected_source_generation_id=HASHES[2])

    assert first == repeated == restarted == current
    assert first.manifest_sha256 == hashlib.sha256(first.manifest_path.read_bytes()).hexdigest()
    assert "manifest_sha256" not in first.manifest
    assert first.pointer["lod_ids"] == [value.record.lod_id for value in lods]
    assert first.pointer["delivery_authorization"]["delivery_authorized"] is True
    assert first.root_path == tmp_path / "appearance" / HASHES[0] / revision.appearance_revision_id
    assert not list(tmp_path.rglob("*.partial"))


def test_store_recomputes_authorization_and_rejects_stale_current_or_corrupt_artifacts(tmp_path):
    revision, lods = _generation(coverage=799999, views=1)
    store = TileAppearanceStore(tmp_path)
    published = store.publish(revision, lods)

    assert published.pointer["delivery_authorization"]["delivery_authorized"] is False
    with pytest.raises(ValueError, match="stale"):
        store.current(HASHES[0], expected_source_generation_id=HASHES[3])

    atlas = published.root_path / "lods" / lods[0].record.lod_id / "atlas-0.png"
    atlas.write_bytes(atlas.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact|hash|verification"):
        store.resolve(HASHES[0], revision.appearance_revision_id)


def test_existing_identity_conflict_is_rejected(tmp_path):
    revision, lods = _generation()
    store = TileAppearanceStore(tmp_path)
    published = store.publish(revision, lods)
    published.manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        store.publish(revision, lods)


def test_store_rejects_tampered_publication_authorization(tmp_path):
    revision, lods = _generation()
    store = TileAppearanceStore(tmp_path)
    published = store.publish(
        revision, lods, building_delivery_eligible=False
    )
    manifest = dict(published.manifest)
    manifest["publication_authorization"] = {
        "building_delivery_eligible": False,
        "building_authorized": True,
        "export_authorized": True,
        "blockers": [],
    }
    published.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publication authorization"):
        store.resolve(HASHES[0], revision.appearance_revision_id)


@pytest.mark.parametrize("point", ["after_files_fsynced", "after_generation_rename", "after_pointer_replace"])
def test_crash_points_leave_only_recoverable_complete_state(tmp_path, point):
    revision, lods = _generation()

    def crash(actual):
        if actual == point:
            raise RuntimeError(point)

    store = TileAppearanceStore(tmp_path, fault_injector=crash)
    with pytest.raises(RuntimeError, match=point):
        store.publish(revision, lods)

    restarted = TileAppearanceStore(tmp_path)
    if point == "after_files_fsynced":
        with pytest.raises(FileNotFoundError):
            restarted.resolve(HASHES[0], revision.appearance_revision_id)
    else:
        resolved = restarted.resolve(HASHES[0], revision.appearance_revision_id)
        assert resolved.appearance_revision_id == revision.appearance_revision_id
        retried = restarted.publish(revision, lods)
        assert restarted.current(HASHES[0]) == retried
    assert not list(tmp_path.rglob("*.partial"))


def test_concurrent_different_revisions_leave_both_generations_valid_and_one_complete_pointer(tmp_path):
    first_revision, first_lods = _generation()
    second_revision, second_lods = _generation(coverage=800001, png=_png((80, 20, 120, 255)))
    store = TileAppearanceStore(tmp_path)
    errors = []

    def publish(revision, lods):
        try:
            store.publish(revision, lods)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=publish, args=(first_revision, first_lods)),
        threading.Thread(target=publish, args=(second_revision, second_lods)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.resolve(HASHES[0], first_revision.appearance_revision_id)
    assert store.resolve(HASHES[0], second_revision.appearance_revision_id)
    assert store.current(HASHES[0]).appearance_revision_id in {
        first_revision.appearance_revision_id,
        second_revision.appearance_revision_id,
    }
