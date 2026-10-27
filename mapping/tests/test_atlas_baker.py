import hashlib
from io import BytesIO
import struct

import numpy as np
from PIL import Image
import pytest

from mapping.pc.go2_mapping_appearance.atlas_codec import (
    parse_render_indices_v1,
    parse_render_uvs_v1,
    parse_render_vertices_v1,
)
from mapping.pc.go2_mapping_appearance.baker import (
    StaleTileGenerationError,
    TileAtlasBakeRequest,
    TileAtlasBaker,
)
from mapping.pc.go2_mapping_appearance.tile_store import TileAppearanceStore
from mapping.pc.go2_mapping_appearance.tile_vertex_color import TileVertexColorStore
from mapping.pc.go2_mapping_rgbd.texture import VertexColor
from mapping.pc.go2_mapping_rgbd.augmentation import AugmentedLayer


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3", "4"))


def _tile_body():
    positions = np.asarray([[0, 0, 0], [0.04, 0, 0], [0.04, 0.04, 0], [0, 0.04, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 4, dtype="<f4")
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<u4")
    scales = np.asarray([0.004, 0.004], dtype="<f4")
    return b"".join((struct.pack("<4sIIII", b"G2TM", 1, 4, 2, 2), positions.tobytes(), normals.tobytes(), triangles.tobytes(), scales.tobytes()))


def _metadata(body, *, generation=HASHES[2]):
    return {
        "revision_id": HASHES[0],
        "geometry_revision_id": HASHES[1],
        "source_generation_id": generation,
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "vertex_count": 4,
        "triangle_count": 2,
        "building_id": "building-a",
        "floor_id": "floor-1",
        "component_id": "component-a",
        "submap_id": "submap-a",
        "tile_id": [0, 0, 0],
    }


def _color(index, *, views=2):
    return VertexColor(
        vertex_id=f"{HASHES[0]}:{index}",
        mean_linear_rgb=(0.2, 0.4 + index * 0.03, 0.6),
        channel_stddev=(0.01, 0.01, 0.01),
        accepted_count=views,
        rejected_count=0,
        independent_views=views,
        source_frame_sha256=(HASHES[6], HASHES[7]),
        accepted_for_texture=True,
    )


def _request(*, coverage=800000, views=2, building_delivery_eligible=True):
    return TileAtlasBakeRequest(
        tile_revision_id=HASHES[0],
        geometry_revision_id=HASHES[1],
        source_generation_id=HASHES[2],
        calibration_id="cal-1",
        calibration_sha256=HASHES[3],
        physical_depth_validation_revision=HASHES[4],
        pose_revision_id=HASHES[5],
        source_frame_sha256=(HASHES[6], HASHES[7]),
        appearance_config_sha256=HASHES[8],
        measured_surface_coverage_ppm=coverage,
        minimum_independent_view_count=views,
        triangle_linear_rgb=np.asarray([[0.8, 0.2, 0.1], [0.1, 0.7, 0.3]], dtype=np.float32),
        triangle_authorized=(True, True),
        vertex_colors=tuple(_color(index, views=views) for index in range(4)),
        atlas_dimension=64,
        padding_pixels=0,
        mip_filter_radius_pixels=0,
        building_delivery_eligible=building_delivery_eligible,
    )


def _baker(tmp_path, loader, events=None):
    return TileAtlasBaker(
        source_loader=loader,
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
        progress_hook=None if events is None else events.append,
    )


def test_end_to_end_bake_publishes_preview_first_and_exact_complete_lods(tmp_path):
    body = _tile_body()
    metadata = _metadata(body)
    events = []
    baker = _baker(tmp_path, lambda _: (dict(metadata), body), events)

    result = baker.bake(_request())
    repeated = baker.bake(_request())

    assert events.index("PREVIEW_PUBLISHED") < events.index("ATLAS_PUBLISHED")
    assert result == repeated
    assert result.state == "TEXTURED_READY"
    assert result.building_authorized is True
    assert result.export_authorized is True
    assert result.preview.appearance_revision_id == result.published.appearance_revision_id
    assert len(result.published.manifest["lods"]) == 3
    for lod_id, lod_root in result.published.lod_paths.items():
        vertices = parse_render_vertices_v1(
            (lod_root / "render-vertices.bin").read_bytes(),
            expected_tile_revision_id=HASHES[0],
            expected_appearance_revision_id=result.revision.appearance_revision_id,
            source_vertex_count=4,
        )
        indices = parse_render_indices_v1(
            (lod_root / "render-indices.bin").read_bytes(),
            expected_tile_revision_id=HASHES[0],
            expected_appearance_revision_id=result.revision.appearance_revision_id,
            render_vertex_count=len(vertices.positions),
            source_triangle_count=2,
        )
        uvs = parse_render_uvs_v1(
            (lod_root / "render-uvs.bin").read_bytes(),
            expected_tile_revision_id=HASHES[0],
            expected_appearance_revision_id=result.revision.appearance_revision_id,
            atlas_count=1,
        )
        image = Image.open(BytesIO((lod_root / "atlas-0.png").read_bytes()))
        assert len(vertices.positions) == len(uvs.uvs)
        assert len(indices.triangles) >= 1
        assert image.mode == "RGBA"
        assert set(image.getchannel("A").get_flattened_data()).issubset({0, 255})


@pytest.mark.parametrize(
    ("coverage", "views", "authorized"),
    [(799999, 2, False), (800000, 1, False), (800000, 2, True)],
)
def test_delivery_boundaries_never_promote_diagnostic_atlas(tmp_path, coverage, views, authorized):
    body = _tile_body()
    result = _baker(tmp_path, lambda _: (_metadata(body), body)).bake(
        _request(coverage=coverage, views=views)
    )

    assert result.revision.delivery_authorization.delivery_authorized is authorized
    assert (result.state == "TEXTURED_READY") is authorized
    assert result.building_authorized is authorized
    assert result.export_authorized is authorized


def test_detached_tile_publication_persists_local_only_authorization(tmp_path):
    body = _tile_body()
    result = _baker(tmp_path, lambda _: (_metadata(body), body)).bake(
        _request(building_delivery_eligible=False)
    )

    assert result.state == "TEXTURED_READY"
    assert result.revision.delivery_authorization.delivery_authorized is True
    assert result.building_authorized is False
    assert result.export_authorized is False
    assert result.published.manifest["publication_authorization"] == {
        "building_delivery_eligible": False,
        "building_authorized": False,
        "export_authorized": False,
        "blockers": ["TILE_SCOPE_NOT_BUILDING_ELIGIBLE"],
    }
    assert result.published.pointer["publication_authorization"] == (
        result.published.manifest["publication_authorization"]
    )


def test_generation_change_after_preview_cancels_atlas_without_mutating_source(tmp_path):
    body = _tile_body()
    original_metadata = _metadata(body)
    calls = 0

    def loader(_):
        nonlocal calls
        calls += 1
        generation = HASHES[2] if calls < 3 else HASHES[9]
        return _metadata(body, generation=generation), body

    baker = _baker(tmp_path, loader)
    with pytest.raises(StaleTileGenerationError, match="source generation"):
        baker.bake(_request())

    assert original_metadata == _metadata(body)
    assert body == _tile_body()
    previews = list((tmp_path / "appearance-previews").rglob("vertex-color.bin"))
    assert len(previews) == 1
    with pytest.raises(FileNotFoundError):
        TileAppearanceStore(tmp_path).current(HASHES[0])


def test_evidence_change_after_preview_cancels_atlas_publication(tmp_path):
    body = _tile_body()
    checks = 0

    def revalidate_evidence():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("appearance evidence changed during bake")

    request = _request()
    request = TileAtlasBakeRequest(
        **{**request.__dict__, "evidence_revalidator": revalidate_evidence}
    )

    with pytest.raises(ValueError, match="evidence changed"):
        _baker(tmp_path, lambda _: (_metadata(body), body)).bake(request)

    assert checks == 2
    assert len(list((tmp_path / "appearance-previews").rglob("vertex-color.bin"))) == 1
    with pytest.raises(FileNotFoundError):
        TileAppearanceStore(tmp_path).current(HASHES[0])


def test_invalid_source_evidence_fails_before_any_appearance_publication(tmp_path):
    body = _tile_body()
    bad = _metadata(body)
    bad["geometry_revision_id"] = HASHES[9]

    with pytest.raises(ValueError, match="geometry revision"):
        _baker(tmp_path, lambda _: (bad, body)).bake(_request())

    assert not list(tmp_path.rglob("vertex-color.bin"))
    assert not list(tmp_path.rglob("CURRENT.json"))


def test_optional_augmentation_is_exact_separate_and_non_authoritative(tmp_path):
    body = _tile_body()
    layer = AugmentedLayer(
        name="RGBD_AUGMENTED",
        positions=np.asarray([[0.01, 0.01, 0.005], [0.02, 0.01, 0.005], [0.01, 0.02, 0.005]]),
        normals=np.asarray([[0, 0, 1]] * 3),
        triangles=np.asarray([[0, 1, 2]], dtype=np.uint32),
        uncertainty_m=np.asarray([0.002, 0.002, 0.002]),
        source_vertex_ids=("aug-v0", "aug-v1", "aug-v2"),
        source_triangle_ids=("aug-t0",),
        geometry_authority=False,
        source="PHYSICAL_RGBD_CONSENSUS",
    )

    class Published:
        source_generation_id = HASHES[2]
        rgbd_augmented_revision_id = HASHES[9]
        manifest_sha256 = HASHES[8]
        manifest = {"geometry_authority": False}

        def __init__(self):
            self.layer = layer

    request = _request()
    request = TileAtlasBakeRequest(
        **{
            **request.__dict__,
            "rgbd_augmented_revision_id": HASHES[9],
            "augmented_triangle_linear_rgb": np.asarray([[0.3, 0.6, 0.9]], dtype=np.float32),
            "augmented_triangle_authorized": (True,),
        }
    )
    calls = []
    baker = TileAtlasBaker(
        source_loader=lambda _: (_metadata(body), body),
        augmentation_resolver=lambda generation, revision: (
            calls.append((generation, revision)) or Published()
        ),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    )

    result = baker.bake(request)

    assert calls == [(HASHES[2], HASHES[9])] * 3
    assert result.revision.revision_core["rgbd_augmented_revision_id"] == HASHES[9]
    for item in result.published.manifest["lods"]:
        primitives = item["core"]["primitives"]
        assert any(value["source_layer"] == "RGBD_AUGMENTED" for value in primitives)
        assert item["core"]["rgbd_augmented_revision_id"] == HASHES[9]
    assert result.published.manifest["revision_core"]["geometry_revision_id"] == HASHES[1]
    assert result.preview.manifest["vertex_count"] == 4


def test_augmentation_identity_mismatch_fails_before_preview(tmp_path):
    body = _tile_body()
    request = _request()
    request = TileAtlasBakeRequest(
        **{**request.__dict__, "rgbd_augmented_revision_id": HASHES[9]}
    )

    class WrongPublished:
        source_generation_id = HASHES[2]
        rgbd_augmented_revision_id = HASHES[8]

    baker = TileAtlasBaker(
        source_loader=lambda _: (_metadata(body), body),
        augmentation_resolver=lambda *_: WrongPublished(),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    )
    with pytest.raises(ValueError, match="augmentation revision"):
        baker.bake(request)

    assert not list(tmp_path.rglob("vertex-color.bin"))
