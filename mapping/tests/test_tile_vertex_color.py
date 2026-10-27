import hashlib
import struct

import pytest

from mapping.pc.go2_mapping_appearance.tile_identity import tile_appearance_revision
from mapping.pc.go2_mapping_appearance.tile_vertex_color import (
    TileVertexColorStore,
    build_tile_vertex_color_sidecar,
    parse_tile_vertex_color_v1,
)
from mapping.pc.go2_mapping_rgbd.texture import VertexColor


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


def _revision(*, coverage=800000, views=2):
    return tile_appearance_revision(
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
        measured_surface_coverage_ppm=coverage,
        minimum_independent_view_count=views,
    )


def _color(index, *, accepted=True, views=2):
    return VertexColor(
        vertex_id=f"{HASHES[0]}:{index}",
        mean_linear_rgb=(0.1 + index * 0.05, 0.25, 0.5),
        channel_stddev=(0.01, 0.01, 0.01),
        accepted_count=views,
        rejected_count=0,
        independent_views=views,
        source_frame_sha256=(HASHES[6], HASHES[7]),
        accepted_for_texture=accepted,
    )


def test_tile_preview_reuses_rgb_validity_records_but_binds_exact_tile_order(tmp_path):
    revision = _revision()
    sidecar = build_tile_vertex_color_sidecar(
        revision=revision,
        vertex_colors=tuple(_color(index) for index in range(4)),
    )
    parsed = parse_tile_vertex_color_v1(
        sidecar.binary,
        expected_tile_revision_id=HASHES[0],
        expected_appearance_revision_id=revision.appearance_revision_id,
    )
    published = TileVertexColorStore(tmp_path).publish(sidecar)

    assert struct.unpack_from("<8sHHI", sidecar.binary) == (b"G2TVCOL1", 1, 80, 4)
    assert parsed.tile_revision_id == HASHES[0]
    assert parsed.appearance_revision_id == revision.appearance_revision_id
    assert parsed.validity.tolist() == [255, 255, 255, 255]
    assert sidecar.manifest["vertex_order"] == "G2TM_V1_EXACT_INDEX_ORDER"
    assert "measured_surface_geometry_revision_id" not in sidecar.manifest
    assert published.binary_path.read_bytes() == sidecar.binary
    assert published.manifest_path.read_bytes()
    assert TileVertexColorStore(tmp_path).resolve(HASHES[0], revision.appearance_revision_id) == published


def test_diagnostic_preview_publishes_below_delivery_gate_without_building_authority(tmp_path):
    revision = _revision(coverage=799999, views=1)
    sidecar = build_tile_vertex_color_sidecar(
        revision=revision,
        vertex_colors=(_color(0, views=1), _color(1, accepted=False, views=1)),
    )
    published = TileVertexColorStore(tmp_path).publish(sidecar)

    assert published.manifest["state"] == "DIAGNOSTIC_VERTEX_COLOR_READY"
    assert published.manifest["delivery_authorized"] is False
    assert published.manifest["building_authorized"] is False
    assert published.manifest["export_authorized"] is False


def test_tile_preview_rejects_wrong_vertex_order_and_corrupt_identity():
    revision = _revision()
    with pytest.raises(ValueError, match="vertex order"):
        build_tile_vertex_color_sidecar(
            revision=revision,
            vertex_colors=(_color(1), _color(0)),
        )

    sidecar = build_tile_vertex_color_sidecar(revision=revision, vertex_colors=(_color(0),))
    corrupt = sidecar.binary[:16] + bytes.fromhex(HASHES[1]) + sidecar.binary[48:]
    with pytest.raises(ValueError, match="tile revision"):
        parse_tile_vertex_color_v1(
            corrupt,
            expected_tile_revision_id=HASHES[0],
            expected_appearance_revision_id=revision.appearance_revision_id,
        )
    assert hashlib.sha256(sidecar.binary).hexdigest() == sidecar.manifest["vertex_color"]["sha256"]
