import struct

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.atlas_codec import (
    encode_render_indices_v1,
    encode_render_uvs_v1,
    encode_render_vertices_v1,
    parse_render_indices_v1,
    parse_render_uvs_v1,
    parse_render_vertices_v1,
)
from mapping.pc.go2_mapping_appearance.tile_codec import decode_g2tm_v1


TILE = "a" * 64
APPEARANCE = "b" * 64


def _tile_body():
    positions = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.01], dtype="<f4")
    return b"".join(
        (
            struct.pack("<4sIIII", b"G2TM", 1, 3, 1, 1),
            positions.tobytes(),
            normals.tobytes(),
            triangles.tobytes(),
            scales.tobytes(),
        )
    )


def _render_payloads():
    positions = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float64)
    normals = np.asarray([[0, 0, 1]] * 3, dtype=np.float64)
    source_vertices = np.asarray([0, 1, 2], dtype=np.uint32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.uint32)
    source_triangles = np.asarray([0], dtype=np.uint32)
    uvs = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    atlas_indices = np.asarray([0, 0, 0], dtype=np.uint16)
    flags = np.asarray([1, 1, 1], dtype=np.uint16)
    return positions, normals, source_vertices, triangles, source_triangles, uvs, atlas_indices, flags


def test_decodes_current_g2tm_exactly_and_returns_immutable_arrays():
    tile = decode_g2tm_v1(_tile_body())

    np.testing.assert_array_equal(
        tile.positions,
        np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype="<f4"),
    )
    assert tile.normals.tolist() == [[0.0, 0.0, 1.0]] * 3
    assert tile.triangles.tolist() == [[0, 1, 2]]
    assert tile.triangle_scale_m.tolist() == pytest.approx([0.01])
    assert tile.positions.flags.writeable is False
    assert tile.triangles.flags.writeable is False


def test_atlas_v1_round_trip_uses_exact_common_header_and_source_mappings():
    positions, normals, source_vertices, triangles, source_triangles, uvs, atlas_indices, flags = _render_payloads()
    vertices_body = encode_render_vertices_v1(
        TILE, APPEARANCE, positions, normals, source_vertices
    )
    indices_body = encode_render_indices_v1(
        TILE, APPEARANCE, triangles, source_triangles
    )
    uvs_body = encode_render_uvs_v1(TILE, APPEARANCE, uvs, atlas_indices, flags)

    assert len(vertices_body[:80]) == 80
    assert struct.unpack_from("<8sHHI", vertices_body) == (b"G2RVERT1", 1, 80, 3)
    assert struct.unpack_from("<8sHHI", indices_body) == (b"G2RIDX01", 1, 80, 1)
    assert struct.unpack_from("<8sHHI", uvs_body) == (b"G2RUV001", 1, 80, 3)
    assert vertices_body[16:48].hex() == TILE
    assert vertices_body[48:80].hex() == APPEARANCE

    vertices = parse_render_vertices_v1(
        vertices_body,
        expected_tile_revision_id=TILE,
        expected_appearance_revision_id=APPEARANCE,
        source_vertex_count=3,
    )
    indices = parse_render_indices_v1(
        indices_body,
        expected_tile_revision_id=TILE,
        expected_appearance_revision_id=APPEARANCE,
        render_vertex_count=3,
        source_triangle_count=1,
    )
    parsed_uvs = parse_render_uvs_v1(
        uvs_body,
        expected_tile_revision_id=TILE,
        expected_appearance_revision_id=APPEARANCE,
        atlas_count=1,
    )

    assert vertices.source_vertex_ids.tolist() == [0, 1, 2]
    assert indices.triangles.tolist() == [[0, 1, 2]]
    assert indices.source_triangle_ids.tolist() == [0]
    assert parsed_uvs.uvs == pytest.approx(uvs.astype(np.float32))
    assert parsed_uvs.flags.tolist() == [1, 1, 1]
    assert all(not array.flags.writeable for array in (vertices.positions, indices.triangles, parsed_uvs.uvs))


def test_base_material_uv_records_require_the_exact_sentinel_and_zero_flags():
    _, _, _, _, _, uvs, _, _ = _render_payloads()
    valid = encode_render_uvs_v1(
        TILE,
        APPEARANCE,
        uvs,
        np.full(3, 65535, dtype=np.uint16),
        np.zeros(3, dtype=np.uint16),
    )
    assert parse_render_uvs_v1(
        valid,
        expected_tile_revision_id=TILE,
        expected_appearance_revision_id=APPEARANCE,
        atlas_count=1,
    ).atlas_indices.tolist() == [65535] * 3

    with pytest.raises(ValueError, match="sentinel|flags"):
        encode_render_uvs_v1(
            TILE,
            APPEARANCE,
            uvs,
            np.full(3, 65535, dtype=np.uint16),
            np.ones(3, dtype=np.uint16),
        )


@pytest.mark.parametrize("mutation", ["truncated", "extra", "bad_magic", "bad_identity"])
def test_parser_rejects_corrupt_length_magic_and_identity(mutation):
    positions, normals, source_vertices, *_ = _render_payloads()
    body = bytearray(encode_render_vertices_v1(TILE, APPEARANCE, positions, normals, source_vertices))
    if mutation == "truncated":
        body = body[:-1]
    elif mutation == "extra":
        body.extend(b"x")
    elif mutation == "bad_magic":
        body[:8] = b"BADMAGIC"

    with pytest.raises(ValueError):
        parse_render_vertices_v1(
            bytes(body),
            expected_tile_revision_id="c" * 64 if mutation == "bad_identity" else TILE,
            expected_appearance_revision_id=APPEARANCE,
            source_vertex_count=3,
        )


def test_codecs_reject_non_unit_normals_and_out_of_range_source_or_render_ids():
    positions, normals, source_vertices, triangles, source_triangles, *_ = _render_payloads()
    normals[0] = (0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="unit"):
        encode_render_vertices_v1(TILE, APPEARANCE, positions, normals, source_vertices)
    with pytest.raises(ValueError, match="source vertex"):
        parse_render_vertices_v1(
            encode_render_vertices_v1(TILE, APPEARANCE, positions, np.asarray([[0, 0, 1]] * 3), [0, 1, 5]),
            expected_tile_revision_id=TILE,
            expected_appearance_revision_id=APPEARANCE,
            source_vertex_count=3,
        )
    with pytest.raises(ValueError, match="render vertex"):
        parse_render_indices_v1(
            encode_render_indices_v1(TILE, APPEARANCE, [[0, 1, 4]], source_triangles),
            expected_tile_revision_id=TILE,
            expected_appearance_revision_id=APPEARANCE,
            render_vertex_count=3,
            source_triangle_count=1,
        )
    with pytest.raises(ValueError, match="source triangle"):
        parse_render_indices_v1(
            encode_render_indices_v1(TILE, APPEARANCE, triangles, [2]),
            expected_tile_revision_id=TILE,
            expected_appearance_revision_id=APPEARANCE,
            render_vertex_count=3,
            source_triangle_count=1,
        )


def test_g2tm_rejects_extra_bytes_and_invalid_normals_or_indices():
    with pytest.raises(ValueError):
        decode_g2tm_v1(_tile_body() + b"x")
    bad_normal = bytearray(_tile_body())
    struct.pack_into("<f", bad_normal, 20 + 3 * 3 * 4 + 2 * 4, 0.5)
    with pytest.raises(ValueError, match="unit"):
        decode_g2tm_v1(bytes(bad_normal))
    bad_index = bytearray(_tile_body())
    struct.pack_into("<I", bad_index, 20 + 3 * 3 * 4 * 2, 4)
    with pytest.raises(ValueError, match="index"):
        decode_g2tm_v1(bytes(bad_index))
