from io import BytesIO

import numpy as np
from PIL import Image

from mapping.pc.go2_mapping_appearance.rasterizer import rasterize_rgba8_atlas


def _geometry():
    uvs = np.asarray(
        [[0.20, 0.20], [0.80, 0.20], [0.20, 0.80], [0.55, 0.55], [0.90, 0.55], [0.90, 0.90]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    colors = np.asarray([[0.25, 0.5, 0.75], [0.9, 0.1, 0.1]], dtype=np.float32)
    return uvs, triangles, colors


def test_rasterizer_emits_rgba8_binary_alpha_and_whole_triangle_base_fallback():
    uvs, triangles, colors = _geometry()
    result = rasterize_rgba8_atlas(
        uvs,
        triangles,
        colors,
        triangle_authorized=np.asarray([True, False]),
        width=64,
        height=64,
        padding_pixels=2,
        mip_filter_radius_pixels=1,
    )

    image = Image.open(BytesIO(result.png_bytes))
    rgba = np.asarray(image)
    assert image.mode == "RGBA"
    assert image.size == (64, 64)
    assert set(np.unique(rgba[..., 3]).tolist()).issubset({0, 255})
    assert np.count_nonzero(rgba[..., 3] == 255) > 0
    assert result.triangle_materials == ("VALIDATED_UV_ATLAS", "BASE_MATERIAL")
    assert result.triangle_atlas_indices.tolist() == [0, 65535]
    assert result.rgba.flags.writeable is False


def test_padding_copies_rgb_without_promoting_unauthorized_alpha():
    uvs, triangles, colors = _geometry()
    no_padding = rasterize_rgba8_atlas(
        uvs[:3], triangles[:1], colors[:1], [True], width=32, height=32, padding_pixels=0
    )
    padded = rasterize_rgba8_atlas(
        uvs[:3], triangles[:1], colors[:1], [True], width=32, height=32, padding_pixels=3
    )

    assert np.count_nonzero(padded.rgba[..., :3]) > np.count_nonzero(no_padding.rgba[..., :3])
    assert np.count_nonzero(padded.rgba[..., 3]) == np.count_nonzero(no_padding.rgba[..., 3])
    assert np.any(np.all(padded.rgba[..., :3] > 0, axis=2) & (padded.rgba[..., 3] == 0))


def test_invalid_filter_footprint_demotes_the_complete_triangle_to_base_material():
    uvs = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]], dtype=np.float32)
    result = rasterize_rgba8_atlas(
        uvs,
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32),
        [True],
        width=32,
        height=32,
        padding_pixels=0,
        mip_filter_radius_pixels=2,
    )

    assert result.triangle_materials == ("BASE_MATERIAL",)
    assert result.triangle_atlas_indices.tolist() == [65535]
    assert np.count_nonzero(result.rgba[..., 3]) == 0


def test_rasterizer_rejects_oversized_or_non_rgba_contract_inputs():
    uvs, triangles, colors = _geometry()
    for width, height in ((4097, 8), (8, 4097), (0, 8)):
        try:
            rasterize_rgba8_atlas(uvs, triangles, colors, [True, True], width=width, height=height)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid atlas dimensions were accepted")
