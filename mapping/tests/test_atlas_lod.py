from collections import OrderedDict

import pytest

from mapping.pc.go2_mapping_appearance.lod import (
    build_lod_policies,
    pack_charts_v1,
    select_adaptive_density,
)


def test_density_is_clamped_to_3_5_mm_and_never_finer_than_source_sampling():
    fine = select_adaptive_density(geometric_detail_m=0.001, source_pixel_footprint_m=0.001)
    source_limited = select_adaptive_density(geometric_detail_m=0.003, source_pixel_footprint_m=0.004)
    coarse = select_adaptive_density(geometric_detail_m=0.004, source_pixel_footprint_m=0.006)

    assert fine.accepted is True and fine.mm_per_texel == 3.0
    assert source_limited.accepted is True and source_limited.mm_per_texel == 4.0
    assert coarse.accepted is False and coarse.mm_per_texel is None
    assert "SOURCE_SAMPLING_TOO_COARSE" in coarse.blockers


def test_multi_atlas_packing_is_deterministic_bounded_and_input_order_independent():
    sizes = OrderedDict(((20, (3000, 3000)), (10, (3000, 3000)), (30, (256, 128))))
    first = pack_charts_v1(sizes, padding_pixels=8, maximum_dimension=4096)
    repeated = pack_charts_v1(OrderedDict(reversed(tuple(sizes.items()))), padding_pixels=8, maximum_dimension=4096)

    assert first == repeated
    assert len(first.atlas_sizes) == 2
    assert all(width <= 4096 and height <= 4096 for width, height in first.atlas_sizes)
    assert tuple(value.chart_id for value in first.placements) == (10, 20, 30)
    assert {value.atlas_index for value in first.placements} == {0, 1}
    for placement in first.placements:
        atlas_width, atlas_height = first.atlas_sizes[placement.atlas_index]
        assert placement.x + placement.width <= atlas_width
        assert placement.y + placement.height <= atlas_height


def test_packing_rejects_a_chart_that_cannot_fit_with_padding():
    with pytest.raises(ValueError, match="fit"):
        pack_charts_v1({0: (4090, 4090)}, padding_pixels=8, maximum_dimension=4096)


def test_lod_policies_are_ordered_and_select_only_complete_source_triangles():
    source_ids = tuple(range(12))
    lods = build_lod_policies(
        source_ids,
        near_mm_per_texel=3.5,
        medium_distance_m=3.0,
        far_distance_m=8.0,
    )

    assert tuple(value.name for value in lods) == ("near", "medium", "far")
    assert tuple(value.order for value in lods) == (0, 1, 2)
    assert tuple(value.mm_per_texel for value in lods) == (3.5, 4.5, 5.0)
    assert lods[0].source_triangle_ids == source_ids
    assert set(lods[2].source_triangle_ids).issubset(lods[1].source_triangle_ids)
    assert set(lods[1].source_triangle_ids).issubset(lods[0].source_triangle_ids)
    assert all(value in source_ids for lod in lods for value in lod.source_triangle_ids)
    assert lods[0].maximum_distance_m == 3.0
    assert lods[1].minimum_distance_m == 3.0
    assert lods[1].maximum_distance_m == 8.0
    assert lods[2].maximum_distance_m is None


def test_lod_source_triangle_order_and_subsets_are_stable():
    first = build_lod_policies((9, 1, 5, 3, 7), near_mm_per_texel=4.0)
    repeated = build_lod_policies((7, 3, 5, 1, 9), near_mm_per_texel=4.0)

    assert first == repeated
    assert first[0].source_triangle_ids == (1, 3, 5, 7, 9)
