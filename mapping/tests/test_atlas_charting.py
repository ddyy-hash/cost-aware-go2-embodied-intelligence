import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.charting import chart_mesh_v1


def _corner(reverse=False):
    positions = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    normals = np.asarray(
        [[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=np.float64
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    triangles = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.uint32)
    source_triangle_ids = np.asarray([20, 21], dtype=np.uint32)
    materials = ("VALIDATED_UV_ATLAS", "BASE_MATERIAL")
    if reverse:
        triangles = triangles[::-1]
        source_triangle_ids = source_triangle_ids[::-1]
        materials = materials[::-1]
    return chart_mesh_v1(
        positions,
        normals,
        triangles,
        source_vertex_ids=np.asarray([10, 11, 12, 13], dtype=np.uint32),
        source_triangle_ids=source_triangle_ids,
        triangle_materials=materials,
        triangle_source_layers=("LIDAR_MEASURED", "LIDAR_MEASURED") if not reverse else ("LIDAR_MEASURED", "LIDAR_MEASURED"),
    )


def test_cube_corner_duplicates_seams_and_preserves_exact_source_mapping():
    chart = _corner()

    assert chart.render_triangles.shape == (2, 3)
    assert chart.source_triangle_ids.tolist() == [20, 21]
    assert set(chart.source_vertex_ids.tolist()) == {10, 11, 12, 13}
    assert np.count_nonzero(chart.source_vertex_ids == 10) == 2
    assert np.count_nonzero(chart.source_vertex_ids == 11) == 2
    assert np.all(np.isfinite(chart.uvs))
    assert np.all((chart.uvs >= 0.0) & (chart.uvs <= 1.0))
    assert chart.triangle_materials == ("VALIDATED_UV_ATLAS", "BASE_MATERIAL")
    assert chart.triangle_primitive_ids.tolist() == [0, 1]
    assert chart.atlas_indices[chart.flags == 0].tolist() == [65535] * 3
    assert all(
        not array.flags.writeable
        for array in (
            chart.render_positions,
            chart.render_normals,
            chart.render_triangles,
            chart.uvs,
            chart.source_vertex_ids,
            chart.source_triangle_ids,
        )
    )


def test_charting_is_stable_when_source_triangle_input_order_changes():
    first = _corner()
    repeated = _corner(reverse=True)

    np.testing.assert_array_equal(repeated.source_vertex_ids, first.source_vertex_ids)
    np.testing.assert_array_equal(repeated.source_triangle_ids, first.source_triangle_ids)
    np.testing.assert_array_equal(repeated.render_triangles, first.render_triangles)
    np.testing.assert_allclose(repeated.uvs, first.uvs, atol=1e-6, rtol=0.0)
    assert repeated.triangle_materials == first.triangle_materials


def test_disjoint_triangles_never_create_new_or_cross_primitive_topology():
    positions = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
        dtype=np.float32,
    )
    chart = chart_mesh_v1(
        positions,
        np.asarray([[0, 0, 1]] * 6, dtype=np.float32),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
        source_vertex_ids=np.arange(6, dtype=np.uint32),
        source_triangle_ids=np.asarray([100, 101], dtype=np.uint32),
        triangle_materials=("VALIDATED_UV_ATLAS", "VALIDATED_UV_ATLAS"),
        triangle_source_layers=("LIDAR_MEASURED", "RGBD_AUGMENTED"),
        allowed_augmented_vertex_ids={3, 4, 5},
        allowed_augmented_triangle_ids={101},
    )

    for triangle, source_triangle in zip(chart.render_triangles, chart.source_triangle_ids):
        source_vertices = set(chart.source_vertex_ids[triangle].tolist())
        assert source_vertices == ({0, 1, 2} if source_triangle == 100 else {3, 4, 5})
    assert chart.triangle_primitive_ids.tolist() == [0, 1]


def test_unknown_augmented_source_ids_reject_the_complete_chart():
    with pytest.raises(ValueError, match="augmentation source"):
        chart_mesh_v1(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.asarray([[0, 0, 1]] * 3, dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
            source_vertex_ids=np.asarray([7, 8, 9], dtype=np.uint32),
            source_triangle_ids=np.asarray([10], dtype=np.uint32),
            triangle_materials=("VALIDATED_UV_ATLAS",),
            triangle_source_layers=("RGBD_AUGMENTED",),
            allowed_augmented_vertex_ids={7, 8},
            allowed_augmented_triangle_ids={10},
        )
