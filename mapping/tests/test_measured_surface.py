from dataclasses import replace

import numpy as np

from mapping.pc.go2_mapping_fusion import MeasuredSurfaceBuilder, VoxelEvidence, VoxelState


def _voxel(index, position, state=VoxelState.MEASURED):
    return VoxelEvidence(
        key=index,
        position=np.asarray(position, dtype=np.float64),
        observation_count=3,
        independent_views=2,
        temporal_support_s=1.0,
        principal_stddev_m=0.005,
        point_to_plane_p95_m=0.005,
        static_probability=1.0,
        state=state,
        pose_generation="pg-1",
        source_artifact_sha256=("a" * 64,),
    )


def test_surface_uses_only_measured_voxels_and_keeps_point_authority():
    voxels = [
        _voxel((x, y, 0), (x * 0.015, y * 0.015, 0.0))
        for x in range(8)
        for y in range(8)
    ]
    voxels.append(_voxel((99, 99, 99), (1.0, 1.0, 1.0), VoxelState.CANDIDATE))

    surface = MeasuredSurfaceBuilder().build(voxels)

    assert surface.source_layer == "LIDAR_MEASURED"
    assert surface.point_authority == "LIDAR_MEASURED"
    assert surface.geometry_authority is False
    assert surface.vertex_count == 64
    assert surface.triangle_count > 0
    assert surface.source_states == ("MEASURED",)
    assert np.max(surface.positions[:, 2]) == 0.0


def test_surface_snapshot_arrays_are_bounded_and_self_consistent():
    surface = MeasuredSurfaceBuilder().build(
        [_voxel((x, y, 0), (x * 0.015, y * 0.015, 0.0)) for x in range(4) for y in range(4)]
    )

    snapshot = surface.bounded_snapshot()

    assert len(snapshot["positions"]) == surface.vertex_count * 3
    assert len(snapshot["normals"]) == surface.vertex_count * 3
    assert len(snapshot["triangles"]) == surface.triangle_count * 3
    assert snapshot["source_vertex_ids"] == list(surface.source_vertex_ids)


def test_surface_source_vertex_ids_follow_sorted_and_budgeted_vertex_order():
    builder = MeasuredSurfaceBuilder()
    builder.gates = replace(builder.gates, display_point_budget=2)
    voxels = [
        _voxel((key, 0, 0), (key * 0.015, 0.0, 0.0))
        for key in (4, 1, 3, 0, 2)
    ]

    surface = builder.build(voxels)

    assert surface.source_vertex_ids == ("0:0:0", "3:0:0")
    assert surface.positions[:, 0].tolist() == [0.0, 0.045]


def test_surface_preserves_parallel_top_bottom_and_vertical_side_faces():
    spacing = 0.015
    points = {
        (x, y, z)
        for z in (0, 8)
        for x in range(9)
        for y in range(9)
    }
    points.update((0, y, z) for y in range(9) for z in range(1, 8))
    voxels = [
        _voxel(key, tuple(value * spacing for value in key))
        for key in sorted(points)
    ]

    surface = MeasuredSurfaceBuilder().build(voxels)
    vertices = surface.positions[surface.triangles]
    centres = vertices.mean(axis=1)
    face_normals = np.cross(
        vertices[:, 1] - vertices[:, 0],
        vertices[:, 2] - vertices[:, 0],
    )
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    horizontal = np.abs(face_normals[:, 2]) > 0.8
    vertical_side = np.abs(face_normals[:, 0]) > 0.8

    assert np.count_nonzero(horizontal & (centres[:, 2] < spacing * 2)) >= 20
    assert np.count_nonzero(horizontal & (centres[:, 2] > spacing * 6)) >= 20
    assert np.count_nonzero(vertical_side & (centres[:, 0] < spacing * 2)) >= 20
    assert np.all(np.ptp(vertices[:, :, 2], axis=1) <= spacing * 3 + 1e-9)
