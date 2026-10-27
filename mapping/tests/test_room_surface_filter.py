import numpy as np

from mapping.pc.go2_mapping_fusion.room_surface_filter import filter_room_surface_points


def test_room_filter_keeps_dense_wall_and_rejects_sparse_wave():
    wall = np.asarray([[x * 0.02, y * 0.02, 0.0] for x in range(12) for y in range(12)], dtype=float)
    wave = np.asarray([[2.0 + i * 0.18, 0.0, 0.3 + i * 0.03] for i in range(5)], dtype=float)
    values, keep = filter_room_surface_points(np.vstack((wall, wave)), support_radius_m=0.05, min_neighbors=3, min_component_size=8)
    assert len(values) == len(wall)
    assert len(keep) == len(wall)


def test_room_filter_keeps_small_connected_object():
    object_points = np.asarray([[0.5 + (i % 4) * 0.02, 0.5 + (i // 4) * 0.02, 0.2] for i in range(8)], dtype=float)
    values, _ = filter_room_surface_points(object_points, support_radius_m=0.05, min_neighbors=2, min_component_size=8)
    assert len(values) == 8
