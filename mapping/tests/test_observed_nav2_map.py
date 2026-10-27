import numpy as np

from mapping.pc.go2_mapping_dashboard.observed_nav2_map import build_observed_nav2_map


def _coarse():
    return {
        "schema_version": 2, "map_id": "observed", "coordinate_frame": "SESSION_ODOM_LOCAL",
        "building_id": "building-1", "floor_id": "floor-1",
        "rooms": [{"id": "room-1", "name": "room", "kind": "room", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]], "coverage_percent": 0}],
        "corridors": [], "walls": [], "doors": [], "features": [], "keep_out_zones": [], "anchors": [], "carpets": [], "thresholds": [],
        "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }


def test_observed_points_overlay_occupied_cells_and_preserve_provenance():
    points = np.asarray([[2.0, 1.5, 0.6], [2.02, 1.5, 0.62], [2.0, 1.52, 0.61]], dtype=np.float32)
    result = build_observed_nav2_map(_coarse(), points, resolution=0.1, padding=0.2)
    assert result["metadata"]["coordinate_frame"] == "SESSION_ODOM_LOCAL"
    assert result["metadata"]["source_mode"] == "RECORDED_LIO_SESSION_ODOM_CLOUD"
    assert result["metadata"]["geometry_authority"] is False
    assert result["metadata"]["observed_point_count"] == 3
    assert any(cell == 0 for row in result["grid"] for cell in row)


def test_observed_map_rejects_invalid_points():
    with np.testing.assert_raises(ValueError):
        build_observed_nav2_map(_coarse(), np.zeros((3, 2), dtype=np.float32))
