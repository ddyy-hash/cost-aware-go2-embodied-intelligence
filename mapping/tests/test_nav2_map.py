from __future__ import annotations

import pytest

from mapping.pc.go2_mapping_dashboard.nav2_map import build_nav2_occupancy_map, write_nav2_map


def _room_map():
    return {
        "schema_version": 2,
        "map_id": "room-01",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "building_id": "building-1",
        "floor_id": "room-01",
        "rooms": [{"id": "room", "name": "Room", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}],
        "corridors": [],
        "walls": [{"id": "wall", "geometry": [[2, 0], [2, 3]]}],
        "doors": [],
        "features": [],
        "keep_out_zones": [{"id": "keep-out", "polygon": [[3, 1], [3.5, 1], [3.5, 2], [3, 2]]}],
        "anchors": [],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 1,
    }


def test_coarse_map_rasterizes_free_room_unknown_outside_and_occupied_obstacles():
    result = build_nav2_occupancy_map(_room_map(), resolution=0.5, padding=0.5)

    assert result["metadata"]["resolution"] == 0.5
    assert result["metadata"]["origin"] == [-0.5, -0.5, 0.0]
    assert result["metadata"]["provenance"] == "USER_PRIOR"
    assert result["metadata"]["geometry_authority"] is False
    assert result["grid"][result["height"] // 2][result["width"] // 4] == 254
    assert result["grid"][0][0] == 205
    assert any(value == 0 for row in result["grid"] for value in row)


def test_coarse_map_export_writes_nav2_yaml_and_pgm_atomically(tmp_path):
    exported = write_nav2_map(_room_map(), tmp_path, resolution=0.5, padding=0.5)

    assert exported["yaml_path"].is_file()
    assert exported["pgm_path"].is_file()
    assert exported["yaml_path"].read_text(encoding="utf-8").startswith("image: map.pgm\n")
    assert exported["pgm_path"].read_bytes().startswith(b"P5\n")
    assert not list(tmp_path.glob("*.partial"))


def test_coarse_map_export_rejects_unbounded_raster_size(tmp_path):
    with pytest.raises(ValueError, match="max_cells"):
        write_nav2_map(_room_map(), tmp_path, resolution=0.001, padding=10.0, max_cells=100)
