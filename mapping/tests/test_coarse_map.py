import json

import pytest

from mapping.pc.go2_mapping_dashboard.coarse_map import (
    CoarseBuildingStore,
    CoarseMapStore,
    assess_corner_diagnostics,
    build_task_graph,
    migrate_coarse_map_v1_to_v2,
    validate_coarse_map,
    merge_room_polygons,
    split_room_polygon,
)


def test_split_and_merge_axis_aligned_function_zones_are_deterministic():
    polygon = [[0, 0], [4, 0], [4, 3], [0, 3]]
    left, right = split_room_polygon(polygon, direction="vertical")
    assert left == [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]
    assert right == [[2.0, 0.0], [4.0, 0.0], [4.0, 3.0], [2.0, 3.0]]
    assert merge_room_polygons(left, right) == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]


def test_split_accepts_valid_freeform_polygon_but_rejects_degenerate_zone():
    left, right = split_room_polygon([[0, 0], [3, 0], [2, 2], [0, 2]], direction="vertical")
    assert len(left) >= 3
    assert len(right) >= 3
    with pytest.raises(ValueError, match="positive area"):
        split_room_polygon([[0, 0], [1, 0], [2, 0]], direction="vertical")


def test_stair_and_landing_are_valid_function_zone_kinds():
    payload = {
        "schema_version": 2, "map_id": "floor-01", "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-a", "floor_id": "floor-01",
        "rooms": [
            {"id": "stairs-1", "name": "Main stairs", "kind": "stair", "polygon": [[0, 0], [2, 0], [2, 4], [0, 4]]},
            {"id": "landing-1", "name": "Landing", "kind": "landing", "polygon": [[3, 0], [5, 0], [5, 2], [3, 2]]},
        ],
        "corridors": [], "walls": [], "doors": [], "features": [], "keep_out_zones": [], "anchors": [],
        "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }
    validated = validate_coarse_map(payload)
    assert [room["kind"] for room in validated["rooms"]] == ["stair", "landing"]


def test_equipment_furniture_dimensions_and_rotation_are_normalized():
    payload = {
        "schema_version": 2, "map_id": "floor-01", "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-a", "floor_id": "floor-01", "rooms": [], "corridors": [], "walls": [], "doors": [],
        "features": [{"id": "sofa-1", "kind": "equipment", "position": [1, 2], "dimensions_m": [2, 0.8, 0.9], "rotation_rad": 1.5707963}],
        "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }
    validated = validate_coarse_map(payload)
    assert validated["features"][0]["dimensions_m"] == [2.0, 0.8, 0.9]
    assert validated["features"][0]["rotation_rad"] == 1.570796


def test_carpet_and_threshold_objects_are_preserved_as_typed_v2_geometry():
    payload = {
        "schema_version": 2, "map_id": "floor-01", "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-a", "floor_id": "floor-01", "rooms": [], "corridors": [], "walls": [], "doors": [],
        "features": [], "carpets": [{"id": "carpet-1", "polygon": [[0, 0], [2, 0], [2, 1], [0, 1]], "material": "wool"}],
        "thresholds": [{"id": "threshold-1", "geometry": [[1, 0], [1, 0.2]], "height_m": 0.02}],
        "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }
    validated = validate_coarse_map(payload)
    assert validated["carpets"][0]["kind"] == "carpet"
    assert validated["carpets"][0]["material"] == "wool"
    assert validated["thresholds"][0]["kind"] == "threshold"


def test_coarse_building_store_registers_independent_floor_maps_and_connections(tmp_path):
    store = CoarseBuildingStore(tmp_path / "building.json")
    floor = {"schema_version": 2, "map_id": "floor-01", "building_id": "building-a", "floor_id": "floor-01", "coordinate_frame": "BUILDING_LOCAL", "rooms": [], "corridors": [], "walls": [], "doors": [], "features": [], "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1}
    created = store.register_floor(floor, label="一层")
    assert created["floors"] == [{"map_id": "floor-01", "floor_id": "floor-01", "label": "一层", "revision": 1}]
    updated = store.update_connections({"connections": [{"id": "lift-1", "kind": "elevator", "from_floor_id": "floor-01", "to_floor_id": "floor-02"}]})
    assert updated["connections"] == []
    store.register_floor({"schema_version": 2, "map_id": "floor-02", "building_id": "building-a", "floor_id": "floor-02", "coordinate_frame": "BUILDING_LOCAL", "rooms": [], "corridors": [], "walls": [], "doors": [], "features": [], "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1}, label="二层")
    updated = store.update_connections({"connections": [{"id": "lift-1", "kind": "elevator", "from_floor_id": "floor-01", "to_floor_id": "floor-02"}]})
    assert updated["connections"][0]["id"] == "lift-1"
    assert store.load()["building_id"] == "building-a"


def test_coarse_building_store_creates_a_persisted_empty_floor_map(tmp_path):
    store = CoarseBuildingStore(tmp_path / "building.json")
    created = store.create_floor("building-a", "floor-02", label="二层")
    assert created["map_id"] == "floor-02"
    assert created["floor_id"] == "floor-02"
    assert created["rooms"] == []
    assert (tmp_path / "coarse-floors" / "floor-02.json").is_file()
    assert store.load()["floors"][0]["label"] == "二层"


def test_build_task_graph_creates_robot_vacuum_style_room_tasks():
    coarse = {
        "schema_version": 1,
        "map_id": "demo",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [
            {
                "id": "room-1",
                "name": "Room 1",
                "kind": "room",
                "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "coverage_percent": 0,
            }
        ],
        "features": [
            {"id": "door-1", "kind": "door", "label": "Door", "position": [0, 1.5]},
            {"id": "cabinet-1", "kind": "equipment", "label": "Cabinet", "position": [3.5, 2.5]},
        ],
    }

    result = build_task_graph(coarse)

    assert [task["kind"] for task in result["tasks"]] == [
        "perimeter",
        "corners",
        "feature",
        "feature",
        "floor_coverage",
    ]
    assert [task["mode"] for task in result["tasks"]] == [
        "ASSISTED",
        "ASSISTED",
        "MANUAL",
        "MANUAL",
        "ASSISTED",
    ]
    assert result["tasks"][0]["room_id"] == "room-1"
    assert result["tasks"][1]["estimated_minutes"] > 0
    assert result["tasks"][2]["feature_id"] == "door-1"


def test_coarse_map_store_round_trips_and_rejects_bad_polygon(tmp_path):
    path = tmp_path / "coarse-map.json"
    store = CoarseMapStore(path)
    coarse = {
        "schema_version": 1,
        "map_id": "demo",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [{"id": "room-1", "name": "Room", "kind": "room", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
        "features": [],
    }
    saved = store.save(build_task_graph(coarse))
    assert json.loads(path.read_text(encoding="utf-8"))["map_id"] == "demo"
    assert saved["tasks"][0]["status"] == "PENDING"
    assert store.load()["rooms"][0]["id"] == "room-1"

    coarse["rooms"][0]["polygon"] = [[0, 0], [1, 0]]
    with pytest.raises(ValueError, match="polygon"):
        validate_coarse_map(coarse)


def test_corner_diagnostics_quarantines_pose_offset_before_autonomous_coverage():
    result = assess_corner_diagnostics({
        "point_to_trajectory_p95_m": 0.62,
        "plane_residual_p95_m": 0.012,
        "vertical_support_count": 180,
        "horizontal_support_count": 220,
    })
    assert result["state"] == "QUARANTINED"
    assert result["reason"] == "POSE_RESIDUAL_TOO_LARGE"


def test_coarse_map_v1_migrates_to_v2_without_granting_geometry_authority():
    legacy = {
        "schema_version": 1,
        "map_id": "floor-01",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [{"id": "room-1", "name": "Room 1", "kind": "room", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
        "features": [{"id": "door-1", "kind": "door", "label": "Door", "position": [0, 1]}],
    }

    migrated = migrate_coarse_map_v1_to_v2(legacy)
    validated = validate_coarse_map(migrated)

    assert validated["schema_version"] == 2
    assert validated["building_id"] == "building-1"
    assert validated["floor_id"] == "floor-01"
    assert validated["revision"] == 1
    assert validated["provenance"] == "USER_PRIOR"
    assert validated["geometry_authority"] is False
    assert validated["doors"][0]["id"] == "door-1"
    assert validated["rooms"][0]["id"] == "room-1"


def test_coarse_map_v2_round_trips_typed_entities_and_revision():
    payload = {
        "schema_version": 2,
        "map_id": "floor-01",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-01",
        "floor_id": "floor-01",
        "rooms": [{"id": "room-1", "name": "Room 1", "kind": "room", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
        "corridors": [],
        "walls": [{"id": "wall-1", "geometry": [[0, 0], [2, 0]]}],
        "doors": [{"id": "door-1", "position": [0, 1], "room_id": "room-1"}],
        "features": [],
        "keep_out_zones": [],
        "anchors": [{"id": "anchor-1", "position": [1, 1]}],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 12,
    }

    validated = validate_coarse_map(payload)

    assert validated["schema_version"] == 2
    assert validated["coordinate_frame"] == "BUILDING_LOCAL"
    assert validated["revision"] == 12
    assert validated["walls"][0]["id"] == "wall-1"
    assert validated["doors"][0]["room_id"] == "room-1"
    assert validated["anchors"][0]["position"] == [1.0, 1.0]
    assert validated["geometry_authority"] is False


def test_v2_task_graph_includes_corridor_and_typed_door_entities():
    payload = {
        "schema_version": 2,
        "map_id": "floor-01",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-01",
        "floor_id": "floor-01",
        "rooms": [],
        "corridors": [{"id": "corridor-1", "name": "Main", "polygon": [[0, 0], [6, 0], [6, 2], [0, 2]]}],
        "walls": [],
        "doors": [{"id": "door-1", "label": "Door", "position": [0, 1], "room_id": "corridor-1"}],
        "features": [],
        "keep_out_zones": [],
        "anchors": [],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 2,
    }

    result = build_task_graph(payload)

    assert any(task["room_id"] == "corridor-1" for task in result["tasks"])
    assert any(task.get("feature_id") == "door-1" for task in result["tasks"])


def test_v2_entities_receive_shared_frame_provenance_and_revision_metadata():
    payload = {
        "schema_version": 2,
        "map_id": "floor-01",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-01",
        "floor_id": "floor-01",
        "rooms": [{"id": "room-1", "name": "Room", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
        "corridors": [],
        "walls": [{"id": "wall-1", "geometry": [[0, 0], [2, 0]]}],
        "doors": [{"id": "door-1", "position": [0, 1]}],
        "features": [{"id": "equipment-1", "kind": "equipment", "position": [1, 1]}],
        "keep_out_zones": [],
        "anchors": [{"id": "anchor-1", "position": [1, 1]}],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": 12,
    }

    validated = validate_coarse_map(payload)

    entities = validated["rooms"] + validated["walls"] + validated["doors"] + validated["features"] + validated["anchors"]
    assert all(entity["coordinate_frame"] == "BUILDING_LOCAL" for entity in entities)
    assert all(entity["provenance"] == "USER_PRIOR" for entity in entities)
    assert all(entity["revision"] == 12 for entity in entities)
    assert all("confidence" in entity for entity in entities)


def test_stair_profile_and_endpoints_are_normalized_and_validated():
    payload = {
        "schema_version": 2, "map_id": "floor-01", "building_id": "building-a",
        "floor_id": "floor-01", "coordinate_frame": "BUILDING_LOCAL",
        "rooms": [{"id": "stairs-1", "kind": "stair", "polygon": [[0, 0], [2, 0], [2, 4], [0, 4]],
                   "stair_profile": {"step_count": 12, "riser_height_m": 0.17, "tread_depth_m": 0.28,
                                     "width_m": 1.1, "direction": "up"},
                   "entry": [1, 0], "exit": [1, 4]}],
        "corridors": [], "walls": [], "doors": [], "features": [], "keep_out_zones": [], "anchors": [],
        "provenance": "USER_PRIOR", "geometry_authority": False, "revision": 1,
    }
    normalized = validate_coarse_map(payload)
    stair = normalized["rooms"][0]
    assert stair["stair_profile"]["step_count"] == 12
    assert stair["entry"] == [1.0, 0.0]
    assert stair["exit"] == [1.0, 4.0]


def test_building_connection_preserves_anchors_and_rejects_same_floor(tmp_path):
    store = CoarseBuildingStore(tmp_path / "building.json")
    for floor_id in ("floor-01", "floor-02"):
        store.register_floor({"schema_version": 2, "map_id": floor_id, "building_id": "building-a", "floor_id": floor_id,
                              "coordinate_frame": "BUILDING_LOCAL", "rooms": [], "corridors": [], "walls": [], "doors": [],
                              "features": [], "keep_out_zones": [], "anchors": [], "provenance": "USER_PRIOR",
                              "geometry_authority": False, "revision": 1})
    connection = {"id": "stairs-1", "kind": "stairs", "from_floor_id": "floor-01", "to_floor_id": "floor-02",
                  "from_anchor": [1, 2], "to_anchor": [1, 2],
                  "stair_profile": {"step_count": 10, "riser_height_m": 0.18, "tread_depth_m": 0.28, "width_m": 1.0}}
    saved = store.update_connections({"connections": [connection]})
    assert saved["connections"][0]["from_anchor"] == [1.0, 2.0]
    with pytest.raises(ValueError, match="different floors"):
        store.update_connections({"connections": [{**connection, "from_floor_id": "floor-01", "to_floor_id": "floor-01"}]})
