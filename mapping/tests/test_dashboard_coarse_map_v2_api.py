from fastapi.testclient import TestClient

from mapping.pc.go2_mapping_dashboard.server import app


def _v2_map(revision=1):
    return {
        "schema_version": 2,
        "map_id": "floor-01",
        "coordinate_frame": "BUILDING_LOCAL",
        "building_id": "building-01",
        "floor_id": "floor-01",
        "rooms": [{"id": "room-1", "name": "Room 1", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}],
        "corridors": [],
        "walls": [],
        "doors": [],
        "features": [],
        "keep_out_zones": [],
        "anchors": [],
        "provenance": "USER_PRIOR",
        "geometry_authority": False,
        "revision": revision,
    }


def test_canonical_coarse_map_api_validates_imports_and_preserves_revision_history(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    client = TestClient(app)

    current = client.get("/api/maps/coarse/floor-01")
    assert current.status_code == 200
    assert current.json()["schema_version"] == 2

    validated = client.post("/api/maps/coarse/floor-01/validate", json=_v2_map())
    assert validated.status_code == 200
    assert validated.json()["valid"] is True
    assert validated.json()["map"]["geometry_authority"] is False

    imported = client.post("/api/maps/coarse/import", json=_v2_map())
    assert imported.status_code == 200
    assert imported.json()["accepted"] is True
    assert client.get("/api/maps/coarse/floor-01").json()["revision"] == 1

    replacement = _v2_map()
    replacement["rooms"][0]["name"] = "Room 1 revised"
    updated = client.put(
        "/api/maps/coarse/floor-01",
        json={**replacement, "expected_revision": 1},
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["rooms"][0]["name"] == "Room 1 revised"

    revisions = client.get("/api/maps/coarse/floor-01/revisions")
    assert revisions.status_code == 200
    assert revisions.json()["map_id"] == "floor-01"
    assert [item["revision"] for item in revisions.json()["revisions"]] == [1, 2]

    historical = client.get("/api/maps/coarse/floor-01/revisions/1")
    assert historical.status_code == 200
    assert historical.json()["map"]["revision"] == 1
    assert historical.json()["map"]["rooms"][0]["name"] == "Room 1"


def test_building_api_creates_and_revises_an_independent_floor_map(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseBuildingStore, CoarseMapStore

    floor_store = CoarseMapStore(tmp_path / "coarse-map.json")
    floor_store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", floor_store)
    monkeypatch.setattr(server, "_coarse_building_store", CoarseBuildingStore(tmp_path / "coarse-building.json"))
    client = TestClient(app)

    current = client.get("/api/buildings/coarse/building-01")
    assert current.status_code == 200
    assert current.json()["floors"][0]["map_id"] == "floor-01"

    created = client.post(
        "/api/buildings/coarse/building-01/floors",
        json={"floor_id": "floor-02", "label": "Second floor"},
    )
    assert created.status_code == 200
    assert created.json()["map"]["rooms"] == []
    assert [item["map_id"] for item in created.json()["building"]["floors"]] == ["floor-01", "floor-02"]

    floor_two = client.get("/api/maps/coarse/floor-02")
    assert floor_two.status_code == 200
    replacement = floor_two.json()
    replacement["rooms"] = [{"id": "zone-1", "name": "Office", "kind": "room", "polygon": [[0, 0], [3, 0], [3, 2], [0, 2]]}]
    revised = client.put("/api/maps/coarse/floor-02", json={**replacement, "expected_revision": 1})
    assert revised.status_code == 200
    assert revised.json()["revision"] == 2
    assert revised.json()["rooms"][0]["name"] == "Office"
    building_after_revision = client.get("/api/buildings/coarse/building-01").json()
    assert next(item for item in building_after_revision["floors"] if item["floor_id"] == "floor-02")["revision"] == 2


def test_geometry_api_splits_and_merges_function_zones_with_revision_guard(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    client = TestClient(app)
    split = client.post("/api/maps/coarse/floor-01/geometry/split", json={"entity_id": "room-1", "direction": "vertical", "expected_revision": 1})
    assert split.status_code == 200
    assert [room["id"] for room in split.json()["rooms"]] == ["room-1-a", "room-1-b"]
    assert split.json()["revision"] == 2
    merge = client.post("/api/maps/coarse/floor-01/geometry/merge", json={"first_entity_id": "room-1-a", "second_entity_id": "room-1-b", "expected_revision": 2})
    assert merge.status_code == 200
    assert merge.json()["rooms"][0]["polygon"] == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    assert merge.json()["revision"] == 3


def test_canonical_coarse_map_revision_detail_rejects_unknown_revision(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)

    response = TestClient(app).get("/api/maps/coarse/floor-01/revisions/99")

    assert response.status_code == 404


def test_canonical_coarse_map_api_rejects_map_id_mismatch_and_invalid_import(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    client = TestClient(app)

    mismatch = client.put("/api/maps/coarse/other", json=_v2_map())
    assert mismatch.status_code == 422

    invalid = client.post("/api/maps/coarse/import", json={"schema_version": 2, "map_id": "bad"})
    assert invalid.status_code == 422

    stale = client.put(
        "/api/maps/coarse/floor-01",
        json={**_v2_map(), "expected_revision": 99},
    )
    assert stale.status_code == 409


def test_canonical_coarse_map_can_export_a_nav2_prior_without_promoting_authority(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    monkeypatch.setattr(server, "_nav2_map_root", tmp_path / "nav2-map")
    client = TestClient(app)

    response = client.post("/api/maps/coarse/floor-01/nav2-export", json={"resolution": 0.25, "padding": 0.5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "AVAILABLE"
    assert payload["provenance"] == "USER_PRIOR"
    assert payload["geometry_authority"] is False
    assert payload["source_revision"] == 1
    assert (tmp_path / "nav2-map" / "floor-01" / "map.yaml").is_file()
    assert (tmp_path / "nav2-map" / "floor-01" / "map.pgm").is_file()


def test_canonical_coarse_map_frontier_preview_is_read_only_and_does_not_create_a_goal(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    client = TestClient(app)

    response = client.post(
        "/api/maps/coarse/floor-01/frontiers",
        json={"robot_pose": {"x": 1.0, "y": 1.0}, "resolution": 0.25, "padding": 0.5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "PREVIEW_ONLY"
    assert payload["read_only"] is True
    assert payload["goal_available"] is False
    assert payload["source_revision"] == 1
    assert payload["coordinate_frame"] == "BUILDING_LOCAL"
    assert payload["candidates"]
    assert client.get("/api/maps/coarse/floor-01").json()["revision"] == 1


def test_canonical_nav2_export_rejects_map_id_path_traversal(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server

    monkeypatch.setattr(server, "_nav2_map_root", tmp_path / "nav2-map")
    response = TestClient(app).post("/api/maps/coarse/../escape/nav2-export", json={})

    assert response.status_code in {404, 422}
    assert not (tmp_path / "escape").exists()


def test_canonical_coarse_map_task_routes_inherit_legacy_task_contract(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseMapStore

    store = CoarseMapStore(tmp_path / "coarse-map.json")
    store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", store)
    client = TestClient(app)

    rebuilt = client.post("/api/maps/coarse/floor-01/tasks/rebuild")
    assert rebuilt.status_code == 200
    task_id = next(task["id"] for task in rebuilt.json()["tasks"] if task["kind"] == "floor_coverage")
    task = next(task for task in rebuilt.json()["tasks"] if task["id"] == task_id)

    mode = client.post(
        f"/api/maps/coarse/floor-01/tasks/{task_id}/mode",
        json={"mode": "AUTONOMOUS", "lease_id": "lease-room-1", "expected_revision": task["revision"]},
    )
    assert mode.status_code == 200
    updated = next(item for item in mode.json()["tasks"] if item["id"] == task_id)
    assert updated["mode"] == "AUTONOMOUS"
    assert updated["lease_id"] == "lease-room-1"
    assert updated["lease_state"] == "LEASED"

    status = client.post(
        f"/api/maps/coarse/floor-01/tasks/{task_id}",
        json={"status": "ACTIVE", "expected_revision": updated["revision"]},
    )
    assert status.status_code == 200
    assert next(item for item in status.json()["tasks"] if item["id"] == task_id)["status"] == "ACTIVE"

    lease = client.post(
        f"/api/maps/coarse/floor-01/tasks/{task_id}/lease",
        json={"lease_id": "", "expected_revision": updated["revision"] + 1},
    )
    assert lease.status_code == 200
    released = next(item for item in lease.json()["tasks"] if item["id"] == task_id)
    assert released["lease_id"] is None
    assert released["lease_state"] == "UNLEASED"


def test_building_mission_api_compiles_floor_and_vertical_tasks_without_activation(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseBuildingStore, CoarseMapStore
    from mapping.pc.go2_mapping_dashboard.mission import MissionStore

    root = tmp_path / "maps"
    floor_store = CoarseMapStore(root / "coarse-map.json")
    floor_store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", floor_store)
    building_store = CoarseBuildingStore(root / "coarse-building.json")
    monkeypatch.setattr(server, "_coarse_building_store", building_store)
    mission_store = MissionStore(root / "missions")
    monkeypatch.setattr(server, "_mission_store", mission_store)
    client = TestClient(app)
    assert client.get("/api/buildings/coarse/building-01").status_code == 200
    client.post("/api/buildings/coarse/building-01/floors", json={"floor_id": "floor-02", "label": "Second"})
    building = client.get("/api/buildings/coarse/building-01").json()
    building["connections"] = [{"id": "stairs-1", "kind": "stairs", "from_floor_id": "floor-01", "to_floor_id": "floor-02",
                                 "from_anchor": [1, 1], "to_anchor": [1, 1]}]
    updated_building = client.put("/api/buildings/coarse/building-01", json=building)
    assert updated_building.status_code == 200
    assert updated_building.json()["connections"], updated_building.json()
    response = client.post("/api/buildings/coarse/building-01/mission")
    assert response.status_code == 200
    payload = response.json()
    assert payload["building_id"] == "building-01"
    assert any(task["kind"] == "vertical_connection" for task in payload["tasks"])
    assert all(task["status"] != "ACTIVE" for task in payload["tasks"])


def test_building_connection_api_requires_anchors_inside_declared_stair_zones(tmp_path, monkeypatch):
    from mapping.pc.go2_mapping_dashboard import server
    from mapping.pc.go2_mapping_dashboard.coarse_map import CoarseBuildingStore, CoarseMapStore

    root = tmp_path / "maps"
    floor_store = CoarseMapStore(root / "coarse-map.json")
    floor_store.save(_v2_map())
    monkeypatch.setattr(server, "_coarse_map_store", floor_store)
    building_store = CoarseBuildingStore(root / "coarse-building.json")
    monkeypatch.setattr(server, "_coarse_building_store", building_store)
    client = TestClient(app)
    client.get("/api/buildings/coarse/building-01")
    client.post("/api/buildings/coarse/building-01/floors", json={"floor_id": "floor-02", "label": "Second"})
    for map_id in ("floor-01", "floor-02"):
        current = client.get(f"/api/maps/coarse/{map_id}").json()
        current["rooms"] = [{"id": f"stairs-{map_id}", "kind": "stair", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
        assert client.put(f"/api/maps/coarse/{map_id}", json={**current, "expected_revision": 1}).status_code == 200
    building = client.get("/api/buildings/coarse/building-01").json()
    building["connections"] = [{"id": "stairs-1", "kind": "stairs", "from_floor_id": "floor-01", "to_floor_id": "floor-02",
                                 "from_anchor": [1, 1], "to_anchor": [1, 1]}]
    assert client.put("/api/buildings/coarse/building-01", json=building).status_code == 200
    building["connections"][0]["to_anchor"] = [9, 9]
    rejected = client.put("/api/buildings/coarse/building-01", json=building)
    assert rejected.status_code == 422
    assert "inside a stair zone" in rejected.json()["detail"]


def test_furniture_catalog_api_is_local_and_non_authoritative():
    response = TestClient(app).get("/api/catalog/furniture")
    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_authority"] == "LOCAL_ONLY"
    assert {item["id"] for item in payload["items"]} >= {"sofa", "table", "bed", "chair", "cabinet", "plant"}
    items = {item["id"]: item for item in payload["items"]}
    assert items["sofa"]["asset_source"] == "LOCAL_GLTF"
    assert items["sofa"]["asset_url"].startswith("/static/assets/furniture/")
    assert all(item["asset_source"] == "LOCAL_GLTF" for item in items.values())
    assert all(item["asset_url"].startswith("/static/assets/furniture/") for item in items.values())
    assert all(item.get("license") == "CC0-1.0" for key, item in items.items() if key != "sofa")


def test_coarse_map_template_catalog_is_user_prior_and_contains_stairs():
    response = TestClient(app).get("/api/catalog/coarse-map-templates")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provenance"] == "USER_PRIOR"
    assert payload["geometry_authority"] is False
    templates = {item["id"]: item for item in payload["templates"]}
    assert {"single-room", "office-floor", "teaching-floor"} <= templates.keys()
    stairs = next(room for room in templates["teaching-floor"]["rooms"] if room["kind"] == "stair")
    assert stairs["stair_profile"]["step_count"] == 16
    assert stairs["entry"] and stairs["exit"]


def test_pbr_material_catalog_is_local_cc0_and_contains_runtime_maps():
    response = TestClient(app).get("/api/catalog/pbr-materials")
    assert response.status_code == 200
    payload = response.json()
    assert payload["license"] == "CC0-1.0"
    materials = {item["id"]: item for item in payload["materials"]}
    assert {"wood", "tile", "concrete", "carpet"} <= materials.keys()
    for item in materials.values():
        assert set(item["maps"]) >= {"color", "normal_gl", "roughness"}
        assert all(value["url"].startswith("/static/assets/textures/pbr/") for value in item["maps"].values())
