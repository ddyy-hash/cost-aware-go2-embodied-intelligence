import hashlib
import json
import sqlite3

import pytest

from mapping.pc.go2_mapping_live.building_project import BuildingProject, SubmapLifecycle
from mapping.pc.go2_mapping_live.live_protocol import FrameIdentity


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()
SESSION_ID = "session-a"


def _write_generation_manifest(generation_root, **overrides):
    generation_root.mkdir(parents=True, exist_ok=True)
    arrays = generation_root / "fixture-surfels.npz"
    witnesses = generation_root / "fixture-witnesses.json"
    arrays.write_bytes(b"surface fixture")
    witnesses.write_text("[]", encoding="utf-8")
    core = {
        "schema_version": 1,
        "map_id": "map-a",
        "source_config_sha256": SOURCE_HASH,
        "calibration_id": "PREVIEW_UNCALIBRATED",
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "witnesses_sha256": hashlib.sha256(witnesses.read_bytes()).hexdigest(),
    }
    core.update(overrides)
    generation_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    generation = generation_root / generation_id
    generation.mkdir(parents=True)
    arrays.replace(generation / "surfels.npz")
    witnesses.replace(generation / "witnesses.json")
    (generation / "manifest.json").write_text(
        json.dumps({**core, "generation_id": generation_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return generation_id


def _identity(sequence: int) -> FrameIdentity:
    return FrameIdentity("building-a", "session-a", "map-a", "producer-a", sequence)


def _route_request(session_id="session-a", *, sequence=1, map_id="map-a", producer_id="producer-a", calibration_id="cal-a", source_hash=SOURCE_HASH):
    return {
        "session_id": session_id,
        "map_id": map_id,
        "producer_instance_id": producer_id,
        "source_config_sha256": source_hash,
        "calibration_id": calibration_id,
        "start_sequence": sequence,
    }


def test_empty_project_bootstraps_explicit_hierarchy_and_restores_active_route(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    assert project.snapshot()["state"] == "BOOTSTRAPPING"

    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="corridor-east",
        zone_kind="CORRIDOR",
        submap_id="submap-001",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=10,
    )
    assert project.route_frame(_identity(10)) == "submap-001"
    assert project.route_frame(_identity(10)) == "submap-001"
    with pytest.raises(ValueError, match="session"):
        project.route_frame(FrameIdentity("building-a", "session-other", "map-a", "producer-a", 10))
    assert project.submap("submap-001")["session_id"] == SESSION_ID

    restarted = BuildingProject(tmp_path, building_id="building-a")
    assert restarted.snapshot()["active_submap_id"] == "submap-001"
    assert restarted.route_frame(_identity(11)) == "submap-001"


def test_route_cutoff_atomically_drains_old_submap_and_routes_next_sequence_to_new_submap(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=20,
    )
    project.begin_handoff(
        old_submap_id="submap-old",
        new_submap_id="submap-new",
        cutoff_sequence=22,
    )

    assert project.submap("submap-old")["lifecycle"] == SubmapLifecycle.DRAINING.value
    assert project.submap("submap-old")["route_cutoff_sequence"] == 22
    assert project.submap("submap-new")["session_id"] == SESSION_ID
    assert project.route_frame(_identity(22)) == "submap-old"
    assert project.route_frame(_identity(23)) == "submap-new"
    assert BuildingProject(tmp_path, building_id="building-a").route_frame(_identity(24)) == "submap-new"


def test_cached_frame_route_cannot_bypass_submap_session_provenance(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-001",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=10,
    )
    with sqlite3.connect(project.database_path) as connection:
        connection.execute(
            "INSERT INTO frame_routes VALUES(?,?,?,?,?,?)",
            ("building-a", "session-other", "map-a", "producer-a", 10, "submap-001"),
        )
        connection.commit()

    with pytest.raises(ValueError, match="session"):
        project.route_frame(FrameIdentity("building-a", "session-other", "map-a", "producer-a", 10))


def test_sealed_generation_is_immutable_and_quarantine_never_becomes_active_route(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-old", new_submap_id="submap-new", cutoff_sequence=3)
    generation_id = _write_generation_manifest(
        tmp_path / "submaps" / "submap-old" / "generations",
        building_id="building-a",
        session_id=SESSION_ID,
        floor_id="floor-01",
        submap_id="submap-old",
    )

    project.seal_submap(
        "submap-old",
        generation_id=generation_id,
        bounds_min=(-1.0, -2.0, 0.0),
        bounds_max=(3.0, 4.0, 2.5),
    )
    project.create_quarantine(
        submap_id="ambiguous-branch",
        floor_id="floor-01",
        zone_id="room-101",
        reason="AMBIGUOUS_RELOCALIZATION",
        source_config_sha256=SOURCE_HASH,
    )

    sealed = project.submap("submap-old")
    assert sealed["lifecycle"] == SubmapLifecycle.SEALED.value
    assert sealed["generation_id"] == generation_id
    authority = project.sealed_generation("submap-old", generation_id)
    assert authority["building_id"] == "building-a"
    assert authority["session_id"] == SESSION_ID
    assert authority["generation_id"] == generation_id
    assert authority["path"] == tmp_path / "submaps" / "submap-old" / "generations" / generation_id
    assert project.submap("ambiguous-branch")["lifecycle"] == SubmapLifecycle.QUARANTINED.value
    assert project.snapshot()["quarantined_submaps"] == 1
    with pytest.raises(ValueError, match="immutable"):
        project.seal_submap(
            "submap-old",
            generation_id="b" * 64,
            bounds_min=(-1.0, -2.0, 0.0),
            bounds_max=(3.0, 4.0, 2.5),
        )


def test_schema_v1_project_migrates_without_inventing_session_provenance(tmp_path):
    database = tmp_path / "building.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata VALUES('building_id', 'building-a');
        INSERT INTO metadata VALUES('schema_version', '1');
        CREATE TABLE floors(floor_id TEXT PRIMARY KEY, label TEXT NOT NULL, elevation_m REAL);
        INSERT INTO floors VALUES('floor-01', 'Level 1', NULL);
        CREATE TABLE zones(zone_id TEXT PRIMARY KEY, floor_id TEXT NOT NULL, kind TEXT NOT NULL);
        INSERT INTO zones VALUES('room-101', 'floor-01', 'ROOM');
        CREATE TABLE submaps(
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            submap_id TEXT UNIQUE NOT NULL, floor_id TEXT NOT NULL, zone_id TEXT NOT NULL,
            map_id TEXT, producer_instance_id TEXT, source_config_sha256 TEXT NOT NULL,
            lifecycle TEXT NOT NULL, route_start_sequence INTEGER, route_cutoff_sequence INTEGER,
            generation_id TEXT, generation_path TEXT, quarantine_reason TEXT,
            bounds_min_json TEXT, bounds_max_json TEXT
        );
        INSERT INTO submaps(
            submap_id,floor_id,zone_id,map_id,producer_instance_id,source_config_sha256,lifecycle,route_start_sequence
        ) VALUES('legacy-submap','floor-01','room-101','map-a','producer-a',
                 'PLACEHOLDER','ACTIVE',1);
        """.replace("PLACEHOLDER", SOURCE_HASH)
    )
    connection.commit()
    connection.close()

    project = BuildingProject(tmp_path, building_id="building-a")

    assert project.submap("legacy-submap")["session_id"] is None
    with sqlite3.connect(database) as migrated:
        assert migrated.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "3"


def _downgrade_to_schema_v2(project_root):
    database = project_root / "building.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX IF EXISTS one_root_component")
        connection.execute("DROP INDEX IF EXISTS one_open_route_per_stream")
        connection.execute("UPDATE metadata SET value='2' WHERE key='schema_version'")
        connection.commit()


def _add_schema_v2_session(
    project_root,
    *,
    session_id,
    submap_id,
    map_id,
    producer_id,
    start_sequence,
    lifecycle="ACTIVE",
    end_sequence=None,
):
    with sqlite3.connect(project_root / "building.sqlite3") as connection:
        connection.execute(
            "INSERT INTO floors(floor_id,label,elevation_m) VALUES(?,?,NULL)",
            (f"floor-{session_id}", f"Floor {session_id}"),
        )
        connection.execute(
            "INSERT INTO zones(zone_id,floor_id,kind) VALUES(?,?,?)",
            (f"zone-{session_id}", f"floor-{session_id}", "OTHER"),
        )
        connection.execute(
            """INSERT INTO submaps(
                submap_id,floor_id,zone_id,session_id,map_id,producer_instance_id,
                source_config_sha256,lifecycle,route_start_sequence,route_cutoff_sequence
            ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                submap_id,
                f"floor-{session_id}",
                f"zone-{session_id}",
                session_id,
                map_id,
                producer_id,
                SOURCE_HASH,
                lifecycle,
                start_sequence,
                end_sequence,
            ),
        )
        route_columns = {row[1] for row in connection.execute("PRAGMA table_info(route_ranges)")}
        if "building_id" in route_columns:
            connection.execute(
                """INSERT INTO route_ranges(
                    building_id,session_id,component_id,map_id,producer_instance_id,
                    submap_id,start_sequence,end_sequence
                ) VALUES(?,?,?,?,?,?,?,?)""",
                ("building-a", session_id, None, map_id, producer_id, submap_id, start_sequence, end_sequence),
            )
        else:
            connection.execute(
                "INSERT INTO route_ranges(map_id,producer_instance_id,submap_id,start_sequence,end_sequence) VALUES(?,?,?,?,?)",
                (map_id, producer_id, submap_id, start_sequence, end_sequence),
            )
        connection.commit()


def _schema_v2_project_with_two_sessions(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-session-first",
        floor_label="First",
        zone_id="zone-session-first",
        zone_kind="OTHER",
        submap_id="submap-session-first",
        session_id="session-first",
        map_id="map-first",
        producer_instance_id="producer-first",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.close()
    _downgrade_to_schema_v2(tmp_path)
    _add_schema_v2_session(
        tmp_path,
        session_id="session-second",
        submap_id="submap-session-second",
        map_id="map-second",
        producer_id="producer-second",
        start_sequence=1,
        lifecycle="SEALED",
        end_sequence=5,
    )
    return BuildingProject(tmp_path, building_id="building-a")


def test_schema_v2_migrates_known_sessions_into_root_and_detached_components(tmp_path):
    project = _schema_v2_project_with_two_sessions(tmp_path)

    snapshot = project.snapshot()

    assert snapshot["root_component_id"]
    assert snapshot["active_component_id"] == snapshot["root_component_id"]
    assert snapshot["detached_component_count"] == 1
    assert project.component(snapshot["root_component_id"])["attachment_state"] == "ROOT"
    detached = [
        project.component(row["component_id"])
        for row in (project.submap("submap-session-first"), project.submap("submap-session-second"))
        if row["component_id"] != snapshot["root_component_id"]
    ]
    assert detached[0]["attachment_state"] == "DETACHED"


def test_schema_v2_active_successor_keeps_component_open(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-old", new_submap_id="submap-new", cutoff_sequence=3)
    project.close()
    _downgrade_to_schema_v2(tmp_path)

    restarted = BuildingProject(tmp_path, building_id="building-a")
    component = restarted.component(restarted.snapshot()["active_component_id"])

    assert component["route_state"] == "OPEN"
    assert component["closing_cutoff_sequence"] is None


def test_schema_v2_multiple_active_submaps_fail_closed_during_migration(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-a",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.close()
    _downgrade_to_schema_v2(tmp_path)
    _add_schema_v2_session(
        tmp_path,
        session_id=SESSION_ID,
        submap_id="submap-b",
        map_id="map-a",
        producer_id="producer-a",
        start_sequence=100,
    )

    with pytest.raises(ValueError, match="schema-v3 migration"):
        BuildingProject(tmp_path, building_id="building-a")


def test_schema_v2_overlapping_route_ranges_fail_closed_during_migration(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-old", new_submap_id="submap-new", cutoff_sequence=3)
    project.close()
    _downgrade_to_schema_v2(tmp_path)
    with sqlite3.connect(tmp_path / "building.sqlite3") as connection:
        connection.execute("UPDATE route_ranges SET start_sequence=3 WHERE submap_id='submap-new'")
        connection.commit()

    with pytest.raises(ValueError, match="schema-v3 migration"):
        BuildingProject(tmp_path, building_id="building-a")


def test_schema_v3_overlapping_route_ranges_fail_closed_on_restart(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-old", new_submap_id="submap-new", cutoff_sequence=3)
    project.close()
    with sqlite3.connect(tmp_path / "building.sqlite3") as connection:
        connection.execute("UPDATE route_ranges SET start_sequence=3 WHERE submap_id='submap-new'")
        connection.commit()

    with pytest.raises(ValueError, match="route ranges"):
        BuildingProject(tmp_path, building_id="building-a")


def test_same_stream_sequence_in_two_sessions_routes_to_separate_components(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")

    first = project.ensure_session_route(**_route_request("session-a", sequence=1))
    second = project.ensure_session_route(**_route_request("session-b", sequence=1))

    assert first != second
    assert project.submap(first)["component_id"] != project.submap(second)["component_id"]
    assert project.component(project.submap(first)["component_id"])["attachment_state"] == "ROOT"
    assert project.component(project.submap(second)["component_id"])["attachment_state"] == "DETACHED"


def test_existing_session_rejects_map_producer_config_or_calibration_change(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.ensure_session_route(**_route_request("session-a"))

    conflicting = (
        _route_request("session-a", map_id="map-other"),
        _route_request("session-a", producer_id="producer-other"),
        _route_request("session-a", source_hash=hashlib.sha256(b"other-config").hexdigest()),
        _route_request("session-a", calibration_id="cal-other"),
    )
    for request in conflicting:
        with pytest.raises(ValueError, match="session route provenance"):
            project.ensure_session_route(**request)


def test_sealed_generation_fails_closed_for_non_authoritative_or_corrupt_targets(tmp_path):
    project = BuildingProject(tmp_path, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-old",
        session_id=SESSION_ID,
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    with pytest.raises(ValueError, match="SEALED"):
        project.sealed_generation("submap-old", "a" * 64)

    project.begin_handoff(old_submap_id="submap-old", new_submap_id="submap-new", cutoff_sequence=3)
    generation_id = _write_generation_manifest(
        tmp_path / "submaps" / "submap-old" / "generations",
        building_id="building-a",
        session_id=SESSION_ID,
        floor_id="floor-01",
        submap_id="submap-old",
    )
    project.seal_submap(
        "submap-old",
        generation_id=generation_id,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
    )

    with pytest.raises(ValueError, match="exact.*generation"):
        project.sealed_generation("submap-old", "b" * 64)

    manifest_path = tmp_path / "submaps" / "submap-old" / "generations" / generation_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["building_id"] = "building-other"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        project.sealed_generation("submap-old", generation_id)
