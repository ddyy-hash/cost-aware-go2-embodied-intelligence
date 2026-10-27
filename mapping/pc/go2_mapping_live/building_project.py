"""Transactional building hierarchy and immutable local-submap lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from types import MappingProxyType
from typing import Iterator, Optional

from .live_protocol import FrameIdentity


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ZONE_KINDS = {"ROOM", "CORRIDOR", "STAIR", "LANDING", "ELEVATOR_HALL", "SERVICE", "OTHER"}


class SubmapLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    SEALED = "SEALED"
    QUARANTINED = "QUARANTINED"


class AttachmentState(str, Enum):
    ROOT = "ROOT"
    DETACHED = "DETACHED"


class RouteState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


def _identifier(value: str, name: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{name} is invalid")
    return result


def _sha256(value: str, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _component_id(building_id: str, session_id: str) -> str:
    canonical = json.dumps(
        {"building_id": building_id, "session_id": session_id},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"component-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _session_submap_id(building_id: str, session_id: str, map_id: str, producer_instance_id: str) -> str:
    canonical = json.dumps(
        {
            "building_id": building_id,
            "session_id": session_id,
            "map_id": map_id,
            "producer_instance_id": producer_instance_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"submap-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


class BuildingProject:
    def __init__(self, root: str | Path, *, building_id: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "submaps").mkdir(exist_ok=True)
        self.building_id = _identifier(building_id, "building_id")
        self.database_path = self.root / "building.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        with self._transaction():
            row = self._connection.execute("SELECT value FROM metadata WHERE key='building_id'").fetchone()
            if row is None:
                self._connection.execute("INSERT INTO metadata(key,value) VALUES('building_id',?)", (self.building_id,))
                self._connection.execute("INSERT INTO metadata(key,value) VALUES('schema_version','3')")
            elif row["value"] != self.building_id:
                raise ValueError("project building_id does not match")
            self._migrate_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS floors(
                floor_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                elevation_m REAL
            );
            CREATE TABLE IF NOT EXISTS zones(
                zone_id TEXT PRIMARY KEY,
                floor_id TEXT NOT NULL REFERENCES floors(floor_id),
                kind TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS components(
                component_id TEXT PRIMARY KEY,
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                source_config_sha256 TEXT NOT NULL,
                calibration_id TEXT,
                attachment_state TEXT NOT NULL,
                route_state TEXT NOT NULL,
                closing_cutoff_sequence INTEGER
            );
            CREATE TABLE IF NOT EXISTS submaps(
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                submap_id TEXT UNIQUE NOT NULL,
                floor_id TEXT NOT NULL REFERENCES floors(floor_id),
                zone_id TEXT NOT NULL REFERENCES zones(zone_id),
                component_id TEXT REFERENCES components(component_id),
                session_id TEXT,
                map_id TEXT,
                producer_instance_id TEXT,
                source_config_sha256 TEXT NOT NULL,
                lifecycle TEXT NOT NULL,
                route_start_sequence INTEGER,
                route_cutoff_sequence INTEGER,
                generation_id TEXT,
                generation_path TEXT,
                quarantine_reason TEXT,
                bounds_min_json TEXT,
                bounds_max_json TEXT
            );
            CREATE TABLE IF NOT EXISTS route_ranges(
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                building_id TEXT NOT NULL,
                session_id TEXT,
                component_id TEXT REFERENCES components(component_id),
                map_id TEXT NOT NULL,
                producer_instance_id TEXT NOT NULL,
                submap_id TEXT NOT NULL REFERENCES submaps(submap_id),
                start_sequence INTEGER NOT NULL,
                end_sequence INTEGER,
                UNIQUE(session_id, map_id, producer_instance_id, start_sequence)
            );
            CREATE TABLE IF NOT EXISTS frame_routes(
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                map_id TEXT NOT NULL,
                producer_instance_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                submap_id TEXT NOT NULL REFERENCES submaps(submap_id),
                PRIMARY KEY(building_id, session_id, map_id, producer_instance_id, sequence)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS submap_bounds USING rtree(
                row_id,
                min_x, max_x,
                min_y, max_y,
                min_z, max_z
            );
            """
        )

    def _migrate_schema(self) -> None:
        row = self._connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if row is None:
            raise ValueError("project schema version is missing")
        version = str(row["value"])
        if version not in {"1", "2", "3"}:
            raise ValueError(f"unsupported project schema version: {version}")
        columns = {str(item["name"]) for item in self._connection.execute("PRAGMA table_info(submaps)")}
        if "session_id" not in columns:
            self._connection.execute("ALTER TABLE submaps ADD COLUMN session_id TEXT")
        if "component_id" not in columns:
            self._connection.execute("ALTER TABLE submaps ADD COLUMN component_id TEXT REFERENCES components(component_id)")
        if version in {"1", "2"}:
            try:
                self._migrate_components_from_v2()
                self._replace_route_ranges_v3()
                self._validate_route_ranges()
                self._initialize_operational_pointer()
            except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
                raise ValueError(f"schema-v3 migration failed: {exc}") from exc
            self._connection.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")
        else:
            self._validate_route_ranges()
        self._ensure_v3_indexes()

    def _migrate_components_from_v2(self) -> None:
        rows = self._connection.execute(
            "SELECT * FROM submaps WHERE session_id IS NOT NULL ORDER BY row_id"
        ).fetchall()
        if not rows:
            return
        sessions: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            sessions.setdefault(str(row["session_id"]), []).append(row)
        root_session = str(rows[0]["session_id"])
        for session_id, session_rows in sessions.items():
            configurations = {str(row["source_config_sha256"]) for row in session_rows}
            if len(configurations) != 1:
                raise ValueError(f"session {session_id} has conflicting source configurations")
            component_id = _component_id(self.building_id, session_id)
            route_state, cutoff = self._migrated_route_state(session_id, session_rows)
            attachment = AttachmentState.ROOT if session_id == root_session else AttachmentState.DETACHED
            existing = self._connection.execute(
                "SELECT * FROM components WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """INSERT INTO components(
                        component_id,building_id,session_id,source_config_sha256,calibration_id,
                        attachment_state,route_state,closing_cutoff_sequence
                    ) VALUES(?,?,?,?,NULL,?,?,?)""",
                    (
                        component_id,
                        self.building_id,
                        session_id,
                        next(iter(configurations)),
                        attachment.value,
                        route_state.value,
                        cutoff,
                    ),
                )
            else:
                if existing["component_id"] != component_id or existing["building_id"] != self.building_id:
                    raise ValueError(f"session {session_id} has conflicting component identity")
                self._connection.execute(
                    """UPDATE components SET source_config_sha256=?,attachment_state=?,
                       route_state=?,closing_cutoff_sequence=? WHERE component_id=?""",
                    (next(iter(configurations)), attachment.value, route_state.value, cutoff, component_id),
                )
            self._connection.execute(
                "UPDATE submaps SET component_id=? WHERE session_id=?",
                (component_id, session_id),
            )

    def _migrated_route_state(
        self, session_id: str, session_rows: list[sqlite3.Row]
    ) -> tuple[RouteState, Optional[int]]:
        submap_ids = [str(row["submap_id"]) for row in session_rows]
        placeholders = ",".join("?" for _ in submap_ids)
        routes = self._connection.execute(
            f"SELECT * FROM route_ranges WHERE submap_id IN ({placeholders}) ORDER BY start_sequence",
            submap_ids,
        ).fetchall()
        routes_by_submap: dict[str, list[sqlite3.Row]] = {}
        for route in routes:
            routes_by_submap.setdefault(str(route["submap_id"]), []).append(route)
        active = [row for row in session_rows if row["lifecycle"] == SubmapLifecycle.ACTIVE.value]
        draining = [row for row in session_rows if row["lifecycle"] == SubmapLifecycle.DRAINING.value]
        sealed = [row for row in session_rows if row["lifecycle"] == SubmapLifecycle.SEALED.value]
        if len(active) > 1:
            raise ValueError(f"session {session_id} has multiple active submaps")
        open_routes = [route for route in routes if route["end_sequence"] is None]
        if active:
            active_routes = routes_by_submap.get(str(active[0]["submap_id"]), [])
            if len(open_routes) != 1 or len(active_routes) != 1 or active_routes[0]["end_sequence"] is not None:
                raise ValueError(f"session {session_id} active submap lacks exactly one open route")
            for row in draining:
                predecessor_routes = routes_by_submap.get(str(row["submap_id"]), [])
                if len(predecessor_routes) != 1 or predecessor_routes[0]["end_sequence"] is None:
                    raise ValueError(f"session {session_id} draining predecessor has an open or missing route")
                if row["route_cutoff_sequence"] != predecessor_routes[0]["end_sequence"]:
                    raise ValueError(f"session {session_id} draining predecessor cutoff conflicts")
            return RouteState.OPEN, None
        if open_routes:
            raise ValueError(f"session {session_id} has an open route without an active submap")
        if len(draining) == 1:
            route_rows = routes_by_submap.get(str(draining[0]["submap_id"]), [])
            if len(route_rows) != 1 or route_rows[0]["end_sequence"] is None:
                raise ValueError(f"session {session_id} draining route is invalid")
            cutoff = draining[0]["route_cutoff_sequence"]
            if cutoff is None or int(cutoff) != int(route_rows[0]["end_sequence"]):
                raise ValueError(f"session {session_id} draining cutoff conflicts")
            return RouteState.CLOSING, int(cutoff)
        if len(draining) > 1:
            raise ValueError(f"session {session_id} has multiple unfinished draining submaps")
        if sealed:
            if not routes or any(route["end_sequence"] is None for route in routes):
                raise ValueError(f"session {session_id} sealed routes lack final cutoffs")
            return RouteState.CLOSED, max(int(route["end_sequence"]) for route in routes)
        raise ValueError(f"session {session_id} has no authoritative routed lifecycle")

    def _replace_route_ranges_v3(self) -> None:
        columns = {str(row["name"]) for row in self._connection.execute("PRAGMA table_info(route_ranges)")}
        if {"building_id", "session_id", "component_id"}.issubset(columns):
            self._connection.execute(
                """UPDATE route_ranges
                   SET session_id=COALESCE(
                           session_id,
                           (SELECT s.session_id FROM submaps AS s WHERE s.submap_id=route_ranges.submap_id)
                       ),
                       component_id=COALESCE(
                           component_id,
                           (SELECT s.component_id FROM submaps AS s WHERE s.submap_id=route_ranges.submap_id)
                       )
                   WHERE EXISTS(
                       SELECT 1 FROM submaps AS s
                       WHERE s.submap_id=route_ranges.submap_id AND s.session_id IS NOT NULL
                   )"""
            )
            return
        self._connection.execute("ALTER TABLE route_ranges RENAME TO route_ranges_v2")
        self._connection.execute(
            """CREATE TABLE route_ranges(
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                building_id TEXT NOT NULL,
                session_id TEXT,
                component_id TEXT REFERENCES components(component_id),
                map_id TEXT NOT NULL,
                producer_instance_id TEXT NOT NULL,
                submap_id TEXT NOT NULL REFERENCES submaps(submap_id),
                start_sequence INTEGER NOT NULL,
                end_sequence INTEGER,
                UNIQUE(session_id,map_id,producer_instance_id,start_sequence)
            )"""
        )
        self._connection.execute(
            """INSERT INTO route_ranges(
                route_id,building_id,session_id,component_id,map_id,producer_instance_id,
                submap_id,start_sequence,end_sequence
            )
            SELECT r.route_id,?,s.session_id,s.component_id,r.map_id,r.producer_instance_id,
                   r.submap_id,r.start_sequence,r.end_sequence
            FROM route_ranges_v2 AS r JOIN submaps AS s ON s.submap_id=r.submap_id""",
            (self.building_id,),
        )
        self._connection.execute("DROP TABLE route_ranges_v2")

    def _validate_route_ranges(self) -> None:
        rows = self._connection.execute(
            """SELECT r.*,
                      s.session_id AS submap_session_id,
                      s.component_id AS submap_component_id,
                      s.map_id AS submap_map_id,
                      s.producer_instance_id AS submap_producer_instance_id,
                      s.source_config_sha256 AS submap_source_config_sha256,
                      c.building_id AS component_building_id,
                      c.session_id AS component_session_id,
                      c.source_config_sha256 AS component_source_config_sha256
               FROM route_ranges AS r
               LEFT JOIN submaps AS s ON s.submap_id=r.submap_id
               LEFT JOIN components AS c ON c.component_id=r.component_id
               ORDER BY r.session_id,r.map_id,r.producer_instance_id,r.start_sequence,r.route_id"""
        ).fetchall()
        streams: dict[tuple[str, str, str], list[tuple[int, Optional[int]]]] = {}
        for row in rows:
            if row["building_id"] != self.building_id:
                raise ValueError("route ranges contain a foreign building_id")
            start = int(row["start_sequence"])
            end = None if row["end_sequence"] is None else int(row["end_sequence"])
            if start < 0:
                raise ValueError("route ranges contain a negative start_sequence")
            if end is not None and end < start:
                raise ValueError("route ranges contain an inverted sequence interval")

            session_id = row["session_id"]
            if session_id is None:
                continue
            if row["submap_session_id"] != session_id:
                raise ValueError("route ranges conflict with submap session provenance")
            if row["component_id"] is None or row["submap_component_id"] != row["component_id"]:
                raise ValueError("route ranges conflict with submap component provenance")
            if row["component_session_id"] != session_id or row["component_building_id"] != self.building_id:
                raise ValueError("route ranges conflict with component project provenance")
            if row["component_source_config_sha256"] != row["submap_source_config_sha256"]:
                raise ValueError("route ranges conflict with component source configuration")
            if row["map_id"] != row["submap_map_id"]:
                raise ValueError("route ranges conflict with submap map provenance")
            if row["producer_instance_id"] != row["submap_producer_instance_id"]:
                raise ValueError("route ranges conflict with submap producer provenance")
            key = (str(session_id), str(row["map_id"]), str(row["producer_instance_id"]))
            streams.setdefault(key, []).append((start, end))

        for ranges in streams.values():
            previous_end: Optional[int] = None
            previous_was_open = False
            open_count = 0
            for start, end in ranges:
                if previous_was_open:
                    raise ValueError("route ranges contain an open interval before the final range")
                if previous_end is not None and start <= previous_end:
                    raise ValueError("route ranges overlap within one session stream")
                if end is None:
                    open_count += 1
                    previous_was_open = True
                previous_end = end
            if open_count > 1:
                raise ValueError("route ranges contain multiple open intervals in one session stream")

    def _initialize_operational_pointer(self) -> None:
        self._connection.execute(
            "DELETE FROM metadata WHERE key IN ('operational_session_id','operational_component_id')"
        )
        open_components = self._connection.execute(
            "SELECT component_id,session_id FROM components WHERE route_state=?",
            (RouteState.OPEN.value,),
        ).fetchall()
        if len(open_components) > 1:
            raise ValueError("multiple authoritative active components require explicit operational selection")
        if len(open_components) == 1:
            component = open_components[0]
            active_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM submaps WHERE component_id=? AND lifecycle=?",
                    (component["component_id"], SubmapLifecycle.ACTIVE.value),
                ).fetchone()[0]
            )
            if active_count != 1:
                raise ValueError("operational component does not have exactly one active submap")
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES('operational_session_id',?)",
                (component["session_id"],),
            )
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES('operational_component_id',?)",
                (component["component_id"],),
            )

    def _ensure_v3_indexes(self) -> None:
        self._connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS one_root_component
               ON components(building_id) WHERE attachment_state='ROOT'"""
        )
        self._connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS one_open_route_per_stream
               ON route_ranges(session_id,map_id,producer_instance_id)
               WHERE session_id IS NOT NULL AND end_sequence IS NULL"""
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def bootstrap(
        self,
        *,
        floor_id: str,
        floor_label: str,
        zone_id: str,
        zone_kind: str,
        submap_id: str,
        session_id: str,
        map_id: str,
        producer_instance_id: str,
        source_config_sha256: str,
        start_sequence: int,
        calibration_id: str = "PREVIEW_UNCALIBRATED",
    ) -> None:
        floor_id = _identifier(floor_id, "floor_id")
        zone_id = _identifier(zone_id, "zone_id")
        submap_id = _identifier(submap_id, "submap_id")
        session_id = _identifier(session_id, "session_id")
        map_id = _identifier(map_id, "map_id")
        producer_instance_id = _identifier(producer_instance_id, "producer_instance_id")
        source_config_sha256 = _sha256(source_config_sha256, "source_config_sha256")
        calibration_id = _identifier(calibration_id, "calibration_id")
        zone_kind = str(zone_kind).upper()
        if zone_kind not in _ZONE_KINDS:
            raise ValueError("zone_kind is invalid")
        if isinstance(start_sequence, bool) or int(start_sequence) < 0:
            raise ValueError("start_sequence must be non-negative")
        if not str(floor_label).strip():
            raise ValueError("floor_label is required")
        with self._transaction():
            if self._connection.execute("SELECT 1 FROM submaps LIMIT 1").fetchone() is not None:
                raise ValueError("building project is already bootstrapped")
            component_id = _component_id(self.building_id, session_id)
            self._connection.execute(
                """INSERT INTO components(
                    component_id,building_id,session_id,source_config_sha256,calibration_id,
                    attachment_state,route_state,closing_cutoff_sequence
                ) VALUES(?,?,?,?,?,?,?,NULL)""",
                (
                    component_id,
                    self.building_id,
                    session_id,
                    source_config_sha256,
                    calibration_id,
                    AttachmentState.ROOT.value,
                    RouteState.OPEN.value,
                ),
            )
            self._connection.execute(
                "INSERT INTO floors(floor_id,label,elevation_m) VALUES(?,?,NULL)",
                (floor_id, str(floor_label).strip()),
            )
            self._connection.execute(
                "INSERT INTO zones(zone_id,floor_id,kind) VALUES(?,?,?)",
                (zone_id, floor_id, zone_kind),
            )
            self._connection.execute(
                """INSERT INTO submaps(
                    submap_id,floor_id,zone_id,component_id,session_id,map_id,producer_instance_id,
                    source_config_sha256,lifecycle,route_start_sequence
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    submap_id,
                    floor_id,
                    zone_id,
                    component_id,
                    session_id,
                    map_id,
                    producer_instance_id,
                    source_config_sha256,
                    SubmapLifecycle.ACTIVE.value,
                    int(start_sequence),
                ),
            )
            self._connection.execute(
                """INSERT INTO route_ranges(
                    building_id,session_id,component_id,map_id,producer_instance_id,submap_id,start_sequence,end_sequence
                ) VALUES(?,?,?,?,?,?,?,NULL)""",
                (self.building_id, session_id, component_id, map_id, producer_instance_id, submap_id, int(start_sequence)),
            )
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES('operational_session_id',?)",
                (session_id,),
            )
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES('operational_component_id',?)",
                (component_id,),
            )
        (self.root / "submaps" / submap_id / "generations").mkdir(parents=True, exist_ok=True)

    def ensure_session_route(
        self,
        *,
        session_id: str,
        map_id: str,
        producer_instance_id: str,
        source_config_sha256: str,
        calibration_id: str,
        start_sequence: int,
    ) -> str:
        session_id = _identifier(session_id, "session_id")
        map_id = _identifier(map_id, "map_id")
        producer_instance_id = _identifier(producer_instance_id, "producer_instance_id")
        source_config_sha256 = _sha256(source_config_sha256, "source_config_sha256")
        calibration_id = _identifier(calibration_id, "calibration_id")
        if isinstance(start_sequence, bool) or int(start_sequence) < 0:
            raise ValueError("start_sequence must be non-negative")
        start_sequence = int(start_sequence)
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM components WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["building_id"] != self.building_id
                    or existing["source_config_sha256"] != source_config_sha256
                    or existing["calibration_id"] != calibration_id
                ):
                    raise ValueError("session route provenance conflicts with component identity")
                if existing["route_state"] != RouteState.OPEN.value:
                    raise ValueError("session route provenance is closed and cannot be reopened")
                active_rows = self._connection.execute(
                    "SELECT * FROM submaps WHERE component_id=? AND lifecycle=?",
                    (existing["component_id"], SubmapLifecycle.ACTIVE.value),
                ).fetchall()
                if len(active_rows) != 1:
                    raise ValueError("session route provenance lacks one active submap")
                active = active_rows[0]
                if active["map_id"] != map_id or active["producer_instance_id"] != producer_instance_id:
                    raise ValueError("session route provenance conflicts with map or producer identity")
                route = self._connection.execute(
                    """SELECT * FROM route_ranges WHERE building_id=? AND session_id=?
                       AND component_id=? AND map_id=? AND producer_instance_id=?
                       AND submap_id=? AND end_sequence IS NULL""",
                    (
                        self.building_id,
                        session_id,
                        existing["component_id"],
                        map_id,
                        producer_instance_id,
                        active["submap_id"],
                    ),
                ).fetchall()
                if len(route) != 1:
                    raise ValueError("session route provenance lacks one open route")
                self._set_operational_pointer(session_id, str(existing["component_id"]))
                return str(active["submap_id"])

            component_id = _component_id(self.building_id, session_id)
            submap_id = _session_submap_id(self.building_id, session_id, map_id, producer_instance_id)
            is_root = self._connection.execute("SELECT 1 FROM components LIMIT 1").fetchone() is None
            attachment = AttachmentState.ROOT if is_root else AttachmentState.DETACHED
            floor_id = "floor-unassigned"
            zone_id = "zone-unassigned"
            floor = self._connection.execute("SELECT * FROM floors WHERE floor_id=?", (floor_id,)).fetchone()
            if floor is None:
                self._connection.execute(
                    "INSERT INTO floors(floor_id,label,elevation_m) VALUES(?,?,NULL)",
                    (floor_id, "Unassigned floor"),
                )
            zone = self._connection.execute("SELECT * FROM zones WHERE zone_id=?", (zone_id,)).fetchone()
            if zone is None:
                self._connection.execute(
                    "INSERT INTO zones(zone_id,floor_id,kind) VALUES(?,?,?)",
                    (zone_id, floor_id, "OTHER"),
                )
            elif zone["floor_id"] != floor_id:
                raise ValueError("unassigned zone belongs to a different floor")
            self._connection.execute(
                """INSERT INTO components(
                    component_id,building_id,session_id,source_config_sha256,calibration_id,
                    attachment_state,route_state,closing_cutoff_sequence
                ) VALUES(?,?,?,?,?,?,?,NULL)""",
                (
                    component_id,
                    self.building_id,
                    session_id,
                    source_config_sha256,
                    calibration_id,
                    attachment.value,
                    RouteState.OPEN.value,
                ),
            )
            self._connection.execute(
                """INSERT INTO submaps(
                    submap_id,floor_id,zone_id,component_id,session_id,map_id,producer_instance_id,
                    source_config_sha256,lifecycle,route_start_sequence
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    submap_id,
                    floor_id,
                    zone_id,
                    component_id,
                    session_id,
                    map_id,
                    producer_instance_id,
                    source_config_sha256,
                    SubmapLifecycle.ACTIVE.value,
                    start_sequence,
                ),
            )
            self._connection.execute(
                """INSERT INTO route_ranges(
                    building_id,session_id,component_id,map_id,producer_instance_id,
                    submap_id,start_sequence,end_sequence
                ) VALUES(?,?,?,?,?,?,?,NULL)""",
                (
                    self.building_id,
                    session_id,
                    component_id,
                    map_id,
                    producer_instance_id,
                    submap_id,
                    start_sequence,
                ),
            )
            self._set_operational_pointer(session_id, component_id)
        (self.root / "submaps" / submap_id / "generations").mkdir(parents=True, exist_ok=True)
        return submap_id

    def _set_operational_pointer(self, session_id: str, component_id: str) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('operational_session_id',?)",
            (session_id,),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('operational_component_id',?)",
            (component_id,),
        )

    def begin_handoff(self, *, old_submap_id: str, new_submap_id: str, cutoff_sequence: int) -> None:
        old_submap_id = _identifier(old_submap_id, "old_submap_id")
        new_submap_id = _identifier(new_submap_id, "new_submap_id")
        if isinstance(cutoff_sequence, bool) or int(cutoff_sequence) < 0:
            raise ValueError("cutoff_sequence must be non-negative")
        cutoff = int(cutoff_sequence)
        with self._transaction():
            old = self._connection.execute("SELECT * FROM submaps WHERE submap_id=?", (old_submap_id,)).fetchone()
            if old is None or old["lifecycle"] != SubmapLifecycle.ACTIVE.value:
                raise ValueError("old submap must be ACTIVE")
            if cutoff < int(old["route_start_sequence"]):
                raise ValueError("cutoff_sequence precedes the active route")
            self._connection.execute(
                "UPDATE submaps SET lifecycle=?,route_cutoff_sequence=? WHERE submap_id=?",
                (SubmapLifecycle.DRAINING.value, cutoff, old_submap_id),
            )
            updated = self._connection.execute(
                "UPDATE route_ranges SET end_sequence=? WHERE submap_id=? AND end_sequence IS NULL",
                (cutoff, old_submap_id),
            )
            if updated.rowcount != 1:
                raise ValueError("active route range is missing")
            self._connection.execute(
                """INSERT INTO submaps(
                    submap_id,floor_id,zone_id,component_id,session_id,map_id,producer_instance_id,
                    source_config_sha256,lifecycle,route_start_sequence
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_submap_id,
                    old["floor_id"],
                    old["zone_id"],
                    old["component_id"],
                    old["session_id"],
                    old["map_id"],
                    old["producer_instance_id"],
                    old["source_config_sha256"],
                    SubmapLifecycle.ACTIVE.value,
                    cutoff + 1,
                ),
            )
            self._connection.execute(
                """INSERT INTO route_ranges(
                    building_id,session_id,component_id,map_id,producer_instance_id,submap_id,start_sequence,end_sequence
                ) VALUES(?,?,?,?,?,?,?,NULL)""",
                (
                    self.building_id,
                    old["session_id"],
                    old["component_id"],
                    old["map_id"],
                    old["producer_instance_id"],
                    new_submap_id,
                    cutoff + 1,
                ),
            )
        (self.root / "submaps" / new_submap_id / "generations").mkdir(parents=True, exist_ok=True)

    def session_source(self, session_id: str) -> dict:
        session_id = _identifier(session_id, "session_id")
        component = self._connection.execute(
            "SELECT * FROM components WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if component is None:
            raise KeyError(session_id)
        lifecycle = {
            RouteState.OPEN.value: SubmapLifecycle.ACTIVE.value,
            RouteState.CLOSING.value: SubmapLifecycle.DRAINING.value,
            RouteState.CLOSED.value: SubmapLifecycle.SEALED.value,
        }[str(component["route_state"])]
        rows = self._connection.execute(
            "SELECT * FROM submaps WHERE component_id=? AND lifecycle=? ORDER BY row_id DESC",
            (component["component_id"], lifecycle),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("session component does not have one lifecycle-authoritative submap")
        return {
            **self.component(str(component["component_id"])),
            **self.submap(str(rows[0]["submap_id"])),
        }

    def surface_sources(self) -> tuple[dict, ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT submap_id FROM submaps
                   WHERE component_id IS NOT NULL AND session_id IS NOT NULL AND lifecycle<>?
                   ORDER BY row_id""",
                (SubmapLifecycle.QUARANTINED.value,),
            ).fetchall()
            sources = []
            for row in rows:
                submap = self.submap(str(row["submap_id"]))
                sources.append(
                    {
                        **self.component(str(submap["component_id"])),
                        **submap,
                    }
                )
            return tuple(sources)

    def highest_routed_sequence(self, session_id: str) -> Optional[int]:
        session_id = _identifier(session_id, "session_id")
        row = self._connection.execute(
            "SELECT MAX(sequence) AS sequence FROM frame_routes WHERE building_id=? AND session_id=?",
            (self.building_id, session_id),
        ).fetchone()
        return None if row is None or row["sequence"] is None else int(row["sequence"])

    def begin_session_close(self, session_id: str, cutoff_sequence: int) -> dict:
        session_id = _identifier(session_id, "session_id")
        if isinstance(cutoff_sequence, bool) or int(cutoff_sequence) < 0:
            raise ValueError("cutoff_sequence must be non-negative")
        cutoff = int(cutoff_sequence)
        with self._transaction():
            component = self._connection.execute(
                "SELECT * FROM components WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if component is None:
                raise KeyError(session_id)
            if component["route_state"] in {RouteState.CLOSING.value, RouteState.CLOSED.value}:
                if component["closing_cutoff_sequence"] != cutoff:
                    raise ValueError("session close cutoff conflicts with durable closing cutoff")
                return self.session_source(session_id)
            active = self._connection.execute(
                "SELECT * FROM submaps WHERE component_id=? AND lifecycle=?",
                (component["component_id"], SubmapLifecycle.ACTIVE.value),
            ).fetchall()
            if len(active) != 1:
                raise ValueError("session close requires exactly one active submap")
            submap = active[0]
            route = self._connection.execute(
                """SELECT * FROM route_ranges WHERE component_id=? AND submap_id=?
                   AND end_sequence IS NULL""",
                (component["component_id"], submap["submap_id"]),
            ).fetchall()
            if len(route) != 1 or cutoff < int(route[0]["start_sequence"]):
                raise ValueError("session close cutoff does not cover the active route")
            self._connection.execute(
                "UPDATE components SET route_state=?,closing_cutoff_sequence=? WHERE component_id=?",
                (RouteState.CLOSING.value, cutoff, component["component_id"]),
            )
            self._connection.execute(
                "UPDATE submaps SET lifecycle=?,route_cutoff_sequence=? WHERE submap_id=?",
                (SubmapLifecycle.DRAINING.value, cutoff, submap["submap_id"]),
            )
            self._connection.execute(
                "UPDATE route_ranges SET end_sequence=? WHERE route_id=?",
                (cutoff, route[0]["route_id"]),
            )
            selected = self._connection.execute(
                "SELECT value FROM metadata WHERE key='operational_component_id'"
            ).fetchone()
            if selected is not None and selected["value"] == component["component_id"]:
                self._connection.execute(
                    "DELETE FROM metadata WHERE key IN ('operational_session_id','operational_component_id')"
                )
        return self.session_source(session_id)

    def complete_session_close(
        self,
        submap_id: str,
        *,
        generation_id: str,
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
        cutoff_sequence: int,
    ) -> dict:
        submap_id = _identifier(submap_id, "submap_id")
        generation_id = _sha256(generation_id, "generation_id")
        cutoff = int(cutoff_sequence)
        lower = tuple(float(value) for value in bounds_min)
        upper = tuple(float(value) for value in bounds_max)
        if len(lower) != 3 or len(upper) != 3 or not all(math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("submap bounds must contain finite XYZ values")
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError("submap bounds are inverted")
        with self._transaction():
            row = self._connection.execute(
                """SELECT s.*,c.route_state,c.closing_cutoff_sequence,
                          c.calibration_id AS component_calibration_id
                   FROM submaps AS s JOIN components AS c ON c.component_id=s.component_id
                   WHERE s.submap_id=?""",
                (submap_id,),
            ).fetchone()
            if row is None:
                raise ValueError("submap does not exist")
            if row["lifecycle"] == SubmapLifecycle.SEALED.value and row["route_state"] == RouteState.CLOSED.value:
                if row["generation_id"] != generation_id or row["closing_cutoff_sequence"] != cutoff:
                    raise ValueError("completed session close is immutable")
                return self.session_source(str(row["session_id"]))
            if row["lifecycle"] != SubmapLifecycle.DRAINING.value or row["route_state"] != RouteState.CLOSING.value:
                raise ValueError("session close completion requires DRAINING+CLOSING state")
            if row["closing_cutoff_sequence"] != cutoff or row["route_cutoff_sequence"] != cutoff:
                raise ValueError("session close cutoff conflicts with durable state")
            generation_path = self.root / "submaps" / submap_id / "generations" / generation_id
            try:
                manifest = json.loads((generation_path / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("complete immutable surface generation is missing") from exc
            core = {key: value for key, value in manifest.items() if key != "generation_id"}
            canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if manifest.get("generation_id") != generation_id or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != generation_id:
                raise ValueError("surface generation identity is invalid")
            expected = {
                "building_id": self.building_id,
                "session_id": row["session_id"],
                "floor_id": row["floor_id"],
                "submap_id": submap_id,
                "map_id": row["map_id"],
                "source_config_sha256": row["source_config_sha256"],
                "calibration_id": row["component_calibration_id"],
            }
            if any(manifest.get(name) != value for name, value in expected.items()):
                raise ValueError("surface generation provenance is invalid")
            artifacts = [("surfels.npz", "arrays_sha256")]
            if manifest.get("schema_version") != 4:
                artifacts.append(("witnesses.json", "witnesses_sha256"))
            for filename, digest_key in artifacts:
                try:
                    digest = hashlib.sha256((generation_path / filename).read_bytes()).hexdigest()
                except OSError as exc:
                    raise ValueError("surface generation artifacts are incomplete") from exc
                if digest != manifest.get(digest_key):
                    raise ValueError("surface generation artifact hash is invalid")
            self._connection.execute(
                """UPDATE submaps SET lifecycle=?,generation_id=?,generation_path=?,
                   bounds_min_json=?,bounds_max_json=? WHERE submap_id=?""",
                (
                    SubmapLifecycle.SEALED.value,
                    generation_id,
                    generation_path.relative_to(self.root).as_posix(),
                    json.dumps(lower, separators=(",", ":")),
                    json.dumps(upper, separators=(",", ":")),
                    submap_id,
                ),
            )
            self._connection.execute(
                "UPDATE components SET route_state=? WHERE component_id=?",
                (RouteState.CLOSED.value, row["component_id"]),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO submap_bounds VALUES(?,?,?,?,?,?,?)",
                (row["row_id"], lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]),
            )
        return self.session_source(str(row["session_id"]))

    def route_frame(self, identity: FrameIdentity) -> str:
        if identity.building_id != self.building_id:
            raise ValueError("frame building_id does not match the project")
        with self._transaction():
            existing = self._connection.execute(
                """SELECT r.submap_id,s.session_id AS submap_session_id,
                          s.component_id AS submap_component_id,s.lifecycle,
                          c.route_state,c.closing_cutoff_sequence
                   FROM frame_routes AS r
                   LEFT JOIN submaps AS s ON s.submap_id=r.submap_id
                   LEFT JOIN components AS c ON c.component_id=s.component_id
                   WHERE r.building_id=? AND r.session_id=? AND r.map_id=?
                     AND r.producer_instance_id=? AND r.sequence=?""",
                identity.as_tuple(),
            ).fetchone()
            if existing is not None:
                if existing["submap_session_id"] != identity.session_id:
                    raise ValueError("cached frame route session does not match submap session provenance")
                replayable_closing = (
                    existing["route_state"] == RouteState.CLOSING.value
                    and existing["lifecycle"] == SubmapLifecycle.DRAINING.value
                    and existing["closing_cutoff_sequence"] is not None
                    and identity.sequence <= int(existing["closing_cutoff_sequence"])
                )
                if existing["submap_component_id"] is None or (
                    existing["route_state"] != RouteState.OPEN.value and not replayable_closing
                ):
                    raise ValueError("cached frame route component is not open or authoritative")
                return str(existing["submap_id"])
            route = self._connection.execute(
                """SELECT r.submap_id FROM route_ranges AS r
                   JOIN submaps AS s ON s.submap_id=r.submap_id
                   JOIN components AS c ON c.component_id=r.component_id
                   WHERE r.building_id=? AND r.session_id=? AND r.map_id=? AND r.producer_instance_id=?
                     AND s.session_id=r.session_id AND s.component_id=r.component_id
                     AND c.session_id=r.session_id AND c.route_state=? AND r.start_sequence<=?
                     AND (r.end_sequence IS NULL OR r.end_sequence>=?)
                   ORDER BY r.start_sequence DESC LIMIT 1""",
                (
                    self.building_id,
                    identity.session_id,
                    identity.map_id,
                    identity.producer_instance_id,
                    RouteState.OPEN.value,
                    identity.sequence,
                    identity.sequence,
                ),
            ).fetchone()
            if route is None:
                any_route = self._connection.execute(
                    """SELECT r.session_id FROM route_ranges AS r
                       JOIN submaps AS s ON s.submap_id=r.submap_id
                       WHERE r.building_id=? AND r.map_id=? AND r.producer_instance_id=? AND r.start_sequence<=?
                         AND (r.end_sequence IS NULL OR r.end_sequence>=?)
                       ORDER BY r.start_sequence DESC LIMIT 1""",
                    (self.building_id, identity.map_id, identity.producer_instance_id, identity.sequence, identity.sequence),
                ).fetchone()
                if any_route is not None:
                    raise ValueError("frame session does not match durable submap session provenance")
                raise ValueError("no durable submap route covers this frame")
            submap_id = str(route["submap_id"])
            self._connection.execute(
                "INSERT INTO frame_routes VALUES(?,?,?,?,?,?)",
                (*identity.as_tuple(), submap_id),
            )
            return submap_id

    def seal_submap(
        self,
        submap_id: str,
        *,
        generation_id: str,
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
    ) -> None:
        submap_id = _identifier(submap_id, "submap_id")
        generation_id = _sha256(generation_id, "generation_id")
        lower = tuple(float(value) for value in bounds_min)
        upper = tuple(float(value) for value in bounds_max)
        if len(lower) != 3 or len(upper) != 3 or not all(math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("submap bounds must contain finite XYZ values")
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError("submap bounds are inverted")
        generation_path = self.root / "submaps" / submap_id / "generations" / generation_id
        if not (generation_path / "manifest.json").is_file():
            raise ValueError("complete immutable surface generation is missing")
        with self._transaction():
            row = self._connection.execute(
                """SELECT s.*,c.calibration_id AS component_calibration_id
                   FROM submaps AS s LEFT JOIN components AS c ON c.component_id=s.component_id
                   WHERE s.submap_id=?""",
                (submap_id,),
            ).fetchone()
            if row is None:
                raise ValueError("submap does not exist")
            if row["lifecycle"] == SubmapLifecycle.SEALED.value:
                raise ValueError("sealed submap generation is immutable")
            if row["lifecycle"] != SubmapLifecycle.DRAINING.value:
                raise ValueError("only a DRAINING submap can be sealed")
            self._connection.execute(
                """UPDATE submaps SET lifecycle=?,generation_id=?,generation_path=?,bounds_min_json=?,bounds_max_json=?
                   WHERE submap_id=?""",
                (
                    SubmapLifecycle.SEALED.value,
                    generation_id,
                    generation_path.relative_to(self.root).as_posix(),
                    json.dumps(lower, separators=(",", ":")),
                    json.dumps(upper, separators=(",", ":")),
                    submap_id,
                ),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO submap_bounds VALUES(?,?,?,?,?,?,?)",
                (row["row_id"], lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]),
            )

    def create_quarantine(
        self,
        *,
        submap_id: str,
        floor_id: str,
        zone_id: str,
        reason: str,
        source_config_sha256: str,
    ) -> None:
        submap_id = _identifier(submap_id, "submap_id")
        floor_id = _identifier(floor_id, "floor_id")
        zone_id = _identifier(zone_id, "zone_id")
        source_config_sha256 = _sha256(source_config_sha256, "source_config_sha256")
        if not str(reason).strip():
            raise ValueError("quarantine reason is required")
        with self._transaction():
            self._connection.execute(
                """INSERT INTO submaps(
                    submap_id,floor_id,zone_id,source_config_sha256,lifecycle,quarantine_reason
                ) VALUES(?,?,?,?,?,?)""",
                (
                    submap_id,
                    floor_id,
                    zone_id,
                    source_config_sha256,
                    SubmapLifecycle.QUARANTINED.value,
                    str(reason).strip(),
                ),
            )
        (self.root / "submaps" / submap_id / "generations").mkdir(parents=True, exist_ok=True)

    def submap(self, submap_id: str) -> dict:
        row = self._connection.execute("SELECT * FROM submaps WHERE submap_id=?", (submap_id,)).fetchone()
        if row is None:
            raise KeyError(submap_id)
        return {
            "submap_id": row["submap_id"],
            "floor_id": row["floor_id"],
            "zone_id": row["zone_id"],
            "component_id": row["component_id"],
            "session_id": row["session_id"],
            "map_id": row["map_id"],
            "producer_instance_id": row["producer_instance_id"],
            "source_config_sha256": row["source_config_sha256"],
            "lifecycle": row["lifecycle"],
            "route_start_sequence": row["route_start_sequence"],
            "route_cutoff_sequence": row["route_cutoff_sequence"],
            "generation_id": row["generation_id"],
            "generation_path": row["generation_path"],
            "quarantine_reason": row["quarantine_reason"],
            "bounds_min": None if row["bounds_min_json"] is None else json.loads(row["bounds_min_json"]),
            "bounds_max": None if row["bounds_max_json"] is None else json.loads(row["bounds_max_json"]),
        }

    def component(self, component_id: str) -> dict:
        component_id = _identifier(component_id, "component_id")
        row = self._connection.execute(
            "SELECT * FROM components WHERE component_id=?",
            (component_id,),
        ).fetchone()
        if row is None:
            raise KeyError(component_id)
        return {
            "component_id": row["component_id"],
            "building_id": row["building_id"],
            "session_id": row["session_id"],
            "source_config_sha256": row["source_config_sha256"],
            "calibration_id": row["calibration_id"],
            "attachment_state": row["attachment_state"],
            "route_state": row["route_state"],
            "closing_cutoff_sequence": row["closing_cutoff_sequence"],
        }

    def sealed_generation(self, submap_id: str, generation_id: str):
        submap_id = _identifier(submap_id, "submap_id")
        generation_id = _sha256(generation_id, "generation_id")
        with self._lock:
            row = self._connection.execute(
                """SELECT s.*,c.calibration_id AS component_calibration_id
                   FROM submaps AS s LEFT JOIN components AS c ON c.component_id=s.component_id
                   WHERE s.submap_id=?""",
                (submap_id,),
            ).fetchone()
            if row is None:
                raise ValueError("submap does not exist")
            if row["lifecycle"] != SubmapLifecycle.SEALED.value:
                raise ValueError("relocalization target must be SEALED")
            if row["session_id"] is None:
                raise ValueError("sealed submap lacks session provenance")
            if row["generation_id"] != generation_id:
                raise ValueError("sealed submap does not reference the exact sealed generation")
            relative_path = Path("submaps") / submap_id / "generations" / generation_id
            if row["generation_path"] != relative_path.as_posix():
                raise ValueError("sealed generation path is invalid")
            generation_path = self.root / relative_path
            manifest_path = generation_path / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("sealed generation manifest is unreadable") from exc
            if not isinstance(manifest, dict) or manifest.get("generation_id") != generation_id:
                raise ValueError("sealed generation manifest identity mismatch")
            manifest_core = {key: value for key, value in manifest.items() if key != "generation_id"}
            canonical = json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != generation_id:
                raise ValueError("sealed generation manifest hash mismatch")
            expected_identities = {
                "building_id": self.building_id,
                "session_id": row["session_id"],
                "floor_id": row["floor_id"],
                "submap_id": submap_id,
            }
            for name, expected in expected_identities.items():
                if manifest.get(name) != expected:
                    raise ValueError(f"sealed generation manifest {name} mismatch")
            if manifest.get("map_id") != row["map_id"]:
                raise ValueError("sealed generation manifest map_id mismatch")
            if manifest.get("source_config_sha256") != row["source_config_sha256"]:
                raise ValueError("sealed generation manifest configuration mismatch")
            if row["component_calibration_id"] is None:
                raise ValueError("sealed submap lacks calibration provenance")
            if manifest.get("calibration_id") != row["component_calibration_id"]:
                raise ValueError("sealed generation manifest calibration mismatch")
            artifacts = [("surfels.npz", "arrays_sha256")]
            if manifest.get("schema_version") != 4:
                artifacts.append(("witnesses.json", "witnesses_sha256"))
            for filename, digest_key in artifacts:
                artifact = generation_path / filename
                try:
                    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                except OSError as exc:
                    raise ValueError("sealed generation manifest is incomplete") from exc
                if manifest.get(digest_key) != digest:
                    raise ValueError(f"sealed generation manifest {digest_key} mismatch")
            return MappingProxyType(
                {
                    "building_id": self.building_id,
                    "session_id": row["session_id"],
                    "floor_id": row["floor_id"],
                    "zone_id": row["zone_id"],
                    "submap_id": submap_id,
                    "generation_id": generation_id,
                    "source_map_id": row["map_id"],
                    "map_id": row["map_id"],
                    "source_config_sha256": row["source_config_sha256"],
                    "calibration_id": row["component_calibration_id"],
                    "path": generation_path,
                }
            )

    def snapshot(self) -> dict:
        counts = {
            row["lifecycle"]: int(row["count"])
            for row in self._connection.execute("SELECT lifecycle,COUNT(*) AS count FROM submaps GROUP BY lifecycle")
        }
        root = self._connection.execute(
            "SELECT component_id FROM components WHERE attachment_state=?",
            (AttachmentState.ROOT.value,),
        ).fetchone()
        operational_session = self._connection.execute(
            "SELECT value FROM metadata WHERE key='operational_session_id'"
        ).fetchone()
        operational_component = self._connection.execute(
            "SELECT value FROM metadata WHERE key='operational_component_id'"
        ).fetchone()
        if (operational_session is None) != (operational_component is None):
            raise ValueError("operational session pointer is incomplete")
        active = None
        if operational_component is not None:
            component = self._connection.execute(
                "SELECT * FROM components WHERE component_id=? AND session_id=?",
                (operational_component["value"], operational_session["value"]),
            ).fetchone()
            if component is None or component["route_state"] != RouteState.OPEN.value:
                raise ValueError("operational session pointer is not open")
            active = self._connection.execute(
                """SELECT submap_id,floor_id,zone_id FROM submaps
                   WHERE component_id=? AND lifecycle=? ORDER BY row_id DESC""",
                (component["component_id"], SubmapLifecycle.ACTIVE.value),
            ).fetchall()
            if len(active) != 1:
                raise ValueError("operational component does not have one active submap")
            active = active[0]
        total = sum(counts.values())
        component_count = int(self._connection.execute("SELECT COUNT(*) FROM components").fetchone()[0])
        detached_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM components WHERE attachment_state=?",
                (AttachmentState.DETACHED.value,),
            ).fetchone()[0]
        )
        return {
            "building_id": self.building_id,
            "state": "BOOTSTRAPPING" if total == 0 else "MAPPING",
            "floor_count": int(self._connection.execute("SELECT COUNT(*) FROM floors").fetchone()[0]),
            "zone_count": int(self._connection.execute("SELECT COUNT(*) FROM zones").fetchone()[0]),
            "submap_count": total,
            "root_component_id": None if root is None else root["component_id"],
            "active_component_id": None if operational_component is None else operational_component["value"],
            "active_attachment_state": (
                None
                if operational_component is None
                else self.component(operational_component["value"])["attachment_state"]
            ),
            "active_submap_id": None if active is None else active["submap_id"],
            "active_floor_id": None if active is None else active["floor_id"],
            "active_zone_id": None if active is None else active["zone_id"],
            "sealed_submaps": counts.get(SubmapLifecycle.SEALED.value, 0),
            "draining_submaps": counts.get(SubmapLifecycle.DRAINING.value, 0),
            "quarantined_submaps": counts.get(SubmapLifecycle.QUARANTINED.value, 0),
            "component_count": component_count,
            "detached_component_count": detached_count,
        }
