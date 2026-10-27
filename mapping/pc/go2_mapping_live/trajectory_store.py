"""Durable Building/floor/session trajectory history from ordered LIO keyframes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional

import numpy as np

from .live_protocol import LiveKeyframeMessage


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{name} must be a printable identifier")
    return value.strip()


class TrajectoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS trajectory_samples(
                   building_id TEXT NOT NULL,
                   session_id TEXT NOT NULL,
                   map_id TEXT NOT NULL,
                   producer_instance_id TEXT NOT NULL,
                   sequence INTEGER NOT NULL,
                   timestamp_ns INTEGER NOT NULL,
                   floor_id TEXT NOT NULL,
                   submap_id TEXT NOT NULL,
                   x_m REAL NOT NULL,
                   y_m REAL NOT NULL,
                   z_m REAL NOT NULL,
                   heading_x REAL NOT NULL,
                   heading_y REAL NOT NULL,
                   pose_json TEXT NOT NULL,
                   frame_record_sha256 TEXT NOT NULL,
                   PRIMARY KEY(building_id,session_id,map_id,producer_instance_id,sequence)
               )"""
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS trajectory_floor_session ON trajectory_samples(building_id,floor_id,session_id,timestamp_ns)"
        )
        self._connection.commit()

    def append(self, frame: LiveKeyframeMessage, *, floor_id: str, submap_id: str) -> bool:
        if not isinstance(frame, LiveKeyframeMessage):
            raise TypeError("frame must be a LiveKeyframeMessage")
        floor = _identifier(floor_id, "floor_id")
        submap = _identifier(submap_id, "submap_id")
        identity = frame.identity
        pose = np.asarray(frame.pose_map_from_lidar, dtype=np.float64)
        position = pose[:3, 3]
        heading = pose[:2, 0]
        heading_norm = float(np.linalg.norm(heading))
        if heading_norm <= 1e-9:
            heading = np.asarray([1.0, 0.0])
        else:
            heading = heading / heading_norm
        frame_digest = hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest()
        values = (
            identity.building_id,
            identity.session_id,
            identity.map_id,
            identity.producer_instance_id,
            identity.sequence,
            frame.timestamp_ns,
            floor,
            submap,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(heading[0]),
            float(heading[1]),
            json.dumps(pose.reshape(-1).tolist(), separators=(",", ":"), allow_nan=False),
            frame_digest,
        )
        key = values[:5]
        with self._lock:
            existing = self._connection.execute(
                """SELECT * FROM trajectory_samples
                   WHERE building_id=? AND session_id=? AND map_id=? AND producer_instance_id=? AND sequence=?""",
                key,
            ).fetchone()
            if existing is not None:
                existing_values = tuple(existing[name] for name in existing.keys())
                if existing_values == values:
                    return False
                raise ValueError("trajectory sample identity conflict")
            self._connection.execute(
                """INSERT INTO trajectory_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
            self._connection.commit()
            return True

    def query(
        self,
        *,
        building_id: str,
        floor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        building = _identifier(building_id, "building_id")
        floor = None if floor_id is None else _identifier(floor_id, "floor_id")
        session = None if session_id is None else _identifier(session_id, "session_id")
        clauses = ["building_id=?"]
        parameters: list[Any] = [building]
        if floor is not None:
            clauses.append("floor_id=?")
            parameters.append(floor)
        if session is not None:
            clauses.append("session_id=?")
            parameters.append(session)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM trajectory_samples WHERE {' AND '.join(clauses)} "
                "ORDER BY session_id,floor_id,map_id,producer_instance_id,timestamp_ns,sequence",
                parameters,
            ).fetchall()
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row["session_id"]),
                str(row["floor_id"]),
                str(row["submap_id"]),
                str(row["map_id"]),
                str(row["producer_instance_id"]),
            )
            grouped.setdefault(key, []).append({
                "sequence": int(row["sequence"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "position_m": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "heading_xy": [float(row["heading_x"]), float(row["heading_y"])],
            })
        trajectories = [
            {
                "session_id": key[0],
                "floor_id": key[1],
                "submap_id": key[2],
                "map_id": key[3],
                "producer_instance_id": key[4],
                "samples": samples,
            }
            for key, samples in grouped.items()
        ]
        result = {
            "schema_version": 1,
            "building_id": building,
            "filters": {"floor_id": floor, "session_id": session},
            "trajectories": trajectories,
        }
        result["revision_id"] = hashlib.sha256(self.canonical_revision_payload(result).encode("utf-8")).hexdigest()
        return result

    def resolve_submap_near(
        self,
        *,
        building_id: str,
        session_id: str,
        map_id: str,
        producer_instance_id: str,
        timestamp_ns: int,
        maximum_skew_ns: int = 250_000_000,
    ) -> dict[str, Any]:
        """Resolve a texture pose to the exact durable LIO route without crossing submaps."""
        building = _identifier(building_id, "building_id")
        session = _identifier(session_id, "session_id")
        map_identifier = _identifier(map_id, "map_id")
        producer = _identifier(producer_instance_id, "producer_instance_id")
        if isinstance(timestamp_ns, bool) or int(timestamp_ns) <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        if isinstance(maximum_skew_ns, bool) or int(maximum_skew_ns) < 0:
            raise ValueError("maximum_skew_ns must be a non-negative integer")
        target = int(timestamp_ns)
        maximum_skew = int(maximum_skew_ns)
        with self._lock:
            row = self._connection.execute(
                """SELECT submap_id,floor_id,sequence,timestamp_ns
                   FROM trajectory_samples
                   WHERE building_id=? AND session_id=? AND map_id=? AND producer_instance_id=?
                   ORDER BY ABS(timestamp_ns-?),sequence LIMIT 1""",
                (building, session, map_identifier, producer, target),
            ).fetchone()
        if row is None:
            raise ValueError("no durable LIO trajectory sample can route the texture observation")
        skew = abs(int(row["timestamp_ns"]) - target)
        if skew > maximum_skew:
            raise ValueError("nearest LIO trajectory sample exceeds the maximum skew")
        return {
            "submap_id": str(row["submap_id"]),
            "floor_id": str(row["floor_id"]),
            "sequence": int(row["sequence"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "skew_ns": skew,
        }

    @staticmethod
    def canonical_revision_payload(payload: dict[str, Any]) -> str:
        core = dict(payload)
        core.pop("revision_id", None)
        return json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
