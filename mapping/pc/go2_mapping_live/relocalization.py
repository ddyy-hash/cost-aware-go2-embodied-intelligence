"""Durable, fail-closed cross-session relocalization evidence decisions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Mapping, Optional

import numpy as np

from .building_project import BuildingProject


_IDENTIFIER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_SHA256_LENGTH = 64


class RelocalizationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or any(c not in _IDENTIFIER_CHARS for c in value):
        raise ValueError(f"{name} is invalid")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _canonical(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("record is not canonical JSON") from exc


def _record_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _rigid_transform(values: tuple[float, ...], name: str = "transform_target_from_source_frame") -> tuple[float, ...]:
    if len(values) != 16:
        raise ValueError(f"{name} must be a 16-value rigid transform")
    try:
        matrix = np.asarray(tuple(float(value) for value in values), dtype=np.float64).reshape((4, 4))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 16-value rigid transform") from exc
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must be a rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or float(np.linalg.det(rotation)) <= 0:
        raise ValueError(f"{name} must be a rigid transform")
    return tuple(float(value) for value in matrix.reshape(-1))


@dataclass(frozen=True)
class SessionFrame:
    frame_id: str
    building_id: str
    session_id: str
    source_config_sha256: str
    calibration_id: str
    frame_kind: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.frame_id, "frame_id"),
            (self.building_id, "building_id"),
            (self.session_id, "session_id"),
            (self.calibration_id, "calibration_id"),
            (self.frame_kind, "frame_kind"),
        ):
            _identifier(value, name)
        _sha256(self.source_config_sha256, "source_config_sha256")

    def payload(self) -> dict[str, str]:
        return {
            "frame_id": self.frame_id,
            "building_id": self.building_id,
            "session_id": self.session_id,
            "source_config_sha256": self.source_config_sha256,
            "calibration_id": self.calibration_id,
            "frame_kind": self.frame_kind,
        }


@dataclass(frozen=True)
class SourceGenerationAuthority:
    frame_id: str
    building_id: str
    session_id: str
    submap_id: str
    generation_id: str
    source_map_id: str
    source_config_sha256: str
    calibration_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.frame_id, "frame_id"),
            (self.building_id, "building_id"),
            (self.session_id, "session_id"),
            (self.submap_id, "submap_id"),
            (self.source_map_id, "source_map_id"),
            (self.calibration_id, "calibration_id"),
        ):
            _identifier(value, name)
        _sha256(self.generation_id, "generation_id")
        _sha256(self.source_config_sha256, "source_config_sha256")

    def payload(self) -> dict[str, str]:
        return {
            "frame_id": self.frame_id,
            "building_id": self.building_id,
            "session_id": self.session_id,
            "submap_id": self.submap_id,
            "generation_id": self.generation_id,
            "source_map_id": self.source_map_id,
            "source_config_sha256": self.source_config_sha256,
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True)
class RelocalizationPolicy:
    min_overlap_m2: float
    min_independent_keyframes: int
    max_translation_residual_m: float
    max_rotation_residual_deg: float
    max_covariance_trace: float
    algorithm_version: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.min_overlap_m2, "min_overlap_m2"),
            (self.max_translation_residual_m, "max_translation_residual_m"),
            (self.max_rotation_residual_deg, "max_rotation_residual_deg"),
            (self.max_covariance_trace, "max_covariance_trace"),
        ):
            _finite_nonnegative(value, name)
        _positive_count(self.min_independent_keyframes, "min_independent_keyframes")
        _identifier(self.algorithm_version, "algorithm_version")

    def payload(self) -> dict[str, Any]:
        return {
            "min_overlap_m2": float(self.min_overlap_m2),
            "min_independent_keyframes": int(self.min_independent_keyframes),
            "max_translation_residual_m": float(self.max_translation_residual_m),
            "max_rotation_residual_deg": float(self.max_rotation_residual_deg),
            "max_covariance_trace": float(self.max_covariance_trace),
            "algorithm_version": self.algorithm_version,
        }


@dataclass(frozen=True)
class RelocalizationCandidate:
    candidate_id: str
    attempt_id: str
    building_id: str
    session_id: str
    floor_id: str
    target_submap_id: str
    target_generation_id: str
    source_frame_id: str
    transform_target_from_source_frame: tuple[float, ...]
    overlap_m2: float
    independent_keyframes: int
    translation_residual_m: float
    rotation_residual_deg: float
    covariance_trace: float
    identity_compatible: bool
    graph_consistent: bool
    forbidden_component_bridge: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.attempt_id, "attempt_id"),
            (self.building_id, "building_id"),
            (self.session_id, "session_id"),
            (self.floor_id, "floor_id"),
            (self.target_submap_id, "target_submap_id"),
            (self.source_frame_id, "source_frame_id"),
        ):
            _identifier(value, name)
        _sha256(self.target_generation_id, "target_generation_id")
        _rigid_transform(tuple(self.transform_target_from_source_frame))
        _finite_nonnegative(self.overlap_m2, "overlap_m2")
        _positive_count(self.independent_keyframes, "independent_keyframes")
        for value, name in (
            (self.translation_residual_m, "translation_residual_m"),
            (self.rotation_residual_deg, "rotation_residual_deg"),
            (self.covariance_trace, "covariance_trace"),
        ):
            _finite_nonnegative(value, name)
        if not all(isinstance(value, bool) for value in (self.identity_compatible, self.graph_consistent, self.forbidden_component_bridge)):
            raise ValueError("candidate gates must be boolean")
        _sha256(self.evidence_sha256, "evidence_sha256")

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "attempt_id": self.attempt_id,
            "building_id": self.building_id,
            "session_id": self.session_id,
            "floor_id": self.floor_id,
            "target_submap_id": self.target_submap_id,
            "target_generation_id": self.target_generation_id,
            "source_frame_id": self.source_frame_id,
            "transform_target_from_source_frame": list(_rigid_transform(tuple(self.transform_target_from_source_frame))),
            "overlap_m2": float(self.overlap_m2),
            "independent_keyframes": int(self.independent_keyframes),
            "translation_residual_m": float(self.translation_residual_m),
            "rotation_residual_deg": float(self.rotation_residual_deg),
            "covariance_trace": float(self.covariance_trace),
            "identity_compatible": self.identity_compatible,
            "graph_consistent": self.graph_consistent,
            "forbidden_component_bridge": self.forbidden_component_bridge,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class RelocalizationDecision:
    decision_id: str
    attempt_id: str
    building_id: str
    session_id: str
    status: RelocalizationStatus
    reason_code: str
    accepted_candidate_id: Optional[str]
    accepted_target_submap_id: Optional[str]
    accepted_target_generation_id: Optional[str]
    source_frame_id: Optional[str]
    transform_target_from_source_frame: Optional[tuple[float, ...]]
    policy_sha256: str
    candidate_ids: tuple[str, ...]


class RelocalizationStore:
    def __init__(self, root: str | Path, *, project: BuildingProject) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "relocalization.sqlite3"
        self.project = project
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_frames(
                frame_id TEXT PRIMARY KEY,
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_config_sha256 TEXT NOT NULL,
                calibration_id TEXT NOT NULL,
                frame_kind TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policies(
                policy_sha256 TEXT PRIMARY KEY,
                canonical_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_generations(
                frame_id TEXT PRIMARY KEY REFERENCES session_frames(frame_id),
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                submap_id TEXT NOT NULL,
                generation_id TEXT NOT NULL,
                source_map_id TEXT NOT NULL,
                source_config_sha256 TEXT NOT NULL,
                calibration_id TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts(
                attempt_id TEXT PRIMARY KEY,
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_frame_id TEXT NOT NULL REFERENCES session_frames(frame_id),
                policy_sha256 TEXT NOT NULL REFERENCES policies(policy_sha256),
                content_json TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                state TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_attempt_per_session
                ON attempts(building_id, session_id) WHERE state='OPEN';
            CREATE TABLE IF NOT EXISTS candidates(
                candidate_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                canonical_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions(
                decision_id TEXT PRIMARY KEY,
                attempt_id TEXT UNIQUE NOT NULL REFERENCES attempts(attempt_id),
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS authoritative_acceptances(
                building_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                decision_id TEXT UNIQUE NOT NULL REFERENCES decisions(decision_id),
                PRIMARY KEY(building_id, session_id)
            );
            """
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

    @staticmethod
    def _frame_from_payload(payload: dict[str, Any]) -> SessionFrame:
        return SessionFrame(**payload)

    @staticmethod
    def _policy_from_payload(payload: dict[str, Any]) -> RelocalizationPolicy:
        return RelocalizationPolicy(**payload)

    @staticmethod
    def _candidate_from_payload(payload: dict[str, Any]) -> RelocalizationCandidate:
        payload = dict(payload)
        payload["transform_target_from_source_frame"] = tuple(payload["transform_target_from_source_frame"])
        return RelocalizationCandidate(**payload)

    @staticmethod
    def _decision_from_payload(payload: dict[str, Any]) -> RelocalizationDecision:
        values = dict(payload)
        transform = values.get("transform_target_from_source_frame")
        values["transform_target_from_source_frame"] = None if transform is None else tuple(transform)
        values["status"] = RelocalizationStatus(values["status"])
        values["candidate_ids"] = tuple(values["candidate_ids"])
        return RelocalizationDecision(**values)

    def register_session_frame(self, frame: SessionFrame) -> SessionFrame:
        canonical = _canonical(frame.payload())
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._transaction():
            existing = self._connection.execute("SELECT * FROM session_frames WHERE frame_id=?", (frame.frame_id,)).fetchone()
            if existing is not None:
                if existing["record_sha256"] != digest or existing["canonical_json"] != canonical:
                    raise ValueError("conflicting session frame ID reuse")
                return frame
            if frame.building_id != self.project.building_id:
                raise ValueError("session frame building_id does not match project")
            self._connection.execute(
                "INSERT INTO session_frames VALUES(?,?,?,?,?,?,?,?)",
                (frame.frame_id, frame.building_id, frame.session_id, frame.source_config_sha256, frame.calibration_id, frame.frame_kind, canonical, digest),
            )
        return frame

    def session_frame(self, frame_id: str) -> SessionFrame:
        with self._lock:
            row = self._connection.execute("SELECT * FROM session_frames WHERE frame_id=?", (frame_id,)).fetchone()
            if row is None:
                raise KeyError(frame_id)
            if hashlib.sha256(row["canonical_json"].encode("utf-8")).hexdigest() != row["record_sha256"]:
                raise ValueError("session frame record is corrupt")
            payload = json.loads(row["canonical_json"])
            frame = self._frame_from_payload(payload)
            expected_columns = {
                "frame_id": frame.frame_id,
                "building_id": frame.building_id,
                "session_id": frame.session_id,
                "source_config_sha256": frame.source_config_sha256,
                "calibration_id": frame.calibration_id,
                "frame_kind": frame.frame_kind,
            }
            if any(row[name] != value for name, value in expected_columns.items()):
                raise ValueError("session frame indexed identity is corrupt")
            return frame

    @staticmethod
    def _source_authority_from_payload(payload: dict[str, Any]) -> SourceGenerationAuthority:
        return SourceGenerationAuthority(**payload)

    def bind_source_generation(
        self,
        frame_id: str,
        submap_id: str,
        generation_id: str,
    ) -> SourceGenerationAuthority:
        frame_id = _identifier(frame_id, "frame_id")
        submap_id = _identifier(submap_id, "submap_id")
        generation_id = _sha256(generation_id, "generation_id")
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM source_generations WHERE frame_id=?",
                (frame_id,),
            ).fetchone()
            if existing is not None and (
                existing["submap_id"] != submap_id or existing["generation_id"] != generation_id
            ):
                raise ValueError("conflicting source generation binding")
            frame = self.session_frame(frame_id)
            if frame.frame_kind != "LIO_MAP":
                raise ValueError("source generation binding requires an LIO_MAP frame")
            source = self.project.sealed_generation(submap_id, generation_id)
            if (
                source["building_id"] != frame.building_id
                or source["session_id"] != frame.session_id
                or source["source_config_sha256"] != frame.source_config_sha256
                or source["calibration_id"] != frame.calibration_id
            ):
                raise ValueError("source generation binding provenance does not match session frame")
            authority = SourceGenerationAuthority(
                frame_id=frame.frame_id,
                building_id=frame.building_id,
                session_id=frame.session_id,
                submap_id=submap_id,
                generation_id=generation_id,
                source_map_id=str(source["source_map_id"]),
                source_config_sha256=frame.source_config_sha256,
                calibration_id=frame.calibration_id,
            )
            canonical = _canonical(authority.payload())
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if existing is not None:
                if existing["canonical_json"] != canonical or existing["record_sha256"] != digest:
                    raise ValueError("conflicting source generation binding")
                return authority
            self._connection.execute(
                "INSERT INTO source_generations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    authority.frame_id,
                    authority.building_id,
                    authority.session_id,
                    authority.submap_id,
                    authority.generation_id,
                    authority.source_map_id,
                    authority.source_config_sha256,
                    authority.calibration_id,
                    canonical,
                    digest,
                ),
            )
            return authority

    def source_generation(self, frame_id: str) -> SourceGenerationAuthority:
        frame_id = _identifier(frame_id, "frame_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM source_generations WHERE frame_id=?",
                (frame_id,),
            ).fetchone()
            if row is None:
                raise ValueError("source generation binding is required")
            if hashlib.sha256(row["canonical_json"].encode("utf-8")).hexdigest() != row["record_sha256"]:
                raise ValueError("source generation binding is corrupt")
            authority = self._source_authority_from_payload(json.loads(row["canonical_json"]))
            indexed = {
                "frame_id": row["frame_id"],
                "building_id": row["building_id"],
                "session_id": row["session_id"],
                "submap_id": row["submap_id"],
                "generation_id": row["generation_id"],
                "source_map_id": row["source_map_id"],
                "source_config_sha256": row["source_config_sha256"],
                "calibration_id": row["calibration_id"],
            }
            if authority.payload() != indexed:
                raise ValueError("source generation binding indexed identity is corrupt")
            frame = self.session_frame(frame_id)
            source = self.project.sealed_generation(authority.submap_id, authority.generation_id)
            if (
                frame.frame_kind != "LIO_MAP"
                or frame.building_id != authority.building_id
                or frame.session_id != authority.session_id
                or frame.source_config_sha256 != authority.source_config_sha256
                or frame.calibration_id != authority.calibration_id
                or source["building_id"] != authority.building_id
                or source["session_id"] != authority.session_id
                or source["source_map_id"] != authority.source_map_id
                or source["source_config_sha256"] != authority.source_config_sha256
                or source["calibration_id"] != authority.calibration_id
            ):
                raise ValueError("source generation binding provenance is invalid")
            return authority

    def _policy_row(self, policy: RelocalizationPolicy) -> tuple[str, str]:
        canonical = _canonical(policy.payload())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical

    def start_attempt(self, building_id: str, session_id: str, attempt_id: str, source_frame_id: str, policy: RelocalizationPolicy) -> str:
        building_id = _identifier(building_id, "building_id")
        session_id = _identifier(session_id, "session_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        source_frame_id = _identifier(source_frame_id, "source_frame_id")
        policy_sha256, policy_json = self._policy_row(policy)
        with self._transaction():
            frame = self.session_frame(source_frame_id)
            self.source_generation(source_frame_id)
            if frame.building_id != building_id or frame.session_id != session_id:
                raise ValueError("source frame provenance does not match attempt")
            accepted = self._connection.execute(
                "SELECT 1 FROM authoritative_acceptances WHERE building_id=? AND session_id=?",
                (building_id, session_id),
            ).fetchone()
            if accepted is not None:
                raise ValueError("session already has an accepted decision")
            self._connection.execute("INSERT OR IGNORE INTO policies VALUES(?,?)", (policy_sha256, policy_json))
            persisted_policy = self._connection.execute(
                "SELECT canonical_json FROM policies WHERE policy_sha256=?", (policy_sha256,)
            ).fetchone()
            if persisted_policy is None or persisted_policy["canonical_json"] != policy_json:
                raise ValueError("policy record is corrupt")
            content = {
                "attempt_id": attempt_id,
                "building_id": building_id,
                "session_id": session_id,
                "source_frame_id": source_frame_id,
                "policy_sha256": policy_sha256,
            }
            content_json = _canonical(content)
            content_hash = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
            existing = self._connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing is not None:
                if existing["content_sha256"] != content_hash or existing["content_json"] != content_json:
                    raise ValueError("conflicting attempt ID reuse")
                if existing["state"] == "OPEN":
                    return attempt_id
                raise ValueError("attempt is already sealed")
            open_attempt = self._connection.execute(
                "SELECT attempt_id FROM attempts WHERE building_id=? AND session_id=? AND state='OPEN'",
                (building_id, session_id),
            ).fetchone()
            if open_attempt is not None:
                raise ValueError(f"session already has OPEN attempt {open_attempt['attempt_id']}")
            self._connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)",
                (attempt_id, building_id, session_id, source_frame_id, policy_sha256, content_json, content_hash, "OPEN"),
            )
        return attempt_id

    def _load_policy(self, policy_sha256: str) -> RelocalizationPolicy:
        row = self._connection.execute("SELECT canonical_json FROM policies WHERE policy_sha256=?", (policy_sha256,)).fetchone()
        if row is None or hashlib.sha256(row["canonical_json"].encode("utf-8")).hexdigest() != policy_sha256:
            raise ValueError("policy record is corrupt")
        return self._policy_from_payload(json.loads(row["canonical_json"]))

    def propose(self, candidate: RelocalizationCandidate) -> RelocalizationCandidate:
        with self._transaction():
            attempt = self._connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (candidate.attempt_id,)).fetchone()
            if attempt is None:
                raise ValueError("attempt does not exist")
            if attempt["state"] != "OPEN":
                raise ValueError("attempt is not OPEN")
            frame = self.session_frame(attempt["source_frame_id"])
            self.source_generation(frame.frame_id)
            if candidate.building_id != attempt["building_id"] or candidate.session_id != attempt["session_id"]:
                raise ValueError("candidate provenance does not match attempt")
            if candidate.source_frame_id != frame.frame_id:
                raise ValueError("candidate source frame does not match attempt")
            self.project.sealed_generation(candidate.target_submap_id, candidate.target_generation_id)
            canonical = _canonical(candidate.payload())
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            existing = self._connection.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate.candidate_id,)).fetchone()
            if existing is not None:
                if existing["attempt_id"] != candidate.attempt_id or existing["record_sha256"] != digest or existing["canonical_json"] != canonical:
                    raise ValueError("conflicting candidate ID reuse")
                return candidate
            self._connection.execute(
                "INSERT INTO candidates VALUES(?,?,?,?)",
                (candidate.candidate_id, candidate.attempt_id, canonical, digest),
            )
        return candidate

    def _candidate_rows(self, attempt_id: str) -> list[RelocalizationCandidate]:
        rows = self._connection.execute("SELECT * FROM candidates WHERE attempt_id=? ORDER BY candidate_id", (attempt_id,)).fetchall()
        result = []
        for row in rows:
            if hashlib.sha256(row["canonical_json"].encode("utf-8")).hexdigest() != row["record_sha256"]:
                raise ValueError("candidate record is corrupt")
            candidate = self._candidate_from_payload(json.loads(row["canonical_json"]))
            if candidate.candidate_id != row["candidate_id"] or candidate.attempt_id != row["attempt_id"]:
                raise ValueError("candidate indexed identity is corrupt")
            result.append(candidate)
        return result

    @staticmethod
    def _passes(candidate: RelocalizationCandidate, policy: RelocalizationPolicy, target: Mapping[str, Any]) -> bool:
        return (
            candidate.building_id == target["building_id"]
            and candidate.floor_id == target["floor_id"]
            and candidate.identity_compatible
            and candidate.graph_consistent
            and not candidate.forbidden_component_bridge
            and candidate.overlap_m2 >= policy.min_overlap_m2
            and candidate.independent_keyframes >= policy.min_independent_keyframes
            and candidate.translation_residual_m <= policy.max_translation_residual_m
            and candidate.rotation_residual_deg <= policy.max_rotation_residual_deg
            and candidate.covariance_trace <= policy.max_covariance_trace
        )

    def seal_and_resolve(self, attempt_id: str) -> RelocalizationDecision:
        attempt_id = _identifier(attempt_id, "attempt_id")
        with self._transaction():
            existing_decision = self._connection.execute("SELECT * FROM decisions WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing_decision is not None:
                return self._decision_from_row(existing_decision)
            attempt = self._connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt["state"] != "OPEN":
                raise ValueError("attempt is already sealed without a decision")
            accepted = None
            status = RelocalizationStatus.INSUFFICIENT_EVIDENCE
            authority = self._connection.execute(
                "SELECT 1 FROM authoritative_acceptances WHERE building_id=? AND session_id=?",
                (attempt["building_id"], attempt["session_id"]),
            ).fetchone()
            if authority is not None:
                reason = "SUPERSEDED_BY_ACCEPTED_SESSION"
            else:
                reason = "ATTEMPT_REVALIDATION_FAILED"
                try:
                    content = json.loads(attempt["content_json"])
                    if hashlib.sha256(attempt["content_json"].encode("utf-8")).hexdigest() != attempt["content_sha256"]:
                        raise ValueError("attempt hash mismatch")
                    indexed = {
                        "attempt_id": attempt["attempt_id"],
                        "building_id": attempt["building_id"],
                        "session_id": attempt["session_id"],
                        "source_frame_id": attempt["source_frame_id"],
                        "policy_sha256": attempt["policy_sha256"],
                    }
                    if content != indexed:
                        raise ValueError("attempt indexed identity mismatch")
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                else:
                    reason = "SOURCE_FRAME_REVALIDATION_FAILED"
                    try:
                        frame = self.session_frame(attempt["source_frame_id"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        pass
                    else:
                        reason = "SOURCE_GENERATION_REVALIDATION_FAILED"
                        try:
                            self.source_generation(frame.frame_id)
                        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                            pass
                        else:
                            reason = "POLICY_REVALIDATION_FAILED"
                            try:
                                policy = self._load_policy(attempt["policy_sha256"])
                            except (TypeError, ValueError, json.JSONDecodeError):
                                pass
                            else:
                                reason = "CANDIDATE_REVALIDATION_FAILED"
                                try:
                                    candidates = self._candidate_rows(attempt_id)
                                except (TypeError, ValueError, json.JSONDecodeError):
                                    pass
                                else:
                                    passing = []
                                    reason = "TARGET_REVALIDATION_FAILED"
                                    targets_valid = True
                                    for candidate in candidates:
                                        try:
                                            target = self.project.sealed_generation(
                                                candidate.target_submap_id, candidate.target_generation_id
                                            )
                                        except (KeyError, TypeError, ValueError):
                                            targets_valid = False
                                            break
                                        if (
                                            candidate.source_frame_id != frame.frame_id
                                            or candidate.building_id != frame.building_id
                                            or candidate.session_id != frame.session_id
                                        ):
                                            reason = "SOURCE_FRAME_REVALIDATION_FAILED"
                                            targets_valid = False
                                            break
                                        if self._passes(candidate, policy, target):
                                            passing.append(candidate)
                                    if targets_valid:
                                        if len(passing) == 0:
                                            reason = "NO_CANDIDATE_PASSED"
                                        elif len(passing) == 1:
                                            status = RelocalizationStatus.ACCEPTED
                                            reason = "ACCEPTED_UNIQUE_CANDIDATE"
                                            accepted = passing[0]
                                        else:
                                            status = RelocalizationStatus.AMBIGUOUS
                                            reason = "MULTIPLE_CANDIDATES_PASSED"
            candidate_ids = tuple(row["candidate_id"] for row in self._connection.execute("SELECT candidate_id FROM candidates WHERE attempt_id=? ORDER BY candidate_id", (attempt_id,)))
            decision_payload = {
                "attempt_id": attempt_id,
                "building_id": attempt["building_id"],
                "session_id": attempt["session_id"],
                "status": status.value,
                "reason_code": reason,
                "accepted_candidate_id": None if accepted is None else accepted.candidate_id,
                "accepted_target_submap_id": None if accepted is None else accepted.target_submap_id,
                "accepted_target_generation_id": None if accepted is None else accepted.target_generation_id,
                "source_frame_id": None if accepted is None else accepted.source_frame_id,
                "transform_target_from_source_frame": None if accepted is None else list(accepted.transform_target_from_source_frame),
                "policy_sha256": attempt["policy_sha256"],
                "candidate_ids": list(candidate_ids),
            }
            decision_id = _record_hash(decision_payload)
            decision_payload["decision_id"] = decision_id
            canonical = _canonical(decision_payload)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            self._connection.execute("UPDATE attempts SET state='SEALED' WHERE attempt_id=?", (attempt_id,))
            self._connection.execute(
                "INSERT INTO decisions VALUES(?,?,?,?,?,?)",
                (decision_id, attempt_id, attempt["building_id"], attempt["session_id"], canonical, digest),
            )
            if status is RelocalizationStatus.ACCEPTED:
                self._connection.execute(
                    "INSERT INTO authoritative_acceptances VALUES(?,?,?)",
                    (attempt["building_id"], attempt["session_id"], decision_id),
                )
            return self._decode_decision_json(canonical)

    def _decode_decision_json(self, canonical: str) -> RelocalizationDecision:
        payload = json.loads(canonical)
        decision_id = payload.get("decision_id")
        without_id = dict(payload)
        without_id.pop("decision_id", None)
        if decision_id != _record_hash(without_id):
            raise ValueError("decision record is corrupt")
        return self._decision_from_payload(payload)

    def _decision_from_row(self, row: sqlite3.Row) -> RelocalizationDecision:
        digest = hashlib.sha256(row["canonical_json"].encode("utf-8")).hexdigest()
        if digest != row["record_sha256"]:
            raise ValueError("decision record is corrupt")
        decision = self._decode_decision_json(row["canonical_json"])
        indexed = {
            "decision_id": row["decision_id"],
            "attempt_id": row["attempt_id"],
            "building_id": row["building_id"],
            "session_id": row["session_id"],
        }
        if any(getattr(decision, name) != value for name, value in indexed.items()):
            raise ValueError("decision indexed identity is corrupt")
        return decision

    def decision(self, attempt_id: str) -> RelocalizationDecision:
        with self._lock:
            row = self._connection.execute("SELECT * FROM decisions WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            return self._decision_from_row(row)

    def authoritative_decision(self, building_id: str, session_id: str) -> Optional[RelocalizationDecision]:
        building_id = _identifier(building_id, "building_id")
        session_id = _identifier(session_id, "session_id")
        with self._lock:
            row = self._connection.execute(
                """SELECT
                       a.building_id AS authority_building_id,
                       a.session_id AS authority_session_id,
                       a.decision_id AS authority_decision_id,
                       d.*
                   FROM authoritative_acceptances AS a
                   JOIN decisions AS d ON d.decision_id=a.decision_id
                   WHERE a.building_id=? AND a.session_id=?""",
                (building_id, session_id),
            ).fetchone()
            if row is None:
                return None
            try:
                decision = self._decision_from_row(row)
                if (
                    row["authority_building_id"] != building_id
                    or row["authority_session_id"] != session_id
                    or row["authority_decision_id"] != decision.decision_id
                    or decision.building_id != building_id
                    or decision.session_id != session_id
                    or decision.status is not RelocalizationStatus.ACCEPTED
                ):
                    raise ValueError("acceptance mapping identity or status mismatch")
                if any(
                    value is None
                    for value in (
                        decision.accepted_candidate_id,
                        decision.accepted_target_submap_id,
                        decision.accepted_target_generation_id,
                        decision.source_frame_id,
                        decision.transform_target_from_source_frame,
                    )
                ):
                    raise ValueError("accepted decision is incomplete")
                attempt = self._connection.execute(
                    "SELECT * FROM attempts WHERE attempt_id=?", (decision.attempt_id,)
                ).fetchone()
                if attempt is None or attempt["state"] != "SEALED":
                    raise ValueError("accepted attempt is missing or not sealed")
                content = json.loads(attempt["content_json"])
                if hashlib.sha256(attempt["content_json"].encode("utf-8")).hexdigest() != attempt["content_sha256"]:
                    raise ValueError("accepted attempt hash mismatch")
                indexed_attempt = {
                    "attempt_id": attempt["attempt_id"],
                    "building_id": attempt["building_id"],
                    "session_id": attempt["session_id"],
                    "source_frame_id": attempt["source_frame_id"],
                    "policy_sha256": attempt["policy_sha256"],
                }
                if content != indexed_attempt:
                    raise ValueError("accepted attempt indexed identity mismatch")
                if (
                    attempt["building_id"] != decision.building_id
                    or attempt["session_id"] != decision.session_id
                    or attempt["source_frame_id"] != decision.source_frame_id
                    or attempt["policy_sha256"] != decision.policy_sha256
                ):
                    raise ValueError("accepted attempt does not match decision")
                frame = self.session_frame(decision.source_frame_id)
                if frame.building_id != decision.building_id or frame.session_id != decision.session_id:
                    raise ValueError("accepted source frame provenance mismatch")
                self.source_generation(decision.source_frame_id)
                policy = self._load_policy(decision.policy_sha256)
                candidates = self._candidate_rows(decision.attempt_id)
                if tuple(candidate.candidate_id for candidate in candidates) != decision.candidate_ids:
                    raise ValueError("accepted candidate set mismatch")
                accepted = next(
                    (candidate for candidate in candidates if candidate.candidate_id == decision.accepted_candidate_id),
                    None,
                )
                if accepted is None:
                    raise ValueError("accepted candidate is missing")
                if (
                    accepted.building_id != decision.building_id
                    or accepted.session_id != decision.session_id
                    or accepted.source_frame_id != decision.source_frame_id
                    or accepted.target_submap_id != decision.accepted_target_submap_id
                    or accepted.target_generation_id != decision.accepted_target_generation_id
                    or accepted.transform_target_from_source_frame != decision.transform_target_from_source_frame
                ):
                    raise ValueError("accepted candidate does not match decision")
                target = self.project.sealed_generation(
                    decision.accepted_target_submap_id, decision.accepted_target_generation_id
                )
                if not self._passes(accepted, policy, target):
                    raise ValueError("accepted candidate no longer passes its persisted policy")
                return decision
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("authoritative relocalization evidence is invalid") from exc
