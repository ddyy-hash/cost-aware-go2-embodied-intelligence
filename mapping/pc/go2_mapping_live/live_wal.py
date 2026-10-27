"""Bounded durable admission log for detailed live keyframes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Optional

from .live_protocol import (
    FrameIdentity,
    LiveKeyframeMessage,
    StreamKey,
    SurfaceSourceStatus,
    validate_live_keyframe,
    validate_surface_source_status,
)


@dataclass(frozen=True)
class WalAdmission:
    disposition: str
    record_sha256: Optional[str]
    next_sequence: int
    error_code: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.disposition != "REJECTED"


@dataclass(frozen=True)
class WalRecord:
    frame: LiveKeyframeMessage
    record_sha256: str
    path: Path


def _stream_id(stream: StreamKey) -> str:
    encoded = json.dumps(stream.as_tuple(), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class LiveKeyframeWal:
    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.records_root = self.root / "records"
        self.state_path = self.root / "state.json"
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()
        self.records_root.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._records: dict[Path, WalRecord] = {}
        self._used_bytes_cache = 0
        self._load_record_index()

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported WAL state schema")
            if not isinstance(payload.get("streams"), dict) or not isinstance(payload.get("statuses"), dict):
                raise ValueError("invalid WAL state")
            return payload
        except FileNotFoundError:
            return {"schema_version": 1, "streams": {}, "statuses": {}}

    def _save_state(self) -> None:
        body = json.dumps(self._state, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        _atomic_write(self.state_path, body)

    @staticmethod
    def _identity_dict(frame: LiveKeyframeMessage) -> dict[str, Any]:
        identity = frame.identity
        return {
            "building_id": identity.building_id,
            "session_id": identity.session_id,
            "map_id": identity.map_id,
            "producer_instance_id": identity.producer_instance_id,
            "sequence": identity.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "calibration_id": frame.calibration_id,
            "point_bytes_sha256": frame.point_bytes_sha256,
            "sensor_frame_transform_id": frame.sensor_frame_transform_id,
        }

    @staticmethod
    def _same_identity(frame: LiveKeyframeMessage, identity: dict[str, Any]) -> bool:
        expected = LiveKeyframeWal._identity_dict(frame)
        actual = dict(identity)
        actual.setdefault("sensor_frame_transform_id", "IDENTITY_SENSOR_FRAME_v1")
        return expected == actual

    def _stream_state(self, stream: StreamKey, *, create: bool = False) -> Optional[dict[str, Any]]:
        key = _stream_id(stream)
        state = self._state["streams"].get(key)
        if state is None and create:
            state = {
                "stream": list(stream.as_tuple()),
                "integrated_sequence": None,
                "checkpoint_id": None,
                "journal": [],
            }
            self._state["streams"][key] = state
        return state

    def _record_path(self, frame: LiveKeyframeMessage, digest: str) -> Path:
        return self.records_root / f"{_stream_id(frame.identity.stream_key())}-{frame.identity.sequence:020d}-{digest}.json"

    def _load_record_index(self) -> None:
        for path in sorted(self.records_root.glob("*.json")):
            try:
                self._used_bytes_cache += path.stat().st_size
            except OSError:
                continue
            try:
                wrapper = json.loads(path.read_text(encoding="utf-8"))
                payload = wrapper["payload"]
                digest = str(wrapper["record_sha256"])
                frame = validate_live_keyframe(payload)
                if hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest() != digest:
                    continue
                self._records[path] = WalRecord(frame, digest, path)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def _all_records(self) -> list[WalRecord]:
        return list(self._records.values())

    def _stream_records(self, stream: StreamKey) -> list[WalRecord]:
        records = [record for record in self._all_records() if record.frame.identity.stream_key() == stream]
        return sorted(records, key=lambda record: record.frame.identity.sequence)

    def _used_bytes(self) -> int:
        return self._used_bytes_cache

    def admit(self, frame: LiveKeyframeMessage) -> WalAdmission:
        with self._lock:
            stream = frame.identity.stream_key()
            state = self._stream_state(stream)
            records = self._stream_records(stream)
            queued_by_sequence = {record.frame.identity.sequence: record for record in records}
            integrated_sequence = None if state is None else state.get("integrated_sequence")
            queued_sequences = [sequence for sequence in queued_by_sequence if integrated_sequence is None or sequence > integrated_sequence]
            if queued_sequences:
                expected = max(queued_sequences) + 1
            elif integrated_sequence is not None:
                expected = int(integrated_sequence) + 1
            else:
                expected = frame.identity.sequence

            sequence = frame.identity.sequence
            if sequence < expected:
                journal = [] if state is None else state.get("journal", [])
                if integrated_sequence is not None and sequence <= int(integrated_sequence):
                    if any(self._same_identity(frame, identity) for identity in journal):
                        return WalAdmission("DUPLICATE_INTEGRATED", None, expected)
                    return WalAdmission("REJECTED", None, expected, "IDENTITY_CONFLICT")
                queued = queued_by_sequence.get(sequence)
                if queued is not None and queued.record_sha256 == hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest():
                    return WalAdmission("DUPLICATE_QUEUED", queued.record_sha256, expected)
                if any(self._same_identity(frame, identity) for identity in journal):
                    return WalAdmission("DUPLICATE_INTEGRATED", None, expected)
                return WalAdmission("REJECTED", None, expected, "IDENTITY_CONFLICT")
            if sequence > expected:
                return WalAdmission("REJECTED", None, expected, "SEQUENCE_GAP")

            digest = hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest()
            wrapper = {"schema_version": 1, "record_sha256": digest, "payload": frame.payload}
            body = json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            if self._used_bytes() + len(body) > self.max_bytes:
                return WalAdmission("REJECTED", None, expected, "WAL_CAPACITY_EXCEEDED")
            path = self._record_path(frame, digest)
            _atomic_write(path, body)
            self._records[path] = WalRecord(frame, digest, path)
            self._used_bytes_cache += len(body)
            self._stream_state(stream, create=True)
            return WalAdmission("QUEUED_NEW", digest, sequence + 1)

    def pending(self) -> list[WalRecord]:
        with self._lock:
            pending: list[WalRecord] = []
            for record in self._all_records():
                state = self._stream_state(record.frame.identity.stream_key())
                cursor = None if state is None else state.get("integrated_sequence")
                if cursor is None or record.frame.identity.sequence > int(cursor):
                    pending.append(record)
            return sorted(pending, key=lambda record: (*record.frame.identity.stream_key().as_tuple(), record.frame.identity.sequence))

    def highest_sequence(self, stream: StreamKey) -> Optional[int]:
        with self._lock:
            state = self._stream_state(stream)
            values = [record.frame.identity.sequence for record in self._stream_records(stream)]
            if state is not None and state.get("integrated_sequence") is not None:
                values.append(int(state["integrated_sequence"]))
            return None if not values else max(values)

    def commit_integrated(self, frame: LiveKeyframeMessage, *, checkpoint_id: str) -> None:
        self.commit_integrated_batch((frame,), checkpoint_id=checkpoint_id)

    def commit_integrated_batch(
        self,
        frames: tuple[LiveKeyframeMessage, ...] | list[LiveKeyframeMessage],
        *,
        checkpoint_id: str,
    ) -> None:
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required")
        frames = tuple(frames)
        if not frames:
            return
        with self._lock:
            cursors: dict[StreamKey, Optional[int]] = {}
            journals: dict[StreamKey, list[dict[str, Any]]] = {}
            for frame in frames:
                stream = frame.identity.stream_key()
                state = self._stream_state(stream, create=True)
                current = cursors.get(stream, state.get("integrated_sequence"))
                pending_sequences = [
                    record.frame.identity.sequence
                    for record in self._stream_records(stream)
                    if current is None or record.frame.identity.sequence > int(current)
                ]
                expected = min(pending_sequences) if current is None else int(current) + 1
                if frame.identity.sequence != expected:
                    raise ValueError(f"integrated cursor requires sequence {expected}")
                cursors[stream] = frame.identity.sequence
                journals.setdefault(stream, []).append(self._identity_dict(frame))
            for stream, cursor in cursors.items():
                state = self._stream_state(stream, create=True)
                state["integrated_sequence"] = cursor
                state["checkpoint_id"] = str(checkpoint_id)
                journal = list(state.get("journal", []))
                journal.extend(journals[stream])
                state["journal"] = journal[-4096:]
            self._save_state()

    def reclaim_integrated(self, stream: StreamKey) -> None:
        with self._lock:
            state = self._stream_state(stream)
            if state is None or state.get("integrated_sequence") is None:
                return
            cursor = int(state["integrated_sequence"])
            for record in self._stream_records(stream):
                if record.frame.identity.sequence <= cursor:
                    try:
                        size = record.path.stat().st_size
                    except OSError:
                        size = 0
                    record.path.unlink(missing_ok=True)
                    self._records.pop(record.path, None)
                    self._used_bytes_cache = max(0, self._used_bytes_cache - size)
            _fsync_directory(self.records_root)

    def accept_status(self, status: SurfaceSourceStatus) -> bool:
        with self._lock:
            key = _stream_id(status.identity.stream_key())
            existing = self._state["statuses"].get(key)
            if existing is not None and int(existing.get("status_sequence", -1)) >= status.status_sequence:
                return False
            self._state["statuses"][key] = dict(status.payload)
            self._save_state()
            return True

    def latest_status(self, stream: StreamKey) -> Optional[SurfaceSourceStatus]:
        with self._lock:
            payload = self._state["statuses"].get(_stream_id(stream))
            return None if payload is None else validate_surface_source_status(payload)

    def latest_status_any(self) -> Optional[SurfaceSourceStatus]:
        with self._lock:
            statuses = [validate_surface_source_status(payload) for payload in self._state["statuses"].values()]
            if not statuses:
                return None
            return max(
                statuses,
                key=lambda status: (
                    status.timestamp_ns,
                    status.status_sequence,
                    status.identity.stream_key().as_tuple(),
                ),
            )
