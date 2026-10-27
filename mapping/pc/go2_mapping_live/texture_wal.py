"""Independent content-addressed WAL for RGB texture keyframes."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Optional

from .live_protocol import TextureKeyframeMessage, validate_texture_keyframe
from .live_wal import WalAdmission, _atomic_write, _fsync_directory


TextureStreamKey = tuple[str, str, str, str, int]


@dataclass(frozen=True)
class TextureWalRecord:
    frame: TextureKeyframeMessage
    record_sha256: str
    path: Path


def _stream_id(stream: TextureStreamKey) -> str:
    encoded = json.dumps(stream, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TextureKeyframeWal:
    """Preserve ordered texture evidence independently from LiDAR admission."""

    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.records_root = self.root / "records"
        self.blobs_root = self.root / "blobs"
        self.state_path = self.root / "state.json"
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()
        self.records_root.mkdir(parents=True, exist_ok=True)
        self.blobs_root.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1 or not isinstance(payload.get("streams"), dict):
                raise ValueError("invalid texture WAL state")
            return payload
        except FileNotFoundError:
            return {"schema_version": 1, "streams": {}}

    def _save_state(self) -> None:
        body = json.dumps(self._state, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        _atomic_write(self.state_path, body)

    @staticmethod
    def _identity_dict(frame: TextureKeyframeMessage) -> dict[str, Any]:
        identity = frame.identity
        return {
            "building_id": identity.building_id,
            "session_id": identity.session_id,
            "map_id": identity.map_id,
            "producer_instance_id": identity.producer_instance_id,
            "producer_epoch": identity.producer_epoch,
            "sequence": identity.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "calibration_id": frame.calibration_id,
            "content_sha256": frame.content_sha256,
        }

    @classmethod
    def _same_identity(cls, frame: TextureKeyframeMessage, identity: dict[str, Any]) -> bool:
        return cls._identity_dict(frame) == identity

    def _stream_state(self, stream: TextureStreamKey, *, create: bool = False) -> Optional[dict[str, Any]]:
        key = _stream_id(stream)
        state = self._state["streams"].get(key)
        if state is None and create:
            state = {
                "stream": list(stream),
                "integrated_sequence": None,
                "checkpoint_id": None,
                "journal": [],
            }
            self._state["streams"][key] = state
        return state

    def _blob_path(self, digest: str, suffix: str) -> Path:
        return self.blobs_root / digest[:2] / f"{digest}{suffix}"

    def _record_path(self, frame: TextureKeyframeMessage, digest: str) -> Path:
        stream_id = _stream_id(frame.identity.stream_key())
        return self.records_root / stream_id[:2] / (
            f"{stream_id}-{frame.identity.sequence:020d}-{digest}.json"
        )

    @staticmethod
    def _externalized_wrapper(frame: TextureKeyframeMessage, digest: str) -> tuple[dict[str, Any], bytes]:
        payload = json.loads(frame.canonical_json)
        color = dict(payload["color"])
        color.pop("jpeg_b64")
        payload["color"] = color
        wrapper: dict[str, Any] = {
            "schema_version": 1,
            "record_sha256": digest,
            "color_blob_sha256": frame.content_sha256,
            "payload": payload,
        }
        if frame.depth_compressed_bytes is not None:
            depth = dict(payload["depth"])
            depth.pop("data_b64")
            payload["depth"] = depth
            wrapper["depth_blob_sha256"] = depth["compressed_bytes_sha256"]
        body = json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return wrapper, body

    def _load_record(self, path: Path) -> Optional[TextureWalRecord]:
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            if wrapper.get("schema_version") != 1:
                return None
            payload = dict(wrapper["payload"])
            color_digest = str(wrapper["color_blob_sha256"])
            color_bytes = self._blob_path(color_digest, ".jpg").read_bytes()
            if hashlib.sha256(color_bytes).hexdigest() != color_digest:
                return None
            color = dict(payload["color"])
            color["jpeg_b64"] = base64.b64encode(color_bytes).decode("ascii")
            payload["color"] = color
            depth_digest = wrapper.get("depth_blob_sha256")
            if depth_digest is not None:
                depth_bytes = self._blob_path(str(depth_digest), ".depth.zlib").read_bytes()
                if hashlib.sha256(depth_bytes).hexdigest() != depth_digest:
                    return None
                depth = dict(payload["depth"])
                depth["data_b64"] = base64.b64encode(depth_bytes).decode("ascii")
                payload["depth"] = depth
            frame = validate_texture_keyframe(payload)
            digest = str(wrapper["record_sha256"])
            if hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest() != digest:
                return None
            return TextureWalRecord(frame, digest, path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return None

    def _all_records(self) -> list[TextureWalRecord]:
        records: list[TextureWalRecord] = []
        for path in self.records_root.rglob("*.json"):
            record = self._load_record(path)
            if record is not None:
                records.append(record)
        return records

    def _stream_records(self, stream: TextureStreamKey) -> list[TextureWalRecord]:
        records = [record for record in self._all_records() if record.frame.identity.stream_key() == stream]
        return sorted(records, key=lambda record: record.frame.identity.sequence)

    def _used_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for root in (self.records_root, self.blobs_root)
            for path in root.rglob("*")
            if path.is_file()
        )

    def records(self) -> list[TextureWalRecord]:
        """Return every retained, integrity-checked record in stable stream order."""
        with self._lock:
            return sorted(
                self._all_records(),
                key=lambda record: (*record.frame.identity.stream_key(), record.frame.identity.sequence),
            )

    def admit(self, frame: TextureKeyframeMessage) -> WalAdmission:
        with self._lock:
            stream = frame.identity.stream_key()
            state = self._stream_state(stream)
            records = self._stream_records(stream)
            queued_by_sequence = {record.frame.identity.sequence: record for record in records}
            integrated_sequence = None if state is None else state.get("integrated_sequence")
            queued_sequences = [
                sequence
                for sequence in queued_by_sequence
                if integrated_sequence is None or sequence > int(integrated_sequence)
            ]
            if queued_sequences:
                expected = max(queued_sequences) + 1
            elif integrated_sequence is not None:
                expected = int(integrated_sequence) + 1
            else:
                expected = frame.identity.sequence

            sequence = frame.identity.sequence
            digest = hashlib.sha256(frame.canonical_json.encode("utf-8")).hexdigest()
            if sequence < expected:
                journal = [] if state is None else state.get("journal", [])
                if integrated_sequence is not None and sequence <= int(integrated_sequence):
                    if any(self._same_identity(frame, identity) for identity in journal):
                        return WalAdmission("DUPLICATE_INTEGRATED", None, expected)
                    return WalAdmission("REJECTED", None, expected, "IDENTITY_CONFLICT")
                queued = queued_by_sequence.get(sequence)
                if queued is not None and queued.record_sha256 == digest:
                    return WalAdmission("DUPLICATE_QUEUED", queued.record_sha256, expected)
                return WalAdmission("REJECTED", None, expected, "IDENTITY_CONFLICT")
            if sequence > expected:
                return WalAdmission("REJECTED", None, expected, "SEQUENCE_GAP")

            _, body = self._externalized_wrapper(frame, digest)
            blobs: list[tuple[Path, bytes]] = [
                (self._blob_path(frame.content_sha256, ".jpg"), frame.color_jpeg),
            ]
            if frame.depth_compressed_bytes is not None:
                depth_hash = hashlib.sha256(frame.depth_compressed_bytes).hexdigest()
                blobs.append((self._blob_path(depth_hash, ".depth.zlib"), frame.depth_compressed_bytes))
            additional_bytes = len(body) + sum(len(blob) for path, blob in blobs if not path.exists())
            if self._used_bytes() + additional_bytes > self.max_bytes:
                return WalAdmission("REJECTED", None, expected, "WAL_CAPACITY_EXCEEDED")
            for path, blob in blobs:
                if not path.exists():
                    _atomic_write(path, blob)
            _atomic_write(self._record_path(frame, digest), body)
            self._stream_state(stream, create=True)
            return WalAdmission("QUEUED_NEW", digest, sequence + 1)

    def pending(self) -> list[TextureWalRecord]:
        with self._lock:
            pending: list[TextureWalRecord] = []
            for record in self._all_records():
                state = self._stream_state(record.frame.identity.stream_key())
                cursor = None if state is None else state.get("integrated_sequence")
                if cursor is None or record.frame.identity.sequence > int(cursor):
                    pending.append(record)
            return sorted(
                pending,
                key=lambda record: (*record.frame.identity.stream_key(), record.frame.identity.sequence),
            )

    def highest_sequence(self, stream: TextureStreamKey) -> Optional[int]:
        with self._lock:
            state = self._stream_state(stream)
            values = [record.frame.identity.sequence for record in self._stream_records(stream)]
            if state is not None and state.get("integrated_sequence") is not None:
                values.append(int(state["integrated_sequence"]))
            return None if not values else max(values)

    def commit_integrated(self, frame: TextureKeyframeMessage, *, checkpoint_id: str) -> None:
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required")
        with self._lock:
            state = self._stream_state(frame.identity.stream_key(), create=True)
            current = state.get("integrated_sequence")
            pending_sequences = [
                record.frame.identity.sequence
                for record in self._stream_records(frame.identity.stream_key())
                if current is None or record.frame.identity.sequence > int(current)
            ]
            expected = min(pending_sequences) if current is None else int(current) + 1
            if frame.identity.sequence != expected:
                raise ValueError(f"integrated cursor requires sequence {expected}")
            state["integrated_sequence"] = frame.identity.sequence
            state["checkpoint_id"] = str(checkpoint_id)
            journal = list(state.get("journal", []))
            journal.append(self._identity_dict(frame))
            state["journal"] = journal[-4096:]
            self._save_state()

    def reclaim_integrated(self, stream: TextureStreamKey) -> None:
        """Reclaim record metadata only; immutable content blobs remain retained."""
        with self._lock:
            state = self._stream_state(stream)
            if state is None or state.get("integrated_sequence") is None:
                return
            cursor = int(state["integrated_sequence"])
            integrated_records = [
                record
                for record in self._stream_records(stream)
                if record.frame.identity.sequence <= cursor
            ]
            parent_directories = sorted(
                {record.path.parent for record in integrated_records},
                key=lambda path: path.as_posix(),
            )
            for record in integrated_records:
                record.path.unlink(missing_ok=True)
            for directory in parent_directories:
                _fsync_directory(directory)
