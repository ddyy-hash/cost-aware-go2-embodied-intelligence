from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Envelope:
    map_id: str
    sequence_id: int
    payload: bytes
    checksum_sha256: str
    calibration_id: str

    @classmethod
    def create(cls, map_id: str, sequence_id: int, payload: bytes, calibration_id: str) -> "Envelope":
        return cls(map_id, sequence_id, payload, hashlib.sha256(payload).hexdigest(), calibration_id)

    def with_payload(self, payload: bytes) -> "Envelope":
        return replace(self, payload=payload)

    def verify(self) -> None:
        if hashlib.sha256(self.payload).hexdigest() != self.checksum_sha256:
            raise ValueError("checksum mismatch")


class IngestStore:
    """Durably accepts each map/sequence pair at most once."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def accept(self, envelope: Envelope) -> bool:
        envelope.verify()
        if not envelope.map_id or envelope.sequence_id < 0 or not envelope.calibration_id:
            raise ValueError("invalid keyframe envelope identity")
        directory = self.root / envelope.map_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{envelope.sequence_id:020d}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing["checksum_sha256"] != envelope.checksum_sha256:
                raise ValueError("sequence collision with a different checksum")
            return False
        temporary = target.with_suffix(".partial")
        temporary.write_text(json.dumps({
            "map_id": envelope.map_id,
            "sequence_id": envelope.sequence_id,
            "payload_b64": base64.b64encode(envelope.payload).decode("ascii"),
            "checksum_sha256": envelope.checksum_sha256,
            "calibration_id": envelope.calibration_id,
        }, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return True

    def count(self, map_id: str) -> int:
        directory = self.root / map_id
        return len(list(directory.glob("*.json"))) if directory.exists() else 0
