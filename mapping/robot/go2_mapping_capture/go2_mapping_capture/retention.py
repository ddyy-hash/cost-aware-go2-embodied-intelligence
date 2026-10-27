"""Crash-safe storage and retention for robot-side mapping segments.

This module deliberately has no ROS dependency.  A ROS recorder may fail or be
restarted at any point; only fully written MCAP payloads with a checksum
manifest become eligible for transfer or retention decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


class RetentionError(RuntimeError):
    """Base error for capture storage handling."""


class CapacityError(RetentionError):
    """No acknowledged, verified segment can be removed to make room."""


class IntegrityError(RetentionError):
    """A finalized segment no longer matches its checksum manifest."""


@dataclass(frozen=True)
class RetentionPolicy:
    raw_limit_bytes: int
    spool_limit_bytes: int
    recovery_headroom_bytes: int
    max_segment_bytes: int

    def __post_init__(self) -> None:
        if min(
            self.raw_limit_bytes,
            self.spool_limit_bytes,
            self.recovery_headroom_bytes,
            self.max_segment_bytes,
        ) < 0:
            raise ValueError("retention limits must be non-negative")
        if self.max_segment_bytes == 0:
            raise ValueError("max_segment_bytes must be positive")


@dataclass(frozen=True)
class RetentionResult:
    removed_segment_ids: Tuple[str, ...]
    raw_usage_bytes: int
    spool_usage_bytes: int


class SegmentStore:
    """Stores finalized MCAP files and removes only acknowledged valid data."""

    _segment_id_pattern = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, root_dir: Path, policy: RetentionPolicy) -> None:
        self.root_dir = Path(root_dir)
        self.policy = policy
        self.raw_dir = self.root_dir / "raw"
        self.spool_dir = self.root_dir / "spool"
        self.staging_dir = self.root_dir / "staging"
        self.recovery_dir = self.root_dir / "recovery"
        self.quarantine_dir = self.root_dir / "quarantine"

    def initialize(self) -> None:
        for directory in (
            self.raw_dir,
            self.spool_dir,
            self.staging_dir,
            self.recovery_dir,
            self.quarantine_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._require_recovery_headroom()

    def payload_path(self, segment_id: str, tier: str = "raw") -> Path:
        return self._tier_dir(tier) / f"{self._validated_id(segment_id)}.mcap"

    def manifest_path(self, segment_id: str, tier: str = "raw") -> Path:
        return self._tier_dir(tier) / f"{self._validated_id(segment_id)}.manifest.json"

    def commit_bytes(
        self,
        segment_id: str,
        payload: bytes,
        *,
        acknowledged: bool,
        tier: str = "raw",
    ) -> str:
        segment_id = self._validated_id(segment_id)
        if len(payload) > self.policy.max_segment_bytes:
            raise CapacityError(
                f"segment {segment_id} is {len(payload)} bytes, above "
                f"max_segment_bytes={self.policy.max_segment_bytes}"
            )
        staged = self.staging_dir / f"{segment_id}.mcap.partial"
        self._write_bytes_atomic(staged, payload)
        return self.commit_file(segment_id, staged, acknowledged=acknowledged, tier=tier)

    def commit_file(
        self,
        segment_id: str,
        staged_payload: Path,
        *,
        acknowledged: bool,
        tier: str = "raw",
    ) -> str:
        segment_id = self._validated_id(segment_id)
        staged_payload = Path(staged_payload)
        if not staged_payload.is_file():
            raise FileNotFoundError(staged_payload)
        size_bytes = staged_payload.stat().st_size
        if size_bytes > self.policy.max_segment_bytes:
            raise CapacityError(
                f"segment {segment_id} is {size_bytes} bytes, above "
                f"max_segment_bytes={self.policy.max_segment_bytes}"
            )

        payload_path = self.payload_path(segment_id, tier)
        manifest_path = self.manifest_path(segment_id, tier)
        if payload_path.exists() or manifest_path.exists():
            raise FileExistsError(f"segment {segment_id} already exists in {tier}")

        digest = self._sha256(staged_payload)
        os.replace(staged_payload, payload_path)
        manifest = {
            "schema_version": 1,
            "segment_id": segment_id,
            "tier": tier,
            "payload": payload_path.name,
            "size_bytes": size_bytes,
            "sha256": digest,
            "acknowledged": bool(acknowledged),
            "created_unix_ns": time.time_ns(),
        }
        self._write_json_atomic(manifest_path, manifest)
        return segment_id

    def set_acknowledged(self, segment_id: str, acknowledged: bool, tier: str = "raw") -> None:
        path = self.manifest_path(segment_id, tier)
        manifest = self._load_manifest(path)
        manifest["acknowledged"] = bool(acknowledged)
        manifest["acknowledged_unix_ns"] = time.time_ns() if acknowledged else None
        self._write_json_atomic(path, manifest)

    def recover_interrupted_writes(self) -> Tuple[str, ...]:
        """Keep incomplete files for diagnosis but never publish them as segments."""
        recovered = []
        for partial in sorted(self.staging_dir.iterdir()):
            if not partial.is_file():
                continue
            target = self.recovery_dir / partial.name
            if target.exists():
                target = self.recovery_dir / f"{time.time_ns()}-{partial.name}"
            os.replace(partial, target)
            recovered.append(partial.name)
        for tier in ("raw", "spool"):
            directory = self._tier_dir(tier)
            for payload in directory.glob("*.mcap"):
                segment_id = payload.stem
                if not self.manifest_path(segment_id, tier).exists():
                    self._quarantine(segment_id, tier, "orphan-payload")
        return tuple(recovered)

    def enforce_retention(self) -> RetentionResult:
        """Enforce each quota, deleting only oldest acknowledged valid entries."""
        self._require_recovery_headroom()
        removed = []
        for tier, limit in (("raw", self.policy.raw_limit_bytes), ("spool", self.policy.spool_limit_bytes)):
            while self._usage_bytes(tier) > limit:
                candidates = sorted(
                    (entry for entry in self._entries(tier) if entry[1].get("acknowledged")),
                    key=lambda entry: (entry[1].get("created_unix_ns", 0), entry[0]),
                )
                if not candidates:
                    raise CapacityError(
                        f"{tier} quota is exhausted; no acknowledged segment may be deleted "
                        "while unacknowledged segments are retained"
                    )
                segment_id, manifest = candidates[0]
                try:
                    self._verify(segment_id, manifest, tier)
                except IntegrityError:
                    self._quarantine(segment_id, tier, "checksum-mismatch")
                    raise
                self.payload_path(segment_id, tier).unlink()
                self.manifest_path(segment_id, tier).unlink()
                removed.append(segment_id)
        return RetentionResult(
            removed_segment_ids=tuple(removed),
            raw_usage_bytes=self._usage_bytes("raw"),
            spool_usage_bytes=self._usage_bytes("spool"),
        )

    def _entries(self, tier: str) -> Iterable[Tuple[str, Dict[str, object]]]:
        for manifest_path in sorted(self._tier_dir(tier).glob("*.manifest.json")):
            manifest = self._load_manifest(manifest_path)
            segment_id = self._validated_id(str(manifest["segment_id"]))
            if manifest.get("tier") != tier:
                raise IntegrityError(f"manifest tier mismatch for {segment_id}")
            yield segment_id, manifest

    def _verify(self, segment_id: str, manifest: Dict[str, object], tier: str) -> None:
        payload = self.payload_path(segment_id, tier)
        if not payload.is_file():
            raise IntegrityError(f"missing payload for {segment_id}")
        expected_size = manifest.get("size_bytes")
        expected_hash = manifest.get("sha256")
        if payload.stat().st_size != expected_size or self._sha256(payload) != expected_hash:
            raise IntegrityError(f"checksum mismatch for {segment_id}")

    def _quarantine(self, segment_id: str, tier: str, reason: str) -> None:
        payload = self.payload_path(segment_id, tier)
        manifest = self.manifest_path(segment_id, tier)
        for source in (payload, manifest):
            if source.exists():
                suffix = source.suffix if source.suffix != ".json" else ".manifest.json"
                target = self.quarantine_dir / f"{segment_id}{suffix}"
                if target.exists():
                    target = self.quarantine_dir / f"{segment_id}-{time.time_ns()}{suffix}"
                os.replace(source, target)
        self._write_json_atomic(
            self.quarantine_dir / f"{segment_id}.reason.json",
            {"segment_id": segment_id, "tier": tier, "reason": reason, "quarantined_unix_ns": time.time_ns()},
        )

    def _usage_bytes(self, tier: str) -> int:
        return sum(path.stat().st_size for path in self._tier_dir(tier).glob("*.mcap") if path.is_file())

    def _require_recovery_headroom(self) -> None:
        if self.policy.recovery_headroom_bytes == 0:
            return
        usage = shutil.disk_usage(self.root_dir)
        if usage.free < self.policy.recovery_headroom_bytes:
            raise CapacityError(
                f"filesystem free space {usage.free} is below recovery_headroom_bytes="
                f"{self.policy.recovery_headroom_bytes}"
            )

    def _tier_dir(self, tier: str) -> Path:
        if tier == "raw":
            return self.raw_dir
        if tier == "spool":
            return self.spool_dir
        raise ValueError(f"unknown storage tier: {tier}")

    def _validated_id(self, segment_id: str) -> str:
        if not self._segment_id_pattern.fullmatch(segment_id):
            raise ValueError(f"invalid segment id: {segment_id!r}")
        return segment_id

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        with temporary.open("wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _write_json_atomic(path: Path, value: Dict[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(value, target, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as source:
            loaded = json.load(source)
        if not isinstance(loaded, dict):
            raise IntegrityError(f"invalid manifest: {path}")
        return loaded
