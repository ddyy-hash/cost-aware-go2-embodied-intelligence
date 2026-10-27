"""Durable PC authority for releasing reconstructed robot artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Mapping


_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactLifecycleStore:
    """Persist checksum and reconstruction gates shared by PC processes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.registry_root = self.root / ".artifact-lifecycle"

    def registry_path(self, session_id: str, artifact_id: str) -> Path:
        self._validate_component("session_id", session_id)
        self._validate_digest("artifact_id", artifact_id)
        return self.registry_root / session_id / f"{artifact_id}.json"

    def record_verified(
        self,
        session_id: str,
        artifact_id: str,
        relative_path: str,
        size_bytes: int,
    ) -> dict:
        self._validate_component("session_id", session_id)
        self._validate_digest("artifact_id", artifact_id)
        relative = self._validate_relative(relative_path)
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        existing = self.status(session_id, artifact_id)
        if (
            existing.get("state") == "PC_RECONSTRUCTION_INGESTED"
            and existing.get("release_authorized") is True
            and existing.get("relative_path") == relative.as_posix()
            and existing.get("artifact_sha256") == artifact_id
            and existing.get("size_bytes") == size_bytes
        ):
            return existing
        state = {
            "schema_version": 1,
            "session_id": session_id,
            "artifact_id": artifact_id,
            "artifact_sha256": artifact_id,
            "relative_path": relative.as_posix(),
            "size_bytes": size_bytes,
            "state": "SHA256_VERIFIED",
            "release_authorized": False,
            "verified_at_ms": int(time.time() * 1000),
            "reconstructed_at_ms": None,
            "integrated_frames": 0,
            "integrated_points": 0,
        }
        self._write(self.registry_path(session_id, artifact_id), state)
        return dict(state)

    def mark_reconstructed(self, artifact_path: Path, *, result: Mapping[str, object]) -> dict:
        path = Path(artifact_path).resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path is outside lifecycle root") from exc
        if not path.is_file():
            raise ValueError("artifact is missing from PC verified storage")
        size_bytes = path.stat().st_size
        digest = self._sha256(path)
        candidates = list(self.registry_root.glob(f"*/{digest}.json"))
        matching = []
        for candidate in candidates:
            try:
                state = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if state.get("relative_path") == relative.as_posix():
                matching.append((candidate, state))
        if not matching:
            for candidate in self.registry_root.glob("*/*.json"):
                try:
                    state = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, TypeError, json.JSONDecodeError):
                    continue
                if state.get("relative_path") == relative.as_posix():
                    raise ValueError("artifact checksum or size no longer matches verified storage")
            raise LookupError("artifact has no verified lifecycle record")
        if len(matching) != 1:
            raise ValueError("verified lifecycle record is missing or ambiguous")
        registry_path, state = matching[0]
        if state.get("artifact_sha256") != digest or int(state.get("size_bytes", -1)) != size_bytes:
            raise ValueError("artifact checksum or size no longer matches verified storage")
        frames = int(result.get("integrated_frames", 0))
        points = int(result.get("integrated_points", 0))
        duplicate = bool(result.get("duplicate", False))
        if not duplicate and (frames <= 0 or points <= 0):
            raise ValueError("artifact produced no reconstructable LiDAR frames")
        state.update(
            {
                "state": "PC_RECONSTRUCTION_INGESTED",
                "release_authorized": True,
                "reconstructed_at_ms": int(time.time() * 1000),
                "integrated_frames": frames,
                "integrated_points": points,
                "duplicate_ingest": duplicate,
            }
        )
        self._write(registry_path, state)
        return dict(state)

    def status(self, session_id: str, artifact_id: str) -> dict:
        path = self.registry_path(session_id, artifact_id)
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema_version": 1,
                "session_id": session_id,
                "artifact_id": artifact_id,
                "state": "UNKNOWN",
                "release_authorized": False,
            }
        except (OSError, TypeError, json.JSONDecodeError):
            return {
                "schema_version": 1,
                "session_id": session_id,
                "artifact_id": artifact_id,
                "state": "INVALID",
                "release_authorized": False,
            }
        return state if isinstance(state, dict) else {
            "schema_version": 1,
            "session_id": session_id,
            "artifact_id": artifact_id,
            "state": "INVALID",
            "release_authorized": False,
        }

    @staticmethod
    def _validate_component(name: str, value: str) -> None:
        if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
            raise ValueError(f"{name} is unsafe")

    @staticmethod
    def _validate_digest(name: str, value: str) -> None:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @staticmethod
    def _validate_relative(value: str) -> PurePosixPath:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("relative_path is unsafe")
        return relative

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write(path: Path, state: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temporary.write_text(json.dumps(dict(state), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, path)
