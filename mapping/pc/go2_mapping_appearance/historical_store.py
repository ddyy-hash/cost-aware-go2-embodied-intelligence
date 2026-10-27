"""Atomic, content-addressed storage for historical appearance sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256

from .vertex_color import VertexColorSidecar


_REVISION_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


@dataclass(frozen=True)
class PublishedAppearance:
    appearance_revision_id: str
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_path: Path
    binary_path: Path


class HistoricalAppearanceStore:
    """Publishes complete sidecars and never mutates an existing revision."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        sidecar: VertexColorSidecar,
        *,
        expected_geometry_revision_id: str | None = None,
    ) -> PublishedAppearance:
        if not isinstance(sidecar, VertexColorSidecar):
            raise TypeError("sidecar must be a VertexColorSidecar")
        manifest = dict(sidecar.manifest)
        revision = self._revision(manifest.get("appearance_revision_id"))
        geometry = self._revision(manifest.get("measured_surface_geometry_revision_id"))
        if expected_geometry_revision_id is not None and geometry != self._revision(expected_geometry_revision_id):
            raise ValueError("appearance geometry revision does not match the measured surface geometry")
        binary = bytes(sidecar.binary)
        expected_content = manifest.get("vertex_color", {}).get("content_sha256")
        actual_content = hashlib.sha256(binary).hexdigest()
        if expected_content != actual_content:
            raise ValueError("appearance binary content verification failed")

        manifest.pop("manifest_sha256", None)
        manifest["content_sha256"] = actual_content
        manifest_sha256 = canonical_sha256(manifest)
        manifest["manifest_sha256"] = manifest_sha256
        target = self.root / revision
        if target.exists() or target.is_symlink():
            return self._load_existing(
                revision,
                expected_geometry_revision_id=expected_geometry_revision_id,
                expected_manifest=manifest,
                expected_binary=binary,
            )

        staging = Path(tempfile.mkdtemp(prefix=".appearance-", suffix=".partial", dir=self.root))
        try:
            manifest_path = staging / "manifest.json"
            binary_path = staging / "vertex-color.bin"
            self._write_fsync(binary_path, binary)
            self._write_fsync(
                manifest_path,
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
            )
            os.replace(staging, target)
            self._fsync_directory(self.root)
        except FileExistsError:
            self._remove_tree(staging)
            return self._load_existing(
                revision,
                expected_geometry_revision_id=expected_geometry_revision_id,
                expected_manifest=manifest,
                expected_binary=binary,
            )
        except Exception:
            self._remove_tree(staging)
            raise
        return PublishedAppearance(
            revision,
            manifest,
            manifest_sha256,
            target / "manifest.json",
            target / "vertex-color.bin",
        )

    def load(
        self,
        appearance_revision_id: str,
        *,
        expected_geometry_revision_id: str | None = None,
    ) -> PublishedAppearance:
        revision = self._revision(appearance_revision_id)
        target = self._safe_target(revision)
        manifest_path = target / "manifest.json"
        binary_path = target / "vertex-color.bin"
        if not manifest_path.is_file() or not binary_path.is_file():
            raise FileNotFoundError("published appearance revision is unavailable")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("appearance manifest is invalid") from exc
        if not isinstance(manifest, dict) or self._revision(manifest.get("appearance_revision_id")) != revision:
            raise ValueError("appearance manifest revision verification failed")
        geometry = self._revision(manifest.get("measured_surface_geometry_revision_id"))
        if expected_geometry_revision_id is not None and geometry != self._revision(expected_geometry_revision_id):
            raise ValueError("appearance geometry revision is stale")
        try:
            binary = binary_path.read_bytes()
        except OSError as exc:
            raise ValueError("appearance binary is unavailable") from exc
        if hashlib.sha256(binary).hexdigest() != manifest.get("content_sha256"):
            raise ValueError("appearance binary verification failed")
        recorded_manifest_sha = manifest.get("manifest_sha256")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_sha256", None)
        if recorded_manifest_sha != canonical_sha256(unsigned_manifest):
            raise ValueError("appearance manifest verification failed")
        return PublishedAppearance(revision, manifest, recorded_manifest_sha, manifest_path, binary_path)

    def _load_existing(
        self,
        revision: str,
        *,
        expected_geometry_revision_id: str | None,
        expected_manifest: dict[str, Any],
        expected_binary: bytes,
    ) -> PublishedAppearance:
        loaded = self.load(revision, expected_geometry_revision_id=expected_geometry_revision_id)
        if loaded.binary_path.read_bytes() != expected_binary:
            raise ValueError("existing appearance revision has conflicting binary content")
        if loaded.manifest != expected_manifest:
            raise ValueError("existing appearance revision has conflicting identity")
        return loaded

    def _safe_target(self, revision: str) -> Path:
        target = (self.root / revision).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("appearance path escapes the publication root") from exc
        return target

    def _revision(self, value: Any) -> str:
        if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
            raise ValueError("appearance revision must be a lowercase SHA-256")
        return value

    @staticmethod
    def _write_fsync(path: Path, payload: bytes) -> None:
        with path.open("wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                HistoricalAppearanceStore._remove_tree(child)
            else:
                child.unlink()
        path.rmdir()
