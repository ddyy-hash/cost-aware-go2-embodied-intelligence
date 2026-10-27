"""Poll atomically published high-fidelity artifacts into the dense mapper."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from time import monotonic as _monotonic
from typing import Any, Dict, Sequence, Set

import zstandard

from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore


class VerifiedArtifactIngestor:
    """Consumes only receiver-published ``.db3`` files, never chunk staging data."""

    def __init__(
        self,
        root: Path,
        mapper: Any,
        lifecycle_store: ArtifactLifecycleStore | None = None,
        *,
        ingest_batch_size: int = 1,
        lifecycle_batch_size: int = 4,
        derived_refresh_batch_size: int = 1,
        derived_refresh_interval_s: float = 30.0,
    ) -> None:
        if ingest_batch_size < 1 or lifecycle_batch_size < 1 or derived_refresh_batch_size < 1:
            raise ValueError("artifact ingest batch sizes must be positive")
        if derived_refresh_interval_s <= 0:
            raise ValueError("derived artifact refresh interval must be positive")
        self.root = Path(root)
        self.mapper = mapper
        self.lifecycle_store = lifecycle_store or ArtifactLifecycleStore(self.root)
        self._ingest_batch_size = ingest_batch_size
        self._lifecycle_batch_size = lifecycle_batch_size
        self._derived_refresh_batch_size = derived_refresh_batch_size
        self._derived_refresh_interval_s = derived_refresh_interval_s
        self._next_derived_refresh = _monotonic() + derived_refresh_interval_s
        self._lifecycle_cursor = 0
        self._derived_refresh_cursor = 0
        self._processed: Set[Path] = set()
        self._processed_results: Dict[Path, Dict[str, object]] = {}
        self._processed_digests: Dict[Path, str] = {}
        self._release_authorized_paths: Set[Path] = set()
        self._integrated_artifacts = 0
        self._integrated_frames = 0
        self._integrated_points = 0
        self._errors = 0
        self._last_error = None
        self._release_authorized_artifacts = 0

    def ingest_once(self) -> Dict[str, int]:
        ingested = 0
        if not self.root.is_dir():
            return {"ingested_artifacts": 0, "errors": 0}
        candidates = [
            path
            for path in sorted([*self.root.rglob("*.db3"), *self.root.rglob("*.db3.zstd")])
            if self._is_published_artifact(path)
        ]
        attempted_artifacts = 0
        lifecycle_attempted: Set[Path] = set()
        for path in candidates:
            if path in self._processed:
                continue
            if attempted_artifacts >= self._ingest_batch_size:
                break
            attempted_artifacts += 1
            try:
                artifact_sha256 = self._sha256(path)
                with self._sqlite_bag_path(path) as sqlite_bag:
                    result = self._ingest_mapper(
                        sqlite_bag,
                        artifact_sha256=artifact_sha256,
                        source_artifact_path=path,
                    )
            except Exception as exc:
                self._errors += 1
                self._last_error = str(exc)
                self._processed.add(path)
                continue
            self._processed.add(path)
            self._processed_digests[path] = artifact_sha256
            self._processed_results[path] = dict(result)
            lifecycle_attempted.add(path)
            try:
                lifecycle = self.lifecycle_store.mark_reconstructed(path, result=result)
                if lifecycle.get("release_authorized") is True:
                    self._release_authorized_paths.add(path)
                    self._release_authorized_artifacts += 1
            except LookupError:
                # Legacy verified artifacts remain reconstructable, but without
                # a lifecycle record they can never authorize robot deletion.
                pass
            except ValueError as exc:
                self._errors += 1
                self._last_error = str(exc)
                continue
            if not result.get("duplicate", False):
                ingested += 1
                self._integrated_artifacts += 1
                self._integrated_frames += int(result.get("integrated_frames", 0))
                self._integrated_points += int(result.get("integrated_points", 0))
        self._reconcile_lifecycle(candidates, exclude=lifecycle_attempted)
        self._refresh_derived_artifacts(candidates)
        return {"ingested_artifacts": ingested, "errors": self._errors}

    def _reconcile_lifecycle(self, candidates: Sequence[Path], *, exclude: Set[Path]) -> None:
        pending = [
            path
            for path in candidates
            if path in self._processed_results
            and path not in self._release_authorized_paths
            and path not in exclude
        ]
        selected, self._lifecycle_cursor = self._round_robin_batch(
            pending,
            self._lifecycle_cursor,
            self._lifecycle_batch_size,
        )
        for path in selected:
            try:
                lifecycle = self.lifecycle_store.mark_reconstructed(
                    path,
                    result=self._processed_results[path],
                )
                if lifecycle.get("release_authorized") is True:
                    self._release_authorized_paths.add(path)
                    self._release_authorized_artifacts += 1
            except LookupError:
                pass
            except ValueError as exc:
                self._errors += 1
                self._last_error = str(exc)

    def _refresh_derived_artifacts(self, candidates: Sequence[Path]) -> None:
        refresh = getattr(self.mapper, "refresh_derived_artifact", None)
        now = _monotonic()
        if not callable(refresh) or now < self._next_derived_refresh:
            return
        self._next_derived_refresh = now + self._derived_refresh_interval_s
        processed = [path for path in candidates if path in self._processed_results]
        selected, self._derived_refresh_cursor = self._round_robin_batch(
            processed,
            self._derived_refresh_cursor,
            self._derived_refresh_batch_size,
        )
        for path in selected:
            try:
                artifact_sha256 = self._processed_digests.get(path) or self._sha256(path)
                self._processed_digests[path] = artifact_sha256
                with self._sqlite_bag_path(path) as sqlite_bag:
                    refresh(
                        sqlite_bag,
                        artifact_sha256=artifact_sha256,
                        source_artifact_path=path,
                    )
            except Exception as exc:
                self._errors += 1
                self._last_error = str(exc)

    @staticmethod
    def _round_robin_batch(
        paths: Sequence[Path],
        cursor: int,
        batch_size: int,
    ) -> tuple[list[Path], int]:
        if not paths:
            return [], 0
        start = cursor % len(paths)
        count = min(batch_size, len(paths))
        selected = [paths[(start + offset) % len(paths)] for offset in range(count)]
        return selected, (start + count) % len(paths)

    def _ingest_mapper(
        self,
        sqlite_bag: Path,
        *,
        artifact_sha256: str,
        source_artifact_path: Path,
    ) -> Dict[str, object]:
        ingest = self.mapper.ingest_sqlite_bag
        try:
            parameters = inspect.signature(ingest).parameters.values()
        except (TypeError, ValueError):
            return ingest(sqlite_bag)
        supports_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        parameter_names = {parameter.name for parameter in parameters}
        if supports_keywords or {"artifact_sha256", "source_artifact_path"}.issubset(parameter_names):
            return ingest(
                sqlite_bag,
                artifact_sha256=artifact_sha256,
                source_artifact_path=source_artifact_path,
            )
        return ingest(sqlite_bag)

    def _is_published_artifact(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        if any(part.startswith(".") for part in relative.parts):
            return False
        return not self._is_deprecated_segment(path)

    def _is_deprecated_segment(self, path: Path) -> bool:
        """Fail closed when an ancestor explicitly retires a captured segment."""
        current = path.parent
        root = self.root.resolve()
        while True:
            marker = current / "DEPRECATED.json"
            if marker.is_file():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, TypeError, json.JSONDecodeError):
                    return True
                return not isinstance(payload, dict) or payload.get("status") == "DEPRECATED"
            if current == root or root not in current.parents:
                return False
            current = current.parent

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @contextmanager
    def _sqlite_bag_path(self, path: Path):
        if path.suffix != ".zstd":
            yield path
            return
        temporary_root = self.root / ".dense-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=temporary_root, suffix=".db3", delete=False) as destination:
            temporary_path = Path(destination.name)
            with path.open("rb") as source:
                zstandard.ZstdDecompressor().copy_stream(source, destination)
        try:
            yield temporary_path
        finally:
            for temporary_sqlite_path in (
                temporary_path,
                temporary_path.with_name(temporary_path.name + "-shm"),
                temporary_path.with_name(temporary_path.name + "-wal"),
                temporary_path.with_name(temporary_path.name + "-journal"),
            ):
                temporary_sqlite_path.unlink(missing_ok=True)

    def snapshot(self) -> dict:
        return {
            "integrated_artifacts": self._integrated_artifacts,
            "integrated_frames": self._integrated_frames,
            "integrated_points": self._integrated_points,
            "errors": self._errors,
            "last_error": self._last_error,
            "release_authorized_artifacts": self._release_authorized_artifacts,
        }
