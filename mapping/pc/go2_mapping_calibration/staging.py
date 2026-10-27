"""Read-only multi-process ROS bag staging for diagnostic calibration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import zstandard


class DiagnosticStagingError(RuntimeError):
    """Raised when immutable diagnostic inputs cannot be materialized safely."""


@dataclass(frozen=True)
class StagedPart:
    source_relative_path: str
    source_sha256: str
    source_bytes: int
    staged_relative_path: str
    staged_sha256: str
    staged_bytes: int


@dataclass(frozen=True)
class StagedCapture:
    source_path: Path
    staged_path: Path
    parts: tuple[StagedPart, ...]
    source_parts_sha256: str

    @property
    def manifest_path(self) -> Path:
        return self.staged_path / "diagnostic-staging.json"


@dataclass(frozen=True)
class _MaterializationTask:
    source_path: Path
    source_relative_path: str
    staged_path: Path
    staged_relative_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _staged_relative_path(path: Path) -> str:
    if path.name.lower().endswith(".db3.zstd"):
        return path.name[: -len(".zstd")]
    return path.name


def _materialize_part(task: _MaterializationTask) -> StagedPart:
    temporary = task.staged_path.with_name(
        f".{task.staged_path.name}.{os.getpid()}.partial"
    )
    try:
        source_sha256 = _sha256(task.source_path)
        with task.source_path.open("rb") as source, temporary.open("xb") as destination:
            if task.source_path.name.lower().endswith(".zstd"):
                zstandard.ZstdDecompressor().copy_stream(source, destination)
            else:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, task.staged_path)
        return StagedPart(
            source_relative_path=task.source_relative_path,
            source_sha256=source_sha256,
            source_bytes=task.source_path.stat().st_size,
            staged_relative_path=task.staged_relative_path,
            staged_sha256=_sha256(task.staged_path),
            staged_bytes=task.staged_path.stat().st_size,
        )
    except (OSError, zstandard.ZstdError) as exc:
        temporary.unlink(missing_ok=True)
        raise DiagnosticStagingError(
            f"cannot materialize immutable diagnostic part: {task.source_path}"
        ) from exc


def _source_parts(path: Path) -> tuple[Path, ...]:
    if not path.is_dir():
        raise DiagnosticStagingError(f"diagnostic capture is not a directory: {path}")
    parts = tuple(
        item
        for item in sorted(path.iterdir())
        if item.is_file()
        and not item.name.startswith(".")
        and (item.name.lower().endswith(".db3") or item.name.lower().endswith(".db3.zstd"))
    )
    if not parts:
        raise DiagnosticStagingError(
            f"diagnostic capture has no published db3 parts: {path}"
        )
    staged_names = [_staged_relative_path(item) for item in parts]
    if len(set(staged_names)) != len(staged_names):
        raise DiagnosticStagingError(
            f"diagnostic capture has colliding staged part names: {path}"
        )
    return parts


def _parts_identity(parts: tuple[StagedPart, ...]) -> str:
    payload = [
        {
            "relative_path": item.source_relative_path,
            "sha256": item.source_sha256,
            "bytes": item.source_bytes,
        }
        for item in parts
    ]
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _write_manifest(capture: StagedCapture, directory: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "READ_ONLY_UNCOMPRESSED_DIAGNOSTIC_STAGING",
        "source_path": str(capture.source_path),
        "source_parts_sha256": capture.source_parts_sha256,
        "parts": [asdict(item) for item in capture.parts],
    }
    path = directory / capture.manifest_path.name
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _load_reusable_stage(
    source: Path, destination: Path, source_parts: tuple[Path, ...]
) -> StagedCapture:
    manifest_path = destination / "diagnostic-staging.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticStagingError(
            f"existing diagnostic stage has no readable manifest: {destination}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "READ_ONLY_UNCOMPRESSED_DIAGNOSTIC_STAGING"
        or Path(str(payload.get("source_path", ""))).resolve(strict=False) != source
        or not isinstance(payload.get("parts"), list)
    ):
        raise DiagnosticStagingError(
            f"existing diagnostic stage manifest does not match its source: {destination}"
        )
    try:
        staged_parts = tuple(StagedPart(**item) for item in payload["parts"])
    except (TypeError, ValueError) as exc:
        raise DiagnosticStagingError(
            f"existing diagnostic stage manifest has invalid parts: {destination}"
        ) from exc
    if len(staged_parts) != len(source_parts):
        raise DiagnosticStagingError(
            f"existing diagnostic stage has an incomplete part set: {destination}"
        )
    for source_part, staged_part in zip(source_parts, staged_parts):
        if (
            staged_part.source_relative_path != source_part.name
            or staged_part.staged_relative_path != _staged_relative_path(source_part)
            or staged_part.source_bytes != source_part.stat().st_size
            or staged_part.source_sha256 != _sha256(source_part)
        ):
            raise DiagnosticStagingError(
                f"existing diagnostic stage source identity mismatch: {destination}"
            )
        staged_path = destination / staged_part.staged_relative_path
        if (
            not staged_path.is_file()
            or staged_part.staged_bytes != staged_path.stat().st_size
            or staged_part.staged_sha256 != _sha256(staged_path)
        ):
            raise DiagnosticStagingError(
                f"existing diagnostic stage content mismatch: {destination}"
            )
    source_parts_sha256 = _parts_identity(staged_parts)
    if payload.get("source_parts_sha256") != source_parts_sha256:
        raise DiagnosticStagingError(
            f"existing diagnostic stage aggregate identity mismatch: {destination}"
        )
    return StagedCapture(
        source_path=source,
        staged_path=destination,
        parts=staged_parts,
        source_parts_sha256=source_parts_sha256,
    )


def stage_capture_for_diagnostic(
    source_path: str | Path,
    staging_root: str | Path,
    *,
    workers: int,
) -> StagedCapture:
    """Materialize immutable bag parts before promoting one complete stage.

    The source capture stays read-only. A failed part removes only the newly
    created temporary staging directory, so callers never receive a partial
    directory as an analysis input.
    """

    source = Path(source_path).resolve(strict=True)
    root = Path(staging_root).resolve(strict=False)
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    parts = _source_parts(source)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.name
    if destination.exists():
        return _load_reusable_stage(source, destination, parts)
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}.partial-", dir=root))
    tasks = tuple(
        _MaterializationTask(
            source_path=part,
            source_relative_path=part.relative_to(source).as_posix(),
            staged_path=temporary / _staged_relative_path(part),
            staged_relative_path=_staged_relative_path(part),
        )
        for part in parts
    )
    try:
        worker_count = min(int(workers), len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            staged_parts = tuple(executor.map(_materialize_part, tasks))
        staged = StagedCapture(
            source_path=source,
            staged_path=destination,
            parts=staged_parts,
            source_parts_sha256=_parts_identity(staged_parts),
        )
        _write_manifest(staged, temporary)
        temporary.rename(destination)
        return staged
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
