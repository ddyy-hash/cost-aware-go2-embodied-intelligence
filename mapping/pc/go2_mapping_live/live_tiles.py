"""Immutable binary tile publication for browser-side indexed meshes."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
from typing import Callable, Sequence
import uuid

import numpy as np

from .live_surface_mesher import (
    MeshTileRevision,
    TRIANGLE_PROVENANCE_NAMES,
    TriangleProvenance,
)


_HEADER = struct.Struct("<4sIIII")
_AUTHORITY = "LIVE_PROVISIONAL_SURFACE"
_QUALITY = "PREVIEW_UNCALIBRATED"
_PROVENANCE_ENCODING = {
    str(code): name for code, name in sorted(TRIANGLE_PROVENANCE_NAMES.items())
}
_UNSUPPORTED_OPERATION_ERRNOS = {
    value
    for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    if (value := getattr(errno, name, None)) is not None
}
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


def _root_process_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def _is_unsupported_operation(exc: OSError) -> bool:
    return exc.errno in _UNSUPPORTED_OPERATION_ERRNOS or (
        os.name == "nt" and getattr(exc, "winerror", None) in {1, 50, 87}
    )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class DecodedSurfaceTile:
    binary_schema_version: int
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    triangle_provenance: np.ndarray
    triangle_uncertainty_m: np.ndarray


@dataclass(frozen=True)
class TilePublicationRequest:
    building_id: str
    floor_id: str
    component_id: str
    attachment_state: str
    submap_id: str
    revision: MeshTileRevision
    source_generation_id: str | None = None
    pose_revision_id: str | None = None


@dataclass(frozen=True)
class TileBatchPublicationResult:
    metadata: tuple[dict, ...]

    @property
    def publications(self) -> tuple[dict, ...]:
        return self.metadata


@dataclass(frozen=True)
class _PreparedTilePublication:
    body: bytes
    metadata: dict
    revision_id: str
    tile_key: str


@dataclass(frozen=True)
class _RawIndexSnapshot:
    existed: bool
    body: bytes


TileBatchFaultInjector = Callable[[str, int | None], None]


def decode_surface_tile(body: bytes) -> DecodedSurfaceTile:
    """Decode both historical measured-only v1 and evidence-bearing v2 tiles."""
    if len(body) < _HEADER.size:
        raise ValueError("surface tile header is truncated")
    magic, version, vertex_count, triangle_count, scale_count = _HEADER.unpack_from(body)
    if magic != b"G2TM" or version not in {1, 2} or scale_count != triangle_count:
        raise ValueError("surface tile header is invalid")
    positions_offset = _HEADER.size
    normals_offset = positions_offset + vertex_count * 12
    triangles_offset = normals_offset + vertex_count * 12
    scales_offset = triangles_offset + triangle_count * 12
    uncertainty_offset = scales_offset + triangle_count * 4
    provenance_offset = uncertainty_offset + (triangle_count * 4 if version == 2 else 0)
    expected_bytes = provenance_offset + (triangle_count if version == 2 else 0)
    if expected_bytes != len(body):
        raise ValueError("surface tile byte count is invalid")
    positions = np.frombuffer(body, dtype="<f4", count=vertex_count * 3, offset=positions_offset).reshape(-1, 3).copy()
    normals = np.frombuffer(body, dtype="<f4", count=vertex_count * 3, offset=normals_offset).reshape(-1, 3).copy()
    triangles = np.frombuffer(body, dtype="<u4", count=triangle_count * 3, offset=triangles_offset).reshape(-1, 3).copy()
    scales = np.frombuffer(body, dtype="<f4", count=triangle_count, offset=scales_offset).copy()
    if version == 2:
        uncertainty = np.frombuffer(
            body, dtype="<f4", count=triangle_count, offset=uncertainty_offset
        ).copy()
        provenance = np.frombuffer(
            body, dtype="u1", count=triangle_count, offset=provenance_offset
        ).copy()
    else:
        uncertainty = scales.copy()
        provenance = np.full(
            triangle_count, int(TriangleProvenance.MEASURED_LIDAR), dtype="u1"
        )
    if len(triangles) and int(triangles.max()) >= len(positions):
        raise ValueError("surface tile triangle index is out of range")
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError("surface tile geometry contains non-finite values")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("surface tile scales are invalid")
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0) or np.any(uncertainty > 1.0):
        raise ValueError("surface tile uncertainty is invalid")
    if any(int(value) not in TRIANGLE_PROVENANCE_NAMES for value in provenance):
        raise ValueError("surface tile provenance is invalid")
    return DecodedSurfaceTile(
        version,
        positions,
        normals,
        triangles,
        scales,
        provenance,
        uncertainty,
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if _is_unsupported_operation(exc) or (
            os.name == "nt" and exc.errno in {errno.EACCES, errno.EPERM}
        ):
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if not _is_unsupported_operation(exc):
            raise
    finally:
        os.close(descriptor)


class _StoreFileLock:
    def __init__(
        self,
        path: Path,
        release_diagnostic: Callable[[BaseException], None],
    ) -> None:
        self.path = path
        self._release_diagnostic = release_diagnostic
        self._file = None
        self._locked = False

    def _close_file(self) -> None:
        file = self._file
        self._file = None
        if file is not None:
            file.close()

    def _acquire_os_lock(self) -> None:
        if self._file is None:
            raise RuntimeError("tile store lock file is not open")
        if os.name == "nt":
            import msvcrt

            while True:
                self._file.seek(0)
                try:
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
                    self._locked = True
                    return
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            self._locked = True

    def __enter__(self) -> "_StoreFileLock":
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self._file = os.fdopen(descriptor, "a+b")
        try:
            self._file.seek(0, os.SEEK_END)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
                os.fsync(self._file.fileno())
            self._file.seek(0)
            self._acquire_os_lock()
            if not self._locked:
                raise OSError(errno.ENOLCK, "tile store OS lock was not acquired")
            return self
        except BaseException as acquire_error:
            try:
                self._close_file()
            except BaseException as close_error:
                if hasattr(acquire_error, "add_note"):
                    acquire_error.add_note(
                        "tile store lock cleanup failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
            self._locked = False
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self._close_file()
        except BaseException as release_error:
            detail = (
                "tile store lock release failed: "
                f"{type(release_error).__name__}: {release_error}"
            )
            if exc_value is not None and hasattr(exc_value, "add_note"):
                exc_value.add_note(detail)
            elif exc_value is None:
                try:
                    self._release_diagnostic(release_error)
                except BaseException:
                    pass
        finally:
            self._locked = False


class ImmutableTileStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.revisions_root = self.root / "revisions"
        self.index_path = self.root / "current-tiles.json"
        self.lock_path = self.root / ".current-tiles.lock"
        self._lock = _root_process_lock(self.root)
        self._lock_release_failures = 0
        self._last_lock_release_error: str | None = None
        with self._lock:
            self.revisions_root.mkdir(parents=True, exist_ok=True)

    @property
    def last_lock_release_error(self) -> str | None:
        with self._lock:
            return self._last_lock_release_error

    def lock_diagnostics(self) -> dict:
        with self._lock:
            return {
                "release_failures": self._lock_release_failures,
                "last_release_error": self._last_lock_release_error,
            }

    def _record_lock_release_error(self, exc: BaseException) -> None:
        self._lock_release_failures += 1
        self._last_lock_release_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _body(revision: MeshTileRevision) -> bytes:
        _nonnegative_integer(revision.unresolved_surfel_count, "unresolved_surfel_count")
        raw_positions = np.asarray(revision.positions)
        raw_normals = np.asarray(revision.normals)
        if (
            raw_positions.ndim != 2
            or raw_positions.shape[1:] != (3,)
            or raw_normals.shape != raw_positions.shape
        ):
            raise ValueError("tile positions and normals must be matching Nx3 arrays")
        for name, values in (("positions", raw_positions), ("normals", raw_normals)):
            if (
                not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.bool_)
                or np.issubdtype(values.dtype, np.complexfloating)
            ):
                raise ValueError(f"tile {name} must be numeric")
        positions = np.ascontiguousarray(raw_positions, dtype="<f4")
        normals = np.ascontiguousarray(raw_normals, dtype="<f4")
        if not np.isfinite(positions).all() or not np.isfinite(normals).all():
            raise ValueError("tile positions and normals must be finite")

        raw_triangles = np.asarray(revision.triangles)
        if raw_triangles.ndim != 2 or raw_triangles.shape[1:] != (3,):
            raise ValueError("tile triangles have an invalid shape")
        if np.issubdtype(raw_triangles.dtype, np.bool_) or not np.issubdtype(
            raw_triangles.dtype, np.integer
        ):
            raise ValueError("tile triangle indices must be integers")
        if len(raw_triangles) and np.any(raw_triangles < 0):
            raise ValueError("tile triangle index is negative")
        if len(raw_triangles) and int(raw_triangles.max()) >= len(positions):
            raise ValueError("tile triangle index is out of range")
        triangles = np.ascontiguousarray(raw_triangles, dtype="<u4")
        triangle_count = len(triangles)

        raw_scales = np.asarray(revision.triangle_scale_m)
        if raw_scales.shape != (triangle_count,):
            raise ValueError("tile triangle scales have an invalid shape")
        if (
            not np.issubdtype(raw_scales.dtype, np.number)
            or np.issubdtype(raw_scales.dtype, np.bool_)
            or np.issubdtype(raw_scales.dtype, np.complexfloating)
        ):
            raise ValueError("tile triangle scales must be numeric")
        scales = np.ascontiguousarray(raw_scales, dtype="<f4")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("tile triangle scales are invalid")

        if revision.triangle_provenance is None:
            provenance = np.full(
                triangle_count, int(TriangleProvenance.MEASURED_LIDAR), dtype="u1"
            )
        else:
            raw_provenance = np.asarray(revision.triangle_provenance)
            if raw_provenance.shape != (triangle_count,):
                raise ValueError("tile triangle provenance has an invalid shape")
            if np.issubdtype(raw_provenance.dtype, np.bool_) or not np.issubdtype(
                raw_provenance.dtype, np.integer
            ):
                raise ValueError("tile triangle provenance must contain integers")
            if any(int(value) not in TRIANGLE_PROVENANCE_NAMES for value in raw_provenance):
                raise ValueError("tile triangle provenance is invalid")
            provenance = np.ascontiguousarray(raw_provenance, dtype="u1")
        if any(int(value) not in TRIANGLE_PROVENANCE_NAMES for value in provenance):
            raise ValueError("tile triangle provenance is invalid")

        if revision.triangle_uncertainty_m is None:
            uncertainty = scales.copy()
        else:
            raw_uncertainty = np.asarray(revision.triangle_uncertainty_m)
            if raw_uncertainty.shape != (triangle_count,):
                raise ValueError("tile triangle uncertainty has an invalid shape")
            if (
                not np.issubdtype(raw_uncertainty.dtype, np.number)
                or np.issubdtype(raw_uncertainty.dtype, np.bool_)
                or np.issubdtype(raw_uncertainty.dtype, np.complexfloating)
            ):
                raise ValueError("tile triangle uncertainty must be numeric")
            uncertainty = np.ascontiguousarray(raw_uncertainty, dtype="<f4")
        if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0) or np.any(uncertainty > 1.0):
            raise ValueError("tile triangle uncertainty is invalid")
        return b"".join(
            (
                _HEADER.pack(b"G2TM", 2, len(positions), len(triangles), len(scales)),
                positions.tobytes(order="C"),
                normals.tobytes(order="C"),
                triangles.tobytes(order="C"),
                scales.tobytes(order="C"),
                uncertainty.tobytes(order="C"),
                provenance.tobytes(order="C"),
            )
        )

    @staticmethod
    def _identity(
        *,
        request: TilePublicationRequest,
        geometry_revision_id: str,
        content_sha256: str,
        attachment_state: str,
        source_generation_id: str | None,
        pose_revision_id: str | None,
        unresolved_surfel_count: int,
        schema_version: int,
    ) -> dict:
        identity = {
            "building_id": str(request.building_id),
            "floor_id": str(request.floor_id),
            "component_id": str(request.component_id),
            "attachment_state": attachment_state,
            "submap_id": str(request.submap_id),
            "tile_id": [int(value) for value in request.revision.tile_id.as_tuple()],
            "geometry_revision_id": geometry_revision_id,
            "content_sha256": content_sha256,
            "binary_schema_version": 2,
            "provenance_encoding": dict(_PROVENANCE_ENCODING),
            "uncertainty_encoding": "float32-le-meters-per-triangle",
        }
        if source_generation_id is not None:
            identity["source_generation_id"] = source_generation_id
        if pose_revision_id is not None:
            identity["pose_revision_id"] = pose_revision_id
        if schema_version == 5:
            identity.update(
                {
                    "unresolved_surfel_count": unresolved_surfel_count,
                    "authority": _AUTHORITY,
                    "quality": _QUALITY,
                    "geometry_authority": False,
                }
            )
        return identity

    def _prepare(self, request: TilePublicationRequest) -> _PreparedTilePublication:
        if not isinstance(request, TilePublicationRequest):
            raise TypeError("tile batch entries must be TilePublicationRequest values")
        body = self._body(request.revision)
        decoded = decode_surface_tile(body)
        content_sha256 = hashlib.sha256(body).hexdigest()
        geometry_revision_id = _require_sha256(
            request.revision.revision_id, "tile revision_id"
        )
        source_generation_id = (
            None
            if request.source_generation_id is None
            else _require_sha256(request.source_generation_id, "source_generation_id")
        )
        pose_revision_id = (
            None
            if request.pose_revision_id is None
            else _require_sha256(request.pose_revision_id, "pose_revision_id")
        )
        if source_generation_id is None and pose_revision_id is not None:
            raise ValueError("pose revision requires a source generation")
        attachment_state = str(request.attachment_state).upper()
        if attachment_state not in {"ROOT", "DETACHED", "ATTACHED"}:
            raise ValueError("tile attachment_state is invalid")
        unresolved_surfel_count = _nonnegative_integer(
            request.revision.unresolved_surfel_count, "unresolved_surfel_count"
        )
        identity = self._identity(
            request=request,
            geometry_revision_id=geometry_revision_id,
            content_sha256=content_sha256,
            attachment_state=attachment_state,
            source_generation_id=source_generation_id,
            pose_revision_id=pose_revision_id,
            unresolved_surfel_count=unresolved_surfel_count,
            schema_version=5,
        )
        revision_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
        metadata = {
            "schema_version": 5,
            **identity,
            "revision_id": revision_id,
            "vertex_count": int(len(decoded.positions)),
            "triangle_count": int(len(decoded.triangles)),
            "binary_url": f"/api/building/surface/tiles/{revision_id}",
        }
        tile_key = "/".join(
            (
                metadata["building_id"],
                metadata["floor_id"],
                metadata["component_id"],
                metadata["submap_id"],
                *(str(value) for value in metadata["tile_id"]),
            )
        )
        return _PreparedTilePublication(body, metadata, revision_id, tile_key)

    def _load_index(self) -> dict:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") not in {1, 2} or not isinstance(payload.get("current"), dict):
                raise ValueError("tile index is invalid")
            if any(
                not isinstance(key, str)
                or not isinstance(revision_id, str)
                or len(revision_id) != 64
                or any(character not in "0123456789abcdef" for character in revision_id)
                for key, revision_id in payload["current"].items()
            ):
                raise ValueError("tile index entries are invalid")
            return {"schema_version": 2, "current": dict(payload["current"])}
        except FileNotFoundError:
            return {"schema_version": 2, "current": {}}

    def _replace_index_bytes(self, body: bytes) -> None:
        if not isinstance(body, bytes):
            raise TypeError("raw tile index body must be bytes")
        temporary = self.index_path.with_name(f".{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.index_path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_index(self, payload: dict) -> None:
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        self._replace_index_bytes(body)

    def _snapshot_index(self) -> _RawIndexSnapshot:
        try:
            return _RawIndexSnapshot(True, self.index_path.read_bytes())
        except FileNotFoundError:
            return _RawIndexSnapshot(False, b"")

    def _remove_index(self) -> None:
        try:
            self.index_path.unlink()
        except FileNotFoundError:
            return
        _fsync_directory(self.root)

    @staticmethod
    def _fault(
        fault_injector: TileBatchFaultInjector | None,
        phase: str,
        entry_index: int | None,
    ) -> None:
        if fault_injector is not None:
            fault_injector(phase, entry_index)

    def _restore_index(
        self,
        snapshot: _RawIndexSnapshot,
        fault_injector: TileBatchFaultInjector | None,
    ) -> None:
        self._fault(fault_injector, "before_index_restore", None)
        if snapshot.existed:
            try:
                unchanged = self.index_path.read_bytes() == snapshot.body
            except FileNotFoundError:
                unchanged = False
            if not unchanged:
                self._replace_index_bytes(snapshot.body)
        else:
            self._remove_index()
        self._fault(fault_injector, "after_index_restore", None)

    def _stage_revision(self, prepared: _PreparedTilePublication) -> None:
        destination = self.revisions_root / prepared.revision_id
        if destination.exists():
            existing, existing_body = self.load(prepared.revision_id)
            if existing != prepared.metadata or existing_body != prepared.body:
                raise ValueError("immutable tile revision identity conflicts with existing content")
            return
        temporary = self.revisions_root / f".partial-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            binary_path = temporary / "tile.bin"
            with binary_path.open("wb") as output:
                output.write(prepared.body)
                output.flush()
                os.fsync(output.fileno())
            metadata_path = temporary / "metadata.json"
            with metadata_path.open("w", encoding="utf-8", newline="\n") as output:
                json.dump(
                    prepared.metadata,
                    output,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.revisions_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def publish_batch(
        self,
        requests: Sequence[TilePublicationRequest],
        *,
        fault_injector: TileBatchFaultInjector | None = None,
    ) -> TileBatchPublicationResult:
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault_injector must be callable")
        prepared = tuple(self._prepare(request) for request in tuple(requests))
        tile_keys = [entry.tile_key for entry in prepared]
        if len(set(tile_keys)) != len(tile_keys):
            raise ValueError("tile batch contains duplicate current tile keys")
        if not prepared:
            return TileBatchPublicationResult(())

        with self._lock:
            with _StoreFileLock(self.lock_path, self._record_lock_release_error):
                snapshot = self._snapshot_index()
                try:
                    index = self._load_index()
                    self._fault(fault_injector, "after_index_loaded", None)
                    for entry_index, entry in enumerate(prepared):
                        self._fault(fault_injector, "before_revision_stage", entry_index)
                        self._stage_revision(entry)
                        self._fault(fault_injector, "after_revision_staged", entry_index)
                    for entry in prepared:
                        index["current"][entry.tile_key] = entry.revision_id
                    self._fault(fault_injector, "before_index_write", None)
                    self._write_index(index)
                    self._fault(fault_injector, "after_index_write", None)
                except BaseException as transaction_error:
                    try:
                        self._restore_index(snapshot, fault_injector)
                    except BaseException as rollback_error:
                        detail = (
                            "tile index rollback failed: "
                            f"{type(rollback_error).__name__}: {rollback_error}"
                        )
                        if hasattr(transaction_error, "add_note"):
                            transaction_error.add_note(detail)
                    raise
        return TileBatchPublicationResult(tuple(entry.metadata for entry in prepared))

    def publish(
        self,
        building_id: str,
        floor_id: str,
        component_id: str,
        attachment_state: str,
        submap_id: str,
        revision: MeshTileRevision,
        *,
        source_generation_id: str | None = None,
        pose_revision_id: str | None = None,
    ) -> dict:
        result = self.publish_batch(
            (
                TilePublicationRequest(
                    building_id=str(building_id),
                    floor_id=str(floor_id),
                    component_id=str(component_id),
                    attachment_state=str(attachment_state),
                    submap_id=str(submap_id),
                    revision=revision,
                    source_generation_id=source_generation_id,
                    pose_revision_id=pose_revision_id,
                ),
            )
        )
        return result.metadata[0]

    def load(self, revision_id: str) -> tuple[dict, bytes]:
        if len(revision_id) != 64 or any(character not in "0123456789abcdef" for character in revision_id):
            raise ValueError("tile revision_id is invalid")
        root = self.revisions_root / revision_id
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("immutable tile metadata is invalid")
        body = (root / "tile.bin").read_bytes()
        if metadata.get("revision_id") != revision_id or hashlib.sha256(body).hexdigest() != metadata.get("content_sha256"):
            raise ValueError("immutable tile revision failed verification")
        schema_version = metadata.get("schema_version")
        if isinstance(schema_version, bool) or schema_version not in {1, 2, 3, 4, 5}:
            raise ValueError("immutable tile schema is unsupported")
        if schema_version in {1, 2} and ("source_generation_id" in metadata or "pose_revision_id" in metadata):
            raise ValueError("legacy tile cannot claim source generation authority")
        if schema_version == 3 and "pose_revision_id" in metadata:
            raise ValueError("schema v3 tile cannot claim pose revision authority")
        if schema_version == 5 and "pose_revision_id" in metadata and "source_generation_id" not in metadata:
            raise ValueError("pose revision requires a source generation")
        if metadata.get("authority") != _AUTHORITY:
            raise ValueError("immutable tile authority is invalid")
        if metadata.get("quality") != _QUALITY:
            raise ValueError("immutable tile quality is invalid")
        if metadata.get("geometry_authority") is not False:
            raise ValueError("immutable tile geometry authority is invalid")
        if metadata.get("binary_url") != f"/api/building/surface/tiles/{revision_id}":
            raise ValueError("immutable tile binary URL is invalid")
        unresolved_surfel_count = _nonnegative_integer(
            metadata.get("unresolved_surfel_count"), "unresolved_surfel_count"
        )
        if schema_version in {2, 3, 4, 5}:
            identity = {
                name: metadata.get(name)
                for name in (
                    "building_id",
                    "floor_id",
                    "component_id",
                    "attachment_state",
                    "submap_id",
                    "tile_id",
                    "geometry_revision_id",
                    "content_sha256",
                )
            }
            if schema_version == 5 or "binary_schema_version" in metadata:
                identity["binary_schema_version"] = metadata.get("binary_schema_version")
                identity["provenance_encoding"] = metadata.get("provenance_encoding")
                identity["uncertainty_encoding"] = metadata.get("uncertainty_encoding")
                if identity["binary_schema_version"] != 2:
                    raise ValueError("immutable tile binary schema is invalid")
                if schema_version == 5 and identity["provenance_encoding"] != _PROVENANCE_ENCODING:
                    raise ValueError("immutable tile provenance encoding is invalid")
                if (
                    schema_version == 5
                    and identity["uncertainty_encoding"]
                    != "float32-le-meters-per-triangle"
                ):
                    raise ValueError("immutable tile uncertainty encoding is invalid")
            if schema_version in {3, 4} or (
                schema_version == 5 and "source_generation_id" in metadata
            ):
                source_generation_id = metadata.get("source_generation_id")
                if not isinstance(source_generation_id, str) or len(source_generation_id) != 64 or any(
                    character not in "0123456789abcdef" for character in source_generation_id
                ):
                    raise ValueError("immutable tile source generation identity is invalid")
                identity["source_generation_id"] = source_generation_id
            if schema_version == 4 or (
                schema_version == 5 and "pose_revision_id" in metadata
            ):
                pose_revision_id = metadata.get("pose_revision_id")
                if not isinstance(pose_revision_id, str) or len(pose_revision_id) != 64 or any(
                    character not in "0123456789abcdef" for character in pose_revision_id
                ):
                    raise ValueError("immutable tile pose revision identity is invalid")
                identity["pose_revision_id"] = pose_revision_id
            if schema_version == 5:
                identity.update(
                    {
                        "unresolved_surfel_count": unresolved_surfel_count,
                        "authority": metadata.get("authority"),
                        "quality": metadata.get("quality"),
                        "geometry_authority": metadata.get("geometry_authority"),
                    }
                )
            expected = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            if expected != revision_id:
                raise ValueError("immutable tile placement identity failed verification")
        decoded = decode_surface_tile(body)
        declared_binary_version = metadata.get("binary_schema_version")
        if declared_binary_version is not None and declared_binary_version != decoded.binary_schema_version:
            raise ValueError("immutable tile binary schema does not match metadata")
        vertex_count = _nonnegative_integer(metadata.get("vertex_count"), "vertex_count")
        triangle_count = _nonnegative_integer(metadata.get("triangle_count"), "triangle_count")
        if vertex_count != len(decoded.positions) or triangle_count != len(decoded.triangles):
            raise ValueError("immutable tile decoded counts do not match metadata")
        return metadata, body

    def manifest_for_assembly(self, assembly) -> dict:
        assembly_id = str(assembly.assembly_id)
        if len(assembly_id) != 64 or any(character not in "0123456789abcdef" for character in assembly_id):
            raise ValueError("assembly_id is invalid")
        members = tuple(assembly.members)
        if not members:
            raise ValueError("assembly has no members")
        member_by_generation = {
            (str(member.submap_id), str(member.generation_id)): member
            for member in members
        }
        if len(member_by_generation) != len(members):
            raise ValueError("assembly member generations are duplicated")
        with self._lock:
            local_tiles = self.manifest(scope="local")["tiles"]
            tiles = []
            for metadata in local_tiles:
                if metadata.get("schema_version") not in {3, 4, 5}:
                    continue
                source_generation_id = metadata.get("source_generation_id")
                member = member_by_generation.get((metadata.get("submap_id"), source_generation_id))
                if member is None:
                    continue
                if metadata.get("building_id") != member.building_id or metadata.get("component_id") != member.component_id:
                    continue
                transform = list(member.transform_building_from_submap)
                placement_identity = {
                    "assembly_id": assembly_id,
                    "source_revision_id": metadata["revision_id"],
                    "component_id": member.component_id,
                    "submap_id": member.submap_id,
                    "generation_id": member.generation_id,
                    "transform_building_from_submap": transform,
                }
                placement_revision_id = hashlib.sha256(
                    json.dumps(placement_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                ).hexdigest()
                tiles.append({
                    **metadata,
                    "assembly_id": assembly_id,
                    "placement_revision_id": placement_revision_id,
                    "attachment_state": "ROOT" if member.source_decision_id is None else "ATTACHED",
                    "transform_building_from_submap": transform,
                })
            tiles.sort(key=lambda item: (item["building_id"], item["floor_id"], item["submap_id"], item["tile_id"]))
            return {
                "schema_version": 3,
                "scope": "building",
                "assembly_id": assembly_id,
                "tiles": tiles,
            }

    def manifest(self, *, scope: str = "local") -> dict:
        scope = str(scope).lower()
        if scope not in {"local", "building", "diagnostic"}:
            raise ValueError("tile manifest scope is invalid")
        with self._lock:
            index = self._load_index()
            tiles = []
            for revision_id in sorted(set(index["current"].values())):
                metadata, _ = self.load(str(revision_id))
                if scope == "building" and metadata.get("attachment_state") not in {"ROOT", "ATTACHED"}:
                    continue
                tiles.append(metadata)
            tiles.sort(key=lambda item: (item["building_id"], item["floor_id"], item["submap_id"], item["tile_id"]))
            return {"schema_version": 2, "scope": scope, "tiles": tiles}
