"""Immutable, assembly-scoped Building appearance generations."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256

from .tile_identity import DeliveryAuthorization
from .tile_store import PublishedTileAppearance


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_REQUIRED_COVERAGE_PPM = 800_000
_REQUIRED_INDEPENDENT_VIEWS = 2
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$", re.ASCII)
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


def _process_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(path))
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse_point(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_path_chain(path: Path, name: str) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if _is_reparse_point(info):
            raise ValueError(f"{name} contains an unsafe symlink or reparse point")


def _safe_directory(path: Path, root: Path | None, name: str) -> Path:
    _safe_path_chain(path, name)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing") from exc
    if _is_reparse_point(info):
        raise ValueError(f"{name} is an unsafe symlink or reparse point")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} is not a directory")
    resolved = path.resolve(strict=True)
    if root is not None and not _within(resolved, root):
        raise ValueError(f"{name} escapes the Building appearance store root")
    return resolved


def _safe_file(path: Path, root: Path, name: str) -> None:
    _safe_path_chain(path, name)
    if not _lexists(path):
        return
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} disappeared during validation") from exc
    if _is_reparse_point(info):
        raise ValueError(f"{name} is an unsafe symlink or reparse point")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} is not a regular file")
    if not _within(path.resolve(strict=True), root):
        raise ValueError(f"{name} escapes the Building appearance store root")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_safe_file(path: Path, root: Path, name: str) -> bytes:
    _safe_path_chain(path, name)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{name} is unreadable") from exc
    try:
        opened_before = os.fstat(descriptor)
        current_before = path.lstat()
        if (
            _is_reparse_point(current_before)
            or not stat.S_ISREG(current_before.st_mode)
            or not _same_file_identity(opened_before, current_before)
            or not _within(path.resolve(strict=True), root)
        ):
            raise ValueError(f"{name} changed to an unsafe path")
        with os.fdopen(descriptor, "rb") as input_file:
            descriptor = None
            body = input_file.read()
            opened_after = os.fstat(input_file.fileno())
        current_after = path.lstat()
        if (
            _is_reparse_point(current_after)
            or not stat.S_ISREG(current_after.st_mode)
            or not _same_file_identity(opened_after, current_after)
            or opened_before.st_size != opened_after.st_size
            or opened_before.st_mtime_ns != opened_after.st_mtime_ns
        ):
            raise ValueError(f"{name} changed during read")
        return body
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_safe_directory_fd(path: Path, name: str) -> int:
    _safe_path_chain(path, name)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            _is_reparse_point(current)
            or not stat.S_ISDIR(current.st_mode)
            or not _same_file_identity(opened, current)
        ):
            raise ValueError(f"{name} changed to an unsafe path")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _replace_safe(source: Path, destination: Path) -> None:
    if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
        source_parent = _open_safe_directory_fd(source.parent, "Building appearance source directory")
        try:
            destination_parent = _open_safe_directory_fd(
                destination.parent,
                "Building appearance destination directory",
            )
            try:
                os.replace(
                    source.name,
                    destination.name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=destination_parent,
                )
            finally:
                os.close(destination_parent)
        finally:
            os.close(source_parent)
        return
    _safe_path_chain(source.parent, "Building appearance source directory")
    _safe_path_chain(destination.parent, "Building appearance destination directory")
    os.replace(source, destination)


def _lock_windows_file(
    file,
    msvcrt_module,
    *,
    sleep=time.sleep,
    clock=time.monotonic,
    timeout_seconds: float = 300.0,
) -> None:
    deadline = clock() + timeout_seconds
    while True:
        file.seek(0)
        try:
            msvcrt_module.locking(file.fileno(), msvcrt_module.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            if clock() >= deadline:
                raise TimeoutError("timed out waiting for the Building appearance store lock") from exc
            sleep(0.05)


class _StoreFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._process_lock = _process_lock(path)
        self._file = None

    def __enter__(self) -> _StoreFileLock:
        self._process_lock.acquire()
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            self._file = os.fdopen(descriptor, "a+b")
            if _is_reparse_point(self.path.lstat()) or not stat.S_ISREG(self.path.stat().st_mode):
                raise ValueError("Building appearance store lock is an unsafe path")
            self._file.seek(0, os.SEEK_END)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
                os.fsync(self._file.fileno())
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                _lock_windows_file(self._file, msvcrt)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            self.assert_current()
            return self
        except BaseException:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._process_lock.release()
            raise

    def assert_current(self) -> None:
        if self._file is None:
            raise ValueError("Building appearance store lock is not held")
        current = self.path.lstat()
        opened = os.fstat(self._file.fileno())
        if (
            _is_reparse_point(current)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink == 0
            or not _same_file_identity(opened, current)
        ):
            raise ValueError("Building appearance store lock identity changed")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._file is not None:
                if os.name == "nt":
                    import msvcrt

                    self._file.seek(0)
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._process_lock.release()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _rigid(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 16:
        raise ValueError("transform_building_from_submap must contain 16 values")
    try:
        matrix = np.asarray(tuple(float(value) for value in values), dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("transform_building_from_submap must be rigid") from exc
    rotation = matrix[:3, :3]
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("transform_building_from_submap must be rigid")
    return tuple(float(value) for value in matrix.reshape(-1))


@dataclass(frozen=True)
class DistanceLodInterval:
    minimum_distance_m: float
    maximum_distance_m: float | None
    lod_id: str

    def __post_init__(self) -> None:
        minimum = float(self.minimum_distance_m)
        maximum = None if self.maximum_distance_m is None else float(self.maximum_distance_m)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("LOD minimum distance must be finite and non-negative")
        if maximum is not None and (not math.isfinite(maximum) or maximum <= minimum):
            raise ValueError("LOD maximum distance must be greater than its minimum")
        object.__setattr__(self, "minimum_distance_m", minimum)
        object.__setattr__(self, "maximum_distance_m", maximum)
        object.__setattr__(self, "lod_id", _sha256(self.lod_id, "LOD ID"))

    def payload(self) -> dict[str, Any]:
        return {
            "minimum_distance_m": self.minimum_distance_m,
            "maximum_distance_m": self.maximum_distance_m,
            "lod_id": self.lod_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DistanceLodInterval":
        return cls(
            payload["minimum_distance_m"],
            payload.get("maximum_distance_m"),
            payload["lod_id"],
        )


@dataclass(frozen=True)
class BuildingAppearanceMember:
    building_id: str
    component_id: str
    session_id: str
    submap_id: str
    source_generation_id: str
    placement_revision_id: str
    tile_revision_id: str
    tile_appearance_revision_id: str
    authorized_lod_ids: tuple[str, ...]
    distance_policy: tuple[DistanceLodInterval, ...]
    transform_building_from_submap: tuple[float, ...]
    delivery_authorization: DeliveryAuthorization
    rgbd_augmented_revision_id: str | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.building_id, "building_id"),
            (self.component_id, "component_id"),
            (self.session_id, "session_id"),
            (self.submap_id, "submap_id"),
        ):
            _identifier(value, name)
        for value, name in (
            (self.source_generation_id, "source_generation_id"),
            (self.placement_revision_id, "placement_revision_id"),
            (self.tile_revision_id, "tile_revision_id"),
            (self.tile_appearance_revision_id, "tile_appearance_revision_id"),
        ):
            _sha256(value, name)
        lods = tuple(_sha256(value, "authorized LOD ID") for value in self.authorized_lod_ids)
        if not lods or len(set(lods)) != len(lods):
            raise ValueError("authorized LOD IDs must be unique and non-empty")
        policy = tuple(self.distance_policy)
        if not policy:
            raise ValueError("distance policy must not be empty")
        if any(not isinstance(value, DistanceLodInterval) for value in policy):
            raise TypeError("distance policy must contain DistanceLodInterval values")
        if policy[0].minimum_distance_m != 0.0:
            raise ValueError("distance policy must start at zero")
        if policy[-1].maximum_distance_m is not None:
            raise ValueError("distance policy must contain a complete final interval")
        if tuple(value.lod_id for value in policy) != lods:
            raise ValueError("authorized LOD IDs must match distance policy order")
        for previous, current in zip(policy, policy[1:]):
            if previous.maximum_distance_m != current.minimum_distance_m:
                raise ValueError("distance policy intervals must be contiguous")
        if not isinstance(self.delivery_authorization, DeliveryAuthorization):
            raise TypeError("delivery_authorization must be DeliveryAuthorization")
        if self.delivery_authorization.delivery_authorized is not True:
            raise ValueError("Building appearance member delivery is unauthorized")
        augmentation = self.rgbd_augmented_revision_id
        if augmentation is not None:
            _sha256(augmentation, "rgbd_augmented_revision_id")
        object.__setattr__(self, "authorized_lod_ids", lods)
        object.__setattr__(self, "distance_policy", policy)
        object.__setattr__(self, "transform_building_from_submap", _rigid(self.transform_building_from_submap))

    def payload(self) -> dict[str, Any]:
        authorization = self.delivery_authorization
        return {
            "building_id": self.building_id,
            "component_id": self.component_id,
            "session_id": self.session_id,
            "submap_id": self.submap_id,
            "source_generation_id": self.source_generation_id,
            "placement_revision_id": self.placement_revision_id,
            "tile_revision_id": self.tile_revision_id,
            "tile_appearance_revision_id": self.tile_appearance_revision_id,
            "authorized_lod_ids": list(self.authorized_lod_ids),
            "distance_policy": [value.payload() for value in self.distance_policy],
            "transform_building_from_submap": list(self.transform_building_from_submap),
            "delivery_authorization": _authorization_payload(authorization),
            "rgbd_augmented_revision_id": self.rgbd_augmented_revision_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BuildingAppearanceMember":
        return cls(
            building_id=payload["building_id"],
            component_id=payload["component_id"],
            session_id=payload["session_id"],
            submap_id=payload["submap_id"],
            source_generation_id=payload["source_generation_id"],
            placement_revision_id=payload["placement_revision_id"],
            tile_revision_id=payload["tile_revision_id"],
            tile_appearance_revision_id=payload["tile_appearance_revision_id"],
            authorized_lod_ids=tuple(payload["authorized_lod_ids"]),
            distance_policy=tuple(DistanceLodInterval.from_payload(value) for value in payload["distance_policy"]),
            transform_building_from_submap=tuple(payload["transform_building_from_submap"]),
            delivery_authorization=_authorization_from_payload(payload["delivery_authorization"]),
            rgbd_augmented_revision_id=payload.get("rgbd_augmented_revision_id"),
        )


@dataclass(frozen=True)
class PublishedBuildingAppearance:
    building_appearance_revision_id: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    pointer: Mapping[str, Any]
    root_path: Path
    manifest_path: Path


class BuildingAppearanceStore:
    """Freeze a complete authorized assembly appearance generation."""

    def __init__(
        self,
        root: str | Path,
        *,
        assembly_resolver: Callable[[str], Any],
        tile_manifest_resolver: Callable[[Any], Mapping[str, Any]],
        tile_resolver: Callable[[str], tuple[Mapping[str, Any], bytes]],
        tile_appearance_resolver: Callable[[str, str], PublishedTileAppearance],
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root) / "building-appearance"
        self.revisions_root = self.root / "revisions"
        self.staging_root = self.root / ".staging"
        self.current_path = self.root / "CURRENT.json"
        self.lock_path = self.root / ".store.lock"
        _safe_path_chain(self.root, "Building appearance store root")
        if _lexists(self.root):
            _safe_directory(self.root, None, "Building appearance store root")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self._canonical_root = _safe_directory(
            self.root,
            None,
            "Building appearance store root",
        )
        _safe_file(self.lock_path, self._canonical_root, "Building appearance store lock")
        self.assembly_resolver = assembly_resolver
        self.tile_manifest_resolver = tile_manifest_resolver
        self.tile_resolver = tile_resolver
        self.tile_appearance_resolver = tile_appearance_resolver
        self._fault_injector = fault_injector
        self._lock = threading.RLock()
        self._file_lock = _StoreFileLock(self.lock_path)
        with self._file_lock:
            for path, name in (
                (self.revisions_root, "Building appearance revisions root"),
                (self.staging_root, "Building appearance staging root"),
            ):
                if _lexists(path):
                    _safe_directory(path, self._canonical_root, name)
                else:
                    path.mkdir()
                    _safe_directory(path, self._canonical_root, name)
            self._validate_layout()
            self._cleanup_staging()

    def publish(
        self,
        *,
        assembly: Any,
        members: Sequence[BuildingAppearanceMember],
    ) -> PublishedBuildingAppearance:
        normalized, core = self._validated_core(assembly, tuple(members))
        revision = canonical_sha256(core)
        manifest = {**core, "building_appearance_revision_id": revision}
        manifest_bytes = _canonical_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        pointer = self._pointer(core, revision, manifest_sha256)
        with self._lock, self._file_lock:
            self._validate_layout()
            self._file_lock.assert_current()
            target = self.revisions_root / revision
            if _lexists(target):
                _safe_directory(
                    target,
                    self._canonical_root,
                    "Building appearance revision",
                )
                published = self._load_generation(revision)
                if published.manifest != manifest:
                    raise ValueError("existing Building appearance identity conflicts with content")
            else:
                staging = self.staging_root / f"{uuid.uuid4().hex}.partial"
                staging.mkdir()
                renamed = False
                try:
                    self._write_fsync(staging / "manifest.json", manifest_bytes)
                    _fsync_directory(staging)
                    self._fault("after_files_fsynced")
                    _replace_safe(staging, target)
                    renamed = True
                    _fsync_directory(self.revisions_root)
                    self._fault("after_generation_rename")
                    published = self._load_generation(revision)
                finally:
                    if not renamed and _lexists(staging):
                        self._remove_staging_path(staging)
            self._fault("after_terminal_revalidation")
            self._file_lock.assert_current()
            self._write_pointer(pointer, validator=lambda: self._load_generation(revision))
            self._fault("after_pointer_replace")
            return published

    def resolve(self, building_appearance_revision_id: str) -> PublishedBuildingAppearance:
        return self._load_generation(_sha256(building_appearance_revision_id, "Building appearance revision"))

    def current(self, *, expected_assembly_id: str | None = None) -> PublishedBuildingAppearance | None:
        with self._lock:
            self._validate_layout()
            if not _lexists(self.current_path):
                return None
            _safe_file(
                self.current_path,
                self._canonical_root,
                "Building appearance CURRENT",
            )
            try:
                pointer_bytes = _read_safe_file(
                    self.current_path,
                    self._canonical_root,
                    "Building appearance CURRENT",
                )
                pointer = json.loads(pointer_bytes)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Building appearance CURRENT is unreadable") from exc
            if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
                raise ValueError("Building appearance CURRENT is invalid")
            if pointer_bytes != _canonical_bytes(pointer):
                raise ValueError("Building appearance CURRENT is not canonical")
            assembly_id = _sha256(pointer.get("assembly_id"), "assembly_id")
            if expected_assembly_id is not None and assembly_id != _sha256(expected_assembly_id, "expected assembly_id"):
                raise ValueError("Building appearance CURRENT is stale")
            published = self.resolve(pointer.get("building_appearance_revision_id"))
            if published.manifest_sha256 != pointer.get("manifest_sha256"):
                raise ValueError("Building appearance CURRENT manifest hash is invalid")
            if published.manifest.get("assembly_id") != assembly_id:
                raise ValueError("Building appearance CURRENT assembly identity is invalid")
            if pointer != published.pointer:
                raise ValueError("Building appearance CURRENT pointer is inconsistent")
            return published

    def _validated_core(
        self,
        assembly: Any,
        members: tuple[BuildingAppearanceMember, ...],
    ) -> tuple[tuple[BuildingAppearanceMember, ...], dict[str, Any]]:
        assembly_id = _sha256(getattr(assembly, "assembly_id", None), "assembly_id")
        building_id = _identifier(getattr(assembly, "building_id", None), "building_id")
        resolved = self.assembly_resolver(assembly_id)
        if resolved is None or getattr(resolved, "assembly_id", None) != assembly_id:
            raise ValueError("assembly is stale or unavailable")
        resolved_building_id = _identifier(
            getattr(resolved, "building_id", None),
            "resolved assembly building_id",
        )
        if building_id != resolved_building_id:
            raise ValueError("assembly building identity is mixed or stale")
        if getattr(resolved, "algorithm_version", None) != getattr(assembly, "algorithm_version", None):
            raise ValueError("assembly algorithm version changed")
        if tuple(getattr(resolved, "members", ())) != tuple(getattr(assembly, "members", ())):
            raise ValueError("assembly provenance changed")
        if any(getattr(value, "building_id", None) != resolved_building_id for value in resolved.members):
            raise ValueError("assembly members contain mixed Building identities")
        assembly_members = {
            (value.component_id, value.session_id, value.submap_id, value.generation_id): value
            for value in assembly.members
        }
        if len(assembly_members) != len(assembly.members):
            raise ValueError("assembly member generations are duplicated")
        tile_manifest = self.tile_manifest_resolver(assembly)
        if tile_manifest.get("assembly_id") != assembly_id or tile_manifest.get("scope") != "building":
            raise ValueError("assembly tile manifest provenance is invalid")
        tiles = tuple(tile_manifest.get("tiles", ()))
        tile_by_revision = {str(value.get("revision_id")): value for value in tiles}
        if len(tile_by_revision) != len(tiles) or set(tile_by_revision) != {value.tile_revision_id for value in members}:
            raise ValueError("Building appearance members must cover the complete visible tile set")
        if len({value.tile_revision_id for value in members}) != len(members):
            raise ValueError("Building appearance tile revisions are duplicated")

        normalized = []
        for member in members:
            tile = tile_by_revision.get(member.tile_revision_id)
            if tile is None:
                raise ValueError("Building appearance member tile is omitted or unknown")
            assembly_member = assembly_members.get(
                (member.component_id, member.session_id, member.submap_id, member.source_generation_id)
            )
            if assembly_member is None:
                raise ValueError("Building appearance member source generation is not in assembly")
            self._validate_member_sources(member, tile, assembly_member, assembly_id)
            normalized.append(member)
        normalized.sort(key=lambda value: (value.building_id, value.component_id, value.submap_id, value.tile_revision_id))
        core = {
            "schema_version": 1,
            "building_id": resolved_building_id,
            "assembly_id": assembly_id,
            "geometry_authority": "OBSERVED_SURFACE",
            "volume_filled": False,
            "watertight": False,
            "members": [value.payload() for value in normalized],
        }
        return tuple(normalized), core

    def _validate_member_sources(self, member, tile, assembly_member, assembly_id) -> None:
        if member.building_id != assembly_member.building_id or member.building_id != tile.get("building_id"):
            raise ValueError("Building appearance member building provenance is invalid")
        for name in ("component_id", "submap_id", "source_generation_id"):
            if getattr(member, name) != tile.get(name):
                raise ValueError(f"Building appearance member {name} mismatch")
        if member.transform_building_from_submap != tuple(assembly_member.transform_building_from_submap):
            raise ValueError("Building appearance transform does not match assembly member")
        if member.transform_building_from_submap != tuple(float(value) for value in tile.get("transform_building_from_submap", ())):
            raise ValueError("Building appearance placement transform mismatch")
        expected_placement = _placement_revision(assembly_id, tile, assembly_member)
        if member.placement_revision_id != expected_placement or tile.get("placement_revision_id") != expected_placement:
            raise ValueError("Building appearance placement revision mismatch")
        metadata, body = self.tile_resolver(member.tile_revision_id)
        if metadata.get("revision_id") != member.tile_revision_id or metadata.get("source_generation_id") != member.source_generation_id:
            raise ValueError("Building appearance source tile generation mismatch")
        for name in ("building_id", "component_id", "submap_id"):
            if metadata.get(name) != getattr(member, name):
                raise ValueError(f"Building appearance source tile {name} provenance mismatch")
        if "placement_revision_id" in metadata and metadata.get("placement_revision_id") != member.placement_revision_id:
            raise ValueError("Building appearance source tile placement revision mismatch")
        if "transform_building_from_submap" in metadata and tuple(
            float(value) for value in metadata["transform_building_from_submap"]
        ) != member.transform_building_from_submap:
            raise ValueError("Building appearance source tile placement transform mismatch")
        if metadata.get("content_sha256") != hashlib.sha256(bytes(body)).hexdigest():
            raise ValueError("Building appearance source tile content hash mismatch")
        if metadata.get("geometry_revision_id") != tile.get("geometry_revision_id"):
            raise ValueError("Building appearance source tile geometry mismatch")
        published = self.tile_appearance_resolver(member.tile_revision_id, member.tile_appearance_revision_id)
        manifest = getattr(published, "manifest", published)
        if manifest.get("appearance_revision_id") != member.tile_appearance_revision_id:
            raise ValueError("Building appearance tile revision mismatch")
        core = manifest.get("revision_core")
        if not isinstance(core, Mapping) or core.get("tile_revision_id") != member.tile_revision_id or core.get("source_generation_id") != member.source_generation_id:
            raise ValueError("Building appearance tile provenance mismatch")
        if canonical_sha256(core) != member.tile_appearance_revision_id:
            raise ValueError("Building appearance tile revision hash mismatch")
        if not isinstance(published, PublishedTileAppearance):
            raise ValueError("Building appearance resolver must return a verified tile appearance")
        published_root = Path(published.root_path)
        canonical_published_root = _safe_directory(
            published_root,
            None,
            "Building appearance tile appearance root",
        )
        expected_manifest_path = published_root / "manifest.json"
        if Path(published.manifest_path) != expected_manifest_path:
            raise ValueError("Building appearance tile manifest path is invalid")
        _safe_file(
            expected_manifest_path,
            canonical_published_root,
            "Building appearance tile manifest",
        )
        try:
            manifest_bytes = _read_safe_file(
                published.manifest_path,
                canonical_published_root,
                "Building appearance tile manifest",
            )
            on_disk_manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Building appearance tile manifest is unreadable") from exc
        if (
            on_disk_manifest != dict(manifest)
            or manifest_bytes != _canonical_bytes(on_disk_manifest)
            or hashlib.sha256(manifest_bytes).hexdigest() != published.manifest_sha256
            or published.tile_revision_id != member.tile_revision_id
            or published.appearance_revision_id != member.tile_appearance_revision_id
        ):
            raise ValueError("Building appearance tile manifest verification failed")
        if set(published.lod_paths) != {
            value.get("lod_id") for value in manifest.get("lods", ())
        }:
            raise ValueError("Building appearance tile LOD path set is invalid")
        for lod in manifest.get("lods", ()):
            lod_id = _sha256(lod.get("lod_id"), "tile appearance LOD ID")
            lod_core = lod.get("core")
            if not isinstance(lod_core, Mapping) or canonical_sha256(lod_core) != lod_id:
                raise ValueError("Building appearance tile LOD identity verification failed")
            artifact_records = lod_core.get("artifacts")
            if not isinstance(artifact_records, list):
                raise ValueError("Building appearance tile LOD artifacts are invalid")
            artifact_names = [
                value.get("name") if isinstance(value, Mapping) else None
                for value in artifact_records
            ]
            if any(not isinstance(value, str) for value in artifact_names) or len(set(artifact_names)) != len(artifact_names):
                raise ValueError("Building appearance tile LOD artifact names are invalid")
            required_artifacts = {"render-vertices.bin", "render-indices.bin", "render-uvs.bin"}
            atlas_count = lod_core.get("atlas_count")
            if not isinstance(atlas_count, int) or atlas_count < 1:
                raise ValueError("Building appearance tile LOD atlas count is invalid")
            expected_atlases = {f"atlas-{index}.png" for index in range(atlas_count)}
            if not required_artifacts.issubset(artifact_names) or {
                value for value in artifact_names if value.startswith("atlas-")
            } != expected_atlases:
                raise ValueError("Building appearance tile LOD artifact set is incomplete")
            lod_root = published.lod_paths.get(lod_id)
            if lod_root is None:
                raise ValueError("Building appearance tile LOD path is missing")
            expected_lod_root = published_root / "lods" / lod_id
            if Path(lod_root) != expected_lod_root:
                raise ValueError("Building appearance tile LOD path is invalid")
            canonical_lod_root = _safe_directory(
                expected_lod_root,
                canonical_published_root,
                "Building appearance tile LOD path",
            )
            for artifact in lod.get("core", {}).get("artifacts", ()):
                artifact_name = artifact.get("name")
                if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
                    raise ValueError("Building appearance tile artifact path is invalid")
                artifact_path = expected_lod_root / artifact_name
                _safe_file(
                    artifact_path,
                    canonical_lod_root,
                    "Building appearance tile artifact",
                )
                try:
                    artifact_body = _read_safe_file(
                        artifact_path,
                        canonical_lod_root,
                        "Building appearance tile artifact",
                    )
                except OSError as exc:
                    raise ValueError("Building appearance tile artifact is unavailable") from exc
                if len(artifact_body) != artifact["bytes"] or hashlib.sha256(artifact_body).hexdigest() != artifact["sha256"]:
                    raise ValueError("Building appearance tile artifact hash verification failed")
        expected_auth = _authorization_from_core(core)
        if manifest.get("delivery_authorization") != _authorization_payload(expected_auth) or member.delivery_authorization != expected_auth:
            raise ValueError("Building appearance delivery authorization mismatch")
        publication = manifest.get("publication_authorization", {})
        if publication.get("building_authorized") is not True or publication.get("export_authorized") is not True:
            raise ValueError("Building appearance tile delivery is unauthorized")
        manifest_lods = tuple(value.get("lod_id") for value in manifest.get("lods", ()))
        if tuple(member.authorized_lod_ids) != manifest_lods or any(value not in manifest_lods for value in member.authorized_lod_ids):
            raise ValueError("Building appearance authorized LOD set mismatch")
        expected_pointer = {
            "schema_version": 1,
            "tile_revision_id": member.tile_revision_id,
            "source_generation_id": member.source_generation_id,
            "appearance_revision_id": member.tile_appearance_revision_id,
            "manifest_sha256": published.manifest_sha256,
            "lod_ids": list(manifest_lods),
            "delivery_authorization": _authorization_payload(expected_auth),
            "publication_authorization": publication,
        }
        if dict(published.pointer) != expected_pointer:
            raise ValueError("Building appearance tile pointer verification failed")
        if member.rgbd_augmented_revision_id != core.get("rgbd_augmented_revision_id"):
            raise ValueError("Building appearance augmentation revision mismatch")

    def _load_generation(self, revision: str) -> PublishedBuildingAppearance:
        target = self.revisions_root / _sha256(revision, "Building appearance revision")
        self._validate_layout()
        _safe_directory(
            target,
            self._canonical_root,
            "Building appearance revision",
        )
        manifest_path = target / "manifest.json"
        _safe_file(
            manifest_path,
            self._canonical_root,
            "Building appearance manifest",
        )
        try:
            body = _read_safe_file(
                manifest_path,
                self._canonical_root,
                "Building appearance manifest",
            )
            manifest = json.loads(body)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Building appearance manifest is unreadable") from exc
        if body != _canonical_bytes(manifest) or "manifest_sha256" in manifest:
            raise ValueError("Building appearance manifest is not canonical")
        if manifest.get("building_appearance_revision_id") != revision:
            raise ValueError("Building appearance manifest identity is invalid")
        core = dict(manifest)
        core.pop("building_appearance_revision_id", None)
        if canonical_sha256(core) != revision:
            raise ValueError("Building appearance manifest hash is invalid")
        assembly = self.assembly_resolver(_sha256(core.get("assembly_id"), "assembly_id"))
        if assembly is None:
            raise ValueError("Building appearance assembly is unavailable")
        members = tuple(BuildingAppearanceMember.from_payload(value) for value in core.get("members", ()))
        normalized, expected_core = self._validated_core(assembly, members)
        if expected_core != core or normalized != members:
            raise ValueError("Building appearance manifest provenance is invalid")
        manifest_sha256 = hashlib.sha256(body).hexdigest()
        pointer = self._pointer(core, revision, manifest_sha256)
        return PublishedBuildingAppearance(
            revision,
            manifest,
            manifest_sha256,
            pointer,
            target,
            manifest_path,
        )

    def _pointer(self, core: Mapping[str, Any], revision: str, manifest_sha256: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "assembly_id": core["assembly_id"],
            "building_id": core["building_id"],
            "building_appearance_revision_id": revision,
            "manifest_sha256": manifest_sha256,
        }

    def _write_pointer(
        self,
        pointer: Mapping[str, Any],
        *,
        validator: Callable[[], PublishedBuildingAppearance],
    ) -> None:
        self._validate_layout()
        _safe_file(
            self.current_path,
            self._canonical_root,
            "Building appearance CURRENT",
        )
        previous = (
            _read_safe_file(
                self.current_path,
                self._canonical_root,
                "Building appearance CURRENT",
            )
            if _lexists(self.current_path)
            else None
        )
        temporary = self.current_path.with_name(f".{self.current_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._write_fsync(temporary, _canonical_bytes(pointer))
            validator()
            self._file_lock.assert_current()
            _replace_safe(temporary, self.current_path)
            _fsync_directory(self.root)
            try:
                validator()
            except BaseException:
                self._restore_pointer(previous)
                raise
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_pointer(self, previous: bytes | None) -> None:
        if previous is None:
            if _lexists(self.current_path):
                _safe_file(
                    self.current_path,
                    self._canonical_root,
                    "Building appearance CURRENT",
                )
                self.current_path.unlink()
                _fsync_directory(self.root)
            return
        temporary = self.current_path.with_name(f".{self.current_path.name}.{uuid.uuid4().hex}.restore")
        try:
            self._write_fsync(temporary, previous)
            _replace_safe(temporary, self.current_path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    @staticmethod
    def _write_fsync(path: Path, body: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        parent_descriptor = None
        try:
            if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
                parent_descriptor = _open_safe_directory_fd(
                    path.parent,
                    "Building appearance output directory",
                )
                descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            else:
                _safe_path_chain(path.parent, "Building appearance output directory")
                descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            if _lexists(path) and _is_reparse_point(path.lstat()):
                raise ValueError("Building appearance output path is an unsafe symlink or reparse point") from exc
            raise
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
        with os.fdopen(descriptor, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
            opened = os.fstat(output.fileno())
            current = path.lstat()
            if _is_reparse_point(current) or not _same_file_identity(opened, current):
                raise ValueError("Building appearance output path changed during write")

    def _cleanup_staging(self) -> None:
        _safe_directory(
            self.staging_root,
            self._canonical_root,
            "Building appearance staging root",
        )
        for path in self.staging_root.glob("*.partial"):
            self._remove_staging_path(path)

    def _remove_staging_path(self, path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if _is_reparse_point(info):
            raise ValueError("Building appearance staging contains an unsafe symlink or reparse point")
        if not _within(path.resolve(strict=True), self._canonical_root):
            raise ValueError("Building appearance staging path escapes the store root")
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(path)
        elif stat.S_ISREG(info.st_mode):
            path.unlink()
        else:
            raise ValueError("Building appearance staging contains an unsafe path")

    def _validate_layout(self) -> None:
        canonical = _safe_directory(
            self.root,
            None,
            "Building appearance store root",
        )
        if canonical != self._canonical_root:
            raise ValueError("Building appearance store root identity changed")
        _safe_directory(
            self.revisions_root,
            self._canonical_root,
            "Building appearance revisions root",
        )
        _safe_directory(
            self.staging_root,
            self._canonical_root,
            "Building appearance staging root",
        )
        _safe_file(
            self.lock_path,
            self._canonical_root,
            "Building appearance store lock",
        )


def _placement_revision(assembly_id: str, tile: Mapping[str, Any], member: Any) -> str:
    identity = {
        "assembly_id": assembly_id,
        "source_revision_id": tile["revision_id"],
        "component_id": member.component_id,
        "submap_id": member.submap_id,
        "generation_id": member.generation_id,
        "transform_building_from_submap": list(member.transform_building_from_submap),
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _authorization_from_core(core: Mapping[str, Any]) -> DeliveryAuthorization:
    evidence = core.get("delivery_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("tile appearance delivery evidence is missing")
    coverage = evidence.get("measured_surface_coverage_ppm")
    views = evidence.get("minimum_independent_view_count")
    if evidence.get("required_coverage_ppm") != _REQUIRED_COVERAGE_PPM or evidence.get("required_independent_view_count") != _REQUIRED_INDEPENDENT_VIEWS:
        raise ValueError("tile appearance delivery thresholds are not immutable")
    if isinstance(coverage, bool) or not isinstance(coverage, int) or not 0 <= coverage <= 1_000_000 or isinstance(views, bool) or not isinstance(views, int) or views < 0:
        raise ValueError("tile appearance delivery evidence is invalid")
    blockers = []
    if coverage < _REQUIRED_COVERAGE_PPM:
        blockers.append("MEASURED_SURFACE_COVERAGE_REQUIRED")
    if views < _REQUIRED_INDEPENDENT_VIEWS:
        blockers.append("INDEPENDENT_COLOR_VIEWS_REQUIRED")
    return DeliveryAuthorization(
        coverage,
        views,
        _REQUIRED_COVERAGE_PPM,
        _REQUIRED_INDEPENDENT_VIEWS,
        not blockers,
        tuple(blockers),
    )


def _authorization_payload(value: DeliveryAuthorization) -> dict[str, Any]:
    return {
        "measured_surface_coverage_ppm": value.measured_surface_coverage_ppm,
        "minimum_independent_view_count": value.minimum_independent_view_count,
        "required_coverage_ppm": value.required_coverage_ppm,
        "required_independent_view_count": value.required_independent_view_count,
        "delivery_authorized": value.delivery_authorized,
        "blockers": list(value.blockers),
    }


def _authorization_from_payload(payload: Mapping[str, Any]) -> DeliveryAuthorization:
    value = DeliveryAuthorization(
        payload["measured_surface_coverage_ppm"],
        payload["minimum_independent_view_count"],
        payload["required_coverage_ppm"],
        payload["required_independent_view_count"],
        payload["delivery_authorized"],
        tuple(payload["blockers"]),
    )
    if _authorization_payload(value) != dict(payload):
        raise ValueError("Building appearance delivery authorization is not canonical")
    return value


__all__ = [
    "BuildingAppearanceMember",
    "BuildingAppearanceStore",
    "DistanceLodInterval",
    "PublishedBuildingAppearance",
]
