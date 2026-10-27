"""Atomic all-LOD transaction store for immutable tile appearance."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Callable, Mapping, Sequence
import uuid

from PIL import Image

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256

from .atlas_codec import parse_render_indices_v1, parse_render_uvs_v1, parse_render_vertices_v1
from .tile_identity import (
    ArtifactRef,
    DeliveryAuthorization,
    LodRecord,
    PrimitiveRecord,
    TileAppearanceRevision,
    build_lod_record,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


@dataclass(frozen=True)
class TileLodPayload:
    record: LodRecord
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if not isinstance(self.record, LodRecord):
            raise TypeError("record must be a LodRecord")
        payloads = {str(name): bytes(value) for name, value in self.artifacts.items()}
        expected = {value.name for value in self.record.artifacts}
        if set(payloads) != expected:
            raise ValueError("LOD payload artifacts must exactly match the LOD record")
        for reference in self.record.artifacts:
            payload = payloads[reference.name]
            if len(payload) != reference.bytes or hashlib.sha256(payload).hexdigest() != reference.sha256:
                raise ValueError(f"LOD artifact {reference.name} does not match its immutable reference")
        object.__setattr__(self, "artifacts", payloads)


@dataclass(frozen=True)
class PublishedTileAppearance:
    tile_revision_id: str
    appearance_revision_id: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    pointer: Mapping[str, Any]
    root_path: Path
    manifest_path: Path
    lod_paths: Mapping[str, Path] = field(compare=False)


class TileAppearanceStore:
    def __init__(
        self,
        root: str | Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root) / "appearance"
        self.root.mkdir(parents=True, exist_ok=True)
        self._fault_injector = fault_injector
        self._lock = threading.RLock()
        self._cleanup_staging()

    def publish(
        self,
        revision: TileAppearanceRevision,
        lods: Sequence[TileLodPayload],
        *,
        building_delivery_eligible: bool = True,
    ) -> PublishedTileAppearance:
        if not isinstance(revision, TileAppearanceRevision):
            raise TypeError("revision must be a TileAppearanceRevision")
        payloads = tuple(sorted(lods, key=lambda value: int(value.record.policy.get("order", -1))))
        if not payloads or any(not isinstance(value, TileLodPayload) for value in payloads):
            raise ValueError("appearance publication requires TileLodPayload records")
        if len({value.record.lod_id for value in payloads}) != len(payloads):
            raise ValueError("LOD IDs must be unique")
        tile = _sha256(revision.revision_core.get("tile_revision_id"), "tile revision")
        appearance = _sha256(revision.appearance_revision_id, "appearance revision")
        if canonical_sha256(revision.revision_core) != appearance:
            raise ValueError("appearance revision core does not match its identity")
        publication_authorization = _publication_authorization(
            revision.delivery_authorization, building_delivery_eligible
        )
        for payload in payloads:
            if payload.record.core.get("tile_revision_id") != tile:
                raise ValueError("LOD tile revision does not match appearance")
            if payload.record.core.get("appearance_revision_id") != appearance:
                raise ValueError("LOD appearance revision does not match appearance")
        manifest = _manifest(revision, payloads, publication_authorization)
        manifest_bytes = _canonical_bytes(manifest)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        pointer = _pointer(revision, payloads, manifest_sha, publication_authorization)
        expected_artifacts = {
            payload.record.lod_id: dict(payload.artifacts) for payload in payloads
        }

        with self._lock:
            tile_root = self._tile_root(tile)
            staging_root = tile_root / ".staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            target = self._generation_root(tile, appearance)
            if target.exists() or target.is_symlink():
                published = self._load_existing(
                    target, tile, appearance, manifest, expected_artifacts
                )
                self._write_pointer(tile_root, pointer)
                return published

            staging = staging_root / f"{uuid.uuid4().hex}.partial"
            staging.mkdir()
            renamed = False
            try:
                self._write_fsync(staging / "manifest.json", manifest_bytes)
                for payload in payloads:
                    lod_root = staging / "lods" / payload.record.lod_id
                    lod_root.mkdir(parents=True)
                    for name, body in payload.artifacts.items():
                        self._write_fsync(lod_root / name, body)
                    _fsync_directory(lod_root)
                _fsync_directory(staging / "lods")
                _fsync_directory(staging)
                self._load_path(staging, tile, appearance)
                self._fault("after_files_fsynced")
                os.replace(staging, target)
                renamed = True
                _fsync_directory(tile_root)
                self._fault("after_generation_rename")
                self._write_pointer(tile_root, pointer)
                self._fault("after_pointer_replace")
            except FileExistsError:
                return self._load_existing(target, tile, appearance, manifest, expected_artifacts)
            finally:
                if not renamed and staging.exists():
                    shutil.rmtree(staging)
            return self._load_path(target, tile, appearance)

    def resolve(self, tile_revision_id: str, appearance_revision_id: str) -> PublishedTileAppearance:
        tile = _sha256(tile_revision_id, "tile revision")
        appearance = _sha256(appearance_revision_id, "appearance revision")
        target = self._generation_root(tile, appearance)
        if not target.is_dir():
            raise FileNotFoundError("tile appearance generation is unavailable")
        return self._load_path(target, tile, appearance)

    def current(
        self,
        tile_revision_id: str,
        *,
        expected_source_generation_id: str | None = None,
    ) -> PublishedTileAppearance:
        tile = _sha256(tile_revision_id, "tile revision")
        pointer_path = self._tile_root(tile) / "CURRENT.json"
        try:
            pointer_bytes = pointer_path.read_bytes()
            pointer = json.loads(pointer_bytes)
        except FileNotFoundError:
            raise FileNotFoundError("tile appearance CURRENT pointer is unavailable") from None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("tile appearance CURRENT pointer is invalid") from exc
        if pointer_bytes != _canonical_bytes(pointer):
            raise ValueError("tile appearance CURRENT pointer is not canonical")
        if pointer.get("schema_version") != 1 or pointer.get("tile_revision_id") != tile:
            raise ValueError("tile appearance CURRENT pointer identity is invalid")
        appearance = _sha256(pointer.get("appearance_revision_id"), "current appearance revision")
        if expected_source_generation_id is not None:
            expected = _sha256(expected_source_generation_id, "expected source generation")
            if pointer.get("source_generation_id") != expected:
                raise ValueError("tile appearance CURRENT source generation is stale")
        published = self.resolve(tile, appearance)
        if dict(published.pointer) != pointer:
            raise ValueError("tile appearance CURRENT pointer does not match the complete generation")
        return published

    def _load_existing(
        self,
        target: Path,
        tile: str,
        appearance: str,
        expected_manifest: Mapping[str, Any],
        expected_artifacts: Mapping[str, Mapping[str, bytes]],
    ) -> PublishedTileAppearance:
        loaded = self._load_path(target, tile, appearance)
        if dict(loaded.manifest) != dict(expected_manifest):
            raise ValueError("existing appearance generation has conflicting manifest identity")
        for lod_id, artifacts in expected_artifacts.items():
            for name, payload in artifacts.items():
                if (target / "lods" / lod_id / name).read_bytes() != payload:
                    raise ValueError("existing appearance generation has conflicting artifact content")
        return loaded

    def _load_path(self, target: Path, tile: str, appearance: str) -> PublishedTileAppearance:
        manifest_path = target / "manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except FileNotFoundError:
            raise FileNotFoundError("tile appearance manifest is unavailable") from None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("tile appearance manifest is invalid") from exc
        if manifest_bytes != _canonical_bytes(manifest):
            raise ValueError("tile appearance manifest is not canonical")
        if manifest.get("schema_version") != 1 or "manifest_sha256" in manifest:
            raise ValueError("tile appearance manifest schema or self-hash is invalid")
        revision_core = manifest.get("revision_core")
        if not isinstance(revision_core, dict) or revision_core.get("tile_revision_id") != tile:
            raise ValueError("tile appearance revision core is invalid")
        if manifest.get("appearance_revision_id") != appearance or canonical_sha256(revision_core) != appearance:
            raise ValueError("tile appearance revision verification failed")
        authorization = _authorization_from_core(revision_core)
        if manifest.get("delivery_authorization") != _authorization_dict(authorization):
            raise ValueError("tile appearance delivery authorization verification failed")
        publication = manifest.get("publication_authorization")
        if not isinstance(publication, dict):
            raise ValueError("tile appearance publication authorization is missing")
        expected_publication = _publication_authorization(
            authorization, publication.get("building_delivery_eligible")
        )
        if publication != expected_publication:
            raise ValueError("tile appearance publication authorization verification failed")
        lod_manifests = manifest.get("lods")
        if not isinstance(lod_manifests, list) or not lod_manifests:
            raise ValueError("tile appearance must contain a complete LOD list")
        lod_paths: dict[str, Path] = {}
        loaded_lods: list[LodRecord] = []
        for item in lod_manifests:
            if not isinstance(item, dict) or set(item) != {"lod_id", "core"}:
                raise ValueError("tile appearance LOD manifest is invalid")
            lod_id = _sha256(item.get("lod_id"), "LOD ID")
            core = item.get("core")
            if not isinstance(core, dict) or canonical_sha256(core) != lod_id:
                raise ValueError("tile appearance LOD identity verification failed")
            record = _rebuild_lod(core)
            if record.lod_id != lod_id:
                raise ValueError("tile appearance LOD core is inconsistent")
            lod_root = target / "lods" / lod_id
            self._validate_lod_artifacts(lod_root, record, tile, appearance)
            lod_paths[lod_id] = lod_root
            loaded_lods.append(record)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        pointer = {
            "schema_version": 1,
            "tile_revision_id": tile,
            "source_generation_id": revision_core["source_generation_id"],
            "appearance_revision_id": appearance,
            "manifest_sha256": manifest_sha,
            "lod_ids": [value.lod_id for value in loaded_lods],
            "delivery_authorization": _authorization_dict(authorization),
            "publication_authorization": expected_publication,
        }
        return PublishedTileAppearance(
            tile,
            appearance,
            manifest,
            manifest_sha,
            pointer,
            target,
            manifest_path,
            lod_paths,
        )

    @staticmethod
    def _validate_lod_artifacts(
        lod_root: Path,
        record: LodRecord,
        tile: str,
        appearance: str,
    ) -> None:
        payloads: dict[str, bytes] = {}
        for reference in record.artifacts:
            try:
                payload = (lod_root / reference.name).read_bytes()
            except OSError as exc:
                raise ValueError(f"appearance artifact {reference.name} is unavailable") from exc
            if len(payload) != reference.bytes or hashlib.sha256(payload).hexdigest() != reference.sha256:
                raise ValueError(f"appearance artifact {reference.name} hash verification failed")
            payloads[reference.name] = payload
        required = {"render-vertices.bin", "render-indices.bin", "render-uvs.bin"}
        if not required.issubset(payloads):
            raise ValueError("appearance LOD is missing required render artifacts")
        core = record.core
        vertices = parse_render_vertices_v1(
            payloads["render-vertices.bin"],
            expected_tile_revision_id=tile,
            expected_appearance_revision_id=appearance,
            source_vertex_count=2**32,
        )
        if len(vertices.positions) != core["vertex_count"]:
            raise ValueError("render vertex count verification failed")
        indices = parse_render_indices_v1(
            payloads["render-indices.bin"],
            expected_tile_revision_id=tile,
            expected_appearance_revision_id=appearance,
            render_vertex_count=core["vertex_count"],
            source_triangle_count=2**32,
        )
        if len(indices.triangles) != core["triangle_count"]:
            raise ValueError("render triangle count verification failed")
        uvs = parse_render_uvs_v1(
            payloads["render-uvs.bin"],
            expected_tile_revision_id=tile,
            expected_appearance_revision_id=appearance,
            atlas_count=core["atlas_count"],
        )
        if len(uvs.uvs) != core["vertex_count"]:
            raise ValueError("render UV count verification failed")
        expected_atlases = {f"atlas-{index}.png" for index in range(core["atlas_count"])}
        if {name for name in payloads if name.startswith("atlas-")} != expected_atlases:
            raise ValueError("appearance atlas artifact set is incomplete")
        for name in expected_atlases:
            try:
                image = Image.open(BytesIO(payloads[name]))
                image.load()
            except Exception as exc:
                raise ValueError(f"appearance atlas {name} cannot be decoded") from exc
            if image.mode != "RGBA" or image.width > 4096 or image.height > 4096:
                raise ValueError("appearance atlas must be RGBA and at most 4096 pixels")
            alpha = set(image.getchannel("A").get_flattened_data())
            if not alpha.issubset({0, 255}):
                raise ValueError("appearance atlas alpha must be binary")

    def _write_pointer(self, tile_root: Path, pointer: Mapping[str, Any]) -> None:
        temporary = tile_root / f".CURRENT.{uuid.uuid4().hex}.partial"
        try:
            self._write_fsync(temporary, _canonical_bytes(pointer))
            os.replace(temporary, tile_root / "CURRENT.json")
            _fsync_directory(tile_root)
        finally:
            temporary.unlink(missing_ok=True)

    def _tile_root(self, tile: str) -> Path:
        target = (self.root / tile).resolve()
        _contained(target, self.root.resolve())
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _generation_root(self, tile: str, appearance: str) -> Path:
        tile_root = self._tile_root(tile)
        target = (tile_root / appearance).resolve()
        _contained(target, tile_root)
        return target

    def _cleanup_staging(self) -> None:
        for tile_root in self.root.iterdir():
            if not tile_root.is_dir() or tile_root.is_symlink():
                continue
            staging = tile_root / ".staging"
            if staging.is_dir() and not staging.is_symlink():
                for candidate in staging.iterdir():
                    if candidate.name.endswith(".partial") and candidate.is_dir():
                        shutil.rmtree(candidate)
            for candidate in tile_root.glob(".CURRENT.*.partial"):
                candidate.unlink(missing_ok=True)

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    @staticmethod
    def _write_fsync(path: Path, payload: bytes) -> None:
        with path.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())


def _manifest(
    revision: TileAppearanceRevision,
    lods: Sequence[TileLodPayload],
    publication_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "revision_core": dict(revision.revision_core),
        "appearance_revision_id": revision.appearance_revision_id,
        "delivery_authorization": _authorization_dict(revision.delivery_authorization),
        "publication_authorization": dict(publication_authorization),
        "lods": [{"lod_id": value.record.lod_id, "core": dict(value.record.core)} for value in lods],
    }


def _pointer(
    revision: TileAppearanceRevision,
    lods: Sequence[TileLodPayload],
    manifest_sha: str,
    publication_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tile_revision_id": revision.revision_core["tile_revision_id"],
        "source_generation_id": revision.revision_core["source_generation_id"],
        "appearance_revision_id": revision.appearance_revision_id,
        "manifest_sha256": manifest_sha,
        "lod_ids": [value.record.lod_id for value in lods],
        "delivery_authorization": _authorization_dict(revision.delivery_authorization),
        "publication_authorization": dict(publication_authorization),
    }


def _publication_authorization(
    delivery: DeliveryAuthorization, building_delivery_eligible: bool
) -> dict[str, Any]:
    if building_delivery_eligible is not True and building_delivery_eligible is not False:
        raise ValueError("building_delivery_eligible must be boolean")
    authorized = delivery.delivery_authorized and building_delivery_eligible
    blockers = list(delivery.blockers)
    if not building_delivery_eligible:
        blockers.append("TILE_SCOPE_NOT_BUILDING_ELIGIBLE")
    return {
        "building_delivery_eligible": building_delivery_eligible,
        "building_authorized": authorized,
        "export_authorized": authorized,
        "blockers": blockers,
    }


def _authorization_from_core(core: Mapping[str, Any]) -> DeliveryAuthorization:
    evidence = core.get("delivery_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("appearance delivery evidence is missing")
    coverage = evidence.get("measured_surface_coverage_ppm")
    views = evidence.get("minimum_independent_view_count")
    if evidence.get("required_coverage_ppm") != 800000 or evidence.get("required_independent_view_count") != 2:
        raise ValueError("appearance delivery thresholds are not immutable V1 values")
    if isinstance(coverage, bool) or not isinstance(coverage, int) or not 0 <= coverage <= 1_000_000:
        raise ValueError("appearance coverage evidence is invalid")
    if isinstance(views, bool) or not isinstance(views, int) or views < 0:
        raise ValueError("appearance view evidence is invalid")
    blockers = []
    if coverage < 800000:
        blockers.append("MEASURED_SURFACE_COVERAGE_REQUIRED")
    if views < 2:
        blockers.append("INDEPENDENT_COLOR_VIEWS_REQUIRED")
    return DeliveryAuthorization(coverage, views, 800000, 2, not blockers, tuple(blockers))


def _authorization_dict(value: DeliveryAuthorization) -> dict[str, Any]:
    return {
        "measured_surface_coverage_ppm": value.measured_surface_coverage_ppm,
        "minimum_independent_view_count": value.minimum_independent_view_count,
        "required_coverage_ppm": value.required_coverage_ppm,
        "required_independent_view_count": value.required_independent_view_count,
        "delivery_authorized": value.delivery_authorized,
        "blockers": list(value.blockers),
    }


def _rebuild_lod(core: Mapping[str, Any]) -> LodRecord:
    artifacts = tuple(ArtifactRef(value["name"], value["sha256"], value["bytes"]) for value in core.get("artifacts", ()))
    primitives = tuple(
        PrimitiveRecord(
            value["primitive_id"],
            value["material"],
            value["atlas_index"],
            value["index_offset"],
            value["index_count"],
            value["source_layer"],
            tuple(value["source_triangle_ids"]),
        )
        for value in core.get("primitives", ())
    )
    return build_lod_record(
        tile_revision_id=core.get("tile_revision_id"),
        appearance_revision_id=core.get("appearance_revision_id"),
        policy=core.get("policy"),
        vertex_count=core.get("vertex_count"),
        triangle_count=core.get("triangle_count"),
        atlas_count=core.get("atlas_count"),
        artifacts=artifacts,
        primitives=primitives,
        rgbd_augmented_revision_id=core.get("rgbd_augmented_revision_id"),
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _contained(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError("tile appearance path escapes its root") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
