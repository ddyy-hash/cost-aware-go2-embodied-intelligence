"""Atomic content-addressed storage for immutable RGB-D augmentation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256

from .augmentation import AugmentedLayer
from .augmentation_identity import RgbdAugmentedIdentity, rgbd_augmented_identity


_REVISION_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_SOURCE_IDS_MAGIC = b"RGBDSID1"


@dataclass(frozen=True)
class PublishedAugmentation:
    source_generation_id: str
    rgbd_augmented_revision_id: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    root_path: Path
    manifest_path: Path
    positions_path: Path
    normals_path: Path
    indices_path: Path
    uncertainty_path: Path
    source_ids_path: Path
    layer: AugmentedLayer = field(compare=False)


class RgbdAugmentationStore:
    """Publishes complete revisions and resolves only exact immutable identities."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "rgbd-augmentation"
        self.root.mkdir(parents=True, exist_ok=True)
        self._cleanup_interrupted_publications()

    def publish(
        self,
        layer: AugmentedLayer,
        identity: RgbdAugmentedIdentity,
        *,
        expected_source_generation_id: str | None = None,
    ) -> PublishedAugmentation:
        if not isinstance(layer, AugmentedLayer):
            raise TypeError("layer must be an AugmentedLayer")
        if not isinstance(identity, RgbdAugmentedIdentity):
            raise TypeError("identity must be an RgbdAugmentedIdentity")
        self._validate_layer_identity(layer, identity)
        generation = self._revision(identity.core.get("source_generation_id"), "source generation")
        revision = self._revision(identity.revision_id, "augmentation revision")
        if expected_source_generation_id is not None:
            expected = self._revision(expected_source_generation_id, "expected source generation")
            if expected != generation:
                raise ValueError("augmentation source generation is stale")

        payloads = self._payloads(layer)
        manifest = self._manifest(layer, identity, payloads)
        generation_root = self._safe_generation(generation)
        generation_root.mkdir(parents=True, exist_ok=True)
        target = self._safe_revision(generation, revision)
        if target.exists() or target.is_symlink():
            return self._load_existing(target, generation, revision, manifest, payloads)

        staging = Path(tempfile.mkdtemp(prefix=".rgbd-", suffix=".partial", dir=generation_root))
        try:
            for name, payload in payloads.items():
                self._write_fsync(staging / name, payload)
            self._write_fsync(staging / "manifest.json", self._canonical_manifest_bytes(manifest))
            self._fsync_directory(staging)
            self._load_path(staging, generation, revision)
            os.replace(staging, target)
            self._fsync_directory(generation_root)
        except FileExistsError:
            self._remove_tree(staging)
            return self._load_existing(target, generation, revision, manifest, payloads)
        except Exception:
            self._remove_tree(staging)
            raise

        published = self.resolve(generation, revision)
        self._write_current(generation_root, published)
        return published

    def resolve(
        self,
        source_generation_id: str,
        rgbd_augmented_revision_id: str,
    ) -> PublishedAugmentation:
        generation = self._revision(source_generation_id, "source generation")
        revision = self._revision(rgbd_augmented_revision_id, "augmentation revision")
        target = self._safe_revision(generation, revision)
        if not target.is_dir():
            raise FileNotFoundError("published RGB-D augmentation revision is unavailable")
        return self._load_path(target, generation, revision)

    def _load_existing(
        self,
        target: Path,
        generation: str,
        revision: str,
        expected_manifest: Mapping[str, Any],
        expected_payloads: Mapping[str, bytes],
    ) -> PublishedAugmentation:
        loaded = self._load_path(target, generation, revision)
        if dict(loaded.manifest) != dict(expected_manifest):
            raise ValueError("existing augmentation revision has conflicting identity")
        for name, payload in expected_payloads.items():
            if (target / name).read_bytes() != payload:
                raise ValueError(f"existing augmentation revision has conflicting {name} content")
        return loaded

    def _load_path(self, target: Path, generation: str, revision: str) -> PublishedAugmentation:
        paths = {
            "manifest": target / "manifest.json",
            "positions.bin": target / "positions.bin",
            "normals.bin": target / "normals.bin",
            "indices.bin": target / "indices.bin",
            "uncertainty.bin": target / "uncertainty.bin",
            "source-ids.bin": target / "source-ids.bin",
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError("published RGB-D augmentation revision is incomplete")
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("augmentation manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("augmentation manifest must be an object")
        if manifest.get("rgbd_augmented_revision_id") != revision:
            raise ValueError("augmentation manifest revision verification failed")
        if manifest.get("source_generation_id") != generation:
            raise ValueError("augmentation manifest source generation verification failed")
        recorded_manifest_sha = manifest.get("manifest_sha256")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_sha256", None)
        if recorded_manifest_sha != canonical_sha256(unsigned_manifest):
            raise ValueError("augmentation manifest verification failed")

        payloads: dict[str, bytes] = {}
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("augmentation artifact manifest is invalid")
        for name in ("positions.bin", "normals.bin", "indices.bin", "uncertainty.bin", "source-ids.bin"):
            try:
                payload = paths[name].read_bytes()
            except OSError as exc:
                raise ValueError(f"augmentation {name} is unavailable") from exc
            record = artifacts.get(name)
            if not isinstance(record, dict) or record.get("bytes") != len(payload):
                raise ValueError(f"augmentation {name} size verification failed")
            if record.get("sha256") != hashlib.sha256(payload).hexdigest():
                label = name.removesuffix(".bin")
                raise ValueError(f"augmentation {label} verification failed")
            payloads[name] = payload

        vertex_count = self._count(manifest.get("vertex_count"), "vertex_count")
        triangle_count = self._count(manifest.get("triangle_count"), "triangle_count")
        positions = self._decode_array(payloads["positions.bin"], "<f4", (vertex_count, 3), "positions")
        normals = self._decode_array(payloads["normals.bin"], "<f4", (vertex_count, 3), "normals")
        triangles = self._decode_array(payloads["indices.bin"], "<u4", (triangle_count, 3), "indices")
        uncertainty = self._decode_array(
            payloads["uncertainty.bin"], "<f4", (vertex_count,), "uncertainty"
        )
        vertex_ids, triangle_ids = _decode_source_ids(payloads["source-ids.bin"])
        if len(vertex_ids) != vertex_count or len(triangle_ids) != triangle_count:
            raise ValueError("augmentation source ID count verification failed")

        identity = rgbd_augmented_identity(
            source_generation_id=generation,
            source_revision_id=manifest.get("source_revision_id"),
            calibration_sha256=manifest.get("calibration_sha256"),
            physical_depth_validation_revision=manifest.get("physical_depth_validation_revision"),
            model_revision_id=manifest.get("model_revision_id"),
            assistance_config_sha256=manifest.get("assistance_config_sha256"),
            source_frame_sha256=manifest.get("source_frame_sha256", ()),
            positions=positions,
            normals=normals,
            triangles=triangles,
            uncertainty_m=uncertainty,
            source_vertex_ids=vertex_ids,
            source_triangle_ids=triangle_ids,
            algorithm_version=manifest.get("algorithm_version"),
            uncertainty_policy=manifest.get("uncertainty_policy"),
        )
        if identity.revision_id != revision:
            raise ValueError("augmentation identity verification failed")
        if identity.core != manifest.get("identity_core"):
            raise ValueError("augmentation identity core verification failed")
        if manifest.get("name") != "RGBD_AUGMENTED" or manifest.get("geometry_authority") is not False:
            raise ValueError("augmentation authority label verification failed")
        layer = AugmentedLayer(
            name="RGBD_AUGMENTED",
            positions=positions,
            normals=normals,
            triangles=triangles,
            uncertainty_m=uncertainty,
            source_vertex_ids=vertex_ids,
            source_triangle_ids=triangle_ids,
            geometry_authority=False,
            source=manifest.get("source"),
        )
        return PublishedAugmentation(
            source_generation_id=generation,
            rgbd_augmented_revision_id=revision,
            manifest=manifest,
            manifest_sha256=recorded_manifest_sha,
            root_path=target,
            manifest_path=paths["manifest"],
            positions_path=paths["positions.bin"],
            normals_path=paths["normals.bin"],
            indices_path=paths["indices.bin"],
            uncertainty_path=paths["uncertainty.bin"],
            source_ids_path=paths["source-ids.bin"],
            layer=layer,
        )

    @staticmethod
    def _validate_layer_identity(layer: AugmentedLayer, identity: RgbdAugmentedIdentity) -> None:
        if layer.name != "RGBD_AUGMENTED" or layer.geometry_authority is not False:
            raise ValueError("RGB-D augmentation must remain separately labeled and non-authoritative")
        core = identity.core
        rebuilt = rgbd_augmented_identity(
            source_generation_id=core.get("source_generation_id"),
            source_revision_id=core.get("source_revision_id"),
            calibration_sha256=core.get("calibration_sha256"),
            physical_depth_validation_revision=core.get("physical_depth_validation_revision"),
            model_revision_id=core.get("model_revision_id"),
            assistance_config_sha256=core.get("assistance_config_sha256"),
            source_frame_sha256=core.get("source_frame_sha256", ()),
            positions=layer.positions,
            normals=layer.normals,
            triangles=layer.triangles,
            uncertainty_m=layer.uncertainty_m,
            source_vertex_ids=layer.source_vertex_ids,
            source_triangle_ids=layer.source_triangle_ids,
            algorithm_version=core.get("algorithm_version"),
            uncertainty_policy=core.get("uncertainty_policy"),
        )
        if rebuilt != identity:
            raise ValueError("augmentation layer content does not match its identity")

    @staticmethod
    def _payloads(layer: AugmentedLayer) -> dict[str, bytes]:
        return {
            "positions.bin": np.ascontiguousarray(layer.positions, dtype="<f4").tobytes(order="C"),
            "normals.bin": np.ascontiguousarray(layer.normals, dtype="<f4").tobytes(order="C"),
            "indices.bin": np.ascontiguousarray(layer.triangles, dtype="<u4").tobytes(order="C"),
            "uncertainty.bin": np.ascontiguousarray(layer.uncertainty_m, dtype="<f4").tobytes(order="C"),
            "source-ids.bin": _encode_source_ids(layer.source_vertex_ids, layer.source_triangle_ids),
        }

    @staticmethod
    def _manifest(
        layer: AugmentedLayer,
        identity: RgbdAugmentedIdentity,
        payloads: Mapping[str, bytes],
    ) -> dict[str, Any]:
        core = dict(identity.core)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "name": "RGBD_AUGMENTED",
            "geometry_authority": False,
            "source": layer.source,
            "rgbd_augmented_revision_id": identity.revision_id,
            **core,
            "identity_core": core,
            "artifacts": {
                name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                for name, payload in sorted(payloads.items())
            },
        }
        manifest_sha256 = canonical_sha256(manifest)
        manifest["manifest_sha256"] = manifest_sha256
        return manifest

    @staticmethod
    def _canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
        return json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    def _write_current(self, generation_root: Path, published: PublishedAugmentation) -> None:
        payload = self._canonical_manifest_bytes(
            {
                "schema_version": 1,
                "diagnostic_only": True,
                "source_generation_id": published.source_generation_id,
                "rgbd_augmented_revision_id": published.rgbd_augmented_revision_id,
                "manifest_sha256": published.manifest_sha256,
            }
        )
        temporary = generation_root / ".CURRENT.json.partial"
        try:
            self._write_fsync(temporary, payload)
            os.replace(temporary, generation_root / "CURRENT.json")
            self._fsync_directory(generation_root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _safe_generation(self, generation: str) -> Path:
        target = (self.root / generation).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("augmentation generation path escapes the store root") from exc
        return target

    def _safe_revision(self, generation: str, revision: str) -> Path:
        generation_root = self._safe_generation(generation)
        target = (generation_root / revision).resolve()
        try:
            target.relative_to(generation_root)
        except ValueError as exc:
            raise ValueError("augmentation revision path escapes its source generation") from exc
        return target

    @staticmethod
    def _revision(value: Any, name: str) -> str:
        if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
        return value

    @staticmethod
    def _count(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"augmentation {name} is invalid")
        return value

    @staticmethod
    def _decode_array(payload: bytes, dtype: str, shape: tuple[int, ...], name: str) -> np.ndarray:
        expected_size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(payload) != expected_size:
            raise ValueError(f"augmentation {name} byte length verification failed")
        result = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        result.setflags(write=False)
        return result

    def _cleanup_interrupted_publications(self) -> None:
        for generation_root in self.root.iterdir():
            if not generation_root.is_dir() or generation_root.is_symlink():
                continue
            for candidate in generation_root.iterdir():
                if candidate.name.startswith(".rgbd-") and candidate.name.endswith(".partial"):
                    self._remove_tree(candidate)
                elif candidate.name == ".CURRENT.json.partial" and candidate.is_file():
                    candidate.unlink()

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
        if path.exists() and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def _encode_source_ids(vertex_ids: Sequence[str], triangle_ids: Sequence[str]) -> bytes:
    payload = bytearray(_SOURCE_IDS_MAGIC)
    for values in (vertex_ids, triangle_ids):
        payload.extend(struct.pack("<I", len(values)))
        for value in values:
            encoded = value.encode("utf-8")
            payload.extend(struct.pack("<I", len(encoded)))
            payload.extend(encoded)
    return bytes(payload)


def _decode_source_ids(payload: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not payload.startswith(_SOURCE_IDS_MAGIC):
        raise ValueError("augmentation source ID magic verification failed")
    offset = len(_SOURCE_IDS_MAGIC)
    groups: list[tuple[str, ...]] = []
    try:
        for _ in range(2):
            count = struct.unpack_from("<I", payload, offset)[0]
            offset += 4
            values: list[str] = []
            for _ in range(count):
                length = struct.unpack_from("<I", payload, offset)[0]
                offset += 4
                end = offset + length
                values.append(payload[offset:end].decode("utf-8"))
                if end > len(payload):
                    raise ValueError("truncated source ID")
                offset = end
            groups.append(tuple(values))
    except (struct.error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("augmentation source ID payload is invalid") from exc
    if offset != len(payload):
        raise ValueError("augmentation source ID payload has trailing bytes")
    return groups[0], groups[1]
