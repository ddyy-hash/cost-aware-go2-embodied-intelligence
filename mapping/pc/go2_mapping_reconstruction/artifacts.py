"""Atomic point/mesh artifact export with explicit layer authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from mapping.pc.go2_mapping_fusion.surface import MeasuredSurface


@dataclass(frozen=True)
class PublishedLayer:
    ply_path: Path
    metadata_path: Path
    ply_sha256: str
    metadata: dict[str, Any]


class LayerArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        surface: MeasuredSurface,
        identity: object,
        *,
        annotations: Mapping[str, Any] | None = None,
    ) -> PublishedLayer:
        identity_payload = self._identity_payload(identity)
        cache_key = str(identity_payload.get("cache_key", ""))
        if len(cache_key) != 64:
            raise ValueError("derived identity must contain a 64-character cache_key")
        stem = f"{surface.source_layer.lower()}-{cache_key[:16]}"
        ply_path = self.root / f"{stem}.ply"
        metadata_path = self.root / f"{stem}.json"
        self._atomic_write(ply_path, self._ply_bytes(surface))
        ply_sha256 = hashlib.sha256(ply_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": 1,
            "source_layer": surface.source_layer,
            "geometry_authority": surface.geometry_authority,
            "point_authority": surface.point_authority,
            "vertex_count": surface.vertex_count,
            "triangle_count": surface.triangle_count,
            "ply_sha256": ply_sha256,
            "derived_identity": identity_payload,
            "derived_cache_key": cache_key,
        }
        protected = set(metadata)
        supplied = {} if annotations is None else dict(annotations)
        overlap = protected.intersection(supplied)
        if overlap:
            raise ValueError(f"artifact annotations cannot replace protected metadata: {sorted(overlap)}")
        metadata.update(supplied)
        self._atomic_write(metadata_path, json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return PublishedLayer(ply_path, metadata_path, ply_sha256, metadata)

    def list_verified(self) -> list[dict[str, Any]]:
        records = []
        for metadata_path in sorted(self.root.glob("*.json")):
            try:
                verified_metadata = self.resolve(metadata_path.name)
                metadata = json.loads(verified_metadata.read_text(encoding="utf-8"))
                ply_path = self.resolve(f"{metadata_path.stem}.ply")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            records.append({
                **metadata,
                "metadata_name": metadata_path.name,
                "ply_name": ply_path.name,
            })
        return records

    def resolve(self, name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise FileNotFoundError("artifact name is unsafe")
        path = (self.root / name).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise FileNotFoundError("artifact path escapes the publication root") from exc
        if path.suffix not in {".ply", ".json"} or not path.is_file():
            raise FileNotFoundError("published reconstruction artifact is unavailable")
        metadata_path = path if path.suffix == ".json" else path.with_suffix(".json")
        ply_path = path if path.suffix == ".ply" else path.with_suffix(".ply")
        if not metadata_path.is_file() or not ply_path.is_file():
            raise FileNotFoundError("published reconstruction pair is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("ply_sha256")
        actual = hashlib.sha256(ply_path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("published reconstruction PLY checksum changed")
        return path

    @staticmethod
    def _identity_payload(identity: object) -> dict[str, Any]:
        if is_dataclass(identity):
            return asdict(identity)
        if isinstance(identity, Mapping):
            return dict(identity)
        raise TypeError("derived identity must be a dataclass or mapping")

    @staticmethod
    def _ply_bytes(surface: MeasuredSurface) -> bytes:
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {surface.vertex_count}",
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            f"element face {surface.triangle_count}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        lines.extend(
            " ".join(format(float(value), ".17g") for value in (*point, *normal))
            for point, normal in zip(surface.positions, surface.normals)
        )
        lines.extend(f"3 {int(face[0])} {int(face[1])} {int(face[2])}" for face in surface.triangles)
        return ("\n".join(lines) + "\n").encode("ascii")

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
