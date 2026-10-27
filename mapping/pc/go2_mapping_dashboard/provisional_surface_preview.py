"""Publish a verified non-deliverable RGB-D color sidecar under Dashboard static files.

The normal Dashboard appearance APIs deliberately remain unavailable for an
immutable observed LiDAR surface.  This publisher is a separate static-only
inspection route.  It mirrors a preview only after binding it to the current
formal read-only surface by surface SHA, geometry revision, and vertex count.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

from mapping.pc.go2_mapping_appearance.derived_provisional_preview import load_provisional_surface_preview
from mapping.pc.go2_mapping_quality.reconstruction_config import is_sha256

from .derived_surface import load_automatic_reconstruction_preview


STATIC_PROVISIONAL_SURFACE_PREVIEW_PROTOCOL = "go2-static-provisional-rgbd-surface-preview-v1"
_DIRECTORY = "provisional-surface-preview"


def publish_current_provisional_surface_preview(
    *,
    status_path: str | Path,
    output_root: str | Path,
    preview_manifest_path: str | Path,
    static_root: str | Path,
) -> dict[str, Any]:
    """Atomically publish only one provenance-verified preview to ``/static``."""

    preview, pointer = load_automatic_reconstruction_preview(status_path, output_root=output_root)
    status = preview.status
    source_surface = _read_surface(preview.surface_path)
    generation_manifest = _read_surface(preview.manifest_path)
    geometry_revision = _lower_sha256(
        source_surface.get("measured_surface_geometry_revision_id"),
        "measured surface geometry revision",
    )
    vertex_count = status.get("vertex_count")
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, int) or vertex_count <= 0:
        raise ValueError("current measured surface vertex count is invalid")
    if not isinstance(source_surface.get("source_vertex_ids"), list) or len(source_surface["source_vertex_ids"]) != vertex_count:
        raise ValueError("current measured surface vertex identity is invalid")
    surface_sha256 = _lower_sha256(status.get("surface_sha256"), "current measured surface SHA-256")
    pointer_before = _read_bytes(Path(status_path).expanduser().resolve(strict=False), "active automatic reconstruction pointer")
    published_preview = load_provisional_surface_preview(
        preview_manifest_path,
        expected_surface_sha256=surface_sha256,
        expected_geometry_revision_id=geometry_revision,
    )
    manifest = published_preview.manifest
    if manifest.get("vertex_count") != vertex_count:
        raise ValueError("provisional preview vertex count does not match current measured surface")
    if manifest.get("session_id") != pointer.get("session_id"):
        raise ValueError("provisional preview session does not match current measured surface")
    if manifest.get("source_vertex_ids_sha256") != _source_vertex_ids_sha256(source_surface["source_vertex_ids"]):
        raise ValueError("provisional preview vertex ordering does not match current measured surface")
    if _artifact_hashes(manifest.get("source_artifact_sha256"), "provisional preview") != _artifact_hashes(
        generation_manifest.get("source_artifacts"),
        "current measured surface",
    ):
        raise ValueError("provisional preview source artifact inventory does not match current measured surface")

    sidecar_bytes = _read_bytes(published_preview.vertex_color_path, "provisional preview vertex-color payload")
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    preview_manifest_bytes = _read_bytes(published_preview.manifest_path, "provisional preview manifest")
    preview_manifest_sha256 = hashlib.sha256(preview_manifest_bytes).hexdigest()
    static = Path(static_root).expanduser().resolve(strict=False)
    static.mkdir(parents=True, exist_ok=True)
    relative_root = Path(
        _DIRECTORY,
        manifest["session_id"],
        surface_sha256,
        geometry_revision,
        published_preview.preview_revision_id,
    )
    vertex_relative = relative_root / "vertex-color.bin"
    manifest_relative = relative_root / "manifest.json"
    vertex_path = _within(static, vertex_relative)
    preview_copy_path = _within(static, manifest_relative)
    _atomic_write(vertex_path, sidecar_bytes)
    _atomic_write(preview_copy_path, preview_manifest_bytes)
    if hashlib.sha256(_read_bytes(vertex_path, "published provisional preview vertex-color payload")).hexdigest() != sidecar_sha256:
        raise ValueError("published provisional preview vertex-color checksum changed")
    if hashlib.sha256(_read_bytes(preview_copy_path, "published provisional preview manifest")).hexdigest() != preview_manifest_sha256:
        raise ValueError("published provisional preview manifest checksum changed")
    _require_pointer_unchanged(Path(status_path).expanduser().resolve(strict=False), pointer_before)

    current = {
        "schema_version": 1,
        "protocol": STATIC_PROVISIONAL_SURFACE_PREVIEW_PROTOCOL,
        "state": manifest["state"],
        "available": True,
        "source_mode": "DERIVED_READ_ONLY",
        "read_only": True,
        "final_store_eligible": False,
        "delivery_authorized": False,
        "measurement_authority": False,
        "runtime_projection_authorized": True,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "session_id": manifest["session_id"],
        "source_surface_sha256": surface_sha256,
        "measured_surface_geometry_revision_id": geometry_revision,
        "vertex_count": vertex_count,
        "colored_vertex_count": manifest["colored_vertex_count"],
        "surface_coverage": manifest["surface_coverage"],
        "preview_revision_id": published_preview.preview_revision_id,
        "preview_manifest_sha256": preview_manifest_sha256,
        "vertex_color_sha256": sidecar_sha256,
        "vertex_color_bytes": len(sidecar_bytes),
        "vertex_color_format": manifest["vertex_color"]["format"],
        "preview_manifest_relative_url": "/static/" + manifest_relative.as_posix(),
        "vertex_color_relative_url": "/static/" + vertex_relative.as_posix(),
        "blockers": list(manifest["blockers"]),
    }
    _atomic_write(
        _within(static, Path(_DIRECTORY, "current.json")),
        _canonical_bytes(current),
    )
    _require_pointer_unchanged(Path(status_path).expanduser().resolve(strict=False), pointer_before)
    return current


def _read_surface(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("current measured surface is unreadable") from exc
    if not isinstance(parsed, dict):
        raise ValueError("current measured surface is invalid")
    return parsed


def _source_vertex_ids_sha256(source_vertex_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps({"source_vertex_ids": source_vertex_ids}, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _artifact_hashes(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} source artifact inventory is invalid")
    if value and all(isinstance(item, str) for item in value):
        hashes = tuple(_lower_sha256(item, f"{label} source artifact SHA-256") for item in value)
    elif value and all(isinstance(item, dict) for item in value):
        hashes = tuple(
            _lower_sha256(item.get("sha256"), f"{label} source artifact SHA-256")
            for item in value
        )
    else:
        raise ValueError(f"{label} source artifact inventory is invalid")
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"{label} source artifact inventory contains duplicates")
    return tuple(sorted(hashes))


def _lower_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _within(root: Path, relative: Path) -> Path:
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("provisional preview static target escapes static root") from exc
    return target


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc


def _require_pointer_unchanged(path: Path, expected: bytes) -> None:
    if _read_bytes(path, "active automatic reconstruction pointer") != expected:
        raise ValueError("active automatic reconstruction pointer changed during provisional preview publication")


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "STATIC_PROVISIONAL_SURFACE_PREVIEW_PROTOCOL",
    "publish_current_provisional_surface_preview",
]
