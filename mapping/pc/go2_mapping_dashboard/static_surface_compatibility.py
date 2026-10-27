"""Static compatibility publication for one verified final LiDAR surface.

The normal Dashboard owns the measured-surface API.  This module is deliberately
smaller: it only mirrors an already verified, final automatic-reconstruction
generation into the Dashboard's shared static root.  It neither starts capture,
performs reconstruction, changes calibration, nor creates geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

from .derived_surface import (
    DERIVED_READ_ONLY,
    PREVIEW_UNCALIBRATED,
    load_automatic_reconstruction_preview,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import is_sha256


STATIC_COMPATIBILITY_PROTOCOL = "go2-static-verified-derived-surface-v1"
_STATIC_COMPATIBILITY_DIRECTORY = "compatibility"
_STATIC_SURFACE_DIRECTORY = "measured-surfaces"


def _lower_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _within(root: Path, relative: Path) -> Path:
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("static compatibility target escapes the static root") from exc
    return target


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_verified_surface(
    *,
    status_path: str | Path,
    output_root: str | Path,
    static_root: str | Path,
) -> dict[str, Any]:
    """Atomically expose one final observed-only generation below ``/static``.

    The automatic pointer loader revalidates the complete immutable generation,
    including its source inventory, mesh artifact hashes, geometry identity, and
    observed-only flags.  This publisher verifies the copied JSON again before
    replacing either static file.
    """

    preview, pointer = load_automatic_reconstruction_preview(
        status_path,
        output_root=output_root,
    )
    if (
        pointer.get("state") != "FINAL_SURFACE_AVAILABLE"
        or pointer.get("finalized") is not True
        or pointer.get("final_manifest_state") != "VERIFIED"
    ):
        raise ValueError("static compatibility requires a final verified automatic surface")

    status = preview.status
    cache_key = _lower_sha256(status.get("derived_cache_key"), "derived cache key")
    surface_sha256 = _lower_sha256(status.get("surface_sha256"), "surface checksum")
    surface_bytes = preview.surface_path.read_bytes()
    if hashlib.sha256(surface_bytes).hexdigest() != surface_sha256:
        raise ValueError("verified surface bytes changed before static compatibility publication")

    static_root_path = Path(static_root).expanduser().resolve(strict=False)
    static_root_path.mkdir(parents=True, exist_ok=True)
    surface_relative_path = Path(
        _STATIC_COMPATIBILITY_DIRECTORY,
        _STATIC_SURFACE_DIRECTORY,
        cache_key,
        f"{surface_sha256}.json",
    )
    surface_path = _within(static_root_path, surface_relative_path)
    _atomic_write(surface_path, surface_bytes)
    if hashlib.sha256(surface_path.read_bytes()).hexdigest() != surface_sha256:
        raise ValueError("static compatibility surface checksum did not survive publication")

    session_id = pointer.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("static compatibility pointer session is invalid")
    manifest = {
        "schema_version": 1,
        "compatibility_protocol": STATIC_COMPATIBILITY_PROTOCOL,
        "available": True,
        "blocker": None,
        "session_id": session_id,
        "state": pointer["state"],
        "final_manifest_state": pointer["final_manifest_state"],
        "source_mode": DERIVED_READ_ONLY,
        "read_only": True,
        "quality": PREVIEW_UNCALIBRATED,
        "point_authority": "LIDAR_MEASURED",
        "geometry_authority": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "volume_filled": False,
        "watertight": False,
        "sensor_frame_transform_id": status["sensor_frame_transform_id"],
        "derived_cache_key": cache_key,
        "surface_sha256": surface_sha256,
        "surface_bytes": len(surface_bytes),
        "ply_sha256": status["ply_sha256"],
        "metadata_sha256": status["metadata_sha256"],
        "vertex_count": status["vertex_count"],
        "triangle_count": status["triangle_count"],
        "surface_relative_url": "/static/" + surface_relative_path.as_posix(),
    }
    manifest_path = _within(
        static_root_path,
        Path(_STATIC_COMPATIBILITY_DIRECTORY, "measured-surface-current.json"),
    )
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return manifest


__all__ = ["STATIC_COMPATIBILITY_PROTOCOL", "publish_verified_surface"]
