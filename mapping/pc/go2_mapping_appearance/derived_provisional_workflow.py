"""Orchestrate one immutable, explicitly uncalibrated RGB-D surface preview.

This module is intentionally an offline inspection workflow.  It derives the
capture path from the current surface session, replays only the source artifacts
bound into that surface, and publishes a content-addressed color sidecar outside
the formal derived-surface and appearance stores.  It never updates a pointer,
mesh, source bag, calibration record, GLB, or final appearance store.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Any

from .derived_provisional_builder import (
    build_provisional_projection_evidence,
    load_active_derived_surface_binding,
    load_operator_forced_runtime_projection,
)
from mapping.pc.go2_mapping_dashboard.derived_surface import load_automatic_reconstruction_preview
from .derived_provisional_preview import publish_provisional_surface_preview


def build_active_provisional_surface_preview(
    *,
    status_path: str | Path,
    output_root: str | Path,
    capture_root: str | Path,
    operator_override_path: str | Path,
    preview_root: str | Path,
    max_pair_delta_ns: int = 100_000_000,
    work_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one preview while proving the formal current pointer was untouched.

    ``preview_root`` must be a sibling store, never the formal derived-surface
    root nor an ancestor/child of it.  The returned JSON-ready mapping keeps
    the non-authority flags explicit for CLIs and static publication code.
    """

    pointer_path = Path(status_path).expanduser().resolve(strict=False)
    formal_output_root = Path(output_root).expanduser().resolve(strict=False)
    source_capture_root = Path(capture_root).expanduser().resolve(strict=False)
    preview_output_root = Path(preview_root).expanduser().resolve(strict=False)
    override_path = Path(operator_override_path).expanduser().resolve(strict=False)
    temporary_root = None if work_root is None else Path(work_root).expanduser().resolve(strict=False)

    _require_separate_preview_root(preview_output_root, formal_output_root)
    pointer_before = _read_pointer_bytes(pointer_path)
    binding = load_active_derived_surface_binding(
        status_path=pointer_path,
        output_root=formal_output_root,
    )
    source_root = _session_capture_root(source_capture_root, binding.session_id)
    projection = load_operator_forced_runtime_projection(override_path)
    evidence = build_provisional_projection_evidence(
        binding=binding,
        source_root=source_root,
        projection=projection,
        max_pair_delta_ns=max_pair_delta_ns,
        work_root=temporary_root,
    )
    _require_pointer_unchanged(pointer_path, pointer_before)
    published = publish_provisional_surface_preview(preview_output_root, evidence)
    _require_pointer_unchanged(pointer_path, pointer_before)

    manifest = published.manifest
    return {
        "schema_version": 1,
        "state": manifest["state"],
        "preview_revision_id": published.preview_revision_id,
        "preview_root": str(preview_output_root),
        "preview_manifest_path": str(published.manifest_path),
        "vertex_color_path": str(published.vertex_color_path),
        "session_id": manifest["session_id"],
        "source_surface_sha256": manifest["source_surface_sha256"],
        "measured_surface_geometry_revision_id": manifest["measured_surface_geometry_revision_id"],
        "vertex_count": manifest["vertex_count"],
        "colored_vertex_count": manifest["colored_vertex_count"],
        "surface_coverage": manifest["surface_coverage"],
        "source_artifact_count": len(manifest["source_artifact_sha256"]),
        "color_source_frame_count": len(manifest["color_source_frame_sha256"]),
        "lidar_frame_count": manifest["lidar_frame_count"],
        "color_frame_count": manifest["color_frame_count"],
        "paired_frame_count": manifest["paired_frame_count"],
        "projected_point_count": manifest["projected_point_count"],
        "matched_surface_point_count": manifest["matched_surface_point_count"],
        "source_mode": manifest["source_mode"],
        "read_only": manifest["read_only"],
        "final_store_eligible": manifest["final_store_eligible"],
        "delivery_authorized": manifest["delivery_authorized"],
        "measurement_authority": manifest["measurement_authority"],
        "runtime_projection_authorized": manifest["runtime_projection_authorized"],
        "geometry_authority": manifest["geometry_authority"],
        "point_authority": manifest["point_authority"],
        "blockers": manifest["blockers"],
        "pointer_sha256": hashlib.sha256(pointer_before).hexdigest(),
        "pointer_bytes_unchanged": True,
    }


def _read_pointer_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError("active automatic reconstruction pointer is unreadable") from exc


def _require_pointer_unchanged(path: Path, expected: bytes) -> None:
    if _read_pointer_bytes(path) != expected:
        raise ValueError("active automatic reconstruction pointer changed during provisional preview replay")


def _session_capture_root(capture_root: Path, session_id: str) -> Path:
    root = capture_root.resolve(strict=False)
    candidate = (root / session_id).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("provisional preview capture path escapes the capture root") from exc
    if not candidate.is_dir():
        raise ValueError("provisional preview session capture is unavailable")
    return candidate


def _require_separate_preview_root(preview_root: Path, formal_output_root: Path) -> None:
    try:
        preview_root.relative_to(formal_output_root)
    except ValueError:
        pass
    else:
        raise ValueError("preview root must be separate from the formal derived surface store")
    try:
        formal_output_root.relative_to(preview_root)
    except ValueError:
        return
    raise ValueError("preview root must be separate from the formal derived surface store")


__all__ = ["build_active_provisional_surface_preview"]
