"""Fail-closed evaluation for exact live textured-mapping evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_IDENTITY_FIELDS = (
    "geometry_revision_id",
    "tile_appearance_revision_id",
    "building_appearance_revision_id",
    "placement_revision_id",
    "source_generation_id",
)


@dataclass(frozen=True)
class FieldAcceptanceResult:
    accepted: bool
    state: str
    blockers: tuple[str, ...]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _at_most(value: Any, maximum: float, blockers: list[str], blocker: str, *, absolute: bool = False) -> None:
    number = _finite(value)
    if number is None or (abs(number) if absolute else number) > maximum:
        blockers.append(blocker)


def _at_least(value: Any, minimum: float, blockers: list[str], blocker: str) -> None:
    number = _finite(value)
    if number is None or number < minimum:
        blockers.append(blocker)


def _evaluate_identity(identity: Mapping[str, Any], blockers: list[str]) -> None:
    if identity.get("restart_performed") is not True:
        blockers.append("OWNED_DASHBOARD_RESTART_REQUIRED")
    before = _mapping(identity.get("pre_restart"))
    after = _mapping(identity.get("post_restart"))
    values = [before.get(field) for field in _IDENTITY_FIELDS]
    values.extend(after.get(field) for field in _IDENTITY_FIELDS)
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in values):
        blockers.append("REVISION_IDENTITY_INVALID")
    if any(before.get(field) != after.get(field) for field in _IDENTITY_FIELDS):
        blockers.append("RESTART_IDENTITY_CHANGED")


def _evaluate_export(name: str, evidence: Mapping[str, Any], blockers: list[str]) -> None:
    prefix = name.upper()
    if evidence.get("pygltflib_version") != "1.16.5":
        blockers.append(f"{prefix}_GLTF_PARSER_VERSION")
    if evidence.get("trimesh_version") != "4.7.1":
        blockers.append(f"{prefix}_TRIMESH_VERSION")
    if evidence.get("pygltflib_loaded") is not True:
        blockers.append(f"{prefix}_PYGLTFLIB_LOAD_FAILED")
    if evidence.get("trimesh_loaded") is not True:
        blockers.append(f"{prefix}_TRIMESH_LOAD_FAILED")
    if evidence.get("threejs_loaded") is not True:
        blockers.append(f"{prefix}_THREEJS_LOAD_FAILED")
    if evidence.get("metres_verified") is not True:
        blockers.append(f"{prefix}_METRE_UNITS_INVALID")
    if evidence.get("coordinate_transform_verified") is not True:
        blockers.append(f"{prefix}_COORDINATE_TRANSFORM_INVALID")
    body = evidence.get("provenance_bytes")
    expected = evidence.get("provenance_sha256")
    if not isinstance(body, bytes) or not isinstance(expected, str) \
            or _SHA256.fullmatch(expected) is None \
            or hashlib.sha256(body).hexdigest() != expected:
        blockers.append(f"{prefix}_PROVENANCE_DIGEST_MISMATCH")


def evaluate_textured_field_acceptance(report: Mapping[str, Any]) -> FieldAcceptanceResult:
    """Evaluate physical evidence without consulting mutable runtime state."""
    blockers: list[str] = []
    evidence = _mapping(report)
    if evidence.get("schema_version") != 1:
        blockers.append("EVIDENCE_SCHEMA_INVALID")
    if evidence.get("evidence_kind") != "PHYSICAL_GO2_CAPTURE":
        blockers.append("PHYSICAL_CAPTURE_REQUIRED")
    if evidence.get("synthetic") is not False:
        blockers.append("SYNTHETIC_EVIDENCE_REJECTED")
    if evidence.get("transport") != "TAILSCALE_DIRECT":
        blockers.append("TRANSPORT_NOT_TAILSCALE_DIRECT")

    surface = _mapping(evidence.get("surface"))
    _at_least(surface.get("coverage_ppm"), 800000, blockers, "SURFACE_COVERAGE_BELOW_THRESHOLD")
    _at_least(
        surface.get("independent_color_views"), 2,
        blockers, "INDEPENDENT_COLOR_VIEWS_BELOW_THRESHOLD",
    )
    error_values = [_finite(surface.get(name)) for name in (
        "ingest_errors", "pose_errors", "rgbd_errors", "surface_errors",
    )]
    if any(value is None or value != 0 for value in error_values):
        blockers.append("PIPELINE_ERRORS_PRESENT")

    physical = _mapping(evidence.get("physical_depth"))
    _at_least(physical.get("correspondence_count"), 100, blockers, "PHYSICAL_DEPTH_CORRESPONDENCE_COUNT")
    _at_most(physical.get("median_m"), 0.025, blockers, "PHYSICAL_DEPTH_MEDIAN")
    _at_most(physical.get("p95_m"), 0.060, blockers, "PHYSICAL_DEPTH_P95")
    _at_most(physical.get("signed_bias_m"), 0.015, blockers, "PHYSICAL_DEPTH_BIAS", absolute=True)
    _at_most(physical.get("occlusion_fraction"), 0.02, blockers, "PHYSICAL_DEPTH_OCCLUSION")
    _at_most(physical.get("free_space_fraction"), 0.01, blockers, "PHYSICAL_DEPTH_FREE_SPACE")

    assistance = _mapping(evidence.get("depth_anything"))
    _at_least(assistance.get("surviving_anchor_count"), 256, blockers, "DEPTH_ANYTHING_ANCHORS")
    _at_least(assistance.get("occupied_cells"), 24, blockers, "DEPTH_ANYTHING_GRID_COVERAGE")
    if assistance.get("total_cells") != 48:
        blockers.append("DEPTH_ANYTHING_GRID_DEFINITION")
    quadrants = assistance.get("quadrants")
    if not isinstance(quadrants, (list, tuple)) or len(quadrants) != 4 or not all(value is True for value in quadrants):
        blockers.append("DEPTH_ANYTHING_QUADRANTS")
    _at_most(assistance.get("held_out_median_m"), 0.025, blockers, "DEPTH_ANYTHING_MEDIAN")
    _at_most(assistance.get("held_out_p95_m"), 0.060, blockers, "DEPTH_ANYTHING_P95")
    _at_most(
        assistance.get("held_out_signed_bias_m"), 0.015,
        blockers, "DEPTH_ANYTHING_BIAS", absolute=True,
    )
    scale = _finite(assistance.get("scale"))
    if scale is None or scale <= 0:
        blockers.append("DEPTH_ANYTHING_SCALE")
    _at_most(
        assistance.get("relative_scale_stddev"), 0.02,
        blockers, "DEPTH_ANYTHING_SCALE_STABILITY",
    )
    _at_most(
        assistance.get("maximum_uncertainty_m"), 0.030,
        blockers, "DEPTH_ANYTHING_UNCERTAINTY",
    )

    _evaluate_identity(_mapping(evidence.get("identity")), blockers)
    exports = _mapping(evidence.get("exports"))
    _evaluate_export("local", _mapping(exports.get("local")), blockers)
    _evaluate_export("building", _mapping(exports.get("building")), blockers)

    unique = tuple(dict.fromkeys(blockers))
    accepted = not unique
    return FieldAcceptanceResult(
        accepted=accepted,
        state="LIVE_TEXTURED_MAPPING_ACCEPTED" if accepted else "LIVE_TEXTURED_MAPPING_REJECTED",
        blockers=unique,
    )


def canonical_acceptance_payload(report: Mapping[str, Any]) -> bytes:
    result = evaluate_textured_field_acceptance(report)
    payload = {
        "schema_version": 1,
        "accepted": result.accepted,
        "state": result.state,
        "blockers": list(result.blockers),
        "transport": report.get("transport"),
        "evidence_kind": report.get("evidence_kind"),
        "synthetic": report.get("synthetic"),
        "identity": report.get("identity"),
    }
    return (json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("utf-8")
