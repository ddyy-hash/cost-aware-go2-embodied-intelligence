"""Verified read-only access to one content-addressed offline surface generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from mapping.pc.go2_mapping_appearance.identity import measured_surface_identity
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.sensor_frame import get_sensor_frame_transform
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, is_sha256


PREVIEW_UNCALIBRATED = "PREVIEW_UNCALIBRATED"
DERIVED_READ_ONLY = "DERIVED_READ_ONLY"
OFFLINE_DERIVATION_ALGORITHM = "go2-offline-observed-surface-v1"
_SESSION_ID_RE = re.compile(r"^segment-[0-9]{8}T[0-9]{6}Z$", re.ASCII)
_OBSERVED_LIDAR_LAYERS = {
    ("LIDAR_MEASURED", "LIDAR_MEASURED"),
    ("LIO_REGISTERED_LIDAR_OBSERVED", "LIO_REGISTERED_LIDAR_OBSERVED"),
}
_RAW_LIDAR_WITH_RECORDED_LIO = "RAW_LIDAR_WITH_RECORDED_LIO"
_RECORDED_LIO_REGISTERED_CLOUD = "RECORDED_LIO_REGISTERED_CLOUD"
_RECORDED_LIO_SESSION_ODOM_CLOUD = "RECORDED_LIO_SESSION_ODOM_CLOUD"
_SESSION_ODOM_LOCAL = "SESSION_ODOM_LOCAL"
_SOURCE_MODE_OBSERVATION_LAYERS = {
    _RAW_LIDAR_WITH_RECORDED_LIO: ("LIDAR_MEASURED", "LIDAR_MEASURED"),
    _RECORDED_LIO_REGISTERED_CLOUD: (
        "LIO_REGISTERED_LIDAR_OBSERVED",
        "LIO_REGISTERED_LIDAR_OBSERVED",
    ),
    _RECORDED_LIO_SESSION_ODOM_CLOUD: (
        "LIO_REGISTERED_LIDAR_OBSERVED",
        "LIO_REGISTERED_LIDAR_OBSERVED",
    ),
}


@dataclass(frozen=True)
class DerivedSurfacePreview:
    surface_path: Path
    manifest_path: Path
    ply_path: Path
    metadata_path: Path
    status: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"derived {label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"derived {label} must contain a JSON object")
    return value, payload


def _lower_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
        raise ValueError(f"derived {label} must be a lowercase SHA-256 digest")
    return value


def _observation_provenance(
    payload: Mapping[str, Any],
    *,
    allow_legacy_measured: bool,
) -> tuple[str, str, str, bool]:
    source_layer = payload.get("source_layer")
    point_authority = payload.get("point_authority")
    if allow_legacy_measured and source_layer is None and point_authority == "LIDAR_MEASURED":
        source_layer = "LIDAR_MEASURED"
    observation_layer = (source_layer, point_authority)
    if observation_layer not in _OBSERVED_LIDAR_LAYERS:
        raise ValueError("must contain an allowed LiDAR observation layer")

    source_mode = payload.get("mapping_source_mode")
    if source_mode is None and allow_legacy_measured and observation_layer == (
        "LIDAR_MEASURED",
        "LIDAR_MEASURED",
    ):
        source_mode = _RAW_LIDAR_WITH_RECORDED_LIO
    if source_mode not in _SOURCE_MODE_OBSERVATION_LAYERS:
        raise ValueError("mapping source mode is invalid")
    if _SOURCE_MODE_OBSERVATION_LAYERS[source_mode] != observation_layer:
        raise ValueError("mapping source mode does not match its LiDAR observation layer")

    transform_applied = payload.get("sensor_frame_transform_applied")
    if transform_applied is None and allow_legacy_measured and source_mode == _RAW_LIDAR_WITH_RECORDED_LIO:
        transform_applied = True
    if not isinstance(transform_applied, bool):
        raise ValueError("sensor frame transform application state is invalid")
    if transform_applied != (source_mode == _RAW_LIDAR_WITH_RECORDED_LIO):
        raise ValueError("mapping source mode does not match sensor transform application")
    if source_mode == _RECORDED_LIO_SESSION_ODOM_CLOUD:
        if (
            payload.get("coordinate_scope") != _SESSION_ODOM_LOCAL
            or payload.get("coordinate_frame") != "odom"
            or payload.get("room_or_building") is not False
            or payload.get("texture_authorized") is not False
        ):
            raise ValueError("local session provenance is incomplete or unsafe")
    return str(source_layer), str(point_authority), str(source_mode), transform_applied


def _require_preview_flags(
    payload: Mapping[str, Any],
    label: str,
    *,
    allow_legacy_measured: bool = False,
) -> tuple[str, str, str, bool]:
    if payload.get("quality") != PREVIEW_UNCALIBRATED or payload.get("status") != PREVIEW_UNCALIBRATED:
        raise ValueError(f"derived {label} must remain {PREVIEW_UNCALIBRATED}")
    if payload.get("geometry_authority") is not False:
        raise ValueError(f"derived {label} cannot claim geometry authority")
    try:
        provenance = _observation_provenance(
            payload,
            allow_legacy_measured=allow_legacy_measured,
        )
    except ValueError as exc:
        raise ValueError(f"derived {label} {exc}") from exc
    if payload.get("volume_filled") is not False or payload.get("watertight") is not False:
        raise ValueError(f"derived {label} cannot fill or close unobserved volume")
    return provenance


def _artifact_path(root: Path, name: object, suffix: str, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or not name.endswith(suffix):
        raise ValueError(f"derived {label} name is invalid")
    candidate = (root / name).resolve(strict=False)
    if candidate.parent != root.resolve(strict=False) or not candidate.is_file():
        raise ValueError(f"derived {label} is unavailable")
    return candidate


def _validate_identity(manifest: Mapping[str, Any], generation_root: Path) -> dict[str, Any]:
    identity = manifest.get("derived_identity")
    if not isinstance(identity, dict):
        raise ValueError("derived manifest identity is missing")
    cache_key = _lower_sha256(identity.get("cache_key"), "identity cache key")
    identity_core = dict(identity)
    identity_core.pop("cache_key", None)
    if canonical_sha256(identity_core) != cache_key:
        raise ValueError("derived identity cache key does not match its canonical payload")
    if generation_root.name != cache_key:
        raise ValueError("derived generation directory does not match the identity cache key")
    if identity.get("algorithm_version") != OFFLINE_DERIVATION_ALGORITHM:
        raise ValueError("derived identity algorithm version is unsupported")

    source_artifacts = identity.get("source_artifacts")
    source_hashes = identity.get("source_artifact_sha256")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ValueError("derived identity source artifact inventory is missing")
    if manifest.get("source_artifacts") != source_artifacts:
        raise ValueError("derived manifest source inventory does not match its identity")
    expected_hashes: list[str] = []
    for index, artifact in enumerate(source_artifacts):
        if not isinstance(artifact, dict):
            raise ValueError("derived source artifact entry is invalid")
        expected_hashes.append(_lower_sha256(artifact.get("sha256"), f"source artifact {index}"))
        relative_path = artifact.get("relative_path")
        byte_count = artifact.get("bytes")
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).name != relative_path:
            raise ValueError("derived source artifact path is invalid")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("derived source artifact byte count is invalid")
    if source_hashes != expected_hashes or len(set(expected_hashes)) != len(expected_hashes):
        raise ValueError("derived identity source hashes do not match its inventory")
    _lower_sha256(identity.get("pose_generation"), "pose generation")
    _lower_sha256(identity.get("fusion_config_sha256"), "fusion configuration")
    source_mode = identity.get("mapping_source_mode")
    if source_mode == _RECORDED_LIO_SESSION_ODOM_CLOUD:
        if identity.get("coordinate_scope") != _SESSION_ODOM_LOCAL:
            raise ValueError("derived local-session identity coordinate scope is invalid")
    elif identity.get("coordinate_scope") not in {None, "GLOBAL_MAP"}:
        raise ValueError("derived identity coordinate scope is invalid")
    return identity


def _validate_transform(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    transform_id = manifest.get("sensor_frame_transform_id")
    if not isinstance(transform_id, str) or identity.get("sensor_frame_transform_id") != transform_id:
        raise ValueError("derived sensor frame transform identity is inconsistent")
    try:
        expected = get_sensor_frame_transform(transform_id)
    except ValueError as exc:
        raise ValueError("derived sensor frame transform is unknown") from exc
    recorded = manifest.get("sensor_frame_transform")
    if not isinstance(recorded, dict) or recorded.get("id") != transform_id:
        raise ValueError("derived sensor frame transform record is missing")
    try:
        rotation = np.asarray(recorded.get("rotation"), dtype=np.float64)
        translation = np.asarray(recorded.get("translation"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("derived sensor frame transform values are invalid") from exc
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.array_equal(rotation, expected.rotation)
        or not np.array_equal(translation, expected.translation)
        or recorded.get("provisional") is not True
        or expected.provisional is not True
    ):
        raise ValueError("derived sensor frame transform does not match the registered provisional transform")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("derived sensor frame transform is not a proper rotation")
    return transform_id


def _validate_surface(
    surface: Mapping[str, Any],
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    transform_id: str,
) -> tuple[int, int]:
    surface_provenance = _require_preview_flags(surface, "surface", allow_legacy_measured=True)
    manifest_provenance = _observation_provenance(manifest, allow_legacy_measured=True)
    if surface_provenance[:3] != manifest_provenance[:3]:
        raise ValueError("derived surface source layer does not match the manifest")
    if surface.get("sensor_frame_transform_id") != transform_id:
        raise ValueError("derived surface transform does not match the manifest")
    if surface.get("pose_generation") != identity.get("pose_generation"):
        raise ValueError("derived surface pose generation does not match the identity")
    surface_hashes = surface.get("artifact_sha256")
    if not isinstance(surface_hashes, list):
        raise ValueError("derived surface source hashes do not match the identity")
    validated_surface_hashes = [
        _lower_sha256(value, f"surface source artifact {index}")
        for index, value in enumerate(surface_hashes)
    ]
    if (
        len(set(validated_surface_hashes)) != len(validated_surface_hashes)
        or sorted(validated_surface_hashes) != sorted(identity.get("source_artifact_sha256", ()))
    ):
        raise ValueError("derived surface source hashes do not match the identity")
    if surface.get("reconstruction_phase") != "IDLE":
        raise ValueError("derived surface generation is not stable")

    try:
        positions = np.asarray(surface.get("positions"), dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(surface.get("normals"), dtype=np.float64).reshape(-1, 3)
        triangle_source = np.asarray(surface.get("triangles"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("derived surface arrays are invalid") from exc
    if triangle_source.ndim != 1 or triangle_source.size % 3 or triangle_source.dtype.kind not in "iu":
        raise ValueError("derived surface triangle array is invalid")
    triangles = triangle_source.astype(np.int64, copy=False).reshape(-1, 3)
    if not len(positions) or not len(triangles) or len(normals) != len(positions):
        raise ValueError("derived surface arrays are empty or inconsistent")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)):
        raise ValueError("derived surface vectors must be finite")
    MeasuredSurface(positions, normals, triangles)

    source_vertex_ids = surface.get("source_vertex_ids")
    if (
        not isinstance(source_vertex_ids, list)
        or len(source_vertex_ids) != len(positions)
        or len(set(source_vertex_ids)) != len(source_vertex_ids)
        or any(not isinstance(value, str) or not value for value in source_vertex_ids)
    ):
        raise ValueError("derived surface measured vertex identity is invalid")
    geometry_identity = measured_surface_identity(
        positions=positions,
        normals=normals,
        triangles=triangles,
        pose_generation=str(identity["pose_generation"]),
        artifact_sha256=identity["source_artifact_sha256"],
        fusion_config_sha256=str(identity["fusion_config_sha256"]),
    )
    surface_manifest = manifest.get("surface")
    if not isinstance(surface_manifest, dict):
        raise ValueError("derived manifest surface summary is missing")
    if (
        surface.get("measured_surface_geometry_revision_id") != geometry_identity.revision_id
        or surface_manifest.get("geometry_revision_id") != geometry_identity.revision_id
        or surface_manifest.get("vertex_count") != len(positions)
        or surface_manifest.get("source_vertex_count") != len(source_vertex_ids)
        or surface_manifest.get("triangle_count") != len(triangles)
    ):
        raise ValueError("derived surface geometry identity or counts do not match the manifest")
    return len(positions), len(triangles)


def _local_session_fields(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    fields = {
        "coordinate_scope": _SESSION_ODOM_LOCAL,
        "coordinate_frame": "odom",
        "room_or_building": False,
        "texture_authorized": False,
    }
    if any(payload.get(key) != value for key, value in fields.items()):
        raise ValueError(f"derived {label} local session scope is invalid")
    return fields


def _validate_local_session_agreement(
    manifest: Mapping[str, Any],
    surface: Mapping[str, Any],
    metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("mapping_source_mode") != _RECORDED_LIO_SESSION_ODOM_CLOUD:
        return {}
    if identity.get("mapping_source_mode") != _RECORDED_LIO_SESSION_ODOM_CLOUD:
        raise ValueError("derived local-session identity source mode is invalid")
    expected = _local_session_fields(manifest, "manifest")
    if identity.get("coordinate_scope") != _SESSION_ODOM_LOCAL:
        raise ValueError("derived local-session identity coordinate scope is invalid")
    for label, payload in (("surface", surface), ("metadata", metadata)):
        if _local_session_fields(payload, label) != expected:
            raise ValueError("derived local-session provenance does not agree")
    return expected


def load_derived_surface_preview(path: str | Path) -> DerivedSurfacePreview:
    surface_path = Path(path).expanduser().resolve(strict=False)
    if surface_path.name != "measured-surface.json" or not surface_path.is_file():
        raise ValueError("GO2_MAPPING_DERIVED_SURFACE_JSON must name an existing measured-surface.json")
    generation_root = surface_path.parent
    manifest_path = generation_root / "manifest.json"
    manifest, _manifest_bytes = _json_object(manifest_path, "manifest")
    if manifest.get("schema_version") != 1 or manifest.get("surface_available") is not True:
        raise ValueError("derived manifest does not describe an available version-1 surface")
    manifest_provenance = _require_preview_flags(manifest, "manifest", allow_legacy_measured=True)
    if manifest.get("hidden_surface_policy") != "OBSERVED_POINTS_ONLY":
        raise ValueError("derived manifest hidden surface policy is invalid")
    identity = _validate_identity(manifest, generation_root)
    transform_id = _validate_transform(manifest, identity)

    surface, surface_bytes = _json_object(surface_path, "surface")
    vertex_count, triangle_count = _validate_surface(surface, manifest, identity, transform_id)

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("derived manifest artifact pair is missing")
    artifacts_root = generation_root / "artifacts"
    ply_path = _artifact_path(artifacts_root, artifact.get("ply_name"), ".ply", "PLY")
    metadata_path = _artifact_path(artifacts_root, artifact.get("metadata_name"), ".json", "metadata")
    if ply_path.stem != metadata_path.stem:
        raise ValueError("derived PLY and metadata names do not form one artifact pair")
    expected_ply_sha256 = _lower_sha256(artifact.get("ply_sha256"), "PLY checksum")
    actual_ply_sha256 = _sha256_file(ply_path)
    if actual_ply_sha256 != expected_ply_sha256:
        raise ValueError("derived PLY checksum changed")

    metadata, metadata_bytes = _json_object(metadata_path, "metadata")
    metadata_provenance = _require_preview_flags(metadata, "metadata", allow_legacy_measured=True)
    if (
        metadata_provenance[:3] != manifest_provenance[:3]
        or metadata.get("hidden_surface_policy") != "OBSERVED_POINTS_ONLY"
        or metadata.get("sensor_frame_transform_id") != transform_id
        or metadata.get("derived_identity") != identity
        or metadata.get("derived_cache_key") != identity["cache_key"]
        or metadata.get("ply_sha256") != actual_ply_sha256
        or metadata.get("vertex_count") != vertex_count
        or metadata.get("triangle_count") != triangle_count
    ):
        raise ValueError("derived metadata does not match the manifest, identity, or PLY")
    canonical_metadata = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if metadata_bytes != canonical_metadata:
        raise ValueError("derived metadata is not in canonical form")
    local_session_fields = _validate_local_session_agreement(manifest, surface, metadata, identity)

    status = {
        "schema_version": 1,
        "available": True,
        "blocker": None,
        "source_mode": DERIVED_READ_ONLY,
        "read_only": True,
        "quality": PREVIEW_UNCALIBRATED,
        "geometry_authority": False,
        "point_authority": manifest_provenance[1],
        "source_layer": manifest_provenance[0],
        "mapping_source_mode": manifest_provenance[2],
        "sensor_frame_transform_applied": manifest_provenance[3],
        "sensor_frame_transform_id": transform_id,
        "derived_cache_key": identity["cache_key"],
        "surface_sha256": hashlib.sha256(surface_bytes).hexdigest(),
        "ply_sha256": actual_ply_sha256,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
    }
    status.update(local_session_fields)
    return DerivedSurfacePreview(
        surface_path=surface_path,
        manifest_path=manifest_path.resolve(),
        ply_path=ply_path,
        metadata_path=metadata_path,
        status=status,
    )


def load_automatic_reconstruction_preview(
    status_path: str | Path,
    *,
    output_root: str | Path,
) -> tuple[DerivedSurfacePreview, dict[str, Any]]:
    """Validate one worker-owned latest-surface pointer without trusting paths.

    The Dashboard reads this on demand rather than at process startup.  The
    worker publishes the pointer with ``os.replace`` after the complete
    generation has passed the ordinary content-addressed surface validator.
    """

    pointer_path = Path(status_path).expanduser().resolve(strict=False)
    root = Path(output_root).expanduser().resolve(strict=False)
    pointer, _pointer_bytes = _json_object(pointer_path, "automatic reconstruction status")
    if pointer.get("schema_version") != 1:
        raise ValueError("automatic reconstruction status schema is unsupported")
    session_id = pointer.get("session_id")
    if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise ValueError("automatic reconstruction status session is invalid")
    if pointer.get("available") is not True or pointer.get("blocker") is not None:
        raise ValueError("automatic reconstruction has no available surface")
    if pointer.get("mode") not in {
        "high_fidelity_incremental_reconstruction",
        "high_fidelity_batch_reconstruction",
        "session_odom_local_observed_surface",
    }:
        raise ValueError("automatic reconstruction mode is invalid")
    if pointer.get("geometry_authority") is not False:
        raise ValueError("automatic reconstruction cannot claim geometry authority")
    try:
        pointer_provenance = _observation_provenance(pointer, allow_legacy_measured=True)
    except ValueError as exc:
        raise ValueError(f"automatic reconstruction {exc}") from exc
    if (
        pointer.get("mode") == "session_odom_local_observed_surface"
        or pointer.get("coordinate_scope") == _SESSION_ODOM_LOCAL
        or (
            pointer_provenance[2] == _RECORDED_LIO_SESSION_ODOM_CLOUD
            and pointer.get("coordinate_scope") == _SESSION_ODOM_LOCAL
        )
    ):
        raise ValueError("automatic reconstruction cannot load SESSION_ODOM_LOCAL surface")
    if pointer_provenance[2] == _RECORDED_LIO_SESSION_ODOM_CLOUD and pointer.get("coordinate_scope") != _SESSION_ODOM_LOCAL:
        raise ValueError("automatic reconstruction local session coordinate scope is invalid")
    if pointer.get("hidden_surface_policy") != "OBSERVED_POINTS_ONLY":
        raise ValueError("automatic reconstruction hidden surface policy is invalid")
    relative_surface_value = pointer.get("surface_json_relative_path")
    if relative_surface_value is not None:
        if not isinstance(relative_surface_value, str) or not relative_surface_value:
            raise ValueError("automatic reconstruction relative surface path is invalid")
        relative_surface_path = Path(relative_surface_value)
        if relative_surface_path.is_absolute() or ".." in relative_surface_path.parts:
            raise ValueError("automatic reconstruction relative surface path is invalid")
        surface_path = (root / relative_surface_path).resolve(strict=False)
    else:
        surface_value = pointer.get("surface_json_path")
        if not isinstance(surface_value, str) or not surface_value:
            raise ValueError("automatic reconstruction surface path is missing")
        surface_path = Path(surface_value).expanduser().resolve(strict=False)
    try:
        surface_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("automatic reconstruction surface path escapes its output root") from exc
    preview = load_derived_surface_preview(surface_path)
    if (
        pointer.get("surface_sha256") != preview.status["surface_sha256"]
        or pointer.get("derived_cache_key") != preview.status["derived_cache_key"]
        or pointer.get("ply_sha256") != preview.status["ply_sha256"]
        or pointer.get("metadata_sha256") != preview.status["metadata_sha256"]
        or pointer.get("vertex_count") != preview.status["vertex_count"]
        or pointer.get("triangle_count") != preview.status["triangle_count"]
        or pointer.get("sensor_frame_transform_id") != preview.status["sensor_frame_transform_id"]
        or pointer_provenance[0] != preview.status["source_layer"]
        or pointer_provenance[1] != preview.status["point_authority"]
        or pointer_provenance[2] != preview.status["mapping_source_mode"]
        or pointer_provenance[3] != preview.status["sensor_frame_transform_applied"]
    ):
        raise ValueError("automatic reconstruction status does not match its verified surface")
    return preview, pointer


def _surface_path_from_pointer(pointer: Mapping[str, Any], root: Path, label: str) -> Path:
    relative_surface_value = pointer.get("surface_json_relative_path")
    if not isinstance(relative_surface_value, str) or not relative_surface_value:
        raise ValueError(f"{label} relative surface path is missing")
    relative_surface_path = Path(relative_surface_value)
    if relative_surface_path.is_absolute() or ".." in relative_surface_path.parts:
        raise ValueError(f"{label} relative surface path is invalid")
    surface_path = (root / relative_surface_path).resolve(strict=False)
    try:
        surface_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} surface path escapes its output root") from exc
    return surface_path


def _require_pointer_match(
    pointer: Mapping[str, Any],
    preview: DerivedSurfacePreview,
    *,
    label: str,
) -> None:
    expected = {
        "surface_sha256": preview.status["surface_sha256"],
        "derived_cache_key": preview.status["derived_cache_key"],
        "ply_sha256": preview.status["ply_sha256"],
        "metadata_sha256": preview.status["metadata_sha256"],
        "vertex_count": preview.status["vertex_count"],
        "triangle_count": preview.status["triangle_count"],
        "sensor_frame_transform_id": preview.status["sensor_frame_transform_id"],
        "source_layer": preview.status["source_layer"],
        "point_authority": preview.status["point_authority"],
        "mapping_source_mode": preview.status["mapping_source_mode"],
        "sensor_frame_transform_applied": preview.status["sensor_frame_transform_applied"],
    }
    if any(pointer.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} does not match its verified surface")


def load_session_odom_local_preview(
    status_path: str | Path,
    *,
    output_root: str | Path,
) -> tuple[DerivedSurfacePreview, dict[str, Any]]:
    """Load the separately published, read-only local ``odom`` validation surface.

    This pointer is intentionally incompatible with the automatic reconstruction
    pointer: it can describe exactly one continuous local LIO session and never
    asserts global, room, building, or texture authority.
    """

    pointer_path = Path(status_path).expanduser().resolve(strict=False)
    root = Path(output_root).expanduser().resolve(strict=False)
    pointer, _pointer_bytes = _json_object(pointer_path, "session odom local status")
    if pointer.get("schema_version") != 1:
        raise ValueError("session odom local status schema is unsupported")
    session_id = pointer.get("session_id")
    if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise ValueError("session odom local status session is invalid")
    if pointer.get("state") != "SESSION_ODOM_LOCAL_SURFACE_AVAILABLE":
        raise ValueError("session odom local status state is invalid")
    if pointer.get("mode") != "session_odom_local_observed_surface":
        raise ValueError("session odom local status mode is invalid")
    if pointer.get("available") is not True or pointer.get("blocker") is not None:
        raise ValueError("session odom local has no available surface")
    if pointer.get("read_only") is not True or pointer.get("geometry_authority") is not False:
        raise ValueError("session odom local must remain read-only")
    if pointer.get("quality") != PREVIEW_UNCALIBRATED:
        raise ValueError("session odom local quality is invalid")
    if pointer.get("hidden_surface_policy") != "OBSERVED_POINTS_ONLY":
        raise ValueError("session odom local hidden surface policy is invalid")
    if pointer.get("mapping_source_mode") != _RECORDED_LIO_SESSION_ODOM_CLOUD:
        raise ValueError("session odom local mapping source mode is invalid")
    local_fields = _local_session_fields(pointer, "pointer")
    if pointer.get("sensor_frame_transform_applied") is not False:
        raise ValueError("session odom local cannot apply a sensor frame transform")
    surface_path = _surface_path_from_pointer(pointer, root, "session odom local")
    preview = load_derived_surface_preview(surface_path)
    if preview.status.get("mapping_source_mode") != _RECORDED_LIO_SESSION_ODOM_CLOUD:
        raise ValueError("session odom local surface source mode is invalid")
    if any(preview.status.get(key) != value for key, value in local_fields.items()):
        raise ValueError("session odom local pointer does not match local surface provenance")
    _require_pointer_match(pointer, preview, label="session odom local status")
    return preview, pointer


def _session_id_from_surface_path(surface_path: Path) -> str:
    candidates = [part for part in surface_path.parts if _SESSION_ID_RE.fullmatch(part)]
    if len(candidates) != 1:
        raise ValueError("local surface path must contain exactly one canonical capture session ID")
    return candidates[0]


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def publish_session_odom_local_preview(
    surface_json: str | Path,
    *,
    output_root: str | Path,
    status_path: str | Path,
) -> dict[str, Any]:
    """Publish a local-only pointer after fully validating its source surface."""

    root = Path(output_root).expanduser().resolve(strict=False)
    surface_path = Path(surface_json).expanduser().resolve(strict=False)
    status_target = Path(status_path).expanduser().resolve(strict=False)
    if status_target.name != "session-odom-local-current.json":
        raise ValueError("local status path must be named session-odom-local-current.json")
    if status_target.name == "automatic-reconstruction-current.json":
        raise ValueError("local status path must not target automatic reconstruction")
    try:
        relative_surface = surface_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("local surface path escapes its output root") from exc
    preview = load_derived_surface_preview(surface_path)
    if preview.status.get("mapping_source_mode") != _RECORDED_LIO_SESSION_ODOM_CLOUD:
        raise ValueError("local publisher accepts only RECORDED_LIO_SESSION_ODOM_CLOUD")
    local_fields = {
        key: preview.status.get(key)
        for key in ("coordinate_scope", "coordinate_frame", "room_or_building", "texture_authorized")
    }
    if local_fields != {
        "coordinate_scope": _SESSION_ODOM_LOCAL,
        "coordinate_frame": "odom",
        "room_or_building": False,
        "texture_authorized": False,
    }:
        raise ValueError("local publisher surface provenance is invalid")
    pointer = {
        "schema_version": 1,
        "session_id": _session_id_from_surface_path(surface_path),
        "state": "SESSION_ODOM_LOCAL_SURFACE_AVAILABLE",
        "mode": "session_odom_local_observed_surface",
        "available": True,
        "blocker": None,
        "read_only": True,
        "quality": PREVIEW_UNCALIBRATED,
        "geometry_authority": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "surface_json_relative_path": relative_surface.as_posix(),
        **local_fields,
        **{
            key: preview.status[key]
            for key in (
                "mapping_source_mode",
                "sensor_frame_transform_applied",
                "sensor_frame_transform_id",
                "source_layer",
                "point_authority",
                "surface_sha256",
                "derived_cache_key",
                "ply_sha256",
                "metadata_sha256",
                "vertex_count",
                "triangle_count",
            )
        },
    }
    _atomic_json_write(status_target, pointer)
    return pointer


__all__ = [
    "DERIVED_READ_ONLY",
    "DerivedSurfacePreview",
    "load_automatic_reconstruction_preview",
    "load_derived_surface_preview",
    "load_session_odom_local_preview",
    "publish_session_odom_local_preview",
]
