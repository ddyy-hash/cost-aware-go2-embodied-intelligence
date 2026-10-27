"""Read-only, content-addressed derivation of a provisional LiDAR surface."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterator, Mapping

import numpy as np
from scipy.spatial import cKDTree
import zstandard

from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.high_fidelity_mapper import (
    HighFidelityMapper,
    RAW_LIDAR_WITH_RECORDED_LIO,
    RECORDED_LIO_REGISTERED_CLOUD,
    RECORDED_LIO_SESSION_ODOM_CLOUD,
)
from mapping.pc.go2_mapping_live.sensor_frame import (
    get_sensor_frame_transform,
    normalize_sensor_frame_transform_id,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import (
    canonical_sha256,
    load_reconstruction_config,
)

from .artifacts import LayerArtifactStore


PREVIEW_UNCALIBRATED = "PREVIEW_UNCALIBRATED"
OFFLINE_DERIVATION_ALGORITHM = "go2-offline-observed-surface-v1"
_PART_RE = re.compile(r"_(\d+)\.db3\.zstd$", re.ASCII)
_HEX64_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_LOCAL_SESSION_SCOPE = "SESSION_ODOM_LOCAL"
_LOCAL_SESSION_FRAME = "odom"


@dataclass(frozen=True)
class SourceArtifact:
    relative_path: str
    path: Path
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class DerivationResult:
    output_root: Path
    manifest_path: Path
    surface_json_path: Path | None
    ply_path: Path | None
    metadata_path: Path | None
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _part_sort_key(path: Path) -> tuple[int, str]:
    match = _PART_RE.search(path.name)
    return (int(match.group(1)), path.name) if match else (2**63 - 1, path.name)


def enumerate_capture_parts(source_root: str | Path) -> tuple[SourceArtifact, ...]:
    """Inventory only closed compressed parts directly in one capture directory."""

    root = Path(source_root).resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"capture source directory is unavailable: {root}")
    if any(path.is_file() for path in root.glob("*.db3")):
        raise ValueError("capture source contains an uncompressed .db3; use closed .db3.zstd parts only")
    paths = sorted((path for path in root.glob("*.db3.zstd") if path.is_file()), key=_part_sort_key)
    if not paths:
        raise ValueError(f"capture source has no closed .db3.zstd parts: {root}")
    indices = [_part_sort_key(path)[0] for path in paths]
    numbered = [index for index in indices if index < 2**63 - 1]
    if len(numbered) != len(set(numbered)):
        raise ValueError("capture source contains duplicate numeric part indices")
    return tuple(
        SourceArtifact(
            relative_path=path.relative_to(root).as_posix(),
            path=path.resolve(),
            sha256=_sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in paths
    )


@contextmanager
def _decompressed_part(source: Path, *, work_root: Path | None = None) -> Iterator[Path]:
    """Materialize exactly one source part and remove it before returning."""

    root = None if work_root is None else Path(work_root)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="go2-derived-part-",
        suffix=".db3",
        dir=None if root is None else str(root),
    )
    sqlite_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file, source.open("rb") as input_file:
            zstandard.ZstdDecompressor().copy_stream(input_file, output_file)
        yield sqlite_path
    except (OSError, zstandard.ZstdError) as exc:
        raise ValueError(f"cannot decompress capture part: {source}") from exc
    finally:
        try:
            sqlite_path.unlink(missing_ok=True)
        except OSError:
            pass


def _materialize_verified_part(part: SourceArtifact, work_root: Path) -> tuple[SourceArtifact, Path]:
    """Hash and decompress one immutable part for the ordered replay consumer."""

    if _sha256_file(part.path) != part.sha256:
        raise ValueError(f"capture part changed before replay: {part.path}")
    work_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="go2-derived-part-",
        suffix=".db3",
        dir=str(work_root),
    )
    sqlite_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file, part.path.open("rb") as input_file:
            zstandard.ZstdDecompressor().copy_stream(input_file, output_file)
        if _sha256_file(part.path) != part.sha256:
            raise ValueError(f"capture part changed during replay: {part.path}")
        return part, sqlite_path
    except (OSError, zstandard.ZstdError) as exc:
        try:
            sqlite_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"cannot decompress capture part: {part.path}") from exc


def _independent_output_root(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve(strict=False)
    output = output_root.resolve(strict=False)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("derived output must use an independent output root")


def _surface_from_snapshot(snapshot: Mapping[str, Any]) -> MeasuredSurface:
    source_layer = snapshot.get("source_layer")
    point_authority = snapshot.get("point_authority")
    allowed_layers = {
        ("LIDAR_MEASURED", "LIDAR_MEASURED"),
        ("LIO_REGISTERED_LIDAR_OBSERVED", "LIO_REGISTERED_LIDAR_OBSERVED"),
    }
    if (source_layer, point_authority) not in allowed_layers:
        raise ValueError("offline output requires an allowed LiDAR observation surface")
    if snapshot.get("geometry_authority") is not False:
        raise ValueError("offline output cannot promote geometry authority")
    positions = np.asarray(snapshot.get("positions", ()), dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(snapshot.get("normals", ()), dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(snapshot.get("triangles", ()), dtype=np.int32).reshape(-1, 3)
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError("offline measured surface contains non-finite vectors")
    return MeasuredSurface(
        positions=positions,
        normals=normals,
        triangles=triangles,
        source_vertex_ids=tuple(snapshot.get("source_vertex_ids", ())),
        source_layer=str(source_layer),
        geometry_authority=False,
        point_authority=str(point_authority),
    )


def _surface_diagnostics(surface: MeasuredSurface, mapper: Any) -> dict[str, Any]:
    positions = np.asarray(surface.positions, dtype=np.float64)
    normals = np.asarray(surface.normals, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "vertex_count": surface.vertex_count,
        "triangle_count": surface.triangle_count,
        "bounds_m": None,
        "horizontal_normal_support_count": 0,
        "vertical_normal_support_count": 0,
        "non_horizontal_triangle_fraction": None,
        "ground_below_trajectory": None,
        "point_to_trajectory_distance_m": None,
        "hidden_surface_generated": False,
        "volume_filled": False,
        "watertight": False,
    }
    if not len(positions):
        return diagnostics
    diagnostics["bounds_m"] = {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
        "extent": (positions.max(axis=0) - positions.min(axis=0)).tolist(),
    }
    normal_z = np.abs(normals[:, 2])
    horizontal = normal_z >= 0.80
    vertical = (normal_z <= 0.35) & (np.linalg.norm(normals[:, :2], axis=1) >= 0.80)
    diagnostics["horizontal_normal_support_count"] = int(np.count_nonzero(horizontal))
    diagnostics["vertical_normal_support_count"] = int(np.count_nonzero(vertical))
    if surface.triangle_count:
        triangles = positions[surface.triangles]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_lengths = np.linalg.norm(face_normals, axis=1)
        valid = face_lengths > 1e-12
        if np.any(valid):
            face_normals = face_normals[valid] / face_lengths[valid, None]
            diagnostics["non_horizontal_triangle_fraction"] = float(
                np.mean(np.abs(face_normals[:, 2]) < 0.80)
            )

    raw_frames = getattr(mapper, "_raw_frames", ())
    solution = getattr(mapper, "_last_pose_graph_solution", None)
    trajectory = []
    for frame in raw_frames:
        pose = getattr(solution, "poses_map_from_keyframe", {}).get(
            frame.node_id,
            frame.initial_pose_map_from_lidar,
        )
        trajectory.append(np.asarray(pose, dtype=np.float64)[:3, 3])
    if trajectory:
        trajectory_array = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
        distances, _ = cKDTree(trajectory_array).query(positions, k=1)
        diagnostics["point_to_trajectory_distance_m"] = {
            "minimum": float(np.min(distances)),
            "p50": float(np.percentile(distances, 50)),
            "p95": float(np.percentile(distances, 95)),
            "maximum": float(np.max(distances)),
        }
        ground_level = float(np.percentile(positions[:, 2], 10))
        diagnostics["ground_below_trajectory"] = bool(
            ground_level < float(np.median(trajectory_array[:, 2])) - 0.05
        )
    return diagnostics


def _identity_payload(
    source_artifacts: tuple[SourceArtifact, ...],
    *,
    pose_generation: str,
    sensor_frame_transform_id: str,
    mapping_source_mode: str,
    coordinate_scope: str | None,
    fusion_config_sha256: str,
) -> dict[str, Any]:
    payload = {
        "identity_schema_version": 1,
        "algorithm_version": OFFLINE_DERIVATION_ALGORITHM,
        "source_artifacts": [item.as_dict() for item in source_artifacts],
        "source_artifact_sha256": [item.sha256 for item in source_artifacts],
        "sensor_frame_transform_id": sensor_frame_transform_id,
        "mapping_source_mode": mapping_source_mode,
        "coordinate_scope": coordinate_scope,
        "fusion_config_sha256": fusion_config_sha256,
        "pose_generation": pose_generation,
    }
    return {**payload, "cache_key": canonical_sha256(payload)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_existing_generation(
    generation_root: Path,
    expected_identity: Mapping[str, Any],
) -> DerivationResult:
    manifest_path = generation_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing derived generation is incomplete: {generation_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing derived manifest is unreadable: {generation_root}") from exc
    if manifest.get("derived_identity") != dict(expected_identity):
        raise ValueError("existing derived generation identity conflicts with requested input")
    surface_json = generation_root / "measured-surface.json"
    artifact = manifest.get("artifact") or {}
    if manifest.get("surface_available") is False:
        if surface_json.exists() or artifact:
            raise ValueError(f"existing derived no-surface generation is inconsistent: {generation_root}")
        return DerivationResult(generation_root, manifest_path, None, None, None, manifest)
    ply_name = artifact.get("ply_name")
    metadata_name = artifact.get("metadata_name")
    ply_path = generation_root / "artifacts" / ply_name if isinstance(ply_name, str) else None
    metadata_path = generation_root / "artifacts" / metadata_name if isinstance(metadata_name, str) else None
    if not surface_json.is_file() or ply_path is None or metadata_path is None:
        raise ValueError(f"existing derived generation is incomplete: {generation_root}")
    if not ply_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"existing derived artifact pair is incomplete: {generation_root}")
    return DerivationResult(generation_root, manifest_path, surface_json, ply_path, metadata_path, manifest)


def derive_capture(
    source_root: str | Path,
    output_root: str | Path,
    *,
    sensor_frame_transform_id: str,
    voxel_m: float = 0.02,
    display_point_budget: int = 100_000,
    max_parts: int | None = None,
    work_root: str | Path | None = None,
    global_optimization: bool = True,
    source_mode: str = RAW_LIDAR_WITH_RECORDED_LIO,
    lidar_topics: tuple[str, ...] = ("/go2/lio/lidar", "/livox/lidar"),
    mapper_factory: Callable[..., Any] = HighFidelityMapper,
    prefetch_workers: int = 2,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> DerivationResult:
    """Replay immutable capture parts into a new provisional surface generation."""

    source = Path(source_root).resolve(strict=False)
    output = Path(output_root).resolve(strict=False)
    decompression_root = None if work_root is None else Path(work_root).resolve(strict=False)
    _independent_output_root(source, output)
    transform_id = normalize_sensor_frame_transform_id(sensor_frame_transform_id)
    if source_mode not in {
        RAW_LIDAR_WITH_RECORDED_LIO,
        RECORDED_LIO_REGISTERED_CLOUD,
        RECORDED_LIO_SESSION_ODOM_CLOUD,
    }:
        raise ValueError(f"unknown mapping source mode: {source_mode}")
    transform = get_sensor_frame_transform(transform_id)
    parts = enumerate_capture_parts(source)
    if max_parts is not None:
        if isinstance(max_parts, bool) or max_parts <= 0:
            raise ValueError("max_parts must be a positive integer")
        parts = parts[:max_parts]
    if not parts:
        raise ValueError("capture has no selected parts")
    if isinstance(prefetch_workers, bool) or not 1 <= int(prefetch_workers) <= 4:
        raise ValueError("prefetch_workers must be between 1 and 4")

    mapper = mapper_factory(
        voxel_m=voxel_m,
        display_point_budget=display_point_budget,
        source_mode=source_mode,
        lidar_topics=lidar_topics,
        sensor_frame_transform_id=transform_id,
        reintegration_work_root=decompression_root,
        prewarm_surface_after_ingest=False,
        defer_pose_graph_optimization=True,
    )
    ingest_results: list[dict[str, Any]] = []
    replay_work_root = decompression_root or Path("/tmp/go2-mapping-reconstruction")
    # I/O and decompression overlap, but mapper submission remains ordered so
    # pose and evidence state cannot be mutated concurrently.
    with ThreadPoolExecutor(max_workers=int(prefetch_workers), thread_name_prefix="go2-replay") as executor:
        pending = [executor.submit(_materialize_verified_part, part, replay_work_root) for part in parts]
        for future in pending:
            part, sqlite_path = future.result()
            if progress_callback is not None:
                progress_callback({"stage": "part_ready", "part": part.relative_path, "parts_total": len(parts)})
            try:
                result = dict(
                    mapper.ingest_sqlite_bag(
                        sqlite_path,
                        artifact_sha256=part.sha256,
                        source_artifact_path=part.path,
                    )
                )
                ingest_results.append(result)
                if progress_callback is not None:
                    progress_callback({
                        "stage": "part_ingested",
                        "part": part.relative_path,
                        "part_index": len(ingest_results),
                        "parts_total": len(parts),
                        "integrated_frames": int(result.get("integrated_frames", 0)),
                        "integrated_points": int(result.get("integrated_points", 0)),
                    })
            finally:
                try:
                    sqlite_path.unlink(missing_ok=True)
                except OSError:
                    pass

    if progress_callback is not None:
        progress_callback({"stage": "reintegration", "parts_total": len(parts)})
    if global_optimization:
        # Preserve the established mapper extension point for the normal
        # globally optimized derivation path.
        mapper.finalize_deferred_batch()
    else:
        mapper.finalize_deferred_batch(global_optimization=False)
    if progress_callback is not None:
        progress_callback({"stage": "surfacing", "parts_total": len(parts)})
    state = dict(mapper.snapshot())
    pose_generation = str(state.get("optimized_pose_generation") or state.get("pose_generation") or "")
    if not pose_generation:
        pose_generation = canonical_sha256({"state": "NO_MATCHED_POSE"})
    if state.get("sensor_frame_transform_id", transform_id) != transform_id:
        raise ValueError("mapper returned a different sensor frame transform")
    config = load_reconstruction_config()
    identity = _identity_payload(
        parts,
        pose_generation=pose_generation,
        sensor_frame_transform_id=transform_id,
        mapping_source_mode=source_mode,
        coordinate_scope=(
            _LOCAL_SESSION_SCOPE if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP"
            if source_mode == RECORDED_LIO_REGISTERED_CLOUD else None
        ),
        fusion_config_sha256=config.fusion_sha256,
    )
    generation_root = output / source.name / f"sensor-frame-{transform_id}" / identity["cache_key"]
    if generation_root.exists():
        if progress_callback is not None:
            progress_callback({"stage": "published_existing", "generation": generation_root.name})
        return _load_existing_generation(generation_root, identity)
    generation_root.mkdir(parents=True, exist_ok=False)

    surface_snapshot = mapper.measured_surface_snapshot()
    source_layer = (
        "LIO_REGISTERED_LIDAR_OBSERVED"
        if source_mode in {RECORDED_LIO_REGISTERED_CLOUD, RECORDED_LIO_SESSION_ODOM_CLOUD}
        else "LIDAR_MEASURED"
    )
    point_authority = (
        "LIO_REGISTERED_LIDAR_OBSERVED"
        if source_mode in {RECORDED_LIO_REGISTERED_CLOUD, RECORDED_LIO_SESSION_ODOM_CLOUD}
        else "LIDAR_MEASURED"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": PREVIEW_UNCALIBRATED,
        "quality": PREVIEW_UNCALIBRATED,
        "geometry_authority": False,
        "source_layer": source_layer,
        "point_authority": point_authority,
        "volume_filled": False,
        "watertight": False,
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
        "sensor_frame_transform_id": transform_id,
        "mapping_source_mode": source_mode,
        "sensor_frame_transform_applied": source_mode == RAW_LIDAR_WITH_RECORDED_LIO,
        "coordinate_scope": (
            _LOCAL_SESSION_SCOPE if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else "GLOBAL_MAP"
            if source_mode == RECORDED_LIO_REGISTERED_CLOUD else None
        ),
        "coordinate_frame": _LOCAL_SESSION_FRAME if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else None,
        "room_or_building": False if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else None,
        "texture_authorized": False if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD else None,
        "sensor_frame_transform": {
            "id": transform.id,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
            "provisional": transform.provisional,
            "description": transform.description,
        },
        "source_root_name": source.name,
        "source_artifacts": [item.as_dict() for item in parts],
        "derived_identity": identity,
        "mapper": {
            key: state.get(key)
            for key in (
                "quality", "points", "artifacts", "frames", "source_points",
                "skipped_pose_frames", "decode_errors", "reconstruction_phase",
                "accepted_loop_closures", "raw_source_artifacts",
            )
        },
        "ingest_results": ingest_results,
        "surface_available": surface_snapshot is not None,
    }
    surface_json_path: Path | None = None
    ply_path: Path | None = None
    metadata_path: Path | None = None
    if surface_snapshot is not None:
        if surface_snapshot.get("sensor_frame_transform_id", transform_id) != transform_id:
            raise ValueError("measured surface returned a different sensor frame transform")
        if surface_snapshot.get("quality", PREVIEW_UNCALIBRATED) != PREVIEW_UNCALIBRATED:
            raise ValueError("measured surface quality is not provisional")
        if source_mode == RECORDED_LIO_SESSION_ODOM_CLOUD:
            observed_frames = surface_snapshot.get("recorded_coordinate_frame_ids")
            if observed_frames not in (None, [_LOCAL_SESSION_FRAME], (_LOCAL_SESSION_FRAME,)):
                raise ValueError("local session surface must contain only recorded odom coordinates")
        surface = _surface_from_snapshot(surface_snapshot)
        diagnostics = _surface_diagnostics(surface, mapper)
        surface_payload = {
            **dict(surface_snapshot),
            "source_layer": source_layer,
            "point_authority": point_authority,
            "mapping_source_mode": source_mode,
            "sensor_frame_transform_applied": source_mode == RAW_LIDAR_WITH_RECORDED_LIO,
            "sensor_frame_transform_id": transform_id,
            "recorded_coordinate_frame_ids": list(surface_snapshot.get("recorded_coordinate_frame_ids", ())),
            "coordinate_scope": manifest["coordinate_scope"],
            "coordinate_frame": manifest["coordinate_frame"],
            "room_or_building": manifest["room_or_building"],
            "texture_authorized": manifest["texture_authorized"],
            "geometry_authority": False,
            "quality": PREVIEW_UNCALIBRATED,
            "status": PREVIEW_UNCALIBRATED,
            "volume_filled": False,
            "watertight": False,
        }
        surface_json_path = generation_root / "measured-surface.json"
        _write_json(surface_json_path, surface_payload)
        artifacts_root = generation_root / "artifacts"
        published = LayerArtifactStore(artifacts_root).publish(
            surface,
            identity,
            annotations={
                "quality": PREVIEW_UNCALIBRATED,
                "status": PREVIEW_UNCALIBRATED,
                "sensor_frame_transform_id": transform_id,
                "volume_filled": False,
                "watertight": False,
                "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
                "mapping_source_mode": source_mode,
                "sensor_frame_transform_applied": source_mode == RAW_LIDAR_WITH_RECORDED_LIO,
                "recorded_coordinate_frame_ids": list(surface_snapshot.get("recorded_coordinate_frame_ids", ())),
                "coordinate_scope": manifest["coordinate_scope"],
                "coordinate_frame": manifest["coordinate_frame"],
                "room_or_building": manifest["room_or_building"],
                "texture_authorized": manifest["texture_authorized"],
            },
        )
        ply_path = published.ply_path
        metadata_path = published.metadata_path
        manifest["surface"] = {
            "vertex_count": surface.vertex_count,
            "triangle_count": surface.triangle_count,
            "source_vertex_count": len(surface.source_vertex_ids),
            "geometry_revision_id": surface_snapshot.get("measured_surface_geometry_revision_id"),
            "diagnostics": diagnostics,
        }
        manifest["artifact"] = {
            "ply_name": ply_path.name,
            "metadata_name": metadata_path.name,
            "ply_sha256": published.ply_sha256,
        }
    else:
        manifest["surface"] = None
        manifest["artifact"] = None
    manifest_path = generation_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return DerivationResult(generation_root, manifest_path, surface_json_path, ply_path, metadata_path, manifest)


__all__ = [
    "OFFLINE_DERIVATION_ALGORITHM",
    "PREVIEW_UNCALIBRATED",
    "RECORDED_LIO_SESSION_ODOM_CLOUD",
    "DerivationResult",
    "SourceArtifact",
    "derive_capture",
    "enumerate_capture_parts",
]
