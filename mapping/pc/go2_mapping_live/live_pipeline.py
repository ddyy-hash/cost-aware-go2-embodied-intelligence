"""Durable WAL-to-building-surface pipeline independent of HTTP handling."""
from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from mapping.pc.go2_mapping_appearance import (
    BuildingAppearanceMember,
    BuildingAppearanceStore,
    PublishedBuildingAppearance,
    TileAppearanceStore,
)
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember, BuildingAssembly, BuildingAssemblyStore
from mapping.pc.go2_mapping_rgbd.multiview_surface import (
    NeuralSurfaceCandidate,
    NeuralSurfaceDecision,
)

from .building_project import BuildingProject
from .live_protocol import LiveKeyframeMessage, StreamKey
from .live_surface import LiveSurfaceMap, MeshingSurfel, TileId
from .live_surface_mesher import (
    TRIANGLE_PROVENANCE_NAMES,
    LiveSurfaceMesher,
    MeshTileRevision,
    TriangleProvenance,
)
from .live_tiles import ImmutableTileStore, TilePublicationRequest, decode_surface_tile
from .live_wal import LiveKeyframeWal, WalAdmission, WalRecord
from .relocalization import RelocalizationStore


MESH_TILE_BATCH_SIZE = 4
MESH_HALO_M = 0.05


@dataclass(frozen=True)
class PipelineUpdate:
    building_id: str
    submap_id: str
    session_id: str
    floor_id: str
    integrated_cells: int
    duplicate_surface_observation: bool
    checkpoint_generation_id: Optional[str]


@dataclass
class _SurfaceRuntime:
    project: BuildingProject
    submap_id: str
    surface: LiveSurfaceMap
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    pending_records: list[WalRecord] = field(default_factory=list)
    mesh_dirty_tiles: set[TileId] = field(default_factory=set)
    frames_since_checkpoint: int = 0
    last_checkpoint_at: float = field(default_factory=time.monotonic)
    last_integrated_at: float = field(default_factory=time.monotonic)
    last_checkpoint_generation_id: Optional[str] = None


@dataclass(frozen=True)
class _FrozenMeshingSurface:
    surfels: tuple[MeshingSurfel, ...]
    tile_m: float

    def surfels_for_tiles(
        self,
        _tiles: set[TileId],
        *,
        halo_m: float,
    ) -> tuple[MeshingSurfel, ...]:
        return self.surfels


def _mesh_frozen_surfel_batch(
    surfels: tuple[MeshingSurfel, ...],
    *,
    tile_m: float,
    mesh_halo_m: float,
    tile_store_root: str,
    publication_building_id: str,
    publication_floor_id: str,
    publication_component_id: str,
    publication_attachment_state: str,
    publication_submap_id: str,
    expected_generation_id: str,
    expected_pose_revision_id: str | None,
    tile_ids: tuple[tuple[int, int, int], ...],
) -> dict:
    surface = _FrozenMeshingSurface(
        surfels=surfels,
        tile_m=float(tile_m),
    )
    result = LiveSurfaceMesher(halo_m=mesh_halo_m).build(
        surface,
        tiles={TileId(*(int(value) for value in tile_id)) for tile_id in tile_ids},
    )
    requests = tuple(
        TilePublicationRequest(
            building_id=publication_building_id,
            floor_id=publication_floor_id,
            component_id=publication_component_id,
            attachment_state=publication_attachment_state,
            submap_id=publication_submap_id,
            revision=revision,
            source_generation_id=expected_generation_id,
            pose_revision_id=expected_pose_revision_id,
        )
        for revision in result.tile_revisions.values()
    )
    published = ImmutableTileStore(tile_store_root).publish_batch(requests).metadata
    return {
        "published": list(published),
        "triangle_count": result.triangle_count,
        "unresolved_surfel_count": result.unresolved_surfel_count,
    }


class LiveSurfacePipeline:
    def __init__(
        self,
        root: str | Path,
        *,
        wal: LiveKeyframeWal,
        checkpoint_batch_frames: int = 5,
        checkpoint_interval_s: float = 2.0,
        default_floor_id: str = "floor-unassigned",
        default_zone_id: str = "zone-unassigned",
        appearance_processor: Callable[[Mapping], None] | None = None,
        tile_appearance_store: TileAppearanceStore | None = None,
        mesh_executor: Executor | None = None,
        mesh_tile_batch_size: int = MESH_TILE_BATCH_SIZE,
    ) -> None:
        if checkpoint_batch_frames < 1:
            raise ValueError("checkpoint_batch_frames must be positive")
        if checkpoint_interval_s <= 0:
            raise ValueError("checkpoint_interval_s must be positive")
        if isinstance(mesh_tile_batch_size, bool) or int(mesh_tile_batch_size) < 1:
            raise ValueError("mesh_tile_batch_size must be a positive integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.wal = wal
        self.checkpoint_batch_frames = int(checkpoint_batch_frames)
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        self.default_floor_id = str(default_floor_id)
        self.default_zone_id = str(default_zone_id)
        self.mesh_tile_batch_size = int(mesh_tile_batch_size)
        self.tile_store = ImmutableTileStore(self.root / "tiles")
        self._projects: dict[str, BuildingProject] = {}
        self._load_persisted_projects()
        self._runtimes: dict[tuple[str, str, str], _SurfaceRuntime] = {}
        self._assembly_stores: dict[str, BuildingAssemblyStore] = {}
        self._building_appearance_stores: dict[tuple[str, bool], BuildingAppearanceStore] = {}
        self._tile_appearance_store = tile_appearance_store
        self._last_meshing_error: Optional[str] = None
        self._lock = threading.RLock()
        self._mesh_lock = threading.Lock()
        if mesh_executor is not None and not callable(getattr(mesh_executor, "submit", None)):
            raise TypeError("mesh_executor must provide submit")
        self._mesh_executor = mesh_executor
        self._close_requested = threading.Event()
        self._appearance_processor = None
        self._appearance_condition = threading.Condition()
        self._appearance_pending: dict[tuple, dict] = {}
        self._appearance_active_key: Optional[tuple[str, str, str]] = None
        self._appearance_completed: set[tuple[str, str, str]] = set()
        self._appearance_retryable: dict[tuple[str, str, str], dict] = {}
        self._appearance_refresh_revision: Optional[str] = None
        self._appearance_closed = False
        self._appearance_scheduled = 0
        self._appearance_processed = 0
        self._appearance_failed = 0
        self._appearance_retried = 0
        self._appearance_replaced = 0
        self._appearance_discovered = 0
        self._appearance_last_error: Optional[str] = None
        self._appearance_schedule_failed = 0
        self._appearance_schedule_last_error: Optional[str] = None
        self._appearance_refresh_required = False
        self._appearance_thread: Optional[threading.Thread] = None
        if appearance_processor is not None:
            self.configure_appearance_processor(appearance_processor)

    def configure_appearance_processor(
        self, processor: Callable[[Mapping], None]
    ) -> None:
        if not callable(processor):
            raise TypeError("appearance processor must be callable")
        with self._appearance_condition:
            if self._appearance_processor is not None or self._appearance_thread is not None:
                raise ValueError("appearance processor is already configured")
            if self._appearance_closed:
                raise ValueError("appearance scheduler is closed")
            self._appearance_processor = processor
            self._appearance_thread = threading.Thread(
                target=self._appearance_worker,
                name="go2-tile-appearance",
                daemon=True,
            )
            self._appearance_thread.start()
        revision = self._capture_appearance_revision()
        failures_before = self._appearance_schedule_failed
        if revision is not None:
            for metadata in self.tile_store.manifest(scope="local")["tiles"]:
                if self._schedule_appearance_nonthrowing(metadata, revision):
                    self._appearance_discovered += 1
        if self._appearance_schedule_failed == failures_before:
            self._mark_appearance_refresh_complete(revision)

    def _appearance_scheduling_revision(self) -> str:
        resolver = getattr(self._appearance_processor, "scheduling_revision", None)
        revision = "0" * 64 if not callable(resolver) else str(resolver())
        if len(revision) != 64 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("appearance evidence revision must be a lowercase SHA-256")
        return revision

    def _record_appearance_schedule_failure(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        with self._appearance_condition:
            self._appearance_schedule_failed += 1
            self._appearance_failed += 1
            self._appearance_schedule_last_error = error
            self._appearance_last_error = error
            self._appearance_refresh_required = True
            self._appearance_refresh_revision = None

    def _capture_appearance_revision(self) -> Optional[str]:
        if self._appearance_processor is None:
            return None
        try:
            return self._appearance_scheduling_revision()
        except Exception as exc:
            self._record_appearance_schedule_failure(exc)
            return None

    def _schedule_appearance_nonthrowing(
        self, metadata: Mapping, evidence_revision: Optional[str]
    ) -> bool:
        if evidence_revision is None:
            return False
        job = dict(metadata)
        job["appearance_evidence_revision_id"] = evidence_revision
        try:
            return self.schedule_appearance(job)
        except Exception as exc:
            self._record_appearance_schedule_failure(exc)
            return False

    def _schedule_published_appearance(
        self, metadata: Sequence[Mapping], evidence_revision: Optional[str]
    ) -> int:
        return sum(
            int(self._schedule_appearance_nonthrowing(item, evidence_revision))
            for item in metadata
        )

    def _mark_appearance_refresh_complete(self, revision: Optional[str]) -> None:
        if revision is None:
            return
        with self._appearance_condition:
            prior_schedule_error = self._appearance_schedule_last_error
            self._appearance_refresh_revision = revision
            self._appearance_refresh_required = False
            self._appearance_schedule_last_error = None
            if self._appearance_last_error == prior_schedule_error:
                self._appearance_last_error = None

    def refresh_appearance_evidence(self) -> int:
        if self._appearance_processor is None:
            return 0
        revision = self._capture_appearance_revision()
        if revision is None:
            return 0
        with self._appearance_condition:
            if (
                revision == self._appearance_refresh_revision
                and not self._appearance_retryable
                and not self._appearance_refresh_required
            ):
                return 0
        failures_before = self._appearance_schedule_failed
        scheduled = 0
        for metadata in self.tile_store.manifest(scope="local")["tiles"]:
            scheduled += int(self._schedule_appearance_nonthrowing(metadata, revision))
        if self._appearance_schedule_failed == failures_before:
            self._mark_appearance_refresh_complete(revision)
        return scheduled

    def _load_persisted_projects(self) -> None:
        for project_root in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not project_root.is_dir() or not (project_root / "building.sqlite3").is_file():
                continue
            project = BuildingProject(project_root, building_id=project_root.name)
            self._projects[project.building_id] = project

    def _project_for_building(self, building_id: str) -> BuildingProject:
        project = self._projects.get(building_id)
        if project is None:
            project = BuildingProject(self.root / building_id, building_id=building_id)
            self._projects[building_id] = project
        return project

    def _ensure_session_route(self, frame: LiveKeyframeMessage) -> tuple[BuildingProject, str]:
        project = self._project_for_building(frame.identity.building_id)
        submap_id = project.ensure_session_route(
            session_id=frame.identity.session_id,
            map_id=frame.identity.map_id,
            producer_instance_id=frame.identity.producer_instance_id,
            source_config_sha256=frame.source_config_sha256,
            calibration_id=frame.calibration_id,
            start_sequence=frame.identity.sequence,
        )
        return project, submap_id

    def admit(self, frame: LiveKeyframeMessage) -> WalAdmission:
        with self._lock:
            self._require_open()
            project, _ = self._ensure_session_route(frame)
            admission = self.wal.admit(frame)
            if admission.accepted:
                project.route_frame(frame.identity)
            return admission

    def _runtime(self, record: WalRecord) -> _SurfaceRuntime:
        with self._lock:
            self._require_open()
            project, submap_id = self._ensure_session_route(record.frame)
            project.route_frame(record.frame.identity)
            submap = project.submap(submap_id)
            component = project.component(str(submap["component_id"]))
        return self._runtime_for_source(
            project,
            {**component, **submap},
            sensor_frame_transform_id=record.frame.sensor_frame_transform_id,
        )

    def _runtime_for_source(
        self,
        project: BuildingProject,
        source: dict,
        *,
        sensor_frame_transform_id: str | None = None,
    ) -> _SurfaceRuntime:
        source = dict(source)
        submap_id = str(source["submap_id"])
        key = (project.building_id, str(source["component_id"]), submap_id)
        with self._lock:
            self._require_open()
            runtime = self._runtimes.get(key)
            if runtime is not None:
                if (
                    sensor_frame_transform_id is not None
                    and runtime.surface.sensor_frame_transform_id != sensor_frame_transform_id
                ):
                    raise ValueError("surface runtime sensor frame transform does not match")
                return runtime

        submap_root = project.root / "submaps" / submap_id
        checkpoint_generation_id = None
        if (submap_root / "CURRENT").is_file():
            checkpoint_generation_id = (submap_root / "CURRENT").read_text(encoding="ascii").strip()
            surface = LiveSurfaceMap.load_checkpoint(
                submap_root,
                expected_map_id=source["map_id"],
                expected_source_config_sha256=source["source_config_sha256"],
                expected_generation_id=checkpoint_generation_id,
                expected_building_id=project.building_id,
                expected_session_id=source["session_id"],
                expected_floor_id=source["floor_id"],
                expected_submap_id=submap_id,
                expected_calibration_id=source["calibration_id"],
                expected_sensor_frame_transform_id=sensor_frame_transform_id,
            )
        else:
            surface = LiveSurfaceMap(
                map_id=source["map_id"],
                source_config_sha256=source["source_config_sha256"],
                sensor_frame_transform_id=sensor_frame_transform_id,
                calibration_id=source["calibration_id"],
            )
        runtime = _SurfaceRuntime(
            project=project,
            submap_id=submap_id,
            surface=surface,
            mesh_dirty_tiles=set(surface.dirty_tiles()),
            last_checkpoint_generation_id=checkpoint_generation_id,
        )
        with self._lock:
            self._require_open()
            return self._runtimes.setdefault(key, runtime)

    def restore_persisted_runtimes(self, *, latest_only: bool = False) -> int:
        if not isinstance(latest_only, bool):
            raise TypeError("latest_only must be a boolean")
        with self._lock:
            self._require_open()
            projects = tuple(self._projects.values())
            existing = set(self._runtimes)
        for project in projects:
            sources = project.surface_sources()
            if latest_only:
                sources = sources[-1:]
            for source in sources:
                submap_root = project.root / "submaps" / str(source["submap_id"])
                if (submap_root / "CURRENT").is_file():
                    self._runtime_for_source(project, source)
        with self._lock:
            self._require_open()
            return len(set(self._runtimes) - existing)

    def integrate(self, record: WalRecord, *, force_checkpoint: bool = False) -> PipelineUpdate:
        runtime = self._runtime(record)
        with runtime.lock:
            self._require_open()
            integration = runtime.surface.integrate(record.frame)
            runtime.mesh_dirty_tiles.update(integration.dirty_tiles)
            runtime.pending_records.append(record)
            if not integration.duplicate:
                runtime.frames_since_checkpoint += 1
            runtime.last_integrated_at = time.monotonic()
            # The checkpoint worker batches durable writes and flushes idle tails,
            # so producer acknowledgements do not wait for serialization.
            generation_id = self._flush(runtime) if force_checkpoint else None
            submap = runtime.project.submap(runtime.submap_id)
            return PipelineUpdate(
                building_id=record.frame.identity.building_id,
                submap_id=runtime.submap_id,
                session_id=record.frame.identity.session_id,
                floor_id=str(submap["floor_id"]),
                integrated_cells=integration.integrated_cells,
                duplicate_surface_observation=integration.duplicate,
                checkpoint_generation_id=generation_id,
            )

    def flush_due(self, *, force: bool = False) -> tuple[tuple[str, str, str], ...]:
        with self._lock:
            self._require_open()
            runtimes = tuple(self._runtimes.items())
        generations = []
        now = time.monotonic()
        for (building_id, _component_id, submap_id), runtime in runtimes:
            with runtime.lock:
                self._require_open()
                elapsed = now - runtime.last_checkpoint_at
                idle = now - runtime.last_integrated_at
                due = (
                    force
                    or (
                        elapsed >= self.checkpoint_interval_s
                        and (
                            runtime.frames_since_checkpoint >= self.checkpoint_batch_frames
                            or idle >= self.checkpoint_interval_s
                        )
                    )
                )
                if runtime.pending_records and due:
                    generations.append((building_id, submap_id, self._flush(runtime)))
        return tuple(generations)

    def recover_pending(self) -> tuple[tuple[str, str, str], ...]:
        """Replay the durable backlog and checkpoint each touched submap once."""
        touched: set[tuple[str, str]] = set()
        for record in self.wal.pending():
            update = self.integrate(record)
            touched.add((update.building_id, update.submap_id))
        if not touched:
            return ()
        generations = self.flush_due(force=True)
        return tuple(
            item for item in generations if (item[0], item[1]) in touched
        )

    def active_targets(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(sorted((building_id, submap_id) for building_id, _component_id, submap_id in self._runtimes))

    def _flush(self, runtime: _SurfaceRuntime) -> str:
        submap = runtime.project.submap(runtime.submap_id)
        checkpoint = runtime.surface.checkpoint(
            runtime.project.root / "submaps" / runtime.submap_id,
            building_id=runtime.project.building_id,
            session_id=submap["session_id"],
            floor_id=submap["floor_id"],
            submap_id=runtime.submap_id,
            calibration_id=runtime.surface.calibration_id,
        )
        streams = set()
        self.wal.commit_integrated_batch(
            [record.frame for record in runtime.pending_records],
            checkpoint_id=checkpoint.generation_id,
        )
        for record in runtime.pending_records:
            streams.add(record.frame.identity.stream_key())
        for stream in streams:
            self.wal.reclaim_integrated(stream)
        runtime.pending_records.clear()
        runtime.frames_since_checkpoint = 0
        runtime.last_checkpoint_at = time.monotonic()
        runtime.last_checkpoint_generation_id = checkpoint.generation_id
        return checkpoint.generation_id

    def mesh(self, building_id: str, submap_id: str) -> dict:
        appearance_revision = self._capture_appearance_revision()
        with self._mesh_lock:
            with self._lock:
                self._require_open()
                runtime = next((
                    runtime
                    for (candidate_building, _component_id, candidate_submap), runtime in self._runtimes.items()
                    if candidate_building == building_id and candidate_submap == submap_id
                ), None)
                if runtime is None:
                    project = self._projects.get(str(building_id))
                    if project is None:
                        raise ValueError("surface mesh building does not exist")
            if runtime is None:
                try:
                    submap = project.submap(str(submap_id))
                    component = project.component(str(submap["component_id"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("surface mesh submap does not exist") from exc
                runtime = self._runtime_for_source(project, {**component, **submap})
            try:
                # Freeze one durable generation, then release the runtime so live
                # integration can continue while the immutable checkpoint is meshed.
                with runtime.lock:
                    self._require_open()
                    if runtime.pending_records:
                        self._flush(runtime)
                    elif runtime.last_checkpoint_generation_id is None:
                        raise ValueError("surface mesh requires a durable checkpoint generation")
                    submap = runtime.project.submap(submap_id)
                    component = runtime.project.component(str(submap["component_id"]))
                    generation_id = runtime.last_checkpoint_generation_id
                    pose_revision_id = runtime.surface.pose_revision_id
                    tile_m = runtime.surface.tile_m
                    supported_tiles = set(
                        runtime.surface.supported_tiles(runtime.mesh_dirty_tiles)
                    )
                    if supported_tiles:
                        runtime.mesh_dirty_tiles.intersection_update(supported_tiles)
                        ranked_tiles = runtime.surface.rank_tiles_by_support(
                            supported_tiles
                        )
                    else:
                        ranked_tiles = runtime.surface.rank_tiles_by_support(
                            runtime.mesh_dirty_tiles
                        )
                    selected_tiles = ranked_tiles[:self.mesh_tile_batch_size]
                    frozen_surfels = runtime.surface.meshing_surfels_for_tiles(
                        set(selected_tiles),
                        halo_m=MESH_HALO_M,
                    )
                if not selected_tiles:
                    self._last_meshing_error = None
                    return {
                        "building_id": building_id,
                        "submap_id": submap_id,
                        "published_tiles": 0,
                        "triangle_count": 0,
                        "unresolved_surfel_count": 0,
                        "remaining_dirty_tiles": 0,
                        "tile_revision_ids": [],
                        "tile_metadata": [],
                    }
                mesh_kwargs = {
                    "tile_m": tile_m,
                    "mesh_halo_m": MESH_HALO_M,
                    "tile_store_root": str(self.tile_store.root),
                    "publication_building_id": building_id,
                    "publication_floor_id": str(submap["floor_id"]),
                    "publication_component_id": str(component["component_id"]),
                    "publication_attachment_state": str(component["attachment_state"]),
                    "publication_submap_id": submap_id,
                    "expected_generation_id": generation_id,
                    "expected_pose_revision_id": pose_revision_id,
                    "tile_ids": tuple(tile.as_tuple() for tile in selected_tiles),
                }
                if self._mesh_executor is None:
                    mesh_publication = _mesh_frozen_surfel_batch(
                        frozen_surfels, **mesh_kwargs
                    )
                else:
                    mesh_publication = self._mesh_executor.submit(
                        _mesh_frozen_surfel_batch, frozen_surfels, **mesh_kwargs
                    ).result()
                published = list(mesh_publication["published"])
                with runtime.lock:
                    generation_is_current = (
                        runtime.last_checkpoint_generation_id == generation_id
                        and not runtime.pending_records
                    )
                    if generation_is_current:
                        runtime.mesh_dirty_tiles.difference_update(selected_tiles)
                    remaining_dirty_tiles = len(runtime.mesh_dirty_tiles)
                self._schedule_published_appearance(published, appearance_revision)
                self._last_meshing_error = None
                return {
                    "building_id": building_id,
                    "submap_id": submap_id,
                    "published_tiles": len(published),
                    "triangle_count": int(mesh_publication["triangle_count"]),
                    "unresolved_surfel_count": int(mesh_publication["unresolved_surfel_count"]),
                    "remaining_dirty_tiles": remaining_dirty_tiles,
                    "tile_revision_ids": [item["revision_id"] for item in published],
                    "tile_metadata": published,
                }
            except Exception as exc:
                self._last_meshing_error = f"{type(exc).__name__}: {exc}"
                raise

    def neural_surface_evidence(self, building_id: str, submap_id: str) -> dict:
        """Freeze the exact measured-tile generation used to gate one DA3 window."""
        with self._mesh_lock:
            runtime = self._runtime_for_submap(building_id, submap_id)
            with runtime.lock:
                generation_id = runtime.last_checkpoint_generation_id
                pose_revision_id = runtime.surface.pose_revision_id
                tile_m = runtime.surface.tile_m
            if generation_id is None or pose_revision_id is None:
                raise ValueError("neural surface evidence requires a durable measured generation")
            tiles = []
            for metadata in self.tile_store.manifest(scope="local")["tiles"]:
                if metadata.get("building_id") != building_id or metadata.get("submap_id") != submap_id:
                    continue
                if metadata.get("source_generation_id") != generation_id:
                    raise ValueError("neural surface evidence contains a stale source generation")
                verified, body = self.tile_store.load(str(metadata["revision_id"]))
                tiles.append({"metadata": verified, "body": body})
            return {
                "schema_version": 1,
                "building_id": building_id,
                "submap_id": submap_id,
                "source_generation_id": generation_id,
                "pose_revision_id": pose_revision_id,
                "tile_m": tile_m,
                "tiles": tuple(tiles),
            }

    def publish_measured_rgbd_surface(self, building_id: str, submap_id: str, surface: object) -> dict:
        provenance = np.asarray(getattr(surface, "triangle_provenance"))
        if np.issubdtype(provenance.dtype, np.bool_) or not np.issubdtype(
            provenance.dtype, np.integer
        ):
            raise ValueError("RGB-D surface publication provenance must contain integers")
        if np.any(provenance != int(TriangleProvenance.MEASURED_RGBD)):
            raise ValueError("RGB-D surface publication accepts only MEASURED_RGBD triangles")
        return self._publish_surface_faces(
            building_id,
            submap_id,
            positions=np.asarray(getattr(surface, "positions")),
            normals=np.asarray(getattr(surface, "normals")),
            triangles=np.asarray(getattr(surface, "triangles")),
            triangle_scale_m=np.asarray(getattr(surface, "triangle_scale_m")),
            triangle_provenance=provenance,
            triangle_uncertainty_m=np.asarray(getattr(surface, "triangle_uncertainty_m")),
            expected_source_generation_id=None,
        )

    def publish_neural_surface(
        self,
        building_id: str,
        submap_id: str,
        candidate: NeuralSurfaceCandidate,
        decision: NeuralSurfaceDecision,
        *,
        expected_source_generation_id: str,
    ) -> dict:
        if not isinstance(candidate, NeuralSurfaceCandidate):
            raise TypeError("candidate must be a NeuralSurfaceCandidate")
        if not isinstance(decision, NeuralSurfaceDecision):
            raise TypeError("decision must be a NeuralSurfaceDecision")
        selected = decision.validate_for(candidate)
        return self._publish_surface_faces(
            building_id,
            submap_id,
            positions=candidate.positions,
            normals=candidate.normals,
            triangles=candidate.triangles[selected],
            triangle_scale_m=candidate.triangle_scale_m[selected],
            triangle_provenance=decision.triangle_provenance,
            triangle_uncertainty_m=decision.triangle_uncertainty_m,
            expected_source_generation_id=expected_source_generation_id,
        )

    def _runtime_for_submap(self, building_id: str, submap_id: str) -> _SurfaceRuntime:
        with self._lock:
            self._require_open()
            runtime = next((
                value
                for (candidate_building, _component_id, candidate_submap), value in self._runtimes.items()
                if candidate_building == building_id and candidate_submap == submap_id
            ), None)
            project = self._projects.get(str(building_id))
        if runtime is not None:
            return runtime
        if project is None:
            raise ValueError("surface publication building does not exist")
        try:
            submap = project.submap(str(submap_id))
            component = project.component(str(submap["component_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("surface publication submap does not exist") from exc
        return self._runtime_for_source(project, {**component, **submap})

    def _publish_surface_faces(
        self,
        building_id: str,
        submap_id: str,
        *,
        positions: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        triangle_scale_m: np.ndarray,
        triangle_provenance: np.ndarray,
        triangle_uncertainty_m: np.ndarray,
        expected_source_generation_id: Optional[str],
    ) -> dict:
        positions = np.asarray(positions)
        normals = np.asarray(normals)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or normals.shape != positions.shape:
            raise ValueError("surface publication vertex arrays are invalid")
        for name, values in (("positions", positions), ("normals", normals)):
            if (
                not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.bool_)
                or np.issubdtype(values.dtype, np.complexfloating)
            ):
                raise ValueError(f"surface publication {name} must be numeric")
        positions = np.ascontiguousarray(positions, dtype="<f4")
        normals = np.ascontiguousarray(normals, dtype="<f4")
        if not np.isfinite(positions).all() or not np.isfinite(normals).all():
            raise ValueError("surface publication contains non-finite geometry")

        triangles = np.asarray(triangles)
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("surface publication triangle array is invalid")
        if np.issubdtype(triangles.dtype, np.bool_) or not np.issubdtype(
            triangles.dtype, np.integer
        ):
            raise ValueError("surface publication triangle indices must be integers")
        if len(triangles) and np.any(triangles < 0):
            raise ValueError("surface publication triangle index is negative")
        if len(triangles) and int(np.max(triangles)) >= len(positions):
            raise ValueError("surface publication triangle index is out of range")
        triangles = np.ascontiguousarray(triangles, dtype="<u4")
        count = len(triangles)

        triangle_scale_m = np.asarray(triangle_scale_m)
        triangle_provenance = np.asarray(triangle_provenance)
        triangle_uncertainty_m = np.asarray(triangle_uncertainty_m)
        if any(
            values.shape != (count,)
            for values in (
                triangle_scale_m,
                triangle_provenance,
                triangle_uncertainty_m,
            )
        ):
            raise ValueError("surface publication evidence arrays do not match triangles")
        if (
            not np.issubdtype(triangle_scale_m.dtype, np.number)
            or np.issubdtype(triangle_scale_m.dtype, np.bool_)
            or np.issubdtype(triangle_scale_m.dtype, np.complexfloating)
        ):
            raise ValueError("surface publication triangle scales must be numeric")
        triangle_scale_m = np.ascontiguousarray(triangle_scale_m, dtype="<f4")
        if not np.isfinite(triangle_scale_m).all() or np.any(triangle_scale_m <= 0):
            raise ValueError("surface publication triangle scales are invalid")
        if np.issubdtype(triangle_provenance.dtype, np.bool_) or not np.issubdtype(
            triangle_provenance.dtype, np.integer
        ):
            raise ValueError("surface publication triangle provenance must contain integers")
        if any(
            int(value) not in TRIANGLE_PROVENANCE_NAMES
            for value in triangle_provenance
        ):
            raise ValueError("surface publication triangle provenance is invalid")
        triangle_provenance = np.ascontiguousarray(triangle_provenance, dtype="u1")
        if (
            not np.issubdtype(triangle_uncertainty_m.dtype, np.number)
            or np.issubdtype(triangle_uncertainty_m.dtype, np.bool_)
            or np.issubdtype(triangle_uncertainty_m.dtype, np.complexfloating)
        ):
            raise ValueError("surface publication triangle uncertainty must be numeric")
        triangle_uncertainty_m = np.ascontiguousarray(
            triangle_uncertainty_m, dtype="<f4"
        )
        if (
            not np.isfinite(triangle_uncertainty_m).all()
            or np.any(triangle_uncertainty_m < 0)
            or np.any(triangle_uncertainty_m > 1.0)
        ):
            raise ValueError("surface publication triangle uncertainty is invalid")

        appearance_revision = self._capture_appearance_revision()
        with self._mesh_lock:
            runtime = self._runtime_for_submap(building_id, submap_id)
            # Publication lock order is mesh -> runtime -> tile-store batch.
            with runtime.lock:
                generation_id = runtime.last_checkpoint_generation_id
                pose_revision_id = runtime.surface.pose_revision_id
                tile_m = runtime.surface.tile_m
                submap = runtime.project.submap(submap_id)
                component = runtime.project.component(str(submap["component_id"]))
                if generation_id is None or pose_revision_id is None:
                    raise ValueError("surface publication requires a durable measured generation")
                if (
                    expected_source_generation_id is not None
                    and expected_source_generation_id != generation_id
                ):
                    raise ValueError("neural surface source generation is stale")
                current_tiles = {
                    tuple(int(value) for value in metadata["tile_id"]): metadata
                    for metadata in self.tile_store.manifest(scope="local")["tiles"]
                    if metadata.get("building_id") == building_id
                    and metadata.get("submap_id") == submap_id
                }
                if any(
                    metadata.get("source_generation_id") != generation_id
                    for metadata in current_tiles.values()
                ):
                    raise ValueError("surface publication current tile generation is stale")

                grouped: dict[tuple[int, int, int], np.ndarray] = {}
                if count:
                    centres = positions[triangles].mean(axis=1)
                    tile_keys = np.floor(centres / tile_m).astype(np.int64)
                    for key in sorted(
                        {tuple(int(value) for value in row) for row in tile_keys}
                    ):
                        grouped[key] = np.flatnonzero(
                            np.all(tile_keys == np.asarray(key), axis=1)
                        )

                revisions = []
                inserted_triangles = 0
                for tile_key, selected in grouped.items():
                    metadata = current_tiles.get(tile_key)
                    existing = None
                    unresolved_surfel_count = 0
                    if metadata is not None:
                        verified, body = self.tile_store.load(str(metadata["revision_id"]))
                        existing = decode_surface_tile(body)
                        unresolved_surfel_count = int(
                            verified["unresolved_surfel_count"]
                        )
                    revision, inserted = self._merge_tile_faces(
                        TileId(*tile_key),
                        existing,
                        unresolved_surfel_count,
                        positions,
                        normals,
                        triangles[selected],
                        triangle_scale_m[selected],
                        triangle_provenance[selected],
                        triangle_uncertainty_m[selected],
                    )
                    if inserted == 0:
                        continue
                    revisions.append(revision)
                    inserted_triangles += inserted

                requests = tuple(
                    TilePublicationRequest(
                        building_id=building_id,
                        floor_id=str(submap["floor_id"]),
                        component_id=str(component["component_id"]),
                        attachment_state=str(component["attachment_state"]),
                        submap_id=submap_id,
                        revision=revision,
                        source_generation_id=generation_id,
                        pose_revision_id=pose_revision_id,
                    )
                    for revision in revisions
                )
                published = list(self.tile_store.publish_batch(requests).metadata)

            self._schedule_published_appearance(published, appearance_revision)
            return {
                "building_id": building_id,
                "submap_id": submap_id,
                "source_generation_id": generation_id,
                "published_tiles": len(published),
                "triangle_count": inserted_triangles,
                "tile_revision_ids": [item["revision_id"] for item in published],
            }

    @staticmethod
    def _merge_tile_faces(
        tile_id: TileId,
        existing: object,
        unresolved_surfel_count: int,
        positions: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        scales: np.ndarray,
        provenance: np.ndarray,
        uncertainty: np.ndarray,
    ) -> tuple[MeshTileRevision, int]:
        new_face_positions = np.asarray(positions[triangles], dtype="<f4")
        new_face_normals = np.asarray(normals[triangles], dtype="<f4")
        keep_new = np.ones(len(triangles), dtype=bool)
        if existing is None:
            existing_face_positions = np.empty((0, 3, 3), dtype="<f4")
            existing_face_normals = np.empty((0, 3, 3), dtype="<f4")
            existing_scales = np.empty(0, dtype="<f4")
            existing_provenance = np.empty(0, dtype="u1")
            existing_uncertainty = np.empty(0, dtype="<f4")
        else:
            existing_face_positions = np.asarray(
                existing.positions[existing.triangles], dtype="<f4"
            )
            existing_face_normals = np.asarray(
                existing.normals[existing.triangles], dtype="<f4"
            )
            existing_scales = np.asarray(existing.triangle_scale_m, dtype="<f4")
            existing_provenance = np.asarray(existing.triangle_provenance, dtype="u1")
            existing_uncertainty = np.asarray(existing.triangle_uncertainty_m, dtype="<f4")
            existing_centres = existing_face_positions.mean(axis=1)
            new_centres = new_face_positions.mean(axis=1)
            new_authority = int(np.min(provenance))
            protected = existing_provenance <= new_authority
            if np.any(protected) and len(new_centres):
                distances, _ = cKDTree(existing_centres[protected]).query(new_centres, k=1)
                keep_new &= distances > (0.018 if new_authority == int(TriangleProvenance.NEURAL_MULTIVIEW) else 0.012)
            if new_authority == int(TriangleProvenance.MEASURED_RGBD) and np.any(keep_new):
                replacing = existing_provenance == int(TriangleProvenance.NEURAL_MULTIVIEW)
                if np.any(replacing):
                    distances, _ = cKDTree(new_centres[keep_new]).query(existing_centres[replacing], k=1)
                    keep_existing = np.ones(len(existing_provenance), dtype=bool)
                    keep_existing[np.flatnonzero(replacing)[distances <= 0.020]] = False
                    existing_face_positions = existing_face_positions[keep_existing]
                    existing_face_normals = existing_face_normals[keep_existing]
                    existing_scales = existing_scales[keep_existing]
                    existing_provenance = existing_provenance[keep_existing]
                    existing_uncertainty = existing_uncertainty[keep_existing]
        new_face_positions = new_face_positions[keep_new]
        new_face_normals = new_face_normals[keep_new]
        scales = np.asarray(scales, dtype="<f4")[keep_new]
        provenance = np.asarray(provenance, dtype="u1")[keep_new]
        uncertainty = np.asarray(uncertainty, dtype="<f4")[keep_new]
        inserted = len(new_face_positions)
        face_positions = np.concatenate((existing_face_positions, new_face_positions))
        face_normals = np.concatenate((existing_face_normals, new_face_normals))
        face_scales = np.concatenate((existing_scales, scales))
        face_provenance = np.concatenate((existing_provenance, provenance))
        face_uncertainty = np.concatenate((existing_uncertainty, uncertainty))
        order = np.argsort(face_provenance, kind="stable")
        face_positions = face_positions[order]
        face_normals = face_normals[order]
        face_scales = face_scales[order]
        face_provenance = face_provenance[order]
        face_uncertainty = face_uncertainty[order]
        merged_positions = np.ascontiguousarray(face_positions.reshape(-1, 3), dtype="<f4")
        merged_normals = np.ascontiguousarray(face_normals.reshape(-1, 3), dtype="<f4")
        merged_triangles = np.arange(len(merged_positions), dtype="<u4").reshape(-1, 3)
        digest = hashlib.sha256()
        digest.update(json.dumps(tile_id.as_tuple(), separators=(",", ":")).encode("ascii"))
        for values in (
            merged_positions,
            merged_normals,
            merged_triangles,
            face_scales,
            face_provenance,
            face_uncertainty,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        return MeshTileRevision(
            tile_id,
            digest.hexdigest(),
            merged_positions,
            merged_normals,
            merged_triangles,
            np.asarray(face_scales, dtype="<f4"),
            int(unresolved_surfel_count),
            np.asarray(face_provenance, dtype="u1"),
            np.asarray(face_uncertainty, dtype="<f4"),
        ), inserted

    def schedule_appearance(self, metadata: Mapping) -> bool:
        """Queue at most one pending appearance generation for each tile."""

        if self._appearance_processor is None:
            return False
        job = dict(metadata)
        revision = str(job.get("revision_id", ""))
        generation = str(job.get("source_generation_id", ""))
        evidence_revision = str(
            job.get("appearance_evidence_revision_id")
            or self._appearance_scheduling_revision()
        )
        if len(revision) != 64 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("appearance tile revision must be a lowercase SHA-256")
        if len(generation) != 64 or any(value not in "0123456789abcdef" for value in generation):
            raise ValueError("appearance source generation must be a lowercase SHA-256")
        if len(evidence_revision) != 64 or any(value not in "0123456789abcdef" for value in evidence_revision):
            raise ValueError("appearance evidence revision must be a lowercase SHA-256")
        job["appearance_evidence_revision_id"] = evidence_revision
        key = (revision, generation, evidence_revision)
        slot = self._appearance_slot(job)
        with self._appearance_condition:
            if self._appearance_closed:
                return False
            if key in self._appearance_completed or key == self._appearance_active_key:
                return False
            pending = self._appearance_pending.get(slot)
            pending_key = None if pending is None else (
                pending["revision_id"],
                pending["source_generation_id"],
                pending["appearance_evidence_revision_id"],
            )
            if key == pending_key:
                return False
            if pending is not None:
                self._appearance_replaced += 1
            self._appearance_pending[slot] = job
            self._appearance_retryable.pop(key, None)
            self._appearance_scheduled += 1
            self._appearance_condition.notify_all()
            return True

    @staticmethod
    def _appearance_slot(job: Mapping) -> tuple:
        scope_fields = (
            "building_id",
            "floor_id",
            "component_id",
            "submap_id",
            "attachment_state",
        )
        tile_id = job.get("tile_id")
        if all(job.get(name) is not None for name in scope_fields) and isinstance(
            tile_id, (list, tuple)
        ) and len(tile_id) == 3:
            return tuple(str(job[name]) for name in scope_fields) + tuple(
                int(value) for value in tile_id
            )
        return ("revision", str(job["revision_id"]))

    def _appearance_worker(self) -> None:
        while True:
            with self._appearance_condition:
                while not self._appearance_pending and not self._appearance_closed:
                    self._appearance_condition.wait()
                if self._appearance_closed and not self._appearance_pending:
                    return
                slot, job = next(iter(self._appearance_pending.items()))
                del self._appearance_pending[slot]
                key = (
                    job["revision_id"],
                    job["source_generation_id"],
                    job["appearance_evidence_revision_id"],
                )
                self._appearance_active_key = key
            succeeded = False
            try:
                processor_job = dict(job)
                attempt = int(processor_job.pop("_appearance_attempt", 0))
                self._appearance_processor(processor_job)
                succeeded = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            with self._appearance_condition:
                if succeeded:
                    self._appearance_completed.add(key)
                    self._appearance_retryable.pop(key, None)
                    self._appearance_processed += 1
                    if self._appearance_schedule_last_error is None:
                        self._appearance_last_error = None
                else:
                    self._appearance_failed += 1
                    self._appearance_last_error = error
                    if attempt < 2 and not self._appearance_closed:
                        retry = dict(job)
                        retry["_appearance_attempt"] = attempt + 1
                        self._appearance_pending.setdefault(slot, retry)
                        self._appearance_retried += 1
                    else:
                        self._appearance_retryable[key] = dict(job)
                self._appearance_active_key = None
                self._appearance_condition.notify_all()

    def appearance_snapshot(self) -> dict:
        with self._appearance_condition:
            return {
                "enabled": self._appearance_processor is not None,
                "capacity": 1,
                "pending_jobs": len(self._appearance_pending),
                "active_jobs": int(self._appearance_active_key is not None),
                "scheduled_jobs": self._appearance_scheduled,
                "processed_jobs": self._appearance_processed,
                "failed_jobs": self._appearance_failed,
                "retried_jobs": self._appearance_retried,
                "replaced_jobs": self._appearance_replaced,
                "discovered_jobs": self._appearance_discovered,
                "last_error": self._appearance_last_error,
                "schedule_failed_jobs": self._appearance_schedule_failed,
                "schedule_last_error": self._appearance_schedule_last_error,
                "refresh_required": self._appearance_refresh_required,
            }

    def wait_for_appearance_idle(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._appearance_condition:
            while self._appearance_pending or self._appearance_active_key is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._appearance_condition.wait(timeout=remaining)
            return True

    def close(self, *, timeout: float = 2.0) -> None:
        self._close_requested.set()
        with self._mesh_lock:
            with self._lock:
                runtimes = tuple(self._runtimes.values())
            for runtime in runtimes:
                with runtime.lock:
                    pass
        with self._appearance_condition:
            self._appearance_pending.clear()
            self._appearance_closed = True
            self._appearance_condition.notify_all()
        if self._appearance_thread is not None:
            self._appearance_thread.join(timeout=max(0.0, float(timeout)))
        if self._mesh_executor is not None:
            self._mesh_executor.shutdown(wait=True, cancel_futures=True)

    def _require_open(self) -> None:
        if self._close_requested.is_set():
            raise RuntimeError("live surface pipeline is closed")

    def close_session_source(self, building_id: str, session_id: str, cutoff_sequence: int) -> dict:
        if isinstance(cutoff_sequence, bool) or int(cutoff_sequence) < 0:
            raise ValueError("cutoff_sequence must be non-negative")
        cutoff = int(cutoff_sequence)
        with self._lock:
            self._require_open()
            project = self._project_for_building(str(building_id))
            source = project.session_source(str(session_id))
            if source["route_state"] == "CLOSED":
                if source["closing_cutoff_sequence"] != cutoff:
                    raise ValueError("session close cutoff conflicts with durable closed source")
                return source
        runtime = self._runtime_for_source(project, source)
        with self._lock:
            self._require_open()
            source = project.session_source(str(session_id))
            stream = StreamKey(project.building_id, str(source["session_id"]), str(source["map_id"]), str(source["producer_instance_id"]))
            highest = [
                value
                for value in (self.wal.highest_sequence(stream), project.highest_routed_sequence(str(session_id)))
                if value is not None
            ]
            if highest and max(highest) > cutoff:
                raise ValueError("close cutoff is below an admitted or routed sequence")
            source = project.begin_session_close(str(session_id), cutoff)
        with runtime.lock:
            self._require_open()
            pending = [
                record
                for record in self.wal.pending()
                if record.frame.identity.stream_key() == stream and record.frame.identity.sequence <= cutoff
            ]
            pending_keys = {record.frame.identity.as_tuple() for record in runtime.pending_records}
            for record in pending:
                integration = runtime.surface.integrate(record.frame)
                runtime.mesh_dirty_tiles.update(integration.dirty_tiles)
                if record.frame.identity.as_tuple() not in pending_keys:
                    runtime.pending_records.append(record)
                    pending_keys.add(record.frame.identity.as_tuple())
            checkpoint_id = self._flush(runtime)
            surfels = runtime.surface.surfels()
            if not surfels:
                raise ValueError("cannot close an empty surface source")
            centroids = [surfel.centroid for surfel in surfels]
            bounds_min = tuple(float(min(point[index] for point in centroids)) for index in range(3))
            bounds_max = tuple(float(max(point[index] for point in centroids)) for index in range(3))
            closed = project.complete_session_close(
                runtime.submap_id,
                generation_id=checkpoint_id,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                cutoff_sequence=cutoff,
            )
            return closed

    def snapshot(self) -> dict:
        with self._lock:
            runtimes = tuple(self._runtimes.values())
            projects = tuple(self._projects.values())
            meshing_error = self._last_meshing_error
        surface_states = [runtime.surface.snapshot() for runtime in runtimes]
        sensor_frame_transform_ids = sorted(
            {state["sensor_frame_transform_id"] for state in surface_states}
        )
        project_states = [project.snapshot() for project in projects]
        tile_manifest = self.tile_store.manifest()
        primary = project_states[0] if len(project_states) == 1 else None
        return {
                "building_id": None if primary is None else primary["building_id"],
                "state": "WAITING_FOR_SURFACE" if not surface_states else "BUILDING",
                "quality": "PREVIEW_UNCALIBRATED",
                "surface_authority": "LIVE_PROVISIONAL_SURFACE",
                "geometry_authority": False,
                "delivery_authorized": False,
                "sensor_frame_transform_id": (
                    sensor_frame_transform_ids[0] if len(sensor_frame_transform_ids) == 1 else None
                ),
                "sensor_frame_transform_ids": sensor_frame_transform_ids,
                "building_count": len(project_states),
                "floor_count": sum(state["floor_count"] for state in project_states),
                "zone_count": sum(state["zone_count"] for state in project_states),
                "submap_count": sum(state["submap_count"] for state in project_states),
                "active_submap_id": None if primary is None else primary["active_submap_id"],
                "active_component_id": None if primary is None else primary["active_component_id"],
                "active_attachment_state": None if primary is None else primary["active_attachment_state"],
                "root_component_id": None if primary is None else primary["root_component_id"],
                "detached_component_count": sum(state["detached_component_count"] for state in project_states),
                "active_floor_id": None if primary is None else primary["active_floor_id"],
                "active_zone_id": None if primary is None else primary["active_zone_id"],
                "surfel_count": sum(state["surfel_count"] for state in surface_states),
                "supported_surfel_count": sum(state["supported_surfel_count"] for state in surface_states),
                "unresolved_surfel_count": sum(state["unresolved_surfel_count"] for state in surface_states),
                "tile_count": len(tile_manifest["tiles"]),
                "triangle_count": sum(tile["triangle_count"] for tile in tile_manifest["tiles"]),
                "meshing_error": meshing_error,
                "appearance": self.appearance_snapshot(),
        }

    def manifest(self, *, scope: str = "building") -> dict:
        scope = str(scope).lower()
        if scope not in {"local", "building", "diagnostic"}:
            raise ValueError("tile manifest scope is invalid")
        if scope == "building":
            with self._lock:
                building_id = next(iter(self._projects)) if len(self._projects) == 1 else None
            if building_id is not None:
                assembly = self.current_assembly(building_id)
                if assembly is not None:
                    return {"building_scan": self.snapshot(), **self.tile_store.manifest_for_assembly(assembly)}
            return {
                "building_scan": self.snapshot(),
                "schema_version": 3,
                "scope": "building",
                "tiles": [],
            }
        return {"building_scan": self.snapshot(), **self.tile_store.manifest(scope=scope)}

    def _assembly_store(self, building_id: str) -> BuildingAssemblyStore:
        building_id = str(building_id)
        existing = self._assembly_stores.get(building_id)
        if existing is not None:
            return existing
        project = self._project_for_building(building_id)
        relocalization = RelocalizationStore(project.root / "relocalization", project=project)
        existing = BuildingAssemblyStore(
            project.root / "assembly",
            project=project,
            relocalization=relocalization,
        )
        self._assembly_stores[building_id] = existing
        return existing

    def publish_assembly(
        self,
        building_id: str,
        members: tuple[AssemblyMember, ...],
        *,
        algorithm_version: str,
    ) -> BuildingAssembly:
        with self._lock:
            self._require_open()
            return self._assembly_store(building_id).publish(members, algorithm_version=algorithm_version)

    def _building_appearance_store(
        self, building_id: str, *, exact_assembly: bool = False
    ) -> BuildingAppearanceStore:
        building_id = str(building_id)
        key = (building_id, bool(exact_assembly))
        existing = self._building_appearance_stores.get(key)
        if existing is not None:
            return existing
        if self._tile_appearance_store is None:
            raise ValueError("tile appearance store is not configured")
        project = self._project_for_building(building_id)

        def resolve_assembly(assembly_id: str):
            try:
                assembly = self._assembly_store(building_id).load(assembly_id)
            except ValueError:
                return None
            if assembly.building_id != building_id:
                return None
            if exact_assembly:
                return assembly
            current = self._assembly_store(building_id).current()
            return (
                assembly
                if current is not None and current.assembly_id == assembly.assembly_id
                else None
            )

        existing = BuildingAppearanceStore(
            project.root,
            assembly_resolver=resolve_assembly,
            tile_manifest_resolver=self.tile_store.manifest_for_assembly,
            tile_resolver=self.tile_store.load,
            tile_appearance_resolver=self._tile_appearance_store.resolve,
        )
        self._building_appearance_stores[key] = existing
        return existing

    def publish_building_appearance(
        self,
        building_id: str,
        assembly_id: str,
        members: Sequence[BuildingAppearanceMember],
    ) -> PublishedBuildingAppearance:
        with self._lock:
            self._require_open()
            current = self._assembly_store(building_id).current()
            if current is None or current.assembly_id != assembly_id:
                raise ValueError("Building appearance requires the current assembly")
            return self._building_appearance_store(building_id).publish(
                assembly=current, members=tuple(members)
            )

    def current_building_appearance(
        self,
        building_id: str,
        *,
        expected_assembly_id: str | None = None,
        expected_source_generation_id: str | None = None,
    ) -> PublishedBuildingAppearance | None:
        with self._lock:
            current_assembly = self._assembly_store(building_id).current()
            if current_assembly is None:
                return None
            if (
                expected_assembly_id is not None
                and expected_assembly_id != current_assembly.assembly_id
            ):
                raise ValueError("Building appearance expected assembly is stale")
            published = self._building_appearance_store(building_id).current(
                expected_assembly_id=current_assembly.assembly_id
            )
            if published is None:
                return None
            if expected_source_generation_id is not None and expected_source_generation_id not in {
                member["source_generation_id"] for member in published.manifest["members"]
            }:
                raise ValueError("Building appearance source generation is stale")
            return published

    def resolve_building_appearance(
        self, building_id: str, building_appearance_revision_id: str
    ) -> PublishedBuildingAppearance | None:
        with self._lock:
            try:
                return self._building_appearance_store(
                    building_id, exact_assembly=True
                ).resolve(
                    building_appearance_revision_id
                )
            except (FileNotFoundError, ValueError):
                return None

    def current_assembly(self, building_id: str, assembly_id: Optional[str] = None) -> Optional[BuildingAssembly]:
        with self._lock:
            store = self._assembly_store(building_id)
            return store.load(assembly_id) if assembly_id is not None else store.current()
