import asyncio
import base64
from concurrent.futures import ProcessPoolExecutor
import contextlib
from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import quote
import zlib
from typing import Any

import numpy as np

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .telemetry import SessionTelemetry
from .camera_assist import CameraAssistError, CameraAssistState
from .coarse_map import COARSE_MAP_TEMPLATES, CoarseBuildingStore, CoarseMapStore, FURNITURE_CATALOG, _MISSING, build_task_graph, merge_room_polygons, split_room_polygon, update_task_record, validate_coarse_map
from .mission import MissionStore, audit_mission_route, compile_building_mission
from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import (
    NavigationBridgeError,
    NavigationBridgeStore,
    compile_navigation_goal,
)
from .derived_surface import (
    DerivedSurfacePreview,
    load_automatic_reconstruction_preview,
    load_derived_surface_preview,
    load_session_odom_local_preview,
    publish_session_odom_local_preview,
)
from .legacy_session import LegacySessionPreview, load_legacy_session_preview
from .live_transport import LiveTransportMetrics
from .progressive_transfer import ProgressiveTransferTelemetry
from .robot_session import RobotSessionTelemetry
from .room_scan import (
    RoomScanConflict,
    RoomScanError,
    RoomScanInvalid,
    RoomScanService,
    RoomScanUnauthorized,
)
from mapping.pc.go2_mapping_vla.offline import (
    MobileVLADecisionError,
    OfflineVLAAdapter,
    decision_to_navigation_payload,
    parse_mobilevla_decision,
)
from mapping.pc.go2_mapping_vla.navigation_planner import plan_mobilevla_navigation
from mapping.pc.go2_mapping_locomotion.backends import available_backends
from .frontier import plan_frontier_candidates
from .nav2_map import build_nav2_occupancy_map, write_nav2_map
from .stair_survey import StairSurveyStore, aggregate_stair_observations, extract_stair_profile_from_points
from mapping.pc.go2_mapping_quality import MappingQualityGate
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, is_sha256, load_reconstruction_config
from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_live.high_fidelity_mapper import HighFidelityMapper
from mapping.pc.go2_mapping_live.live_protocol import (
    MAX_MESSAGE_BYTES,
    LiveProtocolError,
    build_camera_preview_ack,
    build_live_keyframe_ack,
    build_surface_source_status_ack,
    build_texture_keyframe_ack,
    validate_camera_preview,
    validate_live_keyframe,
    validate_surface_source_status,
    validate_texture_keyframe,
)
from mapping.pc.go2_mapping_live.live_pipeline import LiveSurfacePipeline, PipelineUpdate
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal, WalRecord
from mapping.pc.go2_mapping_live.texture_wal import TextureKeyframeWal, TextureWalRecord
from mapping.pc.go2_mapping_live.trajectory_store import TrajectoryStore
from mapping.pc.go2_mapping_live.progressive_ingest import VerifiedArtifactIngestor
from mapping.pc.go2_mapping_live.progressive_mapper import Keyframe, KeyframeError, ProgressiveMapper
from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
from mapping.pc.go2_mapping_rgbd import (
    ArtifactRgbdRefiner,
    create_depth_anything_worker_from_environment,
)
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore
from mapping.pc.go2_mapping_appearance import (
    HistoricalAppearanceStore,
    TileAppearanceStore,
    TileVertexColorStore,
    ValidatedTileAppearanceProcessor,
    build_historical_vertex_color_sidecar,
    measured_surface_identity,
)
from mapping.pc.go2_mapping_appearance.provisional_store import ProvisionalTextureStore
from mapping.pc.go2_mapping_appearance.render_manifest import AtomicRenderManifestStore
from mapping.pc.go2_mapping_appearance.glb_export import (
    AppearanceExportSources,
    export_building_glb,
    export_local_tile_glb,
)
from mapping.pc.go2_mapping_rgbd.augmentation_store import RgbdAugmentationStore
from mapping.pc.go2_mapping_calibration import (
    CalibrationCaptureState,
    TargetDetector,
    TargetSpecError,
    go2_lab_board_spec,
)

ROOT = Path(__file__).parent
_runtime_config_sha256 = os.environ.get("GO2_MAPPING_RUNTIME_CONFIG_SHA256")
_automatic_reconstruction_pointer_protocol = "automatic-reconstruction-pointer-v1"
CAMERA_PROFILE = json.loads(
    (ROOT.parents[1] / "config" / "go2edu_realsense_d435i_camera_profile.json").read_text(encoding="utf-8")
)
_color_info = CAMERA_PROFILE["color_camera_info"]
calibration_capture = CalibrationCaptureState(
    TargetDetector(
        go2_lab_board_spec(),
        camera_matrix=np.asarray(_color_info["k"], dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(_color_info["d"], dtype=np.float64),
    ),
    expected_size=(int(_color_info["width"]), int(_color_info["height"])),
)
telemetry = SessionTelemetry()
camera_assist = CameraAssistState()
live_transport = LiveTransportMetrics()
quality_gate = MappingQualityGate.from_environment()
_depth_scale_value = os.environ.get("GO2_MAPPING_VERIFIED_DEPTH_SCALE_M")
_depth_scale_m = None if not _depth_scale_value else float(_depth_scale_value)
_reconstruction_root = Path(
    os.environ.get("GO2_MAPPING_RECONSTRUCTION_ROOT", ROOT.parents[2] / "output" / "reconstruction")
)
_trajectory_store = TrajectoryStore(
    os.environ.get("GO2_MAPPING_TRAJECTORY_STORE", _reconstruction_root / "trajectories.sqlite3")
)
_provisional_texture_store = ProvisionalTextureStore(
    os.environ.get("GO2_MAPPING_PROVISIONAL_TEXTURE_ROOT", _reconstruction_root / "provisional-textures")
)
_render_manifest_store = AtomicRenderManifestStore(
    os.environ.get("GO2_MAPPING_RENDER_MANIFEST_ROOT", _reconstruction_root / "render-manifests"),
    provisional_store=_provisional_texture_store,
)
_augmentation_store = RgbdAugmentationStore(_reconstruction_root)
_depth_anything_worker = create_depth_anything_worker_from_environment(
    os.environ,
    augmentation_store=_augmentation_store,
    assistance_config_sha256=load_reconstruction_config().appearance_sha256,
)
rgbd_refiner = ArtifactRgbdRefiner(
    os.environ.get("GO2_MAPPING_CALIBRATION_MANIFEST"),
    depth_scale_m=_depth_scale_m,
    depth_anything_worker=_depth_anything_worker,
)
_display_point_budget = int(os.environ.get("GO2_MAPPING_DISPLAY_POINT_BUDGET", "30000"))
_live_scan_voxel_m = float(os.environ.get("GO2_MAPPING_LIVE_SCAN_VOXEL_M", "0.02"))
_live_scan_display_point_budget = int(os.environ.get("GO2_MAPPING_LIVE_SCAN_DISPLAY_POINT_BUDGET", "50000"))
mapper = ProgressiveMapper(
    voxel_m=_live_scan_voxel_m,
    preview_point_budget=_live_scan_display_point_budget,
)
dense_mapper = HighFidelityMapper(
    display_point_budget=_display_point_budget,
    quality_snapshot=quality_gate.snapshot,
    rgbd_refiner=rgbd_refiner,
)
_derived_surface_value = os.environ.get("GO2_MAPPING_DERIVED_SURFACE_JSON")
_derived_surface_preview: DerivedSurfacePreview | None = (
    load_derived_surface_preview(_derived_surface_value) if _derived_surface_value else None
)
_automatic_reconstruction_status_path = os.environ.get("GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_STATUS_PATH")
_automatic_reconstruction_root_value = os.environ.get("GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_ROOT")
if _automatic_reconstruction_status_path and not _automatic_reconstruction_root_value:
    raise ValueError("GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_ROOT is required with the status path")
_automatic_reconstruction_root = (
    None
    if not _automatic_reconstruction_root_value
    else Path(_automatic_reconstruction_root_value).resolve(strict=False)
)
if _derived_surface_preview is not None and _automatic_reconstruction_status_path:
    raise ValueError("static derived surface and automatic reconstruction modes are mutually exclusive")
_automatic_reconstruction_preview: DerivedSurfacePreview | None = None
_automatic_reconstruction_status: dict | None = None
_automatic_reconstruction_lock = threading.RLock()
_session_odom_local_status_path = os.environ.get("GO2_MAPPING_SESSION_ODOM_LOCAL_STATUS_PATH")
_session_odom_local_root_value = os.environ.get("GO2_MAPPING_SESSION_ODOM_LOCAL_ROOT")
if _session_odom_local_status_path and not _session_odom_local_root_value:
    raise ValueError("GO2_MAPPING_SESSION_ODOM_LOCAL_ROOT is required with the local status path")
_session_odom_local_root = (
    None
    if not _session_odom_local_root_value
    else Path(_session_odom_local_root_value).resolve(strict=False)
)
_session_odom_local_preview: DerivedSurfacePreview | None = None
_session_odom_local_status: dict | None = None
_session_odom_local_lock = threading.RLock()
_legacy_session_value = os.environ.get("GO2_MAPPING_LEGACY_SESSION_ROOT")
if _derived_surface_preview is not None and _legacy_session_value:
    raise ValueError("derived surface and legacy session preview modes are mutually exclusive")
_legacy_surface_preview: LegacySessionPreview | None = (
    load_legacy_session_preview(_legacy_session_value) if _legacy_session_value else None
)
_coarse_map_store = CoarseMapStore(
    os.environ.get("GO2_MAPPING_COARSE_MAP_PATH", ROOT.parents[2] / "mapping-data" / "pc-stack" / "coarse-map.json")
)
_coarse_building_store = CoarseBuildingStore(
    os.environ.get("GO2_MAPPING_COARSE_BUILDING_PATH", ROOT.parents[2] / "mapping-data" / "pc-stack" / "coarse-building.json")
)


def _coarse_store_for_map(map_id: str) -> CoarseMapStore:
    """Resolve the legacy floor file or an independently persisted floor map."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", map_id, re.ASCII) is None:
        raise ValueError("coarse map_id is invalid")
    if _coarse_map_store.load().get("map_id") == map_id:
        return _coarse_map_store
    return CoarseMapStore(_coarse_map_store.path.parent / "coarse-floors" / f"{map_id}.json")


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        if min(y0, y1) <= y <= max(y0, y1) and min(x0, x1) <= x <= max(x0, x1):
            cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
            if abs(cross) <= 1e-8:
                return True
        if (y0 > y) != (y1 > y):
            intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < intersection_x:
                inside = not inside
    return inside


def _validate_building_connection_endpoints(building: dict[str, Any]) -> None:
    floor_by_id = {str(floor.get("floor_id")): floor for floor in building.get("floors") or [] if isinstance(floor, dict)}
    for connection in building.get("connections") or []:
        if not isinstance(connection, dict) or not connection.get("from_anchor") or not connection.get("to_anchor"):
            continue
        for floor_id, anchor, label in ((connection.get("from_floor_id"), connection.get("from_anchor"), "from_anchor"), (connection.get("to_floor_id"), connection.get("to_anchor"), "to_anchor")):
            floor = floor_by_id.get(str(floor_id))
            if not floor:
                continue
            map_id = str(floor.get("map_id") or "")
            floor_map = _coarse_store_for_map(map_id).load_canonical(map_id)
            stair_polygons = [room.get("polygon") for room in floor_map.get("rooms", []) if isinstance(room, dict) and room.get("kind") == "stair"]
            if stair_polygons and not any(_point_in_polygon(anchor, polygon) for polygon in stair_polygons):
                raise ValueError(f"connection {connection.get('id', '')} {label} must lie inside a stair zone")
_nav2_map_root = Path(
    os.environ.get(
        "GO2_MAPPING_NAV2_MAP_ROOT",
        ROOT.parents[2] / "mapping-data" / "pc-stack" / "nav2-map",
    )
).resolve(strict=False)
_mission_store = MissionStore(
    os.environ.get("GO2_MAPPING_MISSION_ROOT", ROOT.parents[2] / "mapping-data" / "pc-stack" / "missions")
)
_navigation_bridge_store = NavigationBridgeStore(
    os.environ.get("GO2_MAPPING_NAVIGATION_ROOT", ROOT.parents[2] / "mapping-data" / "pc-stack" / "navigation")
)
_stair_survey_store = StairSurveyStore(
    os.environ.get("GO2_MAPPING_STAIR_SURVEY_PATH", ROOT.parents[2] / "mapping-data" / "pc-stack" / "stair-survey-current.json")
)
_capture_inbox_root = Path(
    os.environ.get("GO2_MAPPING_CAPTURE_INBOX_ROOT", ROOT.parents[2] / "mapping-data" / "inbox")
).resolve(strict=False)
_mobilevla_manifest_path = Path(
    os.environ.get("GO2_MAPPING_MOBILEVLA_MANIFEST", ROOT.parents[2] / "mapping" / "config" / "mobilevla_r1_manifest.json")
)
try:
    _mobilevla_manifest = json.loads(_mobilevla_manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    _mobilevla_manifest = {}
_mobilevla_preferred_variant = _mobilevla_manifest.get("preferred_variant", "sft")
if _mobilevla_preferred_variant not in {"sft", "rl"}:
    _mobilevla_preferred_variant = "sft"
_mobilevla_checkpoint_path = Path(
    os.environ.get(
        "GO2_MAPPING_MOBILEVLA_CHECKPOINT",
        ROOT.parents[2]
        / "models"
        / "mobilevla-r1-download"
        / "extracted"
        / "weight"
        / _mobilevla_preferred_variant,
    )
)
_mobilevla_adapter = OfflineVLAAdapter(_mobilevla_checkpoint_path)

_SEGMENT_ID_RE = re.compile(r"^segment-[0-9]{8}T[0-9]{6}Z$", re.ASCII)
_SEGMENT_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})\s+\*?(.+)$", re.ASCII)


def _segment_id(value: str) -> str:
    if not isinstance(value, str) or _SEGMENT_ID_RE.fullmatch(value) is None:
        raise HTTPException(status_code=422, detail="segment_id must be a canonical segment timestamp")
    return value


def _segment_quarantine_path(segment_id: str) -> Path:
    return (_reconstruction_root / "segment-quarantine" / f"{segment_id}.json").resolve(strict=False)


def _read_optional_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _segment_record(segment_id: str) -> dict | None:
    segment_id = _segment_id(segment_id)
    worker_status = automatic_reconstruction_status()
    local_status = session_odom_local_status()
    session = robot_session.snapshot()
    source_root = (_capture_inbox_root / "lidar-imu-rgbd" / segment_id).resolve(strict=False)
    source_parts = sorted(source_root.glob("*.db3.zstd")) if source_root.is_dir() else []
    worker_matches = isinstance(worker_status, dict) and worker_status.get("session_id") == segment_id
    local_matches = isinstance(local_status, dict) and local_status.get("session_id") == segment_id
    session_matches = session.get("session_id") == segment_id
    quarantine_path = _segment_quarantine_path(segment_id)
    quarantine = _read_optional_json(quarantine_path) if quarantine_path.is_file() else None
    if not (source_parts or worker_matches or local_matches or session_matches or quarantine):
        return None
    status = worker_status if worker_matches else local_status if local_matches else {}
    state = "QUARANTINED" if quarantine else status.get("state")
    if not state and session_matches:
        state = session.get("state")
    if not state:
        state = "ARTIFACTS_PRESENT" if source_parts else "EVIDENCE_UNKNOWN"
    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "state": state,
        "mode": status.get("mode") or "recorded_session",
        "coordinate_scope": status.get("coordinate_scope") or ("SESSION_ODOM_LOCAL" if local_matches else "UNKNOWN"),
        "quality_state": status.get("quality") or status.get("quality_state") or "UNKNOWN",
        "source_topic": status.get("source_topic") or "/go2/lio/lidar",
        "verified_artifact_count": status.get("verified_artifacts", status.get("artifacts", len(source_parts))),
        "remaining_verified_artifacts": status.get("remaining_verified_artifacts"),
        "final_manifest_state": status.get("final_manifest_state", "ABSENT"),
        "source_root": str(source_root) if source_parts else None,
        "quarantine": quarantine,
        "read_only": True,
    }


def _segment_ids() -> list[str]:
    candidates: set[str] = set()
    source_root = _capture_inbox_root / "lidar-imu-rgbd"
    if source_root.is_dir():
        candidates.update(path.name for path in source_root.iterdir() if path.is_dir() and _SEGMENT_ID_RE.fullmatch(path.name))
    for value in (
        robot_session.snapshot().get("session_id"),
        (automatic_reconstruction_status() or {}).get("session_id"),
        (session_odom_local_status() or {}).get("session_id"),
    ):
        if isinstance(value, str) and _SEGMENT_ID_RE.fullmatch(value):
            candidates.add(value)
    quarantine_root = _reconstruction_root / "segment-quarantine"
    if quarantine_root.is_dir():
        candidates.update(path.stem for path in quarantine_root.glob("segment-*.json") if _SEGMENT_ID_RE.fullmatch(path.stem))
    return sorted(value for value in candidates if _segment_record(value) is not None)


def _segment_quality_record(segment_id: str) -> dict:
    record = _segment_record(segment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="reconstruction segment is unavailable")
    status = automatic_reconstruction_status() or {}
    if status.get("session_id") != segment_id:
        status = session_odom_local_status() or {}
    quality = status.get("quality_report") or status.get("quality_evidence") or {}
    quality_state = record.get("quality_state") or "UNKNOWN"
    blockers = list(quality.get("blockers") or quality.get("blocking_reasons") or [])
    if quality_state in {"UNKNOWN", "PREVIEW_UNCALIBRATED"}:
        blockers.append("QUALITY_EVIDENCE_UNAVAILABLE")
    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "decision": "QUARANTINED" if record.get("state") == "QUARANTINED" else ("UNKNOWN" if blockers else "ACCEPTED"),
        "quality_state": quality_state,
        "state": record.get("state"),
        "coordinate_scope": record.get("coordinate_scope"),
        "clock_model": quality.get("clock_model"),
        "point_to_trajectory": quality.get("point_to_trajectory"),
        "wall_plane": quality.get("wall_plane"),
        "floor_plane": quality.get("floor_plane"),
        "blockers": list(dict.fromkeys(blockers)),
        "read_only": True,
    }


def _segment_closure_blockers(segment_id: str) -> list[str]:
    source_root = (_capture_inbox_root / "lidar-imu-rgbd" / segment_id).resolve(strict=False)
    manifest_path = (_capture_inbox_root / "lidar-imu-rgbd" / f"{segment_id}.sha256").resolve(strict=False)
    blockers: list[str] = []
    if not source_root.is_dir():
        blockers.append("SEGMENT_SOURCE_UNAVAILABLE")
        return blockers
    if not manifest_path.is_file():
        blockers.append("COMPLETE_CHECKSUM_MANIFEST_REQUIRED")
        return blockers
    try:
        lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError):
        return ["COMPLETE_CHECKSUM_MANIFEST_UNREADABLE"]
    expected: dict[str, str] = {}
    for line in lines:
        match = _SEGMENT_MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            blockers.append("CHECKSUM_MANIFEST_INVALID")
            continue
        digest, raw_path = match.groups()
        path = Path(raw_path)
        if path.name in expected or not path.name.startswith(f"{segment_id}_") or not path.name.endswith(".db3.zstd"):
            blockers.append("CHECKSUM_MANIFEST_SCOPE_INVALID")
            continue
        expected[path.name] = digest
    actual = {path.name: path for path in source_root.glob("*.db3.zstd") if path.is_file()}
    if not expected:
        blockers.append("COMPLETE_CHECKSUM_MANIFEST_REQUIRED")
    if set(expected) != set(actual):
        blockers.append("CHECKSUM_MANIFEST_INCOMPLETE")
    for name, path in actual.items():
        if name in expected:
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
                    blockers.append("CHECKSUM_MISMATCH")
            except OSError:
                blockers.append("SEGMENT_SOURCE_UNREADABLE")
    return list(dict.fromkeys(blockers))


def automatic_reconstruction_status() -> dict | None:
    """Refresh an automatic worker pointer while retaining the last valid map."""

    global _automatic_reconstruction_preview, _automatic_reconstruction_status
    if not _automatic_reconstruction_status_path:
        return None
    with _automatic_reconstruction_lock:
        try:
            preview, worker_status = load_automatic_reconstruction_preview(
                _automatic_reconstruction_status_path,
                output_root=_automatic_reconstruction_root,
            )
        except (OSError, ValueError):
            return None if _automatic_reconstruction_status is None else dict(_automatic_reconstruction_status)
        _automatic_reconstruction_preview = preview
        _automatic_reconstruction_status = dict(worker_status)
        return dict(worker_status)


def _active_derived_surface_preview() -> DerivedSurfacePreview | None:
    if _derived_surface_preview is not None:
        return _derived_surface_preview
    automatic_reconstruction_status()
    return _automatic_reconstruction_preview


def session_odom_local_status() -> dict | None:
    """Refresh the explicitly selected local-only preview without touching formal state."""

    global _session_odom_local_preview, _session_odom_local_status
    if not _session_odom_local_status_path:
        return None
    with _session_odom_local_lock:
        try:
            preview, status = load_session_odom_local_preview(
                _session_odom_local_status_path,
                output_root=_session_odom_local_root,
            )
        except (OSError, ValueError):
            return None if _session_odom_local_status is None else dict(_session_odom_local_status)
        _session_odom_local_preview = preview
        _session_odom_local_status = dict(status)
        return dict(status)
_ingest_root = os.environ.get("GO2_MAPPING_INGEST_ROOT")
_lifecycle_root = os.environ.get("GO2_MAPPING_LIFECYCLE_ROOT")
artifact_ingestor = VerifiedArtifactIngestor(
    Path(_ingest_root),
    dense_mapper,
    lifecycle_store=ArtifactLifecycleStore(Path(_lifecycle_root or _ingest_root)),
) if _ingest_root else None
# High-fidelity bag reconstruction can hold the GIL or native worker pool for
# minutes. It is opt-in here so protocol-3 admission and browser telemetry stay
# responsive; production reconstruction is owned by the isolated worker path.
_in_process_artifact_ingest_enabled = (
    os.environ.get("GO2_MAPPING_IN_PROCESS_ARTIFACT_INGEST_ENABLED", "0") == "1"
)
progressive_transfer = ProgressiveTransferTelemetry(os.environ.get("GO2_MAPPING_TRANSFER_STATUS"))
robot_session = RobotSessionTelemetry()
_room_scan_service = RoomScanService(
    operator_token_path=os.environ.get(
        "GO2_MAPPING_ROOM_SCAN_OPERATOR_TOKEN_FILE",
        ROOT.parents[2] / "mapping-data" / "secrets" / "go2-room-scan-operator-token.json",
    ),
    session_control_token_path=os.environ.get(
        "GO2_MAPPING_SESSION_CONTROL_TOKEN_FILE",
        ROOT.parents[2] / "mapping-data" / "secrets" / "go2-session-control-token",
    ),
    calibration_manifest_path=os.environ.get("GO2_MAPPING_CALIBRATION_MANIFEST"),
    robot_tailnet_ip=os.environ.get("GO2_MAPPING_ROOM_SCAN_ROBOT_IP", "192.0.2.11"),
    pc_tailnet_ip=os.environ.get("GO2_MAPPING_PC_TAILNET_IP", "192.0.2.10"),
    control_port=int(os.environ.get("GO2_MAPPING_SESSION_CONTROL_PORT", "8788")),
    readiness=lambda: readiness(),
    runtime_calibration=lambda: runtime_calibration_status(),
    session_snapshot=robot_session.snapshot,
    receiver_health_url=os.environ.get("GO2_MAPPING_RECEIVER_HEALTH_URL", "http://127.0.0.1:8000/health"),
)
reconstruction_store = LayerArtifactStore(_reconstruction_root)
_textured_export_root = _reconstruction_root / "textured-exports"
appearance_store = HistoricalAppearanceStore(
    os.environ.get("GO2_MAPPING_APPEARANCE_ROOT", _reconstruction_root / "appearance" / "historical")
)
_tile_appearance_root = Path(
    os.environ.get(
        "GO2_MAPPING_TILE_APPEARANCE_ROOT",
        _reconstruction_root / "tile-appearance",
    )
)
tile_appearance_store = TileAppearanceStore(_tile_appearance_root)
tile_vertex_color_store = TileVertexColorStore(_tile_appearance_root)
_tile_atlas_enabled = os.environ.get("GO2_MAPPING_TILE_ATLAS_ENABLED", "0") == "1"


def _tile_calibration_sha256():
    return rgbd_refiner.snapshot().get("calibration_sha256")


def _tile_appearance_processor(tile_store):
    if not _tile_atlas_enabled:
        return None
    calibration_sha256 = _tile_calibration_sha256()
    if not is_sha256(calibration_sha256) or calibration_sha256 != calibration_sha256.lower():
        return None
    return ValidatedTileAppearanceProcessor(
        tile_store=tile_store,
        rgbd_refiner=rgbd_refiner,
        appearance_store=tile_appearance_store,
        preview_store=tile_vertex_color_store,
        appearance_config_sha256=load_reconstruction_config().appearance_sha256,
        calibration_sha256_resolver=_tile_calibration_sha256,
        publication_scope="LOCAL",
        atlas_dimension=int(os.environ.get("GO2_MAPPING_TILE_ATLAS_DIMENSION", "256")),
        padding_pixels=int(os.environ.get("GO2_MAPPING_TILE_ATLAS_PADDING_PIXELS", "2")),
        mip_filter_radius_pixels=int(
            os.environ.get("GO2_MAPPING_TILE_ATLAS_MIP_FILTER_RADIUS_PIXELS", "1")
        ),
    )


def _refresh_tile_appearance_scheduler() -> bool:
    pipeline = _live_surface_pipeline
    if not _tile_atlas_enabled or pipeline is None:
        return False
    if not pipeline.appearance_snapshot()["enabled"]:
        processor = _tile_appearance_processor(pipeline.tile_store)
        if processor is None:
            return False
        pipeline.configure_appearance_processor(processor)
    pipeline.refresh_appearance_evidence()
    return True


def _require_building_authorized(published) -> None:
    authorization = published.manifest.get("publication_authorization", {})
    if authorization.get("building_authorized") is not True:
        raise HTTPException(status_code=404, detail="Building tile appearance is not authorized")
_ingest_task = None
_live_ingest_queue = None
_live_ingest_workers = []
_live_ingest_next_sequence = None
_live_ingest_last_admitted = {}
_live_ingest_last_error = None
_live_wal = None
_texture_wal = None
_texture_ingest_task = None
_texture_ingested_frames = 0
_texture_last_error = None
_texture_ingest_batch_size = int(os.environ.get("GO2_MAPPING_TEXTURE_INGEST_BATCH_SIZE", "4"))
_texture_render_refresh_interval_s = float(
    os.environ.get("GO2_MAPPING_TEXTURE_RENDER_REFRESH_INTERVAL_S", "10.0")
)
_texture_render_next_refresh = 0.0
_texture_render_pending_buildings = set()
_live_source_status_stream = None
_live_surface_pipeline = None
_typed_inflight = set()
_typed_processed = set()
_surface_mesh_queue = None
_surface_mesh_pending = set()
_surface_mesh_deferred = set()
_surface_mesh_retry_after = {}
_surface_mesh_retry_delay_s = 0.5
_snapshot_refresh_task = None
_surface_recovery_task = None
_surface_recovery_complete = True
_surface_recovery_error = None
_snapshot_cache_lock = threading.Lock()
_snapshot_cache_json = None
_snapshot_cache_generated_at = 0.0
_snapshot_cache_generation = 0
_snapshot_cache_max_age_s = float(os.environ.get("GO2_MAPPING_LEGACY_SNAPSHOT_REFRESH_INTERVAL_S", "10.0"))
_building_scan_summary_lock = threading.Lock()
_building_scan_summary = {
    "state": "WAITING_FOR_SURFACE",
    "quality": "PREVIEW_UNCALIBRATED",
    "geometry_authority": False,
    "delivery_authorized": False,
    "surfel_count": 0,
    "supported_surfel_count": 0,
    "unresolved_surfel_count": 0,
    "tile_count": 0,
    "triangle_count": 0,
    "pipeline_error": None,
}
_dashboard_telemetry_hz = float(os.environ.get("GO2_MAPPING_DASHBOARD_TELEMETRY_HZ", "15"))
_live_ingest_queue_size = int(os.environ.get("GO2_MAPPING_LIVE_INGEST_QUEUE_SIZE", "256"))
_live_wal_root = Path(os.environ.get("GO2_MAPPING_LIVE_WAL_ROOT", _reconstruction_root / "live-wal"))
_live_wal_max_bytes = int(os.environ.get("GO2_MAPPING_LIVE_WAL_MAX_BYTES", str(512 * 1024 * 1024)))
_texture_wal_root = Path(os.environ.get("GO2_MAPPING_TEXTURE_WAL_ROOT", _reconstruction_root / "texture-wal"))
_texture_wal_max_bytes = int(os.environ.get("GO2_MAPPING_TEXTURE_WAL_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))
_texture_submap_max_skew_ns = int(
    float(os.environ.get("GO2_MAPPING_TEXTURE_SUBMAP_MAX_SKEW_MS", "250")) * 1_000_000
)
_direct_rgb_texture_enabled = os.environ.get("GO2_MAPPING_DIRECT_RGB_TEXTURE", "0") == "1"
_live_building_root = Path(os.environ.get("GO2_MAPPING_BUILDING_ROOT", _reconstruction_root / "buildings"))
_surface_checkpoint_batch_frames = int(os.environ.get("GO2_MAPPING_SURFACE_CHECKPOINT_BATCH_FRAMES", "5"))
_surface_checkpoint_interval_s = float(os.environ.get("GO2_MAPPING_SURFACE_CHECKPOINT_INTERVAL_S", "2.0"))
_surface_mesh_tile_batch_size = int(os.environ.get("GO2_MAPPING_SURFACE_MESH_TILE_BATCH_SIZE", "4"))
_surface_mesh_max_tasks_per_child = int(
    os.environ.get("GO2_MAPPING_SURFACE_MESH_MAX_TASKS_PER_CHILD", "1")
)
_textured_export_root = _reconstruction_root / "exports" / "textured"
_room_direct_rgb_preview_root = Path(os.environ.get("GO2_MAPPING_ROOM_DIRECT_RGB_ROOT", _reconstruction_root / "room-direct-rgb-preview")).resolve(strict=False)
if _dashboard_telemetry_hz <= 0:
    raise ValueError("GO2_MAPPING_DASHBOARD_TELEMETRY_HZ must be positive")
if _live_ingest_queue_size < 1:
    raise ValueError("GO2_MAPPING_LIVE_INGEST_QUEUE_SIZE must be positive")
if _live_wal_max_bytes < 4 * 1024 * 1024:
    raise ValueError("GO2_MAPPING_LIVE_WAL_MAX_BYTES must be at least 4 MiB")
if _texture_wal_max_bytes < 4 * 1024 * 1024:
    raise ValueError("GO2_MAPPING_TEXTURE_WAL_MAX_BYTES must be at least 4 MiB")
if _texture_submap_max_skew_ns <= 0:
    raise ValueError("GO2_MAPPING_TEXTURE_SUBMAP_MAX_SKEW_MS must be positive")
if _texture_ingest_batch_size < 1 or _texture_render_refresh_interval_s <= 0:
    raise ValueError("texture ingest batch and render refresh interval must be positive")
if _snapshot_cache_max_age_s <= 0:
    raise ValueError("legacy snapshot refresh interval must be positive")
if _surface_checkpoint_batch_frames < 1 or _surface_checkpoint_interval_s <= 0:
    raise ValueError("surface checkpoint cadence must be positive")
if _surface_mesh_tile_batch_size < 1 or _surface_mesh_max_tasks_per_child < 1:
    raise ValueError("surface mesh memory bounds must be positive")


async def _progressive_ingest_loop() -> None:
    while True:
        if _in_process_artifact_ingest_enabled and artifact_ingestor is not None:
            await asyncio.to_thread(artifact_ingestor.ingest_once)
        await asyncio.to_thread(_refresh_tile_appearance_scheduler)
        await asyncio.sleep(0.5)


def _current_camera_from_lidar(calibration_id: str) -> tuple[np.ndarray | None, bool]:
    quality = quality_gate.snapshot()
    if quality.get("calibration_id") != calibration_id:
        return None, False
    values = quality.get("camera_from_lidar_row_major")
    if not isinstance(values, list) or len(values) != 16:
        return None, False
    try:
        transform = np.asarray(values, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None, False
    eligible = quality.get("eligible_for", {})
    authority = bool(eligible.get("rgbd_or_neural_geometry_fusion"))
    return transform, authority


def _process_texture_record(record: TextureWalRecord) -> str:
    if _texture_wal is None:
        raise RuntimeError("texture WAL is not ready")
    camera_from_lidar, calibration_authority = _current_camera_from_lidar(record.frame.calibration_id)
    identity = record.frame.identity
    route = _trajectory_store.resolve_submap_near(
        building_id=identity.building_id,
        session_id=identity.session_id,
        map_id=identity.map_id,
        producer_instance_id=identity.producer_instance_id,
        timestamp_ns=record.frame.lio_pose_timestamp_ns,
        maximum_skew_ns=_texture_submap_max_skew_ns,
    )
    published = _provisional_texture_store.ingest(
        record.frame,
        camera_from_lidar=camera_from_lidar,
        calibration_authority=calibration_authority,
        submap_id=route["submap_id"],
    )
    _texture_wal.commit_integrated(record.frame, checkpoint_id=published.revision_id)
    return published.revision_id


def _refresh_render_manifest(building_id: str):
    if _live_surface_pipeline is None:
        raise RuntimeError("live surface pipeline is not ready")
    surface_manifest = _live_surface_pipeline.manifest(scope="building")
    observations_by_submap: dict[str, list[str]] = {}
    for tile in surface_manifest.get("tiles", ()):
        revision_id = tile.get("revision_id")
        submap_id = tile.get("submap_id")
        if isinstance(submap_id, str) and submap_id not in observations_by_submap:
            observations_by_submap[submap_id] = _provisional_texture_store.observation_revisions(
                building_id=building_id,
                submap_id=submap_id,
                projectable_only=True,
            )
        observations = observations_by_submap.get(submap_id, [])
        if not isinstance(revision_id, str) or len(revision_id) != 64 or not observations:
            continue
        try:
            _, body = _live_surface_pipeline.tile_store.load(revision_id)
            _provisional_texture_store.project_and_publish(
                tile_revision_id=revision_id,
                tile_body=body,
                observations=observations,
                submap_id=submap_id,
            )
        except (FileNotFoundError, OSError, ValueError):
            continue
    assignments = {
        revision: manifest
        for revision, manifest in _provisional_texture_store.current_assignments().items()
        if any(
            tile.get("revision_id") == revision
            and tile.get("submap_id") == manifest.get("submap_id")
            for tile in surface_manifest.get("tiles", ())
        )
    }
    validated_manifest = None
    try:
        validated = _live_surface_pipeline.current_building_appearance(building_id)
        if validated is not None:
            validated_manifest = validated.manifest
    except (FileNotFoundError, KeyError, OSError, ValueError):
        pass
    trajectories = _trajectory_store.query(building_id=building_id)
    trajectory_pointer = {
        "revision_id": trajectories["revision_id"],
        "url": f"/api/building/trajectories?building_id={quote(building_id, safe='')}",
    }
    return _render_manifest_store.publish(
        building_id=building_id,
        surface_manifest=surface_manifest,
        provisional_assignments=assignments,
        validated_appearance_manifest=validated_manifest,
        trajectories=trajectory_pointer,
    )


async def _texture_ingest_loop() -> None:
    global _texture_ingested_frames, _texture_last_error, _texture_render_next_refresh
    while True:
        processed_any = False
        if _texture_wal is not None:
            pending = await asyncio.to_thread(_texture_wal.pending)
            for record in pending[:_texture_ingest_batch_size]:
                try:
                    await asyncio.to_thread(_process_texture_record, record)
                    _texture_ingested_frames += 1
                    _texture_last_error = None
                    processed_any = True
                    _texture_render_pending_buildings.add(record.frame.identity.building_id)
                except Exception as exc:
                    _texture_last_error = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            refresh_due = (
                _texture_render_pending_buildings
                and (len(pending) <= _texture_ingest_batch_size or now >= _texture_render_next_refresh)
            )
            if refresh_due:
                buildings = sorted(_texture_render_pending_buildings)
                _texture_render_pending_buildings.clear()
                _texture_render_next_refresh = now + _texture_render_refresh_interval_s
                for building_id in buildings:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(_refresh_render_manifest, building_id)
        await asyncio.sleep(0.02 if processed_any else 0.2)


async def _snapshot_refresh_loop() -> None:
    while True:
        try:
            build_task = asyncio.create_task(asyncio.to_thread(_generate_dashboard_snapshot_json))
            try:
                payload = await asyncio.shield(build_task)
            except asyncio.CancelledError:
                # The worker thread cannot be cancelled. Wait for it before the
                # surface pipeline is closed during application shutdown.
                with contextlib.suppress(Exception):
                    await build_task
                raise
            _publish_snapshot_cache(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("dashboard full-snapshot refresh failed")
        await asyncio.sleep(_snapshot_cache_max_age_s)


async def _recover_surface_then_start_workers() -> None:
    global _ingest_task, _texture_ingest_task, _snapshot_refresh_task, _live_ingest_workers
    global _surface_recovery_complete, _surface_recovery_error, _live_ingest_last_error
    try:
        await asyncio.to_thread(
            _live_surface_pipeline.restore_persisted_runtimes,
            latest_only=True,
        )
        await asyncio.to_thread(_live_surface_pipeline.recover_pending)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _surface_recovery_error = f"{type(exc).__name__}: {exc}"
        _live_ingest_last_error = f"SURFACE_RECOVERY_FAILED: {_surface_recovery_error}"
        logging.exception("dashboard surface recovery failed")
        return

    _surface_recovery_complete = True
    _surface_recovery_error = None
    _ingest_task = asyncio.create_task(_progressive_ingest_loop())
    _texture_ingest_task = asyncio.create_task(_texture_ingest_loop())
    _snapshot_refresh_task = asyncio.create_task(_snapshot_refresh_loop())
    _live_ingest_workers = [
        asyncio.create_task(_live_ingest_worker()),
        asyncio.create_task(_surface_checkpoint_loop()),
        asyncio.create_task(_surface_mesh_worker()),
    ]
    for building_id, submap_id in _live_surface_pipeline.active_targets():
        _queue_surface_mesh(building_id, submap_id)
    _schedule_pending_typed_records()


def _create_surface_mesh_executor() -> ProcessPoolExecutor | None:
    """Create the production meshing pool unless local preview opts out.

    The read-only automatic reconstruction preview does not require live tile
    meshing.  A restricted Windows desktop token can reject the named pipe
    used by ``ProcessPoolExecutor``; setting this explicit local-only switch
    keeps the Dashboard API and validated surface viewer available.  WSL and
    normal production starts retain the isolated process pool by default.
    """

    if os.environ.get("GO2_MAPPING_DISABLE_SURFACE_MESH_EXECUTOR") == "1":
        return None
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=_surface_mesh_max_tasks_per_child,
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _ingest_task, _texture_ingest_task, _snapshot_refresh_task, _live_ingest_queue, _live_ingest_workers
    global _live_ingest_next_sequence, _live_ingest_last_admitted, _live_ingest_last_error
    global _live_wal, _texture_wal, _texture_render_next_refresh, _texture_render_pending_buildings
    global _live_source_status_stream, _typed_inflight, _typed_processed
    global _live_surface_pipeline, _surface_mesh_queue, _surface_mesh_pending, _surface_mesh_deferred
    global _surface_mesh_retry_after
    global _surface_recovery_task, _surface_recovery_complete, _surface_recovery_error
    if _depth_anything_worker is not None:
        _depth_anything_worker.start()
    # One committer preserves the exact preview order.  TCP/WebSocket ACKs may
    # Requests may be pipelined; map mutations must remain serialized.
    _live_ingest_queue = asyncio.Queue(maxsize=_live_ingest_queue_size)
    _live_ingest_next_sequence = None
    _live_ingest_last_admitted = {}
    _live_ingest_last_error = None
    if _live_wal is None:
        _live_wal = LiveKeyframeWal(_live_wal_root, max_bytes=_live_wal_max_bytes)
    if _texture_wal is None:
        _texture_wal = TextureKeyframeWal(_texture_wal_root, max_bytes=_texture_wal_max_bytes)
    _texture_render_pending_buildings = set()
    _texture_render_next_refresh = time.monotonic() + _texture_render_refresh_interval_s
    _live_source_status_stream = None
    if _live_surface_pipeline is None:
        mesh_executor = _create_surface_mesh_executor()
        pipeline = LiveSurfacePipeline(
            _live_building_root,
            wal=_live_wal,
            checkpoint_batch_frames=_surface_checkpoint_batch_frames,
            checkpoint_interval_s=_surface_checkpoint_interval_s,
            default_floor_id=os.environ.get("GO2_MAPPING_FLOOR_ID", "floor-unassigned"),
            default_zone_id=os.environ.get("GO2_MAPPING_ZONE_ID", "zone-unassigned"),
            tile_appearance_store=tile_appearance_store,
            mesh_executor=mesh_executor,
            mesh_tile_batch_size=_surface_mesh_tile_batch_size,
        )
        processor = _tile_appearance_processor(pipeline.tile_store)
        if processor is not None:
            pipeline.configure_appearance_processor(processor)
        _live_surface_pipeline = pipeline
    _typed_inflight = set()
    _typed_processed = set()
    _surface_mesh_queue = asyncio.Queue(maxsize=1)
    _surface_mesh_pending = set()
    _surface_mesh_deferred = set()
    _surface_mesh_retry_after = {}
    _surface_recovery_complete = False
    _surface_recovery_error = None
    _surface_recovery_task = asyncio.create_task(_recover_surface_then_start_workers())
    try:
        yield
    finally:
        if _surface_recovery_task is not None:
            _surface_recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _surface_recovery_task
            _surface_recovery_task = None
        if _texture_ingest_task is not None:
            _texture_ingest_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _texture_ingest_task
            _texture_ingest_task = None
        if _snapshot_refresh_task is not None:
            _snapshot_refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _snapshot_refresh_task
            _snapshot_refresh_task = None
        for worker in _live_ingest_workers:
            worker.cancel()
        for worker in _live_ingest_workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        _live_ingest_workers = []
        _live_ingest_queue = None
        _surface_mesh_queue = None
        if _ingest_task is not None:
            _ingest_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _ingest_task
            _ingest_task = None
        if _live_surface_pipeline is not None:
            pipeline = _live_surface_pipeline
            await asyncio.to_thread(pipeline.close)
            _live_surface_pipeline = None
        if _depth_anything_worker is not None:
            await asyncio.to_thread(_depth_anything_worker.close)


app = FastAPI(title="Go2 Mapping Console", lifespan=_lifespan)
_three_examples = ROOT.parent / "dashboard" / "node_modules" / "three" / "examples" / "jsm"
if _three_examples.is_dir():
    app.mount(
        "/static/three-examples",
        StaticFiles(directory=_three_examples),
        name="three-examples",
    )
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


_SOURCE_STATUS_STALE_AFTER_S = 3.0


def _source_status_snapshot() -> dict:
    status = None
    if _live_wal is not None:
        try:
            if _live_source_status_stream is not None:
                status = _live_wal.latest_status(_live_source_status_stream)
            if status is None:
                status = _live_wal.latest_status_any()
        except (OSError, ValueError):
            status = None
    if status is None:
        return {
            "source_status_age_s": None,
            "source_state": None,
            "source_clipped": None,
            "source_blocker": "SOURCE_STATUS_STALE",
        }

    age_s = max(0.0, (time.time_ns() - status.timestamp_ns) / 1_000_000_000.0)
    source_blocker = status.payload.get("blocker")
    if age_s > _SOURCE_STATUS_STALE_AFTER_S:
        source_blocker = "SOURCE_STATUS_STALE"
    elif status.source_state == "DETAIL_PAYLOAD_LIMIT_EXCEEDED":
        source_blocker = "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
    elif status.source_state == "DETAIL_SOURCE_CLIPPED" or status.source_clipped:
        source_blocker = "DETAIL_SOURCE_CLIPPED"
    elif status.source_state == "SOURCE_READY":
        source_blocker = None
    return {
        "source_status_age_s": age_s,
        "source_state": status.source_state,
        "source_clipped": status.source_clipped,
        "source_blocker": source_blocker,
    }


def _apply_building_scan_precedence(snapshot: dict) -> dict:
    pc_error = _live_ingest_last_error or snapshot.get("meshing_error")
    snapshot["pipeline_error"] = None
    if pc_error:
        error_text = str(pc_error)
        snapshot["pipeline_error"] = error_text
        if error_text == "SURFACE_CAPACITY_BLOCKED" or "WAL_CAPACITY_EXCEEDED" in error_text:
            snapshot["state"] = "SURFACE_CAPACITY_BLOCKED"
            snapshot["source_blocker"] = "SURFACE_CAPACITY_BLOCKED"
        else:
            snapshot["state"] = "FAULTED"
            snapshot["source_blocker"] = "PC_PIPELINE_ERROR"
        return snapshot

    source_blocker = snapshot["source_blocker"]
    if source_blocker == "DETAIL_PAYLOAD_LIMIT_EXCEEDED":
        snapshot["state"] = "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
    elif source_blocker == "DETAIL_SOURCE_CLIPPED":
        snapshot["state"] = "DETAIL_SOURCE_CLIPPED"
    elif source_blocker == "SOURCE_STATUS_STALE":
        snapshot["state"] = "SOURCE_STATUS_STALE"
    return snapshot


def _building_scan_snapshot() -> dict:
    global _building_scan_summary
    if _live_surface_pipeline is not None:
        snapshot = dict(_live_surface_pipeline.snapshot())
    else:
        snapshot = {
            "building_id": None,
            "state": "WAITING_FOR_SURFACE",
            "quality": "PREVIEW_UNCALIBRATED",
            "surface_authority": "LIVE_PROVISIONAL_SURFACE",
            "geometry_authority": False,
            "delivery_authorized": False,
            "building_count": 0,
            "floor_count": 0,
            "zone_count": 0,
            "submap_count": 0,
            "active_submap_id": None,
            "active_floor_id": None,
            "active_zone_id": None,
            "surfel_count": 0,
            "supported_surfel_count": 0,
            "unresolved_surfel_count": 0,
            "tile_count": 0,
            "triangle_count": 0,
            "meshing_error": None,
        }
    snapshot.update(_source_status_snapshot())
    snapshot = _apply_building_scan_precedence(snapshot)
    with _building_scan_summary_lock:
        _building_scan_summary = dict(snapshot)
    return snapshot


def _building_scan_telemetry_snapshot() -> dict:
    with _building_scan_summary_lock:
        return dict(_building_scan_summary)

@app.get("/")
def index():
    # Reading the small shell directly avoids WSL/NTFS FileResponse stat and
    # sendfile stalls while keeping the shell explicitly uncached.
    try:
        content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HTTPException(status_code=503, detail="dashboard shell unavailable") from exc
    return Response(
        content=content,
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

def _dashboard_snapshot_uncached() -> dict:
    state = telemetry.snapshot()
    state["dashboard_protocol"] = "rgbd-producer-epoch-v2"
    live_map = mapper.snapshot()
    room_scan = live_map.get("room_scan") or mapper.room_scan_snapshot()
    building_scan = _building_scan_snapshot()
    dense_map = dense_mapper.snapshot()
    transport = live_transport.snapshot()
    high_fidelity = progressive_transfer.snapshot()
    session = robot_session.snapshot()
    ingest = artifact_ingestor.snapshot() if artifact_ingestor is not None else {
        "integrated_artifacts": 0,
        "integrated_frames": 0,
        "integrated_points": 0,
        "errors": 0,
        "last_error": None,
        "release_authorized_artifacts": 0,
    }
    quality = dense_map.get("quality_evidence") or quality_gate.snapshot()
    rgbd_refinement = dense_map.get("rgbd_refinement")
    camera = camera_assist.snapshot()
    eligible = quality.get("eligible_for", {})
    camera_from_lidar = quality.get("camera_from_lidar_row_major", [])
    calibration_matches = camera.get("calibration_id") == quality.get("calibration_id")
    direct_rgb_texture_ready = (
        _direct_rgb_texture_enabled
        and camera.get("state") not in {"AWAITING_CAMERA_KEYFRAME", "CAMERA_PREVIEW_STALE"}
        and len(camera_from_lidar) == 16
        and isinstance(camera.get("calibration_id"), str)
        and bool(camera.get("calibration_id"))
    )
    texture_authorized = (
        direct_rgb_texture_ready
        or
        bool(rgbd_refinement.get("texture", {}).get("delivery_authorized"))
        if isinstance(rgbd_refinement, dict)
        else bool(
            eligible.get("rgbd_or_neural_geometry_fusion")
            and camera.get("depth_aligned_to_color")
            and calibration_matches
            and len(camera_from_lidar) == 16
        )
    )
    if direct_rgb_texture_ready:
        camera["texture_state"] = "DIRECT_RGB_VERTEX_COLOR"
        camera["texture_blocker"] = None
        camera["camera_from_lidar_row_major"] = camera_from_lidar
    elif texture_authorized:
        camera["texture_state"] = "CALIBRATED_TEXTURE"
        camera["texture_blocker"] = None
        camera["camera_from_lidar_row_major"] = camera_from_lidar
    elif camera.get("state") != "AWAITING_CAMERA_KEYFRAME":
        camera["texture_blocker"] = "CALIBRATION_OR_ALIGNED_DEPTH_REQUIRED"
    camera["texture_authorized"] = texture_authorized
    if isinstance(rgbd_refinement, dict):
        camera["rgbd_refinement"] = rgbd_refinement
    camera["deferred_by_bandwidth"] = bool(transport.get("camera_deferred_by_bandwidth"))
    camera["depth_deferred_by_bandwidth"] = bool(transport.get("camera_depth_deferred_by_bandwidth"))

    if session.get("state") == "FINALIZING":
        state["source"] = f"ROBOT UPLOAD FINALIZING / {session['session_id']}"
    elif session["capture_active"]:
        state["source"] = f"ROBOT SCAN ACTIVE / {session['session_id']}"
    elif room_scan.get("state") == "SCANNING":
        state["source"] = "LIVE ROOM SCAN ACTIVE / RAW CAPTURE STATUS UNKNOWN"
    elif room_scan.get("state") == "STALE":
        state["source"] = "LIVE ROOM SCAN STALE / LAST GLOBAL MAP RETAINED"
    elif transport.get("last_frame_age_ms") is not None:
        state["source"] = "LIVE PREVIEW STALE / RAW CAPTURE STATUS UNKNOWN"
    elif high_fidelity.get("active"):
        state["source"] = "HIGH FIDELITY TRANSFER ACTIVE / RAW CAPTURE STATUS UNKNOWN"

    state["live_map"] = live_map
    state["room_scan"] = room_scan
    state["building_scan"] = building_scan
    state["dense_map"] = dense_map
    state["depth_anything"] = rgbd_refiner.assistance_snapshot()
    state["quality"] = quality
    state["camera"] = camera
    state["session"] = session
    state["calibration_capture"] = calibration_capture.snapshot()
    state["progressive_ingest"] = ingest
    state["transport"] = {**state.get("transport", {}), **transport, "high_fidelity": high_fidelity}
    primary_points = live_map["points"] if room_scan.get("live") else (dense_map["points"] or live_map["points"])
    state["mapping"] = {
        **state.get("mapping", {}),
        "points": primary_points,
        "dense_points": dense_map["points"],
        "update_hz": transport["keyframe_hz"],
        "residual_cm": live_map["residual_cm"],
        "room_scan_state": room_scan["state"],
        "room_extent_m": room_scan["extent_m"],
    }
    state["robot"] = {**state.get("robot", {}), "position": live_map["position"]}
    return state


def dashboard_snapshot() -> dict:
    """Build a fresh REST snapshot; WebSocket clients use the shared cache."""
    return _dashboard_snapshot_uncached()


def _generate_dashboard_snapshot_json() -> str:
    return json.dumps(_dashboard_snapshot_uncached(), separators=(",", ":"), allow_nan=False)


def _read_snapshot_cache() -> tuple[str | None, int, float]:
    with _snapshot_cache_lock:
        return _snapshot_cache_json, _snapshot_cache_generation, _snapshot_cache_generated_at


def _publish_snapshot_cache(payload: str) -> int:
    global _snapshot_cache_json, _snapshot_cache_generated_at, _snapshot_cache_generation
    with _snapshot_cache_lock:
        _snapshot_cache_json = payload
        _snapshot_cache_generated_at = time.monotonic()
        _snapshot_cache_generation += 1
        return _snapshot_cache_generation


def dashboard_snapshot_json(*, force: bool = False) -> str:
    cached, _, generated_at = _read_snapshot_cache()
    if not force and cached is not None and time.monotonic() - generated_at < _snapshot_cache_max_age_s:
        return cached
    payload = _generate_dashboard_snapshot_json()
    _publish_snapshot_cache(payload)
    return payload


def dashboard_telemetry_patch() -> dict:
    state = telemetry.snapshot()
    live_map = mapper.telemetry_snapshot()
    room_scan = live_map.get("room_scan") or mapper.room_scan_snapshot()
    building_scan = _building_scan_telemetry_snapshot()
    transport = live_transport.snapshot()
    high_fidelity = progressive_transfer.snapshot()
    session = robot_session.snapshot()
    ingest = artifact_ingestor.snapshot() if artifact_ingestor is not None else {
        "integrated_artifacts": 0,
        "integrated_frames": 0,
        "integrated_points": 0,
        "errors": 0,
        "last_error": None,
        "release_authorized_artifacts": 0,
    }
    if session.get("state") == "FINALIZING":
        state["source"] = f"ROBOT UPLOAD FINALIZING / {session['session_id']}"
    elif session["capture_active"]:
        state["source"] = f"ROBOT SCAN ACTIVE / {session['session_id']}"
    elif room_scan.get("state") == "SCANNING":
        state["source"] = "LIVE ROOM SCAN ACTIVE / RAW CAPTURE STATUS UNKNOWN"
    elif room_scan.get("state") == "STALE":
        state["source"] = "LIVE ROOM SCAN STALE / LAST GLOBAL MAP RETAINED"
    elif transport.get("last_frame_age_ms") is not None:
        state["source"] = "LIVE PREVIEW STALE / RAW CAPTURE STATUS UNKNOWN"
    elif high_fidelity.get("active"):
        state["source"] = "HIGH FIDELITY TRANSFER ACTIVE / RAW CAPTURE STATUS UNKNOWN"
    state.update({
        "live_map": live_map,
        "room_scan": room_scan,
        "building_scan": building_scan,
        "camera": camera_assist.snapshot(),
        "session": session,
        "progressive_ingest": ingest,
        "depth_anything": rgbd_refiner.assistance_snapshot(),
        "transport": {**state.get("transport", {}), **transport, "high_fidelity": high_fidelity},
        "robot": {**state.get("robot", {}), "position": live_map["position"]},
    })
    return {"message_type": "telemetry", "snapshot": state}


@app.get("/api/snapshot")
def snapshot():
    return dashboard_snapshot()


@app.get("/api/diagnostics")
def diagnostics():
    state = dict(dashboard_telemetry_patch()["snapshot"])
    live_map = dict(state.get("live_map") or {})
    live_map.pop("trajectory", None)
    live_map.pop("preview_points", None)
    live_map.pop("measured_preview_points", None)
    state["live_map"] = live_map
    return state


@app.get("/api/coarse-map")
def coarse_map():
    """Return the operator-authored map prior and generated robot tasks."""

    return _coarse_map_store.load()


@app.put("/api/coarse-map")
async def replace_coarse_map(request: Request):
    """Atomically replace the coarse map without touching measured surfaces."""

    try:
        payload = await request.json()
        validate_coarse_map(payload)
        return _coarse_map_store.save(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/coarse-map/tasks/rebuild")
def rebuild_coarse_map_tasks():
    current = _coarse_map_store.load()
    return _coarse_map_store.save(build_task_graph(current))


def _coarse_map_task_payload(request_payload: Any) -> dict[str, Any]:
    if not isinstance(request_payload, dict):
        raise HTTPException(status_code=422, detail="task payload must be a JSON object")
    return request_payload


def _apply_coarse_map_task_update(current: dict[str, Any], task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return update_task_record(
        current,
        task_id,
        status=payload["status"] if "status" in payload else _MISSING,
        coverage_percent=payload["coverage_percent"] if "coverage_percent" in payload else _MISSING,
        diagnostics=payload["diagnostics"] if "diagnostics" in payload else _MISSING,
        mode=payload["mode"] if "mode" in payload else _MISSING,
        lease_id=payload["lease_id"] if "lease_id" in payload else _MISSING,
        expected_revision=payload["expected_revision"] if "expected_revision" in payload else _MISSING,
    )


def _coarse_map_update_task(task_id: str, payload: dict[str, Any]):
    try:
        updated = _apply_coarse_map_task_update(_coarse_map_store.load(), task_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _coarse_map_store.save(updated)


def _canonical_coarse_map_update_task(map_id: str, task_id: str, payload: dict[str, Any]):
    try:
        store = _coarse_store_for_map(map_id)
        current = store.load_canonical(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    try:
        updated = _apply_coarse_map_task_update(current, task_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return store.save(updated)


@app.post("/api/coarse-map/tasks/{task_id}")
async def update_coarse_map_task(task_id: str, request: Request):
    """Update task state from an operator or robot verification run."""

    payload = _coarse_map_task_payload(await request.json())
    return _coarse_map_update_task(task_id, payload)


@app.post("/api/coarse-map/tasks/{task_id:path}/mode")
async def update_coarse_map_task_mode(task_id: str, request: Request):
    """Update only the task mode and optional lease identifier."""

    payload = _coarse_map_task_payload(await request.json())
    if "mode" not in payload:
        raise HTTPException(status_code=422, detail="mode is required")
    return _coarse_map_update_task(task_id, payload)


@app.post("/api/coarse-map/tasks/{task_id:path}/lease")
async def update_coarse_map_task_lease(task_id: str, request: Request):
    """Update only the task lease identifier."""

    payload = _coarse_map_task_payload(await request.json())
    if "lease_id" not in payload:
        raise HTTPException(status_code=422, detail="lease_id is required")
    return _coarse_map_update_task(task_id, payload)


@app.post("/api/maps/coarse/{map_id}/tasks/rebuild")
def canonical_rebuild_coarse_map_tasks(map_id: str):
    try:
        store = _coarse_store_for_map(map_id)
        current = store.load_canonical(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    return store.save(build_task_graph(current))


@app.post("/api/maps/coarse/{map_id}/tasks/{task_id:path}/mode")
async def canonical_update_coarse_map_task_mode(map_id: str, task_id: str, request: Request):
    payload = _coarse_map_task_payload(await request.json())
    if "mode" not in payload:
        raise HTTPException(status_code=422, detail="mode is required")
    return _canonical_coarse_map_update_task(map_id, task_id, payload)


@app.post("/api/maps/coarse/{map_id}/tasks/{task_id:path}/lease")
async def canonical_update_coarse_map_task_lease(map_id: str, task_id: str, request: Request):
    payload = _coarse_map_task_payload(await request.json())
    if "lease_id" not in payload:
        raise HTTPException(status_code=422, detail="lease_id is required")
    return _canonical_coarse_map_update_task(map_id, task_id, payload)


@app.post("/api/maps/coarse/{map_id}/tasks/{task_id:path}")
async def canonical_update_coarse_map_task(map_id: str, task_id: str, request: Request):
    payload = _coarse_map_task_payload(await request.json())
    return _canonical_coarse_map_update_task(map_id, task_id, payload)


def _canonical_coarse_map_payload(request_payload: Any) -> dict[str, Any]:
    if not isinstance(request_payload, dict):
        raise HTTPException(status_code=422, detail="coarse map payload must be a JSON object")
    return request_payload


def _apply_zone_geometry_operation(current: dict[str, Any], payload: dict[str, Any], operation: str) -> dict[str, Any]:
    zones = [dict(zone) for zone in current.get("rooms", [])]
    by_id = {str(zone.get("id")): zone for zone in zones}
    if operation == "split":
        zone_id = str(payload.get("entity_id") or "")
        zone = by_id.get(zone_id)
        if zone is None:
            raise ValueError("function zone not found")
        first, second = split_room_polygon(zone["polygon"], direction=str(payload.get("direction") or "vertical"))
        base = {key: value for key, value in zone.items() if key not in {"id", "name", "polygon", "coverage_percent"}}
        suffix = len(zones) + 1
        zones = [item for item in zones if item["id"] != zone_id]
        zones.extend([
            {**base, "id": f"{zone_id}-a", "name": f"{zone.get('name', zone_id)} A", "polygon": first, "coverage_percent": 0.0},
            {**base, "id": f"{zone_id}-b", "name": f"{zone.get('name', zone_id)} B", "polygon": second, "coverage_percent": 0.0},
        ])
        return {**current, "rooms": zones}
    if operation == "merge":
        first_id = str(payload.get("first_entity_id") or "")
        second_id = str(payload.get("second_entity_id") or "")
        first = by_id.get(first_id)
        second = by_id.get(second_id)
        if first is None or second is None or first_id == second_id:
            raise ValueError("two distinct function zones are required")
        merged = merge_room_polygons(first["polygon"], second["polygon"])
        merged_zone = {**first, "id": str(payload.get("merged_entity_id") or f"{first_id}-merged"), "name": str(payload.get("name") or f"{first.get('name', first_id)} + {second.get('name', second_id)}"), "polygon": merged, "coverage_percent": 0.0}
        return {**current, "rooms": [item for item in zones if item["id"] not in {first_id, second_id}] + [merged_zone]}
    raise ValueError("unsupported geometry operation")


@app.post("/api/maps/coarse/{map_id}/geometry/{operation}")
async def coarse_geometry_operation(map_id: str, operation: str, request: Request):
    if operation not in {"split", "merge"}:
        raise HTTPException(status_code=404, detail="geometry operation not found")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("geometry operation payload must be an object")
        store = _coarse_store_for_map(map_id)
        current = store.load_canonical(map_id)
        updated = _apply_zone_geometry_operation(current, payload, operation)
        return store.replace_canonical(map_id, updated, expected_revision=payload.get("expected_revision"))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/maps/coarse/import")
async def validate_coarse_map_import(request: Request):
    """Validate an import without overwriting the current map revision."""

    payload = _canonical_coarse_map_payload(await request.json())
    source = payload.get("map", payload)
    if not isinstance(source, dict):
        raise HTTPException(status_code=422, detail="coarse map import must contain an object")
    try:
        coarse = validate_coarse_map(source)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if coarse.get("schema_version") != 2:
        raise HTTPException(status_code=422, detail="coarse map schema_version must be 2")
    return {"accepted": True, "map": coarse}


@app.get("/api/maps/coarse/{map_id}")
def canonical_coarse_map(map_id: str):
    try:
        return _coarse_store_for_map(map_id).load_canonical(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/catalog/furniture")
def coarse_furniture_catalog():
    return {"schema_version": 1, "items": list(FURNITURE_CATALOG), "asset_authority": "LOCAL_ONLY"}


@app.get("/api/catalog/coarse-map-templates")
def coarse_map_template_catalog():
    return {"schema_version": 1, "templates": list(COARSE_MAP_TEMPLATES), "provenance": "USER_PRIOR", "geometry_authority": False}


@app.get("/api/catalog/pbr-materials")
def coarse_pbr_material_catalog():
    path = ROOT / "static" / "assets" / "textures" / "pbr" / "ambientcg" / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="local PBR catalog unavailable") from exc
    for material in payload.get("materials", []):
        for item in (material.get("maps") or {}).values():
            item["url"] = "/static/assets/textures/pbr/ambientcg/" + item["file"]
    return payload


@app.get("/api/buildings/coarse/{building_id}")
def coarse_building_index(building_id: str):
    try:
        building = _coarse_building_store.load()
        if not building["floors"]:
            current = _coarse_map_store.load_canonical(_coarse_map_store.load().get("map_id", "go2-demo-floor"))
            if current.get("building_id") == building_id:
                building = _coarse_building_store.register_floor(
                    current,
                    label=current.get("floor_id") or current.get("map_id"),
                )
        if building["building_id"] != building_id:
            raise KeyError(building_id)
        return building
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse building not found") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/buildings/coarse/{building_id}")
async def replace_coarse_building_index(building_id: str, request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("building_id") != building_id:
            raise ValueError("coarse building building_id does not match request path")
        candidate = {**payload, "connections": payload.get("connections", [])}
        _validate_building_connection_endpoints(candidate)
        return _coarse_building_store.update_connections(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/buildings/coarse/{building_id}/floors")
async def create_coarse_building_floor(building_id: str, request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("floor payload must be an object")
        created = _coarse_building_store.create_floor(
            building_id,
            payload.get("floor_id"),
            label=payload.get("label"),
        )
        return {"accepted": True, "map": created, "building": _coarse_building_store.load()}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/buildings/coarse/{building_id}/mission")
def compile_coarse_building_mission(building_id: str):
    """Compile a building-wide mission without activating any task."""
    try:
        building = _coarse_building_store.load()
        if building.get("building_id") != building_id:
            raise KeyError(building_id)
        floor_maps = {}
        for floor in building.get("floors") or []:
            if not isinstance(floor, dict):
                continue
            map_id = str(floor.get("map_id") or "")
            if map_id:
                floor_maps[map_id] = _coarse_store_for_map(map_id).load_canonical(map_id)
        mission = compile_building_mission(building, floor_maps)
        return _mission_store.save(mission)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse building not found") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/maps/coarse/{map_id}")
async def replace_canonical_coarse_map(map_id: str, request: Request):
    payload = _canonical_coarse_map_payload(await request.json())
    if payload.get("map_id") != map_id:
        raise HTTPException(status_code=422, detail="coarse map map_id does not match request path")
    try:
        store = _coarse_store_for_map(map_id)
        updated = store.replace_canonical(
            map_id,
            payload,
            expected_revision=payload.get("expected_revision"),
        )
        if updated.get("building_id"):
            index = _coarse_building_store.load()
            if not index.get("floors") or index.get("building_id") == updated.get("building_id"):
                _coarse_building_store.register_floor(updated, label=updated.get("floor_id") or map_id)
        return updated
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/maps/coarse/{map_id}/validate")
async def validate_canonical_coarse_map(map_id: str, request: Request):
    payload = _canonical_coarse_map_payload(await request.json())
    try:
        coarse = validate_coarse_map(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if coarse.get("schema_version") != 2:
        raise HTTPException(status_code=422, detail="coarse map schema_version must be 2")
    if coarse.get("map_id") != map_id:
        raise HTTPException(status_code=422, detail="coarse map map_id does not match request path")
    return {"valid": True, "map": coarse}


@app.get("/api/maps/coarse/{map_id}/revisions")
def canonical_coarse_map_revisions(map_id: str):
    try:
        revisions = _coarse_store_for_map(map_id).revision_history(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"map_id": map_id, "revisions": revisions}


@app.get("/api/maps/coarse/{map_id}/revisions/{revision}")
def canonical_coarse_map_revision(map_id: str, revision: int):
    try:
        payload = _coarse_store_for_map(map_id).load_revision(map_id, revision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map revision not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"map_id": map_id, "revision": revision, "map": payload}


@app.post("/api/maps/coarse/{map_id}/nav2-export")
async def export_canonical_coarse_map_nav2(map_id: str, request: Request):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", map_id, re.ASCII) is None:
        raise HTTPException(status_code=422, detail="map_id is invalid for Nav2 export")
    try:
        current = _coarse_map_store.load_canonical(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    try:
        payload = await request.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Nav2 export payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Nav2 export payload must be an object")
    try:
        exported = write_nav2_map(
            current,
            _nav2_map_root / map_id,
            resolution=payload.get("resolution", 0.05),
            padding=payload.get("padding", 1.0),
            wall_thickness=payload.get("wall_thickness", 0.12),
            max_cells=payload.get("max_cells", 4_000_000),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output_root = (_nav2_map_root / map_id).resolve(strict=False)
    return {
        "schema_version": 1,
        "state": "AVAILABLE",
        "map_id": map_id,
        "source_revision": exported["source_revision"],
        "coordinate_frame": exported["coordinate_frame"],
        "provenance": exported["provenance"],
        "geometry_authority": exported["geometry_authority"],
        "resolution": exported["resolution"],
        "width": exported["width"],
        "height": exported["height"],
        "yaml_relative_path": str(exported["yaml_path"].relative_to(output_root)),
        "pgm_relative_path": str(exported["pgm_path"].relative_to(output_root)),
        "provenance_relative_path": str(exported["metadata_path"].relative_to(output_root)),
        "read_only": True,
    }


@app.post("/api/maps/coarse/{map_id}/frontiers")
async def preview_canonical_coarse_map_frontiers(map_id: str, request: Request):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", map_id, re.ASCII) is None:
        raise HTTPException(status_code=422, detail="map_id is invalid for frontier preview")
    try:
        current = _coarse_map_store.load_canonical(map_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="coarse map not found") from exc
    try:
        payload = await request.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="frontier payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="frontier payload must be an object")
    try:
        occupancy = build_nav2_occupancy_map(
            current,
            resolution=payload.get("resolution", 0.05),
            padding=payload.get("padding", 1.0),
            wall_thickness=payload.get("wall_thickness", 0.12),
            max_cells=payload.get("max_cells", 4_000_000),
        )
        candidates = plan_frontier_candidates(
            occupancy["grid"],
            origin=occupancy["metadata"]["origin"][:2],
            resolution=occupancy["metadata"]["resolution"],
            robot_pose=payload.get("robot_pose"),
            frame=occupancy["metadata"]["coordinate_frame"],
            min_cluster_size=payload.get("min_cluster_size", 1),
            max_candidates=payload.get("max_candidates", 32),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "schema_version": 1,
        "state": "PREVIEW_ONLY",
        "map_id": map_id,
        "source_revision": occupancy["metadata"]["source_revision"],
        "coordinate_frame": occupancy["metadata"]["coordinate_frame"],
        "provenance": occupancy["metadata"]["provenance"],
        "geometry_authority": occupancy["metadata"]["geometry_authority"],
        "read_only": True,
        "goal_available": False,
        "candidates": candidates,
    }


@app.post("/api/stairs/survey")
async def survey_stairs_read_only(request: Request):
    try:
        payload = await request.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="stair survey payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="stair survey payload must be an object")
    observations = payload.get("observations", [])
    if not isinstance(observations, list):
        raise HTTPException(status_code=422, detail="stair survey observations must be a list")
    prepared_observations: list[Any] = []
    for observation in observations:
        if not isinstance(observation, dict) or "profile" in observation or "points_xyz" not in observation:
            prepared_observations.append(observation)
            continue
        prepared = dict(observation)
        evidence = {
            key: prepared.get(key)
            for key in (
                "top_landing_supported",
                "bottom_landing_supported",
                "static_scene",
                "unobstructed",
                "non_hollow",
                "straight_flight",
            )
        }
        try:
            prepared["profile"] = extract_stair_profile_from_points(
                prepared["points_xyz"], evidence=evidence
            )
        except (TypeError, ValueError) as exc:
            prepared["profile"] = None
            prepared["extraction_error"] = str(exc)
        prepared.pop("points_xyz", None)
        prepared_observations.append(prepared)
    try:
        report = aggregate_stair_observations(
            prepared_observations,
            source_identity=payload.get("source_identity"),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return _stair_survey_store.save(report)
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/stairs/survey/latest")
def latest_stair_survey():
    report = _stair_survey_store.load()
    if report is not None:
        return report
    return {
        "schema_version": 1,
        "state": "NO_SURVEY",
        "profile": None,
        "blocking_reasons": ["NO_STAIR_SURVEY_ARTIFACT"],
        "source_identity": None,
        "measurement_authority": False,
        "locomotion_authorized": False,
        "motion_lease_required": True,
    }


def _coverage_state(task: dict) -> str:
    status = str(task.get("status") or "PENDING")
    if status == "QUARANTINED":
        return "QUARANTINED"
    if status == "DONE":
        return "VERIFIED"
    if status in {"ACTIVE", "PAUSED"}:
        return "OBSERVED"
    return "UNSEEN"


@app.get("/api/coverage/{mission_id}")
def mission_coverage(mission_id: str):
    try:
        mission = _mission_store.load(mission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc
    tasks = [task for task in mission.get("tasks") or [] if isinstance(task, dict)]
    room_groups: dict[str, list[dict]] = {}
    for task in tasks:
        room_id = str(task.get("room_id") or "unassigned")
        room_groups.setdefault(room_id, []).append(task)
    coverage_rooms = []
    for room_id, room_tasks in sorted(room_groups.items()):
        completed = sum(task.get("status") == "DONE" for task in room_tasks)
        states = {_coverage_state(task) for task in room_tasks}
        coverage_rooms.append({
            "room_id": room_id,
            "room_name": next((task.get("room_name") for task in room_tasks if task.get("room_name")), room_id),
            "task_count": len(room_tasks),
            "completed_task_count": completed,
            "coverage_percent": round(100.0 * completed / len(room_tasks), 2) if room_tasks else 0.0,
            "state": "QUARANTINED" if "QUARANTINED" in states else "VERIFIED" if completed == len(room_tasks) else "OBSERVED" if "OBSERVED" in states else "UNSEEN",
            "blockers": ["QUARANTINED_TASK_PRESENT"] if "QUARANTINED" in states else [],
        })
    return {
        "schema_version": 1,
        "mission_id": mission_id,
        "map_id": mission.get("map_id"),
        "coordinate_frame": mission.get("coordinate_frame"),
        "state": mission.get("state"),
        "task_count": len(tasks),
        "completed_task_count": sum(task.get("status") == "DONE" for task in tasks),
        "coverage_percent": round(100.0 * sum(task.get("status") == "DONE" for task in tasks) / len(tasks), 2) if tasks else 0.0,
        "rooms": coverage_rooms,
        "tasks": [
            {
                "task_id": task.get("task_id"),
                "room_id": task.get("room_id"),
                "kind": task.get("kind"),
                "status": task.get("status"),
                "coverage_state": _coverage_state(task),
                "coverage_percent": 100.0 if task.get("status") == "DONE" else 0.0,
                "diagnostics": task.get("diagnostics"),
            }
            for task in tasks
        ],
    }


@app.get("/api/reconstruction/segments")
def reconstruction_segments():
    return {"schema_version": 1, "segments": [_segment_record(segment_id) for segment_id in _segment_ids()]}


@app.get("/api/reconstruction/segments/{segment_id}")
def reconstruction_segment(segment_id: str):
    record = _segment_record(segment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="reconstruction segment is unavailable")
    return record


@app.get("/api/reconstruction/segments/{segment_id}/quality")
def reconstruction_segment_quality(segment_id: str):
    return _segment_quality_record(segment_id)


@app.post("/api/reconstruction/segments/{segment_id}/quarantine")
async def quarantine_reconstruction_segment(segment_id: str, request: Request):
    segment_id = _segment_id(segment_id)
    if _segment_record(segment_id) is None:
        raise HTTPException(status_code=404, detail="reconstruction segment is unavailable")
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="quarantine payload must be a JSON object")
    reason = str(payload.get("reason") or "OPERATOR_QUARANTINE")
    record = {
        "schema_version": 1,
        "segment_id": segment_id,
        "state": "QUARANTINED",
        "reason": reason,
        "created_at_unix": time.time(),
        "read_only": True,
    }
    _write_json_atomic(_segment_quarantine_path(segment_id), record)
    return {**_segment_record(segment_id), "quarantine": record}


@app.post("/api/reconstruction/segments/{segment_id}/replay")
async def replay_reconstruction_segment(segment_id: str, request: Request):
    segment_id = _segment_id(segment_id)
    record = _segment_record(segment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="reconstruction segment is unavailable")
    if record.get("state") == "QUARANTINED":
        raise HTTPException(
            status_code=423,
            detail={"state": "REPLAY_BLOCKED", "segment_id": segment_id, "blockers": ["SEGMENT_QUARANTINED"]},
        )
    blockers = _segment_closure_blockers(segment_id)
    if blockers:
        raise HTTPException(
            status_code=423,
            detail={"state": "REPLAY_BLOCKED", "segment_id": segment_id, "blockers": blockers},
        )
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="replay payload must be a JSON object")
    from mapping.pc.go2_mapping_reconstruction.session_worker import SessionReconstructionWorker
    from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2

    replay_root = (_reconstruction_root / "segment-replays").resolve(strict=False)
    status_path = replay_root / f"{segment_id}.status.json"
    try:
        worker = SessionReconstructionWorker(
            session_id=segment_id,
            inbox_root=_capture_inbox_root,
            output_root=replay_root,
            status_path=status_path,
            sensor_frame_transform_id=str(payload.get("sensor_frame_transform_id") or PROVISIONAL_GO2_UTLIDAR_PITCH180_V2),
            source_mode=str(payload.get("source_mode") or "RAW_LIDAR_WITH_RECORDED_LIO"),
            final_global_optimization=payload.get("final_global_optimization", True) is True,
        )
        result = await asyncio.to_thread(worker.finalize_verified_session)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=f"segment replay failed: {exc}") from exc
    return {
        **result,
        "segment_id": segment_id,
        "replay_root": str(replay_root),
        "official_pointer_updated": False,
        "read_only": True,
    }


def _mission_payload(request_payload: Any) -> dict[str, Any]:
    if request_payload is None:
        return {}
    if not isinstance(request_payload, dict):
        raise HTTPException(status_code=422, detail="mission payload must be a JSON object")
    return request_payload


def _mission_source_map(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source_coarse_map")
    if source is None:
        return _coarse_map_store.load()
    if not isinstance(source, dict):
        raise HTTPException(status_code=422, detail="source_coarse_map must be an object")
    try:
        validate_coarse_map(source)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return source


def _mission_task_update(mission_id: str, task_id: str, payload: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    for key in ("status", "mode", "lease_id", "expected_revision"):
        if key in payload:
            kwargs[key] = payload[key]
    try:
        return _mission_store.update_task(mission_id, task_id, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _navigation_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="navigation payload must be a JSON object")
    return payload


def _load_mission_task(mission_id: str, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        mission = _mission_store.load(mission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc
    task = next((item for item in mission.get("tasks") or [] if item.get("task_id") == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="mission task not found")
    return mission, task


def _mission_route_audit_source(mission: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Resolve the currently persisted coarse prior used to audit a mission task."""

    map_id = str(task.get("map_id") or task.get("floor_map_id") or mission.get("map_id") or "")
    if map_id and map_id != str(mission.get("building_id") or ""):
        try:
            return _coarse_store_for_map(map_id).load_canonical(map_id)
        except (KeyError, ValueError):
            pass
    floor_id = str(task.get("floor_id") or "")
    if floor_id:
        building = _coarse_building_store.load()
        floor = next((item for item in building.get("floors") or [] if str(item.get("floor_id")) == floor_id), None)
        if isinstance(floor, dict) and floor.get("map_id"):
            return _coarse_store_for_map(str(floor["map_id"])).load_canonical(str(floor["map_id"]))
    return _coarse_map_store.load()


def _enforce_mission_route_audit(mission: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    try:
        report = audit_mission_route(task, _mission_route_audit_source(mission, task))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"state": "NAVIGATION_ROUTE_BLOCKED", "route_audit": {"state": "UNAVAILABLE", "navigation_eligible": False, "findings": [{"code": "ROUTE_AUDIT_UNAVAILABLE", "severity": "BLOCKER"}]}, "reason": str(exc)},
        ) from exc
    if report["navigation_eligible"] is not True:
        raise HTTPException(status_code=422, detail={"state": "NAVIGATION_ROUTE_BLOCKED", "route_audit": report})
    return report


def _compile_navigation_goal_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mission_id = payload.get("mission_id")
    task_id = payload.get("task_id")
    if mission_id and task_id:
        mission, task = _load_mission_task(str(mission_id), str(task_id))
        _enforce_mission_route_audit(mission, task)
        try:
            return compile_navigation_goal(
                mission,
                task,
                goal_id=payload.get("goal_id"),
                lease_id=payload.get("lease_id"),
                max_speed_mps=payload.get("max_speed_mps"),
                max_duration_s=payload.get("max_duration_s"),
            )
        except NavigationBridgeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return _navigation_bridge_store.compile_from_payload(payload)
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/missions")
def list_missions():
    return {"missions": _mission_store.list()}


@app.post("/api/missions")
async def create_mission(request: Request):
    payload = _mission_payload(await request.json())
    source_map = _mission_source_map(payload)
    try:
        return _mission_store.compile(source_map, mission_id=payload.get("mission_id"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/missions/{mission_id}")
def get_mission(mission_id: str):
    try:
        return _mission_store.load(mission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc


@app.post("/api/missions/{mission_id}/compile")
async def compile_mission(mission_id: str, request: Request):
    payload = _mission_payload(await request.json())
    source_map = _mission_source_map(payload)
    try:
        return _mission_store.compile(source_map, mission_id=mission_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/missions/{mission_id}/tasks")
def get_mission_tasks(mission_id: str):
    try:
        mission = _mission_store.load(mission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc
    return {"mission_id": mission_id, "tasks": mission.get("tasks") or []}


@app.post("/api/missions/{mission_id}/tasks/{task_id:path}/route-audit")
async def audit_mission_task_route(mission_id: str, task_id: str, request: Request):
    """Return a read-only safety audit before a mission task can send a goal."""

    mission, task = _load_mission_task(mission_id, task_id)
    payload = _mission_payload(await request.json())
    source_map = _mission_source_map(payload)
    try:
        return audit_mission_route(task, source_map)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/missions/{mission_id}/tasks/{task_id}/activate")
async def activate_mission_task(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    payload["status"] = "ACTIVE"
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/pause")
async def pause_mission_task(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    payload["status"] = "PAUSED"
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/resume")
async def resume_mission_task(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    payload["status"] = "ACTIVE"
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/complete")
async def complete_mission_task(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    payload["status"] = "DONE"
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/quarantine")
async def quarantine_mission_task(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    payload["status"] = "QUARANTINED"
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/mode")
async def update_mission_task_mode(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    if "mode" not in payload:
        raise HTTPException(status_code=422, detail="mode is required")
    return _mission_task_update(mission_id, task_id, payload)


@app.post("/api/missions/{mission_id}/tasks/{task_id}/lease")
async def update_mission_task_lease(mission_id: str, task_id: str, request: Request):
    payload = _mission_payload(await request.json())
    if "lease_id" not in payload:
        raise HTTPException(status_code=422, detail="lease_id is required")
    return _mission_task_update(mission_id, task_id, payload)


@app.get("/api/robot/navigation/status")
def robot_navigation_status():
    return _navigation_bridge_store.status()


@app.get("/api/robot/locomotion/status")
def robot_locomotion_status():
    """Expose backend readiness without authorizing robot actuation."""

    return {
        "schema_version": 1,
        "enabled": False,
        "hardware_verified": False,
        "actuation_authorized": False,
        "active_backend": None,
        "control_frequency_hz": None,
        "max_speed_mps": 0.12,
        "state": "WAITING_FOR_GO2_RUNTIME",
        "backends": available_backends(),
        "required_gates": [
            "GO2_ONLINE",
            "SDK_COMPATIBILITY_VERIFIED",
            "CHECKPOINT_IDENTITY_VERIFIED",
            "TERRAIN_QUALIFIED",
            "OPERATOR_CONFIRMATION",
            "MOTION_LEASE",
            "WATCHDOG_READY",
        ],
    }


@app.get("/api/robot/vla/status")
def robot_vla_status():
    status = _mobilevla_adapter.status()
    return {
        "schema_version": 1,
        "model_id": _mobilevla_manifest.get("model_id", "MobileVLA-R1"),
        "upstream_commit": _mobilevla_manifest.get("upstream_commit"),
        "huggingface_repo": _mobilevla_manifest.get("huggingface_repo"),
        "checkpoint_state": _mobilevla_manifest.get("checkpoint_state", "UNKNOWN"),
        "preferred_variant": _mobilevla_preferred_variant,
        "local_checkpoint_path": str(_mobilevla_checkpoint_path),
        "local_files_only": True,
        "action_boundary": "ASSISTED_NAV2_INTENT_ONLY",
        "forbidden_outputs": list(_mobilevla_manifest.get("forbidden_outputs") or []),
        "variant_capabilities": {
            str(variant): dict(capabilities)
            for variant, capabilities in (_mobilevla_manifest.get("variant_capabilities") or {}).items()
            if isinstance(capabilities, dict)
        },
        "adapter": status,
    }


@app.post("/api/robot/vla/intent")
async def robot_vla_intent(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="VLA intent payload must be an object")
    try:
        decision = parse_mobilevla_decision(payload.get("response"))
        coarse_map = _coarse_map_store.load_canonical(_coarse_map_store.load().get("map_id", "go2-demo-floor"))
        snapshot = dashboard_snapshot()
        current_position = payload.get("current_position") or (snapshot.get("robot") or {}).get("position") or [0.0, 0.0]
        auto_plan = bool(payload.get("auto_plan", not payload.get("waypoints")))
        planned = None
        if auto_plan and decision.intent != "stop_and_request_operator":
            try:
                planned = plan_mobilevla_navigation(
                    decision,
                    coarse_map=coarse_map,
                    current_position=current_position,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        waypoints = (planned or {}).get("waypoints") or payload.get("waypoints") or []
        geofence = (planned or {}).get("geofence") or payload.get("geofence") or {"frame": "SESSION_ODOM_LOCAL", "polygon": []}
        mission_id = str(payload.get("mission_id") or "vla-mission")
        task_id = str(payload.get("task_id") or ((planned or {}).get("target") or {}).get("id") and f"vla-{(planned or {}).get('target', {}).get('id')}" or "vla-task")
        navigation = decision_to_navigation_payload(
            decision,
            mission_id=mission_id,
            task_id=task_id,
            task_revision=payload.get("task_revision", 1),
            waypoints=waypoints,
            geofence=geofence,
        )
        return {"schema_version": 1, "decision": asdict(decision), "navigation": navigation, "planning": planned}
    except MobileVLADecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/robot/capabilities")
def robot_capabilities():
    """Return declared interfaces without promoting them to live-hardware evidence."""

    session = robot_session.snapshot()
    navigation = _navigation_bridge_store.status()
    return {
        "schema_version": 1,
        "measurement_authority": False,
        "coordinate_scope": "SESSION_ODOM_LOCAL",
        "transport": {
            "production": "TAILSCALE_DIRECT",
            "ssh_business_path": False,
        },
        "sensors": {
            "lidar": {"topic": "/go2/lio/lidar", "message_type": "sensor_msgs/msg/PointCloud2", "verified": False},
            "lio_pose": {"topic": "/go2/lio/pose", "message_type": "geometry_msgs/msg/PoseStamped", "verified": False},
            "imu": {"topic": "/dog_imu_raw", "message_type": "sensor_msgs/msg/Imu", "verified": False},
            "color": {"topic": "/camera/color/image_raw", "message_type": "sensor_msgs/msg/Image", "verified": False},
            "depth": {"topic": "/camera/aligned_depth_to_color/image_raw", "message_type": "sensor_msgs/msg/Image", "verified": False},
            "camera_info": {"topic": "/camera/color/camera_info", "message_type": "sensor_msgs/msg/CameraInfo", "verified": False},
            "tf": {"topic": "/tf", "message_type": "tf2_msgs/msg/TFMessage", "verified": False},
        },
        "navigation": {
            "nav2": {
                "verified": False,
                "action_server_state": "UNVERIFIED",
                "bridge_state": navigation.get("state"),
            },
            "manual_takeover": {"verified": False, "state": "UNVERIFIED"},
            "direct_unitree_gait": False,
        },
        "locomotion": {
            "enabled": False,
            "hardware_verified": False,
            "actuation_authorized": False,
            "active_backend": None,
            "state": "WAITING_FOR_GO2_RUNTIME",
            "backends": available_backends(),
        },
        "capture": {
            "session_state": session.get("state"),
            "capture_active": session.get("capture_active") is True,
            "storage_capacity_gib": 40,
            "tailnet_connectivity": "UNVERIFIED",
        },
    }


@app.post("/api/robot/navigation/goal")
async def robot_navigation_goal(request: Request):
    payload = _navigation_payload(await request.json())
    if payload.get("mission_id") and payload.get("task_id"):
        mission, task = _load_mission_task(str(payload["mission_id"]), str(payload["task_id"]))
        _enforce_mission_route_audit(mission, task)
        try:
            record = _navigation_bridge_store.compile_from_mission(
                mission,
                task,
                goal_id=payload.get("goal_id"),
                lease_id=payload.get("lease_id"),
                max_speed_mps=payload.get("max_speed_mps"),
                max_duration_s=payload.get("max_duration_s"),
            )
        except NavigationBridgeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        try:
            record = _navigation_bridge_store.compile_from_payload(payload)
        except NavigationBridgeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record["mode"] == "AUTONOMOUS" and record["state"] == "NAV_READY":
        try:
            return _navigation_bridge_store.activate()
        except NavigationBridgeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record


@app.post("/api/robot/navigation/lease")
async def robot_navigation_lease(request: Request):
    payload = _navigation_payload(await request.json())
    lease_id = payload.get("lease_id")
    if lease_id is None:
        raise HTTPException(status_code=422, detail="lease_id is required")
    if payload.get("mission_id") and payload.get("task_id"):
        mission_id = str(payload["mission_id"])
        task_id = str(payload["task_id"])
        _, task = _load_mission_task(mission_id, task_id)
        try:
            updated = _mission_store.update_task(
                mission_id,
                task_id,
                lease_id=lease_id,
                expected_revision=payload.get("expected_revision", task.get("revision")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        updated_task = next(item for item in updated.get("tasks") or [] if item.get("task_id") == task_id)
        try:
            record = _navigation_bridge_store.compile_from_mission(
                updated,
                updated_task,
                lease_id=lease_id,
            )
        except NavigationBridgeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if record["mode"] == "AUTONOMOUS" and record["state"] == "NAV_READY":
            try:
                return _navigation_bridge_store.activate()
            except NavigationBridgeError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return record
    try:
        return _navigation_bridge_store.update_lease(str(lease_id))
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/navigation/pause")
def robot_navigation_pause():
    try:
        return _navigation_bridge_store.pause()
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/navigation/resume")
def robot_navigation_resume():
    try:
        return _navigation_bridge_store.resume()
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/navigation/cancel")
def robot_navigation_cancel():
    try:
        return _navigation_bridge_store.cancel()
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/navigation/safety-stop")
async def robot_navigation_safety_stop(request: Request):
    payload = _navigation_payload(await request.json())
    try:
        return _navigation_bridge_store.safety_stop(reason=str(payload.get("reason") or "SAFETY_STOP"))
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/navigation/acknowledge")
def robot_navigation_acknowledge():
    try:
        return _navigation_bridge_store.acknowledge_safety_stop()
    except NavigationBridgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robot/safety/acknowledge")
def robot_safety_acknowledge():
    """Compatibility safety endpoint for operator takeover acknowledgement."""

    return robot_navigation_acknowledge()


@app.get("/api/readiness")
def readiness():
    return {
        "ready": True,
        "dashboard_protocol": "rgbd-producer-epoch-v2",
        "runtime_config_sha256": _runtime_config_sha256,
        "room_scan": mapper.room_scan_snapshot(),
        "building_scan": _building_scan_snapshot(),
        "session": robot_session.snapshot(),
    }


@app.get("/api/live/readiness")
async def live_readiness():
    return {
        "ready": True,
        "live_protocol": 3,
        "transport": "tailnet-websocket",
        "texture_keyframes": {
            "endpoint": "/ws/live/texture",
            "message_type": "texture_keyframe",
            "ack_type": "texture_keyframe_ack",
            "independent_wal": True,
            "max_message_bytes": MAX_MESSAGE_BYTES,
        },
    }


@app.get("/api/runtime/calibration-status")
def runtime_calibration_status():
    """Expose the small runtime calibration contract without building a map snapshot."""

    quality = quality_gate.snapshot()
    transform = quality.get("camera_from_lidar_row_major")
    if not isinstance(transform, list):
        transform = []
    eligible = quality.get("eligible_for")
    if not isinstance(eligible, dict):
        eligible = {}
    return {
        "calibration_id": quality.get("calibration_id"),
        "formal_runtime_override": quality.get("formal_runtime_override") is True,
        "measurement_authority": quality.get("measurement_authority") is True,
        "runtime_authority": quality.get("runtime_authority"),
        "camera_from_lidar_row_major": transform,
        "runtime_projection_authorized": (
            len(transform) == 16
            and eligible.get("rgbd_or_neural_geometry_fusion") is True
        ),
    }


@app.get("/api/stack/readiness")
async def stack_readiness():
    return {
        "ready": True,
        "dashboard_protocol": "rgbd-producer-epoch-v2",
        "live_protocol": 3,
        "automatic_reconstruction_pointer_protocol": _automatic_reconstruction_pointer_protocol,
        "runtime_config_sha256": _runtime_config_sha256,
        "capabilities": {
            "building_render_manifest": True,
            "camera_mjpeg": True,
            "field_acceptance_evidence": True,
            "textured_tile_export": True,
            "textured_building_export": True,
        },
    }


@app.get("/api/building/surface/manifest")
def building_surface_manifest(scope: str = "building"):
    scope = str(scope).lower()
    if scope not in {"local", "building", "diagnostic"}:
        raise HTTPException(status_code=400, detail="tile manifest scope is invalid")
    if _live_surface_pipeline is None:
        return {"building_scan": _building_scan_snapshot(), "schema_version": 2, "scope": scope, "tiles": []}
    if scope in {"local", "diagnostic"}:
        manifest = _live_surface_pipeline.tile_store.manifest(scope=scope)
        building_scan = _building_scan_telemetry_snapshot()
        if scope == "local":
            active_submap_id = building_scan.get("active_submap_id")
            manifest["tiles"] = [
                tile
                for tile in manifest.get("tiles", [])
                if active_submap_id is not None and tile.get("submap_id") == active_submap_id
            ]
        manifest["building_scan"] = building_scan
        return manifest
    try:
        manifest = _live_surface_pipeline.manifest(scope=scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest["building_scan"] = _building_scan_snapshot()
    return manifest


@app.get("/api/building/trajectories")
def building_trajectories(
    building_id: str,
    floor_id: str | None = None,
    session_id: str | None = None,
):
    try:
        return _trajectory_store.query(
            building_id=building_id,
            floor_id=floor_id,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/building/render/manifest")
def building_render_manifest(building_id: str | None = None):
    if _active_derived_surface_preview() is not None or _legacy_surface_preview is not None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    selected_building = building_id or _active_building_id()
    try:
        try:
            published = _refresh_render_manifest(selected_building)
        except (KeyError, OSError, RuntimeError, ValueError):
            published = _render_manifest_store.current(selected_building)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"render manifest is unavailable: {exc}") from exc
    body = published.manifest_path.read_bytes()
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "ETag": f'"{published.revision_id}"',
            "Cache-Control": "no-cache",
            "X-Go2-Scene-Revision": published.revision_id,
            "X-Go2-Manifest-SHA256": hashlib.sha256(body).hexdigest(),
        },
    )


@app.get("/api/building/render/provisional/observations/{revision_id}/color.jpg")
def provisional_texture_color(revision_id: str):
    try:
        published = _provisional_texture_store.load(revision_id)
        body = published.color_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="provisional texture is unavailable") from exc
    return Response(
        content=body,
        media_type="image/jpeg",
        headers={
            "ETag": f'"{published.manifest["content_sha256"]}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@app.get("/api/building/render/provisional/assignments/{revision_id}/visibility.bin")
def provisional_texture_visibility(revision_id: str):
    try:
        published = _provisional_texture_store.load_assignment(revision_id)
        body = published.visibility_mask_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="provisional visibility mask is unavailable") from exc
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "ETag": f'"{published.manifest["visibility_mask_sha256"]}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@app.get("/api/building/surface/tiles/{revision_id}")
def building_surface_tile(revision_id: str):
    if _live_surface_pipeline is None:
        raise HTTPException(status_code=404, detail="live building surface is unavailable")
    try:
        metadata, body = _live_surface_pipeline.tile_store.load(revision_id)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="verified live surface tile revision is unavailable") from None
    return Response(
        content=body,
        media_type="application/vnd.go2.surface-tile",
        headers={
            "ETag": f'"{revision_id}"',
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Go2-Tile-SHA256": metadata["content_sha256"],
        },
    )


_LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ATLAS_ARTIFACT_RE = re.compile(r"^atlas-(0|[1-9][0-9]*)\.png$", re.ASCII)
_LOD_ARTIFACTS = {"render-vertices.bin", "render-indices.bin", "render-uvs.bin"}
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


def _appearance_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _LOWER_SHA256_RE.fullmatch(value) is None:
        raise HTTPException(status_code=422, detail=f"{name} must be a lowercase SHA-256")
    return value


def _appearance_json_response(published, *, current: bool) -> Response:
    body = published.manifest_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != published.manifest_sha256:
        raise HTTPException(status_code=404, detail="tile appearance manifest changed after verification")
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Cache-Control": "no-store" if current else _IMMUTABLE_CACHE,
            "ETag": f'"{published.manifest_sha256}"',
        },
    )


def _active_building_id() -> str:
    pipeline = _live_surface_pipeline
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Building appearance is unavailable")
    building_id = _building_scan_telemetry_snapshot().get("building_id")
    if not isinstance(building_id, str) or not building_id:
        raise HTTPException(
            status_code=404, detail="Building appearance requires one active Building"
        )
    return building_id


def _building_appearance_json_response(published, *, current: bool) -> Response:
    try:
        body = published.manifest_path.read_bytes()
    except OSError:
        raise HTTPException(
            status_code=404, detail="Building appearance manifest changed after verification"
        ) from None
    if hashlib.sha256(body).hexdigest() != published.manifest_sha256:
        raise HTTPException(
            status_code=404, detail="Building appearance manifest changed after verification"
        )
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Cache-Control": "no-store" if current else _IMMUTABLE_CACHE,
            "ETag": f'"{published.manifest_sha256}"',
        },
    )


@app.get("/api/building/appearance/status")
def building_appearance_status():
    pipeline = _live_surface_pipeline
    if pipeline is None:
        payload = {
            "schema_version": 1,
            "available": False,
            "building_id": None,
            "building_appearance_revision_id": None,
            "blocker": "BUILDING_PIPELINE_UNAVAILABLE",
        }
    else:
        building_id = pipeline.snapshot().get("building_id")
        published = None
        if isinstance(building_id, str) and building_id:
            try:
                published = pipeline.current_building_appearance(building_id)
            except (FileNotFoundError, OSError, ValueError):
                published = None
        payload = {
            "schema_version": 1,
            "available": published is not None,
            "building_id": building_id if isinstance(building_id, str) else None,
            "building_appearance_revision_id": (
                published.building_appearance_revision_id if published is not None else None
            ),
            "blocker": None if published is not None else "BUILDING_APPEARANCE_UNAVAILABLE",
        }
    return Response(
        content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/building/appearance/manifest")
def building_appearance_current(
    expected_assembly_id: str | None = None,
    expected_source_generation_id: str | None = None,
):
    expected_assembly = (
        None
        if expected_assembly_id is None
        else _appearance_sha256(expected_assembly_id, "expected_assembly_id")
    )
    expected_source = (
        None
        if expected_source_generation_id is None
        else _appearance_sha256(
            expected_source_generation_id, "expected_source_generation_id"
        )
    )
    building_id = _active_building_id()
    try:
        published = _live_surface_pipeline.current_building_appearance(
            building_id,
            expected_assembly_id=expected_assembly,
            expected_source_generation_id=expected_source,
        )
    except ValueError as exc:
        if "stale" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(
            status_code=404, detail="verified Building appearance CURRENT is unavailable"
        ) from None
    if published is None:
        raise HTTPException(status_code=404, detail="Building appearance CURRENT is unavailable")
    return _building_appearance_json_response(published, current=True)


@app.get(
    "/api/building/appearance/revisions/{building_appearance_revision_id}/manifest"
)
def building_appearance_revision_manifest(building_appearance_revision_id: str):
    revision = _appearance_sha256(
        building_appearance_revision_id, "building_appearance_revision_id"
    )
    building_id = _active_building_id()
    try:
        published = _live_surface_pipeline.resolve_building_appearance(
            building_id, revision
        )
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404, detail="verified exact Building appearance is unavailable"
        ) from None
    if published is None:
        raise HTTPException(
            status_code=404, detail="verified exact Building appearance is unavailable"
        )
    return _building_appearance_json_response(published, current=False)


@app.get("/api/building/appearance/tiles/{tile_revision_id}/manifest")
def building_tile_appearance_current(
    tile_revision_id: str,
    expected_source_generation_id: str | None = None,
):
    tile = _appearance_sha256(tile_revision_id, "tile_revision_id")
    expected = None
    if expected_source_generation_id is not None:
        expected = _appearance_sha256(
            expected_source_generation_id, "expected_source_generation_id"
        )
    try:
        published = tile_appearance_store.current(
            tile, expected_source_generation_id=expected
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="tile appearance CURRENT is unavailable") from None
    except ValueError as exc:
        if "stale" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=404, detail="verified tile appearance CURRENT is unavailable") from None
    _require_building_authorized(published)
    return _appearance_json_response(published, current=True)


@app.get("/api/building/appearance/tiles/{tile_revision_id}/{appearance_revision_id}/manifest")
def building_tile_appearance_manifest(
    tile_revision_id: str,
    appearance_revision_id: str,
):
    tile = _appearance_sha256(tile_revision_id, "tile_revision_id")
    appearance = _appearance_sha256(appearance_revision_id, "appearance_revision_id")
    try:
        published = tile_appearance_store.resolve(tile, appearance)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="verified exact tile appearance is unavailable") from None
    _require_building_authorized(published)
    return _appearance_json_response(published, current=False)


@app.get("/api/building/appearance/tiles/{tile_revision_id}/{appearance_revision_id}/vertex-color.bin")
def building_tile_vertex_color(
    tile_revision_id: str,
    appearance_revision_id: str,
):
    tile = _appearance_sha256(tile_revision_id, "tile_revision_id")
    appearance = _appearance_sha256(appearance_revision_id, "appearance_revision_id")
    try:
        _require_building_authorized(tile_appearance_store.resolve(tile, appearance))
        published = tile_vertex_color_store.resolve(tile, appearance)
        body = published.binary_path.read_bytes()
    except (FileNotFoundError, ValueError, OSError):
        raise HTTPException(status_code=404, detail="verified tile vertex color is unavailable") from None
    digest = hashlib.sha256(body).hexdigest()
    reference = published.manifest.get("vertex_color", {})
    if reference.get("bytes") != len(body) or reference.get("sha256") != digest:
        raise HTTPException(status_code=404, detail="tile vertex color changed after verification")
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={"Cache-Control": _IMMUTABLE_CACHE, "ETag": f'"{digest}"'},
    )


@app.get("/api/building/appearance/tiles/{tile_revision_id}/{appearance_revision_id}/{lod_id}/{artifact_name}")
def building_tile_appearance_artifact(
    tile_revision_id: str,
    appearance_revision_id: str,
    lod_id: str,
    artifact_name: str,
):
    atlas_like = artifact_name.startswith("atlas-") and artifact_name.endswith(".png")
    atlas_match = _ATLAS_ARTIFACT_RE.fullmatch(artifact_name)
    if artifact_name not in _LOD_ARTIFACTS and atlas_match is None:
        if atlas_like:
            raise HTTPException(status_code=422, detail="atlas index is malformed")
        raise HTTPException(status_code=404, detail="appearance artifact is not allowlisted")
    tile = _appearance_sha256(tile_revision_id, "tile_revision_id")
    appearance = _appearance_sha256(appearance_revision_id, "appearance_revision_id")
    lod = _appearance_sha256(lod_id, "lod_id")
    try:
        published = tile_appearance_store.resolve(tile, appearance)
        _require_building_authorized(published)
        lod_root = published.lod_paths[lod]
        path = lod_root / artifact_name
        if not path.is_file():
            raise FileNotFoundError
        body = path.read_bytes()
    except (FileNotFoundError, KeyError, ValueError, OSError):
        raise HTTPException(status_code=404, detail="verified exact appearance artifact is unavailable") from None
    digest = hashlib.sha256(body).hexdigest()
    lod_manifest = next(
        (item for item in published.manifest["lods"] if item.get("lod_id") == lod),
        None,
    )
    references = () if lod_manifest is None else lod_manifest.get("core", {}).get("artifacts", ())
    reference = next((item for item in references if item.get("name") == artifact_name), None)
    if reference is None or reference.get("bytes") != len(body) or reference.get("sha256") != digest:
        raise HTTPException(status_code=404, detail="appearance artifact changed after verification")
    return Response(
        content=body,
        media_type="image/png" if atlas_match else "application/octet-stream",
        headers={"Cache-Control": _IMMUTABLE_CACHE, "ETag": f'"{digest}"'},
    )

def ingest_live_keyframe(payload: dict) -> dict:
    """Integrate one keyframe and return only a constant-size acknowledgement."""
    compressed_points = payload.get("points_f32_zlib_b64")
    if compressed_points is None:
        point_bytes = base64.b64decode(payload["points_f32_b64"], validate=True)
    else:
        try:
            point_bytes = zlib.decompress(base64.b64decode(compressed_points, validate=True))
        except (ValueError, TypeError, zlib.error) as exc:
            raise KeyframeError("points_f32_zlib_b64 is not valid lossless point data") from exc
    if len(point_bytes) % (3 * np.dtype("<f4").itemsize) != 0:
        raise KeyframeError("point payload byte count is invalid")
    points = np.frombuffer(point_bytes, dtype="<f4").reshape((-1, 3))
    pose = np.asarray(payload["pose_map_from_lidar"], dtype=np.float64).reshape((4, 4))
    frame = Keyframe(
        map_id=str(payload["map_id"]), sequence=int(payload["sequence"]), timestamp_ns=int(payload["timestamp_ns"]),
        calibration_id=str(payload["calibration_id"]), points_xyz=points,
        imu_quaternion_wxyz=tuple(float(value) for value in payload["imu_quaternion_wxyz"]), pose_map_from_lidar=pose,
        points_in_map_frame=bool(payload.get("points_in_map_frame", False)),
        sensor_frame_transform_id=payload.get("sensor_frame_transform_id"),
    )
    accepted = mapper.integrate(frame, payload.get("registration_residual_m"), bool(payload.get("calibrated_lio", False)))
    duplicate = not accepted and mapper.is_current_duplicate(frame)
    if accepted or duplicate:
        camera_payload = payload.get("camera")
        if camera_payload is not None:
            camera_assist.ingest(camera_payload, frame.calibration_id)
    previous_ack_rtt_ms = payload.get("previous_ack_rtt_ms")
    live_transport.record(
        payload_bytes=len(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")),
        acknowledgement_rtt_ms=None if previous_ack_rtt_ms is None else float(previous_ack_rtt_ms),
        duplicate=duplicate,
        camera_deferred_by_bandwidth=bool(payload.get("camera_deferred_by_bandwidth")),
        camera_depth_deferred_by_bandwidth=bool(payload.get("camera_depth_deferred_by_bandwidth")),
    )
    # The robot needs only an acknowledgement. Returning the full preview map
    # here makes every DERP-backed POST grow with the map and eventually blocks
    # the live producer; snapshots and WebSocket updates carry visualization.
    state = mapper.acknowledgement()
    acknowledgement = {
        "accepted": accepted or duplicate,
        "sequence": frame.sequence,
        "points": state["points"],
        "quality": state["quality"],
    }
    if duplicate:
        acknowledgement["duplicate"] = True
    elif not accepted:
        acknowledgement["next_sequence"] = mapper.next_sequence()
    return acknowledgement


async def _live_ingest_worker() -> None:
    """Commit admitted preview frames in sequence, without loss or reordering."""
    global _live_ingest_last_error
    while True:
        payload = await _live_ingest_queue.get()
        typed_record = payload if isinstance(payload, WalRecord) else None
        try:
            while True:
                try:
                    preview_payload = typed_record.frame.preview_payload() if typed_record is not None else payload
                    result = await asyncio.to_thread(ingest_live_keyframe, preview_payload)
                    sequence = None if typed_record is None else typed_record.frame.identity.sequence
                    next_sequence = result.get("next_sequence")
                    preview_cursor_already_past_record = (
                        typed_record is not None
                        and result.get("accepted") is False
                        and result.get("sequence") == sequence
                        and not isinstance(next_sequence, bool)
                        and isinstance(next_sequence, int)
                        and next_sequence > sequence
                    )
                    if not result.get("accepted") and not preview_cursor_already_past_record:
                        raise RuntimeError("ordered live frame was rejected: %s" % result)
                    if typed_record is not None:
                        update = await asyncio.to_thread(_process_typed_record, typed_record)
                        if update.checkpoint_generation_id is not None:
                            _queue_surface_mesh(update.building_id, update.submap_id)
                        identity = typed_record.frame.identity.as_tuple()
                        _typed_processed.add(identity)
                        _typed_inflight.discard(identity)
                    _live_ingest_last_error = None
                    break
                except Exception as exc:
                    # Do not skip a failed sequence.  Retrying in place keeps
                    # all subsequent frames ordered and makes failure visible.
                    _live_ingest_last_error = "%s: %s" % (type(exc).__name__, exc)
                    await asyncio.sleep(0.1)
        finally:
            _live_ingest_queue.task_done()
            _schedule_pending_typed_records()


def _process_typed_record(record: WalRecord, *, force_checkpoint: bool = False) -> PipelineUpdate:
    if _live_surface_pipeline is None:
        raise RuntimeError("detailed live surface pipeline is not ready")
    update = _live_surface_pipeline.integrate(record, force_checkpoint=force_checkpoint)
    _trajectory_store.append(
        record.frame,
        floor_id=update.floor_id,
        submap_id=update.submap_id,
    )
    return update


def _queue_surface_mesh(building_id: str, submap_id: str) -> None:
    if _surface_mesh_queue is None:
        return
    target = (building_id, submap_id)
    if target in _surface_mesh_pending:
        _surface_mesh_deferred.add(target)
        return
    try:
        _surface_mesh_queue.put_nowait(target)
        _surface_mesh_pending.add(target)
    except asyncio.QueueFull:
        _surface_mesh_deferred.add(target)


def _promote_deferred_surface_mesh() -> None:
    if _surface_mesh_queue is None:
        return
    now = time.monotonic()
    for target in sorted(_surface_mesh_deferred):
        if target in _surface_mesh_pending:
            continue
        if _surface_mesh_retry_after.get(target, 0.0) > now:
            continue
        try:
            _surface_mesh_queue.put_nowait(target)
        except asyncio.QueueFull:
            return
        _surface_mesh_pending.add(target)
        _surface_mesh_deferred.discard(target)


async def _surface_checkpoint_loop() -> None:
    while True:
        await asyncio.sleep(min(0.5, _surface_checkpoint_interval_s))
        _promote_deferred_surface_mesh()
        if _live_surface_pipeline is None:
            continue
        updates = await asyncio.to_thread(_live_surface_pipeline.flush_due)
        for building_id, submap_id, _ in updates:
            _queue_surface_mesh(building_id, submap_id)


async def _surface_mesh_worker() -> None:
    while True:
        target = await _surface_mesh_queue.get()
        failed = False
        try:
            if _live_surface_pipeline is not None:
                result = await asyncio.to_thread(_live_surface_pipeline.mesh, *target)
                if isinstance(result, dict) and result.get("remaining_dirty_tiles", 0) > 0:
                    _surface_mesh_deferred.add(target)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(_refresh_render_manifest, target[0])
        except Exception:
            failed = True
            _surface_mesh_deferred.add(target)
            _surface_mesh_retry_after[target] = time.monotonic() + _surface_mesh_retry_delay_s
        finally:
            _surface_mesh_pending.discard(target)
            if not failed:
                _surface_mesh_retry_after.pop(target, None)
            _promote_deferred_surface_mesh()
            _surface_mesh_queue.task_done()


def _schedule_pending_typed_records() -> None:
    if _surface_recovery_task is not None and not _surface_recovery_complete:
        return
    if _live_wal is None or _live_ingest_queue is None:
        return
    for record in _live_wal.pending():
        identity = record.frame.identity.as_tuple()
        if identity in _typed_inflight or identity in _typed_processed:
            continue
        try:
            _live_ingest_queue.put_nowait(record)
        except asyncio.QueueFull:
            return
        _typed_inflight.add(identity)


def _dispatch_live_message(
    payload: dict,
    *,
    message_bytes: int | None = None,
    schedule_pending: bool = True,
) -> dict:
    """Durably dispatch protocol-v3 input; retain untyped preview compatibility."""
    global _live_ingest_last_error, _live_source_status_stream
    message_type = payload.get("message_type") if isinstance(payload, dict) else None
    if message_type == "live_keyframe":
        frame = validate_live_keyframe(payload, message_bytes=message_bytes)
        if _live_wal is None or _live_surface_pipeline is None:
            raise RuntimeError("detailed live surface pipeline is not ready")
        try:
            admission = _live_surface_pipeline.admit(frame)
        except OSError:
            _live_ingest_last_error = "WAL_STORAGE_ERROR"
            return build_live_keyframe_ack(
                frame,
                accepted=False,
                disposition="REJECTED",
                wal_record_sha256=None,
                next_sequence=frame.identity.sequence,
                error_code="WAL_STORAGE_ERROR",
            )
        if admission.error_code == "WAL_CAPACITY_EXCEEDED":
            _live_ingest_last_error = "SURFACE_CAPACITY_BLOCKED"
        if admission.accepted and schedule_pending:
            _schedule_pending_typed_records()
        return build_live_keyframe_ack(
            frame,
            accepted=admission.accepted,
            disposition=admission.disposition,
            wal_record_sha256=admission.record_sha256,
            next_sequence=admission.next_sequence,
            error_code=admission.error_code,
        )
    if message_type == "surface_source_status":
        status = validate_surface_source_status(payload)
        if _live_wal is None:
            raise RuntimeError("detailed live WAL is not ready")
        try:
            accepted = _live_wal.accept_status(status)
        except OSError:
            _live_ingest_last_error = "STATUS_STORAGE_ERROR"
            return build_surface_source_status_ack(status, accepted=False, error_code="STATUS_STORAGE_ERROR")
        _live_source_status_stream = status.identity.stream_key()
        return build_surface_source_status_ack(
            status,
            accepted=accepted,
            error_code=None if accepted else "STATUS_NOT_NEWER",
        )
    if message_type == "camera_preview":
        preview = validate_camera_preview(payload, message_bytes=message_bytes)
        try:
            camera_assist.ingest(preview.camera, preview.calibration_id)
        except CameraAssistError as exc:
            raise LiveProtocolError(str(exc), "INVALID_CAMERA_PREVIEW") from exc
        return build_camera_preview_ack(preview, accepted=True)
    if message_type is not None:
        raise LiveProtocolError("unsupported live message type", "UNSUPPORTED_MESSAGE_TYPE")
    return _queue_live_keyframe(payload)


def _dispatch_texture_message(payload: dict, *, message_bytes: int | None = None) -> dict:
    """Durably admit a texture frame without sharing LiDAR or preview flow control."""
    frame = validate_texture_keyframe(payload, message_bytes=message_bytes)
    if _texture_wal is None:
        raise RuntimeError("texture keyframe WAL is not ready")
    try:
        admission = _texture_wal.admit(frame)
    except OSError:
        return build_texture_keyframe_ack(
            frame,
            accepted=False,
            disposition="REJECTED",
            wal_record_sha256=None,
            next_sequence=frame.identity.sequence,
            error_code="WAL_STORAGE_ERROR",
        )
    return build_texture_keyframe_ack(
        frame,
        accepted=admission.accepted,
        disposition=admission.disposition,
        wal_record_sha256=admission.record_sha256,
        next_sequence=admission.next_sequence,
        error_code=admission.error_code,
    )


async def _dispatch_live_message_async(payload: dict, *, message_bytes: int | None = None) -> dict:
    acknowledgement = await asyncio.to_thread(
        _dispatch_live_message,
        payload,
        message_bytes=message_bytes,
        schedule_pending=False,
    )
    if payload.get("message_type") == "live_keyframe" and acknowledgement.get("accepted"):
        _schedule_pending_typed_records()
    return acknowledgement


def _queue_live_keyframe(payload: dict) -> dict:
    global _live_ingest_next_sequence
    if not isinstance(payload, dict):
        raise ValueError("live keyframe must be a JSON object")
    try:
        sequence = int(payload["sequence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("live keyframe sequence is required") from exc
    if sequence < 0:
        raise ValueError("live keyframe sequence must be non-negative")
    if _live_ingest_queue is None:
        raise RuntimeError("live ingest queue is not ready")
    # The robot stores its sequence across a Dashboard restart.  An empty
    # in-memory map has no predecessor to protect, so establish the order from
    # the first received persistent sequence instead of demanding a reset.
    accepting_first_sequence = _live_ingest_next_sequence is None and mapper.next_sequence() == 0
    expected_sequence = sequence if accepting_first_sequence else max(mapper.next_sequence(), _live_ingest_next_sequence)
    identity = (str(payload.get("map_id", "")), sequence, int(payload.get("timestamp_ns", 0)), str(payload.get("calibration_id", "")))
    if sequence < expected_sequence:
        if _live_ingest_last_admitted.get(sequence) == identity:
            state = mapper.acknowledgement()
            return {"accepted": True, "sequence": sequence, "points": state["points"], "quality": state["quality"], "duplicate": True}
        state = mapper.acknowledgement()
        return {"accepted": False, "sequence": sequence, "points": state["points"], "quality": state["quality"], "next_sequence": expected_sequence}
    if sequence > expected_sequence:
        state = mapper.acknowledgement()
        return {"accepted": False, "sequence": sequence, "points": state["points"], "quality": state["quality"], "next_sequence": expected_sequence}
    try:
        _live_ingest_queue.put_nowait(payload)
    except asyncio.QueueFull:
        state = mapper.acknowledgement()
        return {
            "accepted": False,
            "sequence": sequence,
            "points": state["points"],
            "quality": state["quality"],
            "error": "live ingest backlog is full; retry this same sequence",
            "retry_after_ms": 100,
        }
    _live_ingest_last_admitted[sequence] = identity
    _live_ingest_next_sequence = sequence + 1
    if len(_live_ingest_last_admitted) > _live_ingest_queue_size * 2:
        oldest = min(_live_ingest_last_admitted)
        del _live_ingest_last_admitted[oldest]
    state = mapper.acknowledgement()
    return {"accepted": True, "sequence": sequence, "points": state["points"], "quality": state["quality"]}


@app.post("/api/live/keyframe")
async def live_keyframe(request: Request):
    token = os.environ.get("GO2_MAPPING_LIVE_TOKEN")
    if token and request.headers.get("X-Go2-Live-Token") != token:
        raise HTTPException(status_code=401, detail="live keyframe token rejected")
    try:
        payload = await request.json()
        if isinstance(payload, dict) and payload.get("message_type") is not None:
            return _dispatch_live_message(payload)
        return ingest_live_keyframe(payload)
    except (KeyError, TypeError, ValueError, KeyframeError, LiveProtocolError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/camera/frame")
def camera_frame():
    frame = camera_assist.frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="camera assist frame is unavailable")
    return frame


@app.get("/api/camera/status")
def camera_status():
    status = camera_assist.snapshot()
    transport = live_transport.snapshot()
    status["deferred_by_bandwidth"] = bool(transport.get("camera_deferred_by_bandwidth"))
    status["depth_deferred_by_bandwidth"] = bool(
        transport.get("camera_depth_deferred_by_bandwidth")
    )
    return status


def _camera_mjpeg_parts(*, wait_timeout: float = 1.0):
    identity = None
    while True:
        frame = camera_assist.wait_for_color_frame(identity, timeout=wait_timeout)
        if frame is None:
            continue
        identity = frame["identity"]
        jpeg = frame["jpeg"]
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n".encode("ascii")
            + f"X-Go2-Camera-Identity: {identity}\r\n\r\n".encode("ascii")
            + jpeg
            + b"\r\n"
        )


@app.get("/api/camera/stream.mjpeg")
def camera_stream():
    return StreamingResponse(
        _camera_mjpeg_parts(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/robot/session")
async def robot_session_heartbeat(request: Request):
    try:
        return robot_session.ingest(await request.json())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


_ROOM_SCAN_REQUEST_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.ASCII,
)


def _room_scan_operator_token(request: Request) -> str:
    return request.headers.get("X-Go2-Room-Scan-Operator-Token", "")


def _raise_room_scan_error(error: RoomScanError) -> None:
    raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@app.post("/api/room-scan/start")
async def room_scan_start(request: Request):
    try:
        payload = await request.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="room scan payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="room scan payload must be an object")
    operator_token = str(payload.get("operator_token") or "").strip()
    if not operator_token:
        raise HTTPException(status_code=401, detail="room scan operator token is required")
    duration = payload.get("duration_seconds", 600)
    if isinstance(duration, bool) or not isinstance(duration, int) or not 60 <= duration <= 3600:
        raise HTTPException(status_code=422, detail="duration_seconds must be between 60 and 3600")
    try:
        result = _room_scan_service.start(duration, operator_token)
    except RoomScanError as exc:
        _raise_room_scan_error(exc)
    return {
        **result,
        "mode": "full",
        "coordinate_scope": "SESSION_ODOM_LOCAL",
        "geometry_only_calibration": True,
        "measurement_authority": False,
        "read_only": True,
    }


@app.get("/api/room-scan/capabilities")
def room_scan_capabilities():
    return _room_scan_service.capabilities()


@app.get("/api/room-scan/status/{request_id}")
def room_scan_status(request_id: str, request: Request):
    if _ROOM_SCAN_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise HTTPException(status_code=422, detail="request_id must be a canonical UUIDv4")
    token = _room_scan_operator_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="room scan operator token is required")
    try:
        return _room_scan_service.status(request_id, token)
    except RoomScanError as exc:
        _raise_room_scan_error(exc)


@app.post("/api/room-scan/stop/{request_id}")
def room_scan_stop(request_id: str, request: Request):
    if _ROOM_SCAN_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise HTTPException(status_code=422, detail="request_id must be a canonical UUIDv4")
    token = _room_scan_operator_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="room scan operator token is required")
    try:
        return _room_scan_service.stop(request_id, token)
    except RoomScanError as exc:
        _raise_room_scan_error(exc)


@app.get("/api/calibration/progress")
def calibration_progress():
    return calibration_capture.snapshot()


@app.get("/api/reconstruction/measured-surface")
def measured_surface():
    active_derived = _active_derived_surface_preview()
    if active_derived is not None:
        return FileResponse(
            str(active_derived.surface_path),
            media_type="application/json",
            headers={
                "Cache-Control": "no-store",
                "X-Go2-Derived-Cache-Key": active_derived.status["derived_cache_key"],
                "X-Content-SHA256": active_derived.status["surface_sha256"],
            },
        )
    if _legacy_surface_preview is not None:
        return FileResponse(
            str(_legacy_surface_preview.surface_path),
            media_type="application/vnd.go2.legacy-surface-ply",
            headers={
                "Cache-Control": "no-store",
                "X-Go2-Legacy-Session": _legacy_surface_preview.status["session_id"],
                "X-Content-SHA256": _legacy_surface_preview.status["surface_sha256"],
            },
        )
    surface = dense_mapper.measured_surface_snapshot()
    if surface is None:
        raise HTTPException(status_code=423, detail="LIDAR_MEASURED surface is not available yet")
    return surface


@app.get("/api/reconstruction/measured-surface/status")
def measured_surface_status():
    active_derived = _active_derived_surface_preview()
    if active_derived is not None:
        payload = dict(active_derived.status)
        if _automatic_reconstruction_status_path:
            payload["automatic_reconstruction_status"] = automatic_reconstruction_status()
    elif _legacy_surface_preview is not None:
        payload = _legacy_surface_preview.status
    else:
        available = dense_mapper.measured_surface_snapshot() is not None
        payload = {
            "schema_version": 1,
            "available": available,
            "blocker": None if available else "LIDAR_MEASURED_SURFACE_UNAVAILABLE",
        }
    return Response(
        content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/reconstruction/session-odom-local")
def session_odom_local_surface():
    """Serve the explicitly selected local validation surface, never the formal map."""

    session_odom_local_status()
    if _session_odom_local_preview is None:
        raise HTTPException(status_code=423, detail="SESSION_ODOM_LOCAL surface is not available yet")
    return FileResponse(
        str(_session_odom_local_preview.surface_path),
        media_type="application/json",
        headers={
            "Cache-Control": "no-store",
            "X-Go2-Derived-Cache-Key": _session_odom_local_preview.status["derived_cache_key"],
            "X-Content-SHA256": _session_odom_local_preview.status["surface_sha256"],
        },
    )


@app.get("/api/reconstruction/session-odom-local/status")
def session_odom_local_surface_status():
    status = session_odom_local_status()
    if status is None or _session_odom_local_preview is None:
        payload = {
            "schema_version": 1,
            "available": False,
            "blocker": "SESSION_ODOM_LOCAL_SURFACE_UNAVAILABLE",
        }
    else:
        payload = {
            **_session_odom_local_preview.status,
            **status,
            "source_mode": "SESSION_ODOM_LOCAL_READ_ONLY",
            "read_only": True,
        }
    return Response(
        content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _room_direct_rgb_manifest() -> tuple[Path, dict] | None:
    manifests = sorted(_room_direct_rgb_preview_root.glob("*/ */ */ */manifest.json".replace(" ", "")), key=lambda p: p.stat().st_mtime)
    if not manifests:
        return None
    path = manifests[-1].resolve(strict=False)
    try:
        path.relative_to(_room_direct_rgb_preview_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return path, payload


@app.get("/api/reconstruction/session-odom-local/appearance")
def session_odom_local_appearance():
    found = _room_direct_rgb_manifest()
    if found is None:
        return _appearance_blocked(["ROOM_DIRECT_RGB_PREVIEW_UNAVAILABLE"])
    path, payload = found
    return {**payload, "delivery_authorized": False, "texture_state": "DIRECT_RGB_VERTEX_COLOR", "manifest_url": "/api/reconstruction/session-odom-local/appearance"}


@app.get("/api/reconstruction/session-odom-local/appearance/vertex-color.bin")
def session_odom_local_vertex_color():
    found = _room_direct_rgb_manifest()
    if found is None:
        raise HTTPException(status_code=404, detail="room direct RGB sidecar is unavailable")
    _manifest_path, payload = found
    binary = Path(str(_manifest_path).replace("manifest.json", "vertex-color.bin")).resolve(strict=False)
    try:
        binary.relative_to(_room_direct_rgb_preview_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="room direct RGB sidecar path is invalid") from None
    if not binary.is_file():
        raise HTTPException(status_code=404, detail="room direct RGB sidecar is unavailable")
    return FileResponse(str(binary), filename="vertex-color.bin", media_type="application/octet-stream", headers={"X-Preview-Revision": str(payload.get("preview_revision_id", ""))})


def _field_identity_digest(field: str, members: list[dict]) -> str:
    return canonical_sha256({
        "schema_version": 1,
        "field": field,
        "members": [member[field] for member in members],
    })


@app.get("/api/reconstruction/field-acceptance-evidence")
def field_acceptance_evidence():
    """Read exact runtime evidence without exporting or mutating reconstruction state."""

    blockers: list[str] = []
    pipeline = _live_surface_pipeline
    published = None
    pipeline_snapshot: dict = {}
    if pipeline is None:
        blockers.append("BUILDING_PIPELINE_UNAVAILABLE")
    else:
        pipeline_snapshot = pipeline.snapshot()
        building_id = pipeline_snapshot.get("building_id")
        if not isinstance(building_id, str) or not building_id:
            blockers.append("ACTIVE_BUILDING_UNAVAILABLE")
        else:
            try:
                published = pipeline.current_building_appearance(building_id)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                published = None
            if published is None:
                blockers.append("BUILDING_APPEARANCE_UNAVAILABLE")

    refiner_evidence = rgbd_refiner.field_acceptance_snapshot()
    reports = refiner_evidence.get("accepted_depth_reports")
    tile_reports = [
        tile
        for report in reports if isinstance(report, dict)
        for tile in report.get("tiles", ()) if isinstance(tile, dict)
    ] if isinstance(reports, list) else []
    if not tile_reports:
        blockers.append("ACCEPTED_PHYSICAL_DEPTH_EVIDENCE_UNAVAILABLE")
        physical_depth = {}
    else:
        try:
            physical_depth = {
                "correspondence_count": sum(int(tile["correspondence_count"]) for tile in tile_reports),
                "median_m": max(float(tile["median_absolute_residual_m"]) for tile in tile_reports),
                "p95_m": max(float(tile["p95_absolute_residual_m"]) for tile in tile_reports),
                "signed_bias_m": max(
                    (float(tile["mean_signed_residual_m"]) for tile in tile_reports),
                    key=abs,
                ),
                "occlusion_fraction": max(float(tile["occlusion_conflict_fraction"]) for tile in tile_reports),
                "free_space_fraction": max(float(tile["free_space_violation_fraction"]) for tile in tile_reports),
            }
        except (KeyError, TypeError, ValueError):
            physical_depth = {}
            blockers.append("ACCEPTED_PHYSICAL_DEPTH_EVIDENCE_INVALID")

    assistance = refiner_evidence.get("depth_anything")
    metrics = assistance.get("validation_metrics") if isinstance(assistance, dict) else None
    if not isinstance(metrics, dict) or assistance.get("fit_accepted") is not True:
        blockers.append("DEPTH_ANYTHING_VALIDATION_EVIDENCE_UNAVAILABLE")
        metrics = {}

    members: list[dict] = []
    identity: dict = {}
    local_requests: list[dict] = []
    building_request: dict = {}
    surface = {}
    if published is not None:
        manifest = published.manifest
        manifest_members = manifest.get("members") if isinstance(manifest, dict) else None
        building_revision = manifest.get("building_appearance_revision_id") if isinstance(manifest, dict) else None
        assembly_id = manifest.get("assembly_id") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest_members, list)
            or not manifest_members
            or not is_sha256(building_revision)
            or building_revision != str(building_revision).lower()
            or not is_sha256(assembly_id)
            or assembly_id != str(assembly_id).lower()
        ):
            blockers.append("BUILDING_APPEARANCE_IDENTITY_INVALID")
        else:
            required_fields = (
                "tile_revision_id",
                "tile_appearance_revision_id",
                "placement_revision_id",
                "source_generation_id",
            )
            for member in manifest_members:
                if not isinstance(member, dict):
                    blockers.append("BUILDING_MEMBER_IDENTITY_INVALID")
                    continue
                lods = member.get("authorized_lod_ids")
                authorization = member.get("delivery_authorization")
                if (
                    any(
                        not is_sha256(member.get(field))
                        or member.get(field) != str(member.get(field)).lower()
                        for field in required_fields
                    )
                    or not isinstance(lods, list)
                    or not lods
                    or any(not is_sha256(value) or value != value.lower() for value in lods)
                    or not isinstance(authorization, dict)
                    or authorization.get("delivery_authorized") is not True
                ):
                    blockers.append("BUILDING_MEMBER_IDENTITY_INVALID")
                    continue
                members.append(dict(member))
            members.sort(key=lambda value: value["tile_revision_id"])
            if len(members) != len(manifest_members):
                blockers.append("BUILDING_MEMBER_SET_INCOMPLETE")
            elif members:
                coverage_values = [
                    member["delivery_authorization"].get("measured_surface_coverage_ppm")
                    for member in members
                ]
                view_values = [
                    member["delivery_authorization"].get("minimum_independent_view_count")
                    for member in members
                ]
                if any(not isinstance(value, int) for value in coverage_values + view_values):
                    blockers.append("BUILDING_MEMBER_DELIVERY_EVIDENCE_INVALID")
                else:
                    surface = {
                        "coverage_ppm": min(coverage_values),
                        "independent_color_views": min(view_values),
                    }
                identity = {
                    "geometry_revision_id": _field_identity_digest("tile_revision_id", members),
                    "tile_appearance_revision_id": _field_identity_digest(
                        "tile_appearance_revision_id", members
                    ),
                    "building_appearance_revision_id": building_revision,
                    "placement_revision_id": _field_identity_digest("placement_revision_id", members),
                    "source_generation_id": _field_identity_digest("source_generation_id", members),
                    "assembly_id": assembly_id,
                    "members": members,
                }
                local_requests = [
                    {
                        "tile_revision_id": member["tile_revision_id"],
                        "appearance_revision_id": member["tile_appearance_revision_id"],
                        "lod_id": member["authorized_lod_ids"][0],
                        "rgbd_augmented_revision_id": member.get("rgbd_augmented_revision_id"),
                    }
                    for member in members
                ]
                building_request = {
                    "assembly_id": assembly_id,
                    "building_appearance_revision_id": building_revision,
                    "lod_by_member": {
                        member["tile_revision_id"]: member["authorized_lod_ids"][0]
                        for member in members
                    },
                }

    ingest_snapshot = artifact_ingestor.snapshot() if artifact_ingestor is not None else None
    dense_snapshot = dense_mapper.snapshot()
    counter_values = {
        "ingest_errors": None if not isinstance(ingest_snapshot, dict) else ingest_snapshot.get("errors"),
        "pose_errors": dense_snapshot.get("skipped_pose_frames"),
        "rgbd_errors": (
            int(refiner_evidence.get("rejected_depth_frames", 0))
            + int(assistance.get("failed_frames", 0) if isinstance(assistance, dict) else 0)
        ),
        "surface_errors": int(bool(
            dense_snapshot.get("last_surface_error") or pipeline_snapshot.get("meshing_error")
        )),
    }
    if any(not isinstance(value, int) or value < 0 for value in counter_values.values()):
        blockers.append("FIELD_PIPELINE_ERROR_COUNTS_UNAVAILABLE")
    surface.update(counter_values)

    unique = list(dict.fromkeys(blockers))
    payload = {
        "schema_version": 1,
        "available": not unique,
        "blockers": unique,
        "surface": surface,
        "physical_depth": physical_depth,
        "depth_anything": metrics,
        "identity": identity,
        "exports": {
            "local_request": local_requests[0] if local_requests else None,
            "local_requests": local_requests,
            "building_request": building_request or None,
        },
    }
    return Response(
        content=json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _appearance_blocked(blockers: list[str], *, geometry_revision_id: str | None = None) -> dict:
    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": 1,
        "state": "TEXTURE_NOT_AUTHORIZED",
        "delivery_authorized": False,
        "measured_surface_geometry_revision_id": geometry_revision_id,
        "blockers": unique_blockers,
    }


@app.get("/api/reconstruction/measured-surface/appearance")
def measured_surface_appearance():
    if _active_derived_surface_preview() is not None:
        return _appearance_blocked(["DERIVED_SURFACE_READ_ONLY"])
    if _legacy_surface_preview is not None:
        return _appearance_blocked(["LEGACY_SESSION_READ_ONLY"])
    surface = dense_mapper.measured_surface_snapshot()
    if surface is None:
        return _appearance_blocked(["MEASURED_SURFACE_REQUIRED"])
    blockers: list[str] = []
    geometry_revision_id = surface.get("measured_surface_geometry_revision_id")
    pose_generation = surface.get("pose_generation")
    if not is_sha256(pose_generation) or str(pose_generation) != str(pose_generation).lower():
        blockers.append("POSE_GENERATION_NOT_CONTENT_ADDRESSED")
    if not is_sha256(geometry_revision_id) or str(geometry_revision_id) != str(geometry_revision_id).lower():
        blockers.append("MEASURED_SURFACE_GEOMETRY_REVISION_MISSING")
    source_vertex_ids = surface.get("source_vertex_ids")
    positions = surface.get("positions")
    normals = surface.get("normals")
    triangles = surface.get("triangles")
    if not isinstance(source_vertex_ids, list) or len(source_vertex_ids) != len(positions or ()) // 3:
        blockers.append("MEASURED_SURFACE_VERTEX_IDS_MISSING")
    if surface.get("appearance_blocker"):
        blockers.append(str(surface["appearance_blocker"]))
    if not blockers:
        try:
            identity = measured_surface_identity(
                positions=np.asarray(positions).reshape(-1, 3),
                normals=np.asarray(normals).reshape(-1, 3),
                triangles=np.asarray(triangles).reshape(-1, 3),
                pose_generation=pose_generation,
                artifact_sha256=surface.get("artifact_sha256", ()),
                fusion_config_sha256=load_reconstruction_config().fusion_sha256,
            )
        except (TypeError, ValueError) as exc:
            blockers.append(f"MEASURED_SURFACE_IDENTITY_INVALID:{exc}")
        else:
            if identity.revision_id != geometry_revision_id:
                blockers.append("MEASURED_SURFACE_GEOMETRY_REVISION_STALE")
    if blockers:
        return _appearance_blocked(blockers, geometry_revision_id=geometry_revision_id if is_sha256(geometry_revision_id) else None)

    evidence = rgbd_refiner.appearance_evidence(tuple(source_vertex_ids))
    if not evidence.ready:
        return _appearance_blocked(list(evidence.blockers), geometry_revision_id=geometry_revision_id)
    config = load_reconstruction_config()
    sidecar = build_historical_vertex_color_sidecar(
        geometry_revision_id=geometry_revision_id,
        pose_generation=pose_generation,
        calibration_id=evidence.calibration_id or "",
        calibration_sha256=evidence.calibration_sha256 or "",
        physical_depth_validation_revision=evidence.physical_depth_validation_revision or "",
        appearance_config_sha256=config.appearance_sha256,
        vertex_colors=evidence.vertex_colors,
    )
    published = appearance_store.publish(sidecar, expected_geometry_revision_id=geometry_revision_id)
    unsigned_manifest = dict(published.manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    return {
        **published.manifest,
        "manifest_sha256": published.manifest_sha256,
        "manifest_canonical_json": json.dumps(
            unsigned_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ),
        "manifest_url": "/api/reconstruction/measured-surface/appearance",
    }


def _appearance_surface_payload(surface_id: str) -> dict:
    if not isinstance(surface_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", surface_id):
        raise HTTPException(status_code=422, detail="surface_id is invalid")
    if surface_id in {"session-odom-local", "local"}:
        return _appearance_blocked(["SESSION_ODOM_LOCAL_READ_ONLY", "TEXTURE_NOT_AUTHORIZED"])
    if surface_id not in {"measured-surface", "current"}:
        raise HTTPException(status_code=404, detail="surface appearance is unavailable")
    return measured_surface_appearance()


@app.get("/api/appearance/{surface_id}/status")
def appearance_status(surface_id: str):
    payload = _appearance_surface_payload(surface_id)
    return {"surface_id": surface_id, **payload, "status": payload.get("state")}


@app.post("/api/appearance/{surface_id}/preview")
def appearance_preview(surface_id: str):
    payload = _appearance_surface_payload(surface_id)
    return {"surface_id": surface_id, "preview": True, **payload}


@app.post("/api/appearance/{surface_id}/validate")
def validate_appearance(surface_id: str):
    payload = _appearance_surface_payload(surface_id)
    if payload.get("delivery_authorized") is not True:
        raise HTTPException(
            status_code=423,
            detail={
                "state": "FORMAL_TEXTURE_VALIDATION_BLOCKED",
                "surface_id": surface_id,
                "blockers": list(payload.get("blockers") or ["TEXTURE_NOT_AUTHORIZED"]),
            },
        )
    return {"surface_id": surface_id, "validated": True, **payload}


@app.get("/api/reconstruction/appearance/{appearance_revision_id}/vertex-color.bin")
def reconstruction_vertex_color(appearance_revision_id: str):
    try:
        published = appearance_store.load(appearance_revision_id)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="verified appearance sidecar is unavailable") from None
    return FileResponse(
        published.binary_path,
        filename="vertex-color.bin",
        media_type="application/octet-stream",
        headers={"X-Manifest-SHA256": published.manifest_sha256},
    )


def _published_payload(metadata: dict, *, ply_name: str, metadata_name: str) -> dict:
    return {
        **metadata,
        "ply_name": ply_name,
        "metadata_name": metadata_name,
        "ply_url": f"/api/reconstruction/artifacts/{ply_name}",
        "metadata_url": f"/api/reconstruction/artifacts/{metadata_name}",
    }


def _appearance_export_sources() -> AppearanceExportSources:
    pipeline = _live_surface_pipeline
    if pipeline is None:
        raise RuntimeError("detailed live surface pipeline is unavailable")
    building_id = _active_building_id()
    return AppearanceExportSources(
        tile_geometry=pipeline.tile_store.load,
        tile_appearance=tile_appearance_store.resolve,
        building_appearance=lambda revision: pipeline.resolve_building_appearance(building_id, revision),
        assembly=lambda assembly_id: pipeline.current_assembly(building_id, assembly_id),
        augmentation=_augmentation_store.resolve,
    )


@app.post("/api/reconstruction/publish-measured")
def publish_measured_reconstruction():
    if _active_derived_surface_preview() is not None:
        raise HTTPException(status_code=405, detail="derived surface preview is read-only")
    if _legacy_surface_preview is not None:
        raise HTTPException(status_code=405, detail="legacy session preview is read-only")
    surface_snapshot = dense_mapper.measured_surface_snapshot()
    if surface_snapshot is None:
        raise HTTPException(status_code=423, detail="LIDAR_MEASURED evidence is not available yet")
    if surface_snapshot.get("reconstruction_phase") != "IDLE":
        raise HTTPException(status_code=409, detail="reconstruction generation is still changing")

    state = dense_mapper.snapshot()
    artifact_hashes = surface_snapshot.get("artifact_sha256")
    pose_generation = surface_snapshot.get("pose_generation")
    if (
        state.get("reconstruction_phase") not in {None, "IDLE"}
        or state.get("pose_generation") not in {None, pose_generation}
        or state.get("artifacts") not in {None, len(artifact_hashes or ())}
    ):
        raise HTTPException(status_code=409, detail="reconstruction changed while preparing the export")
    if not isinstance(artifact_hashes, list) or not artifact_hashes or any(not is_sha256(item) for item in artifact_hashes):
        raise HTTPException(status_code=409, detail="measured surface has no verified source artifact identity")

    quality = state.get("quality_evidence") or {}
    quality_label = str(quality.get("level") or state.get("quality") or "PREVIEW_UNCALIBRATED")
    calibration_hash = quality.get("calibration_sha256")
    if not is_sha256(calibration_hash):
        calibration_hash = canonical_sha256({"state": "MEASURED_CALIBRATION_MISSING"})
    identity = load_reconstruction_config().derived_identity(
        artifact_hashes=artifact_hashes,
        pose_generation=str(pose_generation),
        calibration_hash=calibration_hash,
        roi_hash=canonical_sha256({"scope": "OBSERVED_LIDAR_DOMAIN"}),
        mask_hash=canonical_sha256({"mask": "NO_EXTERNAL_MASK"}),
        algorithm_version=load_reconstruction_config().fusion.algorithm_version,
    )
    surface = MeasuredSurface(
        positions=np.asarray(surface_snapshot["positions"], dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(surface_snapshot["normals"], dtype=np.float64).reshape(-1, 3),
        triangles=np.asarray(surface_snapshot["triangles"], dtype=np.int32).reshape(-1, 3),
    )
    eligible = quality.get("eligible_for") or {}
    blockers = list(quality.get("blocking_reasons") or ())
    published = reconstruction_store.publish(
        surface,
        identity,
        annotations={
            "artifact_kind": "MEASURED_SURFACE_PREVIEW",
            "quality_label": quality_label,
            "quality_blockers": blockers,
            "delivery_authorized": bool(eligible.get("centimetre_geometry")),
            "reconstruction_phase": "IDLE",
            "voxel_size_m": float(surface_snapshot["voxel_size_m"]),
        },
    )
    return _published_payload(
        published.metadata,
        ply_name=published.ply_path.name,
        metadata_name=published.metadata_path.name,
    )


@app.get("/api/reconstruction/artifacts")
def reconstruction_artifacts():
    return {
        "artifacts": [
            _published_payload(
                metadata,
                ply_name=str(metadata["ply_name"]),
                metadata_name=str(metadata["metadata_name"]),
            )
            for metadata in reconstruction_store.list_verified()
        ]
    }


@app.get("/api/reconstruction/artifacts/{name}")
def reconstruction_artifact(name: str):
    if re.fullmatch(
        r"(?:tile-[0-9a-f]{64}-[0-9a-f]{64}-[0-9a-f]{64}|building-[0-9a-f]{64}-[0-9a-f]{64})\.glb(?:\.provenance\.json)?",
        name,
    ):
        candidate = _textured_export_root / name
        if candidate.is_file() and candidate.parent.resolve() == _textured_export_root.resolve():
            return FileResponse(candidate, filename=candidate.name)
    try:
        path = reconstruction_store.resolve(name)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        if (
            Path(name).name != name
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*\.(?:glb|glb\.provenance\.json)", name)
        ):
            raise HTTPException(status_code=404, detail="verified reconstruction artifact is unavailable") from None
        path = _textured_export_root / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="verified reconstruction artifact is unavailable") from None
    return FileResponse(path, filename=path.name)


def _textured_export_sha(value, name: str) -> str:
    if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
        raise HTTPException(status_code=422, detail=f"{name} must be a lowercase SHA-256")
    return value


def _textured_export_sources():
    return _appearance_export_sources()


def _textured_export_response(exported, *, kind: str) -> dict:
    glb_path = Path(exported.glb_path)
    provenance_path = Path(exported.provenance_path)
    if glb_path.suffix != ".glb" or provenance_path.name != glb_path.name + ".provenance.json":
        raise HTTPException(status_code=500, detail="textured export returned unsafe artifact names")
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "glb_name": glb_path.name,
        "provenance_name": provenance_path.name,
        "glb_url": f"/api/reconstruction/artifacts/{glb_path.name}",
        "provenance_url": f"/api/reconstruction/artifacts/{provenance_path.name}",
        "glb_sha256": hashlib.sha256(glb_path.read_bytes()).hexdigest(),
        "glb_path": str(glb_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": exported.provenance_sha256,
        "assembly_id": exported.assembly_id,
        "building_appearance_revision_id": exported.building_appearance_revision_id,
        "ply_name": None,
    }


@app.post("/api/reconstruction/export-textured-tile")
def export_textured_tile(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="textured tile export payload must be an object")
    tile = _textured_export_sha(payload.get("tile_revision_id"), "tile_revision_id")
    appearance = _textured_export_sha(payload.get("appearance_revision_id"), "appearance_revision_id")
    lod = _textured_export_sha(payload.get("lod_id"), "lod_id")
    if "output_path" in payload:
        raise HTTPException(status_code=422, detail="output_path is server-owned")
    sources = _textured_export_sources()
    _textured_export_root.mkdir(parents=True, exist_ok=True)
    output = _textured_export_root / f"tile-{tile}-{appearance}-{lod}.glb"
    try:
        exported = export_local_tile_glb(
            sources=sources,
            tile_revision_id=tile,
            appearance_revision_id=appearance,
            lod_id=lod,
            rgbd_augmented_revision_id=payload.get("rgbd_augmented_revision_id"),
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _textured_export_response(exported, kind="TEXTURED_TILE_GLB")


@app.post("/api/reconstruction/export-textured-building")
def export_textured_building(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="textured Building export payload must be an object")
    assembly = _textured_export_sha(payload.get("assembly_id"), "assembly_id")
    building = _textured_export_sha(
        payload.get("building_appearance_revision_id"),
        "building_appearance_revision_id",
    )
    lod_by_member = payload.get("lod_by_member")
    if not isinstance(lod_by_member, dict) or not lod_by_member:
        raise HTTPException(status_code=422, detail="lod_by_member must be a non-empty exact mapping")
    normalized_lods = {
        _textured_export_sha(member, "lod_by_member member"): _textured_export_sha(lod, "lod_by_member LOD")
        for member, lod in lod_by_member.items()
    }
    if "output_path" in payload:
        raise HTTPException(status_code=422, detail="output_path is server-owned")
    sources = _textured_export_sources()
    _textured_export_root.mkdir(parents=True, exist_ok=True)
    output = _textured_export_root / f"building-{assembly}-{building}.glb"
    try:
        exported = export_building_glb(
            sources=sources,
            assembly_id=assembly,
            building_appearance_revision_id=building,
            lod_by_member=normalized_lods,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _textured_export_response(exported, kind="TEXTURED_BUILDING_GLB")


def _formal_export_gate() -> None:
    blockers: list[str] = []
    pipeline = _live_surface_pipeline
    if pipeline is None:
        blockers.append("BUILDING_APPEARANCE_UNAVAILABLE")
    else:
        building_id = pipeline.snapshot().get("building_id")
        if not isinstance(building_id, str) or not building_id:
            blockers.append("ACTIVE_BUILDING_UNAVAILABLE")
        else:
            try:
                published = pipeline.current_building_appearance(building_id)
            except (FileNotFoundError, OSError, ValueError):
                published = None
            if published is None:
                blockers.append("BUILDING_APPEARANCE_UNAVAILABLE")
    if not blockers:
        try:
            evidence = json.loads(field_acceptance_evidence().body)
        except (TypeError, ValueError, json.JSONDecodeError):
            blockers.append("FIELD_ACCEPTANCE_EVIDENCE_INVALID")
        else:
            blockers.extend(str(value) for value in evidence.get("blockers") or [])
            if evidence.get("available") is not True:
                blockers.append("FORMAL_APPEARANCE_GATE_NOT_PASSED")
    if blockers:
        raise HTTPException(
            status_code=423,
            detail={"state": "FORMAL_EXPORT_BLOCKED", "blockers": list(dict.fromkeys(blockers))},
        )


@app.post("/api/exports/glb")
def export_glb(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="GLB export payload must be an object")
    _formal_export_gate()
    kind = str(payload.get("kind") or "building").lower()
    if kind == "tile":
        return export_textured_tile(payload)
    if kind == "building":
        return export_textured_building(payload)
    raise HTTPException(status_code=422, detail="GLB export kind must be tile or building")


def _export_artifact_path(export_id: str) -> Path:
    if not isinstance(export_id, str) or Path(export_id).name != export_id:
        raise HTTPException(status_code=404, detail="export is unavailable")
    if not re.fullmatch(
        r"(?:tile-[0-9a-f]{64}-[0-9a-f]{64}-[0-9a-f]{64}|building-[0-9a-f]{64}-[0-9a-f]{64})\.glb(?:\.provenance\.json)?",
        export_id,
    ):
        raise HTTPException(status_code=404, detail="export is unavailable")
    path = (_textured_export_root / export_id).resolve(strict=False)
    if path.parent != _textured_export_root.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="export is unavailable")
    return path


@app.get("/api/exports/{export_id}")
def get_export(export_id: str):
    path = _export_artifact_path(export_id)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "export_id": export_id,
        "artifact_name": path.name,
        "artifact_url": f"/api/reconstruction/artifacts/{path.name}",
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "state": "AVAILABLE",
        "read_only": True,
    }


@app.get("/api/exports/{export_id}/audit.json")
def get_export_audit(export_id: str):
    path = _export_artifact_path(export_id)
    if not path.name.endswith(".glb"):
        raise HTTPException(status_code=404, detail="export audit is unavailable")
    audit = path.with_name(path.name + ".provenance.json")
    if not audit.is_file():
        raise HTTPException(status_code=404, detail="export audit is unavailable")
    return FileResponse(audit, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.post("/api/calibration/slot/{slot}")
def calibration_slot(slot: str):
    try:
        calibration_capture.select_slot(slot)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return calibration_capture.snapshot()


@app.post("/api/calibration/detect-latest")
def calibration_detect_latest():
    session = robot_session.snapshot()
    if session.get("state") != "RECORDING" or not session.get("capture_active"):
        raise HTTPException(status_code=409, detail="RAW_CALIBRATION_CAPTURE_NOT_RECORDING")
    frame = camera_assist.frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="camera assist frame is unavailable")
    try:
        return calibration_capture.ingest_camera_payload(frame)
    except TargetSpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/live/keyframe")
async def live_keyframe_websocket(websocket: WebSocket):
    token = os.environ.get("GO2_MAPPING_LIVE_TOKEN")
    if token and websocket.headers.get("X-Go2-Live-Token") != token:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            try:
                body = await websocket.receive_text()
                payload = json.loads(body)
                acknowledgement = await _dispatch_live_message_async(
                    payload, message_bytes=len(body.encode("utf-8"))
                )
            except (KeyError, TypeError, ValueError, KeyframeError, LiveProtocolError, json.JSONDecodeError) as exc:
                acknowledgement = {
                    "message_type": "live_protocol_error",
                    "protocol_version": 3,
                    "accepted": False,
                    "error_code": getattr(exc, "error_code", "INVALID_MESSAGE"),
                    "error": str(exc),
                }
            await websocket.send_json(acknowledgement)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/live/texture")
async def live_texture_websocket(websocket: WebSocket):
    token = os.environ.get("GO2_MAPPING_LIVE_TOKEN")
    if token and websocket.headers.get("X-Go2-Live-Token") != token:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            try:
                body = await websocket.receive_text()
                payload = json.loads(body)
                acknowledgement = await asyncio.to_thread(
                    _dispatch_texture_message,
                    payload,
                    message_bytes=len(body.encode("utf-8")),
                )
            except (KeyError, TypeError, ValueError, LiveProtocolError, json.JSONDecodeError) as exc:
                acknowledgement = {
                    "message_type": "texture_protocol_error",
                    "protocol_version": 3,
                    "accepted": False,
                    "error_code": getattr(exc, "error_code", "INVALID_MESSAGE"),
                    "error": str(exc),
                }
            await websocket.send_json(acknowledgement)
    except WebSocketDisconnect:
        return

@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    query_params = getattr(ws, "query_params", {})
    supports_telemetry_patch = query_params.get("protocol") == "2"
    supports_point_free_map = query_params.get("protocol") == "3"

    async def send_payload(payload: str) -> None:
        send_text = getattr(ws, "send_text", None)
        if send_text is not None:
            await send_text(payload)
            return
        await ws.send_json(json.loads(payload))

    if supports_point_free_map:
        last_session_id = None
        last_trajectory: tuple[tuple[float, float, float], ...] = ()
        last_scene_revision = None
        try:
            while True:
                live_state = mapper.telemetry_snapshot()
                session = robot_session.snapshot()
                building = _building_scan_telemetry_snapshot()
                session_id = session.get("session_id")
                robot_event = {
                    "message_type": "robot_state",
                    "protocol_version": 3,
                    "timestamp_ns": time.time_ns(),
                    "session": {
                        "session_id": session_id,
                        "state": session.get("state"),
                        "capture_active": bool(session.get("capture_active")),
                    },
                    "building_state": building.get("state"),
                    "building_quality": building.get("quality"),
                    "active_floor_id": building.get("active_floor_id"),
                    "active_submap_id": building.get("active_submap_id"),
                    "robot": {
                        "position": list(live_state.get("position") or (0.0, 0.0, 0.0)),
                        "heading_xy": list(live_state.get("heading_xy") or (1.0, 0.0)),
                    },
                }
                await send_payload(json.dumps(robot_event, separators=(",", ":"), allow_nan=False))

                trajectory = tuple(
                    tuple(float(value) for value in point)
                    for point in live_state.get("trajectory", ())
                    if isinstance(point, (list, tuple)) and len(point) == 3
                )
                same_prefix = len(trajectory) >= len(last_trajectory) and trajectory[:len(last_trajectory)] == last_trajectory
                if session_id != last_session_id or not same_prefix:
                    trajectory_event = {
                        "message_type": "trajectory_reset",
                        "protocol_version": 3,
                        "session_id": session_id,
                        "points": [list(point) for point in trajectory],
                    }
                    await send_payload(json.dumps(trajectory_event, separators=(",", ":"), allow_nan=False))
                elif len(trajectory) > len(last_trajectory):
                    trajectory_event = {
                        "message_type": "trajectory_delta",
                        "protocol_version": 3,
                        "session_id": session_id,
                        "start_index": len(last_trajectory),
                        "points": [list(point) for point in trajectory[len(last_trajectory):]],
                    }
                    await send_payload(json.dumps(trajectory_event, separators=(",", ":"), allow_nan=False))
                last_session_id = session_id
                last_trajectory = trajectory

                building_id = building.get("building_id")
                scene = None
                if isinstance(building_id, str) and building_id:
                    try:
                        scene = _render_manifest_store.current(building_id)
                    except (KeyError, OSError, ValueError):
                        scene = None
                if scene is not None and scene.revision_id != last_scene_revision:
                    revision_event = {
                        "message_type": "tile_revision",
                        "protocol_version": 3,
                        "building_id": building_id,
                        "scene_revision_id": scene.revision_id,
                        "manifest_url": f"/api/building/render/manifest?building_id={quote(building_id, safe='')}",
                    }
                    await send_payload(json.dumps(revision_event, separators=(",", ":"), allow_nan=False))
                    last_scene_revision = scene.revision_id
                await asyncio.sleep(1.0 / _dashboard_telemetry_hz)
        except WebSocketDisconnect:
            return

    last_full_snapshot_generation = 0
    try:
        while True:
            full_snapshot, generation, _ = _read_snapshot_cache()
            if full_snapshot is not None and generation != last_full_snapshot_generation:
                await send_payload(full_snapshot)
                last_full_snapshot_generation = generation
            if supports_telemetry_patch:
                payload = json.dumps(dashboard_telemetry_patch(), separators=(",", ":"), allow_nan=False)
                await send_payload(payload)
            await asyncio.sleep(1.0 / _dashboard_telemetry_hz)
    except WebSocketDisconnect:
        return
