"""Isolated projective RGB observations that can never authorize final appearance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import uuid
from typing import Iterable, Optional

import numpy as np

from mapping.pc.go2_mapping_live.live_protocol import TextureKeyframeMessage
from mapping.pc.go2_mapping_live.live_tiles import decode_surface_tile


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{name} must be a printable identifier")
    return value.strip()


def _rigid(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError(f"{name} must be a finite homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{name} must be rigid")
    return matrix


@dataclass(frozen=True)
class PublishedProvisionalObservation:
    revision_id: str
    manifest: dict
    root_path: Path
    color_path: Path


@dataclass(frozen=True)
class PublishedProvisionalAssignment:
    revision_id: str
    manifest: dict
    root_path: Path
    visibility_mask_path: Path


class ProvisionalTextureStore:
    """Durably retains immediate projective color outside the final Building store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.observations_root = self.root / "observations"
        self.assignments_root = self.root / "assignments"
        self.assignment_current_path = self.root / "CURRENT-ASSIGNMENTS.json"
        self.observations_root.mkdir(parents=True, exist_ok=True)
        self.assignments_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def ingest(
        self,
        frame: TextureKeyframeMessage,
        *,
        camera_from_lidar: Optional[np.ndarray],
        calibration_authority: bool,
        submap_id: str,
    ) -> PublishedProvisionalObservation:
        if not isinstance(frame, TextureKeyframeMessage):
            raise TypeError("frame must be a TextureKeyframeMessage")
        submap = _identifier(submap_id, "submap_id")
        camera_from_map = None
        camera_from_lidar_values = None
        if camera_from_lidar is not None:
            transform = _rigid(camera_from_lidar, "camera_from_lidar")
            map_from_lidar = _rigid(frame.pose_map_from_lidar, "pose_map_from_lidar")
            camera_from_map = transform @ np.linalg.inv(map_from_lidar)
            camera_from_lidar_values = transform.reshape(-1).tolist()
        identity = frame.identity
        core = {
            "schema_version": 2,
            "texture_state": "PROVISIONAL",
            "final_store_eligible": False,
            "building_id": identity.building_id,
            "submap_id": submap,
            "session_id": identity.session_id,
            "map_id": identity.map_id,
            "producer_instance_id": identity.producer_instance_id,
            "producer_epoch": identity.producer_epoch,
            "texture_sequence": identity.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "calibration_id": frame.calibration_id,
            "calibration_authority": bool(calibration_authority),
            "source_config_sha256": frame.source_config_sha256,
            "content_sha256": frame.content_sha256,
            "source": frame.source,
            "intrinsics": frame.intrinsics,
            "pose_map_from_lidar_row_major": frame.pose_map_from_lidar.reshape(-1).tolist(),
            "camera_from_lidar_row_major": camera_from_lidar_values,
            "camera_from_map_row_major": None if camera_from_map is None else camera_from_map.reshape(-1).tolist(),
            "projectable": camera_from_map is not None,
            "quality": frame.quality,
        }
        revision_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        manifest = {
            **core,
            "provisional_texture_revision_id": revision_id,
            "color_url": f"/api/building/render/provisional/observations/{revision_id}/color.jpg",
        }
        destination = self.observations_root / revision_id
        with self._lock:
            if destination.exists():
                published = self.load(revision_id)
                if published.manifest != manifest or published.color_path.read_bytes() != frame.color_jpeg:
                    raise ValueError("provisional texture revision conflicts with immutable content")
                return published
            temporary = self.observations_root / f".partial-{uuid.uuid4().hex}"
            temporary.mkdir()
            try:
                self._write_fsync(temporary / "color.jpg", frame.color_jpeg)
                self._write_fsync(temporary / "manifest.json", _canonical_bytes(manifest))
                _fsync_directory(temporary)
                os.replace(temporary, destination)
                _fsync_directory(self.observations_root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return PublishedProvisionalObservation(revision_id, manifest, destination, destination / "color.jpg")

    def load(self, revision_id: str) -> PublishedProvisionalObservation:
        revision = _sha256(revision_id, "provisional texture revision")
        root = self.observations_root / revision
        manifest_path = root / "manifest.json"
        color_path = root / "color.jpg"
        try:
            body = manifest_path.read_bytes()
            manifest = json.loads(body)
            color = color_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("provisional texture revision is unreadable") from exc
        core = dict(manifest)
        if core.pop("provisional_texture_revision_id", None) != revision:
            raise ValueError("provisional texture revision identity is invalid")
        core.pop("color_url", None)
        if body != _canonical_bytes(manifest) or hashlib.sha256(_canonical_bytes(core)).hexdigest() != revision:
            raise ValueError("provisional texture manifest verification failed")
        if hashlib.sha256(color).hexdigest() != manifest.get("content_sha256"):
            raise ValueError("provisional texture content verification failed")
        if manifest.get("texture_state") != "PROVISIONAL" or manifest.get("final_store_eligible") is not False:
            raise ValueError("provisional texture authority is invalid")
        return PublishedProvisionalObservation(revision, manifest, root, color_path)

    def observation_revisions(
        self,
        *,
        building_id: Optional[str] = None,
        session_id: Optional[str] = None,
        submap_id: Optional[str] = None,
        projectable_only: bool = False,
    ) -> list[str]:
        submap = None if submap_id is None else _identifier(submap_id, "submap_id")
        revisions = []
        for path in sorted(self.observations_root.iterdir()):
            if not path.is_dir() or len(path.name) != 64:
                continue
            try:
                published = self.load(path.name)
            except ValueError:
                continue
            manifest = published.manifest
            if building_id is not None and manifest.get("building_id") != building_id:
                continue
            if session_id is not None and manifest.get("session_id") != session_id:
                continue
            if submap is not None and manifest.get("submap_id") != submap:
                continue
            if projectable_only and manifest.get("projectable") is not True:
                continue
            revisions.append(published.revision_id)
        return revisions

    def project_and_publish(
        self,
        *,
        tile_revision_id: str,
        tile_body: bytes,
        observations: Iterable[str],
        submap_id: str,
    ) -> PublishedProvisionalAssignment:
        tile_revision = _sha256(tile_revision_id, "tile_revision_id")
        submap = _identifier(submap_id, "submap_id")
        tile = decode_surface_tile(tile_body)
        candidates = [self.load(revision) for revision in observations]
        if any(candidate.manifest.get("submap_id") != submap for candidate in candidates):
            raise ValueError("provisional texture observation submap does not match the surface tile")
        projectable = [candidate for candidate in candidates if candidate.manifest.get("projectable") is True]
        if not projectable:
            raise ValueError("no projectable provisional texture observation is available")
        best = None
        for observation in projectable:
            visible, score = self._visibility(tile.positions, tile.normals, tile.triangles, observation.manifest)
            candidate = (int(np.count_nonzero(visible)), score, observation.revision_id, visible, observation)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        assert best is not None
        visible_count, score, _, mask, observation = best
        if visible_count == 0:
            raise ValueError("no projectable provisional triangle is visible")
        mask_bytes = np.ascontiguousarray(mask, dtype="u1").tobytes()
        mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
        core = {
            "schema_version": 2,
            "texture_state": "PROVISIONAL",
            "final_store_eligible": False,
            "tile_revision_id": tile_revision,
            "submap_id": submap,
            "observation_revision_id": observation.revision_id,
            "visible_triangle_count": visible_count,
            "triangle_count": len(tile.triangles),
            "visibility_mask_sha256": mask_sha256,
            "projection_score": round(float(score), 6),
            "color_url": observation.manifest["color_url"],
            "color_content_sha256": observation.manifest["content_sha256"],
            "camera_from_map_row_major": observation.manifest["camera_from_map_row_major"],
            "intrinsics": observation.manifest["intrinsics"],
        }
        revision_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        manifest = {
            **core,
            "provisional_assignment_revision_id": revision_id,
            "visibility_mask_url": f"/api/building/render/provisional/assignments/{revision_id}/visibility.bin",
        }
        destination = self.assignments_root / revision_id
        with self._lock:
            if destination.exists():
                published = self.load_assignment(revision_id)
                if published.manifest != manifest or published.visibility_mask_path.read_bytes() != mask_bytes:
                    raise ValueError("provisional assignment conflicts with immutable content")
            else:
                temporary = self.assignments_root / f".partial-{uuid.uuid4().hex}"
                temporary.mkdir()
                try:
                    self._write_fsync(temporary / "visibility.bin", mask_bytes)
                    self._write_fsync(temporary / "manifest.json", _canonical_bytes(manifest))
                    _fsync_directory(temporary)
                    os.replace(temporary, destination)
                    _fsync_directory(self.assignments_root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                published = PublishedProvisionalAssignment(
                    revision_id, manifest, destination, destination / "visibility.bin"
                )
            current = self._load_assignment_index()
            current["tiles"][tile_revision] = revision_id
            _atomic_write(self.assignment_current_path, _canonical_bytes(current))
            return published

    def load_assignment(self, revision_id: str) -> PublishedProvisionalAssignment:
        revision = _sha256(revision_id, "provisional assignment revision")
        root = self.assignments_root / revision
        try:
            body = (root / "manifest.json").read_bytes()
            manifest = json.loads(body)
            mask = (root / "visibility.bin").read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("provisional assignment is unreadable") from exc
        core = dict(manifest)
        if core.pop("provisional_assignment_revision_id", None) != revision:
            raise ValueError("provisional assignment identity is invalid")
        core.pop("visibility_mask_url", None)
        if body != _canonical_bytes(manifest) or hashlib.sha256(_canonical_bytes(core)).hexdigest() != revision:
            raise ValueError("provisional assignment manifest verification failed")
        if hashlib.sha256(mask).hexdigest() != manifest.get("visibility_mask_sha256"):
            raise ValueError("provisional assignment mask verification failed")
        if len(mask) != manifest.get("triangle_count"):
            raise ValueError("provisional assignment mask length is invalid")
        return PublishedProvisionalAssignment(revision, manifest, root, root / "visibility.bin")

    def current_assignments(self) -> dict[str, dict]:
        with self._lock:
            index = self._load_assignment_index()
            return {
                tile: self.load_assignment(revision).manifest
                for tile, revision in index["tiles"].items()
            }

    def _load_assignment_index(self) -> dict:
        try:
            payload = json.loads(self.assignment_current_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1 or not isinstance(payload.get("tiles"), dict):
                raise ValueError("provisional assignment index is invalid")
            return payload
        except FileNotFoundError:
            return {"schema_version": 1, "tiles": {}}

    @staticmethod
    def _visibility(positions, normals, triangles, manifest: dict) -> tuple[np.ndarray, float]:
        camera_from_map = _rigid(
            np.asarray(manifest["camera_from_map_row_major"], dtype=np.float64).reshape(4, 4),
            "camera_from_map",
        )
        map_from_camera = np.linalg.inv(camera_from_map)
        camera_position = map_from_camera[:3, 3]
        centres = positions[triangles].mean(axis=1).astype(np.float64)
        centre_normals = normals[triangles].mean(axis=1).astype(np.float64)
        normal_lengths = np.linalg.norm(centre_normals, axis=1)
        centre_normals[normal_lengths > 0] /= normal_lengths[normal_lengths > 0, None]
        homogeneous = np.column_stack((centres, np.ones(len(centres))))
        camera_points = (camera_from_map @ homogeneous.T).T[:, :3]
        depth = camera_points[:, 2]
        intrinsics = manifest["intrinsics"]
        width, height = int(intrinsics["width"]), int(intrinsics["height"])
        safe_depth = np.where(depth > 0, depth, 1.0)
        u = float(intrinsics["fx"]) * camera_points[:, 0] / safe_depth + float(intrinsics["cx"])
        v = float(intrinsics["fy"]) * camera_points[:, 1] / safe_depth + float(intrinsics["cy"])
        to_camera = camera_position[None, :] - centres
        distance = np.linalg.norm(to_camera, axis=1)
        directions = np.zeros_like(to_camera)
        directions[distance > 0] = to_camera[distance > 0] / distance[distance > 0, None]
        incidence = np.einsum("ij,ij->i", centre_normals, directions)
        projected = (
            (depth > 0.05)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
            & (incidence >= 0.15)
        )
        pixels = np.column_stack((np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)))
        zbuffer: dict[tuple[int, int], float] = {}
        for index in np.flatnonzero(projected):
            pixel = tuple(int(value) for value in pixels[index])
            zbuffer[pixel] = min(zbuffer.get(pixel, math.inf), float(depth[index]))
        visible = np.zeros(len(triangles), dtype=bool)
        for index in np.flatnonzero(projected):
            pixel = tuple(int(value) for value in pixels[index])
            visible[index] = depth[index] <= zbuffer[pixel] + 0.03
        quality = manifest["quality"]
        score = (
            np.count_nonzero(visible) * 1_000_000.0
            + float(quality["sharpness_laplacian"])
            - abs(float(quality["mean_luma"]) - 127.5)
            - float(quality["clipped_fraction"]) * 1000.0
            + float(np.sum(np.maximum(incidence[visible], 0.0)))
        )
        return visible, score

    @staticmethod
    def _write_fsync(path: Path, body: bytes) -> None:
        with path.open("wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
