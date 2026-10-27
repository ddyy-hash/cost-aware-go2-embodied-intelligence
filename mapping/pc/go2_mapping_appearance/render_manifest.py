"""Atomic point-free scene manifest composing geometry and two texture authorities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import uuid
from typing import Any, Mapping, Optional

from .provisional_store import ProvisionalTextureStore, _atomic_write, _canonical_bytes, _fsync_directory


@dataclass(frozen=True)
class PublishedRenderManifest:
    revision_id: str
    manifest: dict[str, Any]
    root_path: Path
    manifest_path: Path


class AtomicRenderManifestStore:
    def __init__(self, root: str | Path, *, provisional_store: ProvisionalTextureStore) -> None:
        self.root = Path(root)
        self.revisions_root = self.root / "revisions"
        self.current_root = self.root / "current"
        self.provisional_store = provisional_store
        self.revisions_root.mkdir(parents=True, exist_ok=True)
        self.current_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _building_key(building_id: str) -> str:
        if not isinstance(building_id, str) or not building_id.strip() or any(ord(value) < 0x20 for value in building_id):
            raise ValueError("building_id must be a printable identifier")
        return hashlib.sha256(building_id.encode("utf-8")).hexdigest()

    def publish(
        self,
        *,
        building_id: str,
        surface_manifest: Mapping[str, Any],
        provisional_assignments: Mapping[str, Mapping[str, Any]],
        validated_appearance_manifest: Optional[Mapping[str, Any]],
        trajectories: Mapping[str, Any],
    ) -> PublishedRenderManifest:
        building_key = self._building_key(building_id)
        if surface_manifest.get("scope") != "building" or not isinstance(surface_manifest.get("tiles"), list):
            raise ValueError("render manifest requires a Building surface manifest")
        validated_members = {}
        validated_revision = None
        if validated_appearance_manifest is not None:
            validated_revision = validated_appearance_manifest.get("building_appearance_revision_id")
            validated_members = {
                member.get("tile_revision_id"): dict(member)
                for member in validated_appearance_manifest.get("members", ())
                if isinstance(member, Mapping) and isinstance(member.get("tile_revision_id"), str)
            }
        tiles = []
        for tile in surface_manifest["tiles"]:
            tile_revision = str(tile.get("revision_id", ""))
            if len(tile_revision) != 64:
                raise ValueError("surface tile revision identity is invalid")
            tile_submap = tile.get("submap_id")
            if not isinstance(tile_submap, str) or not tile_submap:
                raise ValueError("surface tile submap identity is invalid")
            provisional = provisional_assignments.get(tile_revision)
            if provisional is not None:
                provisional = dict(provisional)
                if provisional.get("submap_id") != tile_submap:
                    raise ValueError("provisional texture assignment submap does not match the surface tile")
                if (
                    provisional.get("tile_revision_id") != tile_revision
                    or provisional.get("texture_state") != "PROVISIONAL"
                    or provisional.get("final_store_eligible") is not False
                ):
                    raise ValueError("provisional texture assignment authority is invalid")
            validated = validated_members.get(tile_revision)
            geometry = {
                "revision_id": tile_revision,
                "geometry_revision_id": tile.get("geometry_revision_id"),
                "content_sha256": tile.get("content_sha256"),
                "binary_url": tile.get("binary_url"),
                "binary_schema_version": tile.get("binary_schema_version", 1),
                "provenance_encoding": tile.get("provenance_encoding"),
                "uncertainty_encoding": tile.get("uncertainty_encoding"),
                "lods": [{
                    "lod_id": "surface-native",
                    "binary_url": tile.get("binary_url"),
                    "content_sha256": tile.get("content_sha256"),
                }],
            }
            tiles.append({
                "building_id": building_id,
                "tile_revision_id": tile_revision,
                "floor_id": tile.get("floor_id"),
                "submap_id": tile.get("submap_id"),
                "tile_id": tile.get("tile_id"),
                "placement_revision_id": tile.get("placement_revision_id"),
                "attachment_state": tile.get("attachment_state"),
                "transform_building_from_submap": tile.get("transform_building_from_submap"),
                "geometry": geometry,
                "provisional_texture": provisional,
                "validated_texture": validated,
                "texture_composition": (
                    "VALIDATED_MASK_OVER_PROVISIONAL"
                    if validated is not None and provisional is not None
                    else "VALIDATED_ONLY" if validated is not None
                    else "PROVISIONAL_ONLY" if provisional is not None
                    else "UNTEXTURED_OBSERVED_SURFACE"
                ),
            })
        tiles.sort(key=lambda value: (str(value["floor_id"]), str(value["submap_id"]), str(value["tile_id"])))
        core = {
            "schema_version": 1,
            "building_id": building_id,
            "assembly_id": surface_manifest.get("assembly_id"),
            "surface_manifest_schema_version": surface_manifest.get("schema_version"),
            "building_appearance_revision_id": validated_revision,
            "tiles": tiles,
            "trajectories": dict(trajectories),
            "render_contract": {
                "main_primitives": ["TEXTURED_TRIANGLE_SURFACE", "GO2_POSE", "TRAJECTORY"],
                "backend_only": ["LIDAR_POINTS", "RGBD_DEPTH", "DA3_DEPTH", "VOXELS"],
                "provisional_final_store_eligible": False,
                "trusted_mask_overrides_provisional": True,
            },
        }
        revision_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        manifest = {**core, "scene_revision_id": revision_id}
        destination = self.revisions_root / revision_id
        pointer = {
            "schema_version": 1,
            "building_id": building_id,
            "scene_revision_id": revision_id,
            "manifest_sha256": hashlib.sha256(_canonical_bytes(manifest)).hexdigest(),
        }
        with self._lock:
            if destination.exists():
                published = self._load_revision(revision_id)
                if published.manifest != manifest:
                    raise ValueError("render manifest revision conflicts with immutable content")
            else:
                temporary = self.revisions_root / f".partial-{uuid.uuid4().hex}"
                temporary.mkdir()
                try:
                    path = temporary / "manifest.json"
                    with path.open("wb") as output:
                        output.write(_canonical_bytes(manifest))
                        output.flush()
                        os.fsync(output.fileno())
                    _fsync_directory(temporary)
                    os.replace(temporary, destination)
                    _fsync_directory(self.revisions_root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                published = PublishedRenderManifest(
                    revision_id, manifest, destination, destination / "manifest.json"
                )
            _atomic_write(self.current_root / f"{building_key}.json", _canonical_bytes(pointer))
            return published

    def current(self, building_id: str) -> PublishedRenderManifest:
        building_key = self._building_key(building_id)
        try:
            body = (self.current_root / f"{building_key}.json").read_bytes()
            pointer = json.loads(body)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("render manifest CURRENT is unavailable") from exc
        if body != _canonical_bytes(pointer) or pointer.get("building_id") != building_id:
            raise ValueError("render manifest CURRENT is invalid")
        published = self._load_revision(str(pointer.get("scene_revision_id", "")))
        if hashlib.sha256(published.manifest_path.read_bytes()).hexdigest() != pointer.get("manifest_sha256"):
            raise ValueError("render manifest CURRENT hash is invalid")
        return published

    def _load_revision(self, revision_id: str) -> PublishedRenderManifest:
        if len(revision_id) != 64 or any(value not in "0123456789abcdef" for value in revision_id):
            raise ValueError("scene revision identity is invalid")
        root = self.revisions_root / revision_id
        path = root / "manifest.json"
        try:
            body = path.read_bytes()
            manifest = json.loads(body)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("render manifest revision is unreadable") from exc
        core = dict(manifest)
        if core.pop("scene_revision_id", None) != revision_id:
            raise ValueError("render manifest revision identity is invalid")
        if body != _canonical_bytes(manifest) or hashlib.sha256(_canonical_bytes(core)).hexdigest() != revision_id:
            raise ValueError("render manifest revision verification failed")
        return PublishedRenderManifest(revision_id, manifest, root, path)
