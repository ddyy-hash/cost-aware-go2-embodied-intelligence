"""Exact-revision textured GLB export without live or dashboard dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol
import uuid

import numpy as np
from PIL import Image as PilImage
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    TRIANGLES,
    UNSIGNED_INT,
    Accessor,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Image,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Sampler,
    Scene,
    Texture,
    TextureInfo,
)

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256

from .atlas_codec import (
    RenderIndicesV1,
    RenderUvsV1,
    RenderVerticesV1,
    parse_render_indices_v1,
    parse_render_uvs_v1,
    parse_render_vertices_v1,
)
from .tile_codec import decode_g2tm_v1


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_REP103_TO_GLTF = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_REQUIRED_RENDER_ARTIFACTS = {
    "render-vertices.bin",
    "render-indices.bin",
    "render-uvs.bin",
}


class TileGeometryResolver(Protocol):
    def __call__(self, tile_revision_id: str) -> tuple[Mapping[str, Any], bytes]: ...


class TileAppearanceResolver(Protocol):
    def __call__(self, tile_revision_id: str, appearance_revision_id: str) -> Any: ...


class BuildingAppearanceResolver(Protocol):
    def __call__(self, building_appearance_revision_id: str) -> Any: ...


class BuildingAssemblyResolver(Protocol):
    def __call__(self, assembly_id: str) -> Any: ...


class RgbdAugmentationResolver(Protocol):
    def __call__(self, source_generation_id: str, rgbd_augmented_revision_id: str) -> Any: ...


@dataclass(frozen=True)
class ExportedGlb:
    glb_path: Path
    provenance_path: Path
    provenance_sha256: str
    assembly_id: str | None
    building_appearance_revision_id: str | None


@dataclass(frozen=True)
class AppearanceExportSources:
    tile_geometry: TileGeometryResolver
    tile_appearance: TileAppearanceResolver
    building_appearance: BuildingAppearanceResolver
    assembly: BuildingAssemblyResolver
    augmentation: RgbdAugmentationResolver


@dataclass(frozen=True)
class _VerifiedTile:
    tile_revision_id: str
    appearance_revision_id: str
    lod_id: str
    source_generation_id: str
    metadata: Mapping[str, Any]
    appearance: Any
    appearance_manifest_bytes: bytes
    lod_core: Mapping[str, Any]
    artifacts: Mapping[str, bytes]
    vertices: RenderVerticesV1
    indices: RenderIndicesV1
    uvs: RenderUvsV1
    primitives: tuple[Mapping[str, Any], ...]
    augmentation: Any | None
    augmentation_manifest_bytes: bytes | None


class _BinaryBuilder:
    def __init__(self) -> None:
        self.body = bytearray()
        self.views: list[BufferView] = []
        self.accessors: list[Accessor] = []

    def view(self, body: bytes, *, target: int | None = None) -> int:
        self._align()
        index = len(self.views)
        self.views.append(
            BufferView(
                buffer=0,
                byteOffset=len(self.body),
                byteLength=len(body),
                target=target,
            )
        )
        self.body.extend(body)
        return index

    def accessor(
        self,
        values: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int,
        include_bounds: bool = False,
    ) -> int:
        array = np.ascontiguousarray(values)
        view = self.view(array.tobytes(order="C"), target=target)
        count = int(len(array))
        minimum = None
        maximum = None
        if include_bounds and count:
            minimum = [float(value) for value in np.min(array, axis=0)]
            maximum = [float(value) for value in np.max(array, axis=0)]
        index = len(self.accessors)
        self.accessors.append(
            Accessor(
                bufferView=view,
                byteOffset=0,
                componentType=component_type,
                count=count,
                type=accessor_type,
                min=minimum or [],
                max=maximum or [],
            )
        )
        return index

    def _align(self) -> None:
        self.body.extend(b"\0" * ((-len(self.body)) % 4))


def export_local_tile_glb(
    *,
    sources: AppearanceExportSources,
    tile_revision_id: str,
    appearance_revision_id: str,
    lod_id: str,
    rgbd_augmented_revision_id: str | None,
    output_path: Path,
) -> ExportedGlb:
    """Export one exact immutable tile appearance in source-submap coordinates."""

    if not isinstance(sources, AppearanceExportSources):
        raise TypeError("sources must be AppearanceExportSources")
    tile = _sha256(tile_revision_id, "tile_revision_id")
    appearance = _sha256(appearance_revision_id, "appearance_revision_id")
    lod = _sha256(lod_id, "lod_id")
    requested_augmentation = (
        None
        if rgbd_augmented_revision_id is None
        else _sha256(rgbd_augmented_revision_id, "rgbd_augmented_revision_id")
    )
    verified = _verify_tile(
        sources,
        tile,
        appearance,
        lod,
        requested_augmentation=requested_augmentation,
    )
    layers = ("LIDAR_MEASURED",)
    if requested_augmentation is not None:
        layers += ("RGBD_AUGMENTED",)
    provenance = _local_provenance(verified, layers)
    identities = _tile_identity_extras(verified, requested_augmentation)
    return _write_export(
        output_path=Path(output_path),
        provenance=provenance,
        verified_tiles=(verified,),
        selected_layers={tile: layers},
        member_matrices={tile: None},
        asset_extras={
            "schema_version": 1,
            "export_unit": "LOCAL_TILE",
            "units": "metres",
            "coordinate_mapping": "REP103_XYZ_TO_GLTF_XZ_NEGATIVE_Y",
            "source_layers": list(layers),
            "tile_revision_id": tile,
            "geometry_revision_id": verified.appearance.manifest["revision_core"]["geometry_revision_id"],
            "source_generation_id": verified.source_generation_id,
            "appearance_revision_id": appearance,
            "lod_id": lod,
            "rgbd_augmented_revision_id": requested_augmentation,
            "identities": identities,
        },
        assembly_id=None,
        building_appearance_revision_id=None,
    )


def export_building_glb(
    *,
    sources: AppearanceExportSources,
    assembly_id: str,
    building_appearance_revision_id: str,
    lod_by_member: Mapping[str, str],
    output_path: Path,
) -> ExportedGlb:
    """Export all and only the members frozen by one exact Building appearance."""

    if not isinstance(sources, AppearanceExportSources):
        raise TypeError("sources must be AppearanceExportSources")
    assembly_identity = _sha256(assembly_id, "assembly_id")
    building_revision = _sha256(
        building_appearance_revision_id,
        "building_appearance_revision_id",
    )
    if not isinstance(lod_by_member, Mapping):
        raise TypeError("lod_by_member must be a mapping")

    published = sources.building_appearance(building_revision)
    if published is None:
        raise FileNotFoundError("Building appearance revision is unavailable")
    building_manifest, building_manifest_bytes = _verified_json_file(
        getattr(published, "manifest_path", None),
        getattr(published, "manifest_sha256", None),
        "Building appearance manifest",
    )
    building_core = dict(building_manifest)
    building_core.pop("building_appearance_revision_id", None)
    if (
        building_manifest != dict(getattr(published, "manifest", {}))
        or getattr(published, "building_appearance_revision_id", None) != building_revision
        or building_manifest.get("building_appearance_revision_id") != building_revision
        or canonical_sha256(building_core) != building_revision
        or building_core.get("assembly_id") != assembly_identity
        or building_core.get("geometry_authority") != "OBSERVED_SURFACE"
        or building_core.get("volume_filled") is not False
        or building_core.get("watertight") is not False
    ):
        raise ValueError("Building appearance revision identity is stale or invalid")

    assembly = sources.assembly(assembly_identity)
    if assembly is None:
        raise FileNotFoundError("Building assembly is unavailable")
    assembly_manifest, assembly_manifest_bytes = _verified_assembly(assembly, assembly_identity)
    if (
        getattr(assembly, "assembly_id", None) != assembly_identity
        or getattr(assembly, "building_id", None) != building_core.get("building_id")
        or getattr(assembly, "algorithm_version", None) != assembly_manifest.get("algorithm_version")
    ):
        raise ValueError("Building assembly identity is stale")
    assembly_members = tuple(getattr(assembly, "members", ()))
    if not assembly_members:
        raise ValueError("Building assembly members are incomplete")
    _validate_assembly_members(assembly_members, assembly_manifest)

    raw_members = building_core.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("Building appearance member set is incomplete")
    tile_ids = [value.get("tile_revision_id") if isinstance(value, Mapping) else None for value in raw_members]
    if any(not isinstance(value, str) for value in tile_ids) or len(set(tile_ids)) != len(tile_ids):
        raise ValueError("Building appearance member tile identities are invalid")
    requested_keys = set(lod_by_member)
    expected_keys = set(tile_ids)
    if requested_keys != expected_keys:
        raise ValueError("lod_by_member must contain the complete exact Building member set")

    assembly_by_source = {
        (
            member.component_id,
            member.session_id,
            member.submap_id,
            member.generation_id,
        ): member
        for member in assembly_members
    }
    if len(assembly_by_source) != len(assembly_members):
        raise ValueError("Building assembly member identities are duplicated")
    verified_tiles: list[_VerifiedTile] = []
    selected_layers: dict[str, tuple[str, ...]] = {}
    member_matrices: dict[str, np.ndarray] = {}
    asset_members: list[dict[str, Any]] = []
    for member in raw_members:
        tile = _sha256(member.get("tile_revision_id"), "Building member tile_revision_id")
        appearance = _sha256(
            member.get("tile_appearance_revision_id"),
            "Building member tile_appearance_revision_id",
        )
        selected_lod = _sha256(lod_by_member[tile], "Building member lod_id")
        authorized_lods = member.get("authorized_lod_ids")
        if not isinstance(authorized_lods, list) or selected_lod not in authorized_lods:
            raise ValueError("requested Building member LOD is unauthorized")
        requested_augmentation = member.get("rgbd_augmented_revision_id")
        if requested_augmentation is not None:
            requested_augmentation = _sha256(
                requested_augmentation,
                "Building member rgbd_augmented_revision_id",
            )
        verified = _verify_tile(
            sources,
            tile,
            appearance,
            selected_lod,
            requested_augmentation=requested_augmentation,
        )
        assembly_member = assembly_by_source.get(
            (
                member.get("component_id"),
                member.get("session_id"),
                member.get("submap_id"),
                member.get("source_generation_id"),
            )
        )
        if assembly_member is None:
            raise ValueError("Building appearance member is absent from the exact assembly")
        transform = _rigid_matrix(member.get("transform_building_from_submap"))
        if tuple(float(value) for value in transform.reshape(-1)) != tuple(
            assembly_member.transform_building_from_submap
        ):
            raise ValueError("Building appearance member placement is stale")
        metadata = verified.metadata
        if (
            member.get("building_id") != building_core.get("building_id")
            or metadata.get("building_id") != member.get("building_id")
            or metadata.get("component_id") != member.get("component_id")
            or metadata.get("submap_id") != member.get("submap_id")
            or metadata.get("source_generation_id") != member.get("source_generation_id")
            or verified.source_generation_id != member.get("source_generation_id")
            or _placement_revision(assembly_identity, metadata, assembly_member)
            != member.get("placement_revision_id")
        ):
            raise ValueError("Building appearance member source or placement is stale")
        layers = ("LIDAR_MEASURED",)
        if requested_augmentation is not None:
            layers += ("RGBD_AUGMENTED",)
        verified_tiles.append(verified)
        selected_layers[tile] = layers
        member_matrices[tile] = transform
        asset_members.append(
            {
                "building_id": member["building_id"],
                "component_id": member["component_id"],
                "submap_id": member["submap_id"],
                "placement_revision_id": member["placement_revision_id"],
                **_tile_identity_extras(verified, requested_augmentation),
                "source_layers": list(layers),
            }
        )

    provenance = _building_provenance(
        assembly_identity,
        building_revision,
        assembly_manifest_bytes,
        building_manifest_bytes,
        tuple(verified_tiles),
        tuple(asset_members),
    )
    return _write_export(
        output_path=Path(output_path),
        provenance=provenance,
        verified_tiles=tuple(verified_tiles),
        selected_layers=selected_layers,
        member_matrices=member_matrices,
        asset_extras={
            "schema_version": 1,
            "export_unit": "BUILDING",
            "units": "metres",
            "coordinate_mapping": "REP103_XYZ_TO_GLTF_XZ_NEGATIVE_Y",
            "source_layers": sorted(
                {layer for layers in selected_layers.values() for layer in layers}
            ),
            "building_id": building_core["building_id"],
            "assembly_id": assembly_identity,
            "building_appearance_revision_id": building_revision,
            "members": asset_members,
        },
        assembly_id=assembly_identity,
        building_appearance_revision_id=building_revision,
    )


def _verify_tile(
    sources: AppearanceExportSources,
    tile: str,
    appearance: str,
    lod: str,
    *,
    requested_augmentation: str | None,
) -> _VerifiedTile:
    metadata, tile_body = sources.tile_geometry(tile)
    if not isinstance(metadata, Mapping):
        raise ValueError("tile geometry metadata is invalid")
    tile_bytes = bytes(tile_body)
    if metadata.get("revision_id") != tile or hashlib.sha256(tile_bytes).hexdigest() != metadata.get("content_sha256"):
        raise ValueError("tile geometry content verification failed")
    decoded_tile = decode_g2tm_v1(tile_bytes)
    if metadata.get("vertex_count") != len(decoded_tile.positions) or metadata.get("triangle_count") != len(
        decoded_tile.triangles
    ):
        raise ValueError("tile geometry count verification failed")

    published = sources.tile_appearance(tile, appearance)
    if published is None:
        raise FileNotFoundError("tile appearance revision is unavailable")
    manifest, manifest_bytes = _verified_json_file(
        getattr(published, "manifest_path", None),
        getattr(published, "manifest_sha256", None),
        "tile appearance manifest",
    )
    if manifest != dict(getattr(published, "manifest", {})):
        raise ValueError("tile appearance resolver content is inconsistent")
    core = manifest.get("revision_core")
    if (
        not isinstance(core, Mapping)
        or manifest.get("appearance_revision_id") != appearance
        or canonical_sha256(core) != appearance
        or core.get("tile_revision_id") != tile
        or core.get("geometry_revision_id") != metadata.get("geometry_revision_id")
        or core.get("source_generation_id") != metadata.get("source_generation_id")
        or core.get("pose_revision_id") != metadata.get("pose_revision_id")
    ):
        raise ValueError("tile appearance source identity is stale or invalid")
    authorization = manifest.get("delivery_authorization")
    publication = manifest.get("publication_authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("delivery_authorized") is not True
        or authorization.get("measured_surface_coverage_ppm", -1) < 800_000
        or authorization.get("minimum_independent_view_count", -1) < 2
        or not isinstance(publication, Mapping)
        or publication.get("export_authorized") is not True
    ):
        raise ValueError("tile appearance export is unauthorized")

    matching_lods = [value for value in manifest.get("lods", ()) if isinstance(value, Mapping) and value.get("lod_id") == lod]
    if len(matching_lods) != 1:
        raise ValueError("requested LOD is unavailable")
    lod_core = matching_lods[0].get("core")
    if not isinstance(lod_core, Mapping) or canonical_sha256(lod_core) != lod:
        raise ValueError("requested LOD identity is invalid")
    lod_paths = getattr(published, "lod_paths", {})
    lod_root = Path(lod_paths.get(lod, ""))
    expected_lod_root = Path(getattr(published, "root_path", "")) / "lods" / lod
    if lod_root != expected_lod_root or not lod_root.is_dir():
        raise ValueError("requested LOD path is invalid")
    artifacts = _read_artifacts(lod_root, lod_core)
    vertices = parse_render_vertices_v1(
        artifacts["render-vertices.bin"],
        expected_tile_revision_id=tile,
        expected_appearance_revision_id=appearance,
        source_vertex_count=2**32,
    )
    indices = parse_render_indices_v1(
        artifacts["render-indices.bin"],
        expected_tile_revision_id=tile,
        expected_appearance_revision_id=appearance,
        render_vertex_count=len(vertices.positions),
        source_triangle_count=2**32,
    )
    uvs = parse_render_uvs_v1(
        artifacts["render-uvs.bin"],
        expected_tile_revision_id=tile,
        expected_appearance_revision_id=appearance,
        atlas_count=lod_core.get("atlas_count"),
    )
    if (
        lod_core.get("vertex_count") != len(vertices.positions)
        or lod_core.get("triangle_count") != len(indices.triangles)
        or len(uvs.uvs) != len(vertices.positions)
    ):
        raise ValueError("LOD render geometry count verification failed")
    primitives = _validated_primitives(lod_core, indices, uvs)

    expected_augmentation = core.get("rgbd_augmented_revision_id")
    if lod_core.get("rgbd_augmented_revision_id") != expected_augmentation:
        raise ValueError("LOD augmentation identity is inconsistent")
    augmentation = None
    augmentation_manifest_bytes = None
    if requested_augmentation is not None:
        if requested_augmentation != expected_augmentation:
            raise ValueError("requested augmentation revision does not match the tile appearance")
        augmentation = sources.augmentation(core.get("source_generation_id"), requested_augmentation)
        if augmentation is None:
            raise FileNotFoundError("requested augmentation revision is unavailable")
        augmentation_manifest, augmentation_manifest_bytes = _verified_json_file(
            getattr(augmentation, "manifest_path", None),
            getattr(augmentation, "manifest_sha256", None),
            "augmentation manifest",
        )
        layer = getattr(augmentation, "layer", None)
        if (
            augmentation_manifest != dict(getattr(augmentation, "manifest", {}))
            or getattr(augmentation, "source_generation_id", None) != core.get("source_generation_id")
            or getattr(augmentation, "rgbd_augmented_revision_id", None) != requested_augmentation
            or augmentation_manifest.get("source_generation_id") != core.get("source_generation_id")
            or augmentation_manifest.get("rgbd_augmented_revision_id") != requested_augmentation
            or augmentation_manifest.get("name") != "RGBD_AUGMENTED"
            or augmentation_manifest.get("geometry_authority") is not False
            or getattr(layer, "name", None) != "RGBD_AUGMENTED"
            or getattr(layer, "geometry_authority", None) is not False
        ):
            raise ValueError("requested augmentation revision failed verification")

    return _VerifiedTile(
        tile,
        appearance,
        lod,
        core["source_generation_id"],
        dict(metadata),
        published,
        manifest_bytes,
        dict(lod_core),
        artifacts,
        vertices,
        indices,
        uvs,
        primitives,
        augmentation,
        augmentation_manifest_bytes,
    )


def _read_artifacts(lod_root: Path, lod_core: Mapping[str, Any]) -> dict[str, bytes]:
    records = lod_core.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("LOD artifact manifest is invalid")
    names = [value.get("name") if isinstance(value, Mapping) else None for value in records]
    if any(not isinstance(value, str) or Path(value).name != value for value in names) or len(set(names)) != len(names):
        raise ValueError("LOD artifact names are invalid")
    if not _REQUIRED_RENDER_ARTIFACTS.issubset(names):
        raise ValueError("LOD render artifacts are incomplete")
    atlas_count = lod_core.get("atlas_count")
    if isinstance(atlas_count, bool) or not isinstance(atlas_count, int) or atlas_count < 1:
        raise ValueError("LOD atlas count is invalid")
    expected_atlases = {f"atlas-{index}.png" for index in range(atlas_count)}
    if {value for value in names if value.startswith("atlas-")} != expected_atlases:
        raise ValueError("LOD atlas artifacts are incomplete")
    bodies: dict[str, bytes] = {}
    for record in records:
        name = record["name"]
        path = lod_root / name
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"LOD artifact {name} is unavailable") from exc
        if len(body) != record.get("bytes") or hashlib.sha256(body).hexdigest() != record.get("sha256"):
            raise ValueError(f"LOD artifact {name} content verification failed")
        bodies[name] = body
    for name in expected_atlases:
        try:
            with PilImage.open(BytesIO(bodies[name])) as image:
                image.load()
                if image.mode != "RGBA" or image.width > 4096 or image.height > 4096:
                    raise ValueError("LOD atlas image is invalid")
        except (OSError, ValueError) as exc:
            raise ValueError(f"LOD atlas {name} cannot be decoded") from exc
    return bodies


def _validated_primitives(
    lod_core: Mapping[str, Any],
    indices: RenderIndicesV1,
    uvs: RenderUvsV1,
) -> tuple[Mapping[str, Any], ...]:
    raw = lod_core.get("primitives")
    if not isinstance(raw, list) or not raw:
        raise ValueError("LOD primitive set is invalid")
    result: list[Mapping[str, Any]] = []
    expected_offset = 0
    for expected_id, primitive in enumerate(raw):
        if not isinstance(primitive, Mapping):
            raise ValueError("LOD primitive is invalid")
        count = primitive.get("index_count")
        material = primitive.get("material")
        layer = primitive.get("source_layer")
        atlas = primitive.get("atlas_index")
        if (
            primitive.get("primitive_id") != expected_id
            or primitive.get("index_offset") != expected_offset
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or count % 3
            or material not in {"VALIDATED_UV_ATLAS", "BASE_MATERIAL"}
            or layer not in {"LIDAR_MEASURED", "RGBD_AUGMENTED"}
        ):
            raise ValueError("LOD primitive range or identity is invalid")
        first_triangle = expected_offset // 3
        triangle_count = count // 3
        source_ids = primitive.get("source_triangle_ids")
        if list(indices.source_triangle_ids[first_triangle : first_triangle + triangle_count]) != source_ids:
            raise ValueError("LOD primitive source triangle identity is invalid")
        vertex_ids = indices.triangles[first_triangle : first_triangle + triangle_count].reshape(-1)
        expected_flag = 1 if material == "VALIDATED_UV_ATLAS" else 0
        expected_atlas = atlas if expected_flag else 65535
        if (
            (expected_flag and (not isinstance(atlas, int) or not 0 <= atlas < lod_core["atlas_count"]))
            or (not expected_flag and atlas != 65535)
            or np.any(uvs.flags[vertex_ids] != expected_flag)
            or np.any(uvs.atlas_indices[vertex_ids] != expected_atlas)
        ):
            raise ValueError("LOD primitive UV material identity is invalid")
        expected_offset += count
        result.append(dict(primitive))
    if expected_offset != len(indices.triangles) * 3:
        raise ValueError("LOD primitives do not cover every render index")
    augmented = any(value["source_layer"] == "RGBD_AUGMENTED" for value in result)
    if augmented != (lod_core.get("rgbd_augmented_revision_id") is not None):
        raise ValueError("LOD augmentation primitive coverage is invalid")
    return tuple(result)


def _verified_json_file(path_value: Any, expected_sha256: Any, name: str) -> tuple[dict[str, Any], bytes]:
    if path_value is None:
        raise ValueError(f"{name} path is missing")
    try:
        body = Path(path_value).read_bytes()
        payload = json.loads(body)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is unreadable") from exc
    if not isinstance(payload, dict) or hashlib.sha256(body).hexdigest() != _sha256(expected_sha256, f"{name} SHA"):
        raise ValueError(f"{name} content verification failed")
    if body != _canonical_bytes(payload):
        raise ValueError(f"{name} is not canonical")
    return payload, body


def _local_provenance(verified: _VerifiedTile, layers: tuple[str, ...]) -> dict[str, Any]:
    manifests: dict[str, Any] = {
        "tile_appearance": {
            "sha256": hashlib.sha256(verified.appearance_manifest_bytes).hexdigest(),
            "json_utf8": verified.appearance_manifest_bytes.decode("utf-8"),
        }
    }
    if verified.augmentation_manifest_bytes is not None:
        manifests["rgbd_augmentation"] = {
            "sha256": hashlib.sha256(verified.augmentation_manifest_bytes).hexdigest(),
            "json_utf8": verified.augmentation_manifest_bytes.decode("utf-8"),
        }
    return {
        "schema_version": 1,
        "export_unit": "LOCAL_TILE",
        "identities": {
            "tile_revision_id": verified.tile_revision_id,
            "geometry_revision_id": verified.metadata["geometry_revision_id"],
            "source_generation_id": verified.source_generation_id,
            "appearance_revision_id": verified.appearance_revision_id,
            "lod_id": verified.lod_id,
            "rgbd_augmented_revision_id": (
                getattr(verified.augmentation, "rgbd_augmented_revision_id", None)
                if verified.augmentation is not None
                else None
            ),
            "source_layers": list(layers),
        },
        "manifests": manifests,
    }


def _building_provenance(
    assembly_id: str,
    building_revision: str,
    assembly_manifest_bytes: bytes,
    building_manifest_bytes: bytes,
    verified_tiles: tuple[_VerifiedTile, ...],
    members: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    tile_manifests = {
        value.tile_revision_id: {
            "sha256": hashlib.sha256(value.appearance_manifest_bytes).hexdigest(),
            "json_utf8": value.appearance_manifest_bytes.decode("utf-8"),
        }
        for value in verified_tiles
    }
    augmentation_manifests = {
        value.tile_revision_id: {
            "sha256": hashlib.sha256(value.augmentation_manifest_bytes).hexdigest(),
            "json_utf8": value.augmentation_manifest_bytes.decode("utf-8"),
        }
        for value in verified_tiles
        if value.augmentation_manifest_bytes is not None
    }
    manifests: dict[str, Any] = {
        "assembly": {
            "sha256": hashlib.sha256(assembly_manifest_bytes).hexdigest(),
            "json_utf8": assembly_manifest_bytes.decode("utf-8"),
        },
        "building_appearance": {
            "sha256": hashlib.sha256(building_manifest_bytes).hexdigest(),
            "json_utf8": building_manifest_bytes.decode("utf-8"),
        },
        "tile_appearances": tile_manifests,
    }
    if augmentation_manifests:
        manifests["rgbd_augmentations"] = augmentation_manifests
    return {
        "schema_version": 1,
        "export_unit": "BUILDING",
        "identities": {
            "assembly_id": assembly_id,
            "building_appearance_revision_id": building_revision,
            "members": [dict(value) for value in members],
        },
        "manifests": manifests,
    }


def _tile_identity_extras(
    verified: _VerifiedTile,
    rgbd_augmented_revision_id: str | None,
) -> dict[str, Any]:
    core = verified.appearance.manifest["revision_core"]
    return {
        "tile_revision_id": verified.tile_revision_id,
        "geometry_revision_id": core["geometry_revision_id"],
        "source_generation_id": core["source_generation_id"],
        "pose_revision_id": core["pose_revision_id"],
        "calibration_id": core["calibration_id"],
        "calibration_sha256": core["calibration_sha256"],
        "physical_depth_validation_revision": core["physical_depth_validation_revision"],
        "model_revision_id": core["model_revision_id"],
        "assistance_config_sha256": core["assistance_config_sha256"],
        "source_frame_sha256": list(core["source_frame_sha256"]),
        "appearance_config_sha256": core["appearance_config_sha256"],
        "appearance_revision_id": verified.appearance_revision_id,
        "lod_id": verified.lod_id,
        "rgbd_augmented_revision_id": rgbd_augmented_revision_id,
    }


def _verified_assembly(assembly: Any, assembly_id: str) -> tuple[dict[str, Any], bytes]:
    path = Path(getattr(assembly, "path", "")) / "manifest.json"
    try:
        body = path.read_bytes()
        manifest = json.loads(body)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Building assembly manifest is unreadable") from exc
    if not isinstance(manifest, dict) or body != _canonical_bytes(manifest):
        raise ValueError("Building assembly manifest is not canonical")
    core = dict(manifest)
    if core.pop("assembly_id", None) != assembly_id or canonical_sha256(core) != assembly_id:
        raise ValueError("Building assembly manifest identity is invalid")
    return manifest, body


def _validate_assembly_members(assembly_members: tuple[Any, ...], manifest: Mapping[str, Any]) -> None:
    records = manifest.get("members")
    if not isinstance(records, list) or len(records) != len(assembly_members):
        raise ValueError("Building assembly manifest members are incomplete")
    for member, record in zip(assembly_members, records):
        if not isinstance(record, Mapping):
            raise ValueError("Building assembly manifest member is invalid")
        expected = {
            "building_id": member.building_id,
            "component_id": member.component_id,
            "session_id": member.session_id,
            "submap_id": member.submap_id,
            "generation_id": member.generation_id,
            "transform_building_from_submap": list(member.transform_building_from_submap),
            "source_decision_id": member.source_decision_id,
        }
        if any(record.get(name) != value for name, value in expected.items()):
            raise ValueError("Building assembly manifest member provenance is stale")


def _rigid_matrix(values: Any) -> np.ndarray:
    try:
        matrix = np.asarray(tuple(float(value) for value in values), dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("Building member transform must be a rigid 4x4 matrix") from exc
    rotation = matrix[:3, :3]
    if (
        not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("Building member transform must be rigid")
    return matrix


def _placement_revision(assembly_id: str, tile: Mapping[str, Any], member: Any) -> str:
    identity = {
        "assembly_id": assembly_id,
        "source_revision_id": tile["revision_id"],
        "component_id": member.component_id,
        "submap_id": member.submap_id,
        "generation_id": member.generation_id,
        "transform_building_from_submap": list(member.transform_building_from_submap),
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _write_export(
    *,
    output_path: Path,
    provenance: Mapping[str, Any],
    verified_tiles: tuple[_VerifiedTile, ...],
    selected_layers: Mapping[str, tuple[str, ...]],
    member_matrices: Mapping[str, np.ndarray | None],
    asset_extras: Mapping[str, Any],
    assembly_id: str | None,
    building_appearance_revision_id: str | None,
) -> ExportedGlb:
    provenance_bytes = _canonical_bytes(provenance)
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    extras = {**asset_extras, "provenance_sha256": provenance_sha}
    gltf = _build_gltf(verified_tiles, selected_layers, member_matrices, extras)
    output_path = Path(output_path)
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_glb = output_path.with_name(f".{output_path.name}.{token}.partial")
    temporary_provenance = provenance_path.with_name(f".{provenance_path.name}.{token}.partial")
    provenance_committed = False
    try:
        _write_fsync(temporary_provenance, provenance_bytes)
        gltf.save_binary(temporary_glb)
        _fsync_file(temporary_glb)
        GLTF2().load_binary(temporary_glb)
        os.replace(temporary_provenance, provenance_path)
        provenance_committed = True
        os.replace(temporary_glb, output_path)
        _fsync_directory(output_path.parent)
    except Exception:
        if provenance_committed and not output_path.exists():
            provenance_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_glb.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
    return ExportedGlb(
        output_path,
        provenance_path,
        provenance_sha,
        assembly_id,
        building_appearance_revision_id,
    )


def _build_gltf(
    verified_tiles: tuple[_VerifiedTile, ...],
    selected_layers: Mapping[str, tuple[str, ...]],
    member_matrices: Mapping[str, np.ndarray | None],
    asset_extras: Mapping[str, Any],
) -> GLTF2:
    binary = _BinaryBuilder()
    gltf = GLTF2(asset=Asset(version="2.0", generator="go2-mapping-glb-export-v1", extras=dict(asset_extras)))
    gltf.images = []
    gltf.textures = []
    gltf.samplers = [Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)]
    gltf.materials = []
    gltf.meshes = []
    gltf.nodes = []
    root_index = len(gltf.nodes)
    gltf.nodes.append(Node(name="REP103_TO_GLTF", matrix=_matrix_column_major(_REP103_TO_GLTF), children=[]))
    for verified in verified_tiles:
        tile_node_index = len(gltf.nodes)
        tile_matrix = member_matrices.get(verified.tile_revision_id)
        tile_node = Node(
            name=f"TILE/{verified.tile_revision_id}",
            children=[],
            extras={
                "tile_revision_id": verified.tile_revision_id,
                "appearance_revision_id": verified.appearance_revision_id,
                "lod_id": verified.lod_id,
                "source_generation_id": verified.source_generation_id,
            },
        )
        if tile_matrix is not None:
            tile_node.matrix = _matrix_column_major(tile_matrix)
        gltf.nodes.append(tile_node)
        gltf.nodes[root_index].children.append(tile_node_index)
        _append_tile_layers(
            gltf,
            binary,
            verified,
            selected_layers[verified.tile_revision_id],
            tile_node_index,
        )
    gltf.scenes = [Scene(nodes=[root_index])]
    gltf.scene = 0
    binary._align()
    gltf.bufferViews = binary.views
    gltf.accessors = binary.accessors
    gltf.buffers = [Buffer(byteLength=len(binary.body))]
    gltf.set_binary_blob(bytes(binary.body))
    return gltf


def _append_tile_layers(
    gltf: GLTF2,
    binary: _BinaryBuilder,
    verified: _VerifiedTile,
    layers: tuple[str, ...],
    tile_node_index: int,
) -> None:
    position_accessor = binary.accessor(
        np.ascontiguousarray(verified.vertices.positions, dtype="<f4"),
        component_type=FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
        include_bounds=True,
    )
    normal_accessor = binary.accessor(
        np.ascontiguousarray(verified.vertices.normals, dtype="<f4"),
        component_type=FLOAT,
        accessor_type="VEC3",
        target=ARRAY_BUFFER,
    )
    uv_accessor = binary.accessor(
        np.ascontiguousarray(verified.uvs.uvs, dtype="<f4"),
        component_type=FLOAT,
        accessor_type="VEC2",
        target=ARRAY_BUFFER,
    )
    texture_by_atlas: dict[int, int] = {}
    for atlas in range(verified.lod_core["atlas_count"]):
        view = binary.view(verified.artifacts[f"atlas-{atlas}.png"])
        image_index = len(gltf.images)
        gltf.images.append(Image(name=f"{verified.tile_revision_id}/atlas-{atlas}", bufferView=view, mimeType="image/png"))
        texture_index = len(gltf.textures)
        gltf.textures.append(Texture(source=image_index, sampler=0))
        texture_by_atlas[atlas] = texture_index

    for layer in layers:
        primitives = [value for value in verified.primitives if value["source_layer"] == layer]
        if not primitives:
            continue
        gltf_primitives: list[Primitive] = []
        for primitive in primitives:
            first_triangle = primitive["index_offset"] // 3
            triangle_count = primitive["index_count"] // 3
            primitive_indices = np.ascontiguousarray(
                verified.indices.triangles[first_triangle : first_triangle + triangle_count].reshape(-1),
                dtype="<u4",
            )
            index_accessor = binary.accessor(
                primitive_indices,
                component_type=UNSIGNED_INT,
                accessor_type="SCALAR",
                target=ELEMENT_ARRAY_BUFFER,
            )
            material_index = _append_material(
                gltf,
                primitive,
                texture_by_atlas,
                verified.tile_revision_id,
            )
            gltf_primitives.append(
                Primitive(
                    attributes=Attributes(
                        POSITION=position_accessor,
                        NORMAL=normal_accessor,
                        TEXCOORD_0=uv_accessor,
                    ),
                    indices=index_accessor,
                    material=material_index,
                    mode=TRIANGLES,
                    extras={
                        "primitive_id": primitive["primitive_id"],
                        "source_layer": layer,
                        "material_kind": primitive["material"],
                        "atlas_index": primitive["atlas_index"],
                        "source_triangle_ids": list(primitive["source_triangle_ids"]),
                    },
                )
            )
        mesh_index = len(gltf.meshes)
        gltf.meshes.append(Mesh(name=f"{layer}/{verified.tile_revision_id}", primitives=gltf_primitives))
        node_name = layer if layer == "LIDAR_MEASURED" else f"RGBD_AUGMENTED/{verified.tile_revision_id}"
        node_index = len(gltf.nodes)
        gltf.nodes.append(
            Node(
                name=node_name,
                mesh=mesh_index,
                extras={
                    "tile_revision_id": verified.tile_revision_id,
                    "appearance_revision_id": verified.appearance_revision_id,
                    "lod_id": verified.lod_id,
                    "source_generation_id": verified.source_generation_id,
                    "source_layer": layer,
                },
            )
        )
        gltf.nodes[tile_node_index].children.append(node_index)


def _append_material(
    gltf: GLTF2,
    primitive: Mapping[str, Any],
    texture_by_atlas: Mapping[int, int],
    tile_revision_id: str,
) -> int:
    extras = {
        "tile_revision_id": tile_revision_id,
        "source_layer": primitive["source_layer"],
        "material_kind": primitive["material"],
        "atlas_index": primitive["atlas_index"],
    }
    if primitive["material"] == "VALIDATED_UV_ATLAS":
        pbr = PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=texture_by_atlas[primitive["atlas_index"]]),
            metallicFactor=0.03,
            roughnessFactor=0.78,
        )
        material = Material(
            name=f"VALIDATED_UV_ATLAS/{primitive['source_layer']}/{primitive['atlas_index']}",
            pbrMetallicRoughness=pbr,
            alphaMode="MASK",
            alphaCutoff=0.5,
            doubleSided=True,
            extras=extras,
        )
    else:
        material = Material(
            name=f"BASE_MATERIAL/{primitive['source_layer']}",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[0.62, 0.66, 0.70, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
            extras=extras,
        )
    index = len(gltf.materials)
    gltf.materials.append(material)
    return index


def _matrix_column_major(matrix: np.ndarray) -> list[float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError("node transform must be a finite 4x4 matrix")
    return [float(value) for value in values.reshape(-1, order="F")]


def _write_fsync(path: Path, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


__all__ = [
    "AppearanceExportSources",
    "BuildingAppearanceResolver",
    "BuildingAssemblyResolver",
    "ExportedGlb",
    "RgbdAugmentationResolver",
    "TileAppearanceResolver",
    "TileGeometryResolver",
    "export_building_glb",
    "export_local_tile_glb",
]
