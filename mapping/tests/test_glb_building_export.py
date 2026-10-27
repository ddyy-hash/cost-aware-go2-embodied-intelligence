from dataclasses import dataclass
import hashlib
from io import BytesIO
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pygltflib import GLTF2
import pytest
import trimesh

from mapping.pc.go2_mapping_appearance.atlas_codec import (
    encode_render_indices_v1,
    encode_render_uvs_v1,
    encode_render_vertices_v1,
)
from mapping.pc.go2_mapping_appearance.building_store import (
    BuildingAppearanceMember,
    BuildingAppearanceStore,
    DistanceLodInterval,
)
from mapping.pc.go2_mapping_appearance.tile_identity import (
    ArtifactRef,
    DeliveryAuthorization,
    PrimitiveRecord,
    build_lod_record,
    tile_appearance_revision,
)
from mapping.pc.go2_mapping_appearance.tile_store import TileAppearanceStore, TileLodPayload
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember, BuildingAssembly
from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision
from mapping.pc.go2_mapping_live.live_tiles import ImmutableTileStore
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
ROTATED_TRANSLATED = (
    0.0, -1.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 2.0,
    0.0, 0.0, 1.0, 3.0,
    0.0, 0.0, 0.0, 1.0,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _png(color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (4, 4), color)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _export_api():
    module = importlib.import_module("mapping.pc.go2_mapping_appearance.glb_export")
    if not hasattr(module, "export_building_glb"):
        pytest.fail("Building GLB exporter is not implemented")
    return module.AppearanceExportSources, module.export_building_glb


@dataclass(frozen=True)
class _PublishedMember:
    metadata: dict
    appearance: object
    lod_id: str
    augmentation: object | None


@dataclass(frozen=True)
class _BuildingFixture:
    tile_store: ImmutableTileStore
    appearance_store: TileAppearanceStore
    building_store: BuildingAppearanceStore
    assembly: BuildingAssembly
    assembly_manifest_path: Path
    building_appearance: object
    members: tuple[_PublishedMember, ...]


def _assembly(tmp_path: Path) -> tuple[BuildingAssembly, Path]:
    members = (
        AssemblyMember(
            "building-a",
            "component-root",
            "session-root",
            "submap-root",
            _hash("generation-root"),
            IDENTITY,
        ),
        AssemblyMember(
            "building-a",
            "component-attached",
            "session-attached",
            "submap-attached",
            _hash("generation-attached"),
            ROTATED_TRANSLATED,
            _hash("decision-attached"),
        ),
    )
    manifest_members = []
    for index, member in enumerate(members):
        manifest_members.append(
            {
                **member.payload(),
                "floor_id": "floor-1",
                "zone_id": f"zone-{index}",
                "attachment_state": "ROOT" if index == 0 else "ATTACHED",
                "geometry_authority": "OBSERVED_SURFACE",
            }
        )
    core = {
        "schema_version": 1,
        "building_id": "building-a",
        "algorithm_version": "assembly-v1",
        "geometry_authority": "OBSERVED_SURFACE",
        "volume_filled": False,
        "watertight": False,
        "members": manifest_members,
    }
    assembly_id = canonical_sha256(core)
    root = tmp_path / "assembly" / assembly_id
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_canonical({**core, "assembly_id": assembly_id}))
    return BuildingAssembly(assembly_id, "building-a", "assembly-v1", members, root), manifest_path


def _publish_member(
    *,
    tmp_path: Path,
    index: int,
    member: AssemblyMember,
    tile_store: ImmutableTileStore,
    appearance_store: TileAppearanceStore,
    augmented: bool,
) -> _PublishedMember:
    positions = np.asarray(
        [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]], dtype="<f4"
    ) + np.asarray([index, 0, 0], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.01], dtype="<f4")
    geometry_revision = hashlib.sha256(
        positions.tobytes() + normals.tobytes() + triangles.tobytes() + scales.tobytes()
    ).hexdigest()
    metadata = tile_store.publish(
        member.building_id,
        "floor-1",
        member.component_id,
        "ROOT" if member.source_decision_id is None else "ATTACHED",
        member.submap_id,
        MeshTileRevision(TileId(index, 0, 0), geometry_revision, positions, normals, triangles, scales, 0),
        source_generation_id=member.generation_id,
        pose_revision_id=_hash(f"pose-{index}"),
    )
    augmentation_id = _hash(f"augmentation-{index}") if augmented else None
    revision = tile_appearance_revision(
        tile_revision_id=metadata["revision_id"],
        geometry_revision_id=geometry_revision,
        source_generation_id=member.generation_id,
        calibration_id=f"calibration-{index}",
        calibration_sha256=_hash(f"calibration-{index}"),
        physical_depth_validation_revision=_hash(f"depth-{index}"),
        pose_revision_id=metadata["pose_revision_id"],
        model_revision_id=_hash(f"model-{index}") if augmented else None,
        assistance_config_sha256=_hash(f"assistance-{index}") if augmented else None,
        source_frame_sha256=(_hash(f"frame-{index}-a"), _hash(f"frame-{index}-b")),
        appearance_config_sha256=_hash(f"appearance-{index}"),
        charting={"implementation": "xatlas", "version": "0.0.11", "options": {}},
        atlas_density_policy={"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        padding_policy={"pixels": 8, "mip_levels": 4},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=augmentation_id,
        contains_augmented_primitives=augmented,
        measured_surface_coverage_ppm=800_000,
        minimum_independent_view_count=2,
    )
    render_positions = positions.copy()
    render_normals = normals.copy()
    render_triangles = triangles.copy()
    source_vertex_ids = np.arange(3, dtype="<u4")
    source_triangle_ids = np.asarray([0], dtype="<u4")
    uvs = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype="<f4")
    atlas_indices = np.zeros(3, dtype="<u2")
    flags = np.ones(3, dtype="<u2")
    primitives = [
        PrimitiveRecord(0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,))
    ]
    if augmented:
        augmented_positions = np.asarray(
            [[0.5, 0, 0], [0.7, 0, 0], [0.5, 0.2, 0]], dtype="<f4"
        ) + np.asarray([index, 0, 0], dtype="<f4")
        render_positions = np.concatenate((render_positions, augmented_positions))
        render_normals = np.concatenate((render_normals, normals))
        render_triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<u4")
        source_vertex_ids = np.arange(6, dtype="<u4")
        source_triangle_ids = np.asarray([0, 0], dtype="<u4")
        uvs = np.concatenate((uvs, np.zeros((3, 2), dtype="<f4")))
        atlas_indices = np.asarray([0, 0, 0, 65535, 65535, 65535], dtype="<u2")
        flags = np.asarray([1, 1, 1, 0, 0, 0], dtype="<u2")
        primitives.append(
            PrimitiveRecord(1, "BASE_MATERIAL", 65535, 3, 3, "RGBD_AUGMENTED", (0,))
        )
    artifacts = {
        "render-vertices.bin": encode_render_vertices_v1(
            metadata["revision_id"], revision.appearance_revision_id, render_positions, render_normals, source_vertex_ids
        ),
        "render-indices.bin": encode_render_indices_v1(
            metadata["revision_id"], revision.appearance_revision_id, render_triangles, source_triangle_ids
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            metadata["revision_id"], revision.appearance_revision_id, uvs, atlas_indices, flags
        ),
        "atlas-0.png": _png((40 + index * 80, 90, 160, 255)),
    }
    refs = tuple(
        ArtifactRef(name, hashlib.sha256(body).hexdigest(), len(body))
        for name, body in artifacts.items()
    )
    lod = build_lod_record(
        tile_revision_id=metadata["revision_id"],
        appearance_revision_id=revision.appearance_revision_id,
        policy={"name": "exact", "order": 0, "mm_per_texel": 3},
        vertex_count=len(render_positions),
        triangle_count=len(render_triangles),
        atlas_count=1,
        artifacts=refs,
        primitives=tuple(primitives),
        rgbd_augmented_revision_id=augmentation_id,
    )
    appearance = appearance_store.publish(revision, (TileLodPayload(lod, artifacts),))
    augmentation = None
    if augmented:
        manifest = {
            "schema_version": 1,
            "name": "RGBD_AUGMENTED",
            "geometry_authority": False,
            "source_generation_id": member.generation_id,
            "rgbd_augmented_revision_id": augmentation_id,
        }
        manifest_path = tmp_path / f"augmentation-{index}.json"
        manifest_path.write_bytes(_canonical(manifest))
        augmentation = SimpleNamespace(
            source_generation_id=member.generation_id,
            rgbd_augmented_revision_id=augmentation_id,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            layer=SimpleNamespace(name="RGBD_AUGMENTED", geometry_authority=False),
        )
    return _PublishedMember(metadata, appearance, lod.lod_id, augmentation)


def _building_fixture(tmp_path: Path) -> _BuildingFixture:
    assembly, assembly_manifest_path = _assembly(tmp_path)
    tile_store = ImmutableTileStore(tmp_path / "tiles")
    appearance_store = TileAppearanceStore(tmp_path / "tile-appearance")
    published_members = tuple(
        _publish_member(
            tmp_path=tmp_path,
            index=index,
            member=member,
            tile_store=tile_store,
            appearance_store=appearance_store,
            augmented=index == 1,
        )
        for index, member in enumerate(assembly.members)
    )
    tile_manifest = tile_store.manifest_for_assembly(assembly)
    tiles_by_revision = {value["revision_id"]: value for value in tile_manifest["tiles"]}
    authorization = DeliveryAuthorization(800_000, 2, 800_000, 2, True, ())
    building_members = []
    for assembly_member, published in zip(assembly.members, published_members):
        metadata = tiles_by_revision[published.metadata["revision_id"]]
        augmentation_id = (
            None
            if published.augmentation is None
            else published.augmentation.rgbd_augmented_revision_id
        )
        building_members.append(
            BuildingAppearanceMember(
                building_id=assembly.building_id,
                component_id=assembly_member.component_id,
                session_id=assembly_member.session_id,
                submap_id=assembly_member.submap_id,
                source_generation_id=assembly_member.generation_id,
                placement_revision_id=metadata["placement_revision_id"],
                tile_revision_id=metadata["revision_id"],
                tile_appearance_revision_id=published.appearance.appearance_revision_id,
                authorized_lod_ids=(published.lod_id,),
                distance_policy=(DistanceLodInterval(0.0, None, published.lod_id),),
                transform_building_from_submap=assembly_member.transform_building_from_submap,
                delivery_authorization=authorization,
                rgbd_augmented_revision_id=augmentation_id,
            )
        )
    assemblies = {assembly.assembly_id: assembly}
    building_store = BuildingAppearanceStore(
        tmp_path / "building-store",
        assembly_resolver=assemblies.get,
        tile_manifest_resolver=tile_store.manifest_for_assembly,
        tile_resolver=tile_store.load,
        tile_appearance_resolver=appearance_store.resolve,
    )
    building_appearance = building_store.publish(
        assembly=assembly, members=tuple(building_members)
    )
    return _BuildingFixture(
        tile_store,
        appearance_store,
        building_store,
        assembly,
        assembly_manifest_path,
        building_appearance,
        published_members,
    )


def _sources(
    fixture: _BuildingFixture,
    sources_type,
    *,
    tile_geometry=None,
    building_appearance=None,
    assembly=None,
    augmentation=None,
):
    augmentations = {
        (value.augmentation.source_generation_id, value.augmentation.rgbd_augmented_revision_id): value.augmentation
        for value in fixture.members
        if value.augmentation is not None
    }
    return sources_type(
        tile_geometry=fixture.tile_store.load if tile_geometry is None else tile_geometry,
        tile_appearance=fixture.appearance_store.resolve,
        building_appearance=fixture.building_store.resolve if building_appearance is None else building_appearance,
        assembly=(lambda revision: fixture.assembly if revision == fixture.assembly.assembly_id else None)
        if assembly is None
        else assembly,
        augmentation=(
            lambda generation, revision: augmentations.get((generation, revision))
        )
        if augmentation is None
        else augmentation,
    )


def test_building_glb_round_trips_complete_exact_members_transforms_layers_and_provenance(tmp_path):
    sources_type, export_building = _export_api()
    fixture = _building_fixture(tmp_path)
    lod_by_member = {value.metadata["revision_id"]: value.lod_id for value in fixture.members}
    output = tmp_path / "exports" / "building.glb"

    result = export_building(
        sources=_sources(fixture, sources_type),
        assembly_id=fixture.assembly.assembly_id,
        building_appearance_revision_id=fixture.building_appearance.building_appearance_revision_id,
        lod_by_member=lod_by_member,
        output_path=output,
    )

    assert result.assembly_id == fixture.assembly.assembly_id
    assert result.building_appearance_revision_id == fixture.building_appearance.building_appearance_revision_id
    provenance_bytes = result.provenance_path.read_bytes()
    assert result.provenance_sha256 == hashlib.sha256(provenance_bytes).hexdigest()
    provenance = json.loads(provenance_bytes)
    assert provenance["manifests"]["assembly"]["json_utf8"].encode() == fixture.assembly_manifest_path.read_bytes()
    assert provenance["manifests"]["building_appearance"]["json_utf8"].encode() == (
        fixture.building_appearance.manifest_path.read_bytes()
    )
    assert set(provenance["manifests"]["tile_appearances"]) == set(lod_by_member)

    gltf = GLTF2().load_binary(output)
    assert gltf.asset.extras["export_unit"] == "BUILDING"
    assert gltf.asset.extras["assembly_id"] == fixture.assembly.assembly_id
    assert gltf.asset.extras["building_appearance_revision_id"] == (
        fixture.building_appearance.building_appearance_revision_id
    )
    assert gltf.asset.extras["provenance_sha256"] == result.provenance_sha256
    assert {value["tile_revision_id"] for value in gltf.asset.extras["members"]} == set(lod_by_member)
    attached_asset_member = next(
        value
        for value in gltf.asset.extras["members"]
        if value["tile_revision_id"] == fixture.members[1].metadata["revision_id"]
    )
    attached_core = fixture.members[1].appearance.manifest["revision_core"]
    assert attached_asset_member["calibration_sha256"] == attached_core["calibration_sha256"]
    assert attached_asset_member["physical_depth_validation_revision"] == attached_core[
        "physical_depth_validation_revision"
    ]
    assert attached_asset_member["pose_revision_id"] == attached_core["pose_revision_id"]
    assert attached_asset_member["model_revision_id"] == attached_core["model_revision_id"]
    assert attached_asset_member["assistance_config_sha256"] == attached_core[
        "assistance_config_sha256"
    ]
    assert attached_asset_member["source_frame_sha256"] == attached_core["source_frame_sha256"]
    assert attached_asset_member["appearance_config_sha256"] == attached_core[
        "appearance_config_sha256"
    ]
    root = next(node for node in gltf.nodes if node.name == "REP103_TO_GLTF")
    assert root.matrix == pytest.approx(
        [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    )
    tile_nodes = {
        node.extras["tile_revision_id"]: node
        for node in gltf.nodes
        if node.name and node.name.startswith("TILE/")
    }
    assert set(tile_nodes) == set(lod_by_member)
    root_tile = fixture.members[0].metadata["revision_id"]
    attached_tile = fixture.members[1].metadata["revision_id"]
    assert tile_nodes[root_tile].matrix == pytest.approx(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    assert tile_nodes[attached_tile].matrix == pytest.approx(
        [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]
    )
    augmented = next(
        node for node in gltf.nodes if node.name == f"RGBD_AUGMENTED/{attached_tile}"
    )
    assert augmented.extras["source_layer"] == "RGBD_AUGMENTED"
    assert augmented.extras["tile_revision_id"] == attached_tile
    assert sum(node.name == "LIDAR_MEASURED" for node in gltf.nodes) == 2

    independent = trimesh.load(output, force="scene", process=False)
    assert isinstance(independent, trimesh.Scene)
    assert len(independent.geometry) == 3


@pytest.mark.parametrize(
    "failure",
    (
        "missing-member",
        "extra-member",
        "unauthorized-lod",
        "stale-assembly",
        "changed-tile",
        "missing-augmentation",
        "in-progress",
    ),
)
def test_building_export_failure_matrix_leaves_no_partial_artifacts(tmp_path, failure):
    sources_type, export_building = _export_api()
    fixture = _building_fixture(tmp_path)
    lod_by_member = {value.metadata["revision_id"]: value.lod_id for value in fixture.members}
    tile_geometry = fixture.tile_store.load
    assembly_resolver = None
    augmentation_resolver = None
    building_resolver = None
    if failure == "missing-member":
        lod_by_member.pop(next(iter(lod_by_member)))
    elif failure == "extra-member":
        lod_by_member[_hash("extra-tile")] = next(iter(lod_by_member.values()))
    elif failure == "unauthorized-lod":
        lod_by_member[next(iter(lod_by_member))] = _hash("unauthorized-lod")
    elif failure == "stale-assembly":
        stale = BuildingAssembly(
            fixture.assembly.assembly_id,
            fixture.assembly.building_id,
            "assembly-v2",
            fixture.assembly.members,
            fixture.assembly.path,
        )
        assembly_resolver = lambda _revision: stale
    elif failure == "changed-tile":
        original = tile_geometry
        changed_revision = next(iter(lod_by_member))

        def changed_tile(revision):
            metadata, body = original(revision)
            return (metadata, body + b"changed") if revision == changed_revision else (metadata, body)

        tile_geometry = changed_tile
    elif failure == "missing-augmentation":
        augmentation_resolver = lambda _generation, _revision: None
    else:
        def in_progress(_revision):
            raise RuntimeError("appearance job is in progress")

        building_resolver = in_progress
    output = tmp_path / "rejected-building.glb"

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError), match="member|LOD|lod|assembly|content|augmentation|progress"):
        export_building(
            sources=_sources(
                fixture,
                sources_type,
                tile_geometry=tile_geometry,
                building_appearance=building_resolver,
                assembly=assembly_resolver,
                augmentation=augmentation_resolver,
            ),
            assembly_id=fixture.assembly.assembly_id,
            building_appearance_revision_id=fixture.building_appearance.building_appearance_revision_id,
            lod_by_member=lod_by_member,
            output_path=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".glb.provenance.json").exists()
    assert not list(tmp_path.rglob("*.partial"))
