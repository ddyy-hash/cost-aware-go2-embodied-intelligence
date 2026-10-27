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
from mapping.pc.go2_mapping_appearance.tile_identity import (
    ArtifactRef,
    PrimitiveRecord,
    build_lod_record,
    tile_appearance_revision,
)
from mapping.pc.go2_mapping_appearance.tile_store import TileAppearanceStore, TileLodPayload
from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision
from mapping.pc.go2_mapping_live.live_tiles import ImmutableTileStore


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _png(color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (4, 4), color)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _export_api():
    try:
        module = importlib.import_module("mapping.pc.go2_mapping_appearance.glb_export")
    except ModuleNotFoundError:
        pytest.fail("GLB exporter is not implemented")
    return module.AppearanceExportSources, module.export_local_tile_glb


@dataclass(frozen=True)
class _LocalFixture:
    tile_store: ImmutableTileStore
    appearance_store: TileAppearanceStore
    tile_metadata: dict
    published_appearance: object
    lod_id: str
    augmentation: object
    render_positions: np.ndarray
    render_normals: np.ndarray
    render_triangles: np.ndarray


def _local_fixture(tmp_path: Path) -> _LocalFixture:
    source_positions = np.asarray(
        [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], dtype="<f4"
    )
    source_normals = np.asarray([[0, 0, 1]] * 4, dtype="<f4")
    source_triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<u4")
    source_scales = np.asarray([0.01, 0.02], dtype="<f4")
    geometry_revision_id = hashlib.sha256(
        source_positions.tobytes()
        + source_normals.tobytes()
        + source_triangles.tobytes()
        + source_scales.tobytes()
    ).hexdigest()
    source_generation_id = _hash("generation-local")
    tile_store = ImmutableTileStore(tmp_path / "tiles")
    tile_metadata = tile_store.publish(
        "building-a",
        "floor-1",
        "component-local",
        "ROOT",
        "submap-local",
        MeshTileRevision(
            TileId(2, -1, 0),
            geometry_revision_id,
            source_positions,
            source_normals,
            source_triangles,
            source_scales,
            0,
        ),
        source_generation_id=source_generation_id,
        pose_revision_id=_hash("pose-local"),
    )

    augmentation_id = _hash("augmentation-local")
    revision = tile_appearance_revision(
        tile_revision_id=tile_metadata["revision_id"],
        geometry_revision_id=geometry_revision_id,
        source_generation_id=source_generation_id,
        calibration_id="calibration-local",
        calibration_sha256=_hash("calibration-local"),
        physical_depth_validation_revision=_hash("depth-local"),
        pose_revision_id=tile_metadata["pose_revision_id"],
        model_revision_id=_hash("model-local"),
        assistance_config_sha256=_hash("assistance-local"),
        source_frame_sha256=(_hash("frame-local-a"), _hash("frame-local-b")),
        appearance_config_sha256=_hash("appearance-local"),
        charting={"implementation": "xatlas", "version": "0.0.11", "options": {}},
        atlas_density_policy={"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        padding_policy={"pixels": 8, "mip_levels": 4},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=augmentation_id,
        contains_augmented_primitives=True,
        measured_surface_coverage_ppm=800_000,
        minimum_independent_view_count=2,
    )

    render_positions = np.asarray(
        [
            [1, 2, 3], [2, 2, 3], [1, 3, 3],
            [-1, 0, 1], [-2, 0, 1], [-1, 1, 1],
            [0.5, -1, 2], [1.5, -1, 2], [0.5, -2, 2],
        ],
        dtype="<f4",
    )
    render_normals = np.asarray(
        [[0, 0, 1]] * 6 + [[0, 1, 0]] * 3,
        dtype="<f4",
    )
    render_triangles = np.asarray(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="<u4"
    )
    uvs = np.asarray(
        [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.2, 0.2], [0.8, 0.2], [0.2, 0.8],
        ],
        dtype="<f4",
    )
    atlas_indices = np.asarray([0, 0, 0, 65535, 65535, 65535, 1, 1, 1], dtype="<u2")
    flags = np.asarray([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype="<u2")
    artifacts = {
        "render-vertices.bin": encode_render_vertices_v1(
            tile_metadata["revision_id"],
            revision.appearance_revision_id,
            render_positions,
            render_normals,
            np.arange(9, dtype="<u4"),
        ),
        "render-indices.bin": encode_render_indices_v1(
            tile_metadata["revision_id"],
            revision.appearance_revision_id,
            render_triangles,
            np.asarray([0, 1, 0], dtype="<u4"),
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            tile_metadata["revision_id"],
            revision.appearance_revision_id,
            uvs,
            atlas_indices,
            flags,
        ),
        "atlas-0.png": _png((200, 40, 20, 255)),
        "atlas-1.png": _png((20, 80, 220, 255)),
    }
    references = tuple(
        ArtifactRef(name, hashlib.sha256(body).hexdigest(), len(body))
        for name, body in artifacts.items()
    )
    lod = build_lod_record(
        tile_revision_id=tile_metadata["revision_id"],
        appearance_revision_id=revision.appearance_revision_id,
        policy={"name": "near", "order": 0, "mm_per_texel": 3},
        vertex_count=9,
        triangle_count=3,
        atlas_count=2,
        artifacts=references,
        primitives=(
            PrimitiveRecord(0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,)),
            PrimitiveRecord(1, "BASE_MATERIAL", 65535, 3, 3, "LIDAR_MEASURED", (1,)),
            PrimitiveRecord(2, "VALIDATED_UV_ATLAS", 1, 6, 3, "RGBD_AUGMENTED", (0,)),
        ),
        rgbd_augmented_revision_id=augmentation_id,
    )
    appearance_store = TileAppearanceStore(tmp_path / "appearance")
    published_appearance = appearance_store.publish(revision, (TileLodPayload(lod, artifacts),))

    augmentation_manifest = {
        "schema_version": 1,
        "name": "RGBD_AUGMENTED",
        "geometry_authority": False,
        "source_generation_id": source_generation_id,
        "rgbd_augmented_revision_id": augmentation_id,
    }
    augmentation_manifest_path = tmp_path / "augmentation-manifest.json"
    augmentation_manifest_path.write_bytes(
        json.dumps(
            augmentation_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    augmentation = SimpleNamespace(
        source_generation_id=source_generation_id,
        rgbd_augmented_revision_id=augmentation_id,
        manifest=augmentation_manifest,
        manifest_path=augmentation_manifest_path,
        manifest_sha256=hashlib.sha256(augmentation_manifest_path.read_bytes()).hexdigest(),
        layer=SimpleNamespace(name="RGBD_AUGMENTED", geometry_authority=False),
    )
    return _LocalFixture(
        tile_store,
        appearance_store,
        tile_metadata,
        published_appearance,
        lod.lod_id,
        augmentation,
        render_positions,
        render_normals,
        render_triangles,
    )


_COMPONENT_DTYPES = {5123: "<u2", 5125: "<u4", 5126: "<f4"}
_TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor_array(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    width = _TYPE_WIDTHS[accessor.type]
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    result = np.frombuffer(
        gltf.binary_blob(),
        dtype=_COMPONENT_DTYPES[accessor.componentType],
        count=accessor.count * width,
        offset=offset,
    ).copy()
    return result.reshape(accessor.count, width) if width > 1 else result


def _sources(fixture: _LocalFixture, sources_type, *, tile_loader=None, augmentation=None):
    return sources_type(
        tile_geometry=fixture.tile_store.load if tile_loader is None else tile_loader,
        tile_appearance=fixture.appearance_store.resolve,
        building_appearance=lambda _revision: None,
        assembly=lambda _assembly: None,
        augmentation=(
            lambda generation, revision: fixture.augmentation
            if (generation, revision)
            == (
                fixture.augmentation.source_generation_id,
                fixture.augmentation.rgbd_augmented_revision_id,
            )
            else None
        )
        if augmentation is None
        else augmentation,
    )


def test_local_tile_glb_round_trips_exact_geometry_layers_textures_and_provenance(tmp_path):
    sources_type, export_local = _export_api()
    fixture = _local_fixture(tmp_path)
    output = tmp_path / "exports" / "tile.glb"

    result = export_local(
        sources=_sources(fixture, sources_type),
        tile_revision_id=fixture.tile_metadata["revision_id"],
        appearance_revision_id=fixture.published_appearance.appearance_revision_id,
        lod_id=fixture.lod_id,
        rgbd_augmented_revision_id=fixture.augmentation.rgbd_augmented_revision_id,
        output_path=output,
    )

    assert result.glb_path == output
    assert result.provenance_path == output.with_suffix(".glb.provenance.json")
    assert result.assembly_id is None
    assert result.building_appearance_revision_id is None
    provenance_bytes = result.provenance_path.read_bytes()
    assert result.provenance_sha256 == hashlib.sha256(provenance_bytes).hexdigest()
    provenance = json.loads(provenance_bytes)
    assert provenance["manifests"]["tile_appearance"]["json_utf8"].encode("utf-8") == (
        fixture.published_appearance.manifest_path.read_bytes()
    )

    gltf = GLTF2().load_binary(output)
    assert gltf.asset.extras["schema_version"] == 1
    assert gltf.asset.extras["export_unit"] == "LOCAL_TILE"
    assert gltf.asset.extras["units"] == "metres"
    assert gltf.asset.extras["tile_revision_id"] == fixture.tile_metadata["revision_id"]
    assert gltf.asset.extras["appearance_revision_id"] == fixture.published_appearance.appearance_revision_id
    assert gltf.asset.extras["lod_id"] == fixture.lod_id
    assert gltf.asset.extras["provenance_sha256"] == result.provenance_sha256
    assert gltf.asset.extras["coordinate_mapping"] == "REP103_XYZ_TO_GLTF_XZ_NEGATIVE_Y"
    core = fixture.published_appearance.manifest["revision_core"]
    assert gltf.asset.extras["identities"] == {
        "tile_revision_id": fixture.tile_metadata["revision_id"],
        "geometry_revision_id": core["geometry_revision_id"],
        "source_generation_id": core["source_generation_id"],
        "pose_revision_id": core["pose_revision_id"],
        "calibration_id": core["calibration_id"],
        "calibration_sha256": core["calibration_sha256"],
        "physical_depth_validation_revision": core["physical_depth_validation_revision"],
        "model_revision_id": core["model_revision_id"],
        "assistance_config_sha256": core["assistance_config_sha256"],
        "source_frame_sha256": core["source_frame_sha256"],
        "appearance_config_sha256": core["appearance_config_sha256"],
        "appearance_revision_id": fixture.published_appearance.appearance_revision_id,
        "lod_id": fixture.lod_id,
        "rgbd_augmented_revision_id": fixture.augmentation.rgbd_augmented_revision_id,
    }
    assert len(gltf.images) == 2
    assert all(image.uri is None and image.mimeType == "image/png" for image in gltf.images)
    for image in gltf.images:
        view = gltf.bufferViews[image.bufferView]
        start = view.byteOffset or 0
        assert gltf.binary_blob()[start : start + view.byteLength].startswith(b"\x89PNG\r\n\x1a\n")

    root = next(node for node in gltf.nodes if node.name == "REP103_TO_GLTF")
    assert root.matrix == pytest.approx(
        [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    )
    measured = next(node for node in gltf.nodes if node.name == "LIDAR_MEASURED")
    augmented = next(
        node
        for node in gltf.nodes
        if node.name == f"RGBD_AUGMENTED/{fixture.tile_metadata['revision_id']}"
    )
    assert measured.extras["source_layer"] == "LIDAR_MEASURED"
    assert augmented.extras["source_layer"] == "RGBD_AUGMENTED"
    assert measured.extras["tile_revision_id"] == augmented.extras["tile_revision_id"]
    assert len(gltf.meshes[measured.mesh].primitives) == 2
    assert len(gltf.meshes[augmented.mesh].primitives) == 1

    all_primitives = [
        *gltf.meshes[measured.mesh].primitives,
        *gltf.meshes[augmented.mesh].primitives,
    ]
    assert all(
        primitive.attributes.POSITION is not None
        and primitive.attributes.NORMAL is not None
        and primitive.attributes.TEXCOORD_0 is not None
        for primitive in all_primitives
    )
    np.testing.assert_array_equal(
        _accessor_array(gltf, all_primitives[0].attributes.POSITION), fixture.render_positions
    )
    np.testing.assert_array_equal(
        _accessor_array(gltf, all_primitives[0].attributes.NORMAL), fixture.render_normals
    )
    np.testing.assert_array_equal(
        _accessor_array(gltf, all_primitives[0].indices), fixture.render_triangles[0]
    )
    np.testing.assert_array_equal(
        _accessor_array(gltf, all_primitives[1].indices), fixture.render_triangles[1]
    )
    np.testing.assert_array_equal(
        _accessor_array(gltf, all_primitives[2].indices), fixture.render_triangles[2]
    )
    assert [gltf.materials[value.material].extras["material_kind"] for value in all_primitives] == [
        "VALIDATED_UV_ATLAS",
        "BASE_MATERIAL",
        "VALIDATED_UV_ATLAS",
    ]

    independent = trimesh.load(output, force="scene", process=False)
    assert isinstance(independent, trimesh.Scene)
    assert len(independent.geometry) == 3


def test_local_tile_export_omits_unrequested_augmentation_node(tmp_path):
    sources_type, export_local = _export_api()
    fixture = _local_fixture(tmp_path)
    calls = []
    output = tmp_path / "measured-only.glb"

    export_local(
        sources=_sources(
            fixture,
            sources_type,
            augmentation=lambda generation, revision: calls.append((generation, revision)),
        ),
        tile_revision_id=fixture.tile_metadata["revision_id"],
        appearance_revision_id=fixture.published_appearance.appearance_revision_id,
        lod_id=fixture.lod_id,
        rgbd_augmented_revision_id=None,
        output_path=output,
    )

    gltf = GLTF2().load_binary(output)
    assert calls == []
    assert all(not node.name.startswith("RGBD_AUGMENTED/") for node in gltf.nodes)
    assert gltf.asset.extras["source_layers"] == ["LIDAR_MEASURED"]


@pytest.mark.parametrize("failure", ("unknown-lod", "changed-tile", "wrong-augmentation"))
def test_local_tile_export_rejects_non_exact_sources_without_partial_outputs(tmp_path, failure):
    sources_type, export_local = _export_api()
    fixture = _local_fixture(tmp_path)
    output = tmp_path / "rejected.glb"
    tile_loader = fixture.tile_store.load
    lod_id = fixture.lod_id
    augmentation_id = fixture.augmentation.rgbd_augmented_revision_id
    if failure == "unknown-lod":
        lod_id = _hash("unknown-lod")
    elif failure == "changed-tile":
        original = tile_loader

        def changed_tile(revision):
            metadata, body = original(revision)
            return metadata, body + b"changed"

        tile_loader = changed_tile
    else:
        augmentation_id = _hash("wrong-augmentation")

    with pytest.raises((FileNotFoundError, ValueError), match="LOD|lod|content|augmentation|revision"):
        export_local(
            sources=_sources(fixture, sources_type, tile_loader=tile_loader),
            tile_revision_id=fixture.tile_metadata["revision_id"],
            appearance_revision_id=fixture.published_appearance.appearance_revision_id,
            lod_id=lod_id,
            rgbd_augmented_revision_id=augmentation_id,
            output_path=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".glb.provenance.json").exists()
    assert not list(tmp_path.rglob("*.partial"))
