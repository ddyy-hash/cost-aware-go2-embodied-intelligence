import hashlib
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import threading
import copy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from mapping.pc.go2_mapping_appearance import building_store as building_store_module
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
from mapping.pc.go2_mapping_appearance.tile_store import (
    PublishedTileAppearance,
    TileAppearanceStore,
    TileLodPayload,
)
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember, BuildingAssembly
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


def _hash(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
TRANSLATED = (
    1.0, 0.0, 0.0, 2.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _authorization(*, coverage=800_000, views=2):
    blockers = []
    if coverage < 800_000:
        blockers.append("MEASURED_SURFACE_COVERAGE_REQUIRED")
    if views < 2:
        blockers.append("INDEPENDENT_COLOR_VIEWS_REQUIRED")
    return DeliveryAuthorization(
        coverage,
        views,
        800_000,
        2,
        not blockers,
        tuple(blockers),
    )


def _placement(assembly_id, tile, member):
    payload = {
        "assembly_id": assembly_id,
        "source_revision_id": tile["revision_id"],
        "component_id": member.component_id,
        "submap_id": member.submap_id,
        "generation_id": member.generation_id,
        "transform_building_from_submap": list(member.transform_building_from_submap),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _fixture(tmp_path):
    root_member = AssemblyMember(
        "building-a", "component-root", "session-root", "submap-root",
        _hash("generation-root"), IDENTITY,
    )
    attached_member = AssemblyMember(
        "building-a", "component-attached", "session-attached", "submap-attached",
        _hash("generation-attached"), TRANSLATED, _hash("decision-attached"),
    )
    assembly = BuildingAssembly(
        _hash("assembly-a"),
        "building-a",
        "assembly-v1",
        (root_member, attached_member),
        tmp_path / "assembly-a",
    )
    tile_bodies = {
        _hash("tile-root"): b"tile-root-body",
        _hash("tile-attached"): b"tile-attached-body",
    }
    tiles = []
    appearances = {}
    lod_specs = []
    for order in range(2):
        artifact_bodies = {
            "render-vertices.bin": f"verified-render-vertices-{order}".encode("ascii"),
            "render-indices.bin": f"verified-render-indices-{order}".encode("ascii"),
            "render-uvs.bin": f"verified-render-uvs-{order}".encode("ascii"),
            "atlas-0.png": f"verified-atlas-{order}".encode("ascii"),
        }
        artifacts = [
            {
                "name": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
            for name, body in artifact_bodies.items()
        ]
        core = {
            "policy": {"order": order},
            "atlas_count": 1,
            "artifacts": artifacts,
        }
        lod_specs.append((canonical_sha256(core), core, artifact_bodies))
    lods = tuple(lod_id for lod_id, _, _ in lod_specs)
    appearance_ids = []
    for index, (assembly_member, tile_revision) in enumerate(
        zip(assembly.members, tile_bodies)
    ):
        body = tile_bodies[tile_revision]
        metadata = {
            "schema_version": 4,
            "revision_id": tile_revision,
            "geometry_revision_id": _hash(f"geometry-{index}"),
            "source_generation_id": assembly_member.generation_id,
            "pose_revision_id": _hash(f"pose-{index}"),
            "content_sha256": hashlib.sha256(body).hexdigest(),
            "building_id": assembly_member.building_id,
            "floor_id": "floor-1",
            "component_id": assembly_member.component_id,
            "submap_id": assembly_member.submap_id,
            "tile_id": [index, 0, 0],
            "transform_building_from_submap": list(
                assembly_member.transform_building_from_submap
            ),
            "attachment_state": "ROOT" if index == 0 else "ATTACHED",
        }
        metadata["placement_revision_id"] = _placement(
            assembly.assembly_id, metadata, assembly_member
        )
        tiles.append(metadata)
        authorization = _authorization()
        revision_core = {
            "tile_revision_id": tile_revision,
            "geometry_revision_id": metadata["geometry_revision_id"],
            "source_generation_id": assembly_member.generation_id,
            "calibration_id": "calibration-1",
            "calibration_sha256": _hash("calibration"),
            "physical_depth_validation_revision": _hash("depth-validation"),
            "pose_revision_id": metadata["pose_revision_id"],
            "model_revision_id": None,
            "assistance_config_sha256": None,
            "source_frame_sha256": [_hash("frame-a"), _hash("frame-b")],
            "appearance_config_sha256": _hash("appearance-config"),
            "charting": {"implementation": "xatlas", "version": "0.0.11"},
            "atlas_density_policy": {"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
            "padding_policy": {"pixels": 2},
            "baker_algorithm_version": "tile-atlas-baker-v1",
            "rgbd_augmented_revision_id": None,
            "delivery_evidence": {
                "measured_surface_coverage_ppm": authorization.measured_surface_coverage_ppm,
                "minimum_independent_view_count": authorization.minimum_independent_view_count,
                "required_coverage_ppm": authorization.required_coverage_ppm,
                "required_independent_view_count": authorization.required_independent_view_count,
            },
        }
        appearance_revision = canonical_sha256(revision_core)
        appearance_ids.append(appearance_revision)
        generation_root = (
            tmp_path / "verified-tile-appearance" / tile_revision / appearance_revision
        )
        lod_paths = {}
        lod_manifests = []
        for lod_id, lod_core, artifact_bodies in lod_specs:
            lod_root = generation_root / "lods" / lod_id
            lod_root.mkdir(parents=True, exist_ok=True)
            for name, artifact_body in artifact_bodies.items():
                (lod_root / name).write_bytes(artifact_body)
            lod_manifests.append({
                "lod_id": lod_id,
                "core": lod_core,
            })
            lod_paths[lod_id] = lod_root
        manifest = {
            "schema_version": 1,
            "appearance_revision_id": appearance_revision,
            "revision_core": revision_core,
            "delivery_authorization": {
                "measured_surface_coverage_ppm": authorization.measured_surface_coverage_ppm,
                "minimum_independent_view_count": authorization.minimum_independent_view_count,
                "required_coverage_ppm": authorization.required_coverage_ppm,
                "required_independent_view_count": authorization.required_independent_view_count,
                "delivery_authorized": authorization.delivery_authorized,
                "blockers": list(authorization.blockers),
            },
            "publication_authorization": {
                "building_delivery_eligible": True,
                "building_authorized": True,
                "export_authorized": True,
                "blockers": [],
            },
            "lods": lod_manifests,
        }
        generation_root.mkdir(parents=True, exist_ok=True)
        manifest_path = generation_root / "manifest.json"
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        pointer = {
            "schema_version": 1,
            "tile_revision_id": tile_revision,
            "source_generation_id": assembly_member.generation_id,
            "appearance_revision_id": appearance_revision,
            "manifest_sha256": manifest_sha,
            "lod_ids": list(lods),
            "delivery_authorization": manifest["delivery_authorization"],
            "publication_authorization": manifest["publication_authorization"],
        }
        appearances[(tile_revision, appearance_revision)] = PublishedTileAppearance(
            tile_revision,
            appearance_revision,
            manifest,
            manifest_sha,
            pointer,
            generation_root,
            manifest_path,
            lod_paths,
        )

    def tile_manifest_resolver(candidate):
        assert candidate.assembly_id == assembly.assembly_id
        return {
            "schema_version": 3,
            "scope": "building",
            "assembly_id": assembly.assembly_id,
            "tiles": [dict(value) for value in tiles],
        }

    def tile_resolver(revision_id):
        metadata = next(value for value in tiles if value["revision_id"] == revision_id)
        source_metadata = dict(metadata)
        source_metadata.pop("placement_revision_id", None)
        source_metadata.pop("transform_building_from_submap", None)
        return source_metadata, tile_bodies[revision_id]

    store = BuildingAppearanceStore(
        tmp_path / "building-appearance",
        assembly_resolver=lambda assembly_id: assembly if assembly_id == assembly.assembly_id else None,
        tile_manifest_resolver=tile_manifest_resolver,
        tile_resolver=tile_resolver,
        tile_appearance_resolver=lambda tile, appearance: appearances[(tile, appearance)],
    )
    return store, assembly, tiles, appearances, lods, tuple(appearance_ids)


def _building_member(assembly, tile, appearance_id, lods, *, authorization=None, policy=None):
    assembly_member = next(
        value for value in assembly.members if value.component_id == tile["component_id"]
    )
    authorization = _authorization() if authorization is None else authorization
    policy = (
        DistanceLodInterval(0.0, 3.0, lods[0]),
        DistanceLodInterval(3.0, None, lods[1]),
    ) if policy is None else policy
    return BuildingAppearanceMember(
        building_id=assembly.building_id,
        component_id=assembly_member.component_id,
        session_id=assembly_member.session_id,
        submap_id=assembly_member.submap_id,
        source_generation_id=assembly_member.generation_id,
        placement_revision_id=tile["placement_revision_id"],
        tile_revision_id=tile["revision_id"],
        tile_appearance_revision_id=appearance_id,
        authorized_lod_ids=lods,
        distance_policy=policy,
        transform_building_from_submap=assembly_member.transform_building_from_submap,
        delivery_authorization=authorization,
        rgbd_augmented_revision_id=None,
    )


def _real_tile_generation(tile, index):
    revision = tile_appearance_revision(
        tile_revision_id=tile["revision_id"],
        geometry_revision_id=tile["geometry_revision_id"],
        source_generation_id=tile["source_generation_id"],
        calibration_id=f"calibration-{index}",
        calibration_sha256=_hash(f"calibration-{index}"),
        physical_depth_validation_revision=_hash(f"depth-validation-{index}"),
        pose_revision_id=tile["pose_revision_id"],
        model_revision_id=None,
        assistance_config_sha256=None,
        source_frame_sha256=(_hash(f"frame-{index}-a"), _hash(f"frame-{index}-b")),
        appearance_config_sha256=_hash(f"appearance-config-{index}"),
        charting={"implementation": "xatlas", "version": "0.0.11", "options": {}},
        atlas_density_policy={"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        padding_policy={"pixels": 8, "mip_levels": 4},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=None,
        contains_augmented_primitives=False,
        measured_surface_coverage_ppm=800_000,
        minimum_independent_view_count=2,
    )
    positions = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
    normals = np.asarray([[0, 0, 1]] * 3, dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.uint32)
    source_ids = np.asarray([0, 1, 2], dtype=np.uint32)
    source_triangles = np.asarray([0], dtype=np.uint32)
    uvs = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    atlas_indices = np.zeros(3, dtype=np.uint16)
    flags = np.ones(3, dtype=np.uint16)
    image = Image.new("RGBA", (8, 8), (20 + index, 40, 60, 255))
    png = BytesIO()
    image.save(png, format="PNG", optimize=False, compress_level=9)
    artifacts = {
        "render-vertices.bin": encode_render_vertices_v1(
            tile["revision_id"], revision.appearance_revision_id, positions, normals, source_ids
        ),
        "render-indices.bin": encode_render_indices_v1(
            tile["revision_id"], revision.appearance_revision_id, triangles, source_triangles
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            tile["revision_id"], revision.appearance_revision_id, uvs, atlas_indices, flags
        ),
        "atlas-0.png": png.getvalue(),
    }
    payloads = []
    for order, name in enumerate(("near", "far")):
        references = tuple(
            ArtifactRef(key, hashlib.sha256(body).hexdigest(), len(body))
            for key, body in artifacts.items()
        )
        lod = build_lod_record(
            tile_revision_id=tile["revision_id"],
            appearance_revision_id=revision.appearance_revision_id,
            policy={"name": name, "order": order, "mm_per_texel": 3 + 2 * order},
            vertex_count=3,
            triangle_count=1,
            atlas_count=1,
            artifacts=references,
            primitives=(PrimitiveRecord(0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,)),),
            rgbd_augmented_revision_id=None,
        )
        payloads.append(TileLodPayload(lod, artifacts))
    return revision, tuple(payloads)


def test_building_store_sorts_freezes_reuses_and_reloads_complete_generation(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    first = store.publish(assembly=assembly, members=reversed(members))
    repeated = store.publish(assembly=assembly, members=members)
    restarted = BuildingAppearanceStore(
        store.root.parent,
        assembly_resolver=store.assembly_resolver,
        tile_manifest_resolver=store.tile_manifest_resolver,
        tile_resolver=store.tile_resolver,
        tile_appearance_resolver=store.tile_appearance_resolver,
    ).current(expected_assembly_id=assembly.assembly_id)

    assert first == repeated == restarted
    assert len(first.building_appearance_revision_id) == 64
    assert "manifest_sha256" not in first.manifest
    assert first.manifest_sha256 == hashlib.sha256(first.manifest_path.read_bytes()).hexdigest()
    assert first.pointer["assembly_id"] == assembly.assembly_id
    assert [value["component_id"] for value in first.manifest["members"]] == [
        "component-attached", "component-root"
    ]
    assert not list(store.root.rglob("*.partial"))


@pytest.mark.parametrize(
    "member_change",
    (
        lambda member, lods: {**member.__dict__, "delivery_authorization": _authorization(coverage=799_999)},
        lambda member, lods: {**member.__dict__, "delivery_authorization": _authorization(views=1)},
        lambda member, lods: {**member.__dict__, "authorized_lod_ids": (lods[0],)},
        lambda member, lods: {
            **member.__dict__,
            "distance_policy": (
                DistanceLodInterval(0.0, 2.0, lods[0]),
                DistanceLodInterval(3.0, None, lods[1]),
            ),
        },
        lambda member, lods: {
            **member.__dict__,
            "distance_policy": (
                DistanceLodInterval(0.0, 3.0, lods[0]),
                DistanceLodInterval(3.0, 9.0, lods[1]),
            ),
        },
    ),
)
def test_blocked_member_or_invalid_lod_policy_never_creates_current(tmp_path, member_change):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    valid = _building_member(assembly, tiles[0], appearance_ids[0], lods)
    other = _building_member(assembly, tiles[1], appearance_ids[1], lods)

    with pytest.raises(ValueError):
        changed = BuildingAppearanceMember(**member_change(valid, lods))
        store.publish(assembly=assembly, members=(changed, other))
    assert store.current() is None


def test_store_recomputes_tile_authorization_instead_of_trusting_member(tmp_path):
    store, assembly, tiles, appearances, lods, appearance_ids = _fixture(tmp_path)
    published = appearances[(tiles[0]["revision_id"], appearance_ids[0])]
    published.manifest["revision_core"]["measured_surface_coverage_ppm"] = 799_999
    published.manifest["delivery_authorization"] = {
        "measured_surface_coverage_ppm": 799_999,
        "minimum_independent_view_count": 2,
        "required_coverage_ppm": 800_000,
        "required_independent_view_count": 2,
        "delivery_authorized": False,
        "blockers": ["MEASURED_SURFACE_COVERAGE_REQUIRED"],
    }
    published.manifest["publication_authorization"] = {
        "building_delivery_eligible": True,
        "building_authorized": False,
        "export_authorized": False,
        "blockers": ["MEASURED_SURFACE_COVERAGE_REQUIRED"],
    }
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    with pytest.raises(ValueError, match="hash|authorization|unauthorized"):
        store.publish(assembly=assembly, members=members)
    assert store.current() is None


def test_store_rejects_omitted_placement_augmentation_and_source_mismatch(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    first = _building_member(assembly, tiles[0], appearance_ids[0], lods)
    second = _building_member(assembly, tiles[1], appearance_ids[1], lods)

    with pytest.raises(ValueError, match="complete|visible|member"):
        store.publish(assembly=assembly, members=(first,))
    with pytest.raises(ValueError, match="placement"):
        store.publish(
            assembly=assembly,
            members=(BuildingAppearanceMember(**{**first.__dict__, "placement_revision_id": _hash("wrong")}), second),
        )
    with pytest.raises(ValueError, match="augmentation"):
        store.publish(
            assembly=assembly,
            members=(BuildingAppearanceMember(**{**first.__dict__, "rgbd_augmented_revision_id": _hash("augmentation")}), second),
        )
    with pytest.raises(ValueError, match="generation"):
        store.publish(
            assembly=assembly,
            members=(BuildingAppearanceMember(**{**first.__dict__, "source_generation_id": _hash("wrong-generation")}), second),
        )


def test_manifest_and_current_tampering_fail_closed(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    published = store.publish(
        assembly=assembly,
        members=tuple(
            _building_member(assembly, tile, appearance, lods)
            for tile, appearance in zip(tiles, appearance_ids)
        ),
    )
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    manifest["members"][0]["placement_revision_id"] = _hash("tampered")
    published.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|revision|placement"):
        store.resolve(published.building_appearance_revision_id)

    store.current_path.write_text("not-a-sha", encoding="ascii")
    with pytest.raises(ValueError, match="CURRENT"):
        store.current()


@pytest.mark.parametrize("point", ("after_files_fsynced", "after_generation_rename", "after_pointer_replace"))
def test_crash_points_preserve_only_complete_recoverable_state(tmp_path, point):
    base, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    def crash(actual):
        if actual == point:
            raise RuntimeError(point)

    store = BuildingAppearanceStore(
        base.root.parent,
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=base.tile_appearance_resolver,
        fault_injector=crash,
    )
    with pytest.raises(RuntimeError, match=point):
        store.publish(assembly=assembly, members=members)

    restarted = BuildingAppearanceStore(
        base.root.parent,
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=base.tile_appearance_resolver,
    )
    if point == "after_files_fsynced":
        assert restarted.current() is None
    elif point == "after_generation_rename":
        revisions = list(restarted.revisions_root.iterdir())
        assert len(revisions) == 1
        resolved = restarted.resolve(revisions[0].name)
        assert restarted.current() is None
        assert restarted.publish(assembly=assembly, members=members) == resolved
        assert restarted.current(expected_assembly_id=assembly.assembly_id) == resolved
    else:
        assert restarted.current(expected_assembly_id=assembly.assembly_id) is not None
    assert not list(base.root.rglob("*.partial"))


@pytest.mark.skipif(os.name == "nt", reason="the owned WSL runtime provides POSIX fork semantics")
@pytest.mark.parametrize("point", ("after_files_fsynced", "after_generation_rename", "after_pointer_replace"))
def test_hard_process_exit_points_recover_only_complete_state(tmp_path, point):
    base, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    context = multiprocessing.get_context("fork")

    def publish_and_exit():
        def exit_at(actual):
            if actual == point:
                os._exit(91)

        store = BuildingAppearanceStore(
            base.root.parent,
            assembly_resolver=base.assembly_resolver,
            tile_manifest_resolver=base.tile_manifest_resolver,
            tile_resolver=base.tile_resolver,
            tile_appearance_resolver=base.tile_appearance_resolver,
            fault_injector=exit_at,
        )
        store.publish(assembly=assembly, members=members)

    process = context.Process(target=publish_and_exit)
    process.start()
    process.join(5.0)
    assert not process.is_alive()
    assert process.exitcode == 91

    restarted = BuildingAppearanceStore(
        base.root.parent,
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=base.tile_appearance_resolver,
    )
    if point == "after_files_fsynced":
        assert restarted.current() is None
        assert list(restarted.revisions_root.iterdir()) == []
    elif point == "after_generation_rename":
        revisions = list(restarted.revisions_root.iterdir())
        assert len(revisions) == 1
        resolved = restarted.resolve(revisions[0].name)
        assert restarted.current() is None
        assert restarted.publish(assembly=assembly, members=members) == resolved
    else:
        assert restarted.current(expected_assembly_id=assembly.assembly_id) is not None
    assert not list(restarted.staging_root.glob("*.partial"))


def test_concurrent_complete_revisions_remain_valid_with_one_current_pointer(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    second_store = BuildingAppearanceStore(
        store.root.parent,
        assembly_resolver=store.assembly_resolver,
        tile_manifest_resolver=store.tile_manifest_resolver,
        tile_resolver=store.tile_resolver,
        tile_appearance_resolver=store.tile_appearance_resolver,
    )
    first = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    second = tuple(
        _building_member(
            assembly,
            tile,
            appearance,
            lods,
            policy=(
                DistanceLodInterval(0.0, 5.0, lods[0]),
                DistanceLodInterval(5.0, None, lods[1]),
            ),
        )
        for tile, appearance in zip(tiles, appearance_ids)
    )
    results = []
    errors = []

    def publish(candidate_store, members):
        try:
            results.append(candidate_store.publish(assembly=assembly, members=members))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=publish, args=(candidate_store, members))
        for candidate_store, members in ((store, first), (second_store, second))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len({value.building_appearance_revision_id for value in results}) == 2
    for value in results:
        assert store.resolve(value.building_appearance_revision_id) == value
    assert store.current() in results


def test_second_store_constructor_cannot_clean_active_staging_generation(tmp_path):
    base, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    writer_entered = threading.Event()
    release_writer = threading.Event()
    constructor_finished = threading.Event()
    results = []
    errors = []

    def pause_writer(point):
        if point == "after_files_fsynced":
            writer_entered.set()
            assert release_writer.wait(5.0)

    writer = BuildingAppearanceStore(
        base.root.parent,
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=base.tile_appearance_resolver,
        fault_injector=pause_writer,
    )

    def publish():
        try:
            results.append(writer.publish(assembly=assembly, members=members))
        except Exception as exc:
            errors.append(exc)

    def construct_second_store():
        BuildingAppearanceStore(
            base.root.parent,
            assembly_resolver=base.assembly_resolver,
            tile_manifest_resolver=base.tile_manifest_resolver,
            tile_resolver=base.tile_resolver,
            tile_appearance_resolver=base.tile_appearance_resolver,
        )
        constructor_finished.set()

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert writer_entered.wait(5.0)
    constructor = threading.Thread(target=construct_second_store)
    constructor.start()

    assert not constructor_finished.wait(0.2)
    assert len(list(base.staging_root.glob("*.partial"))) == 1
    release_writer.set()
    publisher.join(5.0)
    constructor.join(5.0)

    assert not publisher.is_alive()
    assert not constructor.is_alive()
    assert errors == []
    assert len(results) == 1
    assert constructor_finished.is_set()
    assert base.current(expected_assembly_id=assembly.assembly_id) == results[0]


def test_independent_stores_concurrently_publish_same_revision_once(tmp_path):
    first_store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    results = []
    errors = []

    def pause_first(point):
        if point == "after_files_fsynced":
            first_entered.set()
            assert release_first.wait(5.0)

    first_store._fault_injector = pause_first
    second_store = BuildingAppearanceStore(
        first_store.root.parent,
        assembly_resolver=first_store.assembly_resolver,
        tile_manifest_resolver=first_store.tile_manifest_resolver,
        tile_resolver=first_store.tile_resolver,
        tile_appearance_resolver=first_store.tile_appearance_resolver,
    )

    def publish(store, finished=None):
        try:
            results.append(store.publish(assembly=assembly, members=members))
        except Exception as exc:
            errors.append(exc)
        finally:
            if finished is not None:
                finished.set()

    first_thread = threading.Thread(target=publish, args=(first_store,))
    second_thread = threading.Thread(target=publish, args=(second_store, second_finished))
    first_thread.start()
    assert first_entered.wait(5.0)
    second_thread.start()
    assert not second_finished.wait(0.2)
    release_first.set()
    first_thread.join(5.0)
    second_thread.join(5.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(list(first_store.revisions_root.iterdir())) == 1
    assert first_store.current(expected_assembly_id=assembly.assembly_id) == results[0]


@pytest.mark.skipif(os.name == "nt", reason="the owned WSL runtime provides POSIX fork and flock")
def test_independent_processes_publish_same_revision_under_file_lock(tmp_path):
    base, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    context = multiprocessing.get_context("fork")
    first_entered = context.Event()
    release_first = context.Event()
    results = context.Queue()

    def publish_first():
        def pause(point):
            if point == "after_files_fsynced":
                first_entered.set()
                if not release_first.wait(5.0):
                    raise RuntimeError("timed out waiting to release first publisher")

        store = BuildingAppearanceStore(
            base.root.parent,
            assembly_resolver=base.assembly_resolver,
            tile_manifest_resolver=base.tile_manifest_resolver,
            tile_resolver=base.tile_resolver,
            tile_appearance_resolver=base.tile_appearance_resolver,
            fault_injector=pause,
        )
        try:
            published = store.publish(assembly=assembly, members=members)
            results.put(("ok", published.building_appearance_revision_id))
        except Exception as exc:
            results.put(("error", repr(exc)))

    def publish_second():
        store = BuildingAppearanceStore(
            base.root.parent,
            assembly_resolver=base.assembly_resolver,
            tile_manifest_resolver=base.tile_manifest_resolver,
            tile_resolver=base.tile_resolver,
            tile_appearance_resolver=base.tile_appearance_resolver,
        )
        try:
            published = store.publish(assembly=assembly, members=members)
            results.put(("ok", published.building_appearance_revision_id))
        except Exception as exc:
            results.put(("error", repr(exc)))

    first = context.Process(target=publish_first)
    second = context.Process(target=publish_second)
    first.start()
    assert first_entered.wait(5.0)
    second.start()
    second.join(0.2)
    assert second.is_alive()
    release_first.set()
    first.join(5.0)
    second.join(5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    outcomes = [results.get(timeout=2.0), results.get(timeout=2.0)]
    assert [status for status, _ in outcomes] == ["ok", "ok"]
    assert len({revision for _, revision in outcomes}) == 1
    assert len(list(base.revisions_root.iterdir())) == 1
    assert base.current(expected_assembly_id=assembly.assembly_id).building_appearance_revision_id == outcomes[0][1]


@pytest.mark.skipif(os.name == "nt", reason="the owned WSL runtime provides POSIX fork and flock")
def test_independent_processes_can_construct_new_store_concurrently(tmp_path):
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    root = tmp_path / "fresh-concurrent-store"

    def construct():
        start.wait(5.0)
        try:
            BuildingAppearanceStore(
                root,
                assembly_resolver=lambda _assembly_id: None,
                tile_manifest_resolver=lambda _assembly: {},
                tile_resolver=lambda _tile_id: ({}, b""),
                tile_appearance_resolver=lambda _tile_id, _appearance_id: None,
            )
            results.put(("ok", None))
        except Exception as exc:
            results.put(("error", repr(exc)))

    processes = [context.Process(target=construct) for _ in range(8)]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(5.0)

    assert all(not process.is_alive() for process in processes)
    outcomes = [results.get(timeout=2.0) for _ in processes]
    assert outcomes == [("ok", None)] * len(processes)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_constructor_rejects_symlinked_parent_component(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe|symlink|reparse"):
        BuildingAppearanceStore(
            linked_parent,
            assembly_resolver=lambda _assembly_id: None,
            tile_manifest_resolver=lambda _assembly: {},
            tile_resolver=lambda _tile_id: ({}, b""),
            tile_appearance_resolver=lambda _tile_id, _appearance_id: None,
        )

    assert not (real_parent / "building-appearance").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_constructor_rejects_symlinked_staging_without_deleting_external_data(tmp_path):
    store, _, _, _, _, _ = _fixture(tmp_path)
    external = tmp_path / "external-staging"
    partial = external / f"{_hash('unsafe-partial')[:32]}.partial"
    partial.mkdir(parents=True)
    sentinel = partial / "keep.txt"
    sentinel.write_text("keep", encoding="ascii")
    store.staging_root.rmdir()
    store.staging_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe|symlink|reparse"):
        BuildingAppearanceStore(
            store.root.parent,
            assembly_resolver=store.assembly_resolver,
            tile_manifest_resolver=store.tile_manifest_resolver,
            tile_resolver=store.tile_resolver,
            tile_appearance_resolver=store.tile_appearance_resolver,
        )

    assert sentinel.read_text(encoding="ascii") == "keep"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_constructor_rejects_symlinked_partial_without_deleting_external_data(tmp_path):
    store, _, _, _, _, _ = _fixture(tmp_path)
    external = tmp_path / "external-partial"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("keep", encoding="ascii")
    partial = store.staging_root / f"{_hash('unsafe-partial-entry')[:32]}.partial"
    partial.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe|symlink|reparse"):
        BuildingAppearanceStore(
            store.root.parent,
            assembly_resolver=store.assembly_resolver,
            tile_manifest_resolver=store.tile_manifest_resolver,
            tile_resolver=store.tile_resolver,
            tile_appearance_resolver=store.tile_appearance_resolver,
        )

    assert sentinel.read_text(encoding="ascii") == "keep"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_resolve_rejects_revision_directory_replaced_by_symlink(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    published = store.publish(assembly=assembly, members=members)
    external = tmp_path / "external-revision"
    published.root_path.rename(external)
    published.root_path.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe|symlink|reparse"):
        store.resolve(published.building_appearance_revision_id)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_current_rejects_pointer_replaced_by_symlink(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    store.publish(assembly=assembly, members=members)
    external = tmp_path / "external-current.json"
    store.current_path.rename(external)
    store.current_path.symlink_to(external)

    with pytest.raises(ValueError, match="unsafe|symlink|reparse|CURRENT"):
        store.current(expected_assembly_id=assembly.assembly_id)


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow file creation is exercised in WSL")
def test_pointer_temporary_symlink_is_rejected_without_overwriting_target(tmp_path, monkeypatch):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    generated = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(
        building_store_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(generated)),
    )
    external = tmp_path / "external-pointer-target.json"
    external.write_text("keep", encoding="ascii")
    temporary = store.current_path.with_name(f".{store.current_path.name}.{'b' * 32}.tmp")
    temporary.symlink_to(external)

    with pytest.raises((ValueError, FileExistsError, OSError)):
        store.publish(assembly=assembly, members=members)

    assert external.read_text(encoding="ascii") == "keep"


def test_windows_lock_helper_retries_past_msvcrt_default_limit():
    attempts = []

    class FakeMsvcrt:
        LK_NBLCK = 1

        @staticmethod
        def locking(_descriptor, mode, length):
            attempts.append((mode, length))
            if len(attempts) <= 12:
                error = OSError("locked")
                error.errno = 13
                raise error

    building_store_module._lock_windows_file(
        SimpleNamespace(fileno=lambda: 7, seek=lambda _offset: None),
        FakeMsvcrt,
        sleep=lambda _seconds: None,
    )

    assert len(attempts) == 13


def test_windows_lock_helper_times_out_with_diagnostic_error():
    class AlwaysLockedMsvcrt:
        LK_NBLCK = 1

        @staticmethod
        def locking(_descriptor, _mode, _length):
            error = OSError("locked")
            error.errno = 13
            raise error

    times = iter((0.0, 1.0))
    with pytest.raises(TimeoutError, match="Building appearance store lock"):
        building_store_module._lock_windows_file(
            SimpleNamespace(fileno=lambda: 7, seek=lambda _offset: None),
            AlwaysLockedMsvcrt,
            sleep=lambda _seconds: None,
            clock=lambda: next(times),
            timeout_seconds=0.5,
        )


def test_store_file_lock_rethrows_acquisition_failure(tmp_path):
    lock = building_store_module._StoreFileLock(tmp_path / "missing-parent" / ".store.lock")

    with pytest.raises(FileNotFoundError):
        with lock:
            pytest.fail("lock body must not run when acquisition fails")


def test_real_tile_store_artifact_corruption_blocks_publish_and_restart_resolve(tmp_path):
    base, assembly, tiles, _, _, _ = _fixture(tmp_path)
    tile_store = TileAppearanceStore(tmp_path / "real-tile-store")
    tile_appearances = []
    members = []
    for index, tile in enumerate(tiles):
        revision, payloads = _real_tile_generation(tile, index)
        published = tile_store.publish(revision, payloads)
        tile_appearances.append((published, payloads))
        members.append(
            _building_member(
                assembly,
                tile,
                published.appearance_revision_id,
                tuple(payload.record.lod_id for payload in payloads),
            )
        )

    store = BuildingAppearanceStore(
        tmp_path / "real-building-store",
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=tile_store.resolve,
    )
    building = store.publish(assembly=assembly, members=tuple(members))
    first_appearance, first_payloads = tile_appearances[0]
    artifact = (
        first_appearance.lod_paths[first_payloads[0].record.lod_id]
        / "render-vertices.bin"
    )
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="artifact|hash|verification"):
        store.publish(assembly=assembly, members=tuple(members))

    restarted = BuildingAppearanceStore(
        store.root.parent,
        assembly_resolver=base.assembly_resolver,
        tile_manifest_resolver=base.tile_manifest_resolver,
        tile_resolver=base.tile_resolver,
        tile_appearance_resolver=TileAppearanceStore(tmp_path / "real-tile-store").resolve,
    )
    with pytest.raises(ValueError, match="artifact|hash|verification"):
        restarted.resolve(building.building_appearance_revision_id)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior is exercised in the owned WSL runtime")
def test_tile_appearance_artifact_symlink_is_rejected_even_when_bytes_match(tmp_path):
    store, assembly, tiles, appearances, lods, appearance_ids = _fixture(tmp_path)
    published = appearances[(tiles[0]["revision_id"], appearance_ids[0])]
    artifact = published.lod_paths[lods[0]] / "render-vertices.bin"
    external = tmp_path / "external-render-vertices.bin"
    artifact.rename(external)
    artifact.symlink_to(external)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    with pytest.raises(ValueError, match="unsafe|symlink|reparse|path"):
        store.publish(assembly=assembly, members=members)


def test_terminal_source_revalidation_preserves_previous_complete_current(tmp_path, monkeypatch):
    store, assembly, tiles, appearances, lods, appearance_ids = _fixture(tmp_path)
    first_members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    first = store.publish(assembly=assembly, members=first_members)
    alternate_ids = []
    for tile, original_id in zip(tiles, appearance_ids):
        original = appearances[(tile["revision_id"], original_id)]
        alternate_manifest = copy.deepcopy(original.manifest)
        alternate_manifest["revision_core"]["baker_algorithm_version"] = "tile-atlas-baker-v2"
        alternate_id = canonical_sha256(alternate_manifest["revision_core"])
        alternate_manifest["appearance_revision_id"] = alternate_id
        alternate_root = tmp_path / "alternate-tile-appearance" / tile["revision_id"] / alternate_id
        alternate_lod_paths = {}
        for lod in alternate_manifest["lods"]:
            lod_id = lod["lod_id"]
            lod_root = alternate_root / "lods" / lod_id
            lod_root.mkdir(parents=True, exist_ok=True)
            for artifact in lod["core"]["artifacts"]:
                source = original.lod_paths[lod_id] / artifact["name"]
                (lod_root / artifact["name"]).write_bytes(source.read_bytes())
            alternate_lod_paths[lod_id] = lod_root
        alternate_root.mkdir(parents=True, exist_ok=True)
        alternate_manifest_path = alternate_root / "manifest.json"
        alternate_manifest_bytes = json.dumps(
            alternate_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        alternate_manifest_path.write_bytes(alternate_manifest_bytes)
        alternate_sha = hashlib.sha256(alternate_manifest_bytes).hexdigest()
        alternate = PublishedTileAppearance(
            tile["revision_id"],
            alternate_id,
            alternate_manifest,
            alternate_sha,
            {**original.pointer, "appearance_revision_id": alternate_id, "manifest_sha256": alternate_sha},
            alternate_root,
            alternate_manifest_path,
            alternate_lod_paths,
        )
        alternate_ids.append(alternate_id)
        appearances[(tile["revision_id"], alternate_id)] = alternate
    alternate_ids = tuple(alternate_ids)
    second_members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, alternate_ids)
    )

    def invalidate_after_rename(point):
        return None

    racing = BuildingAppearanceStore(
        store.root.parent,
        assembly_resolver=store.assembly_resolver,
        tile_manifest_resolver=store.tile_manifest_resolver,
        tile_resolver=store.tile_resolver,
        tile_appearance_resolver=store.tile_appearance_resolver,
        fault_injector=invalidate_after_rename,
    )

    original_replace = building_store_module.os.replace
    replaced = False

    def replace_and_invalidate(source, destination, *args, **kwargs):
        nonlocal replaced
        if Path(destination).name == racing.current_path.name and not replaced:
            alternate = appearances[(tiles[0]["revision_id"], alternate_ids[0])]
            alternate.manifest["publication_authorization"]["building_authorized"] = False
            alternate.manifest["publication_authorization"]["export_authorized"] = False
            manifest_bytes = json.dumps(
                alternate.manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            alternate.manifest_path.write_bytes(manifest_bytes)
            object.__setattr__(
                alternate,
                "manifest_sha256",
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            alternate.pointer["manifest_sha256"] = alternate.manifest_sha256
            replaced = True
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(building_store_module.os, "replace", replace_and_invalidate)

    with pytest.raises(ValueError, match="unauthorized"):
        racing.publish(assembly=assembly, members=second_members)
    assert store.current(expected_assembly_id=assembly.assembly_id) == first


def test_top_level_building_identity_must_match_authoritative_assembly_members(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    mixed = BuildingAssembly(
        assembly.assembly_id,
        "building-b",
        assembly.algorithm_version,
        assembly.members,
        assembly.path,
    )
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    with pytest.raises(ValueError, match="[Bb]uilding|assembly|mixed"):
        store.publish(assembly=mixed, members=members)
    assert store.current() is None
    assert list(store.revisions_root.iterdir()) == []


def test_member_transform_must_match_assembly_member_not_only_tile_manifest(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    different_transform = (
        1.0, 0.0, 0.0, 9.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    tiles[0]["transform_building_from_submap"] = list(different_transform)
    first = _building_member(assembly, tiles[0], appearance_ids[0], lods)
    first = BuildingAppearanceMember(
        **{**first.__dict__, "transform_building_from_submap": different_transform}
    )
    second = _building_member(assembly, tiles[1], appearance_ids[1], lods)

    with pytest.raises(ValueError, match="assembly|transform"):
        store.publish(assembly=assembly, members=(first, second))


def test_direct_source_tile_metadata_must_match_frozen_member_scope(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    original = store.tile_resolver

    def mismatched_source(revision_id):
        metadata, body = original(revision_id)
        if revision_id == tiles[0]["revision_id"]:
            metadata["building_id"] = "building-other"
            metadata["component_id"] = "component-other"
            metadata["submap_id"] = "submap-other"
            metadata["placement_revision_id"] = _hash("wrong-placement")
            metadata["transform_building_from_submap"] = list(TRANSLATED)
        return metadata, body

    store.tile_resolver = mismatched_source
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )

    with pytest.raises(ValueError, match="source tile|provenance|building"):
        store.publish(assembly=assembly, members=members)


def test_structured_current_pointer_corruption_is_rejected(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    store.publish(
        assembly=assembly,
        members=tuple(
            _building_member(assembly, tile, appearance, lods)
            for tile, appearance in zip(tiles, appearance_ids)
        ),
    )
    pointer = json.loads(store.current_path.read_text(encoding="utf-8"))
    pointer["building_id"] = "building-other"
    pointer["unexpected"] = True
    store.current_path.write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CURRENT"):
        store.current()


def test_duplicate_mixed_building_and_reordered_lods_are_rejected(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    first = _building_member(assembly, tiles[0], appearance_ids[0], lods)
    second = _building_member(assembly, tiles[1], appearance_ids[1], lods)
    with pytest.raises(ValueError, match="duplicate|complete"):
        store.publish(assembly=assembly, members=(first, first, second))

    mixed = BuildingAppearanceMember(**{**first.__dict__, "building_id": "building-other"})
    with pytest.raises(ValueError, match="building|assembly"):
        store.publish(assembly=assembly, members=(mixed, second))

    reordered = BuildingAppearanceMember(
        **{
            **first.__dict__,
            "authorized_lod_ids": tuple(reversed(lods)),
            "distance_policy": (
                DistanceLodInterval(0.0, 3.0, lods[1]),
                DistanceLodInterval(3.0, None, lods[0]),
            ),
        }
    )
    with pytest.raises(ValueError, match="LOD"):
        store.publish(assembly=assembly, members=(reordered, second))


def test_existing_conflict_and_corrupt_appearance_resolver_fail_closed(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    members = tuple(
        _building_member(assembly, tile, appearance, lods)
        for tile, appearance in zip(tiles, appearance_ids)
    )
    published = store.publish(assembly=assembly, members=members)
    published.manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|identity|hash"):
        store.publish(assembly=assembly, members=members)

    clean, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path / "corrupt")
    original = clean.tile_appearance_resolver

    def corrupt_artifact(tile, appearance):
        if tile == tiles[0]["revision_id"]:
            raise ValueError("appearance artifact hash verification failed")
        return original(tile, appearance)

    clean.tile_appearance_resolver = corrupt_artifact
    with pytest.raises(ValueError, match="artifact"):
        clean.publish(
            assembly=assembly,
            members=tuple(
                _building_member(assembly, tile, appearance, lods)
                for tile, appearance in zip(tiles, appearance_ids)
            ),
        )
    assert clean.current() is None


def test_assembly_replacement_updates_current_and_stales_expected_old_assembly(tmp_path):
    store, assembly, tiles, _, lods, appearance_ids = _fixture(tmp_path)
    first = store.publish(
        assembly=assembly,
        members=tuple(
            _building_member(assembly, tile, appearance, lods)
            for tile, appearance in zip(tiles, appearance_ids)
        ),
    )
    replacement = BuildingAssembly(
        _hash("assembly-b"),
        assembly.building_id,
        "assembly-v2",
        assembly.members,
        Path(tmp_path) / "assembly-b",
    )
    assemblies = {assembly.assembly_id: assembly, replacement.assembly_id: replacement}
    old_manifest_resolver = store.tile_manifest_resolver

    def replacement_manifest(candidate):
        manifest = old_manifest_resolver(assembly)
        manifest["assembly_id"] = candidate.assembly_id
        for tile, member in zip(manifest["tiles"], candidate.members):
            tile["placement_revision_id"] = _placement(candidate.assembly_id, tile, member)
        return manifest

    store.assembly_resolver = assemblies.get
    store.tile_manifest_resolver = replacement_manifest
    replacement_tiles = replacement_manifest(replacement)["tiles"]
    second = store.publish(
        assembly=replacement,
        members=tuple(
            _building_member(replacement, tile, appearance, lods)
            for tile, appearance in zip(replacement_tiles, appearance_ids)
        ),
    )

    assert second.building_appearance_revision_id != first.building_appearance_revision_id
    assert store.current(expected_assembly_id=replacement.assembly_id) == second
    with pytest.raises(ValueError, match="stale"):
        store.current(expected_assembly_id=assembly.assembly_id)
