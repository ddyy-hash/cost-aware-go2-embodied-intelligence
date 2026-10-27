import hashlib
import json
import sqlite3
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.baker import TileAtlasBaker
from mapping.pc.go2_mapping_appearance.historical_store import HistoricalAppearanceStore
from mapping.pc.go2_mapping_appearance.tile_evidence import (
    TileAppearanceScope,
    ValidatedTileAppearanceProcessor,
    build_validated_tile_bake_request,
)
from mapping.pc.go2_mapping_appearance.tile_store import TileAppearanceStore
from mapping.pc.go2_mapping_appearance.tile_vertex_color import TileVertexColorStore
from mapping.pc.go2_mapping_appearance.vertex_color import build_historical_vertex_color_sidecar
from mapping.pc.go2_mapping_live.live_pipeline import LiveSurfacePipeline
from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision
from mapping.pc.go2_mapping_live.live_tiles import ImmutableTileStore
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal
from mapping.pc.go2_mapping_rgbd.artifact_refinement import AppearanceEvidence
from mapping.pc.go2_mapping_rgbd import ArtifactRgbdRefiner
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_rgbd.texture import VertexColor
from mapping.tests.test_rgbd_artifact_refinement import _bag, _lidar_frame, _manifest
from mapping.tests.test_live_surface_pipeline import _frame as _live_frame
from mapping.pc.go2_mapping_live.live_surface_mesher import LiveSurfaceMesher


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


def _tile():
    positions = np.asarray([[0, 0, 1], [0.015, 0, 1], [0, 0.015, 1]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.004], dtype="<f4")
    body = b"".join((struct.pack("<4sIIII", b"G2TM", 1, 3, 1, 1), positions.tobytes(), normals.tobytes(), triangles.tobytes(), scales.tobytes()))
    metadata = {
        "revision_id": HASHES[0],
        "geometry_revision_id": HASHES[1],
        "source_generation_id": HASHES[1],
        "pose_revision_id": HASHES[2],
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "vertex_count": 3,
        "triangle_count": 1,
        "building_id": "building-a",
        "floor_id": "floor-1",
        "component_id": "component-a",
        "submap_id": "submap-a",
        "attachment_state": "ROOT",
        "tile_id": [0, 0, 0],
    }
    return metadata, body


def _mesh_revision():
    positions = np.asarray([[0, 0, 1], [0.015, 0, 1], [0, 0.015, 1]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.004], dtype="<f4")
    return MeshTileRevision(
        TileId(0, 0, 0), HASHES[1], positions, normals, triangles, scales, 0
    )


def _published_scope(metadata, *, publication_scope):
    return TileAppearanceScope(
        building_id=metadata["building_id"],
        floor_id=metadata["floor_id"],
        component_id=metadata["component_id"],
        submap_id=metadata["submap_id"],
        attachment_state=metadata["attachment_state"],
        publication_scope=publication_scope,
    )


def _evidence(*, generation=HASHES[2], calibration=HASHES[3], views=2, accepted=(True, True, True)):
    ids = ("0:0:66", "1:0:66", "0:1:66")
    colors = tuple(
        VertexColor(
            vertex_id,
            (0.2 + index * 0.1, 0.4, 0.6),
            (0.01, 0.01, 0.01),
            views,
            0,
            views,
            (HASHES[6], HASHES[7]),
            accepted[index],
        )
        for index, vertex_id in enumerate(ids)
    )
    return AppearanceEvidence(
        ready=True,
        vertex_colors=colors,
        source_frame_sha256=(HASHES[6], HASHES[7]),
        minimum_independent_accepted_views=views,
        physical_depth_validation_revision=HASHES[4],
        calibration_id="cal-1",
        calibration_sha256=calibration,
        measured_generation=generation,
        blockers=(),
    )


class _Refiner:
    voxel_size_m = 0.015

    def __init__(self, evidence):
        self.evidence = evidence
        self.requested = None

    def appearance_evidence(self, vertex_ids):
        self.requested = tuple(vertex_ids)
        return self.evidence

    def snapshot(self):
        evidence = self.evidence
        return {
            "calibration_id": evidence.calibration_id,
            "calibration_sha256": evidence.calibration_sha256,
            "accepted_depth_frames": int(evidence.ready),
            "accepted_color_observations": sum(
                color.accepted_count for color in evidence.vertex_colors
            ),
            "depth_correspondence_count": len(evidence.vertex_colors),
            "blockers": list(evidence.blockers),
            "texture": {
                "delivery_authorized": evidence.ready,
                "minimum_independent_accepted_views": (
                    evidence.minimum_independent_accepted_views
                ),
            },
            "depth_anything": {},
        }


def test_processor_scheduling_revision_ignores_transient_inference_telemetry(tmp_path):
    metadata, body = _tile()

    class Refiner(_Refiner):
        inference_ms = 10.0
        processed = 1

        def snapshot(self):
            return {
                "calibration_sha256": HASHES[3],
                "accepted_depth_frames": 2,
                "accepted_color_observations": 6,
                "texture": {"delivery_authorized": True},
                "depth_anything": {
                    "source_generation_id": HASHES[2],
                    "model_revision_id": None,
                    "assistance_config_sha256": None,
                    "fit_accepted": False,
                    "augmentation_reference": None,
                    "blockers": [],
                    "last_inference_ms": self.inference_ms,
                    "processed_frames": self.processed,
                },
            }

    refiner = Refiner(_evidence())
    processor = ValidatedTileAppearanceProcessor(
        tile_store=SimpleNamespace(load=lambda _: (metadata, body)),
        rgbd_refiner=refiner,
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
        appearance_config_sha256=HASHES[5],
        calibration_sha256_resolver=lambda: HASHES[3],
        publication_scope="LOCAL",
    )
    first = processor.scheduling_revision()
    refiner.inference_ms = 42.0
    refiner.processed = 99

    assert processor.scheduling_revision() == first


def _scope(**changes):
    values = dict(
        building_id="building-a",
        floor_id="floor-1",
        component_id="component-a",
        submap_id="submap-a",
        attachment_state="ROOT",
        publication_scope="BUILDING",
    )
    values.update(changes)
    return TileAppearanceScope(**values)


def test_appearance_scope_requires_explicit_publication_scope():
    with pytest.raises(TypeError, match="publication_scope"):
        TileAppearanceScope(
            building_id="building-a",
            floor_id="floor-1",
            component_id="component-a",
            submap_id="submap-a",
            attachment_state="ROOT",
        )


def _request(metadata, body, refiner, **changes):
    values = dict(
        metadata=metadata,
        tile_body=body,
        rgbd_refiner=refiner,
        scope=_scope(),
        expected_calibration_sha256=HASHES[3],
        pose_revision_id=HASHES[2],
        appearance_config_sha256=HASHES[5],
        atlas_dimension=64,
        padding_pixels=0,
        mip_filter_radius_pixels=0,
    )
    values.update(changes)
    return build_validated_tile_bake_request(**values)


def test_validated_evidence_reorders_spatial_colors_to_exact_tile_order_and_bakes(tmp_path):
    metadata, body = _tile()
    before_metadata, before_body = dict(metadata), bytes(body)
    refiner = _Refiner(_evidence())
    request = _request(metadata, body, refiner)
    baker = TileAtlasBaker(
        source_loader=lambda _: (dict(metadata), body),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    )

    result = baker.bake(request)

    assert refiner.requested == ("0:0:66", "1:0:66", "0:1:66")
    assert tuple(color.vertex_id for color in request.vertex_colors) == tuple(
        f"{HASHES[0]}:{index}" for index in range(3)
    )
    assert request.measured_surface_coverage_ppm == 1_000_000
    assert request.triangle_authorized == (True,)
    assert result.state == "TEXTURED_READY"
    assert result.building_authorized is True
    assert result.export_authorized is True
    assert metadata == before_metadata
    assert body == before_body


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"expected_calibration_sha256": HASHES[8]}, "calibration"),
        ({"pose_revision_id": HASHES[8]}, "pose revision"),
        ({"scope": _scope(floor_id="floor-2")}, "scope"),
    ],
)
def test_identity_mismatch_rejects_before_baker(changes, message):
    metadata, body = _tile()
    with pytest.raises(ValueError, match=message):
        _request(metadata, body, _Refiner(_evidence()), **changes)


def test_stale_generation_and_blocked_refiner_never_create_request():
    metadata, body = _tile()
    with pytest.raises(ValueError, match="pose revision"):
        _request(metadata, body, _Refiner(_evidence(generation=HASHES[8])))

    blocked = AppearanceEvidence(False, (), (), 0, None, None, None, None, ("CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING",))
    with pytest.raises(ValueError, match="CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING"):
        _request(metadata, body, _Refiner(blocked))


def test_float32_tile_quantization_does_not_collapse_voxel_boundary_colors():
    metadata, body = _tile()
    refiner = _Refiner(_evidence(accepted=(True, True, False)))

    request = _request(metadata, body, refiner)

    assert refiner.requested == ("0:0:66", "1:0:66", "0:1:66")
    assert request.measured_surface_coverage_ppm == 666666
    assert request.triangle_authorized == (False,)


def test_coverage_and_view_boundaries_and_detached_scope_remain_fail_closed(tmp_path):
    metadata, body = _tile()
    low = _request(metadata, body, _Refiner(_evidence(accepted=(True, True, False))))
    assert low.measured_surface_coverage_ppm == 666666
    assert low.triangle_authorized == (False,)

    one_view = _request(metadata, body, _Refiner(_evidence(views=1)))
    detached_metadata = {**metadata, "attachment_state": "DETACHED"}
    detached = _request(
        detached_metadata,
        body,
        _Refiner(_evidence()),
        scope=_scope(attachment_state="DETACHED"),
    )
    baker = TileAtlasBaker(
        source_loader=lambda _: (dict(detached_metadata), body),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    )
    detached_result = baker.bake(detached)

    assert one_view.minimum_independent_view_count == 1
    assert detached.building_delivery_eligible is False
    assert detached_result.state == "TEXTURED_READY"
    assert detached_result.building_authorized is False
    assert detached_result.export_authorized is False


def test_local_scope_can_finish_diagnostically_without_building_or_export_authority(tmp_path):
    metadata, body = _tile()
    request = _request(
        metadata,
        body,
        _Refiner(_evidence()),
        scope=_scope(publication_scope="LOCAL"),
    )

    result = TileAtlasBaker(
        source_loader=lambda _: (dict(metadata), body),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    ).bake(request)

    assert result.state == "TEXTURED_READY"
    assert result.building_authorized is False
    assert result.export_authorized is False


@pytest.mark.parametrize(
    ("case", "expected_blocker"),
    [
        ("wrong_depth_scale", "DEPTH_TILE_GATE_FAILED"),
        ("camera_info_identity", "CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING"),
        ("extrinsic_transform", "DEPTH_TILE_GATE_FAILED"),
        ("time_association", "DEPTH_LIDAR_TIME_ASSOCIATION_FAILED"),
    ],
)
def test_real_sqlite_rgbd_identity_and_association_failures_never_authorize_appearance(
    tmp_path, case, expected_blocker
):
    tile_store = ImmutableTileStore(tmp_path / "tiles")
    tile_metadata = tile_store.publish(
        "building-a",
        "floor-1",
        "component-a",
        "ROOT",
        "submap-a",
        _mesh_revision(),
        source_generation_id=HASHES[1],
        pose_revision_id=HASHES[2],
    )
    before_metadata, before_body = tile_store.load(tile_metadata["revision_id"])
    manifest_path, color, depth = _manifest(tmp_path / f"{case}-calibration.json")
    lidar = _lidar_frame()
    depth_scale_m = 0.001

    if case == "camera_info_identity":
        color = {**color, "k": [*color["k"]]}
        color["k"][0] = 99.0
    elif case == "extrinsic_transform":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["frame_chain"]["t_c_l"]["row_major"][11] = 0.2
        payload["frame_chain"]["t_c_l"]["sha256"] = canonical_sha256(
            {"row_major": payload["frame_chain"]["t_c_l"]["row_major"]}
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "time_association":
        lidar = {**lidar, "timestamp_ns": 2_000_000_000}
    elif case == "wrong_depth_scale":
        depth_scale_m = 0.002

    bag = tmp_path / f"{case}.db3"
    _bag(bag, color, depth)
    refiner = ArtifactRgbdRefiner(manifest_path, depth_scale_m=depth_scale_m)
    refiner.process_sqlite_bag(
        bag,
        artifact_sha256=HASHES[6],
        lidar_frames=(lidar,),
        measured_generation=HASHES[2],
    )

    state = refiner.snapshot()
    evidence = refiner.appearance_evidence(("0:0:66",))
    assert state["accepted_depth_frames"] == 0
    assert state["accepted_color_observations"] == 0
    assert any(blocker.startswith(expected_blocker) for blocker in state["blockers"])
    assert evidence.ready is False
    with pytest.raises(ValueError):
        build_validated_tile_bake_request(
            metadata=before_metadata,
            tile_body=before_body,
            rgbd_refiner=refiner,
            scope=_published_scope(before_metadata, publication_scope="LOCAL"),
            expected_calibration_sha256=refiner.snapshot()["calibration_sha256"] or HASHES[3],
            pose_revision_id=before_metadata["pose_revision_id"],
            appearance_config_sha256=HASHES[5],
        )
    after_metadata, after_body = tile_store.load(tile_metadata["revision_id"])
    assert after_metadata == before_metadata
    assert after_body == before_body
    assert after_metadata["vertex_count"] == before_metadata["vertex_count"]
    assert after_metadata["triangle_count"] == before_metadata["triangle_count"]
    assert not list(tmp_path.rglob("vertex-color.bin"))


def test_pose_generation_mismatch_rejects_real_accepted_depth_evidence(tmp_path):
    manifest_path, color, depth = _manifest(tmp_path / "pose-calibration.json")
    bag = tmp_path / "pose.db3"
    _bag(bag, color, depth)
    refiner = ArtifactRgbdRefiner(manifest_path, depth_scale_m=0.001)
    refiner.process_sqlite_bag(
        bag,
        artifact_sha256=HASHES[6],
        lidar_frames=(_lidar_frame(),),
        measured_generation=HASHES[2],
    )
    assert refiner.snapshot()["accepted_depth_frames"] == 1

    with pytest.raises(ValueError, match="pose revision"):
        _request(*_tile(), refiner, pose_revision_id=HASHES[8])


def test_coincident_local_detached_and_building_tiles_keep_distinct_immutable_authority(tmp_path):
    tile_store = ImmutableTileStore(tmp_path / "tiles")
    revision = _mesh_revision()
    identities = (
        ("LOCAL", "building-local", "floor-local", "component-local", "ROOT", "submap-local", False),
        ("BUILDING", "building-a", "floor-1", "component-detached", "DETACHED", "submap-detached", False),
        ("BUILDING", "building-a", "floor-1", "component-a", "ROOT", "submap-a", True),
    )
    published = [
        (
            publication_scope,
            tile_store.publish(
                building_id,
                floor_id,
                component_id,
                attachment_state,
                    submap_id,
                    revision,
                    source_generation_id=HASHES[1],
                    pose_revision_id=HASHES[2],
                ),
            expected_authority,
        )
        for (
            publication_scope,
            building_id,
            floor_id,
            component_id,
            attachment_state,
            submap_id,
            expected_authority,
        ) in identities
    ]

    results = []
    for publication_scope, metadata, expected_authority in published:
        stored_metadata, body = tile_store.load(metadata["revision_id"])
        request = _request(
            stored_metadata,
            body,
            _Refiner(_evidence()),
            scope=_published_scope(
                stored_metadata, publication_scope=publication_scope
            ),
        )
        result = TileAtlasBaker(
            source_loader=tile_store.load,
            appearance_store=TileAppearanceStore(tmp_path / "appearance"),
            preview_store=TileVertexColorStore(tmp_path / "appearance"),
        ).bake(request)
        results.append(result)
        assert result.state == "TEXTURED_READY"
        assert result.building_authorized is expected_authority
        assert result.export_authorized is expected_authority

    assert len({metadata["revision_id"] for _, metadata, _ in published}) == 3
    assert {tuple(metadata["tile_id"]) for _, metadata, _ in published} == {(0, 0, 0)}
    assert len({result.published.root_path for result in results}) == 3


def test_appearance_failure_leaves_immutable_tile_bytes_and_measured_counts_unchanged(tmp_path):
    tile_store = ImmutableTileStore(tmp_path / "tiles")
    metadata = tile_store.publish(
        "building-a",
        "floor-1",
        "component-a",
        "ROOT",
        "submap-a",
        _mesh_revision(),
        source_generation_id=HASHES[1],
        pose_revision_id=HASHES[2],
    )
    before_metadata, before_body = tile_store.load(metadata["revision_id"])
    blocked = AppearanceEvidence(
        False,
        (),
        (),
        0,
        None,
        None,
        None,
        HASHES[2],
        ("CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING",),
    )
    processor = ValidatedTileAppearanceProcessor(
        tile_store=tile_store,
        rgbd_refiner=_Refiner(blocked),
        appearance_store=TileAppearanceStore(tmp_path / "appearance"),
        preview_store=TileVertexColorStore(tmp_path / "appearance"),
        appearance_config_sha256=HASHES[5],
        calibration_sha256_resolver=lambda: HASHES[3],
        publication_scope="LOCAL",
    )

    with pytest.raises(ValueError, match="CAMERA_INFO_IDENTITY_MISMATCH_OR_MISSING"):
        processor(metadata)

    after_metadata, after_body = tile_store.load(metadata["revision_id"])
    assert after_body == before_body
    assert after_metadata == before_metadata
    assert after_metadata["vertex_count"] == before_metadata["vertex_count"] == 3
    assert after_metadata["triangle_count"] == before_metadata["triangle_count"] == 1
    assert not list((tmp_path / "appearance").rglob("vertex-color.bin"))


@pytest.mark.parametrize(
    ("coverage_ppm", "views", "expected_blocker"),
    [
        (799_999, 2, "MEASURED_SURFACE_COVERAGE_REQUIRED"),
        (1_000_000, 1, "INDEPENDENT_COLOR_VIEWS_REQUIRED"),
    ],
)
def test_exact_delivery_boundaries_never_gain_building_or_export_authority(
    tmp_path, coverage_ppm, views, expected_blocker
):
    metadata, body = _tile()
    request = _request(metadata, body, _Refiner(_evidence(views=views)))
    request = replace(request, measured_surface_coverage_ppm=coverage_ppm)

    result = TileAtlasBaker(
        source_loader=lambda _: (dict(metadata), body),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    ).bake(request)

    assert result.state == "DIAGNOSTIC_ATLAS_READY"
    assert result.building_authorized is False
    assert result.export_authorized is False
    assert expected_blocker in result.revision.delivery_authorization.blockers


def test_calibrated_sqlite_bag_produces_two_view_tile_evidence_and_exact_atlas(tmp_path):
    manifest_path, color, depth = _manifest(tmp_path / "calibration.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for contract in (color, depth):
        contract["k"][0] = contract["k"][4] = 20.0
        contract["p"][0] = contract["p"][5] = 20.0
    payload["frame_chain"]["color_camera_info"] = color
    payload["frame_chain"]["depth_camera_info"] = depth
    payload["frame_chain"]["color_camera_info_sha256"] = canonical_sha256(color)
    payload["frame_chain"]["depth_camera_info_sha256"] = canonical_sha256(depth)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    first_bag = tmp_path / "rgbd-first.db3"
    second_bag = tmp_path / "rgbd-second.db3"
    _bag(first_bag, color, depth)
    _bag(second_bag, color, depth)
    with sqlite3.connect(second_bag) as database:
        rows = database.execute("SELECT id, data FROM messages").fetchall()
        for message_id, encoded in rows:
            retimed = bytearray(encoded)
            struct.pack_into("<iI", retimed, 4, 2, 0)
            database.execute(
                "UPDATE messages SET timestamp = 2000000000, data = ? WHERE id = ?",
                (bytes(retimed), message_id),
            )
        database.commit()
    world = np.asarray([
        ((u - 10) / 20.0, (v - 10) / 20.0, 1.0)
        for v in range(20) for u in range(20)
    ])
    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[0, 3] = 0.2
    first = {"timestamp_ns": 1_000_000_000, "points_xyz": world, "pose_map_from_lidar": first_pose}
    second = {"timestamp_ns": 2_000_000_000, "points_xyz": world, "pose_map_from_lidar": second_pose}
    refiner = ArtifactRgbdRefiner(manifest_path, depth_scale_m=0.001)
    refiner.process_sqlite_bag(first_bag, artifact_sha256=HASHES[6], lidar_frames=(first,), measured_generation=HASHES[2])
    refiner.process_sqlite_bag(second_bag, artifact_sha256=HASHES[7], lidar_frames=(second,), measured_generation=HASHES[2])
    positions = np.asarray([[0, 0, 1], [0.05, 0, 1], [0, 0.05, 1]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.004], dtype="<f4")
    body = b"".join((struct.pack("<4sIIII", b"G2TM", 1, 3, 1, 1), positions.tobytes(), normals.tobytes(), triangles.tobytes(), scales.tobytes()))
    metadata = {**_tile()[0], "content_sha256": hashlib.sha256(body).hexdigest()}
    state_before_request = refiner.snapshot()
    assert state_before_request["accepted_depth_frames"] == 2, state_before_request

    request = _request(
        metadata, body, refiner,
        expected_calibration_sha256=refiner.snapshot()["calibration_sha256"],
    )
    result = TileAtlasBaker(
        source_loader=lambda _: (metadata, body),
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
    ).bake(request)

    state = refiner.snapshot()
    assert state["accepted_depth_frames"] == 2, state
    assert state["accepted_color_observations"] > 0
    assert request.minimum_independent_view_count == 2
    assert request.measured_surface_coverage_ppm == 1_000_000
    assert result.state == "TEXTURED_READY"


def test_checkpoint_to_local_and_building_appearance_uses_exact_pose_and_historical_evidence(
    tmp_path, monkeypatch
):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert pipeline.admit(_live_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    measured_positions = np.asarray([[0, 0, 1], [0.05, 0, 1], [0, 0.05, 1]], dtype="<f4")
    measured_normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    measured_triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    measured_scales = np.asarray([0.004], dtype="<f4")
    mesh_revision = MeshTileRevision(
        TileId(0, 0, 0),
        canonical_sha256({"fixture": "observed-50mm-grid-triangle"}),
        measured_positions,
        measured_normals,
        measured_triangles,
        measured_scales,
        0,
    )
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={mesh_revision.tile_id: mesh_revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)
    metadata = pipeline.tile_store.manifest(scope="local")["tiles"][0]
    assert metadata["source_generation_id"] == update.checkpoint_generation_id
    assert metadata["pose_revision_id"] != metadata["source_generation_id"]

    manifest_path, color, depth = _manifest(tmp_path / "pipeline-calibration.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for contract in (color, depth):
        contract["k"][0] = contract["k"][4] = 20.0
        contract["p"][0] = contract["p"][5] = 20.0
    payload["frame_chain"]["color_camera_info"] = color
    payload["frame_chain"]["depth_camera_info"] = depth
    payload["frame_chain"]["color_camera_info_sha256"] = canonical_sha256(color)
    payload["frame_chain"]["depth_camera_info_sha256"] = canonical_sha256(depth)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    bags = (tmp_path / "pipeline-first.db3", tmp_path / "pipeline-second.db3")
    _bag(bags[0], color, depth)
    _bag(bags[1], color, depth)
    with sqlite3.connect(bags[1]) as database:
        for message_id, encoded in database.execute("SELECT id, data FROM messages").fetchall():
            retimed = bytearray(encoded)
            struct.pack_into("<iI", retimed, 4, 2, 0)
            database.execute(
                "UPDATE messages SET timestamp = 2000000000, data = ? WHERE id = ?",
                (bytes(retimed), message_id),
            )
        database.commit()
    world = np.asarray([
        ((u - 10) / 20.0, (v - 10) / 20.0, 1.0)
        for v in range(20) for u in range(20)
    ])
    poses = (np.eye(4), np.eye(4))
    poses[1][0, 3] = 0.2
    refiner = ArtifactRgbdRefiner(manifest_path, depth_scale_m=0.001)
    for index, bag in enumerate(bags):
        refiner.process_sqlite_bag(
            bag,
            artifact_sha256=HASHES[6 + index],
            lidar_frames=({
                "timestamp_ns": (index + 1) * 1_000_000_000,
                "points_xyz": world,
                "pose_map_from_lidar": poses[index],
            },),
            measured_generation=metadata["pose_revision_id"],
        )

    appearance_root = tmp_path / "appearance"
    local = ValidatedTileAppearanceProcessor(
        tile_store=pipeline.tile_store,
        rgbd_refiner=refiner,
        appearance_store=TileAppearanceStore(appearance_root),
        preview_store=TileVertexColorStore(appearance_root),
        appearance_config_sha256=HASHES[5],
        calibration_sha256_resolver=lambda: refiner.snapshot()["calibration_sha256"],
        publication_scope="LOCAL",
        atlas_dimension=64,
        padding_pixels=0,
        mip_filter_radius_pixels=0,
    )(metadata)
    building = ValidatedTileAppearanceProcessor(
        tile_store=pipeline.tile_store,
        rgbd_refiner=refiner,
        appearance_store=TileAppearanceStore(tmp_path / "building-appearance"),
        preview_store=TileVertexColorStore(tmp_path / "building-appearance"),
        appearance_config_sha256=HASHES[5],
        calibration_sha256_resolver=lambda: refiner.snapshot()["calibration_sha256"],
        publication_scope="BUILDING",
        atlas_dimension=64,
        padding_pixels=0,
        mip_filter_radius_pixels=0,
    )(metadata)
    evidence = refiner.appearance_evidence(("0:0:66", "3:0:66", "0:3:66"))
    historical = HistoricalAppearanceStore(tmp_path / "historical").publish(
        build_historical_vertex_color_sidecar(
            geometry_revision_id=metadata["geometry_revision_id"],
            pose_generation=metadata["pose_revision_id"],
            calibration_id=evidence.calibration_id,
            calibration_sha256=evidence.calibration_sha256,
            physical_depth_validation_revision=evidence.physical_depth_validation_revision,
            appearance_config_sha256=HASHES[5],
            vertex_colors=evidence.vertex_colors,
        ),
        expected_geometry_revision_id=metadata["geometry_revision_id"],
    )

    assert refiner.snapshot()["accepted_depth_frames"] == 2
    assert refiner.snapshot()["accepted_color_observations"] > 0
    assert local.state == building.state == "TEXTURED_READY", {
        "local": local.revision.delivery_authorization,
        "building": building.revision.delivery_authorization,
        "refiner": refiner.snapshot(),
    }
    assert local.building_authorized is False
    assert building.building_authorized is building.export_authorized is True
    assert building.revision.revision_core["pose_revision_id"] == metadata["pose_revision_id"]
    assert historical.manifest["pose_generation"] == metadata["pose_revision_id"]


def test_validated_processor_reloads_exact_tile_and_rejects_queue_metadata_mutation(tmp_path):
    metadata, body = _tile()
    refiner = _Refiner(_evidence())

    class TileStore:
        def load(self, revision_id):
            assert revision_id == HASHES[0]
            return dict(metadata), body

    processor = ValidatedTileAppearanceProcessor(
        tile_store=TileStore(),
        rgbd_refiner=refiner,
        appearance_store=TileAppearanceStore(tmp_path),
        preview_store=TileVertexColorStore(tmp_path),
        appearance_config_sha256=HASHES[5],
        calibration_sha256_resolver=lambda: HASHES[3],
        atlas_dimension=64,
        padding_pixels=0,
        mip_filter_radius_pixels=0,
        publication_scope="LOCAL",
    )

    result = processor(dict(metadata))
    assert result.state == "TEXTURED_READY"
    assert result.building_authorized is False
    assert result.export_authorized is False

    mutated = dict(metadata)
    mutated["floor_id"] = "floor-elsewhere"
    with pytest.raises(ValueError, match="queue metadata"):
        processor(mutated)

    stale_evidence = {
        **metadata,
        "appearance_evidence_revision_id": "9" * 64,
    }
    with pytest.raises(ValueError, match="evidence revision"):
        processor(stale_evidence)
