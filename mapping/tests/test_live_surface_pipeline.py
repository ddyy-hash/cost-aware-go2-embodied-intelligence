import hashlib
from concurrent.futures import Future
from io import BytesIO
import json
from types import SimpleNamespace
import threading
import time

import numpy as np
from PIL import Image
import pytest

from mapping.pc.go2_mapping_appearance import (
    ArtifactRef,
    BuildingAppearanceMember,
    DistanceLodInterval,
    PrimitiveRecord,
    TileAppearanceStore,
    TileLodPayload,
    build_lod_record,
    tile_appearance_revision,
)
from mapping.pc.go2_mapping_appearance.atlas_codec import (
    encode_render_indices_v1,
    encode_render_uvs_v1,
    encode_render_vertices_v1,
)
from mapping.pc.go2_mapping_live.live_pipeline import LiveSurfacePipeline
from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe
from mapping.pc.go2_mapping_live.live_surface import LiveSurfaceMap, TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import LiveSurfaceMesher, MeshTileRevision
from mapping.pc.go2_mapping_live.live_surface_mesher import TriangleProvenance
from mapping.pc.go2_mapping_live.live_tiles import decode_surface_tile
from mapping.pc.go2_mapping_live.live_wal import LiveKeyframeWal
from mapping.pc.go2_mapping_rgbd.multiview_surface import NeuralSurfaceCandidate, NeuralSurfaceGate
from mapping.pc.go2_mapping_rgbd.neural_surface_runtime import SurfaceTriangleBatch
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import (
    PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    build_live_keyframe_message,
)


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()


def _frame(
    sequence: int,
    *,
    session_id="session-a",
    map_id="map-a",
    producer_id="producer-a",
    origin=(0.0, 0.0, 0.0),
    points=None,
    points_in_map_frame=True,
    sensor_frame_transform_id=None,
):
    pose = np.eye(4)
    pose[:3, 3] = origin
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id=session_id,
        map_id=map_id,
        producer_instance_id=producer_id,
        keyframe_sequence=sequence,
        timestamp_ns=10_000 + sequence,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=np.asarray(
            [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]] if points is None else points,
            dtype="<f4",
        ),
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=points_in_map_frame,
        sensor_frame_transform_id=sensor_frame_transform_id,
    )
    return validate_live_keyframe(encoded.payload)


def test_pipeline_binds_corrected_sensor_frame_into_new_surface_and_checkpoint(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    frame = _frame(
        1,
        points=[[0.0, 0.0, 0.5]],
        points_in_map_frame=False,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
    )
    assert wal.admit(frame).accepted
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)

    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)

    submap_root = tmp_path / "buildings" / "building-a" / "submaps" / update.submap_id
    manifest = json.loads(
        (submap_root / "generations" / update.checkpoint_generation_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert pipeline.snapshot()["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def _mesh_revision(*, unresolved_surfel_count: int = 0) -> MeshTileRevision:
    positions = np.asarray([[0, 0, 0], [0.005, 0, 0], [0, 0.005, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.005], dtype="<f4")
    digest = hashlib.sha256(positions.tobytes() + normals.tobytes() + triangles.tobytes() + scales.tobytes()).hexdigest()
    return MeshTileRevision(
        TileId(0, 0, 0),
        digest,
        positions,
        normals,
        triangles,
        scales,
        unresolved_surfel_count,
    )


def _surface_batch(
    anchors,
    *,
    provenance=TriangleProvenance.MEASURED_RGBD,
    scales=None,
    uncertainties=None,
    normal=(0.0, 0.0, 1.0),
):
    positions = []
    normals = []
    triangles = []
    for anchor in anchors:
        start = len(positions)
        x, y, z = anchor
        positions.extend(((x, y, z), (x + 0.006, y, z), (x, y + 0.006, z)))
        normals.extend((normal, normal, normal))
        triangles.append((start, start + 1, start + 2))
    count = len(triangles)
    scale_values = [0.01] * count if scales is None else scales
    uncertainty_values = [0.008] * count if uncertainties is None else uncertainties
    body = np.asarray(positions, dtype="<f4").tobytes()
    return SurfaceTriangleBatch(
        positions=np.asarray(positions, dtype="<f4").reshape(-1, 3),
        normals=np.asarray(normals, dtype="<f4").reshape(-1, 3),
        triangles=np.asarray(triangles, dtype="<u4").reshape(-1, 3),
        triangle_scale_m=np.asarray(scale_values, dtype="<f4"),
        triangle_provenance=np.full(count, int(provenance), dtype="u1"),
        triangle_uncertainty_m=np.asarray(uncertainty_values, dtype="<f4"),
        source_revision_id=hashlib.sha256(body).hexdigest(),
    )


def _neural_publication(anchors, *, scales=None, uncertainties=None, normal=(0.0, 0.0, 1.0)):
    batch = _surface_batch(
        anchors,
        provenance=TriangleProvenance.NEURAL_MULTIVIEW,
        scales=scales,
        uncertainties=uncertainties,
        normal=normal,
    )
    count = len(batch.triangles)
    candidate = NeuralSurfaceCandidate(
        positions=batch.positions,
        normals=batch.normals,
        triangles=batch.triangles,
        triangle_scale_m=batch.triangle_scale_m,
        supporting_frame_count=np.full(count, 3, dtype="u1"),
        independent_view_count=np.full(count, 2, dtype="u1"),
        holdout_median_residual_m=np.full(count, 0.01, dtype="<f4"),
        holdout_p95_residual_m=np.full(count, 0.02, dtype="<f4"),
        holdout_absolute_bias_m=np.full(count, 0.005, dtype="<f4"),
        uncertainty_m=batch.triangle_uncertainty_m,
        observed_front=np.ones(count, dtype=bool),
        free_space_conflict=np.zeros(count, dtype=bool),
        occlusion_conflict=np.zeros(count, dtype=bool),
        depth_boundary_conflict=np.zeros(count, dtype=bool),
        measured_surface_conflict=np.zeros(count, dtype=bool),
        source_window_sha256=batch.source_revision_id,
    )
    decision = NeuralSurfaceGate().evaluate(candidate)
    return candidate, decision


def _meshed_surface_pipeline(monkeypatch, tmp_path, *, unresolved_surfel_count=0):
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        checkpoint_batch_frames=1,
    )
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
    mesh = _mesh_revision(unresolved_surfel_count=unresolved_surfel_count)
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={mesh.tile_id: mesh},
            triangle_count=1,
            unresolved_surfel_count=unresolved_surfel_count,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)
    return pipeline, update


def _rewrite_index_with_distinct_bytes(index_path):
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    index_path.write_bytes(body)
    return body


def _publish_pipeline_building_appearance(monkeypatch, tmp_path):
    appearance_store = TileAppearanceStore(tmp_path / "tile-appearance")
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        checkpoint_batch_frames=1,
        tile_appearance_store=appearance_store,
    )
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
    mesh = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={mesh.tile_id: mesh},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)
    closed = pipeline.close_session_source("building-a", "session-a", 1)
    project = pipeline._projects["building-a"]
    submap = project.submap(update.submap_id)
    component = project.component(submap["component_id"])
    assembly_member = AssemblyMember(
        "building-a",
        component["component_id"],
        "session-a",
        update.submap_id,
        closed["generation_id"],
        tuple(np.eye(4, dtype=np.float64).reshape(-1)),
    )
    assembly = pipeline.publish_assembly(
        "building-a", (assembly_member,), algorithm_version="pipeline-test-v1"
    )
    tile = pipeline.tile_store.manifest_for_assembly(assembly)["tiles"][0]
    appearance = tile_appearance_revision(
        tile_revision_id=tile["revision_id"],
        geometry_revision_id=tile["geometry_revision_id"],
        source_generation_id=tile["source_generation_id"],
        calibration_id="calibration-1",
        calibration_sha256=hashlib.sha256(b"calibration").hexdigest(),
        physical_depth_validation_revision=hashlib.sha256(b"depth-validation").hexdigest(),
        pose_revision_id=tile["pose_revision_id"],
        model_revision_id=None,
        assistance_config_sha256=None,
        source_frame_sha256=(
            hashlib.sha256(b"frame-a").hexdigest(),
            hashlib.sha256(b"frame-b").hexdigest(),
        ),
        appearance_config_sha256=hashlib.sha256(b"appearance-config").hexdigest(),
        charting={"implementation": "xatlas", "version": "0.0.11", "options": {}},
        atlas_density_policy={"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        padding_policy={"pixels": 2, "mip_levels": 1},
        baker_algorithm_version="tile-atlas-baker-v1",
        rgbd_augmented_revision_id=None,
        contains_augmented_primitives=False,
        measured_surface_coverage_ppm=800_000,
        minimum_independent_view_count=2,
    )
    positions = np.asarray([[0, 0, 0], [0.005, 0, 0], [0, 0.005, 0]], dtype=np.float32)
    image = Image.new("RGBA", (8, 8), (20, 40, 60, 255))
    encoded_image = BytesIO()
    image.save(encoded_image, format="PNG", optimize=False, compress_level=9)
    artifacts = {
        "render-vertices.bin": encode_render_vertices_v1(
            tile["revision_id"], appearance.appearance_revision_id, positions,
            np.asarray([[0, 0, 1]] * 3), [0, 1, 2]
        ),
        "render-indices.bin": encode_render_indices_v1(
            tile["revision_id"], appearance.appearance_revision_id,
            np.asarray([[0, 1, 2]], dtype=np.uint32), [0]
        ),
        "render-uvs.bin": encode_render_uvs_v1(
            tile["revision_id"], appearance.appearance_revision_id,
            np.asarray([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]), [0, 0, 0], [1, 1, 1]
        ),
        "atlas-0.png": encoded_image.getvalue(),
    }
    refs = tuple(
        ArtifactRef(name, hashlib.sha256(body).hexdigest(), len(body))
        for name, body in artifacts.items()
    )
    lod = build_lod_record(
        tile_revision_id=tile["revision_id"],
        appearance_revision_id=appearance.appearance_revision_id,
        policy={"name": "full-detail", "order": 0, "mm_per_texel": 3},
        vertex_count=3,
        triangle_count=1,
        atlas_count=1,
        artifacts=refs,
        primitives=(
            PrimitiveRecord(0, "VALIDATED_UV_ATLAS", 0, 0, 3, "LIDAR_MEASURED", (0,)),
        ),
        rgbd_augmented_revision_id=None,
    )
    appearance_store.publish(
        appearance, (TileLodPayload(lod, artifacts),), building_delivery_eligible=True
    )
    building_member = BuildingAppearanceMember(
        building_id="building-a",
        component_id=assembly_member.component_id,
        session_id=assembly_member.session_id,
        submap_id=assembly_member.submap_id,
        source_generation_id=assembly_member.generation_id,
        placement_revision_id=tile["placement_revision_id"],
        tile_revision_id=tile["revision_id"],
        tile_appearance_revision_id=appearance.appearance_revision_id,
        authorized_lod_ids=(lod.lod_id,),
        distance_policy=(DistanceLodInterval(0.0, None, lod.lod_id),),
        transform_building_from_submap=assembly_member.transform_building_from_submap,
        delivery_authorization=appearance.delivery_authorization,
        rgbd_augmented_revision_id=None,
    )
    return pipeline, appearance_store, assembly, building_member


def test_pipeline_checkpoints_surface_before_advancing_cursor_and_reclaiming_wal(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    frame = _frame(100)
    wal.admit(frame)
    record = wal.pending()[0]
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=5)

    update = pipeline.integrate(record, force_checkpoint=True)

    assert update.integrated_cells == 2
    assert update.checkpoint_generation_id is not None
    manifest_path = (
        tmp_path
        / "buildings"
        / "building-a"
        / "submaps"
        / update.submap_id
        / "generations"
        / update.checkpoint_generation_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {
        name: manifest[name]
        for name in ("building_id", "session_id", "floor_id", "submap_id")
    } == {
        "building_id": "building-a",
        "session_id": "session-a",
        "floor_id": "floor-unassigned",
        "submap_id": update.submap_id,
    }
    assert not wal.pending()
    assert not list((tmp_path / "wal" / "records").glob("*.json"))
    state = pipeline.snapshot()
    assert state["building_id"] == "building-a"
    assert state["active_floor_id"] == "floor-unassigned"
    assert state["surfel_count"] == 2
    assert state["quality"] == "PREVIEW_UNCALIBRATED"
    assert state["geometry_authority"] is False


def test_live_integration_does_not_checkpoint_on_the_frame_ack_path(tmp_path, monkeypatch):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    frame = _frame(100)
    wal.admit(frame)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=wal,
        checkpoint_batch_frames=1,
        checkpoint_interval_s=0.001,
    )
    real_checkpoint = LiveSurfaceMap.checkpoint

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("durable checkpoint ran on the frame ACK path")

    monkeypatch.setattr(LiveSurfaceMap, "checkpoint", unexpected_checkpoint)

    update = pipeline.integrate(wal.pending()[0])

    assert update.checkpoint_generation_id is None
    assert len(wal.pending()) == 1
    monkeypatch.setattr(LiveSurfaceMap, "checkpoint", real_checkpoint)
    assert len(pipeline.flush_due(force=True)) == 1
    assert wal.pending() == []
    pipeline.close()


def test_recovery_batches_pending_wal_into_one_checkpoint_per_submap(tmp_path, monkeypatch):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    for sequence in (100, 101, 102):
        wal.admit(_frame(sequence))
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=wal,
        checkpoint_batch_frames=1,
    )
    checkpoints = 0
    real_checkpoint = LiveSurfaceMap.checkpoint

    def counted_checkpoint(surface, *args, **kwargs):
        nonlocal checkpoints
        checkpoints += 1
        return real_checkpoint(surface, *args, **kwargs)

    monkeypatch.setattr(LiveSurfaceMap, "checkpoint", counted_checkpoint)

    recovered = pipeline.recover_pending()

    assert checkpoints == 1
    assert recovered[0][2]
    assert wal.pending() == []
    pipeline.close()


def test_valid_empty_rgbd_publication_uses_durable_runtime_and_returns_zero(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    original_manifest = pipeline.tile_store.manifest

    def locked_manifest(*args, **kwargs):
        assert pipeline._mesh_lock.locked()
        return original_manifest(*args, **kwargs)

    monkeypatch.setattr(pipeline.tile_store, "manifest", locked_manifest)
    try:
        assert pipeline.publish_measured_rgbd_surface(
            "building-a", update.submap_id, _surface_batch(())
        ) == {
            "building_id": "building-a",
            "submap_id": update.submap_id,
            "source_generation_id": update.checkpoint_generation_id,
            "published_tiles": 0,
            "triangle_count": 0,
            "tile_revision_ids": [],
        }
    finally:
        pipeline.close()


def test_empty_rgbd_publication_rejects_nonexistent_runtime(tmp_path):
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
    )
    try:
        with pytest.raises(ValueError, match="building does not exist"):
            pipeline.publish_measured_rgbd_surface(
                "building-missing", "submap-missing", _surface_batch(())
            )
    finally:
        pipeline.close()


def test_empty_rgbd_publication_rejects_runtime_without_durable_checkpoint(tmp_path):
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        checkpoint_batch_frames=10,
        checkpoint_interval_s=60.0,
    )
    try:
        assert pipeline.admit(_frame(1)).accepted
        update = pipeline.integrate(pipeline.wal.pending()[0])
        assert update.checkpoint_generation_id is None

        with pytest.raises(ValueError, match="durable measured generation"):
            pipeline.publish_measured_rgbd_surface(
                "building-a", update.submap_id, _surface_batch(())
            )
    finally:
        pipeline.close()


def test_empty_rgbd_provenance_cannot_hide_malformed_triangle_evidence(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    malformed = SurfaceTriangleBatch(
        positions=np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype="<f4"),
        normals=np.asarray([[0, 0, 1]] * 3, dtype="<f4"),
        triangles=np.asarray([[0, 1, 2]], dtype="<u4"),
        triangle_scale_m=np.asarray([0.01], dtype="<f4"),
        triangle_provenance=np.empty(0, dtype="u1"),
        triangle_uncertainty_m=np.asarray([0.008], dtype="<f4"),
        source_revision_id="a" * 64,
    )
    try:
        with pytest.raises(ValueError, match="evidence arrays"):
            pipeline.publish_measured_rgbd_surface(
                "building-a", update.submap_id, malformed
            )
    finally:
        pipeline.close()


def test_pipeline_atomically_layers_rgbd_then_gated_neural_triangles(monkeypatch, tmp_path):
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        checkpoint_batch_frames=1,
    )
    try:
        assert pipeline.admit(_frame(1)).accepted
        update = pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
        mesh = _mesh_revision(unresolved_surfel_count=7)
        monkeypatch.setattr(
            LiveSurfaceMesher,
            "build",
            lambda self, surface, *, tiles=None: SimpleNamespace(
                tile_revisions={mesh.tile_id: mesh},
                triangle_count=1,
                unresolved_surfel_count=7,
            ),
        )
        pipeline.mesh("building-a", update.submap_id)
        evidence = pipeline.neural_surface_evidence("building-a", update.submap_id)
        assert evidence["source_generation_id"] == update.checkpoint_generation_id
        assert len(evidence["tiles"]) == 1

        rgbd = SurfaceTriangleBatch(
            positions=np.asarray([[0.05, 0.05, 0.05], [0.06, 0.05, 0.05], [0.05, 0.06, 0.05]], dtype="<f4"),
            normals=np.asarray([[0, 0, 1]] * 3, dtype="<f4"),
            triangles=np.asarray([[0, 1, 2]], dtype="<u4"),
            triangle_scale_m=np.asarray([0.01], dtype="<f4"),
            triangle_provenance=np.asarray([TriangleProvenance.MEASURED_RGBD], dtype="u1"),
            triangle_uncertainty_m=np.asarray([0.008], dtype="<f4"),
            source_revision_id="a" * 64,
        )
        rgbd_result = pipeline.publish_measured_rgbd_surface("building-a", update.submap_id, rgbd)
        assert rgbd_result["triangle_count"] == 1

        candidate = NeuralSurfaceCandidate(
            positions=np.asarray([[0.10, 0.10, 0.10], [0.11, 0.10, 0.10], [0.10, 0.11, 0.10]], dtype="<f4"),
            normals=np.asarray([[0, 0, 1]] * 3, dtype="<f4"),
            triangles=np.asarray([[0, 1, 2]], dtype="<u4"),
            triangle_scale_m=np.asarray([0.01], dtype="<f4"),
            supporting_frame_count=np.asarray([3], dtype="u1"),
            independent_view_count=np.asarray([2], dtype="u1"),
            holdout_median_residual_m=np.asarray([0.01], dtype="<f4"),
            holdout_p95_residual_m=np.asarray([0.02], dtype="<f4"),
            holdout_absolute_bias_m=np.asarray([0.005], dtype="<f4"),
            uncertainty_m=np.asarray([0.01], dtype="<f4"),
            observed_front=np.asarray([True]),
            free_space_conflict=np.asarray([False]),
            occlusion_conflict=np.asarray([False]),
            depth_boundary_conflict=np.asarray([False]),
            measured_surface_conflict=np.asarray([False]),
            source_window_sha256="b" * 64,
        )
        decision = NeuralSurfaceGate().evaluate(candidate)
        neural_result = pipeline.publish_neural_surface(
            "building-a",
            update.submap_id,
            candidate,
            decision,
            expected_source_generation_id=evidence["source_generation_id"],
        )
        assert neural_result["triangle_count"] == 1

        current = [
            tile for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
            if tile["submap_id"] == update.submap_id and tile["tile_id"] == [0, 0, 0]
        ][0]
        assert current["unresolved_surfel_count"] == 7
        _, body = pipeline.tile_store.load(current["revision_id"])
        decoded = decode_surface_tile(body)
        assert decoded.triangle_provenance.tolist() == [
            TriangleProvenance.MEASURED_LIDAR,
            TriangleProvenance.MEASURED_RGBD,
            TriangleProvenance.NEURAL_MULTIVIEW,
        ]
        with pytest.raises(ValueError, match="stale"):
            pipeline.publish_neural_surface(
                "building-a", update.submap_id, candidate, decision,
                expected_source_generation_id="f" * 64,
            )
    finally:
        pipeline.close()


def test_surface_publication_floors_triangle_centres_into_quarter_metre_tiles(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        evidence = pipeline.neural_surface_evidence("building-a", update.submap_id)
        assert evidence["tile_m"] == pytest.approx(0.25)
        batch = _surface_batch(
            (
                (-0.253, 0.10, 0.10),
                (-0.003, 0.10, 0.10),
                (0.247, 0.10, 0.10),
                (0.250, 0.10, 0.10),
            )
        )

        result = pipeline.publish_measured_rgbd_surface(
            "building-a", update.submap_id, batch
        )

        assert result["published_tiles"] == 4
        assert result["triangle_count"] == 4
        tiles = [
            tile
            for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
            if tile["submap_id"] == update.submap_id
        ]
        assert [tile["tile_id"] for tile in tiles] == [
            [-2, 0, 0],
            [-1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]
    finally:
        pipeline.close()


def test_surface_authority_conflicts_keep_evidence_arrays_aligned_and_idempotent(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(
        monkeypatch, tmp_path, unresolved_surfel_count=13
    )
    try:
        generation_id = update.checkpoint_generation_id
        neural, neural_decision = _neural_publication(
            ((0.10, 0.10, 0.10),),
            scales=(0.011,),
            uncertainties=(0.025,),
            normal=(1.0, 0.0, 0.0),
        )
        assert pipeline.publish_neural_surface(
            "building-a",
            update.submap_id,
            neural,
            neural_decision,
            expected_source_generation_id=generation_id,
        )["triangle_count"] == 1

        rgbd = _surface_batch(
            ((0.10, 0.10, 0.10),),
            scales=(0.007,),
            uncertainties=(0.006,),
            normal=(0.0, 1.0, 0.0),
        )
        assert pipeline.publish_measured_rgbd_surface(
            "building-a", update.submap_id, rgbd
        )["triangle_count"] == 1
        assert pipeline.publish_measured_rgbd_surface(
            "building-a", update.submap_id, _surface_batch(((0.0, 0.0, 0.0),))
        )["triangle_count"] == 0
        assert pipeline.publish_neural_surface(
            "building-a",
            update.submap_id,
            neural,
            neural_decision,
            expected_source_generation_id=generation_id,
        )["triangle_count"] == 0

        other_neural, other_decision = _neural_publication(
            ((0.15, 0.15, 0.10),),
            scales=(0.013,),
            uncertainties=(0.029,),
            normal=(1.0, 0.0, 0.0),
        )
        assert pipeline.publish_neural_surface(
            "building-a",
            update.submap_id,
            other_neural,
            other_decision,
            expected_source_generation_id=generation_id,
        )["triangle_count"] == 1

        metadata = pipeline.tile_store.manifest(scope="local")["tiles"][0]
        assert metadata["unresolved_surfel_count"] == 13
        _, body = pipeline.tile_store.load(metadata["revision_id"])
        decoded = decode_surface_tile(body)
        assert decoded.triangle_provenance.tolist() == [
            TriangleProvenance.MEASURED_LIDAR,
            TriangleProvenance.MEASURED_RGBD,
            TriangleProvenance.NEURAL_MULTIVIEW,
        ]
        assert decoded.triangle_scale_m.tolist() == pytest.approx([0.005, 0.007, 0.013])
        assert decoded.triangle_uncertainty_m.tolist() == pytest.approx(
            [0.005, 0.006, 0.029]
        )
        face_normals = decoded.normals[decoded.triangles].mean(axis=1)
        np.testing.assert_allclose(
            face_normals,
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            rtol=1e-6,
            atol=1e-12,
        )

        revision_id = metadata["revision_id"]
        assert pipeline.publish_measured_rgbd_surface(
            "building-a", update.submap_id, rgbd
        )["triangle_count"] == 0
        assert pipeline.publish_neural_surface(
            "building-a",
            update.submap_id,
            other_neural,
            other_decision,
            expected_source_generation_id=generation_id,
        )["triangle_count"] == 0
        assert pipeline.tile_store.manifest(scope="local")["tiles"][0][
            "revision_id"
        ] == revision_id
    finally:
        pipeline.close()


def test_surface_publication_validates_every_tile_before_advancing_any_pointer(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        before = {
            tuple(tile["tile_id"]): tile["revision_id"]
            for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
        }
        batch = _surface_batch(
            ((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10)),
            uncertainties=(0.008, 1.1),
        )

        with pytest.raises(ValueError, match="uncertainty"):
            pipeline.publish_measured_rgbd_surface("building-a", update.submap_id, batch)

        after = {
            tuple(tile["tile_id"]): tile["revision_id"]
            for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
        }
        assert after == before
    finally:
        pipeline.close()


def test_surface_publication_rejects_invalid_scale_before_staging_any_tile(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        index_path = pipeline.tile_store.index_path
        before_exists = index_path.is_file()
        before_bytes = _rewrite_index_with_distinct_bytes(index_path)
        original_publish_batch = pipeline.tile_store.publish_batch
        publish_calls = 0

        def record_publish(requests, **kwargs):
            nonlocal publish_calls
            publish_calls += 1
            return original_publish_batch(requests, **kwargs)

        monkeypatch.setattr(pipeline.tile_store, "publish_batch", record_publish)
        batch = _surface_batch(
            ((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10)),
            scales=(0.01, 0.0),
        )

        with pytest.raises(ValueError, match="scale"):
            pipeline.publish_measured_rgbd_surface("building-a", update.submap_id, batch)

        assert publish_calls == 0
        assert index_path.is_file() is before_exists
        assert index_path.read_bytes() == before_bytes
    finally:
        pipeline.close()


def test_surface_publication_rolls_back_all_current_pointers_on_publish_failure(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        index_path = pipeline.tile_store.index_path
        before_exists = index_path.is_file()
        before_bytes = _rewrite_index_with_distinct_bytes(index_path)
        before = {
            tuple(tile["tile_id"]): tile["revision_id"]
            for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
        }
        original_publish_batch = pipeline.tile_store.publish_batch
        staged_entries = []

        def fail_second_publish(requests):
            def fault(phase, entry_index):
                if phase == "before_revision_stage":
                    staged_entries.append(entry_index)
                    assert pipeline._mesh_lock.locked()
                    if entry_index == 1:
                        raise OSError("simulated second tile publication failure")

            return original_publish_batch(requests, fault_injector=fault)

        monkeypatch.setattr(pipeline.tile_store, "publish_batch", fail_second_publish)
        batch = _surface_batch(((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10)))

        with pytest.raises(OSError, match="second tile"):
            pipeline.publish_measured_rgbd_surface("building-a", update.submap_id, batch)

        after = {
            tuple(tile["tile_id"]): tile["revision_id"]
            for tile in pipeline.tile_store.manifest(scope="local")["tiles"]
        }
        assert staged_entries == [0, 1]
        assert after == before
        assert index_path.is_file() is before_exists
        assert index_path.read_bytes() == before_bytes
    finally:
        pipeline.close()


def test_surface_publication_failure_restores_prior_index_nonexistence_and_keeps_staged_revision(
    monkeypatch, tmp_path
):
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        checkpoint_batch_frames=1,
    )
    try:
        assert pipeline.admit(_frame(1)).accepted
        update = pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
        index_path = pipeline.tile_store.index_path
        assert not index_path.exists()
        original_publish_batch = pipeline.tile_store.publish_batch
        staged_entries = []

        def fail_second_publish(requests):
            def fault(phase, entry_index):
                if phase == "before_revision_stage":
                    staged_entries.append(entry_index)
                    if entry_index == 1:
                        raise OSError("simulated second tile publication failure")

            return original_publish_batch(requests, fault_injector=fault)

        monkeypatch.setattr(pipeline.tile_store, "publish_batch", fail_second_publish)
        with pytest.raises(OSError, match="second tile"):
            pipeline.publish_measured_rgbd_surface(
                "building-a",
                update.submap_id,
                _surface_batch(((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10))),
            )

        assert staged_entries == [0, 1]
        assert not index_path.exists()
        assert len(tuple(pipeline.tile_store.revisions_root.iterdir())) == 1
    finally:
        pipeline.close()


@pytest.mark.parametrize("replace_before_failure", [False, True], ids=["before-replace", "after-replace"])
def test_surface_publication_restores_exact_index_when_final_advance_fails(
    monkeypatch, tmp_path, replace_before_failure
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        index_path = pipeline.tile_store.index_path
        before_exists = index_path.is_file()
        before_bytes = _rewrite_index_with_distinct_bytes(index_path)
        revisions_before = set(pipeline.tile_store.revisions_root.iterdir())
        original_publish_batch = pipeline.tile_store.publish_batch
        writes = 0

        failure_phase = "after_index_write" if replace_before_failure else "before_index_write"

        def fail_final_write(requests):
            def fault(phase, _entry_index):
                nonlocal writes
                if phase != failure_phase:
                    return
                writes += 1
                assert pipeline._mesh_lock.locked()
                raise OSError("simulated final pointer advancement failure")

            return original_publish_batch(requests, fault_injector=fault)

        monkeypatch.setattr(pipeline.tile_store, "publish_batch", fail_final_write)
        with pytest.raises(OSError, match="final pointer"):
            pipeline.publish_measured_rgbd_surface(
                "building-a",
                update.submap_id,
                _surface_batch(((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10))),
            )

        assert writes == 1
        assert index_path.is_file() is before_exists
        assert index_path.read_bytes() == before_bytes
        assert set(pipeline.tile_store.revisions_root.iterdir()) > revisions_before
    finally:
        pipeline.close()


def test_successful_multi_tile_publication_advances_one_complete_index(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        index_path = pipeline.tile_store.index_path
        before_bytes = index_path.read_bytes()
        original_publish_batch = pipeline.tile_store.publish_batch
        index_write_phases = []

        def record_complete_write(requests):
            def fault(phase, _entry_index):
                if phase == "before_index_write":
                    assert index_path.read_bytes() == before_bytes
                    index_write_phases.append(phase)
                elif phase == "after_index_write":
                    index_write_phases.append(phase)

            return original_publish_batch(requests, fault_injector=fault)

        monkeypatch.setattr(pipeline.tile_store, "publish_batch", record_complete_write)
        result = pipeline.publish_measured_rgbd_surface(
            "building-a",
            update.submap_id,
            _surface_batch(((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10))),
        )

        assert result["published_tiles"] == 2
        assert index_write_phases == ["before_index_write", "after_index_write"]
        current = json.loads(index_path.read_text(encoding="utf-8"))["current"]
        assert set(result["tile_revision_ids"]).issubset(set(current.values()))
        assert any(key.endswith("/-1/0/0") for key in current)
        assert any(key.endswith("/1/0/0") for key in current)
    finally:
        pipeline.close()


def test_surface_publication_holds_runtime_generation_through_pointer_advance(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    release_publish = threading.Event()
    publish_entered = threading.Event()
    integration_finished = threading.Event()
    original_publish_batch = pipeline.tile_store.publish_batch
    errors = []
    results = []
    generations = []

    def blocking_publish_batch(requests):
        def fault(phase, _entry_index):
            if phase == "before_index_write":
                publish_entered.set()
                release_publish.wait(timeout=3)

        return original_publish_batch(requests, fault_injector=fault)

    monkeypatch.setattr(pipeline.tile_store, "publish_batch", blocking_publish_batch)

    def publish():
        try:
            results.append(
                pipeline.publish_measured_rgbd_surface(
                    "building-a",
                    update.submap_id,
                    _surface_batch(((0.10, 0.10, 0.10),)),
                )
            )
        except Exception as exc:
            errors.append(exc)

    publisher = threading.Thread(target=publish, name="surface-publisher")
    publisher.start()

    def integrate_next_generation():
        try:
            assert pipeline.admit(_frame(2)).accepted
            record = next(
                record
                for record in pipeline.wal.pending()
                if record.frame.identity.sequence == 2
            )
            generations.append(
                pipeline.integrate(record, force_checkpoint=True).checkpoint_generation_id
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            integration_finished.set()

    integrator = threading.Thread(target=integrate_next_generation, name="surface-integrator")
    try:
        assert publish_entered.wait(timeout=2)
        integrator.start()
        assert not integration_finished.wait(timeout=0.2)
    finally:
        release_publish.set()
        publisher.join(timeout=3)
        integrator.join(timeout=3)
        pipeline.close()

    assert not publisher.is_alive()
    assert not integrator.is_alive()
    assert not errors
    assert len(results) == 1
    assert results[0]["source_generation_id"] == update.checkpoint_generation_id
    assert generations[0] != update.checkpoint_generation_id


def test_empty_neural_publication_still_rejects_stale_source_generation(monkeypatch, tmp_path):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        evidence = pipeline.neural_surface_evidence("building-a", update.submap_id)
        assert pipeline.admit(_frame(2)).accepted
        pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
        candidate, decision = _neural_publication(())

        with pytest.raises(ValueError, match="stale"):
            pipeline.publish_neural_surface(
                "building-a",
                update.submap_id,
                candidate,
                decision,
                expected_source_generation_id=evidence["source_generation_id"],
            )
    finally:
        pipeline.close()


def test_pipeline_restores_persisted_surface_when_wal_is_empty(tmp_path):
    wal_root = tmp_path / "wal"
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(wal_root, max_bytes=1_000_000),
        checkpoint_batch_frames=1,
    )
    assert pipeline.admit(_frame(1)).accepted
    pipeline.integrate(pipeline.wal.pending()[0], force_checkpoint=True)
    assert not pipeline.wal.pending()
    pipeline.close()

    restarted = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(wal_root, max_bytes=1_000_000),
        checkpoint_batch_frames=1,
    )
    restore = getattr(restarted, "restore_persisted_runtimes", None)
    assert callable(restore), "pipeline must expose persisted surface restoration"
    assert restore() == 1
    assert restarted.snapshot()["surfel_count"] == 2
    assert restore() == 0
    restarted.close()


def test_pipeline_mesh_binds_tile_to_latest_checkpoint_without_rewriting_geometry(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    frame = _frame(1)
    wal.admit(frame)
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )

    result = pipeline.mesh("building-a", update.submap_id)

    assert result["published_tiles"] == 1
    metadata = pipeline.tile_store.manifest(scope="local")["tiles"][0]
    assert metadata["source_generation_id"] == update.checkpoint_generation_id
    assert len(metadata["pose_revision_id"]) == 64
    assert metadata["pose_revision_id"] != metadata["source_generation_id"]
    assert metadata["geometry_revision_id"] == revision.revision_id
    assert metadata["content_sha256"] == hashlib.sha256(pipeline.tile_store.load(metadata["revision_id"])[1]).hexdigest()


def test_pipeline_mesh_processes_only_dirty_tiles_and_clears_them_after_publication(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    requested = []

    def build_dirty(_self, _surface, *, tiles=None):
        requested.append(set(tiles or ()))
        return SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        )

    monkeypatch.setattr(LiveSurfaceMesher, "build", build_dirty)

    result = pipeline.mesh("building-a", update.submap_id)

    assert requested == [{TileId(0, 0, 0)}]
    assert result["remaining_dirty_tiles"] == 0
    assert pipeline.snapshot()["surfel_count"] == 2
    assert pipeline.mesh("building-a", update.submap_id)["published_tiles"] == 0


def test_meshing_snapshot_contains_only_requested_tile_halo_and_no_witness_payload():
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(
        _frame(
            1,
            points=[
                [0.101, 0.101, 0.101],
                [5.101, 0.101, 0.101],
            ],
        )
    )

    frozen = surface.meshing_surfels_for_tiles({TileId(0, 0, 0)}, halo_m=0.05)

    assert [surfel.key for surfel in frozen] == [(20, 20, 20)]
    assert not hasattr(frozen[0], "witnesses")
    assert not hasattr(frozen[0], "covariance_accumulator")


def test_pipeline_mesh_batches_multiple_ranked_dirty_tiles(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1
    )
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    runtime = pipeline._runtime_for_submap("building-a", update.submap_id)
    ranked = (TileId(0, 0, 0), TileId(1, 0, 0), TileId(2, 0, 0))
    with runtime.lock:
        runtime.mesh_dirty_tiles.update(ranked)
    monkeypatch.setattr(
        runtime.surface,
        "rank_tiles_by_support",
        lambda tiles: tuple(tile for tile in ranked if tile in tiles),
    )
    requested = []
    revision = _mesh_revision()

    def capture_batch(_self, _surface, *, tiles=None):
        requested.append(set(tiles or ()))
        return SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        )

    monkeypatch.setattr(LiveSurfaceMesher, "build", capture_batch)

    result = pipeline.mesh("building-a", update.submap_id)

    assert requested == [set(ranked)]
    assert result["remaining_dirty_tiles"] == 0


def test_pipeline_prunes_zero_support_tiles_when_supported_tiles_are_meshable(
    monkeypatch, tmp_path
):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10
    )
    assert pipeline.admit(_frame(1)).accepted
    pipeline.integrate(wal.pending()[0])
    assert pipeline.admit(_frame(2, origin=(0.06, 0.0, 0.0))).accepted
    second_record = next(
        record for record in wal.pending() if record.frame.identity.sequence == 2
    )
    update = pipeline.integrate(second_record, force_checkpoint=True)
    runtime = pipeline._runtime_for_submap("building-a", update.submap_id)
    supported = TileId(0, 0, 0)
    unresolved = TileId(9, 0, 0)
    with runtime.lock:
        runtime.mesh_dirty_tiles.add(unresolved)
    requested = []
    revision = _mesh_revision()

    def capture_supported(_self, _surface, *, tiles=None):
        requested.append(set(tiles or ()))
        return SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        )

    monkeypatch.setattr(LiveSurfaceMesher, "build", capture_supported)

    result = pipeline.mesh("building-a", update.submap_id)

    assert requested == [{supported}]
    assert result["remaining_dirty_tiles"] == 0


def test_pipeline_offloads_frozen_tile_batch_without_reloading_full_checkpoint(
    monkeypatch, tmp_path
):
    class _InlineExecutor:
        def __init__(self):
            self.calls = []
            self.results = []

        def submit(self, function, *args, **kwargs):
            self.calls.append((function, args, kwargs))
            future = Future()
            try:
                result = function(*args, **kwargs)
                self.results.append(result)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)
            return future

    executor = _InlineExecutor()
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=wal,
        checkpoint_batch_frames=1,
        mesh_executor=executor,
    )
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )

    def reject_full_checkpoint_reload(cls, *args, **kwargs):
        raise AssertionError("meshing must not reload the complete checkpoint")

    monkeypatch.setattr(
        LiveSurfaceMap,
        "load_checkpoint",
        classmethod(reject_full_checkpoint_reload),
    )

    result = pipeline.mesh("building-a", update.submap_id)

    assert result["published_tiles"] == 1
    assert len(executor.calls) == 1
    assert isinstance(executor.results[0], dict)
    assert "tile_revisions" not in executor.results[0]


def test_pipeline_limits_each_meshing_job_to_the_configured_tile_batch(
    monkeypatch, tmp_path
):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=wal,
        checkpoint_batch_frames=1,
        mesh_tile_batch_size=2,
    )
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    runtime = pipeline._runtime_for_submap("building-a", update.submap_id)
    ranked = tuple(TileId(index, 0, 0) for index in range(5))
    with runtime.lock:
        runtime.mesh_dirty_tiles.update(ranked)
    monkeypatch.setattr(
        runtime.surface,
        "rank_tiles_by_support",
        lambda tiles: tuple(tile for tile in ranked if tile in tiles),
    )
    requested = []
    revision = _mesh_revision()

    def capture_batch(_self, _surface, *, tiles=None):
        requested.append(tuple(sorted(tiles or ())))
        return SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        )

    monkeypatch.setattr(LiveSurfaceMesher, "build", capture_batch)

    try:
        result = pipeline.mesh("building-a", update.submap_id)
        assert requested == [ranked[:2]]
        assert result["remaining_dirty_tiles"] == 3
    finally:
        pipeline.close()


def test_pipeline_mesh_flushes_pending_observations_before_binding_source_generation(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    assert pipeline.admit(_frame(1)).accepted
    first = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    assert pipeline.admit(_frame(2)).accepted
    second_record = next(record for record in wal.pending() if record.frame.identity.sequence == 2)
    pipeline.integrate(second_record)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )

    pipeline.mesh("building-a", first.submap_id)

    metadata = pipeline.tile_store.manifest(scope="local")["tiles"][0]
    current_generation = (
        tmp_path / "buildings" / "building-a" / "submaps" / first.submap_id / "CURRENT"
    ).read_text(encoding="ascii").strip()
    assert current_generation != first.checkpoint_generation_id
    assert metadata["source_generation_id"] == current_generation
    assert not wal.pending()


def test_pipeline_meshing_does_not_block_new_frame_admission(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    mesh_started = threading.Event()
    release_mesh = threading.Event()
    admission_finished = threading.Event()
    admission = []

    def blocking_build(_self, _surface, *, tiles=None):
        mesh_started.set()
        assert release_mesh.wait(timeout=3)
        revision = _mesh_revision()
        return SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        )

    monkeypatch.setattr(LiveSurfaceMesher, "build", blocking_build)
    mesh_thread = threading.Thread(target=lambda: pipeline.mesh("building-a", update.submap_id))
    mesh_thread.start()
    assert mesh_started.wait(timeout=2)

    def admit_next():
        admission.append(pipeline.admit(_frame(2)))
        admission_finished.set()

    admission_thread = threading.Thread(target=admit_next)
    admission_thread.start()
    try:
        assert admission_finished.wait(timeout=0.25), "meshing held the live admission lock"
        assert admission[0].accepted is True
    finally:
        release_mesh.set()
        mesh_thread.join(timeout=3)
        admission_thread.join(timeout=3)
        pipeline.close()


def test_checkpoint_reload_during_integration_does_not_block_admission(monkeypatch, tmp_path):
    first_wal = LiveKeyframeWal(tmp_path / "first-wal", max_bytes=1_000_000)
    first = LiveSurfacePipeline(tmp_path / "buildings", wal=first_wal, checkpoint_batch_frames=1)
    assert first.admit(_frame(1)).accepted
    first.integrate(first_wal.pending()[0], force_checkpoint=True)

    replay_wal = LiveKeyframeWal(tmp_path / "replay-wal", max_bytes=1_000_000)
    assert replay_wal.admit(_frame(2)).accepted
    restarted = LiveSurfacePipeline(tmp_path / "buildings", wal=replay_wal, checkpoint_batch_frames=1)
    load_started = threading.Event()
    release_load = threading.Event()
    admission_finished = threading.Event()
    original_load = LiveSurfaceMap.load_checkpoint.__func__

    def blocking_load(cls, *args, **kwargs):
        load_started.set()
        assert release_load.wait(timeout=3)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(LiveSurfaceMap, "load_checkpoint", classmethod(blocking_load))
    integration_thread = threading.Thread(
        target=lambda: restarted.integrate(replay_wal.pending()[0], force_checkpoint=True)
    )
    integration_thread.start()
    assert load_started.wait(timeout=2)

    def admit_next():
        restarted.admit(_frame(3))
        admission_finished.set()

    admission_thread = threading.Thread(target=admit_next)
    admission_thread.start()
    try:
        assert admission_finished.wait(timeout=0.25), "checkpoint reload held the admission lock"
    finally:
        release_load.set()
        integration_thread.join(timeout=3)
        admission_thread.join(timeout=3)
        restarted.close()


def test_surface_integration_does_not_block_admission(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    assert pipeline.admit(_frame(1)).accepted
    integration_started = threading.Event()
    release_integration = threading.Event()
    admission_finished = threading.Event()
    original_integrate = LiveSurfaceMap.integrate

    def blocking_integrate(self, frame):
        integration_started.set()
        assert release_integration.wait(timeout=3)
        return original_integrate(self, frame)

    monkeypatch.setattr(LiveSurfaceMap, "integrate", blocking_integrate)
    integration_thread = threading.Thread(target=lambda: pipeline.integrate(wal.pending()[0]))
    integration_thread.start()
    assert integration_started.wait(timeout=2)

    admission_thread = threading.Thread(
        target=lambda: (pipeline.admit(_frame(2)), admission_finished.set())
    )
    admission_thread.start()
    try:
        assert admission_finished.wait(timeout=0.25), "surface integration held the admission lock"
    finally:
        release_integration.set()
        integration_thread.join(timeout=3)
        admission_thread.join(timeout=3)
        pipeline.close()


def test_surface_integration_does_not_block_stable_snapshot(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    assert pipeline.admit(_frame(1)).accepted
    pipeline.integrate(wal.pending()[0])
    assert pipeline.admit(_frame(2)).accepted
    second = next(record for record in wal.pending() if record.frame.identity.sequence == 2)
    integration_started = threading.Event()
    release_integration = threading.Event()
    snapshot_finished = threading.Event()
    snapshots = []
    original_integrate = LiveSurfaceMap.integrate

    def blocking_integrate(self, frame):
        if frame.identity.sequence == 2:
            integration_started.set()
            assert release_integration.wait(timeout=3)
        return original_integrate(self, frame)

    monkeypatch.setattr(LiveSurfaceMap, "integrate", blocking_integrate)
    integration_thread = threading.Thread(target=lambda: pipeline.integrate(second))
    integration_thread.start()
    assert integration_started.wait(timeout=2)

    def read_snapshot():
        snapshots.append(pipeline.snapshot())
        snapshot_finished.set()

    snapshot_thread = threading.Thread(target=read_snapshot)
    snapshot_thread.start()
    try:
        assert snapshot_finished.wait(timeout=0.25), "surface integration held the snapshot routing lock"
        assert snapshots[0]["surfel_count"] > 0
    finally:
        release_integration.set()
        integration_thread.join(timeout=3)
        snapshot_thread.join(timeout=3)
        pipeline.close()


def test_pipeline_close_waits_for_active_surface_integration(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    assert pipeline.admit(_frame(1)).accepted
    integration_started = threading.Event()
    release_integration = threading.Event()
    close_finished = threading.Event()
    original_integrate = LiveSurfaceMap.integrate

    def blocking_integrate(self, frame):
        integration_started.set()
        assert release_integration.wait(timeout=3)
        return original_integrate(self, frame)

    monkeypatch.setattr(LiveSurfaceMap, "integrate", blocking_integrate)
    integration_thread = threading.Thread(target=lambda: pipeline.integrate(wal.pending()[0]))
    integration_thread.start()
    assert integration_started.wait(timeout=2)
    close_thread = threading.Thread(target=lambda: (pipeline.close(), close_finished.set()))
    close_thread.start()
    try:
        assert not close_finished.wait(timeout=0.25), "pipeline close abandoned active integration"
        with pytest.raises(RuntimeError, match="pipeline is closed"):
            pipeline.admit(_frame(2))
    finally:
        release_integration.set()
        integration_thread.join(timeout=3)
        close_thread.join(timeout=3)
    assert close_finished.is_set()


def test_pipeline_building_manifest_is_empty_until_an_assembly_is_published(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)

    assert len(pipeline.manifest(scope="local")["tiles"]) == 1
    building = pipeline.manifest(scope="building")
    assert building["scope"] == "building"
    assert building["tiles"] == []
    assert pipeline.manifest(scope="Building")["tiles"] == []


def test_pipeline_building_manifest_resolves_current_assembly_while_local_stays_source_scoped(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    frame = _frame(1)
    wal.admit(frame)
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)
    closed = pipeline.close_session_source("building-a", "session-a", 1)
    project = pipeline._projects["building-a"]
    submap = project.submap(update.submap_id)
    component = project.component(submap["component_id"])
    member = AssemblyMember(
        "building-a",
        component["component_id"],
        "session-a",
        update.submap_id,
        closed["generation_id"],
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    pipeline.publish_assembly("building-a", (member,), algorithm_version="test")

    local = pipeline.manifest(scope="local")
    building = pipeline.manifest(scope="building")

    assert "placement_revision_id" not in local["tiles"][0]
    assert building["assembly_id"]
    assert building["tiles"][0]["placement_revision_id"]
    assert building["tiles"][0]["transform_building_from_submap"] == list(member.transform_building_from_submap)

    restarted = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "restarted-wal", max_bytes=1_000_000),
    )
    restarted_building = restarted.manifest(scope="building")

    assert restarted_building["assembly_id"] == building["assembly_id"]
    assert restarted_building["tiles"][0]["placement_revision_id"] == building["tiles"][0]["placement_revision_id"]

    restarted.mesh("building-a", update.submap_id)
    remeshed = restarted.tile_store.manifest(scope="local")["tiles"]
    assert any(tile["source_generation_id"] == closed["generation_id"] for tile in remeshed)


def test_pipeline_publishes_and_reloads_complete_building_appearance(monkeypatch, tmp_path):
    pipeline, appearance_store, assembly, member = _publish_pipeline_building_appearance(
        monkeypatch, tmp_path
    )

    published = pipeline.publish_building_appearance(
        "building-a", assembly.assembly_id, (member,)
    )
    restarted = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "restart-wal", max_bytes=1_000_000),
        tile_appearance_store=appearance_store,
    )

    assert restarted.current_building_appearance(
        "building-a", expected_assembly_id=assembly.assembly_id
    ) == published
    assert restarted.resolve_building_appearance(
        "building-a", published.building_appearance_revision_id
    ) == published


def test_pipeline_rejects_incomplete_or_stale_building_appearance_without_replacing_current(
    monkeypatch, tmp_path
):
    pipeline, _, assembly, member = _publish_pipeline_building_appearance(monkeypatch, tmp_path)
    published = pipeline.publish_building_appearance(
        "building-a", assembly.assembly_id, (member,)
    )

    with pytest.raises(ValueError, match="complete visible tile set"):
        pipeline.publish_building_appearance("building-a", assembly.assembly_id, ())
    stale_member = BuildingAppearanceMember(
        **{**member.__dict__, "source_generation_id": hashlib.sha256(b"stale").hexdigest()}
    )
    with pytest.raises(ValueError, match="current assembly|source generation"):
        pipeline.publish_building_appearance(
            "building-a", assembly.assembly_id, (stale_member,)
        )

    replacement = pipeline.publish_assembly(
        "building-a", assembly.members, algorithm_version="pipeline-test-v2"
    )
    with pytest.raises(ValueError, match="current assembly"):
        pipeline.publish_building_appearance(
            "building-a", assembly.assembly_id, (member,)
        )
    assert replacement.assembly_id != assembly.assembly_id
    with pytest.raises(ValueError, match="stale"):
        pipeline.current_building_appearance("building-a")
    assert pipeline.resolve_building_appearance(
        "building-a", published.building_appearance_revision_id
    ) == published


def test_pipeline_restart_uses_checkpoint_journal_to_integrate_replayed_wal_exactly_once(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    frame = _frame(7)
    wal.admit(frame)
    first = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    first.integrate(wal.pending()[0], force_checkpoint=True)

    # Simulate the durable surface surviving while WAL state is rebuilt from an
    # exact replay record after a crash around cursor publication.
    replay_wal = LiveKeyframeWal(tmp_path / "replay-wal", max_bytes=1_000_000)
    replay_wal.admit(frame)
    restarted = LiveSurfacePipeline(tmp_path / "buildings", wal=replay_wal, checkpoint_batch_frames=1)
    update = restarted.integrate(replay_wal.pending()[0], force_checkpoint=True)

    assert update.duplicate_surface_observation is True
    assert restarted.snapshot()["surfel_count"] == 2
    assert not replay_wal.pending()


def test_pipeline_isolates_identical_stream_sequences_in_a_new_session(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    first = _frame(1, session_id="session-a")
    second = _frame(1, session_id="session-b")

    assert pipeline.admit(first).accepted
    first_update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    assert pipeline.admit(second).accepted
    second_record = next(record for record in wal.pending() if record.frame.identity.session_id == "session-b")
    second_update = pipeline.integrate(second_record, force_checkpoint=True)

    assert first_update.submap_id != second_update.submap_id
    assert pipeline.snapshot()["active_attachment_state"] == "DETACHED"
    assert pipeline.snapshot()["surfel_count"] == 4


def test_close_session_source_seals_exact_surface_and_rejects_late_admission(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    frame = _frame(1)
    assert pipeline.admit(frame).accepted
    pipeline.integrate(wal.pending()[0])

    closed = pipeline.close_session_source("building-a", "session-a", 1)

    assert closed["route_state"] == "CLOSED"
    assert closed["lifecycle"] == "SEALED"
    assert len(closed["generation_id"]) == 64
    assert pipeline.snapshot()["active_submap_id"] is None
    assert not wal.pending()
    with pytest.raises(ValueError, match="closed|CLOSING|CLOSED"):
        pipeline.admit(_frame(2))


def test_close_session_source_rejects_cutoff_below_admitted_sequence_without_state_change(tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal)
    assert pipeline.admit(_frame(1)).accepted
    assert pipeline.admit(_frame(2)).accepted

    with pytest.raises(ValueError, match="cutoff"):
        pipeline.close_session_source("building-a", "session-a", 1)

    assert pipeline.snapshot()["active_submap_id"] is not None
    assert [record.frame.identity.sequence for record in wal.pending()] == [1, 2]


def test_close_session_source_recovers_from_closing_after_checkpoint_failure(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    assert pipeline.admit(_frame(1)).accepted
    pipeline.integrate(wal.pending()[0])
    runtime = next(iter(pipeline._runtimes.values()))
    original_checkpoint = runtime.surface.checkpoint

    def fail_checkpoint(*args, **kwargs):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(runtime.surface, "checkpoint", fail_checkpoint)
    with pytest.raises(OSError, match="simulated"):
        pipeline.close_session_source("building-a", "session-a", 1)
    with pytest.raises(ValueError, match="cutoff"):
        pipeline.close_session_source("building-a", "session-a", 2)
    assert [record.frame.identity.sequence for record in wal.pending()] == [1]

    monkeypatch.setattr(runtime.surface, "checkpoint", original_checkpoint)
    restarted = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=10)
    closed = restarted.close_session_source("building-a", "session-a", 1)

    assert closed["route_state"] == "CLOSED"
    assert closed["lifecycle"] == "SEALED"
    assert not wal.pending()


def test_appearance_worker_is_bounded_coalesced_and_never_blocks_lidar(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    processed = []

    def process(metadata):
        processed.append(metadata["revision_id"])
        started.set()
        release.wait(timeout=5)

    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=wal,
        checkpoint_batch_frames=1,
        appearance_processor=process,
    )
    assert pipeline.admit(_frame(1)).accepted
    first = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", first.submap_id)
    assert started.wait(timeout=2)

    started_at = time.monotonic()
    assert pipeline.admit(_frame(2)).accepted
    second_record = next(record for record in wal.pending() if record.frame.identity.sequence == 2)
    pipeline.integrate(second_record, force_checkpoint=True)
    elapsed = time.monotonic() - started_at

    metadata = pipeline.tile_store.manifest(scope="local")["tiles"][0]
    older = dict(metadata)
    older["revision_id"] = "8" * 64
    older["source_generation_id"] = "8" * 64
    newer = dict(metadata)
    newer["revision_id"] = "9" * 64
    newer["source_generation_id"] = "9" * 64
    pipeline.schedule_appearance(older)
    pipeline.schedule_appearance(newer)
    snapshot = pipeline.appearance_snapshot()

    assert elapsed < 0.5
    assert snapshot["pending_jobs"] == 1
    assert snapshot["replaced_jobs"] >= 1
    release.set()
    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert processed[-1] == "9" * 64
    pipeline.close()


def test_appearance_worker_discovers_published_tiles_after_restart(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    first = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert first.admit(_frame(1)).accepted
    update = first.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    first.mesh("building-a", update.submap_id)
    expected = first.tile_store.manifest(scope="local")["tiles"][0]["revision_id"]

    processed = []
    restarted = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "restart-wal", max_bytes=1_000_000),
        appearance_processor=lambda metadata: processed.append(metadata["revision_id"]),
    )

    assert restarted.wait_for_appearance_idle(timeout=3)
    assert processed == [expected]
    assert restarted.appearance_snapshot()["discovered_jobs"] == 1
    restarted.close()


def test_appearance_worker_can_be_configured_after_pipeline_construction(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    first = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert first.admit(_frame(1)).accepted
    update = first.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    first.mesh("building-a", update.submap_id)
    expected = first.tile_store.manifest(scope="local")["tiles"][0]["revision_id"]

    processed = []
    restarted = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "restart-wal", max_bytes=1_000_000),
    )
    restarted.configure_appearance_processor(
        lambda metadata: processed.append(metadata["revision_id"])
    )

    assert restarted.wait_for_appearance_idle(timeout=3)
    assert processed == [expected]
    assert restarted.appearance_snapshot()["discovered_jobs"] == 1
    with pytest.raises(ValueError, match="already configured"):
        restarted.configure_appearance_processor(lambda metadata: None)
    restarted.close()


def test_new_rgbd_evidence_revision_reschedules_completed_immutable_tile(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal, checkpoint_batch_frames=1)
    assert pipeline.admit(_frame(1)).accepted
    update = pipeline.integrate(wal.pending()[0], force_checkpoint=True)
    revision = _mesh_revision()
    monkeypatch.setattr(
        LiveSurfaceMesher,
        "build",
        lambda self, surface, *, tiles=None: SimpleNamespace(
            tile_revisions={revision.tile_id: revision},
            triangle_count=1,
            unresolved_surfel_count=0,
        ),
    )
    pipeline.mesh("building-a", update.submap_id)

    class Processor:
        evidence_revision = "a" * 64

        def scheduling_revision(self):
            return self.evidence_revision

        def __call__(self, metadata):
            processed.append((metadata["revision_id"], metadata["appearance_evidence_revision_id"]))

    processed = []
    processor = Processor()
    pipeline.configure_appearance_processor(processor)
    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert len(processed) == 1

    assert pipeline.refresh_appearance_evidence() == 0
    processor.evidence_revision = "b" * 64
    assert pipeline.refresh_appearance_evidence() == 1
    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert processed == [
        (processed[0][0], "a" * 64),
        (processed[0][0], "b" * 64),
    ]
    pipeline.close()


def test_appearance_discovery_keeps_one_pending_job_per_distinct_tile(tmp_path):
    started = threading.Event()
    release = threading.Event()
    processed = []

    def processor(metadata):
        started.set()
        release.wait(timeout=3)
        processed.append(metadata["revision_id"])

    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        appearance_processor=processor,
    )
    first = {
        "revision_id": "a" * 64,
        "source_generation_id": "c" * 64,
    }
    second = {
        "revision_id": "b" * 64,
        "source_generation_id": "c" * 64,
    }

    assert pipeline.schedule_appearance(first) is True
    assert started.wait(timeout=3)
    assert pipeline.schedule_appearance(second) is True
    release.set()

    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert processed == [first["revision_id"], second["revision_id"]]
    pipeline.close()


def test_appearance_worker_retries_transient_failure_and_refreshes_exhausted_job(tmp_path):
    attempts = 0

    class Processor:
        def scheduling_revision(self):
            return "d" * 64

        def __call__(self, metadata):
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise OSError("transient appearance failure")

    pipeline = LiveSurfacePipeline(
        tmp_path / "buildings",
        wal=LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000),
        appearance_processor=Processor(),
    )
    metadata = {
        "revision_id": "a" * 64,
        "source_generation_id": "c" * 64,
    }
    pipeline.tile_store.manifest = lambda **_: {"tiles": [metadata]}

    assert pipeline.schedule_appearance(metadata) is True
    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert attempts == 3
    assert pipeline.appearance_snapshot()["failed_jobs"] == 3

    assert pipeline.refresh_appearance_evidence() == 1
    assert pipeline.wait_for_appearance_idle(timeout=3)
    assert attempts == 4
    assert pipeline.appearance_snapshot()["processed_jobs"] == 1
    pipeline.close()


def test_second_tile_appearance_scheduling_failure_does_not_fail_geometry_commit(
    monkeypatch, tmp_path
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    try:
        pipeline.configure_appearance_processor(lambda metadata: None)
        assert pipeline.wait_for_appearance_idle(timeout=3)
        original_schedule = pipeline.schedule_appearance
        schedule_calls = 0

        def fail_second_schedule(metadata):
            nonlocal schedule_calls
            schedule_calls += 1
            if schedule_calls == 2:
                raise OSError("simulated second tile appearance scheduling failure")
            return original_schedule(metadata)

        monkeypatch.setattr(pipeline, "schedule_appearance", fail_second_schedule)
        result = pipeline.publish_measured_rgbd_surface(
            "building-a",
            update.submap_id,
            _surface_batch(((-0.103, 0.10, 0.10), (0.30, 0.10, 0.10))),
        )

        assert result["published_tiles"] == 2
        assert schedule_calls == 2
        current_metadata = pipeline.tile_store.manifest(scope="local")["tiles"]
        current_ids = {tile["revision_id"] for tile in current_metadata}
        assert set(result["tile_revision_ids"]).issubset(current_ids)
        assert len(
            [
                tile
                for tile in current_metadata
                if tile["revision_id"] in result["tile_revision_ids"]
            ]
        ) == 2
        failed = pipeline.appearance_snapshot()
        assert failed["schedule_failed_jobs"] == 1
        assert "second tile appearance" in failed["schedule_last_error"]
        assert failed["refresh_required"] is True

        monkeypatch.setattr(pipeline, "schedule_appearance", original_schedule)
        assert pipeline.refresh_appearance_evidence() >= 1
        assert pipeline.wait_for_appearance_idle(timeout=3)
        refreshed = pipeline.appearance_snapshot()
        assert refreshed["refresh_required"] is False
        assert refreshed["schedule_last_error"] is None
    finally:
        pipeline.close()


@pytest.mark.parametrize("publication", ["mesh", "rgbd", "neural"])
def test_appearance_revision_resolver_runs_outside_geometry_locks(
    monkeypatch, tmp_path, publication
):
    pipeline, update = _meshed_surface_pipeline(monkeypatch, tmp_path)
    runtime = pipeline._runtime_for_submap("building-a", update.submap_id)
    calls = []
    violations = []

    class Processor:
        def scheduling_revision(self):
            calls.append(publication)
            try:
                assert not pipeline._mesh_lock.locked()
                assert not runtime.lock._is_owned()
            except AssertionError as exc:
                violations.append(exc)
                raise
            return "e" * 64

        def __call__(self, metadata):
            return None

    try:
        pipeline.configure_appearance_processor(Processor())
        assert pipeline.wait_for_appearance_idle(timeout=3)
        calls.clear()
        violations.clear()

        if publication == "mesh":
            pipeline.mesh("building-a", update.submap_id)
        elif publication == "rgbd":
            pipeline.publish_measured_rgbd_surface(
                "building-a", update.submap_id, _surface_batch(((0.30, 0.10, 0.10),))
            )
        else:
            candidate, decision = _neural_publication(((0.30, 0.10, 0.10),))
            pipeline.publish_neural_surface(
                "building-a",
                update.submap_id,
                candidate,
                decision,
                expected_source_generation_id=update.checkpoint_generation_id,
            )

        assert calls == [publication]
        assert violations == []
    finally:
        pipeline.close()


def test_close_waits_for_inflight_mutation_and_rejects_late_thread_work(monkeypatch, tmp_path):
    wal = LiveKeyframeWal(tmp_path / "wal", max_bytes=1_000_000)
    pipeline = LiveSurfacePipeline(tmp_path / "buildings", wal=wal)
    entered = threading.Event()
    release = threading.Event()
    close_finished = threading.Event()
    original_route = pipeline._ensure_session_route

    def blocking_route(frame):
        entered.set()
        release.wait(timeout=3)
        return original_route(frame)

    monkeypatch.setattr(pipeline, "_ensure_session_route", blocking_route)
    admission_thread = threading.Thread(target=lambda: pipeline.admit(_frame(1)))
    admission_thread.start()
    assert entered.wait(timeout=3)

    close_thread = threading.Thread(
        target=lambda: (pipeline.close(), close_finished.set())
    )
    close_thread.start()
    assert not close_finished.wait(timeout=0.1)
    release.set()
    admission_thread.join(timeout=3)
    close_thread.join(timeout=3)

    assert close_finished.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.admit(_frame(2))
