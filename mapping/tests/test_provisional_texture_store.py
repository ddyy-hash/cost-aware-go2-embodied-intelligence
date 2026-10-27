import base64
import hashlib
import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_protocol import validate_texture_keyframe
from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision
from mapping.pc.go2_mapping_live.live_tiles import ImmutableTileStore
from mapping.pc.go2_mapping_live.texture_wal import TextureKeyframeWal
from mapping.pc.go2_mapping_appearance.provisional_store import ProvisionalTextureStore
from mapping.pc.go2_mapping_appearance.render_manifest import AtomicRenderManifestStore
from mapping.pc.go2_mapping_dashboard import server


def _frame(sequence: int = 0, *, pose_x: float = 0.0):
    jpeg = b"\xff\xd8" + f"texture-{sequence}".encode() * 20 + b"\xff\xd9"
    pose = np.eye(4)
    pose[0, 3] = pose_x
    return validate_texture_keyframe({
        "message_type": "texture_keyframe",
        "protocol_version": 3,
        "building_id": "building-a",
        "session_id": "session-a",
        "map_id": "go2-live",
        "producer_instance_id": "producer-a",
        "producer_epoch": 1,
        "texture_sequence": sequence,
        "timestamp_ns": 1_000_000_000 + sequence,
        "calibration_id": "calibration-a",
        "source_config_sha256": "a" * 64,
        "content_sha256": hashlib.sha256(jpeg).hexdigest(),
        "source": {
            "topic": "/camera/color/image_raw",
            "frame_id": "camera_color_optical_frame",
            "encoding": "bgr8",
            "width": 424,
            "height": 240,
        },
        "color": {
            "encoding": "jpeg-base64",
            "jpeg_b64": base64.b64encode(jpeg).decode(),
            "width": 424,
            "height": 240,
        },
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 211.5,
            "cy": 119.5,
            "width": 424,
            "height": 240,
        },
        "lio_pose_timestamp_ns": 1_000_000_000 + sequence,
        "pose_skew_ns": 0,
        "pose_association_clock": "subscriber_monotonic",
        "pose_map_from_lidar": pose.tolist(),
        "quality": {
            "selection_accepted": True,
            "sharpness_laplacian": 90.0 - sequence,
            "mean_luma": 112.0,
            "clipped_fraction": 0.01,
            "viewpoint_baseline_m": 0.1,
            "independent_view_id": f"view-{sequence}",
        },
    })


def _tile_body():
    revision = MeshTileRevision(
        TileId(0, 0, 0),
        "b" * 64,
        np.asarray([[-0.1, -0.1, 2.0], [0.1, -0.1, 2.0], [0.0, 0.1, 2.0]], dtype="<f4"),
        np.asarray([[0.0, 0.0, -1.0]] * 3, dtype="<f4"),
        np.asarray([[0, 1, 2]], dtype="<u4"),
        np.asarray([0.005], dtype="<f4"),
        0,
    )
    return ImmutableTileStore._body(revision)


def test_provisional_texture_store_is_content_addressed_projectable_and_never_final(tmp_path):
    store = ProvisionalTextureStore(tmp_path / "provisional")
    camera_from_lidar = np.eye(4)

    published = store.ingest(
        _frame(),
        camera_from_lidar=camera_from_lidar,
        calibration_authority=False,
        submap_id="submap-a",
    )
    duplicate = store.ingest(
        _frame(),
        camera_from_lidar=camera_from_lidar,
        calibration_authority=False,
        submap_id="submap-a",
    )
    loaded = store.load(published.revision_id)

    assert duplicate == published
    assert loaded.manifest["texture_state"] == "PROVISIONAL"
    assert loaded.manifest["final_store_eligible"] is False
    assert loaded.manifest["calibration_authority"] is False
    assert loaded.color_path.read_bytes() == _frame().color_jpeg
    assert loaded.manifest["content_sha256"] == _frame().content_sha256
    assert loaded.manifest["submap_id"] == "submap-a"
    assert loaded.manifest["camera_from_map_row_major"] == pytest.approx(np.eye(4).reshape(-1).tolist())

    assignment = store.project_and_publish(
        tile_revision_id="c" * 64,
        tile_body=_tile_body(),
        observations=[published.revision_id],
        submap_id="submap-a",
    )
    assert assignment.manifest["texture_state"] == "PROVISIONAL"
    assert assignment.manifest["visible_triangle_count"] == 1
    assert assignment.visibility_mask_path.read_bytes() == b"\x01"
    assert assignment.manifest["observation_revision_id"] == published.revision_id
    assert assignment.manifest["submap_id"] == "submap-a"


def test_unresolved_extrinsics_are_retained_but_cannot_be_projected(tmp_path):
    store = ProvisionalTextureStore(tmp_path / "provisional")
    observation = store.ingest(
        _frame(), camera_from_lidar=None, calibration_authority=False, submap_id="submap-a"
    )

    assert observation.manifest["projectable"] is False
    with pytest.raises(ValueError, match="projectable"):
        store.project_and_publish(
            tile_revision_id="c" * 64,
            tile_body=_tile_body(),
            observations=[observation.revision_id],
            submap_id="submap-a",
        )


def test_provisional_projection_rejects_cross_submap_observations(tmp_path):
    store = ProvisionalTextureStore(tmp_path / "provisional")
    observation = store.ingest(
        _frame(),
        camera_from_lidar=np.eye(4),
        calibration_authority=False,
        submap_id="submap-a",
    )

    with pytest.raises(ValueError, match="submap"):
        store.project_and_publish(
            tile_revision_id="c" * 64,
            tile_body=_tile_body(),
            observations=[observation.revision_id],
            submap_id="submap-b",
        )


def test_atomic_render_manifest_references_geometry_provisional_validated_and_trajectory_layers(tmp_path):
    provisional = ProvisionalTextureStore(tmp_path / "provisional")
    observation = provisional.ingest(
        _frame(),
        camera_from_lidar=np.eye(4),
        calibration_authority=False,
        submap_id="submap-a",
    )
    assignment = provisional.project_and_publish(
        tile_revision_id="c" * 64,
        tile_body=_tile_body(),
        observations=[observation.revision_id],
        submap_id="submap-a",
    )
    store = AtomicRenderManifestStore(tmp_path / "render", provisional_store=provisional)
    surface_manifest = {
        "schema_version": 2,
        "scope": "building",
        "assembly_id": "d" * 64,
        "tiles": [{
            "revision_id": "c" * 64,
            "geometry_revision_id": "e" * 64,
            "binary_url": "/api/building/surface/tiles/" + "c" * 64,
            "content_sha256": "f" * 64,
            "binary_schema_version": 2,
            "floor_id": "floor-01",
            "submap_id": "submap-a",
            "tile_id": [0, 0, 0],
        }],
    }
    validated = {
        "building_appearance_revision_id": "1" * 64,
        "members": [{
            "tile_revision_id": "c" * 64,
            "tile_appearance_revision_id": "2" * 64,
            "authorized_lod_ids": ["lod-0"],
        }],
    }

    published = store.publish(
        building_id="building-a",
        surface_manifest=surface_manifest,
        provisional_assignments={"c" * 64: assignment.manifest},
        validated_appearance_manifest=validated,
        trajectories={"revision_id": "3" * 64, "url": "/api/building/trajectories?building_id=building-a"},
    )
    current = store.current("building-a")

    assert current == published
    assert published.manifest["scene_revision_id"] == published.revision_id
    tile = published.manifest["tiles"][0]
    assert tile["geometry"]["binary_schema_version"] == 2
    assert tile["provisional_texture"]["texture_state"] == "PROVISIONAL"
    assert tile["validated_texture"]["tile_appearance_revision_id"] == "2" * 64
    assert tile["texture_composition"] == "VALIDATED_MASK_OVER_PROVISIONAL"
    assert published.manifest["trajectories"]["revision_id"] == "3" * 64
    serialized = json.dumps(published.manifest).lower()
    assert "preview_points" not in serialized
    assert "point_arrays" not in serialized
    assert published.manifest["render_contract"]["backend_only"] == [
        "LIDAR_POINTS", "RGBD_DEPTH", "DA3_DEPTH", "VOXELS",
    ]

    mismatched = {**assignment.manifest, "submap_id": "submap-b"}
    with pytest.raises(ValueError, match="submap"):
        store.publish(
            building_id="building-a",
            surface_manifest=surface_manifest,
            provisional_assignments={"c" * 64: mismatched},
            validated_appearance_manifest=validated,
            trajectories={"revision_id": "3" * 64, "url": "/api/building/trajectories?building_id=building-a"},
        )


def test_texture_wal_consumer_resolves_current_extrinsics_and_commits_after_provisional_fsync(monkeypatch, tmp_path):
    wal = TextureKeyframeWal(tmp_path / "texture-wal", max_bytes=2_000_000)
    frame = _frame()
    assert wal.admit(frame).accepted is True
    provisional = ProvisionalTextureStore(tmp_path / "provisional")

    monkeypatch.setattr(server, "_texture_wal", wal)
    monkeypatch.setattr(server, "_provisional_texture_store", provisional)
    monkeypatch.setattr(
        server._trajectory_store,
        "resolve_submap_near",
        lambda **_kwargs: {"submap_id": "submap-a", "floor_id": "floor-01", "skew_ns": 0},
        raising=False,
    )
    monkeypatch.setattr(server.quality_gate, "snapshot", lambda: {
        "calibration_id": "calibration-a",
        "camera_from_lidar_row_major": np.eye(4).reshape(-1).tolist(),
        "eligible_for": {"rgbd_or_neural_geometry_fusion": False},
    })

    revision = server._process_texture_record(wal.pending()[0])

    assert len(revision) == 64
    assert wal.pending() == []
    published = provisional.load(revision)
    assert published.manifest["projectable"] is True
    assert published.manifest["calibration_authority"] is False
    assert published.manifest["submap_id"] == "submap-a"


def test_render_and_provisional_artifact_endpoints_return_verified_immutable_bytes(monkeypatch, tmp_path):
    provisional = ProvisionalTextureStore(tmp_path / "provisional")
    observation = provisional.ingest(
        _frame(),
        camera_from_lidar=np.eye(4),
        calibration_authority=False,
        submap_id="submap-a",
    )
    assignment = provisional.project_and_publish(
        tile_revision_id="c" * 64,
        tile_body=_tile_body(),
        observations=[observation.revision_id],
        submap_id="submap-a",
    )
    render = AtomicRenderManifestStore(tmp_path / "render", provisional_store=provisional)
    render.publish(
        building_id="building-a",
        surface_manifest={"schema_version": 2, "scope": "building", "tiles": []},
        provisional_assignments={},
        validated_appearance_manifest=None,
        trajectories={"revision_id": "3" * 64, "url": "/api/building/trajectories?building_id=building-a"},
    )
    monkeypatch.setattr(server, "_provisional_texture_store", provisional)
    monkeypatch.setattr(server, "_render_manifest_store", render)

    manifest_response = server.building_render_manifest("building-a")
    color_response = server.provisional_texture_color(observation.revision_id)
    mask_response = server.provisional_texture_visibility(assignment.revision_id)

    assert json.loads(manifest_response.body)["building_id"] == "building-a"
    assert manifest_response.headers["x-go2-scene-revision"]
    assert color_response.body == observation.color_path.read_bytes()
    assert color_response.headers["cache-control"].endswith("immutable")
    assert mask_response.body == b"\x01"
