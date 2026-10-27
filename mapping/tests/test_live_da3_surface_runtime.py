import base64
from io import BytesIO
import hashlib
import zlib

import numpy as np
from PIL import Image
import pytest

from mapping.pc.go2_mapping_live.live_protocol import validate_texture_keyframe
from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision, TriangleProvenance
from mapping.pc.go2_mapping_live.live_tiles import ImmutableTileStore
from mapping.pc.go2_mapping_rgbd.depth_anything import MultiViewDepthFrame
from mapping.pc.go2_mapping_rgbd.multiview_surface import Da3ViewObservation, Da3WindowAssembler
from mapping.pc.go2_mapping_rgbd.neural_surface_runtime import (
    Da3SurfaceCandidateBuilder,
    LiveDa3SurfaceRuntime,
    build_measured_rgbd_surface,
)


def _observation(index: int, *, with_depth: bool = True) -> Da3ViewObservation:
    height, width = 6, 8
    camera_from_world = np.eye(4, dtype=np.float32)
    camera_from_world[0, 3] = -index * 0.01
    depth = np.full((height, width), np.nan, dtype=np.float32)
    if with_depth:
        depth[:, :4] = 2.0
    return Da3ViewObservation(
        frame_sha256=hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
        independent_view_id=f"view-{index}",
        timestamp_ns=1_000_000_000 + index,
        rectified_rgb=np.full((height, width, 3), 30 + index, dtype=np.uint8),
        intrinsics=np.asarray([[100.0, 0.0, 3.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        extrinsics_world_to_camera=camera_from_world,
        aligned_depth_m=depth,
        building_id="building-a",
        session_id="session-a",
        submap_id="submap-a",
    )


def _window():
    assembler = Da3WindowAssembler(window_size=5, overlap=2)
    window = None
    for index in range(5):
        window = assembler.submit(_observation(index))
    assert window is not None
    return window


def _prediction(window):
    return MultiViewDepthFrame(
        depth_m=np.full((5, 6, 8), 2.0, dtype=np.float32),
        model_revision_id="a" * 64,
        preprocessing_revision="depth-anything-3-preprocess-v1",
        frame_sha256s=tuple(view.frame_sha256 for view in window.views),
        intrinsics=np.stack([view.intrinsics for view in window.views]),
        extrinsics_world_to_camera=np.stack([view.extrinsics_world_to_camera for view in window.views]),
    )


def test_aligned_depth_builds_only_observed_front_rgbd_triangles():
    surface = build_measured_rgbd_surface(_observation(0))

    assert len(surface.triangles) > 0
    assert np.all(surface.triangle_provenance == TriangleProvenance.MEASURED_RGBD)
    assert np.all(surface.positions[:, 0] < 0.02)
    assert np.all(surface.triangle_uncertainty_m > 0)
    assert np.all(surface.triangle_scale_m >= 0.005)


def test_da3_candidate_uses_physical_holdout_and_leaves_measured_rgbd_authoritative():
    window = _window()
    measured = build_measured_rgbd_surface(window.views[2])
    candidate = Da3SurfaceCandidateBuilder().build(window, _prediction(window), [measured])
    decision = Da3SurfaceCandidateBuilder.gate.evaluate(candidate)

    assert len(candidate.triangles) > 0
    assert np.count_nonzero(candidate.measured_surface_conflict) > 0
    assert len(decision.accepted_triangle_indices) > 0
    accepted_centres = candidate.positions[candidate.triangles[decision.accepted_triangle_indices]].mean(axis=1)
    assert np.all(accepted_centres[:, 0] > -0.02)
    assert np.all(decision.triangle_provenance == TriangleProvenance.NEURAL_MULTIVIEW)


def _texture_frame(index: int):
    image = Image.new("RGB", (8, 6), (40 + index, 80, 120))
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=90)
    jpeg = encoded.getvalue()
    depth = np.zeros((6, 8), dtype="<u2")
    depth[:, :4] = 2000
    compressed_depth = zlib.compress(depth.tobytes(), level=1)
    pose = np.eye(4)
    pose[0, 3] = index * 0.01
    return validate_texture_keyframe({
        "message_type": "texture_keyframe",
        "protocol_version": 3,
        "building_id": "building-a",
        "session_id": "session-a",
        "map_id": "map-a",
        "producer_instance_id": "producer-a",
        "producer_epoch": 7,
        "texture_sequence": index,
        "timestamp_ns": 1_000_000_000 + index,
        "calibration_id": "calibration-a",
        "source_config_sha256": "b" * 64,
        "content_sha256": hashlib.sha256(jpeg).hexdigest(),
        "source": {"topic": "/camera/color", "frame_id": "camera", "encoding": "rgb8", "width": 8, "height": 6},
        "color": {"encoding": "jpeg-base64", "jpeg_b64": base64.b64encode(jpeg).decode(), "width": 8, "height": 6},
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 3.5, "cy": 2.5, "width": 8, "height": 6},
        "lio_pose_timestamp_ns": 1_000_000_000 + index,
        "pose_skew_ns": 0,
        "pose_association_clock": "subscriber_monotonic",
        "pose_map_from_lidar": pose.tolist(),
        "quality": {"selection_accepted": True, "sharpness_laplacian": 100.0, "mean_luma": 100.0, "clipped_fraction": 0.0, "viewpoint_baseline_m": 0.1, "independent_view_id": f"view-{index}"},
        "depth": {"encoding": "u16-le-zlib-base64", "data_b64": base64.b64encode(compressed_depth).decode(), "compressed_bytes_sha256": hashlib.sha256(compressed_depth).hexdigest(), "width": 8, "height": 6, "aligned_to_color": True, "scale_m": 0.001},
    })


def test_live_runtime_wires_texture_frames_to_rgbd_and_da3_publishers():
    measured = build_measured_rgbd_surface(_observation(2))
    measured_revision = MeshTileRevision(
        TileId(0, 0, 8),
        "c" * 64,
        measured.positions,
        measured.normals,
        measured.triangles,
        measured.triangle_scale_m,
        0,
        measured.triangle_provenance,
        measured.triangle_uncertainty_m,
    )

    class Provider:
        inference_timeout_s = 30.0

        def infer_multiview(self, images, **kwargs):
            window = type("Window", (), {"views": tuple()})
            del window
            return MultiViewDepthFrame(
                depth_m=np.full((5, 6, 8), 2.0, dtype=np.float32),
                model_revision_id="a" * 64,
                preprocessing_revision="depth-anything-3-preprocess-v1",
                frame_sha256s=tuple(kwargs["frame_sha256s"]),
                intrinsics=kwargs["intrinsics"],
                extrinsics_world_to_camera=kwargs["extrinsics_world_to_camera"],
            )

    class Pipeline:
        def __init__(self):
            self.rgbd = []
            self.neural = []

        def publish_measured_rgbd_surface(self, building_id, submap_id, surface):
            self.rgbd.append((building_id, submap_id, surface))
            return {"published_tiles": 1, "triangle_count": len(surface.triangles)}

        def neural_surface_evidence(self, building_id, submap_id):
            return {
                "source_generation_id": "d" * 64,
                "tiles": ({"body": ImmutableTileStore._body(measured_revision)},),
            }

        def publish_neural_surface(self, building_id, submap_id, candidate, decision, *, expected_source_generation_id):
            self.neural.append((building_id, submap_id, candidate, decision, expected_source_generation_id))
            return {"published_tiles": 1, "triangle_count": len(decision.accepted_triangle_indices)}

    pipeline = Pipeline()
    published = []
    runtime = LiveDa3SurfaceRuntime(Provider(), pipeline, on_publish=published.append)
    for index in range(5):
        assert runtime.ingest(
            _texture_frame(index),
            submap_id="submap-a",
            camera_from_lidar=np.eye(4),
            calibration_authority=True,
        ) is True

    assert len(pipeline.rgbd) == 5
    assert runtime.run_once() is True
    assert len(pipeline.neural) == 1
    assert pipeline.neural[0][4] == "d" * 64
    assert published.count("building-a") >= 2
    assert runtime.snapshot()["published_neural_triangles"] > 0


def test_providerless_runtime_keeps_measured_rgbd_and_never_submits_a_da3_window():
    class Pipeline:
        def __init__(self):
            self.rgbd = []

        def publish_measured_rgbd_surface(self, building_id, submap_id, surface):
            self.rgbd.append((building_id, submap_id, surface))
            return {"published_tiles": 1, "triangle_count": len(surface.triangles)}

        def neural_surface_evidence(self, *_args):
            raise AssertionError("providerless runtime must not request DA3 evidence")

    pipeline = Pipeline()
    runtime = LiveDa3SurfaceRuntime(None, pipeline)
    runtime.start()
    try:
        for index in range(5):
            assert runtime.ingest(
                _texture_frame(index),
                submap_id="submap-a",
                camera_from_lidar=np.eye(4),
                calibration_authority=True,
            ) is True

        state = runtime.snapshot()
        assert len(pipeline.rgbd) == 5
        assert state["state"] == "DA3_SURFACE_UNAVAILABLE"
        assert "DA3_PROVIDER_UNAVAILABLE" in state["blockers"]
        assert state["accepted_observations"] == 5
        assert state["active_view_assemblers"] == 0
        assert state["rgbd_published_triangles"] > 0
        assert state["published_neural_triangles"] == 0
        assert state["last_error"] is None
        assert runtime.run_once() is False
    finally:
        runtime.close()


def test_runtime_ingest_failure_is_visible_in_the_stable_snapshot():
    class Provider:
        def infer_multiview(self, *_args, **_kwargs):
            raise AssertionError("one observation cannot form a DA3 window")

    class Pipeline:
        def publish_measured_rgbd_surface(self, *_args):
            raise RuntimeError("measured RGB-D publication failed")

    runtime = LiveDa3SurfaceRuntime(Provider(), Pipeline())

    with pytest.raises(RuntimeError, match="measured RGB-D publication failed"):
        runtime.ingest(
            _texture_frame(0),
            submap_id="submap-a",
            camera_from_lidar=np.eye(4),
            calibration_authority=True,
        )

    state = runtime.snapshot()
    assert state["state"] == "DA3_SURFACE_DEGRADED"
    assert state["last_error"] == "RuntimeError: measured RGB-D publication failed"
    assert "DA3_SURFACE_INGEST_FAILED" in state["blockers"]
