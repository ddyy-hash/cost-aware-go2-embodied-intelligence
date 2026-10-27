import numpy as np

from mapping.pc.go2_mapping_rgbd import DepthValidator, RelativeDepthFrame
from mapping.pc.go2_mapping_rgbd.monocular_anchors import ConsensusAnchorBuilder


HASH = "a" * 64


def _accepted_report():
    camera = np.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    points = np.asarray(
        [[u / 100.0, v / 100.0, 1.0] for v in range(20) for u in range(20)],
        dtype=np.float64,
    )
    return DepthValidator().cross_sensor(
        points,
        np.ones((20, 20), dtype=np.float64),
        camera,
        np.eye(4),
        calibration_id="cal-1",
        calibration_sha256=HASH,
        measured_generation="generation-1",
        frame_sha256="b" * 64,
        timestamp_ns=1000,
    )


def test_builder_preserves_exact_correspondence_identity_and_model_sample():
    report = _accepted_report()
    relative = RelativeDepthFrame(
        np.arange(1, 401, dtype=np.float32).reshape(20, 20),
        "c" * 64,
        "preprocess-v1",
        "b" * 64,
    )

    result = ConsensusAnchorBuilder().build(
        relative,
        report.correspondences,
        calibration_sha256=HASH,
        measured_generation="generation-1",
        rectified_rgb=np.full((20, 20, 3), 100, dtype=np.uint8),
    )

    assert result.blockers == ()
    assert len(result.anchors) == 400
    assert result.calibration_sha256 == HASH
    assert result.measured_generation == "generation-1"
    first = result.anchors[0]
    assert first.pixel_x == 0
    assert first.pixel_y == 0
    assert first.lidar_point_index == 0
    assert first.model_inverse_depth == 1.0
    assert first.metric_depth_m == 1.0
    assert first.point_camera_xyz == (0.0, 0.0, 1.0)
    assert first.frame_sha256 == "b" * 64
    assert first.accepted_region_id == report.tiles[0].tile_id


def test_builder_rejects_masked_saturated_and_mismatched_evidence():
    report = _accepted_report()
    relative = RelativeDepthFrame(np.ones((20, 20)), "c" * 64, "preprocess-v1", "b" * 64)
    dynamic = np.zeros((20, 20), dtype=bool)
    occlusion = np.zeros((20, 20), dtype=bool)
    free_space = np.zeros((20, 20), dtype=bool)
    discontinuity = np.zeros((20, 20), dtype=bool)
    dynamic[0, 0] = True
    occlusion[0, 1] = True
    free_space[0, 2] = True
    discontinuity[0, 3] = True
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    rgb[0, 4] = 255

    result = ConsensusAnchorBuilder().build(
        relative,
        report.correspondences,
        calibration_sha256=HASH,
        measured_generation="generation-1",
        rectified_rgb=rgb,
        dynamic_mask=dynamic,
        occlusion_mask=occlusion,
        free_space_mask=free_space,
        depth_discontinuity_mask=discontinuity,
    )

    assert len(result.anchors) == 395
    assert [anchor.lidar_point_index for anchor in result.anchors[:2]] == [5, 6]

    blocked = ConsensusAnchorBuilder().build(
        relative,
        report.correspondences,
        calibration_sha256="d" * 64,
        measured_generation="generation-1",
        rectified_rgb=rgb,
    )
    assert blocked.anchors == ()
    assert "CALIBRATION_IDENTITY_MISMATCH" in blocked.blockers
