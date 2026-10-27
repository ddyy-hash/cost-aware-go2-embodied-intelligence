import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.manifest import build_calibration_manifest
from mapping.pc.go2_mapping_quality.reconstruction_config import ConfigError, canonical_sha256


def _fit(digest):
    return {
        "slope": 1.0,
        "offset_ns": 0.0,
        "sample_count": 200,
        "span_s": 30.0,
        "inlier_fraction": 0.99,
        "median_absolute_residual_ms": 0.1,
        "p95_residual_ms": 0.2,
        "maximum_prediction_stddev_ms": 0.1,
        "estimator_version": "test",
        "sha256": digest,
    }


def _lidar_imu_evidence():
    return {
        "source_capture_sha256": ["5" * 64, "6" * 64, "7" * 64],
        "result_artifact_sha256": ["8" * 64, "9" * 64, "a" * 64],
        "transform_sha256": ["b" * 64, "c" * 64, "d" * 64],
        "raw_lidar_topic": "/livox/lidar",
        "raw_imu_topic": "/livox/imu",
        "lidar_frame_id": "livox_frame",
        "imu_frame_id": "imu_link",
        "solver": {
            "name": "li-init",
            "version": "2024.1",
            "implementation_sha256": "e" * 64,
        },
        "uses_vendor_extrinsic": False,
    }


def test_manifest_builder_content_addresses_measured_target_and_transforms():
    identity = [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    t_c_l_hash = canonical_sha256({"row_major": identity})
    t_d_c_hash = canonical_sha256({"row_major": identity, "device_serial": "d435"})
    payload = build_calibration_manifest(
        robot_serial="go2",
        camera_serial="d435",
        estimation_hashes=("a" * 64, "b" * 64, "c" * 64),
        holdout_hash="d" * 64,
        solver_version="test-solver",
        solver_implementation_sha256="e" * 64,
        t_c_l=identity,
        t_d_c=identity,
        t_c_l_sha256=t_c_l_hash,
        t_d_c_sha256=t_d_c_hash,
        color_camera_info_sha256="f" * 64,
        depth_camera_info_sha256="1" * 64,
        color_camera_info={"width": 424, "height": 240, "distortion_model": "plumb_bob", "k": [300,0,212,0,300,120,0,0,1], "d": [0,0,0,0,0], "r": [1,0,0,0,1,0,0,0,1], "p": [300,0,212,0,0,300,120,0,0,0,1,0], "rectified": False},
        depth_camera_info={"width": 424, "height": 240, "distortion_model": "plumb_bob", "k": [300,0,212,0,300,120,0,0,1], "d": [0,0,0,0,0], "r": [1,0,0,0,1,0,0,0,1], "p": [300,0,212,0,0,300,120,0,0,0,1,0], "rectified": False},
        lidar_frame_id="livox_frame",
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_depth_optical_frame",
        clock_fits={"lidar": _fit("2" * 64), "color": _fit("3" * 64), "depth": _fit("4" * 64)},
        lidar_imu_evidence=_lidar_imu_evidence(),
        repeatability={"lidar_imu_translation_mm": 1.0, "lidar_imu_rotation_deg": 0.1, "lidar_camera_translation_mm": 1.0, "lidar_camera_rotation_deg": 0.1},
        timing={"lidar_imu_repeatability_ms": 0.5, "lidar_camera_repeatability_ms": 1.0},
        holdout={"artifact_sha256": "d" * 64, "accepted_board_poses": 3, "correspondences_per_pose": [4000,4000,4000], "total_correspondences": 12000, "validated_range_min_m": 0.6, "validated_range_max_m": 3.9, "camera_reprojection_median_px": 0.2, "camera_reprojection_p95_px": 0.4, "depth_median_absolute_residual_m": 0.01, "depth_p95_absolute_residual_m": 0.03, "absolute_mean_signed_bias_m": 0.005, "occlusion_conflict_fraction": 0.01, "free_space_violation_fraction": 0.005, "t_d_c_validated": True},
    )

    assert payload["status"] == "measured"
    assert payload["target"]["square_length_m"] == 0.020
    assert payload["target"]["marker_length_m"] == 0.0145
    assert payload["target"]["source_template_pdf_sha256"] == "fe05d01ca5fdde303f712dad177f53c3ae75d3493ff7d9a783b010f892a2d743"
    assert payload["target"]["source_pdf_nominal_marker_length_m"] == 0.005
    assert payload["target"]["source_pdf_nominal_square_length_m"] == 0.007
    assert payload["frame_chain"]["t_d_c"]["device_serial"] == "d435"
    assert payload["lidar_imu_evidence"]["source_capture_sha256"] == ["5" * 64, "6" * 64, "7" * 64]
    assert payload["lidar_imu_evidence"]["uses_vendor_extrinsic"] is False


def test_manifest_builder_requires_holdout_and_all_three_unique_artifacts():
    with pytest.raises(ValueError, match="unique"):
        build_calibration_manifest(
            robot_serial="go2", camera_serial="d435", estimation_hashes=("a"*64, "a"*64, "b"*64), holdout_hash="c"*64,
            solver_version="s", solver_implementation_sha256="d"*64, t_c_l=[1]*16, t_d_c=[1]*16,
            t_c_l_sha256="e"*64, t_d_c_sha256="f"*64, color_camera_info_sha256="1"*64, depth_camera_info_sha256="2"*64,
            color_camera_info={}, depth_camera_info={}, lidar_frame_id="l", color_frame_id="c", depth_frame_id="d",
            clock_fits={}, repeatability={}, timing={}, holdout={},
            lidar_imu_evidence=_lidar_imu_evidence(),
        )
