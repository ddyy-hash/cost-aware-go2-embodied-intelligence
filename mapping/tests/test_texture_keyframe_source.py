import base64
import hashlib

import numpy as np

from mapping.pc.go2_mapping_live.live_protocol import validate_texture_keyframe
from mapping.robot.go2_mapping_lio.go2_mapping_lio.texture_keyframe_source import (
    TextureQualitySelector,
    build_texture_keyframe_message,
)


def _checkerboard(height: int = 240, width: int = 424) -> np.ndarray:
    rows, columns = np.indices((height, width))
    gray = (((rows // 8) + (columns // 8)) % 2 * 180 + 35).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_texture_quality_selector_requires_focus_exposure_and_new_viewpoint():
    selector = TextureQualitySelector(
        minimum_sharpness=20.0,
        minimum_luma=20.0,
        maximum_luma=235.0,
        maximum_clipped_fraction=0.1,
        minimum_translation_m=0.08,
        minimum_rotation_deg=5.0,
    )
    pose = np.eye(4)

    first = selector.evaluate(_checkerboard(), pose, independent_view_id="view-0")
    repeated = selector.evaluate(_checkerboard(), pose, independent_view_id="view-1")
    translated = pose.copy()
    translated[0, 3] = 0.1
    next_view = selector.evaluate(_checkerboard(), translated, independent_view_id="view-2")
    blurred = selector.evaluate(np.full((240, 424, 3), 120, dtype=np.uint8), translated, independent_view_id="view-3")
    clipped = selector.evaluate(np.full((240, 424, 3), 255, dtype=np.uint8), translated, independent_view_id="view-4")

    assert first.accepted is True
    assert repeated.accepted is False
    assert repeated.blocker == "INSUFFICIENT_VIEWPOINT_BASELINE"
    assert next_view.accepted is True
    assert next_view.viewpoint_baseline_m == 0.1
    assert blurred.blocker == "INSUFFICIENT_SHARPNESS"
    assert clipped.blocker in {"EXPOSURE_OUT_OF_RANGE", "CLIPPED_EXPOSURE"}


def test_texture_message_builder_preserves_original_frame_identity_pose_and_depth():
    color_jpeg = b"\xff\xd8" + b"jpeg-frame" * 64 + b"\xff\xd9"
    depth = np.full((240, 424), 875, dtype="<u2")
    pose = np.eye(4)
    pose[0, 3] = 1.25

    encoded = build_texture_keyframe_message(
        building_id="lab-building",
        session_id="session-a",
        map_id="go2-live",
        producer_instance_id="texture-producer",
        producer_epoch=1234,
        texture_sequence=5,
        timestamp_ns=9_000_000_000,
        calibration_id="calibration-a",
        source_config_sha256="c" * 64,
        source_topic="/camera/color/image_raw",
        source_frame_id="camera_color_optical_frame",
        source_encoding="bgr8",
        source_width=424,
        source_height=240,
        color_jpeg=color_jpeg,
        intrinsics={"fx": 305.0, "fy": 305.0, "cx": 211.5, "cy": 119.5},
        lio_pose_timestamp_ns=8_999_000_000,
        pose_skew_ns=1_000_000,
        pose_association_clock="subscriber_monotonic",
        pose_map_from_lidar=pose,
        quality={
            "selection_accepted": True,
            "sharpness_laplacian": 80.0,
            "mean_luma": 110.0,
            "clipped_fraction": 0.01,
            "viewpoint_baseline_m": 0.1,
            "viewpoint_rotation_deg": 0.0,
            "independent_view_id": "view-5",
        },
        aligned_depth_u16=depth,
        depth_scale_m=0.001,
    )
    frame = validate_texture_keyframe(encoded.payload)

    assert encoded.message_bytes < 4 * 1024 * 1024
    assert frame.content_sha256 == hashlib.sha256(color_jpeg).hexdigest()
    assert frame.source_dimensions == (424, 240)
    assert frame.pose_skew_ns == 1_000_000
    assert frame.depth_u16_bytes == depth.tobytes()
    assert base64.b64decode(encoded.payload["color"]["jpeg_b64"]) == color_jpeg
