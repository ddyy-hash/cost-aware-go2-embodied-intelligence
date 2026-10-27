import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd import (
    AugmentationBuilder,
    ColorAccumulator,
    ColorObservation,
)


HASH = "a" * 64


def _observation(origin, *, color=(0.3, 0.4, 0.5), residual=0.0, occluded=False):
    return ColorObservation(
        linear_rgb=color,
        depth_residual_m=residual,
        range_m=1.0,
        incidence_cosine=1.0,
        sharpness=1.0,
        occluded=occluded,
        view_origin_xyz=origin,
        view_direction_xyz=(1.0, 0.0, 0.0),
        frame_sha256=HASH,
        calibration_id="cal-1",
    )


def test_color_accumulator_weights_views_and_rejects_occluded_samples():
    accumulator = ColorAccumulator("cal-1")

    assert accumulator.observe("v1", _observation((0.0, 0.0, 0.0))) is True
    assert accumulator.observe("v1", _observation((0.25, 0.0, 0.0))) is True
    assert accumulator.observe("v1", _observation((0.5, 0.0, 0.0), occluded=True)) is False
    vertex = accumulator.vertex("v1")

    assert vertex.accepted_count == 2
    assert vertex.rejected_count == 1
    assert vertex.independent_views == 2
    assert vertex.accepted_for_texture is True
    assert vertex.mean_linear_rgb == pytest.approx((0.3, 0.4, 0.5))


def test_texture_hard_rejects_depth_residual_and_requires_coverage():
    accumulator = ColorAccumulator("cal-1")
    assert accumulator.observe("v1", _observation((0.0, 0.0, 0.0), residual=0.031)) is False
    for vertex_id in ("v1", "v2", "v3", "v4"):
        accumulator.observe(vertex_id, _observation((0.0, 0.0, 0.0)))
        accumulator.observe(vertex_id, _observation((0.25, 0.0, 0.0)))

    accepted = accumulator.delivery_report(("v1", "v2", "v3", "v4", "v5"))
    rejected = accumulator.delivery_report(("v1", "v2", "v3", "v4", "v5", "v6"))

    assert accepted["surface_coverage"] == 0.8
    assert accepted["delivery_authorized"] is True
    assert rejected["delivery_authorized"] is False


def test_color_accumulator_exports_requested_vertex_order_without_exposing_state():
    accumulator = ColorAccumulator("cal-1")
    accumulator.observe("v1", _observation((0.0, 0.0, 0.0)))
    accumulator.observe("v1", _observation((0.25, 0.0, 0.0)))
    accumulator.observe("v2", _observation((0.0, 0.0, 0.0), color=(0.8, 0.2, 0.1)))

    vertices = accumulator.vertices(("v2", "missing", "v1"))

    assert tuple(vertex.vertex_id for vertex in vertices) == ("v2", "missing", "v1")
    assert vertices[0].mean_linear_rgb == pytest.approx((0.8, 0.2, 0.1))
    assert vertices[1].accepted_count == 0
    assert vertices[2].independent_views == 2
    assert isinstance(vertices, tuple)


def test_rgbd_hole_fill_is_never_measured_geometry():
    layer = AugmentationBuilder().fill(
        np.asarray([[1.0, 0.0, 0.0]]),
        quality_level="CALIBRATED_DENSE",
        depth_region_accepted=True,
        depth_frame_count=3,
        independent_views=2,
        depth_stddev_m=0.03,
        completely_bounded=True,
        maximum_boundary_distance_m=0.15,
    )

    assert layer.name == "RGBD_AUGMENTED"
    assert layer.geometry_authority is False
    assert layer.source_vertex_ids == ("compat-vertex-0",)
    assert layer.source_triangle_ids == ()


def test_rgbd_hole_fill_rejects_an_unbounded_or_noisy_hole():
    with pytest.raises(PermissionError):
        AugmentationBuilder().fill(
            np.asarray([[1.0, 0.0, 0.0]]),
            quality_level="CALIBRATED_DENSE",
            depth_region_accepted=True,
            depth_frame_count=3,
            independent_views=2,
            depth_stddev_m=0.031,
            completely_bounded=False,
            maximum_boundary_distance_m=0.15,
        )
