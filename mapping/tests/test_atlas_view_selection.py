import pytest

from mapping.pc.go2_mapping_appearance.view_selection import (
    TexelObservation,
    select_validated_texel,
)


def _observation(frame, view, color=(0.2, 0.4, 0.6), **changes):
    values = {
        "source_frame_sha256": frame * 64,
        "pixel_x": 10,
        "pixel_y": 20,
        "view_id": view,
        "linear_rgb": color,
        "in_frustum": True,
        "orientation_cosine": 0.8,
        "physical_depth_accepted": True,
        "consensus_depth_accepted": False,
        "occluded": False,
        "depth_residual_m": 0.005,
        "maximum_depth_residual_m": 0.03,
        "sharpness": 0.8,
        "exposure": 0.5,
    }
    values.update(changes)
    return TexelObservation(**values)


def test_selection_is_input_order_independent_and_blends_compatible_views_in_linear_rgb():
    observations = (
        _observation("b", "view-b", color=(0.3, 0.5, 0.7), pixel_x=11),
        _observation("a", "view-a", color=(0.2, 0.4, 0.6), pixel_x=9),
    )

    first = select_validated_texel(observations)
    reversed_result = select_validated_texel(tuple(reversed(observations)))

    assert first == reversed_result
    assert first.accepted is True
    assert first.linear_rgb == pytest.approx((0.25, 0.45, 0.65))
    assert first.source_frame_sha256 == ("a" * 64, "b" * 64)
    assert first.independent_view_ids == ("view-a", "view-b")
    assert first.blockers == ()


@pytest.mark.parametrize(
    "change",
    [
        {"in_frustum": False},
        {"orientation_cosine": 0.0},
        {"physical_depth_accepted": False, "consensus_depth_accepted": False},
        {"occluded": True},
        {"depth_residual_m": 0.031},
        {"sharpness": 0.01},
        {"exposure": 0.99},
    ],
)
def test_invalid_visibility_residual_sharpness_or_exposure_never_contributes(change):
    result = select_validated_texel(
        (_observation("a", "view-a", **change), _observation("b", "view-b", **change))
    )

    assert result.accepted is False
    assert result.linear_rgb is None
    assert "INSUFFICIENT_VALIDATED_VIEWS" in result.blockers


def test_consensus_depth_is_valid_visibility_but_duplicate_view_ids_do_not_count_twice():
    consensus = _observation(
        "a", "same-view", physical_depth_accepted=False, consensus_depth_accepted=True
    )
    duplicate = _observation("b", "same-view")
    result = select_validated_texel((duplicate, consensus))

    assert result.accepted is False
    assert result.source_frame_sha256 == ("a" * 64,)
    assert "INSUFFICIENT_VALIDATED_VIEWS" in result.blockers


def test_inconsistent_views_are_rejected_instead_of_averaged_into_a_seam():
    result = select_validated_texel(
        (
            _observation("a", "view-a", color=(0.1, 0.1, 0.1)),
            _observation("b", "view-b", color=(0.9, 0.9, 0.9)),
        ),
        maximum_linear_rgb_channel_stddev=0.1,
    )

    assert result.accepted is False
    assert result.linear_rgb is None
    assert "INCONSISTENT_VIEW_COLOR" in result.blockers


def test_canonical_score_chooses_one_best_observation_per_view_with_hash_pixel_tiebreaks():
    worse = _observation("c", "view-a", sharpness=0.4, pixel_x=1)
    tied_later = _observation("b", "view-a", sharpness=0.9, pixel_x=2)
    tied_first = _observation("a", "view-a", sharpness=0.9, pixel_x=3)
    other = _observation("d", "view-b")

    result = select_validated_texel((worse, tied_later, tied_first, other))

    assert result.accepted is True
    assert result.source_frame_sha256 == ("a" * 64, "d" * 64)
