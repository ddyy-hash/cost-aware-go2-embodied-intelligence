import numpy as np

from mapping.pc.go2_mapping_rgbd.augmentation import bounded_hole_components


def _inputs():
    candidate = np.zeros((5, 5), dtype=bool)
    candidate[2, 2] = True
    boundary = np.zeros((5, 5), dtype=bool)
    boundary[1, 2] = boundary[2, 1] = boundary[2, 3] = boundary[3, 2] = True
    positions = np.zeros((5, 5, 3), dtype=np.float64)
    for y in range(5):
        for x in range(5):
            positions[y, x] = (x * 0.01, y * 0.01, 1.0)
    uncertainty = np.full((5, 5), 0.02, dtype=np.float64)
    distance = np.full((5, 5), 0.10, dtype=np.float64)
    frames = {(2, 2): ("a" * 64, "b" * 64, "c" * 64)}
    views = {(2, 2): ("view-1", "view-2")}
    return candidate, boundary, positions, uncertainty, distance, frames, views


def test_complete_four_connected_component_is_accepted_with_per_point_support():
    candidate, boundary, positions, uncertainty, distance, frames, views = _inputs()

    components = bounded_hole_components(
        candidate,
        accepted_boundary_mask=boundary,
        metric_positions=positions,
        uncertainty_m=uncertainty,
        boundary_distance_m=distance,
        support_frame_ids=frames,
        support_view_ids=views,
    )

    assert len(components) == 1
    component = components[0]
    assert component.accepted is True
    assert component.pixel_ids == ((2, 2),)
    assert component.metric_positions.shape == (1, 3)
    assert component.maximum_boundary_distance_m == 0.10
    assert component.support_frame_ids == ("a" * 64, "b" * 64, "c" * 64)
    assert component.support_view_ids == ("view-1", "view-2")
    assert component.blockers == ()


def test_component_rejects_incomplete_or_forbidden_boundary_and_far_points():
    candidate, boundary, positions, uncertainty, distance, frames, views = _inputs()
    boundary[1, 2] = False
    distance[2, 2] = 0.151
    occlusion = np.zeros((5, 5), dtype=bool)
    occlusion[2, 3] = True

    component = bounded_hole_components(
        candidate,
        accepted_boundary_mask=boundary,
        metric_positions=positions,
        uncertainty_m=uncertainty,
        boundary_distance_m=distance,
        support_frame_ids=frames,
        support_view_ids=views,
        occlusion_edge_mask=occlusion,
    )[0]

    assert component.accepted is False
    assert "COMPLETE_ACCEPTED_BOUNDARY_REQUIRED" in component.blockers
    assert "FORBIDDEN_BOUNDARY_EDGE" in component.blockers
    assert "BOUNDARY_DISTANCE_GATE_FAILED" in component.blockers


def test_diagonal_pixels_are_separate_and_each_point_needs_three_frames_two_views():
    candidate, boundary, positions, uncertainty, distance, frames, views = _inputs()
    candidate[1, 1] = True
    boundary[:, :] = True
    frames[(1, 1)] = ("a" * 64, "b" * 64)
    views[(1, 1)] = ("view-1", "view-2")

    components = bounded_hole_components(
        candidate,
        accepted_boundary_mask=boundary,
        metric_positions=positions,
        uncertainty_m=uncertainty,
        boundary_distance_m=distance,
        support_frame_ids=frames,
        support_view_ids=views,
    )

    assert len(components) == 2
    rejected = next(component for component in components if component.pixel_ids == ((1, 1),))
    assert "PER_POINT_FRAME_SUPPORT_REQUIRED" in rejected.blockers


def test_component_touching_image_edge_is_never_bounded():
    candidate, boundary, positions, uncertainty, distance, frames, views = _inputs()
    candidate[:, :] = False
    candidate[0, 2] = True
    frames[(2, 0)] = frames.pop((2, 2))
    views[(2, 0)] = views.pop((2, 2))
    component = bounded_hole_components(
        candidate,
        accepted_boundary_mask=np.ones((5, 5), dtype=bool),
        metric_positions=positions,
        uncertainty_m=uncertainty,
        boundary_distance_m=distance,
        support_frame_ids=frames,
        support_view_ids=views,
    )[0]
    assert "IMAGE_EDGE_CONTACT" in component.blockers
