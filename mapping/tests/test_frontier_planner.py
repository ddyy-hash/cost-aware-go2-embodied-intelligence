from __future__ import annotations

import pytest

from mapping.pc.go2_mapping_dashboard.frontier import plan_frontier_candidates


def _grid() -> list[list[int]]:
    unknown = 205
    free = 254
    occupied = 0
    grid = [[unknown] * 10 for _ in range(8)]
    for row in range(2, 6):
        for column in range(1, 7):
            grid[row][column] = free
    grid[3][4] = occupied
    return grid


def test_frontier_planner_clusters_unknown_boundary_and_scores_candidates():
    candidates = plan_frontier_candidates(
        _grid(),
        origin=(-1.0, -2.0),
        resolution=0.5,
        robot_pose={"x": 0.0, "y": 0.0},
        frame="SESSION_ODOM_LOCAL",
        min_cluster_size=1,
    )

    assert candidates
    assert candidates == sorted(candidates, key=lambda item: (-item["score"], item["candidate_id"]))
    for candidate in candidates:
        assert candidate["frame"] == "SESSION_ODOM_LOCAL"
        assert candidate["position"]["frame"] == "SESSION_ODOM_LOCAL"
        assert candidate["size_cells"] >= 1
        assert candidate["information_gain_cells"] >= candidate["size_cells"]
        assert candidate["distance_m"] >= 0.0
        assert candidate["score"] > 0.0
        assert candidate["position"]["yaw"] == 0.0


def test_frontier_planner_limits_candidates_and_rejects_invalid_grids():
    candidates = plan_frontier_candidates(
        _grid(), robot_pose={"x": 0.0, "y": 0.0}, max_candidates=1
    )
    assert len(candidates) == 1

    with pytest.raises(ValueError, match="rectangular"):
        plan_frontier_candidates([[254], [205, 205]], robot_pose={"x": 0.0, "y": 0.0})


def test_frontier_planner_returns_no_candidate_for_closed_known_space():
    grid = [[254] * 4 for _ in range(4)]

    assert plan_frontier_candidates(grid, robot_pose={"x": 1.0, "y": 1.0}) == []
