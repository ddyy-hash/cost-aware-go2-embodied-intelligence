from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

import mapping.pc.go2_mapping_core.pose_graph as pose_graph_module
from mapping.pc.go2_mapping_core.pose_graph import PoseGraph


def _pose(x, y=0.0, yaw_deg=0.0):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    value[:3, 3] = (x, y, 0.0)
    return value


def test_pose_graph_local_jacobian_matches_global_finite_difference(monkeypatch):
    comparison = {}

    def fake_least_squares(residual, initial, **kwargs):
        local = kwargs["jac"](initial)
        baseline = residual(initial)
        global_numeric = np.zeros_like(local)
        relative_step = np.sqrt(np.finfo(np.float64).eps)
        for column in range(len(initial)):
            value = float(initial[column])
            step = relative_step * (-1.0 if value < 0.0 else 1.0) * max(1.0, abs(value))
            perturbed = np.array(initial, copy=True)
            perturbed[column] += step
            global_numeric[:, column] = (residual(perturbed) - baseline) / step
        comparison["local"] = local
        comparison["global"] = global_numeric
        return SimpleNamespace(success=True, x=np.array(initial, copy=True), nfev=1, message="ok")

    monkeypatch.setattr(pose_graph_module, "least_squares", fake_least_squares)
    graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
    poses = (_pose(0.0), _pose(1.0, 0.1, -2.0), _pose(2.0, 0.15, 3.0))
    for index, pose in enumerate(poses):
        graph.add_keyframe(str(index), pose, "cal-v1", f"raw/{index}.npz")
    graph.add_lio_constraint(
        "0",
        "1",
        np.linalg.inv(poses[0]) @ poses[1],
        translation_sigma_m=0.03,
        rotation_sigma_deg=0.5,
    )
    graph.add_lio_constraint(
        "1",
        "2",
        np.linalg.inv(poses[1]) @ poses[2],
        translation_sigma_m=0.03,
        rotation_sigma_deg=0.5,
    )

    graph.optimize()

    np.testing.assert_allclose(
        comparison["local"],
        comparison["global"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_local_jacobian_preserves_global_numeric_solution(monkeypatch):
    def build_graph():
        graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
        measured = tuple(_pose(index * 1.01) for index in range(12))
        for index, pose in enumerate(measured):
            graph.add_keyframe(str(index), pose, "cal-v1", f"raw/{index}.npz")
        for index in range(len(measured) - 1):
            graph.add_lio_constraint(
                str(index),
                str(index + 1),
                np.linalg.inv(measured[index]) @ measured[index + 1],
                translation_sigma_m=0.03,
                rotation_sigma_deg=0.5,
            )
        assert graph.add_loop_constraint(
            "0",
            str(len(measured) - 1),
            _pose(len(measured) - 1),
            translation_sigma_m=0.005,
            rotation_sigma_deg=0.1,
        )
        return graph

    optimized_least_squares = pose_graph_module.least_squares

    def global_numeric(residual, initial, **kwargs):
        kwargs.pop("jac", None)
        return optimized_least_squares(residual, initial, **kwargs)

    monkeypatch.setattr(pose_graph_module, "least_squares", global_numeric)
    global_solution = build_graph().optimize()
    monkeypatch.setattr(pose_graph_module, "least_squares", optimized_least_squares)
    local_solution = build_graph().optimize()

    assert local_solution.final_cost == pytest.approx(global_solution.final_cost, abs=1e-10)
    assert local_solution.accepted_loop_closures == global_solution.accepted_loop_closures == 1
    for key in global_solution.poses_map_from_keyframe:
        assert local_solution.poses_map_from_keyframe[key] == pytest.approx(
            global_solution.poses_map_from_keyframe[key],
            abs=1e-7,
        )


def test_pose_graph_uses_sparse_lsmr_for_large_frame_sequences(monkeypatch):
    """A room-scale replay must not allocate a dense frame-by-frame Jacobian."""

    observed = {}

    def fake_least_squares(residual, initial, **kwargs):
        observed["tr_solver"] = kwargs["tr_solver"]
        observed["jac"] = kwargs["jac"]
        observed["jacobian_sparsity"] = kwargs["jac_sparsity"]
        observed["residual_size"] = residual(initial).size
        return SimpleNamespace(success=True, x=np.array(initial, copy=True), nfev=1, message="ok")

    monkeypatch.setattr(pose_graph_module, "least_squares", fake_least_squares)
    graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
    measured = tuple(_pose(index * 0.10) for index in range(80))
    for index, pose in enumerate(measured):
        graph.add_keyframe(str(index), pose, "cal-v1", f"raw/{index}.npz")
    for index in range(len(measured) - 1):
        graph.add_lio_constraint(
            str(index),
            str(index + 1),
            np.linalg.inv(measured[index]) @ measured[index + 1],
            translation_sigma_m=0.03,
            rotation_sigma_deg=0.5,
        )

    solution = graph.optimize()

    assert observed["tr_solver"] == "lsmr"
    assert observed["jac"] == "2-point"
    assert sparse.issparse(observed["jacobian_sparsity"])
    assert observed["jacobian_sparsity"].shape == (observed["residual_size"], 6 * (len(measured) - 1))
    assert solution.final_cost == pytest.approx(0.0)


def test_large_sparse_pose_graph_uses_colored_finite_difference_jacobian(monkeypatch):
    """Large replay must group independent SE(3) variables for differentiation."""

    observed = {}

    def fake_least_squares(residual, initial, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(success=True, x=np.array(initial, copy=True), nfev=1, message="ok")

    monkeypatch.setattr(pose_graph_module, "least_squares", fake_least_squares)
    graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
    measured = tuple(_pose(index * 0.10) for index in range(80))
    for index, pose in enumerate(measured):
        graph.add_keyframe(str(index), pose, "cal-v1", f"raw/{index}.npz")
    for index in range(len(measured) - 1):
        graph.add_lio_constraint(
            str(index),
            str(index + 1),
            np.linalg.inv(measured[index]) @ measured[index + 1],
            translation_sigma_m=0.03,
            rotation_sigma_deg=0.5,
        )

    solution = graph.optimize()

    assert observed["jac"] == "2-point"
    assert observed["tr_solver"] == "lsmr"
    assert sparse.issparse(observed["jac_sparsity"])
    assert observed["jac_sparsity"].shape == (6 * (len(measured) - 1), 6 * (len(measured) - 1))
    assert solution.final_cost == pytest.approx(0.0)
