import sys
from pathlib import Path

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PC_ROOT = ROOT / "mapping" / "pc"
sys.path.insert(0, str(PC_ROOT))

from go2_mapping_bim.register import RegistrationDecision, evaluate_registration  # noqa: E402
from go2_mapping_core.ingest import Envelope, IngestStore  # noqa: E402
from go2_mapping_core.pose_graph import PoseGraph  # noqa: E402
from go2_mapping_core.submaps import SubmapStore  # noqa: E402
from go2_mapping_neural.gaussian_submap import GaussianBudget, GaussianSubmap  # noqa: E402
ROBOT_LIO = ROOT / "mapping" / "robot" / "go2_mapping_lio"
sys.path.insert(0, str(ROBOT_LIO))
from go2_mapping_lio.runtime_manifest import build_manifest, verify_manifest  # noqa: E402


def test_ingest_is_checksum_verified_and_idempotent(tmp_path):
    store = IngestStore(tmp_path)
    envelope = Envelope.create("building-a", 7, b"deskewed-points", calibration_id="cal-v1")

    assert store.accept(envelope) is True
    assert store.accept(envelope) is False
    assert store.count("building-a") == 1

    invalid = envelope.with_payload(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.accept(invalid)


def test_pose_graph_keeps_keyframe_measurement_immutable_and_rejects_bad_loop():
    graph = PoseGraph(loop_translation_limit_m=0.05, loop_rotation_limit_deg=1.0)
    graph.add_keyframe("a", (0.0, 0.0, 0.0), "cal-v1", "raw/a.mcap")
    graph.add_keyframe("b", (1.0, 0.0, 0.0), "cal-v1", "raw/b.mcap")
    graph.add_lio_edge("a", "b", translation_residual_m=0.01, rotation_residual_deg=0.2)

    assert graph.add_loop_edge("a", "b", translation_residual_m=0.06, rotation_residual_deg=0.2) is False
    assert graph.add_loop_edge("a", "b", translation_residual_m=0.03, rotation_residual_deg=0.8) is True
    assert graph.keyframe("a").raw_asset_ref == "raw/a.mcap"


def _pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_pose_graph_optimizes_se3_constraints_without_mutating_raw_measurements():
    graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
    graph.add_keyframe("a", _pose(0.0), "cal-v1", "raw/a.npz")
    graph.add_keyframe("b", _pose(1.05), "cal-v1", "raw/b.npz")
    graph.add_keyframe("c", _pose(2.10), "cal-v1", "raw/c.npz")
    graph.add_lio_constraint("a", "b", _pose(1.05), translation_sigma_m=0.03, rotation_sigma_deg=0.5)
    graph.add_lio_constraint("b", "c", _pose(1.05), translation_sigma_m=0.03, rotation_sigma_deg=0.5)
    assert graph.add_loop_constraint(
        "a",
        "c",
        _pose(2.0),
        translation_sigma_m=0.005,
        rotation_sigma_deg=0.1,
    )

    solution = graph.optimize()

    assert solution.poses_map_from_keyframe["a"] == pytest.approx(_pose(0.0))
    assert solution.poses_map_from_keyframe["c"][0, 3] == pytest.approx(2.0, abs=0.01)
    assert solution.accepted_loop_closures == 1
    assert len(solution.pose_generation) == 64
    assert graph.keyframe("c").measured_pose_map_from_keyframe[0, 3] == pytest.approx(2.10)


def test_pose_graph_rejects_loop_constraint_whose_correction_exceeds_gate():
    graph = PoseGraph(loop_translation_limit_m=0.05, loop_rotation_limit_deg=1.0)
    graph.add_keyframe("a", _pose(0.0), "cal-v1", "raw/a.npz")
    graph.add_keyframe("b", _pose(1.0), "cal-v1", "raw/b.npz")

    assert graph.add_loop_constraint(
        "a",
        "b",
        _pose(0.90),
        translation_sigma_m=0.01,
        rotation_sigma_deg=0.1,
    ) is False
    assert graph.optimize().accepted_loop_closures == 0


def test_pose_graph_solution_is_deterministic_when_constraints_arrive_in_different_orders():
    def solve(constraint_order):
        graph = PoseGraph(loop_translation_limit_m=0.20, loop_rotation_limit_deg=2.0)
        graph.add_keyframe("a", _pose(0.0), "cal-v1", "raw/a.npz")
        graph.add_keyframe("b", _pose(1.05), "cal-v1", "raw/b.npz")
        graph.add_keyframe("c", _pose(2.10), "cal-v1", "raw/c.npz")
        constraints = {
            "ab": lambda: graph.add_lio_constraint(
                "a", "b", _pose(1.05), translation_sigma_m=0.03, rotation_sigma_deg=0.5
            ),
            "bc": lambda: graph.add_lio_constraint(
                "b", "c", _pose(1.05), translation_sigma_m=0.03, rotation_sigma_deg=0.5
            ),
            "ac": lambda: graph.add_loop_constraint(
                "a", "c", _pose(2.0), translation_sigma_m=0.005, rotation_sigma_deg=0.1
            ),
        }
        for name in constraint_order:
            assert constraints[name]() is not False
        return graph.optimize()

    forward = solve(("ab", "bc", "ac"))
    reversed_order = solve(("ac", "bc", "ab"))

    assert reversed_order.pose_generation == forward.pose_generation
    assert reversed_order.initial_cost == pytest.approx(forward.initial_cost)
    assert reversed_order.final_cost == pytest.approx(forward.final_cost)
    for keyframe_id in ("a", "b", "c"):
        assert reversed_order.poses_map_from_keyframe[keyframe_id] == pytest.approx(
            forward.poses_map_from_keyframe[keyframe_id]
        )


def test_submaps_evict_completed_data_to_disk_without_losing_metadata(tmp_path):
    store = SubmapStore(tmp_path, max_active_bytes=10)
    store.add("s1", b"123456", sealed=True)
    store.add("s2", b"abcdef", sealed=True)

    assert store.is_evicted("s1") is True
    assert store.load("s1") == b"123456"
    assert store.metadata("s1")["sealed"] is True


def test_gaussian_submap_refuses_to_exceed_vram_budget_or_overwrite_measurements():
    model = GaussianSubmap(GaussianBudget(max_vram_bytes=100, bytes_per_gaussian=20))
    model.add_lidar_surface("wall-1", point_count=50, residual_m=0.01)
    model.add_gaussians(4, source_surface_id="wall-1")

    with pytest.raises(MemoryError):
        model.add_gaussians(2, source_surface_id="wall-1")
    with pytest.raises(PermissionError, match="as-built"):
        model.overwrite_lidar_surface("wall-1")


def test_bim_registration_requires_independent_support_and_never_deforms_as_built():
    rejected = evaluate_registration(
        groups=("north-wall", "east-wall"),
        support_m=15.0,
        median_residual_m=0.02,
        p95_residual_m=0.05,
        translation_sigma_m=0.02,
        rotation_sigma_deg=0.4,
    )
    assert rejected.decision is RegistrationDecision.UNREGISTERED

    accepted = evaluate_registration(
        groups=("north-wall", "east-wall", "column-grid"),
        support_m=15.0,
        median_residual_m=0.02,
        p95_residual_m=0.05,
        translation_sigma_m=0.02,
        rotation_sigma_deg=0.4,
    )
    assert accepted.decision is RegistrationDecision.ACCEPTED
    assert accepted.geometry_authority == "as_built_map"


def test_lio_runtime_manifest_detects_any_binary_or_dependency_change(tmp_path):
    binary = tmp_path / "lio_node"
    dependency = tmp_path / "liblio.so"
    binary.write_bytes(b"node-v1")
    dependency.write_bytes(b"library-v1")
    expected = build_manifest("/opt/lio", [binary, dependency], ros_distro="foxy")

    assert verify_manifest(expected, build_manifest("/opt/lio", [binary, dependency], ros_distro="foxy")) == []
    dependency.write_bytes(b"library-v2")
    assert "hash mismatch" in " ".join(verify_manifest(expected, build_manifest("/opt/lio", [binary, dependency], ros_distro="foxy")))
