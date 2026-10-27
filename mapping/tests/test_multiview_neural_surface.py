import hashlib
from dataclasses import replace

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision, TriangleProvenance
from mapping.pc.go2_mapping_rgbd.multiview_surface import (
    Da3MultiViewWorker,
    Da3ViewObservation,
    Da3WindowAssembler,
    HybridSurfaceAssembler,
    NeuralSurfaceCandidate,
    NeuralSurfaceDecision,
    NeuralSurfaceGate,
)


def _observation(index: int) -> Da3ViewObservation:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = index * 0.1
    intrinsics = np.asarray([[300.0, 0.0, 2.0], [0.0, 300.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return Da3ViewObservation(
        frame_sha256=hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
        independent_view_id=f"view-{index}",
        timestamp_ns=1_000_000_000 + index,
        rectified_rgb=np.full((3, 4, 3), index, dtype=np.uint8),
        intrinsics=intrinsics,
        extrinsics_world_to_camera=pose,
    )


def test_da3_windows_are_five_views_with_two_view_overlap_and_unique_identity():
    assembler = Da3WindowAssembler(window_size=5, overlap=2)
    windows = []
    for index in range(8):
        window = assembler.submit(_observation(index))
        if window is not None:
            windows.append(window)

    assert len(windows) == 2
    assert [view.independent_view_id for view in windows[0].views] == [f"view-{index}" for index in range(5)]
    assert [view.independent_view_id for view in windows[1].views] == [f"view-{index}" for index in range(3, 8)]
    assert windows[0].reference_index == windows[1].reference_index == 2
    assert assembler.pending_view_count == 2
    assert assembler.submit(_observation(7)) is None
    assert assembler.duplicate_frames == 1


def _candidate() -> NeuralSurfaceCandidate:
    positions = np.asarray(
        [[index, 0.0, 1.0] for index in range(8)] + [[index, 1.0, 1.0] for index in range(8)],
        dtype="<f4",
    )
    normals = np.asarray([[0.0, 0.0, -1.0]] * len(positions), dtype="<f4")
    triangles = np.asarray([[index, index + 1, index + 8] for index in range(6)], dtype="<u4")
    return NeuralSurfaceCandidate(
        positions=positions,
        normals=normals,
        triangles=triangles,
        triangle_scale_m=np.full(6, 0.005, dtype="<f4"),
        supporting_frame_count=np.asarray([3, 2, 3, 3, 3, 3], dtype="u1"),
        independent_view_count=np.asarray([2, 2, 1, 2, 2, 2], dtype="u1"),
        holdout_median_residual_m=np.asarray([0.02, 0.02, 0.02, 0.03, 0.02, 0.02], dtype="<f4"),
        holdout_p95_residual_m=np.asarray([0.05] * 6, dtype="<f4"),
        holdout_absolute_bias_m=np.asarray([0.01] * 6, dtype="<f4"),
        uncertainty_m=np.asarray([0.02] * 6, dtype="<f4"),
        observed_front=np.asarray([True, True, True, True, True, False]),
        free_space_conflict=np.asarray([False, False, False, False, True, False]),
        occlusion_conflict=np.zeros(6, dtype=bool),
        depth_boundary_conflict=np.zeros(6, dtype=bool),
        measured_surface_conflict=np.zeros(6, dtype=bool),
        source_window_sha256="a" * 64,
        near_object_size_holdout_error_mm=4.2,
    )


@pytest.mark.parametrize(
    "field",
    (
        "positions",
        "normals",
        "triangle_scale_m",
        "holdout_median_residual_m",
        "holdout_p95_residual_m",
        "holdout_absolute_bias_m",
        "uncertainty_m",
    ),
)
@pytest.mark.parametrize("invalid_kind", ("nonnumeric", "complex"))
def test_neural_surface_candidate_rejects_invalid_float_field_dtypes(
    field, invalid_kind
):
    candidate = _candidate()
    values = np.asarray(getattr(candidate, field))
    if invalid_kind == "nonnumeric":
        invalid = values.astype(str)
    else:
        invalid = values.astype(np.complex64) + np.complex64(1j)

    with pytest.raises(ValueError, match=fr"{field}.*numeric"):
        candidate.replaced(**{field: invalid})


def test_neural_surface_candidate_rejects_float_triangle_indices_before_conversion():
    candidate = _candidate()
    triangles = candidate.triangles.astype(np.float64)
    triangles[0] = np.asarray([0.9, 1.1, 2.2])

    with pytest.raises(ValueError, match="triangle.*integer"):
        candidate.replaced(triangles=triangles)


@pytest.mark.parametrize(
    "field", ("supporting_frame_count", "independent_view_count")
)
@pytest.mark.parametrize("dtype", (np.float64, np.bool_), ids=("float", "bool"))
def test_neural_surface_candidate_rejects_non_integer_count_dtypes(field, dtype):
    candidate = _candidate()
    invalid = np.asarray(getattr(candidate, field), dtype=dtype)

    with pytest.raises(ValueError, match=fr"{field}.*integer"):
        candidate.replaced(**{field: invalid})


@pytest.mark.parametrize(
    "field",
    (
        "observed_front",
        "free_space_conflict",
        "occlusion_conflict",
        "depth_boundary_conflict",
        "measured_surface_conflict",
    ),
)
def test_neural_surface_candidate_requires_boolean_visibility_and_conflict_fields(field):
    candidate = _candidate()
    invalid = np.asarray(getattr(candidate, field), dtype=np.uint8)

    with pytest.raises(ValueError, match=fr"{field}.*boolean"):
        candidate.replaced(**{field: invalid})


def test_neural_surface_candidate_preserves_valid_dtype_normalization_and_gate_behavior():
    original = _candidate()
    candidate = original.replaced(
        positions=np.asarray(original.positions, dtype=np.float64),
        normals=np.asarray(original.normals, dtype=np.int16),
        triangles=np.asarray(original.triangles, dtype=np.int64),
        triangle_scale_m=np.asarray(original.triangle_scale_m, dtype=np.float64),
        supporting_frame_count=np.asarray(
            original.supporting_frame_count, dtype=np.int16
        ),
        independent_view_count=np.asarray(
            original.independent_view_count, dtype=np.int16
        ),
        holdout_median_residual_m=np.asarray(
            original.holdout_median_residual_m, dtype=np.float64
        ),
        holdout_p95_residual_m=np.asarray(
            original.holdout_p95_residual_m, dtype=np.float64
        ),
        holdout_absolute_bias_m=np.asarray(
            original.holdout_absolute_bias_m, dtype=np.float64
        ),
        uncertainty_m=np.asarray(original.uncertainty_m, dtype=np.float64),
    )

    assert candidate.positions.dtype == np.dtype("<f4")
    assert candidate.normals.dtype == np.dtype("<f4")
    assert candidate.triangles.dtype == np.dtype("<u4")
    assert candidate.supporting_frame_count.dtype == np.dtype("u1")
    assert candidate.independent_view_count.dtype == np.dtype("u1")
    assert NeuralSurfaceGate().evaluate(candidate).accepted_triangle_indices.tolist() == [0]


def test_neural_surface_gate_accepts_only_visible_physically_anchored_nonconflicting_triangles():
    decision = NeuralSurfaceGate().evaluate(_candidate())

    assert decision.accepted_triangle_indices.tolist() == [0]
    assert decision.triangle_provenance.tolist() == [TriangleProvenance.NEURAL_MULTIVIEW]
    assert decision.triangle_uncertainty_m.tolist() == pytest.approx([0.02])
    assert set(decision.blockers) == {
        "MINIMUM_THREE_FRAME_SUPPORT_REQUIRED",
        "MINIMUM_TWO_INDEPENDENT_VIEWS_REQUIRED",
        "HOLDOUT_MEDIAN_RESIDUAL_EXCEEDED",
        "FREE_SPACE_CONFLICT",
        "UNOBSERVED_BACK_SURFACE",
    }
    assert decision.metrics["near_object_size_holdout_error_mm"] == 4.2
    assert decision.metrics["accepted_triangle_count"] == 1


def test_every_physical_conflict_and_threshold_fails_closed():
    candidate = _candidate()
    count = len(candidate.triangles)
    conflict_names = (
        "free_space_conflict",
        "occlusion_conflict",
        "depth_boundary_conflict",
        "measured_surface_conflict",
    )
    for name in conflict_names:
        values = np.zeros(count, dtype=bool)
        values[0] = True
        changed = candidate.replaced(**{name: values})
        decision = NeuralSurfaceGate().evaluate(changed)
        assert 0 not in decision.accepted_triangle_indices

    for name, rejected in (
        ("holdout_p95_residual_m", 0.061),
        ("holdout_absolute_bias_m", 0.016),
        ("uncertainty_m", 0.031),
    ):
        values = np.asarray(getattr(candidate, name)).copy()
        values[0] = rejected
        decision = NeuralSurfaceGate().evaluate(candidate.replaced(**{name: values}))
        assert 0 not in decision.accepted_triangle_indices


def test_hybrid_assembler_keeps_measured_authority_ahead_of_gated_neural_faces():
    measured = MeshTileRevision(
        tile_id=TileId(0, 0, 0),
        revision_id="b" * 64,
        positions=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4"),
        normals=np.asarray([[0, 0, 1]] * 3, dtype="<f4"),
        triangles=np.asarray([[0, 1, 2]], dtype="<u4"),
        triangle_scale_m=np.asarray([0.01], dtype="<f4"),
        unresolved_surfel_count=0,
        triangle_provenance=np.asarray([TriangleProvenance.MEASURED_LIDAR], dtype="u1"),
        triangle_uncertainty_m=np.asarray([0.008], dtype="<f4"),
    )
    decision = NeuralSurfaceGate().evaluate(_candidate())

    merged = HybridSurfaceAssembler().merge(measured, _candidate(), decision)

    assert merged.triangle_provenance.tolist() == [
        TriangleProvenance.MEASURED_LIDAR,
        TriangleProvenance.NEURAL_MULTIVIEW,
    ]
    assert merged.triangle_uncertainty_m.tolist() == pytest.approx([0.008, 0.02])
    assert np.array_equal(merged.triangles[0], measured.triangles[0])
    assert merged.revision_id != measured.revision_id


@pytest.mark.parametrize(
    "indices",
    (
        np.asarray([[0]], dtype=np.int64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([-1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
    ),
    ids=("non-1d", "float", "negative", "duplicate"),
)
def test_neural_surface_decision_rejects_ambiguous_indices(indices):
    decision = NeuralSurfaceGate().evaluate(_candidate())

    with pytest.raises(ValueError, match="indices"):
        replace(decision, accepted_triangle_indices=indices)


def test_neural_surface_decision_rejects_out_of_range_index_for_candidate():
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)
    adversarial = replace(
        decision,
        accepted_triangle_indices=np.asarray([len(candidate.triangles)], dtype=np.int64),
        triangle_provenance=np.asarray(
            [TriangleProvenance.NEURAL_MULTIVIEW], dtype="u1"
        ),
        triangle_uncertainty_m=np.asarray([0.02], dtype="<f4"),
    )

    with pytest.raises(ValueError, match="out of range"):
        adversarial.validate_for(candidate)


def test_neural_surface_decision_is_bound_to_source_candidate_and_metrics():
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)

    with pytest.raises(ValueError, match="source window"):
        decision.validate_for(candidate.replaced(source_window_sha256="b" * 64))

    for name, value in (
        ("candidate_triangle_count", len(candidate.triangles) + 1),
        ("accepted_triangle_count", len(decision.accepted_triangle_indices) + 1),
    ):
        metrics = dict(decision.metrics)
        metrics[name] = value
        with pytest.raises(ValueError, match=name):
            replace(decision, metrics=metrics).validate_for(candidate)


def test_neural_surface_decision_validates_selected_evidence_exactly():
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)

    with pytest.raises(ValueError, match="provenance"):
        NeuralSurfaceDecision(
            accepted_triangle_indices=decision.accepted_triangle_indices,
            triangle_provenance=np.asarray(
                [[TriangleProvenance.NEURAL_MULTIVIEW]], dtype="u1"
            ),
            triangle_uncertainty_m=decision.triangle_uncertainty_m,
            candidate_revision_id=decision.candidate_revision_id,
            gate_policy_revision_id=decision.gate_policy_revision_id,
            blockers=decision.blockers,
            metrics=decision.metrics,
        )

    wrong_uncertainty = np.asarray(decision.triangle_uncertainty_m).copy()
    wrong_uncertainty[0] += np.float32(0.001)
    with pytest.raises(ValueError, match="candidate uncertainty"):
        replace(decision, triangle_uncertainty_m=wrong_uncertainty).validate_for(candidate)


def test_hybrid_assembler_rejects_cross_candidate_decision():
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)
    measured = MeshTileRevision(
        tile_id=TileId(0, 0, 0),
        revision_id="b" * 64,
        positions=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4"),
        normals=np.asarray([[0, 0, 1]] * 3, dtype="<f4"),
        triangles=np.asarray([[0, 1, 2]], dtype="<u4"),
        triangle_scale_m=np.asarray([0.01], dtype="<f4"),
        unresolved_surfel_count=0,
    )

    with pytest.raises(ValueError, match="source window"):
        HybridSurfaceAssembler.merge(
            measured,
            candidate.replaced(source_window_sha256="c" * 64),
            decision,
        )


def _alter_candidate_field(candidate, field):
    if field == "near_object_size_holdout_error_mm":
        return candidate.replaced(near_object_size_holdout_error_mm=4.3)
    values = np.asarray(getattr(candidate, field)).copy()
    if field == "positions":
        values[0, 0] += np.float32(0.01)
    elif field == "normals":
        values[0] = np.asarray([0.0, 0.1, -0.99], dtype="<f4")
    elif field == "triangles":
        values[0] = np.asarray([0, 2, 8], dtype="<u4")
    elif field == "triangle_scale_m":
        values[0] += np.float32(0.001)
    elif field in {"supporting_frame_count", "independent_view_count"}:
        values[0] += 1
    elif field in {
        "holdout_median_residual_m",
        "holdout_p95_residual_m",
        "holdout_absolute_bias_m",
    }:
        values[0] -= np.float32(0.001)
    elif field == "uncertainty_m":
        values[1] += np.float32(0.001)
    else:
        values[0] = ~values[0]
    return candidate.replaced(**{field: values})


@pytest.mark.parametrize(
    "field",
    (
        "positions",
        "normals",
        "triangles",
        "triangle_scale_m",
        "supporting_frame_count",
        "independent_view_count",
        "holdout_median_residual_m",
        "holdout_p95_residual_m",
        "holdout_absolute_bias_m",
        "observed_front",
        "free_space_conflict",
        "occlusion_conflict",
        "depth_boundary_conflict",
        "measured_surface_conflict",
        "near_object_size_holdout_error_mm",
    ),
)
def test_old_decision_rejects_every_same_source_candidate_content_change(field):
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)
    changed = _alter_candidate_field(candidate, field)

    assert changed.source_window_sha256 == candidate.source_window_sha256
    assert len(changed.triangles) == len(candidate.triangles)
    assert np.array_equal(changed.uncertainty_m, candidate.uncertainty_m)
    assert changed.candidate_revision_id != candidate.candidate_revision_id
    with pytest.raises(ValueError, match="candidate revision"):
        decision.validate_for(changed)


def test_candidate_revision_is_deterministic_and_covers_uncertainty():
    first = _candidate()
    same = _candidate()
    changed = _alter_candidate_field(first, "uncertainty_m")
    decision = NeuralSurfaceGate().evaluate(first)

    assert first.candidate_revision_id == same.candidate_revision_id
    assert len(first.candidate_revision_id) == 64
    assert set(first.candidate_revision_id) <= set("0123456789abcdef")
    assert changed.candidate_revision_id != first.candidate_revision_id
    with pytest.raises(ValueError, match="candidate revision"):
        decision.validate_for(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_frames", 4),
        ("minimum_independent_views", 3),
        ("maximum_holdout_median_m", 0.026),
        ("maximum_holdout_p95_m", 0.061),
        ("maximum_absolute_bias_m", 0.016),
        ("maximum_uncertainty_m", 0.031),
        ("gate_algorithm_schema_version", 2),
    ),
)
def test_gate_policy_revision_covers_every_threshold_and_rule_version(
    monkeypatch, field, value
):
    original = NeuralSurfaceGate.policy_revision_id()
    monkeypatch.setattr(NeuralSurfaceGate, field, value)

    assert NeuralSurfaceGate.policy_revision_id() != original


def test_gate_instance_cannot_shadow_the_revision_bound_policy():
    gate = NeuralSurfaceGate()
    candidate = _candidate()
    uncertainty = np.asarray(candidate.uncertainty_m).copy()
    uncertainty[0] = 0.04
    candidate = candidate.replaced(uncertainty_m=uncertainty)
    policy_revision_id = NeuralSurfaceGate.policy_revision_id()

    with pytest.raises(AttributeError):
        gate.maximum_uncertainty_m = 0.05

    decision = gate.evaluate(candidate)

    assert 0 not in decision.accepted_triangle_indices
    assert decision.gate_policy_revision_id == policy_revision_id


def test_decision_rejects_changed_gate_policy(monkeypatch):
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)
    monkeypatch.setattr(NeuralSurfaceGate, "maximum_uncertainty_m", 0.031)

    with pytest.raises(ValueError, match="gate policy revision"):
        decision.validate_for(candidate)


def test_decision_metrics_are_immutable_and_bind_candidate_and_policy_revisions():
    candidate = _candidate()
    decision = NeuralSurfaceGate().evaluate(candidate)
    nested = replace(
        decision,
        metrics={**dict(decision.metrics), "nested": {"values": [1, 2]}},
    )

    assert decision.candidate_revision_id == candidate.candidate_revision_id
    assert decision.gate_policy_revision_id == NeuralSurfaceGate.policy_revision_id()
    with pytest.raises(TypeError):
        decision.metrics["accepted_triangle_count"] = 99
    with pytest.raises(TypeError):
        nested.metrics["nested"]["extra"] = True
    with pytest.raises(TypeError):
        nested.metrics["nested"]["values"][0] = 99


def _window(start: int = 0):
    assembler = Da3WindowAssembler(window_size=5, overlap=2)
    result = None
    for index in range(start, start + 5):
        result = assembler.submit(_observation(index))
    assert result is not None
    return result


def test_da3_multiview_worker_has_capacity_one_and_disables_only_the_failed_session():
    calls = []

    class Provider:
        def __init__(self):
            self.fail = False

        def infer_multiview(self, images, **kwargs):
            calls.append((images, kwargs))
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            return {"window": kwargs["frame_sha256s"]}

    provider = Provider()
    processed = []
    worker = Da3MultiViewWorker(provider, lambda window, prediction: processed.append((window, prediction)))
    first = _window(0)
    latest = _window(10)

    assert worker.submit("session-a", first) is True
    assert worker.submit("session-a", latest) is True
    assert worker.snapshot()["coalesced_windows"] == 1
    assert worker.run_once() is True
    assert processed[0][0].window_sha256 == latest.window_sha256
    assert len(calls[0][0]) == 5
    assert calls[0][1]["intrinsics"].shape == (5, 3, 3)
    assert calls[0][1]["extrinsics_world_to_camera"].shape == (5, 4, 4)

    provider.fail = True
    assert worker.submit("session-a", first) is True
    assert worker.run_once() is False
    failed = worker.snapshot()
    assert failed["state"] == "DA3_SESSION_DISABLED"
    assert failed["disabled_session_id"] == "session-a"
    assert "DA3_SESSION_DISABLED_OOM" in failed["blockers"]
    assert worker.submit("session-a", latest) is False

    provider.fail = False
    assert worker.submit("session-b", latest) is True
    assert worker.snapshot()["disabled_session_id"] is None
    assert worker.run_once() is True
