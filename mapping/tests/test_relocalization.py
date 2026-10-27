from dataclasses import replace
import hashlib
import json
import sqlite3

import pytest

from mapping.pc.go2_mapping_live.building_project import BuildingProject
from mapping.pc.go2_mapping_live.relocalization import (
    RelocalizationCandidate,
    RelocalizationPolicy,
    RelocalizationStatus,
    RelocalizationStore,
    SessionFrame,
)


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()
EVIDENCE_HASH = hashlib.sha256(b"relocalization-evidence").hexdigest()
IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _sealed_project(tmp_path):
    root = tmp_path / "building"
    project = BuildingProject(root, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-anchor",
        session_id="session-anchor",
        map_id="map-a",
        producer_instance_id="producer-a",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-anchor", new_submap_id="submap-live", cutoff_sequence=3)
    generation_root = root / "submaps" / "submap-anchor" / "generations"
    arrays = b"sealed surfels"
    witnesses = b"[]"
    manifest_core = {
        "schema_version": 1,
        "building_id": "building-a",
        "session_id": "session-anchor",
        "floor_id": "floor-01",
        "submap_id": "submap-anchor",
        "map_id": "map-a",
        "source_config_sha256": SOURCE_HASH,
        "calibration_id": "PREVIEW_UNCALIBRATED",
        "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
        "witnesses_sha256": hashlib.sha256(witnesses).hexdigest(),
    }
    generation_id = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    generation = generation_root / generation_id
    generation.mkdir(parents=True)
    (generation / "surfels.npz").write_bytes(arrays)
    (generation / "witnesses.json").write_bytes(witnesses)
    (generation / "manifest.json").write_text(
        json.dumps({**manifest_core, "generation_id": generation_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    project.seal_submap(
        "submap-anchor",
        generation_id=generation_id,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(4.0, 4.0, 3.0),
    )
    return project, generation_id


def _write_source_generation(project, *, submap_id, session_id, map_id, calibration_id):
    arrays = b"source surfels"
    witnesses = b"[]"
    manifest_core = {
        "schema_version": 1,
        "building_id": project.building_id,
        "session_id": session_id,
        "floor_id": "floor-unassigned",
        "submap_id": submap_id,
        "map_id": map_id,
        "source_config_sha256": SOURCE_HASH,
        "calibration_id": calibration_id,
        "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
        "witnesses_sha256": hashlib.sha256(witnesses).hexdigest(),
    }
    generation_id = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    generation = project.root / "submaps" / submap_id / "generations" / generation_id
    generation.mkdir(parents=True)
    (generation / "surfels.npz").write_bytes(arrays)
    (generation / "witnesses.json").write_bytes(witnesses)
    (generation / "manifest.json").write_text(
        json.dumps({**manifest_core, "generation_id": generation_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return generation_id


def _source_generation(
    project,
    *,
    session_id="session-live",
    map_id="map-source",
    producer_id="producer-source",
    calibration_id="calibration-a",
):
    submap_id = project.ensure_session_route(
        session_id=session_id,
        map_id=map_id,
        producer_instance_id=producer_id,
        source_config_sha256=SOURCE_HASH,
        calibration_id=calibration_id,
        start_sequence=1,
    )
    generation_id = _write_source_generation(
        project,
        submap_id=submap_id,
        session_id=session_id,
        map_id=map_id,
        calibration_id=calibration_id,
    )
    project.begin_session_close(session_id, 1)
    project.complete_session_close(
        submap_id,
        generation_id=generation_id,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        cutoff_sequence=1,
    )
    return submap_id, generation_id


def _frame(**overrides):
    values = {
        "frame_id": "frame-session-live-map",
        "building_id": "building-a",
        "session_id": "session-live",
        "source_config_sha256": SOURCE_HASH,
        "calibration_id": "calibration-a",
        "frame_kind": "LIO_MAP",
    }
    values.update(overrides)
    return SessionFrame(**values)


def _policy(**overrides):
    values = {
        "min_overlap_m2": 1.0,
        "min_independent_keyframes": 3,
        "max_translation_residual_m": 0.10,
        "max_rotation_residual_deg": 3.0,
        "max_covariance_trace": 0.20,
        "algorithm_version": "relocalizer-v1",
    }
    values.update(overrides)
    return RelocalizationPolicy(**values)


def _candidate(generation_id, **overrides):
    values = {
        "candidate_id": "candidate-a",
        "attempt_id": "attempt-a",
        "building_id": "building-a",
        "session_id": "session-live",
        "floor_id": "floor-01",
        "target_submap_id": "submap-anchor",
        "target_generation_id": generation_id,
        "source_frame_id": "frame-session-live-map",
        "transform_target_from_source_frame": IDENTITY,
        "overlap_m2": 1.5,
        "independent_keyframes": 4,
        "translation_residual_m": 0.05,
        "rotation_residual_deg": 1.0,
        "covariance_trace": 0.10,
        "identity_compatible": True,
        "graph_consistent": True,
        "forbidden_component_bridge": False,
        "evidence_sha256": EVIDENCE_HASH,
    }
    values.update(overrides)
    return RelocalizationCandidate(**values)


def _open_store(tmp_path):
    project, generation_id = _sealed_project(tmp_path)
    source_submap_id, source_generation_id = _source_generation(project)
    store = RelocalizationStore(tmp_path / "relocalization", project=project)
    frame = _frame()
    store.register_session_frame(frame)
    store.bind_source_generation(frame.frame_id, source_submap_id, source_generation_id)
    store.start_attempt("building-a", "session-live", "attempt-a", frame.frame_id, _policy())
    return project, store, generation_id


def test_unbound_session_frame_cannot_start_a_relocalization_attempt(tmp_path):
    project, _ = _sealed_project(tmp_path)
    store = RelocalizationStore(tmp_path / "relocalization", project=project)
    frame = _frame()
    store.register_session_frame(frame)

    with pytest.raises(ValueError, match="source generation binding"):
        store.start_attempt("building-a", "session-live", "attempt-a", frame.frame_id, _policy())


def test_source_generation_binding_is_idempotent_and_conflicts_are_rejected(tmp_path):
    project, _ = _sealed_project(tmp_path)
    source_submap_id, source_generation_id = _source_generation(project)
    store = RelocalizationStore(tmp_path / "relocalization", project=project)
    frame = _frame()
    store.register_session_frame(frame)

    authority = store.bind_source_generation(frame.frame_id, source_submap_id, source_generation_id)

    assert authority.source_map_id == "map-source"
    assert store.source_generation(frame.frame_id) == authority
    assert store.bind_source_generation(frame.frame_id, source_submap_id, source_generation_id) == authority
    with pytest.raises(ValueError, match="conflicting"):
        store.bind_source_generation(frame.frame_id, "submap-anchor", source_generation_id)


def _accepted_store(tmp_path):
    project, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id))
    decision = store.seal_and_resolve("attempt-a")
    assert decision.status is RelocalizationStatus.ACCEPTED
    return project, store, generation_id, decision


def test_unique_passing_candidate_is_accepted_with_stable_source_frame_and_recovers_after_restart(tmp_path):
    project, store, generation_id = _open_store(tmp_path)
    candidate = _candidate(generation_id)

    assert store.propose(candidate) == candidate
    assert store.propose(candidate) == candidate
    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.ACCEPTED
    assert decision.reason_code == "ACCEPTED_UNIQUE_CANDIDATE"
    assert decision.accepted_candidate_id == candidate.candidate_id
    assert decision.source_frame_id == candidate.source_frame_id
    assert decision.transform_target_from_source_frame == IDENTITY
    assert store.seal_and_resolve("attempt-a") == decision
    assert store.authoritative_decision("building-a", "session-live") == decision
    with pytest.raises(ValueError, match="OPEN"):
        store.propose(replace(candidate, candidate_id="candidate-late"))
    with pytest.raises(ValueError, match="accepted"):
        store.start_attempt("building-a", "session-live", "attempt-b", candidate.source_frame_id, _policy())

    database_path = store.database_path
    store.close()
    restarted = RelocalizationStore(database_path.parent, project=project)
    assert restarted.session_frame(candidate.source_frame_id) == _frame()
    assert restarted.decision("attempt-a") == decision
    assert restarted.authoritative_decision("building-a", "session-live") == decision


def test_no_passing_candidate_is_terminal_and_allows_a_new_attempt(tmp_path):
    _, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id, overlap_m2=0.5))

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code == "NO_CANDIDATE_PASSED"
    assert decision.accepted_candidate_id is None
    assert decision.source_frame_id is None
    assert decision.transform_target_from_source_frame is None
    assert store.authoritative_decision("building-a", "session-live") is None
    store.start_attempt("building-a", "session-live", "attempt-b", _frame().frame_id, _policy())


def test_two_passing_candidates_are_ambiguous_and_candidate_order_is_deterministic(tmp_path):
    _, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id, candidate_id="candidate-z"))
    store.propose(_candidate(generation_id, candidate_id="candidate-a"))

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.AMBIGUOUS
    assert decision.reason_code == "MULTIPLE_CANDIDATES_PASSED"
    assert decision.candidate_ids == ("candidate-a", "candidate-z")
    assert store.authoritative_decision("building-a", "session-live") is None


def test_candidate_gates_are_all_fail_closed(tmp_path):
    gate_overrides = (
        {"identity_compatible": False},
        {"graph_consistent": False},
        {"forbidden_component_bridge": True},
        {"overlap_m2": 0.99},
        {"independent_keyframes": 2},
        {"translation_residual_m": 0.11},
        {"rotation_residual_deg": 3.01},
        {"covariance_trace": 0.21},
    )
    for index, overrides in enumerate(gate_overrides):
        case_root = tmp_path / f"case-{index}"
        _, store, generation_id = _open_store(case_root)
        store.propose(_candidate(generation_id, **overrides))
        assert store.seal_and_resolve("attempt-a").status is RelocalizationStatus.INSUFFICIENT_EVIDENCE


def test_validation_rejects_nonfinite_metrics_invalid_transforms_and_conflicting_ids(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        _policy(min_overlap_m2=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        _policy(min_independent_keyframes=0)
    with pytest.raises(ValueError, match="rigid"):
        _candidate("a" * 64, transform_target_from_source_frame=(0.0,) * 16)
    with pytest.raises(ValueError, match="finite"):
        _candidate("a" * 64, covariance_trace=float("inf"))
    with pytest.raises(ValueError, match="SHA-256"):
        _frame(source_config_sha256="bad")

    _, store, generation_id = _open_store(tmp_path)
    candidate = _candidate(generation_id)
    store.propose(candidate)
    with pytest.raises(ValueError, match="conflicting"):
        store.propose(replace(candidate, overlap_m2=2.0))


def test_session_frame_and_open_attempt_ids_are_immutable(tmp_path):
    project, generation_id = _sealed_project(tmp_path)
    store = RelocalizationStore(tmp_path / "relocalization", project=project)
    frame = _frame()
    assert store.register_session_frame(frame) == frame
    assert store.register_session_frame(frame) == frame
    with pytest.raises(ValueError, match="conflicting"):
        store.register_session_frame(replace(frame, calibration_id="calibration-b"))

    source_submap_id, source_generation_id = _source_generation(project)
    store.bind_source_generation(frame.frame_id, source_submap_id, source_generation_id)
    store.start_attempt("building-a", "session-live", "attempt-a", frame.frame_id, _policy())
    store.start_attempt("building-a", "session-live", "attempt-a", frame.frame_id, _policy())
    with pytest.raises(ValueError, match="OPEN"):
        store.start_attempt("building-a", "session-live", "attempt-b", frame.frame_id, _policy())
    with pytest.raises(ValueError, match="exact.*generation"):
        store.propose(_candidate("b" * 64))
    assert generation_id


def test_target_corruption_at_seal_produces_immutable_terminal_failure(tmp_path):
    _, store, generation_id = _open_store(tmp_path)
    candidate = _candidate(generation_id)
    store.propose(candidate)
    manifest_path = (
        tmp_path / "building" / "submaps" / "submap-anchor" / "generations" / generation_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["building_id"] = "building-corrupt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code == "TARGET_REVALIDATION_FAILED"
    assert store.seal_and_resolve("attempt-a") == decision
    with pytest.raises(ValueError, match="OPEN"):
        store.propose(replace(candidate, candidate_id="candidate-late"))


def test_decision_hash_corruption_fails_closed_after_restart(tmp_path):
    project, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id))
    store.seal_and_resolve("attempt-a")
    database_path = store.database_path
    store.close()
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE decisions SET canonical_json='{}' WHERE attempt_id='attempt-a'")
        connection.commit()

    restarted = RelocalizationStore(database_path.parent, project=project)
    with pytest.raises(ValueError, match="corrupt"):
        restarted.decision("attempt-a")


@pytest.mark.parametrize(
    ("table", "reason_code"),
    (
        ("session_frames", "SOURCE_FRAME_REVALIDATION_FAILED"),
        ("policies", "POLICY_REVALIDATION_FAILED"),
        ("candidates", "CANDIDATE_REVALIDATION_FAILED"),
    ),
)
def test_record_corruption_at_seal_is_terminal_with_specific_reason(tmp_path, table, reason_code):
    _, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id))
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"UPDATE {table} SET canonical_json='{{}}'")
        connection.commit()

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code == reason_code
    with pytest.raises(ValueError, match="OPEN"):
        store.propose(replace(_candidate(generation_id), candidate_id="candidate-late"))


def test_existing_authoritative_acceptance_supersedes_and_seals_open_attempt(tmp_path):
    _, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id))
    with sqlite3.connect(store.database_path) as connection:
        source_frame_id, policy_sha256 = connection.execute(
            "SELECT source_frame_id,policy_sha256 FROM attempts WHERE attempt_id='attempt-a'"
        ).fetchone()
        content = json.dumps({"simulated": "concurrent acceptance"}, separators=(",", ":"))
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)",
            (
                "attempt-authoritative",
                "building-a",
                "session-live",
                source_frame_id,
                policy_sha256,
                content,
                content_sha256,
                "SEALED",
            ),
        )
        decision_json = json.dumps({"simulated": "accepted"}, separators=(",", ":"))
        decision_sha256 = hashlib.sha256(decision_json.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO decisions VALUES(?,?,?,?,?,?)",
            (
                "decision-authoritative",
                "attempt-authoritative",
                "building-a",
                "session-live",
                decision_json,
                decision_sha256,
            ),
        )
        connection.execute(
            "INSERT INTO authoritative_acceptances VALUES(?,?,?)",
            ("building-a", "session-live", "decision-authoritative"),
        )
        connection.commit()

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code == "SUPERSEDED_BY_ACCEPTED_SESSION"
    with pytest.raises(ValueError, match="OPEN"):
        store.propose(replace(_candidate(generation_id), candidate_id="candidate-late"))


@pytest.mark.parametrize("table", ("session_frames", "policies", "candidates"))
def test_authoritative_read_revalidates_linked_evidence_after_acceptance(tmp_path, table):
    _, store, _, decision = _accepted_store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"UPDATE {table} SET canonical_json='{{}}'")
        connection.commit()

    assert store.decision("attempt-a") == decision
    with pytest.raises(ValueError, match="authoritative"):
        store.authoritative_decision("building-a", "session-live")


def test_authoritative_read_revalidates_target_artifacts_after_acceptance(tmp_path):
    _, store, generation_id, decision = _accepted_store(tmp_path)
    target = tmp_path / "building" / "submaps" / "submap-anchor" / "generations" / generation_id
    (target / "surfels.npz").write_bytes(b"corrupt after acceptance")

    assert store.decision("attempt-a") == decision
    with pytest.raises(ValueError, match="authoritative"):
        store.authoritative_decision("building-a", "session-live")


def test_source_generation_corruption_at_seal_is_terminal(tmp_path):
    _, store, generation_id = _open_store(tmp_path)
    store.propose(_candidate(generation_id))
    source = store.source_generation(_frame().frame_id)
    source_path = store.project.root / "submaps" / source.submap_id / "generations" / source.generation_id
    (source_path / "surfels.npz").write_bytes(b"corrupt source")

    decision = store.seal_and_resolve("attempt-a")

    assert decision.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code == "SOURCE_GENERATION_REVALIDATION_FAILED"


def test_authoritative_read_revalidates_source_generation_after_acceptance(tmp_path):
    _, store, _, decision = _accepted_store(tmp_path)
    source = store.source_generation(_frame().frame_id)
    source_path = store.project.root / "submaps" / source.submap_id / "generations" / source.generation_id
    (source_path / "witnesses.json").write_bytes(b"corrupt source witnesses")

    assert store.decision("attempt-a") == decision
    with pytest.raises(ValueError, match="authoritative"):
        store.authoritative_decision("building-a", "session-live")


def test_authoritative_mapping_rejects_a_nonaccepted_decision(tmp_path):
    project, store, _, _ = _accepted_store(tmp_path)
    other_frame = _frame(frame_id="frame-other", session_id="session-other")
    store.register_session_frame(other_frame)
    source_submap_id, source_generation_id = _source_generation(
        project,
        session_id="session-other",
        map_id="map-source-other",
        producer_id="producer-source-other",
    )
    store.bind_source_generation(other_frame.frame_id, source_submap_id, source_generation_id)
    store.start_attempt("building-a", "session-other", "attempt-other", other_frame.frame_id, _policy())
    other = store.seal_and_resolve("attempt-other")
    assert other.status is RelocalizationStatus.INSUFFICIENT_EVIDENCE
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE authoritative_acceptances SET decision_id=? WHERE building_id='building-a' AND session_id='session-live'",
            (other.decision_id,),
        )
        connection.commit()

    with pytest.raises(ValueError, match="authoritative"):
        store.authoritative_decision("building-a", "session-live")


def test_authoritative_mapping_rejects_an_accepted_decision_from_another_session(tmp_path):
    project, store, generation_id, _ = _accepted_store(tmp_path)
    other_frame = _frame(frame_id="frame-other", session_id="session-other")
    store.register_session_frame(other_frame)
    source_submap_id, source_generation_id = _source_generation(
        project,
        session_id="session-other",
        map_id="map-source-other",
        producer_id="producer-source-other",
    )
    store.bind_source_generation(other_frame.frame_id, source_submap_id, source_generation_id)
    store.start_attempt("building-a", "session-other", "attempt-other", other_frame.frame_id, _policy())
    store.propose(
        _candidate(
            generation_id,
            candidate_id="candidate-other",
            attempt_id="attempt-other",
            session_id="session-other",
            source_frame_id=other_frame.frame_id,
        )
    )
    other = store.seal_and_resolve("attempt-other")
    assert other.status is RelocalizationStatus.ACCEPTED
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "DELETE FROM authoritative_acceptances WHERE building_id='building-a' AND session_id='session-other'"
        )
        connection.execute(
            "UPDATE authoritative_acceptances SET decision_id=? WHERE building_id='building-a' AND session_id='session-live'",
            (other.decision_id,),
        )
        connection.commit()

    with pytest.raises(ValueError, match="authoritative"):
        store.authoritative_decision("building-a", "session-live")
