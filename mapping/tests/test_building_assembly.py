import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.building_project import BuildingProject
from mapping.pc.go2_mapping_live.relocalization import (
    RelocalizationCandidate,
    RelocalizationPolicy,
    RelocalizationStore,
    SessionFrame,
)
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember, BuildingAssemblyStore


SOURCE_HASH = hashlib.sha256(b"assembly-surface-config").hexdigest()
EVIDENCE_HASH = hashlib.sha256(b"assembly-relocalization-evidence").hexdigest()
IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _write_generation(
    project: BuildingProject,
    *,
    submap_id: str,
    session_id: str,
    floor_id: str,
    map_id: str,
    calibration_id: str,
    payload: bytes,
) -> str:
    witnesses = b"[]"
    manifest_core = {
        "schema_version": 1,
        "building_id": project.building_id,
        "session_id": session_id,
        "floor_id": floor_id,
        "submap_id": submap_id,
        "map_id": map_id,
        "source_config_sha256": SOURCE_HASH,
        "calibration_id": calibration_id,
        "arrays_sha256": hashlib.sha256(payload).hexdigest(),
        "witnesses_sha256": hashlib.sha256(witnesses).hexdigest(),
    }
    generation_id = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    generation = project.root / "submaps" / submap_id / "generations" / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "surfels.npz").write_bytes(payload)
    (generation / "witnesses.json").write_bytes(witnesses)
    (generation / "manifest.json").write_text(
        json.dumps({**manifest_core, "generation_id": generation_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return generation_id


def _project_with_sealed_root(tmp_path):
    root = tmp_path / "building"
    project = BuildingProject(root, building_id="building-a")
    project.bootstrap(
        floor_id="floor-01",
        floor_label="Level 1",
        zone_id="room-101",
        zone_kind="ROOM",
        submap_id="submap-root",
        session_id="session-root",
        map_id="map-root",
        producer_instance_id="producer-root",
        source_config_sha256=SOURCE_HASH,
        start_sequence=1,
    )
    project.begin_handoff(old_submap_id="submap-root", new_submap_id="submap-root-live", cutoff_sequence=3)
    generation_id = _write_generation(
        project,
        submap_id="submap-root",
        session_id="session-root",
        floor_id="floor-01",
        map_id="map-root",
        calibration_id="PREVIEW_UNCALIBRATED",
        payload=b"root-surface",
    )
    project.seal_submap(
        "submap-root",
        generation_id=generation_id,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(4.0, 4.0, 3.0),
    )
    return project, generation_id


def _add_sealed_detached_source(project: BuildingProject):
    submap_id = project.ensure_session_route(
        session_id="session-live",
        map_id="map-live",
        producer_instance_id="producer-live",
        source_config_sha256=SOURCE_HASH,
        calibration_id="calibration-live",
        start_sequence=1,
    )
    generation_id = _write_generation(
        project,
        submap_id=submap_id,
        session_id="session-live",
        floor_id="floor-unassigned",
        map_id="map-live",
        calibration_id="calibration-live",
        payload=b"detached-surface",
    )
    project.begin_session_close("session-live", 1)
    project.complete_session_close(
        submap_id,
        generation_id=generation_id,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        cutoff_sequence=1,
    )
    return submap_id, generation_id


def _accepted_relocalization(tmp_path, project, source_submap_id, source_generation_id, target_generation_id):
    store = RelocalizationStore(tmp_path / "relocalization", project=project)
    frame = SessionFrame(
        frame_id="frame-live",
        building_id="building-a",
        session_id="session-live",
        source_config_sha256=SOURCE_HASH,
        calibration_id="calibration-live",
        frame_kind="LIO_MAP",
    )
    store.register_session_frame(frame)
    store.bind_source_generation(frame.frame_id, source_submap_id, source_generation_id)
    policy = RelocalizationPolicy(
        min_overlap_m2=1.0,
        min_independent_keyframes=3,
        max_translation_residual_m=0.1,
        max_rotation_residual_deg=3.0,
        max_covariance_trace=0.2,
        algorithm_version="relocalizer-v1",
    )
    store.start_attempt("building-a", "session-live", "attempt-live", frame.frame_id, policy)
    store.propose(
        RelocalizationCandidate(
            candidate_id="candidate-live",
            attempt_id="attempt-live",
            building_id="building-a",
            session_id="session-live",
            floor_id="floor-01",
            target_submap_id="submap-root",
            target_generation_id=target_generation_id,
            source_frame_id=frame.frame_id,
            transform_target_from_source_frame=IDENTITY,
            overlap_m2=1.5,
            independent_keyframes=4,
            translation_residual_m=0.05,
            rotation_residual_deg=1.0,
            covariance_trace=0.1,
            identity_compatible=True,
            graph_consistent=True,
            forbidden_component_bridge=False,
            evidence_sha256=EVIDENCE_HASH,
        )
    )
    return store, store.seal_and_resolve("attempt-live")


def _member(project, submap_id, generation_id, *, decision_id=None, transform=IDENTITY):
    submap = project.submap(submap_id)
    return AssemblyMember(
        building_id=project.building_id,
        component_id=submap["component_id"],
        session_id=submap["session_id"],
        submap_id=submap_id,
        generation_id=generation_id,
        transform_building_from_submap=transform,
        source_decision_id=decision_id,
    )


def test_root_assembly_is_content_addressed_and_restartable(tmp_path):
    project, root_generation_id = _project_with_sealed_root(tmp_path)
    store = BuildingAssemblyStore(tmp_path / "assemblies", project=project)

    assembly = store.publish(
        (_member(project, "submap-root", root_generation_id),),
        algorithm_version="assembly-v1",
    )

    assert len(assembly.assembly_id) == 64
    assert assembly.members[0].component_id == project.snapshot()["root_component_id"]
    assert assembly.members[0].transform_building_from_submap == IDENTITY
    assert store.current() == assembly
    restarted = BuildingAssemblyStore(tmp_path / "assemblies", project=project)
    assert restarted.current() == assembly
    assert restarted.publish(assembly.members, algorithm_version="assembly-v1") == assembly


def test_attached_member_requires_authoritative_relocalization_and_target_member(tmp_path):
    project, root_generation_id = _project_with_sealed_root(tmp_path)
    source_submap_id, source_generation_id = _add_sealed_detached_source(project)
    relocalization, decision = _accepted_relocalization(
        tmp_path, project, source_submap_id, source_generation_id, root_generation_id
    )
    store = BuildingAssemblyStore(tmp_path / "assemblies", project=project, relocalization=relocalization)
    root = _member(project, "submap-root", root_generation_id)
    attached = _member(
        project,
        source_submap_id,
        source_generation_id,
        decision_id=decision.decision_id,
    )

    assembly = store.publish((attached, root), algorithm_version="assembly-v1")

    assert [member.submap_id for member in assembly.members] == ["submap-root", source_submap_id]
    assert assembly.members[1].source_decision_id == decision.decision_id
    assert store.load(assembly.assembly_id) == assembly


def test_assembly_rejects_unsealed_or_unverified_members_without_changing_current(tmp_path):
    project, root_generation_id = _project_with_sealed_root(tmp_path)
    source_submap_id, source_generation_id = _add_sealed_detached_source(project)
    store = BuildingAssemblyStore(tmp_path / "assemblies", project=project)
    root = _member(project, "submap-root", root_generation_id)

    with pytest.raises(ValueError, match="authoritative relocalization"):
        store.publish(
            (
                root,
                _member(project, source_submap_id, source_generation_id),
            ),
            algorithm_version="assembly-v1",
        )
    assert store.current() is None


def test_assembly_manifest_and_current_pointer_tampering_fail_closed(tmp_path):
    project, root_generation_id = _project_with_sealed_root(tmp_path)
    store = BuildingAssemblyStore(tmp_path / "assemblies", project=project)
    assembly = store.publish((_member(project, "submap-root", root_generation_id),), algorithm_version="assembly-v1")

    manifest_path = tmp_path / "assemblies" / "assemblies" / assembly.assembly_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["algorithm_version"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="assembly manifest"):
        store.load(assembly.assembly_id)

    (tmp_path / "assemblies" / "CURRENT").write_text("not-a-sha\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CURRENT"):
        store.current()
