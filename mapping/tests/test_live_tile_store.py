import errno
import hashlib
import json
import multiprocessing
import os
from queue import Empty
import struct
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mapping.pc.go2_mapping_live.live_surface import TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import MeshTileRevision, TriangleProvenance
from mapping.pc.go2_mapping_live import live_tiles
from mapping.pc.go2_mapping_live.live_tiles import (
    ImmutableTileStore,
    TilePublicationRequest,
    decode_surface_tile,
)
from mapping.pc.go2_mapping_core.building_assembly import AssemblyMember


def _revision() -> MeshTileRevision:
    positions = np.asarray([[0, 0, 0], [0.005, 0, 0], [0, 0.005, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 3, dtype="<f4")
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    scales = np.asarray([0.005], dtype="<f4")
    digest = hashlib.sha256(positions.tobytes() + normals.tobytes() + triangles.tobytes() + scales.tobytes()).hexdigest()
    return MeshTileRevision(TileId(0, 0, 0), digest, positions, normals, triangles, scales, 2)


class _StringifiableSha256:
    def __str__(self) -> str:
        return "a" * 64


@pytest.mark.parametrize(
    "field", ("revision_id", "source_generation_id", "pose_revision_id")
)
def test_tile_store_publication_rejects_non_string_sha256_values(tmp_path, field):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()
    source_generation_id = "b" * 64
    pose_revision_id = "c" * 64
    invalid = _StringifiableSha256()
    if field == "revision_id":
        revision = replace(revision, revision_id=invalid)
    elif field == "source_generation_id":
        source_generation_id = invalid
    else:
        pose_revision_id = invalid
    before_revisions = set(store.revisions_root.iterdir())

    with pytest.raises(ValueError, match=field):
        store.publish(
            "building-a",
            "floor-01",
            "component-root",
            "ROOT",
            "submap-01",
            revision,
            source_generation_id=source_generation_id,
            pose_revision_id=pose_revision_id,
        )

    assert not store.index_path.exists()
    assert set(store.revisions_root.iterdir()) == before_revisions


def test_tile_store_publishes_immutable_browser_binary_and_manifest(tmp_path):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()

    metadata = store.publish("building-a", "floor-01", "component-root", "ROOT", "submap-01", revision)
    loaded, body = store.load(metadata["revision_id"])

    assert metadata == loaded
    assert metadata["authority"] == "LIVE_PROVISIONAL_SURFACE"
    assert metadata["quality"] == "PREVIEW_UNCALIBRATED"
    assert metadata["vertex_count"] == 3
    assert metadata["triangle_count"] == 1
    assert metadata["geometry_revision_id"] == revision.revision_id
    assert metadata["component_id"] == "component-root"
    assert struct.unpack_from("<4sIIII", body, 0) == (b"G2TM", 2, 3, 1, 1)
    decoded = decode_surface_tile(body)
    assert decoded.binary_schema_version == 2
    assert decoded.triangle_provenance.tolist() == [TriangleProvenance.MEASURED_LIDAR]
    assert decoded.triangle_uncertainty_m.tolist() == pytest.approx([0.005])
    assert metadata["binary_schema_version"] == 2
    assert metadata["provenance_encoding"] == {
        "1": "MEASURED_LIDAR",
        "2": "MEASURED_RGBD",
        "3": "NEURAL_MULTIVIEW",
    }
    assert store.manifest()["tiles"][0]["revision_id"] == metadata["revision_id"]
    assert store.publish("building-a", "floor-01", "component-root", "ROOT", "submap-01", revision) == metadata


@pytest.mark.parametrize(
    "scale",
    [0.0, -0.01, np.nan, np.inf],
    ids=["zero", "negative", "nan", "infinite"],
)
def test_tile_store_rejects_invalid_scale_without_replacing_current(tmp_path, scale):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()
    current = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", revision
    )
    before_exists = store.index_path.is_file()
    before_bytes = store.index_path.read_bytes()
    invalid = replace(
        revision,
        revision_id=hashlib.sha256(
            np.asarray([scale], dtype="<f4").tobytes()
        ).hexdigest(),
        triangle_scale_m=np.asarray([scale], dtype="<f4"),
        triangle_uncertainty_m=np.asarray([0.008], dtype="<f4"),
    )

    with pytest.raises(ValueError, match="scale"):
        store.publish(
            "building-a", "floor-01", "component-root", "ROOT", "submap-01", invalid
        )

    assert store.index_path.is_file() is before_exists
    assert store.index_path.read_bytes() == before_bytes
    assert store.manifest()["tiles"][0]["revision_id"] == current["revision_id"]


def test_surface_tile_v2_round_trips_mixed_per_triangle_evidence_and_v1_remains_readable():
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 4, dtype="<f4")
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype="<u4")
    scales = np.asarray([0.02, 0.005], dtype="<f4")
    provenance = np.asarray(
        [TriangleProvenance.MEASURED_RGBD, TriangleProvenance.NEURAL_MULTIVIEW],
        dtype="u1",
    )
    uncertainty = np.asarray([0.008, 0.025], dtype="<f4")
    revision = MeshTileRevision(
        TileId(0, 0, 0),
        "d" * 64,
        positions,
        normals,
        triangles,
        scales,
        0,
        provenance,
        uncertainty,
    )

    decoded = decode_surface_tile(ImmutableTileStore._body(revision))
    assert decoded.triangle_provenance.tolist() == provenance.tolist()
    assert decoded.triangle_uncertainty_m.tolist() == pytest.approx(uncertainty.tolist())

    legacy_body = b"".join(
        (
            struct.pack("<4sIIII", b"G2TM", 1, 4, 2, 2),
            positions.tobytes(),
            normals.tobytes(),
            triangles.tobytes(),
            scales.tobytes(),
        )
    )
    legacy = decode_surface_tile(legacy_body)
    assert legacy.binary_schema_version == 1
    assert legacy.triangle_provenance.tolist() == [
        TriangleProvenance.MEASURED_LIDAR,
        TriangleProvenance.MEASURED_LIDAR,
    ]
    assert legacy.triangle_uncertainty_m.tolist() == pytest.approx(scales.tolist())


def test_identical_geometry_has_component_scoped_revision_and_building_scope_excludes_detached(tmp_path):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()

    root = store.publish("building-a", "floor-01", "component-root", "ROOT", "submap-root", revision)
    detached = store.publish(
        "building-a", "floor-unassigned", "component-detached", "DETACHED", "submap-detached", revision
    )

    assert root["geometry_revision_id"] == detached["geometry_revision_id"] == revision.revision_id
    assert root["revision_id"] != detached["revision_id"]
    assert [tile["revision_id"] for tile in store.manifest(scope="building")["tiles"]] == [root["revision_id"]]
    assert len(store.manifest(scope="local")["tiles"]) == 2


def test_schema_v1_tile_index_remains_diagnostic_only(tmp_path):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()
    published = store.publish("building-a", "floor-01", "component-root", "ROOT", "submap-root", revision)
    metadata_path = tmp_path / "revisions" / published["revision_id"] / "metadata.json"
    legacy = {
        key: published[key]
        for key in (
            "schema_version",
            "building_id",
            "floor_id",
            "submap_id",
            "tile_id",
            "revision_id",
            "content_sha256",
            "vertex_count",
            "triangle_count",
            "unresolved_surfel_count",
            "authority",
            "quality",
            "geometry_authority",
            "binary_url",
        )
    }
    legacy["schema_version"] = 1
    metadata_path.write_text(json.dumps(legacy), encoding="utf-8")
    (tmp_path / "current-tiles.json").write_text(
        json.dumps({"schema_version": 1, "current": {"legacy-key": published["revision_id"]}}),
        encoding="utf-8",
    )

    assert len(store.manifest(scope="diagnostic")["tiles"]) == 1
    assert store.manifest(scope="building")["tiles"] == []


def test_schema_v2_tile_metadata_tampering_cannot_promote_detached_geometry(tmp_path):
    store = ImmutableTileStore(tmp_path)
    metadata = store.publish(
        "building-a", "floor-unassigned", "component-detached", "DETACHED", "submap-detached", _revision()
    )
    metadata_path = tmp_path / "revisions" / metadata["revision_id"] / "metadata.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["attachment_state"] = "ROOT"
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        store.manifest(scope="building")


def test_tile_store_binds_source_generation_without_changing_binary_geometry(tmp_path):
    store = ImmutableTileStore(tmp_path)
    generation_id = "a" * 64

    metadata = store.publish(
        "building-a",
        "floor-01",
        "component-root",
        "ROOT",
        "submap-root",
        _revision(),
        source_generation_id=generation_id,
    )

    loaded, body = store.load(metadata["revision_id"])

    assert metadata["source_generation_id"] == generation_id
    assert loaded["source_generation_id"] == generation_id
    assert hashlib.sha256(body).hexdigest() == metadata["content_sha256"]
    tampered = json.loads((tmp_path / "revisions" / metadata["revision_id"] / "metadata.json").read_text())
    tampered["source_generation_id"] = "b" * 64
    (tmp_path / "revisions" / metadata["revision_id"] / "metadata.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="identity"):
        store.load(metadata["revision_id"])


@pytest.mark.parametrize("schema_version", [2, 99])
def test_non_v3_tile_cannot_inject_source_generation_authority(tmp_path, schema_version):
    store = ImmutableTileStore(tmp_path)
    metadata = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-root", _revision()
    )
    metadata_path = tmp_path / "revisions" / metadata["revision_id"] / "metadata.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["schema_version"] = schema_version
    tampered["source_generation_id"] = "a" * 64
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")

    member = AssemblyMember(
        "building-a",
        "component-root",
        "session-root",
        "submap-root",
        "a" * 64,
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="schema|source generation"):
        store.manifest_for_assembly(SimpleNamespace(assembly_id="b" * 64, members=(member,)))


def test_assembly_manifest_resolves_only_referenced_generation_bound_tiles(tmp_path):
    store = ImmutableTileStore(tmp_path)
    revision = _revision()
    root_generation = "a" * 64
    attached_generation = "b" * 64
    ignored_generation = "c" * 64
    root = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-root", revision,
        source_generation_id=root_generation,
    )
    attached = store.publish(
        "building-a", "floor-unassigned", "component-attached", "DETACHED", "submap-attached", revision,
        source_generation_id=attached_generation,
    )
    store.publish(
        "building-a", "floor-unassigned", "component-ignored", "DETACHED", "submap-ignored", revision,
        source_generation_id=ignored_generation,
    )
    root_member = AssemblyMember(
        "building-a", "component-root", "session-root", "submap-root", root_generation,
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    attached_transform = (1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    attached_member = AssemblyMember(
        "building-a", "component-attached", "session-attached", "submap-attached", attached_generation,
        attached_transform, "d" * 64,
    )
    manifest = store.manifest_for_assembly(
        SimpleNamespace(assembly_id="e" * 64, members=(root_member, attached_member))
    )

    assert manifest["scope"] == "building"
    assert manifest["assembly_id"] == "e" * 64
    assert {tile["submap_id"] for tile in manifest["tiles"]} == {"submap-root", "submap-attached"}
    attached_metadata = next(tile for tile in manifest["tiles"] if tile["submap_id"] == "submap-attached")
    assert attached_metadata["revision_id"] == attached["revision_id"]
    assert attached_metadata["placement_revision_id"] != attached["revision_id"]
    assert attached_metadata["attachment_state"] == "ATTACHED"
    assert attached_metadata["transform_building_from_submap"] == list(attached_transform)
    assert root["revision_id"] in {tile["revision_id"] for tile in manifest["tiles"]}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("positions", np.asarray([[np.nan, 0, 0], [0.005, 0, 0], [0, 0.005, 0]]), "finite"),
        ("normals", np.asarray([[0, 0, np.inf]] * 3), "finite"),
        ("triangles", np.asarray([[-1, 1, 2]], dtype=np.int64), "index"),
        ("triangles", np.asarray([[0, 1, 3]], dtype=np.int64), "index"),
        ("triangles", np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64), "integer"),
        ("triangle_scale_m", np.asarray([[0.005]], dtype="<f4"), "shape"),
        ("triangle_provenance", np.asarray([[1]], dtype="u1"), "shape"),
        ("triangle_provenance", np.asarray([257], dtype=np.int64), "provenance"),
        ("triangle_provenance", np.asarray([1.0], dtype=np.float64), "integer"),
        ("triangle_uncertainty_m", np.asarray([[0.005]], dtype="<f4"), "shape"),
        ("triangle_uncertainty_m", np.asarray([np.nan], dtype="<f4"), "uncertainty"),
        ("triangle_uncertainty_m", np.asarray([-0.01], dtype="<f4"), "uncertainty"),
        ("triangle_uncertainty_m", np.asarray([1.01], dtype="<f4"), "uncertainty"),
        ("unresolved_surfel_count", -1, "unresolved"),
        ("unresolved_surfel_count", 1.0, "unresolved"),
        ("unresolved_surfel_count", True, "unresolved"),
    ),
)
def test_direct_publication_rejects_malformed_revision_before_filesystem_mutation(
    tmp_path, field, value, message
):
    store = ImmutableTileStore(tmp_path)
    current = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )
    store.index_path.write_bytes(
        (json.dumps(json.loads(store.index_path.read_text()), indent=2) + "\n").encode()
    )
    index_exists = store.index_path.exists()
    index_bytes = store.index_path.read_bytes()
    revision_directories = set(store.revisions_root.iterdir())

    with pytest.raises((TypeError, ValueError), match=message):
        store.publish(
            "building-a",
            "floor-01",
            "component-root",
            "ROOT",
            "submap-01",
            replace(_revision(), **{field: value}),
        )

    assert store.index_path.exists() is index_exists
    assert store.index_path.read_bytes() == index_bytes
    assert set(store.revisions_root.iterdir()) == revision_directories
    assert store.manifest()["tiles"][0]["revision_id"] == current["revision_id"]


def test_malformed_direct_publication_preserves_prior_index_nonexistence(tmp_path):
    store = ImmutableTileStore(tmp_path)
    before_revisions = set(store.revisions_root.iterdir())

    with pytest.raises(ValueError, match="finite"):
        store.publish(
            "building-a",
            "floor-01",
            "component-root",
            "ROOT",
            "submap-01",
            replace(_revision(), positions=np.full((3, 3), np.nan, dtype="<f4")),
        )

    assert not store.index_path.exists()
    assert set(store.revisions_root.iterdir()) == before_revisions
    assert not (tmp_path / ".current-tiles.lock").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("vertex_count", 4),
        ("triangle_count", 2),
        ("unresolved_surfel_count", 99),
        ("authority", "VALIDATED_SURFACE"),
        ("quality", "FINAL"),
        ("geometry_authority", True),
        ("binary_url", "/wrong"),
    ),
)
def test_load_rejects_tampered_counts_and_immutable_metadata(tmp_path, field, value):
    store = ImmutableTileStore(tmp_path)
    metadata = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )
    assert metadata["schema_version"] == 5
    metadata_path = store.revisions_root / metadata["revision_id"] / "metadata.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered[field] = value
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="identity|count|authority|quality|binary|metadata"):
        store.load(metadata["revision_id"])


def test_v5_identity_protects_authoritative_unresolved_surfel_count(tmp_path):
    store = ImmutableTileStore(tmp_path)
    first = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )
    second = store.publish(
        "building-a",
        "floor-01",
        "component-root",
        "ROOT",
        "submap-01",
        replace(_revision(), unresolved_surfel_count=7),
    )

    assert first["revision_id"] != second["revision_id"]
    assert store.load(second["revision_id"])[0]["unresolved_surfel_count"] == 7


def _request(tile_x: int) -> TilePublicationRequest:
    revision = replace(
        _revision(),
        tile_id=TileId(tile_x, 0, 0),
        revision_id=hashlib.sha256(f"tile-{tile_x}".encode()).hexdigest(),
    )
    return TilePublicationRequest(
        building_id="building-a",
        floor_id="floor-01",
        component_id="component-root",
        attachment_state="ROOT",
        submap_id="submap-01",
        revision=revision,
        source_generation_id="a" * 64,
        pose_revision_id="b" * 64,
    )


def _publish_request_in_subprocess(root, tile_x, start, results):
    try:
        store = ImmutableTileStore(root)
        start.wait(timeout=10)

        def hold_after_load(phase, _entry_index):
            if phase == "after_index_loaded":
                time.sleep(0.25)

        metadata = store.publish_batch(
            (_request(tile_x),), fault_injector=hold_after_load
        ).metadata[0]
        results.put(("ok", metadata["revision_id"]))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


@pytest.mark.parametrize("failure_phase", ["before_index_write", "after_index_write"])
def test_batch_publication_durably_restores_exact_prior_index_on_final_failure(
    tmp_path, failure_phase
):
    store = ImmutableTileStore(tmp_path)
    store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )
    prior_bytes = (json.dumps(json.loads(store.index_path.read_text()), indent=2) + "\n").encode()
    store.index_path.write_bytes(prior_bytes)
    before_revisions = set(store.revisions_root.iterdir())

    def fail(phase, _entry_index):
        if phase == failure_phase:
            raise OSError(f"simulated {failure_phase}")

    with pytest.raises(OSError, match=failure_phase):
        store.publish_batch((_request(-1), _request(1)), fault_injector=fail)

    assert store.index_path.read_bytes() == prior_bytes
    assert set(store.revisions_root.iterdir()) > before_revisions
    assert ImmutableTileStore(tmp_path).index_path.read_bytes() == prior_bytes


def test_batch_failure_restores_prior_index_nonexistence_and_keeps_staged_revisions(tmp_path):
    store = ImmutableTileStore(tmp_path)

    def fail(phase, _entry_index):
        if phase == "after_index_write":
            raise OSError("simulated post-replace failure")

    with pytest.raises(OSError, match="post-replace"):
        store.publish_batch((_request(-1), _request(1)), fault_injector=fail)

    assert not store.index_path.exists()
    assert len(tuple(store.revisions_root.iterdir())) == 2
    assert ImmutableTileStore(tmp_path).manifest()["tiles"] == []


def test_batch_staging_failure_preserves_exact_index_and_first_staged_revision(tmp_path):
    store = ImmutableTileStore(tmp_path)
    store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )
    prior_bytes = store.index_path.read_bytes()
    before_revisions = set(store.revisions_root.iterdir())

    def fail(phase, entry_index):
        if phase == "before_revision_stage" and entry_index == 1:
            raise OSError("simulated second revision staging failure")

    with pytest.raises(OSError, match="second revision"):
        store.publish_batch((_request(-1), _request(1)), fault_injector=fail)

    assert store.index_path.read_bytes() == prior_bytes
    assert len(set(store.revisions_root.iterdir()) - before_revisions) == 1


def test_batch_rollback_failure_preserves_original_exception_and_adds_detail(tmp_path):
    store = ImmutableTileStore(tmp_path)

    def fail(phase, _entry_index):
        if phase == "after_index_write":
            raise OSError("original transaction failure")
        if phase == "before_index_restore":
            raise OSError("rollback storage failure")

    with pytest.raises(OSError, match="original transaction failure") as raised:
        store.publish_batch((_request(1),), fault_injector=fail)

    assert any("rollback storage failure" in note for note in raised.value.__notes__)


def test_index_directory_fsync_eio_after_replace_rolls_back_for_restart(
    monkeypatch, tmp_path
):
    store = ImmutableTileStore(tmp_path)
    original_request = _request(1)
    alternate_request = replace(
        original_request,
        revision=replace(
            original_request.revision,
            revision_id=hashlib.sha256(b"alternate-tile-1").hexdigest(),
        ),
    )
    original = store.publish_batch((original_request,)).metadata[0]
    alternate = store.publish_batch((alternate_request,)).metadata[0]
    assert original["revision_id"] != alternate["revision_id"]
    prior_bytes = store.index_path.read_bytes()
    real_fsync_directory = live_tiles._fsync_directory
    calls = 0

    def fail_first_directory_fsync(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, "simulated post-replace storage failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(live_tiles, "_fsync_directory", fail_first_directory_fsync)
    with pytest.raises(OSError) as raised:
        store.publish_batch((original_request,))

    assert raised.value.errno == errno.EIO
    assert calls == 2
    assert store.index_path.read_bytes() == prior_bytes
    restarted = ImmutableTileStore(tmp_path)
    assert restarted.manifest()["tiles"][0]["revision_id"] == alternate["revision_id"]


def test_successful_batch_advances_one_complete_index(tmp_path):
    store = ImmutableTileStore(tmp_path)
    phases = []

    result = store.publish_batch(
        (_request(-1), _request(1)),
        fault_injector=lambda phase, entry_index: phases.append((phase, entry_index)),
    )

    assert len(result.metadata) == 2
    assert phases.count(("before_index_write", None)) == 1
    assert phases.count(("after_index_write", None)) == 1
    current = json.loads(store.index_path.read_text())["current"]
    assert {item["revision_id"] for item in result.metadata}.issubset(current.values())


def test_two_store_instances_concurrently_publish_distinct_tiles_without_lost_updates(tmp_path):
    first_store = ImmutableTileStore(tmp_path)
    second_store = ImmutableTileStore(tmp_path)
    start = threading.Barrier(3)
    errors = []

    def publish(store, request):
        try:
            start.wait(timeout=3)
            store.publish_batch((request,))
        except BaseException as exc:
            errors.append(exc)

    threads = (
        threading.Thread(target=publish, args=(first_store, _request(-1))),
        threading.Thread(target=publish, args=(second_store, _request(1))),
    )
    for thread in threads:
        thread.start()
    start.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert {tuple(tile["tile_id"]) for tile in first_store.manifest()["tiles"]} == {
        (-1, 0, 0),
        (1, 0, 0),
    }


def test_directory_fsync_propagates_real_storage_errors_and_ignores_unsupported(
    monkeypatch, tmp_path
):
    descriptor = os.open(tmp_path / "directory-fsync-probe", os.O_RDWR | os.O_CREAT)

    def fail_with(error_number):
        def failing_fsync(_descriptor):
            raise OSError(error_number, "simulated directory fsync failure")

        return failing_fsync

    monkeypatch.setattr(live_tiles.os, "open", lambda _path, _flags: os.dup(descriptor))
    monkeypatch.setattr(live_tiles.os, "fsync", fail_with(errno.EIO))
    try:
        with pytest.raises(OSError) as raised:
            live_tiles._fsync_directory(tmp_path)
        assert raised.value.errno == errno.EIO

        monkeypatch.setattr(live_tiles.os, "fsync", fail_with(errno.EINVAL))
        live_tiles._fsync_directory(tmp_path)
    finally:
        os.close(descriptor)


def test_directory_fsync_and_publication_complete_on_current_platform(tmp_path):
    live_tiles._fsync_directory(tmp_path)
    store = ImmutableTileStore(tmp_path / "tiles")

    metadata = store.publish(
        "building-a", "floor-01", "component-root", "ROOT", "submap-01", _revision()
    )

    assert store.load(metadata["revision_id"])[0] == metadata


def test_store_fails_closed_when_os_lock_acquisition_is_unsupported(monkeypatch, tmp_path):
    store = ImmutableTileStore(tmp_path)

    def unsupported_lock(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "simulated unsupported OS lock")

    if os.name == "nt":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", unsupported_lock)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", unsupported_lock)

    with pytest.raises(OSError, match="unsupported OS lock"):
        store.publish_batch((_request(1),))

    assert not store.index_path.exists()
    assert tuple(store.revisions_root.iterdir()) == ()


def _make_close_failing_fdopen(real_fdopen):
    class CloseFailingFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def close(self):
            self._wrapped.close()
            raise OSError(errno.EIO, "simulated lock descriptor close failure")

    def open_failing_file(*args, **kwargs):
        return CloseFailingFile(real_fdopen(*args, **kwargs))

    return open_failing_file


def test_lock_release_failure_after_commit_is_diagnostic_not_publication_failure(
    monkeypatch, tmp_path
):
    store = ImmutableTileStore(tmp_path)
    real_fdopen = live_tiles.os.fdopen
    monkeypatch.setattr(
        live_tiles.os, "fdopen", _make_close_failing_fdopen(real_fdopen)
    )

    result = store.publish_batch((_request(1),))

    assert len(result.metadata) == 1
    assert store.manifest()["tiles"][0]["revision_id"] == result.metadata[0]["revision_id"]
    assert "close failure" in store.last_lock_release_error


def test_lock_release_failure_adds_note_to_active_transaction_error(monkeypatch, tmp_path):
    store = ImmutableTileStore(tmp_path)
    real_fdopen = live_tiles.os.fdopen
    monkeypatch.setattr(
        live_tiles.os, "fdopen", _make_close_failing_fdopen(real_fdopen)
    )

    def fail_transaction(phase, _entry_index):
        if phase == "before_index_write":
            raise RuntimeError("simulated transaction failure")

    with pytest.raises(RuntimeError, match="transaction failure") as raised:
        store.publish_batch((_request(1),), fault_injector=fail_transaction)

    assert any("close failure" in note for note in raised.value.__notes__)
    assert store.last_lock_release_error is None


def test_two_processes_contending_on_store_lock_preserve_distinct_current_updates(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_publish_request_in_subprocess,
            args=(str(tmp_path), tile_x, start, results),
        )
        for tile_x in (-1, 1)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
    messages = []
    try:
        for _ in processes:
            messages.append(results.get(timeout=3))
    except Empty:
        pass
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        results.close()
        results.join_thread()

    assert all(process.exitcode == 0 for process in processes)
    assert len(messages) == 2
    assert {status for status, _detail in messages} == {"ok"}
    assert {
        tuple(tile["tile_id"])
        for tile in ImmutableTileStore(tmp_path).manifest()["tiles"]
    } == {(-1, 0, 0), (1, 0, 0)}
