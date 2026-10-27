import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mapping.pc.go2_mapping_dashboard import server


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d"))


def _published(tmp_path):
    manifest = {
        "schema_version": 1,
        "building_id": "building-a",
        "assembly_id": HASHES[0],
        "building_appearance_revision_id": HASHES[1],
        "geometry_authority": "OBSERVED_SURFACE",
        "volume_filled": False,
        "watertight": False,
        "members": [
            {
                "source_generation_id": HASHES[2],
                "tile_revision_id": HASHES[3],
            }
        ],
    }
    body = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    path = tmp_path / "manifest.json"
    path.write_bytes(body)
    return SimpleNamespace(
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=hashlib.sha256(body).hexdigest(),
        building_appearance_revision_id=HASHES[1],
    )


class _Pipeline:
    def __init__(self, published):
        self.published = published
        self.current_calls = []
        self.exact_calls = []

    def snapshot(self):
        return {"building_id": "building-a"}

    def current_building_appearance(
        self, building_id, *, expected_assembly_id=None, expected_source_generation_id=None
    ):
        self.current_calls.append(
            (building_id, expected_assembly_id, expected_source_generation_id)
        )
        return self.published

    def resolve_building_appearance(self, building_id, revision_id):
        self.exact_calls.append((building_id, revision_id))
        return self.published


def _activate(monkeypatch, pipeline, *, building_id="building-a"):
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)
    monkeypatch.setattr(server, "_building_scan_summary", {"building_id": building_id})


def test_current_and_exact_building_manifests_use_frozen_pipeline_revisions_and_cache_headers(
    tmp_path, monkeypatch
):
    published = _published(tmp_path)
    pipeline = _Pipeline(published)
    _activate(monkeypatch, pipeline)

    current = server.building_appearance_current(
        expected_assembly_id=HASHES[0], expected_source_generation_id=HASHES[2]
    )
    exact = server.building_appearance_revision_manifest(HASHES[1])

    assert pipeline.current_calls == [("building-a", HASHES[0], HASHES[2])]
    assert pipeline.exact_calls == [("building-a", HASHES[1])]
    assert current.media_type == exact.media_type == "application/json"
    assert current.headers["cache-control"] == "no-store"
    assert exact.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert current.headers["etag"] == exact.headers["etag"] == f'"{published.manifest_sha256}"'
    assert current.body == exact.body == published.manifest_path.read_bytes()


def test_building_manifest_routes_enforce_malformed_unknown_and_stale_statuses(
    tmp_path, monkeypatch
):
    published = _published(tmp_path)
    pipeline = _Pipeline(published)
    _activate(monkeypatch, pipeline)

    malformed = (
        lambda: server.building_appearance_current(expected_assembly_id="BAD"),
        lambda: server.building_appearance_current(expected_source_generation_id="BAD"),
        lambda: server.building_appearance_revision_manifest("BAD"),
    )
    for invoke in malformed:
        with pytest.raises(HTTPException) as error:
            invoke()
        assert error.value.status_code == 422

    def stale(*args, **kwargs):
        raise ValueError("Building appearance source generation is stale")

    monkeypatch.setattr(pipeline, "current_building_appearance", stale)
    with pytest.raises(HTTPException) as error:
        server.building_appearance_current(expected_source_generation_id=HASHES[2])
    assert error.value.status_code == 409

    def unknown(*args, **kwargs):
        raise FileNotFoundError("unknown Building appearance revision")

    monkeypatch.setattr(pipeline, "resolve_building_appearance", unknown)
    with pytest.raises(HTTPException) as error:
        server.building_appearance_revision_manifest(HASHES[3])
    assert error.value.status_code == 404


def test_building_manifest_routes_fail_closed_without_one_active_building(
    tmp_path, monkeypatch
):
    pipeline = _Pipeline(_published(tmp_path))
    monkeypatch.setattr(pipeline, "snapshot", lambda: {"building_id": None})
    _activate(monkeypatch, pipeline, building_id=None)

    for invoke in (
        server.building_appearance_current,
        lambda: server.building_appearance_revision_manifest(HASHES[1]),
    ):
        with pytest.raises(HTTPException) as error:
            invoke()
        assert error.value.status_code == 404


def test_building_manifest_response_fails_closed_when_verified_file_disappears(
    tmp_path, monkeypatch
):
    published = _published(tmp_path)
    pipeline = _Pipeline(published)
    _activate(monkeypatch, pipeline)
    published.manifest_path.unlink()

    with pytest.raises(HTTPException) as error:
        server.building_appearance_revision_manifest(HASHES[1])

    assert error.value.status_code == 404


def test_building_appearance_status_is_no_store_and_avoids_expected_404_polling(
    tmp_path, monkeypatch
):
    pipeline = _Pipeline(_published(tmp_path))
    monkeypatch.setattr(server, "_live_surface_pipeline", pipeline)

    available = server.building_appearance_status()
    assert available.status_code == 200
    assert available.headers["cache-control"] == "no-store"
    assert json.loads(available.body) == {
        "schema_version": 1,
        "available": True,
        "building_id": "building-a",
        "building_appearance_revision_id": HASHES[1],
        "blocker": None,
    }

    pipeline.published = None
    unavailable = server.building_appearance_status()
    assert unavailable.status_code == 200
    assert json.loads(unavailable.body)["blocker"] == "BUILDING_APPEARANCE_UNAVAILABLE"

    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    inactive = server.building_appearance_status()
    assert json.loads(inactive.body)["blocker"] == "BUILDING_PIPELINE_UNAVAILABLE"
