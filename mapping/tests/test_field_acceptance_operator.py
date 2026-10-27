from __future__ import annotations

from pathlib import Path

import pytest

from mapping.pc.go2_mapping_appearance.field_acceptance_operator import (
    build_physical_report,
    resolve_artifact_url,
    validate_capture_context,
)


SHA = tuple(char * 64 for char in "abcde")


def _context(root: Path) -> dict:
    return {
        "schema_version": 1,
        "evidence_kind": "PHYSICAL_GO2_CAPTURE",
        "transport": "TAILSCALE_DIRECT",
        "synthetic": False,
        "dashboard_url": "http://100.64.0.10:8765",
        "worktree_root": str(root),
        "owned_dashboard_pid": 42,
    }


def _evidence(identity: dict) -> dict:
    return {
        "available": True,
        "blockers": [],
        "surface": {
            "coverage_ppm": 900_000,
            "independent_color_views": 3,
            "ingest_errors": 0,
            "pose_errors": 0,
            "rgbd_errors": 0,
            "surface_errors": 0,
        },
        "physical_depth": {"correspondence_count": 200},
        "depth_anything": {"surviving_anchor_count": 300},
        "identity": identity,
    }


def _parser_record(name: str, threejs_loaded: bool = True) -> dict:
    return {
        "artifact_name": name,
        "pygltflib_version": "1.16.5",
        "trimesh_version": "4.7.1",
        "pygltflib_loaded": True,
        "trimesh_loaded": True,
        "threejs_loaded": threejs_loaded,
        "metres_verified": True,
        "coordinate_transform_verified": True,
        "provenance_path": f"{name}.provenance.json",
        "provenance_sha256": SHA[0],
    }


def test_capture_context_is_bound_to_exact_worktree(tmp_path):
    context = _context(tmp_path)
    assert validate_capture_context(context, tmp_path)["dashboard_url"].endswith(":8765")

    context["worktree_root"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="worktree_root"):
        validate_capture_context(context, tmp_path)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://other.example/artifact.glb", "relative"),
        ("/outside/artifact.glb", "artifact route"),
        ("/api/reconstruction/artifacts/../secret", "canonical"),
    ],
)
def test_artifact_download_url_must_be_same_origin_and_bounded(url: str, message: str):
    with pytest.raises(ValueError, match=message):
        resolve_artifact_url("http://100.64.0.10:8765", url)


def test_multi_tile_report_fails_local_aggregate_when_any_threejs_load_fails(tmp_path):
    identity = {
        "geometry_revision_id": SHA[0],
        "tile_appearance_revision_id": SHA[1],
        "building_appearance_revision_id": SHA[2],
        "placement_revision_id": SHA[3],
        "source_generation_id": SHA[4],
    }
    report = build_physical_report(
        context=_context(tmp_path),
        pre_evidence=_evidence(identity),
        post_evidence=_evidence(identity),
        local_exports=[_parser_record("tile-a.glb"), _parser_record("tile-b.glb", False)],
        building_export=_parser_record("building.glb"),
        restart_performed=True,
    )

    assert report["identity"]["restart_performed"] is True
    assert report["identity"]["pre_restart"] == identity
    assert report["identity"]["post_restart"] == identity
    assert len(report["exports"]["local_tiles"]) == 2
    assert report["exports"]["local"]["threejs_loaded"] is False
    assert report["exports"]["building"]["threejs_loaded"] is True
