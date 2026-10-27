"""Loopback-only synthetic appearance server for browser contract tests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from mapping.pc.go2_mapping_appearance.atlas_codec import (
    encode_render_indices_v1,
    encode_render_uvs_v1,
    encode_render_vertices_v1,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


MARKER = "SYNTHETIC_TEST_ONLY"
ROOT = PROJECT_ROOT / "mapping"
STATIC_ROOT = ROOT / "pc" / "go2_mapping_dashboard" / "static"


def _require_test_mode() -> Path:
    if os.environ.get("GO2_MAPPING_SYNTHETIC_TEST_ONLY") != "1":
        raise RuntimeError("synthetic browser server requires GO2_MAPPING_SYNTHETIC_TEST_ONLY=1")
    raw = os.environ.get("GO2_MAPPING_APPEARANCE_FIXTURE_ROOT")
    if not raw:
        raise RuntimeError("synthetic browser server requires an explicit fixture root")
    fixture_root = Path(raw).resolve()
    contract_path = fixture_root / "fixture.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract != {
        "marker": MARKER,
        "purpose": "browser appearance loading and responsive rendering only",
        "geometry_authority": False,
        "acceptance_authority": False,
    }:
        raise RuntimeError("synthetic browser fixture contract is invalid")
    return fixture_root


def _quad(
    positions: list[list[float]],
    normals: list[list[float]],
    triangles: list[list[int]],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> None:
    start = len(positions)
    edge_ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    edge_ac = np.asarray(c, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    normal = np.cross(edge_ab, edge_ac)
    normal /= np.linalg.norm(normal)
    positions.extend([list(a), list(b), list(c), list(d)])
    normals.extend([normal.tolist()] * 4)
    triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])


def _box(
    positions: list[list[float]],
    normals: list[list[float]],
    triangles: list[list[int]],
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> None:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    _quad(positions, normals, triangles, (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
    _quad(positions, normals, triangles, (x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))
    _quad(positions, normals, triangles, (x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0))
    _quad(positions, normals, triangles, (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
    _quad(positions, normals, triangles, (x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
    _quad(positions, normals, triangles, (x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0))


def _fine_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []
    _quad(positions, normals, triangles, (-3.0, -2.5, 0.0), (3.0, -2.5, 0.0), (3.0, 2.5, 0.0), (-3.0, 2.5, 0.0))
    _quad(positions, normals, triangles, (-3.0, 2.5, 0.0), (3.0, 2.5, 0.0), (3.0, 2.5, 2.8), (-3.0, 2.5, 2.8))
    _box(positions, normals, triangles, (-0.75, -0.45, 0.72), (0.75, 0.45, 0.86))
    for x0, y0 in [(-0.68, -0.38), (0.54, -0.38), (-0.68, 0.24), (0.54, 0.24)]:
        _box(positions, normals, triangles, (x0, y0, 0.02), (x0 + 0.14, y0 + 0.14, 0.74))
    _box(positions, normals, triangles, (1.35, 2.38, 0.02), (2.65, 2.48, 2.35))
    _box(positions, normals, triangles, (1.92, 2.20, 1.02), (2.48, 2.40, 1.14))
    _box(positions, normals, triangles, (2.38, 2.08, 0.94), (2.50, 2.28, 1.22))
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
    )


def _coarse_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []
    _quad(positions, normals, triangles, (-3.0, -2.5, 0.0), (3.0, -2.5, 0.0), (3.0, 2.5, 0.0), (-3.0, 2.5, 0.0))
    _quad(positions, normals, triangles, (-3.0, 2.5, 0.0), (3.0, 2.5, 0.0), (3.0, 2.5, 2.8), (-3.0, 2.5, 2.8))
    _box(positions, normals, triangles, (-0.75, -0.45, 0.02), (0.75, 0.45, 0.86))
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
    )


def _atlas_png() -> bytes:
    image = Image.new("RGBA", (64, 64))
    pixels = image.load()
    colors = ((53, 125, 184, 255), (224, 178, 72, 255), (84, 154, 108, 255), (188, 78, 68, 255))
    for y in range(64):
        for x in range(64):
            pixels[x, y] = colors[((x // 8) + (y // 8)) % len(colors)]
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _build_fixture() -> dict[str, Any]:
    tile_revision = hashlib.sha256(b"synthetic-browser-detail-tile").hexdigest()
    source_generation = hashlib.sha256(b"synthetic-browser-source-generation").hexdigest()
    placement_revision = hashlib.sha256(b"synthetic-browser-placement").hexdigest()
    assembly_id = hashlib.sha256(b"synthetic-browser-assembly").hexdigest()
    depth_revision = hashlib.sha256(b"synthetic-physical-depth-contract-only").hexdigest()
    appearance_core = {
        "atlas_density_policy": {"selected_mm_per_texel": 4.0},
        "physical_depth_validation_revision": depth_revision,
        "rgbd_augmented_revision_id": None,
        "source_generation_id": source_generation,
        "synthetic_test_marker": MARKER,
        "tile_revision_id": tile_revision,
    }
    appearance_revision = canonical_sha256(appearance_core)
    atlas = _atlas_png()
    lods: list[dict[str, Any]] = []
    artifacts_by_lod: dict[str, dict[str, bytes]] = {}
    for order, (name, geometry) in enumerate((("detail", _fine_geometry()), ("coarse", _coarse_geometry()))):
        positions, normals, triangles = geometry
        vertex_count = len(positions)
        triangle_count = len(triangles)
        uvs = np.tile(np.asarray([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]], dtype=np.float32), (vertex_count // 4, 1))
        artifacts = {
            "render-vertices.bin": encode_render_vertices_v1(
                tile_revision, appearance_revision, positions, normals, np.arange(vertex_count, dtype=np.uint32)
            ),
            "render-indices.bin": encode_render_indices_v1(
                tile_revision, appearance_revision, triangles, np.arange(triangle_count, dtype=np.uint32)
            ),
            "render-uvs.bin": encode_render_uvs_v1(
                tile_revision, appearance_revision, uvs, np.zeros(vertex_count, dtype=np.uint16), np.ones(vertex_count, dtype=np.uint16)
            ),
            "atlas-0.png": atlas,
        }
        references = [
            {"name": artifact_name, "sha256": _sha(body), "bytes": len(body)}
            for artifact_name, body in sorted(artifacts.items())
        ]
        core = {
            "appearance_revision_id": appearance_revision,
            "artifacts": references,
            "atlas_count": 1,
            "policy": {"name": name, "order": order},
            "primitives": [{
                "primitive_id": 0,
                "material": "VALIDATED_UV_ATLAS",
                "atlas_index": 0,
                "index_offset": 0,
                "index_count": triangle_count * 3,
                "source_layer": "LIDAR_MEASURED",
                "source_triangle_ids": list(range(triangle_count)),
            }],
            "rgbd_augmented_revision_id": None,
            "synthetic_test_marker": MARKER,
            "tile_revision_id": tile_revision,
            "triangle_count": triangle_count,
            "vertex_count": vertex_count,
        }
        lod_id = canonical_sha256(core)
        lods.append({"lod_id": lod_id, "core": core})
        artifacts_by_lod[lod_id] = artifacts
    authorization = {
        "measured_surface_coverage_ppm": 850000,
        "minimum_independent_view_count": 3,
        "required_coverage_ppm": 800000,
        "required_independent_view_count": 2,
        "delivery_authorized": True,
        "blockers": [],
    }
    tile_manifest = {
        "schema_version": 1,
        "appearance_revision_id": appearance_revision,
        "revision_core": appearance_core,
        "delivery_authorization": authorization,
        "publication_authorization": {
            "building_authorized": True,
            "building_delivery_eligible": True,
            "export_authorized": False,
            "blockers": [MARKER],
        },
        "lods": lods,
        "synthetic_test_marker": MARKER,
    }
    member = {
        "building_id": "synthetic-browser-building",
        "component_id": "synthetic-detail-room",
        "session_id": "synthetic-browser-session",
        "submap_id": "synthetic-room-001",
        "source_generation_id": source_generation,
        "placement_revision_id": placement_revision,
        "tile_revision_id": tile_revision,
        "tile_appearance_revision_id": appearance_revision,
        "authorized_lod_ids": [lods[0]["lod_id"], lods[1]["lod_id"]],
        "distance_policy": [
            {"minimum_distance_m": 0, "maximum_distance_m": 16, "lod_id": lods[0]["lod_id"]},
            {"minimum_distance_m": 16, "maximum_distance_m": None, "lod_id": lods[1]["lod_id"]},
        ],
        "transform_building_from_submap": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "delivery_authorization": authorization,
        "rgbd_augmented_revision_id": None,
        "synthetic_test_marker": MARKER,
    }
    building_core = {
        "schema_version": 1,
        "building_id": "synthetic-browser-building",
        "assembly_id": assembly_id,
        "geometry_authority": "OBSERVED_SURFACE",
        "volume_filled": False,
        "watertight": False,
        "members": [member],
        "synthetic_test_marker": MARKER,
    }
    building_manifest = {**building_core, "building_appearance_revision_id": canonical_sha256(building_core)}
    return {
        "building_manifest": building_manifest,
        "tile_manifest": tile_manifest,
        "artifacts_by_lod": artifacts_by_lod,
        "detail_lod_id": lods[0]["lod_id"],
        "coarse_lod_id": lods[1]["lod_id"],
    }


FIXTURE_ROOT = _require_test_mode()
FIXTURE = _build_fixture()
app = FastAPI(title="GO2 Synthetic Browser Appearance Fixture")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.middleware("http")
async def _mark_every_response(request, call_next):
    response = await call_next(request)
    response.headers["X-Go2-Test-Marker"] = MARKER
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


def _manifest_response(manifest: dict[str, Any]) -> Response:
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return Response(
        body,
        media_type="application/json",
        headers={"ETag": '"' + _sha(body) + '"'},
    )


def _snapshot() -> dict[str, Any]:
    return {
        "synthetic_test_marker": MARKER,
        "timestamp_ms": 0,
        "source": "SYNTHETIC_TEST_ONLY / BROWSER RENDER CONTRACT",
        "live_map": {"points": 0, "preview_points": [], "trajectory": [], "position": [0, 0, 0], "heading_xy": [1, 0], "last_sequence": -1},
        "room_scan": {"state": "STOPPED", "frames": 0, "source_points": 0, "trajectory_length_m": 0, "extent_m": [6, 5, 2.8]},
        "building_scan": {"state": "SYNTHETIC_TEST_ONLY", "quality": "SYNTHETIC_BROWSER_FIXTURE", "active_floor_id": "F01", "active_submap_id": "synthetic-room-001", "surfel_count": 0, "triangle_count": 72, "tile_count": 1, "unresolved_surfel_count": 0},
        "dense_map": {"points": 0, "preview_points": [], "artifacts": 0, "reconstruction_phase": "IDLE", "rgbd_refinement": {"texture": {"surface_coverage": 0.85}, "accepted_depth_frames": 3, "depth_correspondence_count": 300, "depth_residual_p95_m": 0.02}},
        "depth_anything": {"state": "ASSISTANCE_ACCEPTED", "assistance_coverage": 0.84, "blockers": []},
        "transport": {"last_frame_age_ms": None, "high_fidelity": {"active": False}},
        "progressive_ingest": {"release_authorized_artifacts": 0},
        "session": {"state": "STOPPED", "capture_active": False, "session_id": "synthetic-browser-session"},
        "camera": {"state": "SYNTHETIC_TEST_ONLY", "sequence": -1},
    }


@app.get("/api/snapshot")
def snapshot():
    return _snapshot()


@app.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({**_snapshot(), "message_type": "snapshot"})
    try:
        while True:
            await asyncio.sleep(30)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return


@app.get("/api/building/surface/manifest")
def surface_manifest(scope: str = "building"):
    return {"schema_version": 2, "scope": scope, "tiles": [], "synthetic_test_marker": MARKER, "building_scan": _snapshot()["building_scan"]}


@app.get("/api/reconstruction/measured-surface")
def measured_surface():
    return Response(status_code=204)


@app.get("/api/reconstruction/measured-surface/status")
def measured_surface_status():
    return {"schema_version": 1, "available": False, "blocker": MARKER}


@app.get("/api/building/appearance/status")
def appearance_status():
    manifest = FIXTURE["building_manifest"]
    return {"available": True, "building_id": manifest["building_id"], "building_appearance_revision_id": manifest["building_appearance_revision_id"], "blocker": None, "synthetic_test_marker": MARKER}


@app.get("/api/building/appearance/manifest")
def appearance_current():
    return _manifest_response(FIXTURE["building_manifest"])


@app.get("/api/building/appearance/revisions/{revision}/manifest")
def appearance_revision(revision: str):
    manifest = FIXTURE["building_manifest"]
    if revision != manifest["building_appearance_revision_id"]:
        raise HTTPException(status_code=404, detail="unknown synthetic fixture revision")
    return _manifest_response(manifest)


@app.get("/api/building/appearance/tiles/{tile_revision}/{appearance_revision}/manifest")
def tile_manifest(tile_revision: str, appearance_revision: str):
    manifest = FIXTURE["tile_manifest"]
    if tile_revision != manifest["revision_core"]["tile_revision_id"] or appearance_revision != manifest["appearance_revision_id"]:
        raise HTTPException(status_code=404, detail="unknown synthetic fixture tile")
    return _manifest_response(manifest)


@app.get("/api/building/appearance/tiles/{tile_revision}/{appearance_revision}/{lod_id}/{artifact_name}")
def tile_artifact(tile_revision: str, appearance_revision: str, lod_id: str, artifact_name: str):
    manifest = FIXTURE["tile_manifest"]
    if tile_revision != manifest["revision_core"]["tile_revision_id"] or appearance_revision != manifest["appearance_revision_id"]:
        raise HTTPException(status_code=404, detail="unknown synthetic fixture tile")
    body = FIXTURE["artifacts_by_lod"].get(lod_id, {}).get(artifact_name)
    if body is None:
        raise HTTPException(status_code=404, detail="unknown synthetic fixture artifact")
    media_type = "image/png" if artifact_name.endswith(".png") else "application/octet-stream"
    return Response(body, media_type=media_type, headers={"X-Go2-Artifact-SHA256": _sha(body)})


@app.get("/__fixture__/meta")
def fixture_meta():
    return {"synthetic_test_marker": MARKER, "detail_lod_id": FIXTURE["detail_lod_id"], "coarse_lod_id": FIXTURE["coarse_lod_id"], "geometry_authority": False, "acceptance_authority": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("synthetic fixture server is loopback-only")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
