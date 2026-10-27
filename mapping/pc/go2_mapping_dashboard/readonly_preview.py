"""Serve a verified offline LiDAR surface as a loopback Dashboard preview.

This endpoint does not own capture, control, reconstruction, calibration, or
export. It displays a content-addressed result while the PC stack reloads.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .derived_surface import DerivedSurfacePreview, load_automatic_reconstruction_preview


ROOT = Path(__file__).parent
STATIC_ROOT = ROOT / "static"


def _surface_bounds(surface_path: Path) -> tuple[list[float], list[float]]:
    """Read only position extrema needed for inert Dashboard telemetry."""

    try:
        payload = json.loads(surface_path.read_text(encoding="utf-8"))
        values = payload["positions"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("verified surface positions are unavailable") from exc
    if not isinstance(values, list) or not values or len(values) % 3:
        raise ValueError("verified surface positions are invalid")
    try:
        points = [tuple(float(value) for value in values[index : index + 3]) for index in range(0, len(values), 3)]
    except (TypeError, ValueError) as exc:
        raise ValueError("verified surface positions are invalid") from exc
    if any(not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("verified surface positions are invalid")
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return minimum, maximum


def _snapshot(preview: DerivedSurfacePreview, pointer: dict[str, Any]) -> dict[str, Any]:
    minimum, maximum = _surface_bounds(preview.surface_path)
    extent = [maximum[index] - minimum[index] for index in range(3)]
    status = preview.status
    return {
        "timestamp_ms": int(time.time() * 1000),
        "source": "READ_ONLY_DERIVED_SURFACE_PREVIEW",
        "dashboard_protocol": "rgbd-producer-epoch-v2",
        "live_map": {
            "points": 0,
            "preview_points": [],
            "measured_preview_points": [],
            "trajectory": [],
            "position": [0.0, 0.0, 0.0],
            "heading_xy": [1.0, 0.0],
            "last_sequence": -1,
            "quality": status["quality"],
        },
        "room_scan": {
            "state": "OFFLINE_READ_ONLY_PREVIEW",
            "frames": 0,
            "source_points": status["vertex_count"],
            "trajectory_length_m": None,
            "extent_m": extent,
        },
        "building_scan": {
            "state": "DERIVED_SURFACE_READ_ONLY",
            "quality": status["quality"],
            "geometry_authority": False,
            "delivery_authorized": False,
            "active_floor_id": None,
            "active_submap_id": pointer["session_id"],
            "surfel_count": status["vertex_count"],
            "triangle_count": status["triangle_count"],
            "tile_count": 0,
            "unresolved_surfel_count": 0,
        },
        "dense_map": {
            "points": status["vertex_count"],
            "preview_points": [],
            "artifacts": 0,
            "reconstruction_phase": pointer["state"],
            "quality": status["quality"],
            "rgbd_refinement": {
                "state": "UNAVAILABLE_READ_ONLY",
                "texture": {"surface_coverage": None},
                "texture_authorization_blocker": "DERIVED_SURFACE_READ_ONLY",
            },
        },
        "transport": {
            "last_frame_age_ms": None,
            "high_fidelity": {"active": False, "verified_artifacts": 0},
        },
        "progressive_ingest": {"release_authorized_artifacts": 0},
        "session": {
            "state": "STOPPED",
            "capture_active": False,
            "session_id": pointer["session_id"],
        },
        "camera": {
            "state": "UNAVAILABLE_READ_ONLY",
            "sequence": -1,
            "deferred_by_bandwidth": False,
            "texture_authorized": False,
        },
        "depth_anything": {"state": "UNAVAILABLE_READ_ONLY", "blockers": ["DERIVED_SURFACE_READ_ONLY"]},
    }


def create_app(*, status_path: str | Path, output_root: str | Path) -> FastAPI:
    """Create a no-write Dashboard endpoint around an automatic worker pointer."""

    status_path = Path(status_path).resolve(strict=False)
    output_root = Path(output_root).resolve(strict=False)

    def current() -> tuple[DerivedSurfacePreview, dict[str, Any]]:
        return load_automatic_reconstruction_preview(status_path, output_root=output_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Validate before accepting a listener. No mapper or writer starts here.
        current()
        yield

    app = FastAPI(title="Go2 Read-Only Surface Preview", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_ROOT / "index.html", headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/reconstruction/measured-surface")
    def measured_surface():
        preview, _pointer = current()
        return FileResponse(
            preview.surface_path,
            media_type="application/json",
            headers={
                "Cache-Control": "no-store",
                "X-Go2-Derived-Cache-Key": preview.status["derived_cache_key"],
                "X-Content-SHA256": preview.status["surface_sha256"],
            },
        )

    @app.get("/api/reconstruction/measured-surface/status")
    def measured_surface_status():
        preview, pointer = current()
        return {**preview.status, "automatic_reconstruction_status": pointer}

    @app.get("/api/reconstruction/measured-surface/appearance")
    def measured_surface_appearance():
        return {"state": "TEXTURE_NOT_AUTHORIZED", "blockers": ["DERIVED_SURFACE_READ_ONLY"]}

    @app.get("/api/diagnostics")
    @app.get("/api/snapshot")
    def diagnostics():
        preview, pointer = current()
        return _snapshot(preview, pointer)

    @app.get("/api/camera/status")
    def camera_status():
        return {"state": "UNAVAILABLE_READ_ONLY", "sequence": -1, "deferred_by_bandwidth": False}

    @app.get("/api/building/surface/manifest")
    @app.get("/api/building/render/manifest")
    def building_surface_manifest(scope: str = "building"):
        # Return an explicit empty manifest rather than 204. The shared browser
        # bundle keeps this path on a polling loop while the read-only mesh is
        # displayed, and an empty JSON manifest keeps that loop quiet.
        preview, pointer = current()
        return {
            "schema_version": 2,
            "scope": scope,
            "building_id": None,
            "assembly_id": None,
            "scene_revision_id": None,
            "tiles": [],
            "building_scan": _snapshot(preview, pointer)["building_scan"],
        }

    @app.get("/api/building/appearance/status")
    def building_appearance_status():
        return {"available": False, "blocker": "DERIVED_SURFACE_READ_ONLY"}

    @app.post("/api/reconstruction/publish-measured")
    def publish_measured():
        raise HTTPException(status_code=405, detail="derived surface preview is read-only")

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        preview, pointer = current()
        await websocket.send_json({"message_type": "snapshot", "snapshot": _snapshot(preview, pointer)})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one verified Go2 derived surface on loopback only")
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8782)
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(
        create_app(status_path=args.status_path, output_root=args.output_root),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
