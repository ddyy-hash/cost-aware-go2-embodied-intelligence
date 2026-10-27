"""Regression coverage for the loopback-only derived-surface viewer."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_reconstruction_config_import_does_not_eagerly_load_calibration_stack():
    """A read-only surface validator must not require OpenCV calibration code."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mapping.pc.go2_mapping_quality.reconstruction_config; "
                "assert 'mapping.pc.go2_mapping_calibration.target' not in sys.modules"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_readonly_preview_serves_only_the_verified_surface(monkeypatch):
    module = importlib.import_module("mapping.pc.go2_mapping_dashboard.readonly_preview")
    surface_path = (
        PROJECT_ROOT
        / "mapping"
        / "tests"
        / "fixtures"
        / "appearance"
        / "browser"
        / "readonly-preview-surface.json"
    )
    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    preview_status = {
        "schema_version": 1,
        "available": True,
        "blocker": None,
        "source_mode": "DERIVED_READ_ONLY",
        "read_only": True,
        "quality": "PREVIEW_UNCALIBRATED",
        "geometry_authority": False,
        "sensor_frame_transform_id": "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2",
        "derived_cache_key": "a" * 64,
        "surface_sha256": "b" * 64,
        "ply_sha256": "c" * 64,
        "metadata_sha256": "d" * 64,
        "vertex_count": 3,
        "triangle_count": 1,
    }
    pointer_status = {
        "schema_version": 1,
        "session_id": "segment-20260812T083241Z",
        "state": "FINAL_SURFACE_AVAILABLE",
        "available": True,
        "blocker": None,
        "geometry_authority": False,
        "point_authority": "LIDAR_MEASURED",
        "hidden_surface_policy": "OBSERVED_POINTS_ONLY",
    }
    preview = SimpleNamespace(surface_path=surface_path, status=preview_status)
    calls = []

    def load_verified_pointer(path, *, output_root):
        calls.append((Path(path), Path(output_root)))
        return preview, pointer_status

    monkeypatch.setattr(module, "load_automatic_reconstruction_preview", load_verified_pointer)
    app = module.create_app(status_path=surface_path, output_root=surface_path.parent)

    with TestClient(app) as client:
        status_response = client.get("/api/reconstruction/measured-surface/status")
        surface_response = client.get("/api/reconstruction/measured-surface")
        diagnostics_response = client.get("/api/diagnostics")
        blocked_response = client.post("/api/reconstruction/publish-measured")
        manifest_response = client.get("/api/building/surface/manifest")
        with client.websocket_connect("/ws/live") as socket:
            websocket_snapshot = socket.receive_json()

    assert status_response.status_code == 200
    assert status_response.json() == {
        **preview_status,
        "automatic_reconstruction_status": pointer_status,
    }
    assert surface_response.status_code == 200
    assert surface_response.json() == surface_payload
    assert diagnostics_response.status_code == 200
    assert diagnostics_response.json()["dense_map"]["points"] == 3
    assert diagnostics_response.json()["building_scan"]["triangle_count"] == 1
    assert blocked_response.status_code == 405
    assert manifest_response.status_code == 200
    assert manifest_response.json()["tiles"] == []
    assert websocket_snapshot["message_type"] == "snapshot"
    assert websocket_snapshot["snapshot"]["source"] == "READ_ONLY_DERIVED_SURFACE_PREVIEW"
    assert calls
