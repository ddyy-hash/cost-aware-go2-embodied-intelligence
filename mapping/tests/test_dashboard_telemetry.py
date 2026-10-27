import os
import sys
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mapping" / "pc"))

from go2_mapping_dashboard import telemetry


def test_dashboard_is_truthful_when_no_verified_mapping_session_is_available():
    assert hasattr(telemetry, "SessionTelemetry")
    first = telemetry.SessionTelemetry().snapshot()

    assert first["source"] == "AWAITING VERIFIED LIDAR"
    assert first["mapping"]["points"] == 0
    assert first["mapping"]["coverage_m2"] == 0.0
    assert first["robot"]["position"] == [0.0, 0.0, 0.0]
    assert first["mapping"]["as_built_authority"] is False


def test_session_runner_never_marks_a_provisional_live_preview_as_as_built_authority():
    session = (ROOT / "tools" / "start_go2_mapping_session.ps1").read_text(encoding="utf-8")

    assert "as_built_authority = $false" in session


def test_full_capture_rejects_insufficient_space_before_loading_ros(tmp_path):
    if os.name != "posix":
        pytest.skip("the Go2 capture shell contract is exercised on Linux")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1G-blocks Used Available Use%% Mounted on\\n'\n"
        "printf 'fake 100G 56G 44G 56%% /capture\\n'\n",
        encoding="utf-8",
    )
    fake_df.chmod(0o755)
    capture = ROOT / "tools" / "start_go2_mapping_capture.sh"
    environment = {"PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}", "GO2_MAPPING_DATA_ROOT": str(tmp_path / "data")}

    result = subprocess.run(
        ["bash", str(capture), "1", "full"],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 5
    assert "need at least 45GiB" in result.stderr
    assert "/opt/ros/foxy/setup.bash" not in result.stderr


def test_dashboard_launcher_leaves_process_ownership_to_the_port_specific_pc_entrypoint():
    launcher = (ROOT / "mapping" / "pc" / "start_dashboard_wsl.sh").read_text(encoding="utf-8")

    assert "uvicorn mapping.pc.go2_mapping_dashboard.server:app" in launcher
    assert "pkill" not in launcher
    assert "pgrep" not in launcher


def test_dashboard_ui_labels_a_provisional_map_as_a_preview_not_an_as_built_deliverable():
    index = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "PROGRESSIVE AS-BUILT" not in index
    assert 'id="geometry-provenance"' in index
    assert ">PREVIEW_UNCALIBRATED</strong>" in index
    assert "not survey-grade" not in index


def test_unified_camera_and_map_use_live_socket_with_on_demand_diagnostics():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "async function refreshSnapshot(force = false)" in app
    assert 'cameraStream.src = cameraStream.dataset.src;' in app
    assert 'fetch("/api/diagnostics", {cache:\'no-store\'})' in app
    assert "if (!diagnosticsDrawer.open) return;" in app
    assert "snapshotRequestInFlight" in app
    assert "Date.now() - lastSocketMessageAt < 10000" in app
    assert 'new WebSocket(`${protocol}://${location.host}/ws/live?protocol=3`)' in app
    assert "setInterval(refreshSnapshot" not in app
    assert "setInterval(refreshSnapshot, 1000)" not in app
    assert 'fetch("/api/camera/status", { cache: "no-store" })' in app
    assert "setInterval(refreshCameraStatus, 1000)" in app


def test_dashboard_ui_labels_idle_uncalibrated_and_finalizing_states_explicitly():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'session.state === "FINALIZING"' in app
    assert "UPLOAD VERIFY / FINALIZING" in app
    assert "RAW RECORDING COMPLETE / UPLOAD VERIFY ACTIVE" in app
    assert "SESSION IDLE" in app
    assert "RAW RECORDING IDLE" in app
    assert 'metric.residual.textContent = unavailable(live.residual_cm) ? "UNCALIBRATED"' in app


def test_map_always_keeps_surface_dog_and_trajectory_but_never_renders_point_cloud():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "const liveSessionOwnsScene" not in app
    assert "buildingAppearanceObject.visible = surfaceViewMode === \"building\"" in app
    assert "measuredSurfaceMesh.visible = (localMeasuredSurface" in app
    assert "visibleAppearanceTiles === 0 && visibleTileMeshes === 0" in app
    assert "cloud.visible = false" in app
    assert "mapRenderEnabled" not in app
    assert "cameraRenderEnabled" not in app
    assert "scene.add(trajectory);" in app
    assert "scene.add(robot);" in app
    assert "const surfaceReady = updateSurfaceVisibility();" in app


def test_camera_panel_displays_the_independently_measured_preview_fps():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "const cameraPreviewFps = Number(cameraStatus.preview_fps);" in app
    assert 'metric.cameraState.textContent += ` / ${cameraPreviewFps.toFixed(1)} FPS`;' in app


def test_camera_stream_reconnects_after_the_dashboard_process_restarts():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'cameraStream.addEventListener("load", () => syncCameraMedia());' in app
    assert "function ensureCameraStream(status)" in app
    assert 'cameraStream.src = `${cameraStream.dataset.src}?reconnect=${Date.now()}`;' in app
    assert "const streamHasPixels = cameraStream.naturalWidth > 0" in app


def test_dashboard_ui_applies_point_free_robot_state_before_the_first_manifest_arrives():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'const previous = latestSnapshot || {};' in app
    assert 'if (payload.message_type === "robot_state") {' in app
    robot_branch = app.split('if (payload.message_type === "robot_state") {', 1)[1].split(
        '} else if (payload.message_type === "trajectory_reset") {', 1
    )[0]
    assert "apply({" in robot_branch
    assert "position: payload.robot?.position" in robot_branch
    assert "active_floor_id: payload.active_floor_id" in robot_branch
    assert "active_submap_id: payload.active_submap_id" in robot_branch
    assert "quality: payload.building_quality" in robot_branch
    assert "preview_points" not in robot_branch


def test_untextured_camera_frames_do_not_rebuild_the_full_point_cloud():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'const cameraProjectionIdentity = cameraAssist.texture_authorized ? cameraIdentity : "untextured";' in app
    assert 'camera:${cameraProjectionIdentity}' in app
    assert 'camera:${cameraIdentity}' not in app


def test_camera_frame_replacement_releases_the_previous_image_bitmap():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "const previousImage = latestCameraFrame?.image;" in app
    assert "previousImage?.close();" in app


def test_camera_assist_removes_calibration_guide_from_normal_scan_panel():
    static = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
    index = (static / "index.html").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")
    stylesheet = (static / "style.css").read_text(encoding="utf-8")

    assert 'id="camera-guide"' not in index
    assert 'id="camera-board-guide"' not in index
    assert "cameraGuideButton" not in app
    assert "cameraBoardGuide" not in app
    assert "BOARD ALIGNMENT" not in app
    assert 'id="camera-rgb-view"' in index
    assert 'id="camera-depth-view"' in index
    assert 'id="clear-measurements"' in index
    assert 'id="camera-fullscreen"' in index
    assert 'id="camera-stream"' in index
    assert 'data-src="/api/camera/stream.mjpeg"' in index
    assert ".camera-board-guide" not in stylesheet
    assert "18082" not in app
    assert 'fetch("/api/camera/frame", { cache: "no-store" })' in app
    assert "const cameraFramePayloadNeeded = !sessionOdomLocalPreview && (cameraView === \"depth\"" in app
    assert "if (cameraIdentity && cameraFramePayloadNeeded) loadCameraFrame(cameraIdentity)" in app
    assert "producer_instance_id" in app
    assert "CAMERA-RELATIVE ONLY" in app


def test_dashboard_ui_renders_an_interactive_three_dimensional_live_submap():
    index = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'type="module"' in index
    assert "three.module.js" in app
    assert "THREE.WebGLRenderer" in app
    assert "THREE.Points" in app
    assert "heading_xy" in app
    assert "N/A" in app


def test_dashboard_ui_exposes_hash_bound_measured_layer_export():
    index = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'id="export-measured"' in index
    assert "/api/reconstruction/publish-measured" in app
    assert "metadata_url" in app
    assert "MEASURED_SURFACE_PREVIEW" in app


def test_dashboard_vendors_three_module_dependencies_for_offline_browser_loading():
    vendor = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "vendor"
    module = (vendor / "three.module.js").read_text(encoding="utf-8")

    assert "three.core.js" in module
    assert (vendor / "three.core.js").is_file()


def test_dashboard_vendors_matching_gltf_loader_and_local_furniture_asset():
    import json

    static = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
    loader_root = static / "three-examples"
    core = (static / "vendor" / "three.core.js").read_text(encoding="utf-8")
    loader = (loader_root / "loaders" / "GLTFLoader.js").read_text(encoding="utf-8")
    sofa = json.loads((static / "assets" / "furniture" / "unit-sofa.gltf").read_text(encoding="utf-8"))

    assert "const REVISION = '172'" in core
    assert "class GLTFLoader" in loader
    assert (loader_root / "utils" / "BufferGeometryUtils.js").is_file()
    assert (loader_root / "utils" / "SkeletonUtils.js").is_file()
    assert (loader_root / "LICENSE").is_file()
    assert sofa["asset"]["version"] == "2.0"
    assert sofa["buffers"][0]["uri"].startswith("data:application/octet-stream;base64,")
    assert sofa["meshes"][0]["primitives"][0]["attributes"]["POSITION"] == 0

    manifest = json.loads((static / "assets" / "furniture" / "cc0" / "manifest.json").read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        path = static / "assets" / "furniture" / "cc0" / asset["file"]
        assert path.is_file()
        assert path.stat().st_size == asset["bytes"]
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        assert path.read_bytes()[:4] == b"glTF"
    assert manifest["license"] == "CC0-1.0"


def test_dashboard_cc0_pbr_manifest_matches_local_texture_assets():
    import hashlib
    import json

    static = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
    asset_root = static / "assets" / "textures" / "pbr" / "ambientcg"
    manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["license"] == "CC0-1.0"
    assert manifest["license_url"] == "https://ambientcg.com/license"
    assert {asset["id"] for asset in manifest["materials"]} == {"wood", "tile", "concrete", "carpet"}
    for asset in manifest["materials"]:
        assert len(asset["real_size_m"]) == 2
        assert all(float(value) > 0 for value in asset["real_size_m"])
        for texture in asset["maps"].values():
            path = asset_root / texture["file"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == texture["sha256"]


def test_three_canvas_fills_its_spatial_scene_container_instead_of_using_default_canvas_size():
    stylesheet = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "style.css").read_text(
        encoding="utf-8"
    )

    assert ".scene-panel {\n  position: relative;" in stylesheet
    assert "#scene { position: absolute; inset: 0;" in stylesheet


def test_dashboard_prioritizes_verified_dense_lidar_over_the_five_percent_preview_cloud():
    index = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert 'id="dense-points"' in index
    assert 'id="raw-ingress"' in index
    assert "snapshot.dense_map" in app
    assert "transport.high_fidelity" in app
    assert "DENSE_LIDAR_PREVIEW_UNCALIBRATED" in app


def test_dashboard_uses_a_stationary_pose_filter_for_the_display_marker():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "class StationaryPoseFilter" in app
    assert "stationaryRadiusM: 0.08" in app
    assert "releaseSamples: 3" in app
    assert "stationaryPoseFilter.update(position, headingXY)" in app


def test_dashboard_uses_a_compact_robot_marker_at_indoor_map_scale():
    app = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "new THREE.SphereGeometry(0.045, 12, 10)" in app
    assert "new THREE.ConeGeometry(0.05, 0.16, 12)" in app


def test_dashboard_binds_capture_and_dynamic_queues_to_structured_runtime_state():
    static = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
    index = (static / "index.html").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="session-state"' in index
    assert 'id="link-budget"' in index
    assert 'id="observed-capacity"' in index
    assert 'id="reconstructed-artifacts"' in index
    assert 'id="preview-budget"' in index
    assert 'id="high-budget"' in index
    assert "session.capture_active" in app
    assert "highFidelity.active" in app
    assert "progressive_ingest" in app
    assert "/CAPTURE RUNNING/.test(snapshot.source" not in app


def test_dashboard_sidebar_scrolls_without_covering_the_spatial_scene():
    stylesheet = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "style.css").read_text(
        encoding="utf-8"
    )

    assert ".telemetry" in stylesheet
    assert "overflow-y: auto" in stylesheet


def test_dashboard_snapshot_and_patch_publish_depth_assistance_telemetry():
    server = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "server.py").read_text(
        encoding="utf-8"
    )

    assert 'state["depth_anything"] = rgbd_refiner.assistance_snapshot()' in server
    assert '"depth_anything": rgbd_refiner.assistance_snapshot()' in server
    assert "create_depth_anything_worker_from_environment" in server
    assert "depth_anything_worker=_depth_anything_worker" in server


def test_dashboard_exposes_compact_exact_appearance_telemetry():
    static = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
    index = (static / "index.html").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "appearance-state",
        "colored-coverage",
        "textured-coverage",
        "appearance-views",
        "appearance-thresholds",
        "appearance-delivery",
        "physical-depth-acceptance",
        "assistance-coverage",
        "appearance-density",
        "appearance-lod",
        "appearance-revision",
        "building-appearance-revision",
        "appearance-blocker",
    ):
        assert f'id="{element_id}"' in index
        assert f'byId("{element_id}")' in app

    assert 'class="telemetry-group appearance-telemetry"' in index
    assert "map-only" not in index
    assert "snapshot.depth_anything" in app
    assert "renderAppearanceTelemetry" in app


def test_dashboard_appearance_telemetry_has_explicit_responsive_contracts():
    stylesheet = (
        ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "style.css"
    ).read_text(encoding="utf-8")

    assert ".appearance-state, .appearance-value" in stylesheet
    assert "overflow-wrap: anywhere" in stylesheet
    assert "@media (max-width: 24.375rem)" in stylesheet  # 390 px
    assert "@media (min-width: 48rem)" in stylesheet  # 768 px
    assert "@media (min-width: 64rem)" in stylesheet  # 1024 px
    assert "@media (min-width: 90rem)" in stylesheet  # 1440 px
    assert "grid-template-columns: minmax(24rem, 1fr) 20rem;" in stylesheet
    assert "grid-template-columns: minmax(0, 1fr) 21rem;" in stylesheet
    assert "grid-template-columns: minmax(0, 1fr) 23rem;" in stylesheet


def test_narrow_mobile_scene_status_and_tools_have_separate_rows():
    stylesheet = (
        ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static" / "style.css"
    ).read_text(encoding="utf-8")
    mobile = stylesheet.split("@media (max-width: 24.375rem)", 1)[1].split(
        "@media (min-width: 48rem)", 1
    )[0]

    assert "right: 0.5rem" in mobile.split(".scene-label", 1)[1].split("}", 1)[0]
    assert "max-width: none" in mobile.split(".scene-label", 1)[1].split("}", 1)[0]
    assert "top: 3rem" in mobile.split(".scene-tools", 1)[1].split("}", 1)[0]
    assert "top: 6rem" in mobile.split(".surface-controls", 1)[1].split("}", 1)[0]
