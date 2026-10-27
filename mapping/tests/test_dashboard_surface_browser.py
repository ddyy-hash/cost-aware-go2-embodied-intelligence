from pathlib import Path
import json
import shutil
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2] / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
NODE = shutil.which("node") or shutil.which("node.exe") or "node"


def _javascript_module_uri(path: Path) -> str:
    resolved = path.resolve()
    node_target = Path(NODE).resolve() if Path(NODE).exists() else Path(NODE)
    if str(resolved).startswith("/") and str(node_target).lower().endswith(".exe"):
        converted = subprocess.run(
            ["wslpath", "-w", str(resolved)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return "file:///" + quote(converted.replace("\\", "/"), safe="/:")
    return resolved.as_uri()


def _javascript_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    async_start = start - len("async ")
    if async_start >= 0 and source[async_start:start] == "async ":
        start = async_start
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated JavaScript function: {name}")


def test_browser_uses_immutable_indexed_mesh_tiles_as_primary_geometry():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    worker = (ROOT / "surface-tile-worker.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")
    style = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "/api/building/surface/manifest" in app
    assert "new DataView" in app
    assert "new THREE.Mesh(" in app
    assert "geometry.setIndex" in app
    assert "tileMeshes" in app
    assert ".geometry.dispose()" in app
    assert "revision_id" in app
    assert "binarySchemaVersion === 1 || binarySchemaVersion === 2" in worker
    assert "triangleProvenance" in worker
    assert "triangleUncertainty" in worker
    assert "MEASURED_LIDAR" in worker
    assert "NEURAL_MULTIVIEW" in worker
    assert 'id="surface-view-building"' in index
    assert 'id="surface-view-local"' in index
    assert 'id="surface-floor"' in index
    assert 'id="point-diagnostics"' not in index
    assert "[hidden] { display: none !important; }" in style


def test_browser_defaults_to_local_surface_until_a_building_assembly_is_selected():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'let surfaceViewMode = "local";' in app
    assert 'id="surface-view-building" type="button" aria-pressed="false"' in index
    assert 'id="surface-view-local" type="button" aria-pressed="true"' in index


def test_dashboard_uses_one_canonical_workspace_for_map_and_camera():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="view-map"' not in index
    assert 'id="view-camera"' not in index
    assert '?view=map' not in index
    assert '?view=camera' not in index
    assert 'target="_blank"' not in index
    assert 'class="map-only"' not in index
    assert 'class="camera-assist camera-only"' not in index
    assert 'id="scene"' in index
    assert 'id="camera-assist" class="camera-assist"' in index
    assert 'normalizeViewMode' not in app
    assert 'mapRenderEnabled' not in app
    assert 'cameraRenderEnabled' not in app
    assert 'searchParams.delete("view")' in app
    assert 'history.replaceState(null, "", canonicalUrl);' in app
    assert 'cameraStream.src = cameraStream.dataset.src;' in app
    assert 'let liveRenderEnabled = false;' in app
    assert 'surfaceRenderEnabled' not in app
    assert 'if (liveRenderEnabled && points.length > 0' in _javascript_function(app, "apply")
    assert 'connect();' in app
    assert 'refreshSurfaceManifest();' in app.split("connect();", 1)[1]
    assert 'syncCameraMedia();' in app.split("connect();", 1)[1]
    assert '/api/building/render/manifest' in app
    assert '/ws/live?protocol=3' in app
    assert 'new Worker("/static/surface-tile-worker.js"' in app
    assert 'message_type === "tile_revision"' in app


def test_map_supports_google_style_top_down_and_oblique_projection_modes():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")
    style = (ROOT / "style.css").read_text(encoding="utf-8")

    assert 'id="map-view-top" type="button" aria-pressed="true"' in index
    assert 'id="map-view-3d" type="button" aria-pressed="false"' in index
    assert 'let mapProjectionMode = "top";' in app
    assert "new THREE.OrthographicCamera" in app
    assert "topDownPanDelta" in app
    assert "mapImageryLabel" in app
    assert '.map-projection-controls' in style
    assert "touch-action: none" in style.split("#scene", 1)[1].split("}", 1)[0]

    projection = _javascript_function(app, "mapProjectionPreset")
    pan = _javascript_function(app, "topDownPanDelta")
    imagery = _javascript_function(app, "mapImageryLabel")
    script = f"""
{projection}
{pan}
{imagery}
const top = mapProjectionPreset("top", -0.4, 0.9);
if (top.mode !== "top" || top.theta !== 0 || top.phi <= 0 || top.phi >= 0.02) {{
  throw new Error(`top preset ${{JSON.stringify(top)}}`);
}}
const oblique = mapProjectionPreset("oblique", -0.4, 0.9);
if (oblique.mode !== "oblique" || oblique.theta !== -0.4 || oblique.phi !== 0.9) {{
  throw new Error(`oblique preset ${{JSON.stringify(oblique)}}`);
}}
const delta = topDownPanDelta(100, -50, 500, 20);
if (JSON.stringify(delta) !== JSON.stringify({{x: -4, z: 2}})) {{
  throw new Error(`pan ${{JSON.stringify(delta)}}`);
}}
for (const [state, expected] of [
  ["VALIDATED", "PHOTO ATLAS"], ["PROVISIONAL", "LIVE RGB"],
  ["VERTEX_COLOR", "VERTEX COLOR"], ["UNTEXTURED", "NO IMAGERY"],
]) {{
  if (mapImageryLabel(state) !== expected) throw new Error(`imagery ${{state}}`);
}}
"""
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_building_surface_metrics_are_visible_without_embedding_tile_geometry_in_telemetry():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "building-scan-state",
        "building-floor",
        "building-submap",
        "surface-surfels",
        "surface-triangles",
        "surface-tiles",
        "surface-unresolved",
    ):
        assert f'id="{element_id}"' in index
        assert element_id.replace("-", "") in app.lower().replace("-", "")
    assert "building_scan:" in app
    assert "payload.active_floor_id" in app
    assert 'payload.message_type === "robot_state"' in app


def test_browser_renders_verified_artifact_surface_when_live_tiles_are_unavailable():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "/api/reconstruction/measured-surface" in app
    assert "buildMeasuredSurfacePreview" in app
    assert "measuredSurfaceMesh" in app
    assert 'surface.source_layer !== "LIDAR_MEASURED"' in app
    assert "surface.geometry_authority !== false" in app
    assert 'status.quality !== "PREVIEW_UNCALIBRATED"' in app
    assert "visibleTileMeshes === 0" in app
    assert "cloud.visible = false;" in app
    assert "measuredSurfaceSignature" in app
    assert "candidateSignature" in app
    assert "surfaceGroup.add(candidate.mesh)" in app
    assert "disposeMeasuredSurfacePreview()" in app
    assert 'status.source_mode === "LEGACY_SESSION_READ_ONLY"' in app
    assert "buildMeasuredSurfacePreview(surfacePayload)" in app
    assert "snapshotArtifactCount === measuredSurfaceState.artifactCount" in app
    assert "snapshotPhase === measuredSurfaceState.reconstructionPhase" in app
    assert "candidate.state.geometryRevisionId" in app
    assert "loadMeasuredAppearance(" in app
    assert "candidate.mesh, candidate.state.geometryRevisionId" in app
    assert "measuredAppearanceSurfaceActive" in app
    assert 'id="geometry-provenance"' in index
    assert 'id="geometry-quality"' in index
    assert 'id="geometry-transform"' in index
    assert 'status.source_mode === "DERIVED_READ_ONLY"' in app
    assert 'surfaceViewMode = "building";' in app
    assert "measuredSurfaceState.readOnly" in app
    assert "measuredSurfaceState?.readOnly" in _javascript_function(app, "refreshSurfaceManifest")
    assert "measuredSurfaceState?.readOnly" in _javascript_function(app, "refreshBuildingAppearance")
    assert 'measuredSurfaceState.quality || "PREVIEW_UNCALIBRATED"' in app
    assert "measuredSurfaceState.sensorFrameTransformId" in app


def test_current_surface_never_auto_falls_back_to_static_compatibility_geometry():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")

    assert "officialStatus.available === false" in refresh
    assert "disposeMeasuredSurfacePreview();" in refresh
    assert "measuredSurfaceSourceStatus = officialStatus;" in refresh
    assert "loadStaticMeasuredSurfaceCompatibility" not in refresh


def test_measured_preview_has_one_primary_provenance_label():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "MEASURED_SURFACE_PREVIEW_UNCALIBRATED" not in app
    assert "MEASURED SURFACE PREVIEW / UNCALIBRATED" not in app
    assert "Preview only, not survey-grade" not in index
    assert '"DERIVED SURFACE"' in app
    assert '"OBSERVED SURFACE / METERS"' in app
    assert '"DERIVED READ-ONLY ARTIFACT"' in app
    assert '"LEGACY SURFACE"' in app
    assert '"LEGACY RECONSTRUCTION / METERS"' in app
    assert '"LEGACY SESSION / READ-ONLY"' in app
    assert 'const displayQuality = (runtimeAuthority === "MEASURED")' in app
    assert "geometryQuality.textContent = displayQuality;" in app
    assert 'geometryTransform.textContent = measuredSurfaceState.sensorFrameTransformId;' in app


def test_mobile_provenance_stays_inside_the_map_lane_above_camera_assist():
    style = (ROOT / "style.css").read_text(encoding="utf-8")
    mobile = style.split("@media (max-width: 47.99rem)", 1)[1].split(
        "@media (max-width: 24.375rem)", 1
    )[0]

    assert ".geometry-provenance" in mobile
    provenance = mobile.split(".geometry-provenance", 1)[1].split("}", 1)[0]
    assert "top: 15rem;" in provenance
    assert "bottom: auto;" in provenance
    assert mobile.index("#scene { top: 0; bottom: auto; height: 20rem; }") < mobile.index(
        ".geometry-provenance"
    )
    assert mobile.index(".geometry-provenance") < mobile.index(".camera-assist { top: 21rem;")


def test_mobile_empty_state_stays_inside_the_map_lane_above_camera_assist():
    style = (ROOT / "style.css").read_text(encoding="utf-8")
    mobile = style.split("@media (max-width: 47.99rem)", 1)[1].split(
        "@media (max-width: 24.375rem)", 1
    )[0]

    assert ".scene-empty { top: 10rem; }" in mobile
    assert mobile.index(".scene-empty") < mobile.index(".camera-assist { top: 21rem;")


def test_surface_manifest_rejects_stale_scope_and_building_view_filters_detached_tiles():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "const requestedMode = surfaceViewMode;" in app
    assert "scope=${requestedMode}" in app
    assert "requestedMode !== surfaceViewMode" in app
    assert "manifest.scope !== requestedMode" in app
    assert 'metadata.attachment_state === "ROOT"' in app
    assert 'metadata.attachment_state === "ATTACHED"' in app


def test_surface_camera_frames_only_visible_geometry():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    frame_surface = _javascript_function(app, "frameSurface")

    assert "surfaceGroup.children" in frame_surface
    assert "if (child.visible)" in frame_surface
    assert "Math.max(1.2" in frame_surface


def test_map_has_floor_cutaway_follow_history_and_collapsed_diagnostics_contracts():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "follow-robot",
        "surface-floor",
        "trajectory-current",
        "trajectory-history",
        "trajectory-session",
        "map-texture-status",
        "diagnostics-drawer",
    ):
        assert f'id="{element_id}"' in index
    assert '<details id="diagnostics-drawer"' in index
    assert '<details id="diagnostics-drawer" open' not in index
    assert "renderer.localClippingEnabled = true" in app
    assert "currentFloorCutaway" in app
    assert "/api/building/trajectories" in app
    assert "historicalTrajectoryGroup" in app
    assert "followRobotEnabled" in app


def test_validated_atlas_masks_overlay_provisional_without_cross_space_projection():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    atlas_material = _javascript_function(app, "buildAuthorizedAtlasMaterial")
    appearance_tile = _javascript_function(app, "buildAppearanceTileObject")
    surface_tile = _javascript_function(app, "buildSurfaceTile")
    visibility = _javascript_function(app, "updateSurfaceVisibility")

    assert "sampledDiffuseColor.a < 0.5" in atlas_material
    assert "discard" in atlas_material
    assert "polygonOffset: true" in atlas_material
    assert 'primitive.material === "BASE_MATERIAL"' in appearance_tile
    assert 'primitive.source_layer === "LIDAR_MEASURED"' in appearance_tile
    assert "expandedMapPositions.set(mapPositions.subarray(" in surface_tile
    assert "expandedMapPositions.set(buildingPositions.subarray(" not in surface_tile
    assert "visibleAppearanceTiles > 0" not in visibility.split("mesh.visible =", 1)[1].split(";", 1)[0]


def test_surface_tile_worker_hashes_and_parses_v1_v2_off_the_main_thread():
    worker = (ROOT / "surface-tile-worker.js").read_text(encoding="utf-8")

    assert 'import { sha256Hex } from "./sha256.js";' in worker
    assert "binarySchemaVersion === 1 || binarySchemaVersion === 2" in worker
    assert "triangleProvenance" in worker
    assert "triangleUncertainty" in worker
    assert "postMessage" in worker
    assert "transfer" in worker


def test_legacy_ply_worker_verifies_and_parses_full_color_mesh_off_the_main_thread():
    worker_path = ROOT / "legacy-ply-worker.js"
    worker_url = _javascript_module_uri(worker_path)
    script = f"""
import {{ parseLegacyPlySurface }} from {json.dumps(worker_url)};
const header = new TextEncoder().encode(
  "ply\\n" +
  "format binary_little_endian 1.0\\n" +
  "element vertex 3\\n" +
  "property double x\\nproperty double y\\nproperty double z\\n" +
  "property double nx\\nproperty double ny\\nproperty double nz\\n" +
  "property uchar red\\nproperty uchar green\\nproperty uchar blue\\n" +
  "element face 1\\nproperty list uchar uint vertex_indices\\nend_header\\n"
);
const buffer = new ArrayBuffer(header.length + 3 * 51 + 13);
new Uint8Array(buffer).set(header);
const view = new DataView(buffer);
let offset = header.length;
const vertices = [
  [1, 2, 3, 0, 0, 1, 255, 0, 0],
  [4, 5, 6, 0, 1, 0, 0, 255, 0],
  [7, 8, 9, 1, 0, 0, 0, 0, 255],
];
for (const vertex of vertices) {{
  for (let axis = 0; axis < 6; axis += 1) {{
    view.setFloat64(offset, vertex[axis], true);
    offset += 8;
  }}
  for (let channel = 6; channel < 9; channel += 1) view.setUint8(offset++, vertex[channel]);
}}
view.setUint8(offset++, 3);
view.setUint32(offset, 0, true); offset += 4;
view.setUint32(offset, 1, true); offset += 4;
view.setUint32(offset, 2, true);
const parsed = await parseLegacyPlySurface(buffer, {{
  expectedSha256: "skip-for-unit-test",
  skipSha256: true,
  expectedVertexCount: 3,
  expectedTriangleCount: 1,
}});
if (JSON.stringify([...parsed.positions]) !== JSON.stringify([1, 3, -2, 4, 6, -5, 7, 9, -8])) {{
  throw new Error(`positions ${{[...parsed.positions]}}`);
}}
if (JSON.stringify([...parsed.normals]) !== JSON.stringify([0, 1, 0, 0, 0, -1, 1, 0, 0])) {{
  throw new Error(`normals ${{[...parsed.normals]}}`);
}}
if (JSON.stringify([...parsed.colors]) !== JSON.stringify([255, 0, 0, 0, 255, 0, 0, 0, 255])) {{
  throw new Error(`colors ${{[...parsed.colors]}}`);
}}
if (JSON.stringify([...parsed.indices]) !== JSON.stringify([0, 1, 2])) {{
  throw new Error(`indices ${{[...parsed.indices]}}`);
}}
"""

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_browser_loads_legacy_session_as_read_only_uncalibrated_surface():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'new Worker("/static/legacy-ply-worker.js"' in app
    assert 'status.source_mode === "LEGACY_SESSION_READ_ONLY"' in app
    assert "response.arrayBuffer()" in _javascript_function(app, "refreshMeasuredSurfacePreview")
    assert "buildLegacySurfacePreview" in app
    assert "LEGACY SESSION / READ-ONLY" in app
    assert "LEGACY RECONSTRUCTION / METERS" in app


def test_legacy_session_uses_building_scale_framing_and_unlit_vertex_colors():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    legacy_surface = _javascript_function(app, "buildLegacySurfacePreview")
    frame_surface = _javascript_function(app, "frameSurface")

    assert "THREE.MeshBasicMaterial" in legacy_surface
    assert "vertexColors: true" in legacy_surface
    assert "legacySurfaceFrame" in frame_surface
    assert "orbit.phi = 0.72" in frame_surface
    assert "size.length() * 0.88" in frame_surface
    assert "canvas.clientHeight > canvas.clientWidth" in frame_surface
    assert "size.length() * 1.45" in frame_surface


def test_legacy_session_hides_unavailable_live_controls_and_stops_camera_stream():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    presentation = _javascript_function(app, "setLegacySessionPresentation")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")

    assert "cameraAssist.hidden = true" in presentation
    assert 'cameraStream.removeAttribute("src")' in presentation
    assert "surfaceControls.hidden = false" in presentation
    assert "for (const control of surfaceControls.children)" in presentation
    assert "control.hidden = control !== mapTextureStatus" in presentation
    assert 'renderMapImageryState("VERTEX_COLOR")' in presentation
    assert "followRobotButton.hidden = true" in presentation
    assert refresh.index("setLegacySessionPresentation()") < refresh.index(
        'fetch("/api/reconstruction/measured-surface", { cache: "no-store" })'
    )


def test_legacy_session_mobile_canvas_reclaims_hidden_camera_space():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    style = (ROOT / "style.css").read_text(encoding="utf-8")
    presentation = _javascript_function(app, "setLegacySessionPresentation")

    assert 'document.documentElement.dataset.legacySession = "true"' in presentation
    assert 'html[data-legacy-session="true"] #scene' in style
    assert "bottom: 0;" in style.split('html[data-legacy-session="true"] #scene', 1)[1].split("}", 1)[0]
    assert "height: 100%;" in style.split('html[data-legacy-session="true"] #scene', 1)[1].split("}", 1)[0]
    legacy_provenance = style.split(
        'html[data-legacy-session="true"] .geometry-provenance', 1
    )[1].split("}", 1)[0]
    assert "top: auto;" in legacy_provenance
    assert "bottom: 3.5rem;" in legacy_provenance


def test_legacy_session_reframes_when_viewport_orientation_changes():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    resize = _javascript_function(app, "resize")
    frame_surface = _javascript_function(app, "frameSurface")

    assert "lastLegacySurfacePortrait" in resize
    assert "legacySurfacePortrait !== lastLegacySurfacePortrait" in resize
    assert "requestAnimationFrame(updateSurfaceVisibility)" in resize
    assert "lastLegacySurfacePortrait = canvas.clientHeight > canvas.clientWidth" in frame_surface


def test_browser_applies_assembly_placement_before_scene_conversion_and_caches_by_placement():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "placement_revision_id" in app
    assert "transform_building_from_submap" in app
    assert "applyAssemblyTransform" in app
    assert "THREE.Matrix4" in app
    assert ".applyMatrix4" in app
    assert "placement_revision_id || metadata.revision_id" in app
    assert "transform_building_from_submap" in app.split("positionToScene", 1)[0]
    assert 'import { sha256Hex } from "./sha256.js";' in app
    worker = (ROOT / "surface-tile-worker.js").read_text(encoding="utf-8")
    assert "await sha256Hex(buffer) !== expectedSha256" in worker
    assert "Promise.allSettled" in app
    assert "stagedTileMeshes" in app
    assert "surface tile assembly update failed" in app
    assert 'const assemblyPlaced = metadata.attachment_state === "ROOT"' in app
    assert "Boolean(metadata.assembly_id && metadata.placement_revision_id)" in app
    assert "? aligned && assemblyPlaced" in app
    assert "!assemblyPlaced && metadata.submap_id === activeSurfaceSubmap" in app


def test_assembly_transform_helper_executes():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    three_url = _javascript_module_uri(ROOT / "vendor" / "three.module.js")
    script = f"""
import * as THREE from {json.dumps(three_url)};
{_javascript_function(app, "applyAssemblyTransform")}
const transform = [
  0, -1, 0, 2,
  1, 0, 0, 3,
  0, 0, 1, 4,
  0, 0, 0, 1,
];
const metadata = {{ transform_building_from_submap: transform }};
const position = applyAssemblyTransform([1, 0, 2], metadata).toArray();
const normal = applyAssemblyTransform([1, 0, 0], metadata, true).toArray();
const local = applyAssemblyTransform([1, 2, 3], {{}}).toArray();
if (JSON.stringify(position) !== JSON.stringify([2, 4, 6])) throw new Error(`position ${{position}}`);
if (Math.abs(normal[0]) > 1e-12 || Math.abs(normal[1] - 1) > 1e-12 || Math.abs(normal[2]) > 1e-12) {{
  throw new Error(`normal ${{normal}}`);
}}
if (JSON.stringify(local) !== JSON.stringify([1, 2, 3])) throw new Error(`local ${{local}}`);
"""

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_sha256_helper_verifies_known_vectors_without_web_crypto():
    sha256_url = _javascript_module_uri(ROOT / "sha256.js")
    script = f"""
import {{ sha256Hex }} from {json.dumps(sha256_url)};
Object.defineProperty(globalThis, "crypto", {{ value: undefined, configurable: true }});
const vectors = [
  ["", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
  ["abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
  [
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
  ],
];
for (const [source, expected] of vectors) {{
  const buffer = new TextEncoder().encode(source).buffer;
  const actual = await sha256Hex(buffer);
  if (actual !== expected) throw new Error(`${{source.length}} bytes -> ${{actual}}`);
}}
"""

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_building_appearance_polling_checks_no_store_status_before_current_manifest():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    status_fetch = 'fetch("/api/building/appearance/status", { cache: "no-store" })'
    manifest_fetch = 'fetch("/api/building/appearance/manifest", { cache: "no-store" })'
    assert status_fetch in app
    assert manifest_fetch in app
    assert app.index(status_fetch) < app.index(manifest_fetch)
    assert 'if (status.available !== true)' in app


def test_measured_surface_polling_checks_no_store_status_before_locked_resource():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    status_fetch = 'fetch("/api/reconstruction/measured-surface/status", { cache: "no-store" })'
    surface_fetch = 'fetch("/api/reconstruction/measured-surface", { cache: "no-store" })'
    assert status_fetch in app
    assert surface_fetch in app
    assert app.index(status_fetch) < app.index(surface_fetch)
    assert 'if (status.available !== true)' in app


def test_browser_hashes_raw_manifest_responses_and_matches_current_to_exact():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "parseVerifiedManifestResponse" in app
    assert 'response.headers.get("etag")' in app
    assert "await response.arrayBuffer()" in app
    assert "manifest response checksum mismatch" in app
    assert "currentManifest.digest !== exactManifest.digest" in app
    assert "tile appearance manifest" in app


def test_appearance_state_machine_enforces_exact_delivery_boundaries():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    normalize = _javascript_function(app, "normalizeAppearanceTelemetry")
    script = f"""
{normalize}
const exactStates = [
  "TEXTURE_NOT_AUTHORIZED", "DEPTH_VALIDATING", "VERTEX_COLOR_READY",
  "ATLAS_BUILDING", "TEXTURED_READY", "TEXTURE_STALE", "TEXTURE_FAILED",
];
for (const state of exactStates) {{
  const input = state === "TEXTURED_READY"
    ? {{state, texturedCoveragePpm: 800000, minimumIndependentViewCount: 2,
        requiredCoveragePpm: 800000, requiredIndependentViewCount: 2,
        deliveryAuthorized: true}}
    : {{state}};
  if (normalizeAppearanceTelemetry(input).state !== state) throw new Error(`state ${{state}}`);
}}
const boundaries = [
  [799999, 1, false], [799999, 2, false],
  [800000, 1, false], [800000, 2, true],
];
for (const [coverage, views, ready] of boundaries) {{
  const telemetry = normalizeAppearanceTelemetry({{
    state: "TEXTURED_READY",
    coloredCoveragePpm: coverage,
    texturedCoveragePpm: coverage,
    minimumIndependentViewCount: views,
    requiredCoveragePpm: 800000,
    requiredIndependentViewCount: 2,
    deliveryAuthorized: ready,
  }});
  if ((telemetry.state === "TEXTURED_READY") !== ready) {{
    throw new Error(`gate ${{coverage}}/${{views}} -> ${{telemetry.state}}`);
  }}
}}
for (const spoofed of [
  {{requiredCoveragePpm: 799999, requiredIndependentViewCount: 2}},
  {{requiredCoveragePpm: 800000, requiredIndependentViewCount: 1}},
  {{requiredCoveragePpm: 800000, requiredIndependentViewCount: 2, deliveryAuthorized: false}},
]) {{
  const telemetry = normalizeAppearanceTelemetry({{
    state: "TEXTURED_READY", texturedCoveragePpm: 800000,
    minimumIndependentViewCount: 2, deliveryAuthorized: true, ...spoofed,
  }});
  if (telemetry.state === "TEXTURED_READY" || !telemetry.blocker) {{
    throw new Error("spoofed delivery threshold was accepted");
  }}
}}
if (normalizeAppearanceTelemetry({{state: "UNKNOWN"}}).state !== "TEXTURE_FAILED") {{
  throw new Error("unknown appearance state did not fail closed");
}}
"""

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_building_runtime_binds_rendered_state_to_verified_manifest_evidence():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    candidate_state = _javascript_function(app, "buildingAppearanceCandidateState")
    verifier = _javascript_function(app, "verifyBuildingAppearanceManifest")

    assert "normalizeAppearanceTelemetry" in candidate_state
    assert "delivery_authorization" in candidate_state
    assert "atlas_density_policy" in candidate_state
    assert "lodId" in candidate_state
    assert "building_appearance_revision_id" in candidate_state
    assert "required_coverage_ppm" in verifier
    assert "required_independent_view_count" in verifier
    assert "800000" in verifier
    assert "renderAppearanceTelemetry" in _javascript_function(app, "apply")

    script = f"""
{_javascript_function(app, "normalizeAppearanceTelemetry")}
{candidate_state}
const LOWER_SHA256 = /^[0-9a-f]{{64}}$/;
const latestSnapshot = {{depth_anything: {{assistance_coverage: 0.25}}}};
const h = "a".repeat(64);
function candidate(coverage, views, authorized) {{
  const authorization = {{
    measured_surface_coverage_ppm: coverage,
    minimum_independent_view_count: views,
    required_coverage_ppm: 800000,
    required_independent_view_count: 2,
    delivery_authorized: authorized,
    blockers: authorized ? [] : ["DELIVERY_GATE_FAILED"],
  }};
  const member = {{
    tile_appearance_revision_id: h,
    authorized_lod_ids: [h],
    delivery_authorization: authorization,
  }};
  const tileManifest = {{revision_core: {{
    physical_depth_validation_revision: h,
    atlas_density_policy: {{selected_mm_per_texel: 4}},
  }}}};
  return {{
    children: [{{userData: {{baseFallback: false, lodId: h, tileManifest}}}}],
    userData: {{manifest: {{building_appearance_revision_id: h, members: [member]}}}},
  }};
}}
for (const [coverage, views, authorized, ready] of [
  [799999, 1, false, false], [799999, 2, false, false],
  [800000, 1, false, false], [800000, 2, true, true],
]) {{
  const telemetry = buildingAppearanceCandidateState(candidate(coverage, views, authorized));
  if ((telemetry.state === "TEXTURED_READY") !== ready) {{
    throw new Error(`candidate gate ${{coverage}}/${{views}} -> ${{telemetry.state}}`);
  }}
}}
"""
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
