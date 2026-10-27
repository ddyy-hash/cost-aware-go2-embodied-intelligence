import * as THREE from "./vendor/three.module.js";
import { sha256Hex } from "./sha256.js";

const CURRENT_PREVIEW_URL = "/static/provisional-surface-preview/current.json";
const SURFACE_STATUS_URL = "/api/reconstruction/measured-surface/status";
const SURFACE_URL = "/api/reconstruction/measured-surface";
const SHA256_RE = /^[0-9a-f]{64}$/;
const G2PVCOL_HEADER_BYTES = 80;
const neutral = [134, 153, 155];

const byId = (id) => document.getElementById(id);
const ui = {
  canvas: byId("surface-canvas"),
  empty: byId("surface-empty"),
  status: byId("surface-status"),
  verificationState: byId("verification-state"),
  title: byId("evidence-title"),
  evidenceState: byId("evidence-state"),
  session: byId("evidence-session"),
  vertices: byId("evidence-vertices"),
  colored: byId("evidence-colored"),
  coverage: byId("evidence-coverage"),
  artifacts: byId("evidence-artifacts"),
  surfaceSha: byId("surface-sha"),
  geometryRevision: byId("geometry-revision"),
  previewRevision: byId("preview-revision"),
  sidecarSha: byId("sidecar-sha"),
  projection: byId("projection-state"),
  blockers: byId("blocker-list"),
  updatedAt: byId("updated-at"),
  top: byId("top-view"),
  three: byId("three-view"),
  reset: byId("reset-view"),
  reload: byId("reload-evidence"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x071014);
scene.fog = new THREE.Fog(0x071014, 8, 38);
const camera = new THREE.PerspectiveCamera(48, 1, 0.02, 1000);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true, powerPreference: "high-performance" });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const hemisphere = new THREE.HemisphereLight(0xc9f5f3, 0x172a31, 2.2);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(3, -4, 7);
const fillLight = new THREE.DirectionalLight(0x65cfd2, 0.7);
fillLight.position.set(-4, 2, 2);
scene.add(hemisphere, keyLight, fillLight);

let mesh = null;
let geometryBounds = null;
let dragging = false;
let pointer = null;
let viewMode = "3d";
const orbit = { azimuth: -0.78, elevation: 0.76, radius: 2.2, target: new THREE.Vector3() };

function fail(message) {
  throw new Error(message);
}

function assert(condition, message) {
  if (!condition) fail(message);
}

function lowerSha(value, label) {
  assert(typeof value === "string" && SHA256_RE.test(value), `${label} must be a lowercase SHA-256`);
  return value;
}

function positiveInteger(value, label) {
  assert(Number.isInteger(value) && value > 0, `${label} must be a positive integer`);
  return value;
}

function staticPreviewUrl(value, label, fileName) {
  assert(typeof value === "string" && value.startsWith("/static/provisional-surface-preview/"), `${label} is outside the provisional preview route`);
  assert(!value.includes("..") && !value.includes("\\"), `${label} is unsafe`);
  assert(value.endsWith(fileName), `${label} has an unexpected file name`);
  return value;
}

function canonicalState(text, level) {
  ui.status.textContent = text;
  ui.status.dataset.level = level;
  ui.verificationState.textContent = level === "accepted" ? "VERIFIED PREVIEW" : level === "rejected" ? "COLOR REJECTED" : "VERIFYING";
  ui.verificationState.dataset.state = level;
  ui.evidenceState.textContent = ui.verificationState.textContent;
  ui.updatedAt.textContent = new Date().toLocaleTimeString();
}

function shortSha(value) {
  return typeof value === "string" && value.length === 64 ? `${value.slice(0, 12)}...${value.slice(-8)}` : "--";
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(3)}%` : "--";
}

async function fetchBytes(url, label) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) fail(`${label} request failed: HTTP ${response.status}`);
  return { response, buffer: await response.arrayBuffer() };
}

async function fetchJson(url, label) {
  const { buffer } = await fetchBytes(url, label);
  try {
    return JSON.parse(new TextDecoder().decode(buffer));
  } catch {
    fail(`${label} is not valid JSON`);
  }
}

function validateSurfaceStatus(status) {
  assert(status && typeof status === "object", "measured surface status is invalid");
  assert(status.available === true, "measured surface is unavailable");
  assert(status.source_mode === "DERIVED_READ_ONLY", "measured surface is not the read-only derived surface");
  assert(status.read_only === true, "measured surface is not read-only");
  assert(status.quality === "PREVIEW_UNCALIBRATED", "measured surface quality is not PREVIEW_UNCALIBRATED");
  assert(status.geometry_authority === false, "measured surface geometry authority changed");
  return {
    surfaceSha256: lowerSha(status.surface_sha256, "measured surface SHA-256"),
    vertexCount: positiveInteger(status.vertex_count, "measured surface vertex count"),
    triangleCount: positiveInteger(status.triangle_count, "measured surface triangle count"),
  };
}

function validateSurfacePayload(surface, status) {
  assert(surface && typeof surface === "object", "measured surface payload is invalid");
  assert(surface.source_layer === "LIDAR_MEASURED", "measured surface layer changed");
  assert(surface.point_authority === "LIDAR_MEASURED", "measured surface authority changed");
  assert(surface.geometry_authority === false, "measured surface geometry authority changed");
  assert(surface.quality === "PREVIEW_UNCALIBRATED", "measured surface quality changed");
  assert(surface.volume_filled === false && surface.watertight === false, "measured surface fill state changed");
  const geometryRevision = lowerSha(surface.measured_surface_geometry_revision_id, "measured surface geometry revision");
  assert(Array.isArray(surface.positions) && surface.positions.length === status.vertexCount * 3, "measured surface position count does not match status");
  assert(Array.isArray(surface.normals) && surface.normals.length === status.vertexCount * 3, "measured surface normal count does not match status");
  assert(Array.isArray(surface.triangles) && surface.triangles.length === status.triangleCount * 3, "measured surface triangle count does not match status");
  assert(Array.isArray(surface.source_vertex_ids) && surface.source_vertex_ids.length === status.vertexCount, "measured surface vertex ordering is invalid");
  for (const value of surface.positions) assert(Number.isFinite(value), "measured surface positions are not finite");
  for (const value of surface.normals) assert(Number.isFinite(value), "measured surface normals are not finite");
  for (const index of surface.triangles) assert(Number.isInteger(index) && index >= 0 && index < status.vertexCount, "measured surface triangle index is invalid");
  return { ...status, geometryRevision, surface };
}

async function loadCurrentSurface() {
  const status = validateSurfaceStatus(await fetchJson(SURFACE_STATUS_URL, "measured surface status"));
  const { response, buffer } = await fetchBytes(SURFACE_URL, "measured surface");
  const advertisedSha = response.headers.get("X-Content-SHA256");
  assert(advertisedSha === status.surfaceSha256, "measured surface response checksum header does not match status");
  assert(await sha256Hex(buffer) === status.surfaceSha256, "measured surface checksum does not match status");
  let surface;
  try {
    surface = JSON.parse(new TextDecoder().decode(buffer));
  } catch {
    fail("measured surface payload is not valid JSON");
  }
  return validateSurfacePayload(surface, status);
}

function validateCurrentPreview(current, surface) {
  assert(current && typeof current === "object", "current provisional preview manifest is invalid");
  assert(current.protocol === "go2-static-provisional-rgbd-surface-preview-v1", "current provisional preview protocol is invalid");
  assert(current.available === true, "current provisional preview is unavailable");
  assert(current.state === "PROVISIONAL_VERTEX_COLOR_UNCALIBRATED", "current provisional preview state is invalid");
  assert(current.source_mode === "DERIVED_READ_ONLY" && current.read_only === true, "current provisional preview read-only boundary is invalid");
  assert(current.delivery_authorized !== false ? false : true, "current provisional preview delivery authority is invalid");
  assert(current.measurement_authority !== false ? false : true, "current provisional preview measurement authority is invalid");
  assert(current.final_store_eligible !== false ? false : true, "current provisional preview final-store boundary is invalid");
  assert(current.runtime_projection_authorized === true, "current provisional preview runtime projection authority is invalid");
  assert(current.geometry_authority === false && current.point_authority === "LIDAR_MEASURED", "current provisional preview geometry boundary is invalid");
  assert(current.source_surface_sha256 === surface.surfaceSha256, "current provisional preview source surface does not match measured surface");
  assert(current.measured_surface_geometry_revision_id === surface.geometryRevision, "current provisional preview geometry revision does not match measured surface");
  assert(current.vertex_count === surface.vertexCount, "current provisional preview vertex_count does not match measured surface");
  assert(Number.isInteger(current.colored_vertex_count) && current.colored_vertex_count >= 0 && current.colored_vertex_count <= surface.vertexCount, "current provisional preview colored count is invalid");
  assert(current.surface_coverage === current.colored_vertex_count / surface.vertexCount, "current provisional preview coverage is invalid");
  lowerSha(current.preview_revision_id, "current provisional preview revision");
  lowerSha(current.preview_manifest_sha256, "current provisional preview manifest SHA-256");
  lowerSha(current.vertex_color_sha256, "current provisional preview sidecar SHA-256");
  assert(current.vertex_color_bytes === G2PVCOL_HEADER_BYTES + surface.vertexCount * 4, "current provisional preview sidecar length is invalid");
  assert(current.vertex_color_format === "G2PVCOL1_RGB8_VALIDITY8", "current provisional preview sidecar format is invalid");
  assert(Array.isArray(current.blockers) && current.blockers.length > 0, "current provisional preview blockers are missing");
  return {
    ...current,
    previewManifestUrl: staticPreviewUrl(current.preview_manifest_relative_url, "preview manifest URL", "/manifest.json"),
    vertexColorUrl: staticPreviewUrl(current.vertex_color_relative_url, "vertex-color URL", "/vertex-color.bin"),
  };
}

function validatePreviewManifest(manifest, current, surface) {
  assert(manifest && typeof manifest === "object", "provisional preview manifest is invalid");
  assert(manifest.state === "PROVISIONAL_VERTEX_COLOR_UNCALIBRATED", "provisional preview manifest state is invalid");
  assert(manifest.preview_revision_id === current.preview_revision_id, "provisional preview revision does not match current manifest");
  assert(manifest.source_surface_sha256 === surface.surfaceSha256, "provisional preview manifest source surface does not match measured surface");
  assert(manifest.measured_surface_geometry_revision_id === surface.geometryRevision, "provisional preview manifest geometry revision does not match measured surface");
  assert(manifest.vertex_count === surface.vertexCount, "provisional preview manifest vertex count does not match measured surface");
  assert(manifest.colored_vertex_count === current.colored_vertex_count, "provisional preview manifest colored count does not match current manifest");
  assert(manifest.delivery_authorized !== false ? false : true, "provisional preview manifest delivery authority is invalid");
  assert(manifest.measurement_authority !== false ? false : true, "provisional preview manifest measurement authority is invalid");
  assert(manifest.final_store_eligible !== false ? false : true, "provisional preview manifest final-store boundary is invalid");
  assert(manifest.runtime_projection_authorized === true, "provisional preview manifest runtime projection authority is invalid");
  assert(manifest.vertex_color?.format === "G2PVCOL1_RGB8_VALIDITY8", "provisional preview manifest sidecar format is invalid");
  assert(manifest.vertex_color?.content_sha256 === current.vertex_color_sha256, "provisional preview manifest sidecar checksum does not match current manifest");
  assert(Array.isArray(manifest.source_artifact_sha256) && manifest.source_artifact_sha256.length > 0, "provisional preview source artifacts are missing");
  return manifest;
}

function bytesToHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function parseG2Pvcol1(buffer, current, surface) {
  const bytes = new Uint8Array(buffer);
  assert(bytes.byteLength === current.vertex_color_bytes, "provisional RGB-D sidecar byte length does not match current manifest");
  assert(bytes.byteLength >= G2PVCOL_HEADER_BYTES, "provisional RGB-D sidecar is shorter than its header");
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(bytes.slice(0, 8));
  assert(magic === "G2PVCOL1", "provisional RGB-D sidecar is not G2PVCOL1");
  assert(view.getUint16(8, true) === 1 && view.getUint16(10, true) === G2PVCOL_HEADER_BYTES, "provisional RGB-D sidecar header version is invalid");
  const vertexCount = view.getUint32(12, true);
  assert(vertexCount === surface.vertexCount, "provisional RGB-D sidecar vertex count does not match measured surface");
  assert(bytesToHex(bytes.slice(16, 48)) === surface.geometryRevision, "provisional RGB-D sidecar geometry revision does not match measured surface");
  assert(bytesToHex(bytes.slice(48, 80)) === current.preview_revision_id, "provisional RGB-D sidecar preview revision does not match current manifest");
  const rgb = new Uint8Array(vertexCount * 3);
  let colored = 0;
  for (let index = 0; index < vertexCount; index += 1) {
    const source = G2PVCOL_HEADER_BYTES + index * 4;
    const valid = bytes[source + 3];
    assert(valid === 0 || valid === 255, "provisional RGB-D sidecar validity byte is invalid");
    if (valid === 0) {
      assert(bytes[source] === 0 && bytes[source + 1] === 0 && bytes[source + 2] === 0, "uncolored provisional RGB-D sidecar record is nonzero");
      continue;
    }
    rgb[index * 3] = bytes[source];
    rgb[index * 3 + 1] = bytes[source + 1];
    rgb[index * 3 + 2] = bytes[source + 2];
    colored += 1;
  }
  assert(colored === current.colored_vertex_count, "provisional RGB-D sidecar colored count does not match current manifest");
  return { rgb, colored };
}

async function loadVerifiedProvisionalColor(surface) {
  const current = validateCurrentPreview(await fetchJson(CURRENT_PREVIEW_URL, "current provisional preview"), surface);
  const { buffer: manifestBuffer } = await fetchBytes(current.previewManifestUrl, "provisional preview manifest");
  assert(await sha256Hex(manifestBuffer) === current.preview_manifest_sha256, "provisional preview manifest checksum does not match current manifest");
  let rawManifest;
  try {
    rawManifest = JSON.parse(new TextDecoder().decode(manifestBuffer));
  } catch {
    fail("provisional preview manifest is not valid JSON");
  }
  const manifest = validatePreviewManifest(rawManifest, current, surface);
  const { buffer: sidecarBuffer } = await fetchBytes(current.vertexColorUrl, "provisional RGB-D sidecar");
  assert(await sha256Hex(sidecarBuffer) === current.vertex_color_sha256, "provisional RGB-D sidecar checksum does not match current manifest");
  assert(manifest.vertex_color_payload_sha256 === await sha256Hex(new Uint8Array(sidecarBuffer, G2PVCOL_HEADER_BYTES)), "provisional RGB-D payload checksum does not match preview manifest");
  return { current, manifest, colors: parseG2Pvcol1(sidecarBuffer, current, surface) };
}

function disposeMesh() {
  if (!mesh) return;
  scene.remove(mesh);
  mesh.geometry.dispose();
  mesh.material.dispose();
  mesh = null;
}

function renderSurface(surface, colorEvidence = null) {
  disposeMesh();
  const position = new Float32Array(surface.surface.positions);
  const normal = new Float32Array(surface.surface.normals);
  const index = new Uint32Array(surface.surface.triangles);
  const colors = new Uint8Array(surface.vertexCount * 3);
  for (let vertex = 0; vertex < surface.vertexCount; vertex += 1) {
    const offset = vertex * 3;
    if (colorEvidence?.rgb) {
      colors[offset] = colorEvidence.rgb[offset] || neutral[0];
      colors[offset + 1] = colorEvidence.rgb[offset + 1] || neutral[1];
      colors[offset + 2] = colorEvidence.rgb[offset + 2] || neutral[2];
    } else {
      colors[offset] = neutral[0];
      colors[offset + 1] = neutral[1];
      colors[offset + 2] = neutral[2];
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geometry.setAttribute("color", new THREE.Uint8BufferAttribute(colors, 3, true));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.02, side: THREE.DoubleSide });
  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
  geometryBounds = geometry.boundingBox?.clone() || null;
  frameSurface();
  ui.empty.dataset.ready = "true";
}

function frameSurface() {
  if (!geometryBounds) return;
  const size = geometryBounds.getSize(new THREE.Vector3());
  geometryBounds.getCenter(orbit.target);
  orbit.radius = Math.max(size.length() * 1.15, 1.25);
  orbit.azimuth = -0.78;
  orbit.elevation = viewMode === "top" ? 1.34 : 0.76;
  updateCamera();
}

function updateCamera() {
  const horizontal = orbit.radius * Math.cos(orbit.elevation);
  camera.position.set(
    orbit.target.x + horizontal * Math.cos(orbit.azimuth),
    orbit.target.y + horizontal * Math.sin(orbit.azimuth),
    orbit.target.z + orbit.radius * Math.sin(orbit.elevation),
  );
  camera.lookAt(orbit.target);
}

function setViewMode(mode) {
  viewMode = mode;
  ui.top.setAttribute("aria-pressed", String(mode === "top"));
  ui.three.setAttribute("aria-pressed", String(mode === "3d"));
  if (geometryBounds) {
    orbit.elevation = mode === "top" ? 1.34 : 0.76;
    updateCamera();
  }
}

function resize() {
  const width = Math.max(ui.canvas.clientWidth, 1);
  const height = Math.max(ui.canvas.clientHeight, 1);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function populateSurfaceEvidence(surface) {
  ui.session.textContent = "--";
  ui.vertices.textContent = surface.vertexCount.toLocaleString();
  ui.colored.textContent = "0";
  ui.coverage.textContent = "0.000%";
  ui.artifacts.textContent = "--";
  ui.surfaceSha.textContent = shortSha(surface.surfaceSha256);
  ui.geometryRevision.textContent = shortSha(surface.geometryRevision);
  ui.previewRevision.textContent = "--";
  ui.sidecarSha.textContent = "--";
  ui.projection.textContent = "PENDING";
  ui.blockers.textContent = "RGB-D COLOR NOT ACCEPTED UNTIL ALL PROVENANCE CHECKS PASS";
}

function populateColorEvidence(preview) {
  const { current, manifest, colors } = preview;
  ui.title.textContent = "Verified Provisional Overlay";
  ui.session.textContent = current.session_id;
  ui.vertices.textContent = current.vertex_count.toLocaleString();
  ui.colored.textContent = colors.colored.toLocaleString();
  ui.coverage.textContent = percent(current.surface_coverage);
  ui.artifacts.textContent = manifest.source_artifact_sha256.length.toLocaleString();
  ui.previewRevision.textContent = shortSha(current.preview_revision_id);
  ui.sidecarSha.textContent = shortSha(current.vertex_color_sha256);
  ui.projection.textContent = "RUNTIME ONLY / AUTHORIZED";
  ui.blockers.textContent = current.blockers.join(" / ");
}

async function reloadEvidence() {
  ui.reload.disabled = true;
  canonicalState("VALIDATING CURRENT READ-ONLY LIDAR SURFACE", "pending");
  ui.title.textContent = "Current Surface";
  try {
    const surface = await loadCurrentSurface();
    renderSurface(surface);
    populateSurfaceEvidence(surface);
    try {
      const preview = await loadVerifiedProvisionalColor(surface);
      renderSurface(surface, preview.colors);
      populateColorEvidence(preview);
      canonicalState("RGB-D SIDECAR VERIFIED / PROVISIONAL COLORS APPLIED", "accepted");
    } catch (error) {
      canonicalState(`RGB-D COLOR REJECTED / NEUTRAL LIDAR ONLY: ${error.message}`, "rejected");
    }
  } catch (error) {
    disposeMesh();
    ui.empty.dataset.ready = "false";
    canonicalState(`SURFACE REJECTED: ${error.message}`, "rejected");
    ui.blockers.textContent = "MEASURED SURFACE VERIFICATION FAILED";
  } finally {
    ui.reload.disabled = false;
  }
}

ui.canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  pointer = { x: event.clientX, y: event.clientY };
  ui.canvas.setPointerCapture(event.pointerId);
});
ui.canvas.addEventListener("pointermove", (event) => {
  if (!dragging || !pointer) return;
  const deltaX = event.clientX - pointer.x;
  const deltaY = event.clientY - pointer.y;
  orbit.azimuth -= deltaX * 0.008;
  orbit.elevation = THREE.MathUtils.clamp(orbit.elevation + deltaY * 0.008, 0.1, 1.48);
  pointer = { x: event.clientX, y: event.clientY };
  updateCamera();
});
for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
  ui.canvas.addEventListener(eventName, () => { dragging = false; pointer = null; });
}
ui.canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.radius = THREE.MathUtils.clamp(orbit.radius * Math.exp(event.deltaY * 0.001), 0.25, 120);
  updateCamera();
}, { passive: false });
ui.top.addEventListener("click", () => setViewMode("top"));
ui.three.addEventListener("click", () => setViewMode("3d"));
ui.reset.addEventListener("click", frameSurface);
ui.reload.addEventListener("click", reloadEvidence);
new ResizeObserver(resize).observe(ui.canvas);

function render() {
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

resize();
render();
reloadEvidence();
