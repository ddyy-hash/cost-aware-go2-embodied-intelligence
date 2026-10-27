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
            ["wslpath", "-w", str(resolved)], capture_output=True, text=True, check=True
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


def _run_node(script: str) -> None:
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vertex_color_v1_parser_rejects_corrupt_stale_and_mismatched_payloads():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "bytesToHex")}
{_javascript_function(app, "parseVertexColorV1")}
const geometry = "aa".repeat(32);
const appearance = "bb".repeat(32);
const payload = new ArrayBuffer(88);
const bytes = new Uint8Array(payload);
bytes.set(new TextEncoder().encode("G2PVCOL1"), 0);
const view = new DataView(payload);
view.setUint16(8, 1, true);
view.setUint16(10, 80, true);
view.setUint32(12, 2, true);
bytes.set(Uint8Array.from({{length: 32}}, () => 0xaa), 16);
bytes.set(Uint8Array.from({{length: 32}}, () => 0xbb), 48);
bytes.set([10, 20, 30, 255, 0, 0, 0, 0], 80);
const parsed = parseVertexColorV1(payload, geometry, appearance, 2);
if (parsed.vertexCount !== 2) throw new Error("vertex count");
if (JSON.stringify([...parsed.rgb]) !== JSON.stringify([10, 20, 30, 0, 0, 0])) throw new Error("rgb");
if (JSON.stringify([...parsed.validity]) !== JSON.stringify([255, 0])) throw new Error("validity");
for (const mutation of [
  () => bytes[0] = 0,
  () => bytes[87] = 1,
]) {{
  const copy = payload.slice(0);
  mutation();
  let rejected = false;
  try {{ parseVertexColorV1(payload, geometry, appearance, 2); }} catch (_error) {{ rejected = true; }}
  if (!rejected) throw new Error("corrupt payload accepted");
  new Uint8Array(payload).set(new Uint8Array(copy));
}}
for (const args of [["cc".repeat(32), appearance, 2], [geometry, "cc".repeat(32), 2], [geometry, appearance, 3]]) {{
  let rejected = false;
  try {{ parseVertexColorV1(payload, ...args); }} catch (_error) {{ rejected = true; }}
  if (!rejected) throw new Error("identity mismatch accepted");
}}
"""
    _run_node(script)


def test_vertex_color_candidate_commits_atomically_and_rejects_stale_request():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    three_url = _javascript_module_uri(ROOT / "vendor" / "three.module.js")
    script = f"""
import * as THREE from {json.dumps(three_url)};
{_javascript_function(app, "buildVertexColorMaterial")}
{_javascript_function(app, "buildMeasuredAppearanceCandidate")}
{_javascript_function(app, "commitMeasuredAppearance")}
const geometryRevision = "aa".repeat(32);
const appearanceRevision = "bb".repeat(32);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0,0,0, 1,0,0]), 3));
const baseMaterial = new THREE.MeshStandardMaterial({{color: 0x9fd8cf}});
const mesh = new THREE.Mesh(geometry, baseMaterial);
mesh.userData.measured_surface_geometry_revision_id = geometryRevision;
const parsed = {{
  vertexCount: 2, geometryRevisionId: geometryRevision, appearanceRevisionId: appearanceRevision,
  rgb: new Uint8Array([10,20,30, 0,0,0]), validity: new Uint8Array([255,0]),
}};
const first = buildMeasuredAppearanceCandidate(mesh, parsed, baseMaterial);
const linearColor = first.geometry.getAttribute("color");
if (linearColor.normalized) throw new Error("linear color must not be normalized bytes");
const expectedLinearRed = 10 / 255 / 12.92;
if (Math.abs(linearColor.array[0] - expectedLinearRed) > 1e-7) throw new Error("sRGB was not linearized");
if (!first.geometry.getAttribute("appearanceValidity").normalized) throw new Error("validity not normalized");
if (first.material.userData.measuredAppearanceMaterial !== true) throw new Error("material identity");
if (!commitMeasuredAppearance(mesh, first, geometryRevision, "request-1", "request-1", baseMaterial)) {{
  throw new Error("valid candidate rejected");
}}
if (mesh.material !== first.material || mesh.geometry !== first.geometry) throw new Error("candidate not committed");
let oldMaterialDisposed = false;
first.material.addEventListener("dispose", () => oldMaterialDisposed = true);
const second = buildMeasuredAppearanceCandidate(mesh, parsed, baseMaterial);
if (!commitMeasuredAppearance(mesh, second, geometryRevision, "request-2", "request-2", baseMaterial)) {{
  throw new Error("replacement rejected");
}}
if (!oldMaterialDisposed) throw new Error("old appearance material not disposed");
const retainedMaterial = mesh.material;
const stale = buildMeasuredAppearanceCandidate(mesh, parsed, baseMaterial);
if (commitMeasuredAppearance(mesh, stale, geometryRevision, "request-3", "request-4", baseMaterial)) {{
  throw new Error("stale request committed");
}}
if (mesh.material !== retainedMaterial) throw new Error("stale request mutated visible material");
"""
    _run_node(script)


def test_browser_loads_content_addressed_appearance_without_coupling_points():
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "/api/reconstruction/measured-surface/appearance" in app
    assert "loadMeasuredAppearance" in app
    assert "verifyAppearanceManifest" in app
    assert "manifest.manifest_canonical_json" in app
    assert "manifest.manifest_sha256" in app
    assert "manifest_sha256" in app
    assert "manifest.vertex_color.content_sha256" in app
    assert "digest !== manifest.vertex_color.content_sha256" in app
    assert "measured_surface_geometry_revision_id" in app
    assert "appearance_revision_id" in app
    assert "TEXTURE_NOT_AUTHORIZED" in app
    assert "TEXTURE_STALE" in app
    assert "TEXTURE_FAILED" in app
    appearance = _javascript_function(app, "loadMeasuredAppearance")
    assert "pointDiagnosticsToggle" not in appearance
    assert "cloud.visible" not in appearance


def test_browser_verifies_backend_canonical_manifest_and_rejects_mutation():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    sha256_url = _javascript_module_uri(ROOT / "sha256.js")
    script = f"""
import {{ sha256Hex }} from {json.dumps(sha256_url)};
{_javascript_function(app, "verifyAppearanceManifest")}
const unsigned = {{
  appearance_revision_id: "bb".repeat(32),
  delivery_authorized: false,
  measured_surface_geometry_revision_id: "aa".repeat(32),
  blockers: ["CALIBRATION_MISSING"],
}};
const canonical = JSON.stringify(unsigned, Object.keys(unsigned).sort());
const manifest = {{
  ...unsigned,
  manifest_canonical_json: canonical,
  manifest_sha256: await sha256Hex(new TextEncoder().encode(canonical).buffer),
}};
const verified = await verifyAppearanceManifest(manifest);
if (verified.appearance_revision_id !== unsigned.appearance_revision_id) throw new Error("valid manifest rejected");
const mutated = {{ ...manifest, manifest_canonical_json: canonical.replace("CALIBRATION_MISSING", "TAMPERED") }};
let rejected = false;
try {{ await verifyAppearanceManifest(mutated); }} catch (_error) {{ rejected = true; }}
if (!rejected) throw new Error("mutated manifest accepted");
"""
    _run_node(script)
