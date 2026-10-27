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
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{": depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0: return source[start:index + 1]
    raise AssertionError(f"unterminated JavaScript function: {name}")


def _run_node(script: str) -> None:
    result = subprocess.run([NODE, "--input-type=module", "--eval", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_authorized_lod_selection_uses_complete_distance_intervals_and_coarser_fallback_only():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "chooseAuthorizedLod")}
{_javascript_function(app, "authorizedLodFallbacks")}
const LOWER_SHA256 = /^[0-9a-f]{{64}}$/;
const member = {{authorized_lod_ids: ["a".repeat(64), "b".repeat(64), "c".repeat(64)], distance_policy: [
  {{minimum_distance_m: 0, maximum_distance_m: 3, lod_id: "a".repeat(64)}},
  {{minimum_distance_m: 3, maximum_distance_m: 8, lod_id: "b".repeat(64)}},
  {{minimum_distance_m: 8, maximum_distance_m: null, lod_id: "c".repeat(64)}},
]}};
for (const [distance, expected] of [[0, "a"], [2.999, "a"], [3, "b"], [7.999, "b"], [8, "c"], [999, "c"]]) {{
  if (chooseAuthorizedLod(distance, member) !== expected.repeat(64)) throw new Error(`distance ${{distance}}`);
}}
if (JSON.stringify(authorizedLodFallbacks(member, "b".repeat(64))) !== JSON.stringify(["b".repeat(64), "c".repeat(64)])) throw new Error("fallback order");
for (const invalid of [-1, NaN, Infinity]) {{ let rejected = false; try {{ chooseAuthorizedLod(invalid, member); }} catch (_error) {{ rejected = true; }} if (!rejected) throw new Error("invalid distance accepted"); }}
"""
    _run_node(script)


def test_lod_transition_requires_stable_dwell_and_resets_when_camera_crosses_back():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "updateLodDebounce")}
let state = {{currentLodId: "a", pendingLodId: null, pendingSinceMs: 0}};
let result = updateLodDebounce(state, "b", 1000, 300); state = result.state;
if (result.commit || state.pendingLodId !== "b") throw new Error("transition did not stage");
result = updateLodDebounce(state, "a", 1100, 300); state = result.state;
if (result.commit || state.pendingLodId !== null) throw new Error("cross-back did not cancel");
result = updateLodDebounce(state, "b", 1200, 300); state = result.state;
result = updateLodDebounce(state, "b", 1499, 300); state = result.state;
if (result.commit) throw new Error("transition committed early");
result = updateLodDebounce(state, "b", 1500, 300);
if (!result.commit || result.lodId !== "b") throw new Error("stable transition did not commit");
"""
    _run_node(script)


def test_stale_building_candidate_is_disposed_without_mutating_visible_scene():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    three_url = _javascript_module_uri(ROOT / "vendor" / "three.module.js")
    script = f"""
import * as THREE from {json.dumps(three_url)};
{_javascript_function(app, "disposeAppearanceObject")}
{_javascript_function(app, "commitBuildingAppearanceCandidate")}
const scene = new THREE.Group(); const previous = new THREE.Group(); scene.add(previous);
const candidate = new THREE.Group(); const geometry = new THREE.BufferGeometry(); const material = new THREE.MeshBasicMaterial(); const texture = new THREE.Texture(); material.map = texture; candidate.add(new THREE.Mesh(geometry, material));
let geometryDisposed = false; let materialDisposed = false; let textureDisposed = false;
geometry.addEventListener("dispose", () => geometryDisposed = true); material.addEventListener("dispose", () => materialDisposed = true); texture.addEventListener("dispose", () => textureDisposed = true);
const stale = commitBuildingAppearanceCandidate(scene, previous, candidate, "request-1", "request-2");
if (stale.committed || scene.children[0] !== previous || !geometryDisposed || !materialDisposed || !textureDisposed) throw new Error("stale candidate leaked or committed");
const next = new THREE.Group(); const committed = commitBuildingAppearanceCandidate(scene, previous, next, "request-3", "request-3");
if (!committed.committed || scene.children[0] !== next) throw new Error("complete candidate not swapped");
"""
    _run_node(script)


def test_runtime_uses_exact_routes_and_keeps_points_toggle_independent():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    assert "/api/building/appearance/manifest" in app
    assert "/api/building/appearance/revisions/" in app
    assert "/api/building/appearance/tiles/" in app
    assert "pointDiagnosticsToggle" not in _javascript_function(app, "loadBuildingAppearanceCandidate")
    assert "cloud.visible" not in _javascript_function(app, "loadBuildingAppearanceCandidate")
    assert "refreshBuildingAppearanceLods" in app
    assert "commitBuildingAppearanceCandidate" in app
