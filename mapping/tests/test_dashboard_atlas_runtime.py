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


def _run_node(script: str) -> None:
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_building_and_tile_appearance_manifests_are_content_addressed_and_closed():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    sha256_url = _javascript_module_uri(ROOT / "sha256.js")
    script = f"""
import {{ sha256Hex }} from {json.dumps(sha256_url)};
{_javascript_function(app, "canonicalJson")}
{_javascript_function(app, "chooseAuthorizedLod")}
{_javascript_function(app, "verifyBuildingAppearanceManifest")}
{_javascript_function(app, "verifyTileAppearanceManifest")}
const h = (value) => value.repeat(64);
const LOWER_SHA256 = /^[0-9a-f]{{64}}$/;
const revisionCore = {{tile_revision_id: h("b"), source_generation_id: h("d")}};
const appearanceRevision = await sha256Hex(new TextEncoder().encode(canonicalJson(revisionCore)).buffer);
const lodCore = {{
  schema_version: 1, tile_revision_id: h("b"), appearance_revision_id: appearanceRevision,
  policy: {{order: 0}}, vertex_count: 3, triangle_count: 1, atlas_count: 1,
  artifacts: [
    {{name: "atlas-0.png", sha256: h("1"), bytes: 8}},
    {{name: "render-indices.bin", sha256: h("2"), bytes: 96}},
    {{name: "render-uvs.bin", sha256: h("3"), bytes: 116}},
    {{name: "render-vertices.bin", sha256: h("4"), bytes: 164}},
  ],
  primitives: [{{primitive_id: 0, material: "VALIDATED_UV_ATLAS", atlas_index: 0, index_offset: 0, index_count: 3, source_layer: "LIDAR_MEASURED", source_triangle_ids: [0]}}],
  rgbd_augmented_revision_id: null,
}};
const lodId = await sha256Hex(new TextEncoder().encode(canonicalJson(lodCore)).buffer);
const member = {{
  building_id: "building-a", component_id: "component-a", session_id: "session-a", submap_id: "submap-a",
  source_generation_id: h("d"), placement_revision_id: h("e"), tile_revision_id: h("b"),
  tile_appearance_revision_id: appearanceRevision, authorized_lod_ids: [lodId],
  distance_policy: [{{minimum_distance_m: 0, maximum_distance_m: null, lod_id: lodId}}],
  transform_building_from_submap: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
  delivery_authorization: {{measured_surface_coverage_ppm: 800000, minimum_independent_view_count: 2, required_coverage_ppm: 800000, required_independent_view_count: 2, delivery_authorized: true, blockers: []}},
  rgbd_augmented_revision_id: null,
}};
const core = {{schema_version: 1, building_id: "building-a", assembly_id: h("a"), geometry_authority: "OBSERVED_SURFACE", volume_filled: false, watertight: false, members: [member]}};
const buildingRevision = await sha256Hex(new TextEncoder().encode(canonicalJson(core)).buffer);
const building = await verifyBuildingAppearanceManifest({{...core, building_appearance_revision_id: buildingRevision}});
if (building.members[0].tile_revision_id !== h("b")) throw new Error("building rejected");
const tileManifest = {{
  schema_version: 1, appearance_revision_id: appearanceRevision,
  revision_core: revisionCore,
  delivery_authorization: member.delivery_authorization,
  publication_authorization: {{building_authorized: true, building_delivery_eligible: true, export_authorized: true, blockers: []}},
  lods: [{{lod_id: lodId, core: lodCore}}],
}};
await verifyTileAppearanceManifest(tileManifest, member);
for (const invalid of [
  {{...building, volume_filled: true}},
  {{...building, building_appearance_revision_id: h("f")}},
  {{...building, members: [{{...member, source_generation_id: h("f")}}]}},
]) {{
  let rejected = false; try {{ await verifyBuildingAppearanceManifest(invalid); }} catch (_error) {{ rejected = true; }}
  if (!rejected) throw new Error("invalid building manifest accepted");
}}
for (const invalid of [
  {{...tileManifest, appearance_revision_id: h("f")}},
  {{...tileManifest, publication_authorization: {{...tileManifest.publication_authorization, building_authorized: false}}}},
  {{...tileManifest, lods: []}},
  {{...tileManifest, lods: [{{...tileManifest.lods[0], core: {{...lodCore, artifacts: [...lodCore.artifacts, lodCore.artifacts[0]]}}}}]}},
  {{...tileManifest, revision_core: {{...revisionCore, rgbd_augmented_revision_id: h("f")}}}},
]) {{
  let rejected = false; try {{ await verifyTileAppearanceManifest(invalid, member); }} catch (_error) {{ rejected = true; }}
  if (!rejected) throw new Error("invalid tile manifest accepted");
}}
"""
    _run_node(script)


def test_render_v1_parsers_reject_corrupt_identity_range_uv_and_base_sentinel():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "bytesToHex")}
{_javascript_function(app, "parseAtlasHeaderV1")}
{_javascript_function(app, "parseRenderVerticesV1")}
{_javascript_function(app, "parseRenderIndicesV1")}
{_javascript_function(app, "parseRenderUvsV1")}
const tile = "aa".repeat(32); const appearance = "bb".repeat(32);
function header(magic, count, recordBytes) {{
  const body = new ArrayBuffer(80 + count * recordBytes); const bytes = new Uint8Array(body); const view = new DataView(body);
  bytes.set(new TextEncoder().encode(magic), 0); view.setUint16(8, 1, true); view.setUint16(10, 80, true); view.setUint32(12, count, true);
  bytes.set(Uint8Array.from({{length: 32}}, () => 0xaa), 16); bytes.set(Uint8Array.from({{length: 32}}, () => 0xbb), 48); return body;
}}
const vertices = header("G2RVERT1", 3, 28); const vv = new DataView(vertices);
for (let i = 0; i < 3; i++) {{ const o = 80 + i * 28; vv.setFloat32(o, i === 1 ? 1 : 0, true); vv.setFloat32(o + 16, 0, true); vv.setFloat32(o + 20, 1, true); vv.setUint32(o + 24, i, true); }}
const indices = header("G2RIDX01", 1, 16); const iv = new DataView(indices); iv.setUint32(80, 0, true); iv.setUint32(84, 1, true); iv.setUint32(88, 2, true); iv.setUint32(92, 0, true);
const uvs = header("G2RUV001", 3, 12); const uv = new DataView(uvs);
for (let i = 0; i < 3; i++) {{ const o = 80 + i * 12; uv.setFloat32(o, i / 2, true); uv.setFloat32(o + 4, i / 2, true); uv.setUint16(o + 8, 0, true); uv.setUint16(o + 10, 1, true); }}
const pv = parseRenderVerticesV1(vertices, tile, appearance, 3);
const pi = parseRenderIndicesV1(indices, tile, appearance, pv.count, 1);
const pu = parseRenderUvsV1(uvs, tile, appearance, 1);
if (pv.count !== 3 || pi.count !== 1 || pu.count !== 3) throw new Error("valid buffers rejected");
const cases = [
  () => parseRenderVerticesV1(vertices, "cc".repeat(32), appearance, 3),
  () => {{ iv.setUint32(88, 3, true); return parseRenderIndicesV1(indices, tile, appearance, 3, 1); }},
  () => {{ iv.setUint32(88, 2, true); uv.setFloat32(80, 1.1, true); return parseRenderUvsV1(uvs, tile, appearance, 1); }},
  () => {{ uv.setFloat32(80, 0, true); uv.setUint16(88, 65535, true); return parseRenderUvsV1(uvs, tile, appearance, 1); }},
];
for (const invoke of cases) {{ let rejected = false; try {{ invoke(); }} catch (_error) {{ rejected = true; }} if (!rejected) throw new Error("invalid buffer accepted"); }}
"""
    _run_node(script)


def test_primitive_validation_covers_every_index_and_separates_augmented_nodes():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "validateAppearancePrimitiveSet")}
const parsed = {{vertices: {{count: 6}}, indices: {{count: 2, triangles: new Uint32Array([0,1,2,3,4,5]), sourceTriangleIds: new Uint32Array([0,1])}}, uvs: {{count: 6, atlasIndices: new Uint16Array([0,0,0,65535,65535,65535]), flags: new Uint16Array([1,1,1,0,0,0])}}}};
const primitives = [
  {{primitive_id: 0, material: "VALIDATED_UV_ATLAS", atlas_index: 0, index_offset: 0, index_count: 3, source_layer: "LIDAR_MEASURED", source_triangle_ids: [0]}},
  {{primitive_id: 1, material: "BASE_MATERIAL", atlas_index: 65535, index_offset: 3, index_count: 3, source_layer: "RGBD_AUGMENTED", source_triangle_ids: [1]}},
];
const groups = validateAppearancePrimitiveSet({{atlas_count: 1, triangle_count: 2, primitives, rgbd_augmented_revision_id: "aa".repeat(32)}}, parsed);
if (groups.measured.length !== 1 || groups.augmented.length !== 1) throw new Error("source layers not separated");
for (const changed of [
  [{{...primitives[0], index_count: 6}}, primitives[1]],
  [{{...primitives[0], atlas_index: 1}}, primitives[1]],
  [primitives[0], {{...primitives[1], atlas_index: 0}}],
]) {{ let rejected = false; try {{ validateAppearancePrimitiveSet({{atlas_count: 1, triangle_count: 2, primitives: changed, rgbd_augmented_revision_id: "aa".repeat(32)}}, parsed); }} catch (_error) {{ rejected = true; }} if (!rejected) throw new Error("invalid primitive set accepted"); }}
"""
    _run_node(script)


def test_candidate_state_never_reports_textured_ready_for_whole_tile_base_fallback():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    script = f"""
{_javascript_function(app, "buildingAppearanceCandidateState")}
const ready = {{children: [{{userData: {{baseFallback: false}}}}, {{userData: {{baseFallback: false}}}}]}};
const fallback = {{children: [{{userData: {{baseFallback: false}}}}, {{userData: {{baseFallback: true}}}}]}};
if (buildingAppearanceCandidateState(ready).state !== "TEXTURED_READY") throw new Error("authorized candidate not ready");
const blocked = buildingAppearanceCandidateState(fallback);
if (blocked.state === "TEXTURED_READY" || !blocked.details.includes("BASE FALLBACK")) throw new Error("base fallback misreported");
"""
    _run_node(script)
