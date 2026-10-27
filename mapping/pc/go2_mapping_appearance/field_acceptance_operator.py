"""Operator-side collection for physical textured-mapping acceptance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping, Sequence
import urllib.parse
import urllib.request

from pygltflib import GLTF2
import pygltflib
import trimesh

from .field_acceptance import canonical_acceptance_payload, evaluate_textured_field_acceptance


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ARTIFACT_NAME = re.compile(
    r"^(?:tile-[0-9a-f]{64}-[0-9a-f]{64}-[0-9a-f]{64}"
    r"|building-[0-9a-f]{64}-[0-9a-f]{64})\.glb(?:\.provenance\.json)?$",
    re.ASCII,
)
_IDENTITY_FIELDS = (
    "geometry_revision_id",
    "tile_appearance_revision_id",
    "building_appearance_revision_id",
    "placement_revision_id",
    "source_generation_id",
)


def _canonical_root(value: str | Path) -> str:
    raw = str(value)
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        windows = PureWindowsPath(raw)
        return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
    return str(Path(raw).resolve())


def validate_capture_context(context: object, workspace: Path) -> dict[str, Any]:
    """Validate that physical context belongs to this exact worktree."""

    if not isinstance(context, dict):
        raise ValueError("capture context must be a JSON object")
    if context.get("schema_version") != 1:
        raise ValueError("capture context schema_version must be 1")
    if context.get("synthetic") is not False:
        raise ValueError("synthetic evidence cannot satisfy physical acceptance")
    if context.get("evidence_kind") != "PHYSICAL_GO2_CAPTURE":
        raise ValueError("evidence_kind must be PHYSICAL_GO2_CAPTURE")
    if context.get("transport") != "TAILSCALE_DIRECT":
        raise ValueError("transport must be TAILSCALE_DIRECT")
    if _canonical_root(str(context.get("worktree_root", ""))).casefold() != _canonical_root(workspace).casefold():
        raise ValueError("capture context worktree_root does not match this worktree")
    parsed = urllib.parse.urlsplit(str(context.get("dashboard_url", "")))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
        raise ValueError("dashboard_url must be an explicit HTTP(S) origin")
    pid = context.get("owned_dashboard_pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("owned_dashboard_pid must be a positive integer")
    return copy.deepcopy(context)


def resolve_artifact_url(dashboard_url: str, artifact_url: str) -> str:
    """Resolve only a canonical same-origin reconstruction artifact URL."""

    parsed = urllib.parse.urlsplit(artifact_url)
    if parsed.scheme or parsed.netloc:
        raise ValueError("artifact URL must be relative to the dashboard origin")
    if parsed.query or parsed.fragment:
        raise ValueError("artifact URL must be canonical without query or fragment")
    prefix = "/api/reconstruction/artifacts/"
    if not parsed.path.startswith(prefix):
        raise ValueError("artifact URL must use the reconstruction artifact route")
    name = parsed.path[len(prefix):]
    if urllib.parse.unquote(name) != name or _ARTIFACT_NAME.fullmatch(name) is None:
        raise ValueError("artifact URL is not a canonical owned artifact name")
    return urllib.parse.urljoin(dashboard_url.rstrip("/") + "/", parsed.path.lstrip("/"))


def _json_request(url: str, *, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise ValueError(f"dashboard returned HTTP {response.status} for {url}")
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"dashboard returned a non-object JSON response for {url}")
    return decoded


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"invalid expected SHA-256 for {destination.name}")
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    digest = hashlib.sha256()
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            if response.status != 200:
                raise ValueError(f"artifact download returned HTTP {response.status}")
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"downloaded artifact SHA-256 mismatch: {destination.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_glb(glb_path: Path, provenance_path: Path, response: Mapping[str, Any]) -> dict[str, Any]:
    gltf = GLTF2().load_binary(str(glb_path))
    scene = trimesh.load(str(glb_path), force="scene", process=False)
    geometries = tuple(scene.geometry.values())
    if not geometries or not any(len(geometry.vertices) > 0 for geometry in geometries):
        raise ValueError(f"trimesh loaded no geometry from {glb_path.name}")
    extras = gltf.asset.extras if gltf.asset is not None else None
    if not isinstance(extras, dict):
        raise ValueError(f"GLB asset extras are unavailable: {glb_path.name}")
    export_unit = extras.get("export_unit")
    if export_unit not in {"LOCAL_TILE", "BUILDING"}:
        raise ValueError(f"GLB export_unit is invalid: {glb_path.name}")
    provenance_sha = str(response.get("provenance_sha256", ""))
    provenance_body = provenance_path.read_bytes()
    if _SHA256.fullmatch(provenance_sha) is None or hashlib.sha256(provenance_body).hexdigest() != provenance_sha:
        raise ValueError(f"provenance SHA-256 mismatch: {provenance_path.name}")
    return {
        "artifact_name": glb_path.name,
        "export_unit": export_unit,
        "pygltflib_version": pygltflib.__version__,
        "trimesh_version": trimesh.__version__,
        "pygltflib_loaded": True,
        "trimesh_loaded": True,
        "threejs_loaded": False,
        "metres_verified": extras.get("units") == "metres",
        "coordinate_transform_verified": extras.get("coordinate_mapping") == "REP103_XYZ_TO_GLTF_XZ_NEGATIVE_Y",
        "glb_path": str(glb_path),
        "glb_sha256": str(response["glb_sha256"]),
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha,
    }


def _collect_export(
    *, dashboard_url: str, route: str, request_payload: Mapping[str, Any],
    output: Path, expected_kind: str,
) -> dict[str, Any]:
    response = _json_request(dashboard_url.rstrip("/") + route, payload=request_payload)
    if response.get("artifact_kind") != expected_kind:
        raise ValueError(f"export response kind is not {expected_kind}")
    glb_name = response.get("glb_name")
    provenance_name = response.get("provenance_name")
    if (
        not isinstance(glb_name, str) or not isinstance(provenance_name, str)
        or provenance_name != glb_name + ".provenance.json"
        or _ARTIFACT_NAME.fullmatch(glb_name) is None
        or _ARTIFACT_NAME.fullmatch(provenance_name) is None
    ):
        raise ValueError("export response returned unsafe paired artifact names")
    glb_url = resolve_artifact_url(dashboard_url, str(response.get("glb_url", "")))
    provenance_url = resolve_artifact_url(dashboard_url, str(response.get("provenance_url", "")))
    if urllib.parse.urlsplit(glb_url).path.rsplit("/", 1)[-1] != glb_name:
        raise ValueError("GLB response name and URL do not match")
    if urllib.parse.urlsplit(provenance_url).path.rsplit("/", 1)[-1] != provenance_name:
        raise ValueError("provenance response name and URL do not match")
    glb_path = output / glb_name
    provenance_path = output / provenance_name
    _download(glb_url, glb_path, str(response.get("glb_sha256", "")))
    _download(provenance_url, provenance_path, str(response.get("provenance_sha256", "")))
    record = _parse_glb(glb_path, provenance_path, response)
    record["glb_url"] = glb_url
    return record


def _identity(evidence: Mapping[str, Any]) -> dict[str, str]:
    raw = evidence.get("identity")
    if not isinstance(raw, Mapping):
        raise ValueError("field evidence identity is unavailable")
    identity = {field: raw.get(field) for field in _IDENTITY_FIELDS}
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in identity.values()):
        raise ValueError("field evidence identity is invalid")
    return identity  # type: ignore[return-value]


def _require_available(evidence: Mapping[str, Any]) -> None:
    blockers = evidence.get("blockers")
    if evidence.get("available") is not True or blockers != []:
        detail = ",".join(str(value) for value in blockers) if isinstance(blockers, list) else "UNKNOWN"
        raise ValueError(f"physical field evidence is unavailable: {detail}")


def _aggregate_local(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one local tile export is required")
    aggregate = dict(records[0])
    for field in (
        "pygltflib_loaded", "trimesh_loaded", "threejs_loaded",
        "metres_verified", "coordinate_transform_verified",
    ):
        aggregate[field] = all(record.get(field) is True for record in records)
    aggregate["artifact_count"] = len(records)
    return aggregate


def build_physical_report(
    *, context: Mapping[str, Any], pre_evidence: Mapping[str, Any],
    post_evidence: Mapping[str, Any], local_exports: Sequence[Mapping[str, Any]],
    building_export: Mapping[str, Any], restart_performed: bool,
) -> dict[str, Any]:
    """Build evaluator input while preserving all per-tile parser records."""

    return {
        "schema_version": 1,
        "evidence_kind": context.get("evidence_kind"),
        "transport": context.get("transport"),
        "synthetic": context.get("synthetic"),
        "surface": copy.deepcopy(pre_evidence.get("surface", {})),
        "physical_depth": copy.deepcopy(pre_evidence.get("physical_depth", {})),
        "depth_anything": copy.deepcopy(pre_evidence.get("depth_anything", {})),
        "identity": {
            "restart_performed": restart_performed,
            "pre_restart": _identity(pre_evidence),
            "post_restart": _identity(post_evidence),
            "members": copy.deepcopy(pre_evidence.get("identity", {}).get("members", [])),
        },
        "exports": {
            "local": _aggregate_local(local_exports),
            "local_tiles": copy.deepcopy(list(local_exports)),
            "building": copy.deepcopy(dict(building_export)),
        },
    }


def prepare(context_path: Path, output: Path, workspace: Path) -> None:
    context = validate_capture_context(json.loads(context_path.read_text(encoding="utf-8")), workspace)
    dashboard_url = str(context["dashboard_url"]).rstrip("/")
    evidence = _json_request(dashboard_url + "/api/reconstruction/field-acceptance-evidence")
    _require_available(evidence)
    exports = evidence.get("exports")
    if not isinstance(exports, Mapping):
        raise ValueError("field evidence export requests are unavailable")
    local_requests = exports.get("local_requests")
    building_request = exports.get("building_request")
    if not isinstance(local_requests, list) or not local_requests or not isinstance(building_request, dict):
        raise ValueError("field evidence export request set is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    local = [
        _collect_export(
            dashboard_url=dashboard_url,
            route="/api/reconstruction/export-textured-tile",
            request_payload=request,
            output=output,
            expected_kind="TEXTURED_TILE_GLB",
        )
        for request in local_requests if isinstance(request, dict)
    ]
    if len(local) != len(local_requests):
        raise ValueError("field evidence contains an invalid local export request")
    building = _collect_export(
        dashboard_url=dashboard_url,
        route="/api/reconstruction/export-textured-building",
        request_payload=building_request,
        output=output,
        expected_kind="TEXTURED_BUILDING_GLB",
    )
    payload = {"schema_version": 1, "pre_evidence": evidence, "local_exports": local, "building_export": building}
    (output / "prepared-evidence.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def finalize(
    context_path: Path, output: Path, workspace: Path, browser_results_path: Path,
    *, restart_performed: bool,
) -> None:
    context = validate_capture_context(json.loads(context_path.read_text(encoding="utf-8")), workspace)
    prepared = json.loads((output / "prepared-evidence.json").read_text(encoding="utf-8"))
    browser_results = json.loads(browser_results_path.read_text(encoding="utf-8"))
    if not isinstance(browser_results, dict):
        raise ValueError("Three.js browser results must be an object")
    local = prepared["local_exports"]
    building = prepared["building_export"]
    for record in [*local, building]:
        record["threejs_loaded"] = browser_results.get(record["artifact_name"]) is True
    post = _json_request(str(context["dashboard_url"]).rstrip("/") + "/api/reconstruction/field-acceptance-evidence")
    _require_available(post)
    report = build_physical_report(
        context=context,
        pre_evidence=prepared["pre_evidence"],
        post_evidence=post,
        local_exports=local,
        building_export=building,
        restart_performed=restart_performed,
    )
    (output / "acceptance-report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    evaluated = copy.deepcopy(report)
    for scope in ("local", "building"):
        export = evaluated["exports"][scope]
        export["provenance_bytes"] = Path(export["provenance_path"]).read_bytes()
    result = evaluate_textured_field_acceptance(evaluated)
    (output / "acceptance.json").write_bytes(canonical_acceptance_payload(evaluated))
    print(result.state)
    if not result.accepted:
        print(" / ".join(result.blockers))
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--browser-results", type=Path)
    parser.add_argument("--restart-performed", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.context, args.output, args.workspace)
    else:
        if args.browser_results is None:
            parser.error("finalize requires --browser-results")
        finalize(
            args.context, args.output, args.workspace, args.browser_results,
            restart_performed=args.restart_performed,
        )


if __name__ == "__main__":
    main()
