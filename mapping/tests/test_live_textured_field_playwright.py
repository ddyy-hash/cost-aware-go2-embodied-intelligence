from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageStat
import pytest

from mapping.pc.go2_mapping_appearance.field_acceptance import (
    evaluate_textured_field_acceptance,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "output" / "field-acceptance" / "validated-texture"
CONTEXT_ENV = "GO2_MAPPING_LIVE_ACCEPTANCE_CONTEXT"
SYNTHETIC_MARKER = "SYNTHETIC_TEST_ONLY"


def _validate_live_context(context: object) -> dict:
    if not isinstance(context, dict):
        raise ValueError("live acceptance context must be a JSON object")
    if SYNTHETIC_MARKER in json.dumps(context, sort_keys=True):
        raise ValueError("synthetic fixture identity is forbidden in live acceptance")
    if context.get("schema_version") != 1:
        raise ValueError("live acceptance context schema_version must be 1")
    if context.get("synthetic") is not False:
        raise ValueError("synthetic evidence cannot satisfy live acceptance")
    if context.get("evidence_kind") != "PHYSICAL_GO2_CAPTURE":
        raise ValueError("live acceptance requires PHYSICAL_GO2_CAPTURE evidence")
    if context.get("transport") != "TAILSCALE_DIRECT":
        raise ValueError("live acceptance requires the TAILSCALE_DIRECT business path")
    dashboard_url = context.get("dashboard_url")
    parsed = urllib.parse.urlsplit(dashboard_url if isinstance(dashboard_url, str) else "")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("live acceptance requires an explicit dashboard_url")
    return context


def _physical_report(context: dict, context_path: Path) -> dict:
    report = copy.deepcopy(context)
    for scope in ("local", "building"):
        export = report.get("exports", {}).get(scope, {})
        provenance_path = export.pop("provenance_path", None)
        if provenance_path is not None:
            path = Path(provenance_path)
            if not path.is_absolute():
                path = context_path.parent / path
            export["provenance_bytes"] = path.read_bytes()
    return report


def _powershell_npx() -> str:
    command_shell = shutil.which("cmd.exe")
    if not command_shell or not shutil.which("powershell.exe"):
        pytest.fail("Windows command shell and PowerShell are required for live Playwright acceptance")
    probe = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-Command npx.cmd -ErrorAction Stop).Source",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.fail("Windows npx.cmd is required for live Playwright acceptance")
    return command_shell


def _cli(command_shell: str, session: str, *arguments: str, timeout_s: float = 60.0) -> subprocess.CompletedProcess:
    command = [
        command_shell,
        "/d",
        "/c",
        "npx.cmd",
        "--yes",
        "--package",
        "@playwright/cli",
        "playwright-cli",
        f"-s={session}",
        *arguments,
    ]
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
            timeout=timeout_s,
        )
        stdout_file.seek(0)
        stderr_file.seek(0)
        result = subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            stdout_file.read().decode("utf-8", errors="replace"),
            stderr_file.read().decode("utf-8", errors="replace"),
        )
    assert result.returncode == 0, (
        f"Playwright command failed: {arguments}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    return result


def _windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _assert_nonblank_canvas(path: Path) -> None:
    with Image.open(path).convert("RGB") as image:
        assert image.width >= 300 and image.height >= 250
        extrema = ImageStat.Stat(image).extrema
        assert max(high - low for low, high in extrema) >= 48
        colors = image.resize((96, 96)).getcolors(maxcolors=96 * 96)
        assert colors is not None and len(colors) >= 32


def _assert_not_synthetic_server(base_url: str) -> None:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/__fixture__/meta")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return
    raise AssertionError(f"live dashboard exposed a synthetic fixture endpoint: {body[:200]}")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("synthetic", True, "synthetic evidence"),
        ("evidence_kind", "SYNTHETIC_TEST_ONLY", "synthetic fixture identity"),
        ("transport", "SSH", "TAILSCALE_DIRECT"),
    ],
)
def test_live_context_rejects_nonphysical_or_non_tailnet_evidence(field: str, value: object, message: str):
    context = {
        "schema_version": 1,
        "synthetic": False,
        "evidence_kind": "PHYSICAL_GO2_CAPTURE",
        "transport": "TAILSCALE_DIRECT",
        "dashboard_url": "http://127.0.0.1:8765",
    }
    context[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_live_context(context)


def test_live_textured_field_in_real_browser():
    raw_context_path = os.environ.get(CONTEXT_ENV)
    if not raw_context_path:
        pytest.skip(
            "Go2 offline: GO2_MAPPING_LIVE_ACCEPTANCE_CONTEXT is unset; "
            "no physical room or whole-building acceptance is claimed"
        )

    context_path = Path(raw_context_path)
    if not context_path.is_absolute():
        context_path = ROOT / context_path
    assert context_path.is_file(), f"live acceptance context does not exist: {context_path}"
    context = _validate_live_context(json.loads(context_path.read_text(encoding="utf-8")))
    result = evaluate_textured_field_acceptance(_physical_report(context, context_path))
    assert result.accepted, f"physical numeric/identity/export acceptance failed: {result.blockers}"

    dashboard_url = context["dashboard_url"].rstrip("/")
    _assert_not_synthetic_server(dashboard_url)
    command_shell = _powershell_npx()
    session = "go2-live-acceptance-" + uuid.uuid4().hex[:10]
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        _cli(command_shell, session, "open", f"{dashboard_url}/?render=surface", timeout_s=120)
        ready_code = """async (page) => {
          await page.waitForFunction(() => document.querySelector('#appearance-state')?.textContent === 'TEXTURED_READY', null, {timeout: 30000});
          const source = document.querySelector('#source')?.textContent || '';
          const delivery = document.querySelector('#appearance-delivery')?.textContent || '';
          if (source.includes('SYNTHETIC') || delivery !== 'AUTHORIZED') throw new Error(`invalid live source: ${source} / ${delivery}`);
        }"""
        _cli(command_shell, session, "run-code", ready_code)

        for width, height in ((1440, 1000), (1024, 768), (768, 1024), (390, 844)):
            _cli(command_shell, session, "resize", str(width), str(height))
            layout_code = """async (page) => {
              await page.waitForTimeout(500);
              const result = await page.evaluate(() => {
                const rect = (selector) => { const r = document.querySelector(selector).getBoundingClientRect(); return {left:r.left,right:r.right,top:r.top,bottom:r.bottom}; };
                const overlap = (a,b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
                const label = rect('.scene-label'); const tools = rect('.scene-tools'); const controls = rect('.surface-controls');
                return {overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth, labelTools:overlap(label,tools), labelControls:overlap(label,controls), toolsControls:overlap(tools,controls), state:document.querySelector('#appearance-state').textContent, ready:document.querySelector('#texture-state').dataset.ready};
              });
              if (result.overflow > 1 || result.labelTools || result.labelControls || result.toolsControls) throw new Error(`layout ${JSON.stringify(result)}`);
              if (result.state !== 'TEXTURED_READY' || result.ready !== 'true') throw new Error(`render ${JSON.stringify(result)}`);
            }"""
            _cli(command_shell, session, "run-code", layout_code)
            screenshot = EVIDENCE_ROOT / f"live-textured-{width}x{height}.png"
            canvas = EVIDENCE_ROOT / f"live-textured-canvas-{width}x{height}.png"
            _cli(command_shell, session, "screenshot", "--filename", _windows_path(screenshot), "--full-page")
            canvas_code = f"async (page) => {{ await page.locator('#scene').screenshot({{path: {json.dumps(_windows_path(canvas))}}}); }}"
            _cli(command_shell, session, "run-code", canvas_code)
            _assert_nonblank_canvas(canvas)

        surface_only_code = """async (page) => {
          if (await page.locator('#point-diagnostics').count()) throw new Error('Surface lane exposed point diagnostics');
          if (await page.locator('#appearance-state').textContent() !== 'TEXTURED_READY') throw new Error('Surface-only view changed appearance state');
          await page.waitForTimeout(12000);
        }"""
        _cli(command_shell, session, "run-code", surface_only_code, timeout_s=30)
        console = _cli(command_shell, session, "console", "warning").stdout.lower()
        assert "console messages" not in console or "0 messages" in console, console
    finally:
        try:
            _cli(command_shell, session, "close", timeout_s=30)
        except (AssertionError, subprocess.SubprocessError):
            pass
