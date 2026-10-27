from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

from PIL import Image, ImageStat
import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "mapping" / "tests" / "appearance_fixture_server.py"
FIXTURE_ROOT = ROOT / "mapping" / "tests" / "fixtures" / "appearance" / "browser"
STATIC_ROOT = ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static"
EVIDENCE_ROOT = ROOT / "output" / "playwright" / "validated-texture-final"
MARKER = "SYNTHETIC_TEST_ONLY"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_ready(url: str, process: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            raise AssertionError(f"fixture server exited early\nstdout={stdout}\nstderr={stderr}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("fixture server did not become ready")


def _powershell_npx() -> tuple[str, str]:
    command_shell = shutil.which("cmd.exe")
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not command_shell or not powershell:
        pytest.skip("Windows command shell and PowerShell are required for Playwright CLI discovery")
    probe = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", "(Get-Command npx.cmd -ErrorAction Stop).Source"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.skip("Windows npx.cmd is unavailable for the repository Playwright CLI")
    npx = probe.stdout.strip()
    discovery = subprocess.run(
        [
            command_shell,
            "/d",
            "/s",
            "/c",
            subprocess.list2cmdline([npx, "--yes", "--package", "@playwright/cli", "where", "playwright-cli.cmd"]),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    candidates = [line.strip() for line in discovery.stdout.splitlines() if line.strip()]
    if discovery.returncode != 0 or not candidates:
        pytest.skip("@playwright/cli could not be resolved through npx.cmd")
    return command_shell, candidates[0]


def _cli(playwright: tuple[str, str], session: str, *arguments: str, timeout_s: float = 60.0) -> subprocess.CompletedProcess:
    command_shell, playwright_cli = playwright
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cli-", dir=EVIDENCE_ROOT) as temporary_directory:
        effective_arguments = list(arguments)
        if len(effective_arguments) == 2 and effective_arguments[0] == "run-code":
            script = Path(temporary_directory) / "snippet.js"
            script.write_text(effective_arguments[1], encoding="utf-8")
            effective_arguments = ["run-code", "--filename", _windows_path(script)]

        values = [
            playwright_cli,
            f"-s={session}",
            *effective_arguments,
        ]
        command = subprocess.list2cmdline(values)
        # The CLI keeps a persistent browser daemon alive. Captured pipes remain
        # inherited by that daemon, so communicate() can wait for EOF indefinitely.
        with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
            completed = subprocess.run(
                [command_shell, "/d", "/s", "/c", command],
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
    assert result.returncode == 0, f"command failed: {arguments}\nstdout={result.stdout}\nstderr={result.stderr}"
    return result


def _windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    result = subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _assert_nonblank_canvas(path: Path) -> None:
    with Image.open(path).convert("RGB") as image:
        assert image.width >= 300 and image.height >= 250
        extrema = ImageStat.Stat(image).extrema
        assert max(high - low for low, high in extrema) >= 48
        colors = image.resize((96, 96)).getcolors(maxcolors=96 * 96)
        assert colors is not None and len(colors) >= 32


def test_fixture_is_test_only_and_not_discoverable_from_production_runtime():
    contract = json.loads((FIXTURE_ROOT / "fixture.json").read_text(encoding="utf-8"))
    assert contract["marker"] == MARKER
    assert contract["geometry_authority"] is False
    assert contract["acceptance_authority"] is False
    production = (ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "server.py").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert MARKER not in production
    assert "GO2_MAPPING_APPEARANCE_FIXTURE_ROOT" not in production
    assert "GO2_MAPPING_SYNTHETIC_TEST_ONLY" not in production
    assert "GO2_MAPPING_APPEARANCE_FIXTURE_ROOT" not in app


def test_fixture_server_refuses_normal_runtime_without_both_test_switches(tmp_path):
    env = os.environ.copy()
    env.pop("GO2_MAPPING_SYNTHETIC_TEST_ONLY", None)
    env.pop("GO2_MAPPING_APPEARANCE_FIXTURE_ROOT", None)
    result = subprocess.run(
        [sys.executable, str(SERVER), "--port", str(_free_port())],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert "GO2_MAPPING_SYNTHETIC_TEST_ONLY=1" in result.stderr


def test_authorized_texture_browser_real_render_lods_points_and_responsive_layout():
    powershell = _powershell_npx()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["GO2_MAPPING_SYNTHETIC_TEST_ONLY"] = "1"
    env["GO2_MAPPING_APPEARANCE_FIXTURE_ROOT"] = str(FIXTURE_ROOT)
    server = subprocess.Popen(
        [sys.executable, str(SERVER), "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    session = "go2-appearance-" + uuid.uuid4().hex[:10]
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        _wait_ready(f"{base_url}/__fixture__/meta", server)
        with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=3) as response:
            routes = json.load(response)["paths"]
            assert response.headers["X-Go2-Test-Marker"] == MARKER
        assert "/api/reconstruction/export-textured-tile" not in routes
        assert "/api/reconstruction/export-textured-building" not in routes
        assert "/v1/sessions" not in routes
        _cli(powershell, session, "open", f"{base_url}/?render=surface", timeout_s=120)
        ready_code = """async (page) => {
          await page.waitForFunction(() => document.querySelector('#appearance-state')?.textContent === 'TEXTURED_READY', null, {timeout: 20000});
          const source = await page.locator('#source').textContent() || '';
          if (!source.includes('SYNTHETIC_TEST_ONLY')) throw new Error('fixture marker is not visible');
          const delivery = await page.locator('#appearance-delivery').textContent();
          if (delivery !== 'AUTHORIZED') throw new Error(`delivery telemetry: ${delivery}`);
        }"""
        _cli(powershell, session, "run-code", ready_code)

        viewports = [(1440, 1000), (1024, 768), (768, 1024), (390, 844)]
        for width, height in viewports:
            _cli(powershell, session, "resize", str(width), str(height))
            layout_code = """async (page) => {
              await page.waitForTimeout(300);
              const result = await page.evaluate(() => {
                const rect = (selector) => { const r = document.querySelector(selector).getBoundingClientRect(); return {left:r.left,right:r.right,top:r.top,bottom:r.bottom}; };
                const overlap = (a,b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
                const label = rect('.scene-label'); const tools = rect('.scene-tools'); const controls = rect('.surface-controls');
                return {
                  viewport: {width: window.innerWidth, height: window.innerHeight},
                  overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
                  labelTools: overlap(label, tools), labelControls: overlap(label, controls), toolsControls: overlap(tools, controls),
                  label, tools, controls,
                  state: document.querySelector('#appearance-state').textContent,
                  ready: document.querySelector('#texture-state').dataset.ready,
                  canvas: {width: document.querySelector('#scene').width, height: document.querySelector('#scene').height},
                };
              });
              if (result.overflow > 1 || result.labelTools || result.labelControls || result.toolsControls) throw new Error(`layout ${JSON.stringify(result)}`);
              if (result.state !== 'TEXTURED_READY' || result.ready !== 'true' || result.canvas.width < 300 || result.canvas.height < 250) throw new Error(`render ${JSON.stringify(result)}`);
            }"""
            _cli(powershell, session, "run-code", layout_code)
            screenshot = EVIDENCE_ROOT / f"synthetic-textured-{width}x{height}.png"
            canvas = EVIDENCE_ROOT / f"synthetic-textured-canvas-{width}x{height}.png"
            _cli(powershell, session, "screenshot", "--filename", _windows_path(screenshot), "--full-page")
            canvas_code = f"async (page) => {{ await page.locator('#scene').screenshot({{path: {json.dumps(_windows_path(canvas))}}}); }}"
            _cli(powershell, session, "run-code", canvas_code)
            _assert_nonblank_canvas(canvas)

        surface_only_code = """async (page) => {
          if (await page.locator('#point-diagnostics').count()) throw new Error('Surface lane exposed point diagnostics');
          if (await page.locator('#appearance-state').textContent() !== 'TEXTURED_READY') throw new Error('Surface-only view changed appearance state');
        }"""
        _cli(powershell, session, "run-code", surface_only_code)

        lod_code = """async (page) => {
          const page_url = await page.url();
          const meta_url = await page.evaluate(() => new URL('/__fixture__/meta', location.href).href);
          const meta = await (await page.request.get(meta_url)).json();
          const lod = page.locator('#appearance-lod');
          if (!(await lod.textContent()).includes(meta.detail_lod_id)) throw new Error('detail LOD was not initially selected');
          await page.locator('#scene').evaluate((element) => element.dispatchEvent(new WheelEvent('wheel', {deltaY: 5000, bubbles: true, cancelable: true})));
          await page.waitForFunction((id) => document.querySelector('#appearance-lod')?.textContent?.includes(id), meta.coarse_lod_id, {timeout: 5000});
          await page.locator('#scene').evaluate((element) => element.dispatchEvent(new WheelEvent('wheel', {deltaY: -5000, bubbles: true, cancelable: true})));
          await page.waitForFunction((id) => document.querySelector('#appearance-lod')?.textContent?.includes(id), meta.detail_lod_id, {timeout: 5000});
          if (await page.locator('#appearance-state').textContent() !== 'TEXTURED_READY') throw new Error('LOD transition lost readiness');
        }"""
        _cli(powershell, session, "run-code", lod_code)
        console = _cli(powershell, session, "console", "warning").stdout.lower()
        assert "console messages" not in console or "0 messages" in console, console
    finally:
        try:
            _cli(powershell, session, "close", timeout_s=30)
        except (AssertionError, subprocess.SubprocessError):
            pass
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
