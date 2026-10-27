from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_derp_benchmark_collects_both_endpoints_and_is_report_only_by_default():
    script = (ROOT / "tools/measure_go2_derp_candidates.ps1").read_text(encoding="utf-8")

    assert "tailscale.exe netcheck" in script
    assert "tailscale netcheck" in script
    assert "derp_selector" in script
    assert "CurAddr" in script
    assert "[switch]$ApplyPreference" in script
    assert "if ($ApplyPreference" in script
    assert "force-prefer-derp" in script
    assert "official DERP fallback remains enabled" in script


def test_derp_benchmark_never_applies_a_relay_when_selector_reports_direct():
    script = (ROOT / "tools/measure_go2_derp_candidates.ps1").read_text(encoding="utf-8")

    assert "$decision.preferred_region_id -gt 0" in script
    assert "P2P/direct or automatic DERP selection remains active" in script
    assert "BatchMode=yes" in script
    assert "ConnectTimeout=12" in script
