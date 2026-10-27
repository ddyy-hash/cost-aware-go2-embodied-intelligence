from pathlib import Path


STATIC = Path(__file__).parents[1] / "pc" / "go2_mapping_dashboard" / "static"


def test_dashboard_exposes_explicit_simulation_marker_for_provisional_streams():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="simulation-badge"' in html
    assert "PROVISIONAL_SIMULATION" in app
    assert "simulation-badge" in app
