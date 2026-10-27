from pathlib import Path


STYLE = Path(__file__).parents[1] / "pc" / "go2_mapping_dashboard" / "static" / "style.css"


def test_dashboard_uses_cohesive_visual_tokens_and_touch_targets():
    css = STYLE.read_text(encoding="utf-8")
    assert "--color-bg:" in css
    assert "--color-accent:" in css
    assert "--color-warning:" in css
    assert "--color-danger:" in css
    assert "--radius-card: 1rem" in css
    assert "min-height: 2.75rem" in css
    assert "border-radius: var(--radius-card)" in css


def test_dashboard_does_not_collapse_status_colors_into_one_accent():
    css = STYLE.read_text(encoding="utf-8")
    assert "--lime: var(--color-success)" in css
    assert "--amber: var(--color-warning)" in css
    assert "--coral: var(--color-danger)" in css
