from pathlib import Path


STATIC = Path(__file__).parents[1] / "pc" / "go2_mapping_dashboard" / "static"


def test_observe_surface_hides_overlapping_observation_reference():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert 'html[data-workspace-mode="observe"] #coarse-observation-reference' in css
    assert "display: none !important" in css


def test_primary_observe_surface_labels_are_chinese():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "空间建图控制台" in html
    assert "实时空间重建" in html
    assert "观测" in html
    assert "粗建" in html
