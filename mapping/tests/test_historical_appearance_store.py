import json

import pytest

from mapping.pc.go2_mapping_appearance.historical_store import HistoricalAppearanceStore
from mapping.pc.go2_mapping_appearance.vertex_color import build_historical_vertex_color_sidecar
from mapping.pc.go2_mapping_rgbd.texture import VertexColor


def _sidecar(geometry="a" * 64, color=(0.2, 0.4, 0.6)):
    vertex = VertexColor("0:0:0", color, (0.0, 0.0, 0.0), 2, 0, 2, ("f" * 64,), True)
    return build_historical_vertex_color_sidecar(
        geometry_revision_id=geometry,
        pose_generation="b" * 64,
        calibration_id="cal-1",
        calibration_sha256="c" * 64,
        physical_depth_validation_revision="d" * 64,
        appearance_config_sha256="e" * 64,
        vertex_colors=(vertex,),
    )


def test_store_publishes_reuses_and_reloads_verified_revision(tmp_path):
    store = HistoricalAppearanceStore(tmp_path)
    first = store.publish(_sidecar(), expected_geometry_revision_id="a" * 64)
    second = store.publish(_sidecar(), expected_geometry_revision_id="a" * 64)
    restarted = HistoricalAppearanceStore(tmp_path).load(first.appearance_revision_id, expected_geometry_revision_id="a" * 64)

    assert second == first == restarted
    assert first.binary_path.read_bytes()
    assert first.manifest["manifest_sha256"] == first.manifest_sha256
    assert not list(tmp_path.rglob("*.partial"))


def test_store_rejects_stale_geometry_tamper_and_path_escape(tmp_path):
    store = HistoricalAppearanceStore(tmp_path)
    published = store.publish(_sidecar(), expected_geometry_revision_id="a" * 64)

    with pytest.raises(ValueError, match="geometry"):
        store.load(published.appearance_revision_id, expected_geometry_revision_id="9" * 64)
    with pytest.raises(ValueError):
        store.load("../escape")

    payload = bytearray(published.binary_path.read_bytes())
    payload[-1] ^= 1
    published.binary_path.write_bytes(payload)
    with pytest.raises(ValueError, match="verification"):
        store.load(published.appearance_revision_id)


def test_store_rejects_conflicting_existing_revision(tmp_path):
    store = HistoricalAppearanceStore(tmp_path)
    published = store.publish(_sidecar())
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = "CORRUPT"
    published.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        store.publish(_sidecar())
