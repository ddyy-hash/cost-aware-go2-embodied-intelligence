from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.nav2_map_preflight import validate_nav2_map
from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.nav2_params import prepare_nav2_params


def _write_map(root: Path, *, geometry_authority: bool = False, width: int = 4, height: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "map.pgm").write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes([254]) * (width * height))
    (root / "map.yaml").write_text(
        "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n",
        encoding="utf-8",
    )
    (root / "map.provenance.json").write_text(
        json.dumps({
            "schema_version": 1,
            "map_id": "room-01",
            "coordinate_frame": "SESSION_ODOM_LOCAL",
            "provenance": "USER_PRIOR",
            "geometry_authority": geometry_authority,
            "source_revision": 1,
            "width": width,
            "height": height,
            "resolution": 0.1,
        }),
        encoding="utf-8",
    )
    return root / "map.yaml"


def test_nav2_map_preflight_accepts_paired_prior_artifacts(tmp_path):
    result = validate_nav2_map(_write_map(tmp_path / "map"))

    assert result["map_id"] == "room-01"
    assert result["coordinate_frame"] == "SESSION_ODOM_LOCAL"
    assert result["provenance"] == "USER_PRIOR"
    assert result["geometry_authority"] is False


def test_nav2_params_overlay_uses_map_provenance_frame_without_mutating_source(tmp_path):
    source = tmp_path / "nav2_params.yaml"
    source.write_text(
        "map_server:\n  ros__parameters:\n    frame_id: map\n"
        "amcl:\n  ros__parameters:\n    global_frame_id: map\n"
        "bt_navigator:\n  ros__parameters:\n    global_frame: map\n"
        "global_costmap:\n  global_costmap:\n    ros__parameters:\n      global_frame: map\n",
        encoding="utf-8",
    )

    overlay = prepare_nav2_params(source, "SESSION_ODOM_LOCAL", tmp_path / "generated")
    second_overlay = prepare_nav2_params(source, "BUILDING_LOCAL", tmp_path / "generated")

    assert overlay.is_file()
    assert second_overlay.is_file()
    assert overlay != source
    assert second_overlay != source
    assert second_overlay != overlay
    assert "frame_id: map" in source.read_text(encoding="utf-8")
    params = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    assert params["map_server"]["ros__parameters"]["frame_id"] == "SESSION_ODOM_LOCAL"
    assert params["amcl"]["ros__parameters"]["global_frame_id"] == "SESSION_ODOM_LOCAL"
    assert params["bt_navigator"]["ros__parameters"]["global_frame"] == "SESSION_ODOM_LOCAL"
    assert params["global_costmap"]["global_costmap"]["ros__parameters"]["global_frame"] == "SESSION_ODOM_LOCAL"
    second_params = yaml.safe_load(second_overlay.read_text(encoding="utf-8"))
    assert second_params["map_server"]["ros__parameters"]["frame_id"] == "BUILDING_LOCAL"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda root: (root / "map.provenance.json").write_text(json.dumps({**json.loads((root / "map.provenance.json").read_text(encoding="utf-8")), "geometry_authority": True}), encoding="utf-8"), "geometry_authority"),
        (lambda root: (root / "map.pgm").write_bytes(b"P5\n2 2\n255\n" + bytes(4)), "dimensions"),
        (lambda root: (root / "map.yaml").write_text("image: ../outside.pgm\nresolution: 0.1\norigin: [0,0,0]\n", encoding="utf-8"), "inside map directory"),
    ],
)
def test_nav2_map_preflight_rejects_untrusted_or_mismatched_artifacts(tmp_path, mutation, match):
    root = tmp_path / "map"
    yaml_path = _write_map(root)
    mutation(root)

    with pytest.raises(ValueError, match=match):
        validate_nav2_map(yaml_path)
