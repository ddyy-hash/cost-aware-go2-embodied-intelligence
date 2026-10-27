import json

import pytest

from mapping.pc.go2_mapping_locomotion.policy_catalog import (
    PolicyCatalogError,
    policy_asset,
    select_policy_level,
)


def test_catalog_selects_and_verifies_all_downloaded_levels():
    for level in ("03", "07", "10", "13", "17", "20"):
        asset = policy_asset("mapping/config/pgtt_go2_policy_catalog.json", level, root=".")
        assert asset.input_size == 153
        assert asset.action_size == 12
        assert asset.executable is False


@pytest.mark.parametrize(
    ("terrain", "risk", "expected"),
    [("STAIR_UP", "low", "03"), ("STAIR_DOWN", "high", "13"), ("ROUGH_TERRAIN", "medium", "10")],
)
def test_policy_selection_is_conservative(terrain, risk, expected):
    assert select_policy_level(terrain_state=terrain, risk_level=risk) == expected


def test_unknown_policy_level_fails_closed():
    with pytest.raises(PolicyCatalogError, match="not cataloged"):
        policy_asset("mapping/config/pgtt_go2_policy_catalog.json", "99", root=".")
