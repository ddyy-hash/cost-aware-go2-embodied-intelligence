from pathlib import Path


def test_room_scan_entrypoint_defaults_to_local_geometry_and_no_motion():
    script = Path("tools/start_go2_room_scan.ps1").read_text(encoding="utf-8")
    assert "CoordinateScope = 'SESSION_ODOM_LOCAL'" in script
    assert "GeometryOnlyCalibration = $true" in script
    assert "start_go2_mapping_session_tailnet.ps1" in script
    assert "sport" not in script.lower()


def test_room_scan_operator_token_generator_is_bounded_and_short_lived():
    script = Path("tools/create_go2_room_scan_operator_token.ps1").read_text(encoding="utf-8")
    assert "TtlMinutes = 15" in script
    assert "RandomNumberGenerator" in script
    assert "RandomNumberGenerator]::Create" in script
    assert ".GetBytes($bytes)" in script
    assert "expires_at_unix" in script
    assert "mapping-data\\secrets" in script
    assert "StartsWith($secretsRoot" in script
