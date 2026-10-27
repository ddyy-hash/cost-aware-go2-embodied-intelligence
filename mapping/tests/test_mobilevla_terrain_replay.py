import json

from tools.replay_mobilevla_terrain import replay


TERRAIN = {
    "state": "STAIR_UP",
    "confidence": 0.95,
    "riser_height_m": 0.12,
    "tread_depth_m": 0.30,
    "width_m": 0.90,
    "pitch_deg": 22.0,
    "step_count": 6,
    "top_landing_supported": True,
    "bottom_landing_supported": True,
}


def test_replay_defaults_to_fail_closed_without_operator_or_lease():
    result = replay(
        json.dumps({"intent": "inspect_room", "target": "room_a", "confidence": 0.9}),
        TERRAIN,
        terrain_task="STAIR_ASCEND",
    )

    assert result["supervisor"]["state"] == "WAITING_OPERATOR"
    assert result["supervisor"]["command"] is None
    assert result["executable"] is False
    assert result["transport_authorized"] is False


def test_replay_records_ready_supervisor_request_but_never_authorizes_transport():
    result = replay(
        {"intent": "inspect_room", "target": "room_a", "confidence": 0.9},
        TERRAIN,
        terrain_task="STAIR_ASCEND",
        operator_confirmed=True,
        lease_id="replay-lease-1",
    )

    assert result["supervisor"]["state"] == "READY"
    assert result["supervisor"]["command"]["controller"] == "pgtt-go2"
    assert result["executable"] is False
    assert result["transport_authorized"] is False
    assert result["navigation_activation"] is False
