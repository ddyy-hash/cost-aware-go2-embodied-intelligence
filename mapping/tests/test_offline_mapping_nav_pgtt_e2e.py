from tools.run_offline_mapping_nav_pgtt_e2e import build_e2e_report


def test_e2e_report_requires_all_layers_and_preserves_non_production_boundary():
    report = build_e2e_report(
        mapping={"accepted": True, "map_id": "map-1", "coordinate_frame": "SESSION_ODOM_LOCAL"},
        nav2={"accepted": True, "goal_state": "DONE", "waypoint_count": 3},
        vla={"accepted": True, "state": "READY", "controller": "pgtt-go2"},
        locomotion={"accepted": True},
        robustness={"accepted": True},
        safety={"accepted": True},
    )

    assert report["accepted"] is True
    assert report["executable"] is False
    assert report["transport_authorized"] is False
    assert report["ros_nav2_runtime"] is False
    assert report["layers"]["vla"]["state"] == "READY"


def test_e2e_report_rejects_an_unaccepted_layer():
    report = build_e2e_report(
        mapping={"accepted": True},
        nav2={"accepted": True},
        vla={"accepted": True},
        locomotion={"accepted": False},
        robustness={"accepted": True},
        safety={"accepted": True},
    )

    assert report["accepted"] is False
