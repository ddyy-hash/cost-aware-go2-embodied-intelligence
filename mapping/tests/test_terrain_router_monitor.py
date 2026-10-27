from mapping.robot.go2_mapping_locomotion.terrain_router_monitor import _status_should_refresh


def test_router_status_refreshes_unchanged_route_before_executor_stale_window():
    assert _status_should_refresh(
        ("FLAT", "03", True, None, False, 0.5),
        ("FLAT", "03", True, None, False, 0.5),
        now=0.30,
        last_written_at=0.0,
        refresh_period=0.10,
    ) is True


def test_router_status_writes_immediately_when_route_changes():
    assert _status_should_refresh(
        ("FLAT", "03", True, None, False, 0.5),
        ("STAIR_UP", "07", True, None, False, 0.5),
        now=0.01,
        last_written_at=0.0,
        refresh_period=0.10,
    ) is True
