from mapping.pc.go2_mapping_dashboard.robot_session import RobotSessionTelemetry


def test_robot_session_heartbeat_is_active_only_inside_its_ttl():
    session = RobotSessionTelemetry(max_age_s=10.0)
    session.ingest(
        {
            "session_id": "segment-20260731T094647Z",
            "mode": "full",
            "state": "RECORDING",
            "capture_limit_gib": 10,
            "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity",
            "mission_id": "mission-20260825T120000Z",
            "task_id": "room-1:corners",
            "task_revision": 3,
            "lease_id": "lease-room-1",
        },
        now=100.0,
    )

    active = session.snapshot(now=105.0)
    stale = session.snapshot(now=111.0)

    assert active["capture_active"] is True
    assert active["heartbeat_age_ms"] == 5000
    assert active["preview_policy"] == "realtime_preview_priority_high_fidelity_remaining_capacity"
    assert active["mission_id"] == "mission-20260825T120000Z"
    assert active["task_id"] == "room-1:corners"
    assert active["task_revision"] == 3
    assert active["lease_id"] == "lease-room-1"
    assert active["preview_share"] is None
    assert active["high_fidelity_share"] is None
    assert stale["capture_active"] is False
    assert stale["state"] == "STALE"


def test_robot_session_rejects_unsafe_or_inconsistent_heartbeat():
    session = RobotSessionTelemetry()

    for payload in (
        {"session_id": "../escape", "mode": "full", "state": "RECORDING", "capture_limit_gib": 10, "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity"},
        {"session_id": "segment-safe", "mode": "bad", "state": "RECORDING", "capture_limit_gib": 10, "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity"},
        {"session_id": "segment-safe", "mode": "full", "state": "RECORDING", "capture_limit_gib": 10, "preview_policy": "exact_5pct"},
        {"session_id": "segment-safe", "mode": "full", "state": "RECORDING", "capture_limit_gib": 10, "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity", "task_revision": 0},
    ):
        try:
            session.ingest(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid heartbeat was accepted")


def test_robot_session_accepts_an_active_legacy_heartbeat_during_priority_policy_rollout():
    session = RobotSessionTelemetry()

    session.ingest(
        {
            "session_id": "segment-legacy",
            "mode": "full",
            "state": "RECORDING",
            "capture_limit_gib": 10,
            "preview_policy": "strict_dynamic_5pct_preview_95pct_high_fidelity",
        }
    )

    assert session.snapshot()["capture_active"] is True


def test_robot_session_reports_finalization_without_claiming_active_capture():
    session = RobotSessionTelemetry()

    snapshot = session.ingest(
        {
            "session_id": "segment-finalizing",
            "mode": "full",
            "state": "FINALIZING",
            "capture_limit_gib": 10,
            "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity",
        }
    )

    assert snapshot["state"] == "FINALIZING"
    assert snapshot["capture_active"] is False


def test_robot_session_reports_startup_without_claiming_active_capture():
    session = RobotSessionTelemetry()

    snapshot = session.ingest(
        {
            "session_id": "segment-starting",
            "mode": "full",
            "state": "STARTING",
            "capture_limit_gib": 10,
            "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity",
        }
    )

    assert snapshot["state"] == "STARTING"
    assert snapshot["capture_active"] is False
