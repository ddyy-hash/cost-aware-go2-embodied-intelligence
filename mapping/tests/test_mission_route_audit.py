from mapping.pc.go2_mapping_dashboard.mission import audit_mission_route


def _coarse_map():
    return {
        "schema_version": 1,
        "map_id": "route-audit-floor",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [
            {
                "id": "room-1",
                "name": "Room 1",
                "kind": "room",
                "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]],
            }
        ],
        "corridors": [],
        "walls": [],
        "doors": [],
        "features": [],
        "anchors": [],
        "keep_out_zones": [],
    }


def _task(**overrides):
    task = {
        "task_id": "room-1:floor",
        "room_id": "room-1",
        "kind": "floor_coverage",
        "room_polygon": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "waypoints": [[1, 1], [3, 1], [5, 3]],
        "inspection_targets": [],
        "status": "READY",
    }
    task.update(overrides)
    return task


def _finding_codes(report):
    return {finding["code"] for finding in report["findings"]}


def test_route_audit_allows_complete_route_for_human_review():
    report = audit_mission_route(_task(), _coarse_map())

    assert report["state"] == "READY_FOR_HUMAN_REVIEW"
    assert report["route_source"] == "WAYPOINTS"
    assert report["navigation_eligible"] is True
    assert report["route_length_m"] == 4.83
    assert _finding_codes(report) == {"HUMAN_REVIEW_REQUIRED"}


def test_route_audit_blocks_polygon_fallback_from_navigation():
    report = audit_mission_route(_task(waypoints=[]), _coarse_map())

    assert report["state"] == "REVIEW_REQUIRED"
    assert report["route_source"] == "POLYGON_FALLBACK"
    assert report["navigation_eligible"] is False
    assert "POLYGON_FALLBACK" in _finding_codes(report)


def test_route_audit_blocks_keep_out_crossing_and_route_escape():
    coarse_map = _coarse_map()
    coarse_map["keep_out_zones"] = [
        {"id": "keep-out-1", "polygon": [[2, 0.5], [4, 0.5], [4, 2.5], [2, 2.5]]}
    ]
    report = audit_mission_route(_task(waypoints=[[1, 1], [5, 1], [7, 3]]), coarse_map)

    assert report["state"] == "BLOCKED"
    assert report["navigation_eligible"] is False
    assert {"KEEP_OUT_ZONE_CROSSING", "ROUTE_OUTSIDE_TASK_ZONE"}.issubset(_finding_codes(report))
