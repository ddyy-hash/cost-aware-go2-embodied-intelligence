from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import mapping.pc.go2_mapping_dashboard.stair_survey as stair_survey
from mapping.pc.go2_mapping_dashboard.stair_survey import (
    StairSurveyStore,
    aggregate_stair_observations,
    extract_stair_profile_from_points,
    qualify_stair_profile,
    replay_stair_capture,
)


def _profile(**overrides):
    profile = {
        "step_count": 5,
        "riser_height_m": 0.16,
        "tread_depth_m": 0.28,
        "width_m": 0.82,
        "pitch_deg": 29.7,
        "geometry_confidence": 0.94,
        "top_landing_supported": True,
        "bottom_landing_supported": True,
        "static_scene": True,
        "unobstructed": True,
        "non_hollow": True,
        "straight_flight": True,
    }
    profile.update(overrides)
    return profile


def test_stair_qualifier_accepts_only_a_stable_supported_straight_flight():
    report = qualify_stair_profile(_profile())

    assert report["state"] == "STAIR_QUALIFIED"
    assert report["blocking_reasons"] == []
    assert report["profile"]["step_count"] == 5
    assert report["locomotion_authorized"] is False


def test_stair_qualifier_rejects_unsafe_geometry_and_records_each_reason():
    report = qualify_stair_profile(
        _profile(
            riser_height_m=0.31,
            width_m=0.30,
            top_landing_supported=False,
            non_hollow=False,
            unobstructed=False,
        )
    )

    assert report["state"] == "STAIR_REJECTED"
    assert report["locomotion_authorized"] is False
    assert {
        "RISER_OUTSIDE_TESTED_ENVELOPE",
        "STAIR_WIDTH_OUTSIDE_TESTED_ENVELOPE",
        "TOP_LANDING_UNSUPPORTED",
        "HOLLOW_STAIR_UNACCEPTED",
        "STAIR_OBSTRUCTED_OR_UNMODELED",
    }.issubset(report["blocking_reasons"])


def test_stair_survey_requires_repeated_source_identified_fresh_observations():
    observations = [
        {
            "timestamp_ns": 1_000_000_000,
            "profile": _profile(),
            "imu_fresh": True,
            "low_state_fresh": True,
        },
        {
            "timestamp_ns": 2_000_000_000,
            "profile": _profile(riser_height_m=0.165),
            "imu_fresh": True,
            "low_state_fresh": True,
        },
        {
            "timestamp_ns": 3_000_000_000,
            "profile": _profile(tread_depth_m=0.275),
            "imu_fresh": True,
            "low_state_fresh": True,
        },
    ]

    report = aggregate_stair_observations(
        observations,
        source_identity={"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "survey-1"},
    )

    assert report["state"] == "STAIR_QUALIFIED"
    assert report["observation_count"] == 3
    assert report["source_identity"]["capture_id"] == "survey-1"
    assert report["measurement_authority"] is False
    assert report["locomotion_authorized"] is False


def test_stair_survey_preserves_profile_blockers_in_the_aggregate_report():
    report = aggregate_stair_observations(
        [
            {"timestamp_ns": index, "profile": _profile(riser_height_m=0.40), "imu_fresh": True, "low_state_fresh": True}
            for index in (1, 2, 3)
        ],
        source_identity={"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "bad-geometry"},
    )

    assert report["state"] == "STAIR_REJECTED"
    assert "STAIR_PROFILE_REJECTED" in report["blocking_reasons"]
    assert "RISER_OUTSIDE_TESTED_ENVELOPE" in report["blocking_reasons"]


def test_stair_survey_rejects_missing_freshness_and_unstable_profiles():
    report = aggregate_stair_observations(
        [
            {"timestamp_ns": 1, "profile": _profile(), "imu_fresh": False, "low_state_fresh": True},
            {"timestamp_ns": 2, "profile": _profile(riser_height_m=0.22), "imu_fresh": True, "low_state_fresh": False},
        ],
        source_identity={"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "survey-2"},
    )

    assert report["state"] == "STAIR_REJECTED"
    assert "INSUFFICIENT_REPEATED_OBSERVATIONS" in report["blocking_reasons"]
    assert "IMU_NOT_FRESH" in report["blocking_reasons"]
    assert "LOW_STATE_NOT_FRESH" in report["blocking_reasons"]
    assert "STAIR_GEOMETRY_UNSTABLE" in report["blocking_reasons"]


def test_stair_point_cloud_extractor_reconstructs_repeated_treads_for_qualification():
    points = []
    for step in range(5):
        start_x = 0.5 + step * 0.28
        for sample in range(12):
            x = start_x + sample * 0.02
            for lateral in range(9):
                y = -0.4 + lateral * 0.1
                points.append([x, y, step * 0.16])
    for sample in range(20):
        x = sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.0])
        x = 1.9 + sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.64])

    profile = extract_stair_profile_from_points(
        points,
        evidence={
            "top_landing_supported": True,
            "bottom_landing_supported": True,
            "static_scene": True,
            "unobstructed": True,
            "non_hollow": True,
            "straight_flight": True,
        },
    )

    assert profile["step_count"] >= 3
    assert 0.10 <= profile["riser_height_m"] <= 0.20
    assert 0.20 <= profile["tread_depth_m"] <= 0.36
    assert profile["width_m"] >= 0.7
    assert qualify_stair_profile(profile)["state"] == "STAIR_QUALIFIED"


def test_stair_survey_store_rejects_raw_points_or_authority_and_ignores_corrupt_artifacts(tmp_path):
    store = StairSurveyStore(tmp_path / "stair-survey.json")
    valid = {
        "schema_version": 1,
        "state": "STAIR_REJECTED",
        "measurement_authority": False,
        "locomotion_authorized": False,
    }

    store.save(valid)
    assert store.load() == valid

    invalid_raw = {**valid, "points_xyz": [[0.0, 0.0, 0.0]]}
    try:
        store.save(invalid_raw)
    except ValueError as exc:
        assert "raw point cloud" in str(exc)
    else:
        raise AssertionError("raw point cloud report was accepted")
    try:
        store.save({**valid, "locomotion_authorized": True})
    except ValueError as exc:
        assert "locomotion authorization" in str(exc)
    else:
        raise AssertionError("motion-authorized report was accepted")

    assert store.load() == valid
    store.path.write_text(
        '{"schema_version":99,"measurement_authority":false,"locomotion_authorized":false,"state":"STAIR_REJECTED"}',
        encoding="utf-8",
    )
    assert store.load() is None

    store.path.write_text("not-json", encoding="utf-8")
    assert store.load() is None


def test_stair_capture_replay_pairs_lidar_imu_and_low_state_without_authorizing_motion(monkeypatch):
    points = []
    for step in range(5):
        start_x = 0.5 + step * 0.28
        for sample in range(12):
            x = start_x + sample * 0.02
            for lateral in range(9):
                points.append([x, -0.4 + lateral * 0.1, step * 0.16])
    for sample in range(20):
        x = sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.0])
        x = 1.9 + sample * 0.02
        for lateral in range(9):
            points.append([x, -0.4 + lateral * 0.1, 0.64])

    class _Capture:
        def iter_messages(self, stream):
            if stream == "lidar":
                yield 1_000_000_000, b"lidar-1"
                yield 2_000_000_000, b"lidar-2"
                yield 3_000_000_000, b"lidar-3"
            elif stream == "imu":
                yield 1_000_000_000, b"imu-1"
                yield 2_000_000_000, b"imu-2"
                yield 3_000_000_000, b"imu-3"
            else:
                return

    monkeypatch.setattr(
        stair_survey,
        "decode_pointcloud",
        lambda payload: SimpleNamespace(header_timestamp_ns=int(payload[-1:]), points_xyz=np.asarray(points)),
    )
    monkeypatch.setattr(
        stair_survey,
        "decode_imu",
        lambda payload: SimpleNamespace(header_timestamp_ns=int(payload[-1:]) * 1_000_000_000),
    )
    report = replay_stair_capture(
        _Capture(),
        source_identity={"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "bag-1"},
        lidar_to_gravity=np.eye(4),
        transform_identity={
            "transform_id": "gravity-test-v1",
            "coordinate_frame": "GRAVITY_ALIGNED_BASE",
            "measurement_authority": False,
        },
        low_state_timestamps_ns=(1_000_000_000, 2_000_000_000, 3_000_000_000),
        evidence={
            "top_landing_supported": True,
            "bottom_landing_supported": True,
            "static_scene": True,
            "unobstructed": True,
            "non_hollow": True,
            "straight_flight": True,
        },
    )

    assert report["state"] == "STAIR_QUALIFIED"
    assert report["replay"]["lidar_frames_considered"] == 3
    assert report["replay"]["imu_samples"] == 3
    assert report["replay"]["transform_identity"]["transform_id"] == "gravity-test-v1"
    assert report["measurement_authority"] is False
    assert report["locomotion_authorized"] is False


def test_stair_capture_replay_requires_transform_provenance_metadata():
    class _Capture:
        def iter_messages(self, _stream):
            if False:
                yield 0, b"unused"

    try:
        replay_stair_capture(
            _Capture(),
            source_identity={"lidar_topic": "/livox/lidar", "frame_id": "base_link", "capture_id": "missing-transform-id"},
            lidar_to_gravity=np.eye(4),
            transform_identity=None,
        )
    except ValueError as exc:
        assert "transform identity" in str(exc)
    else:
        raise AssertionError("stair replay accepted transform without provenance metadata")
