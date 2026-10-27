"""Read-only stair survey and qualification contracts.

This module is deliberately a geometry/evidence qualifier, not a locomotion
controller. It can reject a profile or describe a conservative survey result,
but it never emits a velocity, joint target, or motion lease.
"""
from __future__ import annotations

from bisect import bisect_left
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from mapping.pc.go2_mapping_calibration.cdr import decode_imu, decode_pointcloud


_MIN_STEPS = 3
_MIN_RISER_M, _MAX_RISER_M = 0.05, 0.22
_MIN_TREAD_M, _MAX_TREAD_M = 0.15, 0.55
_MIN_WIDTH_M, _MAX_WIDTH_M = 0.45, 1.50
_MIN_PITCH_DEG, _MAX_PITCH_DEG = 5.0, 40.0
_MIN_GEOMETRY_CONFIDENCE = 0.80
_MAX_RISER_SPREAD_M = 0.03
_MAX_TREAD_SPREAD_M = 0.05
_MAX_WIDTH_SPREAD_M = 0.10
_MAX_PITCH_SPREAD_DEG = 5.0


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _normalise_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise ValueError("stair profile must be an object")
    step_count = profile.get("step_count")
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise ValueError("step_count must be an integer")
    result = {
        "step_count": step_count,
        "riser_height_m": _number(profile.get("riser_height_m"), "riser_height_m"),
        "tread_depth_m": _number(profile.get("tread_depth_m"), "tread_depth_m"),
        "width_m": _number(profile.get("width_m"), "width_m"),
        "pitch_deg": _number(profile.get("pitch_deg"), "pitch_deg"),
        "geometry_confidence": _number(profile.get("geometry_confidence"), "geometry_confidence"),
        "top_landing_supported": _boolean(profile.get("top_landing_supported"), "top_landing_supported"),
        "bottom_landing_supported": _boolean(profile.get("bottom_landing_supported"), "bottom_landing_supported"),
        "static_scene": _boolean(profile.get("static_scene"), "static_scene"),
        "unobstructed": _boolean(profile.get("unobstructed"), "unobstructed"),
        "non_hollow": _boolean(profile.get("non_hollow"), "non_hollow"),
        "straight_flight": _boolean(profile.get("straight_flight"), "straight_flight"),
    }
    return result


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def qualify_stair_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the deterministic, non-actuating stair geometry envelope."""

    try:
        normalised = _normalise_profile(profile)
    except ValueError as exc:
        return {
            "schema_version": 1,
            "state": "STAIR_REJECTED",
            "profile": None,
            "blocking_reasons": ["PROFILE_INVALID", str(exc)],
            "measurement_authority": False,
            "locomotion_authorized": False,
        }

    reasons: list[str] = []
    if normalised["step_count"] < _MIN_STEPS:
        _append_reason(reasons, "INSUFFICIENT_STAIR_STEPS")
    if not _MIN_RISER_M <= normalised["riser_height_m"] <= _MAX_RISER_M:
        _append_reason(reasons, "RISER_OUTSIDE_TESTED_ENVELOPE")
    if not _MIN_TREAD_M <= normalised["tread_depth_m"] <= _MAX_TREAD_M:
        _append_reason(reasons, "TREAD_OUTSIDE_TESTED_ENVELOPE")
    if not _MIN_WIDTH_M <= normalised["width_m"] <= _MAX_WIDTH_M:
        _append_reason(reasons, "STAIR_WIDTH_OUTSIDE_TESTED_ENVELOPE")
    if not _MIN_PITCH_DEG <= normalised["pitch_deg"] <= _MAX_PITCH_DEG:
        _append_reason(reasons, "STAIR_PITCH_OUTSIDE_TESTED_ENVELOPE")
    if normalised["geometry_confidence"] < _MIN_GEOMETRY_CONFIDENCE:
        _append_reason(reasons, "INSUFFICIENT_GEOMETRY_CONFIDENCE")
    if not normalised["top_landing_supported"]:
        _append_reason(reasons, "TOP_LANDING_UNSUPPORTED")
    if not normalised["bottom_landing_supported"]:
        _append_reason(reasons, "BOTTOM_LANDING_UNSUPPORTED")
    if not normalised["static_scene"]:
        _append_reason(reasons, "STAIR_SCENE_NOT_STATIC")
    if not normalised["unobstructed"]:
        _append_reason(reasons, "STAIR_OBSTRUCTED_OR_UNMODELED")
    if not normalised["non_hollow"]:
        _append_reason(reasons, "HOLLOW_STAIR_UNACCEPTED")
    if not normalised["straight_flight"]:
        _append_reason(reasons, "NON_STRAIGHT_FLIGHT_UNACCEPTED")
    return {
        "schema_version": 1,
        "state": "STAIR_QUALIFIED" if not reasons else "STAIR_REJECTED",
        "profile": normalised,
        "blocking_reasons": reasons,
        "measurement_authority": False,
        "locomotion_authorized": False,
        "control_owner": "NONE",
        "motion_lease_required": True,
    }


def extract_stair_profile_from_points(
    points_xyz: Sequence[Sequence[Any]],
    *,
    resolution_m: float = 0.02,
    min_points_per_bin: int = 4,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a conservative stair profile from gravity-aligned local points.

    Points use ``x`` as the survey direction, ``y`` as lateral distance, and
    ``z`` as gravity-aligned height. The extractor estimates repeated positive
    height transitions only; it is intentionally unable to authorize motion.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("stair point cloud must be a non-empty (N, 3) array")
    if not np.isfinite(points).all():
        raise ValueError("stair point cloud must contain only finite values")
    resolution = _number(resolution_m, "stair point resolution")
    if resolution <= 0.0:
        raise ValueError("stair point resolution must be positive")
    if isinstance(min_points_per_bin, bool) or not isinstance(min_points_per_bin, int) or min_points_per_bin < 2:
        raise ValueError("stair min_points_per_bin must be at least two")

    # Keep the estimator local and reject points that are unlikely to describe
    # the traversable flight rather than allowing distant walls to dominate.
    local = points[
        (points[:, 0] >= -0.2)
        & (points[:, 0] <= 6.0)
        & (np.abs(points[:, 1]) <= 2.0)
        & (points[:, 2] >= -0.5)
        & (points[:, 2] <= 2.0)
    ]
    if len(local) < min_points_per_bin * 3:
        raise ValueError("stair point cloud has insufficient local support")

    minimum_x = float(np.min(local[:, 0]))
    bins = np.floor((local[:, 0] - minimum_x) / resolution).astype(np.int64)
    columns: list[tuple[int, float, int]] = []
    for bin_index in sorted(set(int(value) for value in bins)):
        values = local[bins == bin_index, 2]
        if len(values) >= min_points_per_bin:
            columns.append((bin_index, float(np.median(values)), int(len(values))))
    if len(columns) < 3:
        raise ValueError("stair point cloud has insufficient supported x bins")

    transitions: list[tuple[int, float]] = []
    for left, right in zip(columns, columns[1:]):
        bin_gap = right[0] - left[0]
        if bin_gap > 3:
            continue
        delta = right[1] - left[1]
        if 0.04 <= delta <= 0.30:
            transitions.append((right[0], delta))
    if not transitions:
        raise ValueError("stair point cloud has no repeated positive height transitions")

    # Collapse a single riser observed in adjacent bins into one transition.
    transition_groups: list[list[tuple[int, float]]] = []
    for transition in transitions:
        if not transition_groups or transition[0] - transition_groups[-1][-1][0] > 3:
            transition_groups.append([transition])
        else:
            transition_groups[-1].append(transition)
    risers = [float(np.median([item[1] for item in group])) for group in transition_groups]
    transition_bins = [int(round(np.median([item[0] for item in group]))) for group in transition_groups]
    if len(risers) < 2:
        raise ValueError("stair point cloud has fewer than two repeated risers")

    tread_distances = [
        (right - left) * resolution
        for left, right in zip(transition_bins, transition_bins[1:])
        if right > left
    ]
    if not tread_distances:
        raise ValueError("stair point cloud cannot estimate tread depth")
    tread_depth = float(np.median(tread_distances))
    riser_height = float(np.median(risers))
    width = float(np.percentile(local[:, 1], 95) - np.percentile(local[:, 1], 5))
    pitch = math.degrees(math.atan2(riser_height, tread_depth))
    first_bin = columns[0][0]
    last_bin = columns[-1][0]
    bottom_support = (transition_bins[0] - first_bin) * resolution >= 0.25
    top_support = (last_bin - transition_bins[-1]) * resolution >= 0.25
    density = min(1.0, len(local) / max(1.0, len(columns) * min_points_per_bin * 4.0))
    confidence = min(0.99, 0.80 + 0.18 * density)
    supplied = dict(evidence or {})
    boolean_fields = (
        "static_scene",
        "unobstructed",
        "non_hollow",
        "straight_flight",
    )
    profile = {
        "step_count": len(risers) + 1,
        "riser_height_m": riser_height,
        "tread_depth_m": tread_depth,
        "width_m": width,
        "pitch_deg": pitch,
        "geometry_confidence": confidence,
        "top_landing_supported": supplied.get("top_landing_supported") is True and top_support,
        "bottom_landing_supported": supplied.get("bottom_landing_supported") is True and bottom_support,
        **{field: supplied.get(field) is True for field in boolean_fields},
    }
    return profile


def _transform_stair_points(points_xyz: Sequence[Sequence[Any]], lidar_to_gravity: Any) -> np.ndarray:
    matrix = np.asarray(lidar_to_gravity, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("stair LiDAR-to-gravity transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError("stair LiDAR-to-gravity transform must be rigid")
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError("stair LiDAR-to-gravity transform must be a proper rigid transform")
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("stair LiDAR points must be a finite non-empty (N, 3) array")
    return points @ rotation.T + matrix[:3, 3]


def _nearest_timestamp(timestamp_ns: int, timestamps: Sequence[int]) -> int | None:
    if not timestamps:
        return None
    index = bisect_left(timestamps, timestamp_ns)
    candidates = []
    if index < len(timestamps):
        candidates.append(timestamps[index])
    if index:
        candidates.append(timestamps[index - 1])
    return min(candidates, key=lambda value: abs(value - timestamp_ns))


def replay_stair_capture(
    capture: Any,
    *,
    source_identity: Mapping[str, Any] | None,
    lidar_to_gravity: Any,
    transform_identity: Mapping[str, Any],
    low_state_timestamps_ns: Sequence[Any] = (),
    evidence: Mapping[str, Any] | None = None,
    maximum_sensor_gap_ns: int = 100_000_000,
    max_frames: int = 32,
) -> dict[str, Any]:
    """Replay an existing read-only capture into the stair qualifier.

    ``capture`` is compatible with :class:`CalibrationCapture` and must expose
    only its read-only ``iter_messages`` method. Low-state timestamps are
    supplied by the caller because the standard calibration bag contract does
    not claim that low state was recorded in every capture.
    """

    if not hasattr(capture, "iter_messages"):
        raise ValueError("stair replay capture must expose iter_messages")
    if not isinstance(transform_identity, Mapping):
        raise ValueError("stair transform identity must be an object")
    if not isinstance(transform_identity.get("transform_id"), str) or not transform_identity["transform_id"].strip():
        raise ValueError("stair transform identity requires transform_id")
    if transform_identity.get("coordinate_frame") != "GRAVITY_ALIGNED_BASE":
        raise ValueError("stair transform identity must target GRAVITY_ALIGNED_BASE")
    if transform_identity.get("measurement_authority") is not False:
        raise ValueError("stair transform measurement authority must be false")
    if isinstance(maximum_sensor_gap_ns, bool) or not isinstance(maximum_sensor_gap_ns, int) or maximum_sensor_gap_ns < 0:
        raise ValueError("maximum_sensor_gap_ns must be a non-negative integer")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    try:
        low_states = tuple(sorted({int(value) for value in low_state_timestamps_ns}))
    except (TypeError, ValueError) as exc:
        raise ValueError("low-state timestamps must be integers") from exc
    if any(value <= 0 for value in low_states):
        raise ValueError("low-state timestamps must be positive")
    # Decode IMU timestamps first so each LiDAR frame gets an auditable freshness flag.
    imu_timestamps: list[int] = []
    for _, payload in capture.iter_messages("imu"):
        try:
            record = decode_imu(payload)
        except (TypeError, ValueError):
            continue
        if record.header_timestamp_ns > 0:
            imu_timestamps.append(int(record.header_timestamp_ns))
    imu_timestamps = sorted(set(imu_timestamps))

    observations: list[dict[str, Any]] = []
    decoded_lidar = 0
    extraction_failures = 0
    for record_timestamp_ns, payload in capture.iter_messages("lidar"):
        if len(observations) >= max_frames:
            break
        timestamp_ns = int(record_timestamp_ns)
        try:
            cloud = decode_pointcloud(payload)
            transformed = _transform_stair_points(cloud.points_xyz, lidar_to_gravity)
            profile = extract_stair_profile_from_points(transformed, evidence=evidence)
            decoded_lidar += 1
            extraction_error = None
        except (TypeError, ValueError) as exc:
            profile = None
            extraction_error = str(exc)
            extraction_failures += 1
        imu_timestamp = _nearest_timestamp(timestamp_ns, imu_timestamps)
        low_state_timestamp = _nearest_timestamp(timestamp_ns, low_states)
        observations.append(
            {
                "timestamp_ns": timestamp_ns,
                "profile": profile,
                "extraction_error": extraction_error,
                "imu_fresh": imu_timestamp is not None and abs(imu_timestamp - timestamp_ns) <= maximum_sensor_gap_ns,
                "low_state_fresh": low_state_timestamp is not None and abs(low_state_timestamp - timestamp_ns) <= maximum_sensor_gap_ns,
            }
        )
    report = aggregate_stair_observations(observations, source_identity=source_identity)
    report["replay"] = {
        "source_type": "CALIBRATION_CAPTURE",
        "lidar_frames_considered": len(observations),
        "lidar_frames_decoded": decoded_lidar,
        "profile_extraction_failures": extraction_failures,
        "imu_samples": len(imu_timestamps),
        "low_state_samples": len(low_states),
        "maximum_sensor_gap_ns": maximum_sensor_gap_ns,
        "transform_applied": True,
        "transform_identity": dict(transform_identity),
        "read_only": True,
    }
    return report


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def aggregate_stair_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    source_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate repeated local survey profiles into an auditable report."""

    reasons: list[str] = []
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        observations = []
    if len(observations) < 3:
        _append_reason(reasons, "INSUFFICIENT_REPEATED_OBSERVATIONS")
    if not isinstance(source_identity, Mapping) or any(
        not isinstance(source_identity.get(key), str) or not source_identity.get(key).strip()
        for key in ("lidar_topic", "frame_id", "capture_id")
    ):
        _append_reason(reasons, "SOURCE_IDENTITY_MISSING")

    profiles: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            _append_reason(reasons, "OBSERVATION_INVALID")
            continue
        if observation.get("extraction_error"):
            _append_reason(reasons, "POINT_CLOUD_PROFILE_EXTRACTION_FAILED")
        timestamp = observation.get("timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            _append_reason(reasons, "OBSERVATION_TIMESTAMP_INVALID")
        else:
            timestamps.append(timestamp)
        try:
            profile_report = qualify_stair_profile(observation.get("profile"))
        except (TypeError, ValueError):
            profile_report = {
                "state": "STAIR_REJECTED",
                "profile": None,
                "blocking_reasons": ["PROFILE_INVALID"],
            }
        if profile_report.get("state") != "STAIR_QUALIFIED":
            _append_reason(reasons, "STAIR_PROFILE_REJECTED")
            for reason in profile_report.get("blocking_reasons") or []:
                _append_reason(reasons, str(reason))
        else:
            profiles.append(profile_report["profile"])
        if observation.get("imu_fresh") is not True:
            _append_reason(reasons, "IMU_NOT_FRESH")
        if observation.get("low_state_fresh") is not True:
            _append_reason(reasons, "LOW_STATE_NOT_FRESH")

    aggregate_profile: dict[str, Any] | None = None
    spreads: dict[str, float] = {}
    if profiles:
        numeric_fields = ("riser_height_m", "tread_depth_m", "width_m", "pitch_deg", "geometry_confidence")
        aggregate_profile = dict(profiles[0])
        for field in numeric_fields:
            values = [float(profile[field]) for profile in profiles]
            aggregate_profile[field] = round(_median(values), 6)
            spreads[field] = round(_spread(values), 6)
        aggregate_profile["step_count"] = min(int(profile["step_count"]) for profile in profiles)
        for field in ("top_landing_supported", "bottom_landing_supported", "static_scene", "unobstructed", "non_hollow", "straight_flight"):
            aggregate_profile[field] = all(profile[field] for profile in profiles)
        if spreads.get("riser_height_m", 0.0) > _MAX_RISER_SPREAD_M:
            _append_reason(reasons, "STAIR_GEOMETRY_UNSTABLE")
        if spreads.get("tread_depth_m", 0.0) > _MAX_TREAD_SPREAD_M:
            _append_reason(reasons, "STAIR_GEOMETRY_UNSTABLE")
        if spreads.get("width_m", 0.0) > _MAX_WIDTH_SPREAD_M:
            _append_reason(reasons, "STAIR_GEOMETRY_UNSTABLE")
        if spreads.get("pitch_deg", 0.0) > _MAX_PITCH_SPREAD_DEG:
            _append_reason(reasons, "STAIR_GEOMETRY_UNSTABLE")
        profile_report = qualify_stair_profile(aggregate_profile)
        reasons.extend(reason for reason in profile_report["blocking_reasons"] if reason not in reasons)
    else:
        _append_reason(reasons, "NO_QUALIFIED_PROFILE")

    if timestamps and timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        _append_reason(reasons, "OBSERVATION_TIMESTAMPS_NOT_MONOTONIC")
    if not timestamps and observations:
        _append_reason(reasons, "OBSERVATION_TIMESTAMP_INVALID")
    state = "STAIR_QUALIFIED" if aggregate_profile is not None and not reasons else "STAIR_REJECTED"
    return {
        "schema_version": 1,
        "state": state,
        "observation_count": len(observations),
        "qualified_profile_count": len(profiles),
        "profile": aggregate_profile,
        "profile_spreads": spreads,
        "timestamp_start_ns": min(timestamps) if timestamps else None,
        "timestamp_end_ns": max(timestamps) if timestamps else None,
        "source_identity": dict(source_identity) if isinstance(source_identity, Mapping) else None,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "provenance": "OBSERVED_LIDAR_IMU_LOWSTATE",
        "measurement_authority": False,
        "locomotion_authorized": False,
        "control_owner": "NONE",
        "motion_lease_required": True,
        "limitations": [
            "READ_ONLY_GEOMETRY_QUALIFIER",
            "NO_FOOTHOLD_PLANNER",
            "NO_BALANCE_CONTROLLER",
            "NO_LOCOMOTION_COMMAND",
            "ASCENT_AND_DESCENT_REQUIRE_SEPARATE_ACCEPTANCE",
        ],
    }


class StairSurveyStore:
    """Persist only the bounded qualification report, never raw sensor points."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, report: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(report, Mapping):
            raise ValueError("stair survey report must be an object")
        normalized = dict(report)
        if normalized.get("schema_version") != 1:
            raise ValueError("stair survey report schema_version must be 1")
        if normalized.get("state") not in {"STAIR_QUALIFIED", "STAIR_REJECTED", "NO_SURVEY"}:
            raise ValueError("stair survey report state is unsupported")
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if "points_xyz" in encoded:
            raise ValueError("stair survey report must not contain raw point cloud data")
        if normalized.get("measurement_authority") is not False:
            raise ValueError("stair survey report measurement authority must be false")
        if normalized.get("locomotion_authorized") is not False:
            raise ValueError("stair survey report locomotion authorization must be false")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".partial", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(encoded + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return normalized

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != 1 or payload.get("state") not in {
            "STAIR_QUALIFIED",
            "STAIR_REJECTED",
            "NO_SURVEY",
        }:
            return None
        if payload.get("measurement_authority") is not False or payload.get("locomotion_authorized") is not False:
            return None
        if "points_xyz" in json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True):
            return None
        return payload


__all__ = [
    "StairSurveyStore",
    "aggregate_stair_observations",
    "extract_stair_profile_from_points",
    "qualify_stair_profile",
    "replay_stair_capture",
]
