"""Builder for the measured Go2 calibration manifest contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mapping.pc.go2_mapping_quality.reconstruction_config import (
    canonical_sha256,
    parse_calibration_manifest,
)
from .target import go2_lab_board_spec


def _hashes(values: Sequence[str], field: str) -> list[str]:
    result = list(values)
    if len(result) != 3 or len(set(result)) != 3:
        raise ValueError(f"{field} must contain three unique artifacts")
    return result


def build_calibration_manifest(
    *,
    robot_serial: str,
    camera_serial: str,
    estimation_hashes: Sequence[str],
    holdout_hash: str,
    solver_version: str,
    solver_implementation_sha256: str,
    t_c_l: Sequence[float],
    t_d_c: Sequence[float],
    t_c_l_sha256: str,
    t_d_c_sha256: str,
    color_camera_info_sha256: str,
    depth_camera_info_sha256: str,
    color_camera_info: Mapping[str, Any],
    depth_camera_info: Mapping[str, Any],
    lidar_frame_id: str,
    color_frame_id: str,
    depth_frame_id: str,
    clock_fits: Mapping[str, Mapping[str, Any]],
    lidar_imu_evidence: Mapping[str, Any],
    repeatability: Mapping[str, Any],
    timing: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate a production calibration manifest.

    ``parse_calibration_manifest`` applies the runtime quality gates before the
    payload is returned.
    """
    estimates = _hashes(estimation_hashes, "estimation_hashes")
    identity_l = list(t_c_l)
    identity_d = list(t_d_c)
    board = go2_lab_board_spec()
    target_geometry = {
        "family": "charuco",
        "dictionary": board.dictionary,
        "rows": board.squares_y,
        "columns": board.squares_x,
        "marker_length_m": board.marker_length_m,
        "square_length_m": board.square_length_m,
        "pattern_width_m": board.pattern_width_m,
        "pattern_height_m": board.pattern_height_m,
        "source_template_pdf_sha256": board.source_template_sha256,
        "source_pdf_nominal_marker_length_m": board.template_marker_length_m,
        "source_pdf_nominal_square_length_m": board.template_square_length_m,
        "dimension_authority": "physical_print",
    }
    target = dict(target_geometry)
    target["geometry_sha256"] = canonical_sha256(target_geometry)
    frame = {
        "lidar_frame_id": lidar_frame_id,
        "color_optical_frame_id": color_frame_id,
        "depth_optical_frame_id": depth_frame_id,
        "device_serial": camera_serial,
        "color_camera_info_sha256": color_camera_info_sha256,
        "color_camera_info": dict(color_camera_info),
        "depth_camera_info_sha256": depth_camera_info_sha256,
        "depth_camera_info": dict(depth_camera_info),
        "t_c_l": {"sha256": t_c_l_sha256, "row_major": identity_l},
        "t_d_c": {"sha256": t_d_c_sha256, "row_major": identity_d, "device_serial": camera_serial},
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "measured",
        "calibration_id": canonical_sha256({"robot": robot_serial, "camera": camera_serial, "estimation": estimates, "holdout": holdout_hash})[:24],
        "robot": {"model": "go2_edu", "serial": robot_serial},
        "capture_sets": {"estimation_artifact_sha256": estimates, "holdout_artifact_sha256": holdout_hash},
        "target": target,
        "solver": {"version": solver_version, "implementation_sha256": solver_implementation_sha256},
        "frame_chain": frame,
        "clock_fits": {key: dict(value) for key, value in clock_fits.items()},
        "lidar_imu_evidence": dict(lidar_imu_evidence),
        "repeatability": dict(repeatability),
        "timing": dict(timing),
        "holdout_evidence": dict(holdout),
    }
    parsed = parse_calibration_manifest(payload)
    if parsed.status != "measured":
        raise ValueError("calibration manifest did not pass measured quality gates")
    return payload
