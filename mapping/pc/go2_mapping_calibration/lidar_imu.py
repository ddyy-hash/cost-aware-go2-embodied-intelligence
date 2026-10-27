"""Strict independent raw LiDAR--IMU calibration evidence contract.

The LIO odometry output is intentionally not accepted here: using a vendor
extrinsic to produce LIO and then using that LIO to validate the same
extrinsic is circular evidence. This module accepts results from a dedicated
raw-sensor solver (for example LI-Init/OA-LICalib) only when three independent
source captures and their immutable solver artifacts are present.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


class LidarImuEvidenceError(ValueError):
    pass


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise LidarImuEvidenceError(f"{field} must be a SHA-256 hex digest")
    return value.lower()


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LidarImuEvidenceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise LidarImuEvidenceError(f"{field} is outside finite bounds")
    return result


def _matrix(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 16:
        raise LidarImuEvidenceError(f"{field} must contain 16 row-major values")
    matrix = np.asarray([_number(item, f"{field}[{index}]") for index, item in enumerate(value)], dtype=np.float64).reshape(4, 4)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise LidarImuEvidenceError(f"{field} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise LidarImuEvidenceError(f"{field} rotation must be proper")
    return matrix


@dataclass(frozen=True)
class LidarImuRun:
    source_capture_sha256: str
    result_artifact_sha256: str
    transform: np.ndarray
    transform_sha256: str
    timing_offset_ms: float
    imu_lidar_translation_repeatability_mm: float
    imu_lidar_rotation_repeatability_deg: float
    timing_uncertainty_ms: float
    solver_name: str
    solver_version: str
    implementation_sha256: str
    uses_vendor_extrinsic: bool

    def __post_init__(self) -> None:
        value = np.asarray(self.transform, dtype=np.float64).reshape(4, 4)
        value.setflags(write=False)
        object.__setattr__(self, "transform", value)


@dataclass(frozen=True)
class LidarImuEvidence:
    robot_serial: str
    lidar_frame_id: str
    imu_frame_id: str
    raw_lidar_topic: str
    raw_imu_topic: str
    runs: tuple[LidarImuRun, ...]
    translation_repeatability_mm: float
    rotation_repeatability_deg: float
    timing_repeatability_ms: float
    blockers: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.blockers

    def manifest_repeatability(self) -> dict[str, float]:
        if not self.accepted:
            raise LidarImuEvidenceError("blocked LiDAR-IMU evidence cannot enter a manifest")
        return {
            "lidar_imu_translation_mm": self.translation_repeatability_mm,
            "lidar_imu_rotation_deg": self.rotation_repeatability_deg,
        }

    def manifest_timing(self) -> float:
        if not self.accepted:
            raise LidarImuEvidenceError("blocked LiDAR-IMU evidence cannot enter a manifest")
        return self.timing_repeatability_ms

    def manifest_evidence(self) -> dict[str, Any]:
        if not self.accepted:
            raise LidarImuEvidenceError("blocked LiDAR-IMU evidence cannot enter a manifest")
        first = self.runs[0]
        return {
            "source_capture_sha256": [run.source_capture_sha256 for run in self.runs],
            "result_artifact_sha256": [run.result_artifact_sha256 for run in self.runs],
            "transform_sha256": [run.transform_sha256 for run in self.runs],
            "raw_lidar_topic": self.raw_lidar_topic,
            "raw_imu_topic": self.raw_imu_topic,
            "lidar_frame_id": self.lidar_frame_id,
            "imu_frame_id": self.imu_frame_id,
            "solver": {
                "name": first.solver_name,
                "version": first.solver_version,
                "implementation_sha256": first.implementation_sha256,
            },
            "uses_vendor_extrinsic": False,
        }


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3] @ right[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def parse_lidar_imu_evidence(value: Mapping[str, Any], *, expected_robot_serial: str | None = None) -> LidarImuEvidence:
    if not isinstance(value, Mapping):
        raise LidarImuEvidenceError("LiDAR-IMU evidence must be an object")
    expected_top = {"schema_version", "status", "robot_serial", "lidar_frame_id", "imu_frame_id", "provenance", "solver", "runs"}
    if set(value) != expected_top:
        raise LidarImuEvidenceError("LiDAR-IMU evidence has unknown or missing fields")
    if value["schema_version"] != 1 or value["status"] != "independent_raw":
        raise LidarImuEvidenceError("LiDAR-IMU evidence schema/status is unsupported")
    robot_serial = str(value["robot_serial"]).strip()
    if not robot_serial or (expected_robot_serial is not None and robot_serial != expected_robot_serial):
        raise LidarImuEvidenceError("LiDAR-IMU robot serial mismatch")
    lidar_frame_id, imu_frame_id = str(value["lidar_frame_id"]).strip(), str(value["imu_frame_id"]).strip()
    if not lidar_frame_id or not imu_frame_id or lidar_frame_id == imu_frame_id:
        raise LidarImuEvidenceError("LiDAR and IMU frame IDs must be distinct")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {"raw_lidar_topic", "raw_imu_topic", "uses_vendor_extrinsic"}:
        raise LidarImuEvidenceError("raw sensor provenance is incomplete")
    if not all(isinstance(provenance[key], str) and provenance[key].startswith("/") for key in ("raw_lidar_topic", "raw_imu_topic")):
        raise LidarImuEvidenceError("raw sensor topics must be absolute ROS names")
    if provenance["uses_vendor_extrinsic"] is not False:
        raise LidarImuEvidenceError("vendor/provisional extrinsics cannot be independent evidence")
    solver = value["solver"]
    if not isinstance(solver, Mapping) or set(solver) != {"name", "version", "implementation_sha256"}:
        raise LidarImuEvidenceError("independent solver provenance is incomplete")
    solver_name, solver_version = str(solver["name"]).strip(), str(solver["version"]).strip()
    implementation_sha = _hash(solver["implementation_sha256"], "solver.implementation_sha256")
    if not solver_name or not solver_version:
        raise LidarImuEvidenceError("solver name/version are required")
    raw_runs = value["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 3:
        raise LidarImuEvidenceError("exactly three independent raw LiDAR-IMU runs are required")
    runs: list[LidarImuRun] = []
    for index, raw in enumerate(raw_runs, start=1):
        if not isinstance(raw, Mapping):
            raise LidarImuEvidenceError(f"runs[{index}] must be an object")
        fields = {"source_capture_sha256", "result_artifact_sha256", "transform", "transform_sha256", "timing_offset_ms", "translation_repeatability_mm", "rotation_repeatability_deg", "timing_uncertainty_ms", "accepted"}
        if set(raw) != fields:
            raise LidarImuEvidenceError(f"runs[{index}] has unknown or missing fields")
        if raw["accepted"] is not True:
            raise LidarImuEvidenceError(f"runs[{index}] is not independently accepted")
        transform_payload = raw["transform"]
        matrix = _matrix(transform_payload, f"runs[{index}].transform")
        transform_hash = _hash(raw["transform_sha256"], f"runs[{index}].transform_sha256")
        if transform_hash != canonical_sha256({"row_major": matrix.reshape(-1).tolist()}):
            raise LidarImuEvidenceError(f"runs[{index}].transform_sha256 is not bound to its matrix")
        runs.append(LidarImuRun(
            source_capture_sha256=_hash(raw["source_capture_sha256"], f"runs[{index}].source_capture_sha256"),
            result_artifact_sha256=_hash(raw["result_artifact_sha256"], f"runs[{index}].result_artifact_sha256"),
            transform=matrix,
            transform_sha256=transform_hash,
            timing_offset_ms=_number(raw["timing_offset_ms"], f"runs[{index}].timing_offset_ms"),
            imu_lidar_translation_repeatability_mm=_number(raw["translation_repeatability_mm"], f"runs[{index}].translation_repeatability_mm", nonnegative=True),
            imu_lidar_rotation_repeatability_deg=_number(raw["rotation_repeatability_deg"], f"runs[{index}].rotation_repeatability_deg", nonnegative=True),
            timing_uncertainty_ms=_number(raw["timing_uncertainty_ms"], f"runs[{index}].timing_uncertainty_ms", nonnegative=True),
            solver_name=solver_name,
            solver_version=solver_version,
            implementation_sha256=implementation_sha,
            uses_vendor_extrinsic=False,
        ))
    if len({run.source_capture_sha256 for run in runs}) != 3 or len({run.result_artifact_sha256 for run in runs}) != 3:
        raise LidarImuEvidenceError("raw runs must use three unique source/result artifacts")
    translation_spread = max(float(np.linalg.norm(left.transform[:3, 3] - right.transform[:3, 3])) * 1000.0 for index, left in enumerate(runs) for right in runs[index + 1:])
    rotation_spread = max(_rotation_distance_deg(left.transform, right.transform) for index, left in enumerate(runs) for right in runs[index + 1:])
    timing_spread = max(abs(left.timing_offset_ms - right.timing_offset_ms) for index, left in enumerate(runs) for right in runs[index + 1:])
    blockers: list[str] = []
    if translation_spread > 3.0:
        blockers.append("LIDAR_IMU_TRANSLATION_REPEATABILITY_EXCEEDED")
    if rotation_spread > 0.15:
        blockers.append("LIDAR_IMU_ROTATION_REPEATABILITY_EXCEEDED")
    if timing_spread > 1.0:
        blockers.append("LIDAR_IMU_TIMING_REPEATABILITY_EXCEEDED")
    return LidarImuEvidence(
        robot_serial,
        lidar_frame_id,
        imu_frame_id,
        str(provenance["raw_lidar_topic"]),
        str(provenance["raw_imu_topic"]),
        tuple(runs),
        translation_spread,
        rotation_spread,
        timing_spread,
        tuple(blockers),
    )


def load_lidar_imu_evidence(path: str | Path, *, expected_robot_serial: str | None = None) -> LidarImuEvidence:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LidarImuEvidenceError(f"cannot read LiDAR-IMU evidence: {path}") from exc
    return parse_lidar_imu_evidence(payload, expected_robot_serial=expected_robot_serial)
