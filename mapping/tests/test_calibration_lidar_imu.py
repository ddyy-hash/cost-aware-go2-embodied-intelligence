import copy

import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.lidar_imu import (
    LidarImuEvidenceError,
    parse_lidar_imu_evidence,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


def _transform(x_m: float = 0.0, yaw_deg: float = 0.0) -> list[float]:
    yaw = np.deg2rad(yaw_deg)
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    return [
        cosine, -sine, 0.0, x_m,
        sine, cosine, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _evidence() -> dict:
    runs = []
    for index, (x_m, yaw_deg, timing_ms) in enumerate(
        ((0.0, 0.0, 0.0), (0.001, 0.05, 0.2), (0.002, 0.10, 0.4)),
        start=1,
    ):
        transform = _transform(x_m, yaw_deg)
        runs.append(
            {
                "source_capture_sha256": f"{index}" * 64,
                "result_artifact_sha256": f"{index + 3}" * 64,
                "transform": transform,
                "transform_sha256": canonical_sha256({"row_major": transform}),
                "timing_offset_ms": timing_ms,
                "translation_repeatability_mm": 0.0,
                "rotation_repeatability_deg": 0.0,
                "timing_uncertainty_ms": 0.0,
                "accepted": True,
            }
        )
    return {
        "schema_version": 1,
        "status": "independent_raw",
        "robot_serial": "go2-test",
        "lidar_frame_id": "livox_frame",
        "imu_frame_id": "imu_link",
        "provenance": {
            "raw_lidar_topic": "/livox/lidar",
            "raw_imu_topic": "/livox/imu",
            "uses_vendor_extrinsic": False,
        },
        "solver": {
            "name": "li-init",
            "version": "2024.1",
            "implementation_sha256": "a" * 64,
        },
        "runs": runs,
    }


def test_three_independent_raw_lidar_imu_runs_are_accepted_and_auditable():
    evidence = parse_lidar_imu_evidence(_evidence(), expected_robot_serial="go2-test")

    assert evidence.accepted
    assert evidence.translation_repeatability_mm == pytest.approx(2.0)
    assert evidence.rotation_repeatability_deg == pytest.approx(0.1)
    assert evidence.timing_repeatability_ms == pytest.approx(0.4)
    manifest = evidence.manifest_evidence()
    assert manifest["source_capture_sha256"] == ["1" * 64, "2" * 64, "3" * 64]
    assert manifest["uses_vendor_extrinsic"] is False
    assert len(manifest["transform_sha256"]) == 3


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["runs"][1].update(source_capture_sha256=value["runs"][0]["source_capture_sha256"]), "unique"),
        (lambda value: value["runs"][1].update(result_artifact_sha256=value["runs"][0]["result_artifact_sha256"]), "unique"),
        (lambda value: value["provenance"].update(uses_vendor_extrinsic=True), "vendor"),
        (lambda value: value["runs"][0].update(transform_sha256="f" * 64), "bound"),
        (lambda value: value.update(robot_serial="another-go2"), "serial"),
        (lambda value: value.update(unexpected=True), "unknown"),
    ],
)
def test_lidar_imu_evidence_rejects_non_independent_or_unbound_inputs(mutator, message):
    payload = copy.deepcopy(_evidence())
    mutator(payload)

    with pytest.raises(LidarImuEvidenceError, match=message):
        parse_lidar_imu_evidence(payload, expected_robot_serial="go2-test")


@pytest.mark.parametrize(
    ("translation_m", "yaw_deg", "timing_ms", "blocker"),
    [
        (0.003001, 0.0, 0.0, "LIDAR_IMU_TRANSLATION_REPEATABILITY_EXCEEDED"),
        (0.0, 0.150001, 0.0, "LIDAR_IMU_ROTATION_REPEATABILITY_EXCEEDED"),
        (0.0, 0.0, 1.000001, "LIDAR_IMU_TIMING_REPEATABILITY_EXCEEDED"),
    ],
)
def test_lidar_imu_repeatability_gates_fail_closed(translation_m, yaw_deg, timing_ms, blocker):
    payload = _evidence()
    transform = _transform(translation_m, yaw_deg)
    payload["runs"][2]["transform"] = transform
    payload["runs"][2]["transform_sha256"] = canonical_sha256({"row_major": transform})
    payload["runs"][2]["timing_offset_ms"] = timing_ms

    evidence = parse_lidar_imu_evidence(payload)

    assert not evidence.accepted
    assert blocker in evidence.blockers
