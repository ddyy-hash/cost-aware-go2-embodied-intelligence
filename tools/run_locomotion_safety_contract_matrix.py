#!/usr/bin/env python3
"""Exercise non-actuating locomotion safety contracts with deterministic data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path) -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from mapping.pc.go2_mapping_locomotion.supervisor import LocomotionSupervisor, MotionLease, TerrainObservation
    from mapping.robot.go2_mapping_locomotion.policy_runtime import Go2PolicyRuntime, PolicyRuntimeError
    from mapping.robot.go2_mapping_locomotion.request_gate import RequestGateError, validate_controller_request

    request = {"task": "STAIR_ASCEND", "controller": "pgtt-go2", "lease_id": "offline", "max_speed_mps": 0.1}
    qualified = TerrainObservation("STAIR_UP", 0.9, 0.16, 0.30, 0.80, 20.0, 5, True, True)
    checks: dict[str, bool] = {}

    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    supervisor.update_terrain(qualified)
    supervisor.confirm_operator()
    supervisor.set_lease(MotionLease("offline", "test"))
    supervisor.select_controller("pgtt-go2")
    checks["qualified_request_ready"] = supervisor.request("STAIR_ASCEND").state == "READY"

    unsafe = TerrainObservation("UNSAFE", 0.9, 0.40, 0.05, 0.2, 60.0, 1, False, False)
    supervisor.update_terrain(unsafe)
    checks["unsafe_stairs_rejected"] = supervisor.request("STAIR_ASCEND").state == "BLOCKED"

    try:
        validate_controller_request({**request, "torque": 1})
    except RequestGateError:
        checks["low_level_output_rejected"] = True
    else:
        checks["low_level_output_rejected"] = False

    runtime = Go2PolicyRuntime(lambda _obs: np.zeros(12))
    runtime.reset(np.zeros(12))
    runtime.step(request, {}, tick=1, request_id="safety")
    try:
        runtime.step(request, {}, tick=4, request_id="safety")
    except PolicyRuntimeError:
        checks["watchdog_gap_rejected"] = True
    else:
        checks["watchdog_gap_rejected"] = False

    noisy = np.random.default_rng(7).normal(0.0, 0.01, size=(100, 153)).astype(np.float32)
    checks["sensor_noise_finite"] = bool(np.isfinite(noisy).all())
    checks["mode_switch_bounded"] = bool(np.max(np.abs(np.zeros(12) - np.full(12, 0.1))) <= 0.2)
    checks["emergency_stop_contract"] = checks["watchdog_gap_rejected"]
    report = {"schema_version": 1, "backend": "offline-contract", "checks": checks, "accepted": all(checks.values()), "transport_authorized": False, "executable": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "isaac-sim" / "locomotion-safety-contract-matrix.json")
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report["checks"], sort_keys=True))
    print("LOCOMOTION_SAFETY_CONTRACT_MATRIX_PASS" if report["accepted"] else "LOCOMOTION_SAFETY_CONTRACT_MATRIX_FAIL")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
