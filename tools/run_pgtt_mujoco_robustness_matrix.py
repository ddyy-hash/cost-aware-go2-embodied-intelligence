#!/usr/bin/env python3
"""Run deterministic MuJoCo robustness scenarios for the PGTT policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools" / "run_pgtt_mujoco_reference.py"
XML = ROOT / "third_party" / "phase_guided_terrain_traversal" / "go2" / "xmls" / "terrain_test_mjx.xml"
SCENARIOS = {
    "friction": {"friction_scale": (0.7, 1.0, 1.3), "minimum": 0.8},
    "mass": {"mass_scale": (0.9, 1.0, 1.1), "minimum": 0.8},
    "com": {"com_offset_x": (-0.02, 0.0, 0.02), "minimum": 0.8},
    "sensor_noise": {"sensor_noise_std": (0.002, 0.005, 0.01), "minimum": 0.8},
    "latency": {"action_delay_ticks": (1, 2), "minimum": 0.8},
}


def run_matrix(*, episodes: int, steps: int, output: Path) -> dict[str, object]:
    records = []
    for scenario, spec in SCENARIOS.items():
        for variant, value in enumerate(spec[next(key for key in spec if key != "minimum")]):
            for episode in range(episodes):
                path = output.parent / "mujoco-robustness-episodes" / f"{scenario}-{variant}-{episode}.json"
                command = [sys.executable, str(REFERENCE), "--xml", str(XML), "--direction", "forward", "--seed", str(variant * 100 + episode), "--steps", str(steps), "--output", str(path)]
                key = next(key for key in spec if key != "minimum")
                command.extend([f"--{key.replace('_', '-')}", str(value)])
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    record = {"stable": False, "finite": False}
                record.update({"scenario": scenario, "variant": variant, "episode": episode, "process_returncode": completed.returncode})
                records.append(record)
    summary = {}
    accepted = True
    for scenario, spec in SCENARIOS.items():
        key = next(key for key in spec if key != "minimum")
        variants = []
        for variant, value in enumerate(spec[key]):
            selected = [record for record in records if record["scenario"] == scenario and record["variant"] == variant]
            successes = sum(bool(record.get("finite")) and bool(record.get("stable")) for record in selected)
            rate = successes / episodes
            passed = rate >= spec["minimum"]
            accepted = accepted and passed
            variants.append({"value": value, "episodes": episodes, "successes": successes, "success_rate": rate, "minimum": spec["minimum"], "accepted": passed})
        summary[scenario] = {"parameter": key, "variants": variants, "accepted": all(item["accepted"] for item in variants)}
    report = {"schema_version": 1, "backend": "mujoco-official-pgtt", "scenarios": summary, "episodes": records, "accepted": accepted}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "isaac-sim" / "pgtt-mujoco-robustness-matrix.json")
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        parser.error("episodes and steps must be positive")
    report = run_matrix(episodes=args.episodes, steps=args.steps, output=args.output)
    print(json.dumps(report["scenarios"], sort_keys=True))
    print("PGTT_MUJOCO_ROBUSTNESS_MATRIX_PASS" if report["accepted"] else "PGTT_MUJOCO_ROBUSTNESS_MATRIX_FAIL")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
