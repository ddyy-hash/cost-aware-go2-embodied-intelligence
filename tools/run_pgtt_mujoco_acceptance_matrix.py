#!/usr/bin/env python3
"""Run the PGTT acceptance matrix on the official MuJoCo dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools" / "run_pgtt_mujoco_reference.py"
PGTT = ROOT / "third_party" / "phase_guided_terrain_traversal" / "go2" / "xmls"
SCENARIOS = {
    "flat_stand": ("scene_mjx_feetonly.xml", "forward", True, 1.0),
    "flat_forward": ("scene_mjx_feetonly.xml", "forward", False, 0.8),
    "low_steps": ("terrain_scene_mjx.xml", "up", False, 0.9),
    "stair_up": ("terrain_test_mjx.xml", "up", False, 0.85),
    "stair_down": ("terrain_test_mjx_1.xml", "down", False, 0.8),
}
DEFAULT_COUNTS = {"flat_stand": 1, "flat_forward": 20, "low_steps": 20, "stair_up": 50, "stair_down": 50}


def run_matrix(*, episodes: int | None, steps: int, output: Path) -> dict[str, object]:
    counts = {name: episodes if episodes is not None else DEFAULT_COUNTS[name] for name in SCENARIOS}
    records = []
    for scenario, (xml_name, direction, neutral, minimum) in SCENARIOS.items():
        for index in range(counts[scenario]):
            episode_output = output.parent / "mujoco-episodes" / f"{scenario}-{index:03d}.json"
            command = [sys.executable, str(REFERENCE), "--xml", str(PGTT / xml_name), "--direction", direction, "--seed", str(index), "--steps", str(steps if not neutral else max(steps, 3000)), "--output", str(episode_output)]
            if neutral:
                command.append("--neutral")
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            try:
                record = json.loads(episode_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                record = {"finite": False, "stable": False, "traversal_success": False, "failure_reason": "no_result_artifact"}
            record.update({"scenario": scenario, "episode": index, "process_returncode": completed.returncode})
            records.append(record)
    scenarios = {}
    accepted = True
    for scenario, (_, _, neutral, minimum) in SCENARIOS.items():
        selected = [record for record in records if record["scenario"] == scenario]
        successes = sum(bool(record.get("stable")) and (neutral or bool(record.get("traversal_success"))) for record in selected)
        rate = successes / counts[scenario]
        passed = rate >= minimum
        accepted = accepted and passed
        scenarios[scenario] = {"episodes": counts[scenario], "successes": successes, "success_rate": rate, "minimum": minimum, "accepted": passed}
    report = {"schema_version": 1, "backend": "mujoco-official-pgtt", "scenarios": scenarios, "episodes": records, "accepted": accepted}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, help="override every scenario count for a short smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "isaac-sim" / "pgtt-mujoco-acceptance-matrix.json")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    report = run_matrix(episodes=args.episodes, steps=args.steps, output=args.output)
    print(json.dumps(report["scenarios"], sort_keys=True))
    print("PGTT_MUJOCO_ACCEPTANCE_MATRIX_PASS" if report["accepted"] else "PGTT_MUJOCO_ACCEPTANCE_MATRIX_FAIL")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
