#!/usr/bin/env python3
"""Run the offline PGTT acceptance matrix and fail below declared thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tools" / "run_isaac_go2_pgtt_smoke.py"
DEFAULT_USD = ROOT / "output" / "isaac-sim" / "go2_driven" / "go2.usd" / "go2" / "go2.usda"
DEFAULT_URDF_PARENT_ROOT_HEIGHT = 0.20
DEFAULT_STAIR_START_X = -0.80
DEFAULT_COUNTS = {"flat_stand": 1, "flat_forward": 20, "low_steps": 20, "up": 50, "down": 50}
THRESHOLDS = {"flat_stand": 1.0, "flat_forward": 0.8, "low_steps": 0.9, "up": 0.85, "down": 0.8}


def run_matrix(*, episodes: int | None, steps: int, output: Path, usd: Path = DEFAULT_USD, robot_root: str = "/Go2", root_height: float = DEFAULT_URDF_PARENT_ROOT_HEIGHT, stair_start_x: float = DEFAULT_STAIR_START_X, start_yaw: float = 0.0, feet_only_collision: bool = False, physical_terrain: bool = True, action_smoothing: float = 1.0, policy_backend: str = "pgtt", warmup_ticks: int = 10, heightmap_mode: str = "analytic", smoke: Path = SMOKE) -> dict[str, object]:
    if policy_backend != "pgtt":
        raise ValueError("Genesis backend is disabled; only pgtt is supported")
    counts = {name: episodes if episodes is not None else count for name, count in DEFAULT_COUNTS.items()}
    records: list[dict[str, object]] = []
    for name, count in counts.items():
        terrain = "flat" if name.startswith("flat") else "up" if name == "up" else "down"
        for index in range(count):
            episode_output = output.parent / "episodes" / f"{name}-{index:03d}.json"
            command = [sys.executable, str(smoke), "--usd", str(usd), "--robot-root", robot_root, "--root-height", str(root_height), "--start-x", str(stair_start_x if terrain != "flat" else 0.0), "--start-yaw", str(start_yaw), "--action-smoothing", str(action_smoothing), "--policy-backend", str(policy_backend), "--warmup-ticks", str(warmup_ticks), "--seed", str(index), "--heightmap-mode", heightmap_mode, "--terrain", terrain, "--steps", str(steps), "--output", str(episode_output)]
            if not physical_terrain:
                command.append("--heightmap-only")
            if feet_only_collision:
                command.append("--feet-only-collision")
            if name == "flat_forward":
                command.extend(["--vx", "0.6", "--vy", "0.0"])
            if name == "flat_stand":
                command.append("--zero-command")
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if episode_output.is_file():
                record = json.loads(episode_output.read_text(encoding="utf-8"))
            else:
                record = {"stable": False, "traversal_success": False, "failure_reason": "no_result_artifact"}
            record.update({"scenario": name, "episode": index, "process_returncode": completed.returncode})
            records.append(record)
    summary: dict[str, object] = {"schema_version": 1, "episodes": records, "scenarios": {}}
    scenarios = summary["scenarios"]
    assert isinstance(scenarios, dict)
    matrix_pass = True
    for name, count in counts.items():
        selected = [record for record in records if record["scenario"] == name]
        successes = 0
        for record in selected:
            if not record.get("stable"):
                continue
            if name == "flat_stand":
                successes += 1
            elif name == "flat_forward":
                successes += int(float(record.get("local_velocity_x_mean_mps", -999.0)) >= 0.30 and float(record.get("local_velocity_x_p05_mps", -999.0)) > -0.20)
            else:
                successes += int(bool(record.get("traversal_success")))
        rate = successes / count if count else 0.0
        accepted = rate >= THRESHOLDS[name]
        matrix_pass = matrix_pass and accepted
        scenarios[name] = {"episodes": count, "successes": successes, "success_rate": rate, "minimum": THRESHOLDS[name], "accepted": accepted}
    summary["accepted"] = matrix_pass
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, help="override every scenario count for a short smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "isaac-sim" / "pgtt-acceptance-matrix.json")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--robot-root", default="/Go2")
    parser.add_argument("--root-height", type=float, default=DEFAULT_URDF_PARENT_ROOT_HEIGHT)
    parser.add_argument("--stair-start-x", type=float, default=DEFAULT_STAIR_START_X)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--feet-only-collision", action="store_true")
    parser.add_argument("--heightmap-only", action="store_true", help="omit synthetic physical stairs")
    parser.add_argument("--action-smoothing", type=float, default=1.0)
    parser.add_argument("--policy-backend", choices=("pgtt",), default="pgtt")
    parser.add_argument("--warmup-ticks", type=int, default=10)
    parser.add_argument("--heightmap-mode", choices=("analytic", "raycast"), default="analytic")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    report = run_matrix(episodes=args.episodes, steps=args.steps, output=args.output, usd=args.usd, robot_root=args.robot_root, root_height=args.root_height, stair_start_x=args.stair_start_x, start_yaw=args.start_yaw, feet_only_collision=args.feet_only_collision, physical_terrain=not args.heightmap_only, action_smoothing=args.action_smoothing, policy_backend=args.policy_backend, warmup_ticks=args.warmup_ticks, heightmap_mode=args.heightmap_mode)
    print(json.dumps(report["scenarios"], sort_keys=True))
    print("ISAAC_PGTT_ACCEPTANCE_MATRIX_PASS" if report["accepted"] else "ISAAC_PGTT_ACCEPTANCE_MATRIX_FAIL")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
