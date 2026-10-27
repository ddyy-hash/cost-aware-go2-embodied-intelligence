#!/usr/bin/env python3
"""Aggregate offline acceptance artifacts without merging simulator claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "accepted": False, "path": str(path)}
    if not isinstance(value, dict):
        return {"available": False, "accepted": False, "path": str(path)}
    classification = value.get("classification")
    accepted = bool(value.get("accepted", False))
    if isinstance(classification, dict):
        accepted = accepted or classification.get("state") == "ACCEPTED"
    return {"available": True, "accepted": accepted, "path": str(path), "backend": value.get("backend", value.get("simulator", value.get("artifact_type")))}


def build(output: Path) -> dict[str, object]:
    layers = {
        "mujoco_nominal": load(ROOT / "output/isaac-sim/pgtt-mujoco-acceptance-matrix-full.json"),
        "mujoco_robustness": load(ROOT / "output/isaac-sim/pgtt-mujoco-robustness-matrix-com.json"),
        "safety_contract": load(ROOT / "output/isaac-sim/locomotion-safety-contract-matrix.json"),
        "mobilevla_rl": load(ROOT / "output/assets/mobilevla-rl-gpu-audit.json"),
        "isaac_latest": load(ROOT / "output/isaac-sim/pgtt-urdf-raycast-stair-1000.json"),
    }
    report = {"schema_version": 1, "layers": layers, "accepted": all(bool(layer["available"]) and bool(layer["accepted"]) for layer in layers.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/isaac-sim/offline-acceptance-report.json")
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps(report, sort_keys=True))
    print("OFFLINE_ACCEPTANCE_PASS" if report["accepted"] else "OFFLINE_ACCEPTANCE_INCOMPLETE")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
