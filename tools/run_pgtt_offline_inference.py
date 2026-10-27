"""Run the verified Go2 PGTT checkpoint without connecting to a robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mapping.pc.go2_mapping_locomotion.pgtt_adapter import (
    PgttObservationSpec,
    build_observation,
    compose_policy_input,
    load_pgtt_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("mapping/config/pgtt_go2_level03_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("output/assets/pgtt-go2-inference-result.json"))
    args = parser.parse_args()
    root = ROOT
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    checkpoint = root / manifest["policy_file"]
    policy = load_pgtt_policy(checkpoint, manifest["sha256"])
    observation = build_observation(
        heightmap=np.zeros(tuple(manifest["heightmap_shape"]), dtype=np.float32),
        proprioception=np.zeros(24, dtype=np.float32),
        command=[0.0, 0.0, 0.0],
        spec=PgttObservationSpec(tuple(manifest["heightmap_shape"]), 24, manifest["control_frequency_hz"]),
    )
    vector = compose_policy_input(
        observation,
        angular_velocity=[0.0, 0.0, 0.0],
        gravity_orientation=[0.0, 0.0, -1.0],
        phase=[0.0] * 8,
        gait_frequency=2.0,
        last_action=[0.0] * 12,
    )
    action = policy(vector[None, :]).detach().cpu().numpy()[0]
    result = {
        "schema_version": 1,
        "backend": "pgtt-go2",
        "checkpoint": policy.pgtt_metadata,
        "input": {"shape": list(vector.shape), "finite": bool(np.isfinite(vector).all())},
        "action": {"shape": list(action.shape), "finite": bool(np.isfinite(action).all()), "values": action.tolist()},
        "executable": False,
        "transport_authorized": False,
        "robot_connected": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "input_size": 153, "action_size": 12, "executable": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
