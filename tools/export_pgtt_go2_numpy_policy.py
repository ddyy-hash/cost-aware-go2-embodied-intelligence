"""Export the verified PGTT Go2 checkpoint to a portable NumPy weight bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/pgtt-go2/policy_go2_pgtt_level03_run0"))
    parser.add_argument("--output", type=Path, default=Path("output/robot-release/pgtt-go2/policy_level03_run0.npz"))
    args = parser.parse_args()
    checkpoint = ROOT / args.checkpoint
    output = ROOT / args.output
    with checkpoint.open("rb") as source:
        payload = pickle.load(source)
    statistics, network, _ = payload
    parameters = network["params"]
    arrays = {
        "mean": np.array(statistics.mean["state"], dtype=np.float32, copy=True),
        "std": np.array(statistics.std["state"], dtype=np.float32, copy=True),
    }
    for name in ("hidden_0", "hidden_1", "hidden_2", "hidden_3"):
        arrays[f"{name}_kernel"] = np.array(parameters[name]["kernel"], dtype=np.float32, copy=True)
        arrays[f"{name}_bias"] = np.array(parameters[name]["bias"], dtype=np.float32, copy=True)
    if arrays["mean"].shape != (153,) or arrays["hidden_3_kernel"].shape != (128, 24):
        raise ValueError("unexpected PGTT Go2 checkpoint shape")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("PGTT checkpoint contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "asset_type": "pgtt_go2_numpy_policy",
        "source_checkpoint": str(args.checkpoint).replace("\\", "/"),
        "input_size": 153,
        "action_size": 12,
        "heightmap_shape": [11, 9],
        "control_frequency_hz": 50,
        "action_scale": 0.5,
        "path": str(args.output).replace("\\", "/"),
        "sha256": digest,
        "executable": False,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": digest, "input_size": 153, "action_size": 12}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
