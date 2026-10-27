"""SDK2-only PGTT runner using a file snapshot from the ROS heightmap node."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from .numpy_pgtt import NumpyPGTTPolicy
from .readonly_pgtt_runner import compose_go2_observation
from .sdk2_adapter import UnitreeSDK2Go2Adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heightmap-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/pgtt-heightmap.bin"))
    parser.add_argument("--policy", type=Path, default=Path("/home/unitree/go2_mapping/current/mapping/robot/pgtt-go2/policy_level03_run0.npz"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    root = args.policy.parents[3] if args.policy.is_absolute() else Path("/home/unitree/go2_mapping/current")
    expected = "f3fe6655e9fa13e06ecccf13bf1aa13b5e62de50be4539781ac7a360b97fbedd"
    policy = NumpyPGTTPolicy(args.policy, expected)
    adapter = UnitreeSDK2Go2Adapter()
    adapter.initialize_readonly()
    deadline = time.time() + args.duration
    heightmap = None
    actions = []
    inference_ms = []
    default = np.asarray([0, .9, -1.8, 0, .9, -1.8, 0, .9, -1.8, 0, .9, -1.8], np.float32)
    while time.time() < deadline:
        if args.heightmap_file.is_file() and args.heightmap_file.stat().st_size == 11 * 9 * 4:
            candidate = np.fromfile(args.heightmap_file, dtype="<f4").reshape(11, 9)
            if np.isfinite(candidate).all():
                heightmap = candidate
        if adapter.latest_lowstate is not None and heightmap is not None:
            observation = compose_go2_observation(
                lowstate=adapter.latest_lowstate,
                heightmap=heightmap,
                phase=np.zeros(8, np.float32),
                last_action=np.zeros(12, np.float32),
                command=np.zeros(3, np.float32),
                default_position=default,
            )
            started = time.perf_counter()
            actions.append(policy(observation[None, :])[0])
            inference_ms.append((time.perf_counter() - started) * 1000.0)
            if len(actions) >= args.samples:
                break
        time.sleep(0.05)
    if adapter.latest_lowstate is None or heightmap is None or not actions:
        adapter.close()
        raise SystemExit("REAL_PGTT_INPUT_NOT_READY")
    action = actions[-1]
    print("REAL_PGTT_SENSOR_INPUT shape=(153,) finite=true REAL_PGTT_ACTION shape=(12,) finite=%s samples=%d latency_ms_p50=%.3f min=%s max=%s" % (
        bool(all(np.isfinite(item).all() for item in actions)), len(actions),
        float(np.percentile(inference_ms, 50)), float(action.min()), float(action.max())
    ))
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
