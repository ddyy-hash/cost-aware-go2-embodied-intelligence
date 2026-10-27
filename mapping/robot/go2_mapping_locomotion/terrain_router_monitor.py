"""Read-only Go2 remote/heightmap monitor for automatic PGTT terrain routing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .remote_control import decode_remote
from .sdk2_adapter import SDK2AdapterConfig, UnitreeSDK2Go2Adapter
from .terrain_router import StableTerrainRouter, classify_stair_profile


def _status_should_refresh(
    previous_state: tuple | None,
    current_state: tuple,
    *,
    now: float,
    last_written_at: float,
    refresh_period: float,
) -> bool:
    """Keep the executor's route freshness gate alive without log spam."""
    return previous_state != current_state or now - last_written_at >= refresh_period


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heightmap-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/pgtt-heightmap.bin"))
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--period", type=float, default=0.05)
    parser.add_argument("--simulate-forward", type=float, default=None, help="Diagnostic only; bypass remote intent without authorizing transport")
    parser.add_argument("--status-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/terrain-router-status.json"))
    parser.add_argument("--stair-probe-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/stair-probe-profile.bin"))
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.period <= 0:
        parser.error("duration and period must be positive")
    adapter = UnitreeSDK2Go2Adapter(config=SDK2AdapterConfig(network_interface=args.network_interface))
    adapter.initialize_readonly()
    router = StableTerrainRouter()
    deadline = time.monotonic() + args.duration
    last_state = None
    last_written_at = 0.0
    status_refresh_period = 0.10
    emitted = 0
    try:
        while time.monotonic() < deadline:
            lowstate = adapter.latest_lowstate
            if lowstate is None or not args.heightmap_file.is_file():
                time.sleep(args.period)
                continue
            stat = args.heightmap_file.stat()
            heightmap = np.fromfile(args.heightmap_file, dtype="<f4")
            if heightmap.size != 99 or not np.isfinite(heightmap).all():
                time.sleep(args.period)
                continue
            remote = decode_remote(lowstate.wireless_remote)
            intent = float(args.simulate_forward) if args.simulate_forward is not None else remote.ly
            if args.stair_probe_file.is_file() and args.stair_probe_file.stat().st_mtime >= stat.st_mtime:
                probe = np.fromfile(args.stair_probe_file, dtype="<f4")
                route = classify_stair_profile(probe.reshape(21, 9), forward_intent=intent)
            else:
                route = router.update(heightmap.reshape(11, 9), forward_intent=intent)
            state = (route.state, route.policy_level, route.executable, route.blocker, remote.deadman, round(intent, 2))
            now = time.monotonic()
            payload = {
                "state": route.state,
                "task": route.task,
                "policy_level": route.policy_level,
                "executable": route.executable,
                "blocker": route.blocker,
                "confidence": round(route.confidence, 3),
                "relief_m": round(route.relief_m, 4),
                "maximum_step_m": round(route.maximum_step_m, 4),
                "remote": {"deadman": remote.deadman, "arm_chord": remote.arm_chord, "emergency_stop": remote.emergency_stop, "forward": round(remote.ly, 3)},
                "heightmap_age_ms": round((time.time() - stat.st_mtime) * 1000.0, 1),
                "transport_authorized": False,
                "updated_at_unix": time.time(),
            }
            if _status_should_refresh(last_state, state, now=now, last_written_at=last_written_at, refresh_period=status_refresh_period):
                if state != last_state:
                    print(json.dumps(payload, sort_keys=True), flush=True)
                temporary = args.status_file.with_name(args.status_file.name + ".partial")
                args.status_file.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                os.replace(temporary, args.status_file)
                last_written_at = now
                last_state = state
                emitted += 1
            if remote.emergency_stop:
                print(json.dumps({"state": "EMERGENCY_STOP", "transport_authorized": False}), flush=True)
                return 2
            time.sleep(args.period)
    finally:
        adapter.close()
    if emitted == 0:
        raise SystemExit("AUTOMATIC_TERRAIN_ROUTER_INPUT_NOT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
