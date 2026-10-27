"""Explicitly armed PGTT hardware runner for supervised terrain traversal."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from .action_contract import TargetLimiter, pgtt_joint_targets
from .numpy_pgtt import NumpyPGTTPolicy
from .readonly_pgtt_runner import compose_go2_observation, gravity_orientation
from .remote_control import decode_remote
from .sdk2_adapter import PGTT_TO_UNITREE_MOTOR_INDEX
from .terrain_router import StableTerrainRouter


CONFIRMATION = "GO2_PGTT_FIELD_TEST_CONFIRMED"
POLICY_HASHES = {
    "03": "f3fe6655e9fa13e06ecccf13bf1aa13b5e62de50be4539781ac7a360b97fbedd",
    "07": "02940e033a08fec97473cfd36d279d5854a38a3679145162d8c7149c74a4e9e3",
    "10": "749e8a927fb51a3853e78919deb9bae6bac217b205d74231433fc6c676ee6502",
    "13": "45e6a294880f705ad98113479f3830a1107fbc5e74c4adf033b7ea4a17aa8f4e",
    "17": "dd3c84acfa982253913f7c99d83a877b565a156358b12f773e8c042efd82299c",
    "20": "b416a2112aebab81b846f4a5dcefed977f362aa6cf37ac7ede45c17b2dda96fd",
}
DEFAULT = np.asarray([0.0, 0.9, -1.8] * 4, dtype=np.float32)
PHASES = np.asarray([0.0, math.pi, math.pi, 0.0], dtype=np.float32)
QUALIFIED = {"STAIR_UP", "STAIR_DOWN", "ROUGH_TERRAIN"}


class HardwareRunnerError(RuntimeError):
    pass


def _mode(helper: Path, command: str) -> None:
    args = [str(helper), "--command", command, "--network", "eth0"]
    if command == "release":
        args += ["--confirm", "I_UNDERSTAND_MOTION_MODE_WILL_CHANGE"]
    elif command == "select":
        args += ["--mode", "normal", "--confirm", "I_UNDERSTAND_MOTION_MODE_WILL_CHANGE"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=8)
    if result.returncode:
        raise HardwareRunnerError((result.stderr or result.stdout).strip() or f"motion mode {command} failed")
    print(result.stdout.strip(), flush=True)


def _write_status(path: Path, **payload: Any) -> None:
    payload["updated_at_unix"] = time.time()
    temporary = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-hardware", action="store_true")
    parser.add_argument("--confirmation-file", type=Path, required=True)
    parser.add_argument("--heightmap-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/pgtt-heightmap.bin"))
    parser.add_argument("--policy-root", type=Path, default=Path("/home/unitree/go2_mapping/current/mapping/robot/pgtt-go2"))
    parser.add_argument("--motion-helper", type=Path, default=Path("/home/unitree/go2_mapping/runtime/bin/go2_motion_mode_helper"))
    parser.add_argument("--status-file", type=Path, default=Path("/home/unitree/go2_mapping/runtime/pgtt-hardware-status.json"))
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--maximum-duration", type=float, default=180.0)
    args = parser.parse_args(argv)
    if not args.arm_hardware:
        raise SystemExit("HARDWARE_NOT_ARMED")
    if not args.confirmation_file.is_file() or args.confirmation_file.read_text(encoding="utf-8").strip() != CONFIRMATION:
        raise SystemExit("FIELD_CONFIRMATION_REQUIRED")
    if args.maximum_duration <= 0 or args.maximum_duration > 300:
        parser.error("maximum duration must be in (0, 300]")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, args.network_interface)
    latest: dict[str, Any] = {"state": None, "at": 0.0}

    def receive(message: Any) -> None:
        latest["state"] = message
        latest["at"] = time.monotonic()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(receive, 10)
    input_deadline = time.monotonic() + 8.0
    while latest["state"] is None and time.monotonic() < input_deadline:
        time.sleep(0.02)
    if latest["state"] is None:
        raise SystemExit("LOWSTATE_NOT_READY")

    policies = {
        level: NumpyPGTTPolicy(args.policy_root / f"policy_level{level}_run0.npz", digest)
        for level, digest in POLICY_HASHES.items()
    }
    router = StableTerrainRouter(enter_frames=8, exit_frames=8, minimum_hold_frames=12)
    armed_frames = 0
    route = None
    arm_deadline = time.monotonic() + 30.0
    while time.monotonic() < arm_deadline:
        state = latest["state"]
        remote = decode_remote(state.wireless_remote)
        if remote.emergency_stop:
            raise SystemExit("REMOTE_EMERGENCY_STOP")
        if args.heightmap_file.is_file() and time.time() - args.heightmap_file.stat().st_mtime <= 0.25:
            grid = np.fromfile(args.heightmap_file, dtype="<f4")
            if grid.size == 99 and np.isfinite(grid).all():
                route = router.update(grid.reshape(11, 9), forward_intent=remote.ly)
                ready = route.state in QUALIFIED and route.executable and abs(remote.ly) >= 0.08
                armed_frames = armed_frames + 1 if ready else 0
                if armed_frames >= 20:
                    break
        time.sleep(0.02)
    if armed_frames < 20 or route is None:
        raise SystemExit("TERRAIN_ROUTE_ARM_TIMEOUT")

    command = unitree_go_msg_dds__LowCmd_()
    command.head[0], command.head[1], command.level_flag, command.gpio = 0xFE, 0xEF, 0xFF, 0
    for motor in command.motor_cmd:
        motor.mode, motor.q, motor.kp = 0x01, 2.146e9, 0.0
        motor.dq, motor.kd, motor.tau = 16000.0, 0.0, 0.0
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    crc = CRC()
    measured = np.asarray([latest["state"].motor_state[i].q for i in PGTT_TO_UNITREE_MOTOR_INDEX], dtype=np.float32)
    limiter = TargetLimiter(maximum_step_rad=0.025)
    limiter.reset(measured)
    target = measured.copy()
    phase = PHASES.copy()
    previous_action = np.zeros(12, dtype=np.float32)
    active_level = route.policy_level
    released = False
    started = time.monotonic()
    next_policy = started
    next_publish = started
    blocker = None
    try:
        _mode(args.motion_helper, "release")
        released = True
        while time.monotonic() - started < args.maximum_duration:
            now = time.monotonic()
            state = latest["state"]
            remote = decode_remote(state.wireless_remote)
            if remote.emergency_stop:
                blocker = "REMOTE_EMERGENCY_STOP"
                break
            if now - float(latest["at"]) > 0.10:
                blocker = "LOWSTATE_STALE"
                break
            if not args.heightmap_file.is_file() or time.time() - args.heightmap_file.stat().st_mtime > 0.25:
                blocker = "HEIGHTMAP_STALE"
                break
            gravity = gravity_orientation(state.imu_state.quaternion)
            if abs(float(gravity[0])) > 0.50 or abs(float(gravity[1])) > 0.50 or float(gravity[2]) > -0.70:
                blocker = "POSTURE_GATE"
                break
            if now >= next_policy:
                grid = np.fromfile(args.heightmap_file, dtype="<f4").reshape(11, 9)
                route = router.update(grid, forward_intent=remote.ly)
                if route.state not in QUALIFIED or not route.executable:
                    blocker = route.blocker or "TERRAIN_ROUTE_EXIT"
                    break
                active_level = route.policy_level
                phase = np.fmod(phase + 2.0 * math.pi * 2.0 * 0.02, 2.0 * math.pi)
                phase_observation = np.concatenate((np.cos(phase), np.sin(phase))).astype(np.float32)
                velocity = np.asarray([0.5 * remote.ly, -0.5 * remote.lx, -0.8 * remote.rx], dtype=np.float32)
                observation = compose_go2_observation(lowstate=state, heightmap=grid, phase=phase_observation,
                    last_action=previous_action, command=velocity, default_position=DEFAULT)
                previous_action = policies[active_level](observation[None, :])[0]
                target = limiter.apply(pgtt_joint_targets(previous_action))
                next_policy += 0.02
            if now >= next_publish:
                for policy_index, motor_index in enumerate(PGTT_TO_UNITREE_MOTOR_INDEX):
                    motor = command.motor_cmd[motor_index]
                    motor.q, motor.kp, motor.dq, motor.kd, motor.tau = float(target[policy_index]), 50.0, 0.0, 0.5, 0.0
                command.crc = crc.Crc(command)
                publisher.Write(command)
                next_publish += 0.005
            _write_status(args.status_file, state=route.state, policy_level=active_level,
                transport_authorized=True, blocker=None, remote_contract="DIRECTION_ONLY")
            time.sleep(max(0.0, min(next_policy, next_publish) - time.monotonic()))
    finally:
        _write_status(args.status_file, state="STOPPED", policy_level=active_level,
            transport_authorized=False, blocker=blocker or "MAXIMUM_DURATION")
        if released:
            _mode(args.motion_helper, "select")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
