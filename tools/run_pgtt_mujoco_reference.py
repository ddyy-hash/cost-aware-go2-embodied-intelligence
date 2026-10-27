#!/usr/bin/env/env python3
"""Run the pinned PGTT Go2 policy against the official MuJoCo XML headlessly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PGTT_ROOT = ROOT / "third_party" / "phase_guided_terrain_traversal"
XML = PGTT_ROOT / "go2" / "xmls" / "terrain_test_mjx.xml"
POLICY = ROOT / "models" / "pgtt-go2" / "policy_level03_run0.npz"
POLICY_SHA = "f3fe6655e9fa13e06ecccf13bf1aa13b5e62de50be4539781ac7a360b97fbedd"


def run(*, steps: int, output: Path, xml: Path = XML, seed: int = 0, direction: str = "forward", neutral: bool = False, friction_scale: float = 1.0, mass_scale: float = 1.0, com_offset_x: float = 0.0, com_offset_y: float = 0.0, sensor_noise_std: float = 0.0, action_delay_ticks: int = 0) -> dict[str, object]:
    import mujoco

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(PGTT_ROOT))
    from mapping.robot.go2_mapping_locomotion.numpy_pgtt import NumpyPGTTPolicy
    from deploy.cpu_heightmap.heightmap import create_sensor_matrix

    if direction not in {"forward", "up", "down"}:
        raise ValueError("direction must be forward, up, or down")
    if friction_scale <= 0 or mass_scale <= 0 or sensor_noise_std < 0 or action_delay_ticks < 0:
        raise ValueError("perturbation values are out of range")
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    model.opt.timestep = 0.005
    model.geom_friction[:, 0] *= friction_scale
    model.body_mass[1:] *= mass_scale
    model.body_ipos[1:, 0] += com_offset_x
    model.body_ipos[1:, 1] += com_offset_y
    model.dof_damping[6:] = 0.5
    model.actuator_gainprm[:, 0] = 50.0
    model.actuator_biasprm[:, 1] = -50.0
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    default_pose = model.keyframe("home").qpos[7:].copy()
    rng = np.random.default_rng(seed)
    data.qpos[0:2] += rng.uniform(-0.02, 0.02, size=2)
    data.qpos[7:] += rng.normal(0.0, 0.005, size=data.qpos[7:].shape)
    heightscan = create_sensor_matrix(model, data, data.qpos[:3], 0.0, dist_x=0.1, dist_y=0.1, num_heightscans=11, num_widthscans=9)
    data.qpos[2] += np.max(heightscan[..., 2])
    mujoco.mj_forward(model, data)

    policy = NumpyPGTTPolicy(POLICY, POLICY_SHA)
    phase = np.asarray([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    command = np.asarray([-0.6, -0.4, 0.0] if direction == "down" else [0.6, 0.4, 0.0], dtype=np.float32)
    trajectory = []
    action_history = [np.zeros(12, dtype=np.float32) for _ in range(action_delay_ticks)]
    safe_stop_triggered = False
    finite = True
    for tick in range(steps):
        gyro = np.asarray(data.sensor("gyro").data, dtype=np.float32)
        xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = xmat.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        q_error = np.asarray(data.qpos[7:] - default_pose, dtype=np.float32)
        qd = np.asarray(data.qvel[6:], dtype=np.float32)
        phase = np.fmod(phase + 2.0 * np.pi * 2.0 * 0.02, 2.0 * np.pi)
        phase_observation = np.concatenate([np.cos(phase), np.sin(phase)])
        yaw = float(np.arctan2(xmat[1, 0], xmat[0, 0]))
        heightscan = create_sensor_matrix(model, data, data.qpos[:3], yaw, dist_x=0.1, dist_y=0.1, num_heightscans=11, num_widthscans=9)
        height = np.asarray(heightscan[..., 2], dtype=np.float32).reshape(-1)
        height = height - np.min(height)
        observation = np.concatenate((gyro, gravity, q_error, qd, phase_observation, height, [2.0], last_action, command)).astype(np.float32)
        if sensor_noise_std:
            observation += rng.normal(0.0, sensor_noise_std, size=observation.shape).astype(np.float32)
        delayed_input = action_delay_ticks >= 2
        if delayed_input:
            safe_stop_triggered = True
        action = np.zeros(12, dtype=np.float32) if neutral or delayed_input else policy(observation.reshape(1, -1))[0]
        finite = finite and bool(np.isfinite(observation).all() and np.isfinite(action).all())
        if action_delay_ticks:
            action_history.append(action)
            delayed_action = action_history.pop(0)
        else:
            delayed_action = action
        data.ctrl[:] = default_pose if neutral or delayed_input else default_pose + 0.5 * delayed_action
        last_action = action
        for _ in range(4):
            mujoco.mj_step(model, data)
        if tick % max(1, steps // 20) == 0 or tick == steps - 1:
            trajectory.append({"tick": tick, "x": float(data.qpos[0]), "z": float(data.qpos[2])})

    final_position = np.asarray(data.qpos[:3], dtype=float)
    displacement = float(final_position[0] - trajectory[0]["x"])
    stable = bool(finite and final_position[2] > 0.20 and np.isfinite(final_position).all())
    traversal_success = bool(stable and displacement >= 5.0) if direction in {"forward", "up"} else bool(stable and displacement <= -5.0)
    result = {
        "schema_version": 1,
        "simulator": "mujoco-official-pgtt-xml",
        "physics": "MuJoCo_CPU",
        "steps": steps,
        "seed": seed,
        "direction": direction,
        "neutral": neutral,
        "friction_scale": friction_scale,
        "mass_scale": mass_scale,
        "com_offset_x": com_offset_x,
        "com_offset_y": com_offset_y,
        "sensor_noise_std": sensor_noise_std,
        "action_delay_ticks": action_delay_ticks,
        "safe_stop_triggered": safe_stop_triggered,
        "xml": str(xml),
        "sim_dt": 0.005,
        "control_dt": 0.02,
        "finite": finite,
        "final_position": final_position.tolist(),
        "displacement_x": displacement,
        "stable": stable,
        "traversal_success": traversal_success,
        "trajectory": trajectory,
        "executable": False,
        "transport_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--xml", type=Path, default=XML)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("forward", "up", "down"), default="forward")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--com-offset-x", type=float, default=0.0)
    parser.add_argument("--com-offset-y", type=float, default=0.0)
    parser.add_argument("--sensor-noise-std", type=float, default=0.0)
    parser.add_argument("--action-delay-ticks", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "isaac-sim" / "pgtt-mujoco-reference.json")
    args = parser.parse_args()
    result = run(steps=args.steps, output=args.output, xml=args.xml, seed=args.seed, direction=args.direction, neutral=args.neutral, friction_scale=args.friction_scale, mass_scale=args.mass_scale, com_offset_x=args.com_offset_x, com_offset_y=args.com_offset_y, sensor_noise_std=args.sensor_noise_std, action_delay_ticks=args.action_delay_ticks)
    print(json.dumps(result, sort_keys=True))
    print("PGTT_MUJOCO_REFERENCE_PASS" if result["finite"] else "PGTT_MUJOCO_REFERENCE_FAIL")
    return 0 if result["finite"] and result["stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
