#!/usr/bin/env python3
"""Render a non-authoritative PGTT MuJoCo episode to a portable GIF artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML_ROOT = ROOT / "third_party" / "phase_guided_terrain_traversal" / "go2" / "xmls"
POLICY = ROOT / "models" / "pgtt-go2" / "policy_level03_run0.npz"
POLICY_SHA = "f3fe6655e9fa13e06ecccf13bf1aa13b5e62de50be4539781ac7a360b97fbedd"


def run(*, xml: Path, steps: int, output_dir: Path, fps: int) -> dict[str, object]:
    import imageio.v2 as imageio
    import mujoco

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(XML_ROOT.parents[1]))
    from mapping.robot.go2_mapping_locomotion.numpy_pgtt import NumpyPGTTPolicy
    from deploy.cpu_heightmap.heightmap import create_sensor_matrix

    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    model.opt.timestep = 0.005
    model.dof_damping[6:] = 0.5
    model.actuator_gainprm[:, 0] = 50.0
    model.actuator_biasprm[:, 1] = -50.0
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    default_pose = model.keyframe("home").qpos[7:].copy()
    policy = NumpyPGTTPolicy(POLICY, POLICY_SHA)
    phase = np.asarray([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    command = np.asarray([0.6, 0.4, 0.0], dtype=np.float32)
    renderer = mujoco.Renderer(model, height=360, width=640)
    frames: list[np.ndarray] = []
    trajectory: list[dict[str, float | int]] = []
    stride = max(1, steps // max(1, fps * 12))
    for tick in range(steps):
        gyro = np.asarray(data.sensor("gyro").data, dtype=np.float32)
        xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = xmat.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        q_error = np.asarray(data.qpos[7:] - default_pose, dtype=np.float32)
        qd = np.asarray(data.qvel[6:], dtype=np.float32)
        phase = np.fmod(phase + 2.0 * np.pi * 2.0 * 0.02, 2.0 * np.pi)
        yaw = float(np.arctan2(xmat[1, 0], xmat[0, 0]))
        scan = create_sensor_matrix(model, data, data.qpos[:3], yaw, dist_x=0.1, dist_y=0.1, num_heightscans=11, num_widthscans=9)
        height = np.asarray(scan[..., 2], dtype=np.float32).reshape(-1)
        height -= np.min(height)
        observation = np.concatenate((gyro, gravity, q_error, qd, np.cos(phase), np.sin(phase), height, [2.0], last_action, command)).astype(np.float32)
        action = policy(observation.reshape(1, -1))[0]
        data.ctrl[:] = default_pose + 0.5 * action
        last_action = action
        for _ in range(4):
            mujoco.mj_step(model, data)
        if tick % stride == 0 or tick == steps - 1:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
            trajectory.append({"tick": tick, "x": float(data.qpos[0]), "y": float(data.qpos[1]), "z": float(data.qpos[2])})
    renderer.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "pgtt-mujoco-visual.gif"
    imageio.mimsave(gif_path, frames, format="GIF", duration=1.0 / fps, loop=0)
    result = {
        "schema_version": 1,
        "simulation_backend": "MuJoCo_CPU",
        "visual_artifact": str(gif_path),
        "xml": str(xml),
        "steps": steps,
        "frames": len(frames),
        "fps": fps,
        "trajectory": trajectory,
        "stable": bool(data.qpos[2] > 0.20 and np.isfinite(data.qpos).all()),
        "executable": False,
        "transport_authorized": False,
    }
    report_path = output_dir / "pgtt-mujoco-visual.json"
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=XML_ROOT / "terrain_test_mjx.xml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "simulation-live-visual" / "mujoco")
    args = parser.parse_args()
    if args.steps <= 0 or args.fps <= 0:
        parser.error("steps and fps must be positive")
    result = run(xml=args.xml, steps=args.steps, output_dir=args.output_dir, fps=args.fps)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    print("MUJOCO_VISUAL_SIM_PASS" if result["stable"] else "MUJOCO_VISUAL_SIM_FAIL")
    return 0 if result["stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
