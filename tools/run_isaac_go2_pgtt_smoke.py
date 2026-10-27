#!/usr/bin/env python3
"""Run one Isaac Sim Go2 + PGTT policy smoke episode offline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "output" / "isaac-sim" / "go2_driven" / "go2.usd" / "go2" / "go2.usda"
DEFAULT_PGTT_STAIRS_MJCF = ROOT / "third_party" / "phase_guided_terrain_traversal" / "go2" / "xmls" / "pgtt_isaac_stairs.xml"
DEFAULT_POLICY = ROOT / "models" / "pgtt-go2" / "policy_level03_run0.npz"
DEFAULT_POLICY_SHA = "f3fe6655e9fa13e06ecccf13bf1aa13b5e62de50be4539781ac7a360b97fbedd"
DEFAULT_OUTPUT = ROOT / "output" / "isaac-sim" / "pgtt-smoke.json"
DEFAULT_DRIVE_STIFFNESS = 50.0
DEFAULT_DRIVE_DAMPING = 0.5
DEFAULT_DRIVE_FORCE_LIMIT = 35.55
DEFAULT_STEP_HEIGHT = 0.16
DEFAULT_STEP_TREAD = 0.35
DEFAULT_TERRAIN_FRICTION = 0.6
DEFAULT_TERRAIN_RESTITUTION = 0.0
PGTT_FOOT_NAMES = frozenset({"FL_foot", "FR_foot", "RL_foot", "RR_foot"})
GO2_JOINT_LIMITS = tuple(
    [(-1.0472, 1.0472)] * 4
    + [(-1.5708, 3.4907)] * 4
    + [(-2.7227, -0.83776)] * 4
)

PGTT_SIM_DT = 0.005
PGTT_CTRL_DT = 0.02
PGTT_PHYSICS_STEPS_PER_CONTROL = int(PGTT_CTRL_DT / PGTT_SIM_DT)
SUPPORTED_POLICY_BACKENDS = ("pgtt",)
# The official PGTT Go2 XML home keyframe, reordered to Isaac's FL/FR/RL/RR layout.
DEFAULT_JOINT_VALUES = (0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8)
# Isaac's imported articulation is grouped by joint type. MuJoCo qpos follows
# body order FL, FR, RL, RR, while PGTT actions follow actuator order FR, FL,
# RR, RL. These are intentionally separate contracts.
ISAAC_TO_QPOS = (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
ISAAC_TO_ACTUATOR = (1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10)
ACTUATOR_TO_ISAAC = tuple(ISAAC_TO_ACTUATOR.index(index) for index in range(12))


def simulation_app_config(visual: bool, *, livestream: bool = False) -> dict[str, object]:
    """Return the explicit Isaac Sim app mode without changing physics settings."""
    if not isinstance(visual, bool):
        raise TypeError("visual must be a boolean")
    if not isinstance(livestream, bool):
        raise TypeError("livestream must be a boolean")
    return {
        "headless": not (visual and not livestream),
        "renderer": "RayTracedLighting" if (visual or livestream) else "None",
        "physics_gpu": False,
    }


def streaming_experience_path(isaacsim_file: str | Path) -> Path:
    """Locate the installed Isaac Sim WebRTC experience file."""
    package_root = Path(isaacsim_file).resolve().parent
    experience = package_root / "apps" / "isaacsim.exp.full.streaming.kit"
    if not experience.is_file():
        raise FileNotFoundError(f"Isaac Sim streaming experience is unavailable: {experience}")
    return experience


def validate_policy_backend(policy_backend: str) -> str:
    """Keep the production Isaac path single-backend and fail closed."""
    if policy_backend not in SUPPORTED_POLICY_BACKENDS:
        raise ValueError("Genesis backend is disabled; only pgtt is supported")
    return policy_backend


def stair_step_center_x(index: int, step_tread: float) -> float:
    """Return the physical step center used by the official PGTT XML scene."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("step index must be a non-negative integer")
    if not math.isfinite(step_tread) or step_tread <= 0:
        raise ValueError("step tread must be positive and finite")
    return float((index + 1) * step_tread)


def quaternion_up_z(quaternion) -> float:
    """Return the world-up projection for an Isaac wxyz quaternion."""
    w, x, y, z = [float(value) for value in quaternion]
    return 1.0 - 2.0 * (x * x + y * y)


def quaternion_yaw(quaternion) -> float:
    """Return yaw for an Isaac wxyz quaternion."""
    w, x, y, z = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_gravity(quaternion):
    """Return world gravity expressed in the base frame for wxyz input."""
    import numpy as np

    w, x, y, z = [float(value) for value in quaternion]
    return np.asarray(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    ) * -1.0


def quaternion_inverse_rotate(quaternion, vector):
    """Transform a world-frame vector into the base frame for wxyz input."""
    import numpy as np

    w, x, y, z = [float(value) for value in quaternion]
    vx, vy, vz = [float(value) for value in vector]
    return np.asarray(
        [
            (1.0 - 2.0 * (y * y + z * z)) * vx
            + 2.0 * (x * y + w * z) * vy
            + 2.0 * (x * z - w * y) * vz,
            2.0 * (x * y - w * z) * vx
            + (1.0 - 2.0 * (x * x + z * z)) * vy
            + 2.0 * (y * z + w * x) * vz,
            2.0 * (x * z + w * y) * vx
            + 2.0 * (y * z - w * x) * vy
            + (1.0 - 2.0 * (x * x + y * y)) * vz,
        ],
        dtype=np.float32,
    )


def local_forward_displacement(initial_xy, final_xy, initial_yaw: float) -> float:
    """Project world XY displacement onto the initial body-forward axis."""
    dx = float(final_xy[0]) - float(initial_xy[0])
    dy = float(final_xy[1]) - float(initial_xy[1])
    return math.cos(initial_yaw) * dx + math.sin(initial_yaw) * dy


def within_go2_joint_limits(joint_positions, *, tolerance: float = 1e-5) -> bool:
    """Validate measured Isaac DOF positions against the Go2 joint limits."""
    import numpy as np

    values = np.asarray(joint_positions, dtype=np.float32)
    if values.shape != (12,) or not np.isfinite(values).all():
        return False
    return all(
        lower - tolerance <= float(value) <= upper + tolerance
        for value, (lower, upper) in zip(values, GO2_JOINT_LIMITS)
    )


def clip_go2_joint_targets(joint_targets):
    """Clip Isaac joint targets to the Go2 URDF limits in Isaac DOF order."""
    import numpy as np

    values = np.asarray(joint_targets, dtype=np.float32)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("Go2 joint targets must be finite with shape (12,)")
    return np.asarray(
        [np.clip(value, lower, upper) for value, (lower, upper) in zip(values, GO2_JOINT_LIMITS)],
        dtype=np.float32,
    )


def smooth_action(action, previous_action, alpha: float):
    """Blend a policy action with the action actually applied last tick."""
    import numpy as np

    return (
        alpha * np.asarray(action, dtype=np.float32)
        + (1.0 - alpha) * np.asarray(previous_action, dtype=np.float32)
    ).astype(np.float32)


def is_pgtt_foot_collision(prim_path: str) -> bool:
    """Return whether a path names one of the four PGTT foot spheres."""
    parts = str(prim_path).rstrip("/").split("/")
    return (
        len(parts) >= 2
        and (
            (parts[-1] == "sphere" and parts[-2] in PGTT_FOOT_NAMES)
            or (parts[-1] in {name.removesuffix("_foot") for name in PGTT_FOOT_NAMES} and parts[-2].endswith("_calf"))
        )
    )


def is_pgtt_leg_collision(prim_path: str) -> bool:
    """Return whether a path is a thigh/calf collision under a Go2 leg."""
    return any(part.endswith("_thigh") or part.endswith("_calf") for part in str(prim_path).split("/"))


def configure_pgtt_foot_contacts(stage, robot_root: str) -> int:
    """Match PGTT's feet-only terrain contact contract on a composed USD."""
    from pxr import UsdPhysics

    root = robot_root.rstrip("/") + "/"
    changed = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # Isaac requires the parent/base collision chain to remain enabled for
        # a valid articulation contact graph; suppress only non-foot leg shapes.
        enabled = not is_pgtt_leg_collision(path) or is_pgtt_foot_collision(path)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(enabled)
        changed += 1
    return changed


def build_pgtt_observation(*, angular_velocity, gravity, joint_error, joint_velocity,
                           phase_observation, height_observation, gait_frequency,
                           last_action, command):
    """Build the observation in the same field order as the PGTT deployer."""
    import numpy as np

    return np.concatenate(
        (
            np.asarray(angular_velocity, dtype=np.float32),
            np.asarray(gravity, dtype=np.float32),
            np.asarray(joint_error, dtype=np.float32),
            np.asarray(joint_velocity, dtype=np.float32),
            np.asarray(phase_observation, dtype=np.float32),
            np.asarray(height_observation, dtype=np.float32),
            [float(gait_frequency)],
            np.asarray(last_action, dtype=np.float32),
            np.asarray(command, dtype=np.float32),
        )
    ).astype(np.float32)


def analytic_heightmap(robot_x: float, terrain: str, *, robot_yaw: float = 0.0, step_height: float = DEFAULT_STEP_HEIGHT, step_tread: float = DEFAULT_STEP_TREAD) -> "object":
    """Sample the local 11x9 heightmap from the analytic test stairs."""
    import numpy as np

    if terrain not in {"flat", "up", "down"}:
        raise ValueError("terrain must be flat, up, or down")
    heightmap = np.zeros((11, 9), dtype=np.float32)
    if terrain == "flat":
        return heightmap
    heights = np.asarray([step_height * (index + 1) for index in range(5)], dtype=np.float32)
    if terrain == "down":
        heights = heights[::-1]
    forward_offsets = (5.0 - np.arange(11, dtype=np.float32)) * 0.1
    for row, offset in enumerate(forward_offsets):
        x = robot_x + math.cos(robot_yaw) * float(offset)
        step = int(np.floor((x - step_tread / 2.0) / step_tread))
        if 0 <= step < 5:
            heightmap[row, :] = heights[step]
    return heightmap


def run_episode(*, usd_path: Path, policy_path: Path, policy_sha: str, steps: int, robot_root: str = "/Go2", terrain: str = "up", root_height: float | None = None, start_x: float = 0.0, start_yaw: float = 0.0, step_height: float = DEFAULT_STEP_HEIGHT, step_tread: float = DEFAULT_STEP_TREAD, physical_terrain: bool = True, zero_command: bool = False, warmup_ticks: int = 10, action_map: str = "pgtt", obs_map: str = "qpos", command: tuple[float, float, float] = (0.6, 0.4, 0.0), action_scale: float = 0.5, action_smoothing: float = 1.0, policy_backend: str = "pgtt", seed: int = 0, heightmap_mode: str = "analytic", feet_only_collision: bool = False, drive_stiffness: float | None = DEFAULT_DRIVE_STIFFNESS, drive_damping: float | None = DEFAULT_DRIVE_DAMPING, drive_force_limit: float | None = DEFAULT_DRIVE_FORCE_LIMIT, visual: bool = False) -> tuple[dict[str, object], object, object]:
    validate_policy_backend(policy_backend)
    if not isinstance(visual, bool):
        raise TypeError("visual must be a boolean")
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_scene_query_interface
    from omni.usd import get_context

    sys.path.insert(0, str(ROOT))
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PGTT_SIM_DT,
        rendering_dt=PGTT_CTRL_DT,
    )
    print("ISAAC_PGTT_WORLD_READY", flush=True)
    add_reference_to_stage(str(usd_path), "/Go2")
    print("ISAAC_PGTT_REFERENCE_ADDED", flush=True)
    terrain_material = PhysicsMaterial(
        "/World/PGTTTerrainMaterial",
        static_friction=DEFAULT_TERRAIN_FRICTION,
        dynamic_friction=DEFAULT_TERRAIN_FRICTION,
        restitution=DEFAULT_TERRAIN_RESTITUTION,
    )
    world.scene.add(FixedCuboid("/Ground", name="ground", size=20.0, scale=[1, 1, 0.05], translation=[0, 0, -0.025], physics_material=terrain_material))
    if terrain not in {"flat", "up", "down"}:
        raise ValueError("terrain must be flat, up, or down")
    if terrain != "flat" and physical_terrain:
        heights = [step_height * (index + 1) for index in range(5)]
        if terrain == "down":
            heights = list(reversed(heights))
        for index, height in enumerate(heights):
            world.scene.add(
                FixedCuboid(
                    f"/Step{index}",
                    name=f"step{index}",
                    size=1.0,
                    scale=[step_tread, 1.2, height],
                    translation=[stair_step_center_x(index, step_tread), 0, height / 2],
                    physics_material=terrain_material,
                )
            )
    # MJCF references are composed asynchronously. Let Kit materialize the
    # referenced prims before wrapping the articulation view.
    world.reset()
    robot = world.scene.add(SingleArticulation(robot_root, name="go2"))
    contact_prim_count = configure_pgtt_foot_contacts(get_context().get_stage(), robot_root) if feet_only_collision else 0
    print("ISAAC_PGTT_ARTICULATION_WRAPPED", flush=True)
    from mapping.robot.go2_mapping_locomotion.numpy_pgtt import NumpyPGTTPolicy

    policy = NumpyPGTTPolicy(policy_path, policy_sha)
    policy_action_size = 12
    print("ISAAC_PGTT_POLICY_LOADED", flush=True)
    world.reset()
    print("ISAAC_PGTT_WORLD_RESET", flush=True)
    if root_height is not None:
        robot.set_world_pose(
            position=np.asarray([start_x + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), root_height], dtype=np.float32),
            orientation=np.asarray([math.cos(start_yaw / 2.0), 0.0, 0.0, math.sin(start_yaw / 2.0)], dtype=np.float32),
        )
    if warmup_ticks < 0:
        raise ValueError("warmup_ticks must be non-negative")
    if action_map not in {"pgtt", "identity"}:
        raise ValueError("action_map must be pgtt or identity")
    if obs_map not in {"qpos", "identity"}:
        raise ValueError("obs_map must be qpos or identity")
    if len(command) != 3 or not all(np.isfinite(value) for value in command):
        raise ValueError("command must contain three finite values")
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be positive and finite")
    if not np.isfinite(action_smoothing) or not 0.0 < action_smoothing <= 1.0:
        raise ValueError("action_smoothing must be in (0, 1]")
    if heightmap_mode not in {"analytic", "raycast"}:
        raise ValueError("heightmap_mode must be analytic or raycast")
    default_joints = np.asarray(DEFAULT_JOINT_VALUES, dtype=np.float32)
    robot.set_joint_positions(default_joints + rng.normal(0.0, 0.002, size=12).astype(np.float32))
    controller = robot.get_articulation_controller()
    if drive_stiffness is not None or drive_damping is not None:
        controller.set_gains(
            kps=None if drive_stiffness is None else np.full(12, drive_stiffness, dtype=np.float32),
            kds=None if drive_damping is None else np.full(12, drive_damping, dtype=np.float32),
        )
    if drive_force_limit is not None:
        controller.set_max_efforts(np.full(12, drive_force_limit, dtype=np.float32))
    controller.apply_action(ArticulationAction(joint_positions=default_joints))
    if visual:
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=[3.0, -4.0, 2.5], target=[1.0, 0.0, 0.45])
        except Exception as exc:
            print(f"ISAAC_PGTT_CAMERA_UNAVAILABLE {exc}", flush=True)
    last_action = np.zeros(policy_action_size, dtype=np.float32)
    applied_action = np.zeros(policy_action_size, dtype=np.float32)
    phase = np.asarray([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
    gait_frequency = 2.0

    for _ in range(warmup_ticks):
        controller.apply_action(ArticulationAction(joint_positions=default_joints))
        for _ in range(PGTT_PHYSICS_STEPS_PER_CONTROL):
            world.step(render=visual)

    initial_position, initial_quaternion = robot.get_world_pose()
    initial_yaw = quaternion_yaw(initial_quaternion)

    finite = True
    minimum_z = float("inf")
    maximum_z = float("-inf")
    minimum_up_z = float("inf")
    maximum_joint_error = 0.0
    joint_limits_ok = True
    trajectory = []
    control_trace = []
    velocity_x_samples = []
    local_velocity_x_samples = []
    query = get_physx_scene_query_interface() if heightmap_mode == "raycast" else None
    raycast_trace = []
    for tick in range(steps):
        if tick == 0:
            print("ISAAC_PGTT_LOOP_ENTER", flush=True)
        q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        qd = np.asarray(robot.get_joint_velocities(), dtype=np.float32)
        joint_limits_ok = joint_limits_ok and within_go2_joint_limits(q)
        pose_position, pose_quaternion = robot.get_world_pose()
        angular_velocity = quaternion_inverse_rotate(
            pose_quaternion, robot.get_angular_velocity()
        )
        gravity = quaternion_gravity(pose_quaternion)
        if angular_velocity.shape != (3,) or not np.isfinite(angular_velocity).all():
            raise RuntimeError("Isaac articulation returned an invalid angular velocity")
        if heightmap_mode == "analytic":
            heightmap = analytic_heightmap(float(pose_position[0]), terrain, robot_yaw=quaternion_yaw(pose_quaternion), step_height=step_height, step_tread=step_tread)
        else:
            origin = pose_position
            pose_yaw = quaternion_yaw(pose_quaternion)
            heightmap = np.zeros((11, 9), dtype=np.float32)
            for row, forward in enumerate((5.0 - np.arange(11, dtype=np.float32)) * 0.1):
                for col, lateral in enumerate((4.0 - np.arange(9, dtype=np.float32)) * 0.1):
                    sample_x = float(origin[0] + np.cos(pose_yaw) * forward - np.sin(pose_yaw) * lateral)
                    sample_y = float(origin[1] + np.sin(pose_yaw) * forward + np.cos(pose_yaw) * lateral)
                    ray_z = float(origin[2]) + 2.0
                    hit = None
                    last_hit_diagnostic = None
                    for _ in range(8):
                        candidate = query.raycast_closest((sample_x, sample_y, ray_z), (0.0, 0.0, -1.0), 4.0)
                        if not candidate.get("hit", False):
                            raise RuntimeError(f"raycast heightmap sample missed scene geometry; last_hit={last_hit_diagnostic}")
                        hit = candidate
                        hit_z = float(candidate["position"][2])
                        rigid_body = str(candidate.get("rigidBody", ""))
                        collision = str(candidate.get("collision", ""))
                        last_hit_diagnostic = {"rigidBody": rigid_body, "collision": collision, "z": hit_z}
                        if len(raycast_trace) < 16:
                            raycast_trace.append({"rigidBody": rigid_body, "collision": collision, "z": hit_z})
                        is_self_hit = (
                            rigid_body == robot_root
                            or rigid_body.startswith(robot_root.rstrip("/") + "/")
                            or collision.startswith(robot_root.rstrip("/") + "/")
                        )
                        if not is_self_hit:
                            break
                        # Move well below the hit volume; a millimetre offset
                        # can leave the ray origin inside a thick collider.
                        ray_z = hit_z - 0.10
                    if hit is None or is_self_hit:
                        raise RuntimeError(f"raycast heightmap could not reach support surface; last_hit={last_hit_diagnostic}")
                    heightmap[row, col] = float(hit["position"][2])
        effective_command = (0.0, 0.0, 0.0) if zero_command else command
        phase = np.fmod(phase + 2.0 * np.pi * gait_frequency * PGTT_CTRL_DT, 2.0 * np.pi)
        phase_observation = np.concatenate([np.cos(phase), np.sin(phase)])
        # Match the pinned PGTT deployment preprocessing: subtract the local
        # minimum only; do not rescale the height channel.
        height_observation = heightmap.reshape(-1) - np.min(heightmap)
        q_pgtt = q[list(ISAAC_TO_QPOS)] if obs_map == "qpos" else q
        qd_pgtt = qd[list(ISAAC_TO_QPOS)] if obs_map == "qpos" else qd
        default_qpos = default_joints[list(ISAAC_TO_QPOS)]
        default_actuator = default_joints[list(ISAAC_TO_ACTUATOR)]
        observation = build_pgtt_observation(
            angular_velocity=angular_velocity,
            gravity=gravity,
            joint_error=q_pgtt - default_qpos,
            joint_velocity=qd_pgtt,
            phase_observation=phase_observation,
            height_observation=height_observation,
            gait_frequency=gait_frequency,
            last_action=last_action,
            command=effective_command,
        )
        action = np.zeros(policy_action_size, dtype=np.float32) if zero_command else policy(observation.reshape(1, -1))[0]
        finite = finite and bool(np.isfinite(observation).all() and np.isfinite(action).all() and np.isfinite(q).all())
        if not finite:
            break
        if zero_command:
            target_joints = default_joints.copy()
            applied_action = action.copy()
        elif action_map == "identity":
            applied_action = smooth_action(action, applied_action, action_smoothing)
            target_joints = default_joints + action_scale * applied_action
        else:
            applied_action = smooth_action(action, applied_action, action_smoothing)
            raw_target_actuator = default_actuator + action_scale * applied_action
            target_joints = raw_target_actuator[list(ACTUATOR_TO_ISAAC)]
        target_joints = clip_go2_joint_targets(target_joints)
        last_action = applied_action.copy()
        controller.apply_action(ArticulationAction(joint_positions=target_joints))
        for _ in range(PGTT_PHYSICS_STEPS_PER_CONTROL):
            world.step(render=visual)
        if tick == 0:
            print("ISAAC_PGTT_FIRST_STEP", flush=True)
        z = float(robot.get_world_pose()[0][2])
        pose_quaternion = robot.get_world_pose()[1]
        up_z = quaternion_up_z(pose_quaternion)
        base_velocity = np.asarray(robot.get_linear_velocity(), dtype=np.float32)
        velocity_x_samples.append(float(base_velocity[0]))
        yaw = quaternion_yaw(pose_quaternion)
        local_velocity_x = float(np.cos(yaw) * base_velocity[0] + np.sin(yaw) * base_velocity[1])
        local_velocity_x_samples.append(local_velocity_x)
        minimum_z = min(minimum_z, z)
        maximum_z = max(maximum_z, z)
        minimum_up_z = min(minimum_up_z, up_z)
        maximum_joint_error = max(maximum_joint_error, float(np.max(np.abs(q - target_joints))))
        if len(control_trace) < 50:
            control_trace.append({
                "tick": tick,
                "action": action.astype(float).tolist(),
                "applied_action": applied_action.astype(float).tolist(),
                "target_joints": target_joints.astype(float).tolist(),
                "joint_positions": q.astype(float).tolist(),
                "joint_velocities": qd.astype(float).tolist(),
                "target_error_max": float(np.max(np.abs(q - target_joints))),
                "base_up_z": up_z,
                "base_velocity": base_velocity.astype(float).tolist(),
                "local_velocity_x": local_velocity_x,
            })
        if tick % max(1, steps // 20) == 0 or tick == steps - 1:
            trajectory.append({"tick": tick, "x": float(robot.get_world_pose()[0][0]), "z": z})

    position = np.asarray(robot.get_world_pose()[0])
    print("ISAAC_PGTT_POSE_READ", flush=True)
    stable = finite and joint_limits_ok and bool(np.isfinite(position).all()) and float(position[2]) > 0.40 and minimum_z > 0.40 and minimum_up_z > 0.75
    target_vx = 0.0 if zero_command else float(command[0])
    velocity_tracking_mae = float(np.mean(np.abs(np.asarray(local_velocity_x_samples) - target_vx))) if local_velocity_x_samples else None
    velocity_tracking_pass = bool(stable and velocity_tracking_mae is not None and velocity_tracking_mae <= 0.20)
    # The robot must clear the last step and settle on the top platform. The
    # x threshold is the front edge of Step4 plus one body half-length.
    forward_displacement = local_forward_displacement(initial_position, position, initial_yaw)
    traversal_success = terrain == "up" and (
        stable
        and forward_displacement >= 2.05
        and float(position[2]) >= 0.55
        and (maximum_z - minimum_z) < 0.55
    )
    if terrain == "down":
        traversal_success = bool(
            stable
            and forward_displacement <= -2.05
            and float(position[2]) >= 0.20
            and (maximum_z - minimum_z) < 0.55
        )
    result = {
        "schema_version": 1,
        "simulator": "isaac-sim-6.0.1.0",
        "physics": "CPU_PHYSX",
        "steps": steps,
        "control_frequency_hz": 50,
        "sim_dt": PGTT_SIM_DT,
        "control_dt": PGTT_CTRL_DT,
        "physics_steps_per_control": PGTT_PHYSICS_STEPS_PER_CONTROL,
        "terrain": terrain,
        "root_height_override": root_height,
        "start_x": start_x,
        "start_yaw": start_yaw,
        "step_height": step_height,
        "step_tread": step_tread,
        "physical_terrain": physical_terrain,
        "initial_yaw_rad": initial_yaw,
        "forward_displacement_m": forward_displacement,
        "pgtt_contact_prim_count": contact_prim_count,
        "zero_command": zero_command,
        "action_map": action_map,
        "obs_map": obs_map,
        "command": list(command),
        "action_scale": action_scale,
        "action_smoothing": action_smoothing,
        "seed": seed,
        "heightmap_mode": heightmap_mode,
        "visual": visual,
        "raycast_trace": raycast_trace,
        "warmup_ticks": warmup_ticks,
        "drive_override": {
            "stiffness": drive_stiffness,
            "damping": drive_damping,
            "force_limit": drive_force_limit,
        },
        "dof": int(robot.num_dof),
        "joint_order": list(robot.dof_names),
        "policy": "pgtt-go2-level03-numpy",
        "policy_backend": policy_backend,
        "finite": finite,
        "stable": stable,
        "joint_limits_ok": joint_limits_ok,
        "traversal_success": traversal_success,
        "trajectory": trajectory,
        "control_trace": control_trace,
        "failure_reason": None if traversal_success else (
            "flat_or_down_scenario" if terrain != "up" else
            "did_not_clear_top_platform" if forward_displacement < 2.05 else
            "top_platform_height_or_settling_not_met"
        ),
        "final_position": position.astype(float).tolist(),
        "minimum_z": minimum_z,
        "maximum_z": maximum_z,
        "minimum_up_z": minimum_up_z,
        "maximum_joint_error": maximum_joint_error,
        "velocity_x_mean_mps": float(np.mean(velocity_x_samples)) if velocity_x_samples else None,
        "velocity_x_p05_mps": float(np.percentile(velocity_x_samples, 5)) if velocity_x_samples else None,
        "velocity_x_p95_mps": float(np.percentile(velocity_x_samples, 95)) if velocity_x_samples else None,
        "local_velocity_x_mean_mps": float(np.mean(local_velocity_x_samples)) if local_velocity_x_samples else None,
        "local_velocity_x_p05_mps": float(np.percentile(local_velocity_x_samples, 5)) if local_velocity_x_samples else None,
        "local_velocity_x_p95_mps": float(np.percentile(local_velocity_x_samples, 95)) if local_velocity_x_samples else None,
        "target_velocity_x_mps": target_vx,
        "velocity_tracking_mae_mps": velocity_tracking_mae,
        "velocity_tracking_pass": velocity_tracking_pass,
        "executable": False,
        "transport_authorized": False,
    }
    print("ISAAC_PGTT_RESULT_BUILT", flush=True)
    # Keep World alive until the caller has persisted the result. Isaac Sim 6
    # closes Kit when this object is destroyed at function return.
    return result, world, robot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--policy-sha256", default=DEFAULT_POLICY_SHA)
    parser.add_argument("--policy-backend", choices=SUPPORTED_POLICY_BACKENDS, default="pgtt")
    parser.add_argument("--visual", action="store_true", help="open the Isaac Sim window and render each step")
    parser.add_argument("--livestream", action="store_true", help="stream the rendered Isaac Sim viewport over WebRTC")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="keep Isaac Sim alive after the episode for livestream viewing")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--robot-root", default="/Go2")
    parser.add_argument("--terrain", choices=("flat", "up", "down"), default="up")
    parser.add_argument("--root-height", type=float)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--step-height", type=float, default=DEFAULT_STEP_HEIGHT)
    parser.add_argument("--step-tread", type=float, default=DEFAULT_STEP_TREAD)
    parser.add_argument("--heightmap-only", action="store_true", help="diagnostic: omit physical stairs")
    parser.add_argument("--feet-only-collision", action="store_true", help="experimental; disable non-foot leg collisions")
    parser.add_argument("--zero-command", action="store_true")
    parser.add_argument("--warmup-ticks", type=int, default=10)
    parser.add_argument("--action-map", choices=("pgtt", "identity"), default="pgtt")
    parser.add_argument("--obs-map", choices=("qpos", "identity"), default="qpos")
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--vy", type=float, default=0.4)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--action-smoothing", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heightmap-mode", choices=("analytic", "raycast"), default="analytic")
    parser.add_argument("--drive-stiffness", type=float, default=DEFAULT_DRIVE_STIFFNESS)
    parser.add_argument("--drive-damping", type=float, default=DEFAULT_DRIVE_DAMPING)
    parser.add_argument("--drive-force-limit", type=float, default=DEFAULT_DRIVE_FORCE_LIMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        raise ValueError("hold-seconds must be finite and non-negative")
    if not args.usd.is_file() or not args.policy.is_file():
        raise FileNotFoundError("Isaac USD or PGTT policy is unavailable")

    import isaacsim
    from isaacsim import SimulationApp

    # Kit consumes process arguments as its own flags. Keep the CLI arguments
    # out of Kit, while preserving argparse's parsed values above.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    experience = str(streaming_experience_path(isaacsim.__file__)) if args.livestream else ""
    app = SimulationApp(simulation_app_config(args.visual, livestream=args.livestream), experience=experience)
    sys.argv = saved_argv
    print("ISAAC_PGTT_APP_READY", flush=True)
    world = None
    robot = None
    try:
        result, world, robot = run_episode(usd_path=args.usd, policy_path=args.policy, policy_sha=args.policy_sha256, steps=args.steps, robot_root=args.robot_root, terrain=args.terrain, root_height=args.root_height, start_x=args.start_x, start_yaw=args.start_yaw, step_height=args.step_height, step_tread=args.step_tread, physical_terrain=not args.heightmap_only, zero_command=args.zero_command, warmup_ticks=args.warmup_ticks, action_map=args.action_map, obs_map=args.obs_map, command=(args.vx, args.vy, args.yaw_rate), action_scale=args.action_scale, action_smoothing=args.action_smoothing, policy_backend=args.policy_backend, seed=args.seed, heightmap_mode=args.heightmap_mode, feet_only_collision=args.feet_only_collision, drive_stiffness=args.drive_stiffness, drive_damping=args.drive_damping, drive_force_limit=args.drive_force_limit, visual=args.visual or args.livestream)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True), flush=True)
        print("ISAAC_PGTT_SMOKE_PASS" if result["stable"] else "ISAAC_PGTT_SMOKE_FAIL", flush=True)
        print("ISAAC_PGTT_STAIR_TRAVERSAL_PASS" if result["traversal_success"] else "ISAAC_PGTT_STAIR_TRAVERSAL_NOT_VERIFIED", flush=True)
        if args.livestream and args.hold_seconds > 0:
            deadline = time.monotonic() + args.hold_seconds
            print(f"ISAAC_PGTT_LIVESTREAM_HOLD {args.hold_seconds:.1f}s", flush=True)
            while time.monotonic() < deadline and app.is_running():
                app.update()
                time.sleep(0.05)
        exit_code = 0 if result["stable"] else 1
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
