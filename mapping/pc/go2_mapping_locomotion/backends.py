"""Auditable registry for external locomotion backends.

The registry describes integration contracts only. It deliberately does not
download, import, or execute third-party policy code from the Dashboard.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_BACKENDS: tuple[dict[str, Any], ...] = (
    {
        "id": "pgtt-go2",
        "project": "phase_guided_terrain_traversal",
        "upstream": "https://github.com/NtagkasAlex/phase_guided_terrain_traversal",
        "commit": "9bd133884bbf2eeaa40c82aee534246ccaaf02f8",
        "license_required": True,
        "hardware": ["unitree_go2"],
        "inputs": ["heightmap", "imu", "joint_state", "command"],
        "deployment": "unitree_sdk2py",
        "terrain_tasks": ["stairs", "discrete_obstacles", "rough_terrain"],
        "direct_motor_output": False,
        "status": "ADAPTER_REQUIRED",
    },
    {
        "id": "unitree-rl-gym",
        "project": "unitree_rl_gym",
        "upstream": "https://github.com/unitreerobotics/unitree_rl_gym",
        "commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
        "license_required": True,
        "hardware": ["unitree_go2"],
        "inputs": ["imu", "joint_state", "command"],
        "deployment": "unitree_sdk2",
        "terrain_tasks": ["flat", "rough_terrain"],
        "direct_motor_output": False,
        "status": "REFERENCE_TRAINING_BASELINE",
    },
    {
        "id": "walk-these-ways-go2",
        "project": "walk-these-ways-go2",
        "upstream": "https://github.com/Teddy-Liao/walk-these-ways-go2",
        "commit": "d00320778c61567379b4a304047955c0dcae43",
        "license_required": True,
        "hardware": ["unitree_go2"],
        "inputs": ["imu", "joint_state", "command", "joystick"],
        "deployment": "unitree_sdk2_lcm_bridge",
        "terrain_tasks": ["flat", "rough_terrain"],
        "direct_motor_output": False,
        "status": "PC_DEPLOYMENT_REFERENCE",
    },
)


def available_backends() -> list[dict[str, Any]]:
    """Return copies so callers cannot mutate the registry."""

    return deepcopy(list(_BACKENDS))


def backend_manifest(backend_id: str) -> dict[str, Any]:
    for backend in _BACKENDS:
        if backend["id"] == backend_id:
            return deepcopy(backend)
    raise KeyError(f"unknown locomotion backend: {backend_id}")


__all__ = ["available_backends", "backend_manifest"]
