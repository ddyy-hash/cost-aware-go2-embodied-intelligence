"""Robot-side validation boundary for approved terrain-controller requests."""

from .request_gate import RequestGateError, validate_controller_request
from .policy_runtime import (
    GO2_JOINT_ORDER,
    Go2PolicyRuntime,
    JointTargetFrame,
    PolicyRuntimeConfig,
    PolicyRuntimeError,
)
from .sdk2_adapter import (
    PGTT_TO_UNITREE_MOTOR_INDEX,
    SDK2AdapterConfig,
    SDK2AdapterError,
    UnitreeSDK2Go2Adapter,
)
from .readonly_pgtt_runner import (
    ReadonlyPGTTError,
    compose_go2_observation,
    gravity_orientation,
    heightmap_from_pointcloud,
    run_readonly_inference,
)
from .heightmap import HeightmapError, build_pgtt_heightmap, pointcloud_xyz
from .heightmap_ros import create_node
from .action_contract import ActionContractError, DEFAULT_GO2_POSE, TargetLimiter, pgtt_joint_targets
from .remote_control import RemoteState, decode_remote
from .terrain_router import StableTerrainRouter, TerrainRoute, TerrainRouterError, classify_heightmap

__all__ = [
    "RequestGateError",
    "validate_controller_request",
    "GO2_JOINT_ORDER",
    "Go2PolicyRuntime",
    "JointTargetFrame",
    "PolicyRuntimeConfig",
    "PolicyRuntimeError",
    "PGTT_TO_UNITREE_MOTOR_INDEX",
    "SDK2AdapterConfig",
    "SDK2AdapterError",
    "UnitreeSDK2Go2Adapter",
    "ReadonlyPGTTError",
    "compose_go2_observation",
    "gravity_orientation",
    "heightmap_from_pointcloud",
    "run_readonly_inference",
    "HeightmapError",
    "build_pgtt_heightmap",
    "pointcloud_xyz",
    "create_node",
    "ActionContractError",
    "DEFAULT_GO2_POSE",
    "TargetLimiter",
    "pgtt_joint_targets",
    "RemoteState",
    "decode_remote",
    "StableTerrainRouter",
    "TerrainRoute",
    "TerrainRouterError",
    "classify_heightmap",
]
