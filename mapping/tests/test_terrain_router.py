import struct

import numpy as np
import pytest

from mapping.robot.go2_mapping_locomotion.remote_control import decode_remote
from mapping.robot.go2_mapping_locomotion.terrain_router import StableTerrainRouter, classify_heightmap, classify_stair_profile


def grid(rows):
    return np.repeat(np.asarray(rows, dtype=np.float32)[:, None], 9, axis=1)


def test_router_classifies_flat_rough_up_and_down():
    assert classify_heightmap(grid([0] * 11), forward_intent=0.5).state == "FLAT"
    rough = grid([0, 0.02, -0.05, 0.05, -0.04, 0.07, -0.05, 0.05, 0, 0.03, -0.02])
    assert classify_heightmap(rough, forward_intent=0.5).state == "ROUGH_TERRAIN"
    up = grid([0, 0, 0, 0, 0, 0.06, 0.06, 0.12, 0.12, 0.18, 0.18])
    route = classify_heightmap(up, forward_intent=0.5)
    assert (route.state, route.task, route.executable) == ("STAIR_UP", "STAIR_ASCEND", True)
    down = grid([0.24, 0.24, 0.24, 0.18, 0.18, 0.18, 0.12, 0.12, 0.06, 0.06, 0])
    assert classify_heightmap(down, forward_intent=0.5).state == "STAIR_DOWN"


def test_router_reverses_heightmap_for_backward_intent_and_blocks_high_steps():
    profile = grid([0, 0, 0, 0, 0, 0.13, 0.13, 0.26, 0.26, 0.39, 0.39])
    forward = classify_heightmap(profile, forward_intent=0.5)
    backward = classify_heightmap(profile, forward_intent=-0.5)
    assert forward.state == "STAIR_UP"
    assert forward.executable is False
    assert forward.blocker == "STAIR_STEP_EXCEEDS_INITIAL_HARDWARE_GATE"
    assert backward.state == "STAIR_DOWN"


def test_router_detects_stairs_in_extended_forward_probe():
    profile = grid([0, 0, 0, 0, 0, 0.0, 0.06, 0.06, 0.12, 0.12, 0.18, 0.18, 0.24, 0.24, 0.30, 0.30, 0.36, 0.36, 0.42, 0.42, 0.42])
    route = classify_stair_profile(profile, forward_intent=0.5)
    assert (route.state, route.executable) == ("STAIR_UP", True)


def test_router_requires_direction_intent_and_stable_frames_without_l1():
    flat = grid([0] * 11)
    up = grid([0, 0, 0, 0, 0, 0.06, 0.06, 0.12, 0.12, 0.18, 0.18])
    idle = classify_heightmap(flat, forward_intent=0.0)
    assert (idle.state, idle.blocker) == ("IDLE", "DIRECTION_ZERO")
    router = StableTerrainRouter(enter_frames=3, exit_frames=3, minimum_hold_frames=0)
    assert router.update(flat, forward_intent=0.5).state == "FLAT"
    assert router.update(up, forward_intent=0.5).state == "FLAT"
    assert router.update(up, forward_intent=0.5).state == "FLAT"
    assert router.update(up, forward_intent=0.5).state == "STAIR_UP"
    assert router.update(up, forward_intent=0.5).state == "STAIR_UP"


def test_remote_decodes_direction_without_l1_authorization():
    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, (1 << 9))
    struct.pack_into("<f", payload, 4, -0.25)
    struct.pack_into("<f", payload, 8, 0.3)
    struct.pack_into("<f", payload, 20, 0.6)
    remote = decode_remote(payload)
    assert remote.deadman is False
    assert remote.arm_chord is False
    assert remote.emergency_stop is False
    assert (remote.lx, remote.ly, remote.rx) == pytest.approx((-0.25, 0.6, 0.3))
