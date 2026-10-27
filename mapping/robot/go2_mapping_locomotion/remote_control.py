"""Unitree wireless remote decoding for supervised terrain locomotion."""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any


KEY_R1 = 0
KEY_L1 = 1
KEY_A = 8
KEY_B = 9


@dataclass(frozen=True)
class RemoteState:
    keys: int
    lx: float
    ly: float
    rx: float
    emergency_stop: bool
    deadman: bool
    arm_chord: bool


def decode_remote(value: Any) -> RemoteState:
    raw = bytes(value)
    if len(raw) < 24:
        raise ValueError("wireless remote payload is truncated")
    keys = struct.unpack_from("<H", raw, 2)[0]
    pressed = lambda bit: bool(keys & (1 << bit))
    return RemoteState(
        keys=keys,
        lx=float(struct.unpack_from("<f", raw, 4)[0]),
        rx=float(struct.unpack_from("<f", raw, 8)[0]),
        ly=float(struct.unpack_from("<f", raw, 20)[0]),
        emergency_stop=pressed(KEY_A),
        deadman=pressed(KEY_L1),
        arm_chord=pressed(KEY_L1) and pressed(KEY_R1) and pressed(KEY_B),
    )


__all__ = ["RemoteState", "decode_remote"]
