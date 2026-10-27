"""Explicit diagnostic axis conventions for native Go2 LiDAR data.

This module describes only a discrete axis correspondence observed in natural
scene projections. It intentionally contains no mounting translation and is
never a calibration manifest transform.
"""

from __future__ import annotations

import numpy as np


DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
)
DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_ID = (
    "DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_v1"
)


def diagnostic_utlidar_to_camera_optical_rotation() -> np.ndarray:
    """Return the read-only axis-only diagnostic rotation."""

    rotation = np.asarray(
        DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1,
        dtype=np.float64,
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3)) or not np.isclose(
        np.linalg.det(rotation), 1.0
    ):
        raise RuntimeError("diagnostic LiDAR axis contract is not a proper rotation")
    rotation.setflags(write=False)
    return rotation


def diagnostic_camera_from_lidar_seed() -> np.ndarray:
    """Return an axis-only association seed, explicitly without calibration authority."""

    seed = np.eye(4, dtype=np.float64)
    seed[:3, :3] = diagnostic_utlidar_to_camera_optical_rotation()
    seed.setflags(write=False)
    return seed
