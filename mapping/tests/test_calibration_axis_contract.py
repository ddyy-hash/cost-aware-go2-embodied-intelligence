import numpy as np
import pytest

from mapping.pc.go2_mapping_calibration.axis_contract import (
    DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1,
    diagnostic_utlidar_to_camera_optical_rotation,
)
from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


def test_utlidar_axis_projection_is_explicit_diagnostic_only_and_has_no_translation():
    rotation = diagnostic_utlidar_to_camera_optical_rotation()

    assert DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1 == (
        (-1, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
    )
    assert rotation == pytest.approx(np.asarray(DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1))
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert not rotation.flags.writeable
    assert PROVISIONAL_GO2_UTLIDAR_PITCH180_V2 != "DIAGNOSTIC_UTLIDAR_TO_CAMERA_OPTICAL_AXIS_MAP_V1"
