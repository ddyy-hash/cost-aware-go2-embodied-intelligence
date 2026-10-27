import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROBOT_PACKAGE = ROOT / "mapping" / "robot" / "go2_mapping_lio"
sys.path.insert(0, str(ROBOT_PACKAGE))


def _camera_matrix(width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            [float(width) * 1.2, 0.0, float(width) / 2.0],
            [0.0, float(width) * 1.2, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_robot_preflight_accepts_the_locked_complete_charuco_board():
    from go2_mapping_lio.calibration_preflight import detect_charuco_frame, go2_lab_board_spec

    spec = go2_lab_board_spec()
    rendered = spec.create_board().generateImage((1200, 900), marginSize=0, borderBits=1)
    frame_bgr = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)

    report = detect_charuco_frame(
        frame_bgr,
        camera_matrix=_camera_matrix(frame_bgr.shape[1], frame_bgr.shape[0]),
        distortion=np.zeros(5, dtype=np.float64),
    )

    assert report["accepted"] is True
    assert report["dictionary"] == "DICT_4X4_100"
    assert report["markers"] == 54
    assert report["charuco_corners"] == 88
    assert report["blocking_reasons"] == []


def test_robot_preflight_rejects_an_image_without_the_target():
    from go2_mapping_lio.calibration_preflight import detect_charuco_frame

    frame_bgr = np.zeros((240, 424, 3), dtype=np.uint8)
    report = detect_charuco_frame(
        frame_bgr,
        camera_matrix=_camera_matrix(frame_bgr.shape[1], frame_bgr.shape[0]),
        distortion=np.zeros(5, dtype=np.float64),
    )

    assert report["accepted"] is False
    assert report["markers"] == 0
    assert report["charuco_corners"] == 0
    assert report["blocking_reasons"] == ["NO_TARGET_MARKERS"]


def test_robot_preflight_report_is_explicitly_local_and_non_recording():
    from go2_mapping_lio.calibration_preflight import preflight_result

    result = preflight_result(
        camera_report={
            "accepted": True,
            "markers": 53,
            "charuco_corners": 84,
            "blocking_reasons": [],
        },
        lidar_header_advancing=True,
    )

    assert result["status"] == "READY_FOR_DIAGNOSTIC_CALIBRATION_CAPTURE"
    assert result["frame_eligible"] is True
    assert result["recording_started"] is False
    assert result["upload_started"] is False
    assert result["calibration_solved"] is False


def test_robot_preflight_runner_only_subscribes_and_never_starts_capture_or_upload():
    runner = ROOT / "tools" / "run_go2_calibration_preflight.sh"

    assert runner.is_file()
    text = runner.read_text(encoding="utf-8")
    assert "calibration_preflight.py" in text
    assert "rosbag" not in text
    assert "upload" not in text.lower()
    assert "192.0.2.10" not in text


def test_robot_preflight_runner_can_fall_back_to_the_robot_user_python():
    runner = ROOT / "tools" / "run_go2_calibration_preflight.sh"
    text = runner.read_text(encoding="utf-8")

    assert "calibration-preflight-venv/bin/python" in text
    assert 'python_bin=$(command -v python3)' in text
    assert '"$python_bin" -c' in text


def test_robot_preflight_runner_sources_foxy_under_nounset_shells():
    runner = ROOT / "tools" / "run_go2_calibration_preflight.sh"
    text = runner.read_text(encoding="utf-8")

    guard = 'export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}'
    assert guard in text
    assert text.index(guard) < text.index("source /opt/ros/foxy/setup.bash")
    assert text.index("set +u") < text.index("source /opt/ros/foxy/setup.bash")
    assert text.index("source /unitree/module/graph_pid_ws/install/setup.bash") < text.index("set -u")


def test_robot_release_includes_the_local_preflight_runner():
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")

    assert "run_go2_calibration_preflight.sh" in deploy


def test_robot_release_compiles_the_local_preflight_before_activation():
    deploy = (ROOT / "tools" / "deploy_go2_mapping_robot.ps1").read_text(encoding="utf-8")
    entrypoint = "mapping/robot/go2_mapping_lio/go2_mapping_lio/calibration_preflight.py"

    assert f"test -s {entrypoint}" in deploy
    assert f"python3 -m py_compile {entrypoint}" in deploy
    assert deploy.index(f"python3 -m py_compile {entrypoint}") < deploy.index("mv -Tf")


def test_robot_preflight_constructs_the_header_progress_gate_by_keyword():
    source = (ROBOT_PACKAGE / "go2_mapping_lio" / "calibration_preflight.py").read_text(encoding="utf-8")

    assert "HeaderProgressGate(minimum_progress_ms=args.minimum_header_progress_ms)" in source
