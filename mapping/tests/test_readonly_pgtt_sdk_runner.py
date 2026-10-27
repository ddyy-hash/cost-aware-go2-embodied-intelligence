from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sdk_runner_isolated_from_ros_and_lowcmd():
    text = (ROOT / "mapping" / "robot" / "go2_mapping_locomotion" / "readonly_pgtt_sdk_runner.py").read_text(encoding="utf-8")
    assert "unitree_sdk2py" not in text
    assert "rt/lowcmd" not in text
    assert "NumpyPGTTPolicy" in text
    assert "pgtt-heightmap.bin" in text


def test_readonly_runner_launcher_is_not_a_control_entrypoint():
    text = (ROOT / "tools" / "run_go2_pgtt_readonly.sh").read_text(encoding="utf-8")
    assert "readonly_pgtt_sdk_runner" in text
    assert "lowcmd" not in text.lower()
    assert "publishes" in text.lower()
    assert "source /opt/ros" not in text
    assert "LD_LIBRARY_PATH" in text
    assert "cyclonedds_ws/install/cyclonedds" in text
    runner = (ROOT / "mapping" / "robot" / "go2_mapping_locomotion" / "readonly_pgtt_sdk_runner.py").read_text(encoding="utf-8")
    assert "--samples" in runner
