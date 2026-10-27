from pathlib import Path


def test_visual_runner_exists_and_declares_non_authoritative_artifacts():
    script = Path(__file__).parents[2] / "tools" / "run_mujoco_visual_sim.py"
    text = script.read_text(encoding="utf-8")
    assert "mujoco.Renderer" in text
    assert "transport_authorized" in text
    assert "simulation_backend" in text
