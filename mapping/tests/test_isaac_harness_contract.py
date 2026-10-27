import numpy as np
import pytest
from pathlib import Path

from tools.run_isaac_go2_pgtt_smoke import (
    DEFAULT_DRIVE_DAMPING,
    DEFAULT_DRIVE_FORCE_LIMIT,
    DEFAULT_DRIVE_STIFFNESS,
    DEFAULT_TERRAIN_FRICTION,
    DEFAULT_TERRAIN_RESTITUTION,
    analytic_heightmap,
    is_pgtt_foot_collision,
    is_pgtt_leg_collision,
    local_forward_displacement,
    quaternion_inverse_rotate,
)
from tools.run_isaac_go2_pgtt_smoke import DEFAULT_PGTT_STAIRS_MJCF
from tools.run_isaac_pgtt_acceptance_matrix import DEFAULT_STAIR_START_X, DEFAULT_URDF_PARENT_ROOT_HEIGHT
from tools.run_isaac_go2_pgtt_smoke import build_pgtt_observation


def test_analytic_heightmap_is_flat_and_finite_on_flat_terrain():
    heightmap = analytic_heightmap(0.0, "flat")
    assert heightmap.shape == (11, 9)
    assert np.isfinite(heightmap).all()
    assert np.all(heightmap == 0.0)


def test_analytic_heightmap_moves_stair_profile_with_robot():
    at_start = analytic_heightmap(0.0, "up")
    after_first_step = analytic_heightmap(0.35, "up")
    assert at_start[0, 0] == pytest.approx(0.16)
    assert after_first_step[0, 0] == pytest.approx(0.32)
    assert np.all(after_first_step[:, 0] == after_first_step[:, -1])


def test_analytic_heightmap_reverses_profile_for_down_terrain():
    up = analytic_heightmap(0.0, "up")
    down = analytic_heightmap(0.0, "down")
    assert down[0, 0] == pytest.approx(up[0, 0] * 5)


def test_analytic_heightmap_rejects_unknown_terrain():
    with pytest.raises(ValueError, match="terrain"):
        analytic_heightmap(0.0, "unknown")


def test_pgtt_observation_preserves_imu_angular_velocity():
    observation = build_pgtt_observation(
        angular_velocity=np.asarray([0.11, -0.22, 0.33], dtype=np.float32),
        gravity=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        joint_error=np.zeros(12, dtype=np.float32),
        joint_velocity=np.zeros(12, dtype=np.float32),
        phase_observation=np.zeros(8, dtype=np.float32),
        height_observation=np.zeros(99, dtype=np.float32),
        gait_frequency=2.0,
        last_action=np.zeros(12, dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(observation[:3], [0.11, -0.22, 0.33])
    np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])


def test_isaac_world_angular_velocity_is_converted_to_body_frame():
    body_velocity = quaternion_inverse_rotate(
        np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(body_velocity, [0.0, -1.0, 0.0], atol=1e-6)


def test_isaac_runner_uses_pgtt_drive_defaults():
    assert DEFAULT_DRIVE_STIFFNESS == 50.0
    assert DEFAULT_DRIVE_DAMPING == 0.5
    assert DEFAULT_DRIVE_FORCE_LIMIT == 35.55


def test_isaac_matrix_names_the_urdf_parent_root_height():
    assert DEFAULT_URDF_PARENT_ROOT_HEIGHT == 0.20


def test_isaac_episode_accepts_explicit_start_x():
    import inspect

    from tools.run_isaac_go2_pgtt_smoke import run_episode

    assert "start_x" in inspect.signature(run_episode).parameters


def test_isaac_matrix_gives_stairs_an_approach_distance():
    assert DEFAULT_STAIR_START_X == -0.80


def test_isaac_terrain_material_matches_pgtt_contact_contract():
    assert DEFAULT_TERRAIN_FRICTION == 0.6
    assert DEFAULT_TERRAIN_RESTITUTION == 0.0


def test_pgtt_collision_filter_keeps_only_named_foot_spheres():
    assert is_pgtt_foot_collision("/Go2/go2_description/Geometry/FL_foot/sphere")
    assert is_pgtt_foot_collision("/Go2/go2_description/Geometry/RR_foot/sphere")
    assert not is_pgtt_foot_collision("/Go2/go2_description/Geometry/FL_thigh/cylinder")
    assert not is_pgtt_foot_collision("/Go2/go2_description/Geometry/FL_foot/cylinder")
    assert is_pgtt_leg_collision("/Go2/go2_description/Geometry/FL_thigh/cylinder")
    assert is_pgtt_leg_collision("/Go2/go2_description/Geometry/FL_calf/sphere")
    assert not is_pgtt_leg_collision("/Go2/go2_description/Geometry/base/cylinder")
    assert is_pgtt_foot_collision("/Go2/Geometry/base/FL_hip/FL_thigh/FL_calf/FL")


def test_isaac_episode_exposes_shared_stair_geometry_parameters():
    import inspect

    from tools.run_isaac_go2_pgtt_smoke import run_episode

    parameters = inspect.signature(run_episode).parameters
    assert "step_height" in parameters
    assert "step_tread" in parameters
    assert "physical_terrain" in parameters
    assert "start_yaw" in parameters
    assert "action_smoothing" in parameters
    assert "policy_backend" in parameters


def test_isaac_runtime_allows_only_pgtt_backend():
    from tools.run_isaac_go2_pgtt_smoke import validate_policy_backend

    assert validate_policy_backend("pgtt") == "pgtt"
    with pytest.raises(ValueError, match="Genesis backend is disabled"):
        validate_policy_backend("genesis_stair")


def test_isaac_visual_mode_uses_a_window_and_rendering():
    from tools.run_isaac_go2_pgtt_smoke import simulation_app_config

    assert simulation_app_config(False) == {"headless": True, "renderer": "None", "physics_gpu": False}
    assert simulation_app_config(True) == {"headless": False, "renderer": "RayTracedLighting", "physics_gpu": False}
    assert simulation_app_config(False, livestream=True) == {"headless": True, "renderer": "RayTracedLighting", "physics_gpu": False}


def test_traversal_displacement_uses_initial_body_heading():
    assert local_forward_displacement([0.0, 0.0], [0.0, 3.0], np.pi / 2) == pytest.approx(3.0)


def test_analytic_stair_heightmap_accepts_heading():
    from tools.run_isaac_go2_pgtt_smoke import analytic_heightmap

    assert analytic_heightmap(1.0, "up", robot_yaw=np.pi).shape == (11, 9)


def test_isaac_stair_centers_match_official_pgtt_xml():
    from tools.run_isaac_go2_pgtt_smoke import stair_step_center_x

    assert stair_step_center_x(0, 0.35) == pytest.approx(0.35)
    assert stair_step_center_x(4, 0.35) == pytest.approx(1.75)


def test_pgtt_stair_scene_is_native_mjcf():
    assert DEFAULT_PGTT_STAIRS_MJCF.is_file()
    text = DEFAULT_PGTT_STAIRS_MJCF.read_text(encoding="utf-8")
    assert '<include file="go2_mjx_feetonly.xml"/>' in text
    assert text.count('<geom type="box"') == 6
    assert '<body name="floor"' in text


def test_mjcf_importer_preserves_non_robot_terrain_shapes():
    importer = (Path(__file__).parents[2] / "tools" / "import_pgtt_mjcf_wsl.py").read_text(encoding="utf-8")
    assert 'prim.GetTypeName() in {"Cube", "Plane"}' in importer
    assert "not is_robot_geometry" in importer


def test_isaac_matrix_can_reuse_native_asset_modes():
    import inspect

    from tools.run_isaac_pgtt_acceptance_matrix import run_matrix

    parameters = inspect.signature(run_matrix).parameters
    assert "start_yaw" in parameters
    assert "feet_only_collision" in parameters
    assert "physical_terrain" in parameters
    assert "action_smoothing" in parameters


def test_go2_joint_limit_contract_rejects_out_of_range_positions():
    from tools.run_isaac_go2_pgtt_smoke import within_go2_joint_limits

    assert within_go2_joint_limits(np.asarray([0.0] * 4 + [0.0] * 4 + [-1.8] * 4))
    invalid = np.asarray([0.0] * 8 + [-3.0] + [-1.8] * 3)
    assert not within_go2_joint_limits(invalid)


def test_action_smoothing_feedback_is_the_applied_action():
    from tools.run_isaac_go2_pgtt_smoke import smooth_action

    np.testing.assert_allclose(smooth_action([1.0] * 2, [0.0] * 2, 0.5), [0.5] * 2)


def test_go2_target_clipping_respects_joint_limits():
    from tools.run_isaac_go2_pgtt_smoke import clip_go2_joint_targets

    clipped = clip_go2_joint_targets(np.asarray([0.0] * 8 + [-3.0, 0.0, -1.8, -1.8]))
    assert clipped[8] == pytest.approx(-2.7227)
    assert clipped[9] == pytest.approx(-0.83776)
