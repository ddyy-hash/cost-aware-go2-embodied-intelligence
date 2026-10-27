from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "mapping" / "robot" / "go2_mapping_nav_bridge"


def test_navigation_bridge_is_a_complete_ament_python_package():
    package_xml = PACKAGE_ROOT / "package.xml"
    setup_py = PACKAGE_ROOT / "setup.py"
    setup_cfg = PACKAGE_ROOT / "setup.cfg"
    resource_marker = PACKAGE_ROOT / "resource" / "go2_mapping_nav_bridge"
    launch_file = PACKAGE_ROOT / "launch" / "go2_mapping_nav_bridge.launch.py"

    assert package_xml.is_file()
    assert setup_py.is_file()
    assert setup_cfg.is_file()
    assert resource_marker.is_file()
    assert launch_file.is_file(), "ROS package must ship a launch description"

    package = ET.parse(package_xml).getroot()
    assert package.findtext("name") == "go2_mapping_nav_bridge"
    assert package.findtext("buildtool_depend") == "ament_python"
    assert package.findtext("export/build_type") == "ament_python"

    setup = setup_py.read_text(encoding="utf-8")
    assert 'glob("launch/*.launch.py")' in setup
    assert '"navigation_executor = go2_mapping_nav_bridge.navigation_executor:main"' in setup

    launch = launch_file.read_text(encoding="utf-8")
    assert "generate_launch_description" in launch
    assert "LaunchDescription" in launch
    assert 'package="go2_mapping_nav_bridge"' in launch
    assert 'executable="navigation_executor"' in launch


def test_navigation_package_exposes_nav2_bringup_profile_with_map_and_safety_arguments():
    launch_file = PACKAGE_ROOT / "launch" / "go2_nav2_bringup.launch.py"
    params_file = PACKAGE_ROOT / "config" / "nav2_params.yaml"

    assert launch_file.is_file()
    assert params_file.is_file()
    launch = launch_file.read_text(encoding="utf-8")
    params = params_file.read_text(encoding="utf-8")
    for marker in (
        "nav2_bringup",
        "bringup_launch.py",
        "map",
        "params_file",
        "use_sim_time",
        "autostart",
        "navigation_executor",
        "navigation_root",
        "collision_monitor",
    ):
        assert marker in launch or marker in params
    assert "prepare_nav2_params" in launch
    assert "resolved_params_file" in launch
    assert "coordinate_frame" in launch
    assert "SetLaunchConfiguration" in launch
    assert "cmd_vel" not in launch

    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "nav2_bringup" in package_xml
    assert "nav2_collision_monitor" in package_xml
    assert 'glob("config/*.yaml")' in setup
    assert (PACKAGE_ROOT / "go2_mapping_nav_bridge" / "nav2_map_preflight.py").is_file()
    assert (PACKAGE_ROOT / "go2_mapping_nav_bridge" / "nav2_params.py").is_file()
