"""Bring up Nav2 and the mission bridge on the robot-side ROS Foxy graph.

This launch file composes installed Nav2 nodes; it does not own locomotion,
publish velocity commands, or send a navigation goal by itself.
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from go2_mapping_nav_bridge.nav2_map_preflight import validate_nav2_map
from go2_mapping_nav_bridge.nav2_params import prepare_nav2_params


def _validate_map_action(context):
    map_file = LaunchConfiguration("map").perform(context).strip()
    if not map_file:
        raise RuntimeError("Nav2 bringup requires an explicit validated map YAML")
    validated = validate_nav2_map(map_file)
    params_file = LaunchConfiguration("params_file").perform(context).strip()
    overlay_dir = LaunchConfiguration("params_overlay_dir").perform(context).strip()
    if not params_file or not overlay_dir:
        raise RuntimeError("Nav2 bringup requires parameter source and overlay paths")
    prepared = prepare_nav2_params(params_file, validated["coordinate_frame"], overlay_dir)
    return [SetLaunchConfiguration("resolved_params_file", str(prepared))]


def generate_launch_description() -> LaunchDescription:
    nav2_bringup_launch = PathJoinSubstitution(
        [FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]
    )
    default_params = PathJoinSubstitution(
        [FindPackageShare("go2_mapping_nav_bridge"), "config", "nav2_params.yaml"]
    )
    navigation_root = LaunchConfiguration("navigation_root")
    poll_seconds = LaunchConfiguration("poll_seconds")
    timeout_seconds = LaunchConfiguration("timeout_seconds")
    params_file = LaunchConfiguration("params_file")
    resolved_params_file = LaunchConfiguration("resolved_params_file")
    map_file = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    enable_collision_monitor = LaunchConfiguration("enable_collision_monitor")

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_launch),
        launch_arguments={
            "map": map_file,
            "params_file": resolved_params_file,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "slam": "False",
            "rviz": "False",
        }.items(),
    )
    bridge = Node(
        package="go2_mapping_nav_bridge",
        executable="navigation_executor",
        name="go2_mapping_nav_bridge",
        output="screen",
        arguments=[
            "--navigation-root", navigation_root,
            "--poll-seconds", poll_seconds,
            "--timeout-seconds", timeout_seconds,
        ],
    )
    collision_monitor = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[resolved_params_file],
        condition=IfCondition(enable_collision_monitor),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value="", description="Nav2 occupancy map YAML"),
            DeclareLaunchArgument("params_file", default_value=default_params, description="Nav2 parameters"),
            DeclareLaunchArgument("resolved_params_file", default_value=params_file, description="Generated frame-consistent Nav2 parameters"),
            DeclareLaunchArgument("params_overlay_dir", default_value="/tmp/go2-mapping-nav2", description="Temporary generated parameter directory"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("enable_collision_monitor", default_value="false"),
            DeclareLaunchArgument("navigation_root", default_value="/home/unitree/go2_mapping/runtime/navigation"),
            DeclareLaunchArgument("poll_seconds", default_value="1.0"),
            DeclareLaunchArgument("timeout_seconds", default_value="8.0"),
            OpaqueFunction(function=_validate_map_action),
            nav2,
            collision_monitor,
            bridge,
        ]
    )
