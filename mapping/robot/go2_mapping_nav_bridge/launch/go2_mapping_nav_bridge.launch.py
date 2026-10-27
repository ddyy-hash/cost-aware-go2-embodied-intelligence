"""Launch the Go2 mission navigation executor with explicit runtime bounds."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    navigation_root = LaunchConfiguration("navigation_root")
    poll_seconds = LaunchConfiguration("poll_seconds")
    timeout_seconds = LaunchConfiguration("timeout_seconds")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "navigation_root",
                default_value="/home/unitree/go2_mapping/runtime/navigation",
                description="Directory containing the mission navigation bridge state.",
            ),
            DeclareLaunchArgument(
                "poll_seconds",
                default_value="1.0",
                description="Delay between navigation bridge state polls.",
            ),
            DeclareLaunchArgument(
                "timeout_seconds",
                default_value="8.0",
                description="Timeout for Nav2 action-server and goal operations.",
            ),
            Node(
                package="go2_mapping_nav_bridge",
                executable="navigation_executor",
                name="go2_mapping_nav_bridge",
                output="screen",
                arguments=[
                    "--navigation-root",
                    navigation_root,
                    "--poll-seconds",
                    poll_seconds,
                    "--timeout-seconds",
                    timeout_seconds,
                ],
            ),
        ]
    )

