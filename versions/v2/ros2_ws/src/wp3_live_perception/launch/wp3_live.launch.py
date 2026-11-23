"""Launch file for display-only wp3_live_node (attaches to existing camera publishers; zero MoveIt)."""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("wp3_live_perception"))
    default_cfg = str(pkg_share / "config" / "printed_jenga_live.json")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        DeclareLaunchArgument("policy", default_value="merged_corroborated"),
        DeclareLaunchArgument("wrist_ns", default_value="/perception/wrist"),
        DeclareLaunchArgument("side_ns", default_value="/perception/tripod"),
        DeclareLaunchArgument("use_side_verifier", default_value="true"),
        DeclareLaunchArgument("publish_overlay", default_value="true"),
        DeclareLaunchArgument("save_dir", default_value=""),
        DeclareLaunchArgument("rate_hz", default_value="15.0"),
        Node(
            package="wp3_live_perception",
            executable="wp3_live_node",
            name="wp3_live_node",
            output="screen",
            parameters=[{
                "config": LaunchConfiguration("config"),
                "policy": LaunchConfiguration("policy"),
                "wrist_ns": LaunchConfiguration("wrist_ns"),
                "side_ns": LaunchConfiguration("side_ns"),
                "use_side_verifier": LaunchConfiguration("use_side_verifier"),
                "publish_overlay": LaunchConfiguration("publish_overlay"),
                "save_dir": LaunchConfiguration("save_dir"),
                "rate_hz": LaunchConfiguration("rate_hz"),
            }],
        ),
    ])
