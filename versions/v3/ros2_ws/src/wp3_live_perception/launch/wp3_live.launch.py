"""Launch file for display-only wp3_live_node (attaches to existing camera publishers; zero MoveIt)."""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("wp3_live_perception"))
    default_cfg = str(pkg_share / "config" / "printed_jenga_live.json")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        DeclareLaunchArgument("policy", default_value="merged_corroborated"),
        DeclareLaunchArgument("wrist_ns", default_value="/perception/wrist"),
        DeclareLaunchArgument("side_ns", default_value="/perception/tripod"),
        DeclareLaunchArgument("third_ns", default_value=""),
        DeclareLaunchArgument("use_side_verifier", default_value="true"),
        DeclareLaunchArgument("publish_overlay", default_value="true"),
        DeclareLaunchArgument("save_dir", default_value=""),
        DeclareLaunchArgument("rate_hz", default_value="15.0"),
        # V3.1: V2 path by default; V3 region stage and frame skipping are explicit opt-ins.
        DeclareLaunchArgument("enable_v3_regions", default_value="false"),
        DeclareLaunchArgument("v3_solver", default_value=""),
        DeclareLaunchArgument("v3_frame_budget_ms", default_value="-1.0"),
        DeclareLaunchArgument("enable_temporal_skip", default_value="false"),
        DeclareLaunchArgument("persistent_joint_occlusion", default_value="true"),
        DeclareLaunchArgument("reanchor_after", default_value="2"),
        # V3.2: /tf_static re-read period (<= 0: cache for the node lifetime) and the per-camera plane check.
        DeclareLaunchArgument("static_tf_refresh_s", default_value="1.0"),
        DeclareLaunchArgument("calibration_check", default_value="true"),
        DeclareLaunchArgument("tracker_merge_overlapping", default_value=""),
        DeclareLaunchArgument("retire_on_observed_free", default_value=""),
        DeclareLaunchArgument("event_driven_wakeup", default_value="true"),
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
                "third_ns": ParameterValue(LaunchConfiguration("third_ns"), value_type=str),
                "use_side_verifier": LaunchConfiguration("use_side_verifier"),
                "publish_overlay": LaunchConfiguration("publish_overlay"),
                "save_dir": LaunchConfiguration("save_dir"),
                "rate_hz": LaunchConfiguration("rate_hz"),
                "enable_v3_regions": ParameterValue(LaunchConfiguration("enable_v3_regions"), value_type=bool),
                "v3_solver": ParameterValue(LaunchConfiguration("v3_solver"), value_type=str),
                "v3_frame_budget_ms": ParameterValue(LaunchConfiguration("v3_frame_budget_ms"), value_type=float),
                "enable_temporal_skip": ParameterValue(LaunchConfiguration("enable_temporal_skip"), value_type=bool),
                "persistent_joint_occlusion": ParameterValue(LaunchConfiguration("persistent_joint_occlusion"), value_type=bool),
                "reanchor_after": ParameterValue(LaunchConfiguration("reanchor_after"), value_type=int),
                "static_tf_refresh_s": ParameterValue(LaunchConfiguration("static_tf_refresh_s"), value_type=float),
                "calibration_check": ParameterValue(LaunchConfiguration("calibration_check"), value_type=bool),
                "tracker_merge_overlapping": ParameterValue(LaunchConfiguration("tracker_merge_overlapping"), value_type=str),
                "retire_on_observed_free": ParameterValue(LaunchConfiguration("retire_on_observed_free"), value_type=str),
                "event_driven_wakeup": ParameterValue(LaunchConfiguration("event_driven_wakeup"), value_type=bool),
            }],
        ),
    ])
