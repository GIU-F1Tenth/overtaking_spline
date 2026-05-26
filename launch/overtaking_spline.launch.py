"""Standalone launch for the overtaking_spline planner.

For integration into the f1tenth_stack bringup, copy this config into
src/giu_f1t_system/f1tenth_stack/config/overtaking_spline.yaml and add an
equivalent Node block guarded by a use_overtaking_spline launch arg.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("overtaking_spline")
    default_config = os.path.join(pkg_share, "config", "overtaking_spline.yaml")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to overtaking_spline parameter file.",
    )

    node = Node(
        package="overtaking_spline",
        executable="overtaking_spline_node",
        name="overtaking_spline_node",
        parameters=[LaunchConfiguration("config")],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([config_arg, node])
