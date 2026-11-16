# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch physical Khepera IV hardware bridge with visual pursuit perception stack."""

import os
from typing import Tuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _create_physical_pursuit_arguments(
    default_params: str,
) -> Tuple[DeclareLaunchArgument, ...]:
    """Declare launch arguments for physical Khepera IV visual pursuit."""
    return (
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.101",
            description="IP address of the physical Khepera IV on the hotspot/lab network.",
        ),
        DeclareLaunchArgument(
            "robot_port",
            default_value="5000",
            description="TCP port of the onboard control daemon (default: 5000).",
        ),
        DeclareLaunchArgument(
            "camera_port",
            default_value="5001",
            description="TCP port of GStreamer camera streaming daemon (default: 5001).",
        ),
        DeclareLaunchArgument(
            "robot_name",
            default_value="khepera_1",
            description="Robot identifier namespace.",
        ),
        DeclareLaunchArgument(
            "mock",
            default_value="false",
            description="Set true to run hardware bridge in kinematic mock mode.",
        ),
        DeclareLaunchArgument(
            "dashboard",
            default_value="true",
            description="Launch Qt mapping and visual pursuit dashboard.",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Launch RViz2 3D visualizer.",
        ),
        DeclareLaunchArgument(
            "auto_explore",
            default_value="true",
            description="Launch autonomous reactive explorer for target search.",
        ),
        DeclareLaunchArgument(
            "launch_controller",
            default_value="true",
            description="Launch IBVS kinematic controller node.",
        ),
        DeclareLaunchArgument(
            "launch_supervisor",
            default_value="true",
            description="Launch Behavior Tree supervisor and motion arbiter node.",
        ),
        DeclareLaunchArgument(
            "target_image_path",
            default_value="",
            description="Path to target reference image for Mode B verification.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to ROS 2 perception pipeline parameters YAML file.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="auto",
            description="Compute device for neural networks (auto, cuda:0, cpu).",
        ),
    )


def _create_physical_pursuit_includes(
    physical_launch: str,
    pursuit_launch: str,
) -> Tuple[IncludeLaunchDescription, IncludeLaunchDescription]:
    """Create include actions for physical hardware bridge and perception pipeline."""
    hardware_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(physical_launch),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "robot_port": LaunchConfiguration("robot_port"),
            "camera_port": LaunchConfiguration("camera_port"),
            "robot_name": LaunchConfiguration("robot_name"),
            "mock": LaunchConfiguration("mock"),
            "dashboard": LaunchConfiguration("dashboard"),
            "rviz": LaunchConfiguration("rviz"),
            "auto_explore": LaunchConfiguration("auto_explore"),
            "camera": "true",
            "mapper": "true",
            "cmd_vel_topic": "/explorer/cmd_vel",
        }.items(),
    )

    perception_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(pursuit_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "use_sim_time": "false",
            "launch_controller": LaunchConfiguration("launch_controller"),
            "launch_supervisor": LaunchConfiguration("launch_supervisor"),
            "target_image_path": LaunchConfiguration("target_image_path"),
        }.items(),
    )

    return (hardware_include, perception_include)


def generate_launch_description() -> LaunchDescription:
    """Generate end-to-end physical Khepera IV visual pursuit launch description."""
    pkg_control = get_package_share_directory("khepera4_control")
    pkg_pursuit = get_package_share_directory("khepera_visual_pursuit")

    physical_launch = os.path.join(
        pkg_control, "launch", "single_robot_physical.launch.py"
    )
    pursuit_launch = os.path.join(
        pkg_pursuit, "launch", "perception_pipeline.launch.py"
    )
    default_params = os.path.join(
        pkg_pursuit, "config", "perception_params.yaml"
    )

    args = _create_physical_pursuit_arguments(default_params)
    hardware_inc, perception_inc = _create_physical_pursuit_includes(
        physical_launch, pursuit_launch
    )

    return LaunchDescription(list(args) + [hardware_inc, perception_inc])
