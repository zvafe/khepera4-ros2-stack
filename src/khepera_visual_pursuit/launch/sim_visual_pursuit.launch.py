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

"""Launch file combining Gazebo simulation and Visual Pursuit pipeline."""

import os
from typing import Tuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _create_sim_launch_arguments(
    default_params: str, default_world: str
) -> Tuple[DeclareLaunchArgument, ...]:
    """Declare launch arguments for simulated visual pursuit environment."""
    return (
        DeclareLaunchArgument("rviz", default_value="true", description="Launch RViz2"),
        DeclareLaunchArgument("dashboard", default_value="true", description="Launch Dashboard"),
        DeclareLaunchArgument(
            "launch_controller",
            default_value="true",
            description="Launch IBVS controller",
        ),
        DeclareLaunchArgument(
            "target_image_path",
            default_value="",
            description="Path to target image",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Perception params",
        ),
        DeclareLaunchArgument(
            "auto_explore",
            default_value="false",
            description="Launch autonomous reactive explorer enabled",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run Gazebo Harmonic in headless mode (server only, saves ~20% CPU)",
        ),
        DeclareLaunchArgument(
            "launch_supervisor",
            default_value="true",
            description="Launch Behavior Tree supervisor and motion arbiter node",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="auto",
            description="Hardware device for perception models (auto, cuda:0, cpu)",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=default_world,
            description="Path to world SDF file to load in Gazebo Harmonic",
        ),
    )


def _create_sim_pursuit_actions(
    mapping_launch: str,
    pursuit_launch: str,
) -> Tuple[IncludeLaunchDescription, IncludeLaunchDescription]:
    """Instantiate simulation mapping and pursuit pipeline includes."""
    mapping_sim_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mapping_launch),
        launch_arguments={
            "mode": "sim",
            "dashboard": LaunchConfiguration("dashboard"),
            "rviz": LaunchConfiguration("rviz"),
            "auto_explore": LaunchConfiguration("auto_explore"),
            "headless": LaunchConfiguration("headless"),
            "world": LaunchConfiguration("world"),
            "cmd_vel_topic": "/explorer/cmd_vel",
        }.items(),
    )

    pursuit_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(pursuit_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "use_sim_time": "true",
            "launch_controller": LaunchConfiguration("launch_controller"),
            "launch_supervisor": LaunchConfiguration("launch_supervisor"),
            "target_image_path": LaunchConfiguration("target_image_path"),
        }.items(),
    )

    return (mapping_sim_include, pursuit_include)


def generate_launch_description() -> LaunchDescription:
    """Generate complete end-to-end simulation and visual pursuit description."""
    pkg_control = get_package_share_directory("khepera4_control")
    pkg_pursuit = get_package_share_directory("khepera_visual_pursuit")
    pkg_desc = get_package_share_directory("khepera4_description")

    mapping_launch = os.path.join(pkg_control, "launch", "mapping.launch.py")
    pursuit_launch = os.path.join(pkg_pursuit, "launch", "perception_pipeline.launch.py")
    default_params = os.path.join(pkg_pursuit, "config", "perception_params.yaml")
    default_world = os.path.join(pkg_desc, "worlds", "khepera4_exploration_world.sdf")

    args = _create_sim_launch_arguments(default_params, default_world)
    includes = _create_sim_pursuit_actions(mapping_launch, pursuit_launch)

    return LaunchDescription(list(args) + list(includes))
