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

"""Launch file for Khepera IV visual perception and pursuit pipeline."""

from dataclasses import dataclass
import os
from typing import Tuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


@dataclass(frozen=True)
class PipelineLaunchConfigs:
    """Encapsulates launch configuration substitutions."""

    params_file: LaunchConfiguration
    use_sim_time: LaunchConfiguration
    launch_controller: LaunchConfiguration
    target_image_path: LaunchConfiguration
    launch_supervisor: LaunchConfiguration


def _create_launch_arguments(default_params: str) -> Tuple[DeclareLaunchArgument, ...]:
    """Create launch configuration arguments."""
    return (
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to ROS 2 perception pipeline parameters YAML file.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation clock if true.",
        ),
        DeclareLaunchArgument(
            "launch_controller",
            default_value="true",
            description="Launch IBVS kinematic controller node if true.",
        ),
        DeclareLaunchArgument(
            "launch_supervisor",
            default_value="true",
            description="Launch Behavior Tree supervisor and motion arbiter node if true.",
        ),
        DeclareLaunchArgument(
            "target_image_path",
            default_value="",
            description="Path to target reference image for verification.",
        ),
    )


def _create_pipeline_nodes(configs: PipelineLaunchConfigs) -> Tuple[Node, ...]:
    """Create perception and pursuit ROS 2 nodes."""
    common_params = [configs.params_file, {"use_sim_time": configs.use_sim_time}]
    detector_params = [
        configs.params_file,
        {
            "use_sim_time": configs.use_sim_time,
            "target_image_path": configs.target_image_path,
        },
    ]

    detector_node = Node(
        package="khepera_visual_pursuit",
        executable="object_detector_verifier_node",
        name="object_detector_verifier_node",
        output="screen",
        parameters=detector_params,
    )
    tracker_node = Node(
        package="khepera_visual_pursuit",
        executable="lightglue_tracker_node",
        name="lightglue_tracker_node",
        output="screen",
        parameters=detector_params,
    )
    controller_node = Node(
        package="khepera_visual_pursuit",
        executable="ibvs_controller_node",
        name="ibvs_controller_node",
        output="screen",
        parameters=common_params,
        condition=IfCondition(configs.launch_controller),
    )
    supervisor_node = Node(
        package="khepera_visual_pursuit",
        executable="pursuit_bt_supervisor_node",
        name="pursuit_bt_supervisor_node",
        output="screen",
        parameters=common_params,
        condition=IfCondition(configs.launch_supervisor),
    )
    return (detector_node, tracker_node, controller_node, supervisor_node)


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for visual perception pipeline nodes."""
    pkg_share = get_package_share_directory("khepera_visual_pursuit")
    default_params = os.path.join(pkg_share, "config", "perception_params.yaml")

    args = _create_launch_arguments(default_params)
    configs = PipelineLaunchConfigs(
        params_file=LaunchConfiguration("params_file"),
        use_sim_time=LaunchConfiguration("use_sim_time"),
        launch_controller=LaunchConfiguration("launch_controller"),
        target_image_path=LaunchConfiguration("target_image_path"),
        launch_supervisor=LaunchConfiguration("launch_supervisor"),
    )
    nodes = _create_pipeline_nodes(configs)

    return LaunchDescription(list(args) + list(nodes))
