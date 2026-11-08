"""Launch file for Khepera IV Simulation with integrated 3DGS Reconstruction Pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description combining Gazebo simulation, mapping, and online 3DGS node."""
    pkg_desc = get_package_share_directory("khepera4_description")
    pkg_control = get_package_share_directory("khepera4_control")
    pkg_3dgs = get_package_share_directory("khepera_3dgs")

    mapping_launch_path = os.path.join(pkg_control, "launch", "mapping.launch.py")
    default_3dgs_config = os.path.join(pkg_3dgs, "config", "3dgs_params.yaml")

    # Launch arguments
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 visualizer",
    )

    dashboard_arg = DeclareLaunchArgument(
        "dashboard",
        default_value="true",
        description="Launch Qt Control Station dashboard",
    )

    auto_explore_arg = DeclareLaunchArgument(
        "auto_explore",
        default_value="false",
        description="Enable autonomous exploration motion",
    )

    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value="output/khepera_3dgs",
        description="Target output directory for 3DGS PLY and depth data",
    )

    num_iterations_arg = DeclareLaunchArgument(
        "iterations",
        default_value="2000",
        description="Optimization iterations for 3DGS training",
    )

    train_on_shutdown_arg = DeclareLaunchArgument(
        "train_on_shutdown",
        default_value="true",
        description="Automatically trigger 3DGS training when simulation closes",
    )

    # 1. Base Simulation & Mapping Launch (Gazebo Harmonic + TF + Mapper)
    mapping_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mapping_launch_path),
        launch_arguments={
            "mode": "sim",
            "dashboard": LaunchConfiguration("dashboard"),
            "rviz": LaunchConfiguration("rviz"),
            "auto_explore": LaunchConfiguration("auto_explore"),
        }.items(),
    )

    # 2. Online 3DGS Reconstruction Node
    khepera_3dgs_node = Node(
        package="khepera_3dgs",
        executable="pipeline_node",
        name="khepera_3dgs_node",
        output="screen",
        parameters=[
            default_3dgs_config,
            {
                "use_sim_time": True,
                "output_dir": LaunchConfiguration("output_dir"),
                "num_iterations": LaunchConfiguration("iterations"),
                "train_on_shutdown": LaunchConfiguration("train_on_shutdown"),
                "max_height_m": 0.40,
            },
        ],
    )

    return LaunchDescription([
        rviz_arg,
        dashboard_arg,
        auto_explore_arg,
        output_dir_arg,
        num_iterations_arg,
        train_on_shutdown_arg,
        mapping_sim_launch,
        khepera_3dgs_node,
    ])
