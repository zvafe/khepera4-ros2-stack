import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("khepera_3dgs")
    default_config = os.path.join(pkg_share, "config", "3dgs_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_config,
        description="Path to parameters YAML file for khepera_3dgs",
    )

    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value="output/khepera_3dgs",
        description="Target directory for 3DGS outputs and PLY files",
    )

    node = Node(
        package="khepera_3dgs",
        executable="pipeline_node",
        name="khepera_3dgs_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"output_dir": LaunchConfiguration("output_dir")},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        output_dir_arg,
        node,
    ])
