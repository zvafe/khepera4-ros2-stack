import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    desc_pkg = get_package_share_directory('khepera4_description')
    sim_launch = os.path.join(desc_pkg, 'launch', 'khepera4_sim.launch.py')

    # 1. Include Simulation Launch
    sim_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch)
    )

    # 2. Control & Education Dashboard GUI
    dashboard_node = Node(
        package='khepera4_control',
        executable='control_dashboard',
        name='khepera4_dashboard',
        output='screen'
    )

    return LaunchDescription([
        sim_node,
        dashboard_node
    ])
