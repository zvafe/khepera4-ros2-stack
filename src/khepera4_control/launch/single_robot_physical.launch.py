#!/usr/bin/env python3
"""
Single Physical Khepera IV Robot Operations Launch File.
Connects directly to ONE physical Khepera IV mobile robot in the lab via Wi-Fi TCP daemon.

Launches:
1. khepera4_hardware_bridge (TCP client connecting to robot_ip:robot_port)
2. robot_state_publisher (Broadcasting physical URDF frame hierarchy)
3. khepera4_mapper (Optional, default true: building 2D occupancy grid & 2.5D elevation maps)
4. mapping_dashboard (Optional, default true: Qt graphical control station with live map and teleop)
5. rviz2 (Optional, default false: 3D spatial visualizer)
6. reactive_explorer (Optional, default false: autonomous obstacle avoidance exploration)

Usage in Lab:
  ros2 launch khepera4_control single_robot_physical.launch.py robot_ip:=192.168.1.101
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_desc = get_package_share_directory('khepera4_description')
    urdf_path = os.path.join(pkg_desc, 'urdf', 'khepera4.urdf')
    rviz_config_path = os.path.join(pkg_desc, 'config', 'khepera4_mapping.rviz')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # Launch Arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.101',
        description='IP address of the physical Khepera IV robot on the lab network'
    )

    robot_port_arg = DeclareLaunchArgument(
        'robot_port',
        default_value='5000',
        description='TCP communication port of the onboard daemon (default: 5000)'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='khepera_1',
        description='Robot identifier string'
    )

    mock_arg = DeclareLaunchArgument(
        'mock',
        default_value='false',
        description='Set true to emulate hardware without a real robot (default: false)'
    )

    dashboard_arg = DeclareLaunchArgument(
        'dashboard',
        default_value='true',
        description='Launch Qt mapping and control dashboard (default: true)'
    )

    mapper_arg = DeclareLaunchArgument(
        'mapper',
        default_value='true',
        description='Launch multi-sensor mapping engine (default: true)'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Launch RViz2 3D visualizer (default: false)'
    )

    auto_explore_arg = DeclareLaunchArgument(
        'auto_explore',
        default_value='false',
        description='Launch autonomous reactive explorer (default: false)'
    )

    # 1. Hardware Bridge Node
    bridge_node = Node(
        package='khepera4_control',
        executable='khepera4_hardware_bridge',
        name='khepera4_hardware_bridge',
        output='screen',
        parameters=[{
            'robot_name': LaunchConfiguration('robot_name'),
            'robot_ip': LaunchConfiguration('robot_ip'),
            'robot_port': LaunchConfiguration('robot_port'),
            'protocol': 'kteam_ascii',
            'publish_tf': True,
            'single_robot': True,
            'publish_scans': True,
            'mock': LaunchConfiguration('mock'),
            'poll_rate': 25.0,
            'use_sim_time': False
        }]
    )

    # 2. Robot State Publisher (TF tree from URDF)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': False
        }]
    )

    # 3. Spatial Mapping Engine
    mapper_node = Node(
        package='khepera4_control',
        executable='khepera4_mapper',
        name='khepera4_mapper',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'map_resolution': 0.03,
            'map_width': 200,
            'map_height': 200,
            'map_origin_x': -3.0,
            'map_origin_y': -3.0,
            'map_frame': 'map',
            'robot_frame': 'base_footprint',
            'odom_topic': '/odom',
            'imu_topic': '/imu/data',
            'use_imu_yaw': True,
            'scan_publish_rate': 10.0,
            'map_publish_rate': 3.0,
            'log_odds_free': 0.40,
            'log_odds_occ': 0.85,
            'log_odds_min': -5.0,
            'log_odds_max': 5.0,
            'cliff_distance_threshold': 0.025,
            'map_save_dir': 'maps'
        }],
        condition=IfCondition(LaunchConfiguration('mapper'))
    )

    # 4. Qt Graphical Mapping & Exploration Dashboard
    dashboard_node = Node(
        package='khepera4_control',
        executable='mapping_dashboard',
        name='mapping_dashboard',
        output='screen',
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('dashboard'))
    )

    # 5. RViz2 3D Visualizer
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_mapping',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    # 6. Autonomous Reactive Explorer (Optional)
    explorer_node = Node(
        package='khepera4_control',
        executable='reactive_explorer',
        name='reactive_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'enabled': True,
            'linear_speed': 0.12,
            'reverse_speed': 0.05,
            'turn_speed': 0.90,
            'max_turn_speed': 1.50,
            'obstacle_avoid_dist': 0.40,
            'obstacle_stop_dist': 0.08,
            'cliff_backup_duration': 1.2,
            'control_rate': 20.0
        }],
        condition=IfCondition(LaunchConfiguration('auto_explore'))
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        robot_name_arg,
        mock_arg,
        dashboard_arg,
        mapper_arg,
        rviz_arg,
        auto_explore_arg,
        bridge_node,
        robot_state_pub,
        mapper_node,
        dashboard_node,
        rviz_node,
        explorer_node
    ])
