#!/usr/bin/env python3
"""
Mapping & Exploration Launch File for Khepera IV Mobile Robot.

Launches:
1. Gazebo Harmonic Simulation & Sensor Bridges (khepera4_sim.launch.py)
   with optional headless operation and rich exploration world by default.
2. Khepera IV Multi-Sensor Spatial Mapping Engine (khepera4_mapper):
   - 360° Virtual LaserScan (/khepera4/scan)
   - 2D Probabilistic Occupancy Grid (/map)
   - 2.5D Terrain Elevation Map (/elevation_map)
   - Surface & Cliff Hazard Map (/surface_map, /cliff_alert)
   - Persistence services (/save_map, /load_map, /clear_map)
3. Autonomous Reactive Explorer (reactive_explorer) [Optional: auto_explore:=true]
4. Qt Control Station Dashboard (control_dashboard) [Optional: dashboard:=true]
5. RViz2 3D Spatial Visualizer [Optional: rviz:=true]
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_desc = get_package_share_directory('khepera4_description')
    sim_launch_path = os.path.join(pkg_desc, 'launch', 'khepera4_sim.launch.py')

    # Launch arguments
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo Harmonic in headless mode (server only)'
    )

    auto_explore_arg = DeclareLaunchArgument(
        'auto_explore',
        default_value='false',
        description='Launch autonomous reactive explorer (true/false)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_desc, 'worlds', 'khepera4_exploration_world.sdf'),
        description='Path to SDF world file to load in Gazebo Harmonic'
    )

    dashboard_arg = DeclareLaunchArgument(
        'dashboard',
        default_value='true',
        description='Launch Qt Control Station dashboard'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Launch RViz2 with mapping profile (true/false)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # 1. Include Gazebo Harmonic Simulation Launch
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # 2. Multi-Sensor Spatial Mapping Engine
    mapper_node = Node(
        package='khepera4_control',
        executable='khepera4_mapper',
        name='khepera4_mapper',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
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
        }]
    )

    # 3. Autonomous Reactive Explorer Node (Conditional)
    explorer_node = Node(
        package='khepera4_control',
        executable='reactive_explorer',
        name='reactive_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
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

    # 4. Dedicated Environment Mapping & Exploration Dashboard (Conditional)
    dashboard_node = Node(
        package='khepera4_control',
        executable='mapping_dashboard',
        name='mapping_dashboard',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('dashboard'))
    )

    # 5. RViz2 Visualizer (Conditional)
    rviz_config_path = os.path.join(pkg_desc, 'config', 'khepera4_mapping.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_mapping',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        headless_arg,
        auto_explore_arg,
        world_arg,
        dashboard_arg,
        rviz_arg,
        use_sim_time_arg,
        sim_launch,
        mapper_node,
        explorer_node,
        dashboard_node,
        rviz_node
    ])
