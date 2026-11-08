#!/usr/bin/env python3
# Copyright 2026 Mobile Robotics Laboratory.
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

"""
Autonomous Frontier Exploration Pipeline via explore_lite, Nav2, and SLAM.

Architected for Khepera IV mobile robot in simulation and physical deployment.
Equipped with open hooks for zero-shot (OpenFrontier) and single-shot (SplatSearch) VLM seekers.
"""

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for autonomous frontier exploration stack."""
    pkg_control = get_package_share_directory('khepera4_control')
    pkg_desc = get_package_share_directory('khepera4_description')

    sim_launch_path = os.path.join(pkg_desc, 'launch', 'khepera4_sim.launch.py')
    urdf_path = os.path.join(pkg_desc, 'urdf', 'khepera4.urdf')

    default_nav2_config = os.path.join(pkg_control, 'config', 'khepera4_nav2.yaml')
    default_explore_config = os.path.join(pkg_control, 'config', 'khepera4_explore.yaml')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # --------------------------------------------------------------------------
    # Launch Arguments
    # --------------------------------------------------------------------------
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Execution mode: "sim" (Gazebo Harmonic), "physical", or "mock"'
    )

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.101',
        description='IP address of physical Khepera IV robot'
    )

    robot_port_arg = DeclareLaunchArgument(
        'robot_port',
        default_value='5000',
        description='TCP port of Khepera IV onboard daemon'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo Harmonic in headless mode'
    )

    auto_explore_arg = DeclareLaunchArgument(
        'auto_explore',
        default_value='true',
        description='Start exploration active immediately (true/false)'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_desc, 'worlds', 'khepera4_exploration_world.sdf'),
        description='Path to SDF world file'
    )

    dashboard_arg = DeclareLaunchArgument(
        'dashboard',
        default_value='true',
        description='Launch Qt Control Station dashboard (true/false)'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 with mapping profile (true/false)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='auto',
        description='Use simulation clock: "true", "false", or "auto"'
    )

    vlm_target_mode_arg = DeclareLaunchArgument(
        'vlm_target_mode',
        default_value='none',
        description='VLM Target Seeking Mode: "none", "open_frontier", "splat_search"'
    )

    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=default_nav2_config,
        description='Path to Nav2 parameters YAML'
    )

    explore_params_arg = DeclareLaunchArgument(
        'explore_params_file',
        default_value=default_explore_config,
        description='Path to explore_lite parameters YAML'
    )

    # --------------------------------------------------------------------------
    # Conditions & Expressions
    # --------------------------------------------------------------------------
    is_sim_mode = PythonExpression(["'", LaunchConfiguration('mode'), "' == 'sim'"])
    is_hardware_mode = PythonExpression([
        "'", LaunchConfiguration('mode'), "' in ['physical', 'mock']"
    ])

    effective_use_sim_time = PythonExpression([
        "'true' if ('", LaunchConfiguration('use_sim_time'), "' == 'true' or ('",
        LaunchConfiguration('use_sim_time'), "' == 'auto' and '",
        LaunchConfiguration('mode'), "' == 'sim')) else 'false'"
    ])

    # --------------------------------------------------------------------------
    # 1. Simulation / Hardware Nodes
    # --------------------------------------------------------------------------
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
            'bridge_tf': 'true',
        }.items(),
        condition=IfCondition(is_sim_mode)
    )

    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': False
        }],
        condition=IfCondition(is_hardware_mode)
    )

    hardware_bridge_node = Node(
        package='khepera4_control',
        executable='khepera4_hardware_bridge',
        name='khepera4_hardware_bridge',
        output='screen',
        parameters=[{
            'robot_name': 'khepera_1',
            'robot_ip': LaunchConfiguration('robot_ip'),
            'robot_port': LaunchConfiguration('robot_port'),
            'protocol': 'kteam_ascii',
            'publish_tf': True,
            'single_robot': True,
            'publish_scans': True,
            'mock': PythonExpression(["'", LaunchConfiguration('mode'), "' == 'mock'"]),
            'poll_rate': 25.0,
            'use_sim_time': False
        }],
        condition=IfCondition(is_hardware_mode)
    )

    # --------------------------------------------------------------------------
    # 2. Multi-Sensor Spatial Mapping Engine (Option A: Owns /map and map->odom)
    # --------------------------------------------------------------------------
    mapper_node = Node(
        package='khepera4_control',
        executable='khepera4_mapper',
        name='khepera4_mapper',
        output='screen',
        parameters=[{
            'use_sim_time': effective_use_sim_time,
            'map_resolution': 0.03,
            'map_width': 200,
            'map_height': 200,
            'map_origin_x': -3.0,
            'map_origin_y': -3.0,
            'map_frame': 'map',
            'robot_frame': 'base_footprint',
            'publish_map_tf': True,
            'publish_odom_tf': False,
            'enable_scan_matching': False,
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

    visual_orthomapper_node = Node(
        package='khepera4_control',
        executable='visual_orthomapper',
        name='visual_orthomapper',
        output='screen',
        parameters=[{
            'use_sim_time': effective_use_sim_time,
            'map_resolution': 0.03,
            'map_width': 200,
            'map_height': 200,
            'map_origin_x': -3.0,
            'map_origin_y': -3.0,
            'map_frame': 'map',
            'robot_frame': 'base_footprint',
            'odom_topic': '/khepera4/odom',
            'raw_odom_topic': '/odom',
            'image_topic': '/camera/image_raw',
            'camera_info_topic': '/camera/camera_info',
            'cliff_topic': '/cliff_alert',
            'visual_map_topic': '/visual_map',
            'visual_map_image_topic': '/visual_map/image',
            'map_publish_rate': 3.0,
            'alpha': 0.25,
            'pitch_guard_deg': 6.0
        }]
    )

    # --------------------------------------------------------------------------
    # 4. Navigation 2 (Nav2) Stack
    # --------------------------------------------------------------------------
    try:
        nav2_bringup_dir = get_package_share_directory('nav2_bringup')
        nav2_launch_path = os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        nav2_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            launch_arguments={
                'use_sim_time': effective_use_sim_time,
                'params_file': LaunchConfiguration('nav2_params_file'),
                'autostart': 'true',
            }.items()
        )
    except PackageNotFoundError:
        nav2_launch = LogInfo(
            msg='[NOTICE] nav2_bringup not installed. Run scripts to install.'
        )

    # --------------------------------------------------------------------------
    # 5. Frontier Exploration (explore_lite)
    # --------------------------------------------------------------------------
    explore_node = Node(
        package='explore_lite',
        executable='explore',
        name='explore_node',
        output='screen',
        parameters=[
            LaunchConfiguration('explore_params_file'),
            {'use_sim_time': effective_use_sim_time}
        ]
    )

    # --------------------------------------------------------------------------
    # 6. Exploration Adapter & Semantic Arbiter
    # --------------------------------------------------------------------------
    adapter_node = Node(
        package='khepera4_control',
        executable='explore_adapter',
        name='explore_adapter',
        output='screen',
        parameters=[{
            'use_sim_time': effective_use_sim_time,
            'enabled': LaunchConfiguration('auto_explore'),
            'vlm_target_mode': LaunchConfiguration('vlm_target_mode'),
            'enable_vlm_hooks': True,
            'vlm_semantic_weight': 2.0,
            'exemplar_override_threshold': 0.72,
            'cliff_backup_duration': 1.2,
            'cliff_backup_speed': -0.08,
            'cliff_turn_duration': 1.5,
            'cliff_turn_speed': 0.85,
            'control_rate': 20.0,
        }]
    )

    # --------------------------------------------------------------------------
    # 7. Dashboard & RViz2 UI
    # --------------------------------------------------------------------------
    dashboard_node = Node(
        package='khepera4_control',
        executable='mapping_dashboard',
        name='mapping_dashboard',
        output='screen',
        parameters=[{'use_sim_time': effective_use_sim_time}],
        condition=IfCondition(LaunchConfiguration('dashboard'))
    )

    rviz_config_path = os.path.join(pkg_desc, 'config', 'khepera4_mapping.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_mapping',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': effective_use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        mode_arg,
        robot_ip_arg,
        robot_port_arg,
        headless_arg,
        auto_explore_arg,
        world_arg,
        dashboard_arg,
        rviz_arg,
        use_sim_time_arg,
        vlm_target_mode_arg,
        nav2_params_arg,
        explore_params_arg,
        sim_launch,
        robot_state_pub_node,
        hardware_bridge_node,
        mapper_node,
        visual_orthomapper_node,
        nav2_launch,
        explore_node,
        adapter_node,
        dashboard_node,
        rviz_node
    ])
