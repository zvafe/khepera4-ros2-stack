import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    robots = [
        ('khepera_1', 'base_link', 0.0, 0.0),
        ('khepera_2', 'extender',  0.6, 0.0),
        ('khepera_3', 'explorer',  1.2, 0.0),
    ]

    nodes = []

    # Closed-loop controller for each robot
    for name, role, x, y in robots:
        controller_node = Node(
            package='khepera4_control',
            executable='closed_loop_controller',
            name='closed_loop_controller',
            namespace=name,
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_name': name,
                'mode': 'step',
                'goal_x': x + 0.5,
                'goal_y': y,
                'goal_yaw': 0.0,
                'k_rho': 0.8,
                'k_alpha': 2.0,
                'k_beta': -0.4,
                'k_yaw': 1.5,
                'v_max': 0.25,
                'w_max': 1.8,
                'pos_tolerance': 0.03,
                'yaw_tolerance': 0.08
            }]
        )
        nodes.append(controller_node)

    # Swarm Coordinator node
    swarm_node = Node(
        package='khepera4_control',
        executable='swarm_coordinator',
        name='swarm_coordinator',
        output='screen',
        parameters=[{
            'max_comm_range': 2.0,
            'k_p_pos': 0.8,
            'k_p_yaw': 1.5,
            'v_max': 0.20,
            'w_max': 1.5,
            'default_mode': 'relay'
        }]
    )
    nodes.append(swarm_node)

    return LaunchDescription(nodes)
