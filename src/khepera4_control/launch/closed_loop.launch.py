import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='step',
        description='Control mode: step, waypoint, square, or circle'
    )
    goal_x_arg = DeclareLaunchArgument('goal_x', default_value='1.0', description='Goal X position (m)')
    goal_y_arg = DeclareLaunchArgument('goal_y', default_value='0.0', description='Goal Y position (m)')
    goal_yaw_arg = DeclareLaunchArgument('goal_yaw', default_value='0.0', description='Goal yaw orientation (rad)')

    controller_node = Node(
        package='khepera4_control',
        executable='closed_loop_controller',
        name='closed_loop_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'mode': LaunchConfiguration('mode'),
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_yaw': LaunchConfiguration('goal_yaw'),
            'k_rho': 0.8,
            'k_alpha': 2.0,
            'k_beta': -0.4,
            'k_yaw': 1.5,
            'v_max': 0.25,
            'w_max': 1.8,
            'pos_tolerance': 0.02,
            'yaw_tolerance': 0.05
        }]
    )

    telemetry_node = Node(
        package='khepera4_control',
        executable='telemetry_logger',
        name='telemetry_logger',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'output_dir': os.environ.get('KHEPERA_TELEMETRY_DIR', os.path.join(os.getcwd(), 'telemetry')),
            'auto_save_seconds': 10.0
        }]
    )

    return LaunchDescription([
        mode_arg,
        goal_x_arg,
        goal_y_arg,
        goal_yaw_arg,
        controller_node,
        telemetry_node
    ])
