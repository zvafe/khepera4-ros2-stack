import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    maneuver_arg = DeclareLaunchArgument(
        'maneuver',
        default_value='forward',
        description='Maneuver: forward, rotate, or circle'
    )
    speed_arg = DeclareLaunchArgument('linear_speed', default_value='0.15', description='Linear speed (m/s)')
    angular_arg = DeclareLaunchArgument('angular_speed', default_value='1.0', description='Angular speed (rad/s)')
    duration_arg = DeclareLaunchArgument('duration', default_value='4.0', description='Duration (s)')

    benchmark_node = Node(
        package='khepera4_control',
        executable='open_loop_benchmark',
        name='open_loop_benchmark',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'maneuver': LaunchConfiguration('maneuver'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'duration': LaunchConfiguration('duration')
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
            'auto_save_seconds': 5.0
        }]
    )

    return LaunchDescription([
        maneuver_arg,
        speed_arg,
        angular_arg,
        duration_arg,
        benchmark_node,
        telemetry_node
    ])
