import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    k1_ip_arg = DeclareLaunchArgument('k1_ip', default_value='192.168.1.101', description='IP for khepera_1 (base_link)')
    k2_ip_arg = DeclareLaunchArgument('k2_ip', default_value='192.168.1.102', description='IP for khepera_2 (extender)')
    k3_ip_arg = DeclareLaunchArgument('k3_ip', default_value='192.168.1.103', description='IP for khepera_3 (explorer)')
    mock_arg = DeclareLaunchArgument('mock', default_value='false', description='Run in mock hardware mode for testing')
    port_arg = DeclareLaunchArgument('port', default_value='5000', description='TCP port on robots')

    robots = [
        ('khepera_1', 'base_link', LaunchConfiguration('k1_ip'), 0.0, 0.0),
        ('khepera_2', 'extender',  LaunchConfiguration('k2_ip'), 0.6, 0.0),
        ('khepera_3', 'explorer',  LaunchConfiguration('k3_ip'), 1.2, 0.0),
    ]

    nodes = []

    for name, role, ip, init_x, init_y in robots:
        # Hardware bridge for each physical robot
        bridge_node = Node(
            package='khepera4_control',
            executable='khepera4_hardware_bridge',
            name='khepera4_hardware_bridge',
            namespace=name,
            output='screen',
            parameters=[{
                'robot_name': name,
                'role': role,
                'robot_ip': ip,
                'robot_port': LaunchConfiguration('port'),
                'protocol': 'kteam_ascii',
                'publish_tf': True,
                'mock': LaunchConfiguration('mock'),
                'poll_rate': 25.0
            }]
        )

        # Static transform connecting world to robot odom
        static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'world_to_{name}_odom',
            arguments=[
                '--x', str(init_x),
                '--y', str(init_y),
                '--z', '0.0',
                '--yaw', '0.0',
                '--pitch', '0.0',
                '--roll', '0.0',
                '--frame-id', 'world',
                '--child-frame-id', f'{name}/odom'
            ]
        )

        nodes.extend([bridge_node, static_tf])

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

    return LaunchDescription([
        k1_ip_arg,
        k2_ip_arg,
        k3_ip_arg,
        mock_arg,
        port_arg
    ] + nodes)
