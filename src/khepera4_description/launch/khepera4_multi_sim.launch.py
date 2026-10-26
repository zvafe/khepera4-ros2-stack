import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, SetEnvironmentVariable, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_robot_description(xacro_path, robot_name, role, color_rgb):
    cmd = [
        'xacro', xacro_path,
        f'robot_name:={robot_name}',
        f'role:={role}',
        f'color_rgb:={color_rgb}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout

def generate_launch_description():
    pkg_share = get_package_share_directory('khepera4_description')
    pkg_share_parent = os.path.dirname(pkg_share)
    xacro_path = os.path.join(pkg_share, 'urdf', 'khepera4.urdf.xacro')
    default_world_path = os.path.join(pkg_share, 'worlds', 'khepera4_world.sdf')

    # Environment variable for Gazebo resources
    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    combined_resource_path = f"{pkg_share_parent}:{pkg_share}:{gz_resource_path}"
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=combined_resource_path
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_path,
        description='Path to SDF world file to load in Gazebo Harmonic'
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (server only)'
    )

    # 1. Gazebo Harmonic Process
    start_gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', LaunchConfiguration('world')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    start_gazebo_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', LaunchConfiguration('world')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    # Robot specifications: (robot_name, role, color_rgb, x, y, yaw)
    robots = [
        ('khepera_1', 'base_link', '0.10 0.45 0.90', 0.0, 0.0, 0.0),
        ('khepera_2', 'extender',  '0.95 0.55 0.10', 0.6, 0.0, 0.0),
        ('khepera_3', 'explorer',  '0.10 0.85 0.35', 1.2, 0.0, 0.0),
    ]

    spawn_nodes = []
    bridge_topics = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
    ]

    for name, role, color, x, y, yaw in robots:
        desc = generate_robot_description(xacro_path, name, role, color)

        # Robot State Publisher
        rsp_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=name,
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_description': desc,
                'frame_prefix': f'{name}/'
            }]
        )

        # Gazebo Model Spawner
        spawn_node = Node(
            package='ros_gz_sim',
            executable='create',
            name=f'spawn_{name}',
            output='screen',
            arguments=[
                '-name', name,
                '-string', desc,
                '-x', str(x),
                '-y', str(y),
                '-z', '0.01',
                '-Y', str(yaw)
            ]
        )

        # Static transform connecting global world to robot odom
        static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'world_to_{name}_odom',
            arguments=[
                '--x', str(x),
                '--y', str(y),
                '--z', '0.0',
                '--yaw', str(yaw),
                '--pitch', '0.0',
                '--roll', '0.0',
                '--frame-id', 'world',
                '--child-frame-id', f'{name}/odom'
            ]
        )

        spawn_nodes.extend([rsp_node, spawn_node, static_tf])

        # Topics to bridge for this robot
        bridge_topics.extend([
            f'/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            f'/{name}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            f'/{name}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            f'/{name}/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            f'/{name}/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ])

    # Consolidated ROS 2 <-> Gazebo Bridge
    multi_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='multi_robot_parameter_bridge',
        arguments=bridge_topics,
        output='screen'
    )

    ld_actions = [
        set_gz_resource_path,
        world_arg,
        headless_arg,
        start_gazebo_gui,
        start_gazebo_headless,
        multi_bridge_node
    ] + spawn_nodes

    return LaunchDescription(ld_actions)
