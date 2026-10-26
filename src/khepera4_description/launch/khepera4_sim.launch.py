import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, SetEnvironmentVariable, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('khepera4_description')
    pkg_share_parent = os.path.dirname(pkg_share)
    urdf_path = os.path.join(pkg_share, 'urdf', 'khepera4.urdf')
    default_world_path = os.path.join(pkg_share, 'worlds', 'khepera4_world.sdf')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # 0. Set Gazebo Resource Path for mesh and world resolution
    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    combined_resource_path = f"{pkg_share_parent}:{pkg_share}:{gz_resource_path}"
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=combined_resource_path
    )

    # Launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_path,
        description='Path to the SDF world file to load in Gazebo Harmonic'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (server only)'
    )

    # 1. Gazebo Harmonic Simulation World (GUI vs Headless)
    start_gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', LaunchConfiguration('world')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')),
        on_exit=Shutdown()
    )

    start_gazebo_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', LaunchConfiguration('world')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless')),
        on_exit=Shutdown()
    )

    # 2. Robot State Publisher (Publishes /robot_description and TF)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_desc
        }]
    )

    # 3. Spawn Robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'khepera4',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.01'
        ]
    )

    # 4. Bidirectional Bridge (ROS 2 <-> Gazebo Harmonic)
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
            # 5 Ultrasonic scans (0.20m - 2.0m)
            '/khepera4/us_0/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/us_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/us_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/us_3/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/us_4/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # 8 IR proximity scans (0.002m - 0.25m)
            '/khepera4/ir_0/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_3/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_4/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_5/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_6/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ir_7/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # 4 Ground IR scans (cliff / surface detection)
            '/khepera4/ground_ir_0/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ground_ir_1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ground_ir_2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/khepera4/ground_ir_3/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_gz_resource_path,
        world_arg,
        headless_arg,
        start_gazebo_gui,
        start_gazebo_headless,
        robot_state_publisher_node,
        spawn_robot,
        bridge_node
    ])
