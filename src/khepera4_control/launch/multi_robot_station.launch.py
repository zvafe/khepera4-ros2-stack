import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    desc_pkg = get_package_share_directory('khepera4_description')
    ctrl_pkg = get_package_share_directory('khepera4_control')

    sim_launch = os.path.join(desc_pkg, 'launch', 'khepera4_multi_sim.launch.py')
    phys_launch = os.path.join(ctrl_pkg, 'launch', 'multi_robot_physical.launch.py')

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Execution mode: "sim" (Gazebo Harmonic), "physical" (Real Robots), or "mock"'
    )
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Launch Qt Multi-Robot Control Dashboard'
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (when mode:=sim)'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(desc_pkg, 'worlds', 'khepera4_world.sdf'),
        description='Path to SDF world file to load in Gazebo Harmonic (defaults to clean empty world)'
    )

    # 1. Gazebo Harmonic Multi-Robot Simulation
    sim_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
        }.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'sim'"]))
    )

    # 2. Physical Robot Hardware Bridges
    phys_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(phys_launch),
        launch_arguments={'mock': 'false'}.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'physical'"]))
    )

    # 3. Mock Physical Hardware Bridges (for offline hardware testing)
    mock_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(phys_launch),
        launch_arguments={'mock': 'true'}.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'mock'"]))
    )

    # 4. Qt Multi-Robot Dashboard GUI
    dashboard_node = Node(
        package='khepera4_control',
        executable='control_dashboard',
        name='multi_robot_dashboard',
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    # 5. Swarm Coordinator Node (Fleet Autonomous Formation & Chain Relay)
    swarm_coordinator_node = Node(
        package='khepera4_control',
        executable='swarm_coordinator',
        name='swarm_coordinator',
        output='screen',
        parameters=[{
            'use_sim_time': PythonExpression(["'", LaunchConfiguration('mode'), "' == 'sim'"])
        }]
    )

    return LaunchDescription([
        mode_arg,
        gui_arg,
        headless_arg,
        world_arg,
        sim_include,
        phys_include,
        mock_include,
        dashboard_node,
        swarm_coordinator_node
    ])
