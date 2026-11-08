"""Launch file for unified OpenFrontier and SplatSearch hybrid navigation stack."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description with full simulation, RViz2, and hybrid navigation nodes."""
    pkg_control = get_package_share_directory("khepera4_control")
    pkg_desc = get_package_share_directory("khepera4_description")

    sim_launch_path = os.path.join(pkg_desc, "launch", "khepera4_sim.launch.py")
    default_world = os.path.join(pkg_desc, "worlds", "khepera4_exploration_world.sdf")
    rviz_config_path = os.path.join(pkg_desc, "config", "khepera4_mapping.rviz")

    # Launch Arguments
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="true",
        description="Launch Gazebo Harmonic 3D simulation."
    )

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo in headless mode."
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="SDF world file path."
    )

    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 interactive visualizer."
    )

    dashboard_arg = DeclareLaunchArgument(
        "dashboard",
        default_value="true",
        description="Launch Qt Control Station dashboard."
    )

    mapper_arg = DeclareLaunchArgument(
        "mapper",
        default_value="true",
        description="Launch multi-sensor occupancy grid mapper."
    )

    enable_3dgs_arg = DeclareLaunchArgument(
        "enable_3dgs",
        default_value="false",
        description="Toggle 3D Gaussian Splatting and viewpoint synthesis branch."
    )

    target_desc_arg = DeclareLaunchArgument(
        "target_description",
        default_value="fire extinguisher",
        description="Target object description for language-grounded exploration."
    )

    vlm_mode_arg = DeclareLaunchArgument(
        "vlm_mode",
        default_value="hybrid",
        description="VLM execution mode: 'cloud' (Gemini), 'local', or 'hybrid' (failover)."
    )

    cloud_model_arg = DeclareLaunchArgument(
        "cloud_model",
        default_value="gemini-2.5-flash",
        description="Gemini model identifier for cloud reasoning."
    )

    local_model_arg = DeclareLaunchArgument(
        "local_model",
        default_value="gemma4:e2b",
        description="Local model identifier for Ollama inference."
    )

    # Launch configurations
    sim_conf = LaunchConfiguration("sim")
    headless_conf = LaunchConfiguration("headless")
    world_conf = LaunchConfiguration("world")
    rviz_conf = LaunchConfiguration("rviz")
    dashboard_conf = LaunchConfiguration("dashboard")
    mapper_conf = LaunchConfiguration("mapper")
    enable_3dgs = LaunchConfiguration("enable_3dgs")
    target_desc = LaunchConfiguration("target_description")
    vlm_mode = LaunchConfiguration("vlm_mode")
    cloud_model = LaunchConfiguration("cloud_model")
    local_model = LaunchConfiguration("local_model")

    # 1. Gazebo Harmonic Simulation & Robot Bridge
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            "headless": headless_conf,
            "world": world_conf,
            "bridge_tf": "true",
        }.items(),
        condition=IfCondition(sim_conf)
    )

    # 2. 2D Spatial Occupancy Mapper
    mapper_node = Node(
        package="khepera4_control",
        executable="khepera4_mapper",
        name="khepera4_mapper",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "map_resolution": 0.03,
            "map_width": 200,
            "map_height": 200,
            "map_origin_x": -3.0,
            "map_origin_y": -3.0,
            "map_frame": "map",
            "robot_frame": "base_footprint",
            "publish_map_tf": True,
            "odom_topic": "/odom",
            "imu_topic": "/imu/data",
            "use_imu_yaw": True,
            "scan_publish_rate": 10.0,
            "map_publish_rate": 3.0
        }],
        condition=IfCondition(mapper_conf)
    )

    # 3. Visual Orthomapper
    visual_orthomapper_node = Node(
        package="khepera4_control",
        executable="visual_orthomapper",
        name="visual_orthomapper",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "map_resolution": 0.03,
            "map_width": 200,
            "map_height": 200,
            "map_origin_x": -3.0,
            "map_origin_y": -3.0,
            "map_frame": "map",
            "robot_frame": "base_footprint",
            "image_topic": "/camera/image_raw",
            "camera_info_topic": "/camera/camera_info"
        }],
        condition=IfCondition(mapper_conf)
    )

    # 4. RViz2 Visualizer
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz_conf)
    )

    # 5. Dashboard
    dashboard_node = Node(
        package="khepera4_control",
        executable="mapping_dashboard",
        name="mapping_dashboard",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(dashboard_conf)
    )

    # 6. Semantic Reasoner Node
    reasoner_node = Node(
        package="semantic_reasoner",
        executable="reasoner_node",
        name="semantic_reasoner_node",
        output="screen",
        parameters=[{
            "mode": vlm_mode,
            "cloud_model": cloud_model,
            "local_model": local_model,
            "local_endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "timeout_sec": 8.0
        }]
    )

    # 7. OpenFrontier Manager Node
    frontier_manager_node = Node(
        package="openfrontier_ros",
        executable="frontier_manager_node",
        name="frontier_manager_node",
        output="screen",
        parameters=[{
            "target_description": target_desc,
            "evaluation_interval_sec": 2.0,
            "tau_merge_m": 1.8,
            "r_clear_m": 0.5,
            "r_goal_m": 0.8
        }]
    )

    # 8. ActiveSplat 3DGS Node
    gaussian_mapper_node = Node(
        package="activesplat_ros",
        executable="gaussian_mapper_node",
        name="gaussian_mapper_node",
        output="screen",
        parameters=[{
            "enable_3dgs": enable_3dgs,
            "sampling_radius_m": 1.0,
            "num_rendered_views": 8,
            "loftr_match_threshold": 25
        }]
    )

    # 9. Master Hybrid Coordinator Node
    hybrid_coordinator_node = Node(
        package="khepera_hybrid_nav",
        executable="hybrid_coordinator_node",
        name="hybrid_coordinator_node",
        output="screen",
        parameters=[{
            "enable_3dgs": enable_3dgs,
            "target_description": target_desc,
            "weight_distance": 0.25,
            "weight_semantic": 0.45,
            "weight_visual_3dgs": 0.30
        }]
    )

    return LaunchDescription([
        sim_arg,
        headless_arg,
        world_arg,
        rviz_arg,
        dashboard_arg,
        mapper_arg,
        enable_3dgs_arg,
        target_desc_arg,
        vlm_mode_arg,
        cloud_model_arg,
        local_model_arg,
        sim_launch,
        mapper_node,
        visual_orthomapper_node,
        rviz_node,
        dashboard_node,
        reasoner_node,
        frontier_manager_node,
        gaussian_mapper_node,
        hybrid_coordinator_node
    ])
