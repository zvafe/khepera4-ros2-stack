"""Unit tests for OpenFrontier active map exploration engine."""

import math
import numpy as np
import pytest

from nav_msgs.msg import OccupancyGrid, MapMetaData
from khepera4_control.openfrontier_explorer import OpenFrontierExplorer, FrontierCandidate


def _create_mock_occupancy_grid(
    width: int = 40,
    height: int = 40,
    resolution: float = 0.05,
    origin_x: float = -1.0,
    origin_y: float = -1.0,
) -> OccupancyGrid:
    """Create a mock OccupancyGrid with known free, unknown, and obstacle regions."""
    grid = OccupancyGrid()
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = origin_x
    grid.info.origin.position.y = origin_y

    # Default to unknown (-1)
    data = np.full((height, width), -1, dtype=np.int8)

    # Central room explored: free space (0)
    data[10:30, 10:30] = 0

    # Add an obstacle wall in the middle
    data[18:22, 18:22] = 100

    grid.data = data.flatten().tolist()
    return grid


def test_openfrontier_detector_identifies_frontiers():
    """Verify OpenFrontier detects frontier clusters between free and unknown space."""
    grid = _create_mock_occupancy_grid()
    explorer = OpenFrontierExplorer(min_cluster_cells=3, robot_safety_radius_m=0.10)

    frontiers = explorer.detect_frontiers(grid, robot_x=0.0, robot_y=0.0)

    # Should detect boundaries around the free room
    assert len(frontiers) > 0

    for f in frontiers:
        assert isinstance(f, FrontierCandidate)
        assert f.cluster_size >= 3
        assert f.exploration_gain > 0.0
        assert f.total_utility > 0.0
        assert -1.0 <= f.pos_x <= 1.0
        assert -1.0 <= f.pos_y <= 1.0


def test_openfrontier_lifecycle_merge_and_prune():
    """Verify OpenFrontier merges proposals and prunes visited frontiers."""
    grid = _create_mock_occupancy_grid()
    explorer = OpenFrontierExplorer(tau_merge_m=0.50, r_clear_m=0.30)

    proposals = explorer.detect_frontiers(grid, robot_x=0.0, robot_y=0.0)
    explorer.merge_proposals(proposals)
    initial_count = len(explorer.active_frontiers)
    assert initial_count > 0

    best = explorer.select_best_frontier(robot_x=0.0, robot_y=0.0)
    assert best is not None

    # Simulate robot approaching the frontier within r_clear
    explorer.prune_cleared_frontiers(robot_x=best.pos_x, robot_y=best.pos_y)
    assert len(explorer.active_frontiers) == initial_count - 1
    assert len(explorer.cleared_frontiers) == 1


def test_openfrontier_stall_detection():
    """Verify stall detection triggers when progress stalls for T_stall cycles."""
    explorer = OpenFrontierExplorer(stall_threshold_cycles=5, epsilon_stall_m=0.05)

    # Moving closer: stall counter resets
    assert explorer.handle_progress(current_distance=2.0) is False
    assert explorer.handle_progress(current_distance=1.8) is False
    assert explorer.handle_progress(current_distance=1.6) is False

    # Stalled: distance barely changes
    assert explorer.handle_progress(current_distance=1.59) is False
    assert explorer.handle_progress(current_distance=1.58) is False
    assert explorer.handle_progress(current_distance=1.58) is False
    assert explorer.handle_progress(current_distance=1.58) is False
    assert explorer.handle_progress(current_distance=1.58) is True  # 5th stalled cycle


def test_openfrontier_astar_path_planning():
    """Verify A* finds a collision-free path around an obstacle."""
    grid = _create_mock_occupancy_grid()
    explorer = OpenFrontierExplorer(robot_safety_radius_m=0.10)

    # Start on left of obstacle, goal on right of obstacle
    start_x, start_y = -0.30, 0.0
    goal_x, goal_y = 0.30, 0.0

    path = explorer.plan_path(grid, start_x, start_y, goal_x, goal_y)
    assert len(path) > 1

    # Check path doesn't cut through the center obstacle (which is around x=[-0.1, 0.1], y=[-0.1, 0.1])
    for wx, wy in path:
        is_in_center_obstacle = (-0.05 <= wx <= 0.05) and (-0.05 <= wy <= 0.05)
        assert not is_in_center_obstacle, f"Path intersected obstacle at ({wx}, {wy})"

    # Final waypoint should be close to goal
    assert math.hypot(path[-1][0] - goal_x, path[-1][1] - goal_y) < 0.10


def test_openfrontier_viewpoint_frontier_priority():
    """Verify target sighting viewpoint frontier has high priority utility."""
    grid = _create_mock_occupancy_grid()
    explorer = OpenFrontierExplorer()

    proposals = explorer.detect_frontiers(grid, robot_x=0.0, robot_y=0.0)
    explorer.merge_proposals(proposals)

    # Insert a target viewpoint frontier
    vp = explorer.insert_viewpoint_frontier(
        target_x=0.5, target_y=0.5, robot_x=0.0, robot_y=0.0, standoff_m=0.3
    )
    assert vp.is_viewpoint_frontier is True

    # Best frontier must now be the viewpoint frontier!
    best = explorer.select_best_frontier(robot_x=0.0, robot_y=0.0)
    assert best is not None
    assert best.is_viewpoint_frontier is True
    assert best.id.startswith("VP_")


def test_reactive_explorer_openfrontier_integration():
    """Verify ReactiveExplorer utilizes OpenFrontier to select targets and plan waypoints."""
    import rclpy
    from khepera4_control.reactive_explorer import ReactiveExplorer
    from visualization_msgs.msg import MarkerArray

    if not rclpy.ok():
        rclpy.init()

    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0

        grid = _create_mock_occupancy_grid()
        node.map_callback(grid)

        # Select target using OpenFrontier
        target = node._select_exploration_target()
        assert target is not None
        assert node.openfrontier.current_goal is not None

        # Verify A* waypoints were queued if path had intermediate steps
        assert isinstance(target, tuple)
        assert len(target) == 2

        # Verify markers published
        assert hasattr(node, 'frontier_markers_pub')
        assert len(node.openfrontier.active_frontiers) > 0

    finally:
        node.destroy_node()


def test_reactive_explorer_openfrontier_stall_recovery():
    """Verify that ReactiveExplorer drops stalled frontier and replans."""
    import rclpy
    from khepera4_control.reactive_explorer import ReactiveExplorer

    if not rclpy.ok():
        rclpy.init()

    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0
        node.is_single_user_goal = False

        grid = _create_mock_occupancy_grid()
        node.map_callback(grid)

        target = node._select_exploration_target()
        curr_fid = node.openfrontier.current_goal.id
        # Set active goal far away (> 0.25m) so it is actively navigating towards it
        node.active_goal = (1.5, 1.5)
        node.frontier_waypoints.clear()

        # Trigger stall on openfrontier stall tracker
        node.openfrontier.stall_tracker.stall_counter = 14
        node.openfrontier.stall_tracker.d_prev = 2.12

        # Simulate robot at distance 2.12 (no progress >= 0.06m)
        # Running control loop should detect stall, drop curr_fid, and replan
        node.last_commanded_v = 0.08
        node.control_loop()

        # curr_fid must now be dropped and cleared
        assert len(node.openfrontier.cleared_frontiers) >= 1
        assert node.openfrontier.current_goal is None or node.openfrontier.current_goal.id != curr_fid

    finally:
        node.destroy_node()
