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

"""Unit tests for ReactiveExplorer corner stall escape and ground IR."""

import math
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool

from khepera4_control.common import quaternion_from_euler
from khepera4_control.reactive_explorer import ReactiveExplorer


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_ground_ir_sector_integration():
    """Verify ground IR cliff/obstacle detection reduces d_front."""
    node = ReactiveExplorer()
    try:
        # All normal initial distances
        d_left, d_front, d_right = node._extract_sector_distances()
        assert d_front == 2.0
        assert d_left == 2.0
        assert d_right == 2.0

        # Inject normal floor scan on ground_ir_0 (12mm)
        normal_scan = LaserScan()
        normal_scan.ranges = [0.012]
        normal_scan.range_min = 0.001
        normal_scan.range_max = 0.06
        node.ground_ir_callback(0, normal_scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert d_f == 2.0

        # Inject cliff drop-off on ground_ir_1 (40mm > 25mm threshold)
        cliff_scan = LaserScan()
        cliff_scan.ranges = [0.040]
        cliff_scan.range_min = 0.001
        cliff_scan.range_max = 0.06
        node.ground_ir_callback(1, cliff_scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert d_f <= 0.05
        assert d_l <= 0.05

        # Inject obstacle contact on ground_ir_2 (3mm < 5mm threshold)
        obs_scan = LaserScan()
        obs_scan.ranges = [0.003]
        obs_scan.range_min = 0.001
        obs_scan.range_max = 0.06
        node.ground_ir_callback(2, obs_scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert d_f <= 0.05
        assert d_r <= 0.05

    finally:
        node.destroy_node()


def test_corner_stall_detection_and_180_escape():
    """Verify prolonged linear commanding with near-zero disp triggers 180."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE

        # Set robot at (0, 0), facing East (yaw = 0.0)
        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        node.odom_callback(odom)

        imu = Imu()
        imu.orientation = odom.pose.pose.orientation
        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = 9.81
        node.imu_callback(imu)

        # Simulate robot wedged against a corner wall
        now = time.time()
        node.last_commanded_v = 0.10  # Commanded > 0.05 m/s

        node.pos_history.clear()
        for dt_back in range(15, -1, -1):
            t_samp = now - (dt_back * 0.1)
            node.pos_history.append((t_samp, 0.0, 0.0))

        node.accel_history.clear()
        for _ in range(25):
            node.accel_history.append(0.0)

        # First control loop tick detects stall and starts stall timer
        node.control_loop()
        assert node.stall_start_time is not None

        # Advance stall timer past 1.2s threshold
        node.stall_start_time = now - 1.3

        # Next control loop tick must trigger STATE_REVERSE_TURN
        node.control_loop()
        assert node.state == node.STATE_REVERSE_TURN
        assert math.isclose(node.reverse_turn_initial_yaw, 0.0, abs_tol=1e-3)

        # Simulate robot turning until yaw reaches 175° after 1.0s backup
        node.state_start_time = time.time() - 1.2
        rad_175 = math.radians(175.0)
        qx_t, qy_t, qz_t, qw_t = quaternion_from_euler(0.0, 0.0, rad_175)
        imu.orientation.x = qx_t
        imu.orientation.y = qy_t
        imu.orientation.z = qz_t
        imu.orientation.w = qw_t
        node.imu_callback(imu)

        # Should complete 180° escape and transition back to STATE_CRUISE
        node.control_loop()
        assert node.state == node.STATE_CRUISE

    finally:
        node.destroy_node()


def test_virtual_scan_sector_extraction():
    """Verify 360-sample virtual scan maps correctly into front, left, and right sectors."""
    node = ReactiveExplorer()
    try:
        scan = LaserScan()
        scan.angle_min = -math.pi
        scan.angle_max = math.pi - (2.0 * math.pi / 360.0)
        scan.angle_increment = 2.0 * math.pi / 360.0
        scan.range_min = 0.02
        scan.range_max = 2.0
        # Initialize all ranges clear (2.0m)
        scan.ranges = [2.0] * 360

        # Inject obstacle at front (index 180, angle = 0.0 rad) at 0.15m
        scan.ranges[180] = 0.15
        node.scan_callback(scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert math.isclose(d_f, 0.15, abs_tol=1e-3)
        assert d_l >= 1.9 and d_r >= 1.9

        # Inject obstacle on left (index 270, angle = +pi/2 rad) at 0.12m
        scan.ranges[180] = 2.0
        scan.ranges[270] = 0.12
        node.scan_callback(scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert math.isclose(d_l, 0.12, abs_tol=1e-3)
        assert d_f >= 1.9 and d_r >= 1.9

        # Inject obstacle on right (index 90, angle = -pi/2 rad) at 0.10m
        scan.ranges[270] = 2.0
        scan.ranges[90] = 0.10
        node.scan_callback(scan)

        d_l, d_f, d_r = node._extract_sector_distances()
        assert math.isclose(d_r, 0.10, abs_tol=1e-3)
        assert d_f >= 1.9 and d_l >= 1.9

    finally:
        node.destroy_node()


def test_proactive_obstacle_stopping_distance():
    """Verify that when an obstacle is within safe_dist (<=0.18m), linear speed is 0."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.last_commanded_v = 0.10

        # Inject obstacle in front at 0.15m (< 0.18m safe_dist threshold)
        scan = LaserScan()
        scan.ranges = [0.15]
        scan.range_min = 0.002
        scan.range_max = 0.25
        node.ir_callback(0, scan)
        node.ir_callback(7, scan)

        node.control_loop()
        # Must command ZERO forward linear speed to prevent collision and slip!
        assert node.last_commanded_v == 0.0

    finally:
        node.destroy_node()


def test_frontier_heading_guidance():
    """Verify that occupancy grid analysis produces an exploration heading towards unknown space."""
    node = ReactiveExplorer()
    try:
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0
        node.has_odom = True

        # Construct a 200x200 grid (6m x 6m, res 0.03m, origin -3, -3)
        grid_msg = OccupancyGrid()
        grid_msg.info.resolution = 0.03
        grid_msg.info.width = 200
        grid_msg.info.height = 200
        grid_msg.info.origin.position.x = -3.0
        grid_msg.info.origin.position.y = -3.0

        # Entire arena explored (0 = free) except the East area (col > 120) which is unknown (-1)
        grid = np.zeros((200, 200), dtype=np.int8)
        grid[:, 120:] = -1  # East is unknown frontier
        grid_msg.data = grid.flatten().tolist()

        node.map_callback(grid_msg)
        heading = node._compute_frontier_heading(time.time())

        assert heading is not None
        # East corresponds to heading ~ 0.0 rad (|heading| < 0.5 rad)
        assert abs(heading) < 0.6, f'Expected heading towards East (~0 rad), got {heading}'

    finally:
        node.destroy_node()


def test_perimeter_repulsion_near_walls():
    """Verify that robot close to arena perimeter walls experiences inward repulsive torque."""
    node = ReactiveExplorer()
    try:
        node.has_odom = True
        node.current_yaw = 0.0  # Facing East

        # Robot at x = +2.25m (close to East wall at +2.405m)
        node.current_x = 2.25
        node.current_y = 0.0

        w_perim = node._compute_perimeter_repulsion()
        # Repulsion pointing West (pi rad) when facing East (0 rad) should steer away (+ or - torque)
        assert abs(w_perim) > 0.0

    finally:
        node.destroy_node()


def test_smooth_acceleration_anti_slip():
    """Verify linear acceleration is smoothly rate-limited from standstill to prevent wheel slip."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.last_commanded_v = 0.0  # At rest

        # All sensors clear
        node.control_loop()
        # Should not instantly jump to full speed 0.12 m/s; should be rate-limited
        assert node.last_commanded_v <= 0.030
        assert node.last_commanded_v > 0.0

    finally:
        node.destroy_node()


def test_user_goal_setting_and_guidance():
    """Verify setting a user goal steers robot towards it."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0  # Facing East

        # Target directly North (0, 1.0) -> +pi/2 heading
        node.set_user_goal(0.0, 1.0)
        assert node.active_goal == (0.0, 1.0)

        node.control_loop()
        # Angular speed must steer left (+Z) towards North
        assert node.last_commanded_v > 0.0
        # angular.z should be positive (turning left toward North)
        assert node.state == node.STATE_CRUISE

    finally:
        node.destroy_node()


def test_user_path_waypoints_queue():
    """Verify multiple waypoints in user path are queued and loaded sequentially."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0

        pts = [(0.5, 0.5), (1.0, 1.0), (1.5, 0.5)]
        node.set_user_path(pts)

        assert node.active_goal == (0.5, 0.5)
        assert len(node.user_waypoints) == 2

        # Clear path
        node.clear_user_path()
        assert node.active_goal is None
        assert len(node.user_waypoints) == 0

    finally:
        node.destroy_node()


def test_state_inspect_at_goal():
    """Verify reaching goal within 0.22m triggers STATE_INSPECT and rotates to scan."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 1.0
        node.current_y = 1.0
        node.current_yaw = 0.0

        # Goal is at (1.05, 1.05), distance ~ 0.07m <= 0.22m
        node.set_user_goal(1.05, 1.05)

        node.control_loop()
        # Must enter STATE_INSPECT
        assert node.state == node.STATE_INSPECT
        assert node.last_commanded_v == 0.0

        # While inspecting (< inspect_duration), commands zero linear speed and rotational sweep
        node.state_start_time = time.time()
        node.control_loop()
        assert node.state == node.STATE_INSPECT
        assert node.last_commanded_v == 0.0

        # Once inspect_duration passes, transitions back to STATE_CRUISE
        node.state_start_time = time.time() - (node.inspect_duration + 0.5)
        node.control_loop()
        assert node.state == node.STATE_CRUISE

    finally:
        node.destroy_node()


def test_stall_condition_no_false_positive_at_moderate_speed():
    """Verify that moving at 0.06 m/s in open space does NOT trigger a false stall."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0
        node.last_commanded_v = 0.06

        now = time.time()
        # Robot traveled 0.015m in 0.25s (speed = 0.06 m/s)
        node.pos_history.clear()
        node.pos_history.append((now - 0.25, -0.015, 0.0))
        node.pos_history.append((now, 0.0, 0.0))

        # Open space, no obstacles near front
        node.control_loop()
        assert node.stall_start_time is None
        assert node.state == node.STATE_CRUISE

    finally:
        node.destroy_node()


def test_user_path_smooth_advancement_without_spin_or_inspect():
    """Verify user-drawn path waypoints advance smoothly without stopping or spinning in inspect."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0

        # Drawn path with 3 points, each 8cm apart
        pts = [(0.08, 0.0), (0.16, 0.0), (0.24, 0.0)]
        node.set_user_path(pts)

        # Initial active goal should be the second point or first point depending on lookahead
        assert node.state == node.STATE_CRUISE
        node.control_loop()

        # Must NOT enter STATE_INSPECT at intermediate waypoints
        assert node.state == node.STATE_CRUISE
        assert node.last_commanded_v > 0.0
        assert not node.is_single_user_goal
    finally:
        node.destroy_node()


def test_steer_hysteresis_prevents_chattering():
    """Verify steer direction does not flip back and forth when sensor noise fluctuates symmetrically."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.front_steer_dir = 1.0

        # Front obstacle present, left and right distances almost identical (difference < 0.02)
        # Tick 1: left slightly smaller
        d_left = 0.300
        d_right = 0.305
        # In control loop, front_steer_dir should remain 1.0 because abs(diff) = 0.005 < 0.02
        if abs(d_left - d_right) > 0.02:
            node.front_steer_dir = 1.0 if d_left >= d_right else -1.0
        assert node.front_steer_dir == 1.0

        # Tick 2: right slightly smaller
        d_left = 0.305
        d_right = 0.300
        if abs(d_left - d_right) > 0.02:
            node.front_steer_dir = 1.0 if d_left >= d_right else -1.0
        assert node.front_steer_dir == 1.0  # Hysteresis maintained!
    finally:
        node.destroy_node()


def test_mutual_exclusion_pursuit_active_suppresses_cmd_vel():
    """Verify that pursuit engages active exploration until target is sighted, then halts and yields."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE

        published_cmds = []
        node.cmd_pub.publish = lambda m: published_cmds.append(m)

        # Baseline: normal exploration commands are published
        node.control_loop()
        assert len(published_cmds) == 1
        published_cmds.clear()

        # Engage visual pursuit: explorer enters active search mode
        node.pursuit_enable_callback(Bool(data=True))
        assert node.pursuit_active is True
        assert node.enabled is True
        assert node.target_in_sight is False

        # Active exploration search continues to drive robot while searching
        node.control_loop()
        assert len(published_cmds) == 1
        assert published_cmds[0].linear.x > 0.0
        published_cmds.clear()

        # Target acquired in sight: explorer must immediately halt and yield control
        node.target_status_callback(Bool(data=True))
        assert node.target_in_sight is True
        assert len(published_cmds) == 1
        assert published_cmds[0].linear.x == 0.0
        assert published_cmds[0].angular.z == 0.0
        published_cmds.clear()

        # Successive control loop ticks while target in sight must suppress publishing
        for _ in range(5):
            node.control_loop()
        assert len(published_cmds) == 0

        # Target lost from sight: active exploration resumes search
        node.target_status_callback(Bool(data=False))
        assert node.target_in_sight is False
        node.control_loop()
        assert len(published_cmds) == 1
        assert published_cmds[0].linear.x > 0.0
        published_cmds.clear()

        # Disengage visual pursuit: explorer disengages
        node.pursuit_enable_callback(Bool(data=False))
        assert node.pursuit_active is False
        assert node.enabled is False
    finally:
        node.destroy_node()


def test_dynamic_speed_scaling_via_topics():
    """Verify speed limits update dynamically when receiving Twist commands."""
    node = ReactiveExplorer()
    try:
        # Initial default speeds
        assert math.isclose(node.linear_speed, 0.12, abs_tol=1e-3)
        assert math.isclose(node.turn_speed, 0.90, abs_tol=1e-3)

        # Send speed update via /explorer/set_speed
        twist_msg = Twist()
        twist_msg.linear.x = 0.22
        twist_msg.angular.z = 1.60
        node.speed_callback(twist_msg)

        assert math.isclose(node.linear_speed, 0.22, abs_tol=1e-3)
        assert math.isclose(node.turn_speed, 1.60, abs_tol=1e-3)
        assert node.max_turn_speed >= 2.40
    finally:
        node.destroy_node()


def test_dynamic_speed_scaling_via_parameters():
    """Verify speed limits update dynamically via ROS 2 parameter reconfigurations."""
    node = ReactiveExplorer()
    try:
        # Reconfigure using limit parameter names
        res = node.set_parameters([
            Parameter('linear_speed_limit', Parameter.Type.DOUBLE, 0.25),
            Parameter('angular_speed_limit', Parameter.Type.DOUBLE, 1.80),
        ])
        assert all(r.successful for r in res)
        assert math.isclose(node.linear_speed, 0.25, abs_tol=1e-3)
        assert math.isclose(node.turn_speed, 1.80, abs_tol=1e-3)

        # Reconfigure using standard parameter names
        res2 = node.set_parameters([
            Parameter('linear_speed', Parameter.Type.DOUBLE, 0.18),
            Parameter('turn_speed', Parameter.Type.DOUBLE, 1.40),
        ])
        assert all(r.successful for r in res2)
        assert math.isclose(node.linear_speed, 0.18, abs_tol=1e-3)
        assert math.isclose(node.turn_speed, 1.40, abs_tol=1e-3)
    finally:
        node.destroy_node()


def test_initial_360_survey_on_map_spawn():
    """Verify robot performs quick 360° survey upon spawn in empty map before exploring."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0
        node.last_commanded_v = 0.0
        assert not node.initial_spin_done

        # First control loop tick on empty map enters STATE_INITIAL_SPIN
        node.control_loop()
        assert node.state == node.STATE_INITIAL_SPIN
        assert node.last_commanded_v == 0.0

        # Simulate 360° rotation in 4 quarter-turn steps
        for step_deg in (90.0, 180.0, 270.0, 360.0):
            node.current_yaw = math.radians(step_deg % 360.0)
            node.control_loop()

        # Once 360° is completed, transitions back to STATE_CRUISE and marks initial_spin_done
        assert node.initial_spin_done
        assert node.state == node.STATE_CRUISE
    finally:
        node.destroy_node()


def test_frontier_target_selection_from_grid():
    """Verify true frontier transition between free and unknown space is selected."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.has_odom = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_yaw = 0.0

        grid_msg = OccupancyGrid()
        grid_msg.info.resolution = 0.05
        grid_msg.info.width = 100
        grid_msg.info.height = 100
        grid_msg.info.origin.position.x = -2.5
        grid_msg.info.origin.position.y = -2.5

        # Robot at (0, 0) [col 50, row 50]. Free space around robot up to x = +1.0m (col 70).
        # Unknown space (-1) East of x = +1.0m (col > 70).
        grid = np.zeros((100, 100), dtype=np.int8)
        grid[:, 70:] = -1
        grid_msg.data = grid.flatten().tolist()
        node.map_callback(grid_msg)

        frontier = node._find_map_frontier_target()
        assert frontier is not None
        # Frontier point must be in free space near the boundary (x ~ 0.90m to 1.05m, |y| <= 2.2m)
        fx, fy = frontier
        assert 0.60 <= fx <= 1.20
        assert abs(fy) <= 2.20
    finally:
        node.destroy_node()

