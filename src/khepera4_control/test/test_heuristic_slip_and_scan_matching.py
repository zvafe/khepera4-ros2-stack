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

"""
Unit tests for heuristic slip filtering and 2D correlative scan matching.

Covers:
- Phase 1: Multi-sensor rate-of-change validation (IR range gradient vs. encoder delta).
- Phase 1: IMU linear acceleration gating (body-X dynamic accel vs. wheel accel).
- Phase 1: Dynamic accelerometer gravity compensation on inclined ramps.
- Phase 2: 2D Correlative Scan Matcher (CSM) pose refinement and recovery from drift.
- Phase 2: CSM fallback gating on sparse obstacle maps and invalid scans.
- Parameter overrides and publish_map_tf gating for modular SLAM integration.
"""

import math

from geometry_msgs.msg import Twist
from khepera4_control.common import quaternion_from_euler
from khepera4_control.khepera4_mapper import Khepera4Mapper
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu, LaserScan


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan_msg(
    distance: float,
    min_range: float = 0.002,
    max_range: float = 0.20
) -> LaserScan:
    """Create a single-ray LaserScan message with a given distance reading."""
    scan = LaserScan()
    scan.ranges = [float(distance)]
    scan.range_min = float(min_range)
    scan.range_max = float(max_range)
    scan.angle_min = 0.0
    scan.angle_max = 0.0
    scan.angle_increment = 0.0
    return scan


def create_odom_msg(x: float, y: float, yaw: float, sec: int, nanosec: int = 0) -> Odometry:
    """Create an Odometry message with given pose and timestamp."""
    odom = Odometry()
    odom.header.frame_id = 'odom'
    odom.child_frame_id = 'base_footprint'
    odom.header.stamp.sec = int(sec)
    odom.header.stamp.nanosec = int(nanosec)
    odom.pose.pose.position.x = float(x)
    odom.pose.pose.position.y = float(y)
    odom.pose.pose.position.z = 0.0
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    odom.pose.pose.orientation.x = qx
    odom.pose.pose.orientation.y = qy
    odom.pose.pose.orientation.z = qz
    odom.pose.pose.orientation.w = qw
    return odom


def create_imu_msg(
    ax: float,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
    sec: int = 0,
    nanosec: int = 0
) -> Imu:
    """Create an Imu message with specified body acceleration and RPY orientation."""
    imu = Imu()
    imu.header.frame_id = 'base_footprint'
    imu.header.stamp.sec = int(sec)
    imu.header.stamp.nanosec = int(nanosec)
    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
    imu.orientation.x = qx
    imu.orientation.y = qy
    imu.orientation.z = qz
    imu.orientation.w = qw
    imu.linear_acceleration.x = float(ax)
    imu.linear_acceleration.y = 0.0
    imu.linear_acceleration.z = 9.80665 * math.cos(pitch)
    return imu


# ==============================================================================
# Phase 1: Parameter Configuration & Defaults
# ==============================================================================

def test_parameter_defaults_and_overrides():
    """Verify that default parameters match task specification and can be overridden."""
    default_node = Khepera4Mapper()
    try:
        assert default_node.enable_ir_gradient_slip_check is True
        assert default_node.enable_imu_accel_gate is True
        assert default_node.ir_gradient_window_s == pytest.approx(0.10)
        assert default_node.ir_gradient_min_cmd_v == pytest.approx(0.04)
        assert default_node.ir_gradient_min_delta_s == pytest.approx(0.002)
        assert default_node.ir_gradient_max_closing_rate == pytest.approx(0.0005)
        assert default_node.ir_gradient_proximity_thresh == pytest.approx(0.08)
        assert default_node.imu_accel_gate_threshold == pytest.approx(0.10)
        assert default_node.imu_accel_proximity_thresh == pytest.approx(0.08)
        assert default_node.enable_scan_matching is True
        assert default_node.publish_map_tf is True
        assert default_node.csm_search_dist == pytest.approx(0.03)
        assert default_node.csm_search_angle == pytest.approx(0.03)
        assert default_node.csm_min_valid_hits == 4
        assert default_node.csm_score_thresh == pytest.approx(0.20)
    finally:
        default_node.destroy_node()

    # Test parameter overrides
    custom_node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_ir_gradient_slip_check', Parameter.Type.BOOL, False),
        Parameter('enable_imu_accel_gate', Parameter.Type.BOOL, False),
        Parameter('enable_scan_matching', Parameter.Type.BOOL, False),
        Parameter('publish_map_tf', Parameter.Type.BOOL, False),
        Parameter('csm_search_dist', Parameter.Type.DOUBLE, 0.05),
    ])
    try:
        assert custom_node.enable_ir_gradient_slip_check is False
        assert custom_node.enable_imu_accel_gate is False
        assert custom_node.enable_scan_matching is False
        assert custom_node.publish_map_tf is False
        assert custom_node.csm_search_dist == pytest.approx(0.05)
    finally:
        custom_node.destroy_node()


# ==============================================================================
# Phase 1: Front IR Range Gradient Slip Detection
# ==============================================================================

def test_front_ir_gradient_slip_detected_when_stagnant():
    """Verify wheel slip is flagged when encoder advances while front distance remains stagnant."""
    node = Khepera4Mapper()
    try:
        # Feed active IMU message
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        # Command forward motion
        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        # Seed initial state at t = 10.0s, front obstacle at 0.05m
        scan_t0 = create_scan_msg(0.050)
        node.ir_callback(0, scan_t0)
        node.ir_callback(7, scan_t0)
        odom_t0 = create_odom_msg(0.0, 0.0, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        assert node.has_odom is True
        assert node.is_wheel_slipping is False

        # Advance odometry 150ms later (> 100ms window), encoder advances 12mm
        # Obstacle distance has NOT closed (still 0.050m)
        odom_t1 = create_odom_msg(0.012, 0.0, 0.0, sec=10, nanosec=150000000)
        node.odom_callback(odom_t1)

        # Verify wheel slip is detected
        assert node.is_wheel_slipping is True
        # Verify perpendicular forward penetration was clamped
        assert node.robot_x < 0.005
    finally:
        node.destroy_node()


def test_front_ir_gradient_no_slip_when_closing_normally():
    """Verify wheel slip is NOT flagged when front distance closes proportionally."""
    node = Khepera4Mapper()
    try:
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        # Initialize odometry at t = 9.90s
        odom_init = create_odom_msg(0.0, 0.0, 0.0, sec=9, nanosec=900000000)
        node.odom_callback(odom_init)

        # Pre-seed cruising velocity at t = 9.95s (v = 0.08 m/s)
        odom_pre = create_odom_msg(0.004, 0.0, 0.0, sec=9, nanosec=950000000)
        node.odom_callback(odom_pre)

        # Cruising step at t = 10.0s (v = 0.08 m/s), front obstacle at 0.060m
        scan_t0 = create_scan_msg(0.060)
        node.ir_callback(0, scan_t0)
        node.ir_callback(7, scan_t0)
        odom_t0 = create_odom_msg(0.008, 0.0, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        # Advance odometry 150ms later at constant speed (12mm in 0.15s = 0.08 m/s, so a_wheel = 0)
        # Robot physically closed distance from 60mm down to 48mm (-12mm delta)
        scan_t1 = create_scan_msg(0.048)
        node.ir_callback(0, scan_t1)
        node.ir_callback(7, scan_t1)

        odom_t1 = create_odom_msg(0.020, 0.0, 0.0, sec=10, nanosec=150000000)
        node.odom_callback(odom_t1)

        # Distance closed significantly (closing rate > 0.0005m), so no slip
        assert node.is_wheel_slipping is False
        # Position should advance cleanly
        assert node.robot_x == pytest.approx(0.020, abs=1e-3)
    finally:
        node.destroy_node()


def test_front_ir_gradient_slip_bypassed_when_disabled():
    """Verify that disabling enable_ir_gradient_slip_check ignores stagnant front distance."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_ir_gradient_slip_check', Parameter.Type.BOOL, False),
        Parameter('enable_imu_accel_gate', Parameter.Type.BOOL, False),
    ])
    try:
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        scan_t0 = create_scan_msg(0.050)
        node.ir_callback(0, scan_t0)
        node.ir_callback(7, scan_t0)
        odom_t0 = create_odom_msg(0.0, 0.0, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        odom_t1 = create_odom_msg(0.012, 0.0, 0.0, sec=10, nanosec=150000000)
        node.odom_callback(odom_t1)

        assert node.is_wheel_slipping is False
    finally:
        node.destroy_node()


# ==============================================================================
# Phase 1: IMU Accelerometer Gating & Incline Gravity Compensation
# ==============================================================================

def test_imu_accel_gate_detects_spinning_wheels():
    """Verify IMU gate detects wheel acceleration without chassis acceleration."""
    node = Khepera4Mapper()
    try:
        # Feed active IMU streaming with zero dynamic acceleration
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)
        assert node.has_imu is True
        assert node.imu_valid is True

        # Obstacle in proximity (front IR = 40mm < 80mm gate threshold)
        scan = create_scan_msg(0.040)
        node.ir_callback(0, scan)
        node.ir_callback(7, scan)

        # Seed initial odom at t = 10.0s
        odom_t0 = create_odom_msg(0.0, 0.0, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        # Step odometry at 50Hz (dt = 0.02s) with sudden jump causing high wheel acceleration
        # Step 1: slight advance at 10.02s
        odom_t1 = create_odom_msg(0.0005, 0.0, 0.0, sec=10, nanosec=20000000)
        node.odom_callback(odom_t1)

        # Stream IMU at 10.04s showing physical chassis acceleration is ~0
        imu_msg2 = create_imu_msg(ax=0.0, sec=10, nanosec=40000000)
        node.imu_callback(imu_msg2)

        # Step 2: large wheel advance jump:
        # delta_s = 0.0045m in 0.02s -> v = 0.225 m/s, a_wheel > 5 m/s^2
        odom_t2 = create_odom_msg(0.0050, 0.0, 0.0, sec=10, nanosec=40000000)
        node.odom_callback(odom_t2)

        # Verify wheel acceleration was calculated and gate triggered
        assert node.wheel_accel_x > 0.50
        assert node.is_wheel_slipping is True
    finally:
        node.destroy_node()


def test_imu_gravity_compensation_on_ramp_prevents_false_collision_shock():
    """Verify gravity compensation on 10° pitch ramp eliminates false collision shocks."""
    node = Khepera4Mapper()
    try:
        # Command cruising velocity
        cmd = Twist()
        cmd.linear.x = 0.10
        node.cmd_vel_callback(cmd)

        pitch_10deg = math.radians(10.0)
        # On a 10 deg incline at constant speed, raw accelerometer along body-X measures:
        # ax_raw = -g * sin(10 deg) = -9.80665 * 0.173648 = -1.70289 m/s^2
        ax_raw = -9.80665 * math.sin(pitch_10deg)
        assert ax_raw < -1.2  # Would falsely trigger impact without gravity compensation

        # Feed several IMU messages with 10 deg pitch to settle the low-pass filter
        for i in range(10):
            imu_msg = create_imu_msg(
                ax=ax_raw,
                pitch=pitch_10deg,
                sec=1,
                nanosec=i * 20000000
            )
            node.imu_callback(imu_msg)

        # Verify dynamic forward acceleration is compensated back to ~0 m/s^2
        assert abs(node.ax_dynamic_filtered) < 0.15
        # Verify collision impact shock was NOT falsely triggered
        assert node.last_impact_time == 0.0

        # Now test a real collision shock on flat terrain: pitch = 0, ax = -2.5 m/s^2
        shock_imu = create_imu_msg(ax=-2.5, pitch=0.0, sec=2, nanosec=0)
        node.imu_callback(shock_imu)

        # Real collision shock must still be detected
        assert node.last_impact_time > 0.0
    finally:
        node.destroy_node()


# ==============================================================================
# Phase 2: 2D Correlative Scan Matcher (CSM) Pose Correction
# ==============================================================================

def test_csm_pose_refinement_recovers_translation_drift():
    """Verify 2D Correlative Scan Matcher recovers from dead-reckoning translation drift."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
    ])
    try:
        # Initialize odometry at (0.40, 0.0, 0.0)
        odom = create_odom_msg(0.40, 0.0, 0.0, sec=1)
        node.odom_callback(odom)

        # Construct a known wall at x = 0.50m:
        # col = (0.50 - (-3.0))/0.03 = 116
        col_wall = int(round((0.50 - node.origin_x) / node.res))
        # Place wall spanning y from -0.15 to +0.15
        row_min = int(round((-0.15 - node.origin_y) / node.res))
        row_max = int(round((0.15 - node.origin_y) / node.res))
        for r in range(row_min, row_max + 1):
            node.log_odds[r, col_wall] = node.l_occ * 2.0

        # Provide front sensor readings corresponding to the wall at x = 0.50m
        # Robot true position is x = 0.40m, so wall is 0.10m away from robot center
        # For US2 (sx=0.067, syaw=0.0): r = 0.10 - 0.067 = 0.033m
        scan_us2 = create_scan_msg(0.033, min_range=0.02, max_range=2.0)
        # For IR0 (sx=0.065, syaw=+0.1745 rad): r = (0.10 - 0.065) / cos(0.1745) = 0.0355m
        scan_ir0 = create_scan_msg(0.0355)
        # For IR7 (sx=0.065, syaw=-0.1745 rad): r = 0.0355m
        scan_ir7 = create_scan_msg(0.0355)
        # For US1 (sx=0.060, syaw=+0.4712 rad): r = (0.10 - 0.060) / cos(0.4712) = 0.0449m
        scan_us1 = create_scan_msg(0.0449, min_range=0.02, max_range=2.0)
        # For US3 (sx=0.060, syaw=-0.4712 rad): r = 0.0449m
        scan_us3 = create_scan_msg(0.0449, min_range=0.02, max_range=2.0)

        node.us_callback(2, scan_us2)
        node.ir_callback(0, scan_ir0)
        node.ir_callback(7, scan_ir7)
        node.us_callback(1, scan_us1)
        node.us_callback(3, scan_us3)

        # Artificially inject -2cm dead-reckoning drift into robot pose: robot_x = 0.38m
        node.robot_x = 0.38
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        # Run correlative scan matching
        refined = node.refine_pose_with_scan_matching()

        # CSM should succeed and shift robot_x forward towards 0.40m
        assert refined is True
        assert node.robot_x > 0.385
        assert node.robot_x == pytest.approx(0.40, abs=0.015)
    finally:
        node.destroy_node()


def test_csm_fallback_when_map_has_insufficient_obstacles():
    """Verify CSM aborts gracefully without pose corruption when occupancy map is unmapped."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
    ])
    try:
        odom = create_odom_msg(0.0, 0.0, 0.0, sec=1)
        node.odom_callback(odom)

        # Occupancy grid has zero occupied cells
        assert np.count_nonzero(node.log_odds >= node.l_occ) == 0

        # Feed some valid sensor hits
        scan = create_scan_msg(0.05)
        for i in range(8):
            node.ir_callback(i, scan)

        initial_pose = (node.robot_x, node.robot_y, node.robot_yaw)
        refined = node.refine_pose_with_scan_matching()

        # Must return False and leave pose untouched
        assert refined is False
        assert (node.robot_x, node.robot_y, node.robot_yaw) == initial_pose
    finally:
        node.destroy_node()


def test_csm_fallback_when_scan_has_insufficient_hits():
    """Verify CSM aborts gracefully when sensor scan has fewer hits than csm_min_valid_hits."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
    ])
    try:
        odom = create_odom_msg(0.0, 0.0, 0.0, sec=1)
        node.odom_callback(odom)

        # Populate map with obstacle cells
        node.log_odds[100:120, 100:120] = node.l_occ

        # Sensors are completely clear / out of range (all None or empty)
        for i in range(8):
            node.latest_ir[i] = None
        for i in range(5):
            node.latest_us[i] = None

        initial_pose = (node.robot_x, node.robot_y, node.robot_yaw)
        refined = node.refine_pose_with_scan_matching()

        assert refined is False
        assert (node.robot_x, node.robot_y, node.robot_yaw) == initial_pose
    finally:
        node.destroy_node()


def test_publish_map_tf_disabled_for_slam_toolbox():
    """Verify that publish_map_tf=False suppresses map TF broadcasting."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('publish_map_tf', Parameter.Type.BOOL, False),
    ])
    try:
        odom = create_odom_msg(1.0, 1.0, 0.0, sec=1)
        node.odom_callback(odom)

        # Broadcast call should exit early without error
        now_stamp = node.get_clock().now().to_msg()
        node._broadcast_map_tf(now_stamp)
        assert node.publish_map_tf is False
    finally:
        node.destroy_node()


# ==============================================================================
# Quality-Scored Ghost Wall Removal & Anti-Jitter Smoothing Tests
# ==============================================================================

def test_ghost_wall_erased_by_robot_traversal():
    """Verify that a phantom wall is completely erased when robot traverses through it."""
    node = Khepera4Mapper()
    try:
        # Set robot at x=0.50m, y=0.0m
        node.has_odom = True
        node.robot_x = 0.50
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        c, r = node.world_to_grid(0.50, 0.0)
        # Paint a ghost wall at (0.50, 0.0)
        node.log_odds[r, c] = node.l_occ + 0.5
        node.hit_count[r, c] = 5

        # Update footprint maps
        node._update_footprint_maps()

        # Cell covered by robot footprint must be cleared to free space
        assert node.log_odds[r, c] <= -node.l_free
        assert node.hit_count[r, c] == 0
    finally:
        node.destroy_node()


def test_ghost_wall_eroded_by_penetrating_rays():
    """Verify low-confidence phantom walls are eroded by penetrating free space rays."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        # Place a low-confidence ghost wall at x = 0.20m (c=107, r=100)
        c_ghost, r_ghost = node.world_to_grid(0.20, 0.0)
        node.log_odds[r_ghost, c_ghost] = node.l_occ  # 0.85
        node.hit_count[r_ghost, c_ghost] = 1

        # Ultrasonic sensor 2 (facing forward East) reads 0.50m (clear past ghost wall)
        node.us_callback(2, create_scan_msg(0.50, min_range=0.02, max_range=2.0))
        node._update_occupancy_rays()

        # The ray should penetrate and decrement the ghost cell by l_free
        assert node.log_odds[r_ghost, c_ghost] < node.l_occ
        assert node.log_odds[r_ghost, c_ghost] == pytest.approx(node.l_occ - node.l_free, abs=1e-3)
    finally:
        node.destroy_node()


def test_pose_quality_score_computation():
    """Verify that pose quality score drops during wheel slip and collision contact."""
    node = Khepera4Mapper()
    try:
        # Clean cruising state
        assert node._compute_pose_quality() == pytest.approx(1.0)

        # Slipping state (1.0 - 0.70 = 0.30)
        node.is_wheel_slipping = True
        q_slip = node._compute_pose_quality()
        assert q_slip == pytest.approx(0.30, abs=1e-4)

        # Contact state (1.0 - 0.35 = 0.65)
        node.is_wheel_slipping = False
        node.is_near_contact = True
        q_contact = node._compute_pose_quality()
        assert q_contact == pytest.approx(0.65, abs=1e-4)
    finally:
        node.destroy_node()


def test_anti_jitter_imu_yaw_preservation():
    """Verify CSM locks yaw to IMU heading when IMU is valid to eliminate heading wobble."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
    ])
    try:
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.450
        node.robot_yaw = 0.450
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.has_odom = True

        # Populate some obstacle cells in the map
        node.log_odds[95:105, 110:115] = node.l_occ

        # Run scan matching
        node.refine_pose_with_scan_matching()

        # Heading must remain strictly locked to IMU yaw without discrete yaw jumps
        assert node.robot_yaw == pytest.approx(0.450)
    finally:
        node.destroy_node()


def test_anti_jitter_alpha_smoothing_step():
    """Verify CSM applies exponential moving average smoothing to eliminate 1cm jumps."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
        Parameter('csm_alpha_smooth', Parameter.Type.DOUBLE, 0.30),
    ])
    try:
        # True robot position is at (0.40, 0.0, 0.0)
        odom = create_odom_msg(0.40, 0.0, 0.0, sec=1)
        node.odom_callback(odom)

        col_wall = int(round((0.50 - node.origin_x) / node.res))
        row_min = int(round((-0.15 - node.origin_y) / node.res))
        row_max = int(round((0.15 - node.origin_y) / node.res))
        for r in range(row_min, row_max + 1):
            node.log_odds[r, col_wall] = node.l_occ * 2.0

        # Feed sensor readings hitting the wall at x=0.50m
        scan_us2 = create_scan_msg(0.033, min_range=0.02, max_range=2.0)
        scan_ir0 = create_scan_msg(0.0355)
        scan_ir7 = create_scan_msg(0.0355)
        scan_us1 = create_scan_msg(0.0449, min_range=0.02, max_range=2.0)
        scan_us3 = create_scan_msg(0.0449, min_range=0.02, max_range=2.0)
        node.us_callback(2, scan_us2)
        node.ir_callback(0, scan_ir0)
        node.ir_callback(7, scan_ir7)
        node.us_callback(1, scan_us1)
        node.us_callback(3, scan_us3)

        # Drift robot_x back by 2cm to 0.38m
        node.robot_x = 0.38
        node.robot_y = 0.0

        # First refinement step: alpha = 0.30 moves robot towards target smoothly
        # without hard 1cm snap
        refined = node.refine_pose_with_scan_matching()
        assert refined is True
        assert node.robot_x > 0.385
        assert node.robot_x == pytest.approx(0.389, abs=0.005)
    finally:
        node.destroy_node()


def test_no_stick_when_driving_parallel_to_wall():
    """Verify robot advances cleanly and does not stick when driving parallel to a wall."""
    node = Khepera4Mapper()
    try:
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        # Pre-seed cruising velocity at t = 9.95s (v = 0.08 m/s)
        odom_pre = create_odom_msg(0.004, -2.35, 0.0, sec=9, nanosec=950000000)
        node.odom_callback(odom_pre)

        # Robot at y = -2.35m, driving East parallel to bottom perimeter wall (y = -2.405m)
        odom_t0 = create_odom_msg(0.008, -2.35, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)
        assert node.has_odom is True

        # Forward velocity commanded
        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        # IR readings detect wall on the right (IR 5 and 6 at ~0.04m), but front IR 0 and 7 are clear (> 0.15m)
        scan_side = create_scan_msg(0.040)
        scan_clear = create_scan_msg(0.180)
        node.ir_callback(0, scan_clear)
        node.ir_callback(7, scan_clear)
        node.ir_callback(5, scan_side)
        node.ir_callback(6, scan_side)

        # Advance 150ms later along x at constant speed (12mm in 0.15s = 0.08 m/s, so a_wheel = 0)
        odom_t1 = create_odom_msg(0.020, -2.35, 0.0, sec=10, nanosec=150000000)
        node.odom_callback(odom_t1)

        # Robot must NOT stick: should advance cleanly along x
        assert node.is_wheel_slipping is False
        assert node.robot_x == pytest.approx(0.020, abs=1e-3)
    finally:
        node.destroy_node()


def test_no_stick_when_sliding_tangentially_along_wall_at_glancing_angle():
    """Verify glancing wall slide preserves tangential motion without sticking."""
    node = Khepera4Mapper()
    try:
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        # Robot at bottom perimeter wall y = -2.38m, heading 15 deg glancing toward wall
        angle_glance = math.radians(-15.0)
        odom_t0 = create_odom_msg(0.0, -2.38, angle_glance, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        # Command forward motion
        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        # Advance odometry forward along heading
        dx = 0.010 * math.cos(angle_glance)
        dy = 0.010 * math.sin(angle_glance)
        odom_t1 = create_odom_msg(dx, -2.38 + dy, angle_glance, sec=10, nanosec=50000000)
        node.odom_callback(odom_t1)

        # Tangential motion along East (x) must be preserved, perpendicular downward (y) clamped
        assert node.robot_x > 0.005
        assert node.robot_y >= node.arena_min_y
    finally:
        node.destroy_node()


def test_no_stick_when_normal_is_none():
    """Verify robot does not stick when in proximity to an obstacle but normal is None."""
    node = Khepera4Mapper()
    try:
        imu_msg = create_imu_msg(ax=0.0, sec=10, nanosec=0)
        node.imu_callback(imu_msg)

        odom_t0 = create_odom_msg(0.0, 0.0, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)

        cmd = Twist()
        cmd.linear.x = 0.08
        node.cmd_vel_callback(cmd)

        # Sensors all report clear (> 0.05m), so no surface normal is generated
        scan_mid = create_scan_msg(0.060)
        for i in range(8):
            node.ir_callback(i, scan_mid)

        # Advance odometry 100ms later
        odom_t1 = create_odom_msg(0.008, 0.0, 0.0, sec=10, nanosec=100000000)
        node.odom_callback(odom_t1)

        # Must advance cleanly without freezing to (0, 0)
        assert node.robot_x == pytest.approx(0.008, abs=1e-3)
    finally:
        node.destroy_node()


def test_ghost_wall_rapid_erosion_after_recalibration():
    """Verify unconfirmed ghost walls dissolve in 2 ray passes once recalibrated."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        # Ghost wall cell created during drift: log_odds = 1.2, hit_count = 0
        c_ghost, r_ghost = node.world_to_grid(0.20, 0.0)
        node.log_odds[r_ghost, c_ghost] = 1.20
        node.hit_count[r_ghost, c_ghost] = 0

        # High quality reading: US2 detects arena wall at 0.60m (beyond ghost cell)
        node.us_callback(2, create_scan_msg(0.60, min_range=0.02, max_range=2.0))

        # First ray pass
        node._update_occupancy_rays()
        assert node.log_odds[r_ghost, c_ghost] == pytest.approx(1.20 - node.l_free, abs=1e-3)

        # Second ray pass
        node._update_occupancy_rays()
        assert node.log_odds[r_ghost, c_ghost] == pytest.approx(1.20 - 2 * node.l_free, abs=1e-3)
        # 1.20 - 2 * 0.40 = 0.40 < 0.50 (no longer occupied!)
        assert node.log_odds[r_ghost, c_ghost] < 0.50
    finally:
        node.destroy_node()


def test_csm_suppressed_during_active_slip_to_prevent_feedback_loop():
    """Verify CSM aborts and preserves dead-reckoning pose when robot is slipping."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
    ])
    try:
        odom = create_odom_msg(0.40, 0.0, 0.0, sec=1)
        node.odom_callback(odom)

        # Map has obstacle cells
        node.log_odds[95:105, 110:115] = node.l_occ * 2.0

        # Sensors report hits
        scan = create_scan_msg(0.035)
        for i in range(8):
            node.ir_callback(i, scan)

        # Trigger wheel slip flag
        node.is_wheel_slipping = True

        initial_pose = (node.robot_x, node.robot_y, node.robot_yaw)
        refined = node.refine_pose_with_scan_matching()

        # Must abort immediately
        assert refined is False
        assert (node.robot_x, node.robot_y, node.robot_yaw) == initial_pose
    finally:
        node.destroy_node()


def test_corner_no_deadlock_when_moving_away_or_parallel():
    """Verify robot never deadlocks in corners when driving along or away from walls."""
    node = Khepera4Mapper()
    try:
        # Place robot in bottom-left corner at (-2.39, -2.39)
        odom_t0 = create_odom_msg(-2.39, -2.39, 0.0, sec=10, nanosec=0)
        node.odom_callback(odom_t0)
        assert node.robot_x == pytest.approx(-2.39, abs=1e-3)
        assert node.robot_y == pytest.approx(-2.39, abs=1e-3)

        # Drive East (+x) away from West wall along South wall
        odom_t1 = create_odom_msg(-2.38, -2.39, 0.0, sec=10, nanosec=50000000)
        node.odom_callback(odom_t1)
        # Advance along +x must NOT be zeroed or blocked
        assert node.robot_x == pytest.approx(-2.38, abs=1e-3)
        assert node.robot_y == pytest.approx(-2.39, abs=1e-3)

        # Drive North (+y) away from South wall with yaw = pi/2
        node.robot_yaw = math.pi / 2
        odom_t2 = create_odom_msg(-2.38, -2.37, math.pi / 2, sec=10, nanosec=100000000)
        node.odom_callback(odom_t2)
        # Advance along +y must NOT be zeroed or blocked
        assert node.robot_y == pytest.approx(-2.37, abs=1e-3)
    finally:
        node.destroy_node()


def test_scan_matching_anti_jitter_normal_projection_along_straight_wall():
    """Verify CSM nullifies tangential displacement along flat walls (resolving aperture problem)."""
    node = Khepera4Mapper(parameter_overrides=[
        Parameter('enable_scan_matching', Parameter.Type.BOOL, True),
        Parameter('csm_search_dist', Parameter.Type.DOUBLE, 0.03),
        Parameter('csm_deadband', Parameter.Type.DOUBLE, 0.002),
        Parameter('csm_score_threshold', Parameter.Type.DOUBLE, 0.10),
    ])
    try:
        node.has_odom = True
        node.robot_x = 0.0
        node.robot_y = -2.30
        node.robot_yaw = 0.0
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True

        # Map has established South perimeter wall at y = -2.40 (row ~ 20)
        _, r_wall = node.world_to_grid(0.0, -2.40)
        node.log_odds[r_wall - 1:r_wall + 2, :] = node.l_occ * 2.0
        node.hit_count[r_wall - 1:r_wall + 2, :] = 20

        # Sensors facing South wall at y = -2.40:
        # IR 4 (-135°): distance (0.10 - 0.049) / sin(45°) = 0.0721m
        # IR 5 (-90°): distance 0.10 - 0.070 = 0.030m
        # IR 6 (-45°): distance (0.10 - 0.049) / sin(45°) = 0.0721m
        # US 4 (-90°): distance 0.10 - 0.070 = 0.030m
        node.ir_callback(4, create_scan_msg(0.0721))
        node.ir_callback(5, create_scan_msg(0.0300))
        node.ir_callback(6, create_scan_msg(0.0721))
        node.us_callback(4, create_scan_msg(0.0300, min_range=0.02, max_range=2.0))

        initial_x = node.robot_x
        node.refine_pose_with_scan_matching()

        # Displacement along X (tangential to the wall) must be strictly 0 to prevent jitter
        assert node.robot_x == pytest.approx(initial_x, abs=1e-4)
    finally:
        node.destroy_node()


def test_reactive_explorer_silent_when_disabled():
    """Verify ReactiveExplorer does not publish 0-velocity stop commands at 20Hz when disabled."""
    from khepera4_control.reactive_explorer import ReactiveExplorer
    node = ReactiveExplorer()
    try:
        node.enabled = False
        published_cmds = []
        node.cmd_pub = type('MockPub', (), {'publish': lambda msg: published_cmds.append(msg)})()

        # Execute control loop tick
        node.control_loop()

        # When disabled, it must be completely silent (0 commands published)
        assert len(published_cmds) == 0, "Disabled explorer must not publish to cmd_vel!"
    finally:
        node.destroy_node()


def test_ghost_wall_capped_during_slip_and_eroded_when_recalibrated():
    """Verify obstacle hits during slip are capped and easily eroded after pose recalibration."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        # Step 1: Active wheel slip occurs
        node.is_wheel_slipping = True
        c_ghost, r_ghost = node.world_to_grid(0.10, 0.0)

        # IR0 reports obstacle at 0.036m (ahead) during wheel slip
        node.ir_callback(0, create_scan_msg(0.036))
        node._update_occupancy_rays()

        # Belief must be capped at l_occ during slip, with 0 hit_count
        assert node.log_odds[r_ghost, c_ghost] <= node.l_occ
        assert node.hit_count[r_ghost, c_ghost] == 0

        # Step 2: Recalibrate with established reference (slip ends, high confidence)
        node.is_wheel_slipping = False
        # IR0 is now clear (no longer right at the wall)
        node.ir_callback(0, create_scan_msg(0.20))
        # Sensor reads past the ghost wall to 0.40m
        node.us_callback(2, create_scan_msg(0.40, min_range=0.02, max_range=2.0))

        # Pass 1 of free ray
        node._update_occupancy_rays()
        assert node.log_odds[r_ghost, c_ghost] < node.l_occ

        # Pass 2 of free ray
        node._update_occupancy_rays()
        # Log odds should drop below 0.50 (eroded back to free space)
        assert node.log_odds[r_ghost, c_ghost] < 0.50
    finally:
        node.destroy_node()


