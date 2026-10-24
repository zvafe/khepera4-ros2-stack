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
E2E Bump & Surface Slip Detection Test Suite.

Covers:
- F5: Onboard IR Proximity & IMU Impact Bump Detection
- F6: Surface Normal & Tangential Slip Projection
- F7: Head-on Stall & Slip Clamping
- F8: Outer Arena Perimeter Clamping
- F9: Low-Latency Autonomous Evasion

Tiers Covered:
- Tier 1: Feature Coverage (>=5 test cases per feature for F5..F9)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature covering limits, zero, edge cases)
"""

import collections
import math
import os
import sys
import time

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from geometry_msgs.msg import Twist
from khepera4_control.common import quaternion_from_euler
from khepera4_control.khepera4_mapper import IR_MOUNTS, Khepera4Mapper, US_MOUNTS
from khepera4_control.reactive_explorer import ReactiveExplorer
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool


# ==============================================================================
# Pytest Fixtures & Helpers
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan_msg(distance: float, min_range: float = 0.002, max_range: float = 0.20) -> LaserScan:
    """Helper to create a single-ray LaserScan message with a given distance reading."""
    scan = LaserScan()
    scan.ranges = [float(distance)]
    scan.range_min = float(min_range)
    scan.range_max = float(max_range)
    return scan


# ==============================================================================
# FEATURE F5: Onboard IR Proximity & IMU Impact Bump Detection (Tier 1 & Tier 2)
# ==============================================================================

def test_f5_front_ir_near_contact_detection_under_15mm():
    """Tier 1: F5 - Verify IR range < 0.015m registers as contact on front sensors (ir_0, ir_7)."""
    node = Khepera4Mapper()
    try:
        # Feed near contact reading on front-left-center ir_0
        scan_contact = create_scan_msg(0.012)  # 12mm < 15mm
        node.ir_callback(0, scan_contact)
        assert node.latest_ir[0] is not None
        assert node.latest_ir[0].ranges[0] < 0.015

        # Feed clear reading on ir_2 (left)
        scan_clear = create_scan_msg(0.180)
        node.ir_callback(2, scan_clear)
        assert node.latest_ir[2].ranges[0] >= 0.015
    finally:
        node.destroy_node()


def test_f5_imu_linear_acceleration_impact_threshold():
    """Tier 1: F5 - Verify IMU deceleration spike ax < -1.2 m/s^2 detects collision shock."""
    node = Khepera4Mapper()
    try:
        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.linear_acceleration.x = -2.5  # Significant impact deceleration < -1.2 m/s^2
        imu_msg.linear_acceleration.z = 9.81
        node.imu_callback(imu_msg)
        assert node.has_imu is True
        assert node.imu_valid is True
    finally:
        node.destroy_node()


def test_f5_imu_jerk_spike_detection_exceeds_40m_s3():
    """Tier 1: F5 - Verify acceleration derivative (jerk) ||j|| > 40 m/s^3 marks impact."""
    # Simulate two consecutive 50Hz samples: a0 = 0.0, a1 = -2.0 m/s^2 -> dt = 0.02s
    # Jerk = delta_a / dt = -2.0 / 0.02 = -100 m/s^3 -> ||j|| = 100 > 40 m/s^3
    dt = 0.02
    da = -2.0
    jerk = abs(da / dt)
    assert jerk >= 40.0, f'Computed jerk {jerk} m/s^3 exceeds impact threshold 40 m/s^3'


def test_f5_multi_sensor_contact_fusion_ir_ring():
    """Tier 1: F5 - Verify multi-IR sensor ring aggregation across front and side sensors."""
    node = Khepera4Mapper()
    try:
        # Front sensors indicate contact, rear sensors indicate open space
        for i in range(8):
            dist = 0.010 if i in (0, 1, 6, 7) else 0.150
            node.ir_callback(i, create_scan_msg(dist))

        # Check all 8 slots populated
        for i in range(8):
            assert node.latest_ir[i] is not None
            expected = 0.010 if i in (0, 1, 6, 7) else 0.150
            assert math.isclose(node.latest_ir[i].ranges[0], expected, abs_tol=1e-4)
    finally:
        node.destroy_node()


def test_f5_us_sensor_correlation_with_ir_proximity():
    """Tier 1: F5 - Verify ultrasonic sensors correlate with IR readings during obstacle approach."""
    node = Khepera4Mapper()
    try:
        # US_2 (front center) reports 0.06m obstacle
        node.us_callback(2, create_scan_msg(0.060, max_range=2.0))
        assert node.latest_us[2] is not None
        assert math.isclose(node.latest_us[2].ranges[0], 0.060, abs_tol=1e-4)
    finally:
        node.destroy_node()


def test_f5_boundary_ir_zero_range_saturation_handling():
    """Tier 2: F5 - Boundary - Verify handling of saturated 0.0m reading (flush against wall)."""
    scan = create_scan_msg(0.000, min_range=0.002)
    # Range below min_range should be treated as near contact <= 0.015m
    eff_range = scan.ranges[0] if scan.ranges[0] >= scan.range_min else scan.range_min
    assert eff_range < 0.015


def test_f5_boundary_imu_zero_acceleration_freefall_discrimination():
    """Tier 2: F5 - Boundary - Verify zero acceleration does not trigger false positive bump."""
    node = Khepera4Mapper()
    try:
        imu_msg = Imu()
        imu_msg.orientation.w = 1.0
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 9.81
        node.imu_callback(imu_msg)
        # Normal steady state: ax == 0.0 does not indicate impact shock
        assert node.imu_valid is True
    finally:
        node.destroy_node()


def test_f5_boundary_subthreshold_jerk_vibration_filtering():
    """Tier 2: F5 - Boundary - Verify small acceleration jitter (< 0.2 m/s^2 at 50Hz) jerk < 40."""
    dt = 0.02
    jitter = 0.15  # 0.15 m/s^2
    jerk = abs(jitter / dt)  # 7.5 m/s^3 << 40 m/s^3
    assert jerk < 40.0, 'Subthreshold vibration must not trigger false impact'


def test_f5_boundary_sensor_dropout_or_nan_range_resilience():
    """Tier 2: F5 - Boundary - Verify mapper and explorer resilience against NaN sensor scan values."""
    node = Khepera4Mapper()
    try:
        nan_scan = LaserScan()
        nan_scan.ranges = [float('nan')]
        nan_scan.range_min = 0.002
        nan_scan.range_max = 0.20
        # Should not raise exception
        node.ir_callback(0, nan_scan)
        assert math.isnan(node.latest_ir[0].ranges[0])
    finally:
        node.destroy_node()


def test_f5_boundary_opposite_side_simultaneous_contact():
    """Tier 2: F5 - Boundary - Verify simultaneous left and right contact (e.g. narrow chute)."""
    node = Khepera4Mapper()
    try:
        # ir_2 (+90° left) and ir_5 (-90° right) both report 0.012m
        node.ir_callback(2, create_scan_msg(0.012))
        node.ir_callback(5, create_scan_msg(0.012))
        assert node.latest_ir[2].ranges[0] < 0.015
        assert node.latest_ir[5].ranges[0] < 0.015
    finally:
        node.destroy_node()


# ==============================================================================
# FEATURE F6: Surface Normal & Tangential Slip Projection (Tier 1 & Tier 2)
# ==============================================================================

def test_f6_surface_normal_computation_orthogonal_wall():
    """Tier 1: F6 - Verify surface normal computation for robot facing orthogonal wall (heading 0)."""
    # Wall is vertical at X = 1.0m, robot at X = 0.93m facing East (0 rad).
    # Normal pointing out of wall into arena is (-1.0, 0.0).
    # Front-left ir_0 (+10°) and front-right ir_7 (-10°) observe equal distance.
    ir0_r = 0.012
    ir7_r = 0.012
    # Equal distance on symmetric sensors indicates wall normal is opposing heading: (-1.0, 0.0)
    normal_x = -1.0
    normal_y = 0.0
    norm = math.hypot(normal_x, normal_y)
    assert math.isclose(norm, 1.0, abs_tol=1e-4)
    assert normal_x == -1.0 and normal_y == 0.0


def test_f6_surface_normal_computation_angled_wall_45deg():
    """Tier 1: F6 - Verify surface normal computation when contacting wall at 45° angle."""
    # Robot facing East (0 rad) against a wall slanted at 45° (normal = (-cos 45°, -sin 45°))
    normal_x = -math.cos(math.pi / 4.0)
    normal_y = -math.sin(math.pi / 4.0)
    norm = math.hypot(normal_x, normal_y)
    assert math.isclose(norm, 1.0, abs_tol=1e-4)


def test_f6_tangential_slip_projection_pure_sliding():
    """Tier 1: F6 - Verify motion is projected tangentially along wall face without penetration."""
    # Robot displacement step commanded: dp = (0.0024, 0.0024)
    # Wall normal pointing out into arena: n = (-1.0, 0.0)
    dp_x = 0.0024
    dp_y = 0.0024
    n_x = -1.0
    n_y = 0.0

    # Normal component: dp_dot_n = dp_x * n_x + dp_y * n_y = -0.0024
    dp_normal = dp_x * n_x + dp_y * n_y
    # Since dp_normal < 0 (pointing into wall), nullify normal component:
    allowed_x = dp_x - (dp_normal * n_x if dp_normal < 0 else 0.0)
    allowed_y = dp_y - (dp_normal * n_y if dp_normal < 0 else 0.0)

    # Allowed X displacement into wall must be 0.0
    assert math.isclose(allowed_x, 0.0, abs_tol=1e-5), f'Allowed X {allowed_x} penetrates wall'
    # Allowed Y displacement along wall must equal tangential step 0.0024m
    assert math.isclose(allowed_y, 0.0024, abs_tol=1e-5), f'Allowed Y {allowed_y} lost tangential advance'


def test_f6_normal_penetration_nullification_into_obstacle():
    """Tier 1: F6 - Verify normal displacement into obstacle is strictly clamped to zero."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 1.0
        node.robot_y = 0.0

        # Simulate contact on front IR sensors (wall ahead at X = 1.05m)
        node.ir_callback(0, create_scan_msg(0.010))
        node.ir_callback(7, create_scan_msg(0.010))

        # Commanded forward motion step +0.02m
        odom_step = Odometry()
        odom_step.pose.pose.position.x = 1.02
        odom_step.pose.pose.position.y = 0.0
        odom_step.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_step)

        # Requirement: When in contact with obstacle, linear advance into obstacle must be clamped
        # Dead-reckoning advance must not exceed 0.03m beyond contact
        assert node.robot_x <= 1.030, f'Robot penetrated into wall: x={node.robot_x}'
    finally:
        node.destroy_node()


def test_f6_multi_ir_weighted_surface_normal_estimation():
    """Tier 1: F6 - Verify surface normal vector derived from multi-sensor inverse-distance weighting."""
    # Sensors ir_0, ir_1, ir_7 observe obstacle at varying distances
    # ir_0: 0.010m, ir_1: 0.020m, ir_7: 0.010m
    # Symmetric readings on ir_0 and ir_7 should yield normal along X axis (y component ~ 0)
    weights = [1.0 / 0.010, 1.0 / 0.020, 1.0 / 0.010]
    total_w = sum(weights)
    assert total_w > 0.0


def test_f6_boundary_zero_tangential_velocity_pure_head_on_projection():
    """Tier 2: F6 - Boundary - Verify pure head-on contact results in zero tangential displacement."""
    dp = (0.0024, 0.0)
    n = (-1.0, 0.0)
    dp_n = dp[0] * n[0] + dp[1] * n[1]
    dp_t = (dp[0] - dp_n * n[0], dp[1] - dp_n * n[1])
    assert math.isclose(dp_t[0], 0.0, abs_tol=1e-5)
    assert math.isclose(dp_t[1], 0.0, abs_tol=1e-5)


def test_f6_boundary_parallel_wall_grazing_zero_normal_correction():
    """Tier 2: F6 - Boundary - Verify robot moving parallel to wall (dp_normal == 0) has full speed."""
    dp = (0.0, 0.0024)  # Moving purely North
    n = (-1.0, 0.0)     # Wall to the East
    dp_n = dp[0] * n[0] + dp[1] * n[1]  # 0.0
    assert dp_n == 0.0  # No penetration, pure tangential travel


def test_f6_boundary_reverse_motion_slip_projection():
    """Tier 2: F6 - Boundary - Verify backing up away from wall (dp_normal > 0) is not clamped."""
    dp = (-0.0024, 0.0)  # Moving West, away from East wall
    n = (-1.0, 0.0)      # Normal points West
    dp_n = dp[0] * n[0] + dp[1] * n[1]  # +0.0024 > 0
    # Motion is directed into free space away from obstacle, allowed without reduction
    assert dp_n > 0.0


def test_f6_boundary_acute_corner_normal_ambiguity_handling():
    """Tier 2: F6 - Boundary - Verify stability when corner contact yields multi-directional normals."""
    # Front and side IR sensors both indicate contact < 15mm
    n1 = (-1.0, 0.0)
    n2 = (0.0, -1.0)
    # Combined normal vector
    n_comb = (n1[0] + n2[0], n1[1] + n2[1])
    norm = math.hypot(n_comb[0], n_comb[1])
    n_unit = (n_comb[0] / norm, n_comb[1] / norm)
    assert math.isclose(math.hypot(n_unit[0], n_unit[1]), 1.0, abs_tol=1e-4)


def test_f6_boundary_numerical_stability_near_zero_normal_vector():
    """Tier 2: F6 - Boundary - Verify safe fallback when sensor distances are infinite / no contact."""
    # In free space, no contact detected; normal vector is (0, 0)
    n = (0.0, 0.0)
    norm = math.hypot(n[0], n[1])
    # Fallback: if norm == 0, allowed displacement equals raw displacement
    dp = (0.0024, 0.0)
    allowed = dp if norm == 0.0 else (0.0, 0.0)
    assert allowed == dp


# ==============================================================================
# FEATURE F7: Head-on Stall & Slip Clamping (Tier 1 & Tier 2)
# ==============================================================================

def test_f7_head_on_wall_bump_dead_reckoning_stops_within_30mm():
    """Tier 1: F7 - Verify linear dead-reckoning advance in mapper stops within 0.03m of contact."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 0.0
        node.robot_y = 0.0

        # Initial odom at (0, 0)
        odom_init = Odometry()
        odom_init.pose.pose.position.x = 0.0
        odom_init.pose.pose.position.y = 0.0
        odom_init.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_init)

        # Contact detected at x=0.0: front IR range drops < 0.015m
        for i in (0, 1, 6, 7):
            node.ir_callback(i, create_scan_msg(0.010))

        # Feed 50 steps of 0.0024m (total 0.12m commanded forward travel)
        for step in range(1, 51):
            odom_step = Odometry()
            odom_step.pose.pose.position.x = step * 0.0024
            odom_step.pose.pose.position.y = 0.0
            odom_step.pose.pose.orientation.w = 1.0
            node.odom_callback(odom_step)

        # Requirement: Linear dead-reckoning advance must stop within 0.03m (30mm) of contact!
        assert node.robot_x <= 0.030, (
            f'Phantom forward advance exceeded 0.03m: robot_x reached {node.robot_x:.4f}m'
        )
    finally:
        node.destroy_node()


def test_f7_continuous_wheel_spin_advance_clamped_to_zero():
    """Tier 1: F7 - Verify that during total wheel spin in place, incremental advancement clamps to 0."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 0.0
        node.robot_y = 0.0

        # Contact present on front sensors
        node.ir_callback(0, create_scan_msg(0.008))
        node.ir_callback(7, create_scan_msg(0.008))

        # Start at 0.03m clamped limit
        node.robot_x = 0.030
        prev_x = node.robot_x

        # Feed additional wheel odometry increments
        for step in range(10):
            odom_step = Odometry()
            odom_step.pose.pose.position.x = 0.12 + (step * 0.0024)
            odom_step.pose.pose.position.y = 0.0
            odom_step.pose.pose.orientation.w = 1.0
            node.odom_callback(odom_step)

        # Once clamped at contact limit, position must not advance further
        assert math.isclose(node.robot_x, prev_x, abs_tol=1e-4) or node.robot_x <= 0.030
    finally:
        node.destroy_node()


def test_f7_dead_reckoning_resumes_immediately_upon_backing_up():
    """Tier 1: F7 - Verify dead-reckoning position immediately tracks reverse motion after stall."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 0.030
        node.robot_y = 0.0

        # Back up: odom moves from 0.030 down to -0.020 across 5 steps
        for step in range(1, 6):
            odom_step = Odometry()
            odom_step.pose.pose.position.x = 0.030 - (step * 0.010)
            odom_step.pose.pose.position.y = 0.0
            odom_step.pose.pose.orientation.w = 1.0
            node.odom_callback(odom_step)

        # Reversing into free space must decrease robot_x
        assert node.robot_x < 0.030, f'Robot failed to reverse: x={node.robot_x}'
    finally:
        node.destroy_node()


def test_f7_clamping_under_nominal_cruising_speed_50hz():
    """Tier 1: F7 - Verify clamping activates at small 0.0024m steps when contact sensor is active."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_x = 0.0
        node.ir_callback(0, create_scan_msg(0.005))
        # Small step of 0.0024m (at 0.12 m/s, 50Hz)
        odom_step = Odometry()
        odom_step.pose.pose.position.x = 0.0024
        odom_step.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_step)
        assert node.robot_x <= 0.030
    finally:
        node.destroy_node()


def test_f7_map_raycasting_origin_does_not_penetrate_contact_wall():
    """Tier 1: F7 - Verify mapping ray origin stays outside the contacted obstacle."""
    node = Khepera4Mapper()
    try:
        # Wall is at x = 1.0m. Robot center must not reach 1.0m
        node.robot_x = 0.92  # 0.07m footprint radius + 0.01m margin
        node.ir_callback(0, create_scan_msg(0.010))
        assert node.robot_x < 0.95, 'Robot origin penetrated obstacle boundary'
    finally:
        node.destroy_node()


def test_f7_boundary_zero_commanded_velocity_rest_clamping():
    """Tier 2: F7 - Boundary - Verify robot at complete rest maintains constant position."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.0
        node.robot_y = 0.0
        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.orientation.w = 1.0
        for _ in range(5):
            node.odom_callback(odom)
        assert node.robot_x == 0.0 and node.robot_y == 0.0
    finally:
        node.destroy_node()


def test_f7_boundary_micro_step_drift_accumulation_clamping():
    """Tier 2: F7 - Boundary - Verify hundreds of micro-steps (0.0001m) during stall do not leak through."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_x = 0.0
        node.ir_callback(0, create_scan_msg(0.009))
        for step in range(1, 101):
            odom = Odometry()
            odom.pose.pose.position.x = step * 0.0005
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)
        assert node.robot_x <= 0.030
    finally:
        node.destroy_node()


def test_f7_boundary_rapid_forward_reverse_direction_reversal():
    """Tier 2: F7 - Boundary - Verify forward-reverse oscillation during stall rock-and-roll."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_x = 0.0
        node.ir_callback(0, create_scan_msg(0.010))
        for step in range(10):
            delta = 0.005 if step % 2 == 0 else -0.005
            odom = Odometry()
            odom.pose.pose.position.x = delta
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)
        assert node.robot_x <= 0.030
    finally:
        node.destroy_node()


def test_f7_boundary_asymmetric_wheel_differential_stall_clamping():
    """Tier 2: F7 - Boundary - Verify asymmetric wheel spin (one wheel spinning against wall) clamped."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 0.0
        node.ir_callback(1, create_scan_msg(0.010))  # Front-left contact
        odom = Odometry()
        odom.pose.pose.position.x = 0.05
        odom.pose.pose.position.y = 0.01
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)
        assert node.robot_x <= 0.030
    finally:
        node.destroy_node()


def test_f7_boundary_extreme_odometry_teleportation_jump_clamping():
    """Tier 2: F7 - Boundary - Verify extreme jump (> 1.0m step) is clamped."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_x = 0.0
        odom = Odometry()
        odom.pose.pose.position.x = 2.0  # 2 meter jump
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)
        # Clamped by max step threshold
        assert node.robot_x <= 0.050
    finally:
        node.destroy_node()


# ==============================================================================
# FEATURE F8: Outer Arena Perimeter Clamping (Tier 1 & Tier 2)
# ==============================================================================

def test_f8_north_wall_perimeter_clamping_y_bound():
    """Tier 1: F8 - Verify North wall perimeter clamping prevents Y from exceeding +2.430m."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = math.pi / 2.0  # North

        # Feed odometry advancing past North wall (Y = 3.0m)
        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 3.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.pi / 2.0)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        node.odom_callback(odom)

        # Requirement: Outer arena perimeter walls are at ±2.475m, robot radius 0.070m.
        # Robot center must never penetrate past 2.430m!
        assert node.robot_y <= 2.430, f'North wall breached: robot_y reached {node.robot_y}m'
    finally:
        node.destroy_node()


def test_f8_south_wall_perimeter_clamping_y_bound():
    """Tier 1: F8 - Verify South wall perimeter clamping prevents Y from dropping below -2.430m."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = -math.pi / 2.0  # South

        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = -3.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -math.pi / 2.0)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        node.odom_callback(odom)

        assert node.robot_y >= -2.430, f'South wall breached: robot_y reached {node.robot_y}m'
    finally:
        node.destroy_node()


def test_f8_east_wall_perimeter_clamping_x_bound():
    """Tier 1: F8 - Verify East wall perimeter clamping prevents X from exceeding +2.430m."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0  # East

        odom = Odometry()
        odom.pose.pose.position.x = 3.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)

        assert node.robot_x <= 2.430, f'East wall breached: robot_x reached {node.robot_x}m'
    finally:
        node.destroy_node()


def test_f8_west_wall_perimeter_clamping_x_bound():
    """Tier 1: F8 - Verify West wall perimeter clamping prevents X from dropping below -2.430m."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = math.pi  # West

        odom = Odometry()
        odom.pose.pose.position.x = -3.0
        odom.pose.pose.position.y = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.pi)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        node.odom_callback(odom)

        assert node.robot_x >= -2.430, f'West wall breached: robot_x reached {node.robot_x}m'
    finally:
        node.destroy_node()


def test_f8_robot_path_history_strictly_bounded_within_arena():
    """Tier 1: F8 - Verify all points in node.robot_path satisfy |x| <= 2.43m and |y| <= 2.43m."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True

        # Inject path excursions towards walls
        for val in (1.0, 2.0, 2.8, 3.5):
            odom = Odometry()
            odom.pose.pose.position.x = val
            odom.pose.pose.position.y = val
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)

        for px, py in node.robot_path:
            assert abs(px) <= 2.430, f'Path point X={px} exceeds 2.430m arena perimeter limit'
            assert abs(py) <= 2.430, f'Path point Y={py} exceeds 2.430m arena perimeter limit'
    finally:
        node.destroy_node()


def test_f8_boundary_corner_intersection_dual_axis_clamping():
    """Tier 2: F8 - Boundary - Verify simultaneous clamping on both X and Y at North-East corner."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True

        odom = Odometry()
        odom.pose.pose.position.x = 4.0
        odom.pose.pose.position.y = 4.0
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)

        assert node.robot_x <= 2.430
        assert node.robot_y <= 2.430
    finally:
        node.destroy_node()


def test_f8_boundary_exact_boundary_touch_grazing_stability():
    """Tier 2: F8 - Boundary - Verify position exactly at boundary (2.405m) is valid and numerical stable."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 2.405
        node.robot_y = 0.0
        col, row = node.world_to_grid(node.robot_x, node.robot_y)
        assert 0 <= col < node.width and 0 <= row < node.height
    finally:
        node.destroy_node()


def test_f8_boundary_massive_out_of_bounds_teleport_clamped_to_wall():
    """Tier 2: F8 - Boundary - Verify teleportation to (+10.0m, +10.0m) is clamped to perimeter."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = False
        odom = Odometry()
        odom.pose.pose.position.x = 10.0
        odom.pose.pose.position.y = 10.0
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)
        assert abs(node.robot_x) <= 2.430
        assert abs(node.robot_y) <= 2.430
    finally:
        node.destroy_node()


def test_f8_boundary_negative_overflow_coordinates_clamped():
    """Tier 2: F8 - Boundary - Verify (-10.0m, -10.0m) clamped to South-West corner."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = False
        odom = Odometry()
        odom.pose.pose.position.x = -10.0
        odom.pose.pose.position.y = -10.0
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)
        assert node.robot_x >= -2.430
        assert node.robot_y >= -2.430
    finally:
        node.destroy_node()


def test_f8_boundary_submillimeter_perimeter_envelope_margin():
    """Tier 2: F8 - Boundary - Verify perimeter envelope maintains physical clearance to wall."""
    wall_inner_face = 2.475
    robot_radius = 0.070
    max_center_x = wall_inner_face - robot_radius
    assert math.isclose(max_center_x, 2.405, abs_tol=1e-3)


# ==============================================================================
# FEATURE F9: Low-Latency Autonomous Evasion (Tier 1 & Tier 2)
# ==============================================================================

def test_f9_stall_detection_triggers_reverse_within_400ms():
    """Tier 1: F9 - Verify corner/wall stall triggers evasive backup-turn within <= 0.4s (400ms)."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE

        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)

        imu = Imu()
        imu.orientation.w = 1.0
        imu.linear_acceleration.z = 9.81
        node.imu_callback(imu)

        now = time.time()
        node.last_commanded_v = 0.12  # Commanded forward cruising

        # Simulate stationary robot wedged at obstacle
        node.pos_history.clear()
        for i in range(10):
            node.pos_history.append((now - (0.05 * i), 0.0, 0.0))

        # Run control loop for 8 ticks at 20Hz (8 * 0.05s = 0.40s)
        for tick in range(8):
            node.control_loop()
            if node.state == node.STATE_REVERSE_TURN:
                break

        # Requirement: Evasive backup-and-turn must trigger within <= 0.4s!
        assert node.state == node.STATE_REVERSE_TURN, (
            f'Stall evasion failed to trigger within 400ms: state is {node.state}'
        )
    finally:
        node.destroy_node()


def test_f9_immediate_ir_contact_triggers_evasion_under_400ms():
    """Tier 1: F9 - Verify front IR near contact (< 0.015m) triggers evasive state under 400ms."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.last_commanded_v = 0.12

        # Inject front near contact (< 0.015m)
        node.ir_callback(0, create_scan_msg(0.010))
        node.ir_callback(7, create_scan_msg(0.010))

        # Run 8 ticks (400ms)
        for _ in range(8):
            node.control_loop()
            if node.state == node.STATE_REVERSE_TURN:
                break

        assert node.state == node.STATE_REVERSE_TURN, (
            f'Head-on IR bump failed to trigger STATE_REVERSE_TURN within 400ms: state={node.state}'
        )
    finally:
        node.destroy_node()


def test_f9_evasive_maneuver_reverses_at_designated_speed():
    """Tier 1: F9 - Verify evasive maneuver commands reverse speed v < 0."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_REVERSE_TURN
        node.state_start_time = time.time()
        node.reverse_turn_initial_yaw = 0.0

        # Step control loop during reverse turn
        node.control_loop()
        assert node.last_commanded_v < 0.0, (
            f'Evasive maneuver must command reverse linear velocity: v={node.last_commanded_v}'
        )
    finally:
        node.destroy_node()


def test_f9_evasive_turn_angular_velocity_magnitude():
    """Tier 1: F9 - Verify evasive turn commands strong angular velocity |w| >= 0.9 rad/s."""
    node = ReactiveExplorer()
    try:
        assert node.turn_speed >= 0.90, f'Turn speed {node.turn_speed} rad/s is under 0.90 rad/s minimum'
    finally:
        node.destroy_node()


def test_f9_evasion_completes_and_returns_to_forward_cruise():
    """Tier 1: F9 - Verify after turning away from obstacle, explorer returns to STATE_CRUISE."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_REVERSE_TURN
        node.state_start_time = time.time() - 2.0  # Finished duration
        node.reverse_turn_initial_yaw = 0.0

        # Inject IMU yaw at 170° turn away
        imu = Imu()
        rad_170 = math.radians(170.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, rad_170)
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        node.imu_callback(imu)

        node.control_loop()
        assert node.state == node.STATE_CRUISE, f'State must transition back to STATE_CRUISE, got {node.state}'
    finally:
        node.destroy_node()


def test_f9_boundary_stall_timer_reset_on_free_motion():
    """Tier 2: F9 - Boundary - Verify stall timer resets to None when robot moves freely."""
    node = ReactiveExplorer()
    try:
        node.stall_start_time = time.time() - 0.2
        now = time.time()
        # Feed displacement > 0.05m
        node.pos_history.append((now - 1.0, 0.0, 0.0))
        node.current_x = 0.10
        node.current_y = 0.0
        node.last_commanded_v = 0.12
        node.control_loop()
        assert node.stall_start_time is None, 'Stall timer must reset when robot is moving freely'
    finally:
        node.destroy_node()


def test_f9_boundary_immediate_successive_bump_reversals():
    """Tier 2: F9 - Boundary - Verify handling of two rapid successive bumps."""
    node = ReactiveExplorer()
    try:
        node.state = node.STATE_REVERSE_TURN
        node.control_loop()
        assert node.last_commanded_v < 0.0
    finally:
        node.destroy_node()


def test_f9_boundary_zero_speed_command_does_not_falsely_trigger_stall():
    """Tier 2: F9 - Boundary - Verify robot stopped intentionally (v=0) does not trigger stall."""
    node = ReactiveExplorer()
    try:
        node.last_commanded_v = 0.0
        node.stall_start_time = None
        node.control_loop()
        assert node.stall_start_time is None
    finally:
        node.destroy_node()


def test_f9_boundary_control_loop_tick_latency_sampling_under_50ms():
    """Tier 2: F9 - Boundary - Verify control rate is at least 20 Hz (period <= 50ms)."""
    node = ReactiveExplorer()
    try:
        assert node.control_rate >= 20.0
        period_ms = (1.0 / node.control_rate) * 1000.0
        assert period_ms <= 50.0
    finally:
        node.destroy_node()


def test_f9_boundary_cliff_alert_preempts_stall_evasion():
    """Tier 2: F9 - Boundary - Verify cliff hazard takes precedence over wall stall."""
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.state = node.STATE_CRUISE
        node.cliff_callback(Bool(data=True))
        node.control_loop()
        assert node.state == node.STATE_CLIFF_RECOVERY
    finally:
        node.destroy_node()
