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
Unit and regression tests for Khepera IV rollover detection, recenter service,
and tangential slip attenuation.
"""

import math
import os
import sys
import time

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from khepera4_control.common import quaternion_from_euler  # noqa: E402
from khepera4_control.khepera4_mapper import Khepera4Mapper  # noqa: E402
from khepera4_control.reactive_explorer import ReactiveExplorer  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def rclpy_init():
    """Ensure rclpy is initialized once for all tests."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan(dist: float = 0.01) -> LaserScan:
    """Create LaserScan message with uniform distance readings."""
    scan = LaserScan()
    scan.header.frame_id = 'base_footprint'
    scan.range_min = 0.002
    scan.range_max = 0.25
    scan.ranges = [dist] * 10
    return scan


# ==============================================================================
# SECTION 1: Rollover Detection & Swarm Distress Signal Architecture
# ==============================================================================

def test_mapper_detects_turned_over_upside_down():
    """Verify accelerometer az < -1.0 m/s^2 correctly triggers is_turned_over."""
    mapper = Khepera4Mapper()
    try:
        # 1. Normal upright IMU reading (az ~ +9.81 m/s^2)
        upright_imu = Imu()
        upright_imu.linear_acceleration.x = 0.0
        upright_imu.linear_acceleration.y = 0.0
        upright_imu.linear_acceleration.z = 9.81
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        upright_imu.orientation.x = qx
        upright_imu.orientation.y = qy
        upright_imu.orientation.z = qz
        upright_imu.orientation.w = qw
        mapper.imu_callback(upright_imu)

        assert mapper.is_turned_over is False

        # 2. Turned upside down: az is negative (-9.81 m/s^2)
        flipped_imu = Imu()
        flipped_imu.linear_acceleration.x = 0.0
        flipped_imu.linear_acceleration.y = 0.0
        flipped_imu.linear_acceleration.z = -9.81
        qx, qy, qz, qw = quaternion_from_euler(math.pi, 0.0, 0.0)  # 180 roll
        flipped_imu.orientation.x = qx
        flipped_imu.orientation.y = qy
        flipped_imu.orientation.z = qz
        flipped_imu.orientation.w = qw
        mapper.imu_callback(flipped_imu)

        assert mapper.is_turned_over is True
    finally:
        mapper.destroy_node()


def test_mapper_zeroes_dead_reckoning_advance_when_turned_over():
    """Verify wheel odometry does not move robot pose on map when robot is turned over."""
    mapper = Khepera4Mapper()
    try:
        mapper.has_odom = True
        mapper.has_imu = True
        mapper.imu_valid = True
        mapper.robot_x = 0.5
        mapper.robot_y = 0.5
        mapper.robot_yaw = 0.0
        mapper.last_odom_x = 0.5
        mapper.last_odom_y = 0.5
        mapper.last_odom_yaw = 0.0
        mapper.is_turned_over = True

        # Wheels spin by 0.20m while robot is upside down
        odom = Odometry()
        odom.pose.pose.position.x = 0.70
        odom.pose.pose.position.y = 0.50
        odom.pose.pose.orientation.w = 1.0
        mapper.odom_callback(odom)

        # Pose must remain completely unchanged!
        assert mapper.robot_x == 0.5
        assert mapper.robot_y == 0.5
    finally:
        mapper.destroy_node()


def test_reactive_explorer_halts_when_turned_over():
    """Verify autonomous explorer commands zero velocity and halts when flipped."""
    explorer = ReactiveExplorer()
    try:
        published_cmds = []
        explorer.cmd_pub.publish = lambda cmd: published_cmds.append(cmd)

        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.enabled = True
        explorer.is_turned_over = True

        explorer.control_loop()

        assert len(published_cmds) >= 1
        assert published_cmds[-1].linear.x == 0.0
        assert published_cmds[-1].angular.z == 0.0
    finally:
        explorer.destroy_node()


# ==============================================================================
# SECTION 2: Recenter Service Tests
# ==============================================================================

def test_mapper_recenter_service_resets_pose_and_clears_rollover():
    """Verify /recenter_robot resets pose to origin and clears is_turned_over."""
    mapper = Khepera4Mapper()
    try:
        mapper.has_odom = True
        mapper.robot_x = -1.95
        mapper.robot_y = 1.70
        mapper.robot_yaw = 1.57
        mapper.is_turned_over = True
        mapper.robot_path = [(0.0, 0.0), (-1.0, 1.0), (-1.95, 1.70)]

        req = Trigger.Request()
        res = Trigger.Response()
        out = mapper.recenter_robot_callback(req, res)

        assert out.success is True
        assert mapper.robot_x == 0.0
        assert mapper.robot_y == 0.0
        assert mapper.robot_yaw == 0.0
        assert mapper.is_turned_over is False
        assert mapper.robot_path == [(0.0, 0.0)]
    finally:
        mapper.destroy_node()


# ==============================================================================
# SECTION 3: Tangential Wall Slide Clamping Tests
# ==============================================================================

def test_straight_into_wall_slip_clamped_to_zero():
    """Verify driving straight into a wall (tangent_ratio < 0.18) clamps sliding to zero."""
    mapper = Khepera4Mapper()
    try:
        mapper.has_odom = True
        mapper.has_imu = True
        mapper.imu_valid = True
        mapper.imu_yaw = math.radians(92.0)
        mapper.robot_x = 0.0
        mapper.robot_y = 2.40  # Approaching North wall
        mapper.robot_yaw = math.radians(92.0)  # 2 degrees from perpendicular normal (0, -1)
        mapper.last_odom_x = 0.0
        mapper.last_odom_y = 2.40
        mapper.last_odom_yaw = math.radians(92.0)

        mapper._compute_surface_normal = lambda: (0.0, -1.0)
        mapper.ir_callback(0, create_scan(0.010))

        # Forward motion into wall: delta_y = 0.010, delta_x = 0.010 * sin(2°) = ~0.00035
        odom = Odometry()
        odom.pose.pose.position.x = -0.00035
        odom.pose.pose.position.y = 2.410
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.radians(92.0))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        mapper.odom_callback(odom)

        # Sliding along X must be strictly zero (clamped)
        assert mapper.robot_x == 0.0
    finally:
        mapper.destroy_node()
