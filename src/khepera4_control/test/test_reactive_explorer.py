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

from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu, LaserScan

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
