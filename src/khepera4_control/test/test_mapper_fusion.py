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

"""Unit tests for IMU heading fusion and dead-reckoning projection."""

import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

from khepera4_control.common import quaternion_from_euler
from khepera4_control.khepera4_mapper import (
    GROUND_IR_MOUNTS,
    IR_MOUNTS,
    Khepera4Mapper,
    US_MOUNTS,
)
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_urdf_sensor_alignment():
    """Verify that US, IR and ground IR mounts match khepera4.urdf."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(test_dir, '..', '..', '..'))
    urdf_path = os.path.join(
        ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf'
    )
    assert os.path.exists(urdf_path), f'URDF not found at {urdf_path}'

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {j.get('name'): j for j in root.findall('joint')}

    # 1. Ultrasonic Mounts
    for i, (exp_x, exp_y, exp_yaw) in enumerate(US_MOUNTS):
        joint_name = f'us_{i}_joint'
        assert joint_name in joints, f'Missing joint {joint_name} in URDF'
        origin = joints[joint_name].find('origin')
        xyz = [float(v) for v in origin.get('xyz').split()]
        rpy = [float(v) for v in origin.get('rpy').split()]
        assert math.isclose(xyz[0], exp_x, abs_tol=1e-3), f'us_{i} x err'
        assert math.isclose(xyz[1], exp_y, abs_tol=1e-3), f'us_{i} y err'
        assert math.isclose(rpy[2], exp_yaw, abs_tol=1e-3), f'us_{i} yaw err'

    # 2. IR Proximity Mounts
    for i, (exp_x, exp_y, exp_yaw) in enumerate(IR_MOUNTS):
        joint_name = f'ir_{i}_joint'
        assert joint_name in joints, f'Missing joint {joint_name} in URDF'
        origin = joints[joint_name].find('origin')
        xyz = [float(v) for v in origin.get('xyz').split()]
        rpy = [float(v) for v in origin.get('rpy').split()]
        assert math.isclose(xyz[0], exp_x, abs_tol=1e-3), f'ir_{i} x err'
        assert math.isclose(xyz[1], exp_y, abs_tol=1e-3), f'ir_{i} y err'
        assert math.isclose(rpy[2], exp_yaw, abs_tol=1e-3), f'ir_{i} yaw err'

    # 3. Ground IR Mounts
    for i, (exp_x, exp_y) in enumerate(GROUND_IR_MOUNTS):
        joint_name = f'ground_ir_{i}_joint'
        assert joint_name in joints, f'Missing joint {joint_name} in URDF'
        origin = joints[joint_name].find('origin')
        xyz = [float(v) for v in origin.get('xyz').split()]
        assert math.isclose(xyz[0], exp_x, abs_tol=1e-3), f'gr_{i} x err'
        assert math.isclose(xyz[1], exp_y, abs_tol=1e-3), f'gr_{i} y err'


def test_imu_callback_orientation_and_validation():
    """Verify IMU quaternion extraction, validation, and heading update."""
    node = Khepera4Mapper()
    try:
        assert node.use_imu_yaw is True
        assert node.has_imu is False
        assert node.imu_valid is False

        # Send valid IMU message at 45 degrees (pi/4 rad)
        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.pi / 4.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.linear_acceleration.z = 9.81

        node.imu_callback(imu_msg)

        assert node.has_imu is True
        assert node.imu_valid is True
        assert math.isclose(node.imu_yaw, math.pi / 4.0, abs_tol=1e-4)
        assert math.isclose(node.robot_yaw, math.pi / 4.0, abs_tol=1e-4)

        # Send invalid/degenerate IMU message (all zeros)
        invalid_msg = Imu()
        invalid_msg.orientation.x = 0.0
        invalid_msg.orientation.y = 0.0
        invalid_msg.orientation.z = 0.0
        invalid_msg.orientation.w = 0.0
        node.imu_callback(invalid_msg)

        assert node.imu_valid is False
        # Heading should not be corrupted by degenerate quaternion
        assert math.isclose(node.robot_yaw, math.pi / 4.0, abs_tol=1e-4)

    finally:
        node.destroy_node()


def test_wheel_slip_dead_reckoning_projection():
    """Simulate wheel slip heading drift and verify IMU projection."""
    node = Khepera4Mapper()
    try:
        node.use_imu_yaw = True

        # Initial pose at (0, 0), true heading is East (0 rad)
        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        node.imu_callback(imu_msg)

        odom_init = Odometry()
        odom_init.pose.pose.position.x = 0.0
        odom_init.pose.pose.position.y = 0.0
        odom_init.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_init)

        assert node.robot_x == 0.0
        assert node.robot_y == 0.0
        assert node.robot_yaw == 0.0

        # Simulate robot moving forward by 1.0m across 50 steps of 0.02m,
        # but wheel slip causes odometry to rotate by 8.0 degrees
        drift_yaw = math.radians(8.0)
        num_steps = 50
        step_dist = 1.0 / num_steps  # 0.02m per step (< 0.05m threshold)
        for i in range(1, num_steps + 1):
            odom_step = Odometry()
            px = (i * step_dist) * math.cos(drift_yaw)
            py = (i * step_dist) * math.sin(drift_yaw)
            odom_step.pose.pose.position.x = px
            odom_step.pose.pose.position.y = py
            dqx, dqy, dqz, dqw = quaternion_from_euler(0.0, 0.0, drift_yaw)
            odom_step.pose.pose.orientation.x = dqx
            odom_step.pose.pose.orientation.y = dqy
            odom_step.pose.pose.orientation.z = dqz
            odom_step.pose.pose.orientation.w = dqw
            node.odom_callback(odom_step)

        # Check: Fused heading must stay 0.0 rad
        assert math.isclose(node.robot_yaw, 0.0, abs_tol=1e-4)

        # Check: Projected along IMU heading, eliminating lateral skew!
        assert math.isclose(node.robot_x, 1.0, abs_tol=1e-3)
        assert math.isclose(node.robot_y, 0.0, abs_tol=1e-3)

    finally:
        node.destroy_node()


def test_reverse_motion_dead_reckoning():
    """Verify that backing up (reversing) properly subtracts displacement."""
    node = Khepera4Mapper()
    try:
        node.use_imu_yaw = True

        # Heading is East (0.0 rad)
        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        node.imu_callback(imu_msg)

        # Start at (1.0, 0.0)
        odom_start = Odometry()
        odom_start.pose.pose.position.x = 1.0
        odom_start.pose.pose.position.y = 0.0
        odom_start.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_start)

        # Back up by 0.3m across 15 steps of 0.02m (< 0.05m threshold)
        num_steps = 15
        step_dist = 0.3 / num_steps
        for i in range(1, num_steps + 1):
            odom_back = Odometry()
            odom_back.pose.pose.position.x = 1.0 - (i * step_dist)
            odom_back.pose.pose.position.y = 0.0
            odom_back.pose.pose.orientation.w = 1.0
            node.odom_callback(odom_back)

        assert math.isclose(node.robot_x, 0.7, abs_tol=1e-3)
        assert math.isclose(node.robot_y, 0.0, abs_tol=1e-3)

    finally:
        node.destroy_node()


def test_odometry_slip_runaway_clamping():
    """Verify displacement jump (> 0.05m at 50Hz) is clamped."""
    node = Khepera4Mapper()
    try:
        node.use_imu_yaw = True
        node.clamp_odometry_slip = True

        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        node.imu_callback(imu_msg)

        odom_init = Odometry()
        odom_init.pose.pose.position.x = 0.0
        odom_init.pose.pose.position.y = 0.0
        odom_init.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_init)

        # Single 50Hz step jump of 0.20m (> 0.05m runaway threshold)
        odom_jump = Odometry()
        odom_jump.pose.pose.position.x = 0.20
        odom_jump.pose.pose.position.y = 0.0
        odom_jump.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_jump)

        # Clamped to 0.25 / 50.0 * 1.5 = 0.0075m
        expected_clamped = 0.25 / 50.0 * 1.5
        assert math.isclose(node.robot_x, expected_clamped, abs_tol=1e-4)

    finally:
        node.destroy_node()


def test_fallback_without_imu():
    """Verify fallback to raw odometry when use_imu_yaw is False."""
    node = Khepera4Mapper()
    try:
        node.use_imu_yaw = False

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = 1.5
        odom_msg.pose.pose.position.y = -0.8
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.5)
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        node.odom_callback(odom_msg)

        assert math.isclose(node.robot_x, 1.5, abs_tol=1e-4)
        assert math.isclose(node.robot_y, -0.8, abs_tol=1e-4)
        assert math.isclose(node.robot_yaw, 0.5, abs_tol=1e-4)

    finally:
        node.destroy_node()


def test_save_map_persistence_and_marker():
    """Verify save_map service creates valid PGM, YAML, and PNG."""
    temp_dir = tempfile.mkdtemp()
    node = Khepera4Mapper()
    try:
        node.map_save_dir = temp_dir

        # Set some grid cells and robot state
        node.robot_x = 0.5
        node.robot_y = 0.5
        node.robot_yaw = math.pi / 2.0  # North
        node.has_odom = True
        node.robot_path = [(0.0, 0.0), (0.5, 0.5)]

        col, row = node.world_to_grid(0.5, 0.5)
        node.log_odds[row, col] = 2.0  # Occupied cell

        req = Trigger.Request()
        resp = Trigger.Response()
        res = node.save_map_callback(req, resp)

        assert res.success is True
        assert os.path.exists(os.path.join(temp_dir, 'khepera_map.pgm'))
        assert os.path.exists(os.path.join(temp_dir, 'khepera_map.yaml'))
        assert os.path.exists(os.path.join(temp_dir, 'khepera_map.png'))

    finally:
        node.destroy_node()
        shutil.rmtree(temp_dir, ignore_errors=True)
