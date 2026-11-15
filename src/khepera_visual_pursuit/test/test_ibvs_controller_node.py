# Copyright 2026 Andrei Berteanu.
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

"""Integration and unit tests for IBVSControllerNode (Node 3)."""

from typing import List

import pytest

from khepera_visual_pursuit.ibvs_controller_node import IBVSControllerNode
from khepera_visual_pursuit.ros_compat import (
    Bool,
    Float32,
    LaserScan,
    Point32,
    PolygonStamped,
    Twist,
    rclpy,
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_init_teardown():
    """Manage ROS 2 context lifecycle for IBVS controller node tests."""
    if rclpy is not None and hasattr(rclpy, "ok"):
        if not rclpy.ok():
            rclpy.init()
        yield
        if rclpy.ok():
            rclpy.shutdown()
    else:
        yield


def test_ibvs_controller_node_initialization() -> None:
    """Verify node initialization, parameters, and communication channels."""
    node = IBVSControllerNode()
    assert node.get_name() == "ibvs_controller_node"
    assert node.keypoints_sub is not None
    assert node.confidence_sub is not None
    assert node.scan_sub is not None
    assert node.cmd_vel_pub is not None
    assert node.status_pub is not None
    assert node.has_active_target is False
    node.destroy_node()


def test_ibvs_controller_node_polygon_processing() -> None:
    """Verify polygon centroid and scale calculation from incoming message."""
    node = IBVSControllerNode()

    # Create polygon in pixel coordinates: center around (320, 240), box 100x100
    msg = PolygonStamped()
    msg.polygon.points = [
        Point32(x=270.0, y=190.0, z=0.0),
        Point32(x=370.0, y=190.0, z=0.0),
        Point32(x=370.0, y=290.0, z=0.0),
        Point32(x=270.0, y=290.0, z=0.0),
    ]

    node._on_target_keypoints(msg)
    assert node.has_active_target is True
    assert abs(node.latest_norm_x) < 1e-3  # Centered horizontally (320 px)
    assert abs(node.latest_norm_y) < 1e-3  # Centered vertically (240 px)
    assert node.latest_area_frac > 0.0

    node.destroy_node()


def test_ibvs_controller_node_laser_scan_obstacle() -> None:
    """Verify minimum obstacle distance extraction from frontal scan rays."""
    node = IBVSControllerNode()

    scan = LaserScan()
    scan.angle_min = -3.14159
    scan.angle_max = 3.14159
    scan.angle_increment = 0.01745
    scan.range_min = 0.02
    scan.range_max = 2.0
    # Create 360 rays, with an obstacle at 0.05m right in front (index ~ 180)
    num_rays = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
    scan.ranges = [1.5] * num_rays
    center_idx = num_rays // 2
    scan.ranges[center_idx] = 0.05  # Obstacle at 5 cm

    node._on_laser_scan(scan)
    assert abs(node.latest_obstacle_distance - 0.05) < 1e-4

    node.destroy_node()


def test_ibvs_controller_node_control_cycle_emergency_stop() -> None:
    """Verify emergency stop publication when obstacle is closer than 8 cm."""
    node = IBVSControllerNode()

    published_twists: List[Twist] = []
    published_status: List[Bool] = []
    node.cmd_vel_pub.publish = lambda m: published_twists.append(m)
    node.status_pub.publish = lambda m: published_status.append(m)

    # Set active target but frontal obstacle at 0.04m
    node.pursuit_enabled = True
    node.has_active_target = True
    node.latest_norm_x = 0.0
    node.latest_area_frac = 0.02
    node.latest_confidence = 0.9
    node.latest_obstacle_distance = 0.04

    node._on_control_cycle()

    assert len(published_twists) == 1
    assert published_twists[0].linear.x == 0.0
    assert published_twists[0].angular.z == 0.0
    assert len(published_status) == 1
    assert published_status[0].data is False

    node.destroy_node()


def test_ibvs_controller_node_control_cycle_active_pursuit() -> None:
    """Verify forward driving and steering when clear and confident."""
    node = IBVSControllerNode()

    published_twists: List[Twist] = []
    published_status: List[Bool] = []
    node.cmd_vel_pub.publish = lambda m: published_twists.append(m)
    node.status_pub.publish = lambda m: published_status.append(m)

    # Target slightly left (-0.2), small area (distant), confident, clear of obstacles
    node.pursuit_enabled = True
    node.has_active_target = True
    node.latest_norm_x = -0.2
    node.latest_area_frac = 0.03
    node.latest_confidence = 0.85
    node.latest_obstacle_distance = 1.2

    node._on_control_cycle()

    assert len(published_twists) == 1
    assert published_twists[0].linear.x > 0.0
    assert published_twists[0].angular.z > 0.0  # Steers left to center target
    assert len(published_status) == 1
    assert published_status[0].data is True

    node.destroy_node()


def test_ibvs_controller_node_confidence_callback() -> None:
    """Verify confidence callback updates state and deactivates when lost."""
    node = IBVSControllerNode()
    node.has_active_target = True

    node._on_tracking_confidence(Float32(data=0.88))
    assert node.latest_confidence == 0.88
    assert node.has_active_target is True

    node._on_tracking_confidence(Float32(data=0.01))
    assert node.latest_confidence == 0.01
    assert node.has_active_target is False

    node.destroy_node()


def test_ibvs_controller_node_auto_start_default_false() -> None:
    """Verify that auto_start parameter defaults to False and pursuit is disabled."""
    node = IBVSControllerNode()
    assert node.get_parameter("auto_start").value is False
    assert node.pursuit_enabled is False

    # Control cycle should not publish anything when pursuit is disabled
    published_twists: List[Twist] = []
    node.cmd_vel_pub.publish = lambda m: published_twists.append(m)
    node._on_control_cycle()
    assert len(published_twists) == 0

    node.destroy_node()


def test_ibvs_controller_node_pursuit_enable_toggle_and_halt() -> None:
    """Verify enabling pursuit engages controller and disabling pursuit halts immediately."""
    node = IBVSControllerNode()

    published_twists: List[Twist] = []
    published_status: List[Bool] = []
    node.cmd_vel_pub.publish = lambda m: published_twists.append(m)
    node.status_pub.publish = lambda m: published_status.append(m)

    # Enable pursuit
    node._on_pursuit_enable(Bool(data=True))
    assert node.pursuit_enabled is True

    # Disable pursuit: must reset target state, publish zero Twist and status False
    node.has_active_target = True
    node._on_pursuit_enable(Bool(data=False))
    assert node.pursuit_enabled is False
    assert node.has_active_target is False
    assert len(published_twists) == 1
    assert published_twists[0].linear.x == 0.0
    assert published_twists[0].angular.z == 0.0
    assert len(published_status) == 1
    assert published_status[0].data is False

    # Verify subsequent control cycle outputs nothing
    published_twists.clear()
    node._on_control_cycle()
    assert len(published_twists) == 0

    node.destroy_node()


def test_ibvs_controller_node_target_lost_initiates_recovery_sweep() -> None:
    """Verify target loss or low confidence triggers search rotation (recovery sweep)."""
    node = IBVSControllerNode()

    published_twists: List[Twist] = []
    published_status: List[Bool] = []
    node.cmd_vel_pub.publish = lambda m: published_twists.append(m)
    node.status_pub.publish = lambda m: published_status.append(m)

    node.pursuit_enabled = True
    node.has_active_target = False  # No target detected
    node.latest_obstacle_distance = 1.0

    node._on_control_cycle()

    assert len(published_twists) == 1
    assert published_twists[0].linear.x == 0.0
    # Recovery sweep rotates to search for target
    assert abs(published_twists[0].angular.z) > 0.0
    assert len(published_status) == 1
    assert published_status[0].data is False

    node.destroy_node()
