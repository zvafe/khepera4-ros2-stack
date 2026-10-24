#!/usr/bin/env python3
"""
Test Suite: Clock, QoS, TF, Collision & Teleop Override Fixes.

Validates:
1. Khepera4Mapper publishers (/map, /elevation_map, /surface_map) use TRANSIENT_LOCAL durability for RViz compatibility.
2. Khepera4Mapper broadcasts map -> odom transform to close the TF tree.
3. ReactiveExplorer suppresses false impact detections when stationary (e.g., spawn contact / physics settle).
4. MappingDashboard automatically pauses auto-exploration when manual WASD or D-Pad teleop is commanded.
5. ReactiveExplorer publishes latched status on /explorer/enable to keep UI in sync.
"""

import os
import sys
import time
import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

import rclpy
from rclpy.qos import DurabilityPolicy
from sensor_msgs.msg import Imu

from khepera4_control.khepera4_mapper import Khepera4Mapper
from khepera4_control.reactive_explorer import ReactiveExplorer
from khepera4_control.mapping_dashboard import MappingDashboardWindow


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    if not rclpy.ok():
        rclpy.init()
    yield
    # Cleanup


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_mapper_map_qos_durability_transient_local():
    """Verify that /map, /elevation_map, /surface_map offer TRANSIENT_LOCAL durability for RViz."""
    mapper = Khepera4Mapper()
    try:
        assert mapper.map_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert mapper.elevation_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert mapper.surface_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    finally:
        mapper.destroy_node()


def test_mapper_publishes_map_to_odom_tf():
    """Verify that _publish_fused_odom broadcasts map -> odom transform."""
    mapper = Khepera4Mapper()
    try:
        assert hasattr(mapper, 'tf_broadcaster')
        assert mapper.tf_broadcaster is not None

        # Mock sendTransform
        sent_transforms = []
        mapper.tf_broadcaster.sendTransform = lambda t: sent_transforms.append(t)

        stamp = mapper.get_clock().now().to_msg()
        mapper._publish_fused_odom(stamp, 0.1, 0.0, 0.0)

        assert len(sent_transforms) == 1
        tf_msg = sent_transforms[0]
        assert tf_msg.header.frame_id == mapper.map_frame  # 'map'
        assert tf_msg.child_frame_id == 'odom'
        assert tf_msg.transform.translation.x == 0.0
        assert tf_msg.transform.translation.y == 0.0
        assert tf_msg.transform.translation.z == 0.0
        assert tf_msg.transform.rotation.w == 1.0
    finally:
        mapper.destroy_node()


def test_reactive_explorer_no_impact_when_stationary():
    """Verify that stationary Khepera (e.g. spawn settling, v=0) ignores IMU shock and stays in CRUISE."""
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.last_commanded_v = 0.0  # Stationary

        imu_msg = Imu()
        imu_msg.header.stamp.sec = 1
        imu_msg.header.stamp.nanosec = 0
        imu_msg.linear_acceleration.x = -3.5  # Large deceleration spike from spawn landing
        imu_msg.linear_acceleration.z = 9.81
        imu_msg.orientation.w = 1.0

        explorer.imu_callback(imu_msg)
        assert explorer.imu_impact_detected is False

        explorer.control_loop()
        assert explorer.state == ReactiveExplorer.STATE_CRUISE
    finally:
        explorer.destroy_node()


def test_explorer_status_latched_publisher():
    """Verify ReactiveExplorer creates a latched /explorer/enable publisher."""
    explorer = ReactiveExplorer()
    try:
        assert hasattr(explorer, 'explorer_status_pub')
        assert explorer.explorer_status_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    finally:
        explorer.destroy_node()


def test_dashboard_teleop_override_pauses_auto_explore(qapp):
    """Verify that pressing WASD keys or D-Pad immediately pauses auto-explore to prevent /cmd_vel fights."""
    window = MappingDashboardWindow()
    try:
        window.is_auto_exploring = True
        window.btn_auto_explore.setText("■ Stop Auto Explore")

        # Simulate user pressing 'W'
        event_w = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
        window.keyPressEvent(event_w)

        # Auto-explore should be deactivated automatically
        assert window.is_auto_exploring is False
        assert "Manual Control" in window.lbl_explore_status.text()
        assert "Start Auto Explore" in window.btn_auto_explore.text()

        # Turn auto-explore back on
        window.is_auto_exploring = True

        # Simulate user pressing D-Pad FORWARD
        window.on_dpad_pressed('FORWARD')
        assert window.is_auto_exploring is False
    finally:
        window.close()
