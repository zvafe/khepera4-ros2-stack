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

"""Unit tests for PursuitBTSupervisorNode Behavior Tree arbitration and motion gating."""

from geometry_msgs.msg import Twist
import pytest
import rclpy
from std_msgs.msg import Bool

from khepera_visual_pursuit.pursuit_bt_supervisor_node import (
    STATE_FRONTIER_EXPLORATION,
    STATE_IDLE,
    STATE_TARGET_RECOVERY,
    STATE_TERMINAL_STOP,
    STATE_VISUAL_PURSUIT,
    PursuitBTSupervisorNode,
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    """Ensure rclpy is initialized during testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_supervisor_initialization():
    """Verify default initial state of the Behavior Tree supervisor node."""
    node = PursuitBTSupervisorNode()
    try:
        assert node.current_state in (STATE_FRONTIER_EXPLORATION, STATE_IDLE)
        assert not node.target_docked_latched
        assert node.standoff_distance_m == pytest.approx(0.05, rel=1e-2)
    finally:
        node.destroy_node()


def test_exploration_command_multiplexing():
    """Verify that when no target is active, explorer velocity is routed to cmd_vel."""
    node = PursuitBTSupervisorNode()
    published_cmds = []
    node.cmd_vel_pub.publish = lambda m: published_cmds.append(m)

    try:
        node.explorer_enabled = True
        node.target_locked = False
        node.tracking_confidence = 0.0

        exp_twist = Twist()
        exp_twist.linear.x = 0.10
        exp_twist.angular.z = 0.50
        node._on_explorer_cmd(exp_twist)

        node._on_supervision_cycle()

        assert node.current_state == STATE_FRONTIER_EXPLORATION
        assert len(published_cmds) > 0
        assert published_cmds[-1].linear.x == pytest.approx(0.10, rel=1e-2)
        assert published_cmds[-1].angular.z == pytest.approx(0.50, rel=1e-2)
    finally:
        node.destroy_node()


def test_visual_pursuit_preemption_and_topic_gating():
    """Verify target lock preempts explorer and routes pursuit command."""
    node = PursuitBTSupervisorNode()
    published_cmds = []
    explorer_enables = []
    node.cmd_vel_pub.publish = lambda m: published_cmds.append(m)
    node.explorer_enable_pub.publish = lambda m: explorer_enables.append(m.data)

    try:
        node.explorer_enabled = True

        # Provide both explorer and pursuit commands
        exp_twist = Twist()
        exp_twist.linear.x = 0.10
        node._on_explorer_cmd(exp_twist)

        pursuit_twist = Twist()
        pursuit_twist.linear.x = 0.08
        pursuit_twist.angular.z = -0.30
        node._on_pursuit_cmd(pursuit_twist)

        # Trigger target lock
        node._on_target_status(Bool(data=True))
        node._on_supervision_cycle()

        assert node.current_state == STATE_VISUAL_PURSUIT
        assert len(published_cmds) > 0
        assert published_cmds[-1].linear.x == pytest.approx(0.08, rel=1e-2)
        assert published_cmds[-1].angular.z == pytest.approx(-0.30, rel=1e-2)
        assert False in explorer_enables
    finally:
        node.destroy_node()


def test_target_recovery_when_target_lost_temporarily():
    """Verify supervisor enters TARGET_RECOVERY before falling back to exploration."""
    node = PursuitBTSupervisorNode()
    published_cmds = []
    node.cmd_vel_pub.publish = lambda m: published_cmds.append(m)

    try:
        node.pursuit_enabled = True
        node.explorer_enabled = True
        node.target_locked = True
        node.target_lost_recovery_sec = 2.0

        # Mark target seen at t=10.0
        node.last_target_seen_time = 10.0
        node._get_current_time = lambda: 11.0  # 1.0s elapsed (< 2.0s timeout)

        node.target_locked = False
        node.tracking_confidence = 0.0

        recovery_twist = Twist()
        recovery_twist.angular.z = 0.40
        node._on_pursuit_cmd(recovery_twist)

        node._on_supervision_cycle()
        assert node.current_state == STATE_TARGET_RECOVERY
        assert published_cmds[-1].angular.z == pytest.approx(0.40, rel=1e-2)

        # Now simulate timeout (t=13.0, 3.0s elapsed > 2.0s timeout)
        node._get_current_time = lambda: 13.0
        node._on_supervision_cycle()
        assert node.current_state == STATE_FRONTIER_EXPLORATION
    finally:
        node.destroy_node()


def test_terminal_standoff_stop_at_surface():
    """Verify arrival at 5 cm surface standoff triggers TERMINAL_STOP and latching."""
    node = PursuitBTSupervisorNode()
    published_cmds = []
    stop_signals = []
    node.cmd_vel_pub.publish = lambda m: published_cmds.append(m)
    node.stop_pub.publish = lambda m: stop_signals.append(m.data)

    try:
        node.pursuit_enabled = True
        node.current_state = STATE_VISUAL_PURSUIT
        node.standoff_distance_m = 0.05
        node.standoff_tolerance_m = 0.015
        node.latest_frontal_distance = 0.055  # Inside 5.0 ± 1.5 cm stop boundary

        # 3 ticks to confirm arrival
        node._on_supervision_cycle()
        node._on_supervision_cycle()
        node._on_supervision_cycle()

        assert node.current_state == STATE_TERMINAL_STOP
        assert node.target_docked_latched
        assert published_cmds[-1].linear.x == 0.0
        assert published_cmds[-1].angular.z == 0.0
        assert True in stop_signals

        # Verify latching on subsequent cycles
        node.latest_frontal_distance = 1.0
        node._on_supervision_cycle()
        assert node.current_state == STATE_TERMINAL_STOP
        assert published_cmds[-1].linear.x == 0.0
    finally:
        node.destroy_node()


def test_speed_limit_clamping():
    """Verify speed dial commands clip output velocities."""
    node = PursuitBTSupervisorNode()
    published_cmds = []
    node.cmd_vel_pub.publish = lambda m: published_cmds.append(m)

    try:
        node.explorer_enabled = True
        # Set dial limit to 0.05 m/s and 0.40 rad/s
        dial_msg = Twist()
        dial_msg.linear.x = 0.05
        dial_msg.angular.z = 0.40
        node._on_speed_limits(dial_msg)

        # Explorer proposes 0.12 m/s and 0.90 rad/s
        exp_twist = Twist()
        exp_twist.linear.x = 0.12
        exp_twist.angular.z = 0.90
        node._on_explorer_cmd(exp_twist)

        node._on_supervision_cycle()

        assert published_cmds[-1].linear.x == pytest.approx(0.05, rel=1e-2)
        assert published_cmds[-1].angular.z == pytest.approx(0.40, rel=1e-2)
    finally:
        node.destroy_node()
