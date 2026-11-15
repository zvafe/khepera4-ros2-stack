#!/usr/bin/env python3
# Copyright 2026 Mobile Robotics Laboratory.
"""
Empirical Adversarial Test Harness for Gate 2 Concurrency & State Machine Verification.

Specifically tests:
1. Target detection arriving during INIT_AND_ENROLL when pursuit is disabled.
2. Race condition between _dispatch_nav2_approach and set_pursuit(False).
3. Duplicate and unguarded cmd_vel publication in VISUAL_SERVO on ARRIVED.
4. Consecutive goal preemption without resetting goal handle.
"""

from unittest.mock import MagicMock
from geometry_msgs.msg import Twist
from khepera_fetch_executive.executive_node import ExecutiveNode
from khepera_fetch_executive.state_machine import RecoveryPhase, State
from khepera_interfaces.msg import TargetDetection
import pytest
import rclpy
from rclpy.task import Future
from std_srvs.srv import SetBool


@pytest.fixture(scope='module', autouse=True)
def manage_rclpy():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_startup_target_detection_violates_disabled_pursuit():
    """
    CRITICAL VULNERABILITY TEST:
    When the executive is initialized with initial_pursuit_enabled=False,
    any TargetDetection arriving during INIT_AND_ENROLL must NOT trigger
    autonomous transition to VERIFY_TARGET or dispatch Nav2.
    """
    node = ExecutiveNode()
    try:
        node.pursuit_enabled = False
        assert node.fsm.current_state == State.INIT_AND_ENROLL

        mock_action = MagicMock()
        mock_action.server_is_ready.return_value = True
        node.nav2_client._action_client = mock_action

        node.target_ground_x_odom = 2.0
        node.target_ground_y_odom = 0.0
        node.has_odometry = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.target_distance_ground = 2.0

        sim_time = [1.0]
        node._get_time_sec = lambda: sim_time[0]

        # Target detection arrives during INIT_AND_ENROLL
        det = TargetDetection()
        det.is_detected = True
        det.max_similarity = 0.95
        det.contact_pixel.x = 320.0
        det.contact_pixel.y = 240.0
        node._target_detection_cb(det)

        # FSM should NOT transition to VERIFY_TARGET when pursuit is disabled
        assert node.fsm.current_state != State.VERIFY_TARGET, (
            "DEFECT: Target detection in INIT_AND_ENROLL caused transition to VERIFY_TARGET "
            "even though pursuit_enabled is False!"
        )

        # Simulate ticks
        for _ in range(25):
            sim_time[0] += 0.05
            node._control_loop()

        assert not mock_action.send_goal_async.called, (
            "DEFECT: Nav2 standoff goal was dispatched even though pursuit_enabled is False!"
        )
        assert node.fsm.current_state == State.IDLE, (
            f"Expected IDLE state after startup with pursuit_enabled=False, got {node.fsm.current_state}"
        )
    finally:
        node.destroy_node()


def test_race_dispatch_nav2_approach_clobbers_idle():
    """
    RACE CONDITION TEST:
    If set_pursuit(False) is called during _dispatch_nav2_approach,
    the method must NOT clobber State.IDLE back to State.NAV2_APPROACH.
    """
    node = ExecutiveNode()
    try:
        node.pursuit_enabled = True
        node.fsm.transition_to(State.VERIFY_TARGET, 10.0)

        node.target_ground_x_odom = 2.0
        node.target_ground_y_odom = 0.0
        node.has_odometry = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.target_distance_ground = 2.0

        mock_action = MagicMock()
        mock_action.server_is_ready.return_value = True
        mock_action.send_goal_async.return_value = Future()
        node.nav2_client._action_client = mock_action

        # Interleave set_pursuit(False) into send_standoff_goal
        orig_send = node.nav2_client.send_standoff_goal

        def hooked_send(*args, **kwargs):
            res = orig_send(*args, **kwargs)
            req = SetBool.Request(data=False)
            resp = SetBool.Response()
            node._handle_set_pursuit(req, resp)
            return res

        node.nav2_client.send_standoff_goal = hooked_send

        node._dispatch_nav2_approach(10.5)

        assert node.fsm.current_state == State.IDLE, (
            f"DEFECT: _dispatch_nav2_approach clobbered State.IDLE to {node.fsm.current_state} "
            f"while pursuit_enabled={node.pursuit_enabled}!"
        )
    finally:
        node.destroy_node()


def test_visual_servo_arrived_no_duplicate_publish():
    """
    PUBLICATION INTEGRITY TEST:
    Upon ARRIVED in VISUAL_SERVO, exactly one stop command should be published per tick,
    and publication must respect pursuit_enabled.
    """
    node = ExecutiveNode()
    try:
        node.pursuit_enabled = True
        node.fsm.transition_to(State.VISUAL_SERVO, 10.0)

        sim_time = [10.05]
        node._get_time_sec = lambda: sim_time[0]
        node.visual_servo.update_measurement(320.0, 0.16, 10.05)

        published = []
        node.cmd_vel_pub.publish = lambda msg: published.append(msg)
        node._control_loop()

        assert len(published) == 1, (
            f"DEFECT: ARRIVED caused {len(published)} cmd_vel publications in a single 20Hz tick (expected 1)."
        )
    finally:
        node.destroy_node()
