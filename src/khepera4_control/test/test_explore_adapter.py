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
Unit tests for ExploreAdapter and Semantic Arbiter.

Verifies:
1. Dashboard toggle & enable services/topics.
2. Active goal forwarding and mirroring to /explorer/active_goal (direct & autonomous).
3. Cliff safety failsafe preemption, reverse backup, and persistent hazard hold.
4. OPEN HOOK 1 (OpenFrontier): Semantic weights ingestion, utility blending,
   marker centroid parsing, and Nav2 goal dispatching.
5. OPEN HOOK 2 (SplatSearch): Visual exemplar preemption override, threshold gating,
   and goal dispatching.
6. Mode gating via vlm_target_mode ('none', 'open_frontier', 'splat_search', 'all')
   and enable_vlm_hooks flag.
7. Lifecycle management: explore_lite resumption after user goal/path and exemplar completion.
8. Nav2 action rejection recovery and clearing.
"""

import math
from typing import List, Tuple
from unittest.mock import MagicMock

from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from khepera4_control.explore_adapter import ExploreAdapter
from nav_msgs.msg import Path
import pytest
import rclpy
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def adapter_node():
    """Create and tear down an ExploreAdapter instance for testing."""
    node = ExploreAdapter()
    yield node
    node.destroy_node()


# ==============================================================================
# 1. Dashboard Compatibility Tests
# ==============================================================================

def test_dashboard_toggle_state(adapter_node):
    """Verify /set_explorer service enables and disables the adapter."""
    node = adapter_node
    assert node.enabled is False

    req = SetBool.Request()
    req.data = True
    resp = SetBool.Response()
    result = node._set_explorer_srv(req, resp)
    assert result.success is True
    assert node.enabled is True
    assert 'enabled' in result.message.lower()

    req.data = False
    result = node._set_explorer_srv(req, resp)
    assert result.success is True
    assert node.enabled is False
    assert 'paused' in result.message.lower()


def test_toggle_explorer_service(adapter_node):
    """Verify /toggle_explorer service flips active state."""
    node = adapter_node
    assert node.enabled is False

    req = Trigger.Request()
    resp = Trigger.Response()

    result = node._toggle_explorer_srv(req, resp)
    assert result.success is True
    assert node.enabled is True

    result = node._toggle_explorer_srv(req, resp)
    assert result.success is True
    assert node.enabled is False


def test_enable_topic_sync(adapter_node):
    """Verify topic-based enable syncs internal state."""
    node = adapter_node
    assert node.enabled is False

    node._enable_topic_cb(Bool(data=True))
    assert node.enabled is True

    node._enable_topic_cb(Bool(data=True))
    assert node.enabled is True

    node._enable_topic_cb(Bool(data=False))
    assert node.enabled is False


# ==============================================================================
# 2. Active Goal Forwarding & Tracking
# ==============================================================================

def test_active_goal_forwarding(adapter_node):
    """Verify active frontier goals are mirrored to /explorer/active_goal."""
    node = adapter_node
    published_goals: List[PointStamped] = []

    orig_pub = node.active_goal_pub.publish
    node.active_goal_pub.publish = lambda msg: published_goals.append(msg)

    try:
        node._dispatch_nav2_goal(1.25, -0.75, frame_id='map')

        assert node.active_goal == (1.25, -0.75)
        assert len(published_goals) >= 1
        last_msg = published_goals[-1]
        assert last_msg.header.frame_id == 'map'
        assert math.isclose(last_msg.point.x, 1.25, abs_tol=1e-4)
        assert math.isclose(last_msg.point.y, -0.75, abs_tol=1e-4)
        assert last_msg.point.z == 0.0

        node._set_enabled(False)
        assert node.active_goal is None
    finally:
        node.active_goal_pub.publish = orig_pub


def test_standard_autonomous_goal_mirroring(adapter_node):
    """Verify standard exploration without VLM mirrors active frontier to dashboard."""
    node = adapter_node
    node.enabled = True
    node.vlm_target_mode = 'none'

    markers_msg = MarkerArray()
    m = Marker()
    m.id = 42
    m.type = Marker.SPHERE
    m.pose.position.x = 2.10
    m.pose.position.y = -0.80

    markers_msg.markers = [m]
    node._frontiers_marker_cb(markers_msg)

    assert node.active_goal is not None
    assert math.isclose(node.active_goal[0], 2.10, abs_tol=1e-3)
    assert math.isclose(node.active_goal[1], -0.80, abs_tol=1e-3)


# ==============================================================================
# 3. Cliff Safety Failsafe Preemption
# ==============================================================================

def test_cliff_failsafe_preemption(adapter_node):
    """Verify /cliff_alert halts robot, cancels goals, backs up, pivots, and blacklists goal."""
    node = adapter_node
    cmd_vel_messages: List[Twist] = []
    resume_messages: List[Bool] = []

    node.cmd_vel_pub.publish = lambda msg: cmd_vel_messages.append(msg)
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    node.enabled = True
    node.active_goal = (2.0, 1.0)

    # Fire cliff alert
    cliff_msg = Bool(data=True)
    node._cliff_alert_cb(cliff_msg)

    assert node.cliff_active is True
    assert node.is_recovering_from_cliff is True
    assert node.cliff_recovery_phase == 'BACKUP'
    assert node.cliff_recovery_start_time > 0.0
    # Active goal must be purged immediately and recorded in blacklist
    assert node.active_goal is None
    assert len(node.cliff_blacklisted_goals) == 1
    assert node.cliff_blacklisted_goals[0][0] == 2.0
    assert node.cliff_blacklisted_goals[0][1] == 1.0

    # Must have told explore_lite to pause
    assert any(m.data is False for m in resume_messages)

    # Must have published reverse velocity command
    assert len(cmd_vel_messages) > 0
    assert cmd_vel_messages[-1].linear.x <= node.cliff_backup_speed
    assert cmd_vel_messages[-1].angular.z == 0.0

    # Coordination loop during BACKUP phase
    cmd_vel_messages.clear()
    node._coordination_loop()
    assert len(cmd_vel_messages) > 0
    assert cmd_vel_messages[-1].linear.x <= node.cliff_backup_speed
    assert node.cliff_recovery_phase == 'BACKUP'

    # Backup completes -> transitions to PIVOT phase
    node.cliff_active = False
    node.cliff_recovery_start_time = (
        node._get_current_time() - (node.cliff_backup_duration + 0.1)
    )
    cmd_vel_messages.clear()
    node._coordination_loop()
    assert node.cliff_recovery_phase == 'PIVOT'
    assert len(cmd_vel_messages) > 0
    assert cmd_vel_messages[-1].linear.x == 0.0
    assert cmd_vel_messages[-1].angular.z == pytest.approx(node.cliff_turn_speed, rel=1e-3)

    # Pivot completes -> recovery finished, robot stopped, exploration resumed
    node.cliff_pivot_start_time = (
        node._get_current_time() - (node.cliff_turn_duration + 0.1)
    )
    cmd_vel_messages.clear()
    resume_messages.clear()
    node._coordination_loop()

    assert node.is_recovering_from_cliff is False
    assert node.cliff_recovery_phase == 'NONE'
    assert len(cmd_vel_messages) > 0
    assert cmd_vel_messages[-1].linear.x == 0.0
    assert cmd_vel_messages[-1].angular.z == 0.0
    assert any(m.data is True for m in resume_messages)


def test_cliff_goal_blacklisting(adapter_node):
    """Verify frontiers near blacklisted cliff incident are filtered out."""
    node = adapter_node
    node.enabled = True
    node.cliff_blacklisted_goals.append((1.5, 0.5, node._get_current_time()))

    assert node._is_goal_near_cliff_blacklist(1.6, 0.55) is True
    assert node._is_goal_near_cliff_blacklist(-1.0, -1.0) is False

    markers_msg = MarkerArray()
    m1 = Marker()
    m1.id = 1
    m1.type = Marker.SPHERE
    m1.pose.position.x = 1.55
    m1.pose.position.y = 0.52
    m1.scale.x = 0.2

    m2 = Marker()
    m2.id = 2
    m2.type = Marker.SPHERE
    m2.pose.position.x = -1.2
    m2.pose.position.y = -0.8
    m2.scale.x = 0.2

    markers_msg.markers = [m1, m2]
    node._frontiers_marker_cb(markers_msg)

    # Marker 1 is near the blacklisted cliff goal and MUST be filtered out
    assert 1 not in node.candidate_frontiers
    # Marker 2 is safe and MUST be retained
    assert 2 in node.candidate_frontiers


def test_cliff_alert_persistent_hazard_hold(adapter_node):
    """Verify exploration does NOT resume if cliff hazard is still detected after recovery."""
    node = adapter_node
    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    node.enabled = True
    node.cliff_active = True
    node.is_recovering_from_cliff = True
    node.cliff_recovery_phase = 'PIVOT'
    node.cliff_pivot_start_time = (
        node._get_current_time() - (node.cliff_turn_duration + 0.5)
    )

    node._coordination_loop()

    assert node.is_recovering_from_cliff is False
    assert node.cliff_recovery_phase == 'NONE'
    # Since cliff_active is STILL True, exploration MUST NOT resume
    assert not any(m.data is True for m in resume_messages)


def test_cliff_alert_false_no_trigger(adapter_node):
    """Verify /cliff_alert with data=False does not trigger recovery."""
    node = adapter_node
    cmd_vel_messages: List[Twist] = []
    node.cmd_vel_pub.publish = lambda msg: cmd_vel_messages.append(msg)

    node._cliff_alert_cb(Bool(data=False))
    assert node.cliff_active is False
    assert node.is_recovering_from_cliff is False
    assert len(cmd_vel_messages) == 0


# ==============================================================================
# 4. OPEN HOOK 1: OpenFrontier Semantic Weighting
# ==============================================================================

def test_openfrontier_hook_ingestion(adapter_node):
    """Verify /vlm/frontier_weights updates internal weight cache."""
    node = adapter_node
    node.vlm_target_mode = 'open_frontier'
    assert len(node.vlm_frontier_weights) == 0

    weights_msg = Float32MultiArray()
    weights_msg.data = [0.0, 0.95, 1.0, 0.42, 2.0, 0.10]
    node._vlm_frontier_weights_cb(weights_msg)

    assert len(node.vlm_frontier_weights) == 3
    assert math.isclose(node.vlm_frontier_weights[0], 0.95, abs_tol=1e-4)
    assert math.isclose(node.vlm_frontier_weights[1], 0.42, abs_tol=1e-4)
    assert math.isclose(node.vlm_frontier_weights[2], 0.10, abs_tol=1e-4)


def test_openfrontier_utility_blending(adapter_node):
    """Verify mathematical formulation of blended frontier score."""
    node = adapter_node
    node.vlm_frontier_weights[5] = 0.85
    node.vlm_semantic_weight = 2.5

    # Utility(F) = alpha * InfoGain + beta * VLM_Score - gamma * Cost
    # alpha=1.5, beta=2.5, gamma=3.0
    # InfoGain=10.0, VLM_Score=0.85, Cost=2.0
    # Expected = (1.5 * 10.0) + (2.5 * 0.85) - (3.0 * 2.0) = 11.125
    expected = (1.5 * 10.0) + (2.5 * 0.85) - (3.0 * 2.0)
    blended = node.compute_blended_frontier_utility(
        frontier_id=5,
        info_gain=10.0,
        travel_cost=2.0,
        alpha=1.5,
        beta=2.5,
        gamma=3.0
    )
    assert math.isclose(blended, expected, abs_tol=1e-4)


def test_openfrontier_frontier_selection_dispatches_nav2(adapter_node):
    """Verify frontier markers with VLM weights command Nav2 and pause explore_lite."""
    node = adapter_node
    node.enabled = True
    node.vlm_target_mode = 'open_frontier'
    node.vlm_frontier_weights[10] = 0.95
    node.vlm_frontier_weights[20] = 0.10

    resume_messages: List[Bool] = []
    dispatched_goals: List[Tuple[float, float]] = []

    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)
    orig_dispatch = node._dispatch_nav2_goal
    node._dispatch_nav2_goal = lambda x, y, f='map': (
        dispatched_goals.append((x, y)), orig_dispatch(x, y, f)
    )

    markers_msg = MarkerArray()
    m1 = Marker()
    m1.id = 10
    m1.points = [Point(x=1.0, y=0.5, z=0.0) for _ in range(8)]

    m2 = Marker()
    m2.id = 20
    m2.points = [Point(x=3.0, y=2.0, z=0.0) for _ in range(8)]

    markers_msg.markers = [m1, m2]
    node._frontiers_marker_cb(markers_msg)

    assert 10 in node.candidate_frontiers
    assert 20 in node.candidate_frontiers
    assert node.active_goal is not None
    assert math.isclose(node.active_goal[0], 1.0, abs_tol=1e-3)
    assert math.isclose(node.active_goal[1], 0.5, abs_tol=1e-3)

    # Must have paused explore_lite and dispatched Nav2 goal
    assert any(m.data is False for m in resume_messages)
    assert len(dispatched_goals) >= 1
    assert math.isclose(dispatched_goals[-1][0], 1.0, abs_tol=1e-3)


# ==============================================================================
# 5. OPEN HOOK 2: SplatSearch Visual Exemplar Preemption
# ==============================================================================

def test_splatsearch_preemption_hook(adapter_node):
    """Verify /vlm/goal_override preempts exploration and redirects to target."""
    node = adapter_node
    node.vlm_target_mode = 'splat_search'
    node.latest_target_confidence = 0.88  # > 0.72

    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    override_msg = PoseStamped()
    override_msg.header.frame_id = 'map'
    override_msg.pose.position.x = 2.45
    override_msg.pose.position.y = -1.15

    node._vlm_goal_override_cb(override_msg)

    assert node.is_preempted_by_vlm is True
    assert node.vlm_override_pose == (2.45, -1.15)
    assert node.active_goal == (2.45, -1.15)
    assert any(m.data is False for m in resume_messages)


def test_splatsearch_confidence_threshold_rejection(adapter_node):
    """Verify goal override is rejected when confidence is below threshold."""
    node = adapter_node
    node.vlm_target_mode = 'splat_search'
    node.latest_target_confidence = 0.50  # < 0.72 threshold

    override_msg = PoseStamped()
    override_msg.pose.position.x = 2.0
    override_msg.pose.position.y = 2.0

    node._vlm_goal_override_cb(override_msg)

    assert node.is_preempted_by_vlm is False
    assert node.active_goal is None


# ==============================================================================
# 6. Mode Gating via vlm_target_mode & enable_vlm_hooks
# ==============================================================================

def test_vlm_target_mode_none_ignores_all_vlm(adapter_node):
    """Verify that when vlm_target_mode='none', both VLM hooks are disabled."""
    node = adapter_node
    node.vlm_target_mode = 'none'

    # OpenFrontier weights ignored
    weights_msg = Float32MultiArray()
    weights_msg.data = [0.0, 0.99]
    node._vlm_frontier_weights_cb(weights_msg)
    assert len(node.vlm_frontier_weights) == 0

    # SplatSearch override ignored
    override_msg = PoseStamped()
    override_msg.pose.position.x = 4.0
    override_msg.pose.position.y = 4.0
    node._vlm_goal_override_cb(override_msg)
    assert node.is_preempted_by_vlm is False
    assert node.active_goal is None


def test_vlm_target_mode_openfrontier_only(adapter_node):
    """Verify open_frontier mode accepts weights but rejects SplatSearch override."""
    node = adapter_node
    node.vlm_target_mode = 'open_frontier'

    weights_msg = Float32MultiArray()
    weights_msg.data = [0.0, 0.85]
    node._vlm_frontier_weights_cb(weights_msg)
    assert len(node.vlm_frontier_weights) == 1

    override_msg = PoseStamped()
    override_msg.pose.position.x = 3.0
    override_msg.pose.position.y = 3.0
    node._vlm_goal_override_cb(override_msg)
    assert node.is_preempted_by_vlm is False


def test_vlm_target_mode_splatsearch_only(adapter_node):
    """Verify splat_search mode accepts override but rejects OpenFrontier weights."""
    node = adapter_node
    node.vlm_target_mode = 'splat_search'

    weights_msg = Float32MultiArray()
    weights_msg.data = [0.0, 0.85]
    node._vlm_frontier_weights_cb(weights_msg)
    assert len(node.vlm_frontier_weights) == 0

    override_msg = PoseStamped()
    override_msg.pose.position.x = 3.0
    override_msg.pose.position.y = 3.0
    node._vlm_goal_override_cb(override_msg)
    assert node.is_preempted_by_vlm is True


def test_vlm_hooks_disabled_gating(adapter_node):
    """Verify that when enable_vlm_hooks=False, VLM inputs are safely ignored."""
    node = adapter_node
    node.vlm_target_mode = 'all'
    node.enable_vlm_hooks = False

    weights_msg = Float32MultiArray()
    weights_msg.data = [0.0, 0.99]
    node._vlm_frontier_weights_cb(weights_msg)
    assert len(node.vlm_frontier_weights) == 0

    override_msg = PoseStamped()
    override_msg.pose.position.x = 5.0
    override_msg.pose.position.y = 5.0
    node._vlm_goal_override_cb(override_msg)
    assert node.is_preempted_by_vlm is False
    assert node.active_goal is None


# ==============================================================================
# 7. User Waypoints, Guidance & Lifecycle Resumption
# ==============================================================================

def test_user_goal_preemption_and_resumption(adapter_node):
    """Verify user goal pauses explore_lite, and reaching goal resumes exploration."""
    node = adapter_node
    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    goal_msg = PoseStamped()
    goal_msg.pose.position.x = 0.88
    goal_msg.pose.position.y = 0.44

    node._user_goal_cb(goal_msg)

    assert node.is_single_user_goal is True
    assert node.active_goal == (0.88, 0.44)
    assert any(m.data is False for m in resume_messages)

    # Simulate Nav2 goal completion
    resume_messages.clear()
    node._nav2_result_cb(MagicMock())

    assert node.is_single_user_goal is False
    assert node.active_goal is None
    # Must have resumed explore_lite
    assert any(m.data is True for m in resume_messages)


def test_user_path_and_clear(adapter_node):
    """Verify user path sequence and clear path reset."""
    node = adapter_node
    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    path_msg = Path()
    p1 = PoseStamped()
    p1.pose.position.x = 0.5
    p1.pose.position.y = 0.5
    p2 = PoseStamped()
    p2.pose.position.x = 1.0
    p2.pose.position.y = 1.0
    path_msg.poses = [p1, p2]

    node._user_path_cb(path_msg)
    assert node.active_goal == (0.5, 0.5)
    assert len(node.user_waypoints) == 1

    # Clear path
    node.enabled = True
    node._clear_path_cb(Bool(data=True))
    assert len(node.user_waypoints) == 0
    assert node.is_single_user_goal is False
    assert any(m.data is True for m in resume_messages)


def test_splatsearch_exemplar_completion_resumes_exploration(adapter_node):
    """Verify that reaching a SplatSearch target exemplar resets flag and resumes exploration."""
    node = adapter_node
    node.enabled = True
    node.is_preempted_by_vlm = True
    node.active_goal = (2.0, 2.0)

    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    node._nav2_result_cb(MagicMock())

    assert node.is_preempted_by_vlm is False
    assert node.active_goal is None
    assert any(m.data is True for m in resume_messages)


def test_nav2_goal_rejected_clears_active_goal(adapter_node):
    """Verify Nav2 goal rejection resets active_goal and resumes exploration."""
    node = adapter_node
    node.enabled = True
    node.active_goal = (1.5, 1.5)

    resume_messages: List[Bool] = []
    node.explore_resume_pub.publish = lambda msg: resume_messages.append(msg)

    mock_future = MagicMock()
    mock_handle = MagicMock()
    mock_handle.accepted = False
    mock_future.result.return_value = mock_handle

    node._nav2_goal_response_cb(mock_future)

    assert node.active_goal is None
    assert any(m.data is True for m in resume_messages)
