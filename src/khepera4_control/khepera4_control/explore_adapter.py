#!/usr/bin/env python3
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
Explore Adapter & Semantic Arbiter Node for Khepera IV Mobile Robot.

Coordinates:
1. explore_lite lifecycle and Nav2 /navigate_to_pose action dispatching.
2. Dashboard backward compatibility:
   - Services: /set_explorer, /toggle_explorer
   - Latched status: /explorer/status, /explorer/enable
   - Active exploration goal tracking: /explorer/active_goal
   - User waypoint & path guidance: /explorer/user_goal, /explorer/user_path, /explorer/clear_path
3. Failsafe drop-off safety:
   - Immediate halt & priority reverse backup upon /cliff_alert.

OPEN VLM EXTENSION HOOKS:
- Hook 1 (OpenFrontier): Subscribes to /vlm/frontier_weights to blend zero-shot
  text-conditioned semantic priors P(target|F_i) into geometric frontier utility.
- Hook 2 (SplatSearch / Visual Exemplar): Subscribes to /vlm/goal_override to
  preempt geometric exploration and direct the robot to a target exemplar hypothesis.
"""

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray

try:
    from explore_lite_msgs.msg import ExploreStatus
    HAVE_EXPLORE_LITE_MSGS = True
except ImportError:
    HAVE_EXPLORE_LITE_MSGS = False


class ExploreAdapter(Node):
    """
    Bridge explore_lite, Nav2, Qt mapping_dashboard, and VLM target seekers.

    Coordinates frontier exploration, safety failsafes, and semantic arbiters.
    """

    def __init__(self):
        """Initialize the ExploreAdapter node, interfaces, and open hooks."""
        super().__init__('explore_adapter')

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self.declare_parameter('enabled', False)
        self.declare_parameter('vlm_target_mode', 'none')
        self.declare_parameter('enable_vlm_hooks', True)
        self.declare_parameter('vlm_semantic_weight', 2.0)
        self.declare_parameter('exemplar_override_threshold', 0.72)
        self.declare_parameter('cliff_backup_duration', 1.2)
        self.declare_parameter('cliff_backup_speed', -0.08)
        self.declare_parameter('cliff_turn_duration', 1.5)
        self.declare_parameter('cliff_turn_speed', 0.85)
        self.declare_parameter('cliff_blacklist_radius', 0.6)
        self.declare_parameter('control_rate', 20.0)

        self.enabled: bool = bool(self.get_parameter('enabled').value)
        self.vlm_target_mode: str = str(
            self.get_parameter('vlm_target_mode').value
        ).strip().lower()
        self.enable_vlm_hooks: bool = bool(self.get_parameter('enable_vlm_hooks').value)
        self.vlm_semantic_weight: float = float(
            self.get_parameter('vlm_semantic_weight').value
        )
        self.exemplar_override_threshold: float = float(
            self.get_parameter('exemplar_override_threshold').value
        )
        self.cliff_backup_duration: float = float(
            self.get_parameter('cliff_backup_duration').value
        )
        self.cliff_backup_speed: float = float(
            self.get_parameter('cliff_backup_speed').value
        )
        self.cliff_turn_duration: float = float(
            self.get_parameter('cliff_turn_duration').value
        )
        self.cliff_turn_speed: float = float(
            self.get_parameter('cliff_turn_speed').value
        )
        self.cliff_blacklist_radius: float = float(
            self.get_parameter('cliff_blacklist_radius').value
        )
        self.control_rate: float = float(self.get_parameter('control_rate').value)

        # ----------------------------------------------------------------------
        # Internal State
        # ----------------------------------------------------------------------
        self.active_goal: Optional[Tuple[float, float]] = None
        self.user_waypoints = collections.deque()
        self.is_single_user_goal: bool = False

        # VLM Hooks State
        self.vlm_frontier_weights: Dict[int, float] = {}
        self.is_preempted_by_vlm: bool = False
        self.latest_target_confidence: float = 0.0
        self.vlm_override_pose: Optional[Tuple[float, float]] = None

        # Cliff Safety State
        self.cliff_active: bool = False
        self.is_recovering_from_cliff: bool = False
        self.cliff_recovery_phase: str = 'NONE'
        self.cliff_recovery_start_time: float = 0.0
        self.cliff_pivot_start_time: float = 0.0
        self.cliff_blacklisted_goals: List[Tuple[float, float, float]] = []

        # Nav2 Goal Handle
        self._current_goal_handle: Optional[ClientGoalHandle] = None
        self._is_nav2_goal_active: bool = False

        # Candidate Frontiers Cache (from explore_lite /explore/frontiers)
        self.candidate_frontiers: Dict[int, Tuple[float, float, float]] = {}

        # ----------------------------------------------------------------------
        # Nav2 Action Client
        # ----------------------------------------------------------------------
        self._nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ----------------------------------------------------------------------
        # Latched QoS Profile for Dashboard Compatibility
        # ----------------------------------------------------------------------
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ----------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------
        self.status_pub = self.create_publisher(Bool, '/explorer/status', qos_latched)
        self.enable_pub = self.create_publisher(Bool, '/explorer/enable', qos_latched)
        self.active_goal_pub = self.create_publisher(PointStamped, '/explorer/active_goal', 10)
        self.explore_resume_pub = self.create_publisher(Bool, '/explore/resume', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vlm_debug_pub = self.create_publisher(String, '/vlm/arbiter_debug', 10)

        # Broadcast initial latched status
        self.status_pub.publish(Bool(data=self.enabled))
        self.enable_pub.publish(Bool(data=self.enabled))

        # ----------------------------------------------------------------------
        # Services (Compatible with mapping_dashboard.py)
        # ----------------------------------------------------------------------
        self.set_srv = self.create_service(SetBool, '/set_explorer', self._set_explorer_srv)
        self.toggle_srv = self.create_service(
            Trigger, '/toggle_explorer', self._toggle_explorer_srv
        )

        # ----------------------------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------------------------
        self.create_subscription(Bool, '/explorer/enable', self._enable_topic_cb, qos_latched)
        self.create_subscription(Bool, '/cliff_alert', self._cliff_alert_cb, 10)
        self.create_subscription(PoseStamped, '/explorer/user_goal', self._user_goal_cb, 10)
        self.create_subscription(Path, '/explorer/user_path', self._user_path_cb, 10)
        self.create_subscription(Bool, '/explorer/clear_path', self._clear_path_cb, 10)
        self.create_subscription(MarkerArray, '/explore/frontiers', self._frontiers_marker_cb, 10)

        if HAVE_EXPLORE_LITE_MSGS:
            self.create_subscription(
                ExploreStatus,
                '/explore/status',
                self._explore_status_cb,
                10
            )

        # ======================================================================
        # OPEN HOOK 1: OpenFrontier Semantic Weighting
        # OpenFrontier publishes candidate frontier semantic relevance probabilities:
        # P(target | frontier_i) conditioned on user query text & visual Set-of-Marks.
        # Format: Float32MultiArray [frontier_id_0, score_0, frontier_id_1, score_1, ...]
        # ======================================================================
        self.create_subscription(
            Float32MultiArray,
            '/vlm/frontier_weights',
            self._vlm_frontier_weights_cb,
            10
        )

        # ======================================================================
        # OPEN HOOK 2: SplatSearch / Visual Exemplar Target Preemption
        # When 3D Gaussian Splatting scene model feature matching or DINOv2 exemplar
        # detector identifies a visual target match exceeding confidence threshold,
        # SplatSearch dispatches a goal override pose to preempt frontier exploration.
        # ======================================================================
        self.create_subscription(
            PoseStamped,
            '/vlm/goal_override',
            self._vlm_goal_override_cb,
            10
        )
        self.create_subscription(
            Float32,
            '/vlm/target_confidence',
            self._vlm_target_confidence_cb,
            10
        )

        # ----------------------------------------------------------------------
        # Periodic Safety & Coordination Timer
        # ----------------------------------------------------------------------
        timer_period = 1.0 / max(1.0, self.control_rate)
        self.timer = self.create_timer(timer_period, self._coordination_loop)

        self.get_logger().info(
            f'ExploreAdapter initialized (enabled={self.enabled}, '
            f'vlm_mode={self.vlm_target_mode}, vlm_weight={self.vlm_semantic_weight}).'
        )

    # ==========================================================================
    # Clock Helper
    # ==========================================================================

    def _get_current_time(self) -> float:
        """Get current clock time in seconds, supporting simulation clock."""
        clock_t = self.get_clock().now().nanoseconds * 1e-9
        if clock_t <= 0.0:
            return time.time()
        return clock_t

    # ==========================================================================
    # VLM Mode Gating Helpers
    # ==========================================================================

    def is_open_frontier_enabled(self) -> bool:
        """Check if OpenFrontier semantic weighting hook is active."""
        if not self.enable_vlm_hooks:
            return False
        return self.vlm_target_mode in ('open_frontier', 'all', 'both')

    def is_splat_search_enabled(self) -> bool:
        """Check if SplatSearch visual exemplar preemption hook is active."""
        if not self.enable_vlm_hooks:
            return False
        return self.vlm_target_mode in ('splat_search', 'all', 'both')

    # ==========================================================================
    # Dashboard Service & Topic Handlers
    # ==========================================================================

    def _set_explorer_srv(self, request, response):
        """Handle /set_explorer service call from mapping_dashboard."""
        self._set_enabled(request.data, source='service /set_explorer')
        response.success = True
        status_str = 'enabled' if self.enabled else 'paused'
        response.message = f'Explore adapter {status_str} successfully.'
        return response

    def _toggle_explorer_srv(self, request, response):
        """Handle /toggle_explorer trigger call from mapping_dashboard."""
        new_state = not self.enabled
        self._set_enabled(new_state, source='service /toggle_explorer')
        response.success = True
        status_str = 'enabled' if self.enabled else 'paused'
        response.message = f'Explore adapter toggled to {status_str}.'
        return response

    def _enable_topic_cb(self, msg: Bool):
        """Handle topic-based enable/disable on /explorer/enable."""
        if msg.data != self.enabled:
            self._set_enabled(msg.data, source='topic /explorer/enable')

    def _set_enabled(self, state: bool, source: str = 'internal'):
        """Update active exploration state and synchronize all interfaces."""
        if not state:
            self.active_goal = None

        if self.enabled == state:
            return

        self.enabled = state
        self.get_logger().info(f'Explorer state changed to {self.enabled} via {source}.')

        # Broadcast state to dashboard
        self.status_pub.publish(Bool(data=self.enabled))
        self.enable_pub.publish(Bool(data=self.enabled))

        if self.enabled:
            can_resume = (
                not self.is_preempted_by_vlm and
                not self.user_waypoints and
                not self.is_recovering_from_cliff and
                not self.cliff_active
            )
            if can_resume:
                self.explore_resume_pub.publish(Bool(data=True))
        else:
            self.explore_resume_pub.publish(Bool(data=False))
            self.cancel_nav2_goals()
            self._stop_robot()

    def _stop_robot(self):
        """Publish zero velocity to /cmd_vel."""
        self.cmd_vel_pub.publish(Twist())

    # ==========================================================================
    # User Goal & Path Management
    # ==========================================================================

    def _user_goal_cb(self, msg: PoseStamped):
        """Receive single user-clicked target from mapping_dashboard."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.get_logger().info(f'Received user target goal: ({x:.2f}, {y:.2f}).')
        self.user_waypoints.clear()
        self.is_single_user_goal = True
        self.is_preempted_by_vlm = False

        self.explore_resume_pub.publish(Bool(data=False))
        if not self.enabled:
            self._set_enabled(True, source='user goal')

        self.cancel_nav2_goals()
        self._dispatch_nav2_goal(x, y)

    def _user_path_cb(self, msg: Path):
        """Receive user-drawn exploration path containing ordered waypoints."""
        self.user_waypoints.clear()
        self.is_single_user_goal = False
        self.is_preempted_by_vlm = False

        for pose in msg.poses:
            self.user_waypoints.append((pose.pose.position.x, pose.pose.position.y))

        if self.user_waypoints:
            self.get_logger().info(
                f'Received user path with {len(self.user_waypoints)} waypoints.'
            )
            self.explore_resume_pub.publish(Bool(data=False))
            if not self.enabled:
                self._set_enabled(True, source='user path')

            next_pt = self.user_waypoints.popleft()
            self.cancel_nav2_goals()
            self._dispatch_nav2_goal(next_pt[0], next_pt[1])

    def _clear_path_cb(self, msg: Bool):
        """Clear user waypoints and resume autonomous frontier exploration."""
        if msg.data:
            self.get_logger().info('User path cleared. Resuming frontier exploration.')
            self.user_waypoints.clear()
            self.is_single_user_goal = False
            self.is_preempted_by_vlm = False
            self.active_goal = None
            self.cancel_nav2_goals()
            if self.enabled and not self.is_recovering_from_cliff and not self.cliff_active:
                self.explore_resume_pub.publish(Bool(data=True))

    # ==========================================================================
    # CLIFF DETECTION TOPIC: /cliff_alert (std_msgs/msg/Bool)
    # ==========================================================================
    # Easy Retrieval Hook: Subscribes to std_msgs/msg/Bool on '/cliff_alert'.
    # Broadcast by khepera4_mapper whenever downward ground IR sensors detect an edge.
    # Preempts explore_lite and Nav2 immediately, executes a two-phase recovery
    # (reverse backup -> turn pivot away), and holds robot safe until clear.
    # ==========================================================================

    def _cliff_alert_cb(self, msg: Bool):
        """
        Handle emergency failsafe for downward cliff drop-off hazards.

        Preempts Nav2 and executes emergency reverse backup when triggered.
        """
        was_cliff = self.cliff_active
        self.cliff_active = bool(msg.data)

        if self.cliff_active:
            if self.active_goal is not None:
                self.cliff_blacklisted_goals.append(
                    (self.active_goal[0], self.active_goal[1], self._get_current_time())
                )
            self.active_goal = None  # Clear active goal immediately upon cliff hazard

        if self.cliff_active and not was_cliff:
            self.get_logger().warn(
                '[SAFETY] Cliff alert triggered! Preempting Nav2 and executing backup.'
            )
            self.explore_resume_pub.publish(Bool(data=False))
            self.cancel_nav2_goals()

            self.is_recovering_from_cliff = True
            self.cliff_recovery_phase = 'BACKUP'
            self.cliff_recovery_start_time = self._get_current_time()

            backup_cmd = Twist()
            backup_cmd.linear.x = self.cliff_backup_speed
            backup_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(backup_cmd)

    # ==========================================================================
    # OPEN HOOK 1: OpenFrontier Zero-Shot Target Seeking
    # ==========================================================================

    def _vlm_frontier_weights_cb(self, msg: Float32MultiArray):
        """
        Ingest zero-shot semantic relevance probabilities for candidate frontiers.

        OPEN HOOK 1: OpenFrontier Integration
        Conditioned on user natural language target prompts (e.g. 'find the extinguisher')
        and vision-language model Set-of-Marks evaluations.
        """
        if not self.is_open_frontier_enabled():
            return

        data = list(msg.data)
        if not data:
            return

        if len(data) % 2 == 0:
            for i in range(0, len(data), 2):
                f_id = int(data[i])
                score = float(data[i + 1])
                self.vlm_frontier_weights[f_id] = score
        else:
            for idx, score in enumerate(data):
                self.vlm_frontier_weights[idx] = float(score)

        count = len(self.vlm_frontier_weights)
        self.get_logger().info(
            f'[OpenFrontier Hook] Ingested semantic weights for {count} frontiers.'
        )

        debug_msg = String()
        debug_msg.data = f'OpenFrontier active. Frontier weights: {self.vlm_frontier_weights}'
        self.vlm_debug_pub.publish(debug_msg)

        # Evaluate candidate frontiers if already populated
        self._evaluate_and_dispatch_openfrontier()

    def compute_blended_frontier_utility(
        self,
        frontier_id: int,
        info_gain: float,
        travel_cost: float,
        semantic_score: Optional[float] = None,
        alpha: float = 1.5,
        beta: Optional[float] = None,
        gamma: float = 3.0,
    ) -> float:
        """
        Compute blended utility combining geometric info gain and VLM semantic prior.

        OPEN HOOK 1: Frontier Scoring Arbitration:
          Utility(F_i) = alpha * InfoGain(F_i) + beta * P(target | F_i) - gamma * Cost(F_i)
        """
        if beta is None:
            beta = self.vlm_semantic_weight

        if semantic_score is None:
            semantic_score = self.vlm_frontier_weights.get(frontier_id, 0.0)

        utility = (alpha * info_gain) + (beta * semantic_score) - (gamma * travel_cost)
        return float(utility)

    def _evaluate_and_dispatch_openfrontier(self):
        """Evaluate candidate frontiers with blended utility and command Nav2."""
        if not self.enabled or self.user_waypoints or self.is_preempted_by_vlm:
            return
        if not self.candidate_frontiers or not self.vlm_frontier_weights:
            return
        if not self.is_open_frontier_enabled():
            return

        best_id = None
        best_score = -1e9
        for f_id, (fx, fy, f_size) in self.candidate_frontiers.items():
            cost = math.hypot(fx, fy)
            score = self.compute_blended_frontier_utility(f_id, f_size, cost)
            if score > best_score:
                best_score = score
                best_id = f_id

        if best_id is not None:
            bx, by, _ = self.candidate_frontiers[best_id]
            is_new = (
                self.active_goal is None or
                math.hypot(self.active_goal[0] - bx, self.active_goal[1] - by) > 0.05
            )
            if is_new:
                self.get_logger().info(
                    f'[OpenFrontier] Selected best semantic frontier {best_id} '
                    f'at ({bx:.2f}, {by:.2f}) with score {best_score:.3f}. '
                    f'Dispatching Nav2 goal.'
                )
                self.explore_resume_pub.publish(Bool(data=False))
                self.cancel_nav2_goals()
                self._dispatch_nav2_goal(bx, by)

    # ==========================================================================
    # OPEN HOOK 2: SplatSearch Single-Shot Exemplar Navigation
    # ==========================================================================

    def _vlm_target_confidence_cb(self, msg: Float32):
        """Update latest target detection confidence score."""
        self.latest_target_confidence = float(msg.data)

    def _vlm_goal_override_cb(self, msg: PoseStamped):
        """
        Execute single-shot visual exemplar preemption override.

        OPEN HOOK 2: SplatSearch Integration
        Preempts explore_lite and commands Nav2 to navigate directly to candidate target.
        """
        if not self.is_splat_search_enabled():
            return

        # Check confidence threshold if confidence is being reported
        if 0.0 < self.latest_target_confidence < self.exemplar_override_threshold:
            self.get_logger().warn(
                f'[SplatSearch Hook] Target confidence {self.latest_target_confidence:.3f} '
                f'below threshold {self.exemplar_override_threshold:.3f}. '
                f'Preemption override rejected.'
            )
            return

        tx = msg.pose.position.x
        ty = msg.pose.position.y
        self.vlm_override_pose = (tx, ty)

        self.get_logger().info(
            f'[SplatSearch Hook] Target exemplar detected at ({tx:.2f}, {ty:.2f}). '
            f'Preempting frontier exploration to pursue target.'
        )

        self.is_preempted_by_vlm = True
        self.explore_resume_pub.publish(Bool(data=False))
        self.cancel_nav2_goals()
        self._dispatch_nav2_goal(tx, ty)

        debug_msg = String()
        debug_msg.data = (
            f'SplatSearch preemption active. Pursuing target at ({tx:.2f}, {ty:.2f}).'
        )
        self.vlm_debug_pub.publish(debug_msg)

    # ==========================================================================
    # Frontier Cache & Visualization Marker Tracking
    # ==========================================================================

    def _is_goal_near_cliff_blacklist(self, gx: float, gy: float) -> bool:
        """Check if target coordinates lie within proximity of a blacklisted cliff incident."""
        for bx, by, _ in self.cliff_blacklisted_goals:
            dist = math.hypot(gx - bx, gy - by)
            if dist <= self.cliff_blacklist_radius:
                return True
        return False

    def _frontiers_marker_cb(self, msg: MarkerArray):
        """
        Track candidate frontiers published by explore_lite on /explore/frontiers.

        Maintains frontier cache for OpenFrontier semantic scoring and dashboard goal mirroring.
        """
        self.candidate_frontiers.clear()
        for marker in msg.markers:
            if marker.points:
                cx = sum(p.x for p in marker.points) / len(marker.points)
                cy = sum(p.y for p in marker.points) / len(marker.points)
                size = float(len(marker.points))
                if not self._is_goal_near_cliff_blacklist(cx, cy):
                    self.candidate_frontiers[marker.id] = (cx, cy, size)
            elif marker.type == Marker.SPHERE:
                cx = marker.pose.position.x
                cy = marker.pose.position.y
                size = max(1.0, marker.scale.x * 10.0)
                if not self._is_goal_near_cliff_blacklist(cx, cy):
                    self.candidate_frontiers[marker.id] = (cx, cy, size)

        if not self.enabled or self.user_waypoints or self.is_preempted_by_vlm:
            return

        if self.is_open_frontier_enabled() and self.vlm_frontier_weights:
            self._evaluate_and_dispatch_openfrontier()
        elif self.candidate_frontiers and not self._is_nav2_goal_active:
            # Standard autonomous exploration mode: mirror explore_lite's active frontier goal
            first_id = next(iter(self.candidate_frontiers))
            fx, fy, _ = self.candidate_frontiers[first_id]
            self.active_goal = (fx, fy)
            self._publish_active_goal()

    def _explore_status_cb(self, msg):
        """Handle exploration lifecycle updates from explore_lite."""
        if hasattr(msg, 'status') and msg.status == 2:  # EXPLORATION_COMPLETE
            self.get_logger().info('[explore_lite] Frontier exploration complete.')
            if not self.user_waypoints and not self.is_preempted_by_vlm:
                self.active_goal = None

    # ==========================================================================
    # Nav2 Action Dispatching & Monitoring
    # ==========================================================================

    def _dispatch_nav2_goal(self, x: float, y: float, frame_id: str = 'map'):
        """Send a NavigateToPose goal to Nav2 action server."""
        self.active_goal = (float(x), float(y))
        self._publish_active_goal()

        if not self._nav2_client.server_is_ready():
            self.get_logger().warn(
                f'Nav2 action server not ready. Goal ({x:.2f}, {y:.2f}) registered in adapter.'
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self._is_nav2_goal_active = True
        send_goal_future = self._nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav2_feedback_cb
        )
        send_goal_future.add_done_callback(self._nav2_goal_response_cb)

    def _nav2_goal_response_cb(self, future):
        """Handle Nav2 goal response callback."""
        goal_handle: Optional[ClientGoalHandle] = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected exploration goal.')
            self._is_nav2_goal_active = False
            self.active_goal = None
            if self.user_waypoints:
                next_pt = self.user_waypoints.popleft()
                self._dispatch_nav2_goal(next_pt[0], next_pt[1])
            elif self.enabled and not self.is_recovering_from_cliff and not self.cliff_active:
                self.explore_resume_pub.publish(Bool(data=True))
            return

        self._current_goal_handle = goal_handle
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._nav2_result_cb)

    def _nav2_feedback_cb(self, feedback_msg):
        """Monitor live Nav2 navigation feedback."""
        pass

    def _nav2_result_cb(self, future):
        """Handle goal completion, abortion, or cancellation from Nav2."""
        self._is_nav2_goal_active = False
        self._current_goal_handle = None

        if self.user_waypoints:
            next_pt = self.user_waypoints.popleft()
            self._dispatch_nav2_goal(next_pt[0], next_pt[1])
        else:
            if self.is_single_user_goal:
                self.get_logger().info('User target goal reached.')
                self.is_single_user_goal = False
            if self.is_preempted_by_vlm:
                self.get_logger().info('[SplatSearch] Target exemplar goal reached.')
                self.is_preempted_by_vlm = False
            self.active_goal = None

            # Resume explore_lite autonomous frontier exploration
            if self.enabled and not self.is_recovering_from_cliff and not self.cliff_active:
                self.explore_resume_pub.publish(Bool(data=True))

    def cancel_nav2_goals(self):
        """Cancel currently active Nav2 navigation goals."""
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f'Failed to cancel Nav2 goal: {e}')
            self._current_goal_handle = None
        self._is_nav2_goal_active = False

    def _publish_active_goal(self):
        """Publish active goal coordinates on /explorer/active_goal."""
        if self.active_goal is None:
            return

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = float(self.active_goal[0])
        msg.point.y = float(self.active_goal[1])
        msg.point.z = 0.0
        self.active_goal_pub.publish(msg)

    # ==========================================================================
    # Periodic Coordination Loop
    # ==========================================================================

    def _finish_cliff_recovery(self):
        """Complete the two-phase cliff recovery maneuver."""
        self._stop_robot()
        self.is_recovering_from_cliff = False
        self.cliff_recovery_phase = 'NONE'
        self.get_logger().info('[SAFETY] Cliff backup recovery complete.')
        # Only resume exploration if cliff hazard is cleared!
        if not self.cliff_active and self.enabled and not self.is_preempted_by_vlm:
            self.explore_resume_pub.publish(Bool(data=True))
        elif self.cliff_active:
            self.get_logger().warn(
                '[SAFETY] Cliff hazard still sensed. Remaining in safety hold.'
            )

    def _coordination_loop(self):
        """Execute periodic safety supervision and state management."""
        now = self._get_current_time()

        if self.is_recovering_from_cliff:
            if self.cliff_recovery_phase == 'BACKUP':
                elapsed = now - self.cliff_recovery_start_time
                if elapsed < self.cliff_backup_duration:
                    backup_cmd = Twist()
                    backup_cmd.linear.x = self.cliff_backup_speed
                    backup_cmd.angular.z = 0.0
                    self.cmd_vel_pub.publish(backup_cmd)
                else:
                    # Transition to PIVOT phase: turn away from cliff edge
                    self.cliff_recovery_phase = 'PIVOT'
                    self.cliff_pivot_start_time = now
                    pivot_cmd = Twist()
                    pivot_cmd.linear.x = 0.0
                    pivot_cmd.angular.z = self.cliff_turn_speed
                    self.cmd_vel_pub.publish(pivot_cmd)
            elif self.cliff_recovery_phase == 'PIVOT':
                elapsed = now - self.cliff_pivot_start_time
                if elapsed < self.cliff_turn_duration:
                    pivot_cmd = Twist()
                    pivot_cmd.linear.x = 0.0
                    pivot_cmd.angular.z = self.cliff_turn_speed
                    self.cmd_vel_pub.publish(pivot_cmd)
                else:
                    self._finish_cliff_recovery()

        if self.active_goal is not None:
            self._publish_active_goal()


def main(args=None):
    """Execute main entry point for explore_adapter node."""
    import rclpy
    rclpy.init(args=args)
    node = ExploreAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
