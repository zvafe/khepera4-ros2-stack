#!/usr/bin/env python3
"""
Reactive Explorer Node for Khepera IV Mobile Robot.

Implements autonomous reactive arena exploration:
- Multi-sensor proximity aggregation (US + IR + Ground IR + 360 scan)
- Braitenberg / potential field obstacle repulsion steering
- Dynamic speed modulation & stuck/corner escape behaviors
- Downward cliff hazard safety: immediate halt and 180° escape
- Corner/wall stall entrapment escape: 180° reverse escape when stuck > 1.2s
- Exploration control services and topic interface
"""

import collections
import math
import time
from typing import List, Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from khepera4_control.common import euler_from_quaternion, normalize_angle
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray

from khepera4_control.openfrontier_explorer import FrontierCandidate, OpenFrontierExplorer


class ReactiveExplorer(Node):
    """Autonomous reactive exploration and obstacle/cliff avoidance node."""

    # Navigation States
    STATE_CRUISE = 0
    STATE_AVOID = 1
    STATE_REVERSE_TURN = 2
    STATE_CLIFF_RECOVERY = 3
    STATE_PAUSED = 4
    STATE_INSPECT = 5
    STATE_INITIAL_SPIN = 6

    def _get_current_time(self) -> float:
        """Get current node clock time in fractional seconds (respects use_sim_time / time.time)."""
        clock_t = self.get_clock().now().nanoseconds * 1e-9
        if clock_t <= 0.0:
            return time.time()
        return clock_t

    def __init__(self):
        """Initialize the reactive explorer node and ROS 2 interfaces."""
        super().__init__('reactive_explorer')

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter('enabled', True)
        self.declare_parameter('linear_speed', 0.12)
        self.declare_parameter('reverse_speed', 0.05)
        self.declare_parameter('turn_speed', 0.90)
        self.declare_parameter('max_turn_speed', 1.50)
        self.declare_parameter('obstacle_avoid_dist', 0.40)
        self.declare_parameter('obstacle_stop_dist', 0.08)
        self.declare_parameter('cliff_backup_duration', 1.2)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('stall_timeout', 0.35)
        self.declare_parameter('linear_speed_limit', 0.12)
        self.declare_parameter('angular_speed_limit', 0.90)
        self.declare_parameter('target_lost_timeout', 4.0)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.enabled = bool(self.get_parameter('enabled').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.reverse_speed = float(self.get_parameter('reverse_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.max_turn_speed = float(
            self.get_parameter('max_turn_speed').value
        )
        self.target_lost_timeout = float(
            self.get_parameter('target_lost_timeout').value
        )
        self.avoid_dist = float(
            self.get_parameter('obstacle_avoid_dist').value
        )
        self.stop_dist = float(self.get_parameter('obstacle_stop_dist').value)
        self.cliff_backup_duration = float(
            self.get_parameter('cliff_backup_duration').value
        )
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.stall_timeout = float(self.get_parameter('stall_timeout').value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ----------------------------------------------------------------------
        # State Variables
        # ----------------------------------------------------------------------
        self.state = self.STATE_CRUISE if self.enabled else self.STATE_PAUSED
        self.cliff_alert = False
        init_time = self._get_current_time()
        self.state_start_time = init_time
        self.start_wall_clock = init_time
        self.escape_turn_dir = 1.0  # +1.0 = left, -1.0 = right

        # Motion tracking & corner stall detection
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.has_odom = False
        self.has_imu = False
        self.pos_history = collections.deque(maxlen=40)
        self.accel_history = collections.deque(maxlen=30)
        self.last_commanded_v = 0.0
        self.stall_start_time: Optional[float] = None
        self.stall_ticks = 0
        self.reverse_turn_initial_yaw = 0.0

        # IMU jerk & impact tracking
        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_imu_time: Optional[float] = None
        self.imu_jerk = 0.0
        self.last_impact_time = -100.0
        self.imu_impact_detected = False

        # Courageous autonomous & user-drawn target exploration
        self.user_waypoints = collections.deque()
        self.active_goal: Optional[Tuple[float, float]] = None
        self.goal_start_time: float = init_time
        self.start_pose: Optional[Tuple[float, float]] = None
        self.visited_goals: List[Tuple[float, float]] = []
        self.inspect_duration: float = 1.0  # seconds to scan surroundings at target
        self.inspect_angular_speed: float = 0.0  # Zero spinning (360-degree sensor vision)
        self.is_single_user_goal: bool = False
        self.front_steer_dir: float = 1.0
        self.front_steer_lock_time: float = 0.0
        self.is_first_target: bool = True
        self.initial_spin_done: bool = False
        self.spin_cumulative_yaw: float = 0.0
        self.initial_spin_speed: float = 1.80

        # Sensor readings caches
        self.latest_us: List[Optional[LaserScan]] = [None] * 5
        self.latest_ir: List[Optional[LaserScan]] = [None] * 8
        self.latest_ground_ir: List[Optional[LaserScan]] = [None] * 4
        self.latest_scan: Optional[LaserScan] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self._last_frontier_calc_time = 0.0
        self._cached_frontier_heading: Optional[float] = None

        # OpenFrontier map-based active exploration engine
        self.openfrontier = OpenFrontierExplorer(
            tau_merge_m=0.70,
            r_clear_m=0.35,
            r_near_m=0.30,
            r_goal_m=0.80,
            min_cluster_cells=4,
            robot_safety_radius_m=0.14,
            stall_threshold_cycles=15,
            epsilon_stall_m=0.06,
        )
        self.frontier_waypoints: collections.deque = collections.deque()
        self.last_frontier_replan_time = 0.0

        # ----------------------------------------------------------------------
        # Publishers & Subscribers
        # ----------------------------------------------------------------------
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        raw_cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.explorer_cmd_pub = self.create_publisher(Twist, '/explorer/cmd_vel', 10)

        class _CmdPublisherProxy:
            """Proxy forwarding Twist messages to primary topic and /explorer/cmd_vel."""

            def __init__(self, primary, secondary):
                self._primary = primary
                self._secondary = secondary

            def publish(self, msg):
                self._primary.publish(msg)
                if self._secondary is not None:
                    self._secondary.publish(msg)

            def __getattr__(self, item):
                return getattr(self._primary, item)

        self.cmd_pub = _CmdPublisherProxy(
            raw_cmd_pub,
            self.explorer_cmd_pub if cmd_topic != '/explorer/cmd_vel' else None,
        )
        self.active_goal_pub = self.create_publisher(
            PointStamped, '/explorer/active_goal', 10
        )
        self.frontier_markers_pub = self.create_publisher(
            MarkerArray, '/openfrontier/markers', 10
        )
        self.planned_path_pub = self.create_publisher(
            Path, '/explorer/planned_path', 10
        )

        # Latched explorer status publisher (so dashboard immediately gets state on connect)
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.explorer_status_pub = self.create_publisher(Bool, '/explorer/status', qos_latched)
        self.explorer_enable_pub = self.create_publisher(Bool, '/explorer/enable', qos_latched)
        self.explorer_status_pub.publish(Bool(data=self.enabled))
        self.explorer_enable_pub.publish(Bool(data=self.enabled))
        self.thinking_pub = self.create_publisher(String, '/robot/thinking_log', 10)
        self.speed_sub = self.create_subscription(
            Twist, '/explorer/set_speed', self.speed_callback, 10
        )
        self.speed_limits_sub = self.create_subscription(
            Twist, '/explorer/set_speed_limits', self.speed_callback, 10
        )
        self.robot_speed_limits_sub = self.create_subscription(
            Twist, '/robot/speed_limits', self.speed_callback, 10
        )
        self._last_thought_text = ""
        self._last_thought_time = 0.0

        self.cliff_sub = self.create_subscription(
            Bool, '/cliff_alert', self.cliff_callback, 10
        )
        self.is_turned_over = False
        self.turned_over_sub = self.create_subscription(
            Bool, '/khepera4/turned_over', self.turned_over_callback, 10
        )
        self.enable_sub = self.create_subscription(
            Bool, '/explorer/enable', self.enable_topic_callback, qos_latched
        )
        self.user_goal_sub = self.create_subscription(
            PoseStamped, '/explorer/user_goal', self.user_goal_callback, 10
        )
        self.user_path_sub = self.create_subscription(
            Path, '/explorer/user_path', self.user_path_callback, 10
        )
        self.clear_path_sub = self.create_subscription(
            Bool, '/explorer/clear_path', self.clear_path_callback, 10
        )
        self.pursuit_active = False
        self.target_in_sight = False
        self.target_lost_time = None
        self.target_stopped = False
        self.stop_sub = self.create_subscription(
            Bool, '/target_found_and_stopped', self.target_stopped_callback, 10
        )
        self.target_status_sub = self.create_subscription(
            Bool, '/perception/target_status', self.target_status_callback, 10
        )
        self.pursuit_sub = self.create_subscription(
            Bool, '/pursuit/enable', self.pursuit_enable_callback, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/khepera4/scan', self.scan_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 5
        )
        self.last_fused_odom_time = 0.0
        self.fused_odom_sub = self.create_subscription(
            Odometry, '/khepera4/odom', self.fused_odom_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # 5 Ultrasonic scans
        self.us_subs = []
        for i in range(5):
            topic = f'/khepera4/us_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.us_callback(idx, msg),
                10
            )
            self.us_subs.append(sub)

        # 8 IR proximity scans
        self.ir_subs = []
        for i in range(8):
            topic = f'/khepera4/ir_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.ir_callback(idx, msg),
                10
            )
            self.ir_subs.append(sub)

        # 4 Ground IR scans (downward cliff and ground obstacle detection)
        self.ground_ir_subs = []
        for i in range(4):
            topic = f'/khepera4/ground_ir_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.ground_ir_callback(idx, msg),
                10
            )
            self.ground_ir_subs.append(sub)

        # Services
        self.toggle_srv = self.create_service(
            Trigger, '/toggle_explorer', self.toggle_explorer_callback
        )
        self.set_srv = self.create_service(
            SetBool, '/set_explorer', self.set_explorer_callback
        )

        # Main Control Timer (20 Hz)
        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            f'ReactiveExplorer online (Enabled={self.enabled}, '
            f'Speed={self.linear_speed}m/s).'
        )

    # ==========================================================================
    # Subscriber Callbacks
    # ==========================================================================

    def speed_callback(self, msg: Twist) -> None:
        """Update linear and angular speeds dynamically from teleoperation sliders or topic."""
        if msg.linear.x > 0.001:
            self.linear_speed = float(msg.linear.x)
            self.linear_speed_limit = float(msg.linear.x)
        if msg.angular.z > 0.001:
            self.turn_speed = float(msg.angular.z)
            self.max_turn_speed = max(self.turn_speed * 1.5, 1.5)
            self.angular_speed_limit = float(msg.angular.z)

    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        """Handle dynamic parameter reconfigurations for exploration speed limits."""
        for param in params:
            if param.name in ('linear_speed', 'linear_speed_limit'):
                val = float(param.value)
                if val > 0.001:
                    self.linear_speed = val
                    self.linear_speed_limit = val
            elif param.name in ('turn_speed', 'angular_speed_limit'):
                val = float(param.value)
                if val > 0.001:
                    self.turn_speed = val
                    self.max_turn_speed = max(self.turn_speed * 1.5, 1.5)
                    self.angular_speed_limit = val
            elif param.name == 'max_turn_speed':
                val = float(param.value)
                if val > 0.001:
                    self.max_turn_speed = val
            elif param.name == 'target_lost_timeout':
                self.target_lost_timeout = float(param.value)
        thought = (
            f"⚡ Exploration Speed Limits Reconfigured: Cruise v = {self.linear_speed:.2f} m/s, "
            f"Turn Rate ω = {self.turn_speed:.2f} rad/s"
        )
        self.publish_thought(thought)
        return SetParametersResult(successful=True)

    def target_stopped_callback(self, msg: Bool) -> None:
        """Halt exploration permanently when target is reached and stopped."""
        if msg.data:
            self.target_stopped = True
            self.get_logger().info(
                'Target reached and terminal stop signaled! Halting explorer.'
            )
            self.publish_thought(
                '🛑 Target reached and terminal stop signaled. Exploration permanently halted.'
            )
            self._set_enabled(False, 'topic /target_found_and_stopped')
            self._stop_robot()
        else:
            self.target_stopped = False

    def pursuit_enable_callback(self, msg: Bool) -> None:
        """Handle pursuit active toggles to enable or suspend active exploration."""
        was_active = self.pursuit_active
        self.pursuit_active = bool(msg.data)
        if self.pursuit_active and not was_active:
            if not self.target_stopped:
                self.target_in_sight = False
                self.get_logger().info(
                    'Visual pursuit enabled. Starting 360° survey and active exploration search.'
                )
                self.publish_thought(
                    '🚀 Pursuit enabled: Executing 360° survey to map frontiers and scan for target.',
                    force=True
                )
                self._set_enabled(True, 'topic /pursuit/enable')
                self.state = self.STATE_INITIAL_SPIN
                self.state_start_time = self._get_current_time()
                self.initial_spin_done = False
                self.spin_cumulative_yaw = 0.0
                self.spin_last_yaw = self.current_yaw
                self.active_goal = None
                self.frontier_waypoints.clear()
        elif not self.pursuit_active:
            self.target_in_sight = False
            if was_active:
                self.get_logger().info(
                    'Visual pursuit disengaged. Halting autonomous exploration.'
                )
                self.publish_thought('🛑 Pursuit disengaged. Halting exploration.')
            self._set_enabled(False, 'topic /pursuit/enable')
            self._stop_robot()

    def target_status_callback(self, msg: Bool) -> None:
        """Handle target detection status to yield or resume autonomous exploration."""
        if not self.pursuit_active:
            self.target_in_sight = False
            self.target_lost_time = None
            return
        is_target = bool(msg.data)
        if is_target:
            self.target_lost_time = None
            if not self.target_in_sight:
                self.target_in_sight = True
                self.get_logger().info(
                    'Target acquired in sight! Halting explorer and yielding to IBVS pursuit.'
                )
                self.publish_thought('🎯 Target in sight! Yielding control to visual pursuit.')
                self._stop_robot()
        else:
            now = self._get_current_time()
            if self.target_lost_time is None:
                self.target_lost_time = now
            if (
                self.target_lost_timeout <= 0.0
                or (now - self.target_lost_time) >= self.target_lost_timeout
            ):
                if self.target_in_sight:
                    self.target_in_sight = False
                    self.get_logger().info(
                        'Target lost from visual sight. Resuming active exploration search.'
                    )
                    self.publish_thought(
                        '🔍 Target out of sight. Resuming active exploration search.'
                    )

    def publish_thought(self, text: str, force: bool = False) -> None:
        """Publish cognitive reasoning log for dashboard telemetry."""
        now = self._get_current_time()
        if not force and text == self._last_thought_text and (now - self._last_thought_time) < 2.0:
            return
        self._last_thought_text = text
        self._last_thought_time = now
        if hasattr(self, 'thinking_pub') and self.thinking_pub:
            msg = String()
            msg.data = text
            self.thinking_pub.publish(msg)

    def fused_odom_callback(self, msg: Odometry):
        """Track filtered, slip-projected odometry on /khepera4/odom (highest priority)."""
        self.last_fused_odom_time = self._get_current_time()
        self._process_odom(msg)

    def odom_callback(self, msg: Odometry):
        """Track raw odometry from /odom, falling back when /khepera4/odom is absent."""
        if (self._get_current_time() - self.last_fused_odom_time) <= 1.0:
            return
        self._process_odom(msg)

    def _process_odom(self, msg: Odometry):
        """Track robot position and displacement history for stall check."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        if self.start_pose is None:
            self.start_pose = (self.current_x, self.current_y)
        if not self.has_imu:
            q = msg.pose.pose.orientation
            _, _, self.current_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.has_odom = True
        self.pos_history.append((self._get_current_time(), self.current_x, self.current_y))

    def imu_callback(self, msg: Imu):
        """Track IMU acceleration variance, heading, jerk and impact for evasion."""
        self.has_imu = True
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        self.accel_history.append(math.hypot(ax, ay))

        # Time extraction from ROS node clock or message stamp
        now_t = self._get_current_time()
        if now_t <= 0.0 and (msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0):
            now_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Jerk computation & collision impact shock detection
        jx = 0.0
        jy = 0.0
        if self.last_imu_time is not None:
            dt = now_t - self.last_imu_time
            if 0.005 <= dt <= 0.2:
                jx = (ax - self.last_ax) / dt
                jy = (ay - self.last_ay) / dt
                self.imu_jerk = math.hypot(jx, jy)
            else:
                self.imu_jerk = 0.0
        else:
            self.imu_jerk = 0.0

        self.last_ax = ax
        self.last_ay = ay
        self.last_imu_time = now_t

        if self.last_commanded_v > 0.03:
            # Impact collision when driving forward produces severe negative deceleration (ax < -2.2 m/s^2)
            if ax < -2.2:
                self.last_impact_time = now_t
                self.imu_impact_detected = True

        az = msg.linear_acceleration.z
        a_mag = math.hypot(math.hypot(ax, ay), az)
        z_gravity_ratio = (az / a_mag) if a_mag > 1e-3 else 0.0

        q = msg.orientation
        norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if norm_sq > 0.5:
            roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            if not (math.isnan(yaw) or math.isinf(yaw)):
                self.current_yaw = yaw
            current_slope_deg = math.degrees(math.hypot(pitch, roll))
        else:
            current_slope_deg = 0.0

        # Physical rollover check:
        # For Khepera IV, +Z points directly upward in normal upright orientation (az ~ +9.81 m/s^2).
        # When turned over / flipped upside down, az becomes negative (~ -9.81 m/s^2) or near 0 on its side.
        if a_mag > 1.0:
            is_flipped_accel = (az < -1.0) or (az < 1.0 and current_slope_deg > 45.0) or (z_gravity_ratio < 0.17 and current_slope_deg > 45.0)
        else:
            is_flipped_accel = False
        is_flipped = is_flipped_accel or (current_slope_deg > 70.0)

        if is_flipped:
            if not self.is_turned_over:
                self.get_logger().error(
                    f'ROBOT TURNED OVER / FLIPPED DETECTED! az={az:.2f} m/s^2, slope={current_slope_deg:.1f}°. '
                    'Halting autonomous explorer motor commands.'
                )
            self.is_turned_over = True
            self._stop_robot()
            # NOTE FOR FUTURE SWARM COORDINATION:
            # From this rollover / turned-over state variable (self.is_turned_over), a distress signal
            # (e.g., publishing an SOS alert on '/swarm/distress' or broadcasting an emergency beacon)
            # can be created to notify neighboring robots in the swarm for search, rescue, or re-routing.
            # (Do not implement the distress signal transmission logic yet per project requirements).
        elif (az >= 2.0 or a_mag <= 1.0) and (current_slope_deg <= 50.0):
            # Restored upright
            self.is_turned_over = False

    def turned_over_callback(self, msg: Bool):
        """Handle turned-over alert from mapper node."""
        self.is_turned_over = msg.data
        if self.is_turned_over:
            self._stop_robot()

    def ground_ir_callback(self, index: int, msg: LaserScan):
        """Cache ground IR scan."""
        self.latest_ground_ir[index] = msg

    def cliff_callback(self, msg: Bool):
        """Handle cliff drop-off alerts from mapping engine."""
        was_alert = self.cliff_alert
        self.cliff_alert = msg.data

        if self.cliff_alert and not was_alert and self.enabled:
            self.get_logger().warn(
                'CLIFF HAZARD! Initiating emergency backup and rotation.'
            )
            self.state = self.STATE_CLIFF_RECOVERY
            now = self._get_current_time()
            self.state_start_time = now
            self.escape_turn_dir = 1.0 if (int(now * 10) % 2 == 0) else -1.0

    def enable_topic_callback(self, msg: Bool):
        """Enable or disable explorer via topic."""
        self._set_enabled(msg.data, 'topic /explorer/enable')

    def scan_callback(self, msg: LaserScan):
        """Cache virtual 360 scan."""
        self.latest_scan = msg

    def map_callback(self, msg: OccupancyGrid):
        """Cache incoming occupancy grid map for frontier exploration."""
        self.latest_map = msg

    def us_callback(self, index: int, msg: LaserScan):
        """Cache ultrasonic scan."""
        self.latest_us[index] = msg

    def ir_callback(self, index: int, msg: LaserScan):
        """Cache infrared scan."""
        self.latest_ir[index] = msg

    # ==========================================================================
    # User-Guided Exploration & Target Selection
    # ==========================================================================

    def user_goal_callback(self, msg: PoseStamped):
        """Receive user-selected exploration target from dashboard or RViz."""
        self.set_user_goal(msg.pose.position.x, msg.pose.position.y)

    def user_path_callback(self, msg: Path):
        """Receive user-drawn exploration path containing ordered waypoints."""
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.set_user_path(points)

    def clear_path_callback(self, msg: Bool):
        """Clear user waypoints and reset active target to autonomous exploration."""
        if msg.data:
            self.clear_user_path()

    def set_user_goal(self, x: float, y: float):
        """Set a single target location to be explored."""
        self.user_waypoints.clear()
        self.frontier_waypoints.clear()
        self.is_single_user_goal = True
        self.initial_spin_done = True
        self.is_first_target = False
        self.active_goal = (float(x), float(y))
        self.goal_start_time = self._get_current_time()
        self.get_logger().info(f'User target set: ({x:.2f}, {y:.2f}).')
        self._publish_active_goal()
        thought = (
            f"I pick user-designated target ({x:.2f}, {y:.2f}) "
            f"because navigating here offers a direct line of sight to spot the object."
        )
        self.publish_thought(thought, force=True)
        if not self.enabled:
            self._set_enabled(True, 'user target set')
        if self.state in (self.STATE_INSPECT, self.STATE_CRUISE, self.STATE_PAUSED):
            self.state = self.STATE_CRUISE

    def set_user_path(self, points: List[Tuple[float, float]]):
        """Set ordered waypoints from user-drawn path."""
        self.user_waypoints.clear()
        self.frontier_waypoints.clear()
        self.is_single_user_goal = False
        self.initial_spin_done = True
        self.is_first_target = False
        for pt in points:
            self.user_waypoints.append((float(pt[0]), float(pt[1])))
        if self.user_waypoints:
            self.active_goal = self.user_waypoints.popleft()
            if self.has_odom:
                while self.user_waypoints and math.hypot(self.active_goal[0] - self.current_x, self.active_goal[1] - self.current_y) < 0.12:
                    self.active_goal = self.user_waypoints.popleft()
            self.goal_start_time = self._get_current_time()
            self.get_logger().info(
                f'User path received with {len(points)} waypoints. Active target: {self.active_goal}.'
            )
            self._publish_active_goal()
            thought = (
                f"I pick user trajectory toward ({self.active_goal[0]:.2f}, {self.active_goal[1]:.2f}) "
                f"to scan for visual verification of the target object along this path."
            )
            self.publish_thought(thought, force=True)
        if not self.enabled:
            self._set_enabled(True, 'user path set')
        if self.state in (self.STATE_INSPECT, self.STATE_CRUISE, self.STATE_PAUSED):
            self.state = self.STATE_CRUISE

    def clear_user_path(self):
        """Clear all user waypoints and active goal."""
        self.user_waypoints.clear()
        self.frontier_waypoints.clear()
        self.is_single_user_goal = False
        self.active_goal = None
        self.get_logger().info('User path cleared. Resuming autonomous exploration.')

    def _publish_active_goal(self):
        """Publish current active exploration target on /explorer/active_goal."""
        if self.active_goal is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = float(self.active_goal[0])
        msg.point.y = float(self.active_goal[1])
        msg.point.z = 0.0
        self.active_goal_pub.publish(msg)

    def _publish_frontier_markers(self) -> None:
        """Publish RViz visualization markers for active OpenFrontier candidates."""
        if not hasattr(self, 'frontier_markers_pub') or not self.frontier_markers_pub:
            return
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)

        for idx, (fid, f) in enumerate(self.openfrontier.active_frontiers.items()):
            # Sphere marker
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'openfrontiers'
            m.id = idx * 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(f.pos_x)
            m.pose.position.y = float(f.pos_y)
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = 0.20
            m.scale.y = 0.20
            m.scale.z = 0.20
            is_goal = (self.openfrontier.current_goal is not None and self.openfrontier.current_goal.id == fid)
            if is_goal:
                m.color.r = 1.0
                m.color.g = 0.85
                m.color.b = 0.0
                m.color.a = 0.95
            else:
                m.color.r = 0.0
                m.color.g = 0.8
                m.color.b = 1.0
                m.color.a = 0.75
            marker_array.markers.append(m)

            # Text label marker
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = stamp
            t.ns = 'openfrontiers'
            t.id = idx * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(f.pos_x)
            t.pose.position.y = float(f.pos_y)
            t.pose.position.z = 0.26
            t.pose.orientation.w = 1.0
            t.scale.z = 0.14
            t.text = f"{f.id} (u={f.total_utility:.1f})"
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            marker_array.markers.append(t)

        self.frontier_markers_pub.publish(marker_array)

    def _publish_planned_path(self, waypoints: List[Tuple[float, float]]) -> None:
        """Publish A* planned path on /explorer/planned_path."""
        if not hasattr(self, 'planned_path_pub') or not self.planned_path_pub:
            return
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for x, y in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.planned_path_pub.publish(path_msg)

    def _find_raycast_frontier_target(self) -> Optional[Tuple[float, float]]:
        """Fallback frontier raycaster from latest_map."""
        if self.latest_map is None:
            return None
        info = self.latest_map.info
        res = info.resolution
        w = info.width
        h = info.height
        ox = info.origin.position.x
        oy = info.origin.position.y
        if len(self.latest_map.data) != w * h:
            return None

        grid = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))
        best_candidate = None
        best_score = -1e9
        num_rays = 36

        for k in range(num_rays):
            heading = -math.pi + k * (2.0 * math.pi / num_rays)
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            frontier_hit = None
            last_free_pt = None

            for d in np.arange(0.15, 2.30, 0.08):
                px = self.current_x + d * cos_h
                py = self.current_y + d * sin_h
                if abs(px) > 2.15 or abs(py) > 2.15:
                    break
                col = int(math.floor((px - ox) / res))
                row = int(math.floor((py - oy) / res))
                if col < 0 or col >= w or row < 0 or row >= h:
                    break
                val = grid[row, col]
                if val >= 50:
                    break
                elif 0 <= val < 25:
                    last_free_pt = (px, py, d)
                elif val == -1 and last_free_pt is not None:
                    if d >= 0.25:
                        frontier_hit = (last_free_pt[0], last_free_pt[1], d)
                    break

            if frontier_hit is not None:
                fx, fy, fd = frontier_hit
                dist_to_visited = min(
                    [math.hypot(fx - vx, fy - vy) for vx, vy in self.visited_goals[-6:]],
                    default=2.0
                )
                f_col = int(math.floor((fx - ox) / res))
                f_row = int(math.floor((fy - oy) / res))
                r_min = max(0, f_row - 8)
                r_max = min(h, f_row + 9)
                c_min = max(0, f_col - 8)
                c_max = min(w, f_col + 9)
                sub_grid = grid[r_min:r_max, c_min:c_max]
                unknown_gain = np.count_nonzero(sub_grid == -1)

                score = (1.5 * fd) + (2.5 * dist_to_visited) + (0.05 * unknown_gain)
                if score > best_score:
                    best_score = score
                    best_candidate = (fx, fy)

        return best_candidate

    def _find_highest_unknown_sector(self) -> Optional[Tuple[float, float]]:
        """
        Identify the arena sector with the highest concentration of unexplored cells.

        Evaluates 13 spatial sectors across the arena, computes information gain,
        and targets the safe centroid of the least explored area.
        """
        if self.latest_map is None:
            return None
        info = self.latest_map.info
        res = info.resolution
        w, h = info.width, info.height
        ox, oy = info.origin.position.x, info.origin.position.y
        if len(self.latest_map.data) != w * h:
            return None

        grid = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))

        # Candidate coordinates prioritizing the four quadrants, then central corridors
        candidate_coords = [
            (1.35, -1.35),
            (1.35, 1.35),
            (-1.35, 1.35),
            (-1.35, -1.35),
            (0.00, -1.35),
            (1.35, 0.00),
            (0.00, 1.35),
            (-1.35, 0.00),
        ]

        best_pt = None
        best_score = 0.0
        sector_radius = 0.85  # meters around sector center

        for cx, cy in candidate_coords:
            c_col = int(math.floor((cx - ox) / res))
            c_row = int(math.floor((cy - oy) / res))
            if not (0 <= c_col < w and 0 <= c_row < h):
                continue

            dist_to_robot = math.hypot(cx - self.current_x, cy - self.current_y)
            # Penalty for any visited location in this session
            visited_penalty = 0.0
            for vx, vy in self.visited_goals:
                d_vis = math.hypot(cx - vx, cy - vy)
                if d_vis < 0.90:
                    visited_penalty += 2000.0 * (1.0 - d_vis / 0.90)

            r0 = max(0, int(math.floor((cy - sector_radius - oy) / res)))
            r1 = min(h, int(math.ceil((cy + sector_radius - oy) / res)))
            c0 = max(0, int(math.floor((cx - sector_radius - ox) / res)))
            c1 = min(w, int(math.ceil((cx + sector_radius - ox) / res)))

            sub = grid[r0:r1, c0:c1]

            # Only count unknown cells inside navigable arena bounds (|x| <= 2.35, |y| <= 2.35)
            col_indices = np.arange(c0, c1)
            row_indices = np.arange(r0, r1)
            col_x = ox + (col_indices + 0.5) * res
            row_y = oy + (row_indices + 0.5) * res
            valid_x = np.abs(col_x) <= 2.35
            valid_y = np.abs(row_y) <= 2.35
            in_arena_mask = np.outer(valid_y, valid_x)
            unknown_count = int(np.count_nonzero((sub == -1) & in_arena_mask))

            if unknown_count < 100:
                continue

            score = float(unknown_count) / (1.0 + 0.30 * dist_to_robot) - visited_penalty
            if score > best_score:
                # Find nearest free cell in case candidate coordinate is obstructed
                target_cand = None
                if grid[c_row, c_col] < 50:
                    target_cand = (cx, cy)
                else:
                    for dr in range(-10, 11):
                        for dc in range(-10, 11):
                            nr, nc = c_row + dr, c_col + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                                target_cand = (ox + (nc + 0.5) * res, oy + (nr + 0.5) * res)
                                break
                        if target_cand is not None:
                            break
                if target_cand is not None:
                    best_score = score
                    best_pt = target_cand

        return best_pt

    def _find_map_frontier_target(self) -> Optional[Tuple[float, float]]:
        """Find candidate frontier point from latest_map using OpenFrontier algorithm."""
        if self.latest_map is None:
            return None
        proposals = self.openfrontier.detect_frontiers(
            self.latest_map, self.current_x, self.current_y
        )
        self.openfrontier.merge_proposals(proposals)
        self.openfrontier.prune_cleared_frontiers(self.current_x, self.current_y)
        best = self.openfrontier.select_best_frontier(self.current_x, self.current_y)
        self._publish_frontier_markers()
        if best is not None:
            return (best.pos_x, best.pos_y)
        return self._find_raycast_frontier_target()

    def _select_exploration_target(self) -> Optional[Tuple[float, float]]:
        """
        Select next exploration target: user waypoints first, OpenFrontier map frontiers, then arena fallback.

        ========================================================================
        HOOK 2: FRONTIER DETECTION ALGORITHM EXTENSION POINT
        Pluggable frontier detection engines:
        1. OpenFrontier cluster detection & A* path planner (default active)
        2. Fast Raycast Frontier Edge Detector (boundary transition scan)
        3. Active Unknown Sector Density Surveyor (direct active unknown exploration)
        4. User-designated target & waypoints (highest priority override)
        ========================================================================
        """
        # 1. User-drawn path waypoints (highest priority manual guidance)
        while self.user_waypoints:
            target = self.user_waypoints.popleft()
            if (not self.has_odom or
                    math.hypot(target[0] - self.current_x, target[1] - self.current_y) >= 0.12 or
                    not self.user_waypoints):
                self.get_logger().info(
                    f'Navigating to user waypoint: ({target[0]:.2f}, {target[1]:.2f}).'
                )
                thought = (
                    f"I pick user waypoint target ({target[0]:.2f}, {target[1]:.2f}) "
                    f"to check if the target object is visible along this path."
                )
                self.publish_thought(thought, force=True)
                return target

        # 2. Existing planned frontier waypoints
        while self.frontier_waypoints:
            wp = self.frontier_waypoints.popleft()
            if (not self.has_odom or
                    math.hypot(wp[0] - self.current_x, wp[1] - self.current_y) >= 0.12 or
                    not self.frontier_waypoints):
                return wp

        # ======================================================================
        # HOOK 2.3: ACTIVE UNKNOWN SECTOR DENSITY SURVEYOR
        # Direct active exploration: targets unvisited sectors with high unknown volume.
        # Prioritized when major unexplored quadrants remain (> 100 unknown cells).
        # ======================================================================
        sector_pt = self._find_highest_unknown_sector()
        if sector_pt is not None:
            waypoints = self.openfrontier.plan_path(
                self.latest_map, self.current_x, self.current_y, sector_pt[0], sector_pt[1]
            )
            self._publish_planned_path(waypoints)
            self.frontier_waypoints.clear()
            if len(waypoints) > 1:
                valid_wps = [
                    pt for pt in waypoints
                    if math.hypot(pt[0] - self.current_x, pt[1] - self.current_y) >= 0.15
                ]
                if valid_wps:
                    target_pt = valid_wps[0]
                    for pt in valid_wps[1:]:
                        self.frontier_waypoints.append(pt)
                else:
                    target_pt = sector_pt
            else:
                target_pt = sector_pt

            fid = self.openfrontier.generate_next_id()
            self.openfrontier.current_goal = FrontierCandidate(
                id=f"SEC_{fid}",
                pos_x=sector_pt[0],
                pos_y=sector_pt[1],
                cluster_size=100,
                exploration_gain=500.0,
                distance=math.hypot(sector_pt[0] - self.current_x, sector_pt[1] - self.current_y),
                semantic_prob=1.0,
                total_utility=500.0,
                is_viewpoint_frontier=False,
            )
            self.openfrontier.stall_tracker.stall_counter = 0
            self.openfrontier.stall_tracker.d_prev = 1e9

            self.get_logger().info(
                f'[SectorSurveyor] Selected active unknown sector target at ({sector_pt[0]:.2f}, {sector_pt[1]:.2f}) '
                f'({len(waypoints)} A* waypoints).'
            )
            thought = (
                f"Navigating to unexplored sector ({sector_pt[0]:.2f}, {sector_pt[1]:.2f}) "
                f"to rapidly expand map coverage."
            )
            self.publish_thought(thought, force=True)
            return target_pt

        # ======================================================================
        # HOOK 2.1: OPENFRONTIER CLUSTER EXTRACTION & COLLISION-FREE A* PLANNING
        # Detects navigable boundary clusters between explored free and unknown cells.
        # ======================================================================
        if self.latest_map is not None:
            proposals = self.openfrontier.detect_frontiers(
                self.latest_map, self.current_x, self.current_y
            )
            self.openfrontier.merge_proposals(proposals)
            self.openfrontier.prune_cleared_frontiers(self.current_x, self.current_y)
            best = self.openfrontier.select_best_frontier(
                self.current_x, self.current_y, visited_history=list(self.visited_goals)
            )
            self._publish_frontier_markers()

            if best is not None and best.total_utility > 0:
                waypoints = self.openfrontier.plan_path(
                    self.latest_map, self.current_x, self.current_y, best.pos_x, best.pos_y
                )
                self._publish_planned_path(waypoints)
                self.frontier_waypoints.clear()

                if len(waypoints) > 1:
                    valid_wps = [
                        pt for pt in waypoints
                        if math.hypot(pt[0] - self.current_x, pt[1] - self.current_y) >= 0.15
                    ]
                    if valid_wps:
                        target_pt = valid_wps[0]
                        for pt in valid_wps[1:]:
                            self.frontier_waypoints.append(pt)
                    else:
                        target_pt = (best.pos_x, best.pos_y)
                else:
                    target_pt = (best.pos_x, best.pos_y)

                if self.openfrontier.current_goal is not None:
                    self.openfrontier.stall_tracker.stall_counter = 0
                    self.openfrontier.stall_tracker.d_prev = 1e9

                self.get_logger().info(
                    f'[OpenFrontier] Selected frontier {best.id} at ({best.pos_x:+.2f}, {best.pos_y:+.2f}) '
                    f'gain={best.exploration_gain:.1f}, dist={best.distance:.2f}m -> Utility={best.total_utility:.1f} '
                    f'({len(waypoints)} A* waypoints).'
                )
                thought = (
                    f"🔭 [OpenFrontier] Selected frontier {best.id} at ({best.pos_x:+.2f}, {best.pos_y:+.2f}) "
                    f"gain={best.exploration_gain:.1f}, dist={best.distance:.2f}m -> Utility={best.total_utility:.1f}. "
                    f"Following collision-free A* trajectory."
                )
                self.publish_thought(thought, force=True)
                return target_pt

        # ======================================================================
        # HOOK 2.2: RAYCAST FRONTIER EDGE DETECTOR (FAST BOUNDARY TRANSITION)
        # Direct line-of-sight unknown boundary detection from current robot pose.
        # ======================================================================
        raycast_pt = self._find_raycast_frontier_target()
        if raycast_pt is not None:
            self.get_logger().info(
                f'Frontier target selected from raycaster: ({raycast_pt[0]:.2f}, {raycast_pt[1]:.2f}).'
            )
            thought = (
                f"I pick frontier target ({raycast_pt[0]:.2f}, {raycast_pt[1]:.2f}) "
                f"because this unexplored boundary offers a vantage where I might see the target object."
            )
            self.publish_thought(thought, force=True)
            return raycast_pt

        # 4. First target fallback: Pick a courageous forward location (1.0m to 1.6m away)
        if self.is_first_target:
            self.is_first_target = False
            sx, sy = self.start_pose if self.start_pose is not None else (self.current_x, self.current_y)
            rand_ang = float(np.random.uniform(-math.pi, math.pi))
            rand_dist = float(np.random.uniform(1.00, 1.60))
            gx = float(np.clip(sx + rand_dist * math.cos(rand_ang), -2.10, 2.10))
            gy = float(np.clip(sy + rand_dist * math.sin(rand_ang), -2.10, 2.10))
            self.get_logger().info(
                f'Courageous initial target selected: ({gx:.2f}, {gy:.2f}) '
                f'at {rand_dist:.2f}m from start ({sx:.2f}, {sy:.2f}).'
            )
            thought = (
                f"I pick initial target ({gx:.2f}, {gy:.2f}) because establishing this forward "
                f"vantage gives a clear angle where I might see the target object."
            )
            self.publish_thought(thought, force=True)
            return (gx, gy)

        # 5. Fallback: Select an unvisited location in arena bounds
        for _ in range(25):
            rand_ang = float(np.random.uniform(-math.pi, math.pi))
            rand_dist = float(np.random.uniform(0.90, 1.60))
            gx = float(np.clip(self.current_x + rand_dist * math.cos(rand_ang), -2.10, 2.10))
            gy = float(np.clip(self.current_y + rand_dist * math.sin(rand_ang), -2.10, 2.10))
            too_close = any(math.hypot(gx - vx, gy - vy) < 0.50 for vx, vy in self.visited_goals[-8:])
            if not too_close:
                self.get_logger().info(f'Autonomous arena target selected: ({gx:.2f}, {gy:.2f}).')
                thought = (
                    f"I pick unvisited arena sector ({gx:.2f}, {gy:.2f}) "
                    f"to search for the target object from a fresh vantage point."
                )
                self.publish_thought(thought, force=True)
                return (gx, gy)

        # Fallback to opposite direction if all candidates close
        escape_ang = normalize_angle(self.current_yaw + math.pi)
        gx = float(np.clip(self.current_x + 1.2 * math.cos(escape_ang), -2.10, 2.10))
        gy = float(np.clip(self.current_y + 1.2 * math.sin(escape_ang), -2.10, 2.10))
        thought = (
            f"Repositioning toward ({gx:.2f}, {gy:.2f}) to clear obstacles and "
            f"reopen wide visual search for the target object."
        )
        self.publish_thought(thought, force=True)
        return (gx, gy)

    # ==========================================================================
    # Service Handlers
    # ==========================================================================

    def toggle_explorer_callback(self, request, response):
        """Toggle explorer on/off via Trigger service."""
        new_state = not self.enabled
        self._set_enabled(new_state, 'service /toggle_explorer')
        response.success = True
        status_str = 'activated' if self.enabled else 'paused'
        response.message = f'Reactive explorer {status_str}.'
        return response

    def set_explorer_callback(self, request, response):
        """Set explorer state explicitly via SetBool service."""
        self._set_enabled(request.data, 'service /set_explorer')
        response.success = True
        response.message = f'Reactive explorer state set to: {self.enabled}.'
        return response

    def _set_enabled(self, state: bool, source: str):
        """Update enabled state and log change."""
        if state and self.target_stopped:
            self.get_logger().info('Cannot enable explorer: target already reached and stopped.')
            return
        if self.enabled != state:
            self.enabled = state
            self.get_logger().info(
                f'Explorer state changed to {self.enabled} via {source}.'
            )
            if hasattr(self, 'explorer_status_pub'):
                self.explorer_status_pub.publish(Bool(data=self.enabled))
            if hasattr(self, 'explorer_enable_pub'):
                self.explorer_enable_pub.publish(Bool(data=self.enabled))
            if not self.enabled:
                self.state = self.STATE_PAUSED
                self._stop_robot()
            else:
                self.state = self.STATE_CRUISE
                self.state_start_time = self._get_current_time()

    def _stop_robot(self):
        """Command zero velocity to halt robot."""
        cmd = Twist()
        self.last_commanded_v = 0.0
        self.cmd_pub.publish(cmd)

    # ==========================================================================
    # Sensor Reading Aggregation
    # ==========================================================================

    def _extract_sector_distances(self) -> Tuple[float, float, float]:
        """Aggregate left, front, and right proximity distances."""
        d_left = 2.0
        d_front = 2.0
        d_right = 2.0

        # 1. Ultrasonic sensor mapping:
        for i, scan in enumerate(self.latest_us):
            if scan is None or not scan.ranges:
                continue
            valid_ranges = [
                r for r in scan.ranges
                if scan.range_min <= r <= scan.range_max
            ]
            if not valid_ranges:
                continue
            r_min = min(valid_ranges)

            if i in (0, 1):
                d_left = min(d_left, r_min)
            if i in (1, 2, 3):
                d_front = min(d_front, r_min if i == 2 else r_min * 1.1)
            if i in (3, 4):
                d_right = min(d_right, r_min)

        # 2. Infrared sensor mapping:
        for i, scan in enumerate(self.latest_ir):
            if scan is None or not scan.ranges:
                continue
            valid_ranges = [
                r for r in scan.ranges
                if scan.range_min <= r <= scan.range_max
            ]
            if not valid_ranges:
                continue
            r_min = min(valid_ranges)

            if i in (0, 1, 2):
                d_left = min(d_left, r_min)
            if i in (0, 7):
                d_front = min(d_front, r_min)
            if i in (5, 6, 7):
                d_right = min(d_right, r_min)

        # 3. Ground IR sensors (downward cliff edge or ground obstacle contact)
        for i, scan in enumerate(self.latest_ground_ir):
            if scan is None or not scan.ranges:
                continue
            r = scan.ranges[0]
            if math.isinf(r) or math.isnan(r) or r > 0.025 or r < 0.0035:
                d_front = min(d_front, 0.04)
                if i in (0, 1):
                    d_left = min(d_left, 0.04)
                if i in (2, 3):
                    d_right = min(d_right, 0.04)

        if self.cliff_alert:
            d_front = min(d_front, 0.04)

        # 4. Virtual scan fallback/supplement (correct angular coordinate indexing)
        if self.latest_scan is not None and self.latest_scan.ranges:
            ranges = self.latest_scan.ranges
            ang_min = self.latest_scan.angle_min
            ang_inc = self.latest_scan.angle_increment

            for idx, r in enumerate(ranges):
                if math.isnan(r) or math.isinf(r) or r < 0.01 or r > 2.0:
                    continue
                # Angle relative to robot heading (0 rad is directly ahead, +pi/2 is left, -pi/2 is right)
                ang = normalize_angle(ang_min + idx * ang_inc)
                if abs(ang) <= 0.61:  # ~35 deg forward cone
                    d_front = min(d_front, r)
                if 0.35 < ang <= 1.83:  # ~20 to 105 deg left sector
                    d_left = min(d_left, r)
                if -1.83 <= ang < -0.35:  # ~-105 to -20 deg right sector
                    d_right = min(d_right, r)

        return d_left, d_front, d_right

    def _compute_frontier_heading(self, now: float) -> Optional[float]:
        """Evaluate occupancy grid map to find direction with largest unexplored frontier."""
        if self.latest_map is None or not self.has_odom:
            return None

        # Cache calculation for 0.25 seconds to maintain high control rate
        if (now - self._last_frontier_calc_time) < 0.25:
            return self._cached_frontier_heading

        self._last_frontier_calc_time = now
        info = self.latest_map.info
        res = info.resolution
        w = info.width
        h = info.height
        ox = info.origin.position.x
        oy = info.origin.position.y

        if len(self.latest_map.data) != w * h:
            self._cached_frontier_heading = None
            return None

        grid = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))

        rx = self.current_x
        ry = self.current_y

        num_rays = 36
        best_score = -1e9
        best_heading = None

        # Sample distance checkpoints from 0.15m to 1.8m
        ray_dists = [0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 1.05, 1.30, 1.55, 1.80]

        for k in range(num_rays):
            heading = -math.pi + k * (2.0 * math.pi / num_rays)
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            unknown_count = 0
            clearance = 1.80

            for d in ray_dists:
                px = rx + d * cos_h
                py = ry + d * sin_h
                col = int(math.floor((px - ox) / res))
                row = int(math.floor((py - oy) / res))

                if col < 0 or col >= w or row < 0 or row >= h:
                    clearance = d
                    break

                val = grid[row, col]
                if val >= 50:  # Obstacle / wall hit
                    clearance = d
                    break
                elif val == -1:  # Unexplored frontier cell
                    unknown_count += 1

            if clearance < 0.30:
                continue

            diff = abs(normalize_angle(heading - self.current_yaw))

            # Exploration utility score:
            # - Reward unknown cells (frontier expansion)
            # - Reward clear corridors (safe open space)
            # - Heading continuity bonus
            score = 3.0 * unknown_count + 2.0 * clearance - 1.2 * diff

            # Penalize rays heading outside arena limits
            dest_x = rx + clearance * cos_h
            dest_y = ry + clearance * sin_h
            if abs(dest_x) > 2.20 or abs(dest_y) > 2.20:
                score -= 8.0

            if score > best_score:
                best_score = score
                best_heading = heading

        self._cached_frontier_heading = best_heading
        return best_heading

    def _compute_perimeter_repulsion(self) -> float:
        """Repel robot away from outer perimeter walls ([-2.405, 2.405]m)."""
        if not self.has_odom:
            return 0.0
        bound = 2.10
        repulse_x = 0.0
        repulse_y = 0.0

        if self.current_x > bound:
            repulse_x = -(self.current_x - bound) / 0.30
        elif self.current_x < -bound:
            repulse_x = (-bound - self.current_x) / 0.30

        if self.current_y > bound:
            repulse_y = -(self.current_y - bound) / 0.30
        elif self.current_y < -bound:
            repulse_y = (-bound - self.current_y) / 0.30

        if abs(repulse_x) < 1e-3 and abs(repulse_y) < 1e-3:
            return 0.0

        desired_ang = math.atan2(repulse_y, repulse_x)
        ang_diff = normalize_angle(desired_ang - self.current_yaw)
        return float(np.clip(1.2 * ang_diff, -1.0, 1.0))

    # ==========================================================================
    # Main Control Loop
    # ==========================================================================

    def control_loop(self):
        """Execute high-frequency reactive steering loop."""
        if self.target_stopped or not self.enabled:
            return

        # When visual pursuit is active and target is in sight, yield control to IBVS controller
        if self.pursuit_active and self.target_in_sight:
            return

        if self.is_turned_over:
            self._stop_robot()
            return

        if self.state == self.STATE_PAUSED:
            self.state = self.STATE_CRUISE
            self.state_start_time = self._get_current_time()

        cmd = Twist()
        now = self._get_current_time()
        if now <= 0.0:
            return
        elapsed_state = now - self.state_start_time
        t_wall = now - self.start_wall_clock

        # ----------------------------------------------------------------------
        # State: CLIFF RECOVERY (Top Priority)
        # ----------------------------------------------------------------------
        if self.state == self.STATE_CLIFF_RECOVERY:
            if elapsed_state < self.cliff_backup_duration:
                cmd.linear.x = -self.reverse_speed
                cmd.angular.z = self.escape_turn_dir * self.turn_speed
                self.last_commanded_v = cmd.linear.x
                self.cmd_pub.publish(cmd)
                return
            if not self.cliff_alert:
                self.get_logger().info(
                    'Cliff hazard cleared. Resuming exploration.'
                )
                self.state = self.STATE_CRUISE
                self.state_start_time = now
                self.stall_start_time = None
            else:
                cmd.linear.x = -self.reverse_speed
                cmd.angular.z = self.escape_turn_dir * self.turn_speed
                self.last_commanded_v = cmd.linear.x
                self.cmd_pub.publish(cmd)
                return

        # ----------------------------------------------------------------------
        # State: REVERSE TURN (180° Emergency Escape from Boxed-in / Corner)
        # ----------------------------------------------------------------------
        if self.state == self.STATE_REVERSE_TURN:
            turn_rate = max(self.turn_speed, 1.2)
            turn_duration_est = math.pi / turn_rate
            diff = self.current_yaw - self.reverse_turn_initial_yaw
            yaw_turned = abs(normalize_angle(diff))

            is_turned = elapsed_state > 0.8 and yaw_turned >= (math.pi - 0.25)
            is_timeout = elapsed_state >= (turn_duration_est + 0.6)
            if is_turned or is_timeout:
                deg_t = math.degrees(yaw_turned)
                self.get_logger().info(
                    f'180° escape turn completed ({deg_t:.1f}°, '
                    f'{elapsed_state:.2f}s). Resuming exploration.'
                )
                self.state = self.STATE_CRUISE
                self.state_start_time = now
                self.stall_start_time = None
                self.last_impact_time = -100.0
                self.imu_impact_detected = False
                self.last_commanded_v = 0.0
            else:
                cmd.linear.x = (
                    -self.reverse_speed if elapsed_state < 0.8 else 0.0
                )
                cmd.angular.z = self.escape_turn_dir * turn_rate
                self.last_commanded_v = cmd.linear.x
                self.cmd_pub.publish(cmd)
                return

        # ----------------------------------------------------------------------
        # State: INSPECT (Arrived at Target: Rotate & Scan Surroundings)
        # ----------------------------------------------------------------------
        if self.state == self.STATE_INSPECT:
            if elapsed_state < self.inspect_duration:
                cmd.linear.x = 0.0
                cmd.angular.z = self.inspect_angular_speed
                self.last_commanded_v = 0.0
                self.cmd_pub.publish(cmd)
                return
            self.get_logger().info('Inspection scan completed. Selecting next exploration target.')
            self.state = self.STATE_CRUISE
            self.state_start_time = now
            self.stall_start_time = None
            self.active_goal = self._select_exploration_target()
            self.goal_start_time = now
            self._publish_active_goal()

        # ----------------------------------------------------------------------
        # State: INITIAL_SPIN (Quick 360° survey upon map spawn)
        # ----------------------------------------------------------------------
        if self.state == self.STATE_INITIAL_SPIN:
            delta_yaw = abs(normalize_angle(self.current_yaw - self.spin_last_yaw))
            self.spin_cumulative_yaw += delta_yaw
            self.spin_last_yaw = self.current_yaw

            if self.spin_cumulative_yaw >= (2.0 * math.pi - 0.05) or elapsed_state > 12.0:
                self.get_logger().info(
                    'Initial 360° survey completed. Seeding SLAM map and mapping frontiers.'
                )
                self.initial_spin_done = True
                self.state = self.STATE_CRUISE
                self.state_start_time = now
                self.stall_start_time = None
                self.active_goal = self._select_exploration_target()
                self.goal_start_time = now
                self._publish_active_goal()
                self._publish_frontier_markers()
                if self.active_goal is not None:
                    self.publish_thought(
                        f"🔭 360° survey complete. Mapped frontiers from SLAM map. "
                        f"Navigating to frontier at ({self.active_goal[0]:+.2f}, {self.active_goal[1]:+.2f}).",
                        force=True
                    )
                else:
                    self.publish_thought(
                        "360° survey complete. Beginning room frontier exploration.",
                        force=True
                    )
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.initial_spin_speed
                self.last_commanded_v = 0.0
                self.cmd_pub.publish(cmd)
                return

        # Read sector distances
        d_left, d_front, d_right = self._extract_sector_distances()

        # ----------------------------------------------------------------------
        # Immediate Bump / Near-Contact (< 0.015m) or IMU Impact Jerk Detection
        # ----------------------------------------------------------------------
        min_ir = float('inf')
        min_ir_idx = -1
        for i, scan in enumerate(self.latest_ir):
            if scan is not None and scan.ranges:
                for r in scan.ranges:
                    if (not math.isnan(r) and not math.isinf(r) and
                            scan.range_min <= r <= scan.range_max):
                        if r < min_ir:
                            min_ir = r
                            min_ir_idx = i

        near_contact = (min_ir < 0.015)
        recent_impact = self.imu_impact_detected or (0.0 <= (now - self.last_impact_time) < 0.35)
        self.imu_impact_detected = False

        if near_contact or recent_impact:
            self.get_logger().warn(
                f'Collision detected (IR min={min_ir:.4f}m, jerk={self.imu_jerk:.1f}m/s^3). '
                'Executing immediate evasive reverse turn.'
            )
            self.state = self.STATE_REVERSE_TURN
            self.state_start_time = now
            self.reverse_turn_initial_yaw = self.current_yaw
            self.last_impact_time = -100.0
            if min_ir_idx in (0, 1, 2, 3):
                self.escape_turn_dir = -1.0  # Left contact -> turn right
            elif min_ir_idx in (4, 5, 6, 7):
                self.escape_turn_dir = 1.0   # Right contact -> turn left
            else:
                self.escape_turn_dir = 1.0 if d_left >= d_right else -1.0
            self.stall_start_time = None
            cmd.linear.x = -self.reverse_speed
            cmd.angular.z = self.escape_turn_dir * self.turn_speed
            self.last_commanded_v = cmd.linear.x
            self.cmd_pub.publish(cmd)
            return

        # ----------------------------------------------------------------------
        # Corner & Wall Stalling Detection (Physical impediment only)
        # ----------------------------------------------------------------------
        is_stalled = False
        if self.last_commanded_v > 0.05:
            oldest_sample = None
            for t_samp, x_samp, y_samp in self.pos_history:
                if t_samp <= (now - 0.25):
                    oldest_sample = (t_samp, x_samp, y_samp)
                    break
            if oldest_sample is None and self.pos_history:
                oldest_sample = min(self.pos_history, key=lambda s: s[0])
                if (now - oldest_sample[0]) < 0.15:
                    oldest_sample = None

            if oldest_sample is not None:
                dt_sample = now - oldest_sample[0]
                ref_x, ref_y = self.current_x, self.current_y

                disp = math.hypot(
                    ref_x - oldest_sample[1],
                    ref_y - oldest_sample[2]
                )
                accel_var = 1.0
                if len(self.accel_history) >= 10:
                    mean_a = sum(self.accel_history) / len(self.accel_history)
                    accel_var = sum(
                        (a - mean_a) ** 2 for a in self.accel_history
                    ) / len(self.accel_history)

                expected_disp = self.last_commanded_v * dt_sample
                has_obstacle_front = (d_front < 0.08 or min_ir < 0.035)
                is_stationary = (disp < 0.008 and expected_disp >= 0.015)
                is_contact_stalled = (disp < 0.018 and has_obstacle_front)
                is_slip_stalled = (disp < 0.020 and accel_var < 0.003 and expected_disp >= 0.025)

                if is_stationary or is_contact_stalled or is_slip_stalled:
                    is_stalled = True

        if is_stalled:
            self.stall_ticks += 1
            if self.stall_start_time is None:
                self.stall_start_time = now
            if (now - self.stall_start_time) >= self.stall_timeout or self.stall_ticks >= 7:
                self.get_logger().warn(
                    f'Corner stall detected ({now - self.stall_start_time:.2f}s, '
                    f'ticks={self.stall_ticks}). Triggering 180° reverse escape.'
                )
                self.state = self.STATE_REVERSE_TURN
                self.state_start_time = now
                self.reverse_turn_initial_yaw = self.current_yaw
                self.escape_turn_dir = 1.0 if d_left >= d_right else -1.0
                self.stall_start_time = None
                self.stall_ticks = 0
                cmd.linear.x = -self.reverse_speed
                cmd.angular.z = self.escape_turn_dir * self.turn_speed
                self.last_commanded_v = cmd.linear.x
                self.cmd_pub.publish(cmd)
                return
        else:
            self.stall_start_time = None
            self.stall_ticks = 0

        # ----------------------------------------------------------------------
        # Check for immediate front obstacle / corner entrapment
        # ----------------------------------------------------------------------
        if d_front <= self.stop_dist or (d_left < 0.04 and d_right < 0.04):
            self.state = self.STATE_REVERSE_TURN
            self.state_start_time = now
            self.reverse_turn_initial_yaw = self.current_yaw
            self.escape_turn_dir = 1.0 if d_left >= d_right else -1.0
            cmd.linear.x = -self.reverse_speed
            cmd.angular.z = self.escape_turn_dir * self.turn_speed
            self.last_commanded_v = cmd.linear.x
            self.cmd_pub.publish(cmd)
            return

        # ----------------------------------------------------------------------
        # Target Selection & Arrival Check
        # ----------------------------------------------------------------------
        # Check if initial survey spin should be triggered (first time spawned in empty SLAM map)
        if (not self.initial_spin_done and self.is_first_target and self.active_goal is None
                and self.state == self.STATE_CRUISE
                and self.has_odom and not self.user_waypoints and not self.is_single_user_goal
                and self.last_commanded_v <= 0.02):
            map_is_empty = (
                self.latest_map is None
                or np.count_nonzero(np.array(self.latest_map.data) != -1) < 200
            )
            if map_is_empty and not getattr(self, '_skip_initial_spin', False):
                self.state = self.STATE_INITIAL_SPIN
                self.state_start_time = now
                self.spin_cumulative_yaw = 0.0
                self.spin_last_yaw = self.current_yaw
                self.publish_thought(
                    "Executing initial 360° scan to survey surroundings and seed SLAM map.",
                    force=True
                )
                cmd.linear.x = 0.0
                cmd.angular.z = self.initial_spin_speed
                self.last_commanded_v = 0.0
                self.cmd_pub.publish(cmd)
                return

        if self.active_goal is None:
            self.active_goal = self._select_exploration_target()
            self.goal_start_time = now
            self._publish_active_goal()

        if self.active_goal is not None and self.has_odom:
            dist_to_goal = math.hypot(
                self.active_goal[0] - self.current_x,
                self.active_goal[1] - self.current_y
            )
            # Smooth user-drawn path advancement (zero stopping, zero spinning)
            while self.user_waypoints and dist_to_goal <= 0.16:
                self.visited_goals.append(self.active_goal)
                self.active_goal = self.user_waypoints.popleft()
                self.goal_start_time = now
                self._publish_active_goal()
                dist_to_goal = math.hypot(
                    self.active_goal[0] - self.current_x,
                    self.active_goal[1] - self.current_y
                )

            # Smooth frontier path advancement (A* waypoints)
            while not self.user_waypoints and self.frontier_waypoints and dist_to_goal <= 0.20:
                self.active_goal = self.frontier_waypoints.popleft()
                self.goal_start_time = now
                self._publish_active_goal()
                if self.openfrontier.current_goal is not None:
                    self.openfrontier.stall_tracker.stall_counter = 0
                    self.openfrontier.stall_tracker.d_prev = 1e9
                dist_to_goal = math.hypot(
                    self.active_goal[0] - self.current_x,
                    self.active_goal[1] - self.current_y
                )

            # OpenFrontier Stall Recovery (Algorithm 3)
            if not self.is_single_user_goal and self.openfrontier.current_goal is not None:
                if d_front > 0.155 and self.openfrontier.handle_progress(dist_to_goal):
                    self.get_logger().warn(

                        f'[OpenFrontier] Goal {self.openfrontier.current_goal.id} stalled! '
                        f'Pruning and selecting next frontier.'
                    )
                    self.openfrontier.drop_frontier(self.openfrontier.current_goal.id)
                    self.visited_goals.append(self.active_goal)
                    self.frontier_waypoints.clear()
                    self.active_goal = self._select_exploration_target()
                    self.goal_start_time = now
                    self._publish_active_goal()
                    if self.active_goal is not None:
                        dist_to_goal = math.hypot(
                            self.active_goal[0] - self.current_x,
                            self.active_goal[1] - self.current_y
                        )

            # Destination arrival check (only for final target or single goal)
            arrival_dist = 0.22 if self.is_single_user_goal else 0.20
            if dist_to_goal <= arrival_dist and not self.user_waypoints and not self.frontier_waypoints:
                self.get_logger().info(
                    f'Arrived at target ({self.active_goal[0]:.2f}, {self.active_goal[1]:.2f}) '
                    f'[dist={dist_to_goal:.2f}m].'
                )
                self.visited_goals.append(self.active_goal)
                self.openfrontier.prune_cleared_frontiers(self.current_x, self.current_y)
                if self.openfrontier.current_goal is not None:
                    self.openfrontier.drop_frontier(self.openfrontier.current_goal.id)
                self.active_goal = None

                if self.is_single_user_goal:
                    self.is_single_user_goal = False
                    self.state = self.STATE_INSPECT
                    self.state_start_time = now
                    self._stop_robot()
                    return
                else:
                    self.state = self.STATE_CRUISE
                    self.state_start_time = now
                    self.stall_start_time = None
                    self.active_goal = self._select_exploration_target()
                    self.goal_start_time = now
                    self._publish_active_goal()

            # Check if target is obstructed or unreachable (> 25s)
            if (now - self.goal_start_time) > 25.0:
                self.get_logger().warn(
                    f'Target {self.active_goal} obstructed/timeout ({now - self.goal_start_time:.1f}s). '
                    'Selecting next target.'
                )
                if self.openfrontier.current_goal is not None:
                    self.openfrontier.drop_frontier(self.openfrontier.current_goal.id)
                self.visited_goals.append(self.active_goal)
                self.frontier_waypoints.clear()
                self.active_goal = self._select_exploration_target()
                self.goal_start_time = now
                self._publish_active_goal()

        # Periodically publish active goal (every 1.0s)
        if self.active_goal is not None and (int(now * 10) % 10 == 0):
            self._publish_active_goal()

        # ----------------------------------------------------------------------
        # State: CRUISE & AVOIDANCE (Courageous Target Seeking + Braitenberg Repulsion)
        # ----------------------------------------------------------------------
        safe_dist = 0.15  # At or below 15cm: ZERO forward motion, pure in-place turn
        slow_dist = max(self.avoid_dist, 0.40)

        w_left = max(0.0, (self.avoid_dist - d_left) / self.avoid_dist)
        w_right = max(0.0, (self.avoid_dist - d_right) / self.avoid_dist)
        w_front = max(0.0, (self.avoid_dist - d_front) / self.avoid_dist)

        w_repulse = (w_right - w_left) * self.turn_speed * 1.5
        if w_front > 0.0:
            if now > getattr(self, 'front_steer_lock_time', 0.0):
                if abs(d_left - d_right) > 0.03:
                    self.front_steer_dir = 1.0 if d_left >= d_right else -1.0
                    self.front_steer_lock_time = now + 0.8
            turn_bias = self.front_steer_dir * w_front * self.turn_speed
            w_repulse += turn_bias

        # Target guidance towards active_goal
        w_target = 0.0
        goal_err = 0.0
        if self.active_goal is not None and self.has_odom:
            dx = self.active_goal[0] - self.current_x
            dy = self.active_goal[1] - self.current_y
            goal_ang = math.atan2(dy, dx)
            goal_err = normalize_angle(goal_ang - self.current_yaw)
            w_target = 1.6 * goal_err
        else:
            # Frontier guidance towards unknown map areas
            frontier_heading = self._compute_frontier_heading(now)
            if frontier_heading is not None and w_front < 0.20 and w_left < 0.30 and w_right < 0.30:
                frontier_diff = normalize_angle(frontier_heading - self.current_yaw)
                w_target = 0.85 * frontier_diff

        # Arena boundary avoidance
        w_perim = self._compute_perimeter_repulsion()

        # Exploration wander perturbation (only when no active goal)
        w_wander = (0.08 * math.sin(0.40 * t_wall) + 0.06 * math.cos(0.17 * t_wall)) if self.active_goal is None else 0.0

        # Blend target guidance and obstacle repulsion
        if w_front > 0.15:
            target_w = w_repulse + 0.60 * w_target + w_perim
        elif (w_left > 0.15 or w_right > 0.15):
            target_w = 0.70 * w_repulse + w_target + w_perim
        else:
            target_w = w_target + w_perim + w_wander

        target_w = float(
            max(-self.max_turn_speed, min(self.max_turn_speed, target_w))
        )

        # Proactive Speed Governor: strictly 0 forward speed when obstacle is closer than safe_dist
        if d_front <= (safe_dist + 1e-3):
            target_v = 0.0
            if now > getattr(self, 'front_steer_lock_time', 0.0):
                self.front_steer_dir = 1.0 if d_left >= d_right else -1.0
                self.front_steer_lock_time = now + 0.8
            target_w = self.front_steer_dir * self.turn_speed
        elif d_front < slow_dist:
            u = (d_front - safe_dist) / (slow_dist - safe_dist)
            speed_factor = max(0.35, u)  # Smooth linear deceleration, maintaining momentum
            turn_ratio = abs(target_w) / self.max_turn_speed
            turn_red = max(0.40, 1.0 - 0.35 * turn_ratio)
            target_v = self.linear_speed * speed_factor * turn_red
        else:
            speed_factor = max(0.45, 1.0 - 0.50 * w_front)
            turn_ratio = abs(target_w) / self.max_turn_speed
            turn_red = max(0.50, 1.0 - 0.30 * turn_ratio)
            target_v = self.linear_speed * speed_factor * turn_red

        # Maintain positive forward speed when heading towards target with clear path ahead
        if d_front > 0.30 and abs(goal_err) < 0.8 and w_front < 0.25:
            target_v = max(target_v, min(self.linear_speed, 0.35))
        target_v = min(target_v, self.linear_speed)

        if target_v < 1e-4:
            target_v = 0.0

        # Rate-limit positive acceleration and deceleration to eliminate wheel slip
        max_dv_pos = 0.028  # Rate-limited to prevent wheel slip (< 0.030 m/s per tick)
        max_dv_neg = 0.040  # ~0.8 m/s^2 at 20Hz

        base_v = max(0.0, self.last_commanded_v)
        if target_v > (base_v + max_dv_pos):
            target_v = base_v + max_dv_pos
        elif target_v < (base_v - max_dv_neg) and d_front > (safe_dist + 1e-3):
            target_v = base_v - max_dv_neg

        cmd.linear.x = float(target_v)
        cmd.angular.z = float(target_w)
        self.last_commanded_v = cmd.linear.x

        self.cmd_pub.publish(cmd)


def main(args=None):
    """Run the ReactiveExplorer node."""
    rclpy.init(args=args)
    node = ReactiveExplorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
