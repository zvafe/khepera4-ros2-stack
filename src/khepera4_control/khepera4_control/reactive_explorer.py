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

from geometry_msgs.msg import Twist
from khepera4_control.common import euler_from_quaternion, normalize_angle
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger


class ReactiveExplorer(Node):
    """Autonomous reactive exploration and obstacle/cliff avoidance node."""

    # Navigation States
    STATE_CRUISE = 0
    STATE_AVOID = 1
    STATE_REVERSE_TURN = 2
    STATE_CLIFF_RECOVERY = 3
    STATE_PAUSED = 4

    def _get_current_time(self) -> float:
        """Get current node clock time in fractional seconds (respects use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

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

        self.enabled = bool(self.get_parameter('enabled').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.reverse_speed = float(self.get_parameter('reverse_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.max_turn_speed = float(
            self.get_parameter('max_turn_speed').value
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

        # ----------------------------------------------------------------------
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
        self.last_impact_time = 0.0
        self.imu_impact_detected = False

        # Sensor readings caches
        self.latest_us: List[Optional[LaserScan]] = [None] * 5
        self.latest_ir: List[Optional[LaserScan]] = [None] * 8
        self.latest_ground_ir: List[Optional[LaserScan]] = [None] * 4
        self.latest_scan: Optional[LaserScan] = None

        # ----------------------------------------------------------------------
        # Publishers & Subscribers
        # ----------------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Latched explorer status publisher (so dashboard immediately gets state on connect)
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.explorer_status_pub = self.create_publisher(Bool, '/explorer/enable', qos_latched)
        self.explorer_status_pub.publish(Bool(data=self.enabled))

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
        self.scan_sub = self.create_subscription(
            LaserScan, '/khepera4/scan', self.scan_callback, 10
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
            if self.imu_jerk > 60.0 or ax < -1.8:
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

    def us_callback(self, index: int, msg: LaserScan):
        """Cache ultrasonic scan."""
        self.latest_us[index] = msg

    def ir_callback(self, index: int, msg: LaserScan):
        """Cache infrared scan."""
        self.latest_ir[index] = msg

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
        if self.enabled != state:
            self.enabled = state
            self.get_logger().info(
                f'Explorer state changed to {self.enabled} via {source}.'
            )
            if hasattr(self, 'explorer_status_pub'):
                self.explorer_status_pub.publish(Bool(data=self.enabled))
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
            if math.isinf(r) or math.isnan(r) or r > 0.025 or r < 0.005:
                d_front = min(d_front, 0.04)
                if i in (0, 1):
                    d_left = min(d_left, 0.04)
                if i in (2, 3):
                    d_right = min(d_right, 0.04)

        if self.cliff_alert:
            d_front = min(d_front, 0.04)

        # 4. Virtual scan fallback/supplement
        if self.latest_scan is not None and self.latest_scan.ranges:
            ranges = self.latest_scan.ranges
            n = len(ranges)
            front_left = list(range(0, int(n * 25 / 360)))
            front_right = list(range(int(n * 335 / 360), n))
            front_indices = front_left + front_right
            left_indices = range(int(n * 25 / 360), int(n * 90 / 360))
            right_indices = range(int(n * 270 / 360), int(n * 335 / 360))

            for idx in front_indices:
                r = ranges[idx]
                if 0.02 <= r <= 2.0:
                    d_front = min(d_front, r)
            for idx in left_indices:
                r = ranges[idx]
                if 0.02 <= r <= 2.0:
                    d_left = min(d_left, r)
            for idx in right_indices:
                r = ranges[idx]
                if 0.02 <= r <= 2.0:
                    d_right = min(d_right, r)

        return d_left, d_front, d_right

    # ==========================================================================
    # Main Control Loop
    # ==========================================================================

    def control_loop(self):
        """Execute high-frequency reactive steering loop."""
        if not self.enabled or self.is_turned_over:
            self._stop_robot()
            return

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
            else:
                cmd.linear.x = (
                    -self.reverse_speed if elapsed_state < 0.8 else 0.0
                )
                cmd.angular.z = self.escape_turn_dir * turn_rate
                self.last_commanded_v = cmd.linear.x
                self.cmd_pub.publish(cmd)
                return

        # Read sector distances
        d_left, d_front, d_right = self._extract_sector_distances()

        # ----------------------------------------------------------------------
        # Immediate Bump / Near-Contact (< 0.020m) or IMU Impact Jerk Detection
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
        recent_impact = self.imu_impact_detected or (now - self.last_impact_time < 0.35)
        self.imu_impact_detected = False

        if near_contact or recent_impact:
            self.get_logger().warn(
                f'Collision detected (IR min={min_ir:.4f}m, jerk={self.imu_jerk:.1f}m/s^3). '
                'Executing immediate evasive reverse turn.'
            )
            self.state = self.STATE_REVERSE_TURN
            self.state_start_time = now
            self.reverse_turn_initial_yaw = self.current_yaw
            if min_ir_idx in (0, 1, 2, 3):
                self.escape_turn_dir = -1.0  # Left contact -> turn right
            elif min_ir_idx in (4, 5, 6, 7):
                self.escape_turn_dir = 1.0   # Right contact -> turn left
            else:
                self.escape_turn_dir = 1.0 if d_left >= d_right else -1.0
            self.stall_start_time = None
            cmd.linear.x = -self.reverse_speed
            cmd.angular.z = self.escape_turn_dir * max(self.turn_speed, 1.2)
            self.last_commanded_v = cmd.linear.x
            self.cmd_pub.publish(cmd)
            return

        # ----------------------------------------------------------------------
        # Corner & Wall Stalling Detection (<= 0.4s at commanded v > 0.05 m/s)
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

                if (disp < 0.025 or
                        (disp < 0.04 and accel_var < 0.005) or
                        (d_front < 0.05 and disp < 0.020) or
                        (min_ir < 0.035 and disp < 0.015)):
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
                cmd.angular.z = self.escape_turn_dir * max(
                    self.turn_speed, 1.2
                )
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
            cmd.angular.z = self.escape_turn_dir * max(self.turn_speed, 1.2)
            self.last_commanded_v = cmd.linear.x
            self.cmd_pub.publish(cmd)
            return

        # ----------------------------------------------------------------------
        # State: CRUISE & AVOIDANCE (Braitenberg Repulsion)
        # ----------------------------------------------------------------------
        w_left = max(0.0, (self.avoid_dist - d_left) / self.avoid_dist)
        w_right = max(0.0, (self.avoid_dist - d_right) / self.avoid_dist)
        w_front = max(0.0, (self.avoid_dist - d_front) / self.avoid_dist)

        w_repulse = (w_right - w_left) * self.turn_speed * 1.5
        if w_front > 0.0:
            steer_sign = 1.0 if d_left >= d_right else -1.0
            turn_bias = steer_sign * w_front * self.turn_speed
            w_repulse += turn_bias

        w_wander = (
            0.18 * math.sin(0.40 * t_wall) + 0.12 * math.cos(0.17 * t_wall)
        )
        target_w = w_repulse + w_wander
        target_w = float(
            max(-self.max_turn_speed, min(self.max_turn_speed, target_w))
        )

        speed_factor = max(0.20, 1.0 - 0.85 * w_front)
        turn_ratio = abs(target_w) / self.max_turn_speed
        turn_red = max(0.30, 1.0 - 0.40 * turn_ratio)
        target_v = self.linear_speed * speed_factor * turn_red

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
