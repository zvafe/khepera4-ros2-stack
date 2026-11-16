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

"""Centralized Behavior Tree supervisor and motion arbiter node for visual pursuit."""

import json
import math
from typing import List, Optional

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

# Behavior Tree state identifiers
STATE_IDLE: str = "IDLE"
STATE_FRONTIER_EXPLORATION: str = "FRONTIER_EXPLORATION"
STATE_VISUAL_PURSUIT: str = "VISUAL_PURSUIT"
STATE_TARGET_RECOVERY: str = "TARGET_RECOVERY"
STATE_TERMINAL_STOP: str = "TERMINAL_STOP"


class PursuitBTSupervisorNode(Node):
    """Centralized Behavior Tree supervisor orchestrating visual pursuit and exploration."""

    def __init__(self) -> None:
        """Initialize parameters, subscribers, publishers, and control evaluation timer."""
        super().__init__("pursuit_bt_supervisor_node")

        self._declare_parameters()
        self._load_parameters()

        # Operational state
        self.current_state: str = (
            STATE_FRONTIER_EXPLORATION if self.explorer_enabled else STATE_IDLE
        )
        self.target_locked: bool = False
        self.tracking_confidence: float = 0.0
        self.latest_frontal_distance: float = 10.0
        self.target_docked_latched: bool = False
        self.last_target_seen_time: float = -100.0
        self.terminal_stop_ticks: int = 0

        # Cached candidate command velocities
        self.latest_explorer_cmd = Twist()
        self.latest_pursuit_cmd = Twist()
        self.has_explorer_cmd: bool = False
        self.has_pursuit_cmd: bool = False

        self._init_interfaces()

        timer_period = 1.0 / max(1.0, self.supervision_rate_hz)
        self.eval_timer = self.create_timer(timer_period, self._on_supervision_cycle)

        self.get_logger().info(
            f"PursuitBTSupervisorNode initialized at {self.supervision_rate_hz:.1f} Hz. "
            f"Active state: {self.current_state}"
        )

    def _declare_parameters(self) -> None:
        """Declare all configurable parameters with default values."""
        self.declare_parameter("supervision_rate_hz", 20.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("explorer_cmd_topic", "/explorer/cmd_vel")
        self.declare_parameter("pursuit_cmd_topic", "/pursuit/cmd_vel_raw")
        self.declare_parameter("target_status_topic", "/perception/target_status")
        self.declare_parameter("confidence_topic", "/perception/tracking_confidence")
        self.declare_parameter("scan_topic", "/khepera4/scan")
        self.declare_parameter("stop_topic", "/target_found_and_stopped")
        self.declare_parameter("executive_status_topic", "/executive/status")
        self.declare_parameter("standoff_distance_m", 0.05)  # 5.0 cm surface stop
        self.declare_parameter("standoff_tolerance_m", 0.015)  # ±1.5 cm tolerance
        self.declare_parameter("target_lost_recovery_sec", 3.0)
        self.declare_parameter("min_tracking_confidence", 0.35)
        self.declare_parameter("auto_enable_pursuit", False)
        self.declare_parameter("auto_enable_explorer", True)
        self.declare_parameter("max_linear_speed", 0.12)
        self.declare_parameter("max_angular_speed", 1.00)

    def _load_parameters(self) -> None:
        """Read and store active parameter values."""
        self.supervision_rate_hz = float(self.get_parameter("supervision_rate_hz").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.explorer_cmd_topic = str(self.get_parameter("explorer_cmd_topic").value)
        self.pursuit_cmd_topic = str(self.get_parameter("pursuit_cmd_topic").value)
        self.target_status_topic = str(self.get_parameter("target_status_topic").value)
        self.confidence_topic = str(self.get_parameter("confidence_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.stop_topic = str(self.get_parameter("stop_topic").value)
        self.executive_status_topic = str(self.get_parameter("executive_status_topic").value)
        self.standoff_distance_m = float(self.get_parameter("standoff_distance_m").value)
        self.standoff_tolerance_m = float(self.get_parameter("standoff_tolerance_m").value)
        self.target_lost_recovery_sec = float(self.get_parameter("target_lost_recovery_sec").value)
        self.min_tracking_confidence = float(self.get_parameter("min_tracking_confidence").value)
        self.pursuit_enabled = bool(self.get_parameter("auto_enable_pursuit").value)
        self.explorer_enabled = bool(self.get_parameter("auto_enable_explorer").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

    def _init_interfaces(self) -> None:
        """Construct ROS 2 publisher and subscriber interfaces."""
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Authoritative motion publisher
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Supervisory state and telemetry publishers
        self.executive_status_pub = self.create_publisher(String, self.executive_status_topic, 10)
        self.stop_pub = self.create_publisher(Bool, self.stop_topic, 10)
        self.pursuit_enable_pub = self.create_publisher(Bool, "/pursuit/enable", 10)
        self.explorer_enable_pub = self.create_publisher(Bool, "/explorer/enable", qos_latched)

        # Subscriptions to candidate velocity generators
        self.explorer_cmd_sub = self.create_subscription(
            Twist, self.explorer_cmd_topic, self._on_explorer_cmd, 10
        )
        self.pursuit_cmd_sub = self.create_subscription(
            Twist, self.pursuit_cmd_topic, self._on_pursuit_cmd, 10
        )

        # Perception signals
        self.status_sub = self.create_subscription(
            Bool, self.target_status_topic, self._on_target_status, 10
        )
        self.conf_sub = self.create_subscription(
            Float32, self.confidence_topic, self._on_confidence, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._on_laser_scan, 10
        )

        # Coordination and control enable topics
        self.pursuit_enable_sub = self.create_subscription(
            Bool, "/pursuit/enable", self._on_pursuit_enable, 10
        )
        self.explorer_enable_sub = self.create_subscription(
            Bool, "/explorer/enable", self._on_explorer_enable, qos_latched
        )
        self.stop_sub = self.create_subscription(
            Bool, self.stop_topic, self._on_stop_signal, 10
        )
        self.speed_limits_sub = self.create_subscription(
            Twist, "/robot/speed_limits", self._on_speed_limits, 10
        )

    # --------------------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------------------

    def _on_explorer_cmd(self, msg: Twist) -> None:
        """Cache candidate velocity from exploration node."""
        self.latest_explorer_cmd = msg
        self.has_explorer_cmd = True

    def _on_pursuit_cmd(self, msg: Twist) -> None:
        """Cache candidate velocity from visual pursuit controller."""
        self.latest_pursuit_cmd = msg
        self.has_pursuit_cmd = True

    def _on_target_status(self, msg: Bool) -> None:
        """Update boolean target lock status from detector verifier."""
        self.target_locked = bool(msg.data)
        if self.target_locked:
            self.last_target_seen_time = self._get_current_time()

    def _on_confidence(self, msg: Float32) -> None:
        """Update tracking confidence metric from feature tracker."""
        self.tracking_confidence = float(msg.data)
        if self.tracking_confidence >= self.min_tracking_confidence:
            self.last_target_seen_time = self._get_current_time()

    def _on_laser_scan(self, msg: LaserScan) -> None:
        """Extract frontal sector proximity to enforce terminal surface standoff."""
        if not msg.ranges:
            return

        # Measure frontal ±15° arc (centered on bearing 0.0 rad)
        num_rays = len(msg.ranges)
        mid_idx = num_rays // 2
        span_idx = max(2, int(round((15.0 * math.pi / 180.0) / max(1e-4, msg.angle_increment))))

        start_idx = max(0, mid_idx - span_idx)
        end_idx = min(num_rays, mid_idx + span_idx + 1)
        frontal_ranges = [
            r for r in msg.ranges[start_idx:end_idx]
            if msg.range_min <= r <= msg.range_max and not math.isnan(r) and not math.isinf(r)
        ]

        if frontal_ranges:
            self.latest_frontal_distance = float(min(frontal_ranges))
        else:
            self.latest_frontal_distance = 10.0

    def _on_pursuit_enable(self, msg: Bool) -> None:
        """Handle external pursuit enable/disable toggles."""
        self.pursuit_enabled = bool(msg.data)
        if self.pursuit_enabled:
            self.target_docked_latched = False
            self.terminal_stop_ticks = 0
            self.last_target_seen_time = -100.0
            self.explorer_enabled = True
        else:
            if self.explorer_enabled:
                self.current_state = STATE_FRONTIER_EXPLORATION
            else:
                self.current_state = STATE_IDLE

    def _on_explorer_enable(self, msg: Bool) -> None:
        """Handle external exploration enable/disable toggles."""
        self.explorer_enabled = bool(msg.data)
        if not self.explorer_enabled and self.current_state == STATE_FRONTIER_EXPLORATION:
            self.current_state = STATE_IDLE

    def _on_stop_signal(self, msg: Bool) -> None:
        """Handle explicit target stop event notification."""
        if msg.data:
            self.target_docked_latched = True
            self.current_state = STATE_TERMINAL_STOP
        else:
            self.target_docked_latched = False
            self.terminal_stop_ticks = 0

    def _on_speed_limits(self, msg: Twist) -> None:
        """Update linear and angular speed ceilings from dashboard dials."""
        if msg.linear.x > 0.001:
            self.max_linear_speed = float(msg.linear.x)
        if msg.angular.z > 0.001:
            self.max_angular_speed = float(msg.angular.z)

    def _get_current_time(self) -> float:
        """Return monotonic ROS simulation or system time in seconds."""
        return float(self.get_clock().now().nanoseconds * 1e-9)

    # --------------------------------------------------------------------------
    # Behavior Tree Arbitration Logic
    # --------------------------------------------------------------------------

    def _on_supervision_cycle(self) -> None:
        """Execute one Behavior Tree evaluation tick and arbitrate cmd_vel."""
        now = self._get_current_time()
        is_target_active = bool(
            self.target_locked or self.tracking_confidence >= self.min_tracking_confidence
        )
        if is_target_active:
            self.last_target_seen_time = now

        selected_cmd = Twist()
        stop_cutoff = self.standoff_distance_m + self.standoff_tolerance_m

        # ----------------------------------------------------------------------
        # Branch 1: Terminal Standoff Stop (Highest Priority)
        # ----------------------------------------------------------------------
        if self.target_docked_latched:
            self.current_state = STATE_TERMINAL_STOP
            self._publish_command(selected_cmd)
            self._publish_telemetry(selected_cmd, is_target_active)
            return

        # Check for arrival at 5 cm surface standoff
        is_in_pursuit_mode = bool(
            self.pursuit_enabled
            and (
                self.current_state in (STATE_VISUAL_PURSUIT, STATE_TARGET_RECOVERY)
                or is_target_active
            )
        )
        if is_in_pursuit_mode and self.latest_frontal_distance <= stop_cutoff:
            self.terminal_stop_ticks += 1
            if self.terminal_stop_ticks >= 3:
                self.target_docked_latched = True
                self.current_state = STATE_TERMINAL_STOP
                self._halt_and_publish_stop()
                self._publish_telemetry(selected_cmd, is_target_active)
                return
            self.current_state = STATE_VISUAL_PURSUIT
            selected_cmd = Twist()
            self._publish_command(selected_cmd)
            self._publish_telemetry(selected_cmd, is_target_active)
            return
        else:
            self.terminal_stop_ticks = 0

        # ----------------------------------------------------------------------
        # Branch 2: Visual Pursuit (Target Confirmed & In View, Pursuit Enabled)
        # ----------------------------------------------------------------------
        if self.pursuit_enabled and is_target_active:
            self.current_state = STATE_VISUAL_PURSUIT
            selected_cmd = self.latest_pursuit_cmd
            self.explorer_enable_pub.publish(Bool(data=False))
            self._publish_command(selected_cmd)
            self._publish_telemetry(selected_cmd, is_target_active)
            return

        # ----------------------------------------------------------------------
        # Branch 3: Target Recovery (Target Lost Temporarily)
        # ----------------------------------------------------------------------
        time_since_seen = now - self.last_target_seen_time
        if self.pursuit_enabled and time_since_seen < self.target_lost_recovery_sec:
            self.current_state = STATE_TARGET_RECOVERY
            selected_cmd = self.latest_pursuit_cmd
            self._publish_command(selected_cmd)
            self._publish_telemetry(selected_cmd, is_target_active)
            return

        # ----------------------------------------------------------------------
        # Branch 4: Frontier Exploration (No Target Visible)
        # ----------------------------------------------------------------------
        if self.explorer_enabled or self.pursuit_enabled:
            self.current_state = STATE_FRONTIER_EXPLORATION
            selected_cmd = self.latest_explorer_cmd
            self._publish_command(selected_cmd)
            self._publish_telemetry(selected_cmd, is_target_active)
            return

        # ----------------------------------------------------------------------
        # Branch 5: Idle / Standby Fallback
        # ----------------------------------------------------------------------
        self.current_state = STATE_IDLE
        self._publish_command(selected_cmd)
        self._publish_telemetry(selected_cmd, is_target_active)

    def _publish_command(self, cmd: Twist) -> None:
        """Apply speed limits and publish authoritative velocity to cmd_vel."""
        safe_cmd = Twist()
        # Clamp linear velocity to speed limit
        lin_v = cmd.linear.x
        if lin_v > 0.0:
            safe_cmd.linear.x = min(lin_v, self.max_linear_speed)
        else:
            safe_cmd.linear.x = max(lin_v, -self.max_linear_speed)

        # Clamp angular velocity to speed limit
        ang_w = cmd.angular.z
        if ang_w > 0.0:
            safe_cmd.angular.z = min(ang_w, self.max_angular_speed)
        else:
            safe_cmd.angular.z = max(ang_w, -self.max_angular_speed)

        self.cmd_vel_pub.publish(safe_cmd)

    def _halt_and_publish_stop(self) -> None:
        """Command zero velocity, emit stop event, and disable explorer and pursuit."""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.stop_pub.publish(Bool(data=True))
        self.explorer_enable_pub.publish(Bool(data=False))
        self.pursuit_enable_pub.publish(Bool(data=False))

    def _publish_telemetry(self, active_cmd: Twist, target_active: bool) -> None:
        """Publish JSON telemetry string detailing current Behavior Tree state."""
        telemetry = {
            "state": self.current_state,
            "pursuit_enabled": self.pursuit_enabled,
            "explorer_enabled": self.explorer_enabled,
            "target_active": target_active,
            "tracking_confidence": round(self.tracking_confidence, 3),
            "frontal_distance_m": round(self.latest_frontal_distance, 3),
            "cmd_linear_x": round(active_cmd.linear.x, 3),
            "cmd_angular_z": round(active_cmd.angular.z, 3),
        }
        msg = String()
        msg.data = json.dumps(telemetry)
        self.executive_status_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    """Run PursuitBTSupervisorNode with standard lifecycle ordering."""
    rclpy.init(args=args)
    node = PursuitBTSupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
