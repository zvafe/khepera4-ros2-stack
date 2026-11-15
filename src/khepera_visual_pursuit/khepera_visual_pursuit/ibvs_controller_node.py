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

"""ROS 2 Node executing differential drive Image-Based Visual Servoing (IBVS)."""

from dataclasses import replace
import logging
import math
import sys
from typing import List, Optional, Tuple

import numpy as np

from khepera_visual_pursuit.ibvs_controller import (
    DEFAULT_CREEP_VELOCITY,
    DEFAULT_DESIRED_AREA_FRACTION,
    DEFAULT_K_OMEGA,
    DEFAULT_KV,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_OBSTACLE_STOP_DISTANCE,
    DEFAULT_OMEGA_MAX,
    DEFAULT_REVERSE_SWEEP_OMEGA,
    DEFAULT_V_MAX,
    IBVSController,
    IBVSGains,
)
from khepera_visual_pursuit.ros_compat import (
    Bool,
    ExternalShutdownException,
    Float32,
    LaserScan,
    Node,
    Parameter,
    PolygonStamped,
    SetParametersResult,
    Twist,
    rclpy,
)
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

logger = logging.getLogger(__name__)

DEFAULT_KEYPOINTS_TOPIC = "/perception/target_keypoints"
DEFAULT_CONFIDENCE_TOPIC = "/perception/tracking_confidence"
DEFAULT_SCAN_TOPIC = "/khepera4/scan"
DEFAULT_ULTRASONIC_TOPIC_PREFIX = "/khepera4/us_"
DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"
DEFAULT_STATUS_TOPIC = "/perception/ibvs_status"
DEFAULT_STOP_TOPIC = "/target_found_and_stopped"
DEFAULT_CONTROL_RATE_HZ: float = 25.0
DEFAULT_IMAGE_WIDTH: float = 640.0
DEFAULT_IMAGE_HEIGHT: float = 480.0
DEFAULT_USE_ULTRASONIC: bool = True
FRONTAL_SECTOR_RAD: float = math.pi / 6.0


class IBVSControllerNode(Node):
    """ROS 2 Node computing real-time Twist commands for visual pursuit."""

    def __init__(self) -> None:
        """Initialize subscriptions, publishers, gains, and control timer."""
        super().__init__("ibvs_controller_node")

        self._declare_and_load_parameters()

        self.controller = IBVSController(self.gains)
        self.latest_norm_x: float = 0.0
        self.latest_norm_y: float = 0.0
        self.latest_area_frac: float = 0.0
        self.latest_confidence: float = 0.0
        self.latest_obstacle_distance: float = 10.0
        self.latest_obstacle_left: float = 10.0
        self.latest_obstacle_right: float = 10.0
        self.latest_us_ranges: List[float] = [10.0] * 5
        self.has_active_target: bool = False
        self.target_reached_and_stopped: bool = False
        self.pursuit_enabled: bool = bool(self.get_parameter("auto_start").value)

        self._init_subscribers_and_publishers()

        timer_period = 1.0 / max(1.0, self.control_rate_hz)
        self.timer = self.create_timer(timer_period, self._on_control_cycle)

    def _declare_and_load_parameters(self) -> None:
        """Declare ROS 2 parameters and construct tuning gains."""
        self.declare_parameter("keypoints_topic", DEFAULT_KEYPOINTS_TOPIC)
        self.declare_parameter("confidence_topic", DEFAULT_CONFIDENCE_TOPIC)
        self.declare_parameter("scan_topic", DEFAULT_SCAN_TOPIC)
        self.declare_parameter("use_ultrasonic", DEFAULT_USE_ULTRASONIC)
        self.declare_parameter("ultrasonic_topic_prefix", DEFAULT_ULTRASONIC_TOPIC_PREFIX)
        self.declare_parameter("cmd_vel_topic", DEFAULT_CMD_VEL_TOPIC)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("cooperate_with_explorer", False)
        self.declare_parameter("control_rate_hz", DEFAULT_CONTROL_RATE_HZ)
        self.declare_parameter("k_v", DEFAULT_KV)
        self.declare_parameter("k_omega", DEFAULT_K_OMEGA)
        self.declare_parameter("v_max", DEFAULT_V_MAX)
        self.declare_parameter("omega_max", DEFAULT_OMEGA_MAX)
        self.declare_parameter("desired_area_fraction", DEFAULT_DESIRED_AREA_FRACTION)
        self.declare_parameter("min_confidence", DEFAULT_MIN_CONFIDENCE)
        self.declare_parameter("reverse_sweep_omega", DEFAULT_REVERSE_SWEEP_OMEGA)
        self.declare_parameter("obstacle_stop_distance", DEFAULT_OBSTACLE_STOP_DISTANCE)
        self.declare_parameter("creep_velocity", DEFAULT_CREEP_VELOCITY)
        self.declare_parameter("image_width", DEFAULT_IMAGE_WIDTH)
        self.declare_parameter("image_height", DEFAULT_IMAGE_HEIGHT)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.img_width = float(self.get_parameter("image_width").value)
        self.img_height = float(self.get_parameter("image_height").value)
        self.cooperate_with_explorer = bool(self.get_parameter("cooperate_with_explorer").value)

        self.gains = IBVSGains(
            k_v=float(self.get_parameter("k_v").value),
            k_omega=float(self.get_parameter("k_omega").value),
            v_max=float(self.get_parameter("v_max").value),
            omega_max=float(self.get_parameter("omega_max").value),
            desired_area_fraction=float(self.get_parameter("desired_area_fraction").value),
            min_confidence=float(self.get_parameter("min_confidence").value),
            reverse_sweep_omega=float(self.get_parameter("reverse_sweep_omega").value),
            obstacle_stop_distance=float(self.get_parameter("obstacle_stop_distance").value),
            creep_velocity=float(self.get_parameter("creep_velocity").value),
        )

    def _init_subscribers_and_publishers(self) -> None:
        """Set up ROS 2 subscriber and publisher interfaces."""
        kp_topic = str(self.get_parameter("keypoints_topic").value)
        conf_topic = str(self.get_parameter("confidence_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.keypoints_sub = self.create_subscription(
            PolygonStamped, kp_topic, self._on_target_keypoints, 10
        )
        self.confidence_sub = self.create_subscription(
            Float32, conf_topic, self._on_tracking_confidence, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._on_laser_scan, 10
        )
        self.us_subs: List[object] = []
        if bool(self.get_parameter("use_ultrasonic").value):
            us_prefix = str(self.get_parameter("ultrasonic_topic_prefix").value)
            for i in range(5):
                topic = f"{us_prefix}{i}/scan"
                sub = self.create_subscription(
                    LaserScan,
                    topic,
                    lambda msg, idx=i: self._on_ultrasonic_scan(idx, msg),
                    10,
                )
                self.us_subs.append(sub)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        raw_cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.raw_pursuit_pub = self.create_publisher(Twist, "/pursuit/cmd_vel_raw", 10)

        class _CmdPublisherProxy:
            """Proxy forwarding Twist messages to primary topic and /pursuit/cmd_vel_raw."""

            def __init__(self, primary, secondary):
                self._primary = primary
                self._secondary = secondary

            def publish(self, msg):
                self._primary.publish(msg)
                if self._secondary is not None:
                    self._secondary.publish(msg)

            def __getattr__(self, item):
                return getattr(self._primary, item)

        self.cmd_vel_pub = _CmdPublisherProxy(
            raw_cmd_pub,
            self.raw_pursuit_pub if cmd_topic != "/pursuit/cmd_vel_raw" else None,
        )
        self.status_pub = self.create_publisher(Bool, DEFAULT_STATUS_TOPIC, 10)
        self.stop_pub = self.create_publisher(Bool, DEFAULT_STOP_TOPIC, 10)
        self.pursuit_enable_pub = self.create_publisher(Bool, "/pursuit/enable", 10)
        self.explorer_enable_pub = self.create_publisher(
            Bool, "/explorer/enable", qos_latched
        )
        self.enable_sub = self.create_subscription(
            Bool, "/pursuit/enable", self._on_pursuit_enable, 10
        )
        self.stop_sub = self.create_subscription(
            Bool, DEFAULT_STOP_TOPIC, self._on_stop_signal_message, 10
        )
        self.pursuit_speed_sub = self.create_subscription(
            Twist, "/pursuit/set_speed_limits", self._on_speed_limits, 10
        )
        self.robot_speed_sub = self.create_subscription(
            Twist, "/robot/speed_limits", self._on_speed_limits, 10
        )
        self.explorer_speed_sub = self.create_subscription(
            Twist, "/explorer/set_speed_limits", self._on_speed_limits, 10
        )
        if hasattr(self, "add_on_set_parameters_callback"):
            self.add_on_set_parameters_callback(self._on_set_parameters)

    def _on_speed_limits(self, msg: Twist) -> None:
        """Update IBVS maximum linear and angular speed limits from speed dial or topic."""
        changes = {}
        if msg.linear.x > 0.001:
            changes["v_max"] = float(msg.linear.x)
        if msg.angular.z > 0.001:
            new_omega_max = float(msg.angular.z)
            changes["omega_max"] = new_omega_max
            if self.gains.reverse_sweep_omega > new_omega_max:
                changes["reverse_sweep_omega"] = new_omega_max

        if changes:
            self.gains = replace(self.gains, **changes)
            if hasattr(self, "controller") and self.controller is not None:
                self.controller.gains = self.gains
            self.get_logger().info(
                f"IBVS speed limits updated: v_max={self.gains.v_max:.2f} m/s, "
                f"omega_max={self.gains.omega_max:.2f} rad/s"
            )

    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        """Handle dynamic parameter reconfigurations for visual pursuit speed limits."""
        changes = {}
        for param in params:
            if param.name in ("v_max", "linear_speed"):
                val = float(param.value)
                if val > 0.001:
                    changes["v_max"] = val
            elif param.name in ("omega_max", "angular_speed", "turn_speed"):
                val = float(param.value)
                if val > 0.001:
                    changes["omega_max"] = val
        if changes:
            self.gains = replace(self.gains, **changes)
            if hasattr(self, "controller") and self.controller is not None:
                self.controller.gains = self.gains
        return SetParametersResult(successful=True)

    def _on_stop_signal_message(self, msg: Bool) -> None:
        """Reset terminal stop state when False is received on stop topic."""
        if not msg.data:
            self.target_reached_and_stopped = False
            self.pursuit_enabled = True

    def _on_pursuit_enable(self, msg: Bool) -> None:
        """Dynamically toggle pursuit control loop execution."""
        self.pursuit_enabled = bool(msg.data)
        self.get_logger().info(f"Pursuit enable set to: {self.pursuit_enabled}")
        if self.pursuit_enabled:
            self.target_reached_and_stopped = False
        else:
            self.has_active_target = False
            self.controller.reset()
            self.cmd_vel_pub.publish(Twist())
            self.status_pub.publish(Bool(data=False))

    def _on_target_keypoints(self, msg: PolygonStamped) -> None:
        """Extract normalized centroid and bounding box area from PolygonStamped."""
        pts = msg.polygon.points
        if not pts:
            self.has_active_target = False
            return

        norm_x, norm_y, area_frac = self._calculate_centroid_and_area(pts)
        self.latest_norm_x = norm_x
        self.latest_norm_y = norm_y
        self.latest_area_frac = area_frac
        was_active = self.has_active_target
        self.has_active_target = True
        if not was_active and self.cooperate_with_explorer:
            self.explorer_enable_pub.publish(Bool(data=False))

    def _calculate_centroid_and_area(
        self,
        points: List[object],
    ) -> Tuple[float, float, float]:
        """Compute normalized coordinates and area fraction from polygon vertices."""
        x_coords = [float(p.x) for p in points]
        y_coords = [float(p.y) for p in points]

        mean_x = float(np.mean(x_coords))
        mean_y = float(np.mean(y_coords))

        # Check if already normalized in [-1, 1] or in pixel space
        if max(x_coords) <= 1.0 and min(x_coords) >= -1.0:
            norm_x = float(np.clip(mean_x, -1.0, 1.0))
            norm_y = float(np.clip(mean_y, -1.0, 1.0))
            width_span = float(max(x_coords) - min(x_coords))
            height_span = float(max(y_coords) - min(y_coords))
            area_frac = float(np.clip((width_span * height_span) / 4.0, 0.0, 1.0))
        else:
            half_w = self.img_width / 2.0
            half_h = self.img_height / 2.0
            norm_x = float(np.clip((mean_x - half_w) / half_w, -1.0, 1.0))
            norm_y = float(np.clip((mean_y - half_h) / half_h, -1.0, 1.0))
            box_w = float(max(x_coords) - min(x_coords))
            box_h = float(max(y_coords) - min(y_coords))
            total_area = self.img_width * self.img_height
            area_frac = float(np.clip((box_w * box_h) / total_area, 0.0, 1.0))

        return norm_x, norm_y, area_frac

    def _on_tracking_confidence(self, msg: Float32) -> None:
        """Update latest tracking confidence score."""
        self.latest_confidence = float(msg.data)
        if self.latest_confidence < 0.05:
            self.has_active_target = False

    def _on_ultrasonic_scan(self, index: int, msg: LaserScan) -> None:
        """Process range scan from individual ultrasonic proximity sensor."""
        if not msg.ranges:
            return
        valid_ranges = [
            r for r in msg.ranges
            if msg.range_min <= r <= msg.range_max and not math.isnan(r) and not math.isinf(r)
        ]
        if valid_ranges:
            self.latest_us_ranges[index] = float(min(valid_ranges))

    def _on_laser_scan(self, msg: LaserScan) -> None:
        """Extract minimum range from frontal and lateral sectors for obstacle avoidance."""
        if not msg.ranges:
            self.latest_obstacle_distance = 10.0
            self.latest_obstacle_left = 10.0
            self.latest_obstacle_right = 10.0
            return

        min_frontal_dist = 10.0
        min_left_dist = 10.0
        min_right_dist = 10.0
        angle = msg.angle_min
        is_body_frame = msg.header.frame_id in ("base_footprint", "base_link")
        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max:
                eff_r = (r - 0.07) if (is_body_frame and r > 0.07) else r
                if -FRONTAL_SECTOR_RAD <= angle <= FRONTAL_SECTOR_RAD:
                    if eff_r < min_frontal_dist:
                        min_frontal_dist = eff_r
                elif FRONTAL_SECTOR_RAD < angle <= math.pi / 2.0:
                    if eff_r < min_left_dist:
                        min_left_dist = eff_r
                elif -math.pi / 2.0 <= angle < -FRONTAL_SECTOR_RAD:
                    if eff_r < min_right_dist:
                        min_right_dist = eff_r
            angle += msg.angle_increment

        self.latest_obstacle_distance = float(min_frontal_dist)
        self.latest_obstacle_left = float(min_left_dist)
        self.latest_obstacle_right = float(min_right_dist)

    def _on_control_cycle(self) -> None:
        """Generate kinematic command and publish to cmd_vel at control loop rate."""
        if self.target_reached_and_stopped:
            return

        if not self.pursuit_enabled:
            return

        if self.cooperate_with_explorer and not self.has_active_target:
            return

        # Fuse frontal and lateral obstacle distances from laser scan and ultrasonic sensors
        us_frontal = min(
            self.latest_us_ranges[1],
            self.latest_us_ranges[2],
            self.latest_us_ranges[3],
        )
        us_left = min(self.latest_us_ranges[0], self.latest_us_ranges[1])
        us_right = min(self.latest_us_ranges[3], self.latest_us_ranges[4])

        effective_dist = min(self.latest_obstacle_distance, us_frontal)
        effective_left = min(self.latest_obstacle_left, us_left)
        effective_right = min(self.latest_obstacle_right, us_right)

        target_state = (
            self.latest_norm_x,
            self.latest_area_frac,
            self.latest_confidence if self.has_active_target else 0.0,
        )

        cmd = self.controller.compute_control(
            target_state,
            effective_dist,
            effective_left,
            effective_right,
        )

        if cmd.is_emergency_stop:
            self.target_reached_and_stopped = True
            self.pursuit_enabled = False
            self.has_active_target = False
            self.controller.reset()
            self.stop_pub.publish(Bool(data=True))
            self.pursuit_enable_pub.publish(Bool(data=False))
            self.explorer_enable_pub.publish(Bool(data=False))
            self.cmd_vel_pub.publish(Twist())
            self.status_pub.publish(Bool(data=False))
            self.get_logger().info(
                f"🎯 [TERMINAL STOP] Target reached! Standoff: {effective_dist*100:.1f} cm"
            )
            return

        twist_msg = Twist()
        twist_msg.linear.x = float(cmd.linear_x)
        twist_msg.angular.z = float(cmd.angular_z)
        self.cmd_vel_pub.publish(twist_msg)

        is_active = bool(not cmd.is_emergency_stop and not cmd.is_recovery_sweep)
        self.status_pub.publish(Bool(data=is_active and self.has_active_target))


def main(args: Optional[List[str]] = None) -> None:
    """Run IBVS controller node with standard lifecycle ordering."""
    rclpy.init(args=args)
    node = IBVSControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main(sys.argv)
