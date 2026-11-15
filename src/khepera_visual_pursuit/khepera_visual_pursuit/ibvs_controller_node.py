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

import logging
import math
import sys
from typing import List, Optional, Tuple

import numpy as np

from khepera_visual_pursuit.ibvs_controller import (
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
    PolygonStamped,
    Twist,
    rclpy,
)

logger = logging.getLogger(__name__)

DEFAULT_KEYPOINTS_TOPIC = "/perception/target_keypoints"
DEFAULT_CONFIDENCE_TOPIC = "/perception/tracking_confidence"
DEFAULT_SCAN_TOPIC = "/khepera4/scan"
DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"
DEFAULT_STATUS_TOPIC = "/perception/ibvs_status"
DEFAULT_CONTROL_RATE_HZ: float = 25.0
DEFAULT_IMAGE_WIDTH: float = 640.0
DEFAULT_IMAGE_HEIGHT: float = 480.0
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
        self.has_active_target: bool = False
        self.pursuit_enabled: bool = bool(self.get_parameter("auto_start").value)

        self._init_subscribers_and_publishers()

        timer_period = 1.0 / max(1.0, self.control_rate_hz)
        self.timer = self.create_timer(timer_period, self._on_control_cycle)

    def _declare_and_load_parameters(self) -> None:
        """Declare ROS 2 parameters and construct tuning gains."""
        self.declare_parameter("keypoints_topic", DEFAULT_KEYPOINTS_TOPIC)
        self.declare_parameter("confidence_topic", DEFAULT_CONFIDENCE_TOPIC)
        self.declare_parameter("scan_topic", DEFAULT_SCAN_TOPIC)
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

        self.cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.status_pub = self.create_publisher(Bool, DEFAULT_STATUS_TOPIC, 10)
        self.enable_sub = self.create_subscription(
            Bool, "/pursuit/enable", self._on_pursuit_enable, 10
        )

    def _on_pursuit_enable(self, msg: Bool) -> None:
        """Dynamically toggle pursuit control loop execution."""
        self.pursuit_enabled = bool(msg.data)
        if not self.pursuit_enabled:
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
        self.has_active_target = True

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
        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max:
                eff_r = (r - 0.07) if r > 0.07 else r
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
        if not self.pursuit_enabled:
            return

        if self.cooperate_with_explorer and not self.has_active_target:
            return

        target_state = (
            self.latest_norm_x,
            self.latest_area_frac,
            self.latest_confidence if self.has_active_target else 0.0,
        )

        cmd = self.controller.compute_control(
            target_state,
            self.latest_obstacle_distance,
            self.latest_obstacle_left,
            self.latest_obstacle_right,
        )

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
