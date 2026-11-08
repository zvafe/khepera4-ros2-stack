"""ROS 2 node for OpenFrontier visual-language grounded frontier exploration."""

import math
import sys
from typing import List, Optional
from geometry_msgs.msg import Point, PoseStamped, Twist
from hybrid_nav_interfaces.msg import Frontier, FrontierArray
from hybrid_nav_interfaces.srv import EvaluateFrontiers, VerifyTarget
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from openfrontier_ros.frontier_detector import RawFrontier2D, VisualFrontierDetector
from openfrontier_ros.frontier_manager import GlobalFrontierManager, ManagedFrontier
from openfrontier_ros.image_utils import cv2_to_ros_image, ros_image_to_cv2


class FrontierManagerNode(Node):
    """Manages visual frontier extraction, VLM scoring, and goal selection."""

    def __init__(self) -> None:
        """Initialize subscriptions, publishers, clients, and execution timer."""
        super().__init__("frontier_manager_node")

        self.declare_parameter("target_description", "fire extinguisher")
        self.declare_parameter("evaluation_interval_sec", 2.0)
        self.declare_parameter("tau_merge_m", 1.8)
        self.declare_parameter("r_clear_m", 0.5)
        self.declare_parameter("r_near_m", 0.5)
        self.declare_parameter("r_goal_m", 0.8)
        self.declare_parameter("linear_speed_mps", 0.15)
        self.declare_parameter("yaw_gain", 1.2)

        self._target_desc = str(self.get_parameter("target_description").value)
        self._eval_interval = float(self.get_parameter("evaluation_interval_sec").value)
        self._tau_merge = float(self.get_parameter("tau_merge_m").value)
        self._r_clear = float(self.get_parameter("r_clear_m").value)
        self._r_near = float(self.get_parameter("r_near_m").value)
        self._r_goal = float(self.get_parameter("r_goal_m").value)
        self._linear_speed = float(self.get_parameter("linear_speed_mps").value)
        self._yaw_gain = float(self.get_parameter("yaw_gain").value)

        self._detector = VisualFrontierDetector()
        self._manager = GlobalFrontierManager(
            tau_merge_m=self._tau_merge,
            r_clear_m=self._r_clear,
            r_near_m=self._r_near,
            r_goal_m=self._r_goal
        )

        # State tracking
        self._current_image: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._current_target: Optional[ManagedFrontier] = None
        self._is_evaluating = False

        # Subscriptions
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._handle_rgb_image,
            10
        )
        self._depth_sub = self.create_subscription(
            Image,
            "/camera/depth",
            self._handle_depth_image,
            10
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._handle_odometry,
            20
        )

        # Service client for VLM
        self._eval_client = self.create_client(
            EvaluateFrontiers,
            "/semantic_reasoner/evaluate_frontiers"
        )
        self._verify_client = self.create_client(
            VerifyTarget,
            "/semantic_reasoner/verify_target"
        )

        # Publishers
        self._frontiers_pub = self.create_publisher(
            FrontierArray,
            "/openfrontier/frontiers",
            10
        )
        self._markers_pub = self.create_publisher(
            MarkerArray,
            "/openfrontier/markers",
            10
        )
        self._goal_pub = self.create_publisher(
            PoseStamped,
            "/openfrontier/goal_pose",
            10
        )
        self._cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # Cyclic exploration & control timer (10 Hz)
        self._control_timer = self.create_timer(0.1, self._handle_control_cycle)
        # Slower VLM frontier reasoning timer
        self._reasoning_timer = self.create_timer(self._eval_interval, self._trigger_vlm_evaluation)

        self.get_logger().info(
            f"FrontierManagerNode initialized. Searching for '{self._target_desc}'."
        )

    def _handle_rgb_image(self, msg: Image) -> None:
        """Cache incoming camera RGB observation."""
        try:
            self._current_image = ros_image_to_cv2(msg)
        except Exception:
            pass

    def _handle_depth_image(self, msg: Image) -> None:
        """Cache incoming camera depth observation."""
        try:
            self._current_depth = ros_image_to_cv2(msg)
        except Exception:
            pass

    def _handle_odometry(self, msg: Odometry) -> None:
        """Update robot global 2D pose and heading."""
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _trigger_vlm_evaluation(self) -> None:
        """Extract candidate frontiers and asynchronously query VLM."""
        if self._current_image is None or self._is_evaluating:
            return
        if not self._eval_client.service_is_ready():
            return

        raw_candidates: List[RawFrontier2D] = self._detector.detect_visual_frontiers(
            self._current_image,
            self._current_depth
        )
        if not raw_candidates:
            return

        proposals: List[ManagedFrontier] = []
        req_candidates: List[Frontier] = []

        for cand in raw_candidates:
            x_cam, _, z_cam = self._detector.backproject_to_camera_frame(
                cand.pixel_u,
                cand.pixel_v,
                cand.estimated_depth_m
            )
            world_x = self._robot_x + (z_cam * math.cos(self._robot_yaw) - x_cam * math.sin(self._robot_yaw))
            world_y = self._robot_y + (z_cam * math.sin(self._robot_yaw) + x_cam * math.cos(self._robot_yaw))

            fid = self._manager.generate_next_id()
            managed = ManagedFrontier(
                id=fid,
                pos_x=world_x,
                pos_y=world_y,
                pos_z=0.0,
                pixel_u=cand.pixel_u,
                pixel_v=cand.pixel_v,
                exploration_gain=cand.exploration_gain
            )
            proposals.append(managed)

            f_msg = Frontier()
            f_msg.id = fid
            f_msg.pixel_u = cand.pixel_u
            f_msg.pixel_v = cand.pixel_v
            f_msg.pose_3d = Point(x=world_x, y=world_y, z=0.0)
            f_msg.exploration_gain = cand.exploration_gain
            req_candidates.append(f_msg)

        # Merge proposals into manager geometry
        self._manager.merge_proposals(proposals)

        # Prepare service call
        req = EvaluateFrontiers.Request()
        req.image = cv2_to_ros_image(self._current_image, encoding="bgr8")
        req.candidates = req_candidates
        req.target_description = self._target_desc

        self._is_evaluating = True
        future = self._eval_client.call_async(req)
        future.add_done_callback(self._on_vlm_evaluation_done)

    def _on_vlm_evaluation_done(self, future: rclpy.task.Future) -> None:
        """Handle scored frontiers returned from VLM reasoning engine."""
        self._is_evaluating = False
        try:
            response = future.result()
            if not response.success:
                return

            score_map = {f.id: f.semantic_prob for f in response.scored_frontiers}
            for active_f in self._manager.get_active_frontiers():
                if active_f.id in score_map:
                    active_f.semantic_prob = score_map[active_f.id]

            self.get_logger().info(
                f"Updated {len(response.scored_frontiers)} frontiers with VLM priors."
            )
        except Exception as exc:
            self.get_logger().warn(f"Frontier evaluation call failed: {exc}")

    def _handle_control_cycle(self) -> None:
        """10 Hz execution cycle: prune frontiers, pick best target, issue motion."""
        self._manager.prune_cleared_frontiers(self._robot_x, self._robot_y)
        best = self._manager.select_best_frontier(self._robot_x, self._robot_y)
        self._current_target = best

        self._publish_frontiers_and_markers()

        if best is None:
            # Idle or emergency in-place turn
            return

        dx = best.pos_x - self._robot_x
        dy = best.pos_y - self._robot_y
        dist = math.hypot(dx, dy)

        # Check stall
        if self._manager.handle_progress(dist):
            self.get_logger().warn(f"Stall detected on frontier '{best.id}'. Dropping.")
            self._manager.drop_frontier(best.id)
            return

        # Low-level pure pursuit guidance
        target_heading = math.atan2(dy, dx)
        yaw_err = target_heading - self._robot_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        cmd = Twist()
        if abs(yaw_err) > 0.4:
            cmd.angular.z = max(-1.0, min(1.0, self._yaw_gain * yaw_err))
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = max(-0.8, min(0.8, self._yaw_gain * yaw_err))
            cmd.linear.x = min(self._linear_speed, max(0.05, 0.2 * dist))

        self._cmd_pub.publish(cmd)

    def _publish_frontiers_and_markers(self) -> None:
        """Publish ROS 2 FrontierArray and visualization markers."""
        active = self._manager.get_active_frontiers()
        array_msg = FrontierArray()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = "odom"
        array_msg.selected_frontier_id = self._current_target.id if self._current_target else ""

        marker_array = MarkerArray()
        for idx, f in enumerate(active):
            f_msg = Frontier()
            f_msg.id = f.id
            f_msg.pose_3d = Point(x=f.pos_x, y=f.pos_y, z=f.pos_z)
            f_msg.pixel_u = f.pixel_u
            f_msg.pixel_v = f.pixel_v
            f_msg.exploration_gain = f.exploration_gain
            f_msg.semantic_prob = f.semantic_prob
            f_msg.visual_match_score = f.visual_match_score
            f_msg.total_utility = f.total_utility
            f_msg.is_viewpoint_frontier = f.is_viewpoint_frontier
            array_msg.frontiers.append(f_msg)

            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = array_msg.header.stamp
            m.ns = "openfrontiers"
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = f.pos_x
            m.pose.position.y = f.pos_y
            m.pose.position.z = 0.1
            m.scale.x = 0.35
            m.scale.y = 0.35
            m.scale.z = 0.05
            # Color by semantic probability (Green = high, Yellow = mid, Blue = low)
            m.color.r = float(1.0 - f.semantic_prob)
            m.color.g = float(f.semantic_prob)
            m.color.b = 0.2
            m.color.a = 0.85
            marker_array.markers.append(m)

        self._frontiers_pub.publish(array_msg)
        self._markers_pub.publish(marker_array)


def main(args: list = None) -> None:
    """Standard ROS 2 entrypoint lifecycle."""
    rclpy.init(args=args)
    node = FrontierManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
