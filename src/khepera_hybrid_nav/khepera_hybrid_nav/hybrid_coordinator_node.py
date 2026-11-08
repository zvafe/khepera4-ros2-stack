"""Master coordinator for hybrid OpenFrontier and SplatSearch navigation."""

import math
import sys
from typing import Dict, List, Optional
from geometry_msgs.msg import Point, PoseStamped
from hybrid_nav_interfaces.msg import Frontier, FrontierArray
from hybrid_nav_interfaces.srv import MatchCandidate3DGS
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String


class HybridCoordinatorNode(Node):
    """Fuses OpenFrontier semantic scores with SplatSearch 3DGS visual correspondence."""

    def __init__(self) -> None:
        """Initialize parameters, publishers, subscribers, and clients."""
        super().__init__("hybrid_coordinator_node")

        self.declare_parameter("enable_3dgs", False)
        self.declare_parameter("weight_distance", 0.25)
        self.declare_parameter("weight_semantic", 0.45)
        self.declare_parameter("weight_visual_3dgs", 0.30)
        self.declare_parameter("target_description", "fire extinguisher")

        self._enable_3dgs = bool(self.get_parameter("enable_3dgs").value)
        self._w_dist = float(self.get_parameter("weight_distance").value)
        self._w_sem = float(self.get_parameter("weight_semantic").value)
        self._w_vis = float(self.get_parameter("weight_visual_3dgs").value)
        self._target_desc = str(self.get_parameter("target_description").value)

        self._robot_x = 0.0
        self._robot_y = 0.0
        self._latest_frontiers: List[Frontier] = []

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self._handle_parameter_change)

        # Subscriptions
        self._frontiers_sub = self.create_subscription(
            FrontierArray,
            "/openfrontier/frontiers",
            self._handle_frontiers,
            10
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._handle_odometry,
            20
        )

        # Publishers
        self._goal_pub = self.create_publisher(
            PoseStamped,
            "/hybrid_nav/selected_goal",
            10
        )
        self._status_pub = self.create_publisher(
            String,
            "/hybrid_nav/status",
            10
        )

        # Cyclic coordinator timer (5 Hz)
        self._timer = self.create_timer(0.2, self._handle_coordination_cycle)

        self.get_logger().info(
            f"HybridCoordinatorNode initialized. 3DGS toggle: {self._enable_3dgs}."
        )

    def _handle_parameter_change(self, params: List[Parameter]) -> SetParametersResult:
        """Handle dynamic parameter updates at runtime."""
        for param in params:
            if param.name == "enable_3dgs":
                self._enable_3dgs = bool(param.value)
                self.get_logger().info(
                    f"Coordinator: enable_3dgs dynamically toggled to {self._enable_3dgs}."
                )
        return SetParametersResult(successful=True)

    def _handle_odometry(self, msg: Odometry) -> None:
        """Update robot position."""
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _handle_frontiers(self, msg: FrontierArray) -> None:
        """Cache incoming frontiers from OpenFrontier manager."""
        self._latest_frontiers = list(msg.frontiers)

    def _handle_coordination_cycle(self) -> None:
        """Evaluate unified utility across frontiers and broadcast best target."""
        if not self._latest_frontiers:
            return

        best_frontier: Optional[Frontier] = None
        best_utility = -float("inf")

        for f in self._latest_frontiers:
            dist = math.hypot(self._robot_x - f.pose_3d.x, self._robot_y - f.pose_3d.y)
            d_norm = 1.0 / (1.0 + dist)
            sem_term = f.semantic_prob * f.exploration_gain

            # Visual term only added if 3DGS is enabled
            vis_term = f.visual_match_score if self._enable_3dgs else 0.0

            total_utility = (
                self._w_dist * d_norm +
                self._w_sem * sem_term +
                (self._w_vis * vis_term if self._enable_3dgs else 0.0)
            )

            if f.is_viewpoint_frontier:
                total_utility += 1000.0  # Priority gate for direct candidate observation

            if total_utility > best_utility:
                best_utility = total_utility
                best_frontier = f

        if best_frontier is not None:
            self._publish_selected_goal(best_frontier, best_utility)

    def _publish_selected_goal(self, frontier: Frontier, utility: float) -> None:
        """Publish the selected 3D goal pose and status telemetry."""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "odom"
        goal_msg.pose.position.x = frontier.pose_3d.x
        goal_msg.pose.position.y = frontier.pose_3d.y
        goal_msg.pose.position.z = frontier.pose_3d.z
        goal_msg.pose.orientation.w = 1.0
        self._goal_pub.publish(goal_msg)

        status_msg = String()
        mode_str = "SplatSearch_3DGS" if self._enable_3dgs else "OpenFrontier_ZeroShot"
        status_msg.data = (
            f"Mode: {mode_str} | Best Frontier: {frontier.id} | "
            f"Utility: {utility:.2f} | Pos: ({frontier.pose_3d.x:.2f}, {frontier.pose_3d.y:.2f})"
        )
        self._status_pub.publish(status_msg)


def main(args: list = None) -> None:
    """Standard ROS 2 entrypoint lifecycle."""
    rclpy.init(args=args)
    node = HybridCoordinatorNode()
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
