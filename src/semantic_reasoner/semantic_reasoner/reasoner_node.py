"""ROS 2 node serving VLM semantic reasoning for visual frontier navigation."""

import sys
from typing import List
from geometry_msgs.msg import Point
from hybrid_nav_interfaces.msg import Frontier
from hybrid_nav_interfaces.srv import EvaluateFrontiers, VerifyTarget
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from semantic_reasoner.base_reasoner import (
    BaseSemanticReasoner,
    FrontierCandidate,
    FrontierScore,
    TargetVerification,
)
from semantic_reasoner.cloud_gemini_provider import CloudGeminiProvider
from semantic_reasoner.image_utils import cv2_to_ros_image, ros_image_to_cv2
from semantic_reasoner.local_vlm_provider import LocalVlmProvider
from semantic_reasoner.som_annotator import SetOfMarksAnnotator


class SemanticReasonerNode(Node):
    """Coordinates Set-of-Marks visual annotation and VLM reasoning services."""

    def __init__(self) -> None:
        """Initialize parameters, reasoner backends, and ROS 2 service servers."""
        super().__init__("semantic_reasoner_node")

        self.declare_parameter("mode", "hybrid")
        self.declare_parameter("cloud_model", "gemini-2.5-flash")
        self.declare_parameter("local_endpoint", "http://127.0.0.1:11434/v1/chat/completions")
        self.declare_parameter("local_model", "gemma4:e2b")
        self.declare_parameter("timeout_sec", 8.0)

        self._mode = str(self.get_parameter("mode").value)
        self._cloud_model = str(self.get_parameter("cloud_model").value)
        self._local_endpoint = str(self.get_parameter("local_endpoint").value)
        self._local_model = str(self.get_parameter("local_model").value)
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)

        self._annotator = SetOfMarksAnnotator()

        self._cloud_provider: Optional[BaseSemanticReasoner] = None
        self._local_provider: Optional[BaseSemanticReasoner] = None
        self._init_providers()

        # Service servers
        self._eval_srv = self.create_service(
            EvaluateFrontiers,
            "/semantic_reasoner/evaluate_frontiers",
            self._handle_evaluate_frontiers
        )
        self._verify_srv = self.create_service(
            VerifyTarget,
            "/semantic_reasoner/verify_target",
            self._handle_verify_target
        )

        # Publisher for visual inspection
        self._som_pub = self.create_publisher(
            Image,
            "/semantic_reasoner/annotated_som_image",
            10
        )

        self.get_logger().info(
            f"Semantic Reasoner Node initialized in mode '{self._mode}'."
        )

    def _init_providers(self) -> None:
        """Instantiate configured VLM providers."""
        try:
            self._cloud_provider = CloudGeminiProvider(model_name=self._cloud_model)
            self.get_logger().info(f"Cloud Gemini provider ready ({self._cloud_model}).")
        except Exception as exc:
            self.get_logger().warn(f"Cloud Gemini init skipped or failed: {exc}")

        self._local_provider = LocalVlmProvider(
            endpoint_url=self._local_endpoint,
            model_name=self._local_model,
            timeout_sec=self._timeout_sec
        )

    def _get_active_provider(self) -> BaseSemanticReasoner:
        """Select provider according to operational mode and availability."""
        if self._mode == "cloud" and self._cloud_provider:
            return self._cloud_provider
        if self._mode == "local":
            return self._local_provider
        if self._mode == "hybrid":
            if self._cloud_provider:
                status = self._cloud_provider.get_health_status()
                if status.get("available") == "True":
                    return self._cloud_provider
            return self._local_provider
        return self._local_provider

    def _handle_evaluate_frontiers(
        self,
        request: EvaluateFrontiers.Request,
        response: EvaluateFrontiers.Response
    ) -> EvaluateFrontiers.Response:
        """Process Set-of-Marks image annotation and obtain frontier scores."""
        try:
            image_bgr = ros_image_to_cv2(request.image)
        except Exception as exc:
            self.get_logger().error(f"Image conversion error: {exc}")
            response.success = False
            response.reasoning = f"Image conversion error: {exc}"
            return response

        # Build candidates and markers
        candidates: List[FrontierCandidate] = []
        markers = []
        for frontier in request.candidates:
            candidates.append(
                FrontierCandidate(
                    id=frontier.id,
                    pixel_coords=(frontier.pixel_u, frontier.pixel_v),
                    world_coords=(
                        frontier.pose_3d.x,
                        frontier.pose_3d.y,
                        frontier.pose_3d.z
                    ),
                    exploration_gain=float(frontier.exploration_gain)
                )
            )
            markers.append((frontier.id, (frontier.pixel_u, frontier.pixel_v)))

        annotated = self._annotator.annotate(image_bgr, markers)

        # Publish debug visualization
        try:
            som_msg = cv2_to_ros_image(annotated, encoding="bgr8")
            som_msg.header = request.image.header
            self._som_pub.publish(som_msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish annotated image: {exc}")

        provider = self._get_active_provider()
        scores: List[FrontierScore] = provider.evaluate_frontiers(
            annotated_image=annotated,
            candidates=candidates,
            target_description=request.target_description
        )

        score_map = {s.id: s for s in scores}
        scored_frontiers: List[Frontier] = []
        for orig in request.candidates:
            scored = Frontier()
            scored.id = orig.id
            scored.pose_3d = Point(
                x=orig.pose_3d.x,
                y=orig.pose_3d.y,
                z=orig.pose_3d.z
            )
            scored.pixel_u = orig.pixel_u
            scored.pixel_v = orig.pixel_v
            scored.exploration_gain = orig.exploration_gain
            scored.is_viewpoint_frontier = orig.is_viewpoint_frontier

            if orig.id in score_map:
                scored.semantic_prob = score_map[orig.id].relevance_prob
                scored.total_utility = score_map[orig.id].total_utility
            else:
                scored.semantic_prob = 0.5
                scored.total_utility = 0.5 * orig.exploration_gain

            scored_frontiers.append(scored)

        response.success = True
        response.scored_frontiers = scored_frontiers
        response.reasoning = f"Evaluated via {provider.get_health_status().get('backend')}"
        return response

    def _handle_verify_target(
        self,
        request: VerifyTarget.Request,
        response: VerifyTarget.Response
    ) -> VerifyTarget.Response:
        """Verify target object visibility and reachability."""
        try:
            image_bgr = ros_image_to_cv2(request.image)
        except Exception as exc:
            self.get_logger().error(f"Image conversion error: {exc}")
            response.success = False
            response.is_present = False
            response.reasoning = f"Image conversion error: {exc}"
            return response

        provider = self._get_active_provider()
        res: TargetVerification = provider.verify_target(
            image_bgr=image_bgr,
            target_description=request.target_description,
            max_distance_m=request.max_distance_m
        )

        response.success = True
        response.is_present = res.is_present
        response.confidence = res.confidence
        response.reasoning = res.reasoning
        return response


def main(args: list = None) -> None:
    """Standard ROS 2 entrypoint lifecycle."""
    rclpy.init(args=args)
    node = SemanticReasonerNode()
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
