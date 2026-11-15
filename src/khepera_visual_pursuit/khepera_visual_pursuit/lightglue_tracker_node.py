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

"""High-frequency sparse correspondence tracking ROS 2 node."""

import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from khepera_visual_pursuit.image_converter import imgmsg_to_cv2
from khepera_visual_pursuit.ros_compat import (
    Detection2D,
    Detection2DArray,
    ExternalShutdownException,
    Float32,
    Image,
    Node,
    Point32,
    PolygonStamped,
    String,
    rclpy,
)
from khepera_visual_pursuit.superpoint_lightglue_engine import (
    MIN_KEYPOINTS_THRESHOLD,
    NOMINAL_MATCH_THRESHOLD,
    SuperPointLightGlueTracker,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TOPIC = "/camera/image_raw"
DEFAULT_TARGET_IMAGE_TOPIC = "/pursuit/target_image"
DEFAULT_TARGET_PROMPT_TOPIC = "/pursuit/target_prompt"
DEFAULT_PURSUIT_MODE_TOPIC = "/pursuit/mode"
DEFAULT_ROIS_TOPIC = "/perception/candidate_rois"
DEFAULT_KEYPOINTS_TOPIC = "/perception/target_keypoints"
DEFAULT_CONFIDENCE_TOPIC = "/perception/tracking_confidence"
DEFAULT_TRACKING_RATE_HZ = 30.0


class LightGlueTrackerNode(Node):
    """ROS 2 Node executing high-speed sparse correspondence tracking."""

    def __init__(self) -> None:
        """Initialize tracker node subscriptions, publishers, and timer."""
        super().__init__("lightglue_tracker_node")

        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("target_image_topic", DEFAULT_TARGET_IMAGE_TOPIC)
        self.declare_parameter("target_prompt_topic", DEFAULT_TARGET_PROMPT_TOPIC)
        self.declare_parameter("pursuit_mode_topic", DEFAULT_PURSUIT_MODE_TOPIC)
        self.declare_parameter("target_image_path", "")
        self.declare_parameter("rois_topic", DEFAULT_ROIS_TOPIC)
        self.declare_parameter("keypoints_topic", DEFAULT_KEYPOINTS_TOPIC)
        self.declare_parameter("confidence_topic", DEFAULT_CONFIDENCE_TOPIC)
        self.declare_parameter("min_keypoints", MIN_KEYPOINTS_THRESHOLD)
        self.declare_parameter("match_threshold", NOMINAL_MATCH_THRESHOLD)
        self.declare_parameter("tracking_rate_hz", DEFAULT_TRACKING_RATE_HZ)
        self.declare_parameter("device", "auto")

        img_topic = str(self.get_parameter("image_topic").value)
        target_img_topic = str(self.get_parameter("target_image_topic").value)
        target_prompt_topic = str(self.get_parameter("target_prompt_topic").value)
        mode_topic = str(self.get_parameter("pursuit_mode_topic").value)
        rois_topic = str(self.get_parameter("rois_topic").value)
        keypoints_topic = str(self.get_parameter("keypoints_topic").value)
        confidence_topic = str(self.get_parameter("confidence_topic").value)
        min_kps = int(self.get_parameter("min_keypoints").value)
        match_thresh = float(self.get_parameter("match_threshold").value)
        rate_hz = float(self.get_parameter("tracking_rate_hz").value)
        device = str(self.get_parameter("device").value)

        self.tracker = SuperPointLightGlueTracker(
            match_threshold=match_thresh,
            min_keypoints=min_kps,
            device=device,
        )

        self.pursuit_mode: str = "semantic"
        self.has_target_lock: bool = False
        self._last_confidence: float = 0.0
        self.latest_verified_roi: Optional[Detection2D] = None
        self.latest_roi_time: float = 0.0
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: str = "camera_optical_frame"

        self._load_target_from_parameter()

        self.image_sub = self.create_subscription(Image, img_topic, self._on_camera_image, 10)
        self.target_sub = self.create_subscription(
            Image, target_img_topic, self._on_target_image, 10
        )
        self.mode_sub = self.create_subscription(
            String, mode_topic, self._on_pursuit_mode, 10
        )
        self.prompt_sub = self.create_subscription(
            String, target_prompt_topic, self._on_target_prompt, 10
        )
        self.rois_sub = self.create_subscription(
            Detection2DArray, rois_topic, self._on_candidate_rois, 10
        )

        self.keypoints_pub = self.create_publisher(PolygonStamped, keypoints_topic, 10)
        self.confidence_pub = self.create_publisher(Float32, confidence_topic, 10)

        timer_period = 1.0 / max(1.0, rate_hz)
        self.timer = self.create_timer(timer_period, self._on_tracking_timer)

    def _load_target_from_parameter(self) -> None:
        """Load static target reference image if configured in parameters."""
        path_param = self.get_parameter("target_image_path").value
        path = str(path_param) if path_param else ""
        if path and os.path.isfile(path):
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                if self.tracker.set_reference_target(img_bgr):
                    self.pursuit_mode = "template"
                    self.has_target_lock = True
                    self.get_logger().info(f"Loaded target reference image from {path}")

    def _on_pursuit_mode(self, msg: String) -> None:
        """Update tracker operational pursuit mode."""
        mode = str(msg.data).strip().lower()
        if mode in ("template", "image", "mode_b", "mode b"):
            self.pursuit_mode = "template"
            self.get_logger().info("Tracker mode set to Mode B: Reference Image Template.")
        elif mode in ("semantic", "text", "mode_a", "mode a"):
            self.pursuit_mode = "semantic"
            self.has_target_lock = False
            self.latest_verified_roi = None
            self.latest_roi_time = 0.0
            self.tracker.reset()
            self.get_logger().info("Tracker mode set to Mode A: Semantic Description.")

    def _on_target_image(self, msg: Image) -> None:
        """Lock tracker reference template directly from target image message."""
        target_bgr = imgmsg_to_cv2(msg)
        if target_bgr is None or target_bgr.size == 0:
            return
        success = self.tracker.set_reference_target(target_bgr)
        if success:
            self.pursuit_mode = "template"
            self.has_target_lock = True
            self.get_logger().info("Target tracker reference lock established from target image.")

    def _on_target_prompt(self, msg: String) -> None:
        """Reset tracker state when new semantic prompt is received (only in Mode A)."""
        if self.pursuit_mode == "template" and self.has_target_lock:
            self.get_logger().info("Ignored target prompt in Mode B (Tracker Template Locked).")
            return
        prompt = str(msg.data).strip()
        if prompt:
            self.pursuit_mode = "semantic"
            self.tracker.reset()
            self.has_target_lock = False
            self.latest_verified_roi = None
            self.latest_roi_time = 0.0
            self.get_logger().info(
                f"Target prompt '{prompt}' received: reset tracker lock for search."
            )

    def _on_camera_image(self, msg: Image) -> None:
        """Cache incoming camera frame for high-frequency tracking."""
        self.latest_frame = imgmsg_to_cv2(msg)
        if msg.header and msg.header.frame_id:
            self.latest_frame_id = msg.header.frame_id

    def _on_candidate_rois(self, msg: Detection2DArray) -> None:
        """Lock tracker reference template onto highest-confidence verified ROI."""
        if not msg.detections or self.latest_frame is None or self.latest_frame.size == 0:
            return

        best_detection = self._select_best_roi(msg.detections)
        if best_detection is None:
            return

        self.latest_verified_roi = best_detection
        if hasattr(self, "get_clock"):
            self.latest_roi_time = float(self.get_clock().now().nanoseconds * 1e-9)
        else:
            self.latest_roi_time = float(time.time())

        # In Mode B (template), the reference template is immutable ground-truth
        if self.pursuit_mode == "template" and self.has_target_lock:
            return

        if self.has_target_lock and self._last_confidence >= 0.30:
            return

        crop_bgr = self._extract_detection_crop(self.latest_frame, best_detection)
        if crop_bgr is None or crop_bgr.size == 0:
            return

        success = self.tracker.set_reference_target(crop_bgr)
        if success:
            self.has_target_lock = True
            self._last_confidence = 0.50
            self.get_logger().info("Target tracker reference lock established.")

    def _select_best_roi(self, detections: List[Detection2D]) -> Optional[Detection2D]:
        """Select detection with highest confidence score."""
        if not detections:
            return None
        return max(
            detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0,
        )

    def _extract_detection_crop(
        self,
        frame_bgr: np.ndarray,
        detection: Detection2D,
    ) -> Optional[np.ndarray]:
        """Crop region of interest safely from camera frame."""
        h_frame, w_frame = frame_bgr.shape[:2]
        c_x = detection.bbox.center.position.x
        c_y = detection.bbox.center.position.y
        s_x = detection.bbox.size_x
        s_y = detection.bbox.size_y

        x1 = max(0, int(c_x - s_x / 2.0))
        y1 = max(0, int(c_y - s_y / 2.0))
        x2 = min(w_frame, int(c_x + s_x / 2.0))
        y2 = min(h_frame, int(c_y + s_y / 2.0))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame_bgr[y1:y2, x1:x2].copy()

    def _on_tracking_timer(self) -> None:
        """Run tracking cycle at nominal rate, publishing polygon and confidence."""
        if self.latest_frame is None:
            self._last_confidence = 0.0
            self.confidence_pub.publish(Float32(data=0.0))
            return

        # Attempt high-frequency feature tracking if locked
        if self.has_target_lock:
            output = self.tracker.track(self.latest_frame)
            if output.is_tracked:
                self._last_confidence = output.confidence
                poly_msg = self._build_polygon_message(output.polygon_points)
                self.keypoints_pub.publish(poly_msg)
                self.confidence_pub.publish(Float32(data=output.confidence))
                return

        # Fallback to verified candidate ROI vertices if present and recent
        now = (
            float(self.get_clock().now().nanoseconds * 1e-9)
            if hasattr(self, "get_clock")
            else float(time.time())
        )
        if self.latest_verified_roi is not None and (now - self.latest_roi_time) < 1.0:
            c_x = float(self.latest_verified_roi.bbox.center.position.x)
            c_y = float(self.latest_verified_roi.bbox.center.position.y)
            s_x = float(self.latest_verified_roi.bbox.size_x)
            s_y = float(self.latest_verified_roi.bbox.size_y)
            half_w = s_x / 2.0
            half_h = s_y / 2.0
            corners = (
                (c_x - half_w, c_y - half_h),
                (c_x + half_w, c_y - half_h),
                (c_x + half_w, c_y + half_h),
                (c_x - half_w, c_y + half_h),
            )
            score = (
                float(self.latest_verified_roi.results[0].hypothesis.score)
                if self.latest_verified_roi.results
                else 0.60
            )
            self._last_confidence = score
            poly_msg = self._build_polygon_message(corners)
            self.keypoints_pub.publish(poly_msg)
            self.confidence_pub.publish(Float32(data=score))
            return

        self._last_confidence = 0.0
        self.confidence_pub.publish(Float32(data=0.0))

    def _build_polygon_message(
        self,
        points: Tuple[Tuple[float, float], ...],
    ) -> PolygonStamped:
        """Assemble PolygonStamped message from bounding polygon vertices."""
        poly_msg = PolygonStamped()
        poly_msg.header.frame_id = self.latest_frame_id
        if hasattr(self, "get_clock"):
            poly_msg.header.stamp = self.get_clock().now().to_msg()
        for pt in points:
            poly_msg.polygon.points.append(Point32(x=float(pt[0]), y=float(pt[1]), z=0.0))
        return poly_msg


def main(args: Optional[List[str]] = None) -> None:
    """Run LightGlue tracker node with standard lifecycle ordering."""
    rclpy.init(args=args)
    node = LightGlueTrackerNode()
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
