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

"""ROS 2 node generating proposals via YOLO-World and verifying with DINOv2."""

import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from khepera_visual_pursuit.dinov2_verifier import DINOv2Verifier
from khepera_visual_pursuit.image_converter import cv2_to_imgmsg, imgmsg_to_cv2
from khepera_visual_pursuit.ros_compat import (
    Bool,
    Detection2D,
    Detection2DArray,
    ExternalShutdownException,
    Image,
    Node,
    ObjectHypothesisWithPose,
    String,
    rclpy,
)
from khepera_visual_pursuit.yolo_world_engine import (
    NEGATIVE_CLASSES,
    BoundingBox,
    YoloWorldEngine,
    calculate_weighted_semantic_score,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TOPIC = "/camera/image_raw"
DEFAULT_TARGET_IMAGE_TOPIC = "/pursuit/target_image"
DEFAULT_TARGET_PROMPT_TOPIC = "/pursuit/target_prompt"
DEFAULT_PURSUIT_MODE_TOPIC = "/pursuit/mode"
DEFAULT_ROIS_TOPIC = "/perception/candidate_rois"
DEFAULT_STATUS_TOPIC = "/perception/target_status"
DEFAULT_VISUAL_TOPIC = "/target/detection_visual"
DEFAULT_THINKING_TOPIC = "/robot/thinking_log"
DEFAULT_STOP_TOPIC = "/target_found_and_stopped"
DEFAULT_DETECTION_RATE_HZ = 10.0
DEFAULT_TAU_VERIFY = 0.55
DEFAULT_TAU_MARGIN = 0.025
DEFAULT_CLASSES: Tuple[str, ...] = ("target object", "cylinder", "box", "can")
GENERIC_OBJECT_CLASSES: Tuple[str, ...] = (
    "object", "block", "box", "cube", "cylinder", "can", "toy", "entity", "structure", "target"
)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate intersection-over-union between two bounding boxes."""
    x1 = max(box1.x_min, box2.x_min)
    y1 = max(box1.y_min, box2.y_min)
    x2 = min(box1.x_max, box2.x_max)
    y2 = min(box1.y_max, box2.y_max)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = float(inter_w * inter_h)
    box1_area = float(max(1, (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)))
    box2_area = float(max(1, (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)))
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def apply_nms(
    proposals: Tuple[BoundingBox, ...],
    iou_threshold: float = 0.35,
) -> Tuple[BoundingBox, ...]:
    """Suppress redundant overlapping bounding box proposals."""
    if not proposals:
        return ()
    sorted_proposals = sorted(proposals, key=lambda p: p.score, reverse=True)
    selected: List[BoundingBox] = []
    for prop in sorted_proposals:
        if any(calculate_iou(prop, s) > iou_threshold for s in selected):
            continue
        selected.append(prop)
    return tuple(selected)


class ObjectDetectorVerifierNode(Node):
    """ROS 2 Node coordinating YOLO-World proposals and DINOv2 verification."""

    def __init__(self) -> None:
        """Initialize detector and verifier node with topics and parameters."""
        super().__init__("object_detector_verifier_node")

        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("target_image_topic", DEFAULT_TARGET_IMAGE_TOPIC)
        self.declare_parameter("target_prompt_topic", DEFAULT_TARGET_PROMPT_TOPIC)
        self.declare_parameter("pursuit_mode_topic", DEFAULT_PURSUIT_MODE_TOPIC)
        self.declare_parameter("target_image_path", "")
        self.declare_parameter("detection_rate_hz", DEFAULT_DETECTION_RATE_HZ)
        self.declare_parameter("tau_verify", DEFAULT_TAU_VERIFY)
        self.declare_parameter("tau_margin", DEFAULT_TAU_MARGIN)
        self.declare_parameter("classes", list(DEFAULT_CLASSES))
        self.declare_parameter("device", "auto")

        img_topic = str(self.get_parameter("image_topic").value)
        tgt_topic = str(self.get_parameter("target_image_topic").value)
        tgt_prompt_topic = str(self.get_parameter("target_prompt_topic").value)
        mode_topic = str(self.get_parameter("pursuit_mode_topic").value)
        rate_hz = float(self.get_parameter("detection_rate_hz").value)
        tau_verify = float(self.get_parameter("tau_verify").value)
        tau_margin = float(self.get_parameter("tau_margin").value)
        device = str(self.get_parameter("device").value)

        self.detector = YoloWorldEngine(device=device)
        self.verifier = DINOv2Verifier(tau_verify=tau_verify, tau_margin=tau_margin)

        self.pursuit_mode: str = "semantic"
        self.is_template_locked: bool = False
        self.target_name: str = "target object"
        self.target_desc: str = ""
        self.target_template_bgr: Optional[np.ndarray] = None
        self.target_embedding: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: str = "camera_optical_frame"

        # Temporal persistence tracking state to prevent target dropping
        self.target_locked: bool = False
        self.last_verified_box: Optional[BoundingBox] = None
        self.last_verified_sim: float = 0.0
        self.missed_detection_count: int = 0
        self.max_coast_frames: int = 5

        # Mode A temporal confirmation buffer and timeout tracking
        self.confirmation_buffer: List[BoundingBox] = []
        self.last_target_sim_time: float = 0.0
        self.CONFIRMATION_FRAMES_REQUIRED: int = 3
        self.CONFIRMATION_IOU_THRESHOLD: float = 0.20
        self.CONFIRMATION_SCORE_THRESHOLD: float = 0.25
        self.LOCK_TIMEOUT_SECONDS: float = 6.0

        # Mode B temporal confirmation buffer and strict verification thresholds
        self.mode_b_confirmation_buffer: List[BoundingBox] = []
        self.TAU_VERIFY_MODE_B_INIT: float = 0.68
        self.TAU_VERIFY_MODE_B_TRACK: float = 0.58
        self.MODE_B_CONFIRMATION_FRAMES: int = 2

        self._load_target_from_parameter()

        self.image_sub = self.create_subscription(Image, img_topic, self._on_camera_image, 10)
        self.target_sub = self.create_subscription(
            Image, tgt_topic, self._on_target_image_message, 10
        )
        self.mode_sub = self.create_subscription(
            String, mode_topic, self._on_pursuit_mode_message, 10
        )

        self.rois_pub = self.create_publisher(Detection2DArray, DEFAULT_ROIS_TOPIC, 10)
        self.status_pub = self.create_publisher(Bool, DEFAULT_STATUS_TOPIC, 10)
        self.visual_pub = self.create_publisher(Image, DEFAULT_VISUAL_TOPIC, 10)
        self.thinking_pub = self.create_publisher(String, DEFAULT_THINKING_TOPIC, 10)
        self.prompt_sub = self.create_subscription(
            String, tgt_prompt_topic, self._on_target_prompt_message, 10
        )
        self.stop_sub = self.create_subscription(
            Bool, DEFAULT_STOP_TOPIC, self._on_target_found_and_stopped, 10
        )
        self.is_halted: bool = False
        self._last_thought_text = ""
        self._last_thought_time = 0.0

        timer_period = 1.0 / max(1.0, rate_hz)
        self.timer = self.create_timer(timer_period, self._on_detection_timer)

    def _load_target_from_parameter(self) -> None:
        """Load static target reference image if configured in parameters."""
        path_param = self.get_parameter("target_image_path").value
        path = str(path_param) if path_param else ""
        if path and os.path.isfile(path):
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                self.target_template_bgr = img_bgr.copy()
                self.target_embedding = self.verifier.extract_embedding(img_bgr)
                self.pursuit_mode = "template"
                self.is_template_locked = True
                self.detector.set_classes(GENERIC_OBJECT_CLASSES)
                self.get_logger().info(f"Loaded target reference image from {path}")

    def set_target_image(self, target_bgr: np.ndarray) -> None:
        """Directly assign reference target template for verification."""
        if target_bgr is not None and target_bgr.size > 0:
            self.target_template_bgr = target_bgr.copy()
            self.target_embedding = self.verifier.extract_embedding(target_bgr)
            self.pursuit_mode = "template"
            self.is_template_locked = True
            self.target_locked = False
            self.last_verified_box = None
            self.missed_detection_count = 0
            self.confirmation_buffer.clear()
            self.mode_b_confirmation_buffer.clear()
            self.detector.set_classes(GENERIC_OBJECT_CLASSES)

    def _on_target_found_and_stopped(self, msg: Bool) -> None:
        """Halt object detector inference when target is reached and stopped."""
        if msg.data:
            self.is_halted = True
            self.confirmation_buffer.clear()
            self.mode_b_confirmation_buffer.clear()
            self._publish_empty_results()
            self.get_logger().info("Target reached and stopped. Halting object detector node.")
        else:
            self.is_halted = False

    def _on_pursuit_mode_message(self, msg: String) -> None:
        """Switch operational pursuit mode between Mode A (semantic) and Mode B (template)."""
        mode = str(msg.data).strip().lower()
        self.is_halted = False
        self.target_locked = False
        self.last_verified_box = None
        self.missed_detection_count = 0
        self.confirmation_buffer.clear()
        self.mode_b_confirmation_buffer.clear()
        self.last_target_sim_time = 0.0
        if mode in ("template", "image", "mode_b", "mode b"):
            self.pursuit_mode = "template"
            self.detector.set_classes(GENERIC_OBJECT_CLASSES)
            self.get_logger().info("Pursuit mode switched to Mode B: Reference Image Template.")
        elif mode in ("semantic", "text", "mode_a", "mode a"):
            self.pursuit_mode = "semantic"
            self.is_template_locked = False
            self.target_embedding = None
            self.get_logger().info("Pursuit mode switched to Mode A: Semantic Description.")

    def _on_camera_image(self, msg: Image) -> None:
        """Cache incoming camera frame without running blocking inference."""
        self.latest_frame = imgmsg_to_cv2(msg)
        if msg.header and msg.header.frame_id:
            self.latest_frame_id = msg.header.frame_id

    def _on_target_image_message(self, msg: Image) -> None:
        """Update target template descriptor from incoming pursuit target message (Mode B)."""
        target_bgr = imgmsg_to_cv2(msg)
        if target_bgr is not None and target_bgr.size > 0:
            self.pursuit_mode = "template"
            self.target_template_bgr = target_bgr.copy()
            self.target_embedding = self.verifier.extract_embedding(target_bgr)
            self.is_template_locked = True
            self.target_locked = False
            self.last_verified_box = None
            self.missed_detection_count = 0
            self.confirmation_buffer.clear()
            self.mode_b_confirmation_buffer.clear()
            self.last_target_sim_time = 0.0
            self.detector.set_classes(GENERIC_OBJECT_CLASSES)
            self._publish_thought(
                "Target reference image received -> DINOv2 verification active in Mode B (Template)"
            )
            self.get_logger().info("Target visual descriptor updated from topic in Mode B.")

    def _on_target_prompt_message(self, msg: String) -> None:
        """Update open-vocabulary proposal classes for Mode A semantic search."""
        if self.pursuit_mode == "template" and self.is_template_locked:
            self.get_logger().info(
                f"Target prompt '{msg.data}' ignored in Mode B (Template Lock Active)."
            )
            return
        raw_prompt = str(msg.data).strip()
        if not raw_prompt:
            return

        self.pursuit_mode = "semantic"
        self.is_template_locked = False
        self.is_halted = False
        self.target_embedding = None
        self.confirmation_buffer.clear()
        self.last_target_sim_time = 0.0

        # Parse target name and description (e.g. "red chair | sits in the office", "chair : red")
        target_name = raw_prompt
        target_desc = ""
        for sep in ("|", "::", ":"):
            if sep in raw_prompt:
                parts = raw_prompt.split(sep, 1)
                target_name = parts[0].strip()
                target_desc = parts[1].strip()
                break

        self.target_name = target_name
        self.target_desc = target_desc

        # Class name represents primary discriminator; description provides context
        candidate_classes = [target_name]
        if target_desc:
            candidate_classes.append(f"{target_name}, {target_desc}")
            candidate_classes.append(target_desc)
        candidate_classes.extend([f"a {target_name}", f"photo of a {target_name}"])
        for shape_name in ("cylinder", "box", "cube", "can", "sphere", "ball"):
            if shape_name in target_name.lower() and shape_name != target_name.lower():
                candidate_classes.append(shape_name)
        candidate_classes.extend(NEGATIVE_CLASSES)
        target_name_lower = target_name.lower()
        for distractor_color in ("green", "blue", "red", "yellow"):
            if distractor_color not in target_name_lower:
                candidate_classes.append(f"{distractor_color} object")
                candidate_classes.append(f"{distractor_color} box")

        # Deduplicate while preserving ordering
        unique_classes = tuple(dict.fromkeys(candidate_classes))
        self.detector.set_classes(unique_classes)
        self._publish_thought(f"Target open-vocab prompt updated: '{raw_prompt}'")
        self.get_logger().info(f"Target open-vocab prompt updated: '{raw_prompt}'")

    def _generate_mode_b_proposals(
        self, frame_bgr: np.ndarray, yolo_proposals: Tuple[BoundingBox, ...]
    ) -> List[BoundingBox]:
        """Generate candidate proposals combining YOLO, template matching, and priors."""
        h, w = frame_bgr.shape[:2]
        negative_class_set = {neg.lower() for neg in NEGATIVE_CLASSES}
        proposals: List[BoundingBox] = [
            p for p in yolo_proposals if p.class_name.strip().lower() not in negative_class_set
        ]

        # 1. Temporal tracking prior if target was recently locked
        if self.last_verified_box is not None:
            proposals.append(self.last_verified_box)
            proposals.append(
                BoundingBox(
                    center_x=self.last_verified_box.center_x,
                    center_y=self.last_verified_box.center_y,
                    size_x=float(self.last_verified_box.size_x * 1.15),
                    size_y=float(self.last_verified_box.size_y * 1.15),
                    confidence=0.85,
                    class_name="template_prior",
                )
            )

        # 2. Multi-scale template matching proposals (downscaled for speed < 15ms)
        if self.target_template_bgr is not None and self.target_template_bgr.size > 0:
            scale_factor = 2.0
            sh_frame = int(h / scale_factor)
            sw_frame = int(w / scale_factor)
            small_frame = cv2.resize(frame_bgr, (sw_frame, sh_frame))
            frame_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            base_size = 100
            base_tpl = cv2.resize(self.target_template_bgr, (base_size, base_size))
            tpl_gray = cv2.cvtColor(base_tpl, cv2.COLOR_BGR2GRAY)

            # Scales covering 1.5m to 0.25m distance
            for s in (0.22, 0.30, 0.40, 0.55, 0.75, 1.00, 1.15, 1.30):
                sw_box = int(base_size * s)
                sh_box = int(base_size * s)
                if sw_box < 12 or sh_box < 12 or sw_box >= sw_frame or sh_box >= sh_frame:
                    continue
                scaled_tpl = cv2.resize(tpl_gray, (sw_box, sh_box), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(frame_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.40:
                    cx = float((max_loc[0] + sw_box / 2.0) * scale_factor)
                    cy = float((max_loc[1] + sh_box / 2.0) * scale_factor)
                    bw = float(sw_box * scale_factor)
                    bh = float(sh_box * scale_factor)
                    proposals.append(
                        BoundingBox(
                            center_x=cx,
                            center_y=cy,
                            size_x=bw,
                            size_y=bh,
                            confidence=float(0.80 + 0.20 * max_val),
                            class_name="template_match",
                        )
                    )

            # 3. Contour / edge proposals on salient foreground
            blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cx_b, cy_b, cw_b, ch_b = cv2.boundingRect(cnt)
                if cw_b > 15 and ch_b > 15 and (cw_b * ch_b) < (sw_frame * sh_frame * 0.45):
                    proposals.append(
                        BoundingBox(
                            center_x=float((cx_b + cw_b / 2.0) * scale_factor),
                            center_y=float((cy_b + ch_b / 2.0) * scale_factor),
                            size_x=float(cw_b * scale_factor),
                            size_y=float(ch_b * scale_factor),
                            confidence=0.30,
                            class_name="contour",
                        )
                    )

        def proposal_priority(p: BoundingBox) -> float:
            weight = 2.0 if ("template" in p.class_name) else 1.0
            return float(p.score * weight)

        nms_proposals = apply_nms(tuple(proposals), iou_threshold=0.35)
        candidates = [p for p in nms_proposals if p.score >= 0.25]
        candidates.sort(key=proposal_priority, reverse=True)
        return list(candidates[:6])

    def _on_detection_timer(self) -> None:
        """Periodically generate candidate proposals and verify target identity."""
        if self.is_halted:
            return

        if self.latest_frame is None or self.latest_frame.size == 0:
            self._publish_empty_results()
            return

        proposals = self.detector.generate_proposals(self.latest_frame)

        # ------------------------------------------------------------------
        # Mode B: Reference Image Template (DINOv2)
        # ------------------------------------------------------------------
        if self.pursuit_mode == "template":
            if self.target_embedding is None:
                self._publish_empty_results()
                return

            top_candidates = self._generate_mode_b_proposals(self.latest_frame, proposals)
            if not top_candidates:
                self.mode_b_confirmation_buffer.clear()
                if (
                    self.target_locked
                    and self.missed_detection_count < self.max_coast_frames
                    and self.last_verified_box is not None
                ):
                    self.missed_detection_count += 1
                    decayed_sim = self.last_verified_sim * (0.96 ** self.missed_detection_count)
                    array_msg = self._build_detection_array(self.last_verified_box, decayed_sim)
                    self.rois_pub.publish(array_msg)
                    self.status_pub.publish(Bool(data=True))
                    self._publish_detection_visual(self.last_verified_box, decayed_sim)
                    thought = (
                        f"⏳ [Mode B Persistence] Holding target lock ({self.missed_detection_count}/"
                        f"{self.max_coast_frames} frames) -> Sim={decayed_sim:.2f}"
                    )
                    self._publish_thought(thought)
                else:
                    self.target_locked = False
                    self.last_verified_box = None
                    self._publish_empty_results()
                return

            crops = [self.detector.crop_roi(self.latest_frame, p) for p in top_candidates]
            embeddings = self.verifier.extract_embeddings_batch(crops)
            verif = self.verifier.verify_candidates(
                embeddings, self.target_embedding, candidate_boxes=tuple(top_candidates)
            )

            tau_req = (
                self.TAU_VERIFY_MODE_B_TRACK
                if self.target_locked
                else self.TAU_VERIFY_MODE_B_INIT
            )
            if (
                verif.is_verified
                and verif.best_candidate_index >= 0
                and verif.best_similarity >= tau_req
            ):
                best_box = top_candidates[verif.best_candidate_index]

                if not self.target_locked:
                    if len(self.mode_b_confirmation_buffer) == 0:
                        self.mode_b_confirmation_buffer.append(best_box)
                    else:
                        last_box = self.mode_b_confirmation_buffer[-1]
                        if calculate_iou(best_box, last_box) >= self.CONFIRMATION_IOU_THRESHOLD:
                            self.mode_b_confirmation_buffer.append(best_box)
                        else:
                            self.mode_b_confirmation_buffer = [best_box]

                    if len(self.mode_b_confirmation_buffer) >= self.MODE_B_CONFIRMATION_FRAMES:
                        self.target_locked = True
                        self.last_verified_box = best_box
                        self.last_verified_sim = verif.best_similarity
                        self.missed_detection_count = 0
                        self.mode_b_confirmation_buffer.clear()

                        array_msg = self._build_detection_array(best_box, verif.best_similarity)
                        self.rois_pub.publish(array_msg)
                        self.status_pub.publish(Bool(data=True))
                        self._publish_detection_visual(best_box, verif.best_similarity)
                        thought = (
                            f"🎯 [Mode B] Target Confirmed by DINOv2! Sim={verif.best_similarity:.2f} "
                            f"(margin={verif.margin:.2f}) -> Visual Servoing active"
                        )
                        self._publish_thought(thought)
                    else:
                        thought = (
                            f"🔍 [Mode B] Target candidate verifying "
                            f"({len(self.mode_b_confirmation_buffer)}/{self.MODE_B_CONFIRMATION_FRAMES} frames) "
                            f"-> Sim={verif.best_similarity:.2f}"
                        )
                        self._publish_thought(thought)
                        array_msg = self._build_detection_array(best_box, verif.best_similarity)
                        self.rois_pub.publish(array_msg)
                        self._publish_detection_visual(best_box, verif.best_similarity)
                else:
                    self.last_verified_box = best_box
                    self.last_verified_sim = verif.best_similarity
                    self.missed_detection_count = 0
                    self.mode_b_confirmation_buffer.clear()

                    array_msg = self._build_detection_array(best_box, verif.best_similarity)
                    self.rois_pub.publish(array_msg)
                    self.status_pub.publish(Bool(data=True))
                    self._publish_detection_visual(best_box, verif.best_similarity)
                    thought = (
                        f"🎯 [Mode B Tracking] Sim={verif.best_similarity:.2f} "
                        f"(margin={verif.margin:.2f}) -> Visual Servoing active"
                    )
                    self._publish_thought(thought)
            elif (
                self.target_locked
                and self.missed_detection_count < self.max_coast_frames
                and self.last_verified_box is not None
            ):
                self.mode_b_confirmation_buffer.clear()
                self.missed_detection_count += 1
                decayed_sim = self.last_verified_sim * (0.96 ** self.missed_detection_count)
                array_msg = self._build_detection_array(self.last_verified_box, decayed_sim)
                self.rois_pub.publish(array_msg)
                self.status_pub.publish(Bool(data=True))
                self._publish_detection_visual(self.last_verified_box, decayed_sim)
                thought = (
                    f"⏳ [Mode B Persistence] Holding target lock ({self.missed_detection_count}/"
                    f"{self.max_coast_frames} frames) -> Sim={decayed_sim:.2f}"
                )
                self._publish_thought(thought)
            else:
                self.target_locked = False
                self.last_verified_box = None
                self.mode_b_confirmation_buffer.clear()
                self._publish_empty_results()
            return

        # ------------------------------------------------------------------
        # Mode A: Semantic Description (YOLO-World + Bootstrapped Tracking)
        # ------------------------------------------------------------------
        now = time.time()
        if not proposals:
            if (
                self.target_embedding is not None
                and (now - self.last_target_sim_time) > self.LOCK_TIMEOUT_SECONDS
            ):
                self._unlock_mode_a("No proposals and lock timeout exceeded")
            self._publish_empty_results()
            return

        nms_proposals = apply_nms(proposals, iou_threshold=0.35)
        negative_class_set = {neg.lower() for neg in NEGATIVE_CLASSES}
        target_name_lower = self.target_name.lower()
        for distractor_color in ("green", "blue", "red", "yellow"):
            if distractor_color not in target_name_lower:
                negative_class_set.add(f"{distractor_color} object")
                negative_class_set.add(f"{distractor_color} box")

        scored_candidates = []
        for p in nms_proposals:
            if p.class_name.strip().lower() in negative_class_set:
                continue
            crop = self.detector.crop_roi(self.latest_frame, p)
            w_score = calculate_weighted_semantic_score(
                p, self.target_name, self.target_desc, crop
            )
            if w_score >= 0.25:
                scored_candidates.append(
                    BoundingBox(
                        center_x=p.center_x,
                        center_y=p.center_y,
                        size_x=p.size_x,
                        size_y=p.size_y,
                        confidence=w_score,
                        class_name=p.class_name,
                    )
                )

        if not scored_candidates:
            if (
                self.target_embedding is not None
                and (now - self.last_target_sim_time) > self.LOCK_TIMEOUT_SECONDS
            ):
                self._unlock_mode_a("No valid candidates and lock timeout exceeded")
            self._publish_empty_results()
            return

        scored_candidates.sort(key=lambda p: p.score, reverse=True)
        best_box = scored_candidates[0]

        # 1. Unlocked open-vocabulary search mode with 3-frame confirmation buffer
        if self.target_embedding is None:
            if best_box.score >= self.CONFIRMATION_SCORE_THRESHOLD:
                if len(self.confirmation_buffer) == 0:
                    self.confirmation_buffer.append(best_box)
                else:
                    last_box = self.confirmation_buffer[-1]
                    if calculate_iou(best_box, last_box) >= self.CONFIRMATION_IOU_THRESHOLD:
                        self.confirmation_buffer.append(best_box)
                    else:
                        self.confirmation_buffer = [best_box]

                if len(self.confirmation_buffer) >= self.CONFIRMATION_FRAMES_REQUIRED:
                    crop = self.detector.crop_roi(self.latest_frame, best_box)
                    self.target_embedding = self.verifier.extract_embedding(crop)
                    self.last_target_sim_time = now
                    self.confirmation_buffer.clear()
                    thought = (
                        f"🎯 Target '{best_box.class_name}' Locked "
                        f"(confirmed {self.CONFIRMATION_FRAMES_REQUIRED} frames)! "
                        f"Weighted Score = {best_box.score:.2f} (70% Name, 30% Desc)"
                    )
                    self._publish_thought(thought)
                else:
                    thought = (
                        f"🔍 Target candidate '{best_box.class_name}' confirming "
                        f"({len(self.confirmation_buffer)}/{self.CONFIRMATION_FRAMES_REQUIRED} "
                        f"frames)... Score = {best_box.score:.2f}"
                    )
                    self._publish_thought(thought)

                array_msg = self._build_detection_array(best_box, best_box.score)
                self.rois_pub.publish(array_msg)
                self.status_pub.publish(Bool(data=True))
                self._publish_detection_visual(best_box, best_box.score)
            else:
                self.confirmation_buffer.clear()
                self._publish_empty_results()
            return

        # 2. Locked Mode A: validate semantic relevance alongside DINOv2
        top_candidates = scored_candidates[:3]
        crops = [self.detector.crop_roi(self.latest_frame, p) for p in top_candidates]
        embeddings = self.verifier.extract_embeddings_batch(crops)
        verif = self.verifier.verify_candidates(embeddings, self.target_embedding)

        chosen_box = None
        chosen_score = 0.0

        if verif.is_verified and verif.best_candidate_index >= 0:
            chosen_box = top_candidates[verif.best_candidate_index]
            chosen_score = verif.best_similarity
            self.last_target_sim_time = now
        elif best_box.score >= self.CONFIRMATION_SCORE_THRESHOLD:
            # Semantic relevance alongside DINOv2: check similarity
            best_crop = self.detector.crop_roi(self.latest_frame, best_box)
            new_emb = self.verifier.extract_embedding(best_crop)
            sim = self.verifier.calculate_cosine_similarity(new_emb, self.target_embedding)
            if sim >= self.verifier.tau_verify or best_box.score >= 0.35:
                self.last_target_sim_time = now
                chosen_box = best_box
                chosen_score = max(sim, best_box.score)
                if new_emb is not None and self.target_embedding is not None and sim >= 0.40:
                    self.target_embedding = 0.8 * self.target_embedding + 0.2 * new_emb
                    norm = float(np.linalg.norm(self.target_embedding))
                    if norm > 1e-6:
                        self.target_embedding = self.target_embedding / norm
            elif (now - self.last_target_sim_time) > self.LOCK_TIMEOUT_SECONDS:
                self._unlock_mode_a(
                    f"Target similarity ({sim:.2f}) dropped below threshold for > {self.LOCK_TIMEOUT_SECONDS}s"
                )
                self._publish_empty_results()
                return
            else:
                self._publish_empty_results()
                return
        else:
            if (now - self.last_target_sim_time) > self.LOCK_TIMEOUT_SECONDS:
                self._unlock_mode_a("Target lost and lock timeout exceeded")
                self._publish_empty_results()
                return

        if chosen_box is not None:
            array_msg = self._build_detection_array(chosen_box, chosen_score)
            self.rois_pub.publish(array_msg)
            self.status_pub.publish(Bool(data=True))
            self._publish_detection_visual(chosen_box, chosen_score)
            thought = (
                f"🎯 Target '{chosen_box.class_name}' Verified! "
                f"Weighted Score = {chosen_score:.2f} (70% Name, 30% Desc)"
            )
            self._publish_thought(thought)
        else:
            self._publish_empty_results()

    def _unlock_mode_a(self, reason: str = "") -> None:
        """Unlock Mode A target template and re-enter open-vocabulary search mode."""
        self.target_embedding = None
        self.confirmation_buffer.clear()
        self.last_target_sim_time = 0.0
        msg = f"🔓 Mode A Unlocked ({reason}) -> Re-entering open-vocabulary search mode"
        self._publish_thought(msg)
        self.get_logger().info(msg)

    def _publish_detection_visual(self, best_box: BoundingBox, score: float) -> None:
        """Publish annotated image frame with detected bounding box."""
        if self.latest_frame is None or not hasattr(self, "visual_pub") or not self.visual_pub:
            return
        vis = self.latest_frame.copy()
        h, w = vis.shape[:2]
        x1 = min(w - 1, max(0, int(best_box.center_x - best_box.size_x / 2.0)))
        y1 = min(h - 1, max(0, int(best_box.center_y - best_box.size_y / 2.0)))
        x2 = min(w, max(x1 + 1, int(best_box.center_x + best_box.size_x / 2.0)))
        y2 = min(h, max(y1 + 1, int(best_box.center_y + best_box.size_y / 2.0)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"{best_box.class_name} {score:.2f}"
        cv2.putText(
            vis, lbl, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        vis_msg = cv2_to_imgmsg(vis, encoding="bgr8", frame_id=self.latest_frame_id)
        if hasattr(self, "get_clock"):
            vis_msg.header.stamp = self.get_clock().now().to_msg()
        self.visual_pub.publish(vis_msg)

    def _publish_thought(self, text: str) -> None:
        """Publish cognitive reasoning log for dashboard telemetry."""
        now = time.time()
        if text == self._last_thought_text and (now - self._last_thought_time) < 2.5:
            return
        self._last_thought_text = text
        self._last_thought_time = now
        if hasattr(self, "thinking_pub") and self.thinking_pub:
            msg = String()
            msg.data = text
            self.thinking_pub.publish(msg)

    def _publish_empty_results(self) -> None:
        """Publish negative tracking state and empty detection array."""
        empty_array = Detection2DArray()
        empty_array.header.frame_id = self.latest_frame_id
        if hasattr(self, "get_clock"):
            empty_array.header.stamp = self.get_clock().now().to_msg()
        self.rois_pub.publish(empty_array)
        self.status_pub.publish(Bool(data=False))
        if self.latest_frame is not None and hasattr(self, "visual_pub") and self.visual_pub:
            vis_msg = cv2_to_imgmsg(
                self.latest_frame, encoding="bgr8", frame_id=self.latest_frame_id
            )
            if hasattr(self, "get_clock"):
                vis_msg.header.stamp = self.get_clock().now().to_msg()
            self.visual_pub.publish(vis_msg)

    def _build_detection_array(self, box: BoundingBox, similarity: float) -> Detection2DArray:
        """Build Detection2DArray message enclosing verified candidate bounding box."""
        msg = Detection2DArray()
        msg.header.frame_id = self.latest_frame_id
        if hasattr(self, "get_clock"):
            msg.header.stamp = self.get_clock().now().to_msg()

        detection = Detection2D()
        detection.header.frame_id = self.latest_frame_id
        if hasattr(self, "get_clock"):
            detection.header.stamp = self.get_clock().now().to_msg()

        detection.bbox.center.position.x = float(box.center_x)
        detection.bbox.center.position.y = float(box.center_y)
        detection.bbox.size_x = float(box.size_x)
        detection.bbox.size_y = float(box.size_y)

        hypo = ObjectHypothesisWithPose()
        hypo.hypothesis.class_id = box.class_name
        hypo.hypothesis.score = float(similarity)
        detection.results.append(hypo)

        msg.detections.append(detection)
        return msg


def main(args: Optional[List[str]] = None) -> None:
    """Run object detector and verifier node with standard lifecycle ordering."""
    rclpy.init(args=args)
    node = ObjectDetectorVerifierNode()
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
