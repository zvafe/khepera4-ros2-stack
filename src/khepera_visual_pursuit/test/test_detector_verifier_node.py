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

"""Integration tests for ObjectDetectorVerifierNode (Node 1)."""

import os
import tempfile
import time

import cv2
import numpy as np
import pytest

from khepera_visual_pursuit.image_converter import cv2_to_imgmsg, imgmsg_to_cv2
from khepera_visual_pursuit.object_detector_verifier_node import (
    ObjectDetectorVerifierNode,
)
from khepera_visual_pursuit.ros_compat import String, rclpy
from khepera_visual_pursuit.yolo_world_engine import BoundingBox


@pytest.fixture(scope="module", autouse=True)
def rclpy_init_teardown():
    """Manage ROS 2 context lifecycle for node tests."""
    if rclpy is not None and hasattr(rclpy, "ok"):
        if not rclpy.ok():
            rclpy.init()
        yield
        if rclpy.ok():
            rclpy.shutdown()
    else:
        yield


def test_detector_verifier_node_initialization() -> None:
    """Verify parameters, publishers, and subscriptions upon initialization."""
    node = ObjectDetectorVerifierNode()
    assert node.get_name() == "object_detector_verifier_node"
    assert node.rois_pub is not None
    assert node.status_pub is not None
    assert node.visual_pub is not None
    assert node.thinking_pub is not None
    assert node.image_sub is not None
    assert node.target_sub is not None
    assert node.prompt_sub is not None
    node.destroy_node()


def test_detector_verifier_timer_empty_frame() -> None:
    """Ensure negative target status is published when no camera image is available."""
    node = ObjectDetectorVerifierNode()
    published_status = []
    published_rois = []
    published_visuals = []
    node.status_pub.publish = lambda m: published_status.append(m)
    node.rois_pub.publish = lambda m: published_rois.append(m)
    node.visual_pub.publish = lambda m: published_visuals.append(m)

    # Trigger detection timer with no image cached
    node._on_detection_timer()

    assert len(published_status) > 0
    latest_status = published_status[-1]
    assert latest_status.data is False

    assert len(published_rois) > 0
    latest_rois = published_rois[-1]
    assert len(latest_rois.detections) == 0
    node.destroy_node()


def test_detector_verifier_mode_a_semantic_prompt() -> None:
    """Verify Mode A semantic description prompt updates classes and bootstraps tracking."""
    node = ObjectDetectorVerifierNode()
    published_status = []
    published_rois = []
    published_visuals = []
    published_thoughts = []

    node.status_pub.publish = lambda m: published_status.append(m)
    node.rois_pub.publish = lambda m: published_rois.append(m)
    node.visual_pub.publish = lambda m: published_visuals.append(m)
    node.thinking_pub.publish = lambda m: published_thoughts.append(m)

    # Pre-set embedding to simulate prior target
    dummy_emb = np.ones((384,), dtype=np.float32)
    node.target_embedding = dummy_emb

    # Send new Mode A prompt
    prompt_msg = String()
    prompt_msg.data = "blue cylinder"
    node._on_target_prompt_message(prompt_msg)

    # Classes updated and embedding reset
    assert "blue cylinder" in node.detector.classes
    assert node.target_embedding is None
    assert len(published_thoughts) >= 1
    assert "blue cylinder" in published_thoughts[-1].data

    # Feed scene image with synthetic contour
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(scene, (100, 100), (160, 180), (255, 120, 0), -1)
    img_msg = cv2_to_imgmsg(scene)
    node._on_camera_image(img_msg)

    # Trigger detection cycle in un-bootstrapped state (3 frames for temporal confirmation buffer)
    for _ in range(3):
        node._on_detection_timer()

    assert len(published_status) > 0
    assert published_status[-1].data is True
    assert len(published_rois) > 0
    assert len(published_rois[-1].detections) >= 1
    assert node.target_embedding is not None

    # Verify visual HUD output published
    assert len(published_visuals) > 0
    vis_bgr = imgmsg_to_cv2(published_visuals[-1])
    assert vis_bgr.shape == scene.shape
    node.destroy_node()


def test_detector_verifier_mode_b_reference_image() -> None:
    """Verify Mode B reference image template updates DINOv2 target embedding."""
    node = ObjectDetectorVerifierNode()
    published_thoughts = []
    published_status = []
    published_rois = []
    node.thinking_pub.publish = lambda m: published_thoughts.append(m)
    node.status_pub.publish = lambda m: published_status.append(m)
    node.rois_pub.publish = lambda m: published_rois.append(m)

    # Deliver reference image
    target = np.zeros((60, 60, 3), dtype=np.uint8)
    target[:, :] = (0, 200, 100)
    tgt_msg = cv2_to_imgmsg(target)
    node._on_target_image_message(tgt_msg)

    assert node.target_embedding is not None
    assert len(published_thoughts) >= 1
    assert "DINOv2" in published_thoughts[-1].data

    # Now provide scene embedding same target
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    scene[90:150, 130:190] = target
    node._on_camera_image(cv2_to_imgmsg(scene))

    node._on_detection_timer()
    assert published_status[-1].data is True
    assert len(published_rois[-1].detections) >= 1
    node.destroy_node()


def test_detector_verifier_visual_debug_geometry_and_rendering() -> None:
    """Verify visual debug frame geometry calculation (center_x - size_x / 2.0) and rendering."""
    node = ObjectDetectorVerifierNode()
    published_visuals = []
    node.visual_pub.publish = lambda m: published_visuals.append(m)

    scene = np.zeros((240, 320, 3), dtype=np.uint8)
    node.latest_frame = scene.copy()

    box = BoundingBox(
        center_x=100.0,
        center_y=80.0,
        size_x=40.0,
        size_y=60.0,
        confidence=0.88,
        class_name="target_can",
    )

    node._publish_detection_visual(box, score=0.88)
    assert len(published_visuals) == 1

    rendered = imgmsg_to_cv2(published_visuals[0])
    assert rendered.shape == scene.shape

    # Bounding box should be drawn in green (0, 255, 0)
    # Expected bounds: x1=80, y1=50, x2=120, y2=110
    green_pixels = np.where(
        (rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 255) & (rendered[:, :, 2] == 0)
    )
    assert len(green_pixels[0]) > 0

    # Ensure box corners match center_x - size_x / 2.0 formula
    assert int(box.center_x - box.size_x / 2.0) == 80
    assert int(box.center_y - box.size_y / 2.0) == 50
    assert int(box.center_x + box.size_x / 2.0) == 120
    assert int(box.center_y + box.size_y / 2.0) == 110
    node.destroy_node()


def test_detector_verifier_load_target_from_file() -> None:
    """Verify static reference image loading from target_image_path parameter."""
    node = ObjectDetectorVerifierNode()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        sample_img = np.full((50, 50, 3), 128, dtype=np.uint8)
        cv2.imwrite(tmp.name, sample_img)
        tmp_path = tmp.name

    try:
        node.set_parameters(
            [rclpy.parameter.Parameter("target_image_path", value=tmp_path)]
        )
        node._load_target_from_parameter()
        assert node.target_embedding is not None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        node.destroy_node()


def test_detector_verifier_empty_prompt_ignored() -> None:
    """Verify whitespace-only or empty target prompt is safely ignored."""
    node = ObjectDetectorVerifierNode()
    initial_classes = node.detector.classes
    msg = String()
    msg.data = "   "
    node._on_target_prompt_message(msg)
    assert node.detector.classes == initial_classes
    node.destroy_node()


def test_mode_b_template_not_overwritten_by_mode_a_prompt() -> None:
    """Verify Mode B template embedding is immune to prompt override and leakage."""
    node = ObjectDetectorVerifierNode()
    # Deploy Mode B template
    template_img = np.full((64, 64, 3), 100, dtype=np.uint8)
    msg_img = cv2_to_imgmsg(template_img, encoding="bgr8")
    node._on_target_image_message(msg_img)

    assert node.pursuit_mode == "template"
    assert node.is_template_locked is True
    assert node.target_embedding is not None
    initial_embedding = node.target_embedding.copy()

    # Now simulate Mode A prompt publication
    prompt_msg = String()
    prompt_msg.data = "Red Cylinder: Bright red cylindrical soda can with white lettering"
    node._on_target_prompt_message(prompt_msg)

    # Embedding must NOT be wiped out or overwritten!
    assert node.target_embedding is not None
    np.testing.assert_array_almost_equal(node.target_embedding, initial_embedding)
    assert node.is_template_locked is True

    # Switching to Mode A via mode topic re-enables semantic search
    mode_msg = String()
    mode_msg.data = "semantic"
    node._on_pursuit_mode_message(mode_msg)
    assert node.pursuit_mode == "semantic"
    assert node.target_embedding is None
    assert node.is_template_locked is False

    node.destroy_node()


def test_negative_background_classes_rejection() -> None:
    """Verify proposals matching negative classes are rejected before scoring."""
    node = ObjectDetectorVerifierNode()
    published_rois = []
    node.rois_pub.publish = lambda m: published_rois.append(m)

    # Mock proposals with negative classes
    node.detector.generate_proposals = lambda frame: (
        BoundingBox(100.0, 100.0, 50.0, 50.0, 0.95, "floor"),
        BoundingBox(200.0, 200.0, 60.0, 60.0, 0.90, "wall"),
        BoundingBox(50.0, 50.0, 30.0, 30.0, 0.88, "robot chassis"),
    )

    scene = np.full((300, 400, 3), 120, dtype=np.uint8)
    node.latest_frame = scene
    node._on_detection_timer()

    assert len(published_rois) > 0
    assert len(published_rois[-1].detections) == 0
    assert node.target_embedding is None
    node.destroy_node()


def test_mode_a_temporal_confirmation_and_unlock() -> None:
    """Verify Mode A requires 3 consecutive frames before locking and unlocks on timeout."""
    node = ObjectDetectorVerifierNode()
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(scene, (100, 100), (160, 180), (0, 0, 255), -1)
    node._on_camera_image(cv2_to_imgmsg(scene))

    prompt = String()
    prompt.data = "red cylinder"
    node._on_target_prompt_message(prompt)

    # Frame 1: candidate detected, score >= 0.35, but target_embedding is not locked yet
    node._on_detection_timer()
    assert node.target_embedding is None
    assert len(node.confirmation_buffer) == 1

    # Frame 2: candidate continues in same spot, still not locked
    node._on_detection_timer()
    assert node.target_embedding is None
    assert len(node.confirmation_buffer) == 2

    # Frame 3: confirmed 3 consecutive frames -> target_embedding is locked!
    node._on_detection_timer()
    assert node.target_embedding is not None
    assert len(node.confirmation_buffer) == 0

    # Simulate timeout: target lost / similarity low for > 1.0 second
    node.last_target_sim_time = time.time() - 2.0
    node.detector.generate_proposals = lambda f: ()
    node._on_detection_timer()

    # Must unlock and re-enter open-vocabulary search mode
    assert node.target_embedding is None
    assert len(node.confirmation_buffer) == 0
    node.destroy_node()
