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

"""Integration tests for LightGlueTrackerNode (Node 2)."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from khepera_visual_pursuit.image_converter import cv2_to_imgmsg
from khepera_visual_pursuit.lightglue_tracker_node import LightGlueTrackerNode
from khepera_visual_pursuit.ros_compat import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    String,
    rclpy,
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_init_teardown():
    """Manage ROS 2 context lifecycle for tracker node tests."""
    if rclpy is not None and hasattr(rclpy, "ok"):
        if not rclpy.ok():
            rclpy.init()
        yield
        if rclpy.ok():
            rclpy.shutdown()
    else:
        yield


def test_lightglue_tracker_node_initialization() -> None:
    """Verify tracker node initialization, parameters, and topics."""
    node = LightGlueTrackerNode()
    assert node.get_name() == "lightglue_tracker_node"
    assert node.image_sub is not None
    assert node.target_sub is not None
    assert node.prompt_sub is not None
    assert node.rois_sub is not None
    assert node.keypoints_pub is not None
    assert node.confidence_pub is not None
    assert node.has_target_lock is False
    node.destroy_node()


def test_tracker_node_timer_without_lock() -> None:
    """Ensure tracker outputs 0.0 confidence when no target has been locked."""
    node = LightGlueTrackerNode()
    published_conf = []
    node.confidence_pub.publish = lambda m: published_conf.append(m)

    node._on_tracking_timer()

    assert len(published_conf) > 0
    latest_conf = published_conf[-1]
    assert latest_conf.data == 0.0
    node.destroy_node()


def test_tracker_node_lock_and_track() -> None:
    """Validate ROI reception, reference template lock, and tracking publication."""
    node = LightGlueTrackerNode()
    published_poly = []
    published_conf = []
    node.keypoints_pub.publish = lambda m: published_poly.append(m)
    node.confidence_pub.publish = lambda m: published_conf.append(m)

    # Create scene with distinct textured target
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(scene, (120, 55), (145, 75), (255, 0, 0), -1)
    cv2.circle(scene, (165, 65), 10, (0, 255, 0), -1)
    cv2.putText(scene, "K4", (125, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(scene, (150, 75), (175, 90), (0, 255, 255), -1)

    img_msg = cv2_to_imgmsg(scene)
    node._on_camera_image(img_msg)

    # Deliver verified ROI encompassing the textured target
    rois_msg = Detection2DArray()
    detection = Detection2D()
    detection.bbox.center.position.x = 150.0
    detection.bbox.center.position.y = 75.0
    detection.bbox.size_x = 70.0
    detection.bbox.size_y = 50.0

    hypo = ObjectHypothesisWithPose()
    hypo.hypothesis.score = 0.92
    hypo.hypothesis.class_id = "target object"
    detection.results.append(hypo)
    rois_msg.detections.append(detection)

    node._on_candidate_rois(rois_msg)
    assert node.has_target_lock is True

    # Execute tracking cycle
    node._on_tracking_timer()

    assert len(published_poly) > 0
    latest_poly = published_poly[-1]
    assert len(latest_poly.polygon.points) == 4

    assert len(published_conf) > 0
    latest_conf = published_conf[-1]
    assert latest_conf.data > 0.0
    node.destroy_node()


def test_tracker_node_mode_b_direct_target_image() -> None:
    """Verify Mode B direct target template initialization via target_image topic."""
    node = LightGlueTrackerNode()

    # Create distinct textured template
    template = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (35, 30), (255, 0, 0), -1)
    cv2.circle(template, (60, 25), 10, (0, 255, 0), -1)
    cv2.putText(template, "K4", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tgt_msg = cv2_to_imgmsg(template)
    node._on_target_image(tgt_msg)

    assert node.has_target_lock is True
    assert node.tracker.has_reference is True

    # In Mode B, receiving a prompt should not reset tracker lock
    prompt_msg = String()
    prompt_msg.data = "new object"
    node._on_target_prompt(prompt_msg)
    assert node.has_target_lock is True

    # Switching to Mode A resets tracker lock
    mode_msg = String()
    mode_msg.data = "semantic"
    node._on_pursuit_mode(mode_msg)
    node._on_target_prompt(prompt_msg)

    assert node.has_target_lock is False
    assert node.tracker.has_reference is False
    node.destroy_node()


def test_tracker_node_load_from_parameter_file() -> None:
    """Verify reference template loading from target_image_path parameter."""
    node = LightGlueTrackerNode()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (35, 30), (255, 0, 0), -1)
        cv2.circle(template, (60, 25), 10, (0, 255, 0), -1)
        cv2.putText(template, "K4", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(tmp.name, template)
        tmp_path = tmp.name

    try:
        node.set_parameters([rclpy.parameter.Parameter("target_image_path", value=tmp_path)])
        node._load_target_from_parameter()
        assert node.has_target_lock is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        node.destroy_node()


def test_select_best_roi_priority() -> None:
    """Ensure candidate ROI with highest hypothesis score is chosen."""
    node = LightGlueTrackerNode()

    det_low = Detection2D()
    hypo_low = ObjectHypothesisWithPose()
    hypo_low.hypothesis.score = 0.5
    det_low.results.append(hypo_low)

    det_high = Detection2D()
    hypo_high = ObjectHypothesisWithPose()
    hypo_high.hypothesis.score = 0.95
    det_high.results.append(hypo_high)

    best = node._select_best_roi([det_low, det_high])
    assert best is det_high
    node.destroy_node()
