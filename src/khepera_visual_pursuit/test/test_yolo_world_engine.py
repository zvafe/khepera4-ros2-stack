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

"""Unit tests for YOLO-World proposal engine and BoundingBox representation."""

import cv2
import numpy as np

from khepera_visual_pursuit.yolo_world_engine import (
    NEGATIVE_CLASSES,
    BoundingBox,
    YoloWorldEngine,
    _is_featureless_surface,
    calculate_weighted_semantic_score,
)


def test_bounding_box_properties() -> None:
    """Verify BoundingBox integer boundary coordinate calculations."""
    box = BoundingBox(
        center_x=100.0,
        center_y=80.0,
        size_x=40.0,
        size_y=30.0,
        confidence=0.85,
        class_name="cylinder",
    )
    assert box.x_min == 80
    assert box.y_min == 65
    assert box.x_max == 120
    assert box.y_max == 95
    assert box.confidence == 0.85
    assert box.class_name == "cylinder"


def test_bounding_box_clamping_non_negative() -> None:
    """Ensure coordinates do not drop below zero when box exceeds left/top boundary."""
    box = BoundingBox(
        center_x=10.0,
        center_y=10.0,
        size_x=50.0,
        size_y=50.0,
        confidence=0.5,
        class_name="target",
    )
    assert box.x_min == 0
    assert box.y_min == 0
    assert box.x_max == 35
    assert box.y_max == 35


def test_yolo_engine_initialization() -> None:
    """Verify engine initialization and class configuration."""
    engine = YoloWorldEngine(confidence_threshold=0.35)
    assert engine.confidence_threshold == 0.35

    new_classes = ("robot", "obstacle")
    engine.set_classes(new_classes)
    assert all(c in engine.classes for c in new_classes)
    assert all(neg in engine.classes for neg in NEGATIVE_CLASSES)


def test_generate_proposals_empty_input() -> None:
    """Ensure engine returns empty tuple on None or zero-sized frames."""
    engine = YoloWorldEngine()
    assert engine.generate_proposals(None) == ()
    assert engine.generate_proposals(np.zeros((0, 0, 3), dtype=np.uint8)) == ()


def test_generate_proposals_with_contours() -> None:
    """Test heuristic proposal extraction on synthetic image with clear contours."""
    engine = YoloWorldEngine()
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 80), (220, 200), (255, 255, 255), -1)

    proposals = engine.generate_proposals(test_image)
    assert len(proposals) >= 1
    best_box = proposals[0]
    assert 90 <= best_box.center_x <= 170
    assert 70 <= best_box.center_y <= 150
    assert best_box.confidence > 0.0


def test_generate_proposals_blank_fallback() -> None:
    """Ensure fallback central proposal is produced when no edges exist."""
    engine = YoloWorldEngine()
    blank_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    proposals = engine.generate_proposals(blank_image)
    assert len(proposals) == 1
    assert proposals[0].center_x == 100.0
    assert proposals[0].center_y == 100.0


def test_crop_roi_safety() -> None:
    """Test safe rectangular ROI extraction within and exceeding image boundaries."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:60, 20:60] = 200

    box_inside = BoundingBox(40.0, 40.0, 30.0, 30.0, 0.9, "object")
    crop = YoloWorldEngine.crop_roi(image, box_inside)
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    # Box exceeding right/bottom boundaries
    box_outside = BoundingBox(95.0, 95.0, 40.0, 40.0, 0.9, "object")
    crop_out = YoloWorldEngine.crop_roi(image, box_outside)
    assert crop_out.shape[0] <= 100
    assert crop_out.shape[1] <= 100


def test_yolo_backend_and_device_handling() -> None:
    """Verify backend identification and device configuration."""
    engine_cpu = YoloWorldEngine(device="cpu")
    assert engine_cpu.device == "cpu"
    assert engine_cpu.backend in ("cpu", "heuristic")
    assert engine_cpu._is_half is False


def test_yolo_oom_fallback() -> None:
    """Ensure CUDA OutOfMemoryError during YOLO inference falls back to CPU."""
    from unittest.mock import MagicMock
    import torch

    engine = YoloWorldEngine(device="cpu")
    mock_model = MagicMock()
    mock_model.predict.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM test")
    engine._model = mock_model
    engine.device = "cuda:0"
    engine._is_half = True

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    proposals = engine.generate_proposals(dummy_frame)
    assert isinstance(proposals, tuple)
    assert engine.device == "cpu"
    assert engine._is_half is False


def test_is_featureless_surface_filters_grey_wall_and_accepts_red_target() -> None:
    """Verify featureless surface filter rejects blank grey walls but accepts colored boxes."""
    # Blank grey wall: low variance, low std, no color saturation -> featureless
    grey_wall = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert _is_featureless_surface(grey_wall) is True

    # Solid red target in Gazebo: low variance, high color saturation -> NOT featureless
    red_box = np.full((64, 64, 3), (0, 0, 220), dtype=np.uint8)
    assert _is_featureless_surface(red_box) is False

    # Uniform solid blue target: NOT featureless
    blue_target = np.full((64, 64, 3), (220, 0, 0), dtype=np.uint8)
    assert _is_featureless_surface(blue_target) is False


def test_calculate_weighted_semantic_score_solid_red_box() -> None:
    """Ensure solid-colored red box in simulation is not discarded with 0 score."""
    box = BoundingBox(
        center_x=160.0,
        center_y=120.0,
        size_x=60.0,
        size_y=60.0,
        confidence=0.85,
        class_name="box",
    )
    red_crop = np.full((60, 60, 3), (0, 0, 220), dtype=np.uint8)
    score = calculate_weighted_semantic_score(
        box, target_name="red box", target_desc="", crop_bgr=red_crop
    )
    assert score >= 0.50
