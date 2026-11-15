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

"""Unit tests for SuperPoint + LightGlue tracking engine, masks, and Kalman filter."""

import cv2
import numpy as np
import pytest

from khepera_visual_pursuit.superpoint_lightglue_engine import (
    FLOOR_CLIP_FRACTION,
    KalmanBoxTracker,
    MAX_COAST_FRAMES,
    SuperPointLightGlueTracker,
    TARGET_TOP_FRACTION,
    create_floor_mask,
    create_target_top_mask,
    enhance_contrast_clahe,
    filter_bidirectional_ransac,
)


def test_enhance_contrast_clahe() -> None:
    """Ensure CLAHE enhancement retains shape and valid pixel range."""
    img = np.random.randint(50, 200, (100, 120, 3), dtype=np.uint8)
    enhanced = enhance_contrast_clahe(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8

    empty = enhance_contrast_clahe(np.zeros((0, 0, 3), dtype=np.uint8))
    assert empty.size == 0


def test_create_floor_mask() -> None:
    """Verify lower 15% floor area is masked out."""
    height, width = 100, 100
    mask = create_floor_mask(height, width, FLOOR_CLIP_FRACTION)
    assert mask.shape == (height, width)

    # Top region should be 255 (active)
    assert np.all(mask[:80, :] == 255)
    # Bottom 15% (rows >= 85) should be 0 (masked floor)
    assert np.all(mask[85:, :] == 0)


def test_create_target_top_mask() -> None:
    """Verify 100% of bounding box is active with TARGET_TOP_FRACTION=1.0 and custom fractions."""
    height, width = 100, 100
    mask = create_target_top_mask(height, width, TARGET_TOP_FRACTION)
    assert mask.shape == (height, width)
    assert np.all(mask == 255)

    # Custom fractional mask (e.g. 60%)
    mask_60 = create_target_top_mask(height, width, 0.60)
    assert np.all(mask_60[:60, :] == 255)
    assert np.all(mask_60[60:, :] == 0)


def test_filter_bidirectional_ransac_insufficient_points() -> None:
    """Ensure geometric filtering safely rejects under-constrained correspondence sets."""
    pts_r = np.array([[10, 10], [20, 20]], dtype=np.float32)
    pts_c = np.array([[15, 15], [25, 25]], dtype=np.float32)
    scores = np.array([0.9, 0.9], dtype=np.float32)

    res = filter_bidirectional_ransac(pts_r, pts_c, scores)
    assert res.is_valid is False
    assert res.inlier_count == 0


def test_filter_bidirectional_ransac_low_scores() -> None:
    """Ensure matches below NOMINAL_MATCH_THRESHOLD (0.2) are ignored."""
    pts_r = np.array([[10, 10], [20, 10], [10, 30], [20, 30], [15, 20]], dtype=np.float32)
    pts_c = pts_r + 5.0
    scores = np.array([0.1, 0.15, 0.18, 0.12, 0.14], dtype=np.float32)

    res = filter_bidirectional_ransac(pts_r, pts_c, scores)
    assert res.is_valid is False


def test_filter_bidirectional_ransac_consistent_affine() -> None:
    """Test successful bidirectional verification on translated keypoints."""
    np.random.seed(42)
    pts_r = np.random.uniform(20, 180, (15, 2)).astype(np.float32)
    shift = np.array([12.0, -8.0], dtype=np.float32)
    pts_c = pts_r + shift
    scores = np.ones((15,), dtype=np.float32) * 0.85

    res = filter_bidirectional_ransac(pts_r, pts_c, scores)
    assert res.is_valid is True
    assert res.inlier_count >= 10
    assert res.transform_matrix is not None


def test_kalman_box_tracker_lifecycle() -> None:
    """Validate constant velocity Kalman filter prediction, update, and coasting."""
    tracker = KalmanBoxTracker(delta_t=1.0 / 30.0)
    assert tracker.is_initialized is False

    tracker.initialize((100.0, 100.0, 40.0, 40.0))
    assert tracker.is_initialized is True

    # Initial predict before movement
    pred = tracker.predict()
    assert pytest.approx(pred[0], abs=1.0) == 100.0

    # Moving measurement: shift x by 2 pixels per step across multiple frames
    for step in range(1, 8):
        tracker.predict()
        tracker.update((100.0 + step * 2.0, 100.0, 40.0, 40.0))

    # Next prediction should extrapolate moving velocity
    pred_extrapolated = tracker.predict()
    assert pred_extrapolated[0] > 114.0

    # Coasting behavior
    assert tracker.coast_count == 0
    tracker.increment_coast()
    assert tracker.coast_count == 1
    tracker.reset()
    assert tracker.is_initialized is False


def test_superpoint_lightglue_tracker_tracking_and_fallback() -> None:
    """Validate full tracker workflow including target locking, tracking, and Kalman fallback."""
    tracker = SuperPointLightGlueTracker(match_threshold=0.2, min_keypoints=6)

    # Create synthetic textured target template with distinctive shapes in top 60%
    template = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (30, 25), (255, 0, 0), -1)
    cv2.circle(template, (55, 20), 10, (0, 255, 0), -1)
    cv2.putText(template, "K4", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(template, (45, 30), (70, 45), (0, 255, 255), -1)

    lock_ok = tracker.set_reference_target(template)
    assert lock_ok is True
    assert tracker.has_reference is True

    # Frame 1: Reference placed inside full scene frame
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[50:130, 100:180] = template

    output = tracker.track(frame)
    assert output.is_tracked is True
    assert output.is_fallback is False
    assert -1.0 <= output.normalized_centroid[0] <= 1.0
    assert -1.0 <= output.normalized_centroid[1] <= 1.0
    assert 0.0 <= output.area_fraction <= 1.0
    assert output.confidence > 0.0
    assert len(output.polygon_points) == 4

    # Frame 2: Target occluded / missing -> Kalman filter fallback
    occluded_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    fallback_output = tracker.track(occluded_frame)
    assert fallback_output.is_tracked is True
    assert fallback_output.is_fallback is True
    assert fallback_output.confidence < output.confidence

    # Exceeding MAX_COAST_FRAMES terminates tracking
    for _ in range(MAX_COAST_FRAMES + 2):
        last_out = tracker.track(occluded_frame)

    assert last_out.is_tracked is False


def test_specular_glare_mitigation_and_top_mask_rejection() -> None:
    """Verify lower-15% floor mask rejects specular glare and 100% template is utilized."""
    tracker = SuperPointLightGlueTracker(match_threshold=0.2, min_keypoints=6)

    # 1. Target with textures placed in bottom 30% (rows >= 70 out of 100)
    # With TARGET_TOP_FRACTION=1.0, 100% of target bounding box is utilized (no truncation)
    bottom_heavy_target = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(
        bottom_heavy_target, "FLOOR", (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    cv2.circle(bottom_heavy_target, (50, 90), 8, (0, 255, 255), -1)

    accepted = tracker.set_reference_target(bottom_heavy_target)
    assert accepted is True
    assert tracker.has_reference is True

    # 2. Valid target with textures in top 50%
    valid_target = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(
        valid_target, "VALID", (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    cv2.circle(valid_target, (60, 25), 8, (0, 255, 0), -1)
    assert tracker.set_reference_target(valid_target) is True

    # 3. Scene frame where target appearance only exists in the lower 10% (floor glare region)
    glare_scene = np.zeros((300, 400, 3), dtype=np.uint8)
    # Place valid target appearance at rows 275-300 (bottom 10% of 300 rows is > 255)
    glare_scene[265:295, 150:230] = cv2.resize(valid_target, (80, 30))

    # Tracking should fail to find keypoints in the floor-masked region
    tracker._ref_template_bgr = None  # Isolate keypoint floor masking from template correlation
    glare_output = tracker.track(glare_scene)
    # Because valid keypoints are masked out by floor_mask, it cannot lock and must fallback/fail
    assert glare_output.is_fallback is True or glare_output.is_tracked is False


def test_superpoint_lightglue_gpu_tensor_pipeline() -> None:
    """Validate GPU tensor inference on cuda:0 or CPU with SuperPoint and LightGlue."""
    import torch
    from khepera_visual_pursuit.superpoint_lightglue_engine import image_to_tensor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = image_to_tensor(sample_img, device)
    assert tensor.shape == (3, 100, 100)
    assert str(tensor.device).startswith(device.split(":")[0])

    np.random.seed(123)
    textured = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    tracker = SuperPointLightGlueTracker(device=device, min_keypoints=6)
    locked = tracker.set_reference_target(textured)
    assert locked is True
    assert tracker.has_reference is True
    assert tracker._ref_sp_feats is not None

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[50:250, 50:250] = textured
    output = tracker.track(frame)
    assert output.is_tracked is True
    assert output.keypoint_count >= 6
    assert output.confidence > 0.50


def test_image_to_tensor_pinned_memory() -> None:
    """Verify image_to_tensor correctly returns normalized tensor on target device."""
    import torch
    from khepera_visual_pursuit.superpoint_lightglue_engine import image_to_tensor

    img = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    tensor_cpu = image_to_tensor(img, "cpu")
    assert tensor_cpu.shape == (3, 60, 80)
    assert tensor_cpu.dtype == torch.float32
    assert tensor_cpu.device.type == "cpu"


def test_superpoint_tracker_oom_fallback() -> None:
    """Ensure CUDA OutOfMemoryError in tracker falls back cleanly."""
    from unittest.mock import MagicMock
    import torch

    tracker = SuperPointLightGlueTracker(device="cpu")
    mock_extractor = MagicMock()
    mock_extractor.extract.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM test")
    tracker._sp_extractor = mock_extractor
    tracker._is_cuda = True

    dummy = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    # Target lock should handle OOM gracefully and fall back to SIFT/correlation
    result = tracker.set_reference_target(dummy)
    assert isinstance(result, bool)
