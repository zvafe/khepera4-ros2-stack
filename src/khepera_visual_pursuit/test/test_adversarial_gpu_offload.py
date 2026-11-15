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

"""Adversarial stress and breaking tests for GPU offloading and fallback resilience."""

import os
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from khepera_visual_pursuit.dinov2_verifier import DINOv2Verifier
from khepera_visual_pursuit.superpoint_lightglue_engine import SuperPointLightGlueTracker
from khepera_visual_pursuit.yolo_world_engine import YoloWorldEngine


class TestAdversarialGpuOffload:
    """Stress tests designed to challenge GPU offloading pipelines and assert graceful recovery."""

    def test_corrupt_tensorrt_engine_file_recovers_gracefully(self):
        """Verify that a corrupted or truncated .engine file triggers automatic fallback."""
        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as f:
            f.write(b"CORRUPT_HEADER_NOT_A_VALID_TENSORRT_ENGINE_BYTECODE_TRUNCATED")
            corrupt_path = f.name

        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            engine = YoloWorldEngine(model_name=corrupt_path, device=device)
            assert engine is not None
            dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            proposals = engine.generate_proposals(dummy_frame)
            assert isinstance(proposals, tuple)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_yolo_world_cuda_oom_recovers_to_fallback(self):
        """Verify that CUDA OutOfMemoryError in YOLO-World is caught and falls back."""
        engine = YoloWorldEngine(device="cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        if engine._model is not None:
            oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
            with patch.object(engine._model, "predict", side_effect=oom_exc):
                proposals = engine.generate_proposals(dummy_frame)
                assert isinstance(proposals, tuple)

    def test_dinov2_cuda_oom_recovers_to_histogram_embedding(self):
        """Verify that CUDA OutOfMemoryError in DINOv2 triggers fallback descriptor."""
        verifier = DINOv2Verifier()
        dummy_crop = np.full((100, 100, 3), 100, dtype=np.uint8)

        if verifier._model is not None:
            oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
            with patch.object(verifier._model, "forward_features", side_effect=oom_exc):
                emb = verifier.extract_embedding(dummy_crop)
                assert isinstance(emb, np.ndarray)
                assert emb.shape == (384,)
                assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    def test_lightglue_cuda_oom_recovers_to_fallback(self):
        """Verify that CUDA OutOfMemoryError in LightGlue triggers graceful fallback."""
        tracker = SuperPointLightGlueTracker()
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracker.set_reference_target(dummy_frame[100:200, 100:200])

        oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        with patch("torch.autocast", side_effect=oom_exc):
            out = tracker.track(dummy_frame)
            assert hasattr(out, "is_tracked")
            assert hasattr(out, "confidence")

    @pytest.mark.parametrize("bad_shape", [
        (0, 0, 3),
        (1, 1, 3),
        (10, 0, 3),
        (0, 10, 3),
    ])
    def test_degenerate_zero_and_micro_frames(self, bad_shape):
        """Ensure all perception engines reject degenerate zero or micro input frames."""
        bad_img = np.zeros(bad_shape, dtype=np.uint8)

        engine = YoloWorldEngine()
        proposals = engine.generate_proposals(bad_img)
        assert proposals == ()

        verifier = DINOv2Verifier()
        emb = verifier.extract_embedding(bad_img)
        assert isinstance(emb, np.ndarray)
        assert np.all(emb == 0.0)

        tracker = SuperPointLightGlueTracker()
        assert tracker.set_reference_target(bad_img) is False
        out = tracker.track(bad_img)
        assert out.is_tracked is False

    def test_nan_and_inf_frame_sanitization(self):
        """Verify frames containing NaNs or Infs do not cause kernel faults."""
        nan_img = np.full((100, 100, 3), np.nan, dtype=np.float32)
        nan_img_uint8 = np.nan_to_num(nan_img).astype(np.uint8)

        verifier = DINOv2Verifier()
        emb = verifier.extract_embedding(nan_img_uint8)
        assert not np.isnan(emb).any()
        assert not np.isinf(emb).any()

    def test_adversarial_fir_jpeg_negative_rejection(self):
        """Verify that fir.jpeg does not produce false positives against arena objects."""
        fir_path = "/home/andy/Downloads/fir.jpeg"
        if not os.path.exists(fir_path):
            pytest.skip("fir.jpeg not available on host")

        fir_img = cv2.imread(fir_path)
        assert fir_img is not None

        verifier = DINOv2Verifier(tau_verify=0.78, tau_margin=0.15)
        fir_emb = verifier.extract_embedding(fir_img)

        red_cylinder = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.circle(red_cylinder, (60, 60), 45, (0, 0, 255), -1)

        gray_wall = np.full((120, 120, 3), 190, dtype=np.uint8)

        red_emb = verifier.extract_embedding(red_cylinder)
        wall_emb = verifier.extract_embedding(gray_wall)

        verif = verifier.verify_candidates((red_emb, wall_emb), fir_emb)
        assert verif.is_verified is False
        assert verif.best_similarity < 0.60

        tracker = SuperPointLightGlueTracker()
        tracker.set_reference_target(fir_img)
        arena_scene = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.circle(arena_scene, (320, 240), 60, (0, 0, 255), -1)

        track_out = tracker.track(arena_scene)
        assert track_out.is_tracked is False
        assert track_out.confidence < 0.20

    def test_backward_compatibility_forced_cpu_execution(self):
        """Verify that forcing device='cpu' operates without any CUDA requirements."""
        engine = YoloWorldEngine(device="cpu")
        assert engine.device == "cpu"
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        proposals = engine.generate_proposals(frame)
        assert isinstance(proposals, tuple)

        tracker = SuperPointLightGlueTracker(device="cpu")
        assert tracker.device == "cpu"
        tracker.set_reference_target(frame[50:150, 50:150])
        out = tracker.track(frame)
        assert hasattr(out, "is_tracked")
