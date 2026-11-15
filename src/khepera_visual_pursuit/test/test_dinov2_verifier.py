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

"""Unit tests for DINOv2 visual verification engine and dual-threshold logic."""

import numpy as np
import pytest

from khepera_visual_pursuit.dinov2_verifier import (
    DEFAULT_EMBEDDING_DIM,
    DINOv2Verifier,
    VerificationResult,
)


def test_verification_result_structure() -> None:
    """Verify attributes of VerificationResult dataclass."""
    res = VerificationResult(
        is_verified=True,
        best_similarity=0.88,
        margin=0.22,
        best_candidate_index=1,
    )
    assert res.is_verified is True
    assert res.best_similarity == 0.88
    assert res.margin == 0.22
    assert res.best_candidate_index == 1


def test_cosine_similarity_edge_cases() -> None:
    """Test cosine similarity on collinear, orthogonal, and zero-norm vectors."""
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vec_zero = np.zeros((3,), dtype=np.float32)

    assert pytest.approx(DINOv2Verifier.calculate_cosine_similarity(vec_a, vec_b), 1e-5) == 1.0
    assert pytest.approx(DINOv2Verifier.calculate_cosine_similarity(vec_a, vec_c), 1e-5) == 0.0
    assert pytest.approx(DINOv2Verifier.calculate_cosine_similarity(vec_a, -vec_b), 1e-5) == -1.0
    assert DINOv2Verifier.calculate_cosine_similarity(vec_a, vec_zero) == 0.0


def test_extract_embedding_fallback() -> None:
    """Validate descriptor extraction on synthetic color patches."""
    verifier = DINOv2Verifier()
    patch = np.zeros((40, 40, 3), dtype=np.uint8)
    patch[:, :] = (0, 120, 255)

    embedding = verifier.extract_embedding(patch)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == DEFAULT_EMBEDDING_DIM
    norm = np.linalg.norm(embedding)
    assert pytest.approx(norm, 1e-4) == 1.0

    empty_emb = verifier.extract_embedding(np.zeros((0, 0, 3), dtype=np.uint8))
    assert empty_emb.shape[0] == DEFAULT_EMBEDDING_DIM


def test_verify_candidates_single_pass() -> None:
    """Ensure single candidate exceeding tau_verify is accepted."""
    verifier = DINOv2Verifier(tau_verify=0.78, tau_margin=0.15)
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidate = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    candidate /= np.linalg.norm(candidate)

    result = verifier.verify_candidates((candidate,), target)
    assert result.is_verified is True
    assert result.best_candidate_index == 0
    assert result.best_similarity >= 0.78


def test_verify_candidates_single_fail() -> None:
    """Ensure candidate below tau_verify is rejected."""
    verifier = DINOv2Verifier(tau_verify=0.78, tau_margin=0.15)
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidate = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    candidate /= np.linalg.norm(candidate)

    result = verifier.verify_candidates((candidate,), target)
    assert result.is_verified is False
    assert result.best_candidate_index == -1


def test_verify_candidates_dual_threshold_margin() -> None:
    """Test candidate selection requiring both high similarity and distinct margin."""
    verifier = DINOv2Verifier(tau_verify=0.78, tau_margin=0.15)
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Candidate 1: 0.90 similarity, Candidate 2: 0.65 similarity (margin = 0.25 > 0.15)
    cand_1 = np.array([0.90, 0.435, 0.0], dtype=np.float32)
    cand_1 /= np.linalg.norm(cand_1)
    cand_2 = np.array([0.65, 0.76, 0.0], dtype=np.float32)
    cand_2 /= np.linalg.norm(cand_2)

    res_pass = verifier.verify_candidates((cand_1, cand_2), target)
    assert res_pass.is_verified is True
    assert res_pass.best_candidate_index == 0
    assert res_pass.margin >= 0.15

    # Ambiguous distractor: Candidate 1: 0.90, Candidate 2: 0.85 (margin = 0.05 < 0.15)
    cand_distractor = np.array([0.85, 0.526, 0.0], dtype=np.float32)
    cand_distractor /= np.linalg.norm(cand_distractor)

    res_fail = verifier.verify_candidates((cand_1, cand_distractor), target)
    assert res_fail.is_verified is False
    assert res_fail.best_candidate_index == -1


def test_verify_candidates_empty() -> None:
    """Verify handling when no candidates are provided."""
    verifier = DINOv2Verifier()
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    res = verifier.verify_candidates((), target)
    assert res.is_verified is False
    assert res.best_candidate_index == -1


def test_dinov2_device_and_fp16() -> None:
    """Verify DINOv2 verifier device configuration and FP16 flag."""
    verifier_cpu = DINOv2Verifier(device="cpu")
    assert verifier_cpu.device_str == "cpu"
    assert verifier_cpu._is_half is False


def test_dinov2_oom_fallback() -> None:
    """Ensure CUDA OutOfMemoryError during DINOv2 forward falls back gracefully."""
    from unittest.mock import MagicMock
    import torch

    verifier = DINOv2Verifier(device="cpu")
    mock_model = MagicMock()
    mock_model.forward_features.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM test")
    mock_model.parameters.return_value = iter([torch.zeros(1, device="cpu")])
    verifier._model = mock_model
    verifier._is_half = True

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    emb = verifier.extract_embedding(crop)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == DEFAULT_EMBEDDING_DIM
    assert verifier._is_half is False
