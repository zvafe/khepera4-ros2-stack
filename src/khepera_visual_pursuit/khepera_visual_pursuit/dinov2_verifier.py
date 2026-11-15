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

"""DINOv2 patch token verification engine for instance identification."""

from dataclasses import dataclass
import logging
import os
from typing import Optional, Sequence, Tuple, Union
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.load.*")

logger = logging.getLogger(__name__)

NOMINAL_TAU_VERIFY: float = 0.70
NOMINAL_TAU_MARGIN: float = 0.025
DINOV2_PATCH_SIZE: int = 14
DEFAULT_EMBEDDING_DIM: int = 384


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of multi-candidate target verification."""

    is_verified: bool
    best_similarity: float
    margin: float
    best_candidate_index: int


def calculate_box_iou(box1: object, box2: object) -> float:
    """Calculate intersection-over-union between two bounding box objects."""
    x1_min = getattr(box1, "center_x", 0.0) - getattr(box1, "size_x", 0.0) / 2.0
    x1_max = getattr(box1, "center_x", 0.0) + getattr(box1, "size_x", 0.0) / 2.0
    y1_min = getattr(box1, "center_y", 0.0) - getattr(box1, "size_y", 0.0) / 2.0
    y1_max = getattr(box1, "center_y", 0.0) + getattr(box1, "size_y", 0.0) / 2.0

    x2_min = getattr(box2, "center_x", 0.0) - getattr(box2, "size_x", 0.0) / 2.0
    x2_max = getattr(box2, "center_x", 0.0) + getattr(box2, "size_x", 0.0) / 2.0
    y2_min = getattr(box2, "center_y", 0.0) - getattr(box2, "size_y", 0.0) / 2.0
    y2_max = getattr(box2, "center_y", 0.0) + getattr(box2, "size_y", 0.0) / 2.0

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h
    area1 = max(1e-6, getattr(box1, "size_x", 0.0) * getattr(box1, "size_y", 0.0))
    area2 = max(1e-6, getattr(box2, "size_x", 0.0) * getattr(box2, "size_y", 0.0))
    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 0.0 else 0.0


class DINOv2Verifier:
    """Extracts DINOv2 feature embeddings and validates instance matches."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        tau_verify: float = NOMINAL_TAU_VERIFY,
        tau_margin: float = NOMINAL_TAU_MARGIN,
        device: str = "auto",
    ) -> None:
        """Initialize verifier with model specifications and thresholds."""
        self.model_name = model_name
        self.tau_verify = tau_verify
        self.tau_margin = tau_margin
        self.device_str = device
        self._is_half: bool = False
        self._model = self._load_model()

    def _load_model(self) -> Optional[object]:
        """Load DINOv2 backbone from torch.hub with fallback to heuristic extractor."""
        try:
            import warnings
            import torch

            warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

            if self.device_str == "auto":
                target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                target_device = self.device_str
            device = torch.device(target_device)

            local_hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
            if os.path.isdir(local_hub):
                model = torch.hub.load(local_hub, self.model_name, source="local")
            else:
                model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
            model.eval()
            model.to(device)
            if device.type == "cuda":
                model.half()
                self._is_half = True
            else:
                self._is_half = False
            return model
        except (torch.cuda.OutOfMemoryError, Exception) as exc:
            logger.warning(
                "DINOv2 backbone unavailable (%s). Using fallback extractor.", exc
            )
            return None

    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compute L2-normalized feature descriptor for an image crop."""
        if (
            crop_bgr is None
            or crop_bgr.size == 0
            or crop_bgr.shape[0] < 4
            or crop_bgr.shape[1] < 4
        ):
            return np.zeros((DEFAULT_EMBEDDING_DIM,), dtype=np.float32)

        if self._model is not None:
            embedding = self._extract_with_model(crop_bgr)
            if embedding is not None and isinstance(embedding, np.ndarray):
                return embedding

        return self._extract_fallback_descriptor(crop_bgr)

    def extract_embeddings_batch(
        self, crops_bgr: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, ...]:
        """Compute L2-normalized feature descriptors for a batch of image crops."""
        if not crops_bgr:
            return ()

        valid_crops = [
            c if (c is not None and c.size > 0 and c.shape[0] >= 4 and c.shape[1] >= 4)
            else np.zeros((32, 32, 3), dtype=np.uint8)
            for c in crops_bgr
        ]

        if self._model is not None:
            embeddings = self._extract_with_model(valid_crops)
            if embeddings is not None and isinstance(embeddings, tuple):
                return embeddings

        return tuple(self._extract_fallback_descriptor(c) for c in valid_crops)

    def _extract_with_model(
        self, crops: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Extract deep visual tokens using DINOv2 PyTorch backbone with stacked batching."""
        try:
            import torch

            is_single = isinstance(crops, np.ndarray) and crops.ndim == 3
            crop_list = [crops] if is_single else list(crops)
            if not crop_list:
                return None if is_single else ()

            device = next(self._model.parameters()).device
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            tensors = []
            for crop in crop_list:
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                normalized = (rgb - mean) / std
                tensors.append(torch.from_numpy(normalized).permute(2, 0, 1))

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            if self._is_half and device.type == "cuda":
                batch_tensor = batch_tensor.half()

            with torch.inference_mode():
                out = self._model.forward_features(batch_tensor)
                cls_token = out["x_norm_clstoken"]
                if hasattr(cls_token, "cpu"):
                    cls_arr = cls_token.cpu().float().numpy()
                else:
                    cls_arr = np.asarray(cls_token, dtype=np.float32)

                if cls_arr.ndim == 1:
                    cls_arr = cls_arr.reshape(1, -1)

                norms = np.linalg.norm(cls_arr, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                normalized = cls_arr / norms

                if is_single:
                    return normalized[0]
                return tuple(normalized[i] for i in range(len(normalized)))
        except torch.cuda.OutOfMemoryError as oom_exc:
            logger.error(
                "CUDA OOM during DINOv2 embedding extraction: %s. Falling back to CPU.",
                oom_exc,
            )
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._fallback_to_cpu()
            return None
        except Exception as exc:
            logger.error("DINOv2 forward failed: %s", exc)
            return None

    def _fallback_to_cpu(self) -> None:
        """Migrate DINOv2 model to CPU and float32 precision."""
        try:
            import torch

            if self._model is not None:
                self._model.to(torch.device("cpu")).float()
                self._is_half = False
                self.device_str = "cpu"
                logger.info("DINOv2 successfully migrated to CPU.")
        except Exception as exc:
            logger.warning("Failed to migrate DINOv2 to CPU: %s. Disabling model.", exc)
            self._model = None

    def _extract_fallback_descriptor(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compute deterministic multi-scale normalized feature vector for testing."""
        resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [128], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [128], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [128], [0, 256]).flatten()
        descriptor = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(descriptor)
        return descriptor / max(norm, 1e-8)

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity S = (a . b) / (||a|| * ||b||)."""
        dot_product = float(np.dot(vec_a, vec_b))
        norm_product = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if norm_product < 1e-8:
            return 0.0
        return float(np.clip(dot_product / norm_product, -1.0, 1.0))

    def verify_candidates(
        self,
        candidate_embeddings: Tuple[np.ndarray, ...],
        target_embedding: np.ndarray,
        candidate_boxes: Optional[Tuple[object, ...]] = None,
    ) -> VerificationResult:
        """
        Evaluate candidates against target embedding using dual-threshold criteria.

        Criteria:
        1. tau_1 > tau_verify (nominal 0.70)
        2. tau_1 - tau_2 > tau_margin against distinct non-overlapping distractors.
        """
        if not candidate_embeddings:
            return VerificationResult(
                is_verified=False,
                best_similarity=0.0,
                margin=0.0,
                best_candidate_index=-1,
            )

        similarities = [
            self.calculate_cosine_similarity(emb, target_embedding)
            for emb in candidate_embeddings
        ]

        indexed_sims = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        best_idx, s1 = indexed_sims[0]

        if candidate_boxes is not None and len(candidate_boxes) == len(candidate_embeddings):
            best_box = candidate_boxes[best_idx]
            distractor_sims = [
                sim for i, (sim, box) in enumerate(zip(similarities, candidate_boxes))
                if i != best_idx and calculate_box_iou(box, best_box) <= 0.20
            ]
            if distractor_sims:
                margin = s1 - max(distractor_sims)
                has_competing_distractors = True
            else:
                margin = s1
                has_competing_distractors = False
            passes_margin = (
                (not has_competing_distractors)
                or (margin >= self.tau_margin)
                or (s1 >= 0.95)
            )
        else:
            if len(indexed_sims) > 1:
                _, s2 = indexed_sims[1]
                margin = s1 - s2
            else:
                margin = s1
            passes_margin = (len(indexed_sims) == 1) or (margin >= self.tau_margin)

        passes_absolute = s1 >= self.tau_verify
        is_verified = bool(passes_absolute and passes_margin)

        return VerificationResult(
            is_verified=is_verified,
            best_similarity=s1,
            margin=margin,
            best_candidate_index=best_idx if is_verified else -1,
        )
