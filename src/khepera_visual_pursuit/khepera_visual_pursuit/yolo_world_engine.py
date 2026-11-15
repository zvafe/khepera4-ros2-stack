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

"""Open-vocabulary object proposal generation engine utilizing YOLO-World."""

from dataclasses import dataclass
import logging
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_CLASSES: Tuple[str, ...] = ("target object", "cylinder", "box", "can")
NEGATIVE_CLASSES: Tuple[str, ...] = (
    "background",
    "floor",
    "wall",
    "arena boundary",
    "shadow",
    "robot chassis",
)


@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box proposal in image coordinates."""

    center_x: float
    center_y: float
    size_x: float
    size_y: float
    confidence: float
    class_name: str

    @property
    def score(self) -> float:
        """Alias for proposal confidence score."""
        return self.confidence

    @property
    def x_min(self) -> int:
        """Calculate minimum horizontal pixel coordinate."""
        return max(0, int(self.center_x - self.size_x / 2.0))

    @property
    def y_min(self) -> int:
        """Calculate minimum vertical pixel coordinate."""
        return max(0, int(self.center_y - self.size_y / 2.0))

    @property
    def x_max(self) -> int:
        """Calculate maximum horizontal pixel coordinate."""
        return max(0, int(self.center_x + self.size_x / 2.0))

    @property
    def y_max(self) -> int:
        """Calculate maximum vertical pixel coordinate."""
        return max(0, int(self.center_y + self.size_y / 2.0))


COLOR_MAP = {
    "red": [((0, 70, 40), (10, 255, 255)), ((170, 70, 40), (180, 255, 255))],
    "blue": [((100, 70, 40), (130, 255, 255))],
    "green": [((35, 70, 40), (85, 255, 255))],
    "yellow": [((20, 70, 40), (35, 255, 255))],
    "orange": [((10, 70, 40), (25, 255, 255))],
}


def _extract_target_color(text: str) -> Optional[str]:
    """Extract standard primary color keyword from target string if present."""
    text_l = text.lower()
    for ck, aliases in (
        ("yellow", ("yellow", "gold", "amber")),
        ("red", ("red", "crimson", "scarlet")),
        ("blue", ("blue", "azure", "cyan")),
        ("green", ("green", "lime", "emerald")),
        ("orange", ("orange",)),
    ):
        if any(a in text_l for a in aliases):
            return ck
    return None


def _is_featureless_surface(crop_bgr: np.ndarray) -> bool:
    """Identify uniform featureless background crops (e.g. walls, floors)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    color_std = float(np.std(crop_bgr))
    return bool(lap_var < 15.0 and color_std < 10.0)


def calculate_weighted_semantic_score(
    box: BoundingBox,
    target_name: str,
    target_desc: str = "",
    crop_bgr: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate composite semantic score with 70% class name and 30% description weighting.

    Class name is the primary categorical discriminator (~70% weight) identifying object
    category and structure. Description provides attribute context (~30% weight) detailing
    color, material, and context. If no description is provided, class name carries 100% weight.
    """
    if not target_name:
        return float(box.confidence)

    if crop_bgr is not None and crop_bgr.size > 0 and _is_featureless_surface(crop_bgr):
        return 0.0

    name_lower = target_name.strip().lower()
    desc_lower = target_desc.strip().lower() if target_desc else ""
    box_cls = box.class_name.strip().lower()

    # 1. Categorical / Structural Score S_name (70% weight)
    if name_lower == box_cls:
        s_name = 1.0
    elif name_lower in box_cls or box_cls in name_lower:
        s_name = 0.90
    else:
        name_tokens = set(name_lower.replace(",", " ").split())
        cls_tokens = set(box_cls.replace(",", " ").split())
        overlap = name_tokens.intersection(cls_tokens)
        if overlap:
            s_name = 0.70 + 0.20 * (len(overlap) / max(len(name_tokens), 1))
        else:
            aspect_ratio = float(box.size_y) / max(1.0, float(box.size_x))
            if any(k in name_lower for k in ("cylinder", "can", "bottle", "person")):
                s_name = 0.70 if aspect_ratio >= 1.05 else 0.10
            elif any(k in name_lower for k in ("cube", "box", "sphere", "ball")):
                s_name = 0.70 if 0.70 <= aspect_ratio <= 1.40 else 0.10
            elif any(k in name_lower for k in ("chair", "seat")):
                s_name = 0.70 if 0.80 <= aspect_ratio <= 2.20 else 0.10
            else:
                s_name = 0.10

    # If target name specifies a color, verify presence in crop
    target_color = _extract_target_color(name_lower)
    if target_color and crop_bgr is not None and crop_bgr.size > 0:
        hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
        for lb, ub in COLOR_MAP[target_color]:
            mask |= cv2.inRange(hsv_crop, lb, ub)
        frac = float(np.count_nonzero(mask)) / max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
        if frac < 0.08:
            s_name *= max(0.05, frac * 5.0)

    s_name = float(np.clip(s_name * (0.50 + 0.50 * box.confidence), 0.0, 1.0))

    if not desc_lower:
        return s_name

    # 2. Attribute / Context Score S_desc (30% weight)
    s_desc = 0.20
    if crop_bgr is not None and crop_bgr.size > 0:
        hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        color_matches = []
        check_colors = [target_color] if target_color else list(COLOR_MAP.keys())
        for ck in check_colors:
            aliases = (ck,) if ck != "red" else ("red", "crimson", "scarlet")
            if (not target_color) and not any(c in desc_lower for c in aliases):
                continue
            mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
            for lb, ub in COLOR_MAP[ck]:
                mask |= cv2.inRange(hsv_crop, lb, ub)
            frac = float(np.count_nonzero(mask)) / max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
            color_matches.append(frac)

        if color_matches:
            best_frac = max(color_matches)
            if best_frac >= 0.20:
                s_desc = float(np.clip(0.60 + best_frac * 0.40, 0.60, 1.0))
            else:
                s_desc = float(np.clip(best_frac * 1.5, 0.05, 0.35))
        else:
            desc_tokens = set(desc_lower.replace(",", " ").split())
            cls_tokens = set(box_cls.replace(",", " ").split())
            s_desc = 0.85 if desc_tokens.intersection(cls_tokens) else 0.50
    else:
        desc_tokens = set(desc_lower.replace(",", " ").split())
        cls_tokens = set(box_cls.replace(",", " ").split())
        s_desc = 0.85 if desc_tokens.intersection(cls_tokens) else 0.50

    composite = 0.70 * s_name + 0.30 * s_desc
    return float(np.clip(composite, 0.0, 1.0))


class YoloWorldEngine:
    """YOLO-World proposal engine with fallback heuristics for offline testing."""

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        device: str = "auto",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize YOLO-World model engine with specified device and confidence."""
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device == "cpu":
            torch.set_num_threads(min(2, os.cpu_count() or 1))
        self.confidence_threshold = confidence_threshold
        self.classes: Tuple[str, ...] = self._inject_negative_classes(DEFAULT_CLASSES)
        self.backend: str = "cpu"
        self._is_half: bool = False
        self._model = self._initialize_model()

    def _find_tensorrt_engine(self) -> Optional[str]:
        """Find candidate TensorRT engine file corresponding to model_name."""
        if self.model_name.endswith(".engine") and os.path.isfile(self.model_name):
            return self.model_name

        base_name = os.path.splitext(self.model_name)[0]
        engine_candidates = [
            f"{base_name}.engine",
            os.path.join(os.getcwd(), f"{base_name}.engine"),
            os.path.join(os.path.expanduser("~/.cache/ultralytics"), f"{base_name}.engine"),
        ]
        for candidate in engine_candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _initialize_model(self) -> Optional[object]:
        """Attempt loading YOLO model with dynamic backend detection and fallback."""
        if self.device.startswith("cuda"):
            # 1. Attempt TensorRT engine if available
            engine_path = self._find_tensorrt_engine()
            if engine_path is not None:
                try:
                    import tensorrt  # noqa: F401
                    from ultralytics import YOLO

                    logger.info("Loading TensorRT YOLO engine from '%s'...", engine_path)
                    model = YOLO(engine_path)
                    dummy = np.zeros((32, 32, 3), dtype=np.uint8)
                    model.predict(
                        source=dummy,
                        conf=self.confidence_threshold,
                        device=self.device,
                        verbose=False,
                    )
                    self.backend = "tensorrt"
                    self._is_half = False
                    logger.info("Successfully loaded TensorRT YOLO-World engine.")
                    return model
                except Exception as trt_exc:
                    logger.warning(
                        "TensorRT engine loading failed (%s). Falling back to PyTorch CUDA FP16.",
                        trt_exc,
                    )

            # 2. PyTorch CUDA FP16
            pytorch_model = (
                "yolov8s-worldv2.pt" if self.model_name.endswith(".engine") else self.model_name
            )
            try:
                from ultralytics import YOLO

                logger.info(
                    "Loading PyTorch YOLO-World model '%s' on %s (FP16)...",
                    pytorch_model,
                    self.device,
                )
                model = YOLO(pytorch_model)
                model.to(self.device)
                if hasattr(model, "model") and hasattr(model.model, "half"):
                    model.model.half()
                self._is_half = True
                self.backend = "cuda_fp16"
                model.set_classes(list(self.classes))

                # Attempt torch.compile if compatible
                if hasattr(torch, "compile"):
                    orig_model = model.model
                    try:
                        compiled = torch.compile(orig_model, mode="reduce-overhead")
                        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                        model.model = compiled
                        model.predict(
                            source=dummy,
                            conf=self.confidence_threshold,
                            device=self.device,
                            quantize=16,
                            verbose=False,
                        )
                        self.backend = "cuda_fp16_compiled"
                        logger.info(
                            "YOLO-World compiled with torch.compile(mode='reduce-overhead')."
                        )
                    except Exception as comp_exc:
                        model.model = orig_model
                        logger.debug("torch.compile fallback to standard CUDA FP16: %s", comp_exc)

                return model
            except (torch.cuda.OutOfMemoryError, Exception) as exc:
                logger.warning(
                    "CUDA initialization failed for YOLO-World (%s). Falling back to CPU.", exc
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._fallback_to_cpu_init()

        # 3. CPU fallback
        return self._fallback_to_cpu_init()

    def _fallback_to_cpu_init(self) -> Optional[object]:
        """Initialize YOLO model on CPU with float32 precision."""
        self.device = "cpu"
        self.backend = "cpu"
        self._is_half = False
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        pytorch_model = (
            "yolov8s-worldv2.pt" if self.model_name.endswith(".engine") else self.model_name
        )
        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO-World model '%s' on CPU...", pytorch_model)
            model = YOLO(pytorch_model)
            model.to("cpu")
            if hasattr(model, "model") and hasattr(model.model, "float"):
                model.model.float()
            model.set_classes(list(self.classes))
            return model
        except Exception as exc:
            logger.warning(
                "YOLO-World could not be loaded on CPU (%s). Operating in heuristic mode.", exc
            )
            self.backend = "heuristic"
            return None

    @staticmethod
    def _inject_negative_classes(
        classes: Sequence[str],
    ) -> Tuple[str, ...]:
        """Inject negative classes to prevent false positive background classifications."""
        class_list = list(classes)
        for neg in NEGATIVE_CLASSES:
            if neg not in class_list:
                class_list.append(neg)
        return tuple(class_list)

    def set_classes(self, classes: Sequence[str]) -> None:
        """Update open-vocabulary detection classes, always injecting negative classes."""
        self.classes = self._inject_negative_classes(classes)
        if self._model is not None and hasattr(self._model, "set_classes"):
            try:
                self._model.set_classes(list(self.classes))
            except Exception as exc:
                logger.warning("Failed to set open-vocab classes on model: %s", exc)

    def generate_proposals(self, image_bgr: np.ndarray) -> Tuple[BoundingBox, ...]:
        """Generate candidate bounding box proposals from incoming image."""
        if (
            image_bgr is None
            or image_bgr.size == 0
            or image_bgr.shape[0] < 4
            or image_bgr.shape[1] < 4
        ):
            return ()

        if self._model is not None:
            proposals = self._infer_with_yolo(image_bgr)
            if len(proposals) > 0:
                return proposals

        return self._generate_heuristic_proposals(image_bgr)

    def _infer_with_yolo(self, image_bgr: np.ndarray) -> Tuple[BoundingBox, ...]:
        """Run deep inference via YOLO-World."""
        try:
            predict_kwargs = {
                "source": image_bgr,
                "conf": self.confidence_threshold,
                "device": self.device,
                "verbose": False,
            }
            if self._is_half and self.device != "cpu":
                predict_kwargs["quantize"] = 16

            results = self._model.predict(**predict_kwargs)
            if not results or len(results) == 0:
                return ()

            proposals = []
            boxes = results[0].boxes
            for idx in range(len(boxes)):
                xywh = boxes.xywh[idx].cpu().numpy()
                conf = float(boxes.conf[idx].cpu().numpy())
                cls_idx = int(boxes.cls[idx].cpu().numpy())
                class_name = self.classes[cls_idx] if cls_idx < len(self.classes) else "object"
                proposals.append(
                    BoundingBox(
                        center_x=float(xywh[0]),
                        center_y=float(xywh[1]),
                        size_x=float(xywh[2]),
                        size_y=float(xywh[3]),
                        confidence=conf,
                        class_name=class_name,
                    )
                )
            return tuple(proposals)
        except torch.cuda.OutOfMemoryError as oom_exc:
            logger.error("CUDA OOM during YOLO inference: %s. Falling back to CPU.", oom_exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.device != "cpu":
                self._model = self._fallback_to_cpu_init()
                if self._model is not None:
                    return self._infer_with_yolo(image_bgr)
            return self._generate_heuristic_proposals(image_bgr)
        except Exception as exc:
            logger.error("YOLO inference failed: %s. Using heuristic fallback.", exc)
            return self._generate_heuristic_proposals(image_bgr)

    def _generate_heuristic_proposals(self, image_bgr: np.ndarray) -> Tuple[BoundingBox, ...]:
        """Generate color-segmented and contour-based proposals for simulation primitives."""
        height, width = image_bgr.shape[:2]
        proposals = []
        min_area = 100.0
        max_area = (width * height) * 0.85

        # 1. Color-based segmentation for simulation target objects
        target_primary = self.classes[0].lower() if self.classes else ""
        is_red = any(k in target_primary for k in ("red", "crimson", "scarlet", "ruby"))
        is_blue = any(k in target_primary for k in ("blue", "azure", "cyan"))
        is_green = any(k in target_primary for k in ("green", "lime", "emerald"))
        is_yellow = any(k in target_primary for k in ("yellow", "gold", "amber"))
        is_orange = "orange" in target_primary
        has_specific_color = bool(is_red or is_blue or is_green or is_yellow or is_orange)

        if not has_specific_color:
            prompt_str = " ".join(self.classes).lower() if self.classes else ""
            is_red = any(k in prompt_str for k in ("red", "crimson", "scarlet", "ruby"))
            is_blue = any(k in prompt_str for k in ("blue", "azure", "cyan"))
            is_green = any(k in prompt_str for k in ("green", "lime", "emerald"))
            is_yellow = any(k in prompt_str for k in ("yellow", "gold", "amber"))
            is_orange = "orange" in prompt_str
            if not (is_red or is_blue or is_green or is_yellow or is_orange):
                is_red = is_blue = is_green = is_yellow = is_orange = True

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        color_masks = []
        if is_red:
            m1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
            m2 = cv2.inRange(hsv, (170, 70, 40), (180, 255, 255))
            color_masks.append((m1 | m2, "red"))
        if is_blue:
            color_masks.append((cv2.inRange(hsv, (100, 70, 40), (130, 255, 255)), "blue"))
        if is_green:
            color_masks.append((cv2.inRange(hsv, (35, 70, 40), (85, 255, 255)), "green"))
        if is_yellow:
            color_masks.append((cv2.inRange(hsv, (20, 70, 40), (35, 255, 255)), "yellow"))
        if is_orange:
            color_masks.append((cv2.inRange(hsv, (10, 70, 40), (25, 255, 255)), "orange"))

        for mask, color_name in color_masks:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(c)
                    target_label = (
                        self.classes[0]
                        if (self.classes and color_name in self.classes[0].lower())
                        else f"{color_name} object"
                    )
                    proposals.append(
                        BoundingBox(
                            center_x=float(x + w / 2.0),
                            center_y=float(y + h / 2.0),
                            size_x=float(w),
                            size_y=float(h),
                            confidence=0.80,
                            class_name=target_label,
                        )
                    )

        # 2. General edge and intensity contours (only if no specific color is sought)
        if not has_specific_color:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center_x = float(x + w / 2.0)
                center_y = float(y + h / 2.0)
                is_dup = any(
                    abs(p.center_x - center_x) < 20 and abs(p.center_y - center_y) < 20
                    for p in proposals
                )
                if not is_dup:
                    proposals.append(
                        BoundingBox(
                            center_x=center_x,
                            center_y=center_y,
                            size_x=float(w),
                            size_y=float(h),
                            confidence=0.50,
                            class_name="candidate",
                        )
                    )

            # Fallback to low-confidence central candidate if no features found on valid resolution
            if not proposals and width >= 32 and height >= 32:
                proposals.append(
                    BoundingBox(
                        center_x=float(width / 2.0),
                        center_y=float(height / 2.0),
                        size_x=float(width * 0.3),
                        size_y=float(height * 0.4),
                        confidence=0.20,
                        class_name="candidate",
                    )
                )

        return tuple(proposals[:10])

    @staticmethod
    def crop_roi(image_bgr: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Extract rectangular region of interest from image safely."""
        height, width = image_bgr.shape[:2]
        x1 = min(max(0, box.x_min), width - 1)
        y1 = min(max(0, box.y_min), height - 1)
        x2 = min(max(x1 + 1, box.x_max), width)
        y2 = min(max(y1 + 1, box.y_max), height)
        return image_bgr[y1:y2, x1:x2].copy()
