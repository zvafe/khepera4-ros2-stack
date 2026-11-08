"""Google Gemini Cloud API provider for visual frontier reasoning."""

import json
import os
from typing import Dict, List, Optional
import cv2
from google import genai
from google.genai import types
import numpy as np

from semantic_reasoner.base_reasoner import (
    BaseSemanticReasoner,
    FrontierCandidate,
    FrontierScore,
    TargetVerification,
)


class CloudGeminiProvider(BaseSemanticReasoner):
    """Client for Google Gemini 2.5 Flash / Pro Multimodal API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ) -> None:
        """Initialize the Gemini client using environment variable or explicit key."""
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model_name
        self._last_latency_ms = 0.0
        self._is_available = True

    def evaluate_frontiers(
        self,
        annotated_image: np.ndarray,
        candidates: List[FrontierCandidate],
        target_description: str
    ) -> List[FrontierScore]:
        """Query Gemini using Set-of-Marks prompt to evaluate candidate frontiers."""
        if not candidates:
            return []

        labels = [c.id for c in candidates]
        prompt = (
            f"Assume the labeled frontiers {labels} represent possible places to go. "
            "Each frontier is a detected boundary between explored and unexplored space.\n"
            f"Estimate the probability that each frontier leads to (or is already around) a "
            f"{target_description} when moving towards it and continuing exploration. "
            "Unseen labels should have a probability of 0.5.\n"
            "Consider long-term navigation potential, not only immediate visibility. "
            "Some frontiers may lead to larger unexplored regions. "
            "Use the local neighborhood context around each frontier.\n"
            "Return only a JSON dictionary mapping each frontier label to a 2-element list: "
            "[probability (float between 0.0 and 1.0), short explanation (string)].\n"
            'Example: {"A": [0.85, "open doorway leading to living room"]}'
        )

        _, buffer = cv2.imencode(".jpg", annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()

        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            raw_text = response.text or "{}"
            parsed = json.loads(raw_text)
            self._is_available = True
        except Exception:
            self._is_available = False
            parsed = {}

        results: List[FrontierScore] = []
        for candidate in candidates:
            score_data = parsed.get(candidate.id, [0.5, "Default fallback prior"])
            prob = float(score_data[0]) if isinstance(score_data, (list, tuple)) else 0.5
            reason = str(score_data[1]) if isinstance(score_data, (list, tuple)) else "N/A"
            prob = max(0.0, min(1.0, prob))
            utility = prob * candidate.exploration_gain

            results.append(
                FrontierScore(
                    id=candidate.id,
                    relevance_prob=prob,
                    reasoning=reason,
                    total_utility=utility
                )
            )

        return results

    def verify_target(
        self,
        image_bgr: np.ndarray,
        target_description: str,
        max_distance_m: float = 3.0
    ) -> TargetVerification:
        """Verify target presence with reaching, reflection, and occlusion checks."""
        prompt = (
            f"Based on this image, estimate the probability that a {target_description} "
            f"is in the camera field of view, within {max_distance_m:.1f} meters, and reachable.\n"
            "If the object is reflected in a mirror, behind glass, barely visible, heavily "
            "occluded, or unreachable, it should be considered absent.\n"
            "Probabilities should be close to 0.0 (absent) or 1.0 (present).\n"
            "Return only a JSON dictionary with keys 'probability' and 'reason'."
        )

        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_bytes = buffer.tobytes()

        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            parsed = json.loads(response.text or "{}")
            prob = float(parsed.get("probability", 0.0))
            reason = str(parsed.get("reason", "No reasoning provided"))
            self._is_available = True
            return TargetVerification(
                is_present=prob >= 0.7,
                confidence=prob,
                reasoning=reason
            )
        except Exception as exc:
            self._is_available = False
            return TargetVerification(
                is_present=False,
                confidence=0.0,
                reasoning=f"Gemini API verification failed: {exc}"
            )

    def get_health_status(self) -> Dict[str, str]:
        """Return provider status telemetry."""
        return {
            "backend": "cloud_gemini",
            "model": self._model_name,
            "available": str(self._is_available),
            "last_latency_ms": f"{self._last_latency_ms:.1f}"
        }
