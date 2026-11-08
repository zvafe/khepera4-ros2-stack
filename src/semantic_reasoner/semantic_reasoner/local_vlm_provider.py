"""Local VLM provider for offline Set-of-Marks and target verification reasoning."""

import base64
import json
from typing import Dict, List, Optional
import urllib.error
import urllib.request
import cv2
import numpy as np

from semantic_reasoner.base_reasoner import (
    BaseSemanticReasoner,
    FrontierCandidate,
    FrontierScore,
    TargetVerification,
)


class LocalVlmProvider(BaseSemanticReasoner):
    """Client for local OpenAI-compatible or OpenFrontier HTTP VLM servers."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:11434/v1/chat/completions",
        model_name: str = "gemma4:e2b",
        timeout_sec: float = 8.0
    ) -> None:
        """Initialize the local HTTP client parameters."""
        self._endpoint_url = endpoint_url
        self._model_name = model_name
        self._timeout_sec = timeout_sec
        self._is_available = True

    def _post_json(self, payload: dict) -> Optional[dict]:
        """Send JSON payload via standard library HTTP client."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as response:
                self._is_available = True
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self._is_available = False
            return None

    def evaluate_frontiers(
        self,
        annotated_image: np.ndarray,
        candidates: List[FrontierCandidate],
        target_description: str
    ) -> List[FrontierScore]:
        """Evaluate candidate frontiers via local server Set-of-Marks query."""
        if not candidates:
            return []

        labels = [c.id for c in candidates]
        _, buffer = cv2.imencode(".jpg", annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"Assume the labeled frontiers {labels} represent possible places to go. "
            f"Estimate the probability that each leads towards {target_description}.\n"
            "Output JSON mapping each label to [probability, reason].\n"
            'Example: {"A": [0.75, "corridor to goal"]}'
        )

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            "temperature": 0.1
        }

        response = self._post_json(payload)
        parsed = {}
        if response and "choices" in response and len(response["choices"]) > 0:
            content_str = response["choices"][0].get("message", {}).get("content", "")
            try:
                # Find start and end of JSON object if wrapped in markdown
                start_idx = content_str.find("{")
                end_idx = content_str.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    parsed = json.loads(content_str[start_idx:end_idx])
            except json.JSONDecodeError:
                parsed = {}

        results: List[FrontierScore] = []
        for candidate in candidates:
            score_data = parsed.get(candidate.id, [0.5, "Default fallback"])
            prob = float(score_data[0]) if isinstance(score_data, (list, tuple)) else 0.5
            reason = str(score_data[1]) if isinstance(score_data, (list, tuple)) else "N/A"
            prob = max(0.0, min(1.0, prob))
            results.append(
                FrontierScore(
                    id=candidate.id,
                    relevance_prob=prob,
                    reasoning=reason,
                    total_utility=prob * candidate.exploration_gain
                )
            )
        return results

    def verify_target(
        self,
        image_bgr: np.ndarray,
        target_description: str,
        max_distance_m: float = 3.0
    ) -> TargetVerification:
        """Verify target presence via local server query."""
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"Is {target_description} in the FOV, reachable, within {max_distance_m:.1f}m?\n"
            'Return JSON: {"probability": 0.0 to 1.0, "reason": "reason"}'
        )

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            "temperature": 0.1
        }

        response = self._post_json(payload)
        if response and "choices" in response and len(response["choices"]) > 0:
            content_str = response["choices"][0].get("message", {}).get("content", "")
            try:
                start_idx = content_str.find("{")
                end_idx = content_str.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    parsed = json.loads(content_str[start_idx:end_idx])
                    prob = float(parsed.get("probability", 0.0))
                    reason = str(parsed.get("reason", "Local verification"))
                    return TargetVerification(is_present=prob >= 0.7, confidence=prob, reasoning=reason)
            except json.JSONDecodeError:
                pass

        return TargetVerification(is_present=False, confidence=0.0, reasoning="Local VLM unreachable")

    def get_health_status(self) -> Dict[str, str]:
        """Return provider status telemetry."""
        return {
            "backend": "local_vlm",
            "model": self._model_name,
            "endpoint": self._endpoint_url,
            "available": str(self._is_available)
        }
