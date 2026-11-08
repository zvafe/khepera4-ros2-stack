"""Abstract interface and data models for VLM semantic reasoning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FrontierCandidate:
    """Represents a single candidate frontier for Set-of-Marks evaluation."""

    id: str
    pixel_coords: Tuple[int, int]
    world_coords: Tuple[float, float, float]
    exploration_gain: float


@dataclass(frozen=True)
class FrontierScore:
    """Represents the semantic evaluation output for a candidate frontier."""

    id: str
    relevance_prob: float
    reasoning: str
    total_utility: float


@dataclass(frozen=True)
class TargetVerification:
    """Represents the result of target object presence verification."""

    is_present: bool
    confidence: float
    reasoning: str
    bbox_2d: Optional[Tuple[int, int, int, int]] = None


class BaseSemanticReasoner(ABC):
    """Abstract base class defining required VLM reasoning primitives."""

    @abstractmethod
    def evaluate_frontiers(
        self,
        annotated_image: np.ndarray,
        candidates: List[FrontierCandidate],
        target_description: str
    ) -> List[FrontierScore]:
        """Query VLM with Set-of-Marks annotated image to obtain frontier scores.

        Args:
            annotated_image: BGR image with visual markers overlaid.
            candidates: List of candidate frontiers to evaluate.
            target_description: Natural language description of the target.

        Returns:
            List of FrontierScore objects with predicted relevance.
        """
        pass

    @abstractmethod
    def verify_target(
        self,
        image_bgr: np.ndarray,
        target_description: str,
        max_distance_m: float = 3.0
    ) -> TargetVerification:
        """Verify if the target object is present, visible, and reachable.

        Args:
            image_bgr: Current camera RGB observation.
            target_description: Target object class or detailed description.
            max_distance_m: Maximum acceptable distance threshold.

        Returns:
            TargetVerification with boolean presence and confidence.
        """
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, str]:
        """Return provider telemetry and operational availability."""
        pass
