"""Feature Extraction Module (FEM) for instance goal image matching."""

from typing import List, Tuple

import cv2
import numpy as np


class DirectSplatFeatureMatcher:
    """Matches rendered 3DGS viewpoints against reference goal image."""

    def __init__(
        self,
        match_threshold: int = 25,
        use_sift_fallback: bool = True
    ) -> None:
        """Initialize matcher configuration."""
        self._threshold = match_threshold
        self._use_sift = use_sift_fallback
        self._sift = cv2.SIFT_create(nfeatures=1000)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match_views(
        self,
        goal_image_bgr: np.ndarray,
        rendered_views: List[np.ndarray]
    ) -> Tuple[int, int, float]:
        """Find the maximum number of keypoint matches across all rendered views.

        Args:
            goal_image_bgr: Reference goal image I^g.
            rendered_views: List of synthetic viewpoints rendered from 3DGS map.

        Returns:
            Tuple of (best_view_idx, max_correspondences, normalized_score).
        """
        if not rendered_views or goal_image_bgr is None:
            return (-1, 0, 0.0)

        # Extract features on goal image
        gray_goal = cv2.cvtColor(goal_image_bgr, cv2.COLOR_BGR2GRAY)
        kp_goal, desc_goal = self._sift.detectAndCompute(gray_goal, None)
        if desc_goal is None or len(desc_goal) < 2:
            return (-1, 0, 0.0)

        best_idx = -1
        max_matches = 0

        for idx, view_bgr in enumerate(rendered_views):
            gray_view = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2GRAY)
            kp_view, desc_view = self._sift.detectAndCompute(gray_view, None)
            if desc_view is None or len(desc_view) < 2:
                continue

            # Lowe's ratio test matching
            raw_matches = self._bf_matcher.knnMatch(desc_goal, desc_view, k=2)
            good_matches = [
                m for m, n in raw_matches
                if len((m, n)) == 2 and m.distance < 0.75 * n.distance
            ]

            num_good = len(good_matches)
            if num_good > max_matches:
                max_matches = num_good
                best_idx = idx

        norm_score = float(min(1.0, max_matches / float(self._threshold)))
        return (best_idx, max_matches, norm_score)
