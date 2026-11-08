"""Frontier detector identifying visual and geometric navigation boundaries."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class RawFrontier2D:
    """Represents a 2D candidate frontier in image pixel coordinates."""

    pixel_u: int
    pixel_v: int
    estimated_depth_m: float
    exploration_gain: float


class VisualFrontierDetector:
    """Extracts candidate visual navigation frontiers from RGB-D observations."""

    def __init__(
        self,
        min_cluster_area: int = 150,
        max_depth_range_m: float = 4.5,
        default_camera_fx: float = 554.25,
        default_camera_fy: float = 554.25,
        default_cx: float = 320.0,
        default_cy: float = 240.0
    ) -> None:
        """Initialize camera intrinsics and geometric filtering thresholds."""
        self._min_area = min_cluster_area
        self._max_depth = max_depth_range_m
        self._fx = default_camera_fx
        self._fy = default_camera_fy
        self._cx = default_cx
        self._cy = default_cy

    def update_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> None:
        """Update pinhole camera intrinsic parameters."""
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

    def detect_visual_frontiers(
        self,
        image_bgr: np.ndarray,
        depth_img: Optional[np.ndarray] = None
    ) -> List[RawFrontier2D]:
        """Detect candidate visual frontiers from visual gradients and open regions."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Morphological dilation to bridge open apertures (doorways, hallways)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        inverted_free = cv2.bitwise_not(dilated)

        # Mask out upper 20% (ceiling) and lower 15% (immediate floor near chassis)
        h, w = gray.shape
        mask = np.zeros_like(inverted_free)
        y_top = int(0.20 * h)
        y_bot = int(0.85 * h)
        mask[y_top:y_bot, :] = 255
        masked_free = cv2.bitwise_and(inverted_free, mask)

        contours, _ = cv2.findContours(masked_free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[RawFrontier2D] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            moments = cv2.moments(cnt)
            if moments["m00"] <= 0:
                continue

            u = int(moments["m10"] / moments["m00"])
            v = int(moments["m01"] / moments["m00"])

            depth_val = self._sample_depth(depth_img, u, v)
            if depth_val <= 0.2 or depth_val > self._max_depth:
                depth_val = 2.0  # Fallback default visual frontier depth

            # Exploration gain proportional to visible aperture size
            gain = float(min(10.0, max(1.0, area / 500.0)))

            candidates.append(
                RawFrontier2D(
                    pixel_u=u,
                    pixel_v=v,
                    estimated_depth_m=depth_val,
                    exploration_gain=gain
                )
            )

        return candidates

    def backproject_to_camera_frame(
        self,
        pixel_u: int,
        pixel_v: int,
        depth_m: float
    ) -> Tuple[float, float, float]:
        """Project a 2D pixel coordinate into 3D camera coordinates (X_right, Y_down, Z_fwd)."""
        x_cam = (float(pixel_u) - self._cx) * depth_m / self._fx
        y_cam = (float(pixel_v) - self._cy) * depth_m / self._fy
        z_cam = depth_m
        return (x_cam, y_cam, z_cam)

    def _sample_depth(
        self,
        depth_img: Optional[np.ndarray],
        u: int,
        v: int
    ) -> float:
        """Sample depth image using median in a 5x5 neighborhood."""
        if depth_img is None:
            return 2.0

        h, w = depth_img.shape[:2]
        u_min = max(0, u - 2)
        u_max = min(w, u + 3)
        v_min = max(0, v - 2)
        v_max = min(h, v + 3)

        patch = depth_img[v_min:v_max, u_min:u_max]
        valid_vals = patch[np.isfinite(patch) & (patch > 0.1)]
        if len(valid_vals) == 0:
            return 2.0
        return float(np.median(valid_vals))
