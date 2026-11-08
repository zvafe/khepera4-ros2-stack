"""Set-of-Marks visual marker annotation utility for RGB observations."""

from typing import List, Tuple
import cv2
import numpy as np


class SetOfMarksAnnotator:
    """Draws high-contrast Set-of-Marks labels at candidate pixel centroids."""

    def __init__(
        self,
        circle_radius: int = 14,
        font_scale: float = 0.55,
        circle_color: Tuple[int, int, int] = (0, 0, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        """Initialize annotation style parameters."""
        self._radius = circle_radius
        self._font_scale = font_scale
        self._circle_color = circle_color
        self._text_color = text_color
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._border_color = (0, 0, 0)

    def annotate(
        self,
        image_bgr: np.ndarray,
        markers: List[Tuple[str, Tuple[int, int]]]
    ) -> np.ndarray:
        """Overlay Set-of-Marks markers at specified pixel coordinates.

        Args:
            image_bgr: Input BGR image (H, W, 3).
            markers: List of tuples containing (label_str, (pixel_u, pixel_v)).

        Returns:
            Annotated BGR image copy.
        """
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        for label, (u, v) in markers:
            u_clamped = max(self._radius, min(w - self._radius, int(u)))
            v_clamped = max(self._radius, min(h - self._radius, int(v)))

            # Outer dark border for visibility on white backgrounds
            cv2.circle(annotated, (u_clamped, v_clamped), self._radius + 1, self._border_color, 2)
            # Solid inner marker
            cv2.circle(annotated, (u_clamped, v_clamped), self._radius, self._circle_color, -1)

            # Center text calculation
            text_size, _ = cv2.getTextSize(label, self._font, self._font_scale, 2)
            text_x = u_clamped - (text_size[0] // 2)
            text_y = v_clamped + (text_size[1] // 2)

            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                self._font,
                self._font_scale,
                self._text_color,
                2,
                cv2.LINE_AA
            )

        return annotated
