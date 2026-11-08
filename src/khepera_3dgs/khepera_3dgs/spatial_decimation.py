"""Spatial decimation and frame-pose synchronization module for Khepera IV."""

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class KeyframeData:
    """Container for synchronized keyframe visual observation and robot state."""

    id: int  # noqa: A003
    timestamp: float
    x: float
    y: float
    yaw: float
    linear_vel: float
    angular_vel: float
    laplacian_var: float
    image_path: str = ""
    depth_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert keyframe metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyframeData":
        """Reconstruct keyframe metadata from dictionary."""
        return cls(**data)


class SpatialDecimator:
    """
    Selects keyframes based on geometric gating and velocity constraints.

    Enforces minimum planar translation (Delta d >= 0.10 m) or rotation (Delta theta >= 10 deg),
    and rejects blurred or high-velocity captures (v > 0.20 m/s or |omega| > 0.50 rad/s).
    """

    def __init__(
        self,
        min_translation_m: float = 0.10,
        min_rotation_deg: float = 10.0,
        max_linear_vel_mps: float = 0.20,
        max_angular_vel_radps: float = 0.50,
        min_laplacian_var: float = 40.0,
    ) -> None:
        """
        Initialize decimation parameters.

        :param min_translation_m: Planar translation threshold in meters.
        :param min_rotation_deg: Heading angle change threshold in degrees.
        :param max_linear_vel_mps: Maximum permissible linear speed.
        :param max_angular_vel_radps: Maximum permissible angular speed.
        :param min_laplacian_var: Minimum Laplacian variance for blur filter.
        """
        self.min_dist = float(min_translation_m)
        self.min_rot_rad = math.radians(float(min_rotation_deg))
        self.max_v = float(max_linear_vel_mps)
        self.max_omega = float(max_angular_vel_radps)
        self.min_blur = float(min_laplacian_var)

        self._keyframes: List[KeyframeData] = []
        self._images: List[np.ndarray] = []
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._last_yaw: Optional[float] = None
        self._next_id = 0

    @property
    def keyframes(self) -> List[KeyframeData]:
        """Return list of selected keyframe records."""
        return self._keyframes

    @property
    def images(self) -> List[np.ndarray]:
        """Return list of keyframe RGB images."""
        return self._images

    def compute_blur_metric(self, image_rgb: np.ndarray) -> float:
        """
        Calculate image sharpness using variance of Laplacian.

        :param image_rgb: RGB image array (H, W, 3).
        :return: Laplacian variance scalar.
        """
        if len(image_rgb.shape) == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())

    @staticmethod
    def normalize_angle(theta: float) -> float:
        """
        Normalize angle to the interval [-pi, pi].

        :param theta: Input angle in radians.
        :return: Wrapped angle in radians.
        """
        return math.atan2(math.sin(theta), math.cos(theta))

    def evaluate_gates(
        self,
        x: float,
        y: float,
        yaw: float,
        linear_vel: float,
        angular_vel: float,
        image_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str, float]:
        """
        Evaluate gating criteria for candidate observation.

        :param x: Current robot x-coordinate in map frame.
        :param y: Current robot y-coordinate in map frame.
        :param yaw: Current robot orientation in map frame.
        :param linear_vel: Linear velocity magnitude.
        :param angular_vel: Angular velocity magnitude.
        :param image_rgb: Candidate RGB image array.
        :return: (is_selected, reason, blur_metric)
        """
        # Velocity sanity gate
        if abs(linear_vel) > self.max_v:
            return False, f"Linear speed ({abs(linear_vel):.2f}) exceeds limit {self.max_v}", 0.0

        if abs(angular_vel) > self.max_omega:
            return False, (
                f"Angular speed ({abs(angular_vel):.2f}) exceeds limit {self.max_omega}"
            ), 0.0

        # Blur gate
        blur_val = 0.0
        if image_rgb is not None and self.min_blur > 0.0:
            blur_val = self.compute_blur_metric(image_rgb)
            if blur_val < self.min_blur:
                msg = f"Image blurred (Laplacian var {blur_val:.1f} < {self.min_blur})"
                return False, msg, blur_val

        # Initial frame is always accepted
        if self._last_x is None or self._last_y is None or self._last_yaw is None:
            return True, "Initial keyframe anchor", blur_val

        # Spatial translation gate
        delta_dist = math.hypot(x - self._last_x, y - self._last_y)
        if delta_dist >= self.min_dist:
            return True, f"Displacement {delta_dist:.3f}m >= {self.min_dist:.3f}m", blur_val

        # Angular rotation gate
        delta_yaw = abs(self.normalize_angle(yaw - self._last_yaw))
        if delta_yaw >= self.min_rot_rad:
            return True, (
                f"Rotation {math.degrees(delta_yaw):.1f}deg >= "
                f"{math.degrees(self.min_rot_rad):.1f}deg"
            ), blur_val

        return False, "Displacement and heading below decimation threshold", blur_val

    def add_frame(
        self,
        timestamp: float,
        image_rgb: np.ndarray,
        x: float,
        y: float,
        yaw: float,
        linear_vel: float = 0.0,
        angular_vel: float = 0.0,
        force: bool = False,
    ) -> Optional[KeyframeData]:
        """
        Process incoming frame and register keyframe if criteria are met.

        :param timestamp: Sensor timestamp in seconds.
        :param image_rgb: (H, W, 3) RGB uint8 image.
        :param x: Robot x in map frame.
        :param y: Robot y in map frame.
        :param yaw: Robot heading in map frame.
        :param linear_vel: Estimated linear speed.
        :param angular_vel: Estimated angular speed.
        :param force: Bypass gating checks.
        :return: KeyframeData instance if registered, None otherwise.
        """
        is_selected, _, blur_val = self.evaluate_gates(
            x, y, yaw, linear_vel, angular_vel, image_rgb
        )
        if not is_selected and not force:
            return None

        kf = KeyframeData(
            id=self._next_id,
            timestamp=float(timestamp),
            x=float(x),
            y=float(y),
            yaw=float(self.normalize_angle(yaw)),
            linear_vel=float(linear_vel),
            angular_vel=float(angular_vel),
            laplacian_var=float(blur_val),
        )

        self._keyframes.append(kf)
        self._images.append(image_rgb.copy())
        self._last_x = float(x)
        self._last_y = float(y)
        self._last_yaw = float(yaw)
        self._next_id += 1
        return kf

    def export_dataset(
        self,
        output_dir: str,
        image_subdir: str = "images",
    ) -> str:
        """
        Persist keyframes and manifest to disk for offline training.

        :param output_dir: Base directory to serialize keyframes.
        :param image_subdir: Relative subfolder for RGB images.
        :return: Path to generated manifest.json file.
        """
        img_dir = os.path.join(output_dir, image_subdir)
        os.makedirs(img_dir, exist_ok=True)

        manifest_records = []
        for kf, img in zip(self._keyframes, self._images):
            img_rel_path = os.path.join(image_subdir, f"frame_{kf.id:04d}.png")
            img_abs_path = os.path.join(output_dir, img_rel_path)
            cv2.imwrite(img_abs_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            rec = kf.to_dict()
            rec["image_path"] = img_rel_path
            manifest_records.append(rec)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"keyframes": manifest_records}, f, indent=2)

        return manifest_path
