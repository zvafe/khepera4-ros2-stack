"""Novel Viewpoint Synthesis Module (NVSM) pose generation."""

import math
from typing import List, Tuple

import numpy as np


class NovelViewSampler:
    """Generates upper-hemisphere camera viewpoints around an object 3D centroid."""

    def __init__(
        self,
        num_views: int = 8,
        sphere_radius_m: float = 1.0,
        elevation_deg_range: Tuple[float, float] = (15.0, 55.0)
    ) -> None:
        """Initialize sampling geometry parameters."""
        self._num_views = max(4, num_views)
        self._radius = sphere_radius_m
        self._elev_min = math.radians(elevation_deg_range[0])
        self._elev_max = math.radians(elevation_deg_range[1])

    def sample_viewpoints(
        self,
        centroid_x: float,
        centroid_y: float,
        centroid_z: float
    ) -> List[np.ndarray]:
        """Compute camera-to-world 4x4 transformation matrices looking at the centroid.

        Args:
            centroid_x: X coordinate of object center.
            centroid_y: Y coordinate of object center.
            centroid_z: Z coordinate of object center.

        Returns:
            List of 4x4 homogenous transformation matrices (T_cam_to_world).
        """
        center = np.array([centroid_x, centroid_y, centroid_z], dtype=np.float32)
        c2w_matrices: List[np.ndarray] = []

        angle_step = 2.0 * math.pi / float(self._num_views)
        elev_mid = 0.5 * (self._elev_min + self._elev_max)

        for i in range(self._num_views):
            azimuth = i * angle_step

            # Spherical coordinate position on upper hemisphere
            px = centroid_x + self._radius * math.cos(azimuth) * math.cos(elev_mid)
            py = centroid_y + self._radius * math.sin(azimuth) * math.cos(elev_mid)
            pz = centroid_z + self._radius * math.sin(elev_mid)
            eye = np.array([px, py, pz], dtype=np.float32)

            # Camera coordinate frame: Z forwards, X right, Y down
            forward = center - eye
            norm_f = np.linalg.norm(forward)
            forward = forward / norm_f if norm_f > 1e-6 else np.array([0, 0, 1])

            up_world = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(forward, up_world)
            norm_r = np.linalg.norm(right)
            right = right / norm_r if norm_r > 1e-6 else np.array([1, 0, 0])

            down = np.cross(forward, right)
            down = down / np.linalg.norm(down)

            # Build 4x4 camera-to-world matrix
            t_c2w = np.eye(4, dtype=np.float32)
            t_c2w[:3, 0] = right
            t_c2w[:3, 1] = down
            t_c2w[:3, 2] = forward
            t_c2w[:3, 3] = eye

            c2w_matrices.append(t_c2w)

        return c2w_matrices
