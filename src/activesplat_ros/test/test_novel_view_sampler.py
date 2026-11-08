"""Unit tests for ActiveSplat NovelViewSampler."""

import math

import numpy as np

from activesplat_ros.novel_view_sampler import NovelViewSampler


def test_sampler_generates_exact_view_count() -> None:
    """Verify that sample_viewpoints outputs the requested number of camera poses."""
    sampler = NovelViewSampler(num_views=8, sphere_radius_m=1.0)
    poses = sampler.sample_viewpoints(centroid_x=0.0, centroid_y=0.0, centroid_z=0.5)

    assert len(poses) == 8
    for pose in poses:
        assert pose.shape == (4, 4)
        # Check that camera is positioned at radius 1.0 from centroid
        eye_pos = pose[:3, 3]
        dist_to_center = math.dist(eye_pos, [0.0, 0.0, 0.5])
        assert abs(dist_to_center - 1.0) < 1e-3


def test_sampler_cameras_face_centroid() -> None:
    """Verify that camera optical axis (forward vector) points towards the centroid."""
    sampler = NovelViewSampler(num_views=6, sphere_radius_m=1.5)
    cx, cy, cz = 1.0, 2.0, 0.3
    poses = sampler.sample_viewpoints(centroid_x=cx, centroid_y=cy, centroid_z=cz)

    center = np.array([cx, cy, cz])
    for pose in poses:
        eye = pose[:3, 3]
        fwd = pose[:3, 2]
        expected_fwd = (center - eye) / np.linalg.norm(center - eye)
        # Dot product between camera forward and direction to center should be close to 1
        dot_prod = float(np.dot(fwd, expected_fwd))
        assert dot_prod > 0.99
