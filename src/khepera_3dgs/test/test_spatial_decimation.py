"""Unit tests for spatial decimation and frame gating."""

import math

import numpy as np

from khepera_3dgs.spatial_decimation import SpatialDecimator


def test_initial_frame_acceptance():
    """Verify first observed frame is unconditionally accepted as origin anchor."""
    decimator = SpatialDecimator(
        min_translation_m=0.10,
        min_rotation_deg=10.0,
        min_laplacian_var=0.0,
    )
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    kf = decimator.add_frame(
        timestamp=1.0,
        image_rgb=dummy_img,
        x=0.0,
        y=0.0,
        yaw=0.0,
        linear_vel=0.05,
        angular_vel=0.02,
    )
    assert kf is not None
    assert kf.id == 0
    assert len(decimator.keyframes) == 1


def test_velocity_rejection():
    """Verify frames captured during high linear or angular speed are rejected."""
    decimator = SpatialDecimator(
        max_linear_vel_mps=0.20,
        max_angular_vel_radps=0.50,
        min_laplacian_var=0.0,
    )
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initial anchor
    decimator.add_frame(1.0, dummy_img, 0.0, 0.0, 0.0, 0.10, 0.10)

    # Excessive linear speed (> 0.20 m/s)
    kf_high_v = decimator.add_frame(
        timestamp=2.0,
        image_rgb=dummy_img,
        x=0.20,
        y=0.0,
        yaw=0.0,
        linear_vel=0.35,  # Exceeds 0.20 m/s
        angular_vel=0.10,
    )
    assert kf_high_v is None

    # Excessive angular speed (> 0.50 rad/s)
    kf_high_w = decimator.add_frame(
        timestamp=3.0,
        image_rgb=dummy_img,
        x=0.20,
        y=0.0,
        yaw=0.0,
        linear_vel=0.10,
        angular_vel=0.80,  # Exceeds 0.50 rad/s
    )
    assert kf_high_w is None


def test_displacement_and_rotation_gating():
    """Verify gating triggers at Delta d >= 0.10m and Delta theta >= 10 deg."""
    decimator = SpatialDecimator(
        min_translation_m=0.10,
        min_rotation_deg=10.0,
        max_linear_vel_mps=0.20,
        max_angular_vel_radps=0.50,
        min_laplacian_var=0.0,
    )
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 0
    decimator.add_frame(1.0, dummy_img, 0.0, 0.0, 0.0)

    # Sub-threshold displacement (0.05 m < 0.10 m)
    kf_small = decimator.add_frame(1.5, dummy_img, 0.05, 0.0, 0.0)
    assert kf_small is None

    # Threshold displacement (0.12 m >= 0.10 m)
    kf_dist = decimator.add_frame(2.0, dummy_img, 0.12, 0.0, 0.0)
    assert kf_dist is not None
    assert kf_dist.id == 1

    # Rotation gate (15 deg >= 10 deg)
    yaw_15deg = math.radians(15.0)
    kf_rot = decimator.add_frame(2.5, dummy_img, 0.12, 0.0, yaw_15deg)
    assert kf_rot is not None
    assert kf_rot.id == 2
