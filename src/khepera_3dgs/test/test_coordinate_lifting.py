"""Unit tests for coordinate lifting, SE(2)->SE(3) compounding, and floor geometry."""

import math
import numpy as np
import pytest

from khepera_3dgs.coordinate_lifting import CoordinateLifter


def test_extrinsic_transform_structure():
    """Verify camera extrinsics and optical frame axes."""
    lifter = CoordinateLifter(
        camera_x_offset_m=0.0682,
        camera_z_offset_m=0.0283,
        camera_pitch_rad=0.0,
    )

    # Base-to-camera optical translation
    assert np.isclose(lifter.t_base_cam_ros[0, 3], 0.0682)
    assert np.isclose(lifter.t_base_cam_ros[1, 3], 0.0)
    assert np.isclose(lifter.t_base_cam_ros[2, 3], 0.0283)

    # In ROS optical frame:
    # +Z_optical is +X_robot (Forward)
    # +X_optical is -Y_robot (Right)
    # +Y_optical is -Z_robot (Down)
    r = lifter.t_base_cam_ros[:3, :3]
    # Unit vector along optical +Z transformed into robot coordinates
    z_opt_in_base = r @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(z_opt_in_base, [1.0, 0.0, 0.0], atol=1e-5)


def test_se2_to_se3_compounding():
    """Verify SE(2) compounding with robot pose (x=1.0, y=2.0, yaw=pi/2)."""
    lifter = CoordinateLifter(camera_x_offset_m=0.0682, camera_z_offset_m=0.0283)
    yaw = math.pi / 2.0
    t_ros, t_gl, viewmat = lifter.compute_pose_matrices(x=1.0, y=2.0, yaw=yaw)

    # In map frame, robot at (1.0, 2.0) facing +Y (yaw = pi/2):
    # Camera is 0.0682m forward (along +Y), so camera position in map:
    # x_cam_map = 1.0, y_cam_map = 2.0 + 0.0682 = 2.0682, z_cam_map = 0.0283
    assert np.isclose(t_ros[0, 3], 1.0, atol=1e-4)
    assert np.isclose(t_ros[1, 3], 2.0682, atol=1e-4)
    assert np.isclose(t_ros[2, 3], 0.0283, atol=1e-4)

    # OpenGL conversion preserves origin but flips Y and Z columns
    assert np.isclose(t_gl[0, 3], 1.0, atol=1e-4)
    assert np.isclose(t_gl[1, 3], 2.0682, atol=1e-4)
    assert np.isclose(t_gl[2, 3], 0.0283, atol=1e-4)

    # Viewmatrix is exact inverse of OpenGL camera pose
    ident = viewmat @ t_gl
    assert np.allclose(ident, np.eye(4), atol=1e-4)


def test_geometric_sanity_floor_at_z_zero():
    """
    Geometric Sanity Test: Floor points must unproject to Z_world approx 0.0.

    Matches Pipeline Specification:
    'Generate a raw point cloud from keyframe 0 and confirm floor points sit at Z approx 0
    relative to the SLAM map frame.'
    """
    lifter = CoordinateLifter(
        camera_x_offset_m=0.0682,
        camera_z_offset_m=0.0283,
        fx=554.256258,
        fy=554.256258,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    t_ros, _, _ = lifter.compute_pose_matrices(x=0.0, y=0.0, yaw=0.0)

    # Synthesize planar ground floor depth
    h, w = 480, 640
    depth_map = np.zeros((h, w), dtype=np.float32)
    img_rgb = np.full((h, w, 3), 128, dtype=np.uint8)

    # For any floor pixel below horizon (v > cy = 240):
    # Expected depth: d = fy * h_floor / (v - cy)
    for v in range(250, 480):
        d_val = (lifter.fy * lifter.zc) / float(v - lifter.cy)
        depth_map[v, :] = d_val

    pts_world, _ = lifter.unproject_depth_map(
        depth_map=depth_map,
        image_rgb=img_rgb,
        t_world_cam_ros=t_ros,
        d_min=0.05,
        d_max=2.0,
    )

    assert len(pts_world) > 0
    # Confirm every single unprojected ground point has Z_world = 0.0 within numerical precision
    z_coords = pts_world[:, 2]
    max_z_deviation = np.max(np.abs(z_coords))
    assert max_z_deviation < 1e-4, f"Floor points deviate from Z=0: max |Z| = {max_z_deviation}"


def test_ground_plane_clamping_and_sky_filtering():
    """Verify floor depth clamping eliminates subterranean points and sky filter drops ceiling."""
    lifter = CoordinateLifter(camera_x_offset_m=0.0682, camera_z_offset_m=0.0283)
    t_ros, _, _ = lifter.compute_pose_matrices(x=0.0, y=0.0, yaw=0.0)

    h, w = 480, 640
    # Artificially set entire image to 2.5m (which would normally send floor subterranean
    # and upper half into the sky ceiling at Z > 1.5m)
    depth_map = np.full((h, w), 2.5, dtype=np.float32)
    img_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

    pts_world, _ = lifter.unproject_depth_map(
        depth_map=depth_map,
        image_rgb=img_rgb,
        t_world_cam_ros=t_ros,
        d_min=0.05,
        d_max=3.50,
        subsample_step=4,
        min_z=-0.02,
        max_z=0.40,
    )

    assert len(pts_world) > 0
    # Every surviving point must satisfy the arena bounds: -0.02 <= Z <= 0.40
    assert np.all(pts_world[:, 2] >= -0.02)
    assert np.all(pts_world[:, 2] <= 0.40)
    # The floor points (below horizon v > 240) must sit within numerical precision of Z=0.0
    floor_pts = pts_world[pts_world[:, 2] <= 0.01]
    assert len(floor_pts) > 0
    assert np.all(np.abs(floor_pts[:, 2]) < 1e-4)

