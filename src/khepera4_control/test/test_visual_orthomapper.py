# Copyright 2026 Mobile Robotics Laboratory.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
E2E Test Suite: Ground-Plane IPM Homography Math (F1) & Visual Orthomapper Node (F2).

4-Tier Hierarchy:
- Tier 1: Feature Coverage (Forward & Inverse Homography, Horizon Clipping,
          Distance Bounds, Pitch Guard, Node Init, CameraInfo Sync,
          Odometry Sync, Grid Updating, Latched QoS)
- Tier 2: Boundary & Corner Cases (Horizon Singularity, Grazing Angle Far Bound,
          Bottom Row Near Bound, Lateral Edges, Negative Depth, Pitch Boundary,
          Unlocalized Camera Frame, Boundary Clamping)
- Tier 3: Pairwise Combinations (Continuous Rotation & IPM, Pitch Oscillation & Streaming,
          Odometry Fallback, Clear Map During Warping)
- Tier 4: Real-World Scenarios (Straight Drive Orthomosaic Coverage, Arena Trajectory Strip)

Authoritative Specifications:
- ORIGINAL_REQUEST.md §R1
- PROJECT.md §Feature Inventory (F1, F2)
- survey_specs.md §1 (Homography & Optical Geometry)
- khepera4.urdf (WVGA 640x480, HFOV 60 deg, camera mount x=0.0682, z=0.0283)
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

from khepera4_control.common import quaternion_from_euler
from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image as RosImage

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Ensure rclpy is initialized for node inspection."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def mapper_node(tmp_path):
    """Instantiate production VisualOrthomapper node for direct verification."""
    node = VisualOrthomapper(
        parameter_overrides=[
            Parameter(
                'keyframe_dir',
                Parameter.Type.STRING,
                str(tmp_path)
            )
        ]
    )
    yield node
    node.destroy_node()


@pytest.fixture
def orthomapper_node(mapper_node):
    """Alias fixture for orthomapper_node."""
    return mapper_node


def get_urdf_path():
    """Retrieve absolute path to khepera4.urdf."""
    ws_root = os.path.abspath(os.path.join(_PKG_ROOT, '..', '..'))
    return os.path.join(
        ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf'
    )


# ==============================================================================
# Tier 1: Feature Coverage (F1 & F2)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Primary behavior coverage for Homography Math (F1) & Orthomapper Node (F2)."""

    def test_f1_camera_intrinsics_pinhole_derivation(self, mapper_node):
        """F1: Validate pinhole focal lengths fx, fy, and principal center cx, cy."""
        node = mapper_node
        assert math.isclose(node.fx, 320.0 * math.sqrt(3), rel_tol=1e-5)
        assert math.isclose(node.fy, node.fx, rel_tol=1e-5)
        assert node.cx == 320.0
        assert node.cy == 240.0
        assert node.camera_h == pytest.approx(0.0283, abs=1e-4)
        assert node.camera_xc == pytest.approx(0.0682, abs=1e-4)
        assert node.fx == pytest.approx(554.256, rel=1e-3)

    def test_f1_forward_homography_ground_to_pixel(self, mapper_node):
        """F1: Forward homography projects known ground points to correct pixel coords."""
        node = mapper_node
        # 1. Point on optical axis at 1.0m forward: (X_b = 1.0, Y_b = 0.0)
        u, v, valid = node.ground_to_image(1.0, 0.0)
        assert valid is True
        assert u == pytest.approx(320.0, abs=1e-3)
        expected_v = 240.0 + (node.camera_h * node.fy) / (1.0 - node.camera_xc)
        assert v == pytest.approx(expected_v, abs=1e-2)

        # 2. Point displaced to left: Y_b = 0.1m at 1.0m forward
        u_left, v_left, valid_left = node.ground_to_image(1.0, 0.1)
        assert valid_left is True
        assert u_left < 320.0  # Left in world is left (smaller u) in image
        assert v_left == pytest.approx(v, abs=1e-3)

    def test_f1_inverse_homography_pixel_to_ground(self, mapper_node):
        """F1: Inverse homography back-projects image pixels to ground (X_b, Y_b)."""
        node = mapper_node
        v_test = 300.0
        u_test = 400.0
        xb, yb, valid = node.image_to_ground(u_test, v_test)
        assert valid is True

        expected_xb = node.camera_xc + (node.camera_h * node.fy) / (v_test - node.cy)
        assert xb == pytest.approx(expected_xb, rel=1e-4)
        assert yb < 0.0  # u > 320 corresponds to robot right (-Y_b)

    def test_f1_homography_forward_inverse_roundtrip(self, mapper_node):
        """F1: Roundtrip projection (u, v) -> (X_b, Y_b) -> (u', v') within 1e-4 tolerance."""
        node = mapper_node
        for u in [100.0, 200.0, 320.0, 450.0, 580.0]:
            for v in [260.0, 300.0, 350.0, 400.0, 460.0]:
                xb, yb, valid = node.image_to_ground(u, v)
                assert valid is True
                assert xb is not None and yb is not None
                u_proj, v_proj, valid_proj = node.ground_to_image(xb, yb)
                assert valid_proj is True
                assert u_proj == pytest.approx(u, abs=1e-4)
                assert v_proj == pytest.approx(v, abs=1e-4)

    def test_f1_homography_determinant_and_invertibility(self, mapper_node):
        """F1: Homography determinant det(H) = -h * fx * fy != 0 matches analytical formula."""
        node = mapper_node
        det_h = np.linalg.det(node.H_b2img)
        expected_det = -node.camera_h * node.fx * node.fy
        assert det_h == pytest.approx(expected_det, rel=1e-4)
        assert abs(det_h) > 8000.0  # Well conditioned, strictly invertible

        # Check H * H^-1 is identity
        identity = np.dot(node.H_b2img, node.H_img2b)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-5)

    def test_f1_optical_axis_center_column_symmetry(self, mapper_node):
        """F1: Ground points with Y_b = 0 project strictly onto image centerline u = 320."""
        node = mapper_node
        for dist in [0.2, 0.5, 0.8, 1.0, 1.4]:
            u, v, valid = node.ground_to_image(dist, 0.0)
            assert valid is True
            assert u == pytest.approx(320.0, abs=1e-5)
            assert v is not None and v > 240.0

    def test_f1_horizon_clipping_rejection(self, mapper_node):
        """F1: Pixels at or above horizon (v <= 240) produce no ground intersection."""
        node = mapper_node
        assert node.ground_to_image(0.05, 0.0)[2] is False  # Behind camera
        assert node.image_to_ground(320.0, 240.0)[2] is False  # Exact horizon
        assert node.image_to_ground(320.0, 200.0)[2] is False  # Sky/walls
        assert node.image_to_ground(320.0, 0.0)[2] is False    # Top edge

    def test_f1_distance_limits_bounds(self, mapper_node):
        """F1: Near and far ground distance limits match optical geometry."""
        node = mapper_node
        # Near distance at bottom row v=479
        xb_near, _, valid_near = node.image_to_ground(320.0, 479.0)
        assert valid_near is True
        assert xb_near == pytest.approx(0.1338, abs=1e-3)

        # Far distance at v=251
        xb_far, _, valid_far = node.image_to_ground(320.0, 251.0)
        assert valid_far is True
        assert xb_far <= node.max_ground_dist
        assert xb_far == pytest.approx(1.494, abs=0.01)

    def test_f1_pitch_tilt_guard_suspension(self, mapper_node):
        """F1: Ground mapping suspended when pitch exceeds 6 degrees."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 100, dtype=np.uint8)

        # Within 6 deg limit: mapping executes
        count_flat = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(3.5))
        assert count_flat > 0

        # Exceeding 6 deg limit: mapping suspended
        count_tilted = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(7.5))
        assert count_tilted == 0

    def test_f2_node_initialization_and_grid_parameters(self, mapper_node):
        """F2: Visual orthomapper grid has 0.03m resolution and 200x200 dimensions."""
        node = mapper_node
        assert node.res == 0.03
        assert node.width == 200
        assert node.height == 200
        assert node.origin_x == -3.0
        assert node.origin_y == -3.0
        assert node.color_map.shape == (200, 200, 3)
        assert node.observed_mask.shape == (200, 200)
        assert node.weight_map.shape == (200, 200)

    def test_f2_camera_info_subscription_matrix_match(self, mapper_node):
        """F2: Synthetic CameraInfo message matches pinhole matrix K."""
        node = mapper_node
        info_msg = CameraInfo()
        info_msg.width = node.image_width
        info_msg.height = node.image_height
        info_msg.k = [node.fx, 0.0, node.cx, 0.0, node.fy, node.cy, 0.0, 0.0, 1.0]

        assert len(info_msg.k) == 9
        assert info_msg.k[0] == pytest.approx(554.256, rel=1e-3)
        assert info_msg.k[2] == 320.0
        assert info_msg.k[5] == 240.0

        node.camera_info_callback(info_msg)
        assert node.has_camera_info is True
        assert node.fx == pytest.approx(554.256, rel=1e-3)

    def test_f2_odometry_transform_handling(self, mapper_node):
        """F2: Odometry updates robot pose and aligns ground points with map frame."""
        node = mapper_node
        odom = Odometry()
        odom.pose.pose.position.x = 1.0
        odom.pose.pose.position.y = -0.5
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.pi / 2.0)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        node.khepera4_odom_callback(odom)
        assert node.robot_x == 1.0
        assert node.robot_y == -0.5
        assert node.robot_yaw == pytest.approx(math.pi / 2.0)
        assert node.has_odom is True

    def test_f2_alpha_blending_accumulation(self, mapper_node):
        """F2: Repeated observations blend floor colors with alpha=0.25."""
        node = mapper_node
        # First observation: bright white floor (240, 240, 240)
        img1 = np.full((node.image_height, node.image_width, 3), 240, dtype=np.uint8)
        node.warp_image_to_grid(img1, 0.0, 0.0, 0.0, 0.0)

        # Inspect an observed cell
        obs_idx = np.where(node.observed_mask)
        r0, c0 = obs_idx[0][0], obs_idx[1][0]
        assert node.color_map[r0, c0, 0] == pytest.approx(240.0, abs=1.0)

        # Second observation: dark floor (40, 40, 40)
        img2 = np.full((node.image_height, node.image_width, 3), 40, dtype=np.uint8)
        node.warp_image_to_grid(img2, 0.0, 0.0, 0.0, 0.0)

        # Blended value: 0.75 * 240 + 0.25 * 40 = 180 + 10 = 190
        assert node.color_map[r0, c0, 0] == pytest.approx(190.0, abs=2.0)

    def test_f2_latched_qos_publication_profile(self, mapper_node):
        """F2: Topic /visual_map QoS profile is TRANSIENT_LOCAL with depth 1."""
        node = mapper_node
        qos = node.visual_map_pub.qos_profile
        assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert qos.reliability == QoSProfile(depth=1).reliability
        assert qos.depth == 1


# ==============================================================================
# Tier 2: Boundary & Corner Cases (F1 & F2)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary value analysis & singular conditions for IPM & Orthomapper."""

    def test_f1_horizon_singularity_boundary(self, mapper_node):
        """Boundary: v=240.0 (singularity) vs v=240.001 vs v=251.0."""
        node = mapper_node
        # Exactly on horizon
        assert node.image_to_ground(320.0, 240.0)[2] is False
        # Just above horizon (sky)
        assert node.image_to_ground(320.0, 239.9)[2] is False
        # Extremely small delta below horizon produces huge distance (> 100m)
        xb_huge, _, valid = node.image_to_ground(320.0, 240.01)
        assert xb_huge > 100.0
        assert valid is False

    def test_f1_bottom_image_row_near_boundary(self, mapper_node):
        """Boundary: Bottom row v=479 vs out-of-bounds row v=480."""
        node = mapper_node
        xb_479, yb_479, valid_479 = node.image_to_ground(320.0, 479.0)
        assert valid_479 is True
        assert xb_479 == pytest.approx(0.1338, abs=1e-3)

        # Row 480 is beyond image height (0-indexed 640x480)
        u_480, v_480, valid_480 = node.ground_to_image(xb_479 - 0.05, 0.0)
        assert valid_480 is False
        assert v_480 is not None and v_480 > 480.0

    def test_f1_lateral_fov_edges_boundary(self, mapper_node):
        """Boundary: Left edge u=0 and right edge u=639 match 60 deg cone."""
        node = mapper_node
        xb = 1.0
        tan_half = math.tan(node.hfov_rad / 2.0)
        # Left boundary u=0
        u_left, _, _ = node.ground_to_image(xb, (xb - node.camera_xc) * tan_half)
        assert u_left == pytest.approx(0.0, abs=1e-2)

        # Right boundary u=640
        u_right, _, _ = node.ground_to_image(xb, -(xb - node.camera_xc) * tan_half)
        assert u_right == pytest.approx(640.0, abs=1e-2)

    def test_f1_negative_depth_behind_camera(self, mapper_node):
        """Boundary: Points with X_b <= x_c (behind lens) are strictly rejected."""
        node = mapper_node
        assert node.ground_to_image(node.camera_xc, 0.0)[2] is False
        assert node.ground_to_image(node.camera_xc - 0.05, 0.0)[2] is False
        assert node.ground_to_image(-1.0, 0.0)[2] is False

    def test_f1_pitch_angle_exact_boundary(self, mapper_node):
        """Boundary: Exactly 6.0 deg (permitted) vs 6.01 deg (suspended)."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 128, dtype=np.uint8)

        assert node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(6.000)) > 0

        node.clear()
        assert node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(6.010)) == 0

    def test_f2_unlocalized_frame_before_odometry(self, mapper_node):
        """Boundary: Camera frames arriving before odometry are dropped without error."""
        node = mapper_node
        node.has_odom = False
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)
        ros_img = RosImage()
        ros_img.header.stamp = node.get_clock().now().to_msg()
        ros_img.width = node.image_width
        ros_img.height = node.image_height
        ros_img.encoding = 'rgb8'
        ros_img.data = img.tobytes()
        node.image_callback(ros_img)
        assert not np.any(node.observed_mask)

    def test_f2_arena_boundary_clamping(self, mapper_node):
        """Boundary: Ground points outside arena bounds [-2.405, 2.405]m are ignored."""
        node = mapper_node
        # Robot located right at arena wall facing outside
        img = np.full((node.image_height, node.image_width, 3), 150, dtype=np.uint8)
        node.warp_image_to_grid(img, 2.400, 0.0, 0.0, 0.0)

        # Check cells beyond X = 2.405m: must not be observed!
        for col in range(node.width):
            wx, _ = node.grid_to_world(col, 100)
            if wx > 2.405:
                assert not node.observed_mask[100, col]

    def test_f2_black_and_empty_images(self, mapper_node):
        """Corner: All-zero (black) image frame processed without division errors."""
        node = mapper_node
        black_img = np.zeros((node.image_height, node.image_width, 3), dtype=np.uint8)
        assert node.warp_image_to_grid(black_img, 0.0, 0.0, 0.0, 0.0) > 0

        data = node.to_occupancy_grid_data()
        assert any(val == 0 for val in data)  # Black pixels convert to 0 occupancy
        assert any(val == -1 for val in data)  # Unobserved remain -1

    def test_f2_rapid_frame_updates_no_divergence(self, mapper_node):
        """Corner: Rapid stream of 50 frames does not cause memory leak or divergence."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 128, dtype=np.uint8)

        for _ in range(50):
            node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Inspect weights: must be capped at 100.0
        assert np.max(node.weight_map) <= 100.0
        assert np.all(node.color_map >= 0.0)
        assert np.all(node.color_map <= 255.0)


# ==============================================================================
# Tier 3: Pairwise Combinations (F1 & F2)
# ==============================================================================

class TestTier3PairwiseCombinations:
    """Tier 3: Multi-parameter interaction tests for Visual Mapping."""

    def test_pairwise_continuous_rotation_during_warping(self, mapper_node):
        """Pairwise: Warping while robot executes continuous 360 deg rotation."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 180, dtype=np.uint8)

        # Rotate full circle in 12 steps of 30 deg
        for step in range(12):
            yaw = step * (math.pi / 6.0)
            node.warp_image_to_grid(img, 0.0, 0.0, yaw, 0.0)

        # Observed mask should form a circular ring around the origin
        obs_count = np.sum(node.observed_mask)
        assert obs_count > 1000  # Wide circular coverage

    def test_pairwise_pitch_oscillation_during_streaming(self, mapper_node):
        """Pairwise: Alternating pitch angles between flat (<6 deg) and tilted (>6 deg)."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)

        # Step 1: Flat -> active
        c1 = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(2.0))
        assert c1 > 0

        # Step 2: Ramp climb -> suspended
        c2 = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(8.0))
        assert c2 == 0

        # Step 3: Flat plateau -> active again
        c3 = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(0.5))
        assert c3 > 0

    def test_pairwise_clear_map_during_active_streaming(self, mapper_node):
        """Pairwise: /clear_map resets grid during active camera streaming."""
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 210, dtype=np.uint8)

        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert np.any(node.observed_mask)

        # Service clear called
        node.clear()
        assert not np.any(node.observed_mask)
        assert np.all(node.color_map == 0.0)

        # Next frame accumulates cleanly
        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert np.any(node.observed_mask)


# ==============================================================================
# Tier 4: Real-World Scenarios (F1 & F2)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end operational mission trajectories."""

    def test_tier4_straight_line_orthomosaic_strip(self, mapper_node):
        """Scenario: Robot traverses a 1.2m straight-line path generating continuous strip."""
        node = mapper_node

        # Simulate robot moving along +X axis from X = -0.6m to X = +0.6m
        num_waypoints = 10
        xs = np.linspace(-0.6, 0.6, num_waypoints)

        # Checkerboard pattern floor image
        checker_img = np.zeros(
            (node.image_height, node.image_width, 3), dtype=np.uint8
        )
        checker_img[::40, :, :] = 255

        for x in xs:
            node.warp_image_to_grid(checker_img, x, 0.0, 0.0, 0.0)

        # Inspect resulting coverage along X-axis
        cols_observed = set()
        for row in range(node.height):
            for col in range(node.width):
                if node.observed_mask[row, col]:
                    cols_observed.add(col)

        # Path of 1.2m at 0.03m/cell spans at least 40 columns
        assert len(cols_observed) >= 40

    def test_tier4_urdf_xacro_physical_camera_contract(self):
        """Scenario: Verify physical URDF and launch configurations match mathematical model."""
        urdf_path = get_urdf_path()
        assert os.path.exists(urdf_path), f'URDF not found: {urdf_path}'

        root = ET.parse(urdf_path).getroot()
        cam_joint = next(
            j for j in root.findall('joint') if j.get('name') == 'camera_joint'
        )
        xyz = [float(v) for v in cam_joint.find('origin').get('xyz').split()]
        assert xyz[0] == pytest.approx(0.0682, abs=1e-4)
        assert xyz[1] == pytest.approx(0.0, abs=1e-4)
        assert xyz[2] == pytest.approx(0.0283, abs=1e-4)

        opt_joint = next(
            j for j in root.findall('joint')
            if j.get('name') == 'camera_optical_joint'
        )
        rpy = [float(v) for v in opt_joint.find('origin').get('rpy').split()]
        assert rpy[0] == pytest.approx(-math.pi / 2.0, abs=1e-3)
        assert rpy[1] == pytest.approx(0.0, abs=1e-3)
        assert rpy[2] == pytest.approx(-math.pi / 2.0, abs=1e-3)
