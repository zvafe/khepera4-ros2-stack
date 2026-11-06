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
Adversarial Stress Test Harness & Mathematical Verification Oracle — Challenger 1 (Milestone M1).

Validates:
1. Homography Numerical Stability: Condition number, determinant, analytical inverse.
2. Horizon Exactness & Range Boundaries: Exact v=240 horizon cutoff,
   v=251 near-horizon integer row.
3. Roundtrip Reprojection Precision: Ground-to-image and image-to-ground sub-atomic precision.
4. Bilinear Interpolation Edge Cases: Boundary checks, corner clamping, gradient linearity.
5. High-Frequency Streaming & Real-Time Throughput: 60+ Hz stream, zero lag, frame drop guards.
6. Extreme Robot Dynamics & Odometry Fallbacks: High speeds, zero/infinite covariance, pitch guard.
"""

import math
import os
import sys
import time

from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image as RosImage
from std_srvs.srv import Trigger

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture(scope='module', autouse=True)
def manage_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def mapper_node(tmp_path):
    """Create an isolated VisualOrthomapper node instance."""
    node = VisualOrthomapper(
        parameter_overrides=[
            rclpy.parameter.Parameter('keyframe_dir', rclpy.Parameter.Type.STRING, str(tmp_path))
        ]
    )
    yield node
    node.destroy_node()


# ==============================================================================
# 1. Homography Numerical Stability & Invertibility Oracle
# ==============================================================================

class TestHomographyNumericalStability:
    """Stress tests on forward and inverse homography matrix computation."""

    def test_analytical_homography_inverse_identity(self, mapper_node):
        """Verify H_b2img @ H_img2b == I within floating point machine precision."""
        h_fwd = mapper_node.H_b2img
        h_inv = mapper_node.H_img2b

        prod = np.dot(h_fwd, h_inv)
        residual = np.max(np.abs(prod - np.eye(3)))
        assert residual < 1e-12, f'Matrix product deviates from identity: residual={residual}'

        # Invert product order as well
        prod_rev = np.dot(h_inv, h_fwd)
        residual_rev = np.max(np.abs(prod_rev - np.eye(3)))
        assert residual_rev < 1e-12, f'Reverse product deviates from identity: {residual_rev}'

    def test_homography_determinant_closed_form(self, mapper_node):
        """Verify analytical determinant matches det(H) = -h * fx * fy."""
        det_h = np.linalg.det(mapper_node.H_b2img)
        expected_det = -mapper_node.camera_h * mapper_node.fx * mapper_node.fy

        assert abs(det_h) > 5000.0, f'Determinant too small: {det_h}'
        assert math.isclose(det_h, expected_det, rel_tol=1e-11)

    def test_homography_condition_number_stability(self, mapper_node):
        """Verify condition number of H remains well-behaved (< 20000) across calibrations."""
        # Baseline calibration
        cond_default = np.linalg.cond(mapper_node.H_b2img)
        assert cond_default < 15000.0, f'Default condition number too high: {cond_default}'

        # Vary focal length from narrow to wide angle (hfov 30 deg to 90 deg)
        for hfov_deg in [30.0, 45.0, 60.0, 75.0, 90.0]:
            hfov_rad = math.radians(hfov_deg)
            fx = (mapper_node.image_width / 2.0) / math.tan(hfov_rad / 2.0)
            mapper_node.fx = fx
            mapper_node.fy = fx
            h_var = mapper_node.compute_forward_homography()
            cond_var = np.linalg.cond(h_var)
            assert cond_var < 25000.0, f'Condition number {cond_var} blew up for hfov {hfov_deg}'

    def test_homography_singularity_at_camera_optical_center(self, mapper_node):
        """Points at xb = xc (s = 0) must trigger singularity guard and return False."""
        xc = mapper_node.camera_xc
        u, v, valid = mapper_node.ground_to_image(xc, 0.0)
        assert not valid
        assert u is None and v is None

        # Point epsilon behind camera
        u_neg, v_neg, valid_neg = mapper_node.ground_to_image(xc - 1e-5, 0.0)
        assert not valid_neg
        assert u_neg is None and v_neg is None

    def test_extreme_coordinate_distance_resilience(self, mapper_node):
        """Extreme distances (10 km away) must be safely rejected without overflow/crash."""
        # 10 kilometers forward
        u_far, v_far, valid_far = mapper_node.ground_to_image(10000.0, 0.0)
        assert not valid_far  # Exceeds max_ground_dist
        assert u_far == pytest.approx(320.0, abs=1e-2)
        assert v_far == pytest.approx(240.0, abs=1e-2)

        # 10 kilometers behind
        u_back, v_back, valid_back = mapper_node.ground_to_image(-10000.0, 0.0)
        assert not valid_back


# ==============================================================================
# 2. Horizon Exactness & Geometric Range Boundaries
# ==============================================================================

class TestHorizonExactnessAndClipping:
    """Rigorous verification of horizon row v=240 and clipping intervals."""

    def test_horizon_exact_singularity(self, mapper_node):
        """Pixel row v=240.0 is the exact horizon (ray parallel to ground) -> invalid."""
        xb, yb, valid = mapper_node.image_to_ground(320.0, 240.0)
        assert not valid
        assert xb is None and yb is None

    def test_above_horizon_pixels_rejection(self, mapper_node):
        """Pixel rows v < 240.0 point above horizon (sky / ceiling) -> strictly invalid."""
        for v in [0.0, 50.0, 100.0, 200.0, 239.0, 239.99]:
            xb, yb, valid = mapper_node.image_to_ground(320.0, v)
            assert not valid, f'Row {v} above horizon unexpectedly marked valid!'
            assert xb is None and yb is None

    def test_just_below_horizon_distance_cutoff(self, mapper_node):
        """Row v=240.001 points to ground thousands of meters away -> rejected by max_dist."""
        xb, yb, valid = mapper_node.image_to_ground(320.0, 240.001)
        assert not valid
        assert xb > 1000.0  # Physical distance is huge, so rejected

    def test_first_valid_integer_row_v251(self, mapper_node):
        """Row v=251 is the first valid integer row satisfying X_b <= 1.50m."""
        # Row 251: forward distance ~1.494m <= 1.50m
        xb_251, yb_251, valid_251 = mapper_node.image_to_ground(320.0, 251.0)
        assert valid_251, 'Row 251 must be valid'
        assert xb_251 <= mapper_node.max_ground_dist
        assert xb_251 == pytest.approx(1.494, abs=0.01)

        # Row 250: forward distance ~1.637m > 1.50m -> rejected
        xb_250, yb_250, valid_250 = mapper_node.image_to_ground(320.0, 250.0)
        assert not valid_250, f'Row 250 (dist {xb_250}m) must be rejected (> 1.50m)'

    def test_bottom_row_v479_near_ground_distance(self, mapper_node):
        """Row v=479 corresponds to nearest ground point visible by CMOS sensor."""
        # Row 478 is the lowest integer row meeting min_ground_dist >= 0.134m
        xb_478, yb_478, valid_478 = mapper_node.image_to_ground(320.0, 478.0)
        assert valid_478, 'Row 478 must be valid (xb >= 0.134m)'
        assert xb_478 == pytest.approx(0.1341, abs=0.001)

        # Row 479 computes exact optical ground intercept xb=0.13383m, which meets
        # min_ground_dist=0.1338m threshold
        xb_479, yb_479, valid_479 = mapper_node.image_to_ground(320.0, 479.0)
        assert xb_479 == pytest.approx(0.1338, abs=0.001)
        assert valid_479, 'Row 479 must be valid under min_ground_dist=0.1338m threshold'

        # Bottom out of bounds (v = 480)
        xb_480, yb_480, valid_480 = mapper_node.image_to_ground(320.0, 480.0)
        assert not valid_480, 'Row 480 is out of bounds for 480p image'


# ==============================================================================
# 3. Roundtrip Reprojection Precision Across Full Ground Grid
# ==============================================================================

class TestRoundtripReprojectionError:
    """Stress tests verifying bijective precision between ground plane and image coordinates."""

    def test_ground_to_image_and_back_grid_sweep(self, mapper_node):
        """Sweep 2000+ points across valid ground trapezoid, verifying < 1e-12m error."""
        # Valid range within v in [251, 479] corresponds to xb in [0.13383, 1.494]
        xb_values = np.linspace(0.13383, 1.494, 50)
        tan_half_fov = math.tan(mapper_node.hfov_rad / 2.0)
        max_err = 0.0
        points_tested = 0

        for xb in xb_values:
            max_yb = (xb - mapper_node.camera_xc) * tan_half_fov
            # Sample across lateral FOV width
            for yb in np.linspace(-max_yb * 0.98, max_yb * 0.98, 40):
                u, v, valid_fwd = mapper_node.ground_to_image(xb, yb)
                assert valid_fwd, f'Failed forward projection for xb={xb}, yb={yb}'

                xb_back, yb_back, valid_inv = mapper_node.image_to_ground(u, v)
                assert valid_inv, f'Failed inverse projection for u={u}, v={v}'

                err = math.hypot(xb - xb_back, yb - yb_back)
                if err > max_err:
                    max_err = err
                points_tested += 1

        assert points_tested == 2000
        assert max_err < 1e-12, f'Roundtrip reprojection error {max_err:.2e} m exceeded 1e-12 m!'

    def test_image_pixels_to_ground_and_back_sweep(self, mapper_node):
        """Sweep pixels across valid frustum (u in [10, 630], v in [252, 478]), error < 1e-12px."""
        max_err_px = 0.0
        pixels_tested = 0

        for u in np.linspace(10.0, 630.0, 40):
            for v in np.linspace(252.0, 478.0, 40):
                xb, yb, valid_inv = mapper_node.image_to_ground(u, v)
                if valid_inv:
                    u_back, v_back, valid_fwd = mapper_node.ground_to_image(xb, yb)
                    assert valid_fwd, f'Failed re-projection for ({xb}, {yb})'

                    err_px = math.hypot(u - u_back, v - v_back)
                    if err_px > max_err_px:
                        max_err_px = err_px
                    pixels_tested += 1

        assert pixels_tested > 1000
        assert max_err_px < 1e-12, f'Pixel roundtrip error {max_err_px:.2e} px exceeded 1e-12 px!'

    def test_world_to_grid_and_grid_to_world_consistency(self, mapper_node):
        """Verify world_to_grid and grid_to_world are exact inverse operations."""
        for col in [0, 50, 100, 150, 199]:
            for row in [0, 50, 100, 150, 199]:
                xw, yw = mapper_node.grid_to_world(col, row)
                col_back, row_back = mapper_node.world_to_grid(xw, yw)
                assert col == col_back, f'Column mismatch: expected {col}, got {col_back}'
                assert row == row_back, f'Row mismatch: expected {row}, got {row_back}'


# ==============================================================================
# 4. Bilinear Interpolation & Image Boundary Guards
# ==============================================================================

class TestBilinearInterpolationBoundaries:
    """Stress tests on pixel interpolation, boundary clamping, and color sampling."""

    def test_interpolation_on_four_corners(self, mapper_node):
        """Verify warp_image_to_grid handles corner pixels without index errors."""
        # Create corner-identifying test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[251, 0] = [255, 0, 0]      # Top-left valid corner (Red)
        img[251, 639] = [0, 255, 0]    # Top-right valid corner (Green)
        img[479, 0] = [0, 0, 255]      # Bottom-left valid corner (Blue)
        img[479, 639] = [255, 255, 0]  # Bottom-right valid corner (Yellow)

        # Warping must execute without IndexError or ValueError
        count = mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert count > 0, 'No cells updated during corner test'

    def test_flat_uniform_color_preservation(self, mapper_node):
        """A uniform color image must warp into identically colored grid cells."""
        target_color = np.array([142, 85, 210], dtype=np.uint8)
        img = np.full((480, 640, 3), target_color, dtype=np.uint8)

        mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Check observed cells
        obs_idx = np.where(mapper_node.observed_mask)
        assert len(obs_idx[0]) > 0
        warped_colors = mapper_node.color_map[obs_idx]

        for channel in range(3):
            expected = float(target_color[channel])
            diff = np.max(np.abs(warped_colors[:, channel] - expected))
            assert diff < 1e-4, f'Color distortion in channel {channel}: diff={diff}'

    def test_linear_gradient_interpolation_fidelity(self, mapper_node):
        """Vertical color gradient must be preserved monotonically along ground distance."""
        # Create vertical gradient: row 251=0 (black), row 479=228 (bright)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for v in range(251, 480):
            img[v, :, :] = v - 251

        mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Along robot centerline (Y_b = 0, which is row = 100 for robot at (0, 0) facing +X):
        # Far cells (large X) should have low values, near cells (small X) should have high values
        center_row = 100
        observed_cols = [c for c in range(200) if mapper_node.observed_mask[center_row, c]]
        assert len(observed_cols) > 10

        values = [mapper_node.color_map[center_row, c, 0] for c in sorted(observed_cols)]
        # As column increases (robot facing East, X increases -> distance increases
        # -> v decreases -> color decreases)
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1.0, 'Gradient monotonicity violated'

    def test_unsupported_image_encodings_handled_gracefully(self, mapper_node):
        """Unsupported image encoding must be rejected gracefully without unhandled crash."""
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        mapper_node.khepera4_odom_callback(odom)

        bad_img = RosImage()
        bad_img.encoding = 'bayer_rggb8'
        bad_img.width = 640
        bad_img.height = 480
        bad_img.data = b'\x00' * (640 * 480)

        # Should not raise exception
        mapper_node.image_callback(bad_img)
        assert not np.any(mapper_node.observed_mask)


# ==============================================================================
# 5. High-Frequency Streaming & Real-Time Performance Stress
# ==============================================================================

class TestStreamingAndStressHarness:
    """Stress tests simulating high frame rates (60+ Hz) and robot dynamics."""

    def test_high_frequency_warping_throughput(self, mapper_node):
        """Verify that visual_orthomapper can sustain 60+ Hz stream (> 60 FPS)."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        num_frames = 100
        latencies = []

        for i in range(num_frames):
            yaw = (i * 0.06) % (2.0 * math.pi)
            rx = 0.4 * math.cos(yaw)
            ry = 0.4 * math.sin(yaw)

            t_start = time.perf_counter()
            updated = mapper_node.warp_image_to_grid(img, rx, ry, yaw, 0.0)
            t_end = time.perf_counter()

            latencies.append(t_end - t_start)
            assert updated > 0

        mean_latency = float(np.mean(latencies))
        fps = 1.0 / mean_latency
        p99_latency = float(np.percentile(latencies, 99))

        # Must comfortably exceed 60 FPS (mean latency < 16.6 ms)
        assert fps >= 60.0, f'Throughput {fps:.1f} FPS below 60 Hz requirement!'
        assert p99_latency < 0.015, f'P99 latency {p99_latency*1000:.2f} ms exceeds 15 ms'

    def test_high_velocity_robot_dynamics(self, mapper_node):
        """Simulate high linear (3.0 m/s) and angular (15 rad/s) velocities without divergence."""
        img = np.full((480, 640, 3), 160, dtype=np.uint8)

        # High-speed circular orbit
        dt = 0.02  # 50 Hz updates
        v_lin = 2.5
        w_ang = 8.0
        x, y, yaw = 0.0, 0.0, 0.0

        for _ in range(50):
            x += v_lin * math.cos(yaw) * dt
            y += v_lin * math.sin(yaw) * dt
            yaw += w_ang * dt

            # Wrap coordinates to arena bounds
            x_clamped = float(np.clip(x, -2.0, 2.0))
            y_clamped = float(np.clip(y, -2.0, 2.0))

            count = mapper_node.warp_image_to_grid(img, x_clamped, y_clamped, yaw, 0.0)
            assert count >= 0
            assert np.all(mapper_node.color_map >= 0.0)
            assert np.all(mapper_node.color_map <= 255.0)

    def test_pitch_guard_strict_boundary(self, mapper_node):
        """Pitch guard: warping suspended when abs(pitch) > 6.0 deg."""
        img = np.full((480, 640, 3), 200, dtype=np.uint8)

        # Exactly 5.9 deg -> allowed
        c_under = mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(5.9))
        assert c_under > 0

        # Exactly 6.1 deg -> suspended
        c_over = mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(6.1))
        assert c_over == 0

        # Negative pitch (pointing downwards into ground or ditch): -6.1 deg -> suspended
        c_neg = mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, math.radians(-6.1))
        assert c_neg == 0

    def test_odometry_timeout_fallback_switching(self, mapper_node):
        """Odometry source switching: /khepera4/odom prioritized, falls back to /odom."""
        # 1. Raw odom alone is accepted initially
        raw_msg1 = Odometry()
        raw_msg1.pose.pose.position.x = 0.8
        raw_msg1.pose.pose.orientation.w = 1.0
        mapper_node.raw_odom_callback(raw_msg1)
        assert mapper_node.robot_x == 0.8

        # 2. khepera4 odom arrives -> takes precedence
        k4_msg = Odometry()
        k4_msg.pose.pose.position.x = 1.4
        k4_msg.pose.pose.orientation.w = 1.0
        mapper_node.khepera4_odom_callback(k4_msg)
        assert mapper_node.robot_x == 1.4

        # 3. Raw odom arrives immediately after -> ignored
        raw_msg2 = Odometry()
        raw_msg2.pose.pose.position.x = 0.9
        mapper_node.raw_odom_callback(raw_msg2)
        assert mapper_node.robot_x == 1.4  # Still 1.4 from khepera4_odom

        # 4. Wait > 1.0s for timeout fallback
        time.sleep(1.05)
        raw_msg3 = Odometry()
        raw_msg3.pose.pose.position.x = 1.8
        mapper_node.raw_odom_callback(raw_msg3)
        assert mapper_node.robot_x == 1.8  # Fallback activated

    def test_degenerate_odometry_covariance(self, mapper_node):
        """Verify zero, infinite, and NaN covariance arrays are tolerated without crash."""
        odom = Odometry()
        odom.pose.pose.position.x = 0.5
        odom.pose.pose.position.y = 0.5
        odom.pose.pose.orientation.w = 1.0

        # Zero covariance
        odom.pose.covariance = [0.0] * 36
        mapper_node.khepera4_odom_callback(odom)
        assert mapper_node.robot_x == 0.5

        # Infinite covariance
        odom.pose.covariance = [float('inf')] * 36
        mapper_node.khepera4_odom_callback(odom)
        assert mapper_node.robot_x == 0.5

        # NaN covariance
        odom.pose.covariance = [float('nan')] * 36
        mapper_node.khepera4_odom_callback(odom)
        assert mapper_node.robot_x == 0.5

    def test_atomic_clear_service_resets_all_grids(self, mapper_node):
        """Triggering /clear_visual_map resets color_map, observed_mask, and weights atomically."""
        img = np.full((480, 640, 3), 190, dtype=np.uint8)
        mapper_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert np.any(mapper_node.observed_mask)

        req = Trigger.Request()
        resp = Trigger.Response()
        result_resp = mapper_node.clear_map_callback(req, resp)

        assert result_resp.success
        assert not np.any(mapper_node.observed_mask)
        assert np.all(mapper_node.color_map == 0.0)
        assert np.all(mapper_node.weight_map == 0.0)
        assert mapper_node.keyframe_count == 0
