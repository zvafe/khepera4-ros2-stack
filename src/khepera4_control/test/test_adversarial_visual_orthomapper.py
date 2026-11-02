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
Adversarial Stress Test Suite: Visual Orthomapper Node (Milestone M1).

Conducted by Challenger 2 (challenger_v2_m1_2).
Empirically stress-tests physical edge cases:
1. Pitch tilt perturbation:
   - Pitch angle <= 6.0 deg permits floor warping.
   - Pitch angle > 6.0 deg cleanly suspends floor warping without error or memory leaks.
   - Rapid oscillation across the 6.0 deg boundary.
   - Extreme tilt angles (+/-89 deg, +/-90 deg, +/-180 deg).
2. Arena boundary clamping:
   - All warped ground points outside [-2.405, 2.405]m are strictly clipped.
   - Array bounds: no writes outside the 200x200 array under any pose or angle.
   - Direct calls with out-of-arena coordinates.
3. Odometry fallback:
   - Priority of /khepera4/odom over /odom.
   - Fallback to /odom when /khepera4/odom ceases publishing (> 1.0s timeout).
   - Instant re-assertion of priority when /khepera4/odom resumes.
   - Intermittent dropout stress testing.
"""

import math
import os
import sys
from typing import Tuple

from khepera4_control.common import quaternion_from_euler
from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image as RosImage

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


# ==============================================================================
# Pytest Fixtures & Helpers
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def rclpy_session():
    """Initialize and shut down rclpy for node testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def orthomapper_node():
    """Create a clean VisualOrthomapper node instance for each test."""
    node = VisualOrthomapper()
    yield node
    node.destroy_node()


def create_test_image(height: int = 480, width: int = 640,
                      rgb: Tuple[int, int, int] = (180, 200, 220)) -> RosImage:
    """Create a synthetic ROS Image message with specified uniform color."""
    msg = RosImage()
    msg.header.stamp = Time(seconds=1, nanoseconds=0).to_msg()
    msg.header.frame_id = 'camera_optical_frame'
    msg.height = height
    msg.width = width
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = width * 3

    img_data = np.full((height, width, 3), rgb, dtype=np.uint8)
    msg.data = img_data.tobytes()
    return msg


def create_odom_msg(x: float, y: float, z: float = 0.0,
                    roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                    stamp_sec: float = 1.0) -> Odometry:
    """Create a synthetic Odometry message."""
    msg = Odometry()
    msg.header.stamp = Time(seconds=int(stamp_sec),
                            nanoseconds=int((stamp_sec % 1.0) * 1e9)).to_msg()
    msg.header.frame_id = 'map'
    msg.child_frame_id = 'base_footprint'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)

    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    return msg


# ==============================================================================
# Category 1: Pitch Tilt Perturbation Stress Tests
# ==============================================================================

class TestPitchTiltPerturbation:
    """Empirical challenge on pitch tilt guard boundary and physical slope behavior."""

    def test_pitch_guard_operational_range_warp_method(self, orthomapper_node):
        """Verify warp_image_to_grid operational behavior across pitch angles."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 150, dtype=np.uint8)

        # Flat: 0 deg -> permits warping
        c_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, 0.0)
        assert c_0 > 0, 'Flat ground (0.0 deg) must permit warping.'

        # Within bounds: 3.0 deg -> permits warping
        c_3_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(3.0))
        assert c_3_0 > 0, 'Pitch of 3.0 deg must permit warping.'

        # Just below boundary: 5.9 deg -> permits warping
        c_5_9 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(5.9))
        assert c_5_9 > 0, 'Pitch of 5.9 deg must permit warping.'

        # 5.99 deg -> permits warping
        c_5_99 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(5.99))
        assert c_5_99 > 0, 'Pitch of 5.99 deg must permit warping.'

        # Exceeding boundary: 6.01 deg -> suspends warping
        c_6_01 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(6.01))
        assert c_6_01 == 0, 'Pitch of 6.01 deg must suspend warping.'

        # Ramp climb: 10.0 deg -> suspends warping
        c_10_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(10.0))
        assert c_10_0 == 0, 'Pitch of 10.0 deg (ramp climb) must suspend warping.'

        # Negative within bounds: -3.0 deg -> permits warping
        c_neg_3_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(-3.0))
        assert c_neg_3_0 > 0, 'Pitch of -3.0 deg must permit warping.'

        # Negative just below boundary: -5.99 deg -> permits warping
        c_neg_5_99 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(-5.99))
        assert c_neg_5_99 > 0, 'Pitch of -5.99 deg must permit warping.'

        # Negative exceeding boundary: -6.01 deg -> suspends warping
        c_neg_6_01 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(-6.01))
        assert c_neg_6_01 == 0, 'Pitch of -6.01 deg must suspend warping.'

        # Negative ramp descent: -10.0 deg -> suspends warping
        c_neg_10_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, math.radians(-10.0))
        assert c_neg_10_0 == 0, 'Pitch of -10.0 deg (ramp descent) must suspend warping.'

    def test_pitch_guard_exact_boundary_warp_method(self, orthomapper_node):
        """Empirically demonstrate that exact 6.0 deg boundary permits warping."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 150, dtype=np.uint8)

        # Exact 6.0 deg in radians
        rad_6_0 = math.radians(6.0)
        c_6_0 = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, rad_6_0)
        assert c_6_0 > 0, (
            f'Specification requires pitch <= 6.0 deg to permit warping, but got {c_6_0}'
        )

    test_pitch_guard_exact_boundary_floating_point_defect = (
        test_pitch_guard_exact_boundary_warp_method
    )

    def test_pitch_guard_boundary_angles_sweep(self, orthomapper_node):
        """Empirically stress-test boundary angles: 0, 5.9, 6.0, 6.0001, 6.01, 10.0 deg."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 150, dtype=np.uint8)

        # Matrix of test angles and expected warping permission
        # Specification and tolerance rule:
        # pitch <= (6.0 + 1e-4) permits warping; pitch > (6.0 + 1e-4) suspends warping.
        angle_cases = [
            (0.0, True, 'Flat ground (0.0 deg) must permit warping'),
            (5.9, True, 'Pitch 5.9 deg must permit warping'),
            (6.0, True, 'Exact 6.0 deg must permit warping'),
            (6.00005, True, 'Pitch 6.00005 deg must permit warping'),
            (6.00010, True, 'Pitch 6.00010 deg within tolerance must permit warping'),
            (6.00011, False, 'Pitch 6.00011 deg beyond tolerance must suspend warping'),
            (6.001, False, 'Pitch 6.001 deg must suspend warping'),
            (6.01, False, 'Pitch 6.01 deg must suspend warping'),
            (10.0, False, 'Pitch 10.0 deg (ramp climb) must suspend warping'),
        ]

        for deg, should_permit, desc in angle_cases:
            # Positive angle
            rad_pos = math.radians(deg)
            cells_pos = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, rad_pos)
            if should_permit:
                assert cells_pos > 0, f'+{deg} deg: {desc} (got {cells_pos} cells)'
            else:
                assert cells_pos == 0, f'+{deg} deg: {desc} (got {cells_pos} cells)'

            # Negative angle (symmetric pitch down / descent)
            rad_neg = math.radians(-deg)
            cells_neg = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, rad_neg)
            if should_permit:
                assert cells_neg > 0, f'-{deg} deg: {desc} (got {cells_neg} cells)'
            else:
                assert cells_neg == 0, f'-{deg} deg: {desc} (got {cells_neg} cells)'

    def test_pitch_guard_boundary_angles_ros_pipeline(self, orthomapper_node):
        """Verify boundary angles 0, 5.9, 6.0, 6.0001, 6.01, 10.0 deg via ROS callbacks."""
        node = orthomapper_node
        node.enable_auto_keyframes = False
        img_msg = create_test_image(rgb=(190, 150, 100))

        angle_cases = [
            (0.0, True),
            (5.9, True),
            (6.0, True),
            (6.00010, True),
            (6.00011, False),
            (6.01, False),
            (10.0, False),
        ]

        for deg, should_permit in angle_cases:
            # Positive pitch
            node.clear()
            odom_pos = create_odom_msg(0.0, 0.0, pitch=math.radians(deg))
            node.khepera4_odom_callback(odom_pos)
            node.image_callback(img_msg)
            obs_pos = int(np.sum(node.observed_mask))
            if should_permit:
                assert obs_pos > 0, f'+{deg} deg via ROS callback must update visual map'
            else:
                assert obs_pos == 0, f'+{deg} deg via ROS callback must NOT update visual map'

            # Negative pitch
            node.clear()
            odom_neg = create_odom_msg(0.0, 0.0, pitch=math.radians(-deg))
            node.khepera4_odom_callback(odom_neg)
            node.image_callback(img_msg)
            obs_neg = int(np.sum(node.observed_mask))
            if should_permit:
                assert obs_neg > 0, f'-{deg} deg via ROS callback must update visual map'
            else:
                assert obs_neg == 0, f'-{deg} deg via ROS callback must NOT update visual map'

    def test_pitch_guard_via_ros_pipeline_callbacks(self, orthomapper_node):
        """Verify end-to-end ROS message callback behavior under pitch tilt."""
        node = orthomapper_node
        img_msg = create_test_image(rgb=(200, 100, 50))

        # 1. Deliver flat odometry (2.0 deg pitch)
        odom_flat = create_odom_msg(0.0, 0.0, pitch=math.radians(2.0))
        node.khepera4_odom_callback(odom_flat)
        assert node.has_odom is True
        assert math.isclose(math.degrees(node.robot_pitch), 2.0, abs_tol=1e-3)

        # Process image frame: must update visual map
        node.image_callback(img_msg)
        obs_count_flat = np.sum(node.observed_mask)
        assert obs_count_flat > 0, 'Flat ground frame must update visual map.'

        # 2. Reset visual map via service
        class MockReq:
            pass

        class MockResp:
            success = False
            message = ''

        node.clear_map_callback(MockReq(), MockResp())
        assert np.sum(node.observed_mask) == 0

        # 3. Deliver ramp pitch odometry (10 deg incline)
        odom_ramp = create_odom_msg(0.0, 0.0, pitch=math.radians(10.0))
        node.khepera4_odom_callback(odom_ramp)
        assert math.isclose(math.degrees(node.robot_pitch), 10.0, abs_tol=1e-3)

        # Process image frame during ramp climb: must NOT update visual map
        node.image_callback(img_msg)
        obs_count_ramp = np.sum(node.observed_mask)
        assert obs_count_ramp == 0, 'Visual map must NOT be updated when pitch is 10.0 deg.'

        # 4. Deliver pitch exceeding threshold (6.01 deg)
        odom_threshold_exceeded = create_odom_msg(0.0, 0.0, pitch=math.radians(6.01))
        node.khepera4_odom_callback(odom_threshold_exceeded)
        node.image_callback(img_msg)
        assert np.sum(node.observed_mask) == 0, 'Visual map must NOT be updated at 6.01 deg pitch.'

        # 5. Deliver pitch within permitted threshold (5.9 deg)
        odom_within_threshold = create_odom_msg(0.0, 0.0, pitch=math.radians(5.9))
        node.khepera4_odom_callback(odom_within_threshold)
        node.image_callback(img_msg)
        assert np.sum(node.observed_mask) > 0, 'Visual map must update when pitch is 5.9 deg.'

    def test_pitch_guard_rapid_oscillation_stability(self, orthomapper_node):
        """Adversarial stress: rapidly cycle pitch across the boundary 500 times."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 128, dtype=np.uint8)

        # 500 iterations alternating across the operational boundary
        angles = [0.0, 3.0, 5.9, 6.01, 10.0, -10.0, -6.01, -5.9, -3.0, 0.0]
        for i in range(500):
            pitch_deg = angles[i % len(angles)]
            pitch_rad = math.radians(pitch_deg)
            cells = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, pitch_rad)

            if abs(pitch_deg) > 6.0:
                assert cells == 0, f'Expected 0 cells at {pitch_deg} deg, got {cells}'
            else:
                assert cells > 0, f'Expected >0 cells at {pitch_deg} deg, got {cells}'

        # State integrity checks
        assert node.color_map.shape == (200, 200, 3)
        assert node.observed_mask.shape == (200, 200)
        assert node.weight_map.shape == (200, 200)

        assert not np.isnan(node.color_map).any()
        assert not np.isinf(node.color_map).any()
        assert np.all(node.color_map >= 0.0)
        assert np.all(node.color_map <= 255.0)

        assert not np.isnan(node.weight_map).any()
        assert np.all(node.weight_map <= 100.0)

    def test_pitch_guard_extreme_pitch_angles(self, orthomapper_node):
        """Test extreme pitch angles: +/-45 deg, +/-89.9 deg, +/-90.0 deg, 180 deg."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 200, dtype=np.uint8)

        extreme_angles = [
            45.0, -45.0, 89.9, -89.9, 90.0, -90.0,
            120.0, -120.0, 179.9, -179.9, 180.0, -180.0
        ]
        for deg in extreme_angles:
            rad = math.radians(deg)
            result = node.warp_image_to_grid(test_img, 0.0, 0.0, 0.0, rad)
            assert result == 0, f'Extreme pitch angle {deg} deg must return 0'


# ==============================================================================
# Category 2: Arena Boundary Clamping Stress Tests
# ==============================================================================

class TestArenaBoundaryClamping:
    """Empirical challenge on arena boundary clamping and memory array bounds."""

    def test_clamping_robot_placed_at_all_perimeter_edges(self, orthomapper_node):
        """Verify robot placed at arena perimeter walls never writes outside arena."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 220, dtype=np.uint8)

        boundary_poses = [
            (2.405, 0.0, 0.0, 'East wall looking East (out of arena)'),
            (-2.405, 0.0, math.pi, 'West wall looking West (out of arena)'),
            (0.0, 2.405, math.pi / 2.0, 'North wall looking North (out of arena)'),
            (0.0, -2.405, -math.pi / 2.0, 'South wall looking South (out of arena)'),
            (2.405, 2.405, math.pi / 4.0, 'NE corner looking outward'),
            (-2.405, 2.405, 3.0 * math.pi / 4.0, 'NW corner looking outward'),
            (-2.405, -2.405, -3.0 * math.pi / 4.0, 'SW corner looking outward'),
            (2.405, -2.405, -math.pi / 4.0, 'SE corner looking outward'),
        ]

        for rx, ry, ryaw, desc in boundary_poses:
            # Clear map before each test
            node.color_map.fill(0.0)
            node.observed_mask.fill(False)
            node.weight_map.fill(0.0)

            node.warp_image_to_grid(test_img, rx, ry, ryaw, 0.0)

            # Inspect all cells in the 200x200 grid
            rows, cols = np.where(node.observed_mask)
            for r, c in zip(rows, cols):
                assert 0 <= r < node.height, f'Row index {r} out of bounds in {desc}'
                assert 0 <= c < node.width, f'Col index {c} out of bounds in {desc}'

                wx, wy = node.grid_to_world(c, r)
                # Cell center world coordinates must strictly be inside arena
                assert -2.405 <= wx <= 2.405, (
                    f'Observed cell (r={r}, c={c}) at wx={wx:.4f}m outside '
                    f'arena X bounds in {desc}'
                )
                assert -2.405 <= wy <= 2.405, (
                    f'Observed cell (r={r}, c={c}) at wy={wy:.4f}m outside '
                    f'arena Y bounds in {desc}'
                )

    def test_clamping_full_360_rotation_at_arena_edge(self, orthomapper_node):
        """Assert full 360 rotation near perimeter never breaches bounds."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 160, dtype=np.uint8)

        for deg in range(0, 360, 10):
            yaw = math.radians(deg)
            node.warp_image_to_grid(test_img, 2.355, 0.0, yaw, 0.0)

        # Verify all accumulated cells are strictly inside [-2.405, 2.405]
        rows, cols = np.where(node.observed_mask)
        assert len(rows) > 0, 'Should have observed cells facing inside the arena.'

        for r, c in zip(rows, cols):
            wx, wy = node.grid_to_world(c, r)
            assert -2.405 <= wx <= 2.405, f'Cell wx={wx:.4f} breached East/West wall'
            assert -2.405 <= wy <= 2.405, f'Cell wy={wy:.4f} breached North/South wall'

    def test_clamping_extreme_out_of_bounds_odometry(self, orthomapper_node):
        """Verify extreme out-of-bounds odometry is clamped safely."""
        node = orthomapper_node
        test_img_msg = create_test_image(rgb=(100, 200, 100))

        extreme_positions = [
            (5.0, 5.0),
            (-10.0, 0.0),
            (0.0, 20.0),
            (-50.0, -50.0),
            (1000.0, 1000.0),
        ]

        for x_raw, y_raw in extreme_positions:
            odom = create_odom_msg(x_raw, y_raw)
            node.khepera4_odom_callback(odom)

            # Robot state must be clipped
            assert -2.405 <= node.robot_x <= 2.405, f'robot_x {node.robot_x} not clamped'
            assert -2.405 <= node.robot_y <= 2.405, f'robot_y {node.robot_y} not clamped'

            # Image callback must execute without IndexError
            node.image_callback(test_img_msg)

            # All observed cells must remain within arena
            rows, cols = np.where(node.observed_mask)
            for r, c in zip(rows, cols):
                wx, wy = node.grid_to_world(c, r)
                assert -2.405 <= wx <= 2.405
                assert -2.405 <= wy <= 2.405

    def test_direct_warp_with_out_of_arena_coordinates(self, orthomapper_node):
        """Verify direct warp invocation with out-of-arena coordinates returns 0."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 100, dtype=np.uint8)

        # Case 1: far positive
        cells_pos = node.warp_image_to_grid(test_img, 10.0, 10.0, 0.0, 0.0)
        assert cells_pos == 0

        # Case 2: far negative
        cells_neg = node.warp_image_to_grid(test_img, -10.0, -10.0, 0.0, 0.0)
        assert cells_neg == 0

        # Grid must remain completely unobserved
        assert np.sum(node.observed_mask) == 0

    def test_arena_boundary_integrity_high_angular_velocities(self, orthomapper_node):
        """Stress-test arena boundary containment under high angular velocities."""
        node = orthomapper_node
        node.enable_auto_keyframes = False
        test_img_msg = create_test_image(rgb=(160, 180, 210))

        positions = [
            (2.405, 0.0, 'East perimeter boundary'),
            (-2.405, 0.0, 'West perimeter boundary'),
            (0.0, 2.405, 'North perimeter boundary'),
            (0.0, -2.405, 'South perimeter boundary'),
            (2.405, 2.405, 'NE corner'),
            (-2.405, 2.405, 'NW corner'),
            (-2.405, -2.405, 'SW corner'),
            (2.405, -2.405, 'SE corner'),
            (2.38, 2.38, 'Near NE corner (interior 2.5cm)'),
            (2.50, 0.0, 'Out-of-bounds East (clamped)'),
            (2.50, 2.50, 'Out-of-bounds NE corner (clamped)'),
        ]

        omega_rates = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, -10.0, -50.0, -100.0]

        for px, py, label in positions:
            for omega in omega_rates:
                node.clear()
                dt = 0.02
                yaw = 0.0
                for step in range(20):
                    t = 1.0 + step * dt
                    yaw += omega * dt
                    odom = create_odom_msg(px, py, yaw=yaw, stamp_sec=t)
                    odom.twist.twist.angular.z = float(omega)
                    node.khepera4_odom_callback(odom)
                    node.image_callback(test_img_msg)

                # Verify containment of all updated cells
                rows, cols = np.where(node.observed_mask)
                for r, c in zip(rows, cols):
                    assert 0 <= r < node.height, f'Row index {r} out of bounds in {label}'
                    assert 0 <= c < node.width, f'Col index {c} out of bounds in {label}'
                    wx, wy = node.grid_to_world(c, r)
                    assert -2.405 <= wx <= 2.405, (
                        f'Cell wx={wx:.4f} breached arena X bound in {label} at omega={omega}'
                    )
                    assert -2.405 <= wy <= 2.405, (
                        f'Cell wy={wy:.4f} breached arena Y bound in {label} at omega={omega}'
                    )

                # Verify state integrity
                assert not np.isnan(node.color_map).any()
                assert not np.isinf(node.color_map).any()
                assert np.all(node.color_map >= 0.0)
                assert np.all(node.color_map <= 255.0)

    def test_arena_boundary_sliding_perimeter_high_angular_velocity(self, orthomapper_node):
        """Stress-test concurrent translation along perimeter with high angular velocity."""
        node = orthomapper_node
        node.enable_auto_keyframes = False
        test_img_msg = create_test_image(rgb=(170, 200, 190))

        perimeter_laps = [
            (-2.405, -2.405, 2.405, -2.405, 25.0),
            (2.405, -2.405, 2.405, 2.405, -30.0),
            (2.405, 2.405, -2.405, 2.405, 40.0),
            (-2.405, 2.405, -2.405, -2.405, -50.0),
            (-2.50, -2.50, 2.50, -2.50, 60.0),
        ]

        node.clear()
        dt = 0.02
        t = 1.0
        yaw = 0.0

        for x0, y0, x1, y1, omega in perimeter_laps:
            xs = np.linspace(x0, x1, 25)
            ys = np.linspace(y0, y1, 25)
            for x, y in zip(xs, ys):
                t += dt
                yaw += omega * dt
                odom = create_odom_msg(x, y, yaw=yaw, stamp_sec=t)
                odom.twist.twist.linear.x = 0.15
                odom.twist.twist.angular.z = float(omega)
                node.khepera4_odom_callback(odom)
                node.image_callback(test_img_msg)

        rows, cols = np.where(node.observed_mask)
        assert len(rows) > 0, 'Perimeter sliding test must update visible cells'
        for r, c in zip(rows, cols):
            wx, wy = node.grid_to_world(c, r)
            assert -2.405 <= wx <= 2.405, f'Cell wx={wx:.4f} out of bounds during sliding spin'
            assert -2.405 <= wy <= 2.405, f'Cell wy={wy:.4f} out of bounds during sliding spin'

    def test_arena_boundary_extreme_yaw_accumulation(self, orthomapper_node):
        """Verify boundary containment under extreme accumulated yaw angles."""
        node = orthomapper_node
        test_img = np.full((480, 640, 3), 150, dtype=np.uint8)

        extreme_yaws = [
            100.0 * math.pi,
            -100.0 * math.pi,
            1000.0 * math.pi,
            -1000.0 * math.pi,
            100000.0,
            -100000.0,
        ]

        for yaw in extreme_yaws:
            node.clear()
            node.warp_image_to_grid(test_img, 2.405, 2.405, yaw, 0.0)
            rows, cols = np.where(node.observed_mask)
            for r, c in zip(rows, cols):
                wx, wy = node.grid_to_world(c, r)
                assert -2.405 <= wx <= 2.405, f'Cell wx={wx:.4f} out of bounds at yaw={yaw}'
                assert -2.405 <= wy <= 2.405, f'Cell wy={wy:.4f} out of bounds at yaw={yaw}'


# ==============================================================================
# Category 3: Odometry Fallback Behavior Stress Tests
# ==============================================================================

class TestOdometryFallbackBehavior:
    """Empirical challenge on dual odometry subscriber fallback (/khepera4/odom -> /odom)."""

    def test_initial_state_raw_odom_fallback(self, orthomapper_node):
        """Verify /odom accepted immediately when /khepera4/odom has never published."""
        node = orthomapper_node
        assert node.has_odom is False
        assert node.last_khepera4_odom_time is None

        # Deliver /odom message
        raw_odom = create_odom_msg(1.15, -0.75, yaw=math.radians(45.0))
        node.raw_odom_callback(raw_odom)

        assert node.has_odom is True, 'Node must accept /odom when /khepera4/odom is absent.'
        assert math.isclose(node.robot_x, 1.15, abs_tol=1e-3)
        assert math.isclose(node.robot_y, -0.75, abs_tol=1e-3)
        assert math.isclose(math.degrees(node.robot_yaw), 45.0, abs_tol=1e-2)
        assert node.last_khepera4_odom_time is None

    def test_khepera4_odom_priority_over_raw_odom(self, orthomapper_node):
        """Verify incoming /odom rejected while /khepera4/odom is active."""
        node = orthomapper_node

        # 1. Deliver /khepera4/odom message at (0.20, 0.30)
        k4_odom = create_odom_msg(0.20, 0.30, yaw=0.0)
        node.khepera4_odom_callback(k4_odom)

        assert node.last_khepera4_odom_time is not None
        assert math.isclose(node.robot_x, 0.20, abs_tol=1e-3)
        assert math.isclose(node.robot_y, 0.30, abs_tol=1e-3)

        # 2. Immediately deliver conflicting /odom message at (1.50, 1.50)
        raw_odom = create_odom_msg(1.50, 1.50, yaw=math.pi / 2.0)
        node.raw_odom_callback(raw_odom)

        # Pose must NOT change — /odom was rejected because /khepera4/odom is active
        assert math.isclose(node.robot_x, 0.20, abs_tol=1e-3), (
            '/odom must be ignored while /khepera4/odom is active'
        )
        assert math.isclose(node.robot_y, 0.30, abs_tol=1e-3)
        assert math.isclose(node.robot_yaw, 0.0, abs_tol=1e-3)

    def test_timeout_fallback_to_raw_odom(self, orthomapper_node):
        """Verify fallback to /odom triggers when /khepera4/odom silent > 1.0s."""
        node = orthomapper_node

        # 1. Deliver /khepera4/odom message
        k4_odom = create_odom_msg(0.50, 0.50)
        node.khepera4_odom_callback(k4_odom)
        assert math.isclose(node.robot_x, 0.50, abs_tol=1e-3)

        # 2. Simulate 1.1s elapsing since last /khepera4/odom
        now = node.get_clock().now()
        node.last_khepera4_odom_time = now - Duration(seconds=1, nanoseconds=100_000_000)

        # 3. Deliver /odom message with new position (0.80, -0.60)
        raw_odom = create_odom_msg(0.80, -0.60)
        node.raw_odom_callback(raw_odom)

        # /odom must be ACCEPTED due to timeout (> 1.0s)
        assert math.isclose(node.robot_x, 0.80, abs_tol=1e-3), (
            '/odom must be accepted after /khepera4/odom timeout > 1.0s'
        )
        assert math.isclose(node.robot_y, -0.60, abs_tol=1e-3)

    def test_instant_priority_recovery_when_khepera4_odom_resumes(self, orthomapper_node):
        """Verify new /khepera4/odom message instantly re-asserts priority after fallback."""
        node = orthomapper_node

        # 1. Trigger fallback state by aging last_khepera4_odom_time
        now = node.get_clock().now()
        node.last_khepera4_odom_time = now - Duration(seconds=2, nanoseconds=0)

        raw_odom1 = create_odom_msg(-0.50, -0.50)
        node.raw_odom_callback(raw_odom1)
        assert math.isclose(node.robot_x, -0.50, abs_tol=1e-3)

        # 2. /khepera4/odom resumes publishing at (-1.20, 1.20)
        k4_odom = create_odom_msg(-1.20, 1.20)
        node.khepera4_odom_callback(k4_odom)
        assert math.isclose(node.robot_x, -1.20, abs_tol=1e-3)
        assert math.isclose(node.robot_y, 1.20, abs_tol=1e-3)

        # 3. Next /odom arrives immediately (within 0.05s) at (0.0, 0.0)
        raw_odom2 = create_odom_msg(0.0, 0.0)
        node.raw_odom_callback(raw_odom2)

        # Must NOT revert to (0.0, 0.0)
        assert math.isclose(node.robot_x, -1.20, abs_tol=1e-3), (
            '/khepera4/odom must retain priority immediately after resuming'
        )
        assert math.isclose(node.robot_y, 1.20, abs_tol=1e-3)

    def test_intermittent_odometry_dropout_stress(self, orthomapper_node):
        """Stress test: 100 interleaved messages with periodic dropouts of /khepera4/odom."""
        node = orthomapper_node
        test_img_msg = create_test_image()

        for cycle in range(10):
            # Phase A: /khepera4/odom active (5 frames)
            for _ in range(5):
                k4 = create_odom_msg(0.1 * cycle, 0.1 * cycle)
                node.khepera4_odom_callback(k4)
                # Competing raw odom
                raw = create_odom_msg(99.0, 99.0)
                node.raw_odom_callback(raw)
                assert math.isclose(node.robot_x, 0.1 * cycle, abs_tol=1e-3)

            # Phase B: /khepera4/odom drops out (age clock by 1.5s)
            now = node.get_clock().now()
            node.last_khepera4_odom_time = now - Duration(seconds=1, nanoseconds=500_000_000)

            # Raw odom delivers update
            raw_fallback = create_odom_msg(-0.1 * cycle, -0.1 * cycle)
            node.raw_odom_callback(raw_fallback)
            assert math.isclose(node.robot_x, -0.1 * cycle, abs_tol=1e-3)

            # Process an image frame in fallback mode
            node.image_callback(test_img_msg)

        assert node.has_odom is True
        assert not np.isnan(node.color_map).any()
