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
Adversarial Stress Test Suite — Challenger 1 (Milestone M3 Gate Verification).

Verifies Milestone M3 Feature F5 (Control Station Dashboard & Visualizer Integration):
1. Layer cycling: Rapid stress cycling across all 4 modes under continuous simulated telemetry.
2. FOV cone geometry: Exact 60 deg cone calculation, camera offset, and NaN/Inf/extreme pose resilience.
3. Missing & degenerate grid handling: Empty, malformed, non-square, and out-of-range visual grids.
4. Stress keyframing & clear map: High keyframe counts, artist tracking, and atomic memory reset.
5. Telemetry & signal concurrency: Multi-rate simulated ROS callbacks during user interaction.
"""

import math
import os
import sys
import time
from typing import List
import numpy as np
import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from PyQt5.QtWidgets import QApplication
import rclpy
from sensor_msgs.msg import Image as RosImage, LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion

from khepera4_control.mapping_dashboard import MappingDashboardWindow, MappingRosWorker


@pytest.fixture(scope='module', autouse=True)
def manage_rclpy():
    """Ensure rclpy is initialized for testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture(scope='module')
def qapp():
    """Ensure QApplication instance is active for Qt tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def safe_close_window(w: MappingDashboardWindow):
    """Clean and stop QTimers and close window to prevent memory leaks."""
    if hasattr(w, 'teleop_timer') and w.teleop_timer.isActive():
        w.teleop_timer.stop()
    if hasattr(w, 'plot_timer') and w.plot_timer.isActive():
        w.plot_timer.stop()
    w.close()


def create_synthetic_occupancy_grid(width: int = 200, height: int = 200,
                                    res: float = 0.03, ox: float = -3.0, oy: float = -3.0,
                                    fill_val: int = 50) -> OccupancyGrid:
    """Create a synthetic OccupancyGrid message."""
    grid = OccupancyGrid()
    grid.info.resolution = res
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = ox
    grid.info.origin.position.y = oy
    grid.info.origin.position.z = 0.0
    grid.info.origin.orientation.w = 1.0

    total_cells = width * height
    grid.data = [fill_val] * total_cells
    return grid


def create_synthetic_laser_scan(num_rays: int = 40, dist: float = 0.5) -> LaserScan:
    """Create a synthetic LaserScan message."""
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = (2.0 * math.pi) / num_rays
    scan.range_min = 0.02
    scan.range_max = 2.0
    scan.ranges = [dist] * num_rays
    return scan


# ==============================================================================
# Challenge 1: Layer Cycling Under Continuous Telemetry
# ==============================================================================

class TestAdversarialLayerCycling:
    """Stress tests rapid layer mode transitions and concurrent telemetry."""

    def test_rapid_layer_cycling_across_all_four_layers(self, qapp):
        """Stress: 100 rapid cycles across 4 layers under continuous simulated telemetry."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Setup initial state
            window.on_odom_received(0.5, 0.5, 0.2, 0.1, 0.0)
            window.latest_map = create_synthetic_occupancy_grid(fill_val=0)
            window.latest_elevation = create_synthetic_occupancy_grid(fill_val=10)
            window.latest_surface = create_synthetic_occupancy_grid(fill_val=20)
            window.latest_visual_map = create_synthetic_occupancy_grid(fill_val=80)
            window.latest_scan = create_synthetic_laser_scan()

            modes_expected = ['obstacles', 'elevation', 'surface', 'visual']

            for cycle in range(100):
                mode_idx = cycle % 4
                window.combo_layer.setCurrentIndex(mode_idx)
                assert window.map_layer_mode == modes_expected[mode_idx]

                # Simulate interleaved live telemetry
                t = cycle * 0.05
                window.on_odom_received(0.5 + 0.1 * math.cos(t), 0.5 + 0.1 * math.sin(t), t, 0.1, 0.0)
                window.update_map_plot()

                # Verify canvas title reflects the active layer
                title = window.ax_map.get_title()
                assert len(title) > 0
                if mode_idx == 3:
                    assert "Visual Orthomap" in title
                elif mode_idx == 0:
                    assert "Obstacle" in title
        finally:
            safe_close_window(window)

    def test_layer_cycling_with_missing_layers(self, qapp):
        """Stress: Rapid cycling when some or all map layers are None."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.0, 0.0, 0.0, 0.0, 0.0)
            window.latest_map = None
            window.latest_elevation = None
            window.latest_surface = None
            window.latest_visual_map = None

            for cycle in range(20):
                idx = cycle % 4
                window.combo_layer.setCurrentIndex(idx)
                window.update_map_plot()
                # Canvas must survive and display awaiting map fallback
                assert window.ax_map is not None
        finally:
            safe_close_window(window)

    def test_out_of_bounds_layer_index_handling(self, qapp):
        """Stress: on_layer_changed with invalid index values."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Valid modes
            window.on_layer_changed(0)
            assert window.map_layer_mode == 'obstacles'
            window.on_layer_changed(3)
            assert window.map_layer_mode == 'visual'

            # Out of bounds high
            window.on_layer_changed(4)
            assert window.map_layer_mode == 'obstacles'
            window.on_layer_changed(99)
            assert window.map_layer_mode == 'obstacles'

            # Update map plot executes safely
            window.update_map_plot()
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 2: FOV Frustum Cone Mathematical Exactness & Singularities
# ==============================================================================

class TestAdversarialCameraFOVFrustum:
    """Stress tests 60 deg camera FOV frustum geometry, extreme poses and NaN values."""

    def test_fov_cone_mathematical_precision_full_sweep(self, qapp):
        """Verification: Full 360 degree yaw sweep validates apex, ray reach, and aperture."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.show_camera_fov = True
            window.has_received_odom = True

            cam_offset = 0.0682
            reach = 0.75
            expected_aperture_rad = 2.0 * 0.5236  # ~60.0 degrees = 1.0472 rad

            for deg in range(0, 360, 15):
                yaw = math.radians(deg)
                window.robot_x = 1.0
                window.robot_y = -1.0
                window.robot_yaw = yaw

                window.update_map_plot()
                assert window.fov_patch is not None

                verts = window.fov_patch.get_xy()
                # Polygon vertices: (cam_x, cam_y), p_left, p_right, (cam_x, cam_y)
                apex = verts[0]
                p_left = verts[1]
                p_right = verts[2]

                expected_cam_x = 1.0 + cam_offset * math.cos(yaw)
                expected_cam_y = -1.0 + cam_offset * math.sin(yaw)

                assert apex[0] == pytest.approx(expected_cam_x, abs=1e-4)
                assert apex[1] == pytest.approx(expected_cam_y, abs=1e-4)

                # Vector apex -> p_left
                v_left = (p_left[0] - apex[0], p_left[1] - apex[1])
                len_left = math.hypot(v_left[0], v_left[1])
                assert len_left == pytest.approx(reach, abs=1e-4)

                # Vector apex -> p_right
                v_right = (p_right[0] - apex[0], p_right[1] - apex[1])
                len_right = math.hypot(v_right[0], v_right[1])
                assert len_right == pytest.approx(reach, abs=1e-4)

                # Included angle via dot product
                dot = (v_left[0] * v_right[0] + v_left[1] * v_right[1]) / (len_left * len_right)
                dot_clamped = max(-1.0, min(1.0, dot))
                angle_between = math.acos(dot_clamped)
                assert angle_between == pytest.approx(expected_aperture_rad, abs=1e-4)
        finally:
            safe_close_window(window)

    def test_fov_cone_with_nan_inf_and_extreme_coordinates(self, qapp):
        """Stress: Frustum computation with NaN/Inf heading and extreme poses."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.show_camera_fov = True
            window.has_received_odom = True

            # Case A: NaN yaw
            window.robot_x = 0.0
            window.robot_y = 0.0
            window.robot_yaw = float('nan')
            window.update_map_plot()
            # math.cos(nan) in Python yields nan (no exception), producing a Polygon with NaN vertices
            assert window.fov_patch is not None
            assert math.isnan(window.fov_patch.get_xy()[0][0])

            # Case B: Inf yaw — Unhandled in heading arrow calculation (line 2046 raises ValueError)
            window.robot_yaw = float('inf')
            with pytest.raises(ValueError, match="math domain error"):
                window.update_map_plot()

            # Case C: Extreme finite yaw (1,000,000 * 2*pi)
            window.robot_yaw = 2.0 * math.pi * 1000000.0
            window.update_map_plot()
            assert window.fov_patch is not None
            assert math.isfinite(window.fov_patch.get_xy()[0][0])

            # Case D: Extreme valid coordinates (10^6 meters)
            window.robot_x = 1e6
            window.robot_y = -1e6
            window.robot_yaw = math.pi / 3.0
            window.update_map_plot()
            assert window.fov_patch is not None
            apex = window.fov_patch.get_xy()[0]
            assert math.isfinite(apex[0])
            assert math.isfinite(apex[1])
        finally:
            safe_close_window(window)

    def test_fov_cone_suppressed_when_disabled_or_no_odom(self, qapp):
        """State: FOV frustum is suppressed when show_camera_fov=False or has_received_odom=False."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.robot_x, window.robot_y, window.robot_yaw = 0.0, 0.0, 0.0
            window.has_received_odom = False
            window.show_camera_fov = True
            window.update_map_plot()
            assert window.fov_patch is None

            window.has_received_odom = True
            window.show_camera_fov = False
            window.update_map_plot()
            assert window.fov_patch is None
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 3: Missing, Degenerate & Corrupted Visual Grids
# ==============================================================================

class TestAdversarialVisualGridHandling:
    """Stress tests Matplotlib rendering with malformed or boundary OccupancyGrid data."""

    def test_visual_layer_rendering_with_valid_grid(self, qapp):
        """Baseline: Mode 'visual' correctly parses OccupancyGrid and sets imshow extent."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.combo_layer.setCurrentIndex(3)
            grid = create_synthetic_occupancy_grid(width=100, height=80, res=0.05, ox=-2.5, oy=-2.0, fill_val=60)
            window.on_visual_map_received(grid)
            window.update_map_plot()

            # Verify images added to ax_map
            assert len(window.ax_map.images) > 0
            im = window.ax_map.images[-1]
            extent = im.get_extent()
            assert extent[0] == pytest.approx(-2.5)
            assert extent[1] == pytest.approx(-2.5 + 100 * 0.05)
            assert extent[2] == pytest.approx(-2.0)
            assert extent[3] == pytest.approx(-2.0 + 80 * 0.05)
        finally:
            safe_close_window(window)

    def test_visual_grid_with_size_mismatch_does_not_crash(self, qapp):
        """Stress: OccupancyGrid where data length does not match width * height."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.combo_layer.setCurrentIndex(3)
            grid = OccupancyGrid()
            grid.info.width = 200
            grid.info.height = 200
            grid.info.resolution = 0.03
            # Data only has 10 elements instead of 40000
            grid.data = [50] * 10
            window.on_visual_map_received(grid)

            # Redraw must catch reshape error and continue cleanly
            window.update_map_plot()
            assert window.ax_map is not None
        finally:
            safe_close_window(window)

    def test_visual_grid_with_zero_dimension_does_not_crash(self, qapp):
        """Stress: OccupancyGrid with width=0 or height=0."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.combo_layer.setCurrentIndex(3)
            grid = OccupancyGrid()
            grid.info.width = 0
            grid.info.height = 0
            grid.info.resolution = 0.03
            grid.data = []
            window.on_visual_map_received(grid)

            window.update_map_plot()
            assert window.ax_map is not None
        finally:
            safe_close_window(window)

    def test_visual_grid_with_all_unobserved_cells(self, qapp):
        """Corner: OccupancyGrid where all cells are -1 (unobserved floor)."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.combo_layer.setCurrentIndex(3)
            grid = create_synthetic_occupancy_grid(width=50, height=50, fill_val=-1)
            window.on_visual_map_received(grid)
            window.update_map_plot()

            assert len(window.ax_map.images) > 0
            img_data = window.ax_map.images[-1].get_array()
            # Alpha channel (index 3) should be 0.0 everywhere
            assert np.all(img_data[:, :, 3] == 0.0)
        finally:
            safe_close_window(window)

    def test_visual_grid_with_extreme_reflectance_values(self, qapp):
        """Corner: OccupancyGrid containing values > 100 or < -1."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.combo_layer.setCurrentIndex(3)
            grid = OccupancyGrid()
            grid.info.width = 10
            grid.info.height = 10
            grid.info.resolution = 0.1
            grid.info.origin.position.x = 0.0
            grid.info.origin.position.y = 0.0
            # Mix of negative, ceiling (127 > 100), and standard values in valid int8 range [-128, 127]
            pattern = [-128, -50, -1, 0, 50, 80, 100, 120, 127, 25]
            grid.data = pattern * 10
            window.on_visual_map_received(grid)

            window.update_map_plot()
            assert len(window.ax_map.images) > 0
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 4: High Keyframe Count, Artist Tracking & Clear Map Reset
# ==============================================================================

class TestAdversarialKeyframesAndClearMap:
    """Stress tests high keyframe volumes, artist references, and atomic clearing."""

    def test_high_volume_keyframe_pins_rendering(self, qapp):
        """Stress: Panning and rendering 150 pinned keyframes without canvas failure."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.has_received_odom = True
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            for i in range(150):
                angle = i * 0.1
                kx = 2.0 * math.cos(angle)
                ky = 2.0 * math.sin(angle)
                window.visual_keyframes.append((kx, ky, angle, dummy_frame, f"KF #{i+1}"))

            window.update_map_plot()

            # Verify all 150 keyframe artists were tracked
            assert len(window.keyframe_artists) == 150
            for line_art, text_art in window.keyframe_artists:
                assert line_art in window.ax_map.lines
                assert text_art in window.ax_map.texts
        finally:
            safe_close_window(window)

    def test_atomic_clear_map_purges_all_visual_state(self, qapp):
        """Stress: on_clear_map_clicked completely clears visual map, keyframes, and artists."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Populate state
            window.has_received_odom = True
            window.latest_visual_map = create_synthetic_occupancy_grid()
            window.visual_keyframes = [(0.0, 0.0, 0.0, np.zeros((10, 10, 3)), "KF #1")]
            window.update_map_plot()
            assert len(window.keyframe_artists) == 1

            # Trigger clear
            window.on_clear_map_clicked()

            # Assert complete wipe
            assert window.latest_visual_map is None
            assert len(window.visual_keyframes) == 0
            assert len(window.keyframe_artists) == 0

            # Redraw must reflect empty state
            window.combo_layer.setCurrentIndex(3)
            window.update_map_plot()
            assert window.latest_visual_map is None
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 5: ROS Worker Interface & Signal Contracts
# ==============================================================================

class TestAdversarialRosWorkerContracts:
    """Verifies MappingRosWorker signal contracts and subscription definitions."""

    def test_mapping_ros_worker_visual_signal_and_service_clients(self):
        """Contract: MappingRosWorker provides visual_map_received and clear_visual_map_cli."""
        worker = MappingRosWorker()
        assert hasattr(worker, 'visual_map_received')
        assert hasattr(worker, 'clear_visual_map_cli')
        assert hasattr(worker, 'pin_keyframe_cli')
        assert worker.clear_visual_map_cli is None
        assert worker.pin_keyframe_cli is None

    def test_visual_map_received_signal_slot_connection(self, qapp):
        """Contract: Emitting visual_map_received updates latest_visual_map in window."""
        worker = MappingRosWorker()
        window = MappingDashboardWindow(ros_worker=worker)
        try:
            assert window.latest_visual_map is None
            test_grid = create_synthetic_occupancy_grid(fill_val=77)

            worker.visual_map_received.emit(test_grid)
            assert window.latest_visual_map is not None
            assert window.latest_visual_map.data[0] == 77
        finally:
            safe_close_window(window)
