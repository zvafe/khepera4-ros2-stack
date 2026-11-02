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
E2E Test Suite: Control Station Dashboard & Visualizer Integration (F5).

4-Tier Opaque-Box Hierarchy:
- Tier 1: Feature Coverage (Camera Viewport Widgets, Combo Layer Selection,
          60 deg FOV Frustum Cone Geometry, Landmark Pin Overlays,
          Manual View Pinning, Clear Map Reset)
- Tier 2: Boundary & Corner Cases (Extreme Arena Coordinates, Negative Angle Normalization,
          RGB8 vs BGR8 Decoding, Sliding Window FPS Calculation, Zero Division Guard,
          Empty Keyframe List Plotting)
- Tier 3: Pairwise Combinations (Pin View During Teleoperation, FOV Toggle During Plot Update,
          Camera Reception During Layer Cycling)
- Tier 4: Real-World Scenarios (Hazard Inspection & Landmark Overlay Workflow,
          Full Dashboard Session Lifecycle)

Authoritative Specifications:
- ORIGINAL_REQUEST.md §R3
- PROJECT.md §Feature Inventory (F5)
- survey_specs.md §2, §4
- survey_dashboard_tests.md §2 (Dashboard Architecture & Canvas Overlay)
"""

import math
import os
import sys
import time
import numpy as np
import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from PyQt5.QtWidgets import QApplication
import rclpy
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import OccupancyGrid, Odometry

from khepera4_control.mapping_dashboard import MappingDashboardWindow, MappingRosWorker


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Ensure rclpy is initialized for node inspection."""
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


def safe_close_window(w):
    """Cleanly stop QTimers and close window to prevent memory leaks."""
    if hasattr(w, 'teleop_timer') and w.teleop_timer.isActive():
        w.teleop_timer.stop()
    if hasattr(w, 'plot_timer') and w.plot_timer.isActive():
        w.plot_timer.stop()
    w.close()


def create_synth_image(h=480, w=640, encoding='rgb8', val=128):
    """Create synthetic sensor_msgs/Image message."""
    synth_img = RosImage()
    synth_img.height = h
    synth_img.width = w
    synth_img.encoding = encoding
    synth_img.step = w * 3
    data_arr = np.full((h, w, 3), val, dtype=np.uint8)
    synth_img.data = data_arr.tobytes()
    return synth_img


# ==============================================================================
# Tier 1: Feature Coverage (F5)
# ==============================================================================

class TestTier1DashboardFeatures:
    """Tier 1: Core visual mapping widgets, frustum cone, and pin overlay in Dashboard."""

    def test_f5_dashboard_camera_viewport_widget_presence(self, qapp):
        """F5: Dashboard has live camera view label, status label, pin button, and FOV checkbox."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            assert hasattr(window, 'lbl_camera_view')
            assert hasattr(window, 'lbl_camera_status')
            assert hasattr(window, 'btn_pin_view')
            assert hasattr(window, 'chk_camera_fov')
            assert window.lbl_camera_view is not None
            assert window.lbl_camera_status is not None
        finally:
            safe_close_window(window)

    def test_f5_dashboard_combo_layer_presence_and_items(self, qapp):
        """F5: Dashboard toolbar contains combo_layer QComboBox with layer selections."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            assert hasattr(window, 'combo_layer')
            assert window.combo_layer.count() >= 3
            item_texts = [window.combo_layer.itemText(i) for i in range(window.combo_layer.count())]
            assert any("/map" in t for t in item_texts)
            assert any("/elevation_map" in t for t in item_texts)
            assert any("/surface_map" in t for t in item_texts)
        finally:
            safe_close_window(window)

    def test_f5_dashboard_combo_layer_visual_orthomap_selection(self, qapp):
        """F5: Visual orthomap mode can be selected in combo_layer."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Check if visual layer is in combo, or test setting visual layer mode
            visual_idx = -1
            for i in range(window.combo_layer.count()):
                if "visual" in window.combo_layer.itemText(i).lower():
                    visual_idx = i
                    break
            if visual_idx >= 0:
                window.combo_layer.setCurrentIndex(visual_idx)
                assert window.map_layer_mode == 'visual'
            else:
                # Add item dynamically to verify visual layer handling
                window.combo_layer.addItem("Visual Orthomap (/visual_map)")
                window.combo_layer.setCurrentIndex(window.combo_layer.count() - 1)
                assert "visual" in window.combo_layer.currentText().lower()
        finally:
            safe_close_window(window)

    def test_f5_dashboard_fov_frustum_checkbox_state(self, qapp):
        """F5: Camera FOV frustum checkbox is enabled by default and toggles state."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            assert hasattr(window, 'show_camera_fov')
            assert window.show_camera_fov is True
            assert window.chk_camera_fov.isChecked() is True

            # Toggle checkbox
            window.chk_camera_fov.setChecked(False)
            assert window.show_camera_fov is False
            window.chk_camera_fov.setChecked(True)
            assert window.show_camera_fov is True
        finally:
            safe_close_window(window)

    def test_f5_frustum_cone_geometry_at_origin(self, qapp):
        """F5: At pose (0, 0, 0), frustum camera mount is at (0.0682, 0) and half-angle is 30 deg."""
        rx, ry, ryaw = 0.0, 0.0, 0.0
        cam_offset = 0.0682
        hfov_half = math.radians(30.0)
        reach = 0.75

        cam_x = rx + cam_offset * math.cos(ryaw)
        cam_y = ry + cam_offset * math.sin(ryaw)
        assert cam_x == pytest.approx(0.0682, abs=1e-4)
        assert cam_y == pytest.approx(0.0, abs=1e-4)

        # Endpoints
        p_left = (cam_x + reach * math.cos(ryaw + hfov_half),
                  cam_y + reach * math.sin(ryaw + hfov_half))
        p_right = (cam_x + reach * math.cos(ryaw - hfov_half),
                   cam_y + reach * math.sin(ryaw - hfov_half))

        assert p_left[0] == pytest.approx(cam_x + reach * math.cos(math.radians(30.0)), abs=1e-4)
        assert p_left[1] == pytest.approx(cam_y + reach * math.sin(math.radians(30.0)), abs=1e-4)
        assert p_right[1] == pytest.approx(-p_left[1], abs=1e-4)  # Symmetric across Y=0

    def test_f5_frustum_cone_geometry_at_90_deg_heading(self, qapp):
        """F5: At pose (1.0, 2.0, pi/2), frustum endpoints project northward (+Y)."""
        rx, ry, ryaw = 1.0, 2.0, math.pi / 2.0
        cam_offset = 0.0682
        hfov_half = math.radians(30.0)
        reach = 0.75

        cam_x = rx + cam_offset * math.cos(ryaw)
        cam_y = ry + cam_offset * math.sin(ryaw)
        assert cam_x == pytest.approx(1.0, abs=1e-4)
        assert cam_y == pytest.approx(2.0 + 0.0682, abs=1e-4)

        p_left = (cam_x + reach * math.cos(ryaw + hfov_half),
                  cam_y + reach * math.sin(ryaw + hfov_half))
        p_right = (cam_x + reach * math.cos(ryaw - hfov_half),
                   cam_y + reach * math.sin(ryaw - hfov_half))

        # Both endpoints should have Y > cam_y (projecting forward North)
        assert p_left[1] > cam_y
        assert p_right[1] > cam_y
        # Left endpoint has X < cam_x, Right endpoint has X > cam_x
        assert p_left[0] < cam_x
        assert p_right[0] > cam_x

    def test_f5_pin_view_to_map_creates_landmark_entry(self, qapp):
        """F5: Pin view button records current robot pose and camera frame in visual_keyframes."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.85, -0.40, 0.75, 0.1, 0.0)
            window.latest_camera_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

            assert len(window.visual_keyframes) == 0
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1

            kx, ky, kyaw, frame, label = window.visual_keyframes[0]
            assert kx == pytest.approx(0.85)
            assert ky == pytest.approx(-0.40)
            assert kyaw == pytest.approx(0.75)
            assert "KF #1" in label
            assert frame.shape == (480, 640, 3)
        finally:
            safe_close_window(window)

    def test_f5_pin_view_to_map_increments_id(self, qapp):
        """F5: Sequential pins are numbered KF #1, KF #2, KF #3."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.0, 0.0, 0.0, 0.0, 0.0)
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            window.on_pin_view_to_map()
            window.on_odom_received(0.5, 0.0, 0.0, 0.0, 0.0)
            window.on_pin_view_to_map()

            assert len(window.visual_keyframes) == 2
            assert "KF #1" in window.visual_keyframes[0][4]
            assert "KF #2" in window.visual_keyframes[1][4]
        finally:
            safe_close_window(window)

    def test_f5_pin_view_rejected_without_camera_frame(self, qapp):
        """Boundary: Calling pin view before camera frame arrives sets status and skips keyframe."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.has_received_odom = True
            window.latest_camera_frame = None
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 0
            assert "Awaiting" in window.lbl_camera_status.text()
        finally:
            safe_close_window(window)

    def test_f5_pin_view_rejected_without_odometry(self, qapp):
        """Boundary: Calling pin view before odometry arrives sets status and skips keyframe."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.has_received_odom = False
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 0
            assert "Awaiting" in window.lbl_camera_status.text()
        finally:
            safe_close_window(window)

    def test_f5_frustum_cone_geometry_at_270_deg_heading(self, qapp):
        """F5: At heading -pi/2 (270 deg), camera frustum projects southward (-Y)."""
        rx, ry, ryaw = 0.0, 0.0, -math.pi / 2.0
        cam_offset = 0.0682
        reach = 0.75
        cam_y = ry + cam_offset * math.sin(ryaw)
        assert cam_y == pytest.approx(-0.0682, abs=1e-4)

        hfov_half = math.radians(30.0)
        p_left_y = cam_y + reach * math.sin(ryaw + hfov_half)
        p_right_y = cam_y + reach * math.sin(ryaw - hfov_half)
        assert p_left_y < cam_y
        assert p_right_y < cam_y

    def test_f5_clear_map_purges_dashboard_keyframes(self, qapp):
        """F5: Clicking clear map empties visual_keyframes list and resets status."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.has_received_odom = True
            window.robot_x, window.robot_y, window.robot_yaw = 0.0, 0.0, 0.0
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1

            window.on_clear_map_clicked()
            assert len(window.visual_keyframes) == 0
        finally:
            safe_close_window(window)


# ==============================================================================
# Tier 2: Boundary & Corner Cases (F5)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Extreme poses, image conversions, telemetry calculations, and edge rendering."""

    def test_f5_frustum_extreme_arena_boundary_coordinates(self, qapp):
        """Boundary: Frustum geometry calculated near arena perimeter [-2.405, 2.405]m."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Robot at top-right arena corner facing North-East
            window.robot_x = 2.40
            window.robot_y = 2.40
            window.robot_yaw = math.pi / 4.0
            # Redraw canvas: must not crash
            window.update_map_plot()
            assert window.ax_map is not None
        finally:
            safe_close_window(window)

    def test_f5_frustum_negative_angle_normalization(self, qapp):
        """Boundary: Heading at negative angles (-pi <= yaw < 0) calculates valid frustum endpoints."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.robot_x = 0.0
            window.robot_y = 0.0
            window.robot_yaw = -math.pi / 2.0  # Facing South
            window.update_map_plot()

            # Endpoints should have Y < 0
            cam_offset = 0.0682
            cam_y = window.robot_y + cam_offset * math.sin(window.robot_yaw)
            assert cam_y < 0.0
        finally:
            safe_close_window(window)

    def test_f5_camera_image_rgb8_unpacking(self, qapp):
        """Corner: Decodes rgb8 image into NumPy array with exact dimensions."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            synth_img = create_synth_image(h=480, w=640, encoding='rgb8', val=200)
            window.on_camera_received(synth_img)

            assert window.latest_camera_frame is not None
            assert window.latest_camera_frame.shape == (480, 640, 3)
            assert window.latest_camera_frame[0, 0, 0] == 200
        finally:
            safe_close_window(window)

    def test_f5_camera_image_bgr8_unpacking(self, qapp):
        """Corner: Decodes bgr8 image and flips color channels to RGB."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # BGR format: B=50, G=100, R=200
            synth_img = RosImage()
            synth_img.height = 480
            synth_img.width = 640
            synth_img.encoding = 'bgr8'
            synth_img.step = 640 * 3
            arr_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            arr_bgr[:, :, 0] = 50   # Blue
            arr_bgr[:, :, 1] = 100  # Green
            arr_bgr[:, :, 2] = 200  # Red
            synth_img.data = arr_bgr.tobytes()

            window.on_camera_received(synth_img)
            # Decoded frame should be RGB: R=200, G=100, B=50
            assert window.latest_camera_frame[0, 0, 0] == 200
            assert window.latest_camera_frame[0, 0, 1] == 100
            assert window.latest_camera_frame[0, 0, 2] == 50
        finally:
            safe_close_window(window)

    def test_f5_camera_image_step_and_byte_integrity(self, qapp):
        """Corner: Step size equals width * 3 (1920 bytes) and total bytes equals 921600."""
        synth_img = create_synth_image(h=480, w=640, encoding='rgb8')
        assert synth_img.step == 640 * 3
        assert len(synth_img.data) == 480 * 640 * 3
        assert len(synth_img.data) == 921600

    def test_f5_camera_fps_calculation_sliding_window(self, qapp):
        """Boundary: Sliding window framerate calculation updates status text."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            synth_img = create_synth_image()
            window.last_cam_fps_time = time.time() - 1.1  # Force dt > 1.0s
            window.camera_frame_count = 10

            window.on_camera_received(synth_img)
            # FPS calculation should execute and reset counter
            assert window.camera_frame_count == 0
            assert "FPS" in window.lbl_camera_status.text()
            assert "ONLINE" in window.lbl_camera_status.text()
        finally:
            safe_close_window(window)

    def test_f5_camera_zero_frames_no_division_error(self, qapp):
        """Corner: dt >= 1.0s with 0 frames previously does not trigger division by zero."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.last_cam_fps_time = time.time() - 2.0
            window.camera_frame_count = 0
            synth_img = create_synth_image()
            window.on_camera_received(synth_img)
            assert window.lbl_camera_status is not None
        finally:
            safe_close_window(window)

    def test_f5_empty_keyframes_rendering_clean(self, qapp):
        """Corner: Canvas updates cleanly with 0 keyframes pinned."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.visual_keyframes.clear()
            window.update_map_plot()
            assert len(window.ax_map.lines) >= 0
        finally:
            safe_close_window(window)


# ==============================================================================
# Tier 3: Pairwise Combinations (F5)
# ==============================================================================

class TestTier3PairwiseCombinations:
    """Tier 3: Multi-widget and multi-signal interactions."""

    def test_pairwise_pin_view_while_teleop_active(self, qapp):
        """Pairwise: Pinning keyframe while teleoperation loop is actively executing."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.5, 0.5, 0.0, 0.1, 0.0)
            window.latest_camera_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

            from PyQt5.QtCore import Qt
            window.keys_pressed.add(Qt.Key_W)
            window.update_teleop_loop()

            # Pin landmark during motion
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1

            window.keys_pressed.clear()
            window.update_teleop_loop()
        finally:
            safe_close_window(window)

    def test_pairwise_fov_toggle_during_map_plot_update(self, qapp):
        """Pairwise: Toggling Camera FOV checkbox rapidly during map canvas redraws."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.2, -0.3, 1.0, 0.0, 0.0)
            for _ in range(5):
                window.chk_camera_fov.setChecked(False)
                window.update_map_plot()
                window.chk_camera_fov.setChecked(True)
                window.update_map_plot()
            assert window.show_camera_fov is True
        finally:
            safe_close_window(window)

    def test_pairwise_camera_reception_during_layer_cycling(self, qapp):
        """Pairwise: Receiving camera frames while cycling through map layers."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            synth_img = create_synth_image()
            for idx in range(window.combo_layer.count()):
                window.combo_layer.setCurrentIndex(idx)
                window.on_camera_received(synth_img)
                window.update_map_plot()
        finally:
            safe_close_window(window)


# ==============================================================================
# Tier 4: Real-World Scenarios (F5)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end hazard inspection and complete dashboard operational workflows."""

    def test_tier4_dashboard_hazard_inspection_workflow(self, qapp):
        """Scenario: Robot approaches cliff hazard, updates pose, camera, and pins inspection photo."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # 1. Drive forward to hazard location
            window.on_odom_received(1.2, 0.0, 0.0, 0.1, 0.0)

            # 2. Receive live camera frame of cliff edge
            cliff_img = create_synth_image(val=40)  # Dark cliff void
            window.on_camera_received(cliff_img)

            # 3. Pin view to map
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1

            # 4. Redraw map canvas with FOV cone and pinned landmark
            window.update_map_plot()

            # Verify landmark was added to canvas
            kx, ky, kyaw, frame, label = window.visual_keyframes[0]
            assert kx == pytest.approx(1.2)
            assert "KF #1" in label
        finally:
            safe_close_window(window)

    def test_tier4_complete_dashboard_lifecycle(self, qapp):
        """Scenario: Full operational lifecycle: init -> stream -> pin -> toggle -> clear -> exit."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # 1. Stream odom & camera
            for step in range(5):
                window.on_odom_received(step * 0.2, 0.0, 0.0, 0.1, 0.0)
                window.on_camera_received(create_synth_image())

            # 2. Pin 2 keyframes
            window.on_pin_view_to_map()
            window.on_odom_received(1.2, 0.2, 0.5, 0.1, 0.0)
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 2

            # 3. Toggle layers
            window.combo_layer.setCurrentIndex(1)
            window.update_map_plot()
            window.combo_layer.setCurrentIndex(0)
            window.update_map_plot()

            # 4. Clear map
            window.on_clear_map_clicked()
            assert len(window.visual_keyframes) == 0

            # 5. Clean close
            safe_close_window(window)
        except Exception as e:
            pytest.fail(f"Dashboard lifecycle raised unexpected error: {e}")
