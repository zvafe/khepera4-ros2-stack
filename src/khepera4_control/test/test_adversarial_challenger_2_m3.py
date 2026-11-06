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
Adversarial Stress Test Suite — Challenger 2 (Milestone M3 Gate Verification).

Verifies Milestone M3 Feature F5 (Control Station Dashboard & Visualizer Integration):
1. Stress test camera feed decoding under corrupt, truncated, empty, and out-of-spec buffers.
2. Stress test manual keyframe pinning, sequential numbering, independence, and extreme coordinates.
3. Stress test canvas clicks, interaction modes, and keyframe pin overlays under user targeting.
4. Stress test clear map resets, multi-service lifecycle triggers, and artist cleanup idempotency.
5. Stress test visual orthomosaic rendering with invalid bounds, negative cells, and mismatched grids.
"""

import math
import os
import sys
import time
from unittest.mock import MagicMock
import numpy as np
import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from PyQt5.QtWidgets import QApplication
import rclpy
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger

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
    """Stop QTimers and close window cleanly to prevent memory leaks."""
    if hasattr(w, 'teleop_timer') and w.teleop_timer.isActive():
        w.teleop_timer.stop()
    if hasattr(w, 'plot_timer') and w.plot_timer.isActive():
        w.plot_timer.stop()
    w.close()


def create_raw_ros_image(height=480, width=640, encoding='rgb8', data=None):
    """Create a sensor_msgs/Image with custom parameters and data buffer."""
    msg = RosImage()
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding in ('rgb8', 'bgr8'):
        msg.step = width * 3
    elif encoding == 'mono8':
        msg.step = width
    else:
        msg.step = width

    if data is not None:
        msg.data = data
    else:
        arr = np.full((height, width, 3), 120, dtype=np.uint8)
        msg.data = arr.tobytes()
    return msg


def create_synthetic_visual_grid(width=200, height=200, res=0.03, ox=-3.0, oy=-3.0,
                                 fill_val=50):
    """Create an OccupancyGrid message with configurable parameters."""
    grid = OccupancyGrid()
    grid.info.resolution = res
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = ox
    grid.info.origin.position.y = oy
    grid.info.origin.position.z = 0.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [fill_val] * (width * height)
    return grid


# ==============================================================================
# Challenge 1: Camera Feed Decoding Under Adversarial & Corrupt Buffers
# ==============================================================================

class TestCameraDecodingAdversarial:
    """Stress test camera stream decoding under empty, truncated, or corrupt image buffers."""

    def test_camera_decoding_empty_data_buffer(self, qapp):
        """Adversarial: Buffer with height/width set but data is 0-byte string."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            msg = create_raw_ros_image(height=480, width=640, encoding='rgb8', data=b'')
            # Must not crash, should be caught by exception handler
            window.on_camera_received(msg)
            assert window.latest_camera_frame is None
        finally:
            safe_close_window(window)

    def test_camera_decoding_truncated_partial_buffer(self, qapp):
        """Adversarial: Buffer truncated halfway through frame (e.g. dropped packet)."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            corrupt_bytes = b'\xaa' * 1024  # Far fewer than 480*640*3 = 921600 bytes
            msg = create_raw_ros_image(height=480, width=640, encoding='rgb8', data=corrupt_bytes)
            window.on_camera_received(msg)
            assert window.latest_camera_frame is None
        finally:
            safe_close_window(window)

    def test_camera_decoding_zero_and_negative_dimensions(self, qapp):
        """Adversarial: Buffer with height=0, width=0, or negative dimensions."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            msg_zero = create_raw_ros_image(height=0, width=0, encoding='rgb8', data=b'')
            window.on_camera_received(msg_zero)
            assert window.latest_camera_frame is None or window.latest_camera_frame.size == 0

            msg_neg = create_raw_ros_image(height=-10, width=640, encoding='rgb8', data=b'')
            window.on_camera_received(msg_neg)
            assert window.latest_camera_frame is None or window.latest_camera_frame.size == 0
        finally:
            safe_close_window(window)

    def test_camera_decoding_unsupported_encodings_ignored_safely(self, qapp):
        """Adversarial: Mono8, RGBA8, Bayer, or empty encodings must not corrupt state."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            for enc in ('mono8', 'rgba8', 'yuv422', 'bayer_rggb8', '', 'unknown_encoding'):
                mono_data = np.full((480, 640), 128, dtype=np.uint8).tobytes()
                msg = create_raw_ros_image(height=480, width=640, encoding=enc, data=mono_data)
                window.on_camera_received(msg)
                # latest_camera_frame must remain unset
                assert window.latest_camera_frame is None
        finally:
            safe_close_window(window)

    def test_camera_decoding_extreme_dimensions(self, qapp):
        """Adversarial: Minimal 1x1 image and large 1080p image decode and scale cleanly."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # 1x1 pixel image
            data_1x1 = np.array([[[255, 0, 128]]], dtype=np.uint8).tobytes()
            msg_1x1 = create_raw_ros_image(height=1, width=1, encoding='rgb8', data=data_1x1)
            window.on_camera_received(msg_1x1)
            assert window.latest_camera_frame is not None
            assert window.latest_camera_frame.shape == (1, 1, 3)

            # 1920x1080 Full HD image
            h_hd, w_hd = 1080, 1920
            arr_hd = np.full((h_hd, w_hd, 3), 75, dtype=np.uint8)
            msg_hd = create_raw_ros_image(height=h_hd, width=w_hd, encoding='rgb8', data=arr_hd.tobytes())
            window.on_camera_received(msg_hd)
            assert window.latest_camera_frame.shape == (1080, 1920, 3)
        finally:
            safe_close_window(window)

    def test_camera_burst_streaming_stability(self, qapp):
        """Stress: Rapid burst of 100 images with changing dimensions and encodings."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.last_cam_fps_time = time.time() - 2.0
            for i in range(100):
                encoding = 'rgb8' if (i % 2 == 0) else 'bgr8'
                h = 120 if (i % 5 == 0) else 480
                w = 160 if (i % 5 == 0) else 640
                arr = np.full((h, w, 3), i % 256, dtype=np.uint8)
                msg = create_raw_ros_image(height=h, width=w, encoding=encoding, data=arr.tobytes())
                window.on_camera_received(msg)

            assert window.latest_camera_frame is not None
            assert "ONLINE" in window.lbl_camera_status.text()
            assert "FPS" in window.lbl_camera_status.text()
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 2: Keyframe Pinning & Pin Clicks Under Stress
# ==============================================================================

class TestKeyframePinningAdversarial:
    """Stress test manual pinning, sequential numbering, deep copying, and canvas rendering."""

    def test_pin_view_rejection_matrix(self, qapp):
        """Adversarial: Test all combinations of missing prerequisites."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Case 1: No odom, no camera
            window.has_received_odom = False
            window.latest_camera_frame = None
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 0
            assert "Awaiting frame / odom" in window.lbl_camera_status.text()

            # Case 2: Odom received, but no camera frame
            window.has_received_odom = True
            window.latest_camera_frame = None
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 0
            assert "Awaiting frame / odom" in window.lbl_camera_status.text()

            # Case 3: Camera frame present, but no odom
            window.has_received_odom = False
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 0
            assert "Awaiting frame / odom" in window.lbl_camera_status.text()
        finally:
            safe_close_window(window)

    def test_pin_view_frame_data_isolation(self, qapp):
        """Integrity: Modifying latest_camera_frame after pinning must NOT mutate pinned keyframe."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.5, 0.5, 0.0, 0.0, 0.0)
            initial_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            window.latest_camera_frame = initial_frame

            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1
            assert window.visual_keyframes[0][3][0, 0, 0] == 100

            # Overwrite active frame buffer in-place
            window.latest_camera_frame[:] = 250
            # Pinned keyframe must retain original value 100
            assert window.visual_keyframes[0][3][0, 0, 0] == 100
        finally:
            safe_close_window(window)

    def test_high_volume_sequential_pinning_and_rendering(self, qapp):
        """Stress: Pin 50 keyframes in sequence and verify canvas artist tracking."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.latest_camera_frame = frame

            for i in range(50):
                x = -2.0 + (i * 0.08)
                y = math.sin(i * 0.2) * 1.5
                yaw = (i * 0.1) % (2.0 * math.pi)
                window.on_odom_received(x, y, yaw, 0.1, 0.0)
                window.on_pin_view_to_map()

            assert len(window.visual_keyframes) == 50
            assert "KF #50" in window.visual_keyframes[-1][4]

            # Redraw plot with all 50 keyframes
            window.update_map_plot()
            assert len(window.keyframe_artists) == 50
            for line_art, txt_art in window.keyframe_artists:
                assert line_art is not None
                assert txt_art is not None
        finally:
            safe_close_window(window)

    def test_pinning_at_extreme_arena_coordinates(self, qapp):
        """Boundary: Pinning keyframes at outer perimeter (+/-2.4m) and extreme coordinates."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.has_received_odom = True

            extremes = [
                (2.40, 2.40, math.pi / 4),
                (-2.40, -2.40, -3 * math.pi / 4),
                (100.0, -100.0, 0.0),  # Way outside arena
            ]
            for x, y, yaw in extremes:
                window.robot_x, window.robot_y, window.robot_yaw = x, y, yaw
                window.on_pin_view_to_map()

            assert len(window.visual_keyframes) == 3
            window.update_map_plot()
            assert len(window.keyframe_artists) == 3
        finally:
            safe_close_window(window)

    def test_canvas_clicks_and_interactions_over_keyframe_pins(self, qapp):
        """Interaction: Mouse clicks on canvas near pins during targeting or path drawing."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            window.on_odom_received(1.0, 1.0, 0.0, 0.0, 0.0)
            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1

            # Mock a Matplotlib mouse click event on top of the keyframe pin (1.0, 1.0)
            mock_event = MagicMock()
            mock_event.inaxes = window.ax_map
            mock_event.xdata = 1.0
            mock_event.ydata = 1.0

            # Mode: Click Goal
            window.map_interaction_mode = 'click_goal'
            window.on_canvas_press(mock_event)
            assert window.active_goal == (1.0, 1.0)
            # Keyframe list must remain intact
            assert len(window.visual_keyframes) == 1

            # Mode: Draw Path
            window.map_interaction_mode = 'draw_path'
            window.on_canvas_press(mock_event)
            assert window.is_drawing_path is True
            assert window.drawn_path == [(1.0, 1.0)]

            mock_move = MagicMock()
            mock_move.inaxes = window.ax_map
            mock_move.xdata = 1.2
            mock_move.ydata = 1.2
            window.on_canvas_motion(mock_move)
            assert len(window.drawn_path) == 2

            window.on_canvas_release(mock_move)
            assert window.is_drawing_path is False
            assert len(window.visual_keyframes) == 1
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 3: Clear Map Reset & Lifecycle Robustness
# ==============================================================================

class TestClearMapResetAdversarial:
    """Stress test clear map lifecycle, service triggers, and memory reset."""

    def test_clear_map_comprehensive_state_purge(self, qapp):
        """Lifecycle: Verify clear map purges robot_path, keyframes, artists, visual map, and updates toast."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            # Populate comprehensive state
            window.robot_path = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
            window.visual_keyframes = [(0.0, 0.0, 0.0, np.zeros((10, 10, 3)), "KF #1")]
            window.latest_visual_map = create_synthetic_visual_grid()
            window.update_map_plot()
            assert len(window.keyframe_artists) == 1

            window.on_clear_map_clicked()

            assert len(window.robot_path) == 0
            assert len(window.visual_keyframes) == 0
            assert len(window.keyframe_artists) == 0
            assert window.latest_visual_map is None
            assert "Resetting and clearing" in window.lbl_map_toast.text()
        finally:
            safe_close_window(window)

    def test_clear_map_idempotency(self, qapp):
        """Adversarial: 10 repeated clear map invocations without crashing."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            for _ in range(10):
                window.on_clear_map_clicked()
                assert len(window.visual_keyframes) == 0
                assert window.latest_visual_map is None
        finally:
            safe_close_window(window)

    def test_clear_map_ros_service_dispatch(self, qapp):
        """Integration: Clear map triggers both clear_map_cli and clear_visual_map_cli."""
        mock_worker = MagicMock()
        mock_worker.clear_map_cli = MagicMock()
        mock_worker.clear_visual_map_cli = MagicMock()

        window = MappingDashboardWindow(ros_worker=mock_worker)
        try:
            window.on_clear_map_clicked()
            # Assert worker.trigger_service was called for both services
            calls = mock_worker.trigger_service.call_args_list
            assert len(calls) == 2
            cli_args = [c[0][0] for c in calls]
            assert mock_worker.clear_map_cli in cli_args
            assert mock_worker.clear_visual_map_cli in cli_args
        finally:
            safe_close_window(window)

    def test_pin_indexing_restarts_cleanly_after_clear(self, qapp):
        """Lifecycle: Pinning after clear_map starts from KF #1 again."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.on_odom_received(0.0, 0.0, 0.0, 0.0, 0.0)
            window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            window.on_pin_view_to_map()
            assert "KF #1" in window.visual_keyframes[0][4]

            window.on_clear_map_clicked()
            assert len(window.visual_keyframes) == 0

            window.on_pin_view_to_map()
            assert len(window.visual_keyframes) == 1
            assert "KF #1" in window.visual_keyframes[0][4]
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 4: Visual Orthomap Layer Rendering & Corrupt Grids
# ==============================================================================

class TestVisualOrthomapRenderingAdversarial:
    """Stress test visual orthomosaic rendering with corrupt or extreme grid messages."""

    def test_visual_orthomap_negative_unobserved_cells(self, qapp):
        """Math: Negative cells (<0) must be rendered transparent, observed (0..100) visible."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            grid = OccupancyGrid()
            grid.info.resolution = 0.03
            grid.info.width = 4
            grid.info.height = 4
            grid.info.origin.position.x = -0.06
            grid.info.origin.position.y = -0.06
            # Half unobserved (-1), half observed (0, 50, 100)
            grid.data = [-1, -1, 0, 50,
                         -1, -1, 50, 100,
                         -1, -1, 25, 75,
                         -1, -1, 10, 90]

            window.on_visual_map_received(grid)
            window.combo_layer.setCurrentIndex(3)
            assert window.map_layer_mode == 'visual'

            # Redraw must execute cleanly
            window.update_map_plot()
            assert window.latest_visual_map is not None
        finally:
            safe_close_window(window)

    def test_visual_orthomap_data_length_mismatch(self, qapp):
        """Adversarial: Grid with width=200, height=200 (40000 cells) but data only has 100 cells."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            grid = OccupancyGrid()
            grid.info.resolution = 0.03
            grid.info.width = 200
            grid.info.height = 200
            grid.data = [50] * 100  # Truncated data

            window.on_visual_map_received(grid)
            window.combo_layer.setCurrentIndex(3)
            # Must catch reshape exception gracefully without crashing GUI
            window.update_map_plot()
        finally:
            safe_close_window(window)

    def test_layer_cycling_stress_with_active_telemetry(self, qapp):
        """Stress: Rapidly cycle all 4 layer modes while streaming odom and camera."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            synth_img = create_raw_ros_image()
            synth_grid = create_synthetic_visual_grid()
            window.on_visual_map_received(synth_grid)

            modes = ['obstacles', 'elevation', 'surface', 'visual']
            for cycle in range(8):
                for idx, mode in enumerate(modes):
                    window.on_odom_received(float(cycle) * 0.1, 0.0, float(cycle) * 0.2, 0.1, 0.0)
                    window.on_camera_received(synth_img)
                    window.combo_layer.setCurrentIndex(idx)
                    assert window.map_layer_mode == mode
                    window.update_map_plot()
        finally:
            safe_close_window(window)

    def test_visual_orthomap_degenerate_origin_and_resolution(self, qapp):
        """Boundary: Visual grid with 0.0 or negative resolution and extreme origins."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            grid = create_synthetic_visual_grid(width=10, height=10, res=0.0, ox=-1000.0, oy=1000.0)
            window.on_visual_map_received(grid)
            window.combo_layer.setCurrentIndex(3)
            window.update_map_plot()
            assert window.latest_visual_map is not None
        finally:
            safe_close_window(window)

    def test_zoom_and_pan_with_visual_orthomap(self, qapp):
        """Stress: Rapid zoom in, zoom out, and fit arena while rendering visual orthomap."""
        window = MappingDashboardWindow(ros_worker=None)
        try:
            grid = create_synthetic_visual_grid()
            window.on_visual_map_received(grid)
            window.combo_layer.setCurrentIndex(3)

            for _ in range(5):
                window.on_zoom_in()
                window.on_zoom_out()
            window.on_fit_arena()
            assert window.zoom_level == 1.0
            assert window.view_offset_x == 0.0
            assert window.view_offset_y == 0.0
        finally:
            safe_close_window(window)


# ==============================================================================
# Challenge 5: High-Frequency Multi-Threaded Concurrency
# ==============================================================================

class TestConcurrencyAndThreadSafetyAdversarial:
    """Stress test multi-thread signals, rapid teleop actions, and concurrent visual events."""

    def test_concurrent_worker_signals_into_dashboard(self, qapp):
        """Concurrency: Emit odometry, camera, and visual map signals concurrently across threads."""
        import threading

        worker = MappingRosWorker()
        window = MappingDashboardWindow(ros_worker=worker)
        try:
            stop_event = threading.Event()

            def emit_odom():
                count = 0
                while not stop_event.is_set() and count < 100:
                    worker.odom_received.emit(0.01 * count, 0.01 * count, 0.05 * count, 0.1, 0.0)
                    count += 1
                    time.sleep(0.001)

            def emit_camera():
                count = 0
                img = create_raw_ros_image()
                while not stop_event.is_set() and count < 100:
                    worker.camera_received.emit(img)
                    count += 1
                    time.sleep(0.001)

            def emit_visual_map():
                count = 0
                grid = create_synthetic_visual_grid(fill_val=60)
                while not stop_event.is_set() and count < 50:
                    worker.visual_map_received.emit(grid)
                    count += 1
                    time.sleep(0.002)

            t1 = threading.Thread(target=emit_odom)
            t2 = threading.Thread(target=emit_camera)
            t3 = threading.Thread(target=emit_visual_map)

            t1.start()
            t2.start()
            t3.start()

            for _ in range(20):
                qapp.processEvents()
                window.update_map_plot()
                time.sleep(0.005)

            stop_event.set()
            t1.join(timeout=1.0)
            t2.join(timeout=1.0)
            t3.join(timeout=1.0)
            qapp.processEvents()

            assert window.has_received_odom is True
            assert window.latest_camera_frame is not None
            assert window.latest_visual_map is not None
        finally:
            safe_close_window(window)

    def test_stress_pinning_and_clearing_under_active_teleop(self, qapp):
        """Stress: Interleave WASD teleop updates, pin clicks, and clear map resets."""
        from PyQt5.QtCore import Qt

        window = MappingDashboardWindow(ros_worker=None)
        try:
            window.latest_camera_frame = np.full((480, 640, 3), 180, dtype=np.uint8)

            for step in range(30):
                # Teleop step
                window.keys_pressed.add(Qt.Key_W if step % 2 == 0 else Qt.Key_D)
                window.update_teleop_loop()

                # Telemetry update
                window.on_odom_received(0.05 * step, 0.02 * step, 0.1 * step, 0.12, 0.0)

                # Pin view every 3rd step
                if step % 3 == 0:
                    window.on_pin_view_to_map()

                # Clear map on step 15
                if step == 15:
                    window.on_clear_map_clicked()
                    assert len(window.visual_keyframes) == 0

            assert len(window.visual_keyframes) > 0
            window.update_map_plot()
            assert len(window.keyframe_artists) == len(window.visual_keyframes)
        finally:
            safe_close_window(window)

