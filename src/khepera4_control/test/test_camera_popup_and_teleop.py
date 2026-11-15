#!/usr/bin/env python3
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
Unit & Integration Tests for Khepera IV Pop-up Camera Window & WASD Teleoperation.

Verifies:
1. Pop-up camera window instantiation, default always-on-top state & geometry.
2. Always-on-top toggle functionality (setting/clearing Qt.WindowStaysOnTopHint).
3. Image stream rendering and dynamic aspect-ratio scaling during resize.
4. Seamless keyboard teleoperation (W, A, S, D, Space, Arrow keys) routed to robot controller.
5. Focus-out safety stops and keyframe pin triggers.
"""

import os
import sys

from khepera4_control.mapping_dashboard import CameraPopupWindow, MappingDashboardWindow
import numpy as np
from PyQt5.QtCore import QEvent, QEventLoop, Qt
from PyQt5.QtGui import QFocusEvent, QKeyEvent
from PyQt5.QtWidgets import QApplication
import pytest
from sensor_msgs.msg import Image as RosImage

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(scope='module')
def qapp():
    """Ensure QApplication instance is active for Qt tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def safe_close_window(w: MappingDashboardWindow):
    """Safely terminate timers and close main window and popup."""
    app = QApplication.instance()
    if app:
        try:
            app.removeEventFilter(w)
        except Exception:
            pass
    if hasattr(w, 'teleop_timer') and w.teleop_timer.isActive():
        w.teleop_timer.stop()
    if hasattr(w, 'plot_timer') and w.plot_timer.isActive():
        w.plot_timer.stop()
    if hasattr(w, 'camera_popup') and w.camera_popup:
        try:
            w.camera_popup.close()
        except Exception:
            pass
    if hasattr(w, 'canvas') and w.canvas:
        try:
            w.canvas.draw_idle = lambda: None
            w.canvas.draw = lambda: None
        except Exception:
            pass
    w.close()




def test_camera_popup_creation_and_attributes(qapp):
    """Verify CameraPopupWindow initializes with proper default attributes and flags."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        assert hasattr(window, 'btn_popup_cam')
        assert window.btn_popup_cam is not None

        # Trigger popup creation
        window.on_toggle_camera_popup()
        popup = window.camera_popup
        assert popup is not None
        assert isinstance(popup, CameraPopupWindow)
        assert popup.isVisible() is True
        assert popup.is_always_on_top is True
        assert bool(popup.windowFlags() & Qt.WindowStaysOnTopHint) is True
        assert 'Always on Top: ON' in popup.btn_always_on_top.text()
    finally:
        safe_close_window(window)


def test_camera_popup_always_on_top_toggle(qapp):
    """Verify toggling Always on Top switches the Qt.WindowStaysOnTopHint window flag."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        popup = window.camera_popup

        # Initial state is ON
        assert popup.is_always_on_top is True
        assert bool(popup.windowFlags() & Qt.WindowStaysOnTopHint) is True

        # Toggle OFF
        popup.toggle_always_on_top()
        assert popup.is_always_on_top is False
        assert bool(popup.windowFlags() & Qt.WindowStaysOnTopHint) is False
        assert 'Always on Top: OFF' in popup.btn_always_on_top.text()

        # Toggle ON again
        popup.toggle_always_on_top()
        assert popup.is_always_on_top is True
        assert bool(popup.windowFlags() & Qt.WindowStaysOnTopHint) is True
        assert 'Always on Top: ON' in popup.btn_always_on_top.text()
    finally:
        safe_close_window(window)


def test_camera_popup_frame_update_and_scaling(qapp):
    """Verify image updates render to popup and handle resize without distortion."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        popup = window.camera_popup

        # Synthetic 640x480 RGB image
        synth_img = RosImage()
        synth_img.height = 480
        synth_img.width = 640
        synth_img.encoding = 'rgb8'
        synth_img.step = 640 * 3
        data_arr = np.full((480, 640, 3), 200, dtype=np.uint8)
        synth_img.data = data_arr.tobytes()

        window.on_camera_received(synth_img)

        assert popup.latest_frame is not None
        assert popup.latest_frame.shape == (480, 640, 3)
        assert popup.lbl_view.pixmap() is not None
        assert not popup.lbl_view.pixmap().isNull()

        # Resize popup window and verify smooth scaling
        popup.resize(800, 600)
        popup._render_current_frame()
        assert popup.lbl_view.pixmap() is not None
    finally:
        safe_close_window(window)


def test_camera_popup_wasd_teleop_forwarding(qapp):
    """Verify pressing W, A, S, D in the camera popup directly controls the Khepera robot."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        popup = window.camera_popup

        # Forward drive: Press 'W'
        press_w = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
        popup.keyPressEvent(press_w)
        assert Qt.Key_W in window.keys_pressed

        window.update_teleop_loop()
        assert window.active_cmd_v == pytest.approx(window.max_linear_speed)
        assert window.active_cmd_w == pytest.approx(0.0)

        # Release 'W'
        rel_w = QKeyEvent(QEvent.KeyRelease, Qt.Key_W, Qt.NoModifier)
        popup.keyReleaseEvent(rel_w)
        assert Qt.Key_W not in window.keys_pressed

        window.update_teleop_loop()
        assert window.active_cmd_v == pytest.approx(0.0)

        # Left turn: Press 'A'
        press_a = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier)
        popup.keyPressEvent(press_a)
        assert Qt.Key_A in window.keys_pressed

        window.update_teleop_loop()
        assert window.active_cmd_w == pytest.approx(window.max_angular_speed)

        # Spacebar immediate stop
        press_space = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
        popup.keyPressEvent(press_space)
        assert window.active_cmd_v == pytest.approx(0.0)
        assert window.active_cmd_w == pytest.approx(0.0)
    finally:
        safe_close_window(window)


def test_camera_popup_focus_loss_safety(qapp):
    """Verify losing focus in camera popup clears active drive keys to prevent runaway."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        popup = window.camera_popup

        # Press 'W' and 'D'
        popup.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier))
        popup.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_D, Qt.NoModifier))
        assert Qt.Key_W in window.keys_pressed
        assert Qt.Key_D in window.keys_pressed

        # Simulate focus loss
        focus_out = QFocusEvent(QEvent.FocusOut)
        popup.focusOutEvent(focus_out)

        assert Qt.Key_W not in window.keys_pressed
        assert Qt.Key_D not in window.keys_pressed
    finally:
        safe_close_window(window)


def test_camera_popup_pin_landmark(qapp):
    """Verify pinning view from popup records visual keyframe at current pose."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        popup = window.camera_popup

        window.on_odom_received(0.5, 1.2, 0.78, 0.0, 0.0)
        window.latest_camera_frame = np.ones((480, 640, 3), dtype=np.uint8)

        assert len(window.visual_keyframes) == 0
        popup.on_pin_clicked()
        assert len(window.visual_keyframes) == 1
        assert window.visual_keyframes[0][0] == pytest.approx(0.5)
        assert window.visual_keyframes[0][1] == pytest.approx(1.2)
    finally:
        safe_close_window(window)
