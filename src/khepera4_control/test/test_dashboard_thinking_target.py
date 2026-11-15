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
Unit tests for Dashboard Target Specification & Cognitive Telemetry features.

Covers:
1. Dynamic exploration speed modification via teleoperation sliders.
2. View mode switching (Map Only, Target/Thinking Only, Split View).
3. Target detail inputs, mode radio buttons, relocation, and randomization.
4. Reference template image selection and deployment via ROS topic.
5. Cognitive thinking stream, spatial intent updates, and perception telemetry.
6. Teleoperation key suppression (WASD, Space, Arrow keys) in all text input fields.
7. Visual pursuit toggle and command dispatching.
"""

import os
import sys
import tempfile

import cv2
from khepera4_control.mapping_dashboard import (
    MappingDashboardWindow,
    MappingRosWorker,
)
import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(scope='module')
def qapp():
    """Ensure QApplication instance is active for Qt tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def safe_close_window(w: MappingDashboardWindow):
    """Safely terminate timers and close window."""
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


class MockRosWorker(MappingRosWorker):
    """Mock ROS worker to capture speed changes, goals, and target deployments."""

    def __init__(self):
        """Initialize mock ROS worker with recorded telemetry call lists."""
        super().__init__()
        self.explorer_speeds = []
        self.published_target_images = []
        self.published_target_prompts = []
        self.spawn_requests = []
        self.dispatched_goals = []
        self.pursuit_enables = []

    def set_explorer_speed(self, linear_v: float, angular_w: float):
        """Capture explorer speed settings."""
        self.explorer_speeds.append((linear_v, angular_w))

    def publish_target_image(self, img_msg):
        """Capture published reference target images."""
        self.published_target_images.append(img_msg)

    def publish_target_prompt(self, prompt: str):
        """Capture published target semantic text prompts."""
        self.published_target_prompts.append(prompt)

    def set_pursuit_enable(self, enable: bool):
        """Capture pursuit enable/disable requests."""
        self.pursuit_enables.append(enable)

    def spawn_sim_target(self, model_name: str, x: float, y: float, z: float, yaw: float):
        """Capture simulated target relocation requests."""
        self.spawn_requests.append((model_name, x, y, z, yaw))

    def send_user_goal(self, x: float, y: float):
        """Capture dispatched user waypoint goals."""
        self.dispatched_goals.append((x, y))


def test_teleop_sliders_dynamic_speed(qapp):
    """Verify teleop linear & angular sliders update explorer speed dynamically."""
    mock_ros = MockRosWorker()
    window = MappingDashboardWindow(ros_worker=mock_ros)
    try:
        window.slider_v.setValue(25)  # 0.25 m/s
        assert len(mock_ros.explorer_speeds) >= 1
        assert mock_ros.explorer_speeds[-1][0] == pytest.approx(0.25, abs=0.01)

        window.slider_w.setValue(180)  # 1.80 rad/s
        assert mock_ros.explorer_speeds[-1][1] == pytest.approx(1.80, abs=0.01)
    finally:
        safe_close_window(window)


def test_view_mode_switching(qapp):
    """Verify switching between SLAM Map View, Target & Thinking View, and Split View."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        assert hasattr(window, 'combo_view_mode')
        assert hasattr(window, 'canvas_container')
        assert hasattr(window, 'thinking_target_panel')

        # Index 0: SLAM Map View (Default)
        window.combo_view_mode.setCurrentIndex(0)
        assert not window.canvas_container.isHidden()
        assert window.thinking_target_panel.isHidden()

        # Index 1: Target & Thinking View (replaces map)
        window.combo_view_mode.setCurrentIndex(1)
        assert window.canvas_container.isHidden()
        assert not window.thinking_target_panel.isHidden()

        # Index 2: Split View (Both visible)
        window.combo_view_mode.setCurrentIndex(2)
        assert not window.canvas_container.isHidden()
        assert not window.thinking_target_panel.isHidden()
    finally:
        safe_close_window(window)


def test_target_specification_inputs_and_relocation(qapp):
    """Verify target specification input fields, radio buttons, and relocation slot."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        assert hasattr(window, 'radio_mode_semantic')
        assert hasattr(window, 'radio_mode_template')
        assert hasattr(window, 'edit_target_name')
        assert hasattr(window, 'edit_target_desc')
        assert hasattr(window, 'spin_target_pos_x')
        assert hasattr(window, 'spin_target_pos_y')
        assert hasattr(window, 'btn_relocate_target_input')
        assert hasattr(window, 'btn_randomize_target')

        # Verify initial button texts and mode states
        assert window.radio_mode_semantic.isChecked() is True
        assert window.radio_mode_template.isChecked() is False
        assert window.btn_relocate_target_input.text() == '📍 Relocate'
        assert window.btn_randomize_target.text() == '🎲 Randomize'

        # Switch to Mode B
        window.radio_mode_template.setChecked(True)
        assert window.radio_mode_template.isChecked() is True
        assert window.radio_mode_semantic.isChecked() is False

        # Switch back to Mode A
        window.radio_mode_semantic.setChecked(True)
        assert window.radio_mode_semantic.isChecked() is True

        window.edit_target_name.setText('Green Canister')
        window.edit_target_desc.setText('Metallic green canister with high contrast badge')
        window.spin_target_pos_x.setValue(1.80)
        window.spin_target_pos_y.setValue(-0.60)

        window.on_relocate_target_from_inputs()
        assert window.target_pose[0] == pytest.approx(1.80, abs=0.01)
        assert window.target_pose[1] == pytest.approx(-0.60, abs=0.01)

        log_text = window.txt_thinking_log.toPlainText()
        assert 'Target Placed' in log_text
        assert 'Green Canister' in log_text
    finally:
        safe_close_window(window)


def test_target_randomization_and_placement_tools(qapp):
    """Verify target randomization and map canvas interactive placement buttons."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        assert hasattr(window, 'btn_randomize_target')
        assert hasattr(window, 'btn_place_target')

        # Set known robot pose
        window.robot_x = 0.50
        window.robot_y = 0.20
        window.robot_yaw = 0.00
        window.edit_target_name.setText('Orange Cone')

        window.on_randomize_target_clicked()

        # Ahead of robot by 1.6m at yaw 0: x ~ 2.10, y ~ 0.20
        assert window.target_pose[0] == pytest.approx(2.10, abs=0.05)
        assert window.target_pose[1] == pytest.approx(0.20, abs=0.05)
        assert window.spin_target_pos_x.value() == pytest.approx(2.10, abs=0.05)
        assert window.spin_target_pos_y.value() == pytest.approx(0.20, abs=0.05)

        log_text = window.txt_thinking_log.toPlainText()
        assert 'Target Randomized' in log_text
        assert 'Orange Cone' in log_text

        # Verify interactive map place target button
        window.btn_place_target.setChecked(True)
        window.on_toggle_place_target_mode()
        assert window.map_interaction_mode == 'place_target'

        window.btn_place_target.setChecked(False)
        window.on_toggle_place_target_mode()
        assert window.map_interaction_mode == 'none'
    finally:
        safe_close_window(window)


def test_template_image_deployment(qapp):
    """Verify target reference template encoding and deployment via ROS topic."""
    mock_ros = MockRosWorker()
    window = MappingDashboardWindow(ros_worker=mock_ros)
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            dummy_img = np.full((64, 64, 3), 128, dtype=np.uint8)
            cv2.imwrite(tmp_path, dummy_img)

        try:
            window.edit_target_img_path.setText(tmp_path)
            window.on_deploy_target_template()

            assert len(mock_ros.published_target_images) == 1
            msg = mock_ros.published_target_images[0]
            assert msg.width == 64
            assert msg.height == 64
            assert msg.encoding == 'bgr8'
            assert len(msg.data) == 64 * 64 * 3

            # Check that deployment was logged to thinking stream
            log_text = window.txt_thinking_log.toPlainText()
            assert 'Reference template' in log_text
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        safe_close_window(window)


def test_cognitive_thinking_and_sighting_telemetry(qapp):
    """Verify live cognitive thoughts, spatial intent, and perception telemetry."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        # 1. Thought: 'Where should I go'
        window.on_thinking_log_received('Where should I go: exploring frontier at (1.2, -0.8)')
        assert 'Where should I go' in window.lbl_thinking_nav_intent.text()
        assert 'Where should I go' in window.txt_thinking_log.toPlainText()

        # 2. Thought: 'Target detected'
        window.on_thinking_log_received('Target detected: candidate at [120, 80, 240, 200]')
        assert 'CANDIDATE DETECTED' in window.lbl_target_sighting_badge.text()
        assert 'Target detected' in window.txt_thinking_log.toPlainText()

        # 3. Thought: 'Relocated target'
        window.on_thinking_log_received('Relocated target to arena position (1.50, 0.00)')
        assert 'Relocated target' in window.lbl_thinking_nav_intent.text()
        assert 'Relocated target' in window.txt_thinking_log.toPlainText()

        # 4. Target verified thought
        window.on_thinking_log_received('Verified target with confidence 0.89! Locking tracker.')
        assert 'TARGET CONFIRMED' in window.lbl_target_sighting_badge.text()
        assert 'VISUAL_PURSUIT' in window.lbl_thinking_motion_badge.text()

        # 5. Direct status and confidence updates
        window.on_target_confidence_received(0.85)
        assert '85.0%' in window.lbl_thinking_dinov2_sim.text()

        window.on_target_status_received(False)
        assert 'NO TARGET IN SIGHT' in window.lbl_target_sighting_badge.text()

        # 6. Clear thoughts
        window.txt_thinking_log.clear()
        assert window.txt_thinking_log.toPlainText() == ''
    finally:
        safe_close_window(window)


def test_teleop_key_suppression_when_focused_on_text_inputs(qapp):
    """Verify teleop keys (WASD, Space, Arrows) are ignored when focused on text fields."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        teleop_keys_to_test = (
            (Qt.Key_W, 'w'),
            (Qt.Key_A, 'a'),
            (Qt.Key_S, 's'),
            (Qt.Key_D, 'd'),
            (Qt.Key_Space, ' '),
            (Qt.Key_Up, ''),
            (Qt.Key_Down, ''),
            (Qt.Key_Left, ''),
            (Qt.Key_Right, ''),
        )

        # 1. Target Description (QLineEdit)
        for key_code, char_str in teleop_keys_to_test:
            press_ev = QKeyEvent(QEvent.KeyPress, key_code, Qt.NoModifier, char_str)
            window.eventFilter(window.edit_target_desc, press_ev)
            assert len(window.keys_pressed) == 0
            rel_ev = QKeyEvent(QEvent.KeyRelease, key_code, Qt.NoModifier, char_str)
            window.eventFilter(window.edit_target_desc, rel_ev)
            assert len(window.keys_pressed) == 0

        # 2. Target Name (QLineEdit)
        for key_code, char_str in teleop_keys_to_test:
            press_ev = QKeyEvent(QEvent.KeyPress, key_code, Qt.NoModifier, char_str)
            window.eventFilter(window.edit_target_name, press_ev)
            assert len(window.keys_pressed) == 0

        # 3. Coordinate Spinbox (QDoubleSpinBox / QAbstractSpinBox)
        for key_code, char_str in teleop_keys_to_test:
            press_ev = QKeyEvent(QEvent.KeyPress, key_code, Qt.NoModifier, char_str)
            window.eventFilter(window.spin_target_pos_x, press_ev)
            assert len(window.keys_pressed) == 0

        # 4. Thinking Log (QTextEdit)
        for key_code, char_str in teleop_keys_to_test:
            press_ev = QKeyEvent(QEvent.KeyPress, key_code, Qt.NoModifier, char_str)
            window.eventFilter(window.txt_thinking_log, press_ev)
            assert len(window.keys_pressed) == 0

        # 5. Direct window key events when text input has focus
        window.edit_target_desc.setFocus()
        QApplication.processEvents()
        press_w = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier, 'w')
        window.keyPressEvent(press_w)
        assert Qt.Key_W not in window.keys_pressed

        press_space = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, ' ')
        window.keyPressEvent(press_space)
        assert Qt.Key_Space not in window.keys_pressed

        # 6. Safety clear: if keys were held and text field receives event, keys are cleared
        window.keys_pressed.add(Qt.Key_W)
        window.keys_pressed.add(Qt.Key_D)
        assert len(window.keys_pressed) == 2
        window.eventFilter(window.edit_target_desc, press_w)
        assert len(window.keys_pressed) == 0

        # 7. Normal text editing verification: typing into edit_target_desc works
        window.edit_target_desc.setText('')
        window.edit_target_desc.insert('wasd target test')
        assert window.edit_target_desc.text() == 'wasd target test'
    finally:
        safe_close_window(window)


def test_pursuit_button_enables_visual_pursuit(qapp):
    """Verify clicking pursuit engages visual pursuit controller and dispatches prompt."""
    mock_ros = MockRosWorker()
    window = MappingDashboardWindow(ros_worker=mock_ros)
    try:
        window.edit_target_name.setText('Red Cylinder')
        window.radio_mode_semantic.setChecked(True)
        window.on_toggle_pursuit_clicked()

        assert window.is_pursuit_active is True
        assert len(mock_ros.pursuit_enables) >= 1
        assert mock_ros.pursuit_enables[-1] is True
        assert len(mock_ros.published_target_prompts) >= 1

        # Toggle off
        window.on_toggle_pursuit_clicked()
        assert window.is_pursuit_active is False
        assert mock_ros.pursuit_enables[-1] is False
    finally:
        safe_close_window(window)


def test_stream_of_consciousness_filters_speed_adjustments_and_shows_cognitive_intent(qapp):
    """Verify speed adjustment spam is suppressed and cognitive target rationale is displayed."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        # 1. Speed adjustment telemetry spam must be suppressed from thinking log
        spam_msg = '⚡ Exploration Speed Adjusted: Cruise v = 0.20 m/s, Turn Rate ω = 1.00 rad/s'
        window.on_thinking_log_received(spam_msg)
        assert spam_msg not in window.txt_thinking_log.toPlainText()

        # 2. Cognitive goal-directed thoughts must be accepted and displayed
        cognitive_thought = (
            'I pick frontier target (1.20, 0.40) because this unexplored boundary '
            'offers a vantage where I might see the target object.'
        )
        window.on_thinking_log_received(cognitive_thought)
        assert cognitive_thought in window.txt_thinking_log.toPlainText()

        # 3. Engaging pursuit emits steering toward target thought
        window.edit_target_name.setText('Red Cylinder')
        window.target_pose = (1.50, 0.00, 0.0)
        window.on_toggle_pursuit_clicked()
        expected_thought = "[Pursuit Active] Robot steering toward 'Red Cylinder'"
        assert expected_thought in window.txt_thinking_log.toPlainText()
    finally:
        safe_close_window(window)


def test_live_camera_view_overlays_square_over_pursued_object(qapp):
    """Verify live camera feed overlays target bounding square when target is in sight."""
    from sensor_msgs.msg import Image as RosImage

    window = MappingDashboardWindow(ros_worker=None)
    try:
        # 1. Perception ROI detection overlay
        window.on_target_roi_received(80, 60, 220, 200, 'Red Cylinder', 0.92)
        bbox = window._get_target_visual_bbox(640, 480)
        assert bbox is not None
        assert bbox[0] == 80 and bbox[1] == 60 and bbox[2] == 220 and bbox[3] == 200
        assert bbox[4] == 'Red Cylinder'

        # 2. Generate blank frame and test overlay rendering
        blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = window._overlay_target_bbox(blank_rgb)
        # Check that green overlay pixels (e.g. 16, 185, 129) were drawn
        assert np.any(annotated[:, :, 1] > 100)

        # 3. Feed frame through ROS camera callback
        img_msg = RosImage()
        img_msg.height = 480
        img_msg.width = 640
        img_msg.encoding = 'rgb8'
        img_msg.step = 640 * 3
        img_msg.data = blank_rgb.tobytes()
        window.on_camera_received(img_msg)

        assert window.lbl_camera_view.pixmap() is not None
        assert not window.lbl_camera_view.pixmap().isNull()

        # 4. Simulation FOV projection when pursuit is active
        window.latest_target_bbox = None
        window.last_target_bbox_time = 0.0
        window.is_pursuit_active = True
        window.robot_x, window.robot_y, window.robot_yaw = 0.0, 0.0, 0.0
        window.target_pose = (1.5, 0.0, 0.0)  # directly in front of camera
        proj_bbox = window._get_target_visual_bbox(640, 480)
        assert proj_bbox is not None
        px1, py1, px2, py2, pname, pconf = proj_bbox
        assert px1 < px2 and py1 < py2
        # Target directly in front should be centered around x = 320
        mid_x = (px1 + px2) // 2
        assert abs(mid_x - 320) < 30
    finally:
        safe_close_window(window)


def test_camera_popup_displays_target_pursuit_overlay(qapp):
    """Verify pop-up camera window initializes with HUD active and renders overlay."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_toggle_camera_popup()
        assert window.camera_popup is not None
        assert window.camera_popup.hud_enabled is True

        # Provide annotated frame to popup
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:150, 100:150, 1] = 200
        window.camera_popup.update_frame(test_frame, 640, 480, 30.0)

        assert window.camera_popup.latest_frame is not None
        assert window.camera_popup.lbl_view.pixmap() is not None
    finally:
        safe_close_window(window)
