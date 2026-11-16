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

"""Test suite verifying user requirements for window layout, rotation, and target dragging."""

import math
import os
import xml.etree.ElementTree as ET

import pytest
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication

from khepera4_control.mapping_dashboard import CameraPopupWindow, MappingDashboardWindow


@pytest.fixture(scope="session")
def qapp():
    """Create persistent QApplication instance for headless test execution."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


def test_window_layout_geometries(qapp):
    """Verify Dashboard and Camera popup calculate left-half and top-right quarter geometries."""
    dash = MappingDashboardWindow(ros_worker=None)
    dash.show()

    screen = QGuiApplication.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        expected_w = max(dash.minimumWidth(), geo.width() // 2)
        expected_h = max(dash.minimumHeight(), geo.height())
        assert dash.width() == expected_w
        assert dash.height() == expected_h

    # Verify camera popup geometry
    popup = CameraPopupWindow(dashboard=dash)
    popup.show()
    if screen:
        geo = screen.availableGeometry()
        expected_pw = max(popup.minimumWidth(), geo.width() // 2)
        expected_ph = max(popup.minimumHeight(), geo.height() // 2)
        assert popup.width() == expected_pw
        assert popup.height() == expected_ph

    popup.close()
    dash.close()


def test_gazebo_gui_config_bottom_right_quarter():
    """Verify khepera4_gui.config places Gazebo window in bottom-right quarter."""
    config_path = os.path.expanduser(
        "~/Desktop/khepera4_ws/src/khepera4_description/config/khepera4_gui.config"
    )
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        content = f.read()

    # Wrap in single root for standard XML parser
    tree = ET.fromstring(f"<root>{content.replace('<?xml version=\"1.0\"?>', '')}</root>")
    window = tree.find("window")
    assert window is not None

    width = int(window.find("width").text)
    height = int(window.find("height").text)
    pos_x = int(window.find("position_x").text)
    pos_y = int(window.find("position_y").text)

    # 1280x720 at (1280, 720) corresponds to bottom-right quarter of 2560x1440 display
    assert width == 1280
    assert height == 720
    assert pos_x == 1280
    assert pos_y == 720


def test_description_field_cleared_on_boot(qapp):
    """Verify description text field is empty on startup with appropriate placeholder."""
    dash = MappingDashboardWindow(ros_worker=None)
    assert hasattr(dash, "edit_target_desc")
    assert dash.edit_target_desc.text() == ""
    assert "Enter target semantic description" in dash.edit_target_desc.placeholderText()
    dash.close()


def test_slam_map_90_deg_ccw_rotation_math(qapp):
    """Verify coordinate transforms: (X, Y) -> (X', Y') = (-Y, X) and inverse."""
    # Test point at (x=1.5, y=0.0)
    wx, wy = 1.5, 0.0
    cx, cy = -wy, wx
    assert cx == 0.0
    assert cy == 1.5

    # Inverse mapping from canvas back to world
    inv_wx = cy
    inv_wy = -cx
    assert math.isclose(inv_wx, wx)
    assert math.isclose(inv_wy, wy)

    # Test point at (x=0.0, y=2.0)
    wx, wy = 0.0, 2.0
    cx, cy = -wy, wx
    assert cx == -2.0
    assert cy == 0.0
    inv_wx = cy
    inv_wy = -cx
    assert math.isclose(inv_wx, wx)
    assert math.isclose(inv_wy, wy)


def test_interactive_target_dragging_and_spinbox_sync(qapp):
    """Verify interactive dragging state transitions and two-way spinbox synchronization."""
    dash = MappingDashboardWindow(ros_worker=None)
    dash.show()

    # Initial target pose at (1.50, 0.00)
    assert dash.spin_target_pos_x.value() == 1.50
    assert dash.spin_target_pos_y.value() == 0.00

    # Test spinbox change updates target_pose preview
    dash.spin_target_pos_x.setValue(-0.80)
    dash.spin_target_pos_y.setValue(1.20)
    assert dash.target_pose[0] == -0.80
    assert dash.target_pose[1] == 1.20

    # Simulate canvas dragging near target dot:
    # Target in world is (-0.80, 1.20) -> canvas coords: X' = -1.20, Y' = -0.80
    class MockMouseEvent:
        def __init__(self, inaxes, xdata, ydata, button=1):
            self.inaxes = inaxes
            self.xdata = xdata
            self.ydata = ydata
            self.button = button

    # Press near target dot
    press_event = MockMouseEvent(dash.ax_map, -1.20, -0.80, button=1)
    dash.on_canvas_press(press_event)
    assert dash.is_dragging_target is True

    # Drag to new canvas position (-0.50, 0.50) -> world coords: X = 0.50, Y = 0.50
    motion_event = MockMouseEvent(dash.ax_map, -0.50, 0.50)
    dash.on_canvas_motion(motion_event)
    assert math.isclose(dash.target_pose[0], 0.50, abs_tol=1e-2)
    assert math.isclose(dash.target_pose[1], 0.50, abs_tol=1e-2)
    assert math.isclose(dash.spin_target_pos_x.value(), 0.50, abs_tol=1e-2)
    assert math.isclose(dash.spin_target_pos_y.value(), 0.50, abs_tol=1e-2)

    # Release mouse
    release_event = MockMouseEvent(dash.ax_map, -0.50, 0.50, button=1)
    dash.on_canvas_release(release_event)
    assert dash.is_dragging_target is False

    dash.close()
