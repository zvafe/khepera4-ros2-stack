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
Test Suite: Khepera IV Camera Sensor Integration & Visual Mapping Pipeline.

Verifies:
1. Physical Camera Sensor Definition in URDF & XACRO (WVGA 640x480, 60 deg HFOV, optical frame).
2. Bidirectional Bridge Configuration (both /camera/image_raw and /camera/camera_info).
3. Live Camera Viewport, Status & Keyframe Pinning in Mapping Dashboard.
4. Camera Frustum (FOV Cone) and Landmark Overlay rendering on Map Canvas.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from PyQt5.QtWidgets import QApplication
import rclpy
from sensor_msgs.msg import Image as RosImage
from khepera4_control.mapping_dashboard import MappingDashboardWindow, MappingRosWorker


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Ensure rclpy is initialized for node inspection."""
    if not rclpy.ok():
        rclpy.init()
    yield


@pytest.fixture(scope='module')
def qapp():
    """Ensure QApplication instance is active for Qt tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def get_desc_path(filename: str) -> str:
    ws_root = os.path.abspath(os.path.join(_PKG_ROOT, '..', '..'))
    return os.path.join(ws_root, 'src', 'khepera4_description', filename)


def test_urdf_camera_sensor_definition():
    """Verify khepera4.urdf defines accurate CMOS camera parameters and optical frame."""
    urdf_path = get_desc_path(os.path.join('urdf', 'khepera4.urdf'))
    root = ET.parse(urdf_path).getroot()

    # Verify link and optical frame joints
    links = {l.get('name') for l in root.findall('link')}
    assert 'camera_link' in links, "Missing camera_link in URDF"
    assert 'camera_optical_frame' in links, "Missing camera_optical_frame in URDF"

    # Find camera gazebo sensor
    cam_sensors = [s for s in root.findall('.//sensor') if s.get('type') == 'camera']
    assert len(cam_sensors) == 1, "Expected exactly 1 camera sensor in khepera4.urdf"
    cam = cam_sensors[0]

    topic_elem = cam.find('topic')
    assert topic_elem is not None and topic_elem.text == '/camera/image_raw'

    camera_elem = cam.find('camera')
    assert camera_elem is not None
    assert float(camera_elem.find('horizontal_fov').text) == pytest.approx(1.047198, rel=1e-3)

    img_elem = camera_elem.find('image')
    assert int(img_elem.find('width').text) == 640
    assert int(img_elem.find('height').text) == 480
    assert img_elem.find('format').text == 'R8G8B8'

    opt_frame_elem = camera_elem.find('optical_frame_id')
    assert opt_frame_elem is not None and opt_frame_elem.text == 'camera_optical_frame'


def test_xacro_camera_sensor_definition():
    """Verify khepera4.urdf.xacro defines matching camera parameters and optical frame."""
    xacro_path = get_desc_path(os.path.join('urdf', 'khepera4.urdf.xacro'))
    with open(xacro_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert '<link name="camera_link"/>' in content
    assert '<link name="camera_optical_frame"/>' in content
    assert '<optical_frame_id>camera_optical_frame</optical_frame_id>' in content
    assert '<horizontal_fov>1.047198</horizontal_fov>' in content
    assert '<width>640</width>' in content
    assert '<height>480</height>' in content


def test_sim_launch_bridges_camera_info_and_image_raw():
    """Verify launch scripts bridge both camera image_raw and camera_info topics."""
    sim_launch = get_desc_path(os.path.join('launch', 'khepera4_sim.launch.py'))
    with open(sim_launch, 'r', encoding='utf-8') as f:
        sim_code = f.read()
    assert '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' in sim_code
    assert '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' in sim_code

    multi_launch = get_desc_path(os.path.join('launch', 'khepera4_multi_sim.launch.py'))
    with open(multi_launch, 'r', encoding='utf-8') as f:
        multi_code = f.read()
    assert 'camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' in multi_code
    assert 'camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' in multi_code


def safe_close_window(w):
    if hasattr(w, 'teleop_timer'):
        w.teleop_timer.stop()
    if hasattr(w, 'plot_timer'):
        w.plot_timer.stop()
    w.close()


def test_dashboard_camera_widgets_and_signals(qapp):
    """Verify MappingDashboardWindow has camera view, status label, pin button and toggle."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        assert hasattr(window, 'lbl_camera_view')
        assert hasattr(window, 'lbl_camera_status')
        assert hasattr(window, 'btn_pin_view')
        assert hasattr(window, 'chk_camera_fov')
        assert hasattr(window, 'show_camera_fov')
        assert window.show_camera_fov is True
        assert window.chk_camera_fov.isChecked() is True
    finally:
        safe_close_window(window)


def test_dashboard_camera_image_processing(qapp):
    """Verify on_camera_received unpacks RGB image and updates camera status."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        # Create synthetic 640x480 RGB image
        synth_img = RosImage()
        synth_img.height = 480
        synth_img.width = 640
        synth_img.encoding = 'rgb8'
        synth_img.step = 640 * 3
        data_arr = np.full((480, 640, 3), 128, dtype=np.uint8)
        synth_img.data = data_arr.tobytes()

        window.on_camera_received(synth_img)
        assert window.latest_camera_frame is not None
        assert window.latest_camera_frame.shape == (480, 640, 3)
        assert window.camera_frame_count >= 1
    finally:
        safe_close_window(window)


def test_dashboard_pin_view_to_map(qapp):
    """Verify pinning camera view creates a visual landmark keyframe at the current pose."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.on_odom_received(1.25, -0.80, 0.50, 0.10, 0.0)
        window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        assert len(window.visual_keyframes) == 0
        window.on_pin_view_to_map()
        assert len(window.visual_keyframes) == 1

        kx, ky, kyaw, frame, label = window.visual_keyframes[0]
        assert kx == pytest.approx(1.25)
        assert ky == pytest.approx(-0.80)
        assert kyaw == pytest.approx(0.50)
        assert "KF #1" in label
    finally:
        safe_close_window(window)


def test_dashboard_clear_map_resets_keyframes(qapp):
    """Verify on_clear_map_clicked purges visual keyframes alongside path."""
    window = MappingDashboardWindow(ros_worker=None)
    try:
        window.has_received_odom = True
        window.robot_x, window.robot_y, window.robot_yaw = 0.5, 0.5, 0.0
        window.latest_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.on_pin_view_to_map()
        assert len(window.visual_keyframes) == 1

        window.on_clear_map_clicked()
        assert len(window.visual_keyframes) == 0
    finally:
        safe_close_window(window)
