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
Unit and Integration Tests for Khepera4CameraBridge.
"""

import socket
import threading
import time

import cv2
from khepera4_control.khepera4_camera_bridge import Khepera4CameraBridge
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image


@pytest.fixture(scope='module')
def rclpy_init():
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_camera_bridge_mock_mode(rclpy_init):
    """Verify camera bridge in mock mode generates synthetic Image and CameraInfo."""
    from rclpy.parameter import Parameter
    node = Khepera4CameraBridge(parameter_overrides=[Parameter('mock', Parameter.Type.BOOL, True)])

    received_images = []
    received_infos = []

    sub_img = node.create_subscription(
        Image, '/camera/image_raw',
        lambda m: received_images.append(m), 10
    )
    sub_info = node.create_subscription(
        CameraInfo, '/camera/camera_info',
        lambda m: received_infos.append(m), 10
    )

    t0 = time.time()
    while time.time() - t0 < 1.0 and len(received_images) < 2:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()

    assert len(received_images) >= 1, "Should receive at least one mock Image"
    assert len(received_infos) >= 1, "Should receive at least one CameraInfo"
    assert received_images[0].width == 640
    assert received_images[0].height == 480
    assert received_images[0].encoding == 'rgb8'
    assert received_images[0].header.frame_id == 'camera_optical_frame'
    assert received_infos[0].width == 640
    assert received_infos[0].height == 480


def test_camera_bridge_live_socket(rclpy_init):
    """Verify camera bridge parses real MJPEG TCP stream accurately."""
    # Create mock TCP MJPEG server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 0))
    port = server.getsockname()[1]
    server.listen(1)

    running_server = True

    def server_thread():
        # Generate dummy 640x480 test JPEG
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        _, jpg_data = cv2.imencode('.jpg', frame)
        jpg_bytes = jpg_data.tobytes()

        try:
            conn, _ = server.accept()
            while running_server:
                conn.sendall(b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
                time.sleep(0.05)
            conn.close()
        except Exception:
            pass

    th = threading.Thread(target=server_thread, daemon=True)
    th.start()

    node = Khepera4CameraBridge(parameter_overrides=[
        Parameter('robot_ip', Parameter.Type.STRING, '127.0.0.1'),
        Parameter('camera_port', Parameter.Type.INTEGER, port),
        Parameter('mock', Parameter.Type.BOOL, False)
    ])

    received_images = []
    sub = node.create_subscription(
        Image, '/camera/image_raw',
        lambda m: received_images.append(m), 10
    )

    t0 = time.time()
    while time.time() - t0 < 1.5 and len(received_images) < 2:
        rclpy.spin_once(node, timeout_sec=0.1)

    running_server = False
    server.close()
    node.destroy_node()

    assert len(received_images) >= 1, "Should receive and decode TCP MJPEG frames"
    assert received_images[0].width == 640
    assert received_images[0].height == 480
    assert received_images[0].encoding == 'rgb8'
