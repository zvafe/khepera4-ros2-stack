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
Khepera IV Camera Hardware Bridge Node.

Connects to the Khepera IV physical robot onboard GStreamer camera streaming
daemon over TCP (default port 5001), receives real-time MJPEG frames, and
publishes standard ROS 2 Image and CameraInfo messages.

Supported Topics:
- /camera/image_raw (sensor_msgs/msg/Image)
- /camera/camera_info (sensor_msgs/msg/CameraInfo)
- /<robot_name>/camera/image_raw
- /<robot_name>/camera/camera_info
"""

import math
import socket
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class Khepera4CameraBridge(Node):
    """Bridge node connecting physical Khepera IV camera TCP stream to ROS 2."""

    def __init__(self, **kwargs):
        super().__init__('khepera4_camera_bridge', **kwargs)

        # Declare parameters
        self.declare_parameter('robot_ip', '192.168.137.43')
        self.declare_parameter('camera_port', 5001)
        self.declare_parameter('robot_name', 'khepera_1')
        self.declare_parameter('single_robot', True)
        self.declare_parameter('camera_frame_id', 'camera_optical_frame')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('hfov_deg', 60.0)
        self.declare_parameter('focal_length', 554.256258)
        self.declare_parameter('mock', False)
        self.declare_parameter('stream_rate', 15.0)

        # Retrieve parameters
        self.robot_ip = str(self.get_parameter('robot_ip').value)
        self.camera_port = int(self.get_parameter('camera_port').value)
        self.robot_name = str(self.get_parameter('robot_name').value)
        self.single_robot = bool(self.get_parameter('single_robot').value)
        self.camera_frame_id = str(self.get_parameter('camera_frame_id').value)
        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.hfov_deg = float(self.get_parameter('hfov_deg').value)
        self.fx = float(self.get_parameter('focal_length').value)
        self.fy = self.fx
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.mock = bool(self.get_parameter('mock').value)
        self.stream_rate = float(self.get_parameter('stream_rate').value)

        # Publishers
        self.image_pubs = []
        self.camera_info_pubs = []

        # Standard topics
        self.image_pubs.append(self.create_publisher(Image, '/camera/image_raw', 10))
        self.camera_info_pubs.append(self.create_publisher(CameraInfo, '/camera/camera_info', 10))

        # Scoped topics for multi-robot or explicit robot name
        scoped_img = f'/{self.robot_name}/camera/image_raw'
        scoped_info = f'/{self.robot_name}/camera/camera_info'
        if scoped_img != '/camera/image_raw':
            self.image_pubs.append(self.create_publisher(Image, scoped_img, 10))
            self.camera_info_pubs.append(self.create_publisher(CameraInfo, scoped_info, 10))

        # Static CameraInfo template
        self.camera_info_template = self._build_camera_info()

        # Threading & lifecycle control
        self.running = True
        self.connected = False
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

        self.get_logger().info(
            f'Khepera4CameraBridge initialized. Target: {self.robot_ip}:{self.camera_port} '
            f'(Frame: {self.camera_frame_id}, Mock: {self.mock})'
        )

    def _build_camera_info(self) -> CameraInfo:
        """Construct static CameraInfo message matching camera calibration."""
        info = CameraInfo()
        info.header.frame_id = self.camera_frame_id
        info.width = self.image_width
        info.height = self.image_height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]
        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        info.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        return info

    def _generate_mock_frame(self, frame_idx: int) -> np.ndarray:
        """Generate synthetic simulated camera feed for off-robot testing."""
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # Gradient ground floor
        for y in range(self.image_height):
            v = int(140 + 70 * (y / self.image_height))
            img[y, :] = [v, v - 10, v - 20]

        # Draw grid lines on ground
        for x in range(0, self.image_width, 80):
            cv2.line(img, (x, 240), (int(320 + (x - 320) * 2.5), self.image_height), (100, 100, 100), 1)

        # Arena boundary line
        cv2.line(img, (0, 240), (self.image_width, 240), (60, 60, 60), 2)

        # Status HUD overlay
        cv2.putText(
            img, f'KHEPERA IV MOCK CAM: {self.robot_name}',
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        t_str = time.strftime('%H:%M:%S')
        cv2.putText(
            img, f'TIME: {t_str} | FRAME: {frame_idx}',
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        return img

    def _publish_frame(self, rgb_frame: np.ndarray):
        """Convert numpy array to ROS 2 Image and publish to all registered camera topics."""
        now = self.get_clock().now().to_msg()
        h, w, _ = rgb_frame.shape

        img_msg = Image()
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.camera_frame_id
        img_msg.height = h
        img_msg.width = w
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = w * 3
        img_msg.data = rgb_frame.tobytes()

        info_msg = self.camera_info_template
        info_msg.header.stamp = now

        for pub in self.image_pubs:
            pub.publish(img_msg)
        for pub in self.camera_info_pubs:
            pub.publish(info_msg)

    def _stream_loop(self):
        """Worker thread maintaining TCP socket connection and decoding live JPEG stream."""
        frame_idx = 0
        interval = 1.0 / max(1.0, self.stream_rate)

        while self.running and rclpy.ok():
            if self.mock:
                frame_idx += 1
                mock_bgr = self._generate_mock_frame(frame_idx)
                mock_rgb = cv2.cvtColor(mock_bgr, cv2.COLOR_BGR2RGB)
                self._publish_frame(mock_rgb)
                time.sleep(interval)
                continue

            sock = None
            try:
                self.get_logger().info(f'Connecting to camera stream at {self.robot_ip}:{self.camera_port}...')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.5)
                sock.connect((self.robot_ip, self.camera_port))
                self.connected = True
                self.get_logger().info(f'Successfully connected to camera stream at {self.robot_ip}:{self.camera_port}')

                buffer = bytearray()
                while self.running and rclpy.ok():
                    chunk = sock.recv(8192)
                    if not chunk:
                        self.get_logger().warn('Camera stream connection closed by remote peer.')
                        break

                    buffer.extend(chunk)

                    # Locate complete JPEG frames marked by SOI (\xff\xd8) and EOI (\xff\xd9)
                    while True:
                        soi = buffer.find(b'\xff\xd8')
                        if soi == -1:
                            # Keep tail byte in case \xff is at the boundary
                            if len(buffer) > 1:
                                buffer = buffer[-1:]
                            break

                        eoi = buffer.find(b'\xff\xd9', soi + 2)
                        if eoi == -1:
                            # Discard any stale prefix before SOI to avoid unbounded buffer growth
                            if soi > 0:
                                del buffer[:soi]
                            break

                        # Extract complete JPEG payload
                        jpg_bytes = buffer[soi:eoi + 2]
                        # Discard through EOI
                        del buffer[:eoi + 2]

                        # Check if newer frames already exist in buffer; if so, skip decoding this frame
                        next_soi = buffer.find(b'\xff\xd8')
                        next_eoi = buffer.find(b'\xff\xd9', next_soi + 2) if next_soi != -1 else -1
                        if next_soi != -1 and next_eoi != -1:
                            # Drop older frame to guarantee zero latency display
                            continue

                        # Decode JPEG
                        np_buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        bgr_frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                        if bgr_frame is not None:
                            # Resize if different from expected parameters
                            if bgr_frame.shape[1] != self.image_width or bgr_frame.shape[0] != self.image_height:
                                bgr_frame = cv2.resize(
                                    bgr_frame,
                                    (self.image_width, self.image_height),
                                    interpolation=cv2.INTER_LINEAR
                                )

                            # Convert BGR to RGB for standard ROS 2 Image encoding
                            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                            self._publish_frame(rgb_frame)

            except Exception as e:
                self.connected = False
                if self.running and rclpy.ok():
                    self.get_logger().warn(f'Camera stream connection error: {e}. Reconnecting in 2.0s...')
                    # If connection refused and not mock, attempt self-healing launch on robot
                    if not self.mock and isinstance(e, (ConnectionRefusedError, socket.error)):
                        try:
                            self.get_logger().info(f'Attempting to launch camera daemon on robot {self.robot_ip}...')
                            subprocess.Popen(
                                [
                                    'ssh', '-o', 'StrictHostKeyChecking=no',
                                    '-o', 'ConnectTimeout=2',
                                    f'root@{self.robot_ip}',
                                    '/home/root/khepera4_camera_streamer.sh > /dev/null 2>&1 &'
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                        except Exception:
                            pass
                    time.sleep(2.0)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

    def destroy_node(self):
        """Clean up threads and sockets on node termination."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Khepera4CameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
