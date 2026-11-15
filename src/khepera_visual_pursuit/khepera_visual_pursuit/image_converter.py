# Copyright 2026 Andrei Berteanu.
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

"""Zero-dependency conversion between ROS Image messages and OpenCV images."""

import cv2
import numpy as np

from khepera_visual_pursuit.ros_compat import Image


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """Convert ROS sensor_msgs.msg.Image to OpenCV BGR numpy array."""
    if msg is None or not msg.data:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    height = int(msg.height)
    width = int(msg.width)
    encoding = (msg.encoding or "bgr8").lower()

    if height <= 0 or width <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    raw_bytes = bytes(msg.data)
    if encoding in ("bgr8", "8uc3"):
        expected_len = height * width * 3
        data_arr = np.frombuffer(raw_bytes[:expected_len], dtype=np.uint8)
        return data_arr.reshape((height, width, 3))

    if encoding == "rgb8":
        expected_len = height * width * 3
        data_arr = (
            np.frombuffer(raw_bytes[:expected_len], dtype=np.uint8).reshape((height, width, 3))
        )
        return cv2.cvtColor(data_arr, cv2.COLOR_RGB2BGR)

    if encoding in ("mono8", "8uc1"):
        expected_len = height * width
        data_arr = (
            np.frombuffer(raw_bytes[:expected_len], dtype=np.uint8).reshape((height, width))
        )
        return cv2.cvtColor(data_arr, cv2.COLOR_GRAY2BGR)

    expected_len = height * width * 3
    if len(raw_bytes) >= expected_len:
        data_arr = np.frombuffer(raw_bytes[:expected_len], dtype=np.uint8)
        return data_arr.reshape((height, width, 3))

    return np.zeros((height, width, 3), dtype=np.uint8)


def cv2_to_imgmsg(
    cv_image: np.ndarray,
    encoding: str = "bgr8",
    frame_id: str = "camera_optical_frame",
) -> Image:
    """Convert OpenCV BGR numpy array to ROS sensor_msgs.msg.Image."""
    msg = Image()
    msg.header.frame_id = frame_id
    if cv_image is None or cv_image.size == 0:
        return msg

    msg.height = int(cv_image.shape[0])
    msg.width = int(cv_image.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0

    if cv_image.ndim == 2:
        msg.step = int(msg.width)
        msg.data = cv_image.tobytes()
    else:
        msg.step = int(msg.width * cv_image.shape[2])
        msg.data = cv_image.tobytes()

    return msg
