"""Robust image conversion utilities independent of OpenCV versioning quirks."""

import cv2
import numpy as np
from sensor_msgs.msg import Image


def cv2_to_ros_image(cv_img: np.ndarray, encoding: str = "bgr8") -> Image:
    """Convert a numpy BGR/mono array to a ROS 2 sensor_msgs/Image message."""
    msg = Image()
    msg.height, msg.width = cv_img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False
    channels = cv_img.shape[2] if len(cv_img.shape) > 2 else 1
    msg.step = cv_img.shape[1] * channels
    msg.data = cv_img.tobytes()
    return msg


def ros_image_to_cv2(msg: Image) -> np.ndarray:
    """Convert a ROS 2 sensor_msgs/Image message to a numpy OpenCV array."""
    if msg.encoding in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    if msg.encoding in ("mono8", "8UC1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
    if msg.encoding in ("32FC1", "passthrough"):
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
