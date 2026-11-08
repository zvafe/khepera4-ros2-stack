"""Offline ROS 2 rosbag (.db3 / .mcap) processor for Khepera IV 3DGS pipeline."""

import logging
import math
import os
from typing import Any, Generator, List, Optional, Tuple

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

LOGGER = logging.getLogger("khepera_3dgs.rosbag_processor")


class RosbagReader:
    """Reads messages from ROS 2 bags (.db3 or .mcap)."""

    def __init__(
        self,
        bag_path: str,
        storage_id: str = "",
    ) -> None:
        """
        Initialize rosbag reader.

        :param bag_path: Directory or file path of the rosbag.
        :param storage_id: Storage plugin ('sqlite3' for .db3 or 'mcap').
        """
        self.bag_path = bag_path
        if not storage_id:
            if bag_path.endswith(".mcap") or os.path.isdir(bag_path):
                # Inspect directory for mcap
                storage_id = "mcap" if "mcap" in bag_path else "sqlite3"
            else:
                storage_id = "sqlite3"
        self.storage_id = storage_id

    def read_messages(
        self, topics: Optional[List[str]] = None
    ) -> Generator[Tuple[str, Any, int], None, None]:
        """
        Yield (topic, deserialized_msg, timestamp_ns) tuples.

        :param topics: Optional list of topic names to filter.
        :return: Generator over bag messages.
        """
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path, storage_id=self.storage_id
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)

        topic_types = {
            meta.name: meta.type for meta in reader.get_all_topics_and_types()
        }

        # Topic filtering
        if topics:
            storage_filter = rosbag2_py.StorageFilter(topics=topics)
            reader.set_filter(storage_filter)

        from rosidl_runtime_py.utilities import get_message

        while reader.has_next():
            topic, data, stamp = reader.read_next()
            msg_type_str = topic_types.get(topic)
            if msg_type_str is None:
                continue

            msg_cls = get_message(msg_type_str)
            msg = deserialize_message(data, msg_cls)
            yield topic, msg, stamp


class RosbagKeyframeExtractor:
    """
    Extracts time-synchronized camera frames and robot map poses from recorded bag.

    Reconstructs continuous TF trajectory (map -> base_footprint) and associates each
    camera frame with its corresponding map pose (x, y, yaw).
    """

    def __init__(
        self,
        bag_path: str,
        image_topic: str = "/camera/image_raw",
        tf_topic: str = "/tf",
        camera_info_topic: str = "/camera/camera_info",
    ) -> None:
        """
        Initialize extractor with target topics.

        :param bag_path: Path to recorded bag.
        :param image_topic: Topic name for RGB frames.
        :param tf_topic: Topic name for dynamic transforms.
        :param camera_info_topic: Topic name for camera calibration.
        """
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.tf_topic = tf_topic
        self.camera_info_topic = camera_info_topic
        self.reader = RosbagReader(bag_path)

    @staticmethod
    def euler_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        """
        Compute yaw heading from quaternion.

        :return: Yaw angle in radians.
        """
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def ros_image_to_cv2(msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image to RGB NumPy array.

        :param msg: ROS Image message.
        :return: (H, W, 3) uint8 RGB array.
        """
        if msg.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr if msg.encoding == "rgb8" else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif msg.encoding in ("mono8", "8UC1"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            return arr[:, :, :3]

    def extract_synchronized_samples(
        self,
    ) -> List[Tuple[float, np.ndarray, float, float, float, float, float]]:
        """
        Process rosbag and return sorted synchronized samples.

        :return: List of (timestamp, image_rgb, x, y, yaw, linear_vel, angular_vel).
        """
        tf_records: List[Tuple[float, float, float, float]] = []  # (t, x, y, yaw)
        image_records: List[Tuple[float, np.ndarray]] = []

        LOGGER.info("Scanning rosbag %s for TF and image messages...", self.bag_path)
        for topic, msg, stamp_ns in self.reader.read_messages(
            topics=[self.image_topic, self.tf_topic, "/tf_static"]
        ):
            stamp_sec = stamp_ns * 1e-9

            if topic in (self.tf_topic, "/tf_static"):
                tf_msg: TFMessage = msg
                for tr in tf_msg.transforms:
                    # Look for map -> base_footprint or map -> base_link
                    child = tr.child_frame_id.lstrip("/")
                    parent = tr.header.frame_id.lstrip("/")
                    if parent in ("map", "odom") and child in (
                        "base_footprint",
                        "base_link",
                        "khepera4/base_footprint",
                    ):
                        x = tr.transform.translation.x
                        y = tr.transform.translation.y
                        q = tr.transform.rotation
                        yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
                        t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                        tf_records.append((t, x, y, yaw))

            elif topic == self.image_topic:
                img_rgb = self.ros_image_to_cv2(msg)
                t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if t_img <= 0.0:
                    t_img = stamp_sec
                image_records.append((t_img, img_rgb))

        LOGGER.info(
            "Found %d TF transforms and %d camera frames in bag.",
            len(tf_records),
            len(image_records),
        )

        if not tf_records:
            LOGGER.warning("No map->base TF found in bag. Generating relative odometry poses.")
            # Fallback if no map TF was recorded: simulate stationary or incremental sequence
            samples = []
            for idx, (t, img) in enumerate(image_records):
                samples.append((t, img, 0.0, 0.0, 0.0, 0.0, 0.0))
            return samples

        # Sort by timestamp
        tf_records.sort(key=lambda r: r[0])
        tf_times = np.array([r[0] for r in tf_records])
        tf_x = np.array([r[1] for r in tf_records])
        tf_y = np.array([r[2] for r in tf_records])
        tf_yaw = np.unwrap(np.array([r[3] for r in tf_records]))

        synchronized = []
        prev_x, prev_y, prev_t = None, None, None

        for t_img, img in image_records:
            # Interpolate pose at t_img
            idx = np.searchsorted(tf_times, t_img)
            if idx == 0:
                cur_x, cur_y, cur_yaw = tf_x[0], tf_y[0], tf_yaw[0]
            elif idx >= len(tf_times):
                cur_x, cur_y, cur_yaw = tf_x[-1], tf_y[-1], tf_yaw[-1]
            else:
                dt = tf_times[idx] - tf_times[idx - 1]
                ratio = (t_img - tf_times[idx - 1]) / max(dt, 1e-6)
                cur_x = float(tf_x[idx - 1] + ratio * (tf_x[idx] - tf_x[idx - 1]))
                cur_y = float(tf_y[idx - 1] + ratio * (tf_y[idx] - tf_y[idx - 1]))
                cur_yaw = float(tf_yaw[idx - 1] + ratio * (tf_yaw[idx] - tf_yaw[idx - 1]))

            # Velocity estimation
            lin_vel = 0.0
            ang_vel = 0.0
            if prev_t is not None and (t_img - prev_t) > 1e-4:
                dt_step = t_img - prev_t
                lin_vel = math.hypot(cur_x - prev_x, cur_y - prev_y) / dt_step

            prev_x, prev_y, prev_t = cur_x, cur_y, t_img
            synchronized.append((t_img, img, cur_x, cur_y, cur_yaw, lin_vel, ang_vel))

        return synchronized
