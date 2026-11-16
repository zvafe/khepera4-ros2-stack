#!/usr/bin/env python3
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

"""
MicProbe Evaluation and Ground-Truth Verification Node.

Independent supervisor node for Khepera IV visual pursuit trials:
1. Tracks simulation clock and trial timeout (120s limit).
2. Computes 3D-to-2D frustum projection of ground-truth target corners.
3. Cross-references 360-degree laser scan to cull wall/obstacle occlusions.
4. Audits detector performance ("Sees vs Detects") and logs BLIND_PASS_EVENT frames.
5. Monitors terminal stop criteria and calculates physical standoff distance.
6. Emits 15-second heartbeat telemetry and per-trial summary artifacts.
"""

from dataclasses import dataclass, field
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy

from khepera4_control.map_raycaster import MapRaycaster
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Bool, Float32
import tf2_geometry_msgs  # noqa: F401
import tf2_ros
from vision_msgs.msg import Detection2DArray

# Default file paths
DEFAULT_TRIAL_CONFIG_PATH = "/tmp/mic_eval/current_trial.json"
DEFAULT_HEARTBEAT_PATH = "/tmp/mic_eval/heartbeat.json"
DEFAULT_FRAMES_DIR = "/tmp/mic_eval/frames"
DEFAULT_SUMMARY_DIR = "/tmp/mic_eval"

# Timing and limits
HEARTBEAT_INTERVAL_SEC: float = 15.0
TRIAL_TIMEOUT_SEC: float = 120.0
PROBE_LOOP_RATE_HZ: float = 10.0

# Physical dimensions and tolerances
ROBOT_RADIUS_METERS: float = 0.07  # Khepera IV chassis front radius (7.0 cm)
STOP_VELOCITY_THRESH_LIN: float = 0.005  # m/s
STOP_VELOCITY_THRESH_ANG: float = 0.015  # rad/s
STOP_CONFIRMATION_TIME_SEC: float = 1.0  # persistence duration to confirm stop
STOP_MAX_DISTANCE_TO_TARGET: float = 0.80  # max center-to-center distance for stop check

# Camera projection defaults
DEFAULT_IMAGE_WIDTH: int = 640
DEFAULT_IMAGE_HEIGHT: int = 480
DEFAULT_FX: float = 554.25
DEFAULT_FY: float = 554.25
DEFAULT_CX: float = 320.0
DEFAULT_CY: float = 240.0
MAX_VIEW_DISTANCE_METERS: float = 4.5
OCCLUSION_MARGIN_METERS: float = 0.18
MIN_CONSECUTIVE_BLIND_FRAMES: int = 2


@dataclass
class TrialConfig:
    """Encapsulates target and robot ground truth configuration for an active trial."""

    trial_id: str = "0"
    x_gt: float = 0.0
    y_gt: float = 0.0
    z_gt: float = 0.20
    lx: float = 0.20
    ly: float = 0.20
    lz: float = 0.40
    x_0: float = 0.0
    y_0: float = 0.0
    yaw_0: float = 0.0


@dataclass
class RobotPose:
    """Robot 2D spatial pose in world/map frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class DetectionProposal:
    """Candidate detection box and verification confidence score."""

    xmin: int = 0
    ymin: int = 0
    xmax: int = 0
    ymax: int = 0
    score: float = 0.0
    label: str = ""


@dataclass
class TargetVisibility:
    """Geometric 3D-to-2D frustum visibility and laser occlusion evaluation."""

    is_in_frustum: bool = False
    is_occluded: bool = False
    is_visible: bool = False
    center_depth_z: float = 0.0
    distance_xy: float = 0.0
    corners_2d: List[Tuple[int, int]] = field(default_factory=list)
    center_2d: Optional[Tuple[int, int]] = None
    bbox_2d: Optional[Tuple[int, int, int, int]] = None


class MicProbeNode(Node):
    """ROS 2 supervisor node evaluating visual pursuit trials against ground truth."""

    def __init__(self) -> None:
        """Initialize subscriptions, TF listener, state tracking, and control timers."""
        super().__init__("mic_probe_node")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.declare_parameter("trial_config_path", DEFAULT_TRIAL_CONFIG_PATH)
        self.declare_parameter("heartbeat_path", DEFAULT_HEARTBEAT_PATH)
        self.declare_parameter("frames_dir", DEFAULT_FRAMES_DIR)
        self.declare_parameter("summary_dir", DEFAULT_SUMMARY_DIR)

        self.trial_config_path = str(self.get_parameter("trial_config_path").value)
        self.heartbeat_path = str(self.get_parameter("heartbeat_path").value)
        self.frames_dir = str(self.get_parameter("frames_dir").value)
        self.summary_dir = str(self.get_parameter("summary_dir").value)

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Simulation clock and state
        self.sim_time: float = 0.0
        self.trial_start_sim_time: float = 0.0
        self.has_clock_received: bool = False
        self.last_heartbeat_time: float = 0.0

        # Current trial tracking
        self.active_trial: Optional[TrialConfig] = None
        self.is_trial_active: bool = False
        self.is_trial_completed: bool = False
        self.trial_config_mtime: float = 0.0

        # Robot kinematic state
        self.robot_pose = RobotPose()
        self.latest_twist = Twist()
        self.consecutive_stop_duration: float = 0.0
        self.stop_signal_received: bool = False

        # Camera and sensory state
        self.camera_info: Optional[CameraInfo] = None
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_stamp: float = 0.0
        self.frame_index: int = 0
        self.latest_scan: Optional[LaserScan] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self.map_raycaster = MapRaycaster(
            default_occ_threshold=50,
            robot_radius=ROBOT_RADIUS_METERS,
            target_radius=0.06,
        )

        # Detector and tracking state
        self.latest_detections: List[DetectionProposal] = []
        self.is_detector_active: bool = False
        self.detector_confidence: float = 0.0
        self.consecutive_detections: int = 0
        self.consecutive_blind_frames: int = 0
        self.blind_pass_count: int = 0
        self.blind_pass_events: List[Dict[str, Any]] = []

        # Statistical counters
        self.total_frames_evaluated: int = 0
        self.frames_camera_saw_target: int = 0
        self.frames_detector_saw_target: int = 0
        self.first_camera_sight_sim_time: Optional[float] = None
        self.first_detector_sight_sim_time: Optional[float] = None

        self._init_subscribers()

        timer_period = 1.0 / PROBE_LOOP_RATE_HZ
        self.eval_timer = self.create_timer(timer_period, self._on_evaluation_cycle)

        self.get_logger().info(
            f"MicProbe initialized. Watching trial config: {self.trial_config_path}"
        )

    def _init_subscribers(self) -> None:
        """Create all sensor, perception, and control topic subscriptions."""
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Simulation clock
        self.clock_sub = self.create_subscription(
            Clock, "/clock", self._on_clock_message, sensor_qos
        )

        # Camera streams and intrinsics
        self.cam_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_camera_info, reliable_qos
        )
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self._on_camera_image, sensor_qos
        )
        self.comp_image_sub = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self._on_compressed_camera_image,
            sensor_qos,
        )

        # Laser scans (check both unified /scan and Khepera-specific /khepera4/scan)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._on_laser_scan, sensor_qos
        )
        self.khepera_scan_sub = self.create_subscription(
            LaserScan, "/khepera4/scan", self._on_laser_scan, sensor_qos
        )

        # Occupancy grid map subscription for raycasting occlusion
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._on_map_message, reliable_qos
        )

        # Odometry fallback
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._on_odometry, sensor_qos
        )
        self.khepera_odom_sub = self.create_subscription(
            Odometry, "/khepera4/odom", self._on_odometry, sensor_qos
        )

        # Control commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._on_cmd_vel, reliable_qos
        )

        # Detection and perception status topics
        self.rois_sub = self.create_subscription(
            Detection2DArray,
            "/perception/candidate_rois",
            self._on_candidate_rois,
            reliable_qos,
        )
        self.status_sub = self.create_subscription(
            Bool, "/perception/target_status", self._on_target_status, reliable_qos
        )
        self.conf_sub = self.create_subscription(
            Float32,
            "/perception/tracking_confidence",
            self._on_tracking_confidence,
            reliable_qos,
        )

        # Stop event triggers
        self.stop_sub = self.create_subscription(
            Bool, "/target_found_and_stopped", self._on_stop_event, reliable_qos
        )
        self.pursuit_stop_sub = self.create_subscription(
            Bool, "/pursuit/target_stopped", self._on_stop_event, reliable_qos
        )

    # --------------------------------------------------------------------------
    # Subscriber Callbacks
    # --------------------------------------------------------------------------

    def _on_clock_message(self, msg: Clock) -> None:
        """Update active simulation time from clock topic."""
        self.sim_time = float(msg.clock.sec + msg.clock.nanosec * 1e-9)
        self.has_clock_received = True

    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Cache latest camera calibration and optical parameters."""
        self.camera_info = msg

    def _on_camera_image(self, msg: Image) -> None:
        """Decode incoming raw camera image to BGR numpy array."""
        self.latest_image_stamp = float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.frame_index += 1
        decoded = self._decode_image(msg)
        if decoded is not None:
            self.latest_image = decoded

    def _on_compressed_camera_image(self, msg: CompressedImage) -> None:
        """Decode incoming compressed image stream if raw is unavailable."""
        if self.latest_image is None or (self.frame_index % 2 == 0):
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.latest_image_stamp = float(
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                )
                self.frame_index += 1
            except Exception as e:
                self.get_logger().debug(f"Compressed decode error: {e}")

    def _on_laser_scan(self, msg: LaserScan) -> None:
        """Store most recent laser scan for line-of-sight occlusion verification."""
        self.latest_scan = msg

    def _on_map_message(self, msg: OccupancyGrid) -> None:
        """Cache incoming occupancy grid for map-referenced line-of-sight raycasting."""
        self.latest_map = msg

    def _on_odometry(self, msg: Odometry) -> None:
        """Update backup robot pose from odometry topic."""
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        self.robot_pose.x = float(msg.pose.pose.position.x)
        self.robot_pose.y = float(msg.pose.pose.position.y)
        self.robot_pose.yaw = float(yaw)
        self.robot_pose.timestamp = float(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Track commanded robot velocities to monitor terminal stop behavior."""
        self.latest_twist = msg

    def _on_candidate_rois(self, msg: Detection2DArray) -> None:
        """Parse predicted 2D bounding boxes and confidence scores from detector."""
        proposals: List[DetectionProposal] = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y
            score = 0.0
            label = "target"
            if det.results:
                score = float(det.results[0].hypothesis.score)
                label = str(det.results[0].hypothesis.class_id)
            xmin = int(cx - sx / 2.0)
            ymin = int(cy - sy / 2.0)
            xmax = int(cx + sx / 2.0)
            ymax = int(cy + sy / 2.0)
            proposals.append(
                DetectionProposal(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score=score, label=label
                )
            )
        self.latest_detections = proposals
        if proposals:
            self.is_detector_active = True

    def _on_target_status(self, msg: Bool) -> None:
        """Record boolean confirmation status published by verifier node."""
        if msg.data:
            self.is_detector_active = True

    def _on_tracking_confidence(self, msg: Float32) -> None:
        """Update tracker confidence metric."""
        self.detector_confidence = float(msg.data)
        if self.detector_confidence >= 0.35:
            self.is_detector_active = True

    def _on_stop_event(self, msg: Bool) -> None:
        """Handle explicit target stop event notification."""
        self.stop_signal_received = bool(msg.data)

    # --------------------------------------------------------------------------
    # Helper Routines
    # --------------------------------------------------------------------------

    def _decode_image(self, msg: Image) -> Optional[np.ndarray]:
        """Convert ROS Image message to BGR numpy array using bridge or manual buffer."""
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

        try:
            if msg.encoding in ("bgr8", "rgb8"):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
            elif msg.encoding == "mono8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().debug(f"Manual image decode failure: {e}")
        return None

    def _poll_trial_config(self) -> None:
        """Inspect and parse /tmp/mic_eval/current_trial.json when modified."""
        if not os.path.isfile(self.trial_config_path):
            return

        try:
            mtime = os.path.getmtime(self.trial_config_path)
            if self.active_trial is not None and mtime <= self.trial_config_mtime:
                return

            with open(self.trial_config_path, "r") as f:
                data = json.load(f)

            trial_id = str(data.get("trial_id", "0"))
            cfg = TrialConfig(
                trial_id=trial_id,
                x_gt=float(data.get("x_gt", 0.0)),
                y_gt=float(data.get("y_gt", 0.0)),
                z_gt=float(data.get("z_gt", 0.20)),
                lx=float(data.get("lx", 0.20)),
                ly=float(data.get("ly", 0.20)),
                lz=float(data.get("lz", 0.40)),
                x_0=float(data.get("x_0", 0.0)),
                y_0=float(data.get("y_0", 0.0)),
                yaw_0=float(data.get("yaw_0", 0.0)),
            )

            # Trigger fresh trial reset when new configuration is detected
            if self.active_trial is None or self.active_trial.trial_id != cfg.trial_id:
                self._start_new_trial(cfg, mtime)
        except Exception as e:
            self.get_logger().warn(f"Failed to read trial configuration: {e}")

    def _start_new_trial(self, cfg: TrialConfig, mtime: float) -> None:
        """Reset trial metrics and begin supervisor monitoring."""
        self.active_trial = cfg
        self.trial_config_mtime = mtime
        self.is_trial_active = True
        self.is_trial_completed = False
        self.trial_start_sim_time = self._get_current_time()
        self.consecutive_stop_duration = 0.0
        self.stop_signal_received = False

        self.consecutive_detections = 0
        self.consecutive_blind_frames = 0
        self.blind_pass_count = 0
        self.blind_pass_events = []

        self.total_frames_evaluated = 0
        self.frames_camera_saw_target = 0
        self.frames_detector_saw_target = 0
        self.first_camera_sight_sim_time = None
        self.first_detector_sight_sim_time = None

        self.get_logger().info(
            f"=== STARTING TRIAL {cfg.trial_id} === Target GT: "
            f"({cfg.x_gt:.3f}, {cfg.y_gt:.3f}, {cfg.z_gt:.3f}) size: ({cfg.lx:.2f}, "
            f"{cfg.ly:.2f}, {cfg.lz:.2f})"
        )
        self._write_heartbeat(status="RUNNING")

    def _get_current_time(self) -> float:
        """Return best available time metric (sim clock or wall clock fallback)."""
        if self.has_clock_received:
            return self.sim_time
        try:
            return float(self.get_clock().now().nanoseconds * 1e-9)
        except Exception:
            return float(time.time())

    def _update_robot_pose_from_tf(self) -> None:
        """Query TF2 buffer to update ground truth robot pose in world coordinates."""
        for world_frame in ("map", "world", "odom"):
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    world_frame,
                    "base_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(nanoseconds=10000000),
                )
                qx = tf_stamped.transform.rotation.x
                qy = tf_stamped.transform.rotation.y
                qz = tf_stamped.transform.rotation.z
                qw = tf_stamped.transform.rotation.w
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                self.robot_pose.x = float(tf_stamped.transform.translation.x)
                self.robot_pose.y = float(tf_stamped.transform.translation.y)
                self.robot_pose.yaw = float(yaw)
                self.robot_pose.timestamp = float(
                    tf_stamped.header.stamp.sec + tf_stamped.header.stamp.nanosec * 1e-9
                )
                return
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue

    # --------------------------------------------------------------------------
    # Frustum Visibility and Occlusion Computation
    # --------------------------------------------------------------------------

    def _calculate_target_visibility(self) -> TargetVisibility:
        """Compute 3D-to-2D frustum projection and verify line-of-sight against laser scan."""
        result = TargetVisibility()
        if self.active_trial is None:
            return result

        cfg = self.active_trial
        dx = cfg.x_gt - self.robot_pose.x
        dy = cfg.y_gt - self.robot_pose.y
        result.distance_xy = float(math.hypot(dx, dy))

        # Resolve camera optical frame transform
        optical_frame = "camera_optical_frame"
        world_frame_used = None
        tf_optical = None

        for w_cand in ("map", "world", "odom"):
            try:
                tf_optical = self.tf_buffer.lookup_transform(
                    optical_frame,
                    w_cand,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(nanoseconds=10000000),
                )
                world_frame_used = w_cand
                break
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue

        if tf_optical is None or world_frame_used is None:
            return result

        # Compute 8 3D bounding box corners and center point in world frame
        hx, hy, hz = cfg.lx / 2.0, cfg.ly / 2.0, cfg.lz / 2.0
        corner_offsets = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        world_corners = [
            (cfg.x_gt + ox, cfg.y_gt + oy, cfg.z_gt + oz) for ox, oy, oz in corner_offsets
        ]

        # Camera intrinsics
        width, height = DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
        fx, fy, cx, cy = DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY
        if self.camera_info is not None:
            width = self.camera_info.width or width
            height = self.camera_info.height or height
            k = self.camera_info.k
            if len(k) == 9 and k[0] > 0.0:
                fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

        # Project center point
        center_pt = PointStamped()
        center_pt.header.frame_id = world_frame_used
        center_pt.point.x = cfg.x_gt
        center_pt.point.y = cfg.y_gt
        center_pt.point.z = cfg.z_gt
        cam_center = tf2_geometry_msgs.do_transform_point(center_pt, tf_optical)
        result.center_depth_z = float(cam_center.point.z)

        center_in_view = False
        if cam_center.point.z > 0.05:
            cu = int(round(fx * (cam_center.point.x / cam_center.point.z) + cx))
            cv = int(round(fy * (cam_center.point.y / cam_center.point.z) + cy))
            result.center_2d = (cu, cv)
            if 0 <= cu < width and 0 <= cv < height:
                center_in_view = True

        # Project 8 corners
        projected_corners: List[Tuple[int, int]] = []
        corners_in_view = 0
        min_u, max_u = width, 0
        min_v, max_v = height, 0

        for wx, wy, wz in world_corners:
            pt = PointStamped()
            pt.header.frame_id = world_frame_used
            pt.point.x = wx
            pt.point.y = wy
            pt.point.z = wz
            pt_cam = tf2_geometry_msgs.do_transform_point(pt, tf_optical)

            if pt_cam.point.z > 0.05:
                u = int(round(fx * (pt_cam.point.x / pt_cam.point.z) + cx))
                v = int(round(fy * (pt_cam.point.y / pt_cam.point.z) + cy))
                projected_corners.append((u, v))
                if 0 <= u < width and 0 <= v < height:
                    corners_in_view += 1
                min_u = min(min_u, max(0, u))
                max_u = max(max_u, min(width - 1, u))
                min_v = min(min_v, max(0, v))
                max_v = max(max_v, min(height - 1, v))
            else:
                projected_corners.append((-100, -100))

        result.corners_2d = projected_corners
        if max_u > min_u and max_v > min_v:
            result.bbox_2d = (min_u, min_v, max_u, max_v)

        # Frustum criteria: target in front, reasonable range, center or corners visible
        in_range = 0.10 <= result.center_depth_z <= MAX_VIEW_DISTANCE_METERS
        if in_range and (center_in_view or corners_in_view >= 2):
            result.is_in_frustum = True

        # Check line-of-sight occlusion using 360 laser scan
        result.is_occluded = self._is_line_of_sight_occluded(cfg, result.distance_xy)
        result.is_visible = result.is_in_frustum and not result.is_occluded

        return result

    def _is_line_of_sight_occluded(self, cfg: TrialConfig, distance_to_target: float) -> bool:
        """Verify whether an obstacle or wall occludes the line of sight to the target."""
        # 1. 2D Map-Referenced Bresenham Raycasting Check (Primary)
        if self.latest_map is not None:
            try:
                self.map_raycaster.target_radius = max(0.12, max(cfg.lx, cfg.ly) / 2.0 + 0.02)
                is_map_blocked = self.map_raycaster.is_line_of_sight_occluded(
                    grid=self.latest_map,
                    start_x=self.robot_pose.x,
                    start_y=self.robot_pose.y,
                    target_x=cfg.x_gt,
                    target_y=cfg.y_gt,
                    occ_threshold=50,
                )
                if is_map_blocked:
                    return True
            except Exception as e:
                self.get_logger().debug(f"Map raycaster evaluation error: {e}")

        # 2. 360-degree Laser Scan Verification (Secondary / Fallback)
        if self.latest_scan is None or not self.latest_scan.ranges:
            return False

        scan = self.latest_scan
        # Compute relative bearing angle in robot base frame
        dx = cfg.x_gt - self.robot_pose.x
        dy = cfg.y_gt - self.robot_pose.y
        target_bearing = math.atan2(dy, dx) - self.robot_pose.yaw
        target_bearing = (target_bearing + math.pi) % (2.0 * math.pi) - math.pi

        # Angular span of the target geometry
        target_half_extent = max(cfg.lx, cfg.ly) / 2.0
        angle_half_width = max(
            0.06, math.atan2(target_half_extent, max(0.1, distance_to_target))
        )

        min_angle = target_bearing - angle_half_width
        max_angle = target_bearing + angle_half_width

        valid_rays = 0
        blocked_rays = 0
        threshold_dist = max(0.15, distance_to_target - target_half_extent - OCCLUSION_MARGIN_METERS)

        curr_angle = scan.angle_min
        for r in scan.ranges:
            if min_angle <= curr_angle <= max_angle:
                if scan.range_min <= r <= scan.range_max:
                    valid_rays += 1
                    if r < threshold_dist:
                        blocked_rays += 1
            curr_angle += scan.angle_increment

        # If a significant majority of valid returns are closer than target, line of sight is blocked
        if valid_rays >= 3 and (blocked_rays / valid_rays) >= 0.70:
            return True

        return False

    # --------------------------------------------------------------------------
    # Main Evaluation Loop
    # --------------------------------------------------------------------------

    def _on_evaluation_cycle(self) -> None:
        """Perform supervisory audit, check stopping/timeout, and emit telemetry."""
        self._poll_trial_config()

        now_sec = self._get_current_time()
        self._update_robot_pose_from_tf()

        # Emit periodic heartbeat even while waiting
        if (now_sec - self.last_heartbeat_time) >= HEARTBEAT_INTERVAL_SEC:
            status_str = "RUNNING" if self.is_trial_active else "IDLE"
            self._write_heartbeat(status=status_str)

        if not self.is_trial_active or self.is_trial_completed or self.active_trial is None:
            return

        elapsed_sim = now_sec - self.trial_start_sim_time
        visibility = self._calculate_target_visibility()

        detector_sees = bool(self.is_detector_active or len(self.latest_detections) > 0)
        camera_sees = visibility.is_visible

        self.total_frames_evaluated += 1
        if camera_sees:
            self.frames_camera_saw_target += 1
            if self.first_camera_sight_sim_time is None:
                self.first_camera_sight_sim_time = elapsed_sim

        if detector_sees:
            self.frames_detector_saw_target += 1
            self.consecutive_detections += 1
            if self.first_detector_sight_sim_time is None:
                self.first_detector_sight_sim_time = elapsed_sim
        else:
            self.consecutive_detections = 0

        # Audit "Sees vs Detects" for Blind Pass Events
        if camera_sees and not detector_sees:
            self.consecutive_blind_frames += 1
            if self.consecutive_blind_frames >= MIN_CONSECUTIVE_BLIND_FRAMES:
                self._handle_blind_pass_event(visibility, elapsed_sim)
        else:
            self.consecutive_blind_frames = 0

        # Check terminal stop conditions
        is_stopped = self._check_terminal_stop(visibility.distance_xy)

        # Check timeout condition (120 seconds simulated)
        if elapsed_sim >= TRIAL_TIMEOUT_SEC:
            self.get_logger().error(f"Trial {self.active_trial.trial_id} TIMEOUT at {elapsed_sim:.1f}s")
            self._finalize_trial(status="FAIL: TIMEOUT_120S", visibility=visibility)
            return

        if is_stopped:
            standoff = self._calculate_standoff_distance()
            self.get_logger().info(
                f"Trial {self.active_trial.trial_id} STOPPED! Standoff distance: {standoff:.3f}m"
            )
            self._finalize_trial(status="SUCCESS", visibility=visibility)
            return

        # Reset per-frame detector flag
        self.is_detector_active = False

    def _check_terminal_stop(self, distance_to_target: float) -> bool:
        """Determine if robot has reached verified terminal halt near target."""
        if self.stop_signal_received and distance_to_target <= STOP_MAX_DISTANCE_TO_TARGET:
            return True

        v_lin = abs(self.latest_twist.linear.x)
        v_ang = abs(self.latest_twist.angular.z)

        # Confirm stationary kinematic velocity
        is_stationary = (
            v_lin <= STOP_VELOCITY_THRESH_LIN and v_ang <= STOP_VELOCITY_THRESH_ANG
        )

        # Check proximity to target
        is_near_target = distance_to_target <= STOP_MAX_DISTANCE_TO_TARGET

        dt = 1.0 / PROBE_LOOP_RATE_HZ
        if is_stationary and is_near_target:
            self.consecutive_stop_duration += dt
            if self.consecutive_stop_duration >= STOP_CONFIRMATION_TIME_SEC:
                return True
        else:
            self.consecutive_stop_duration = 0.0

        return False

    def _calculate_standoff_distance(self) -> float:
        """Compute distance from robot front surface (radius 7cm) to target front surface."""
        if self.active_trial is None:
            return 0.0

        cfg = self.active_trial
        dx = cfg.x_gt - self.robot_pose.x
        dy = cfg.y_gt - self.robot_pose.y
        center_dist = math.hypot(dx, dy)

        if center_dist < 1e-4:
            return 0.0

        # Normal vector from target center toward robot
        ux = -dx / center_dist
        uy = -dy / center_dist

        hx = cfg.lx / 2.0
        hy = cfg.ly / 2.0
        # Distance from target center to its facing boundary
        dist_target_surface = min(hx / (abs(ux) + 1e-6), hy / (abs(uy) + 1e-6))

        standoff = center_dist - ROBOT_RADIUS_METERS - dist_target_surface
        return float(round(standoff, 4))

    # --------------------------------------------------------------------------
    # Event Logging and Artifact Emission
    # --------------------------------------------------------------------------

    def _handle_blind_pass_event(
        self, visibility: TargetVisibility, elapsed_sim: float
    ) -> None:
        """Log BLIND_PASS_EVENT and persist annotated camera snapshot."""
        self.blind_pass_count += 1
        trial_id = self.active_trial.trial_id if self.active_trial else "0"
        frame_id = self.frame_index

        self.get_logger().warn(
            f"BLIND_PASS_EVENT [Trial {trial_id}, Frame {frame_id}]: Camera sees target "
            f"at distance {visibility.distance_xy:.2f}m, but detector missed for "
            f"{self.consecutive_blind_frames} frames!"
        )

        event_record = {
            "trial_id": trial_id,
            "frame_index": frame_id,
            "sim_time": elapsed_sim,
            "distance_xy": visibility.distance_xy,
            "consecutive_blind_frames": self.consecutive_blind_frames,
        }
        self.blind_pass_events.append(event_record)

        img_annotated = self._render_annotated_frame(
            visibility=visibility,
            header_text=f"BLIND PASS EVENT - Frame {frame_id} (SimTime {elapsed_sim:.1f}s)",
        )
        save_path = os.path.join(self.frames_dir, f"trial_{trial_id}_blind_{frame_id}.png")
        if img_annotated is not None:
            cv2.imwrite(save_path, img_annotated)
            event_record["image_path"] = save_path

    def _render_annotated_frame(
        self, visibility: TargetVisibility, header_text: str
    ) -> Optional[np.ndarray]:
        """Render frame with green projected GT 3D/2D bounding box and red detections."""
        if self.latest_image is not None:
            img = self.latest_image.copy()
        else:
            img = np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, 3), dtype=np.uint8)

        # Draw green projected 3D wireframe edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        corners = visibility.corners_2d
        if len(corners) == 8:
            for i, j in edges:
                u1, v1 = corners[i]
                u2, v2 = corners[j]
                if u1 >= 0 and v1 >= 0 and u2 >= 0 and v2 >= 0:
                    cv2.line(img, (u1, v1), (u2, v2), (0, 255, 0), 2)

        # Draw green 2D projected bounding box
        if visibility.bbox_2d is not None:
            u1, v1, u2, v2 = visibility.bbox_2d
            cv2.rectangle(img, (u1, v1), (u2, v2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"GT Target ({visibility.distance_xy:.2f}m)",
                (u1, max(15, v1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw red predicted boxes from detector
        for det in self.latest_detections:
            cv2.rectangle(
                img,
                (det.xmin, det.ymin),
                (det.xmax, det.ymax),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img,
                f"Pred: {det.label} {det.score:.2f}",
                (det.xmin, max(15, det.ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # Header banner
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(
            img,
            header_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return img

    def _write_heartbeat(self, status: str) -> None:
        """Write atomic status telemetry to /tmp/mic_eval/heartbeat.json."""
        self.last_heartbeat_time = self._get_current_time()
        trial_id = self.active_trial.trial_id if self.active_trial else "NONE"
        dist = 0.0
        if self.active_trial:
            dist = math.hypot(
                self.active_trial.x_gt - self.robot_pose.x,
                self.active_trial.y_gt - self.robot_pose.y,
            )

        payload = {
            "timestamp_wall": time.time(),
            "sim_time": self.sim_time,
            "trial_id": trial_id,
            "status": status,
            "robot_pose": {
                "x": round(self.robot_pose.x, 3),
                "y": round(self.robot_pose.y, 3),
                "yaw": round(self.robot_pose.yaw, 3),
            },
            "distance_to_target": round(dist, 3),
            "consecutive_detections": self.consecutive_detections,
            "consecutive_blind_frames": self.consecutive_blind_frames,
            "blind_pass_count": self.blind_pass_count,
            "total_frames_evaluated": self.total_frames_evaluated,
        }

        tmp_path = f"{self.heartbeat_path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.heartbeat_path)
        except Exception as e:
            self.get_logger().debug(f"Failed writing heartbeat: {e}")

    def _finalize_trial(self, status: str, visibility: TargetVisibility) -> None:
        """Emit final annotated frame, write trial summary JSON, and terminate trial."""
        self.is_trial_completed = True
        self.is_trial_active = False

        if self.active_trial is None:
            return

        cfg = self.active_trial
        elapsed = self._get_current_time() - self.trial_start_sim_time
        standoff = self._calculate_standoff_distance()

        # Save final annotated frame
        final_img = self._render_annotated_frame(
            visibility=visibility,
            header_text=f"Trial {cfg.trial_id} [{status}] - Standoff {standoff:.3f}m - {elapsed:.1f}s",
        )
        final_frame_path = os.path.join(self.frames_dir, f"trial_{cfg.trial_id}_final.png")
        if final_img is not None:
            cv2.imwrite(final_frame_path, final_img)

        # Build summary JSON
        summary = {
            "trial_id": cfg.trial_id,
            "status": status,
            "success": status == "SUCCESS",
            "elapsed_sim_time": round(elapsed, 2),
            "final_standoff_distance": standoff,
            "center_to_center_distance": round(visibility.distance_xy, 3),
            "robot_radius_meters": ROBOT_RADIUS_METERS,
            "target_gt": {
                "x": cfg.x_gt,
                "y": cfg.y_gt,
                "z": cfg.z_gt,
                "lx": cfg.lx,
                "ly": cfg.ly,
                "lz": cfg.lz,
            },
            "robot_initial_pose": {"x": cfg.x_0, "y": cfg.y_0, "yaw": cfg.yaw_0},
            "robot_final_pose": {
                "x": round(self.robot_pose.x, 3),
                "y": round(self.robot_pose.y, 3),
                "yaw": round(self.robot_pose.yaw, 3),
            },
            "metrics": {
                "total_frames_evaluated": self.total_frames_evaluated,
                "frames_camera_saw_target": self.frames_camera_saw_target,
                "frames_detector_saw_target": self.frames_detector_saw_target,
                "blind_pass_count": self.blind_pass_count,
                "first_camera_sight_sim_time": self.first_camera_sight_sim_time,
                "first_detector_sight_sim_time": self.first_detector_sight_sim_time,
            },
            "blind_pass_events": self.blind_pass_events,
            "final_frame_image": final_frame_path,
        }

        summary_file = os.path.join(self.summary_dir, f"trial_{cfg.trial_id}_summary.json")
        tmp_summary = f"{summary_file}.tmp"
        try:
            with open(tmp_summary, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_summary, summary_file)
            self.get_logger().info(f"Wrote trial summary to: {summary_file}")
        except Exception as e:
            self.get_logger().error(f"Failed writing trial summary: {e}")

        self._write_heartbeat(status=status)


def main(args: Optional[List[str]] = None) -> None:
    """Run MicProbeNode with standard ROS 2 lifecycle ordering."""
    rclpy.init(args=args)
    node = MicProbeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main(sys.argv)
