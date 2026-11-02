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
Visual Orthomapper Node for Khepera IV Mobile Robot.

Features:
1. Real-time Ground-Plane Inverse Perspective Mapping (IPM) using closed-form
   forward and inverse homographies between image and ground plane (Z=0).
2. Synchronization of camera image and odometry (prioritizing /khepera4/odom
   with automatic fallback to /odom).
3. Inverse grid pull-warping and incremental alpha-blending (alpha=0.25) into a
   continuous 2D visual grid matching the SLAM grid (0.03 m/cell, 200x200).
4. Physical tilt guard (suspending warping if |pitch| > 6.0 deg) and horizon
   cutoff (v in [251, 479], ground distance 0.1338 m <= X_b <= 1.50 m).
5. Publishing /visual_map (OccupancyGrid) and /visual_map/image (RGBA Image)
   with TRANSIENT_LOCAL durability.
6. Spatial inspection keyframe sampling and manifest serialization.
7. Atomic memory and grid reset upon /clear_map.
"""

from datetime import datetime, timezone
import json
import math
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from khepera4_control.common import euler_from_quaternion, normalize_angle
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
from PIL import Image, ImageDraw
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image as RosImage
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class KeyframeManager:
    """
    Manager for spatial camera keyframe sampling, storage, and cataloging.

    Matches survey_specs.md §2 and tracks robot displacement, orientation
    delta, and hazard events.
    """

    def __init__(
        self,
        keyframe_dir: str = 'maps/keyframes',
        map_frame: str = 'map',
        camera_xc: float = 0.0682,
        camera_h: float = 0.0283,
        image_width: int = 640,
        image_height: int = 480,
        hfov_deg: float = 60.0,
        focal_length: float = 554.256258,
        optical_frame: str = 'camera_optical_frame'
    ):
        """Initialize KeyframeManager with storage paths and camera parameters."""
        self.keyframe_dir = keyframe_dir
        self.manifest_path = os.path.join(self.keyframe_dir, 'manifest.json')
        self.map_frame = map_frame

        self.camera_xc = camera_xc
        self.camera_h = camera_h
        self.image_width = image_width
        self.image_height = image_height
        self.hfov_rad = math.radians(hfov_deg)
        self.fx = focal_length
        self.fy = focal_length
        self.camera_optical_frame = optical_frame

        self.keyframes: List[Dict] = []
        self.last_kf_x: Optional[float] = None
        self.last_kf_y: Optional[float] = None
        self.last_kf_yaw: Optional[float] = None

        self.last_hazard_time = 0.0
        self.last_hazard_x = 0.0
        self.last_hazard_y = 0.0

        self.next_id = 1
        self._lock = threading.RLock()
        self._init_manifest()

    def _init_manifest(self):
        """Initialize empty manifest.json catalog."""
        with self._lock:
            self.keyframes.clear()
            self._save_manifest()

    def should_trigger_displacement(
        self, x: float, y: float, threshold: float = 0.35
    ) -> bool:
        """Evaluate if translation distance exceeds threshold."""
        with self._lock:
            if self.last_kf_x is None or self.last_kf_y is None:
                return True
            dist = math.hypot(x - self.last_kf_x, y - self.last_kf_y)
            return dist >= threshold

    def should_trigger_orientation(
        self, yaw: float, threshold_deg: float = 35.0
    ) -> bool:
        """Evaluate if heading change exceeds threshold degrees."""
        with self._lock:
            if self.last_kf_yaw is None:
                return True
            dyaw = abs(normalize_angle(yaw - self.last_kf_yaw))
            return dyaw >= math.radians(threshold_deg)

    def should_trigger_cliff(
        self,
        x: float,
        y: float,
        current_time: float,
        cooldown_sec: float = 1.0,
        min_dist: float = 0.05
    ) -> bool:
        """Evaluate if cliff hazard qualifies after debouncing cooldown."""
        with self._lock:
            if self.last_hazard_time <= 0.0:
                return True
            dt = current_time - self.last_hazard_time
            dist = math.hypot(x - self.last_hazard_x, y - self.last_hazard_y)
            return (dt >= cooldown_sec) and (dist >= min_dist)

    def record_keyframe(
        self,
        image_rgb: np.ndarray,
        x: float,
        y: float,
        yaw: float,
        trigger_type: str,
        current_time: Optional[float] = None,
        linear_vel: float = 0.0,
        angular_vel: float = 0.0,
        cliff_alert: bool = False
    ) -> dict:
        """Capture and serialize keyframe photo and metadata entry."""
        if current_time is None:
            current_time = time.time()

        with self._lock:
            kf_id = self.next_id
            self.next_id += 1

            filename = f'keyframe_{kf_id:04d}.png'
            img_path = os.path.join(self.keyframe_dir, filename)
            rel_path = os.path.join(self.keyframe_dir, filename)

            os.makedirs(self.keyframe_dir, exist_ok=True)
            pil_img = Image.fromarray(image_rgb)
            pil_img.save(img_path, format='PNG')

            entry = {
                'id': kf_id,
                'filename': filename,
                'relative_path': rel_path,
                'image_file': rel_path,
                'timestamp': round(current_time, 3),
                'datetime': datetime.fromtimestamp(
                    current_time, tz=timezone.utc
                ).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'pose': {
                    'x': round(x, 4),
                    'y': round(y, 4),
                    'z': 0.0,
                    'yaw': round(yaw, 4),
                    'yaw_deg': round(math.degrees(yaw), 2),
                    'pitch_deg': 0.0,
                    'roll_deg': 0.0
                },
                'robot_pose': {
                    'x': round(x, 4),
                    'y': round(y, 4),
                    'z': 0.0,
                    'yaw': round(yaw, 4)
                },
                'camera_pose': {
                    'x': round(x + self.camera_xc * math.cos(yaw), 4),
                    'y': round(y + self.camera_xc * math.sin(yaw), 4),
                    'z': round(self.camera_h, 4),
                    'yaw': round(yaw, 4)
                },
                'trigger': trigger_type,
                'trigger_source': trigger_type,
                'telemetry': {
                    'linear_velocity': round(linear_vel, 3),
                    'angular_velocity': round(angular_vel, 3),
                    'cliff_alert': bool(cliff_alert)
                },
                'camera_metadata': {
                    'width': int(self.image_width),
                    'height': int(self.image_height),
                    'hfov_deg': round(math.degrees(self.hfov_rad), 1),
                    'focal_length_px': round(self.fx, 3),
                    'optical_frame': str(self.camera_optical_frame)
                }
            }

            self.keyframes.append(entry)
            self.last_kf_x = x
            self.last_kf_y = y
            self.last_kf_yaw = yaw

            if trigger_type == 'cliff_hazard':
                self.last_hazard_time = current_time
                self.last_hazard_x = x
                self.last_hazard_y = y

            self._save_manifest()
            return entry

    def _save_manifest(self):
        """Write manifest.json atomically via temp file replace."""
        with self._lock:
            manifest_data = {
                'map_frame': self.map_frame,
                'manifest_version': '1.0',
                'total_keyframes': len(self.keyframes),
                'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'keyframes': list(self.keyframes)
            }
            os.makedirs(self.keyframe_dir, exist_ok=True)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.keyframe_dir,
                    prefix='manifest_',
                    suffix='.tmp',
                    delete=False,
                    mode='w',
                    encoding='utf-8'
                ) as f:
                    tmp_path = f.name
                    json.dump(manifest_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.manifest_path)
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

    def clear(self):
        """Purge keyframe images and reset manifest catalog."""
        with self._lock:
            self.keyframes.clear()
            self.last_kf_x = None
            self.last_kf_y = None
            self.last_kf_yaw = None
            self.last_hazard_time = 0.0
            self.last_hazard_x = 0.0
            self.last_hazard_y = 0.0
            self.next_id = 1

            if os.path.exists(self.keyframe_dir):
                for fname in os.listdir(self.keyframe_dir):
                    if fname.endswith('.png') or fname.endswith('.tmp'):
                        try:
                            os.remove(os.path.join(self.keyframe_dir, fname))
                        except OSError:
                            pass
            self._save_manifest()


def generate_composite_map(
    output_path: str,
    visual_grid: np.ndarray,
    log_odds_grid: np.ndarray,
    elevation_grid: np.ndarray,
    surface_grid: np.ndarray,
    robot_path: list,
    keyframes: list,
    res: float = 0.03,
    origin_x: float = -3.0,
    origin_y: float = -3.0,
    current_pose: Optional[Tuple[float, float, float]] = None
) -> str:
    """
    Generate multi-layer composite map matching survey_specs.md §4.

    Blends ground visual orthomap, SLAM obstacles, terrain slopes, cliff hazards,
    trajectory polyline, and inspection keyframe landmark pins.
    """
    h, w = visual_grid.shape[:2]
    # Layer 1: Base visual map (RGB)
    if visual_grid.ndim == 3 and visual_grid.shape[2] >= 3:
        base_rgb = visual_grid[:, :, :3].copy().astype(np.uint8)
    elif visual_grid.ndim == 2:
        base_rgb = np.repeat(visual_grid[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    else:
        base_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

    # Layer 2: SLAM obstacles (dark charcoal)
    obs_mask = (log_odds_grid > 0.5)
    base_rgb[obs_mask] = [20, 20, 25]

    # Layer 3: Slopes > 5 deg (amber tint, alpha blend)
    slope_mask = (elevation_grid > 5) & (~obs_mask)
    if np.any(slope_mask):
        base_rgb[slope_mask] = (
            0.55 * base_rgb[slope_mask] + 0.45 * np.array([255, 175, 50])
        ).astype(np.uint8)

    # Layer 4: Cliff hazard cells (hazard red)
    cliff_mask = (surface_grid == 100)
    base_rgb[cliff_mask] = [230, 25, 25]

    # Convert to PIL Image (flip vertically for standard map orientation)
    img_flipped = np.flipud(base_rgb)
    pil_img = Image.fromarray(img_flipped)
    draw = ImageDraw.Draw(pil_img)

    # Layer 5: Robot trajectory polyline
    if robot_path and len(robot_path) >= 2:
        pts = []
        for wx, wy in robot_path:
            c = int((wx - origin_x) / res)
            r = int((wy - origin_y) / res)
            if 0 <= c < w and 0 <= r < h:
                pts.append((c, h - 1 - r))
        if len(pts) >= 2:
            draw.line(pts, fill=(30, 140, 255), width=2)

    # Layer 6: Visual Inspection Landmark Pins
    for kf in keyframes:
        pose = kf.get('pose', {})
        kx = pose.get('x', 0.0)
        ky = pose.get('y', 0.0)
        c = int((kx - origin_x) / res)
        r = int((ky - origin_y) / res)
        if 0 <= c < w and 0 <= r < h:
            y_img = h - 1 - r
            draw.rectangle(
                [c - 3, y_img - 3, c + 3, y_img + 3],
                fill=(245, 158, 11),
                outline=(255, 255, 255)
            )
            kf_id = kf.get('id', '')
            draw.text((c + 5, y_img - 5), f'KF #{kf_id}', fill=(254, 240, 138))

    # Optional Layer 7: Current robot footprint
    if current_pose is not None:
        rx, ry, ryaw = current_pose
        cur_c = int((rx - origin_x) / res)
        cur_r = int((ry - origin_y) / res)
        if 0 <= cur_c < w and 0 <= cur_r < h:
            cur_y_img = h - 1 - cur_r
            r_rad = 3
            draw.ellipse(
                [cur_c - r_rad, cur_y_img - r_rad, cur_c + r_rad, cur_y_img + r_rad],
                fill=(0, 220, 80),
                outline=(0, 100, 40)
            )
            hx = cur_c + int(6 * math.cos(ryaw))
            hy = cur_y_img - int(6 * math.sin(ryaw))
            draw.line([(cur_c, cur_y_img), (hx, hy)], fill=(0, 100, 40), width=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pil_img.save(output_path, format='PNG')
    return output_path


class VisualOrthomapper(Node):
    """
    Dedicated node for Ground-Plane Inverse Perspective Mapping (IPM).

    Maintains a continuous 2D visual orthomosaic map matching the SLAM grid,
    and supports spatial keyframe inspection capture and persistence.
    """

    def __init__(self, **kwargs):
        """Initialize the VisualOrthomapper node and its parameters."""
        super().__init__('visual_orthomapper', **kwargs)

        # ----------------------------------------------------------------------
        # Parameters: Grid & World Geometry
        # ----------------------------------------------------------------------
        self.declare_parameter('map_resolution', 0.03)      # 3 cm per cell
        self.declare_parameter('map_width', 200)            # 200 cells = 6.0 m
        self.declare_parameter('map_height', 200)           # 200 cells = 6.0 m
        self.declare_parameter('map_origin_x', -3.0)        # Grid origin X (m)
        self.declare_parameter('map_origin_y', -3.0)        # Grid origin Y (m)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter(
            'camera_optical_frame', 'camera_optical_frame'
        )

        # ----------------------------------------------------------------------
        # Parameters: Topics & QoS
        # ----------------------------------------------------------------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('odom_topic', '/khepera4/odom')
        self.declare_parameter('raw_odom_topic', '/odom')
        self.declare_parameter('cliff_topic', '/cliff_alert')
        self.declare_parameter('visual_map_topic', '/visual_map')
        self.declare_parameter('visual_map_image_topic', '/visual_map/image')
        self.declare_parameter('map_publish_rate', 3.0)     # Hz
        self.declare_parameter('clear_service_name', '/clear_visual_map')

        # ----------------------------------------------------------------------
        # Parameters: Camera Intrinsics & Extrinsics
        # ----------------------------------------------------------------------
        # Camera mounted at (x_c=0.0682, y_c=0.0, z_c=0.0283) on base link
        self.declare_parameter('camera_height', 0.0283)     # h = z_c in meters
        self.declare_parameter('camera_x_offset', 0.0682)   # x_c in meters
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('hfov_deg', 60.0)
        self.declare_parameter('focal_length', 554.256258)
        self.declare_parameter('principal_point_x', 320.0)
        self.declare_parameter('principal_point_y', 240.0)

        # ----------------------------------------------------------------------
        # Parameters: IPM Homography Range & Pitch Guard
        # ----------------------------------------------------------------------
        self.declare_parameter('min_ground_dist', 0.1338)    # Closest floor (m)
        self.declare_parameter('max_ground_dist', 1.50)      # Far clipping (m)
        self.declare_parameter('min_row_v', 251)             # Horizon cutoff
        self.declare_parameter('max_row_v', 479)             # Bottom image row
        self.declare_parameter('pitch_guard_deg', 6.0)       # Tilt suspension
        self.declare_parameter('alpha', 0.25)                # Alpha-blending

        # ----------------------------------------------------------------------
        # Parameters: Arena Boundaries
        # ----------------------------------------------------------------------
        self.declare_parameter('arena_min_x', -2.405)
        self.declare_parameter('arena_max_x', 2.405)
        self.declare_parameter('arena_min_y', -2.405)
        self.declare_parameter('arena_max_y', 2.405)

        # ----------------------------------------------------------------------
        # Parameters: Keyframe Inspection & Persistence
        # ----------------------------------------------------------------------
        self.declare_parameter('keyframe_dir', 'maps/keyframes')
        self.declare_parameter('enable_auto_keyframes', True)
        self.declare_parameter('kf_displacement_thresh', 0.35)     # >= 0.35m
        self.declare_parameter('kf_orientation_thresh_deg', 35.0)  # >= 35 deg
        self.declare_parameter('kf_cliff_cooldown', 1.0)           # 1.0s deb
        self.declare_parameter('kf_cliff_dist_thresh', 0.05)       # 0.05m deb

        # Extract parameters into local variables
        self.res = float(self.get_parameter('map_resolution').value)
        self.width = int(self.get_parameter('map_width').value)
        self.height = int(self.get_parameter('map_height').value)
        self.origin_x = float(self.get_parameter('map_origin_x').value)
        self.origin_y = float(self.get_parameter('map_origin_y').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.camera_optical_frame = str(
            self.get_parameter('camera_optical_frame').value
        )

        self.camera_h = float(self.get_parameter('camera_height').value)
        self.camera_xc = float(self.get_parameter('camera_x_offset').value)
        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.hfov_rad = math.radians(
            float(self.get_parameter('hfov_deg').value)
        )
        self.fx = float(self.get_parameter('focal_length').value)
        self.fy = self.fx
        self.cx = float(self.get_parameter('principal_point_x').value)
        self.cy = float(self.get_parameter('principal_point_y').value)
        self.has_camera_info = False

        self.min_ground_dist = float(
            self.get_parameter('min_ground_dist').value
        )
        self.max_ground_dist = float(
            self.get_parameter('max_ground_dist').value
        )
        self.min_row_v = int(self.get_parameter('min_row_v').value)
        self.max_row_v = int(self.get_parameter('max_row_v').value)
        self.pitch_guard_deg = float(
            self.get_parameter('pitch_guard_deg').value
        )
        self.alpha = float(self.get_parameter('alpha').value)

        self.arena_min_x = float(self.get_parameter('arena_min_x').value)
        self.arena_max_x = float(self.get_parameter('arena_max_x').value)
        self.arena_min_y = float(self.get_parameter('arena_min_y').value)
        self.arena_max_y = float(self.get_parameter('arena_max_y').value)

        self.keyframe_dir = str(self.get_parameter('keyframe_dir').value)
        self.enable_auto_keyframes = bool(
            self.get_parameter('enable_auto_keyframes').value
        )
        self.kf_disp_thresh = float(
            self.get_parameter('kf_displacement_thresh').value
        )
        self.kf_yaw_thresh_rad = math.radians(
            float(self.get_parameter('kf_orientation_thresh_deg').value)
        )
        self.kf_cliff_cooldown = float(
            self.get_parameter('kf_cliff_cooldown').value
        )
        self.kf_cliff_dist_thresh = float(
            self.get_parameter('kf_cliff_dist_thresh').value
        )

        # ----------------------------------------------------------------------
        # State Variables: Odometry, IMU & Telemetry
        # ----------------------------------------------------------------------
        self.has_odom = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_roll = 0.0
        self.robot_pitch = 0.0
        self.robot_yaw = 0.0
        self.robot_linear_vel = 0.0
        self.robot_angular_vel = 0.0
        self.last_khepera4_odom_time = None
        self.last_odom_stamp = None

        # Keyframe manager instance
        self.keyframe_manager = KeyframeManager(
            keyframe_dir=self.keyframe_dir,
            map_frame=self.map_frame,
            camera_xc=self.camera_xc,
            camera_h=self.camera_h,
            image_width=self.image_width,
            image_height=self.image_height,
            hfov_deg=float(self.get_parameter('hfov_deg').value),
            focal_length=self.fx,
            optical_frame=self.camera_optical_frame
        )
        self.manifest_path = self.keyframe_manager.manifest_path

        self.cliff_alert = False
        self.last_hazard_time = 0.0
        self.last_hazard_x = 0.0
        self.last_hazard_y = 0.0

        # Latest camera frame for keyframe sampling
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.latest_camera_stamp = None

        # ----------------------------------------------------------------------
        # Visual Map Memory
        # ----------------------------------------------------------------------
        # 2D visual color grid (H, W, 3) stored in float32 RGB [0, 255]
        self.color_map = np.zeros(
            (self.height, self.width, 3), dtype=np.float32
        )
        # Observation mask: True if cell has been observed
        self.observed_mask = np.zeros((self.height, self.width), dtype=bool)
        # Weight map for confidence accumulation
        self.weight_map = np.zeros(
            (self.height, self.width), dtype=np.float32
        )

        # Compute initial analytical homographies
        self.H_b2img = self.compute_forward_homography()
        self.H_img2b = self.compute_inverse_homography()

        # ----------------------------------------------------------------------
        # Publishers: Latched TRANSIENT_LOCAL for RViz & Dashboard
        # ----------------------------------------------------------------------
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.visual_map_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('visual_map_topic').value),
            qos_latched
        )
        self.visual_map_image_pub = self.create_publisher(
            RosImage,
            str(self.get_parameter('visual_map_image_topic').value),
            qos_latched
        )

        # ----------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------------
        # 1. Image feed from onboard CMOS camera
        self.image_sub = self.create_subscription(
            RosImage,
            str(self.get_parameter('image_topic').value),
            self.image_callback,
            10
        )
        # 2. CameraInfo
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self.camera_info_callback,
            5
        )
        # 3. Odometry: Prioritize /khepera4/odom with fallback to /odom
        self.odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self.khepera4_odom_callback,
            20
        )
        self.raw_odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter('raw_odom_topic').value),
            self.raw_odom_callback,
            20
        )
        # 4. Cliff Hazard Alert
        self.cliff_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('cliff_topic').value),
            self.cliff_callback,
            10
        )

        # ----------------------------------------------------------------------
        # Services & Timers
        # ----------------------------------------------------------------------
        # Service: /clear_visual_map
        clear_srv_name = str(self.get_parameter('clear_service_name').value)
        self.clear_visual_map_srv = self.create_service(
            Trigger,
            clear_srv_name,
            self.clear_map_callback
        )
        # Manual keyframe trigger services (/pin_keyframe and /record_keyframe alias)
        self.pin_keyframe_srv = self.create_service(
            Trigger,
            '/pin_keyframe',
            self.pin_keyframe_callback
        )
        self.record_keyframe_srv = self.create_service(
            Trigger,
            '/record_keyframe',
            self.pin_keyframe_callback
        )

        # Service: /save_visual_map
        self.save_visual_map_srv = self.create_service(
            Trigger,
            '/save_visual_map',
            self.save_map_callback
        )

        # Periodic visual map publisher timer (>= 1.0 Hz)
        pub_rate = float(self.get_parameter('map_publish_rate').value)
        publish_period = 1.0 / pub_rate
        self.publish_timer = self.create_timer(
            publish_period, self.publish_visual_maps
        )

        self.get_logger().info(
            f'VisualOrthomapper initialized: Grid {self.width}x{self.height} '
            f'@ {self.res}m (Origin [{self.origin_x}, {self.origin_y}]). '
            f'Camera: xc={self.camera_xc}m, h={self.camera_h}m, '
            f'f={self.fx:.2f}px. IPM ready.'
        )

    # ==========================================================================
    # Closed-Form Homography Calculations
    # ==========================================================================

    def compute_forward_homography(self) -> np.ndarray:
        """
        Compute closed-form 3x3 homography matrix H_b2img.

        Maps ground points in base link frame (X_b, Y_b, 0) to scaled image
        coordinates: s * [u, v, 1]^T = H_b2img * [X_b, Y_b, 1]^T
        where s = X_b - x_c.
        """
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        xc = self.camera_xc
        h = self.camera_h

        return np.array([
            [cx, -fx, -xc * cx],
            [cy, 0.0, h * fy - xc * cy],
            [1.0, 0.0, -xc]
        ], dtype=np.float64)

    def compute_inverse_homography(self) -> np.ndarray:
        """
        Compute closed-form 3x3 inverse homography matrix H_img2b.

        Maps image pixels [u, v, 1]^T to base ground plane coordinates:
        s_g * [X_b, Y_b, 1]^T = H_img2b * [u, v, 1]^T
        where s_g = (v - c_y) / (h * f_y).
        """
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        xc = self.camera_xc
        h = self.camera_h
        h_fy = h * fy

        return np.array([
            [0.0, xc / h_fy, 1.0 - (xc * cy) / h_fy],
            [-1.0 / fx, 0.0, cx / fx],
            [0.0, 1.0 / h_fy, -cy / h_fy]
        ], dtype=np.float64)

    def ground_to_image(
        self, xb: float, yb: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Project robot base ground point (X_b, Y_b, 0) to image coordinates.

        Returns: (u, v, is_valid)
        """
        s = xb - self.camera_xc
        if s <= 1e-4:
            return None, None, False

        u = self.cx - self.fx * (yb / s)
        v = self.cy + (self.camera_h * self.fy) / s

        is_valid = (
            0.0 <= u <= (self.image_width - 1.0) and
            self.min_row_v <= v <= (self.image_height - 1.0) and
            (self.min_ground_dist - 1e-4) <= xb <= self.max_ground_dist
        )
        return float(u), float(v), is_valid

    def image_to_ground(
        self, u: float, v: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Back-project image pixel (u, v) to ground point (X_b, Y_b, 0).

        Returns: (X_b, Y_b, is_valid)
        """
        denom = v - self.cy
        if denom <= 0.0:  # At or above horizon
            return None, None, False

        h_fy = self.camera_h * self.fy
        xb = self.camera_xc + h_fy / denom
        yb = -h_fy * (u - self.cx) / (denom * self.fx)

        # Check valid bounds
        tan_half_hfov = math.tan(self.hfov_rad / 2.0)
        max_half_width = (xb - self.camera_xc) * tan_half_hfov

        is_valid = (
            (self.min_ground_dist - 1e-4) <= xb <= self.max_ground_dist and
            abs(yb) <= max_half_width and
            0.0 <= u <= (self.image_width - 1.0) and
            self.min_row_v <= v <= (self.image_height - 1.0)
        )
        return float(xb), float(yb), is_valid

    # ==========================================================================
    # Coordinate Conversions
    # ==========================================================================

    def world_to_grid(self, xw: float, yw: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices (col, row)."""
        col = int(math.floor((xw - self.origin_x) / self.res))
        row = int(math.floor((yw - self.origin_y) / self.res))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell (col, row) to world coordinates at center."""
        xw = self.origin_x + (col + 0.5) * self.res
        yw = self.origin_y + (row + 0.5) * self.res
        return xw, yw

    # ==========================================================================
    # ROS 2 Callbacks: Odometry & Sensors
    # ==========================================================================

    def camera_info_callback(self, msg: CameraInfo):
        """Update intrinsic calibration parameters from CameraInfo topic."""
        if not self.has_camera_info:
            if msg.k[0] > 0.0:
                self.fx = float(msg.k[0])
                self.fy = float(msg.k[4])
                self.cx = float(msg.k[2])
                self.cy = float(msg.k[5])
                self.image_width = int(msg.width)
                self.image_height = int(msg.height)
                self.H_b2img = self.compute_forward_homography()
                self.H_img2b = self.compute_inverse_homography()
                self.keyframe_manager.fx = self.fx
                self.keyframe_manager.fy = self.fy
                self.keyframe_manager.image_width = self.image_width
                self.keyframe_manager.image_height = self.image_height
                self.has_camera_info = True
                self.get_logger().info(
                    f'Camera calibration updated: fx={self.fx:.2f}, '
                    f'fy={self.fy:.2f}, cx={self.cx:.1f}, cy={self.cy:.1f}'
                )

    def khepera4_odom_callback(self, msg: Odometry):
        """Handle high-accuracy fused odometry from /khepera4/odom."""
        self.last_khepera4_odom_time = self.get_clock().now()
        self._process_odometry_message(msg)

    def raw_odom_callback(self, msg: Odometry):
        """Handle raw odometry from /odom as automatic fallback."""
        use_raw = False
        if self.last_khepera4_odom_time is None:
            use_raw = True
        else:
            time_now = self.get_clock().now()
            last_ns = getattr(self.last_khepera4_odom_time, 'nanoseconds', None)
            if last_ns is None and hasattr(self.last_khepera4_odom_time, 'sec'):
                last_ns = int(
                    self.last_khepera4_odom_time.sec * 1_000_000_000 +
                    self.last_khepera4_odom_time.nanosec
                )
            if last_ns is not None:
                dt = (time_now.nanoseconds - last_ns) * 1e-9
            else:
                dt = 999.0
            if dt > 1.0:
                use_raw = True

        if use_raw:
            self._process_odometry_message(msg)

    def _process_odometry_message(self, msg: Odometry):
        """Extract pose and attitude from odometry message."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # Arena boundary clamping
        self.robot_x = float(np.clip(x, self.arena_min_x, self.arena_max_x))
        self.robot_y = float(np.clip(y, self.arena_min_y, self.arena_max_y))
        self.robot_z = z
        self.robot_roll = roll
        self.robot_pitch = pitch
        self.robot_yaw = yaw
        self.robot_linear_vel = msg.twist.twist.linear.x
        self.robot_angular_vel = msg.twist.twist.angular.z
        self.last_odom_stamp = msg.header.stamp
        self.has_odom = True

    def cliff_callback(self, msg: Bool):
        """Track downward cliff / drop-off alerts."""
        self.cliff_alert = bool(msg.data)

    # ==========================================================================
    # Core Image Processing & Inverse Perspective Mapping (IPM)
    # ==========================================================================

    def image_callback(self, msg: RosImage):
        """
        Process camera image and update visual orthomap.

        Unpacks frame, checks tilt guard, warps floor pixels into 2D visual
        grid, and evaluates keyframe triggers.
        """
        if not self.has_odom:
            return

        # 1. Unpack ROS Image to RGB numpy array
        try:
            h, w = msg.height, msg.width
            if msg.encoding == 'rgb8':
                img_rgb = np.frombuffer(
                    msg.data, dtype=np.uint8
                ).reshape((h, w, 3))
            elif msg.encoding == 'bgr8':
                img_bgr = np.frombuffer(
                    msg.data, dtype=np.uint8
                ).reshape((h, w, 3))
                img_rgb = img_bgr[:, :, ::-1]
            elif msg.encoding == 'rgba8':
                img_rgba = np.frombuffer(
                    msg.data, dtype=np.uint8
                ).reshape((h, w, 4))
                img_rgb = img_rgba[:, :, :3]
            elif msg.encoding == 'bgra8':
                img_bgra = np.frombuffer(
                    msg.data, dtype=np.uint8
                ).reshape((h, w, 4))
                img_rgb = img_bgra[:, :, [2, 1, 0]]
            elif msg.encoding in ('mono8', '8UC1'):
                img_mono = np.frombuffer(
                    msg.data, dtype=np.uint8
                ).reshape((h, w))
                img_rgb = np.repeat(img_mono[:, :, np.newaxis], 3, axis=2)
            else:
                self.get_logger().warn(
                    f'Unsupported image encoding: {msg.encoding}',
                    throttle_duration_sec=5.0
                )
                return
        except Exception as e:
            self.get_logger().error(
                f'Failed to decode camera image: {str(e)}',
                throttle_duration_sec=2.0
            )
            return

        # Store latest frame for keyframe sampling
        self.latest_camera_frame = img_rgb.copy()
        self.latest_camera_stamp = msg.header.stamp

        # 2. Check Pitch Tilt Guard
        # If climbing a 10 deg ramp or tilted > 6 deg, suspend floor warping
        pitch_deg = abs(math.degrees(self.robot_pitch))
        if pitch_deg > (self.pitch_guard_deg + 1e-4):
            self.get_logger().debug(
                f'Suspended IPM floor grid warping: robot pitch '
                f'{pitch_deg:.2f}° > {self.pitch_guard_deg}°',
                throttle_duration_sec=2.0
            )
        else:
            # 3. Perform Inverse Grid Pull-Warping and Alpha Blending
            self.warp_image_to_grid(
                img_rgb,
                self.robot_x,
                self.robot_y,
                self.robot_yaw,
                self.robot_pitch
            )

        # 4. Check Keyframe Triggers
        if self.enable_auto_keyframes:
            self.evaluate_keyframe_triggers(img_rgb, msg.header.stamp)

    def warp_image_to_grid(
        self,
        image_rgb: np.ndarray,
        rx: float,
        ry: float,
        ryaw: float,
        rpitch: float
    ) -> int:
        """
        Perform inverse grid pull-warping from camera image to map grid.

        For every visual map cell inside the camera's ground trapezoid:
        1. Project cell center into image coordinates (u, v) using IPM.
        2. Bilinearly sample pixel color.
        3. Alpha-blend into visual orthomosaic grid layer (alpha = 0.25).
        Returns the number of updated grid cells.
        """
        # Guard: check pitch tilt angle
        if abs(math.degrees(rpitch)) > (self.pitch_guard_deg + 1e-4):
            return 0

        tan_half_hfov = math.tan(self.hfov_rad / 2.0)
        h_fy = self.camera_h * self.fy

        # Define 4 corners of visible ground trapezoid in base frame:
        xb_near = self.min_ground_dist
        yb_near = (xb_near - self.camera_xc) * tan_half_hfov
        xb_far = self.max_ground_dist
        yb_far = (xb_far - self.camera_xc) * tan_half_hfov

        corners_b = np.array([
            [xb_near, yb_near],
            [xb_near, -yb_near],
            [xb_far, -yb_far],
            [xb_far, yb_far]
        ], dtype=np.float64)

        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        # Transform corners to world map coordinates
        corners_m_x = rx + corners_b[:, 0] * cos_yaw - corners_b[:, 1] * sin_yaw
        corners_m_y = ry + corners_b[:, 0] * sin_yaw + corners_b[:, 1] * cos_yaw

        min_xm = max(float(np.min(corners_m_x)), self.arena_min_x)
        max_xm = min(float(np.max(corners_m_x)), self.arena_max_x)
        min_ym = max(float(np.min(corners_m_y)), self.arena_min_y)
        max_ym = min(float(np.max(corners_m_y)), self.arena_max_y)

        # Bounding box of candidate grid cells
        c_min, r_min = self.world_to_grid(min_xm, min_ym)
        c_max, r_max = self.world_to_grid(max_xm, max_ym)

        c_min = max(0, min(self.width - 1, c_min))
        c_max = max(0, min(self.width - 1, c_max))
        r_min = max(0, min(self.height - 1, r_min))
        r_max = max(0, min(self.height - 1, r_max))

        if c_min > c_max or r_min > r_max:
            return 0

        # Create candidate cell meshgrid
        cols = np.arange(c_min, c_max + 1, dtype=np.int32)
        rows = np.arange(r_min, r_max + 1, dtype=np.int32)
        col_grid, row_grid = np.meshgrid(cols, rows)
        c_flat = col_grid.ravel()
        r_flat = row_grid.ravel()

        # Compute cell center coordinates in world map frame
        xm_flat = self.origin_x + (c_flat.astype(np.float64) + 0.5) * self.res
        ym_flat = self.origin_y + (r_flat.astype(np.float64) + 0.5) * self.res

        # Arena boundary check
        in_arena = (
            (xm_flat >= self.arena_min_x) & (xm_flat <= self.arena_max_x) &
            (ym_flat >= self.arena_min_y) & (ym_flat <= self.arena_max_y)
        )
        c_flat = c_flat[in_arena]
        r_flat = r_flat[in_arena]
        xm_flat = xm_flat[in_arena]
        ym_flat = ym_flat[in_arena]

        if len(c_flat) == 0:
            return 0

        # Transform candidate world points to robot base link frame
        dx = xm_flat - rx
        dy = ym_flat - ry
        xb = dx * cos_yaw + dy * sin_yaw
        yb = -dx * sin_yaw + dy * cos_yaw

        # Ground FOV trapezoid check
        dist_c = xb - self.camera_xc
        in_fov = (
            (xb >= self.min_ground_dist) &
            (xb <= self.max_ground_dist) &
            (dist_c > 1e-4) &
            (np.abs(yb) <= dist_c * tan_half_hfov)
        )

        c_sel = c_flat[in_fov]
        r_sel = r_flat[in_fov]
        yb_sel = yb[in_fov]
        dist_c_sel = dist_c[in_fov]

        if len(c_sel) == 0:
            return 0

        # Closed-form pinhole projection from ground to image
        u = self.cx - self.fx * (yb_sel / dist_c_sel)
        v = self.cy + h_fy / dist_c_sel

        # Pixel validity check: horizon cutoff
        img_h, img_w = image_rgb.shape[:2]
        max_v = float(min(self.max_row_v, img_h - 1))
        valid_px = (
            (u >= 0.0) & (u <= float(img_w - 1)) &
            (v >= float(self.min_row_v)) & (v <= max_v)
        )

        c_valid = c_sel[valid_px]
        r_valid = r_sel[valid_px]
        u_valid = u[valid_px]
        v_valid = v[valid_px]

        if len(c_valid) == 0:
            return 0

        # Bilinear color interpolation from image_rgb
        u0 = np.floor(u_valid).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, img_w - 1)
        v0 = np.floor(v_valid).astype(np.int32)
        v1 = np.clip(v0 + 1, 0, img_h - 1)

        wu1 = (u_valid - u0)[:, np.newaxis]
        wu0 = 1.0 - wu1
        wv1 = (v_valid - v0)[:, np.newaxis]
        wv0 = 1.0 - wv1

        top = (
            wu0 * image_rgb[v0, u0].astype(np.float32) +
            wu1 * image_rgb[v0, u1].astype(np.float32)
        )
        bot = (
            wu0 * image_rgb[v1, u0].astype(np.float32) +
            wu1 * image_rgb[v1, u1].astype(np.float32)
        )
        c_obs = wv0 * top + wv1 * bot  # Shape (N, 3)

        # Alpha-Blending Update
        is_first_obs = ~self.observed_mask[r_valid, c_valid]
        idx_first = np.where(is_first_obs)[0]
        idx_update = np.where(~is_first_obs)[0]

        if len(idx_first) > 0:
            rf = r_valid[idx_first]
            cf = c_valid[idx_first]
            self.color_map[rf, cf] = c_obs[idx_first]
            self.observed_mask[rf, cf] = True
            self.weight_map[rf, cf] = 1.0

        if len(idx_update) > 0:
            ru = r_valid[idx_update]
            cu = c_valid[idx_update]
            alpha = self.alpha
            self.color_map[ru, cu] = (
                (1.0 - alpha) * self.color_map[ru, cu] + alpha * c_obs[idx_update]
            )
            self.weight_map[ru, cu] = np.minimum(
                self.weight_map[ru, cu] + 1.0, 100.0
            )

        return len(c_valid)

    # ==========================================================================
    # Map Publishing
    # ==========================================================================

    def publish_visual_maps(self):
        """Publish /visual_map (OccupancyGrid) and /visual_map/image (RGBA)."""
        now = self.get_clock().now().to_msg()

        # ----------------------------------------------------------------------
        # 1. OccupancyGrid (/visual_map)
        # ----------------------------------------------------------------------
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = now
        grid_msg.header.frame_id = self.map_frame
        grid_msg.info.map_load_time = now
        grid_msg.info.resolution = self.res
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # Calculate reflectance 0..100; unobserved is -1
        grid_msg.data = self.to_occupancy_grid_data()
        self.visual_map_pub.publish(grid_msg)

        # ----------------------------------------------------------------------
        # 2. RGBA Orthomosaic Image (/visual_map/image)
        # ----------------------------------------------------------------------
        img_msg = RosImage()
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.map_frame
        img_msg.height = self.height
        img_msg.width = self.width
        img_msg.encoding = 'rgba8'
        img_msg.is_bigendian = 0
        img_msg.step = self.width * 4

        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(self.color_map, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = np.where(self.observed_mask, 255, 0).astype(np.uint8)

        img_msg.data = rgba.tobytes()
        self.visual_map_image_pub.publish(img_msg)

    # ==========================================================================
    # Spatial Keyframe & Inspection Landmark System
    # ==========================================================================

    @property
    def keyframes(self) -> List[Dict]:
        """Return list of captured keyframe records."""
        return self.keyframe_manager.keyframes

    @keyframes.setter
    def keyframes(self, val: List[Dict]):
        self.keyframe_manager.keyframes = val

    @property
    def keyframe_count(self) -> int:
        """Return total number of keyframes."""
        return len(self.keyframe_manager.keyframes)

    @keyframe_count.setter
    def keyframe_count(self, val: int):
        self.keyframe_manager.next_id = val + 1

    @property
    def next_id(self) -> int:
        """Return next keyframe ID."""
        return self.keyframe_manager.next_id

    @next_id.setter
    def next_id(self, val: int):
        self.keyframe_manager.next_id = val

    @property
    def last_kf_x(self) -> Optional[float]:
        """Return last keyframe X coordinate."""
        return self.keyframe_manager.last_kf_x

    @last_kf_x.setter
    def last_kf_x(self, val: Optional[float]):
        self.keyframe_manager.last_kf_x = val

    @property
    def last_kf_y(self) -> Optional[float]:
        """Return last keyframe Y coordinate."""
        return self.keyframe_manager.last_kf_y

    @last_kf_y.setter
    def last_kf_y(self, val: Optional[float]):
        self.keyframe_manager.last_kf_y = val

    @property
    def last_kf_yaw(self) -> Optional[float]:
        """Return last keyframe yaw orientation."""
        return self.keyframe_manager.last_kf_yaw

    @last_kf_yaw.setter
    def last_kf_yaw(self, val: Optional[float]):
        self.keyframe_manager.last_kf_yaw = val

    @property
    def last_hazard_time(self) -> float:
        """Return timestamp of last cliff hazard keyframe."""
        return self.keyframe_manager.last_hazard_time

    @last_hazard_time.setter
    def last_hazard_time(self, val: float):
        self.keyframe_manager.last_hazard_time = val

    @property
    def last_hazard_x(self) -> float:
        """Return X position of last cliff hazard."""
        return self.keyframe_manager.last_hazard_x

    @last_hazard_x.setter
    def last_hazard_x(self, val: float):
        self.keyframe_manager.last_hazard_x = val

    @property
    def last_hazard_y(self) -> float:
        """Return Y position of last cliff hazard."""
        return self.keyframe_manager.last_hazard_y

    @last_hazard_y.setter
    def last_hazard_y(self, val: float):
        self.keyframe_manager.last_hazard_y = val

    def should_trigger_displacement(
        self, x: float, y: float, threshold: Optional[float] = None
    ) -> bool:
        """Check if robot displacement exceeds keyframe threshold."""
        if threshold is None:
            threshold = self.kf_disp_thresh
        return self.keyframe_manager.should_trigger_displacement(x, y, threshold)

    def should_trigger_orientation(
        self, yaw: float, threshold_deg: Optional[float] = None
    ) -> bool:
        """Check if robot heading change exceeds keyframe threshold."""
        if threshold_deg is None:
            threshold_deg = float(self.get_parameter('kf_orientation_thresh_deg').value)
        return self.keyframe_manager.should_trigger_orientation(yaw, threshold_deg)

    def should_trigger_cliff(
        self,
        x: float,
        y: float,
        current_time: float,
        cooldown_sec: Optional[float] = None,
        min_dist: Optional[float] = None
    ) -> bool:
        """Check if cliff hazard qualifies after debouncing cooldown."""
        if cooldown_sec is None:
            cooldown_sec = self.kf_cliff_cooldown
        if min_dist is None:
            min_dist = self.kf_cliff_dist_thresh
        return self.keyframe_manager.should_trigger_cliff(
            x, y, current_time, cooldown_sec, min_dist
        )

    def record_keyframe(
        self,
        image_rgb: np.ndarray,
        x: float,
        y: float,
        yaw: float,
        trigger_type: str,
        current_time: Optional[float] = None,
        linear_vel: Optional[float] = None,
        angular_vel: Optional[float] = None,
        cliff_alert: Optional[bool] = None
    ) -> dict:
        """Record and serialize keyframe photo and metadata entry."""
        if linear_vel is None:
            linear_vel = self.robot_linear_vel
        if angular_vel is None:
            angular_vel = self.robot_angular_vel
        if cliff_alert is None:
            cliff_alert = self.cliff_alert
        return self.keyframe_manager.record_keyframe(
            image_rgb, x, y, yaw, trigger_type,
            current_time=current_time,
            linear_vel=linear_vel,
            angular_vel=angular_vel,
            cliff_alert=cliff_alert
        )

    def clear_keyframes(self):
        """Purge keyframe images on disk and reset manifest catalog."""
        self.keyframe_manager.clear()

    def evaluate_keyframe_triggers(self, img_rgb: np.ndarray, stamp):
        """
        Evaluate automated keyframe capture triggers.

        Triggers checked:
        - Cliff Hazard: cliff_alert True (with 1.0s and 0.05m debounce)
        - Displacement: Delta s >= 0.35m
        - Orientation: Delta yaw >= 35 deg
        """
        current_time = stamp.sec + stamp.nanosec * 1e-9

        # 1. Cliff hazard trigger has highest priority
        if self.cliff_alert:
            if self.should_trigger_cliff(self.robot_x, self.robot_y, current_time):
                self.record_keyframe(
                    img_rgb, self.robot_x, self.robot_y, self.robot_yaw,
                    trigger_type='cliff_hazard',
                    current_time=current_time,
                    linear_vel=self.robot_linear_vel,
                    angular_vel=self.robot_angular_vel,
                    cliff_alert=True
                )
                return

        # 2. Displacement trigger
        if self.should_trigger_displacement(self.robot_x, self.robot_y):
            self.record_keyframe(
                img_rgb, self.robot_x, self.robot_y, self.robot_yaw,
                trigger_type='displacement',
                current_time=current_time,
                linear_vel=self.robot_linear_vel,
                angular_vel=self.robot_angular_vel,
                cliff_alert=bool(self.cliff_alert)
            )
            return

        # 3. Orientation trigger
        if self.should_trigger_orientation(self.robot_yaw):
            self.record_keyframe(
                img_rgb, self.robot_x, self.robot_y, self.robot_yaw,
                trigger_type='orientation',
                current_time=current_time,
                linear_vel=self.robot_linear_vel,
                angular_vel=self.robot_angular_vel,
                cliff_alert=bool(self.cliff_alert)
            )
            return

    def pin_keyframe_callback(self, request, response):
        """Handle manual service trigger to force keyframe capture."""
        if not self.has_odom or self.latest_camera_frame is None:
            response.success = False
            response.message = 'Odometry or camera frame not yet available'
            return response

        current_time = (
            (self.latest_camera_stamp.sec + self.latest_camera_stamp.nanosec * 1e-9)
            if self.latest_camera_stamp is not None
            else (self.get_clock().now().nanoseconds * 1e-9)
        )
        entry = self.record_keyframe(
            self.latest_camera_frame,
            self.robot_x,
            self.robot_y,
            self.robot_yaw,
            trigger_type='user_manual',
            current_time=current_time,
            linear_vel=self.robot_linear_vel,
            angular_vel=self.robot_angular_vel,
            cliff_alert=bool(self.cliff_alert)
        )
        response.success = True
        response.message = f"Pinned Keyframe #{entry['id']}"
        return response

    def clear(self, purge_disk: bool = False):
        """Clear visual map grid arrays and optionally keyframe memory."""
        self.color_map.fill(0.0)
        self.observed_mask.fill(False)
        self.weight_map.fill(0.0)
        if purge_disk:
            self.clear_keyframes()

    def clear_map_callback(self, request, response):
        """Reset visual map memory, clear keyframe index and purge cache."""
        self.get_logger().info('Clearing visual map memory and keyframes...')
        self.clear(purge_disk=True)
        self.publish_visual_maps()
        response.success = True
        response.message = 'Visual map and keyframe catalog cleared.'
        return response

    def save_composite_map(
        self, output_path: str = 'maps/khepera_visual_map.png'
    ) -> str:
        """Generate and save multi-layer composite visual map."""
        lo_grid = np.zeros((self.height, self.width), dtype=np.float32)
        elev_grid = np.full((self.height, self.width), -1, dtype=np.int8)
        surf_grid = np.full((self.height, self.width), -1, dtype=np.int8)
        robot_path = [(self.robot_x, self.robot_y)] if self.has_odom else []
        return generate_composite_map(
            output_path,
            self.color_map,
            lo_grid,
            elev_grid,
            surf_grid,
            robot_path,
            self.keyframes,
            res=self.res,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            current_pose=(
                (self.robot_x, self.robot_y, self.robot_yaw) if self.has_odom else None
            )
        )

    def save_map_callback(self, request, response):
        """Service callback to export composite visual map."""
        try:
            out_file = self.save_composite_map()
            response.success = True
            response.message = f'Composite visual map saved to {out_file}'
        except Exception as e:
            response.success = False
            response.message = f'Error saving composite visual map: {str(e)}'
        return response

    def to_occupancy_grid_data(self) -> List[int]:
        """Convert color grid to 0-100 grayscale occupancy data (-1 for unobserved)."""
        gray = (
            0.299 * self.color_map[:, :, 0] +
            0.587 * self.color_map[:, :, 1] +
            0.114 * self.color_map[:, :, 2]
        )
        occ = np.full((self.height, self.width), -1, dtype=np.int8)
        norm_gray = np.clip(
            np.round(gray * (100.0 / 255.0)), 0, 100
        ).astype(np.int8)
        occ[self.observed_mask] = norm_gray[self.observed_mask]
        return occ.flatten().tolist()

    def is_in_arena(self, xw: float, yw: float) -> bool:
        """Check if world coordinate is within arena boundaries."""
        return (
            self.arena_min_x <= xw <= self.arena_max_x and
            self.arena_min_y <= yw <= self.arena_max_y
        )


def main(args=None):
    """Run the visual orthomapper node."""
    rclpy.init(args=args)
    node = VisualOrthomapper()
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
