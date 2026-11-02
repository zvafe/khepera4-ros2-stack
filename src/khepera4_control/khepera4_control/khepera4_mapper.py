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
Khepera IV Mapping Engine: Multi-Sensor Fusion, Spatial Mapping & Persistent Memory.

Implements:
1. Multi-Sensor Spatial Fusion:
   - 360° Virtual LaserScan (/khepera4/scan) combining 8 IR and 5 Ultrasonic sensors
     accounting for physical mounting radial and angular offsets on base_footprint.
2. 2D Probabilistic Occupancy Grid Mapping (/map):
   - 6m x 6m @ 0.03m resolution (200x200 cells), origin at (-3.0, -3.0).
   - Fast Bresenham 2D ray-casting with log-odds belief updates.
   - Values: 0 (free), 100 (occupied), -1 (unknown).
3. 2.5D Terrain Elevation & Inclination Mapping (/elevation_map):
   - Pitch & roll extraction from IMU gravity vector.
   - Visited terrain slope angle in degrees (normalized 0-100).
4. Surface Classification & Cliff Hazard Detection (/surface_map, /cliff_alert):
   - 4 Downward-facing IR ground sensors for drop-off detection (> 0.025m).
   - High-rate vertical vibration variance estimation for terrain roughness.
5. Persistent Map Memory & Services:
   - /save_map: Standard ROS YAML + PGM + High-resolution color PNG + Composite Map.
   - /load_map: Reloads previous map for spatial memory persistence.
   - /clear_map: Resets maps, keyframes, and memory.
"""

import collections
from datetime import datetime, timezone
import json
import math
import os
import subprocess
from typing import List, Optional, Tuple

from geometry_msgs.msg import TransformStamped, Twist
from khepera4_control.common import (
    euler_from_quaternion,
    normalize_angle,
    quaternion_from_euler,
)
from khepera4_control.visual_orthomapper import generate_composite_map
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
from sensor_msgs.msg import Image as RosImage, Imu, LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


# ==============================================================================
# Sensor Mounting Constants (from Khepera IV CAD / URDF specification)
# (x_offset, y_offset, yaw_offset in radians relative to base_footprint)
# ==============================================================================

US_MOUNTS = [
    (0.0000, 0.0700, 1.570796),     # us_0: +90 deg (Left)
    (0.0448, 0.0454, 0.785398),     # us_1: +45 deg (Front-Left)
    (0.0636, 0.0000, 0.000000),     # us_2:   0 deg (Front)
    (0.0449, -0.0454, -0.785398),   # us_3: -45 deg (Front-Right)
    (0.0000, -0.0700, -1.570796),   # us_4: -90 deg (Right)
]

IR_MOUNTS = [
    (0.0690, 0.0120, 0.174),        # ir_0: ~10 deg (Front-Left-Center)
    (0.0490, 0.0490, 0.785),        # ir_1: ~45 deg (Front-Left)
    (0.0000, 0.0700, 1.571),        # ir_2: ~90 deg (Left)
    (-0.0490, 0.0490, 2.356),       # ir_3: ~135 deg (Rear-Left)
    (-0.0490, -0.0490, -2.356),     # ir_4: -135 deg (Rear-Right)
    (0.0000, -0.0700, -1.571),      # ir_5: -90 deg (Right)
    (0.0490, -0.0490, -0.785),      # ir_6: -45 deg (Front-Right)
    (0.0690, -0.0120, -0.174),      # ir_7: -10 deg (Front-Right-Center)
]

GROUND_IR_MOUNTS = [
    (0.0610, 0.0200),               # ground_ir_0: Front-Left Outer
    (0.0640, 0.0090),               # ground_ir_1: Front-Left Inner
    (0.0640, -0.0090),              # ground_ir_2: Front-Right Inner
    (0.0610, -0.0200),              # ground_ir_3: Front-Right Outer
]


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Generate cell coordinates along a line using 2D Bresenham algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


class Khepera4Mapper(Node):
    """Multi-sensor fusion, probabilistic 2D mapping, 2.5D elevation & terrain classification."""

    def __init__(self, **kwargs):
        super().__init__('khepera4_mapper', **kwargs)

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter('map_resolution', 0.03)      # 3 cm per cell
        self.declare_parameter('map_width', 200)            # 200 cells = 6.0 m
        self.declare_parameter('map_height', 200)           # 200 cells = 6.0 m
        self.declare_parameter('map_origin_x', -3.0)        # Grid origin X (m)
        self.declare_parameter('map_origin_y', -3.0)        # Grid origin Y (m)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_map_tf', True)
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('use_imu_yaw', True)        # Fuse drift-free IMU heading
        self.declare_parameter('scan_publish_rate', 10.0)   # 10 Hz for 360 virtual scan
        self.declare_parameter('map_publish_rate', 3.0)     # 3 Hz for occupancy grids
        self.declare_parameter('log_odds_free', 0.40)       # Decrement for free ray
        self.declare_parameter('log_odds_occ', 0.85)        # Increment for occupied hit
        self.declare_parameter('log_odds_min', -5.0)        # Max free confidence
        self.declare_parameter('log_odds_max', 5.0)         # Max occupied confidence
        self.declare_parameter('cliff_distance_threshold', 0.025)  # 25mm drop-off
        self.declare_parameter('clamp_odometry_slip', True)        # Clamp wheel slip runaway
        self.declare_parameter('max_step_delta_s', 0.05)           # 50Hz max step threshold
        self.declare_parameter('map_save_dir', 'maps')
        self.declare_parameter('fused_odom_topic', '/khepera4/odom')
        self.declare_parameter('contact_dist_threshold', 0.015)
        self.declare_parameter('slip_dist_threshold', 0.035)
        self.declare_parameter('arena_min_x', -2.405)
        self.declare_parameter('arena_max_x', 2.405)
        self.declare_parameter('arena_min_y', -2.405)
        self.declare_parameter('arena_max_y', 2.405)

        # Phase 1: Multi-Sensor Rate-of-Change & IMU Gating Parameters
        self.declare_parameter('enable_ir_gradient_slip_check', True)
        self.declare_parameter('enable_imu_accel_gate', True)
        self.declare_parameter('ir_gradient_window_s', 0.10)          # 100 ms rolling window
        self.declare_parameter('ir_gradient_min_cmd_v', 0.04)         # cmd_linear_x > 0.04 m/s
        self.declare_parameter('ir_gradient_min_delta_s', 0.002)      # encoder advance > 0.002 m
        self.declare_parameter('ir_gradient_max_closing_rate', 0.0005)  # delta approx 0
        self.declare_parameter('ir_gradient_proximity_thresh', 0.08)  # obstacle prox thresh (m)
        self.declare_parameter('imu_accel_gate_threshold', 0.10)      # IMU accel gate (m/s^2)
        self.declare_parameter('imu_accel_proximity_thresh', 0.08)    # prox gate for accel (m)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Phase 2: 2D Correlative Scan Matching & Pose Correction Parameters
        self.declare_parameter('enable_scan_matching', True)
        self.declare_parameter('csm_search_dist', 0.03)               # +/- 3cm translation window
        self.declare_parameter('csm_search_angle', 0.03)              # +/- ~1.7 deg yaw window
        self.declare_parameter('csm_trans_step', 0.01)                # 1cm resolution
        self.declare_parameter('csm_angle_step', 0.015)               # ~0.86 deg resolution
        self.declare_parameter('csm_min_valid_hits', 4)               # min scan points required
        self.declare_parameter('csm_score_threshold', 0.20)           # min improvement score
        self.declare_parameter('csm_alpha_smooth', 0.30)              # EMA smoothing factor
        self.declare_parameter('csm_deadband', 0.005)                 # 5mm deadband
        self.declare_parameter('hit_count_confirm_threshold', 10)     # hits to solidify wall
        self.declare_parameter('enable_footprint_carving', True)      # erase ghost walls

        self.res = float(self.get_parameter('map_resolution').value)
        self.width = int(self.get_parameter('map_width').value)
        self.height = int(self.get_parameter('map_height').value)
        self.origin_x = float(self.get_parameter('map_origin_x').value)
        self.origin_y = float(self.get_parameter('map_origin_y').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.publish_map_tf = bool(self.get_parameter('publish_map_tf').value)
        self.robot_frame = self.get_parameter('robot_frame').value
        self.l_free = float(self.get_parameter('log_odds_free').value)
        self.l_occ = float(self.get_parameter('log_odds_occ').value)
        self.l_min = float(self.get_parameter('log_odds_min').value)
        self.l_max = float(self.get_parameter('log_odds_max').value)
        self.cliff_thresh = float(self.get_parameter('cliff_distance_threshold').value)
        self.clamp_odometry_slip = bool(self.get_parameter('clamp_odometry_slip').value)
        self.max_step_delta_s = float(self.get_parameter('max_step_delta_s').value)
        self.map_save_dir = self.get_parameter('map_save_dir').value
        self._use_imu_yaw = bool(self.get_parameter('use_imu_yaw').value)
        self.fused_odom_topic = str(self.get_parameter('fused_odom_topic').value)
        self.contact_dist_thresh = float(self.get_parameter('contact_dist_threshold').value)
        self.slip_dist_thresh = float(self.get_parameter('slip_dist_threshold').value)
        self.arena_min_x = float(self.get_parameter('arena_min_x').value)
        self.arena_max_x = float(self.get_parameter('arena_max_x').value)
        self.arena_min_y = float(self.get_parameter('arena_min_y').value)
        self.arena_max_y = float(self.get_parameter('arena_max_y').value)

        self.enable_ir_gradient_slip_check = bool(
            self.get_parameter('enable_ir_gradient_slip_check').value
        )
        self.enable_imu_accel_gate = bool(self.get_parameter('enable_imu_accel_gate').value)
        self.ir_gradient_window_s = float(self.get_parameter('ir_gradient_window_s').value)
        self.ir_gradient_min_cmd_v = float(self.get_parameter('ir_gradient_min_cmd_v').value)
        self.ir_gradient_min_delta_s = float(self.get_parameter('ir_gradient_min_delta_s').value)
        self.ir_gradient_max_closing_rate = float(
            self.get_parameter('ir_gradient_max_closing_rate').value
        )
        self.ir_gradient_proximity_thresh = float(
            self.get_parameter('ir_gradient_proximity_thresh').value
        )
        self.imu_accel_gate_threshold = float(self.get_parameter('imu_accel_gate_threshold').value)
        self.imu_accel_proximity_thresh = float(
            self.get_parameter('imu_accel_proximity_thresh').value
        )

        self.enable_scan_matching = bool(self.get_parameter('enable_scan_matching').value)
        self.csm_search_dist = float(self.get_parameter('csm_search_dist').value)
        self.csm_search_angle = float(self.get_parameter('csm_search_angle').value)
        self.csm_trans_step = float(self.get_parameter('csm_trans_step').value)
        self.csm_angle_step = float(self.get_parameter('csm_angle_step').value)
        self.csm_min_valid_hits = int(self.get_parameter('csm_min_valid_hits').value)
        self.csm_score_thresh = float(self.get_parameter('csm_score_threshold').value)
        self.csm_alpha_smooth = float(self.get_parameter('csm_alpha_smooth').value)
        self.csm_deadband = float(self.get_parameter('csm_deadband').value)
        self.hit_count_confirm_threshold = int(
            self.get_parameter('hit_count_confirm_threshold').value
        )
        self.enable_footprint_carving = bool(self.get_parameter('enable_footprint_carving').value)

        # ----------------------------------------------------------------------
        # State Arrays
        # ----------------------------------------------------------------------
        # 2D Occupancy Grid log-odds belief: 0.0 = prior (p=0.5, unknown)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

        # 2D Hit confirmation count matrix (0-255) for quality scoring & ghost wall erosion
        self.hit_count = np.zeros((self.height, self.width), dtype=np.uint8)

        # 2.5D Elevation & slope grid (slope in degrees 0-100, -1 = unvisited)
        self.elevation_grid = np.full((self.height, self.width), -1, dtype=np.int8)

        # Surface type & cliff hazard grid (100 = cliff, 0-85 = roughness, -1 = unvisited)
        self.surface_grid = np.full((self.height, self.width), -1, dtype=np.int8)

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_odom = False
        self.has_imu = False
        self.imu_yaw = 0.0
        self.imu_valid = False
        self.last_odom_x: Optional[float] = None
        self.last_odom_y: Optional[float] = None
        self.last_odom_yaw: Optional[float] = None
        self.robot_path = collections.deque(maxlen=5000)

        # Commanded motion tracking
        self.cmd_linear_x = 0.0
        self.cmd_angular_z = 0.0
        self.last_cmd_vel_time = 0.0

        # Heuristic slip & rate-of-change telemetry state
        self.is_wheel_slipping = False
        self.is_near_contact = False
        self.front_range_history = collections.deque(maxlen=30)
        self.last_odom_time: Optional[float] = None
        self.last_wheel_v: float = 0.0
        self.wheel_accel_x: float = 0.0
        self.ax_dynamic_filtered: float = 0.0

        # IMU state, jerk & impact tracking
        self.current_slope_deg = 0.0
        self.current_roughness = 0
        self.accel_z_buffer = collections.deque(maxlen=30)
        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_imu_time: Optional[float] = None
        self.imu_jerk = 0.0
        self.last_impact_time = 0.0

        # Sensor readings caches: latest message + arrival timestamp
        self.latest_us: List[Optional[LaserScan]] = [None] * 5
        self.latest_ir: List[Optional[LaserScan]] = [None] * 8
        self.latest_ground_ir: List[Optional[LaserScan]] = [None] * 4

        # Cliff alert state
        self.cliff_detected = False

        # Rollover / Turned Over State
        self.is_turned_over = False

        # ----------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # a) 360 Virtual LaserScan
        self.scan_pub = self.create_publisher(LaserScan, '/khepera4/scan', qos_reliable)

        # b) Probabilistic 2D Map (Latched for RViz & Dashboard)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', qos_latched)

        # c) 2.5D Elevation / Slope Map (Latched)
        self.elevation_pub = self.create_publisher(OccupancyGrid, '/elevation_map', qos_latched)

        # d) Surface & Cliff Hazard Map (Latched)
        self.surface_pub = self.create_publisher(OccupancyGrid, '/surface_map', qos_latched)
        self.cliff_alert_pub = self.create_publisher(Bool, '/cliff_alert', 10)
        self.turned_over_pub = self.create_publisher(Bool, '/khepera4/turned_over', 10)

        # e) Fused Odometry with Tangential Slip Projection & Perimeter Clamping
        self.fused_odom_pub = self.create_publisher(
            Odometry, self.fused_odom_topic, qos_reliable
        )

        # f) TF Broadcaster for map -> odom transform
        self.tf_broadcaster = TransformBroadcaster(self)

        # ----------------------------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------------------------
        odom_top = self.get_parameter('odom_topic').value
        self.odom_sub = self.create_subscription(Odometry, odom_top, self.odom_callback, 20)
        self.khepera_odom_sub = None

        cmd_top = str(self.get_parameter('cmd_vel_topic').value)
        self.cmd_sub = self.create_subscription(
            Twist, cmd_top, self.cmd_vel_callback, 10
        )
        if cmd_top != '/khepera4/cmd_vel':
            self.khepera_cmd_sub = self.create_subscription(
                Twist, '/khepera4/cmd_vel', self.cmd_vel_callback, 10
            )
        else:
            self.khepera_cmd_sub = None

        imu_top = self.get_parameter('imu_topic').value
        self.imu_sub = self.create_subscription(Imu, imu_top, self.imu_callback, 30)
        if imu_top != '/khepera4/imu/data':
            self.khepera_imu_sub = self.create_subscription(
                Imu, '/khepera4/imu/data', self.imu_callback, 30
            )
        else:
            self.khepera_imu_sub = None

        # Ultrasonic Scans (5)
        self.us_subs = []
        for i in range(5):
            topic = f'/khepera4/us_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.us_callback(idx, msg),
                10
            )
            self.us_subs.append(sub)

        # IR Proximity Scans (8)
        self.ir_subs = []
        for i in range(8):
            topic = f'/khepera4/ir_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.ir_callback(idx, msg),
                10
            )
            self.ir_subs.append(sub)

        # Ground IR Scans (4)
        self.ground_ir_subs = []
        for i in range(4):
            topic = f'/khepera4/ground_ir_{i}/scan'
            sub = self.create_subscription(
                LaserScan, topic,
                lambda msg, idx=i: self.ground_ir_callback(idx, msg),
                10
            )
            self.ground_ir_subs.append(sub)

        # ----------------------------------------------------------------------
        # Visual Orthomap Cache for Composite Export
        # ----------------------------------------------------------------------
        self.visual_grid = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        self.has_visual_map = False

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.visual_image_sub = self.create_subscription(
            RosImage,
            '/visual_map/image',
            self.visual_image_callback,
            qos_latched
        )
        self.visual_grid_sub = self.create_subscription(
            OccupancyGrid,
            '/visual_map',
            self.visual_grid_callback,
            qos_latched
        )
        self.clear_visual_map_cli = self.create_client(
            Trigger, '/clear_visual_map'
        )

        # ----------------------------------------------------------------------
        # Services (Map Persistence & Memory & Recovery)
        # ----------------------------------------------------------------------
        self.save_map_srv = self.create_service(
            Trigger, '/save_map', self.save_map_callback
        )
        self.load_map_srv = self.create_service(
            Trigger, '/load_map', self.load_map_callback
        )
        self.clear_map_srv = self.create_service(
            Trigger, '/clear_map', self.clear_map_callback
        )
        self.recenter_srv = self.create_service(
            Trigger, '/recenter_robot', self.recenter_robot_callback
        )

        # ----------------------------------------------------------------------
        # Timers
        # ----------------------------------------------------------------------
        scan_period = 1.0 / float(self.get_parameter('scan_publish_rate').value)
        self.scan_timer = self.create_timer(scan_period, self.publish_virtual_scan)

        map_period = 1.0 / float(self.get_parameter('map_publish_rate').value)
        self.map_timer = self.create_timer(map_period, self.update_and_publish_maps)

        self.get_logger().info(
            f'Khepera4Mapper initialized: Grid {self.width}x{self.height} @ {self.res}m '
            f'(Origin [{self.origin_x}, {self.origin_y}]). Fusion & persistence online.'
        )

    # ==========================================================================
    # IMU Fusion & Heading Properties
    # ==========================================================================

    @property
    def use_imu_yaw(self) -> bool:
        """Flag indicating whether IMU heading fusion and dead-reckoning projection are active."""
        try:
            return bool(self.get_parameter('use_imu_yaw').value)
        except Exception:
            return self._use_imu_yaw

    @use_imu_yaw.setter
    def use_imu_yaw(self, value: bool):
        self._use_imu_yaw = bool(value)

    # ==========================================================================
    # Subscriber Callbacks
    # ==========================================================================

    def cmd_vel_callback(self, msg: Twist):
        """Track commanded velocities for slip & stall detection."""
        self.cmd_linear_x = msg.linear.x
        self.cmd_angular_z = msg.angular.z
        self.last_cmd_vel_time = self.get_clock().now().nanoseconds * 1e-9

    def _compute_surface_normal(self) -> Optional[Tuple[float, float]]:
        """Compute surface normal in world coordinates pointing OUT of obstacle toward robot."""
        d_detect = 0.055
        vec_x = 0.0
        vec_y = 0.0
        total_w = 0.0

        for i, mount in enumerate(IR_MOUNTS):
            scan = self.latest_ir[i]
            if scan is None or not scan.ranges:
                continue
            sx, sy, syaw = mount
            for j, r in enumerate(scan.ranges):
                if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                    continue
                if r <= d_detect:
                    ray_ang = syaw + (scan.angle_min + j * scan.angle_increment)
                    hx = sx + r * math.cos(ray_ang)
                    hy = sy + r * math.sin(ray_ang)
                    h_norm = math.hypot(hx, hy)
                    if h_norm > 1e-4:
                        ux = hx / h_norm
                        uy = hy / h_norm
                    else:
                        ux = math.cos(ray_ang)
                        uy = math.sin(ray_ang)
                    w = ((d_detect - r) / d_detect) ** 2
                    vec_x -= w * ux
                    vec_y -= w * uy
                    total_w += w

        norm_sq = vec_x * vec_x + vec_y * vec_y
        if norm_sq < 1e-8:
            # Fallback: check closest valid IR reading within detection threshold
            min_r = float('inf')
            best_dir = None
            for i, mount in enumerate(IR_MOUNTS):
                scan = self.latest_ir[i]
                if scan is None or not scan.ranges:
                    continue
                sx, sy, syaw = mount
                for j, r in enumerate(scan.ranges):
                    if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                        continue
                    if r < min_r and r <= d_detect:
                        min_r = r
                        ray_ang = syaw + (scan.angle_min + j * scan.angle_increment)
                        best_dir = (-math.cos(ray_ang), -math.sin(ray_ang))
            if best_dir is not None:
                vec_x, vec_y = best_dir
                norm_sq = vec_x * vec_x + vec_y * vec_y
            else:
                return None

        inv_len = 1.0 / math.sqrt(norm_sq)
        nx_body = vec_x * inv_len
        ny_body = vec_y * inv_len

        # Transform to world coordinates using self.robot_yaw
        cy = math.cos(self.robot_yaw)
        sy = math.sin(self.robot_yaw)
        nx_world = cy * nx_body - sy * ny_body
        ny_world = sy * nx_body + cy * ny_body

        return (nx_world, ny_world)

    def _publish_fused_odom(self, stamp, vx: float, vy: float, wz: float,
                            pose_cov=None, twist_cov=None):
        """Construct and publish nav_msgs/Odometry on /khepera4/odom."""
        fused = Odometry()
        fused.header.stamp = stamp
        fused.header.frame_id = 'odom'
        fused.child_frame_id = self.robot_frame

        fused.pose.pose.position.x = self.robot_x
        fused.pose.pose.position.y = self.robot_y
        fused.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.robot_yaw)
        fused.pose.pose.orientation.x = qx
        fused.pose.pose.orientation.y = qy
        fused.pose.pose.orientation.z = qz
        fused.pose.pose.orientation.w = qw
        if pose_cov is not None:
            fused.pose.covariance = pose_cov

        fused.twist.twist.linear.x = vx
        fused.twist.twist.linear.y = vy
        fused.twist.twist.linear.z = 0.0
        fused.twist.twist.angular.x = 0.0
        fused.twist.twist.angular.y = 0.0
        fused.twist.twist.angular.z = wz
        if twist_cov is not None:
            fused.twist.covariance = twist_cov

        self.fused_odom_pub.publish(fused)

        self._broadcast_map_tf(stamp)

    def _broadcast_map_tf(self, stamp):
        """
        Broadcast dynamic map -> odom transform.

        T(map -> odom) = T(map -> base) * T(odom -> base)^-1.
        This keeps the visual model in RViz (fixed frame: map) strictly aligned with
        the filtered map pose, completely preventing wall-collision visual penetration
        and sensor beam tearing.
        """
        if not self.publish_map_tf:
            return
        ox = self.last_odom_x if self.last_odom_x is not None else self.robot_x
        oy = self.last_odom_y if self.last_odom_y is not None else self.robot_y
        oyaw = self.last_odom_yaw if self.last_odom_yaw is not None else self.robot_yaw

        d_yaw = normalize_angle(self.robot_yaw - oyaw)
        cos_dyaw = math.cos(d_yaw)
        sin_dyaw = math.sin(d_yaw)

        t_map_x = self.robot_x - (ox * cos_dyaw - oy * sin_dyaw)
        t_map_y = self.robot_y - (ox * sin_dyaw + oy * cos_dyaw)

        t_map = TransformStamped()
        t_map.header.stamp = stamp
        t_map.header.frame_id = self.map_frame
        t_map.child_frame_id = 'odom'
        t_map.transform.translation.x = float(t_map_x)
        t_map.transform.translation.y = float(t_map_y)
        t_map.transform.translation.z = 0.0

        q_map_x, q_map_y, q_map_z, q_map_w = quaternion_from_euler(0.0, 0.0, d_yaw)
        t_map.transform.rotation.x = q_map_x
        t_map.transform.rotation.y = q_map_y
        t_map.transform.rotation.z = q_map_z
        t_map.transform.rotation.w = q_map_w
        self.tf_broadcaster.sendTransform(t_map)

    def odom_callback(self, msg: Odometry):
        """Update robot pose and path history from wheel odometry fused with IMU heading."""
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, odom_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        if not self.has_odom:
            self.robot_x = float(np.clip(odom_x, self.arena_min_x, self.arena_max_x))
            self.robot_y = float(np.clip(odom_y, self.arena_min_y, self.arena_max_y))
            self.last_odom_x = odom_x
            self.last_odom_y = odom_y
            self.last_odom_yaw = odom_yaw
            if self.use_imu_yaw and self.has_imu and self.imu_valid:
                self.robot_yaw = self.imu_yaw
            else:
                self.robot_yaw = odom_yaw
            self.has_odom = True
            if not self.robot_path:
                self.robot_path.append((self.robot_x, self.robot_y))
            self._publish_fused_odom(
                msg.header.stamp,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.angular.z,
                msg.pose.covariance,
                msg.twist.covariance
            )
            return

        if self.last_odom_x is None or self.last_odom_y is None:
            self.last_odom_x = self.robot_x
            self.last_odom_y = self.robot_y
            if self.last_odom_yaw is None:
                self.last_odom_yaw = self.robot_yaw

        dx = odom_x - self.last_odom_x
        dy = odom_y - self.last_odom_y
        self.last_odom_x = odom_x
        self.last_odom_y = odom_y

        if self.is_turned_over:
            # Robot is upside down or flipped on its side; physical wheels have no ground traction.
            # Strictly zero all dead reckoning advance to prevent phantom motion.
            self.last_odom_yaw = odom_yaw
            self._publish_fused_odom(
                msg.header.stamp, 0.0, 0.0, 0.0,
                msg.pose.covariance, msg.twist.covariance
            )
            return

        if self.use_imu_yaw and self.has_imu:
            if self.imu_valid:
                self.robot_yaw = self.imu_yaw

            # Incremental displacement step from wheel odometry step
            # Determine direction (forward vs reverse) by projecting step onto odom heading
            ref_yaw = odom_yaw
            fwd_proj = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
            sign = -1.0 if fwd_proj < 0.0 else 1.0
            delta_s = sign * math.hypot(dx, dy)

            # Odometry Slip Runaway Clamping:
            # When wheels slip against obstacle/ramp and odom step jumps (> 2.5 m/s at 50Hz),
            # clamp delta_s to physically realistic displacement step.
            if self.clamp_odometry_slip and abs(delta_s) > self.max_step_delta_s:
                delta_s = math.copysign(min(abs(delta_s), 0.25 / 50.0 * 1.5), delta_s)

            if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
                now_sec = self.get_clock().now().nanoseconds * 1e-9
            else:
                now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # Wheel acceleration estimation from discrete odometry
            dt_odom = (now_sec - self.last_odom_time) if self.last_odom_time is not None else 0.02
            if not (0.001 <= dt_odom <= 0.25):
                dt_odom = 0.02
            v_wheel = delta_s / dt_odom
            self.wheel_accel_x = (v_wheel - self.last_wheel_v) / dt_odom
            self.last_wheel_v = v_wheel
            self.last_odom_time = now_sec

            # Check IR proximity across all 8 IR ring sensors
            min_ir_range = float('inf')
            for i in range(8):
                scan = self.latest_ir[i]
                if scan is not None and scan.ranges:
                    for r in scan.ranges:
                        if (not math.isnan(r) and not math.isinf(r) and
                                scan.range_min <= r <= scan.range_max):
                            if r < min_ir_range:
                                min_ir_range = r

            # Check front sensor range (IR 0 at +10°, IR 7 at -10°)
            min_front_range = float('inf')
            for i in (0, 7):
                scan = self.latest_ir[i]
                if scan is not None and scan.ranges:
                    for r in scan.ranges:
                        if (not math.isnan(r) and not math.isinf(r) and
                                scan.range_min <= r <= scan.range_max):
                            if r < min_front_range:
                                min_front_range = r

            if min_front_range < float('inf'):
                self.front_range_history.append((now_sec, min_front_range, delta_s))

            effective_cmd_x = self.cmd_linear_x
            if (now_sec - self.last_cmd_vel_time) > 0.5 and self.last_cmd_vel_time > 0.0:
                effective_cmd_x = 0.0

            # 1. Front IR Gradient vs. Encoder Delta Check (Multi-sensor rate-of-change)
            is_ir_gradient_slip = False
            if (self.enable_ir_gradient_slip_check and delta_s > self.ir_gradient_min_delta_s and
                    min_front_range <= self.ir_gradient_proximity_thresh):
                window_old = None
                for t_entry, d_entry, s_entry in self.front_range_history:
                    if (now_sec - t_entry) >= self.ir_gradient_window_s:
                        window_old = (t_entry, d_entry)
                        break
                if window_old is not None:
                    delta_d_sensor = min_front_range - window_old[1]
                    if (effective_cmd_x > self.ir_gradient_min_cmd_v or delta_s > 0.002):
                        if delta_d_sensor >= -self.ir_gradient_max_closing_rate:
                            is_ir_gradient_slip = True

            # 2. IMU Accelerometer Sanity Gate
            is_imu_accel_slip = False
            has_imu_ready = (
                self.enable_imu_accel_gate and self.has_imu and
                self.imu_valid and self.last_imu_time is not None
            )
            near_imu_obs = (
                min_ir_range <= self.imu_accel_proximity_thresh or
                min_front_range <= self.imu_accel_proximity_thresh
            )
            if has_imu_ready and near_imu_obs:
                if delta_s > 0.001 and self.wheel_accel_x > 0.50:
                    if self.ax_dynamic_filtered < self.imu_accel_gate_threshold:
                        is_imu_accel_slip = True

            self.is_wheel_slipping = bool(is_ir_gradient_slip or is_imu_accel_slip)

            # Raw displacement vector along fused IMU yaw
            dx_raw = delta_s * math.cos(self.robot_yaw)
            dy_raw = delta_s * math.sin(self.robot_yaw)

            is_near_contact = (min_ir_range < self.contact_dist_thresh)
            self.is_near_contact = bool(is_near_contact)
            is_recent_impact = (now_sec - self.last_impact_time < 0.25)
            is_stalled_proximity = (
                min_ir_range <= self.slip_dist_thresh and
                (effective_cmd_x > 0.05 or delta_s > 0.001)
            )

            # Surface normal and tangential projection
            needs_projection = (
                is_near_contact or is_recent_impact or
                is_stalled_proximity or self.is_wheel_slipping
            )
            if needs_projection:
                normal = self._compute_surface_normal()

                if normal is not None:
                    nx, ny = normal
                    tx = -ny
                    ty = nx

                    d_par_raw = dx_raw * tx + dy_raw * ty
                    p_into_raw = -(dx_raw * nx + dy_raw * ny)
                    p_into = p_into_raw

                    if p_into > 0.0:
                        # Strictly nullify perpendicular penetration
                        total_disp = math.hypot(dx_raw, dy_raw)
                        tangent_ratio = (
                            abs(d_par_raw) / total_disp if total_disp > 1e-6 else 0.0
                        )

                        # Head-on stall: clamp advance to zero within <= 0.03m of contact
                        # When pushing straight into a wall (tangent_ratio < 0.18,
                        # within ~10 deg of head-on)
                        if tangent_ratio < 0.18:
                            delta_px = 0.0
                            delta_py = 0.0
                        else:
                            # Glancing angle: smooth sliding along wall without sticking
                            if tangent_ratio >= 0.45:
                                slip_factor = 1.0
                            else:
                                u = (tangent_ratio - 0.18) / (0.45 - 0.18)
                                slip_factor = u * u * (3.0 - 2.0 * u)
                            eff_tangent = d_par_raw * slip_factor
                            delta_px = eff_tangent * tx
                            delta_py = eff_tangent * ty
                    else:
                        # Moving away from or parallel to surface (e.g. reverse motion
                        # or tangential sliding)
                        delta_px = dx_raw
                        delta_py = dy_raw
                else:
                    # Near contact, impact, or slip but no normal found:
                    # Never unconditionally zero displacement without confirmed surface contact!
                    if (is_near_contact or is_recent_impact) and (
                        min_ir_range < self.contact_dist_thresh
                    ):
                        if delta_s > 0.0:
                            delta_px = 0.0
                            delta_py = 0.0
                        else:
                            delta_px = dx_raw
                            delta_py = dy_raw
                    elif self.is_wheel_slipping and min_ir_range < self.slip_dist_thresh:
                        # Scaled progression during close unoriented slip
                        delta_px = dx_raw * 0.20
                        delta_py = dy_raw * 0.20
                    else:
                        delta_px = dx_raw
                        delta_py = dy_raw
            else:
                delta_px = dx_raw
                delta_py = dy_raw

            # Axis-aligned arena perimeter wall protection:
            # Strictly prevent motion penetrating into outer arena walls,
            # while fully permitting motion parallel to or away from walls and corners.
            if self.robot_x >= (self.arena_max_x - 0.01) and delta_px > 0.0:
                delta_px = 0.0
            elif self.robot_x <= (self.arena_min_x + 0.01) and delta_px < 0.0:
                delta_px = 0.0

            if self.robot_y >= (self.arena_max_y - 0.01) and delta_py > 0.0:
                delta_py = 0.0
            elif self.robot_y <= (self.arena_min_y + 0.01) and delta_py < 0.0:
                delta_py = 0.0

            new_x = self.robot_x + delta_px
            new_y = self.robot_y + delta_py
        else:
            self.robot_yaw = odom_yaw
            new_x = odom_x
            new_y = odom_y
            delta_px = dx
            delta_py = dy

        self.last_odom_yaw = odom_yaw

        # Outer arena perimeter walls hard-clamping: [-2.405, +2.405] m
        self.robot_x = float(np.clip(new_x, self.arena_min_x, self.arena_max_x))
        self.robot_y = float(np.clip(new_y, self.arena_min_y, self.arena_max_y))

        if not self.robot_path:
            self.robot_path.append((self.robot_x, self.robot_y))
        else:
            last_x, last_y = self.robot_path[-1]
            dist_sq = (self.robot_x - last_x) ** 2 + (self.robot_y - last_y) ** 2
            if dist_sq >= (0.02 * 0.02):
                self.robot_path.append((self.robot_x, self.robot_y))

        # Publish corrected, slip-projected, wall-clamped odometry on /khepera4/odom
        disp_allowed = math.hypot(delta_px, delta_py)
        disp_raw = math.hypot(dx, dy)
        if disp_raw > 1e-6:
            scale = min(1.0, disp_allowed / disp_raw)
        else:
            scale = 1.0 if disp_allowed == 0.0 else 0.0

        filtered_vx = msg.twist.twist.linear.x * scale
        filtered_vy = msg.twist.twist.linear.y * scale
        filtered_wz = msg.twist.twist.angular.z

        self._publish_fused_odom(
            msg.header.stamp,
            filtered_vx,
            filtered_vy,
            filtered_wz,
            msg.pose.covariance,
            msg.twist.covariance
        )

    def imu_callback(self, msg: Imu):
        """Extract yaw heading, terrain inclination, and roughness from vibration."""
        # 1. Orientation / Heading Fusion from IMU quaternion
        q = msg.orientation
        norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if not (math.isnan(norm_sq) or norm_sq < 1e-4 or
                math.isnan(q.x) or math.isnan(q.y) or math.isnan(q.z) or math.isnan(q.w)):
            _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            if not (math.isnan(yaw) or math.isinf(yaw)):
                self.imu_yaw = yaw
                self.has_imu = True
                self.imu_valid = True
                if self.use_imu_yaw:
                    self.robot_yaw = self.imu_yaw
        else:
            self.imu_valid = False

        # 2. Extract terrain inclination from gravity vector and roughness from vibration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        roll = math.atan2(ay, az if az != 0 else 0.001)
        total_slope_rad = math.sqrt(pitch * pitch + roll * roll)
        self.current_slope_deg = min(100.0, math.degrees(total_slope_rad))

        # 2b. Rollover / Turned Over Detection via Accelerometer & Gravity Vector
        # An upright robot on flat ground experiences proper acceleration az ~ 9.81 m/s^2.
        # When turned upside down or rolled onto its side (e.g. falling off a ramp), gravity
        # is not pulling in the correct direction (az <= -1.0 m/s^2 or severe tilt > 70 deg).
        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a > 1.0:
            z_gravity_ratio = az / norm_a
            is_flipped_accel = (
                (az < -1.0) or
                (az < 1.0 and self.current_slope_deg > 45.0) or
                (z_gravity_ratio < 0.17 and self.current_slope_deg > 45.0)
            )
        else:
            is_flipped_accel = False
        self.is_turned_over = bool(is_flipped_accel or (self.current_slope_deg > 70.0))

        # NOTE FOR FUTURE SWARM COORDINATION:
        # From this rollover state variable, a distress signal can be created to
        # notify neighboring robots in the swarm for search, rescue, or re-routing.
        # (Do not implement the distress signal transmission logic yet per project requirements).
        if hasattr(self, 'turned_over_pub'):
            self.turned_over_pub.publish(Bool(data=self.is_turned_over))

        self.accel_z_buffer.append(az)
        if len(self.accel_z_buffer) >= 10:
            std_dev = float(np.std(self.accel_z_buffer))
            self.current_roughness = int(np.clip(std_dev * 100.0, 0, 85))

        # 3. IMU linear accel & jerk impact detection (||j|| > 40 m/s^3 or ax_dynamic < -1.2 m/s^2)
        if msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0:
            now_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            now_t = self.get_clock().now().nanoseconds * 1e-9

        # Gravity compensation for body X-axis dynamic forward acceleration:
        # Accelerometer measures specific force a_meas = a_dynamic - g_body.
        # On a pitch incline theta, g_x = -g * sin(theta), so a_meas = a_dynamic + g * sin(pitch).
        if self.imu_valid:
            sin_p = 2.0 * (q.w * q.y - q.z * q.x)
            sin_p = max(-1.0, min(1.0, sin_p))
            q_pitch = math.asin(sin_p)
            ax_dynamic = ax + 9.80665 * math.sin(q_pitch)
        else:
            ax_dynamic = ax

        self.ax_dynamic_filtered = 0.35 * ax_dynamic + 0.65 * self.ax_dynamic_filtered

        if self.last_imu_time is not None:
            dt = now_t - self.last_imu_time
            if 0.001 <= dt <= 0.2:
                jx = (ax - self.last_ax) / dt
                jy = (ay - self.last_ay) / dt
                self.imu_jerk = math.hypot(jx, jy)
            else:
                self.imu_jerk = 0.0
        else:
            self.imu_jerk = 0.0

        self.last_ax = ax
        self.last_ay = ay
        self.last_imu_time = now_t

        effective_cmd_x = self.cmd_linear_x
        if (now_t - self.last_cmd_vel_time) > 0.5 and self.last_cmd_vel_time > 0.0:
            effective_cmd_x = 0.0

        if self.imu_jerk > 40.0 or (ax_dynamic < -1.2 and effective_cmd_x > 0.05):
            self.last_impact_time = now_t

    def us_callback(self, index: int, msg: LaserScan):
        """Cache latest ultrasonic range scan."""
        self.latest_us[index] = msg

    def ir_callback(self, index: int, msg: LaserScan):
        """Cache latest infrared proximity scan."""
        self.latest_ir[index] = msg

    def ground_ir_callback(self, index: int, msg: LaserScan):
        """Monitor ground IR scan for cliff drop-off and terrain roughness."""
        self.latest_ground_ir[index] = msg

        any_cliff = False
        for k in range(4):
            scan = self.latest_ground_ir[k]
            if scan is not None and len(scan.ranges) > 0:
                dist = scan.ranges[0]
                if math.isinf(dist) or math.isnan(dist) or dist > self.cliff_thresh:
                    any_cliff = True
                    if self.has_odom:
                        gx, gy = GROUND_IR_MOUNTS[k]
                        wx = (self.robot_x + gx * math.cos(self.robot_yaw) -
                              gy * math.sin(self.robot_yaw))
                        wy = (self.robot_y + gx * math.sin(self.robot_yaw) +
                              gy * math.cos(self.robot_yaw))
                        c, r = self.world_to_grid(wx, wy)
                        if self.is_valid_cell(c, r):
                            self.surface_grid[r, c] = 100

        self.cliff_detected = any_cliff
        alert_msg = Bool()
        alert_msg.data = any_cliff
        self.cliff_alert_pub.publish(alert_msg)

    # ==========================================================================
    # Coordinate Transformations & Grid Helpers
    # ==========================================================================

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world metric coordinates (x, y) to grid cell (col, row)."""
        col = int(math.floor((wx - self.origin_x) / self.res))
        row = int(math.floor((wy - self.origin_y) / self.res))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell (col, row) to world metric coordinates (x, y)."""
        wx = self.origin_x + (col + 0.5) * self.res
        wy = self.origin_y + (row + 0.5) * self.res
        return wx, wy

    def is_valid_cell(self, col: int, row: int) -> bool:
        """Check if grid cell (col, row) lies within bounds."""
        return 0 <= col < self.width and 0 <= row < self.height

    # ==========================================================================
    # Task 1.a: 360° Virtual LaserScan Fusion
    # ==========================================================================

    def publish_virtual_scan(self):
        """Combine 8 IR and 5 US scans into a unified 360-sample 2D scan on base_footprint."""
        num_samples = 360
        angle_min = -math.pi
        angle_max = math.pi - (2.0 * math.pi / num_samples)
        angle_inc = 2.0 * math.pi / num_samples

        ranges = [float('inf')] * num_samples
        now = self.get_clock().now().to_msg()

        # 1. Process 8 Infrared Proximity Sensors (0.002m - 0.25m close range)
        for i, mount in enumerate(IR_MOUNTS):
            scan = self.latest_ir[i]
            if scan is None:
                continue
            sx, sy, syaw = mount
            for j, r in enumerate(scan.ranges):
                if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                    continue
                ray_angle = scan.angle_min + j * scan.angle_increment
                px = sx + r * math.cos(syaw + ray_angle)
                py = sy + r * math.sin(syaw + ray_angle)
                dist = math.hypot(px, py)
                angle = math.atan2(py, px)

                idx = int(round((angle - angle_min) / angle_inc)) % num_samples
                if dist < ranges[idx]:
                    ranges[idx] = dist

        # Find minimum detected obstacle distance from IR sensors in robot frame
        min_ir_dist = min([r for r in ranges if not math.isinf(r)], default=2.0)

        # 2. Process 5 Ultrasonic Sensors (0.02m - 2.0m medium range)
        for i, mount in enumerate(US_MOUNTS):
            scan = self.latest_us[i]
            if scan is None:
                continue
            sx, sy, syaw = mount
            for j, r in enumerate(scan.ranges):
                if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                    continue
                ray_angle = scan.angle_min + j * scan.angle_increment
                px = sx + r * math.cos(syaw + ray_angle)
                py = sy + r * math.sin(syaw + ray_angle)
                dist = math.hypot(px, py)
                angle = math.atan2(py, px)

                idx = int(round((angle - angle_min) / angle_inc)) % num_samples

                # Reject ultrasonic hits penetrating obstacles detected at close range by IR
                if min_ir_dist < 0.15 and dist > (min_ir_dist + 0.15):
                    continue

                # Verify hit does not lie outside physical arena bounds
                if self.has_odom:
                    hit_wx = self.robot_x + dist * math.cos(self.robot_yaw + angle)
                    hit_wy = self.robot_y + dist * math.sin(self.robot_yaw + angle)
                    out_x = (
                        hit_wx < (self.arena_min_x - 0.05) or
                        hit_wx > (self.arena_max_x + 0.05)
                    )
                    out_y = (
                        hit_wy < (self.arena_min_y - 0.05) or
                        hit_wy > (self.arena_max_y + 0.05)
                    )
                    if out_x or out_y:
                        continue

                if dist < ranges[idx]:
                    ranges[idx] = dist

        scan_msg = LaserScan()
        scan_msg.header.stamp = now
        scan_msg.header.frame_id = self.robot_frame
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = angle_inc
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / float(self.get_parameter('scan_publish_rate').value)
        scan_msg.range_min = 0.002
        scan_msg.range_max = 2.0
        scan_msg.ranges = ranges

        self.scan_pub.publish(scan_msg)

    # ==========================================================================
    # Task 1.b, 1.c, 1.d: Map Updates & Publishing
    # ==========================================================================
    # Phase 2: 2D Correlative Scan Matching (CSM) & Pose Correction
    # ==========================================================================

    def _collect_scan_endpoints(self) -> List[Tuple[float, float]]:
        """Collect valid obstacle hit points in base_footprint frame from IR and US sensors."""
        endpoints = []
        for i, mount in enumerate(IR_MOUNTS):
            scan = self.latest_ir[i]
            if scan is None or not scan.ranges:
                continue
            sx, sy, syaw = mount
            for j, r in enumerate(scan.ranges):
                valid_r = not math.isnan(r) and not math.isinf(r)
                if valid_r and scan.range_min <= r <= scan.range_max:
                    ray_ang = syaw + (scan.angle_min + j * scan.angle_increment)
                    px = sx + r * math.cos(ray_ang)
                    py = sy + r * math.sin(ray_ang)
                    endpoints.append((px, py))

        for i, mount in enumerate(US_MOUNTS):
            scan = self.latest_us[i]
            if scan is None or not scan.ranges:
                continue
            sx, sy, syaw = mount
            for j, r in enumerate(scan.ranges):
                valid_r = not math.isnan(r) and not math.isinf(r)
                if valid_r and scan.range_min <= r <= scan.range_max:
                    if r > 1.2:
                        continue
                    ray_ang = syaw + (scan.angle_min + j * scan.angle_increment)
                    px = sx + r * math.cos(ray_ang)
                    py = sy + r * math.sin(ray_ang)
                    endpoints.append((px, py))
        return endpoints

    def _compute_pose_quality(self) -> float:
        """
        Compute continuous pose quality score Q in [0.15, 1.0].

        Downweights obstacle reinforcement during wheel slip or contact,
        preventing phantom/ghost obstacles from establishing solid belief.
        """
        penalty = 0.0
        if self.is_wheel_slipping:
            penalty += 0.70
        if self.is_near_contact:
            penalty += 0.35
        if abs(self.wheel_accel_x) > 0.50:
            penalty += min(0.30, (abs(self.wheel_accel_x) - 0.50) / 5.0 * 0.30)
        return float(np.clip(1.0 - penalty, 0.15, 1.0))

    def refine_pose_with_scan_matching(self) -> bool:
        """Refine robot pose via 2D Correlative Scan Matching with anti-jitter smoothing."""
        if not self.enable_scan_matching or not self.has_odom:
            return False

        # Do not run CSM during active wheel slip or collision contact:
        # Prevents matching against phantom obstacles and feedback loops
        if self.is_wheel_slipping or self.is_near_contact:
            return False

        endpoints = self._collect_scan_endpoints()
        if len(endpoints) < self.csm_min_valid_hits:
            return False

        # Only match if map contains sufficient confirmed obstacle cells
        if np.count_nonzero(self.log_odds >= self.l_occ) < 10:
            return False

        seed_x, seed_y, seed_yaw = self.robot_x, self.robot_y, self.robot_yaw
        pts_b = np.array(endpoints, dtype=np.float32)

        d_trans = self.csm_search_dist
        step_trans = self.csm_trans_step

        # When IMU heading is valid, lock yaw search to 0 to prevent 50Hz/3Hz heading war & jitter
        has_stable_imu = bool(self.use_imu_yaw and self.has_imu and self.imu_valid)
        if has_stable_imu:
            dyaw_vals = np.array([0.0], dtype=np.float32)
        else:
            d_yaw = self.csm_search_angle
            step_yaw = self.csm_angle_step
            dyaw_vals = np.arange(-d_yaw, d_yaw + 1e-5, step_yaw, dtype=np.float32)

        dx_vals = np.arange(-d_trans, d_trans + 1e-5, step_trans, dtype=np.float32)
        dy_vals = np.arange(-d_trans, d_trans + 1e-5, step_trans, dtype=np.float32)

        grid_dx, grid_dy, grid_dyaw = np.meshgrid(dx_vals, dy_vals, dyaw_vals, indexing='ij')
        cand_dx = grid_dx.flatten()
        cand_dy = grid_dy.flatten()
        cand_dyaw = grid_dyaw.flatten()

        cand_x = seed_x + cand_dx
        cand_y = seed_y + cand_dy
        cand_yaw = np.array([normalize_angle(seed_yaw + y) for y in cand_dyaw], dtype=np.float32)

        cos_y = np.cos(cand_yaw)
        sin_y = np.sin(cand_yaw)

        # Vectorized coordinate transformation to world coordinates
        px_b = pts_b[:, 0][None, :]
        py_b = pts_b[:, 1][None, :]
        cos_y_k = cos_y[:, None]
        sin_y_k = sin_y[:, None]

        w_x = cand_x[:, None] + px_b * cos_y_k - py_b * sin_y_k
        w_y = cand_y[:, None] + px_b * sin_y_k + py_b * cos_y_k

        # Map to grid cell indices
        cols = np.floor((w_x - self.origin_x) / self.res).astype(np.int32)
        rows = np.floor((w_y - self.origin_y) / self.res).astype(np.int32)

        valid_mask = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        safe_cols = np.clip(cols, 0, self.width - 1)
        safe_rows = np.clip(rows, 0, self.height - 1)

        # Retrieve log-odds belief values
        cell_vals = self.log_odds[safe_rows, safe_cols]
        clipped_vals = np.clip(cell_vals, -self.l_free, self.l_occ * 2.0)
        scores_per_point = np.where(valid_mask, clipped_vals, -self.l_free)
        total_scores = np.sum(scores_per_point, axis=1)

        # Locate seed pose score (dx=0, dy=0, dyaw=0)
        seed_mask = (
            (np.abs(cand_dx) < 1e-4) &
            (np.abs(cand_dy) < 1e-4) &
            (np.abs(cand_dyaw) < 1e-4)
        )
        seed_indices = np.where(seed_mask)[0]
        seed_score = total_scores[seed_indices[0]] if len(seed_indices) > 0 else total_scores[0]

        best_idx = int(np.argmax(total_scores))
        best_score = total_scores[best_idx]

        # Analyze spatial eigenvalue conditioning of scan endpoints (aperture problem)
        # When scan points form a single line (e.g. driving along a straight wall),
        # translation along the wall is degenerate and unobservable.
        # Project displacement strictly onto the wall normal to eliminate jagged zigzag jitter!
        raw_dx = float(cand_dx[best_idx])
        raw_dy = float(cand_dy[best_idx])

        corr_dx = raw_dx
        corr_dy = raw_dy
        if len(pts_b) >= 3:
            cov = np.cov(pts_b, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            if float(eigvals[1]) > 1e-5 and (float(eigvals[0]) / float(eigvals[1])) < 0.08:
                # 1D single wall geometry: normal to the wall in robot body frame
                n_b = eigvecs[:, 0]
                cos_s = math.cos(seed_yaw)
                sin_s = math.sin(seed_yaw)
                # Rotate normal to map frame
                nx_m = float(n_b[0] * cos_s - n_b[1] * sin_s)
                ny_m = float(n_b[0] * sin_s + n_b[1] * cos_s)
                norm_m = math.hypot(nx_m, ny_m)
                if norm_m > 1e-4:
                    nx_m /= norm_m
                    ny_m /= norm_m
                    # Project displacement purely onto the wall normal; tangential drift is zeroed
                    proj = raw_dx * nx_m + raw_dy * ny_m
                    corr_dx = proj * nx_m
                    corr_dy = proj * ny_m

        disp = float(math.hypot(corr_dx, corr_dy))

        if best_score > (seed_score + self.csm_score_thresh) and disp >= self.csm_deadband:
            target_x = float(seed_x + corr_dx)
            target_y = float(seed_y + corr_dy)
            target_yaw = float(cand_yaw[best_idx])

            # Exponential moving average (alpha smoothing) to eliminate 1cm staircase jumps
            alpha = self.csm_alpha_smooth
            smoothed_x = (1.0 - alpha) * self.robot_x + alpha * target_x
            smoothed_y = (1.0 - alpha) * self.robot_y + alpha * target_y

            # Apply arena perimeter bounds
            self.robot_x = float(np.clip(smoothed_x, self.arena_min_x, self.arena_max_x))
            self.robot_y = float(np.clip(smoothed_y, self.arena_min_y, self.arena_max_y))
            if not has_stable_imu:
                self.robot_yaw = target_yaw

            # Broadcast updated TF immediately
            now_stamp = self.get_clock().now().to_msg()
            self._broadcast_map_tf(now_stamp)
            return True
        return False

    # ==========================================================================
    # Task 1.b, 1.c, 1.d: Map Updates & Publishing
    # ==========================================================================

    def update_and_publish_maps(self):
        """Update 2D occupancy grid, 2.5D elevation grid, and surface grid, then publish."""
        if not self.has_odom:
            return

        now = self.get_clock().now().to_msg()

        # Phase 2: Refine robot pose with 2D Correlative Scan Matching before painting rays
        if self.enable_scan_matching:
            self.refine_pose_with_scan_matching()

        self._update_occupancy_rays()
        self._update_footprint_maps()

        self._publish_grid(self.map_pub, self._build_occupancy_grid_data(), now)
        self._publish_grid(self.elevation_pub, self.elevation_grid.flatten().tolist(), now)
        self._publish_grid(self.surface_pub, self.surface_grid.flatten().tolist(), now)

    def _update_occupancy_rays(self):
        """Cast rays for all active IR and US sensors and update log-odds."""
        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw

        # 1. Cast IR rays
        for i, (sx, sy, syaw) in enumerate(IR_MOUNTS):
            scan = self.latest_ir[i]
            if scan is None:
                continue
            wsx = rx + sx * math.cos(ryaw) - sy * math.sin(ryaw)
            wsy = ry + sx * math.sin(ryaw) + sy * math.cos(ryaw)
            c0, r0 = self.world_to_grid(wsx, wsy)

            for j, r in enumerate(scan.ranges):
                ray_ang = ryaw + syaw + (scan.angle_min + j * scan.angle_increment)
                is_hit = (not math.isnan(r) and not math.isinf(r) and
                          scan.range_min <= r <= scan.range_max)
                end_r = r if is_hit else scan.range_max

                wex = wsx + end_r * math.cos(ray_ang)
                wey = wsy + end_r * math.sin(ray_ang)
                wex = max(self.arena_min_x, min(self.arena_max_x, wex))
                wey = max(self.arena_min_y, min(self.arena_max_y, wey))
                c1, r1 = self.world_to_grid(wex, wey)

                q_score = self._compute_pose_quality()
                cells = bresenham_line(c0, r0, c1, r1)
                for k, (col, row) in enumerate(cells):
                    if not self.is_valid_cell(col, row):
                        break
                    if is_hit and (k == len(cells) - 1):
                        delta_l = q_score * self.l_occ
                        if self.is_wheel_slipping or q_score < 0.60:
                            # Capped obstacle reinforcement during wheel slip
                            self.log_odds[row, col] = min(
                                self.l_occ, self.log_odds[row, col] + delta_l
                            )
                        else:
                            self.log_odds[row, col] = min(
                                self.l_max, self.log_odds[row, col] + delta_l
                            )
                            self.hit_count[row, col] = min(255, int(self.hit_count[row, col]) + 1)
                        break

                    # Check if cell was previously believed to be occupied
                    if self.log_odds[row, col] >= self.l_occ:
                        # Erode the cell with incoming free space observation
                        delta_free = q_score * self.l_free
                        self.log_odds[row, col] = max(
                            self.l_min, self.log_odds[row, col] - delta_free
                        )
                        if self.hit_count[row, col] > 0:
                            self.hit_count[row, col] -= 1

                        # Stop ray if cell is confirmed solid wall
                        is_solid_wall = (
                            self.hit_count[row, col] >= self.hit_count_confirm_threshold or
                            self.log_odds[row, col] >= self.l_occ
                        )
                        if is_solid_wall:
                            break
                    else:
                        delta_free = q_score * self.l_free
                        self.log_odds[row, col] = max(
                            self.l_min, self.log_odds[row, col] - delta_free
                        )

        # Collect minimum detected obstacle distance from IR sensors
        min_ir_dist = 2.0
        for scan in self.latest_ir:
            if scan is not None and scan.ranges:
                for r in scan.ranges:
                    valid_r = not math.isnan(r) and not math.isinf(r)
                    if valid_r and scan.range_min <= r <= scan.range_max:
                        if r < min_ir_dist:
                            min_ir_dist = r

        # 2. Cast Ultrasonic rays
        for i, (sx, sy, syaw) in enumerate(US_MOUNTS):
            scan = self.latest_us[i]
            if scan is None:
                continue
            wsx = rx + sx * math.cos(ryaw) - sy * math.sin(ryaw)
            wsy = ry + sx * math.sin(ryaw) + sy * math.cos(ryaw)
            c0, r0 = self.world_to_grid(wsx, wsy)

            for j, r in enumerate(scan.ranges):
                ray_ang = ryaw + syaw + (scan.angle_min + j * scan.angle_increment)
                is_hit = (not math.isnan(r) and not math.isinf(r) and
                          scan.range_min <= r <= scan.range_max)

                # Cross-sensor occlusion: if an obstacle is very close by IR,
                # reject ultrasonic rays that penetrate through it
                if min_ir_dist < 0.15 and is_hit and r > (min_ir_dist + 0.15):
                    continue
                if not is_hit and min_ir_dist < 0.15:
                    continue

                end_r = r if is_hit else min(scan.range_max, 1.2)

                wex = wsx + end_r * math.cos(ray_ang)
                wey = wsy + end_r * math.sin(ray_ang)
                wex = max(self.arena_min_x, min(self.arena_max_x, wex))
                wey = max(self.arena_min_y, min(self.arena_max_y, wey))
                c1, r1 = self.world_to_grid(wex, wey)

                q_score = self._compute_pose_quality()
                cells = bresenham_line(c0, r0, c1, r1)
                for k, (col, row) in enumerate(cells):
                    if not self.is_valid_cell(col, row):
                        break
                    if is_hit and (k == len(cells) - 1):
                        delta_l = q_score * self.l_occ
                        if self.is_wheel_slipping or q_score < 0.60:
                            # Capped obstacle reinforcement during wheel slip
                            self.log_odds[row, col] = min(
                                self.l_occ, self.log_odds[row, col] + delta_l
                            )
                        else:
                            self.log_odds[row, col] = min(
                                self.l_max, self.log_odds[row, col] + delta_l
                            )
                            self.hit_count[row, col] = min(255, int(self.hit_count[row, col]) + 1)
                        break

                    # Check if cell was previously believed to be occupied
                    if self.log_odds[row, col] >= self.l_occ:
                        # Erode the cell with incoming free space observation
                        delta_free = q_score * self.l_free
                        self.log_odds[row, col] = max(
                            self.l_min, self.log_odds[row, col] - delta_free
                        )
                        if self.hit_count[row, col] > 0:
                            self.hit_count[row, col] -= 1

                        # Stop ray if cell is confirmed solid wall
                        is_solid_wall = (
                            self.hit_count[row, col] >= self.hit_count_confirm_threshold or
                            self.log_odds[row, col] >= self.l_occ
                        )
                        if is_solid_wall:
                            break
                    else:
                        delta_free = q_score * self.l_free
                        self.log_odds[row, col] = max(
                            self.l_min, self.log_odds[row, col] - delta_free
                        )

    def _update_footprint_maps(self):
        """Update elevation map, roughness map, and carve free space under robot footprint."""
        cr, rr = self.world_to_grid(self.robot_x, self.robot_y)
        robot_radius_cells = int(math.ceil(0.070 / self.res))
        # 60 mm inner radius protects perimeter walls during glancing scrapes
        carve_radius_sq = (0.060) ** 2

        slope_val = int(np.clip(round(self.current_slope_deg), 0, 100))
        roughness_val = self.current_roughness

        slip_contact = self.is_wheel_slipping and self.is_near_contact
        can_carve = self.enable_footprint_carving and not slip_contact

        for dcol in range(-robot_radius_cells, robot_radius_cells + 1):
            for drow in range(-robot_radius_cells, robot_radius_cells + 1):
                dist_sq = (dcol * self.res) ** 2 + (drow * self.res) ** 2
                if dist_sq <= (0.070 ** 2):
                    c = cr + dcol
                    r = rr + drow
                    if self.is_valid_cell(c, r):
                        self.elevation_grid[r, c] = slope_val
                        if self.surface_grid[r, c] != 100:
                            self.surface_grid[r, c] = roughness_val
                        if can_carve and dist_sq <= carve_radius_sq:
                            # Erase any phantom / ghost wall that the robot physically traverses
                            if self.log_odds[r, c] > -self.l_free:
                                self.log_odds[r, c] = -self.l_free
                            self.hit_count[r, c] = 0
                            self.surface_grid[r, c] = roughness_val

    def _build_occupancy_grid_data(self) -> List[int]:
        """Convert log-odds belief matrix into ROS occupancy grid standard values."""
        grid_data = np.full((self.height, self.width), -1, dtype=np.int8)
        grid_data[self.log_odds > 0.5] = 100   # Occupied
        grid_data[self.log_odds < -0.4] = 0    # Free
        return grid_data.flatten().tolist()

    def _publish_grid(self, publisher, data: List[int], stamp):
        """Construct and publish a nav_msgs/OccupancyGrid message."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = stamp
        msg.info.resolution = self.res
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        publisher.publish(msg)

    def visual_image_callback(self, msg: RosImage):
        """Update cached high-res visual orthomap from /visual_map/image."""
        try:
            h, w = msg.height, msg.width
            if h == self.height and w == self.width:
                if msg.encoding == 'rgba8':
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 4))
                    self.visual_grid = arr[:, :, :3].copy()
                    self.has_visual_map = True
                elif msg.encoding == 'rgb8':
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                    self.visual_grid = arr.copy()
                    self.has_visual_map = True
        except Exception as e:
            self.get_logger().debug(f'Error unpacking /visual_map/image: {e}')

    def visual_grid_callback(self, msg: OccupancyGrid):
        """Fallback updater for visual map from /visual_map (OccupancyGrid)."""
        if not self.has_visual_map:
            try:
                h, w = msg.info.height, msg.info.width
                if h == self.height and w == self.width:
                    data = np.array(msg.data, dtype=np.int8).reshape((h, w))
                    obs_mask = (data >= 0)
                    gray = np.clip(
                        np.round(data.astype(np.float32) * (255.0 / 100.0)),
                        0,
                        255
                    ).astype(np.uint8)
                    v_rgb = np.full((h, w, 3), 200, dtype=np.uint8)
                    v_rgb[obs_mask] = np.repeat(gray[obs_mask, np.newaxis], 3, axis=1)
                    self.visual_grid = v_rgb
            except Exception as e:
                self.get_logger().debug(f'Error unpacking /visual_map: {e}')

    # ==========================================================================
    # Task 1.e: Map Persistence & Memory (save, load, clear)
    # ==========================================================================

    def save_map_callback(self, request, response):
        """Save map as ROS standard PGM + YAML, color PNG, and composite visual map."""
        try:
            save_dir = os.path.abspath(self.map_save_dir)
            os.makedirs(save_dir, exist_ok=True)

            pgm_path = os.path.join(save_dir, 'khepera_map.pgm')
            yaml_path = os.path.join(save_dir, 'khepera_map.yaml')
            png_path = os.path.join(save_dir, 'khepera_map.png')
            visual_map_path = os.path.join(save_dir, 'khepera_visual_map.png')

            # 1. Generate ROS PGM file
            grid = np.full((self.height, self.width), 205, dtype=np.uint8)
            grid[self.log_odds < -0.4] = 254
            grid[self.log_odds > 0.5] = 0
            grid_flipped = np.flipud(grid)

            header = f'P5\n{self.width} {self.height}\n255\n'.encode('ascii')
            with open(pgm_path, 'wb') as f:
                f.write(header)
                f.write(grid_flipped.tobytes())

            # 2. Generate ROS YAML metadata file
            yaml_content = (
                f'image: khepera_map.pgm\n'
                f'resolution: {self.res:.4f}\n'
                f'origin: [{self.origin_x:.4f}, {self.origin_y:.4f}, 0.0]\n'
                f'negate: 0\n'
                f'occupied_thresh: 0.65\n'
                f'free_thresh: 0.196\n'
                f'mode: trinary\n'
            )
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)

            # 3. Generate color-coded PNG visualization
            img_rgb = np.full((self.height, self.width, 3), 215, dtype=np.uint8)
            img_rgb[self.log_odds < -0.4] = [255, 255, 255]
            img_rgb[self.log_odds > 0.5] = [20, 20, 20]

            slope_mask = (self.elevation_grid > 5) & (self.log_odds < -0.4)
            img_rgb[slope_mask] = [255, 180, 50]

            cliff_mask = (self.surface_grid == 100)
            img_rgb[cliff_mask] = [240, 20, 20]

            img_flipped = np.flipud(img_rgb)
            pil_img = Image.fromarray(img_flipped)
            draw = ImageDraw.Draw(pil_img)

            if len(self.robot_path) >= 2:
                img_pts = []
                for wx, wy in self.robot_path:
                    col, row = self.world_to_grid(wx, wy)
                    img_pts.append((col, self.height - 1 - row))
                draw.line(img_pts, fill=(0, 120, 255), width=2)

            if self.has_odom:
                cur_c, cur_r = self.world_to_grid(self.robot_x, self.robot_y)
                cur_y_img = self.height - 1 - cur_r
                r_rad = 3
                draw.ellipse(
                    [cur_c - r_rad, cur_y_img - r_rad, cur_c + r_rad, cur_y_img + r_rad],
                    fill=(0, 220, 80), outline=(0, 100, 40)
                )
                hx = cur_c + int(6 * math.cos(self.robot_yaw))
                hy = cur_y_img - int(6 * math.sin(self.robot_yaw))
                draw.line([(cur_c, cur_y_img), (hx, hy)], fill=(0, 100, 40), width=2)

            pil_img.save(png_path)

            # 4. Generate composite high-resolution visual map (khepera_visual_map.png)
            kf_dir = os.path.join(save_dir, 'keyframes')
            manifest_path = os.path.join(kf_dir, 'manifest.json')
            keyframes = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        m_data = json.load(f)
                        keyframes = m_data.get('keyframes', [])
                except Exception as e:
                    self.get_logger().warn(f'Failed to load keyframe manifest: {e}')

            generate_composite_map(
                visual_map_path,
                self.visual_grid,
                self.log_odds,
                self.elevation_grid,
                self.surface_grid,
                self.robot_path,
                keyframes,
                res=self.res,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
                current_pose=(
                    (self.robot_x, self.robot_y, self.robot_yaw) if self.has_odom else None
                )
            )

            self.get_logger().info(f'Successfully saved persistent map files to {save_dir}/')
            response.success = True
            response.message = f'Map successfully saved to {save_dir}/ (pgm, yaml, png)'
        except Exception as e:
            self.get_logger().error(f'Failed to save map: {str(e)}')
            response.success = False
            response.message = f'Error saving map: {str(e)}'
        return response

    def load_map_callback(self, request, response):
        """Reload previously saved map from PGM/YAML into persistent memory."""
        try:
            save_dir = os.path.abspath(self.map_save_dir)
            pgm_path = os.path.join(save_dir, 'khepera_map.pgm')

            if not os.path.exists(pgm_path):
                response.success = False
                response.message = f'Map file not found: {pgm_path}'
                return response

            with open(pgm_path, 'rb') as f:
                magic = f.readline().decode('ascii').strip()
                if magic != 'P5':
                    raise ValueError(f'Unsupported PGM format: {magic}')
                line = f.readline().decode('ascii').strip()
                while line.startswith('#'):
                    line = f.readline().decode('ascii').strip()
                w, h = map(int, line.split())
                _ = int(f.readline().decode('ascii').strip())
                raw_bytes = f.read()

            if len(raw_bytes) != w * h:
                raise ValueError(f'Byte count mismatch: expected {w*h}, got {len(raw_bytes)}')

            data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
            data_restored = np.flipud(data)

            new_log_odds = np.zeros((self.height, self.width), dtype=np.float32)
            new_log_odds[data_restored == 254] = self.l_min  # Free
            new_log_odds[data_restored == 0] = self.l_max    # Occupied

            self.log_odds = new_log_odds
            self.hit_count.fill(0)
            self.hit_count[new_log_odds == self.l_max] = 20

            now = self.get_clock().now().to_msg()
            self._publish_grid(self.map_pub, self._build_occupancy_grid_data(), now)

            self.get_logger().info(f'Loaded persistent map from {pgm_path}')
            response.success = True
            response.message = f'Successfully loaded map from {pgm_path}'
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {str(e)}')
            response.success = False
            response.message = f'Error loading map: {str(e)}'
        return response

    def clear_map_callback(self, request, response):
        """Reset map, elevation map, surface map, trajectory, and keyframes."""
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.hit_count.fill(0)
        self.elevation_grid = np.full((self.height, self.width), -1, dtype=np.int8)
        self.surface_grid = np.full((self.height, self.width), -1, dtype=np.int8)
        self.visual_grid = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        self.has_visual_map = False
        if self.has_odom:
            self.robot_path = [(self.robot_x, self.robot_y)]
        else:
            self.robot_path = []

        now = self.get_clock().now().to_msg()
        self._publish_grid(self.map_pub, self._build_occupancy_grid_data(), now)
        self._publish_grid(self.elevation_pub, self.elevation_grid.flatten().tolist(), now)
        self._publish_grid(self.surface_pub, self.surface_grid.flatten().tolist(), now)

        # Clean maps/keyframes directory and reset manifest
        try:
            save_dir = os.path.abspath(self.map_save_dir)
            kf_dir = os.path.join(save_dir, 'keyframes')
            if os.path.exists(kf_dir):
                for fname in os.listdir(kf_dir):
                    if fname.endswith('.png'):
                        try:
                            os.remove(os.path.join(kf_dir, fname))
                        except Exception:
                            pass
            else:
                os.makedirs(kf_dir, exist_ok=True)

            manifest_path = os.path.join(kf_dir, 'manifest.json')
            empty_manifest = {
                'map_frame': self.map_frame,
                'manifest_version': '1.0',
                'total_keyframes': 0,
                'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'keyframes': []
            }
            tmp_path = manifest_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(empty_manifest, f, indent=2)
            os.replace(tmp_path, manifest_path)
        except Exception as e:
            self.get_logger().warn(f'Failed to clear keyframe directory: {e}')

        # Notify visual_orthomapper service if available
        try:
            cli = getattr(self, 'clear_visual_map_cli', None)
            if cli is not None and cli.service_is_ready():
                cli.call_async(Trigger.Request())
        except Exception:
            pass

        self.get_logger().info('Map memory and keyframe manifest cleared.')
        response.success = True
        response.message = 'Maps reset and cleared successfully.'
        return response

    def recenter_robot_callback(self, request, response):
        """Respawn Khepera IV at arena center rightside up in Gazebo and reset mapper state."""
        self.get_logger().info(
            'Recenter service triggered: respawning robot at arena center rightside up.'
        )

        # 1. Teleport model in Gazebo Harmonic via UserCommands set_pose service
        pose_req = (
            'name: "khepera4", position: {x: 0.0, y: 0.0, z: 0.03}, '
            'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}'
        )
        for world_name in ['khepera4_world', 'default']:
            try:
                cmd = [
                    'gz', 'service', '-s', f'/world/{world_name}/set_pose',
                    '--reqtype', 'gz.msgs.Pose',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '1500',
                    '--req', pose_req
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
                if 'data: true' in res.stdout:
                    self.get_logger().info(
                        f'Gazebo set_pose succeeded on /world/{world_name}/set_pose'
                    )
                    break
            except Exception as e:
                self.get_logger().debug(f'Gazebo set_pose on {world_name} note: {e}')

        # 2. Reset internal mapper state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None
        self.is_turned_over = False
        self.robot_path = [(0.0, 0.0)]

        # 3. Publish reset status and zero-velocity odometry
        if hasattr(self, 'turned_over_pub'):
            self.turned_over_pub.publish(Bool(data=False))
        now_stamp = self.get_clock().now().to_msg()
        self._publish_fused_odom(now_stamp, 0.0, 0.0, 0.0)

        response.success = True
        response.message = 'Khepera IV successfully recentered at origin (0, 0) rightside up.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Khepera4Mapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
