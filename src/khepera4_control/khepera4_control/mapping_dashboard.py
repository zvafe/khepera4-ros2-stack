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
Khepera IV Dedicated Mapping & Spatial Exploration Dashboard.

Specialized Single-Robot Control Station for 2D/2.5D SLAM, Autonomous Frontier Exploration,
and Hazard Monitoring.

Features:
- Single-robot direct control: /cmd_vel, /odom, /imu/data, /khepera4/scan
- Clickable tactile Interactive D-Pad + Smooth Keyboard WASD Teleoperation
- Autonomous Exploration Toggle (/set_explorer, /toggle_explorer, /explorer/enable)
- Attitude & Terrain HUD: Pitch, Roll, Terrain Slope, and Cliff Hazard Warning
- Real-time Multi-Layer Spatial Map Canvas:
  * 2D Probabilistic Obstacle Grid (/map)
  * 2.5D Terrain Slope Elevation Map (/elevation_map)
  * Surface Classification & Cliff Hazard Map (/surface_map)
  * Ground-Plane Visual Orthomap (/visual_map)
  * Real-time Khepera IV footprint, heading vector, trajectory path history
  * IR close-range (orange) and Ultrasonic medium-range (cyan) sensor beam overlay
- Map Persistence Memory: Save, Load, Clear map services
- Complete tactile click visual feedback (:hover, :pressed) on all UI buttons
"""

import atexit
import collections
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

# Safeguard: Neutralize opencv-python's intrusive QT_QPA_PLATFORM_PLUGIN_PATH override
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ and 'cv2' in os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import numpy as np

import cv2

try:
    import pyqtgraph as pg
    pg.setConfigOptions(useOpenGL=True, antialias=True)
    PYQTGRAPH_AVAILABLE = True
except Exception:
    PYQTGRAPH_AVAILABLE = False

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QGuiApplication, QImage, QKeyEvent, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QRadioButton,
    QRubberBand, QScrollArea, QSlider, QSplitter, QTextEdit,
    QVBoxLayout, QWidget
)

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from khepera4_control.common import euler_from_quaternion, normalize_angle
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan, Image as RosImage
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ and 'cv2' in os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']


# ==============================================================================
# Sensor Mounting Specification (Khepera IV physical specs)
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


# ==============================================================================
# Global Modern Dark Stylesheet with Distinct Click Effects (:hover, :pressed)
# ==============================================================================
MODERN_STYLE = """
QMainWindow {
    background-color: #0b0f19;
}
QWidget {
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 11px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    padding-bottom: 4px;
    font-weight: bold;
    font-size: 11px;
    background-color: #111827;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #38bdf8;
    background-color: #111827;
}
/* Base PushButton styling with tactile depressed click effect */
QPushButton {
    background-color: #1a2332;
    border: 1px solid #3b82f6;
    border-radius: 5px;
    padding: 5px 10px;
    color: #f8fafc;
    font-weight: bold;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #2563eb;
    border-color: #60a5fa;
}
QPushButton:pressed {
    background-color: #1d4ed8;
    border-color: #93c5fd;
    padding-top: 7px;
    padding-bottom: 3px;
}
QPushButton:disabled {
    background-color: #131b26;
    border-color: #263345;
    color: #4b5563;
}

/* Specialized D-Pad Directional Buttons */
QPushButton#dpad_btn {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 4px;
    color: #38bdf8;
    font-size: 11px;
    font-weight: bold;
}
QPushButton#dpad_btn:hover {
    background-color: #334155;
    border-color: #38bdf8;
    color: #ffffff;
}
QPushButton#dpad_btn:pressed {
    background-color: #0284c7;
    border-color: #7dd3fc;
    color: #ffffff;
    padding-top: 8px;
    padding-bottom: 4px;
}

/* Stop Button (Red Warning) */
QPushButton#stop_btn {
    background-color: #450a0a;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 6px 4px;
    color: #fca5a5;
    font-size: 11px;
    font-weight: bold;
}
QPushButton#stop_btn:hover {
    background-color: #dc2626;
    border-color: #f87171;
    color: #ffffff;
}
QPushButton#stop_btn:pressed {
    background-color: #991b1b;
    border-color: #fca5a5;
    color: #ffffff;
    padding-top: 8px;
    padding-bottom: 4px;
}

/* Auto Explore Toggle Button */
QPushButton#auto_explore_idle {
    background-color: #0f172a;
    border: 2px solid #38bdf8;
    border-radius: 6px;
    padding: 8px 14px;
    color: #38bdf8;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#auto_explore_idle:hover {
    background-color: #0369a1;
    border-color: #7dd3fc;
    color: #ffffff;
}
QPushButton#auto_explore_idle:pressed {
    background-color: #0284c7;
    border-color: #bae6fd;
    color: #ffffff;
    padding-top: 10px;
    padding-bottom: 6px;
}

QPushButton#auto_explore_active {
    background-color: #7f1d1d;
    border: 2px solid #ef4444;
    border-radius: 6px;
    padding: 8px 14px;
    color: #ffffff;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#auto_explore_active:hover {
    background-color: #b91c1c;
    border-color: #fca5a5;
}
QPushButton#auto_explore_active:pressed {
    background-color: #991b1b;
    border-color: #fee2e2;
    padding-top: 10px;
    padding-bottom: 6px;
}

/* User-Guided Exploration Buttons (Draw Path, Set Target, Clear) */
QPushButton#user_draw_btn {
    background-color: #1e293b;
    border: 1px solid #0284c7;
    border-radius: 4px;
    padding: 4px 6px;
    color: #38bdf8;
    font-size: 10px;
    font-weight: bold;
}
QPushButton#user_draw_btn:hover {
    background-color: #0369a1;
    border-color: #38bdf8;
    color: #ffffff;
}
QPushButton#user_draw_btn:checked {
    background-color: #0284c7;
    border-color: #7dd3fc;
    color: #ffffff;
}
QPushButton#user_clear_btn {
    background-color: #27272a;
    border: 1px solid #71717a;
    border-radius: 4px;
    padding: 4px 6px;
    color: #e4e4e7;
    font-size: 10px;
    font-weight: bold;
}
QPushButton#user_clear_btn:hover {
    background-color: #52525b;
    border-color: #a1a1aa;
    color: #ffffff;
}

/* Persistence Buttons (Green, Slate, Amber) */
QPushButton#save_btn {
    background-color: #064e3b;
    border: 1px solid #10b981;
    border-radius: 5px;
    color: #6ee7b7;
    font-weight: bold;
    padding: 5px 8px;
}
QPushButton#save_btn:hover {
    background-color: #059669;
    border-color: #34d399;
    color: #ffffff;
}
QPushButton#save_btn:pressed {
    background-color: #047857;
    border-color: #a7f3d0;
    padding-top: 7px;
    padding-bottom: 3px;
}

QPushButton#load_btn {
    background-color: #1e293b;
    border: 1px solid #64748b;
    border-radius: 5px;
    color: #cbd5e1;
    font-weight: bold;
    padding: 5px 8px;
}
QPushButton#load_btn:hover {
    background-color: #475569;
    border-color: #94a3b8;
    color: #ffffff;
}
QPushButton#load_btn:pressed {
    background-color: #334155;
    border-color: #cbd5e1;
    padding-top: 7px;
    padding-bottom: 3px;
}

QPushButton#clear_btn {
    background-color: #451a03;
    border: 1px solid #f59e0b;
    border-radius: 5px;
    color: #fcd34d;
    font-weight: bold;
    padding: 5px 8px;
}
QPushButton#clear_btn:hover {
    background-color: #d97706;
    border-color: #fbbf24;
    color: #ffffff;
}
QPushButton#clear_btn:pressed {
    background-color: #b45309;
    border-color: #fde68a;
    padding-top: 7px;
    padding-bottom: 3px;
}

QPushButton#export_ply_btn {
    background-color: #4c1d95;
    border: 1px solid #a855f7;
    border-radius: 5px;
    color: #e9d5ff;
    font-weight: bold;
    padding: 6px 10px;
    font-size: 11px;
}
QPushButton#export_ply_btn:hover {
    background-color: #7e22ce;
    border-color: #c084fc;
    color: #ffffff;
}
QPushButton#export_ply_btn:pressed {
    background-color: #6b21a8;
    border-color: #f3e8ff;
    padding-top: 8px;
    padding-bottom: 4px;
}

/* Recenter Button (Indigo / Violet) */
QPushButton#recenter_btn {
    background-color: #1e1b4b;
    border: 1px solid #6366f1;
    border-radius: 6px;
    color: #c7d2fe;
    font-weight: bold;
    padding: 6px 10px;
    font-size: 11px;
}
QPushButton#recenter_btn:hover {
    background-color: #4338ca;
    border-color: #818cf8;
    color: #ffffff;
}
QPushButton#recenter_btn:pressed {
    background-color: #3730a3;
    border-color: #a5b4fc;
    padding-top: 8px;
    padding-bottom: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #1e293b;
    height: 5px;
    background: #0f172a;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    border: 1px solid #7dd3fc;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #7dd3fc;
}

/* Checkboxes */
QCheckBox {
    spacing: 5px;
    font-weight: 500;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #334155;
    background: #0f172a;
}
QCheckBox::indicator:hover {
    border-color: #38bdf8;
}
QCheckBox::indicator:checked {
    background: #0284c7;
    border-color: #38bdf8;
}

/* ComboBox */
QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 3px 20px 3px 8px;
    color: #f8fafc;
    font-weight: bold;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #38bdf8;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #334155;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    color: #e2e8f0;
    padding: 2px;
}

/* Splitter Handles */
QSplitter::handle {
    background-color: #1e293b;
}
QSplitter::handle:horizontal {
    width: 6px;
    margin: 0px 2px;
    border-radius: 2px;
}
QSplitter::handle:horizontal:hover {
    background-color: #38bdf8;
}
QSplitter::handle:horizontal:pressed {
    background-color: #0284c7;
}
QSplitter::handle:vertical {
    height: 6px;
    margin: 2px 0px;
    border-radius: 2px;
}
QSplitter::handle:vertical:hover {
    background-color: #38bdf8;
}
QSplitter::handle:vertical:pressed {
    background-color: #0284c7;
}

/* Autonomous Pursuit & Object Interaction Styles */
QPushButton#pursuit_btn_idle {
    background-color: #064e3b;
    border: 2px solid #10b981;
    border-radius: 6px;
    padding: 6px 12px;
    color: #6ee7b7;
    font-size: 11px;
    font-weight: bold;
}
QPushButton#pursuit_btn_idle:hover {
    background-color: #059669;
    border-color: #34d399;
    color: #ffffff;
}
QPushButton#pursuit_btn_idle:pressed {
    background-color: #047857;
    border-color: #a7f3d0;
    padding-top: 8px;
    padding-bottom: 4px;
}
QPushButton#pursuit_btn_active {
    background-color: #7f1d1d;
    border: 2px solid #ef4444;
    border-radius: 6px;
    padding: 6px 12px;
    color: #ffffff;
    font-size: 11px;
    font-weight: bold;
}
QPushButton#pursuit_btn_active:hover {
    background-color: #b91c1c;
    border-color: #fca5a5;
}
QPushButton#pursuit_btn_active:pressed {
    background-color: #991b1b;
    border-color: #fee2e2;
    padding-top: 8px;
    padding-bottom: 4px;
}
QPushButton#re_enroll_btn {
    background-color: #4c1d95;
    border: 1px solid #8b5cf6;
    border-radius: 5px;
    color: #ddd6fe;
    font-weight: bold;
    padding: 5px 8px;
}
QPushButton#re_enroll_btn:hover {
    background-color: #6d28d9;
    border-color: #a78bfa;
    color: #ffffff;
}
QPushButton#re_enroll_btn:pressed {
    background-color: #5b21b6;
    border-color: #c4b5fd;
    padding-top: 7px;
    padding-bottom: 3px;
}
"""


# ==============================================================================
# ROS 2 Single-Robot Dedicated Worker
# ==============================================================================
class MappingRosWorker(QObject):
    """ROS 2 node running in a background thread for single-robot mapping & control."""

    odom_received = pyqtSignal(float, float, float, float, float)  # x, y, yaw, v, w
    imu_received = pyqtSignal(float, float, float)  # pitch_deg, roll_deg, az
    map_received = pyqtSignal(object)              # nav_msgs/OccupancyGrid
    elevation_received = pyqtSignal(object)        # nav_msgs/OccupancyGrid
    surface_received = pyqtSignal(object)          # nav_msgs/OccupancyGrid
    scan_received = pyqtSignal(object)             # sensor_msgs/LaserScan
    cliff_alert_received = pyqtSignal(bool)
    turned_over_received = pyqtSignal(bool)
    feedback_received = pyqtSignal(str)
    explorer_state_received = pyqtSignal(bool)
    active_goal_received = pyqtSignal(float, float)
    screenshot_requested = pyqtSignal(str)
    camera_received = pyqtSignal(object)              # sensor_msgs/Image
    visual_map_received = pyqtSignal(object)          # nav_msgs/OccupancyGrid
    executive_state_received = pyqtSignal(object)     # khepera_interfaces/ExecutiveState
    detection_visual_received = pyqtSignal(object)    # sensor_msgs/Image
    target_spawn_response_received = pyqtSignal(bool, str, float, float, float)
    target_enroll_response_received = pyqtSignal(bool, str)
    pursuit_response_received = pyqtSignal(bool, str)
    thinking_log_received = pyqtSignal(str)
    target_status_received = pyqtSignal(bool)
    target_confidence_received = pyqtSignal(float)
    target_roi_received = pyqtSignal(int, int, int, int, str, float)
    target_keypoints_count_received = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.node: Optional[Node] = None
        self.cmd_pub = None
        self.explorer_enable_pub = None
        self.user_goal_pub = None
        self.user_path_pub = None
        self.clear_path_pub = None
        self.save_map_cli = None
        self.load_map_cli = None
        self.clear_map_cli = None
        self.clear_visual_map_cli = None
        self.reconstruct_3dgs_cli = None
        self.pin_keyframe_cli = None
        self.recenter_cli = None
        self.set_exp_cli = None
        self.toggle_exp_cli = None
        self.set_pursuit_cli = None
        self.spawn_target_cli = None
        self.enroll_target_cli = None
        self.is_running = True
        self.last_fused_odom_time = 0.0
        self._explorer_proc: Optional[subprocess.Popen] = None
        self.auto_spawn_explorer: bool = False

    def _get_now(self) -> float:
        if self.node and self.node.context.ok():
            return self.node.get_clock().now().nanoseconds * 1e-9
        return time.time()

    def init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('khepera4_mapping_dashboard_node')
        if not self.node.has_parameter('use_sim_time'):
            self.node.declare_parameter('use_sim_time', False)

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 1. Direct Single-Robot Command Publisher & User Exploration Publishers
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.explorer_enable_pub = self.node.create_publisher(Bool, '/explorer/enable', qos_latched)
        self.explorer_speed_pub = self.node.create_publisher(Twist, '/explorer/set_speed', 10)
        self.explorer_speed_limits_pub = self.node.create_publisher(
            Twist, '/explorer/set_speed_limits', 10
        )
        self.pursuit_speed_limits_pub = self.node.create_publisher(
            Twist, '/pursuit/set_speed_limits', 10
        )
        self.robot_speed_limits_pub = self.node.create_publisher(
            Twist, '/robot/speed_limits', 10
        )
        self.target_image_pub = self.node.create_publisher(RosImage, '/pursuit/target_image', 10)
        self.target_prompt_pub = self.node.create_publisher(String, '/pursuit/target_prompt', 10)
        self.pursuit_mode_pub = self.node.create_publisher(String, '/pursuit/mode', 10)
        self.pursuit_enable_pub = self.node.create_publisher(Bool, '/pursuit/enable', 10)
        self.stop_reset_pub = self.node.create_publisher(Bool, '/target_found_and_stopped', 10)
        self.user_goal_pub = self.node.create_publisher(PoseStamped, '/explorer/user_goal', 10)
        self.user_path_pub = self.node.create_publisher(Path, '/explorer/user_path', 10)
        self.clear_path_pub = self.node.create_publisher(Bool, '/explorer/clear_path', 10)

        # 2. Service Clients for Spatial Persistence & Autonomy & Recenter & Object Pursuit
        self.save_map_cli = self.node.create_client(Trigger, '/save_map')
        self.load_map_cli = self.node.create_client(Trigger, '/load_map')
        self.clear_map_cli = self.node.create_client(Trigger, '/clear_map')
        self.clear_visual_map_cli = self.node.create_client(Trigger, '/clear_visual_map')
        self.reconstruct_3dgs_cli = self.node.create_client(
            Trigger, '/khepera_3dgs/trigger_reconstruction'
        )
        self.pin_keyframe_cli = self.node.create_client(Trigger, '/pin_keyframe')
        self.recenter_cli = self.node.create_client(Trigger, '/recenter_robot')
        self.set_exp_cli = self.node.create_client(SetBool, '/set_explorer')
        self.toggle_exp_cli = self.node.create_client(Trigger, '/toggle_explorer')

        # 3. Subscriptions (Direct single-robot topics + fallbacks)
        self.node.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.node.create_subscription(Odometry, '/khepera4/odom', self._fused_odom_cb, 20)
        self.node.create_subscription(Imu, '/imu/data', self._imu_cb, 30)
        self.node.create_subscription(Imu, '/khepera4/imu/data', self._imu_cb, 30)
        self.node.create_subscription(OccupancyGrid, '/map', lambda m: self.map_received.emit(m), qos_latched)
        self.node.create_subscription(OccupancyGrid, '/elevation_map', lambda m: self.elevation_received.emit(m), qos_latched)
        self.node.create_subscription(OccupancyGrid, '/surface_map', lambda m: self.surface_received.emit(m), qos_latched)
        self.node.create_subscription(OccupancyGrid, '/visual_map', lambda m: self.visual_map_received.emit(m), qos_latched)
        self.node.create_subscription(LaserScan, '/khepera4/scan', lambda m: self.scan_received.emit(m), 10)
        self.node.create_subscription(Bool, '/cliff_alert', lambda m: self.cliff_alert_received.emit(m.data), 10)
        self.node.create_subscription(Bool, '/khepera4/turned_over', lambda m: self.turned_over_received.emit(m.data), 10)
        self.node.create_subscription(Bool, '/explorer/status', lambda m: self.explorer_state_received.emit(m.data), qos_latched)
        self.node.create_subscription(Bool, '/explorer/enable', lambda m: self.explorer_state_received.emit(m.data), 10)
        self.node.create_subscription(PointStamped, '/explorer/active_goal', self._active_goal_cb, 10)
        self.node.create_subscription(String, '/dashboard/screenshot', lambda m: self.screenshot_requested.emit(m.data), 10)
        self.node.create_subscription(RosImage, '/camera/image_raw', lambda m: self.camera_received.emit(m), 10)
        self.node.create_subscription(RosImage, '/khepera4/camera/image_raw', lambda m: self.camera_received.emit(m), 10)
        self.node.create_subscription(RosImage, '/target/detection_visual', lambda m: self.detection_visual_received.emit(m), 10)
        self.node.create_subscription(String, '/robot/thinking_log', lambda m: self.thinking_log_received.emit(m.data), 10)
        self.node.create_subscription(Bool, '/perception/target_status', lambda m: self.target_status_received.emit(m.data), 10)
        self.node.create_subscription(Float32, '/perception/tracking_confidence', lambda m: self.target_confidence_received.emit(m.data), 10)
        self.node.create_subscription(
            String, '/executive/status', lambda m: self.executive_state_received.emit(m.data), 10
        )

        # Perception ROI subscriptions (with graceful fallback)
        try:
            from vision_msgs.msg import Detection2DArray
            self.node.create_subscription(
                Detection2DArray, '/perception/candidate_rois',
                self._candidate_rois_cb, 10
            )
        except Exception:
            pass

        try:
            from khepera_interfaces.msg import TargetDetection
            self.node.create_subscription(
                TargetDetection, '/target/detection',
                self._target_detection_cb, 10
            )
        except Exception:
            pass

        try:
            from geometry_msgs.msg import PolygonStamped
            self.node.create_subscription(
                PolygonStamped, '/perception/target_keypoints',
                self._target_keypoints_cb, 10
            )
        except Exception:
            pass

    def _candidate_rois_cb(self, msg):
        if not hasattr(msg, 'detections') or not msg.detections:
            return
        best = max(
            msg.detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0
        )
        score = best.results[0].hypothesis.score if best.results else 1.0
        class_id = best.results[0].hypothesis.class_id if best.results else "target"
        cx = best.bbox.center.position.x
        cy = best.bbox.center.position.y
        sx = best.bbox.size_x
        sy = best.bbox.size_y
        x1 = max(0, int(cx - sx / 2.0))
        y1 = max(0, int(cy - sy / 2.0))
        x2 = int(cx + sx / 2.0)
        y2 = int(cy + sy / 2.0)
        self.target_roi_received.emit(x1, y1, x2, y2, str(class_id), float(score))

    def _target_detection_cb(self, msg):
        if not getattr(msg, 'is_detected', False):
            return
        bbox = msg.bbox
        x1 = int(bbox.x_offset)
        y1 = int(bbox.y_offset)
        x2 = int(bbox.x_offset + bbox.width)
        y2 = int(bbox.y_offset + bbox.height)
        score = float(getattr(msg, 'max_similarity', 1.0))
        self.target_roi_received.emit(x1, y1, x2, y2, "target", score)

    def _target_keypoints_cb(self, msg):
        if not hasattr(msg, 'polygon') or not msg.polygon.points:
            self.target_keypoints_count_received.emit(0)
            return
        xs = [p.x for p in msg.polygon.points]
        ys = [p.y for p in msg.polygon.points]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        self.target_roi_received.emit(x1, y1, x2, y2, "target", 0.90)
        self.target_keypoints_count_received.emit(len(msg.polygon.points))

    @property
    def using_fused_odom(self) -> bool:
        """Return True if /khepera4/odom was received within the last 1.0 second."""
        return (self._get_now() - self.last_fused_odom_time) <= 1.0

    @using_fused_odom.setter
    def using_fused_odom(self, value: bool):
        if value:
            self.last_fused_odom_time = self._get_now()
        else:
            self.last_fused_odom_time = 0.0

    def _fused_odom_cb(self, msg: Odometry):
        """Handle filtered, slip-projected odometry on /khepera4/odom (highest priority)."""
        self.last_fused_odom_time = self._get_now()
        self._process_odom(msg)

    def _odom_cb(self, msg: Odometry):
        """Handle raw odometry from /odom, falling back only when /khepera4/odom is absent."""
        if not self.using_fused_odom:
            # Suppress raw odometry that has penetrated beyond outer arena perimeter walls
            if abs(msg.pose.pose.position.x) > 2.405 or abs(msg.pose.pose.position.y) > 2.405:
                return
            self._process_odom(msg)

    def _process_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        # Enforce arena bounds so visual marker never breaks outside walls
        cx = float(np.clip(pos.x, -2.405, 2.405))
        cy = float(np.clip(pos.y, -2.405, 2.405))
        self.odom_received.emit(cx, cy, yaw, twist.linear.x, twist.angular.z)

    def _imu_cb(self, msg: Imu):
        orient = msg.orientation
        roll, pitch, _ = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        self.imu_received.emit(math.degrees(pitch), math.degrees(roll), msg.linear_acceleration.z)

    def publish_cmd(self, linear_v: float, angular_w: float):
        """Publish velocity command directly to /cmd_vel."""
        if self.node and self.cmd_pub and self.node.context.ok():
            msg = Twist()
            msg.linear.x = float(linear_v)
            msg.angular.z = float(angular_w)
            self.cmd_pub.publish(msg)

    def set_robot_speed(self, linear_v: float, angular_w: float):
        """Publish dynamic speed limits across all robot algorithms."""
        if self.node and self.node.context.ok():
            msg = Twist()
            msg.linear.x = float(linear_v)
            msg.angular.z = float(angular_w)
            if hasattr(self, 'robot_speed_limits_pub') and self.robot_speed_limits_pub:
                self.robot_speed_limits_pub.publish(msg)
            if hasattr(self, 'explorer_speed_pub') and self.explorer_speed_pub:
                self.explorer_speed_pub.publish(msg)
            if hasattr(self, 'explorer_speed_limits_pub') and self.explorer_speed_limits_pub:
                self.explorer_speed_limits_pub.publish(msg)
            if hasattr(self, 'pursuit_speed_limits_pub') and self.pursuit_speed_limits_pub:
                self.pursuit_speed_limits_pub.publish(msg)
        self.set_explorer_speed(linear_v, angular_w)

    def set_explorer_speed(self, linear_v: float, angular_w: float):
        """Publish dynamic speed limits (alias for backward compatibility)."""
        pass

    def publish_target_image(self, img_msg: RosImage):
        """Publish reference target template image for visual pursuit verification."""
        if (
            self.node
            and hasattr(self, 'target_image_pub')
            and self.target_image_pub
            and self.node.context.ok()
        ):
            self.target_image_pub.publish(img_msg)

    def publish_target_prompt(self, prompt_text: str):
        """Publish text prompt for open-vocabulary detection."""
        if (
            self.node
            and hasattr(self, 'target_prompt_pub')
            and self.target_prompt_pub
            and self.node.context.ok()
        ):
            msg = String()
            msg.data = str(prompt_text)
            self.target_prompt_pub.publish(msg)

    def publish_pursuit_mode(self, mode: str):
        """Publish target pursuit mode ('semantic' or 'template') to /pursuit/mode."""
        if (
            self.node
            and hasattr(self, 'pursuit_mode_pub')
            and self.pursuit_mode_pub
            and self.node.context.ok()
        ):
            msg = String()
            msg.data = str(mode)
            self.pursuit_mode_pub.publish(msg)

    def set_pursuit_enable(self, enable: bool):
        """Publish enable state to /pursuit/enable."""
        if (
            self.node
            and hasattr(self, 'pursuit_enable_pub')
            and self.pursuit_enable_pub
            and self.node.context.ok()
        ):
            msg = Bool()
            msg.data = bool(enable)
            self.pursuit_enable_pub.publish(msg)

    def publish_target_stopped(self, stopped: bool):
        """Publish stop reset or halt state on /target_found_and_stopped."""
        if (
            self.node
            and hasattr(self, 'stop_reset_pub')
            and self.stop_reset_pub
            and self.node.context.ok()
        ):
            msg = Bool()
            msg.data = bool(stopped)
            self.stop_reset_pub.publish(msg)

    def set_explorer_enable(self, enable: bool):
        """Publish directly to /explorer/enable topic."""
        if (
            self.node
            and hasattr(self, 'explorer_enable_pub')
            and self.explorer_enable_pub
            and self.node.context.ok()
        ):
            msg = Bool()
            msg.data = bool(enable)
            self.explorer_enable_pub.publish(msg)

    def kill_explorer_instances(self):
        """Immediately stop and kill all active reactive_explorer instances."""
        if self.cmd_pub and self.node and self.node.context.ok():
            self.cmd_pub.publish(Twist())
        if self.explorer_enable_pub and self.node and self.node.context.ok():
            self.explorer_enable_pub.publish(Bool(data=False))

        if self._explorer_proc is not None:
            try:
                if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                    pgid = os.getpgid(self._explorer_proc.pid)
                    # Only killpg if this pgid is strictly an isolated child process group
                    if pgid != os.getpgrp() and pgid != os.getpid():
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            self._explorer_proc.wait(timeout=0.5)
                        except Exception:
                            os.killpg(pgid, signal.SIGKILL)
                    else:
                        self._explorer_proc.terminate()
                else:
                    self._explorer_proc.terminate()
            except Exception:
                try:
                    self._explorer_proc.kill()
                except Exception:
                    pass
            self._explorer_proc = None

        try:
            out = subprocess.run(
                ["pgrep", "-f", "khepera4_control.*reactive_explorer"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if out.stdout:
                my_pid = os.getpid()
                my_ppid = os.getppid()
                for pid_str in out.stdout.split():
                    try:
                        pid = int(pid_str)
                        if pid != my_pid and pid != my_ppid:
                            # Terminate specific PID directly; never kill parent or launch group
                            os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
        except Exception:
            pass
        self.feedback_received.emit("[Explorer] Terminated all active explorer instances.")
        self.explorer_state_received.emit(False)

    def _ensure_explorer_running(self) -> bool:
        """Verify explorer is running; spawn fallback subprocess only if absent from graph."""
        # 1. First check if node is already launched in ROS 2 graph
        if self.node and self.node.context.ok():
            try:
                node_names = self.node.get_node_names()
                if "reactive_explorer" in node_names:
                    return True
            except Exception:
                pass
        # 2. Check if local tracked subprocess is already running
        if self._explorer_proc is not None and self._explorer_proc.poll() is None:
            return True
        if not self.auto_spawn_explorer:
            return False
        # 3. Fallback: Spawn with isolated process group
        try:
            use_sim = False
            if self.node and self.node.has_parameter('use_sim_time'):
                use_sim = bool(self.node.get_parameter('use_sim_time').value)
            cmd = [
                'ros2', 'run', 'khepera4_control', 'reactive_explorer',
                '--ros-args',
                '-p', f'use_sim_time:={str(use_sim).lower()}',
                '-p', 'enabled:=false'
            ]
            self._explorer_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            self.feedback_received.emit("[Explorer] Auto-spawning reactive explorer (fallback)...")
            return True
        except Exception as e:
            self.feedback_received.emit(f"[Error] Failed to auto-spawn explorer: {e}")
            return False

    def set_auto_explore(self, enable: bool):
        """Call /set_explorer service and publish to /explorer/enable or kill instances."""
        if not enable:
            self.kill_explorer_instances()
            return

        if self.node and self.explorer_enable_pub and self.node.context.ok():
            bmsg = Bool()
            bmsg.data = bool(enable)
            self.explorer_enable_pub.publish(bmsg)

        def _call():
            if not self.set_exp_cli or not self.node or not self.node.context.ok():
                return
            if not self.set_exp_cli.wait_for_service(timeout_sec=0.4):
                if enable and self._ensure_explorer_running():
                    if not self.set_exp_cli.wait_for_service(timeout_sec=2.5):
                        if self.toggle_exp_cli and self.toggle_exp_cli.wait_for_service(timeout_sec=0.5):
                            self.trigger_service(self.toggle_exp_cli, "Toggle Explorer")
                        else:
                            self.feedback_received.emit("[Warning] Explorer Service Offline")
                            self.explorer_state_received.emit(False)
                        return
                else:
                    if self.toggle_exp_cli and self.toggle_exp_cli.wait_for_service(timeout_sec=0.5):
                        self.trigger_service(self.toggle_exp_cli, "Toggle Explorer")
                    else:
                        self.feedback_received.emit("[Warning] Explorer Service Offline")
                        self.explorer_state_received.emit(False)
                    return
            req = SetBool.Request()
            req.data = bool(enable)
            future = self.set_exp_cli.call_async(req)
            t0 = time.time()
            while not future.done() and (time.time() - t0 < 2.5):
                time.sleep(0.04)
            if future.done():
                try:
                    res = future.result()
                    self.feedback_received.emit(f"[Explorer] {res.message}")
                    self.explorer_state_received.emit(enable)
                except Exception as e:
                    self.feedback_received.emit(f"[Error] Explorer: {e}")
            else:
                self.feedback_received.emit("[Timeout] Explorer service call")

        threading.Thread(target=_call, daemon=True).start()

    def trigger_service(self, client, action_name: str):
        """Asynchronously trigger ROS standard Trigger service."""
        def _call():
            if not client or not self.node or not self.node.context.ok():
                return
            if not client.wait_for_service(timeout_sec=1.2):
                self.feedback_received.emit(f"[Offline] {action_name} service")
                return
            req = Trigger.Request()
            future = client.call_async(req)
            t0 = time.time()
            while not future.done() and (time.time() - t0 < 3.0):
                time.sleep(0.04)
            if future.done():
                try:
                    res = future.result()
                    status_tag = "[OK]" if res.success else "[Failed]"
                    self.feedback_received.emit(f"{status_tag} {action_name}: {res.message}")
                except Exception as e:
                    self.feedback_received.emit(f"[Error] {action_name}: {e}")
            else:
                self.feedback_received.emit(f"[Timeout] {action_name} request")

        threading.Thread(target=_call, daemon=True).start()

    def recenter_robot(self):
        """Trigger /recenter_robot service to respawn Khepera IV at origin rightside up."""
        self.trigger_service(self.recenter_cli, "Recenter Robot")

    def _active_goal_cb(self, msg: PointStamped):
        """Forward active target to dashboard GUI."""
        self.active_goal_received.emit(msg.point.x, msg.point.y)

    def send_user_goal(self, x: float, y: float):
        """Dispatch a single user-clicked target to /explorer/user_goal."""
        if self.node and self.user_goal_pub and self.node.context.ok():
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.orientation.w = 1.0
            self.user_goal_pub.publish(msg)

        if self.auto_spawn_explorer and self.is_running and self.set_exp_cli and not self.set_exp_cli.service_is_ready():
            def _ensure_goal():
                try:
                    if not self.is_running or not self.node or not self.node.context.ok():
                        return
                    if self._ensure_explorer_running():
                        if not self.is_running or not self.node or not self.node.context.ok():
                            return
                        if self.set_exp_cli and self.set_exp_cli.wait_for_service(timeout_sec=2.5):
                            if self.is_running and self.node and self.node.context.ok() and self.user_goal_pub:
                                msg = PoseStamped()
                                msg.header.stamp = self.node.get_clock().now().to_msg()
                                msg.header.frame_id = 'map'
                                msg.pose.position.x = float(x)
                                msg.pose.position.y = float(y)
                                msg.pose.orientation.w = 1.0
                                self.user_goal_pub.publish(msg)
                except Exception:
                    pass
            threading.Thread(target=_ensure_goal, daemon=True).start()

    def send_user_path(self, points: List[Tuple[float, float]]):
        """Dispatch user-drawn exploration path to /explorer/user_path."""
        if self.node and self.user_path_pub and self.node.context.ok():
            msg = Path()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            for pt in points:
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose.position.x = float(pt[0])
                ps.pose.position.y = float(pt[1])
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
            self.user_path_pub.publish(msg)

        if self.auto_spawn_explorer and self.is_running and self.set_exp_cli and not self.set_exp_cli.service_is_ready():
            def _ensure_path():
                try:
                    if not self.is_running or not self.node or not self.node.context.ok():
                        return
                    if self._ensure_explorer_running():
                        if not self.is_running or not self.node or not self.node.context.ok():
                            return
                        if self.set_exp_cli and self.set_exp_cli.wait_for_service(timeout_sec=2.5):
                            if self.is_running and self.node and self.node.context.ok() and self.user_path_pub:
                                msg = Path()
                                msg.header.stamp = self.node.get_clock().now().to_msg()
                                msg.header.frame_id = 'map'
                                for pt in points:
                                    ps = PoseStamped()
                                    ps.header = msg.header
                                    ps.pose.position.x = float(pt[0])
                                    ps.pose.position.y = float(pt[1])
                                    ps.pose.orientation.w = 1.0
                                    msg.poses.append(ps)
                                self.user_path_pub.publish(msg)
                except Exception:
                    pass
            threading.Thread(target=_ensure_path, daemon=True).start()

    def send_clear_path(self):
        """Dispatch clear command to /explorer/clear_path."""
        if self.node and self.clear_path_pub and self.node.context.ok():
            msg = Bool()
            msg.data = True
            self.clear_path_pub.publish(msg)

    def spin(self):
        while self.is_running and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.02)
            except Exception:
                break

    def stop(self):
        self.is_running = False
        if self.node:
            self.publish_cmd(0.0, 0.0)
        if self._explorer_proc is not None:
            try:
                import signal
                if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                    os.killpg(os.getpgid(self._explorer_proc.pid), signal.SIGTERM)
                else:
                    self._explorer_proc.terminate()
            except Exception:
                try:
                    self._explorer_proc.kill()
                except Exception:
                    pass
            self._explorer_proc = None
            self.publish_cmd(0.0, 0.0)

    def enable_pursuit(self, enable: bool):
        """Publish pursuit enable state on /pursuit/enable."""
        if self.pursuit_enable_pub and self.node and self.node.context.ok():
            self.pursuit_enable_pub.publish(Bool(data=enable))

    def send_target_prompt(self, prompt: str):
        """Publish target semantic prompt on /pursuit/target_prompt."""
        if self.target_prompt_pub and self.node and self.node.context.ok():
            self.target_prompt_pub.publish(String(data=prompt))

    def send_target_image(self, img_msg: RosImage):
        """Publish target template image on /pursuit/target_image."""
        if self.target_image_pub and self.node and self.node.context.ok():
            self.target_image_pub.publish(img_msg)


# ==============================================================================
# Floating, Resizable & Always-on-Top Pop-up Camera Window
# ==============================================================================
class CameraPopupWindow(QWidget):
    """Floating, resizable pop-up camera monitor for Khepera IV.

    Features:
    - Dynamic aspect-ratio-preserving video frame rendering
    - Toggleable 'Always on Top' window state (pinned above Gazebo/RViz2)
    - Full keyboard WASD/Arrow/Space teleoperation forwarding to the robot controller
    - Click-to-focus and automatic safety stop on focus-loss
    - Direct Pin-View-to-Map visual keyframe registration
    """

    def __init__(self, dashboard: 'MappingDashboardWindow', parent=None):
        super().__init__(parent, Qt.Window)
        self.dashboard = dashboard
        self.is_always_on_top = True
        self.hud_enabled: bool = True
        self.pursuit_active: bool = False
        if self.dashboard and hasattr(self.dashboard, 'is_pursuit_active'):
            self.pursuit_active = bool(self.dashboard.is_pursuit_active)

        # Active rendered frame
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_qimage: Optional[QImage] = None
        self.latest_fps: float = 0.0
        self.latest_res: Tuple[int, int] = (640, 480)

        # Raw camera stream cache
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_raw_qimage: Optional[QImage] = None
        self.latest_raw_fps: float = 0.0
        self.latest_raw_res: Tuple[int, int] = (640, 480)

        # Detection HUD annotated stream cache (/target/detection_visual)
        self.latest_hud_frame: Optional[np.ndarray] = None
        self.latest_hud_qimage: Optional[QImage] = None
        self.latest_hud_fps: float = 0.0
        self.latest_hud_res: Tuple[int, int] = (640, 480)

        self.setWindowTitle("Khepera IV - Live Camera Stream [WASD Teleop Active]")
        self.setMinimumSize(260, 210)
        self.setFocusPolicy(Qt.StrongFocus)

        primary_screen = QGuiApplication.primaryScreen()
        if primary_screen:
            geom = primary_screen.availableGeometry()
            sw, sh = geom.width(), geom.height()
            self.setGeometry(geom.x() + sw // 2, geom.y(), sw // 2, sh // 2)
        else:
            self.resize(540, 430)

        if self.is_always_on_top:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0b0f19;
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                font-size: 11px;
            }
            QPushButton {
                background-color: #1a2332;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                padding: 4px 8px;
                color: #f8fafc;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header controls bar
        header_bar = QWidget()
        hb_layout = QHBoxLayout(header_bar)
        hb_layout.setContentsMargins(0, 0, 0, 0)
        hb_layout.setSpacing(6)

        # Always-on-top toggle button
        self.btn_always_on_top = QPushButton()
        self.btn_always_on_top.setCursor(Qt.PointingHandCursor)
        self.btn_always_on_top.setFocusPolicy(Qt.NoFocus)
        self.btn_always_on_top.setToolTip("Toggle keeping this camera stream always above other windows (e.g. Gazebo/RViz2)")
        self.btn_always_on_top.clicked.connect(self.toggle_always_on_top)
        self._update_on_top_button_style()

        # Detection HUD Overlay Toggle Button (Green Boundaries)
        self.btn_hud_toggle = QPushButton("Detection HUD: OFF")
        self.btn_hud_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_hud_toggle.setFocusPolicy(Qt.NoFocus)
        self.btn_hud_toggle.setToolTip("Toggle target detection visual overlay (green bounding box, contact crosshair, similarity score)")
        self.btn_hud_toggle.clicked.connect(self.toggle_hud)
        self._update_hud_button_style()

        # Pin view button
        self.btn_pin = QPushButton("Pin View to Map")
        self.btn_pin.setCursor(Qt.PointingHandCursor)
        self.btn_pin.setFocusPolicy(Qt.NoFocus)
        self.btn_pin.setStyleSheet(
            "background-color: #0f766e; border: 1px solid #14b8a6; border-radius: 4px; "
            "color: #ccfbf1; font-weight: bold; font-size: 10px; padding: 4px 8px;"
        )
        self.btn_pin.setToolTip("Pin the current camera frame to the SLAM map as a visual landmark")
        self.btn_pin.clicked.connect(self.on_pin_clicked)

        # Stream status
        self.lbl_status = QLabel("ONLINE | 640x480 | On-Top: ON")
        self.lbl_status.setStyleSheet("color: #38bdf8; font-size: 9.5px; font-family: monospace; font-weight: bold;")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        hb_layout.addWidget(self.btn_always_on_top)
        hb_layout.addWidget(self.btn_hud_toggle)
        hb_layout.addWidget(self.btn_pin)
        hb_layout.addWidget(self.lbl_status, stretch=1)
        layout.addWidget(header_bar)

        # Video frame display area
        self.lbl_view = QLabel("Awaiting camera stream...")
        self.lbl_view.setAlignment(Qt.AlignCenter)
        self.lbl_view.setStyleSheet(
            "background-color: #020617; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #64748b; font-family: monospace; font-size: 11px;"
        )
        self.lbl_view.setFocusPolicy(Qt.NoFocus)
        self.lbl_view.mousePressEvent = lambda e: self.setFocus()
        layout.addWidget(self.lbl_view, stretch=1)

        # Teleop instruction footer bar
        self.lbl_teleop_hint = QLabel("Click window to drive: [W][A][S][D] / [Arrows]  |  [Space] to stop")
        self.lbl_teleop_hint.setAlignment(Qt.AlignCenter)
        self.lbl_teleop_hint.setStyleSheet(
            "background-color: #111827; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #38bdf8; font-size: 9.5px; font-weight: bold; padding: 2px;"
        )
        layout.addWidget(self.lbl_teleop_hint)

    def update_pursuit_state(self, active: bool):
        """Update pursuit state and adjust HUD button styling and prompt."""
        self.pursuit_active = bool(active)
        self._update_hud_button_style()
        self._update_status_text()

    def toggle_always_on_top(self):
        self.set_always_on_top(not self.is_always_on_top)

    def set_always_on_top(self, enabled: bool):
        self.is_always_on_top = enabled
        pos = self.pos()
        size = self.size()
        was_visible = self.isVisible()

        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        self._update_on_top_button_style()

        if was_visible:
            self.show()
            self.move(pos)
            self.resize(size)
            self.raise_()
            self.activateWindow()

    def toggle_hud(self):
        """Toggle between raw camera stream and detection visual overlay."""
        self.set_hud_enabled(not self.hud_enabled)

    def set_hud_enabled(self, enabled: bool):
        """Set detection HUD overlay enabled state."""
        self.hud_enabled = enabled
        self._update_hud_button_style()
        self._refresh_active_frame()

    def _update_hud_button_style(self):
        if not hasattr(self, 'btn_hud_toggle'):
            return
        if self.hud_enabled:
            self.btn_hud_toggle.setText("Detection HUD: ON (Green Bounds)")
            self.btn_hud_toggle.setStyleSheet(
                "background-color: #059669; border: 2px solid #10b981; border-radius: 4px; "
                "color: #ffffff; font-weight: bold; font-size: 10px; padding: 4px 8px;"
            )
            self.btn_hud_toggle.setToolTip("Target detection visual overlay active (green bounding box & contact crosshairs). Click to disable.")
        else:
            self.btn_hud_toggle.setText("Detection HUD: OFF")
            if getattr(self, 'pursuit_active', False):
                self.btn_hud_toggle.setStyleSheet(
                    "background-color: #064e3b; border: 1px solid #10b981; border-radius: 4px; "
                    "color: #a7f3d0; font-weight: bold; font-size: 10px; padding: 4px 8px;"
                )
                self.btn_hud_toggle.setToolTip("Pursuit is ACTIVE! Click to enable green notation contact bounding box.")
            else:
                self.btn_hud_toggle.setStyleSheet(
                    "background-color: #1e293b; border: 1px solid #475569; border-radius: 4px; "
                    "color: #94a3b8; font-weight: bold; font-size: 10px; padding: 4px 8px;"
                )
                self.btn_hud_toggle.setToolTip("Toggle target detection visual overlay (green bounding box & contact crosshairs).")
        self._update_status_text()

    def _refresh_active_frame(self):
        """Switch currently rendered QImage between raw camera and detection visual."""
        if self.hud_enabled and self.latest_hud_qimage is not None:
            self.latest_frame = self.latest_hud_frame
            self.latest_qimage = self.latest_hud_qimage
            self.latest_res = self.latest_hud_res
            self.latest_fps = self.latest_hud_fps
        elif self.latest_raw_qimage is not None:
            self.latest_frame = self.latest_raw_frame
            self.latest_qimage = self.latest_raw_qimage
            self.latest_res = self.latest_raw_res
            self.latest_fps = self.latest_raw_fps
        self._render_current_frame()
        self._update_status_text()

    def _update_on_top_button_style(self):
        if self.is_always_on_top:
            self.btn_always_on_top.setText("Always on Top: ON")
            self.btn_always_on_top.setStyleSheet(
                "background-color: #0284c7; border: 1px solid #38bdf8; border-radius: 4px; "
                "color: #ffffff; font-weight: bold; font-size: 10px; padding: 4px 8px;"
            )
        else:
            self.btn_always_on_top.setText("Always on Top: OFF")
            self.btn_always_on_top.setStyleSheet(
                "background-color: #1e293b; border: 1px solid #475569; border-radius: 4px; "
                "color: #94a3b8; font-weight: bold; font-size: 10px; padding: 4px 8px;"
            )
        self._update_status_text()

    def _update_status_text(self):
        w, h = self.latest_res
        on_top_str = "ON" if self.is_always_on_top else "OFF"
        hud_str = " | HUD: ON" if self.hud_enabled else " | HUD: OFF"
        pursuit_str = " | Pursuit: ACTIVE" if getattr(self, 'pursuit_active', False) else " | Pursuit: STANDBY"
        fps_str = f" @ {self.latest_fps:.1f} FPS" if self.latest_fps > 0 else ""
        teleop_str = ""
        if self.dashboard and (abs(self.dashboard.active_cmd_v) > 0.001 or abs(self.dashboard.active_cmd_w) > 0.001):
            teleop_str = f" | [v={self.dashboard.active_cmd_v:+.2f} m/s, ω={self.dashboard.active_cmd_w:+.2f} rad/s]"
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(f"ONLINE | {w}x{h}{fps_str}{teleop_str}{hud_str}{pursuit_str} | On-Top: {on_top_str}")

    def on_pin_clicked(self):
        if self.dashboard:
            self.dashboard.on_pin_view_to_map()

    def update_frame(self, rgb_arr: np.ndarray, w: int, h: int, fps: float):
        """Update live camera frame (with target pursuit overlay when visible)."""
        if not rgb_arr.flags['C_CONTIGUOUS']:
            rgb_arr = np.ascontiguousarray(rgb_arr)
        self.latest_raw_frame = rgb_arr
        self.latest_raw_res = (w, h)
        self.latest_raw_fps = fps
        self.latest_raw_qimage = QImage(rgb_arr.data, w, h, w * 3, QImage.Format_RGB888).copy()

        now = time.time()
        hud_is_fresh = (
            self.hud_enabled
            and self.latest_hud_qimage is not None
            and (now - getattr(self, '_last_hud_recv_time', 0.0)) < 0.6
        )
        if not hud_is_fresh:
            self.latest_frame = self.latest_raw_frame
            self.latest_res = self.latest_raw_res
            self.latest_fps = self.latest_raw_fps
            self.latest_qimage = self.latest_raw_qimage
            self._render_current_frame()
            self._update_status_text()

    def update_detection_frame(self, rgb_arr: np.ndarray, w: int, h: int, fps: float):
        """Update detection visual overlay frame with green bounding box boundaries."""
        if not rgb_arr.flags['C_CONTIGUOUS']:
            rgb_arr = np.ascontiguousarray(rgb_arr)
        self.latest_hud_frame = rgb_arr
        self.latest_hud_res = (w, h)
        self.latest_hud_fps = fps
        self.latest_hud_qimage = QImage(rgb_arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._last_hud_recv_time = time.time()

        if self.hud_enabled:
            self.latest_frame = self.latest_hud_frame
            self.latest_res = self.latest_hud_res
            self.latest_fps = self.latest_hud_fps
            self.latest_qimage = self.latest_hud_qimage
            self._render_current_frame()
            self._update_status_text()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_active_frame()

    def _render_current_frame(self):
        if not self.isVisible() or self.latest_qimage is None:
            return
        lbl_w = max(100, self.lbl_view.width())
        lbl_h = max(75, self.lbl_view.height())
        pixmap = QPixmap.fromImage(self.latest_qimage).scaled(
            lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.lbl_view.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current_frame()

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        teleop_keys = (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                       Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
                       Qt.Key_Space)
        if key in teleop_keys:
            if not event.isAutoRepeat():
                if self.dashboard:
                    self.dashboard.keys_pressed.add(key)
                    if self.dashboard.is_auto_exploring:
                        self.dashboard.on_toggle_auto_explore()
                    if key == Qt.Key_Space:
                        self.dashboard.on_stop_clicked()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        teleop_keys = (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                       Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
                       Qt.Key_Space)
        if key in teleop_keys:
            if not event.isAutoRepeat():
                if self.dashboard and key in self.dashboard.keys_pressed:
                    self.dashboard.keys_pressed.remove(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        if self.dashboard:
            drive_keys = (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                          Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right)
            for k in drive_keys:
                self.dashboard.keys_pressed.discard(k)
            if self.dashboard.ros:
                self.dashboard.ros.publish_cmd(0.0, 0.0)
        super().focusOutEvent(event)

    def closeEvent(self, event):
        if self.dashboard:
            drive_keys = (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                          Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right)
            for k in drive_keys:
                self.dashboard.keys_pressed.discard(k)
            if hasattr(self.dashboard, 'btn_popup_cam'):
                self.dashboard.btn_popup_cam.setText("Pop-out Feed")
                self.dashboard.btn_popup_cam.setStyleSheet(
                    "background-color: #1a2332; border: 1px solid #3b82f6; border-radius: 4px; "
                    "color: #f8fafc; font-weight: bold; font-size: 10px; padding: 3px 6px;"
                )
        super().closeEvent(event)


# ==============================================================================
# Live Camera Frame Crop & Target Template Selection Dialog
# ==============================================================================
class CropImageLabel(QLabel):
    """Interactive QLabel providing rubber-band ROI selection on live camera image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.selected_rect: Optional[QRect] = None

    def mousePressEvent(self, event):
        """Begin rubber-band selection rectangle on left click."""
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        """Update rubber-band rectangle dimensions while dragging."""
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        """Finalize rubber-band selection rectangle on mouse release."""
        if event.button() == Qt.LeftButton:
            self.selected_rect = self.rubber_band.geometry()


class TargetCropDialog(QDialog):
    """Modal dialog to capture and crop target template from live Khepera IV camera."""

    def __init__(self, rgb_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📸 Capture & Crop Target from Camera")
        self.setModal(True)
        self.resize(680, 560)
        self.setStyleSheet("background-color: #0b1120; color: #f8fafc;")

        self.rgb_image = rgb_image
        self.img_h, self.img_w = rgb_image.shape[:2]
        self.cropped_bgr: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        lbl_header = QLabel("🎯 Drag a rectangle around the target object, or use full frame:")
        lbl_header.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        layout.addWidget(lbl_header)

        # Image view container
        self.lbl_view = CropImageLabel(self)
        self.lbl_view.setAlignment(Qt.AlignCenter)
        self.lbl_view.setStyleSheet("border: 2px solid #334155; background-color: #020617;")

        qimg = QImage(
            self.rgb_image.data, self.img_w, self.img_h, self.img_w * 3, QImage.Format_RGB888
        )
        self.pixmap = QPixmap.fromImage(qimg)
        self.display_w = min(640, self.img_w)
        self.display_h = int(self.img_h * (self.display_w / max(1, self.img_w)))
        scaled_pix = self.pixmap.scaled(
            self.display_w, self.display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_view.setPixmap(scaled_pix)
        self.lbl_view.setFixedSize(self.display_w, self.display_h)
        layout.addWidget(self.lbl_view, alignment=Qt.AlignCenter)

        # Action Buttons
        btn_row = QWidget()
        b_layout = QHBoxLayout(btn_row)
        b_layout.setContentsMargins(0, 4, 0, 0)
        b_layout.setSpacing(8)

        self.btn_confirm = QPushButton("✂️ Crop & Select Region")
        self.btn_confirm.setCursor(Qt.PointingHandCursor)
        self.btn_confirm.setStyleSheet(
            "background-color: #0284c7; border: 1px solid #38bdf8; border-radius: 4px; "
            "color: white; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_confirm.clicked.connect(self._on_confirm_crop)

        self.btn_full = QPushButton("🖼️ Use Full Frame")
        self.btn_full.setCursor(Qt.PointingHandCursor)
        self.btn_full.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #475569; border-radius: 4px; "
            "color: #cbd5e1; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_full.clicked.connect(self._on_use_full_frame)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet(
            "background-color: #334155; border: 1px solid #64748b; border-radius: 4px; "
            "color: #e2e8f0; padding: 6px 12px;"
        )
        self.btn_cancel.clicked.connect(self.reject)

        b_layout.addStretch()
        b_layout.addWidget(self.btn_full)
        b_layout.addWidget(self.btn_confirm)
        b_layout.addWidget(self.btn_cancel)
        layout.addWidget(btn_row)

    def _on_confirm_crop(self):
        """Extract user-drawn bounding box and convert to OpenCV BGR."""
        rect = self.lbl_view.selected_rect
        if rect is None or rect.width() < 10 or rect.height() < 10:
            self._on_use_full_frame()
            return

        scale_x = self.img_w / float(self.display_w)
        scale_y = self.img_h / float(self.display_h)

        x1 = max(0, int(rect.x() * scale_x))
        y1 = max(0, int(rect.y() * scale_y))
        x2 = min(self.img_w, int((rect.x() + rect.width()) * scale_x))
        y2 = min(self.img_h, int((rect.y() + rect.height()) * scale_y))

        if x2 - x1 < 10 or y2 - y1 < 10:
            self._on_use_full_frame()
            return

        cropped_rgb = self.rgb_image[y1:y2, x1:x2]
        self.cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
        self.accept()

    def _on_use_full_frame(self):
        """Use complete uncropped frame as reference template."""
        self.cropped_bgr = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        self.accept()


# ==============================================================================
# Main Mapping Dashboard Window
# ==============================================================================
class MappingDashboardWindow(QMainWindow):
    """Dedicated Single-Robot Environment Mapping & Exploration Dashboard."""

    def __init__(self, ros_worker: Optional[MappingRosWorker] = None):
        super().__init__()
        self.ros = ros_worker
        if self.ros:
            self.ros.auto_spawn_explorer = True

        # Robot State & Telemetry
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_v = 0.0
        self.odom_w = 0.0
        self.has_received_odom = False
        self.robot_path = collections.deque(maxlen=3000)

        # IMU & Attitude
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_az = 9.81
        self.is_cliff_alert = False
        self.is_turned_over = False

        # Autonomy State
        self.is_auto_exploring = False

        # Interactive Map Drawing & User-Target State
        self.active_goal: Optional[Tuple[float, float]] = None
        self.drawn_path: List[Tuple[float, float]] = []
        self.is_drawing_path: bool = False
        self.is_dragging_target: bool = False
        self.map_interaction_mode: str = 'none'  # 'none', 'click_goal', 'draw_path'

        # Teleoperation Limits & Keyboard State
        self.max_linear_speed = 0.35   # m/s (quick explore cruise speed)
        self.max_angular_speed = 1.0   # rad/s
        self.active_cmd_v = 0.0
        self.active_cmd_w = 0.0
        self.keys_pressed = set()
        self.dpad_direction: Optional[str] = None

        # Map Layers Caches
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_elevation: Optional[OccupancyGrid] = None
        self.latest_surface: Optional[OccupancyGrid] = None
        self.latest_visual_map: Optional[OccupancyGrid] = None
        self.latest_scan: Optional[LaserScan] = None

        # Canvas View Settings
        self.map_layer_mode = 'obstacles'   # 'obstacles', 'elevation', 'surface', 'visual'
        self.show_sensor_beams = True
        self.show_trajectory = True
        self.show_grid = True
        self.show_camera_fov = True
        self.zoom_level = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0

        # Camera & Visual Mapping State
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.camera_frame_count = 0
        self.last_cam_fps_time = time.time()
        self.cam_fps = 0.0
        self.last_camera_render_time: float = 0.0
        self.visual_keyframes: List[Tuple[float, float, float, np.ndarray, str]] = []
        self.fov_patch: Optional[Polygon] = None
        self.keyframe_artists: List[Tuple[Any, Any]] = []
        self.camera_popup: Optional[CameraPopupWindow] = None

        # Object Pursuit & Custom Target Object State
        self.is_pursuit_active: bool = False
        self.latest_executive_state = None
        self.target_pose: Optional[Tuple[float, float, float]] = (1.5, 0.0, 0.0)
        self.latest_detection_frame: Optional[np.ndarray] = None
        self.detection_frame_count: int = 0
        self.detection_fps: float = 0.0
        self.last_det_fps_time: float = time.time()
        self.latest_target_bbox: Optional[Tuple[int, int, int, int, str, float]] = None
        self.selected_target_shape: str = "Cylinder"
        self.selected_target_color = QColor(225, 29, 72)  # Default Red (#e11d48)
        self.last_detection_frame_time: float = 0.0

        # Throttled & Event-Driven Rendering Flags
        self._map_dirty: bool = True
        self._map_layer_dirty: bool = True
        self._map_img_artist = None
        self._last_rendered_map_msg = None
        self._last_rendered_layer_mode: Optional[str] = None
        self._last_rendered_x: float = 0.0
        self._last_rendered_y: float = 0.0
        self._last_rendered_yaw: float = 0.0
        self.use_gpu_canvas: bool = False

        # Build UI
        self.init_ui()

        # Connect ROS signals if worker is supplied
        if self.ros:
            self.ros.odom_received.connect(self.on_odom_received)
            self.ros.imu_received.connect(self.on_imu_received)
            self.ros.map_received.connect(self.on_map_received)
            self.ros.elevation_received.connect(self.on_elevation_received)
            self.ros.surface_received.connect(self.on_surface_received)
            self.ros.scan_received.connect(self.on_scan_received)
            self.ros.cliff_alert_received.connect(self.on_cliff_alert_received)
            self.ros.turned_over_received.connect(self.on_turned_over_received)
            self.ros.feedback_received.connect(self.on_feedback_received)
            self.ros.explorer_state_received.connect(self.on_explorer_state_received)
            self.ros.active_goal_received.connect(self.on_active_goal_received)
            self.ros.screenshot_requested.connect(self._on_save_screenshot)
            self.ros.camera_received.connect(self.on_camera_received)
            self.ros.visual_map_received.connect(self.on_visual_map_received)
            self.ros.executive_state_received.connect(self.on_executive_state_received)
            self.ros.detection_visual_received.connect(self.on_detection_visual_received)
            self.ros.thinking_log_received.connect(self.on_thinking_log_received)
            self.ros.target_status_received.connect(self.on_target_status_received)
            self.ros.target_confidence_received.connect(self.on_target_confidence_received)
            if hasattr(self.ros, 'target_roi_received'):
                self.ros.target_roi_received.connect(self.on_target_roi_received)
            if hasattr(self.ros, 'target_keypoints_count_received'):
                self.ros.target_keypoints_count_received.connect(self.on_keypoints_count_received)

        # Initial pursuit button state
        self._update_pursuit_button_style()

        # Install application event filter for uninterrupted WASD keyboard teleop
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        # Teleop control timer (20 Hz loop for smooth keyboard/D-Pad response)
        self.teleop_timer = QTimer(self)
        self.teleop_timer.timeout.connect(self.update_teleop_loop)
        self.teleop_timer.start(50)

        # Throttled map plot timer: 3 Hz (333 ms) with dirty-flag gating
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self._on_plot_timer_tick)
        self.plot_timer.start(333)

        # Initial plot draw
        self.update_map_plot()

        # Automatically open Live Camera Stream pop-up window on bootup
        QTimer.singleShot(100, self.on_toggle_camera_popup)

    def _on_plot_timer_tick(self):
        """Throttled timer tick (3 Hz): redraws map only if sensor data or pose changed."""
        if self._map_dirty:
            self.update_map_plot()

    def _on_save_screenshot(self, filepath: str):
        """Save a crisp full screenshot of the dashboard window and the map canvas."""
        try:
            target_path = os.path.abspath(filepath)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            self.grab().save(target_path)
            canvas_path = os.path.splitext(target_path)[0] + "_map.png"
            self.canvas.figure.savefig(canvas_path, dpi=120, bbox_inches='tight')
            if hasattr(self, 'lbl_draw_info') and self.lbl_draw_info:
                self.lbl_draw_info.setText(f"Screenshot saved: {os.path.basename(target_path)}")
        except Exception as e:
            print(f"[Dashboard] Error saving screenshot: {e}")

    def init_ui(self):
        self.setWindowTitle("Khepera IV Mapping & Spatial Exploration Station")
        self.setMinimumSize(600, 480)
        primary_screen = QGuiApplication.primaryScreen()
        if primary_screen:
            geom = primary_screen.availableGeometry()
            sw, sh = geom.width(), geom.height()
            self.setGeometry(geom.x(), geom.y(), sw // 2, sh)
        else:
            self.resize(1280, 800)
        self.setStyleSheet(MODERN_STYLE)
        self.setFocusPolicy(Qt.StrongFocus)

        # Central Horizontal Splitter Layout (Draggable between Left Panel & SLAM Map)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("main_horizontal_splitter")

        # ----------------------------------------------------------------------
        # Left Control & Telemetry Panel (Wrapped in ScrollArea for responsiveness)
        # ----------------------------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(750)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 8, 2)
        left_layout.setSpacing(6)

        # 1. Header & Live Connection Badge
        header_widget = QWidget()
        h_layout = QVBoxLayout(header_widget)
        h_layout.setContentsMargins(0, 0, 0, 2)
        h_layout.setSpacing(1)

        lbl_title = QLabel("KHEPERA IV SLAM STATION")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: 900; color: #38bdf8; letter-spacing: 0.5px;")
        lbl_subtitle = QLabel("Single-Robot Spatial Mapping & Autonomous Exploration")
        lbl_subtitle.setStyleSheet("font-size: 10px; color: #94a3b8;")

        self.lbl_connection = QLabel("● ROS 2 JAZZY ONLINE (/cmd_vel DIRECT)")
        self.lbl_connection.setStyleSheet(
            "background-color: #064e3b; border: 1px solid #10b981; border-radius: 4px; "
            "color: #6ee7b7; font-size: 10px; font-weight: bold; padding: 2px 6px; margin-top: 2px;"
        )

        h_layout.addWidget(lbl_title)
        h_layout.addWidget(lbl_subtitle)
        h_layout.addWidget(self.lbl_connection)
        left_layout.addWidget(header_widget)

        # 2. Manual Teleoperation GroupBox
        teleop_group = QGroupBox("Manual Teleoperation (WASD / Interactive D-Pad)")
        teleop_layout = QVBoxLayout(teleop_group)
        teleop_layout.setSpacing(5)

        # Speed Sliders
        speed_widget = QWidget()
        sp_layout = QGridLayout(speed_widget)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.setSpacing(4)

        lbl_v = QLabel("Linear (v):")
        lbl_v.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.slider_v = QSlider(Qt.Horizontal)
        self.slider_v.setRange(2, 30)  # 0.02 - 0.30 m/s
        self.slider_v.setValue(int(self.max_linear_speed * 100))
        self.slider_v.valueChanged.connect(self.on_v_slider_changed)
        self.lbl_v_val = QLabel(f"{self.max_linear_speed:.2f} m/s")
        self.lbl_v_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace;")
        self.lbl_v_val.setFixedWidth(75)

        lbl_w = QLabel("Angular (ω):")
        lbl_w.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.slider_w = QSlider(Qt.Horizontal)
        self.slider_w.setRange(20, 200)  # 0.20 - 2.00 rad/s
        self.slider_w.setValue(int(self.max_angular_speed * 100))
        self.slider_w.valueChanged.connect(self.on_w_slider_changed)
        self.lbl_w_val = QLabel(f"{self.max_angular_speed:.2f} rad/s")
        self.lbl_w_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace;")
        self.lbl_w_val.setFixedWidth(75)

        sp_layout.addWidget(lbl_v, 0, 0)
        sp_layout.addWidget(self.slider_v, 0, 1)
        sp_layout.addWidget(self.lbl_v_val, 0, 2)
        sp_layout.addWidget(lbl_w, 1, 0)
        sp_layout.addWidget(self.slider_w, 1, 1)
        sp_layout.addWidget(self.lbl_w_val, 1, 2)
        teleop_layout.addWidget(speed_widget)

        # Interactive D-Pad Grid
        dpad_container = QWidget()
        dpad_grid = QGridLayout(dpad_container)
        dpad_grid.setContentsMargins(6, 2, 6, 2)
        dpad_grid.setSpacing(5)

        self.btn_fwd = QPushButton("▲ W\nForward")
        self.btn_fwd.setObjectName("dpad_btn")
        self.btn_fwd.pressed.connect(lambda: self.on_dpad_pressed('FORWARD'))
        self.btn_fwd.released.connect(self.on_dpad_released)

        self.btn_left = QPushButton("◄ A\nLeft")
        self.btn_left.setObjectName("dpad_btn")
        self.btn_left.pressed.connect(lambda: self.on_dpad_pressed('LEFT'))
        self.btn_left.released.connect(self.on_dpad_released)

        self.btn_stop = QPushButton("■ Space\nSTOP")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.clicked.connect(self.on_stop_clicked)

        self.btn_right = QPushButton("► D\nRight")
        self.btn_right.setObjectName("dpad_btn")
        self.btn_right.pressed.connect(lambda: self.on_dpad_pressed('RIGHT'))
        self.btn_right.released.connect(self.on_dpad_released)

        self.btn_rev = QPushButton("▼ S\nReverse")
        self.btn_rev.setObjectName("dpad_btn")
        self.btn_rev.pressed.connect(lambda: self.on_dpad_pressed('REVERSE'))
        self.btn_rev.released.connect(self.on_dpad_released)

        dpad_grid.addWidget(self.btn_fwd, 0, 1)
        dpad_grid.addWidget(self.btn_left, 1, 0)
        dpad_grid.addWidget(self.btn_stop, 1, 1)
        dpad_grid.addWidget(self.btn_right, 1, 2)
        dpad_grid.addWidget(self.btn_rev, 2, 1)
        teleop_layout.addWidget(dpad_container)

        # Velocity Status Display
        self.lbl_teleop_cmd = QLabel("Cmd: v = +0.00 m/s | ω = +0.00 rad/s")
        self.lbl_teleop_cmd.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 10px; font-weight: bold; text-align: center;")
        self.lbl_teleop_cmd.setAlignment(Qt.AlignCenter)
        teleop_layout.addWidget(self.lbl_teleop_cmd)

        left_layout.addWidget(teleop_group)

        # 3. Autonomous & User-Guided Exploration GroupBox
        explore_group = QGroupBox("Autonomous && User-Guided Exploration")
        exp_layout = QVBoxLayout(explore_group)
        exp_layout.setSpacing(5)

        self.btn_auto_explore = QPushButton("Start Auto Explore")
        self.btn_auto_explore.setObjectName("auto_explore_idle")
        self.btn_auto_explore.setCursor(Qt.PointingHandCursor)
        self.btn_auto_explore.clicked.connect(self.on_toggle_auto_explore)

        self.lbl_explore_status = QLabel("Mode: Manual Control (Click button to start exploration)")
        self.lbl_explore_status.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.lbl_explore_status.setWordWrap(True)

        exp_layout.addWidget(self.btn_auto_explore)
        exp_layout.addWidget(self.lbl_explore_status)

        # User-guided path and target buttons
        user_draw_row = QWidget()
        udr_layout = QHBoxLayout(user_draw_row)
        udr_layout.setContentsMargins(0, 0, 0, 0)
        udr_layout.setSpacing(4)

        self.btn_click_goal = QPushButton("Goal on Map")
        self.btn_click_goal.setObjectName("user_draw_btn")
        self.btn_click_goal.setCheckable(True)
        self.btn_click_goal.setCursor(Qt.PointingHandCursor)
        self.btn_click_goal.setToolTip("Click anywhere on the 2D map canvas to direct robot navigation.")
        self.btn_click_goal.clicked.connect(self.on_toggle_click_goal_mode)

        self.btn_draw_path = QPushButton("Draw Path")
        self.btn_draw_path.setObjectName("user_draw_btn")
        self.btn_draw_path.setCheckable(True)
        self.btn_draw_path.setCursor(Qt.PointingHandCursor)
        self.btn_draw_path.clicked.connect(self.on_toggle_draw_path_mode)

        self.btn_clear_user_path = QPushButton("Clear")
        self.btn_clear_user_path.setObjectName("user_clear_btn")
        self.btn_clear_user_path.setCursor(Qt.PointingHandCursor)
        self.btn_clear_user_path.clicked.connect(self.on_clear_user_path_clicked)

        self.btn_place_target = QPushButton("Place Target")
        self.btn_place_target.setObjectName("user_draw_btn")
        self.btn_place_target.setCheckable(True)
        self.btn_place_target.setCursor(Qt.PointingHandCursor)
        self.btn_place_target.setToolTip(
            "Click anywhere on the map to relocate target object in simulation."
        )
        self.btn_place_target.clicked.connect(self.on_toggle_place_target_mode)

        udr_layout.addWidget(self.btn_click_goal)
        udr_layout.addWidget(self.btn_draw_path)
        udr_layout.addWidget(self.btn_place_target)
        udr_layout.addWidget(self.btn_clear_user_path)

        self.lbl_draw_info = QLabel("Map Draw: Click 'Goal on Map' or 'Draw Path' to guide robot.")
        self.lbl_draw_info.setStyleSheet("color: #fbbf24; font-size: 9.5px;")
        self.lbl_draw_info.setWordWrap(True)

        exp_layout.addWidget(user_draw_row)
        exp_layout.addWidget(self.lbl_draw_info)
        left_layout.addWidget(explore_group)

        # 3b. Onboard CMOS Camera Feed & Visual Landmarks GroupBox
        cam_group = QGroupBox("Onboard CMOS Camera (640x480)")
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setSpacing(4)

        self.lbl_camera_view = QLabel("Awaiting /camera/image_raw stream...")
        self.lbl_camera_view.setAlignment(Qt.AlignCenter)
        self.lbl_camera_view.setFixedHeight(160)
        self.lbl_camera_view.setStyleSheet(
            "background-color: #020617; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #64748b; font-family: monospace; font-size: 10px;"
        )
        self.lbl_camera_view.setCursor(Qt.PointingHandCursor)
        self.lbl_camera_view.setToolTip("Click to open in a floating, resizable, always-on-top window")
        self.lbl_camera_view.mousePressEvent = lambda e: self.on_toggle_camera_popup()
        cam_layout.addWidget(self.lbl_camera_view)

        cam_bar = QWidget()
        cb_layout = QHBoxLayout(cam_bar)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setSpacing(4)

        self.lbl_camera_status = QLabel("STANDBY | 640x480 | 60° FOV")
        self.lbl_camera_status.setStyleSheet("color: #94a3b8; font-size: 9px; font-family: monospace;")

        self.btn_popup_cam = QPushButton("Pop-out Feed")
        self.btn_popup_cam.setCursor(Qt.PointingHandCursor)
        self.btn_popup_cam.setToolTip("Open camera stream in a separate resizable, always-on-top window")
        self.btn_popup_cam.setStyleSheet(
            "background-color: #1a2332; border: 1px solid #3b82f6; border-radius: 4px; "
            "color: #f8fafc; font-weight: bold; font-size: 10px; padding: 3px 6px;"
        )
        self.btn_popup_cam.clicked.connect(self.on_toggle_camera_popup)

        self.btn_pin_view = QPushButton("Pin View")
        self.btn_pin_view.setCursor(Qt.PointingHandCursor)
        self.btn_pin_view.setStyleSheet(
            "background-color: #0f766e; border: 1px solid #14b8a6; border-radius: 4px; "
            "color: #ccfbf1; font-weight: bold; font-size: 10px; padding: 3px 6px;"
        )
        self.btn_pin_view.clicked.connect(self.on_pin_view_to_map)

        self.btn_snap_cam = QPushButton("📸 Crop / Snap")
        self.btn_snap_cam.setCursor(Qt.PointingHandCursor)
        self.btn_snap_cam.setToolTip(
            "Freeze live camera frame and crop ROI for Mode B pursuit template"
        )
        self.btn_snap_cam.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #38bdf8; border-radius: 4px; "
            "color: #38bdf8; font-weight: bold; font-size: 10px; padding: 3px 6px;"
        )
        self.btn_snap_cam.clicked.connect(self.on_snap_camera_template)

        cb_layout.addWidget(self.lbl_camera_status, stretch=1)
        cb_layout.addWidget(self.btn_popup_cam)
        cb_layout.addWidget(self.btn_pin_view)
        cb_layout.addWidget(self.btn_snap_cam)
        cam_layout.addWidget(cam_bar)

        left_layout.addWidget(cam_group)

        # 4. Attitude & Terrain HUD GroupBox
        hud_group = QGroupBox("Attitude && Terrain HUD")
        hud_layout = QVBoxLayout(hud_group)
        hud_layout.setSpacing(5)

        # Pitch & Roll Cards
        gauges_widget = QWidget()
        g_layout = QHBoxLayout(gauges_widget)
        g_layout.setContentsMargins(0, 0, 0, 0)
        g_layout.setSpacing(6)

        self.card_pitch = QFrame()
        self.card_pitch.setStyleSheet("background-color: #1a2332; border: 1px solid #263345; border-radius: 5px; padding: 4px;")
        cp_layout = QVBoxLayout(self.card_pitch)
        cp_layout.setContentsMargins(4, 2, 4, 2)
        lbl_p_title = QLabel("PITCH")
        lbl_p_title.setStyleSheet("color: #94a3b8; font-size: 9px; font-weight: bold;")
        self.lbl_pitch_val = QLabel("+0.0°")
        self.lbl_pitch_val.setStyleSheet("color: #38bdf8; font-size: 13px; font-weight: 900; font-family: monospace;")
        cp_layout.addWidget(lbl_p_title)
        cp_layout.addWidget(self.lbl_pitch_val)

        self.card_roll = QFrame()
        self.card_roll.setStyleSheet("background-color: #1a2332; border: 1px solid #263345; border-radius: 5px; padding: 4px;")
        cr_layout = QVBoxLayout(self.card_roll)
        cr_layout.setContentsMargins(4, 2, 4, 2)
        lbl_r_title = QLabel("ROLL")
        lbl_r_title.setStyleSheet("color: #94a3b8; font-size: 9px; font-weight: bold;")
        self.lbl_roll_val = QLabel("+0.0°")
        self.lbl_roll_val.setStyleSheet("color: #38bdf8; font-size: 13px; font-weight: 900; font-family: monospace;")
        cr_layout.addWidget(lbl_r_title)
        cr_layout.addWidget(self.lbl_roll_val)

        g_layout.addWidget(self.card_pitch)
        g_layout.addWidget(self.card_roll)
        hud_layout.addWidget(gauges_widget)

        # Terrain Slope Indicator
        self.lbl_terrain_slope = QLabel("Terrain: Flat Floor (Slope: 0.0°)")
        self.lbl_terrain_slope.setStyleSheet(
            "background-color: #131b26; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #cbd5e1; font-size: 10px; font-weight: bold; padding: 4px;"
        )
        hud_layout.addWidget(self.lbl_terrain_slope)

        # Cliff Alert Badge
        self.lbl_cliff_badge = QLabel("CLIFF SAFE")
        self.lbl_cliff_badge.setAlignment(Qt.AlignCenter)
        self.lbl_cliff_badge.setStyleSheet(
            "background-color: #064e3b; border: 1px solid #10b981; border-radius: 5px; "
            "color: #6ee7b7; font-size: 11px; font-weight: 900; padding: 4px;"
        )
        hud_layout.addWidget(self.lbl_cliff_badge)

        # Rollover / Turned Over Badge
        self.lbl_rollover_badge = QLabel("ORIENTATION: UPRIGHT & STABLE")
        self.lbl_rollover_badge.setAlignment(Qt.AlignCenter)
        self.lbl_rollover_badge.setStyleSheet(
            "background-color: #064e3b; border: 1px solid #10b981; border-radius: 5px; "
            "color: #6ee7b7; font-size: 10px; font-weight: bold; padding: 4px;"
        )
        hud_layout.addWidget(self.lbl_rollover_badge)

        # Arena Center Respawn & Hot Reload Buttons
        respawn_row = QWidget()
        rr_layout = QHBoxLayout(respawn_row)
        rr_layout.setContentsMargins(0, 0, 0, 0)
        rr_layout.setSpacing(5)

        self.btn_recenter = QPushButton("Recenter (Origin)")
        self.btn_recenter.setObjectName("recenter_btn")
        self.btn_recenter.setCursor(Qt.PointingHandCursor)
        self.btn_recenter.clicked.connect(self.on_recenter_clicked)
        rr_layout.addWidget(self.btn_recenter)

        self.btn_hot_reload = QPushButton("🔥 Hot Reload")
        self.btn_hot_reload.setObjectName("hot_reload_btn")
        self.btn_hot_reload.setCursor(Qt.PointingHandCursor)
        self.btn_hot_reload.setToolTip(
            "Kill all orphan processes, reset robot to origin, clear maps & pipeline."
        )
        self.btn_hot_reload.setStyleSheet(
            "background-color: #7c2d12; border: 2px solid #ea580c; border-radius: 5px; "
            "color: #ffedd5; font-size: 11px; font-weight: bold; padding: 6px;"
        )
        self.btn_hot_reload.clicked.connect(self.on_hot_reload_clicked)
        rr_layout.addWidget(self.btn_hot_reload)

        hud_layout.addWidget(respawn_row)

        # Physical Robot Network & IP Auto-Discovery Row
        net_row = QWidget()
        net_layout = QHBoxLayout(net_row)
        net_layout.setContentsMargins(0, 0, 0, 0)
        net_layout.setSpacing(5)

        self.lbl_robot_ip = QLabel("📡 IP: Standby")
        self.lbl_robot_ip.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; border-radius: 4px; "
            "color: #38bdf8; font-size: 10px; font-weight: bold; padding: 4px;"
        )
        self.btn_detect_ip = QPushButton("🔍 Auto-Detect IP")
        self.btn_detect_ip.setCursor(Qt.PointingHandCursor)
        self.btn_detect_ip.setToolTip(
            "Scan local hotspot / network ARP cache for active Khepera IV port 5000"
        )
        self.btn_detect_ip.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #0284c7; border-radius: 4px; "
            "color: #e2e8f0; font-size: 10px; font-weight: bold; padding: 4px;"
        )
        self.btn_detect_ip.clicked.connect(self.on_auto_detect_robot_ip)
        net_layout.addWidget(self.lbl_robot_ip, stretch=1)
        net_layout.addWidget(self.btn_detect_ip)
        hud_layout.addWidget(net_row)

        left_layout.addWidget(hud_group)

        # 5. Map Persistence & Spatial Memory GroupBox
        persist_group = QGroupBox("Map Persistence && Memory")
        persist_layout = QVBoxLayout(persist_group)
        persist_layout.setSpacing(5)

        btn_row = QWidget()
        br_layout = QHBoxLayout(btn_row)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.setSpacing(5)

        self.btn_save = QPushButton("Save Map")
        self.btn_save.setObjectName("save_btn")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(self.on_save_map_clicked)

        self.btn_load = QPushButton("Load Map")
        self.btn_load.setObjectName("load_btn")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.on_load_map_clicked)

        self.btn_clear = QPushButton("Clear Map")
        self.btn_clear.setObjectName("clear_btn")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.on_clear_map_clicked)

        br_layout.addWidget(self.btn_save)
        br_layout.addWidget(self.btn_load)
        br_layout.addWidget(self.btn_clear)
        persist_layout.addWidget(btn_row)

        self.btn_export_ply = QPushButton("Export 3DGS .PLY (SuperSplat)")
        self.btn_export_ply.setObjectName("export_ply_btn")
        self.btn_export_ply.setCursor(Qt.PointingHandCursor)
        self.btn_export_ply.setToolTip("Trigger 3D Gaussian Splatting reconstruction & export .ply")
        self.btn_export_ply.clicked.connect(self.on_export_ply_clicked)
        persist_layout.addWidget(self.btn_export_ply)

        self.lbl_map_toast = QLabel("[Ready] Persistence & 3DGS services online.")
        self.lbl_map_toast.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #f59e0b; font-family: monospace; font-size: 9px; padding: 4px;"
        )
        self.lbl_map_toast.setWordWrap(True)
        persist_layout.addWidget(self.lbl_map_toast)

        left_layout.addWidget(persist_group)

        left_layout.addStretch()
        scroll_area.setWidget(left_container)
        self.main_splitter.addWidget(scroll_area)

        # ----------------------------------------------------------------------
        # Right Center Panel: Prominent Real-Time Map Display
        # ----------------------------------------------------------------------
        right_panel = QWidget()
        right_panel.setMinimumWidth(500)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 0, 2, 0)
        right_layout.setSpacing(6)

        # Map Toolbar & Layer Selector
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(2, 2, 2, 2)
        tb_layout.setSpacing(6)

        lbl_layer = QLabel("Map Layer:")
        lbl_layer.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.combo_layer = QComboBox()
        self.combo_layer.addItems([
            '2D Obstacle Map (/map)',
            '2.5D Elevation Map (/elevation_map)',
            'Surface & Cliff Map (/surface_map)',
            'Visual Orthomap (/visual_map)'
        ])
        self.combo_layer.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_layer.setMinimumWidth(180)
        self.combo_layer.view().setMinimumWidth(240)
        self.combo_layer.currentIndexChanged.connect(self.on_layer_changed)

        self.chk_sensor_rays = QCheckBox("Sensor Rays")
        self.chk_sensor_rays.setChecked(True)
        self.chk_sensor_rays.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_sensor_rays.toggled.connect(self.on_sensor_rays_toggled)

        self.chk_camera_fov = QCheckBox("Camera FOV")
        self.chk_camera_fov.setChecked(True)
        self.chk_camera_fov.setStyleSheet("color: #0ea5e9; font-weight: bold;")
        self.chk_camera_fov.toggled.connect(self.on_camera_fov_toggled)

        self.chk_trajectory = QCheckBox("Trajectory")
        self.chk_trajectory.setChecked(True)
        self.chk_trajectory.setStyleSheet("color: #a855f7; font-weight: bold;")
        self.chk_trajectory.toggled.connect(self.on_trajectory_toggled)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.setStyleSheet("color: #94a3b8;")
        self.chk_grid.toggled.connect(self.on_grid_toggled)

        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_in.clicked.connect(self.on_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_out.clicked.connect(self.on_zoom_out)

        self.btn_fit = QPushButton("Fit Arena")
        self.btn_fit.clicked.connect(self.on_fit_arena)

        lbl_view_mode = QLabel("View:")
        lbl_view_mode.setStyleSheet("font-weight: bold; color: #a855f7; margin-left: 6px;")
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems([
            'SLAM Map View',
            'Target & Thinking View',
            'Split View (Both)'
        ])
        self.combo_view_mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_view_mode.setMinimumWidth(150)
        self.combo_view_mode.view().setMinimumWidth(180)
        self.combo_view_mode.setToolTip(
            "Toggle between SLAM Map View, Target & Thinking View, or Split View"
        )
        self.combo_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)

        tb_layout.addWidget(lbl_layer)
        tb_layout.addWidget(self.combo_layer)
        tb_layout.addWidget(lbl_view_mode)
        tb_layout.addWidget(self.combo_view_mode)
        tb_layout.addWidget(self.chk_sensor_rays)
        tb_layout.addWidget(self.chk_camera_fov)
        tb_layout.addWidget(self.chk_trajectory)
        tb_layout.addWidget(self.chk_grid)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_zoom_in)
        tb_layout.addWidget(self.btn_zoom_out)
        tb_layout.addWidget(self.btn_fit)
        right_layout.addWidget(toolbar)

        # Center Vertical Splitter: supports Map Only, Target & Thinking Only, or Split View
        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #1e293b; height: 5px; }"
        )

        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        # Matplotlib Canvas for High-Performance Spatial Map Rendering
        self.figure = Figure(figsize=(8, 7), facecolor='#0b0f19')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #0b0f19;")
        self.ax_map = self.figure.add_subplot(111, facecolor='#0f172a')
        self.figure.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.07)
        canvas_layout.addWidget(self.canvas)

        # Hardware-Accelerated PyQtGraph OpenGL Canvas (Optional GPU Mode)
        if PYQTGRAPH_AVAILABLE:
            try:
                self.pg_plot = pg.PlotWidget()
                self.pg_plot.setBackground('#0f172a')
                self.pg_plot.showGrid(x=True, y=True, alpha=0.3)
                self.pg_image = pg.ImageItem()
                self.pg_plot.addItem(self.pg_image)
                self.pg_plot.hide()
                canvas_layout.addWidget(self.pg_plot)
            except Exception:
                self.pg_plot = None
                self.pg_image = None
        else:
            self.pg_plot = None
            self.pg_image = None

        # Connect Matplotlib Canvas mouse events for user targeting and path drawing
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)

        # Build Dedicated Target Detail & Cognitive Thinking Panel
        self.thinking_target_panel = self._build_thinking_target_panel()

        self.center_splitter.addWidget(self.canvas_container)
        self.center_splitter.addWidget(self.thinking_target_panel)
        self.thinking_target_panel.hide()

        right_layout.addWidget(self.center_splitter)

        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([380, 900])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)

        main_layout.addWidget(self.main_splitter)

    # --------------------------------------------------------------------------
    # Teleoperation Speed Controls & Slider Slots
    # --------------------------------------------------------------------------
    def on_v_slider_changed(self, val: int):
        self.max_linear_speed = val / 100.0
        self.lbl_v_val.setText(f"{self.max_linear_speed:.2f} m/s")
        if self.ros:
            self.ros.set_robot_speed(self.max_linear_speed, self.max_angular_speed)

    def on_w_slider_changed(self, val: int):
        self.max_angular_speed = val / 100.0
        self.lbl_w_val.setText(f"{self.max_angular_speed:.2f} rad/s")
        if self.ros:
            self.ros.set_robot_speed(self.max_linear_speed, self.max_angular_speed)

    # --------------------------------------------------------------------------
    # Interactive D-Pad & Keyboard Event Handling
    # --------------------------------------------------------------------------
    def on_dpad_pressed(self, direction: str):
        self.dpad_direction = direction
        if self.is_auto_exploring:
            self.on_toggle_auto_explore()

    def on_dpad_released(self):
        self.dpad_direction = None

    def on_stop_clicked(self):
        self.dpad_direction = None
        self.keys_pressed.clear()
        self.active_cmd_v = 0.0
        self.active_cmd_w = 0.0
        if self.is_auto_exploring:
            self.on_toggle_auto_explore()
        if self.ros:
            self.ros.kill_explorer_instances()
        self.lbl_teleop_cmd.setText("Cmd: v = +0.00 m/s | ω = +0.00 rad/s [STOPPED]")

    def _is_text_input_focused(self, widget: Optional[QObject] = None) -> bool:
        """Check whether the given widget or the active focus widget is a text input."""
        text_classes = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
        candidates = [widget, QApplication.focusWidget(), self.focusWidget()]
        for w in candidates:
            curr = w
            while curr is not None:
                if isinstance(curr, text_classes):
                    return True
                try:
                    if isinstance(curr, QWidget):
                        curr = curr.parentWidget()
                    elif hasattr(curr, 'parent') and callable(curr.parent):
                        curr = curr.parent()
                    else:
                        curr = None
                except (TypeError, AttributeError):
                    curr = None

        text_widgets = (
            getattr(self, 'edit_target_desc', None),
            getattr(self, 'edit_target_name', None),
            getattr(self, 'edit_target_img_path', None),
            getattr(self, 'spin_target_pos_x', None),
            getattr(self, 'spin_target_pos_y', None),
            getattr(self, 'txt_thinking_log', None),
        )
        for tw in text_widgets:
            if tw is not None and hasattr(tw, 'hasFocus') and tw.hasFocus():
                return True
        return False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Pass keys through without teleop capture when typing in text fields."""
        if self._is_text_input_focused(obj):
            if self.keys_pressed:
                self.keys_pressed.clear()
            return super().eventFilter(obj, event)

        teleop_keys = (
            Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
            Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
            Qt.Key_Space
        )

        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in teleop_keys:
                if not event.isAutoRepeat():
                    self.keys_pressed.add(key)
                    if self.is_auto_exploring:
                        self.on_toggle_auto_explore()
                return True
        elif event.type() == QEvent.KeyRelease:
            key = event.key()
            if key in teleop_keys:
                if not event.isAutoRepeat() and key in self.keys_pressed:
                    self.keys_pressed.remove(key)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle dashboard key press with suppression for active text inputs."""
        if self._is_text_input_focused():
            super().keyPressEvent(event)
            return

        teleop_keys = (
            Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
            Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
            Qt.Key_Space
        )
        key = event.key()
        if not event.isAutoRepeat() and key in teleop_keys:
            self.keys_pressed.add(key)
            if self.is_auto_exploring:
                self.on_toggle_auto_explore()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle dashboard key release with suppression for active text inputs."""
        if self._is_text_input_focused():
            super().keyReleaseEvent(event)
            return

        if not event.isAutoRepeat() and event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())
        super().keyReleaseEvent(event)

    def update_teleop_loop(self):
        """20 Hz Teleoperation controller combining keyboard WASD and tactile D-Pad."""
        if self.is_turned_over:
            if abs(self.active_cmd_v) > 0.001 or abs(self.active_cmd_w) > 0.001:
                self.active_cmd_v = 0.0
                self.active_cmd_w = 0.0
                if self.ros:
                    self.ros.publish_cmd(0.0, 0.0)
            self.lbl_teleop_cmd.setText("Cmd: v = +0.00 m/s | ω = +0.00 rad/s [LOCKED - ROBOT TURNED OVER]")
            return

        target_v = 0.0
        target_w = 0.0

        k_fwd = (Qt.Key_W in self.keys_pressed) or (Qt.Key_Up in self.keys_pressed)
        k_rev = (Qt.Key_S in self.keys_pressed) or (Qt.Key_Down in self.keys_pressed)
        k_left = (Qt.Key_A in self.keys_pressed) or (Qt.Key_Left in self.keys_pressed)
        k_right = (Qt.Key_D in self.keys_pressed) or (Qt.Key_Right in self.keys_pressed)
        k_stop = (Qt.Key_Space in self.keys_pressed)

        active_border = "border: 2px solid #38bdf8; background-color: #0284c7; color: white;"
        idle_border = ""
        self.btn_fwd.setStyleSheet(active_border if k_fwd else idle_border)
        self.btn_left.setStyleSheet(active_border if k_left else idle_border)
        self.btn_right.setStyleSheet(active_border if k_right else idle_border)
        self.btn_rev.setStyleSheet(active_border if k_rev else idle_border)
        self.btn_stop.setStyleSheet(active_border if k_stop else idle_border)

        if k_stop:
            self.on_stop_clicked()
            return

        if k_fwd and not k_rev:
            target_v = self.max_linear_speed
        elif k_rev and not k_fwd:
            target_v = -self.max_linear_speed

        if k_left and not k_right:
            target_w = self.max_angular_speed
        elif k_right and not k_left:
            target_w = -self.max_angular_speed

        if self.dpad_direction == 'FORWARD':
            target_v = self.max_linear_speed
        elif self.dpad_direction == 'REVERSE':
            target_v = -self.max_linear_speed
        elif self.dpad_direction == 'LEFT':
            target_w = self.max_angular_speed
        elif self.dpad_direction == 'RIGHT':
            target_w = -self.max_angular_speed

        if abs(target_v) > 0.001 or abs(target_w) > 0.001 or abs(self.active_cmd_v) > 0.001 or abs(self.active_cmd_w) > 0.001:
            self.active_cmd_v = target_v
            self.active_cmd_w = target_w
            if self.ros:
                self.ros.publish_cmd(target_v, target_w)
            self.lbl_teleop_cmd.setText(f"Cmd: v = {target_v:+.2f} m/s | ω = {target_w:+.2f} rad/s")
            if hasattr(self, 'camera_popup') and self.camera_popup and self.camera_popup.isVisible():
                self.camera_popup._update_status_text()

    # --------------------------------------------------------------------------
    # Autonomous Exploration Slots
    # --------------------------------------------------------------------------
    def on_toggle_auto_explore(self):
        new_state = not self.is_auto_exploring
        self.is_auto_exploring = new_state
        if new_state:
            # Auto explore uses quick explore algorithm with NO pursue
            if getattr(self, 'is_pursuit_active', False):
                self.on_toggle_pursuit_clicked()
            elif self.ros:
                self.ros.set_pursuit_enable(False)
        if self.ros:
            self.ros.set_auto_explore(new_state)
            if new_state:
                self.ros.set_robot_speed(self.max_linear_speed, self.max_angular_speed)

        if new_state:
            self.btn_auto_explore.setObjectName("auto_explore_active")
            self.btn_auto_explore.setText("■ Stop Auto Explore")
            self.lbl_explore_status.setText("Mode: Autonomous Frontier Exploration Active")
            self.lbl_explore_status.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px;")
        else:
            self.btn_auto_explore.setObjectName("auto_explore_idle")
            self.btn_auto_explore.setText("▶ Start Auto Explore")
            self.lbl_explore_status.setText("Mode: Manual Control (Click button to start exploration)")
            self.lbl_explore_status.setStyleSheet("color: #94a3b8; font-size: 10px;")

        self.btn_auto_explore.style().unpolish(self.btn_auto_explore)
        self.btn_auto_explore.style().polish(self.btn_auto_explore)

    def on_explorer_state_received(self, active: bool):
        if self.is_auto_exploring != active:
            self.is_auto_exploring = active
            if active:
                self.btn_auto_explore.setObjectName("auto_explore_active")
                self.btn_auto_explore.setText("■ Stop Auto Explore")
                self.lbl_explore_status.setText("Mode: Autonomous Frontier Exploration Active")
                self.lbl_explore_status.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px;")
            else:
                self.btn_auto_explore.setObjectName("auto_explore_idle")
                self.btn_auto_explore.setText("▶ Start Auto Explore")
                self.lbl_explore_status.setText("Mode: Manual Control (Click button to start exploration)")
                self.lbl_explore_status.setStyleSheet("color: #94a3b8; font-size: 10px;")
            self.btn_auto_explore.style().unpolish(self.btn_auto_explore)
            self.btn_auto_explore.style().polish(self.btn_auto_explore)

    def on_active_goal_received(self, gx: float, gy: float):
        """Update active exploration target received from /explorer/active_goal."""
        self.active_goal = (gx, gy)

    def on_toggle_click_goal_mode(self):
        """Toggle mode to click anywhere on map to set exploration target."""
        if self.btn_click_goal.isChecked():
            self.map_interaction_mode = 'click_goal'
            self.btn_draw_path.setChecked(False)
            if hasattr(self, 'btn_place_target'):
                self.btn_place_target.setChecked(False)
            self.lbl_draw_info.setText("Click anywhere on the map to set a courageous target.")
            self.lbl_draw_info.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 9.5px;")
        else:
            self.map_interaction_mode = 'none'
            self.lbl_draw_info.setText("Interactive: Select tool above to guide robot.")
            self.lbl_draw_info.setStyleSheet("color: #94a3b8; font-size: 9.5px;")

    def on_toggle_draw_path_mode(self):
        """Toggle mode to draw an exploration path on map with mouse drag."""
        if self.btn_draw_path.isChecked():
            self.map_interaction_mode = 'draw_path'
            self.btn_click_goal.setChecked(False)
            if hasattr(self, 'btn_place_target'):
                self.btn_place_target.setChecked(False)
            self.lbl_draw_info.setText("Drag mouse across map to draw exploration path.")
            self.lbl_draw_info.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 9.5px;")
        else:
            self.map_interaction_mode = 'none'
            self.lbl_draw_info.setText("Interactive: Select tool above to guide robot.")
            self.lbl_draw_info.setStyleSheet("color: #94a3b8; font-size: 9.5px;")

    def on_toggle_place_target_mode(self):
        """Toggle mode to click on 2D map canvas to relocate target object in Gazebo."""
        if hasattr(self, 'btn_place_target') and self.btn_place_target.isChecked():
            self.map_interaction_mode = 'place_target'
            self.btn_click_goal.setChecked(False)
            self.btn_draw_path.setChecked(False)
            self.lbl_draw_info.setText("Click anywhere on the map canvas to place target object.")
            self.lbl_draw_info.setStyleSheet("color: #10b981; font-weight: bold; font-size: 9.5px;")
        else:
            self.map_interaction_mode = 'none'
            self.lbl_draw_info.setText("Interactive: Select tool above to guide robot or place target.")
            self.lbl_draw_info.setStyleSheet("color: #94a3b8; font-size: 9.5px;")

    def on_clear_user_path_clicked(self):
        """Clear user path and waypoints and revert to autonomous exploration."""
        self.drawn_path.clear()
        self.active_goal = None
        self.btn_click_goal.setChecked(False)
        self.btn_draw_path.setChecked(False)
        if hasattr(self, 'btn_place_target'):
            self.btn_place_target.setChecked(False)
        self.map_interaction_mode = 'none'
        if self.ros:
            self.ros.send_clear_path()
        self.lbl_draw_info.setText("User path and targets cleared. Resumed autonomous mode.")
        self.lbl_draw_info.setStyleSheet("color: #94a3b8; font-size: 9.5px;")
        self.update_map_plot()

    def on_canvas_press(self, event):
        """Handle canvas mouse press for goal clicking, path drawing, or target drag/placement."""
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return
        x_canvas = float(event.xdata)
        y_canvas = float(event.ydata)
        x_world = float(np.clip(y_canvas, -2.40, 2.40))
        y_world = float(np.clip(-x_canvas, -2.40, 2.40))

        # Check for direct interactive target object dragging anytime
        if hasattr(self, 'target_pose') and self.target_pose is not None:
            tx, ty = self.target_pose[0], self.target_pose[1]
            c_tx = -ty
            c_ty = tx
            if math.hypot(x_canvas - c_tx, y_canvas - c_ty) <= 0.25:
                self.is_dragging_target = True
                self.canvas.setCursor(Qt.ClosedHandCursor)
                if hasattr(self, 'lbl_draw_info'):
                    self.lbl_draw_info.setText(
                        f"Dragging target object... ({x_world:+.2f}, {y_world:+.2f})"
                    )
                    self.lbl_draw_info.setStyleSheet("color: #38bdf8; font-size: 9.5px;")
                return

        if self.map_interaction_mode == 'click_goal':
            self.active_goal = (x_world, y_world)
            if self.ros:
                self.ros.send_user_goal(x_world, y_world)
            self.lbl_draw_info.setText(
                f"Target set at ({x_world:.2f}, {y_world:.2f}). Starting autonomous pursuit."
            )
            self.btn_click_goal.setChecked(False)
            self.map_interaction_mode = 'none'
            if not self.is_auto_exploring:
                self.on_toggle_auto_explore()
            self.update_map_plot()

        elif self.map_interaction_mode == 'place_target':
            self.target_pose = (x_world, y_world, 0.0)
            if hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
                self.spin_target_pos_x.blockSignals(True)
                self.spin_target_pos_x.setValue(x_world)
                self.spin_target_pos_x.blockSignals(False)
                self.spin_target_pos_y.blockSignals(True)
                self.spin_target_pos_y.setValue(y_world)
                self.spin_target_pos_y.blockSignals(False)
            if hasattr(self, 'lbl_target_pose'):
                self.lbl_target_pose.setText(
                    f"Target Pose: (X: {x_world:+.2f}m, Y: {y_world:+.2f}m)"
                )
            if hasattr(self, 'lbl_draw_info'):
                self.lbl_draw_info.setText(f"Target relocated to ({x_world:.2f}, {y_world:.2f}).")
                self.lbl_draw_info.setStyleSheet(
                    "color: #10b981; font-weight: bold; font-size: 9.5px;"
                )
            if hasattr(self, 'btn_place_target'):
                self.btn_place_target.setChecked(False)
            self.map_interaction_mode = 'none'
            self.relocate_target(x_world, y_world, 0.0)
            self.update_map_plot()

        elif self.map_interaction_mode == 'draw_path':
            self.is_drawing_path = True
            self.drawn_path = [(x_world, y_world)]
            self.lbl_draw_info.setText("Drawing path... (1 point)")
            self.update_map_plot()

    def on_canvas_motion(self, event):
        """Handle canvas mouse drag for target dragging or continuous path drawing."""
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return
        x_canvas = float(event.xdata)
        y_canvas = float(event.ydata)
        x_world = float(np.clip(y_canvas, -2.40, 2.40))
        y_world = float(np.clip(-x_canvas, -2.40, 2.40))

        if getattr(self, 'is_dragging_target', False):
            self.target_pose = (x_world, y_world, 0.0)
            if hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
                self.spin_target_pos_x.blockSignals(True)
                self.spin_target_pos_x.setValue(x_world)
                self.spin_target_pos_x.blockSignals(False)
                self.spin_target_pos_y.blockSignals(True)
                self.spin_target_pos_y.setValue(y_world)
                self.spin_target_pos_y.blockSignals(False)
            if hasattr(self, 'lbl_target_pose'):
                self.lbl_target_pose.setText(
                    f"Target Pose: (X: {x_world:+.2f}m, Y: {y_world:+.2f}m)"
                )
            if hasattr(self, 'lbl_draw_info'):
                self.lbl_draw_info.setText(f"Dragging target: ({x_world:+.2f}, {y_world:+.2f})")
            self._update_target_marker_canvas()
            return

        # Cursor hover feedback when near target
        if not self.is_drawing_path and self.map_interaction_mode == 'none':
            if hasattr(self, 'target_pose') and self.target_pose is not None:
                tx, ty = self.target_pose[0], self.target_pose[1]
                c_tx = -ty
                c_ty = tx
                if math.hypot(x_canvas - c_tx, y_canvas - c_ty) <= 0.25:
                    self.canvas.setCursor(Qt.OpenHandCursor)
                else:
                    self.canvas.unsetCursor()

        if not self.is_drawing_path or self.map_interaction_mode != 'draw_path':
            return

        if not self.drawn_path:
            self.drawn_path.append((x_world, y_world))
            return

        lx, ly = self.drawn_path[-1]
        if math.hypot(x_world - lx, y_world - ly) >= 0.08:  # 8cm spacing
            self.drawn_path.append((x_world, y_world))
            self.lbl_draw_info.setText(f"Drawing path: {len(self.drawn_path)} waypoints...")
            self.update_map_plot()

    def on_canvas_release(self, event):
        """Handle canvas mouse release to finalize target drag or dispatch drawn path."""
        if getattr(self, 'is_dragging_target', False):
            self.is_dragging_target = False
            self.canvas.unsetCursor()
            if event.inaxes == self.ax_map and event.xdata is not None and event.ydata is not None:
                x_world = float(np.clip(event.ydata, -2.40, 2.40))
                y_world = float(np.clip(-event.xdata, -2.40, 2.40))
                self.target_pose = (x_world, y_world, 0.0)
            if hasattr(self, 'target_pose') and self.target_pose is not None:
                tx, ty = self.target_pose[0], self.target_pose[1]
                if hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
                    self.spin_target_pos_x.blockSignals(True)
                    self.spin_target_pos_x.setValue(tx)
                    self.spin_target_pos_x.blockSignals(False)
                    self.spin_target_pos_y.blockSignals(True)
                    self.spin_target_pos_y.setValue(ty)
                    self.spin_target_pos_y.blockSignals(False)
                if hasattr(self, 'lbl_target_pose'):
                    self.lbl_target_pose.setText(f"Target Pose: (X: {tx:+.2f}m, Y: {ty:+.2f}m)")
                if hasattr(self, 'lbl_draw_info'):
                    self.lbl_draw_info.setText(
                        f"Target placed at ({tx:.2f}, {ty:.2f}). Click '📍 Relocate' to commit."
                    )
                    self.lbl_draw_info.setStyleSheet(
                        "color: #10b981; font-weight: bold; font-size: 9.5px;"
                    )
            self._update_target_marker_canvas()
            return

        if self.is_drawing_path:
            self.is_drawing_path = False
            if len(self.drawn_path) >= 2:
                if self.ros:
                    self.ros.send_user_path(self.drawn_path)
                self.lbl_draw_info.setText(
                    f"Path dispatched ({len(self.drawn_path)} waypoints). Starting exploration!"
                )
                self.lbl_draw_info.setStyleSheet(
                    "color: #10b981; font-weight: bold; font-size: 9.5px;"
                )
                if not self.is_auto_exploring:
                    self.on_toggle_auto_explore()
            self.btn_draw_path.setChecked(False)
            self.map_interaction_mode = 'none'
            self.update_map_plot()

    def on_toggle_pursuit_clicked(self):
        """Toggle autonomous visual pursuit mode on and off."""
        new_state = not self.is_pursuit_active
        self.is_pursuit_active = new_state
        if hasattr(self, 'lbl_pursuit_status'):
            status_str = "ACTIVE" if new_state else "STANDBY"
            self.lbl_pursuit_status.setText(f"Pursuit: {status_str}")
            self.lbl_pursuit_status.setStyleSheet(
                "color: #10b981; font-weight: bold; font-size: 10px;" if new_state
                else "color: #94a3b8; font-weight: bold; font-size: 10px;"
            )
        self._update_pursuit_button_style()

        target_name = (
            self.edit_target_name.text().strip()
            if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
            else "Target Object"
        )
        target_desc = (
            self.edit_target_desc.text().strip()
            if hasattr(self, 'edit_target_desc') and self.edit_target_desc.text().strip()
            else ""
        )

        if new_state:
            if self.ros:
                self.ros._ensure_explorer_running()

            tx, ty = (1.50, 0.00)
            if hasattr(self, 'target_pose') and self.target_pose is not None:
                tx, ty = self.target_pose[0], self.target_pose[1]
            elif hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
                tx, ty = self.spin_target_pos_x.value(), self.spin_target_pos_y.value()

            self.on_thinking_log_received(
                f"[Pursuit Active] Robot steering toward '{target_name}' at arena coords "
                f"(x={tx:+.2f}, y={ty:+.2f}) while scanning for visual verification!"
            )

            # 2. Configure pursuit mode (Semantic Description vs Reference Image)
            is_semantic = (
                self.radio_mode_semantic.isChecked()
                if hasattr(self, 'radio_mode_semantic') else True
            )
            if is_semantic:
                if self.ros and hasattr(self.ros, 'publish_pursuit_mode'):
                    self.ros.publish_pursuit_mode("semantic")
                # Class name is primary discriminator; description is secondary attribute
                prompt_text = f"{target_name} | {target_desc}" if target_desc else target_name
                if self.ros:
                    self.ros.publish_target_prompt(prompt_text)
                self.on_thinking_log_received(
                    f"🎯 [Pursuit] Mode A Semantic YOLO-World engaged for '{prompt_text}'."
                )
                self._ensure_textured_gazebo_target(force_option_a=True)
            else:
                if self.ros and hasattr(self.ros, 'publish_pursuit_mode'):
                    self.ros.publish_pursuit_mode("template")
                self.on_deploy_target_template()
                self.on_thinking_log_received(
                    f"🎯 [Pursuit] Mode B Reference Image Template engaged for '{target_name}'."
                )

            # 3. Reset stop state and enable visual pursuit controller
            if self.ros:
                if hasattr(self.ros, 'publish_target_stopped'):
                    self.ros.publish_target_stopped(False)
                if hasattr(self.ros, 'set_explorer_enable'):
                    self.ros.set_explorer_enable(True)
                if hasattr(self.ros, 'set_pursuit_enable'):
                    self.ros.set_pursuit_enable(True)
                if hasattr(self, '_sync_speed_limits_to_robot'):
                    self._sync_speed_limits_to_robot()

            if hasattr(self, 'lbl_thinking_motion_badge'):
                self.lbl_thinking_motion_badge.setText("TARGET_PURSUIT")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 9.5px; "
                    "padding: 2px 6px; border-radius: 3px;"
                )
        else:
            if self.ros:
                if hasattr(self.ros, 'set_pursuit_enable'):
                    self.ros.set_pursuit_enable(False)
                if hasattr(self.ros, 'set_explorer_enable'):
                    self.ros.set_explorer_enable(False)
                if hasattr(self.ros, 'publish_target_stopped'):
                    self.ros.publish_target_stopped(True)
                if hasattr(self.ros, 'kill_explorer_instances'):
                    self.ros.kill_explorer_instances()
            self.on_thinking_log_received("[Pursuit] Target pursuit disengaged.")
            if hasattr(self, 'lbl_thinking_motion_badge'):
                self.lbl_thinking_motion_badge.setText("MANUAL_OR_IDLE")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #1e293b; color: #94a3b8; font-weight: bold; font-size: 9.5px; "
                    "padding: 2px 6px; border-radius: 3px;"
                )
            self.on_stop_clicked()

        if hasattr(self, 'camera_popup') and self.camera_popup:
            self.camera_popup.update_pursuit_state(new_state)
        self.update_map_plot()

    def on_executive_state_received(self, msg: Any):
        """Update pursuit status indicator from /executive/status topic."""
        self.latest_executive_state = msg
        state_str = "UNKNOWN"
        try:
            if isinstance(msg, str):
                data = json.loads(msg)
                if isinstance(data, dict):
                    state_str = data.get("state", "UNKNOWN")
            elif hasattr(msg, "data") and isinstance(msg.data, str):
                data = json.loads(msg.data)
                if isinstance(data, dict):
                    state_str = data.get("state", "UNKNOWN")
        except Exception:
            state_str = "UNKNOWN"
        if state_str == "UNKNOWN":
            state_str = getattr(msg, 'current_state', getattr(msg, 'state', 'UNKNOWN'))

        if hasattr(self, 'lbl_pursuit_status'):
            self.lbl_pursuit_status.setText(f"Pursuit: {state_str}")
            if state_str in ('IDLE', 'STANDBY', 'TERMINAL_STOP'):
                self.is_pursuit_active = False
                self.lbl_pursuit_status.setStyleSheet(
                    "color: #94a3b8; font-weight: bold; font-size: 10px;"
                )
            else:
                self.is_pursuit_active = True
                self.lbl_pursuit_status.setStyleSheet(
                    "color: #10b981; font-weight: bold; font-size: 10px;"
                )
            self._update_pursuit_button_style()

        if hasattr(self, 'lbl_thinking_motion_badge'):
            if state_str == 'TERMINAL_STOP':
                self.lbl_thinking_motion_badge.setText("TARGET_DOCKED")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #065f46; color: #6ee7b7; font-weight: bold; "
                    "font-size: 9.5px; padding: 2px 6px; border-radius: 3px;"
                )
            elif state_str in ('VISUAL_PURSUIT', 'TARGET_RECOVERY'):
                self.lbl_thinking_motion_badge.setText("TARGET_PURSUIT")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; font-weight: bold; "
                    "font-size: 9.5px; padding: 2px 6px; border-radius: 3px;"
                )
            elif state_str == 'FRONTIER_EXPLORATION':
                self.lbl_thinking_motion_badge.setText("FRONTIER_SEARCH")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #1e3a8a; color: #60a5fa; font-weight: bold; "
                    "font-size: 9.5px; padding: 2px 6px; border-radius: 3px;"
                )
            elif state_str in ('IDLE', 'STANDBY'):
                self.lbl_thinking_motion_badge.setText("IDLE / STANDBY")
                self.lbl_thinking_motion_badge.setStyleSheet(
                    "background-color: #1e293b; color: #94a3b8; font-weight: bold; "
                    "font-size: 9.5px; padding: 2px 6px; border-radius: 3px;"
                )

        if hasattr(self, 'camera_popup') and self.camera_popup:
            self.camera_popup.update_pursuit_state(self.is_pursuit_active)

    def _update_pursuit_button_style(self):
        if hasattr(self, 'btn_toggle_pursuit'):
            if self.is_pursuit_active:
                self.btn_toggle_pursuit.setText("Stop Object Pursuit")
                self.btn_toggle_pursuit.setObjectName("pursuit_btn_active")
                self.btn_toggle_pursuit.setStyleSheet(
                    "background-color: #7f1d1d; border: 2px solid #ef4444; border-radius: 6px; "
                    "color: #ffffff; font-size: 11px; font-weight: bold; padding: 6px 12px;"
                )
            else:
                self.btn_toggle_pursuit.setText("Start Object Pursuit")
                self.btn_toggle_pursuit.setObjectName("pursuit_btn_idle")
                self.btn_toggle_pursuit.setStyleSheet(
                    "background-color: #064e3b; border: 2px solid #10b981; border-radius: 6px; "
                    "color: #6ee7b7; font-size: 11px; font-weight: bold; padding: 6px 12px;"
                )

        for btn, lbl in [
            (
                getattr(self, 'btn_target_panel_pursuit', None),
                getattr(self, 'lbl_target_panel_pursuit_status', None),
            ),
            (
                getattr(self, 'btn_option_a_pursuit', None),
                getattr(self, 'lbl_option_a_pursuit_status', None),
            ),
        ]:
            if btn is not None and lbl is not None:
                if self.is_pursuit_active:
                    btn.setText("Stop Pursuit")
                    btn.setStyleSheet(
                        "background-color: #7f1d1d; border: 1px solid #ef4444; color: #ffffff; font-weight: bold;"
                    )
                    lbl.setText("Pursuit: ACTIVE")
                    lbl.setStyleSheet("color: #10b981; font-weight: bold; font-size: 10px;")
                else:
                    btn.setText("Start Pursuit")
                    btn.setStyleSheet(
                        "background-color: #064e3b; border: 1px solid #10b981; color: #6ee7b7; font-weight: bold;"
                    )
                    lbl.setText("Pursuit: STANDBY")
                    lbl.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 10px;")

    def on_detection_visual_received(self, msg: RosImage):
        """Handle annotated detection visual stream (/target/detection_visual) with green bounding boxes."""
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                h, w = msg.height, msg.width
                img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                if msg.encoding == 'bgr8':
                    rgb_arr = np.ascontiguousarray(img_arr[:, :, ::-1])
                else:
                    rgb_arr = np.ascontiguousarray(img_arr)
                self.latest_detection_frame = rgb_arr.copy()
                self.last_detection_frame_time = time.time()

                self.detection_frame_count += 1
                now = time.time()
                dt = now - self.last_det_fps_time
                if dt >= 1.0:
                    self.detection_fps = self.detection_frame_count / dt
                    self.detection_frame_count = 0
                    self.last_det_fps_time = now

                if hasattr(self, 'camera_popup') and self.camera_popup:
                    self.camera_popup.update_detection_frame(rgb_arr, w, h, self.detection_fps)
        except Exception:
            pass

    def _ensure_textured_gazebo_target(
        self,
        img_path: Optional[str] = None,
        target_name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        yaw: float = 0.0,
        force_option_a: bool = False,
        force_option_b: bool = False
    ):
        """Generate SDF model and spawn/replace in Gazebo Sim.

        Option A spawns solid colored 3D primitives (Cylinder, Cube, Sphere) with NO texture image.
        Option B wraps the selected reference template image around a Cube.
        """
        if x is None or y is None:
            if hasattr(self, 'target_pose') and self.target_pose is not None:
                x, y = self.target_pose[0], self.target_pose[1]
            elif hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
                x, y = self.spin_target_pos_x.value(), self.spin_target_pos_y.value()
            else:
                x, y = 1.50, 0.00

        if target_name is None:
            target_name = (
                self.edit_target_name.text().strip()
                if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
                else "Target Object"
            )

        if force_option_b:
            is_mode_a = False
        elif force_option_a:
            is_mode_a = True
        elif img_path is not None and os.path.isfile(img_path):
            is_mode_a = False
        else:
            is_mode_a = (
                hasattr(self, 'radio_mode_semantic') and self.radio_mode_semantic.isChecked()
            )

        # Thread safety: extract all GUI widget states on main thread
        shape_str = (
            self.combo_target_shape.currentText().lower()
            if hasattr(self, 'combo_target_shape') else ""
        )
        col = getattr(self, 'selected_target_color', QColor(225, 29, 72))
        col_r, col_g, col_b = col.redF(), col.greenF(), col.blueF()

        effective_img = img_path
        if (effective_img is None or not os.path.isfile(effective_img)) and hasattr(
            self, 'edit_target_img_path'
        ):
            effective_img = self.edit_target_img_path.text().strip()
        if not effective_img or not os.path.isfile(effective_img):
            default_path = self._get_default_target_template_path()
            if default_path and os.path.isfile(default_path):
                effective_img = default_path

        def _spawn_bg():
            target_dir = "/tmp/khepera4_target"
            os.makedirs(target_dir, exist_ok=True)
            model_config_file = os.path.join(target_dir, "model.config")
            sdf_file = os.path.join(target_dir, "target.sdf")
            with open(model_config_file, "w", encoding="utf-8") as f:
                f.write("""<?xml version="1.0"?>
<model>
  <name>custom_target_object</name>
  <version>1.0</version>
  <sdf version="1.8">target.sdf</sdf>
  <description>Custom Target Object</description>
</model>""")

            has_texture = False
            if is_mode_a:
                # Option A: Do NOT use custom image when deploying/spawning for Option A
                if "cylinder" in shape_str:
                    geom_xml = "<cylinder><radius>0.05</radius><length>0.12</length></cylinder>"
                elif "cube" in shape_str or "box" in shape_str:
                    geom_xml = "<box><size>0.12 0.12 0.12</size></box>"
                elif "sphere" in shape_str:
                    geom_xml = "<sphere><radius>0.06</radius></sphere>"
                elif "cylinder" in target_name.lower():
                    geom_xml = "<cylinder><radius>0.05</radius><length>0.12</length></cylinder>"
                elif "sphere" in target_name.lower():
                    geom_xml = "<sphere><radius>0.06</radius></sphere>"
                else:
                    geom_xml = "<box><size>0.12 0.12 0.12</size></box>"

                mat_xml = f"""<material>
          <ambient>{col_r:.3f} {col_g:.3f} {col_b:.3f} 1.0</ambient>
          <diffuse>{col_r:.3f} {col_g:.3f} {col_b:.3f} 1.0</diffuse>
          <specular>0.3 0.3 0.3 1.0</specular>
        </material>"""
            else:
                # Option B: Input a template image and wrap it around a cube as a texture
                geom_xml = "<box><size>0.12 0.12 0.12</size></box>"
                # Unique timestamp filename to bust Ogre2 GPU texture cache
                ts = int(time.time() * 1000)
                tex_file = os.path.join(target_dir, f"target_texture_{ts}_target_texture.png")
                canonical_tex = os.path.join(target_dir, "target_texture.png")

                if effective_img and os.path.isfile(effective_img):
                    try:
                        src_img = cv2.imread(effective_img)
                        if src_img is not None:
                            h, w = src_img.shape[:2]
                            max_dim = 512
                            scale = min(1.0, max_dim / max(h, w))
                            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                            resized = cv2.resize(
                                src_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                            )
                            cv2.imwrite(tex_file, resized)
                            cv2.imwrite(canonical_tex, resized)
                            has_texture = True
                    except Exception:
                        has_texture = False

                if has_texture:
                    mat_xml = f"""<material>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
          <pbr>
            <metal>
              <albedo_map>{tex_file}</albedo_map>
              <roughness>0.6</roughness>
              <metalness>0.1</metalness>
            </metal>
          </pbr>
        </material>"""
                else:
                    mat_xml = """<material>
          <ambient>0.9 0.1 0.15 1.0</ambient>
          <diffuse>0.95 0.15 0.2 1.0</diffuse>
          <specular>0.3 0.3 0.3 1.0</specular>
        </material>"""

            sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="custom_target_object">
    <static>false</static>
    <pose>{x:.4f} {y:.4f} 0.06 0 0 {yaw:.4f}</pose>
    <link name="target_link">
      <inertial>
        <mass>0.250</mass>
        <inertia>
          <ixx>0.0004</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.0004</iyy><iyz>0</iyz><izz>0.0002</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>{geom_xml}</geometry>
        <surface>
          <friction>
            <ode><mu>1.2</mu><mu2>1.2</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>{geom_xml}</geometry>
        {mat_xml}
      </visual>
    </link>
  </model>
</sdf>"""
            with open(sdf_file, "w", encoding="utf-8") as f:
                f.write(sdf_content)

            if not self.is_simulation:
                return

            # Discover active Gazebo world name dynamically
            world_name = "khepera4_world"
            try:
                proc = subprocess.run(
                    ['gz', 'service', '-l'],
                    capture_output=True, text=True, timeout=2.0
                )
                for line in proc.stdout.splitlines():
                    parts = line.strip().split('/')
                    if len(parts) >= 4 and parts[1] == 'world' and parts[3] in ('create', 'remove'):
                        world_name = parts[2]
                        break
            except Exception:
                pass

            # Remove previous target instance from world if it exists
            subprocess.run([
                'gz', 'service',
                '-s', f'/world/{world_name}/remove',
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "custom_target_object" type: MODEL'
            ], capture_output=True, timeout=3.0)

            time.sleep(0.15)  # Allow Gazebo one simulation cycle to finalize entity deletion

            # Create new model
            subprocess.run([
                'gz', 'service',
                '-s', f'/world/{world_name}/create',
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '2500',
                '--req', f'sdf_filename: "{sdf_file}"'
            ], capture_output=True, timeout=3.5)

        threading.Thread(target=_spawn_bg, daemon=True).start()

    def relocate_target(self, x: float, y: float, yaw: float = 0.0):
        """Relocate custom target object in Gazebo Harmonic immediately and update map marker."""
        if not math.isfinite(x):
            x = 1.5
        if not math.isfinite(y):
            y = 0.0
        if not math.isfinite(yaw):
            yaw = 0.0
        x = float(np.clip(x, -2.40, 2.40))
        y = float(np.clip(y, -2.40, 2.40))
        self.target_pose = (x, y, yaw)
        if hasattr(self, 'lbl_target_pose'):
            self.lbl_target_pose.setText(f"Target Pose: (X: {x:+.2f}m, Y: {y:+.2f}m)")

        def _relocate_bg():
            if not self.is_simulation:
                return
            target_z = 0.06
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            req_str = (
                f'name: "custom_target_object" '
                f'position {{ x: {x:.4f} y: {y:.4f} z: {target_z:.4f} }} '
                f'orientation {{ x: 0.0 y: 0.0 z: {qz:.4f} w: {qw:.4f} }}'
            )
            cmd = [
                'gz', 'service',
                '-s', '/world/khepera4_world/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1500',
                '--req', req_str
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
                # If set_pose fails (model not yet spawned or removed), recreate it
                if res.returncode != 0 or 'data: true' not in res.stdout:
                    self._ensure_textured_gazebo_target(x=x, y=y, yaw=yaw)
            except Exception:
                pass

        threading.Thread(target=_relocate_bg, daemon=True).start()
        self.update_map_plot()

    # --------------------------------------------------------------------------
    # Dedicated Target Detail & Cognitive Telemetry Panel & Slots
    # --------------------------------------------------------------------------
    def _build_thinking_target_panel(self) -> QWidget:
        """Construct the dedicated Target Specification & Cognitive Telemetry panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #0b0f19; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_target_spec_group())
        layout.addWidget(self._build_cognitive_telemetry_group())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_target_mode_row(self) -> QWidget:
        """Build mode selection radio button controls for pursuit pipeline."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl_mode = QLabel("Pursuit Mode:")
        lbl_mode.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.radio_mode_semantic = QRadioButton("Mode A: Semantic Description (YOLO-World)")
        self.radio_mode_semantic.setChecked(True)
        self.radio_mode_semantic.setStyleSheet("color: #f8fafc; font-weight: bold;")
        self.radio_mode_template = QRadioButton(
            "Mode B: Reference Image Template (DINOv2 + LightGlue)"
        )
        self.radio_mode_template.setStyleSheet("color: #f8fafc; font-weight: bold;")
        self.pursuit_mode_group = QButtonGroup(self)
        self.pursuit_mode_group.addButton(self.radio_mode_semantic)
        self.pursuit_mode_group.addButton(self.radio_mode_template)
        self.radio_mode_semantic.toggled.connect(self._on_pursuit_mode_toggled)
        layout.addWidget(lbl_mode)
        layout.addWidget(self.radio_mode_semantic)
        layout.addWidget(self.radio_mode_template)
        layout.addStretch()
        return row

    def _on_pursuit_mode_toggled(self, is_semantic: bool):
        """Broadcast active pursuit mode to ROS network and update stream log."""
        mode_str = "semantic" if is_semantic else "template"
        if self.ros and hasattr(self.ros, 'publish_pursuit_mode'):
            self.ros.publish_pursuit_mode(mode_str)
        if hasattr(self, 'option_a_row') and self.option_a_row:
            self.option_a_row.setVisible(is_semantic)
        if hasattr(self, 'option_b_row') and self.option_b_row:
            self.option_b_row.setVisible(not is_semantic)
        if is_semantic:
            self._ensure_textured_gazebo_target(force_option_a=True)
            self.on_thinking_log_received(
                "🔄 Pursuit Mode switched to Mode A: Semantic Description (Solid 3D Primitive)."
            )
        else:
            self.on_thinking_log_received(
                "🔄 Pursuit Mode switched to Mode B: Reference Image Template (DINOv2 + LightGlue)."
            )

    def _build_target_placement_row(self) -> QWidget:
        """Build target class name, coordinate inputs, and placement/randomize buttons."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl_name = QLabel("Class / Name:")
        lbl_name.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.edit_target_name = QLineEdit("Red Cylinder")
        self.edit_target_name.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 4px; padding: 3px;"
        )
        self.spin_target_pos_x = QDoubleSpinBox()
        self.spin_target_pos_x.setRange(-2.40, 2.40)
        self.spin_target_pos_x.setSingleStep(0.1)
        self.spin_target_pos_x.setValue(1.50)
        self.spin_target_pos_x.valueChanged.connect(self._on_target_spinbox_changed)
        self.spin_target_pos_y = QDoubleSpinBox()
        self.spin_target_pos_y.setRange(-2.40, 2.40)
        self.spin_target_pos_y.setSingleStep(0.1)
        self.spin_target_pos_y.setValue(0.00)
        self.spin_target_pos_y.valueChanged.connect(self._on_target_spinbox_changed)
        self.btn_relocate_target_input = QPushButton("📍 Relocate")
        self.btn_relocate_target_input.setCursor(Qt.PointingHandCursor)
        self.btn_relocate_target_input.setToolTip("Relocate target to designated (X, Y) coords.")
        self.btn_relocate_target_input.clicked.connect(self.on_relocate_target_from_inputs)

        self.btn_randomize_target = QPushButton("🎲 Randomize")
        self.btn_randomize_target.setCursor(Qt.PointingHandCursor)
        self.btn_randomize_target.setToolTip(
            "Randomize target placement in front of robot within arena bounds."
        )
        self.btn_randomize_target.clicked.connect(self.on_randomize_target_clicked)

        layout.addWidget(lbl_name)
        layout.addWidget(self.edit_target_name, stretch=2)
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.spin_target_pos_x)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.spin_target_pos_y)
        layout.addWidget(self.btn_relocate_target_input)
        layout.addWidget(self.btn_randomize_target)
        return row

    def _build_target_desc_row(self) -> QWidget:
        """Build target semantic description text input row."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl_desc = QLabel("Description:")
        lbl_desc.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.edit_target_desc = QLineEdit("")
        self.edit_target_desc.setPlaceholderText("Enter target semantic description...")
        self.edit_target_desc.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 4px; padding: 3px;"
        )
        layout.addWidget(lbl_desc)
        layout.addWidget(self.edit_target_desc, stretch=1)
        return row

    def _on_target_spinbox_changed(self, value: float = 0.0):
        """Synchronize target pose preview from spinbox adjustments when not dragging."""
        if getattr(self, 'is_dragging_target', False):
            return
        if hasattr(self, 'spin_target_pos_x') and hasattr(self, 'spin_target_pos_y'):
            x = self.spin_target_pos_x.value()
            y = self.spin_target_pos_y.value()
            self.target_pose = (x, y, 0.0)
            if hasattr(self, 'lbl_target_pose'):
                self.lbl_target_pose.setText(f"Target Pose: (X: {x:+.2f}m, Y: {y:+.2f}m)")
            self._update_target_marker_canvas()

    def _update_target_marker_canvas(self):
        """Update target marker and text position on the map canvas smoothly."""
        if not hasattr(self, '_target_rect') or not hasattr(self, '_target_marker'):
            self.update_map_plot()
            return
        if hasattr(self, 'target_pose') and self.target_pose is not None:
            tx, ty = self.target_pose[0], self.target_pose[1]
            box_size = 0.12
            c_tx = -ty
            c_ty = tx
            self._target_rect.set_xy((c_tx - box_size / 2.0, c_ty - box_size / 2.0))
            self._target_marker.set_data([c_tx], [c_ty])
            if hasattr(self, '_target_text'):
                self._target_text.set_position((c_tx + 0.08, c_ty + 0.08))
                self._target_text.set_text(f"Target Object ({tx:.2f}, {ty:.2f})")
                self._target_text.set_visible(True)
            self._target_rect.set_visible(True)
            self._target_marker.set_visible(True)
            self.canvas.draw_idle()

    def _update_color_button_style(self):
        """Update color picker button display with current selected QColor."""
        col = getattr(self, 'selected_target_color', QColor(225, 29, 72))
        lum = col.red() * 0.299 + col.green() * 0.587 + col.blue() * 0.114
        text_col = "#ffffff" if lum < 140 else "#000000"
        if hasattr(self, 'btn_target_color') and self.btn_target_color:
            self.btn_target_color.setStyleSheet(
                f"background-color: {col.name()}; color: {text_col}; font-weight: bold; "
                f"border: 1px solid #475569; border-radius: 4px; padding: 3px 8px;"
            )
            self.btn_target_color.setText(f"🎨 {col.name().upper()}")

    def on_pick_target_color(self):
        """Open interactive color picker wheel dialog for Option A."""
        init_col = getattr(self, 'selected_target_color', QColor(225, 29, 72))
        color = QColorDialog.getColor(
            init_col, self, "Select 3D Object Color (Option A)",
            QColorDialog.ShowAlphaChannel if False else QColorDialog.ColorDialogOptions()
        )
        if color.isValid():
            self.selected_target_color = color
            self._update_color_button_style()
            self.on_thinking_log_received(
                f"🎨 [Color Picked] Option A color set to {color.name()} (RGB: {color.red()}, {color.green()}, {color.blue()})"
            )
            self._ensure_textured_gazebo_target(force_option_a=True)

    def on_target_shape_changed(self, shape_name: str):
        """Handle 3D shape dropdown selection changes for Option A."""
        self.selected_target_shape = shape_name
        if hasattr(self, 'edit_target_name') and any(
            s in self.edit_target_name.text() for s in ("Cylinder", "Cube", "Sphere")
        ):
            self.edit_target_name.setText(shape_name)
        self.on_thinking_log_received(f"📐 [Shape Selected] Option A shape set to '{shape_name}'.")
        self._ensure_textured_gazebo_target(target_name=shape_name, force_option_a=True)

    def on_deploy_target_shape(self):
        """Deploy selected 3D primitive shape and solid color to Gazebo simulation (Option A)."""
        shape = (
            self.combo_target_shape.currentText()
            if hasattr(self, 'combo_target_shape') else "Cylinder"
        )
        target_name = (
            self.edit_target_name.text().strip()
            if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
            else shape
        )
        target_desc = (
            self.edit_target_desc.text().strip()
            if hasattr(self, 'edit_target_desc') and self.edit_target_desc.text().strip()
            else ""
        )
        self._ensure_textured_gazebo_target(target_name=target_name, force_option_a=True)
        if hasattr(self, 'btn_deploy_shape'):
            self.btn_deploy_shape.setText("✓ Deployed!")
            QTimer.singleShot(1500, lambda: self.btn_deploy_shape.setText("🚀 Deploy Shape"))
        col = getattr(self, 'selected_target_color', QColor(225, 29, 72))
        prompt_text = f"{target_name} | {target_desc}" if target_desc else target_name
        if self.ros:
            if hasattr(self.ros, 'publish_pursuit_mode'):
                self.ros.publish_pursuit_mode("semantic")
            if hasattr(self.ros, 'publish_target_prompt'):
                self.ros.publish_target_prompt(prompt_text)
        self.on_feedback_received(f"[Target Operator] Deployed 3D {shape} ({col.name()})")
        self.on_thinking_log_received(
            f"🎯 [Shape Deployed] 3D {shape} ({col.name()}) active for Option A "
            f"(No texture image used). Prompt: '{prompt_text}'"
        )

    def _build_option_a_row(self) -> QWidget:
        """Build Option A controls: 3D shape dropdown, color picker wheel, deploy shape."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl_shape = QLabel("3D Shape:")
        lbl_shape.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.combo_target_shape = QComboBox()
        self.combo_target_shape.addItems(["Cylinder", "Cube", "Sphere"])
        self.combo_target_shape.setCurrentText("Cylinder")
        self.combo_target_shape.currentTextChanged.connect(self.on_target_shape_changed)

        lbl_col = QLabel("Color:")
        lbl_col.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_target_color = QPushButton("🎨 #E11D48")
        self.btn_target_color.setCursor(Qt.PointingHandCursor)
        self.btn_target_color.setToolTip("Open Color Picker Wheel to select 3D object color")
        self.btn_target_color.clicked.connect(self.on_pick_target_color)
        self._update_color_button_style()

        self.btn_deploy_shape = QPushButton("🚀 Deploy Shape")
        self.btn_deploy_shape.setCursor(Qt.PointingHandCursor)
        self.btn_deploy_shape.setStyleSheet(
            "background-color: #065f46; border: 1px solid #10b981; "
            "color: #ffffff; font-weight: bold; padding: 4px 8px; border-radius: 4px;"
        )
        self.btn_deploy_shape.setToolTip("Spawn selected solid 3D shape and color into Gazebo (No texture)")
        self.btn_deploy_shape.clicked.connect(self.on_deploy_target_shape)

        self.btn_option_a_pursuit = QPushButton("Start Pursuit")
        self.btn_option_a_pursuit.setCursor(Qt.PointingHandCursor)
        self.btn_option_a_pursuit.clicked.connect(self.on_toggle_pursuit_clicked)
        self.lbl_option_a_pursuit_status = QLabel("Pursuit: STANDBY")
        self.lbl_option_a_pursuit_status.setStyleSheet(
            "color: #94a3b8; font-weight: bold; font-size: 10px;"
        )

        layout.addWidget(lbl_shape)
        layout.addWidget(self.combo_target_shape)
        layout.addWidget(lbl_col)
        layout.addWidget(self.btn_target_color)
        layout.addWidget(self.btn_deploy_shape)
        layout.addWidget(self.btn_option_a_pursuit)
        layout.addWidget(self.lbl_option_a_pursuit_status)
        layout.addStretch()
        return row

    def _build_target_template_row(self) -> QWidget:
        """Build target template image picker and pursuit activation controls."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl_tmpl = QLabel("Template:")
        lbl_tmpl.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.edit_target_img_path = QLineEdit()
        self.edit_target_img_path.setPlaceholderText("Path to .png / .jpg template...")
        self.edit_target_img_path.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 4px; padding: 3px;"
        )
        self.btn_browse_template = QPushButton("Browse...")
        self.btn_browse_template.clicked.connect(self.on_browse_target_template)

        self.btn_snap_template = QPushButton("📸 Snap Live")
        self.btn_snap_template.setCursor(Qt.PointingHandCursor)
        self.btn_snap_template.setToolTip(
            "Capture live frame from Khepera IV camera and crop target object"
        )
        self.btn_snap_template.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #38bdf8; "
            "color: #38bdf8; font-weight: bold; padding: 3px 6px;"
        )
        self.btn_snap_template.clicked.connect(self.on_snap_camera_template)

        self.btn_deploy_template = QPushButton("Deploy Template")
        self.btn_deploy_template.setStyleSheet(
            "background-color: #065f46; border: 1px solid #10b981; "
            "color: #ffffff; font-weight: bold;"
        )
        self.btn_deploy_template.clicked.connect(self.on_deploy_target_template)

        self.lbl_target_img_preview = QLabel("No Img")
        self.lbl_target_img_preview.setFixedSize(45, 45)
        self.lbl_target_img_preview.setAlignment(Qt.AlignCenter)
        self.lbl_target_img_preview.setStyleSheet(
            "border: 1px solid #334155; border-radius: 4px; "
            "background-color: #0f172a; color: #64748b; font-size: 8px;"
        )

        self.btn_target_panel_pursuit = QPushButton("Start Pursuit")
        self.btn_target_panel_pursuit.clicked.connect(self.on_toggle_pursuit_clicked)
        self.lbl_target_panel_pursuit_status = QLabel("Pursuit: STANDBY")
        self.lbl_target_panel_pursuit_status.setStyleSheet(
            "color: #94a3b8; font-weight: bold; font-size: 10px;"
        )

        layout.addWidget(lbl_tmpl)
        layout.addWidget(self.edit_target_img_path, stretch=1)
        layout.addWidget(self.btn_browse_template)
        layout.addWidget(self.btn_snap_template)
        layout.addWidget(self.btn_deploy_template)
        layout.addWidget(self.lbl_target_img_preview)
        layout.addWidget(self.btn_target_panel_pursuit)
        layout.addWidget(self.lbl_target_panel_pursuit_status)

        default_tmpl = self._get_default_target_template_path()
        if default_tmpl and os.path.isfile(default_tmpl):
            self.edit_target_img_path.setText(default_tmpl)
            pix = QPixmap(default_tmpl)
            if not pix.isNull():
                self.lbl_target_img_preview.setPixmap(
                    pix.scaled(45, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        return row

    def _build_target_spec_group(self) -> QGroupBox:
        """Build target specification, image template picker, and deployment controls."""
        group = QGroupBox("Target Specification && Pursuit Enrollment")
        layout = QVBoxLayout(group)
        layout.addWidget(self._build_target_mode_row())
        layout.addWidget(self._build_target_placement_row())
        layout.addWidget(self._build_target_desc_row())
        self.option_a_row = self._build_option_a_row()
        self.option_b_row = self._build_target_template_row()
        layout.addWidget(self.option_a_row)
        layout.addWidget(self.option_b_row)

        is_semantic = (
            self.radio_mode_semantic.isChecked()
            if hasattr(self, 'radio_mode_semantic') else True
        )
        self.option_a_row.setVisible(is_semantic)
        self.option_b_row.setVisible(not is_semantic)
        return group

    def _build_spatial_intent_card(self) -> QFrame:
        """Build spatial reasoning and navigational intent card."""
        card = QFrame()
        card.setStyleSheet(
            "background-color: #111827; border: 1px solid #1e293b; "
            "border-radius: 6px; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setSpacing(3)
        h_a = QHBoxLayout()
        lbl_title = QLabel("Spatial Intent (Where should I go?):")
        lbl_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.lbl_thinking_motion_badge = QLabel("AUTONOMOUS_EXPLORATION")
        self.lbl_thinking_motion_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-weight: bold; font-size: 9.5px; "
            "padding: 2px 6px; border-radius: 3px;"
        )
        h_a.addWidget(lbl_title)
        h_a.addStretch()
        h_a.addWidget(self.lbl_thinking_motion_badge)
        layout.addLayout(h_a)

        self.lbl_thinking_nav_intent = QLabel(
            "Exploring frontiers: Seeking informative areas in arena..."
        )
        self.lbl_thinking_nav_intent.setWordWrap(True)
        self.lbl_thinking_nav_intent.setStyleSheet(
            "color: #cbd5e1; font-size: 10.5px; font-style: italic;"
        )
        layout.addWidget(self.lbl_thinking_nav_intent)
        return card

    def _build_perception_card(self) -> QFrame:
        """Build target perception and feature matching status card."""
        card = QFrame()
        card.setStyleSheet(
            "background-color: #111827; border: 1px solid #1e293b; "
            "border-radius: 6px; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setSpacing(3)
        h_b = QHBoxLayout()
        lbl_title = QLabel("Perception (Did I see target?):")
        lbl_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.lbl_target_sighting_badge = QLabel("● NO TARGET IN SIGHT")
        self.lbl_target_sighting_badge.setStyleSheet(
            "background-color: #450a0a; color: #f87171; font-weight: bold; font-size: 9.5px; "
            "padding: 2px 6px; border-radius: 3px;"
        )
        h_b.addWidget(lbl_title)
        h_b.addStretch()
        h_b.addWidget(self.lbl_target_sighting_badge)
        layout.addLayout(h_b)

        m_row = QHBoxLayout()
        self.lbl_thinking_dinov2_sim = QLabel("-- %")
        self.lbl_thinking_dinov2_sim.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.lbl_thinking_keypoints = QLabel("--")
        self.lbl_thinking_keypoints.setStyleSheet("color: #a855f7; font-weight: bold;")
        m_row.addWidget(QLabel("DINOv2 Match:"))
        m_row.addWidget(self.lbl_thinking_dinov2_sim)
        m_row.addSpacing(10)
        m_row.addWidget(QLabel("LightGlue Inliers:"))
        m_row.addWidget(self.lbl_thinking_keypoints)
        m_row.addStretch()
        layout.addLayout(m_row)
        return card

    def _build_stream_log_box(self) -> QWidget:
        """Build cognitive stream of consciousness log display and clear button."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        log_h = QHBoxLayout()
        lbl_log = QLabel("Stream of Consciousness (/robot/thinking_log):")
        lbl_log.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px;")
        btn_clr = QPushButton("Clear Thoughts")
        btn_clr.clicked.connect(lambda: self.txt_thinking_log.clear())
        log_h.addWidget(lbl_log)
        log_h.addStretch()
        log_h.addWidget(btn_clr)
        layout.addLayout(log_h)

        self.txt_thinking_log = QTextEdit()
        self.txt_thinking_log.setReadOnly(True)
        self.txt_thinking_log.setMinimumHeight(130)
        self.txt_thinking_log.setStyleSheet(
            "background-color: #030712; border: 1px solid #1f2937; border-radius: 4px; "
            "color: #a7f3d0; font-family: monospace; font-size: 10px; padding: 4px;"
        )
        layout.addWidget(self.txt_thinking_log)
        return box

    def _build_cognitive_telemetry_group(self) -> QGroupBox:
        """Build telemetry group showing what the Khepera robot is thinking."""
        group = QGroupBox("Khepera IV Cognitive Telemetry && Live Reasoning State")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        cards = QWidget()
        c_layout = QHBoxLayout(cards)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(8)
        c_layout.addWidget(self._build_spatial_intent_card(), stretch=1)
        c_layout.addWidget(self._build_perception_card(), stretch=1)
        layout.addWidget(cards)

        layout.addWidget(self._build_stream_log_box())
        return group

    def on_view_mode_changed(self, index: int):
        """Switch between SLAM Map View, Target & Thinking View, or Split View."""
        if index == 0:  # SLAM Map View
            self.canvas_container.show()
            self.thinking_target_panel.hide()
        elif index == 1:  # Target & Thinking View
            self.canvas_container.hide()
            self.thinking_target_panel.show()
        elif index == 2:  # Split View (Both)
            self.canvas_container.show()
            self.thinking_target_panel.show()
            total_h = self.center_splitter.height()
            if total_h <= 100:
                total_h = 700
            self.center_splitter.setSizes([total_h // 2, total_h // 2])

    def on_thinking_log_received(self, text: str):
        """Append thought to cognitive log and update intent/sighting status."""
        lower_t = text.lower()
        # Suppress internal speed adjustment telemetry to maintain cognitive reasoning focus
        if "speed adjusted" in lower_t or "cruise v =" in lower_t:
            return

        timestamp = time.strftime("%H:%M:%S")
        self.txt_thinking_log.append(f"[{timestamp}] {text}")
        if "verified target" in lower_t or "target confirmed" in lower_t or "saw target" in lower_t:
            self.lbl_target_sighting_badge.setText("● TARGET CONFIRMED & TRACKED")
            self.lbl_target_sighting_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            self.lbl_thinking_nav_intent.setText(text)
            self.lbl_thinking_motion_badge.setText("VISUAL_PURSUIT")
            self.lbl_thinking_motion_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
        elif "candidate" in lower_t or "yolo" in lower_t or "target detected" in lower_t:
            self.lbl_target_sighting_badge.setText("● CANDIDATE DETECTED (Verifying...)")
            self.lbl_target_sighting_badge.setStyleSheet(
                "background-color: #78350f; color: #fde047; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            self.lbl_thinking_nav_intent.setText(text)
        elif "lost" in lower_t or "searching" in lower_t or "frontier" in lower_t:
            if "lost" in lower_t or "searching" in lower_t:
                self.lbl_target_sighting_badge.setText("● SEARCHING FOR TARGET")
                self.lbl_target_sighting_badge.setStyleSheet(
                    "background-color: #450a0a; color: #f87171; font-weight: bold; font-size: 9.5px; "
                    "padding: 2px 6px; border-radius: 3px;"
                )
            self.lbl_thinking_nav_intent.setText(text)
            self.lbl_thinking_motion_badge.setText("AUTONOMOUS_EXPLORATION")
            self.lbl_thinking_motion_badge.setStyleSheet(
                "background-color: #1e293b; color: #38bdf8; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
        elif ("where should i go" in lower_t or "relocated target" in lower_t or
              "target relocated" in lower_t or "target placed" in lower_t or
              "target randomized" in lower_t):
            self.lbl_thinking_nav_intent.setText(text)

    def on_target_status_received(self, verified: bool):
        """Handle perception target verification status updates."""
        if verified:
            self.lbl_target_sighting_badge.setText("● TARGET CONFIRMED & TRACKED")
            self.lbl_target_sighting_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            self.lbl_thinking_motion_badge.setText("VISUAL_PURSUIT")
            self.lbl_thinking_motion_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )
        else:
            self.lbl_target_sighting_badge.setText("● NO TARGET IN SIGHT")
            self.lbl_target_sighting_badge.setStyleSheet(
                "background-color: #450a0a; color: #f87171; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )

    def on_target_confidence_received(self, conf: float):
        """Update DINOv2 / LightGlue similarity telemetry."""
        pct = max(0.0, min(100.0, conf * 100.0))
        self.lbl_thinking_dinov2_sim.setText(f"{pct:.1f}%")
        if conf >= 0.78:
            self.lbl_thinking_dinov2_sim.setStyleSheet("color: #34d399; font-weight: bold;")
        else:
            self.lbl_thinking_dinov2_sim.setStyleSheet("color: #94a3b8;")

    def on_target_roi_received(
        self, x1: int, y1: int, x2: int, y2: int, label: str, score: float
    ) -> None:
        """Cache latest target bounding box coordinates and update verified DINOv2 match."""
        self.latest_target_bbox = (x1, y1, x2, y2, label, score)
        self.last_target_bbox_time = time.time()
        if hasattr(self, 'lbl_thinking_dinov2_sim') and score > 0.0:
            pct = max(0.0, min(100.0, score * 100.0))
            self.lbl_thinking_dinov2_sim.setText(f"{pct:.1f}%")
            if score >= 0.78:
                self.lbl_thinking_dinov2_sim.setStyleSheet("color: #34d399; font-weight: bold;")
            else:
                self.lbl_thinking_dinov2_sim.setStyleSheet("color: #94a3b8;")

    def on_keypoints_count_received(self, count: int) -> None:
        """Update LightGlue inlier count telemetry display."""
        if hasattr(self, 'lbl_thinking_keypoints'):
            self.lbl_thinking_keypoints.setText(str(count) if count > 0 else "--")

    @property
    def is_simulation(self) -> bool:
        """Return True if running in Gazebo simulation (use_sim_time=True)."""
        ros_node = getattr(self.ros, 'node', None) if self.ros else None
        if ros_node and hasattr(ros_node, 'has_parameter') and ros_node.has_parameter('use_sim_time'):
            try:
                return bool(ros_node.get_parameter('use_sim_time').value)
            except Exception:
                return False
        return False

    def on_auto_detect_robot_ip(self):
        """Scan local hotspot / network ARP cache for active Khepera IV TCP daemon."""
        if hasattr(self, 'btn_detect_ip'):
            self.btn_detect_ip.setEnabled(False)
            self.btn_detect_ip.setText("Scanning...")
        if hasattr(self, 'lbl_robot_ip'):
            self.lbl_robot_ip.setText("📡 Scanning ARP table & subnets...")

        def _scan_worker():
            found_ip = None
            try:
                candidates = []
                if os.path.isfile('/proc/net/arp'):
                    with open('/proc/net/arp', 'r', encoding='utf-8') as f:
                        for line in f.readlines()[1:]:
                            parts = line.split()
                            if parts and len(parts) >= 4 and parts[2] != '0x0':
                                ip = parts[0].strip()
                                if (
                                    ip not in ('0.0.0.0', '255.255.255.255')
                                    and not ip.startswith('127.')
                                ):
                                    candidates.append(ip)

                for base in ('10.42.0.', '192.168.137.', '192.168.1.'):
                    for host in range(2, 60):
                        cand = f"{base}{host}"
                        if cand not in candidates:
                            candidates.append(cand)

                for ip in candidates:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(0.08)
                        res = s.connect_ex((ip, 5000))
                        s.close()
                        if res == 0:
                            found_ip = ip
                            break
                    except Exception:
                        continue
            except Exception:
                pass

            def _update_ui():
                if hasattr(self, 'btn_detect_ip'):
                    self.btn_detect_ip.setEnabled(True)
                    self.btn_detect_ip.setText("🔍 Auto-Detect IP")
                if hasattr(self, 'lbl_robot_ip'):
                    if found_ip:
                        self.lbl_robot_ip.setText(f"📡 IP: {found_ip} (Online)")
                        self.lbl_robot_ip.setStyleSheet(
                            "background-color: #064e3b; border: 1px solid #10b981; "
                            "border-radius: 4px; color: #6ee7b7; font-size: 10px; "
                            "font-weight: bold; padding: 4px;"
                        )
                        self.on_thinking_log_received(
                            f"✅ [Hotspot IP] Found active Khepera IV at {found_ip}:5000"
                        )
                        self.on_feedback_received(f"[Network] Khepera IV detected at {found_ip}")
                    else:
                        self.lbl_robot_ip.setText("📡 IP: Not Detected")
                        self.lbl_robot_ip.setStyleSheet(
                            "background-color: #450a0a; border: 1px solid #ef4444; "
                            "border-radius: 4px; color: #fca5a5; font-size: 10px; "
                            "font-weight: bold; padding: 4px;"
                        )
                        self.on_thinking_log_received(
                            "⚠️ [Hotspot IP] No active Khepera IV found on port 5000."
                        )

            QTimer.singleShot(0, _update_ui)

        threading.Thread(target=_scan_worker, daemon=True).start()

    def on_snap_camera_template(self):
        """Open interactive camera snapshot and ROI crop dialog to deploy target template."""
        if not hasattr(self, 'latest_camera_frame') or self.latest_camera_frame is None:
            self.on_feedback_received("⚠️ No camera frame received on /camera/image_raw yet.")
            self.on_thinking_log_received(
                "⚠️ [Snapshot] Awaiting /camera/image_raw stream before capturing."
            )
            return

        dialog = TargetCropDialog(self.latest_camera_frame, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.cropped_bgr is not None:
            cropped = dialog.cropped_bgr
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                self.on_feedback_received("⚠️ Cropped region too small. Please select a larger area.")
                return

            save_path = "/tmp/khepera_live_target_template.png"
            cv2.imwrite(save_path, cropped)

            if hasattr(self, 'edit_target_img_path'):
                self.edit_target_img_path.setText(save_path)
            if hasattr(self, 'lbl_target_img_preview'):
                pix = QPixmap(save_path)
                if not pix.isNull():
                    self.lbl_target_img_preview.setPixmap(
                        pix.scaled(45, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )

            target_name = (
                self.edit_target_name.text().strip()
                if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
                else "Live Target"
            )
            self.on_thinking_log_received(
                f"📸 [Live Target Captured] Reference template ({cropped.shape[1]}x{cropped.shape[0]} px) "
                f"captured from Khepera IV camera for '{target_name}'."
            )
            self.on_deploy_target_template()

    def on_browse_target_template(self):
        """Open file dialog to pick reference target template image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target Template Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.edit_target_img_path.setText(path)
            pix = QPixmap(path)
            if not pix.isNull():
                self.lbl_target_img_preview.setPixmap(
                    pix.scaled(45, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    def _get_default_target_template_path(self) -> str:
        """Find default simulated target template snapshot in workspace."""
        candidates = [
            os.path.expanduser(
                "~/Desktop/khepera4_ws/src/khepera_visual_pursuit/sample_targets/snapshots/target_1.jpg"
            ),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return ""

    def on_deploy_target_template(self):
        """Deploy selected template image to perception pipeline via ROS topic."""
        path = self.edit_target_img_path.text().strip()
        if not path or not os.path.isfile(path):
            default_path = self._get_default_target_template_path()
            if default_path and os.path.isfile(default_path):
                path = default_path
                self.edit_target_img_path.setText(path)
                pix = QPixmap(path)
                if not pix.isNull():
                    self.lbl_target_img_preview.setPixmap(
                        pix.scaled(45, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )

        if not path or not os.path.isfile(path):
            self.on_feedback_received(
                "[Template Deploy] Please browse and select a template image (.jpg / .png)."
            )
            self.on_thinking_log_received("⚠️ [Template] Please browse and select a template image.")
            return

        img = cv2.imread(path)
        if img is None:
            self.on_feedback_received(f"[Template Deploy] Failed to decode image: {path}")
            return

        target_name = self.edit_target_name.text().strip() or "Target Object"

        if self.ros:
            msg = RosImage()
            if self.ros.node and self.ros.node.context.ok():
                msg.header.stamp = self.ros.node.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            msg.height = img.shape[0]
            msg.width = img.shape[1]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = img.shape[1] * 3
            msg.data = img.tobytes()
            if hasattr(self.ros, 'publish_target_image'):
                self.ros.publish_target_image(msg)
            if hasattr(self.ros, 'publish_pursuit_mode'):
                self.ros.publish_pursuit_mode('template')

            thought = (
                f"🎯 [Template Deployed] Reference template '{os.path.basename(path)}' "
                f"({img.shape[1]}x{img.shape[0]} px) active for '{target_name}'. "
                f"Perception engines primed!"
            )
            self.on_thinking_log_received(thought)
            self.on_feedback_received(
                f"[Target Operator] Reference template deployed: {os.path.basename(path)}"
            )
            self.btn_deploy_template.setText("✓ Deployed!")
            QTimer.singleShot(2000, lambda: self.btn_deploy_template.setText("🚀 Deploy Template"))

        if hasattr(self, 'radio_mode_template'):
            self.radio_mode_template.setChecked(True)

        # Spawn or update textured target in Gazebo simulation
        self._ensure_textured_gazebo_target(
            img_path=path, target_name=target_name, force_option_b=True
        )

    def on_relocate_target_from_inputs(self):
        """Relocate simulated target object using coordinates from input spin boxes."""
        tx = self.spin_target_pos_x.value()
        ty = self.spin_target_pos_y.value()
        self.relocate_target(tx, ty, 0.0)
        self.on_feedback_received(f"[Target Operator] Relocated target to ({tx:.2f}, {ty:.2f})")
        if hasattr(self, 'btn_relocate_target_input'):
            self.btn_relocate_target_input.setText("✓ Placed!")
            QTimer.singleShot(1500, lambda: self.btn_relocate_target_input.setText("📍 Relocate"))
        target_name = (
            self.edit_target_name.text().strip()
            if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
            else "Target Object"
        )
        self.on_thinking_log_received(
            f"📍 [Target Placed] '{target_name}' positioned at arena coords "
            f"(x={tx:+.2f}m, y={ty:+.2f}m) in simulation."
        )

    @staticmethod
    def is_pose_colliding_with_geometries(tx: float, ty: float) -> bool:
        """Check if coordinates intersect static obstacles or geometries in arena."""
        # 1. Outer arena bounds (|x| > 2.10 or |y| > 2.10)
        if abs(tx) > 2.10 or abs(ty) > 2.10:
            return True
        # 2. Inclined ramp: x in [0.35, 1.85], y in [0.85, 1.75]
        if 0.35 <= tx <= 1.85 and 0.85 <= ty <= 1.75:
            return True
        # 3. Raised platform & approach ramp: x in [-2.05, -0.25], y in [0.45, 1.75]
        if -2.05 <= tx <= -0.25 and 0.45 <= ty <= 1.75:
            return True
        # 4. Obstacle box 1 at (0.8, -1.1) (0.5 x 0.4)
        if 0.45 <= tx <= 1.15 and -1.40 <= ty <= -0.80:
            return True
        # 5. Obstacle box 2 at (-1.1, -1.2) (0.6 x 0.35)
        if -1.50 <= tx <= -0.70 and -1.45 <= ty <= -0.95:
            return True
        # 6. Obstacle barrier at (0.0, -1.7) (1.2 x 0.15)
        if -0.70 <= tx <= 0.70 and -1.85 <= ty <= -1.55:
            return True
        # 7. Obstacle cylinder 1 at (1.6, -0.3) radius 0.15
        if math.hypot(tx - 1.60, ty - (-0.30)) < 0.28:
            return True
        # 8. Obstacle cylinder 2 at (-0.2, 0.6) radius 0.12
        if math.hypot(tx - (-0.20), ty - 0.60) < 0.25:
            return True
        # 9. Obstacle cylinder 3 at (-1.5, -0.2) radius 0.10
        if math.hypot(tx - (-1.50), ty - (-0.20)) < 0.22:
            return True
        return False

    def get_valid_random_target_pose(self) -> Tuple[float, float]:
        """Generate random arena coordinates avoiding obstacles, walls, and robot footprint."""
        import random
        for _ in range(300):
            tx = round(random.uniform(-2.10, 2.10), 2)
            ty = round(random.uniform(-2.10, 2.10), 2)
            if self.is_pose_colliding_with_geometries(tx, ty):
                continue
            if math.hypot(tx - self.robot_x, ty - self.robot_y) < 0.70:
                continue
            return (tx, ty)
        return (1.50, 0.00)

    def on_randomize_target_clicked(self):
        """Randomize target object placement in front of the robot within arena bounds."""
        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
        dist = 1.6
        tx_front = float(np.clip(rx + dist * math.cos(ryaw), -2.35, 2.35))
        ty_front = float(np.clip(ry + dist * math.sin(ryaw), -2.35, 2.35))

        # Check if front placement is unobstructed by any arena geometry
        is_front_clear = (
            not self.is_pose_colliding_with_geometries(tx_front, ty_front)
            and math.hypot(tx_front - rx, ty_front - ry) >= 0.70
        )
        if is_front_clear:
            tx, ty = tx_front, ty_front
        else:
            tx, ty = self.get_valid_random_target_pose()

        if hasattr(self, 'spin_target_pos_x'):
            self.spin_target_pos_x.setValue(tx)
        if hasattr(self, 'spin_target_pos_y'):
            self.spin_target_pos_y.setValue(ty)
        self.relocate_target(tx, ty, ryaw + math.pi)
        if hasattr(self, 'lbl_draw_info'):
            self.lbl_draw_info.setText(f"Target randomized to ({tx:.2f}, {ty:.2f}).")
            self.lbl_draw_info.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 9.5px;")
        if hasattr(self, 'btn_randomize_target'):
            self.btn_randomize_target.setText("✓ Randomized!")
            QTimer.singleShot(1500, lambda: self.btn_randomize_target.setText("🎲 Randomize"))
        target_name = (
            self.edit_target_name.text().strip()
            if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
            else "Target Object"
        )
        self.on_thinking_log_received(
            f"🎲 [Target Randomized] '{target_name}' spawned at "
            f"(x={tx:+.2f}m, y={ty:+.2f}m) in simulation."
        )
        self.on_feedback_received(f"[Target Operator] Randomized target to ({tx:.2f}, {ty:.2f})")

    # --------------------------------------------------------------------------
    # Persistence Slots
    # --------------------------------------------------------------------------
    def on_save_map_clicked(self):
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Saving map to disk...")
        if self.ros:
            self.ros.trigger_service(self.ros.save_map_cli, "Save Map")

    def on_load_map_clicked(self):
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Loading map memory from disk...")
        if self.ros:
            self.ros.trigger_service(self.ros.load_map_cli, "Load Map")

    def on_clear_map_clicked(self):
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Resetting and clearing spatial maps...")
        self.robot_path.clear()
        self.visual_keyframes.clear()
        self.keyframe_artists.clear()
        self.latest_visual_map = None
        self._map_layer_dirty = True
        if hasattr(self, '_map_img_artist') and self._map_img_artist is not None:
            if self._map_img_artist in self.ax_map.images:
                try:
                    self._map_img_artist.remove()
                except Exception:
                    pass
            self._map_img_artist = None
        self._last_rendered_map_msg = None
        if self.ros:
            self.ros.trigger_service(self.ros.clear_map_cli, "Clear Map")
            if hasattr(self.ros, 'clear_visual_map_cli') and self.ros.clear_visual_map_cli:
                self.ros.trigger_service(self.ros.clear_visual_map_cli, 'Clear Visual Map')
        self.update_map_plot()

    def on_export_ply_clicked(self):
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Triggering 3DGS training & PLY export...")
        if self.ros and self.ros.reconstruct_3dgs_cli:
            self.ros.trigger_service(self.ros.reconstruct_3dgs_cli, "Export 3DGS PLY")
        else:
            self.lbl_map_toast.setText(f"[{timestamp}] 3DGS pipeline service not available.")

    def on_feedback_received(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] {msg}")

    # --------------------------------------------------------------------------
    # Sensor & Telemetry Slots
    # --------------------------------------------------------------------------
    def on_odom_received(self, x: float, y: float, yaw: float, v: float, w: float):
        self.robot_x = x
        self.robot_y = y
        self.robot_yaw = yaw
        self.odom_v = v
        self.odom_w = w
        self.has_received_odom = True

        dx = x - getattr(self, '_last_rendered_x', 0.0)
        dy = y - getattr(self, '_last_rendered_y', 0.0)
        dyaw = abs(yaw - getattr(self, '_last_rendered_yaw', 0.0))
        if math.hypot(dx, dy) > 0.015 or dyaw > 0.02:
            self._map_dirty = True

        if not self.robot_path or math.hypot(x - self.robot_path[-1][0], y - self.robot_path[-1][1]) > 0.03:
            self.robot_path.append((x, y))

    def on_imu_received(self, pitch: float, roll: float, az: float):
        self.imu_pitch = pitch
        self.imu_roll = roll
        self.imu_az = az
        self.lbl_pitch_val.setText(f"{pitch:+.1f}°")
        self.lbl_roll_val.setText(f"{roll:+.1f}°")

        slope = math.hypot(pitch, roll)
        is_flipped = (az < -1.0) or (az < 1.0 and slope > 45.0) or (slope > 70.0)
        if is_flipped:
            if not self.is_turned_over:
                self.on_turned_over_received(True)
        elif self.is_turned_over and (az >= 2.0) and (slope <= 50.0):
            self.on_turned_over_received(False)

        if abs(pitch) > 5.0 or abs(roll) > 5.0:
            self.lbl_terrain_slope.setText(f"Terrain: Inclined Ramp (Slope: {slope:.1f}°)")
            self.lbl_terrain_slope.setStyleSheet(
                "background-color: #451a03; border: 1px solid #f59e0b; border-radius: 4px; "
                "color: #fde68a; font-size: 10px; font-weight: bold; padding: 4px;"
            )
        elif abs(az - 9.81) > 1.2:
            self.lbl_terrain_slope.setText(f"Terrain: Rough Floor (Δaz: {abs(az - 9.81):.2f})")
            self.lbl_terrain_slope.setStyleSheet(
                "background-color: #311042; border: 1px solid #c084fc; border-radius: 4px; "
                "color: #f3e8ff; font-size: 10px; font-weight: bold; padding: 4px;"
            )
        else:
            self.lbl_terrain_slope.setText(f"Terrain: Flat Floor (Slope: {slope:.1f}°)")
            self.lbl_terrain_slope.setStyleSheet(
                "background-color: #131b26; border: 1px solid #1e293b; border-radius: 4px; "
                "color: #cbd5e1; font-size: 10px; font-weight: bold; padding: 4px;"
            )

    def on_recenter_clicked(self):
        """Handle Recenter button click: respawn robot at origin rightside up and reset path/state."""
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Teleporting robot to origin rightside up...")
        self.is_turned_over = False
        self.keys_pressed.clear()
        self.dpad_direction = None
        self.active_cmd_v = 0.0
        self.active_cmd_w = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_path.clear()
        self._map_dirty = True
        self.on_turned_over_received(False)
        if self.ros:
            self.ros.recenter_robot()

    def on_hot_reload_clicked(self):
        """Execute a full clean hot-reload: kill orphans, reset simulation & pipeline."""
        timestamp = time.strftime("%H:%M:%S")
        self.lbl_map_toast.setText(f"[{timestamp}] Executing hot-reload reset...")

        # 1. Stop all active motion immediately
        self.active_cmd_v = 0.0
        self.active_cmd_w = 0.0
        self.keys_pressed.clear()
        self.dpad_direction = None
        self.on_stop_clicked()

        # 2. Terminate orphan explorer and navigation processes
        if self.ros and hasattr(self.ros, 'kill_explorer_instances'):
            self.ros.kill_explorer_instances()
        try:
            my_pid = os.getpid()
            my_ppid = os.getppid()
            out = subprocess.run(
                ["pgrep", "-f", "reactive_explorer|openfrontier_explorer"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if out.stdout:
                for pid_str in out.stdout.split():
                    try:
                        p = int(pid_str)
                        if p != my_pid and p != my_ppid:
                            os.kill(p, signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass

        # 3. Reset ROS pipeline topics
        if self.ros:
            if hasattr(self.ros, 'publish_cmd'):
                self.ros.publish_cmd(0.0, 0.0)
            if hasattr(self.ros, 'set_pursuit_enable'):
                self.ros.set_pursuit_enable(False)
            if hasattr(self.ros, 'set_explorer_enable'):
                self.ros.set_explorer_enable(False)
            if hasattr(self.ros, 'publish_target_stopped'):
                self.ros.publish_target_stopped(False)
            if hasattr(self.ros, 'set_auto_explore'):
                self.ros.set_auto_explore(False)

        # 4. Teleport robot to arena origin rightside up & clear maps
        self.on_recenter_clicked()
        self.on_clear_map_clicked()

        # 5. Remove custom target model from Gazebo world (simulation only)
        if self.is_simulation:
            def _remove_gz_target():
                try:
                    world_name = "khepera4_world"
                    proc = subprocess.run(
                        ['gz', 'service', '-l'],
                        capture_output=True, text=True, timeout=2.0
                    )
                    for line in proc.stdout.splitlines():
                        parts = line.strip().split('/')
                        if (
                            len(parts) >= 4
                            and parts[1] == 'world'
                            and parts[3] in ('create', 'remove')
                        ):
                            world_name = parts[2]
                            break
                    subprocess.run([
                        'gz', 'service',
                        '-s', f'/world/{world_name}/remove',
                        '--reqtype', 'gz.msgs.Entity',
                        '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '2000',
                        '--req', 'name: "custom_target_object" type: MODEL'
                    ], capture_output=True, timeout=2.5)
                except Exception:
                    pass

            threading.Thread(target=_remove_gz_target, daemon=True).start()

        # 6. Reset UI buttons, badges, logs, and dials to initial boot defaults
        self.is_pursuit_active = False
        self.is_auto_exploring = False
        self.is_turned_over = False
        self._update_pursuit_button_style()
        if hasattr(self, 'btn_auto_explore'):
            self.btn_auto_explore.setObjectName("auto_explore_idle")
            self.btn_auto_explore.setText("▶ Start Auto Explore")
            self.btn_auto_explore.style().unpolish(self.btn_auto_explore)
            self.btn_auto_explore.style().polish(self.btn_auto_explore)
        if hasattr(self, 'lbl_explore_status'):
            self.lbl_explore_status.setText(
                "Mode: Manual Control (Click button to start exploration)"
            )
            self.lbl_explore_status.setStyleSheet("color: #94a3b8; font-size: 10px;")

        if hasattr(self, 'lbl_thinking_motion_badge'):
            self.lbl_thinking_motion_badge.setText("IDLE / STANDBY")
            self.lbl_thinking_motion_badge.setStyleSheet(
                "background-color: #1e293b; color: #94a3b8; font-weight: bold; font-size: 9.5px; "
                "padding: 2px 6px; border-radius: 3px;"
            )

        if hasattr(self, 'lbl_target_pose'):
            self.lbl_target_pose.setText("Target Pose: (X: +1.50m, Y: +0.00m)")
        self.target_pose = (1.50, 0.00, 0.0)

        self.max_linear_speed = 0.12
        self.max_angular_speed = 1.00
        if hasattr(self, 'dial_linear_speed'):
            self.dial_linear_speed.setValue(12)
        if hasattr(self, 'dial_angular_speed'):
            self.dial_angular_speed.setValue(100)
        if hasattr(self, '_sync_speed_limits_to_robot'):
            self._sync_speed_limits_to_robot()

        self.lbl_map_toast.setText(
            f"[{timestamp}] 🔥 Hot-reload complete: 0 orphans, robot at origin, fresh state."
        )
        self.on_thinking_log_received(
            "🔥 [Hot Reload] Clean reset: 0 orphans, robot at origin, fresh boot state."
        )
        self.on_feedback_received("[System] Hot-reload reset to fresh boot state completed.")
        self.update_map_plot()

    def on_turned_over_received(self, flipped: bool):
        """Update UI and state when turned over / flipped state changes."""
        self.is_turned_over = flipped
        if flipped:
            self.lbl_rollover_badge.setText("🚨 FLIPPED / TURNED OVER")
            self.lbl_rollover_badge.setStyleSheet(
                "background-color: #7f1d1d; border: 2px solid #ef4444; border-radius: 5px; "
                "color: #fee2e2; font-size: 10px; font-weight: 900; padding: 4px;"
            )
            if self.ros:
                self.ros.publish_cmd(0.0, 0.0)
            self.lbl_teleop_cmd.setText("Cmd: v = +0.00 m/s | ω = +0.00 rad/s [LOCKED - ROBOT TURNED OVER]")
        else:
            self.lbl_rollover_badge.setText("ORIENTATION: UPRIGHT & STABLE")
            self.lbl_rollover_badge.setStyleSheet(
                "background-color: #064e3b; border: 1px solid #10b981; border-radius: 5px; "
                "color: #6ee7b7; font-size: 10px; font-weight: bold; padding: 4px;"
            )
            if "LOCKED" in self.lbl_teleop_cmd.text():
                self.lbl_teleop_cmd.setText("Cmd: v = +0.00 m/s | ω = +0.00 rad/s")

    def on_cliff_alert_received(self, alert: bool):
        self.is_cliff_alert = alert
        if alert:
            self.lbl_cliff_badge.setText("! CLIFF HAZARD DETECTED !")
            self.lbl_cliff_badge.setStyleSheet(
                "background-color: #991b1b; border: 2px solid #ef4444; border-radius: 5px; "
                "color: #ffffff; font-size: 11px; font-weight: 900; padding: 4px;"
            )
        else:
            self.lbl_cliff_badge.setText("CLIFF SAFE")
            self.lbl_cliff_badge.setStyleSheet(
                "background-color: #064e3b; border: 1px solid #10b981; border-radius: 5px; "
                "color: #6ee7b7; font-size: 11px; font-weight: 900; padding: 4px;"
            )

    def on_map_received(self, msg: OccupancyGrid):
        self.latest_map = msg
        self._map_dirty = True
        self._map_layer_dirty = True

    def on_elevation_received(self, msg: OccupancyGrid):
        self.latest_elevation = msg
        self._map_dirty = True
        self._map_layer_dirty = True

    def on_surface_received(self, msg: OccupancyGrid):
        self.latest_surface = msg
        self._map_dirty = True
        self._map_layer_dirty = True

    def on_visual_map_received(self, msg: OccupancyGrid):
        self.latest_visual_map = msg
        self._map_dirty = True
        self._map_layer_dirty = True

    def on_scan_received(self, msg: LaserScan):
        self.latest_scan = msg
        self._map_dirty = True

    def _get_target_visual_bbox(
        self, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int, str, float]]:
        """Return (x1, y1, x2, y2, label, score) if target is detected or seen in FOV."""
        now = time.time()
        # 1. Perception pipeline detection ROI (YOLO-World, DINOv2, LightGlue)
        if self.latest_target_bbox is not None and (now - self.last_target_bbox_time) < 0.8:
            return self.latest_target_bbox

        # 2. In simulation or pursuit mode: project target if within camera FOV
        if (
            self.is_pursuit_active
            and self.target_pose is not None
            and img_w > 10
            and img_h > 10
        ):
            tx, ty = self.target_pose[0], self.target_pose[1]
            rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
            dx = tx - rx
            dy = ty - ry
            dist = math.hypot(dx, dy)
            target_bearing = normalize_angle(math.atan2(dy, dx) - ryaw)
            hfov = 1.047  # ~60 degrees horizontal FOV
            if 0.15 < dist < 3.8 and abs(target_bearing) < (hfov / 2.0):
                # Camera projection onto image plane
                cx_px = int((img_w / 2.0) - (target_bearing / (hfov / 2.0)) * (img_w / 2.0))
                focal_px = (img_w / 2.0) / math.tan(hfov / 2.0)
                w_box = int(np.clip(0.08 * focal_px / dist, 24, img_w * 0.80))
                h_box = int(np.clip(0.12 * focal_px / dist, 32, img_h * 0.85))
                cy_px = int(img_h * 0.56)
                x1 = int(np.clip(cx_px - w_box // 2, 0, img_w - 1))
                y1 = int(np.clip(cy_px - h_box // 2, 0, img_h - 1))
                x2 = int(np.clip(cx_px + w_box // 2, 0, img_w - 1))
                y2 = int(np.clip(cy_px + h_box // 2, 0, img_h - 1))
                name = (
                    self.edit_target_name.text().strip()
                    if hasattr(self, 'edit_target_name') and self.edit_target_name.text().strip()
                    else "Target Object"
                )
                conf = max(0.85, 1.0 - (dist / 4.0) * 0.2)
                return (x1, y1, x2, y2, name, conf)
        return None

    def _overlay_target_bbox(self, frame: np.ndarray) -> np.ndarray:
        """Render prominent target pursuit bounding square and HUD tag over live camera feed."""
        h, w = frame.shape[:2]
        bbox_info = self._get_target_visual_bbox(w, h)
        if bbox_info is None:
            return frame

        x1, y1, x2, y2, label, score = bbox_info
        if (x2 - x1) < 6 or (y2 - y1) < 6:
            return frame

        vis = frame.copy()
        # 1. Emerald green bounding square
        cv2.rectangle(vis, (x1, y1), (x2, y2), (16, 185, 129), 2)

        # 2. High-visibility tactical corner accents
        c_len = max(6, min(18, (x2 - x1) // 4))
        # Top-left
        cv2.line(vis, (x1, y1), (x1 + c_len, y1), (52, 211, 153), 3)
        cv2.line(vis, (x1, y1), (x1, y1 + c_len), (52, 211, 153), 3)
        # Top-right
        cv2.line(vis, (x2, y1), (x2 - c_len, y1), (52, 211, 153), 3)
        cv2.line(vis, (x2, y1), (x2, y1 + c_len), (52, 211, 153), 3)
        # Bottom-left
        cv2.line(vis, (x1, y2), (x1 + c_len, y2), (52, 211, 153), 3)
        cv2.line(vis, (x1, y2), (x1, y2 - c_len), (52, 211, 153), 3)
        # Bottom-right
        cv2.line(vis, (x2, y2), (x2 - c_len, y2), (52, 211, 153), 3)
        cv2.line(vis, (x2, y2), (x2, y2 - c_len), (52, 211, 153), 3)

        # 3. Center targeting crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(
            vis, (cx, cy), (52, 211, 153),
            markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1
        )

        # 4. Target identification tag with solid contrast background
        tag_text = f"TARGET: {label} [{int(score * 100)}%]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, thickness)
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = max(th + 8, y1)
        cv2.rectangle(vis, (x1, tag_y1), (min(w - 1, x1 + tw + 10), tag_y2), (6, 78, 59), -1)
        cv2.rectangle(vis, (x1, tag_y1), (min(w - 1, x1 + tw + 10), tag_y2), (16, 185, 129), 1)
        cv2.putText(
            vis, tag_text, (x1 + 4, tag_y2 - 4),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        return vis

    def on_camera_received(self, msg: RosImage):
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                h, w = msg.height, msg.width
                img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                if msg.encoding == 'bgr8':
                    rgb_arr = np.ascontiguousarray(img_arr[:, :, ::-1])
                else:
                    rgb_arr = np.ascontiguousarray(img_arr)
                self.latest_camera_frame = rgb_arr.copy()

                now = time.time()
                self.camera_frame_count += 1
                dt = now - self.last_cam_fps_time
                if dt >= 1.0:
                    self.cam_fps = self.camera_frame_count / dt
                    self.camera_frame_count = 0
                    self.last_cam_fps_time = now
                    self.lbl_camera_status.setText(f"ONLINE | {w}x{h} @ {self.cam_fps:.1f} FPS")
                    self.lbl_camera_status.setStyleSheet(
                        "color: #10b981; font-weight: bold; font-size: 9px;"
                    )

                # Throttle camera preview rendering to at most 10 FPS (100 ms)
                if (now - getattr(self, 'last_camera_render_time', 0.0)) >= 0.100:
                    self.last_camera_render_time = now

                    # Prefer recent annotated visual from /target/detection_visual if active (< 0.5s)
                    if (
                        hasattr(self, 'latest_detection_frame')
                        and self.latest_detection_frame is not None
                        and (now - getattr(self, 'last_detection_frame_time', 0.0)) < 0.5
                    ):
                        display_frame = self.latest_detection_frame
                    else:
                        display_frame = self._overlay_target_bbox(rgb_arr)

                    qimg = QImage(display_frame.data, w, h, w * 3, QImage.Format_RGB888).copy()
                    lbl_w = max(160, self.lbl_camera_view.width())
                    lbl_h = max(120, self.lbl_camera_view.height())
                    pixmap = QPixmap.fromImage(qimg).scaled(
                        lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.FastTransformation
                    )
                    self.lbl_camera_view.setPixmap(pixmap)

                    if hasattr(self, 'camera_popup') and self.camera_popup:
                        self.camera_popup.update_frame(display_frame, w, h, self.cam_fps)
        except Exception:
            pass

    def on_toggle_camera_popup(self):
        """Open or focus the floating, resizable, always-on-top camera window."""
        if not hasattr(self, 'camera_popup') or self.camera_popup is None:
            self.camera_popup = CameraPopupWindow(self)

        # Pre-seed popup with latest raw frame and detection HUD frame if available
        if self.latest_camera_frame is not None:
            h, w = self.latest_camera_frame.shape[:2]
            self.camera_popup.update_frame(self.latest_camera_frame, w, h, self.cam_fps)

        if hasattr(self, 'latest_detection_frame') and self.latest_detection_frame is not None:
            dh, dw = self.latest_detection_frame.shape[:2]
            self.camera_popup.update_detection_frame(self.latest_detection_frame, dw, dh, self.detection_fps)

        self.camera_popup.update_pursuit_state(self.is_pursuit_active)

        if self.camera_popup.isVisible():
            self.camera_popup.raise_()
            self.camera_popup.activateWindow()
        else:
            self.camera_popup.show()
            self.camera_popup.raise_()
            self.camera_popup.activateWindow()

        if hasattr(self, 'btn_popup_cam'):
            self.btn_popup_cam.setText("⧉ Pop-out (Active)")
            self.btn_popup_cam.setStyleSheet(
                "background-color: #0284c7; border: 1px solid #38bdf8; border-radius: 4px; "
                "color: #ffffff; font-weight: bold; font-size: 10px; padding: 3px 6px;"
            )

    def on_pin_view_to_map(self):
        """Pin current camera view to the generated map at current robot pose."""
        if not self.has_received_odom or self.latest_camera_frame is None:
            self.lbl_camera_status.setText("Awaiting frame / odom")
            return
        kf_idx = len(self.visual_keyframes) + 1
        t_str = time.strftime("%H:%M:%S")
        self.visual_keyframes.append((self.robot_x, self.robot_y, self.robot_yaw, self.latest_camera_frame.copy(), f"KF #{kf_idx} ({t_str})"))
        self.lbl_camera_status.setText(f"Pinned Keyframe #{kf_idx} ({t_str})")
        self.lbl_camera_status.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 9px;")
        self.update_map_plot()

    # --------------------------------------------------------------------------
    # Map Toolbar Slots
    # --------------------------------------------------------------------------
    def on_layer_changed(self, idx: int):
        modes = ['obstacles', 'elevation', 'surface', 'visual']
        self.map_layer_mode = modes[idx] if idx < len(modes) else 'obstacles'
        self._map_layer_dirty = True
        self.update_map_plot()

    def on_sensor_rays_toggled(self, checked: bool):
        self.show_sensor_beams = checked
        self.update_map_plot()

    def on_camera_fov_toggled(self, checked: bool):
        self.show_camera_fov = checked
        self.update_map_plot()

    def on_trajectory_toggled(self, checked: bool):
        self.show_trajectory = checked
        self.update_map_plot()

    def on_grid_toggled(self, checked: bool):
        self.show_grid = checked
        self.update_map_plot()

    def on_zoom_in(self):
        self.zoom_level = max(0.2, self.zoom_level * 0.8)
        self.update_map_plot()

    def on_zoom_out(self):
        self.zoom_level = min(5.0, self.zoom_level * 1.25)
        self.update_map_plot()

    def on_fit_arena(self):
        self.zoom_level = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.update_map_plot()

    # --------------------------------------------------------------------------
    # Matplotlib High-Quality Map & Spatial Rendering
    # --------------------------------------------------------------------------
    def update_map_plot(self):
        """Redraw map canvas using persistent artists and cached background layers."""
        need_axes_init = (
            not hasattr(self, '_axes_initialized')
            or not self._axes_initialized
            or len(self.ax_map.spines) == 0
        )
        if need_axes_init:
            self.ax_map.set_facecolor('#0f172a')
            self.ax_map.tick_params(colors='#64748b', labelsize=8)
            for spine in self.ax_map.spines.values():
                spine.set_color('#1e293b')
            self.ax_map.set_xlabel("X' [meters]", color='#94a3b8', fontsize=8, labelpad=2)
            self.ax_map.set_ylabel("Y' [meters]", color='#94a3b8', fontsize=8, labelpad=2)
            self._axes_initialized = True

        if self.show_grid:
            self.ax_map.grid(True, color='#1e293b', linestyle='--', linewidth=0.6, alpha=0.8)
        else:
            self.ax_map.grid(False)

        layer_title_map = {
            'obstacles': '2D Probabilistic SLAM Obstacle Grid (/map)',
            'elevation': '2.5D Terrain Slope Elevation Map (/elevation_map)',
            'surface': 'Surface Classification & Cliff Hazard Map (/surface_map)',
            'visual': 'Visual Orthomap (/visual_map)'
        }
        target_title = f"Khepera IV Spatial SLAM — {layer_title_map.get(self.map_layer_mode, '')}"
        if self.ax_map.get_title() != target_title:
            self.ax_map.set_title(
                target_title, color='#38bdf8', fontsize=10, fontweight='bold', pad=8
            )

        # 1. Active Map Layer (persistent AxesImage)
        if self.map_layer_mode == 'obstacles':
            current_map_msg = self.latest_map
        elif self.map_layer_mode == 'elevation':
            current_map_msg = self.latest_elevation
        elif self.map_layer_mode == 'surface':
            current_map_msg = self.latest_surface
        elif self.map_layer_mode == 'visual':
            current_map_msg = self.latest_visual_map
        else:
            current_map_msg = None

        need_rebuild_layer = (
            getattr(self, '_map_layer_dirty', True)
            or getattr(self, '_last_rendered_layer_mode', None) != self.map_layer_mode
            or getattr(self, '_last_rendered_map_msg', None) is not current_map_msg
            or getattr(self, '_map_img_artist', None) is None
            or self._map_img_artist not in self.ax_map.images
        )

        has_rendered_grid = False
        if current_map_msg is None:
            if getattr(self, '_map_img_artist', None) is not None:
                if self._map_img_artist in self.ax_map.images:
                    try:
                        self._map_img_artist.remove()
                    except Exception:
                        pass
                self._map_img_artist = None
            self._last_rendered_map_msg = None
            self._map_layer_dirty = False
            has_rendered_grid = False
        elif need_rebuild_layer:
            try:
                info = current_map_msg.info
                w, h = info.width, info.height
                res = info.resolution
                ox = info.origin.position.x
                oy = info.origin.position.y
                data = np.array(current_map_msg.data, dtype=np.int8).reshape((h, w))
                rgba = np.zeros((h, w, 4), dtype=np.float32)

                if self.map_layer_mode == 'obstacles':
                    # Free cells (0): clean dark arena floor
                    free_mask = (data == 0)
                    rgba[free_mask] = [0.12, 0.18, 0.28, 0.85]
                    # Occupied cells (100): vibrant electric cyan
                    occ_mask = (data == 100)
                    rgba[occ_mask] = [0.00, 0.96, 0.83, 0.98]

                elif self.map_layer_mode == 'elevation':
                    visited = (data >= 0)
                    slopes = np.clip(data[visited], 0, 30) / 30.0
                    # Green (flat) -> Yellow (moderate) -> Red (steep)
                    rgba[visited, 0] = np.clip(slopes * 2.0, 0.0, 1.0)
                    rgba[visited, 1] = np.clip(2.0 * (1.0 - slopes), 0.0, 1.0)
                    rgba[visited, 2] = 0.2
                    rgba[visited, 3] = 0.85

                elif self.map_layer_mode == 'surface':
                    # Visited roughness (0 - 85): Soft purple-blue
                    rough = (data >= 0) & (data < 100)
                    rgba[rough] = [0.35, 0.40, 0.65, 0.80]
                    # Cliff drops (100): High-intensity hazard red
                    cliffs = (data == 100)
                    rgba[cliffs] = [1.00, 0.08, 0.25, 0.98]

                elif self.map_layer_mode == 'visual':
                    # Observed floor cells (reflectance 0..100)
                    observed = (data >= 0)
                    gray_val = (data[observed] / 100.0).astype(np.float32)
                    rgba[observed, 0] = gray_val
                    rgba[observed, 1] = gray_val
                    rgba[observed, 2] = gray_val
                    rgba[observed, 3] = 0.90

                # 90-degree CCW map rotation:
                # World (X, Y) -> Canvas (X', Y') = (-Y, X)
                # Extent [xmin, xmax, ymin, ymax] -> [-Ymax, -Ymin, Xmin, Xmax]
                # In Cartesian space with origin='lower', 90 deg CCW rotation from +X to +Y
                # is np.rot90(rgba, 1, axes=(1, 0)).
                rgba_rot = np.rot90(rgba, 1, axes=(1, 0))
                extent = [-(oy + h * res), -oy, ox, ox + w * res]
                if (
                    getattr(self, '_map_img_artist', None) is None
                    or self._map_img_artist not in self.ax_map.images
                ):
                    self._map_img_artist = self.ax_map.imshow(
                        rgba_rot, origin='lower', extent=extent,
                        interpolation='nearest', zorder=1
                    )
                else:
                    self._map_img_artist.set_data(rgba_rot)
                    self._map_img_artist.set_extent(extent)
                    self._map_img_artist.set_visible(True)

                has_rendered_grid = True
                self._last_rendered_map_msg = current_map_msg
                self._last_rendered_layer_mode = self.map_layer_mode
                self._map_layer_dirty = False
                self._last_rendered_rgba = rgba_rot
                self._last_grid_meta = (-(oy + h * res), ox, h, w, res)
            except Exception:
                has_rendered_grid = False
                if getattr(self, '_map_img_artist', None) is not None and self._map_img_artist in self.ax_map.images:
                    try:
                        self._map_img_artist.remove()
                    except Exception:
                        pass
                self._map_img_artist = None
                self._last_rendered_map_msg = None
                self._map_layer_dirty = False
        else:
            has_rendered_grid = (
                getattr(self, '_map_img_artist', None) is not None
                and self._map_img_artist in self.ax_map.images
                and self._map_img_artist.get_visible()
            )

        # Draw Arena Outline if no map loaded yet
        if not hasattr(self, '_arena_rect') or self._arena_rect not in self.ax_map.patches:
            self._arena_rect = plt.Rectangle(
                (-3.0, -3.0), 6.0, 6.0, fill=True,
                facecolor='#111827', edgecolor='#334155',
                linestyle='--', linewidth=1.2, alpha=0.9, zorder=1
            )
            self.ax_map.add_patch(self._arena_rect)
        if not hasattr(self, '_awaiting_text') or self._awaiting_text not in self.ax_map.texts:
            self._awaiting_text = self.ax_map.text(
                0, 0,
                "Awaiting Map Stream from /map\n(Run khepera4_mapper or launch mapping.launch.py)",
                ha='center', va='center', color='#64748b', fontsize=11,
                fontweight='bold', zorder=2
            )

        if has_rendered_grid:
            self._arena_rect.set_visible(False)
            self._awaiting_text.set_visible(False)
        else:
            self._arena_rect.set_visible(True)
            self._awaiting_text.set_visible(True)

        # 4. Draw Explored Trajectory History
        if not hasattr(self, '_traj_line') or self._traj_line not in self.ax_map.lines:
            self._traj_line = self.ax_map.plot(
                [], [], color='#38bdf8', linewidth=1.6, alpha=0.75, linestyle='-', zorder=3
            )[0]
        if self.show_trajectory and len(self.robot_path) > 1:
            px, py = zip(*self.robot_path)
            c_px = [-y for y in py]
            c_py = list(px)
            self._traj_line.set_data(c_px, c_py)
            self._traj_line.set_visible(True)
        else:
            self._traj_line.set_visible(False)

        # 4b. Draw User-Drawn Exploration Path
        if not hasattr(self, '_drawn_line') or self._drawn_line not in self.ax_map.lines:
            self._drawn_line = self.ax_map.plot(
                [], [], color='#fbbf24', linewidth=2.0, linestyle='--', alpha=0.90, zorder=4
            )[0]
            self._drawn_pts = self.ax_map.plot(
                [], [], 'o', color='#f59e0b', markersize=4.0, alpha=0.90, zorder=4
            )[0]
        if len(self.drawn_path) > 1:
            ux, uy = zip(*self.drawn_path)
            c_ux = [-y for y in uy]
            c_uy = list(ux)
            self._drawn_line.set_data(c_ux, c_uy)
            self._drawn_pts.set_data(c_ux, c_uy)
            self._drawn_line.set_visible(True)
            self._drawn_pts.set_visible(True)
        else:
            self._drawn_line.set_visible(False)
            self._drawn_pts.set_visible(False)

        # 4c. Draw Active Exploration Frontier Target (Bullseye)
        if not hasattr(self, '_goal_circle') or self._goal_circle not in self.ax_map.lines:
            self._goal_circle = self.ax_map.plot(
                [], [], 'o', color='#f43f5e', markersize=11, fillstyle='none',
                markeredgewidth=2.2, zorder=7
            )[0]
            self._goal_cross = self.ax_map.plot(
                [], [], '+', color='#f43f5e', markersize=14, markeredgewidth=1.8, zorder=7
            )[0]
            self._goal_dot = self.ax_map.plot(
                [], [], '.', color='#ffffff', markersize=3.5, zorder=7
            )[0]
            self._goal_text = self.ax_map.text(
                0, 0, "", color='#fecdd3', fontsize=8, fontweight='bold',
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#4c0519",
                    "edgecolor": "#f43f5e",
                    "alpha": 0.85
                },
                zorder=8
            )
        if self.active_goal is not None and not getattr(self, 'is_pursuit_active', False):
            gx, gy = self.active_goal
            c_gx = -gy
            c_gy = gx
            self._goal_circle.set_data([c_gx], [c_gy])
            self._goal_cross.set_data([c_gx], [c_gy])
            self._goal_dot.set_data([c_gx], [c_gy])
            self._goal_text.set_position((c_gx + 0.08, c_gy + 0.08))
            self._goal_text.set_text(f"Frontier ({gx:.2f}, {gy:.2f})")
            self._goal_circle.set_visible(True)
            self._goal_cross.set_visible(True)
            self._goal_dot.set_visible(True)
            self._goal_text.set_visible(True)
        else:
            self._goal_circle.set_visible(False)
            self._goal_cross.set_visible(False)
            self._goal_dot.set_visible(False)
            self._goal_text.set_visible(False)

        # 4d. Draw Custom Target Object on Map Canvas (Distinct green marker/box)
        if not hasattr(self, '_target_rect') or self._target_rect not in self.ax_map.patches:
            self._target_rect = plt.Rectangle(
                (0, 0), 0.12, 0.12,
                facecolor='#10b981', edgecolor='#34d399',
                linewidth=2.0, alpha=0.85, zorder=7
            )
            self.ax_map.add_patch(self._target_rect)
            self._target_marker = self.ax_map.plot(
                [], [], 'x', color='#ffffff', markersize=6, markeredgewidth=1.8, zorder=8
            )[0]
            self._target_text = self.ax_map.text(
                0, 0, "", color='#a7f3d0', fontsize=8, fontweight='bold',
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#064e3b",
                    "edgecolor": "#10b981",
                    "alpha": 0.85
                },
                zorder=8
            )
        if hasattr(self, 'target_pose') and self.target_pose is not None:
            tx, ty, tyaw = self.target_pose
            if math.isfinite(tx) and math.isfinite(ty):
                box_size = 0.12
                c_tx = -ty
                c_ty = tx
                self._target_rect.set_xy((c_tx - box_size / 2.0, c_ty - box_size / 2.0))
                self._target_rect.set_width(box_size)
                self._target_rect.set_height(box_size)
                self._target_rect.set_visible(True)
                self._target_marker.set_data([c_tx], [c_ty])
                self._target_marker.set_visible(True)
                self._target_text.set_position((c_tx + 0.08, c_ty + 0.08))
                self._target_text.set_text(f"Target Object ({tx:.2f}, {ty:.2f})")
                self._target_text.set_visible(True)
            else:
                self._target_rect.set_visible(False)
                self._target_marker.set_visible(False)
                self._target_text.set_visible(False)
        else:
            self._target_rect.set_visible(False)
            self._target_marker.set_visible(False)
            self._target_text.set_visible(False)

        # 5. Draw Multi-Sensor Scan Rays (IR in orange, US in cyan)
        if not hasattr(self, '_scan_ir_rays') or self._scan_ir_rays not in self.ax_map.lines:
            self._scan_ir_rays = self.ax_map.plot(
                [], [], color='#f59e0b', linewidth=0.8, alpha=0.45, zorder=4
            )[0]
            self._scan_ir_pts = self.ax_map.plot(
                [], [], '.', color='#f59e0b', markersize=3.5, alpha=0.85, zorder=4
            )[0]
            self._scan_us_rays = self.ax_map.plot(
                [], [], color='#06b6d4', linewidth=0.8, alpha=0.45, zorder=4
            )[0]
            self._scan_us_pts = self.ax_map.plot(
                [], [], '.', color='#06b6d4', markersize=3.5, alpha=0.85, zorder=4
            )[0]

        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
        c_rx = -ry
        c_ry = rx
        has_scan = False
        if self.show_sensor_beams and self.latest_scan is not None:
            try:
                ranges = self.latest_scan.ranges
                num_rays = len(ranges)
                angle_min = self.latest_scan.angle_min
                angle_inc = self.latest_scan.angle_increment
                step = max(1, num_rays // 40)

                ir_x, ir_y, ir_pt_x, ir_pt_y = [], [], [], []
                us_x, us_y, us_pt_x, us_pt_y = [], [], [], []

                for i in range(0, num_rays, step):
                    r = ranges[i]
                    if math.isnan(r) or math.isinf(r) or r < 0.005 or r >= 1.95:
                        continue
                    beam_angle = ryaw + angle_min + i * angle_inc
                    hit_x = rx + r * math.cos(beam_angle)
                    hit_y = ry + r * math.sin(beam_angle)
                    c_hit_x = -hit_y
                    c_hit_y = hit_x

                    if r < 0.25:
                        ir_x.extend([c_rx, c_hit_x, np.nan])
                        ir_y.extend([c_ry, c_hit_y, np.nan])
                        if r < 1.95:
                            ir_pt_x.append(c_hit_x)
                            ir_pt_y.append(c_hit_y)
                    else:
                        us_x.extend([c_rx, c_hit_x, np.nan])
                        us_y.extend([c_ry, c_hit_y, np.nan])
                        if r < 1.95:
                            us_pt_x.append(c_hit_x)
                            us_pt_y.append(c_hit_y)

                self._scan_ir_rays.set_data(ir_x, ir_y)
                self._scan_ir_pts.set_data(ir_pt_x, ir_pt_y)
                self._scan_us_rays.set_data(us_x, us_y)
                self._scan_us_pts.set_data(us_pt_x, us_pt_y)
                self._scan_ir_rays.set_visible(True)
                self._scan_ir_pts.set_visible(True)
                self._scan_us_rays.set_visible(True)
                self._scan_us_pts.set_visible(True)
                has_scan = True
            except Exception:
                pass

        if not has_scan:
            self._scan_ir_rays.set_visible(False)
            self._scan_ir_pts.set_visible(False)
            self._scan_us_rays.set_visible(False)
            self._scan_us_pts.set_visible(False)

        # 5b. Draw Camera Sightline / FOV Frustum
        if not hasattr(self, '_fov_wedge') or self._fov_wedge not in self.ax_map.patches:
            self._fov_wedge = Polygon(
                [(0, 0), (0, 0), (0, 0)], closed=True,
                facecolor='#38bdf8', edgecolor='#0284c7',
                linewidth=1.2, alpha=0.18, linestyle='--', zorder=4
            )
            self.ax_map.add_patch(self._fov_wedge)

        self.fov_patch = None
        if self.show_camera_fov and self.has_received_odom:
            try:
                cam_x = rx + 0.0682 * math.cos(ryaw)
                cam_y = ry + 0.0682 * math.sin(ryaw)
                fov_dist = 0.75
                half_fov = 0.5236  # 30 degrees (pi/6)
                p_left = (cam_x + fov_dist * math.cos(ryaw + half_fov),
                          cam_y + fov_dist * math.sin(ryaw + half_fov))
                p_right = (cam_x + fov_dist * math.cos(ryaw - half_fov),
                           cam_y + fov_dist * math.sin(ryaw - half_fov))

                c_cam = (-cam_y, cam_x)
                c_left = (-p_left[1], p_left[0])
                c_right = (-p_right[1], p_right[0])

                self._fov_wedge.set_xy([c_cam, c_left, c_right])
                self._fov_wedge.set_visible(True)
                self.fov_patch = self._fov_wedge
            except Exception:
                self._fov_wedge.set_visible(False)
        else:
            self._fov_wedge.set_visible(False)

        # 5c. Draw Visual Keyframe Photo Pins on Map
        need_sync_keyframes = (
            len(self.visual_keyframes) != len(self.keyframe_artists)
            or any(
                line_art not in self.ax_map.lines or text_art not in self.ax_map.texts
                for line_art, text_art in self.keyframe_artists
            )
        )
        if need_sync_keyframes:
            for line_art, text_art in self.keyframe_artists:
                try:
                    line_art.remove()
                    text_art.remove()
                except Exception:
                    pass
            self.keyframe_artists.clear()
            for idx, (kx, ky, kyaw, _, label) in enumerate(self.visual_keyframes):
                c_kx = -ky
                c_ky = kx
                line_artist = self.ax_map.plot(
                    c_kx, c_ky, 's', color='#f59e0b', markersize=6.5,
                    markeredgecolor='#ffffff', markeredgewidth=1.2, zorder=6
                )[0]
                text_artist = self.ax_map.text(
                    c_kx + 0.04, c_ky + 0.04, f"[CAM #{idx + 1}]",
                    color='#fef08a', fontsize=7.5, fontweight='bold', zorder=7
                )
                self.keyframe_artists.append((line_artist, text_artist))

        # 6. Draw Khepera IV True-Scale Footprint (Radius: 0.070 m = 140 mm diameter)
        c_yaw = ryaw + math.pi / 2.0
        if not hasattr(self, '_khep_circle') or self._khep_circle not in self.ax_map.patches:
            self._khep_circle = Circle(
                (c_rx, c_ry), 0.070, facecolor='#0284c7', edgecolor='#7dd3fc',
                linewidth=2.0, alpha=0.85, zorder=5
            )
            self.ax_map.add_patch(self._khep_circle)
            self._khep_arrow = self.ax_map.annotate(
                '', xy=(c_rx, c_ry), xytext=(c_rx, c_ry),
                arrowprops={
                    "arrowstyle": "-|>", "color": "#ffffff", "lw": 2.2, "mutation_scale": 14
                },
                zorder=6
            )
            self._khep_dot = self.ax_map.plot(
                [c_rx], [c_ry], 'o', color='#ffffff', markersize=3, zorder=7
            )[0]

        self._khep_circle.center = (c_rx, c_ry)
        arrow_len = 0.16
        hx = c_rx + arrow_len * math.cos(c_yaw)
        hy = c_ry + arrow_len * math.sin(c_yaw)
        self._khep_arrow.xy = (hx, hy)
        self._khep_arrow.set_position((c_rx, c_ry))
        self._khep_dot.set_data([c_rx], [c_ry])

        # 7. Apply View Bounds & Zoom
        base_half_span = 3.5 * self.zoom_level
        self.ax_map.set_xlim(
            self.view_offset_x - base_half_span, self.view_offset_x + base_half_span
        )
        self.ax_map.set_ylim(
            self.view_offset_y - base_half_span, self.view_offset_y + base_half_span
        )
        self.ax_map.set_aspect('equal', adjustable='box')

        # 8. Canvas Overlay HUD Text (Pose & System Summary)
        if not hasattr(self, '_hud_text') or self._hud_text not in self.ax_map.texts:
            self._hud_text = self.ax_map.text(
                0.02, 0.02, "", transform=self.ax_map.transAxes,
                color='#e2e8f0', fontsize=8.5, fontfamily='monospace', fontweight='bold',
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#0b0f19",
                    "edgecolor": "#334155",
                    "alpha": 0.85
                },
                zorder=8
            )

        odom_source = "FUSED (/khepera4/odom)" if (self.ros and self.ros.using_fused_odom) else "RAW (/odom)"
        hud_text = (
            f"Pose: X: {rx:+.2f} m | Y: {ry:+.2f} m | Yaw: {math.degrees(ryaw):+.1f}°\n"
            f"Speed: {self.odom_v:+.2f} m/s | Odom: {odom_source} | Path: {len(self.robot_path)} pts"
        )
        self._hud_text.set_text(hud_text)

        if self.ros and hasattr(self, 'lbl_connection'):
            if self.ros.using_fused_odom:
                self.lbl_connection.setText("● ROS 2 JAZZY ONLINE (PRIORITY: /khepera4/odom FUSED)")
                self.lbl_connection.setStyleSheet(
                    "background-color: #064e3b; border: 1px solid #10b981; border-radius: 4px; "
                    "color: #6ee7b7; font-size: 10px; font-weight: bold; padding: 2px 6px; margin-top: 2px;"
                )
            else:
                self.lbl_connection.setText("● ROS 2 JAZZY ONLINE (FALLBACK: /odom RAW)")
                self.lbl_connection.setStyleSheet(
                    "background-color: #451a03; border: 1px solid #f59e0b; border-radius: 4px; "
                    "color: #fde68a; font-size: 10px; font-weight: bold; padding: 2px 6px; margin-top: 2px;"
                )

        self._map_dirty = False
        self._last_rendered_x = rx
        self._last_rendered_y = ry
        self._last_rendered_yaw = ryaw

        if (
            getattr(self, 'use_gpu_canvas', False)
            and getattr(self, 'pg_image', None) is not None
            and has_rendered_grid
            and hasattr(self, '_last_rendered_rgba')
            and hasattr(self, '_last_grid_meta')
        ):
            try:
                ox, oy, w, h, res = self._last_grid_meta
                self.pg_image.setImage(self._last_rendered_rgba.transpose((1, 0, 2)))
                self.pg_image.setRect(QRectF(ox, oy, w * res, h * res))
            except Exception:
                pass

        self.canvas.draw_idle()

    def enable_gpu_opengl_canvas(self, enable: bool = True):
        """Toggle between hardware-accelerated PyQtGraph OpenGL canvas and Matplotlib."""
        if not PYQTGRAPH_AVAILABLE or getattr(self, 'pg_plot', None) is None:
            return
        self.use_gpu_canvas = enable
        if enable:
            self.canvas.hide()
            self.pg_plot.show()
        else:
            self.pg_plot.hide()
            self.canvas.show()
        self.update_map_plot()

    def closeEvent(self, event):
        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        if hasattr(self, 'plot_timer') and self.plot_timer.isActive():
            self.plot_timer.stop()
        if hasattr(self, 'teleop_timer') and self.teleop_timer.isActive():
            self.teleop_timer.stop()
        if hasattr(self, 'camera_popup') and self.camera_popup:
            self.camera_popup.close()
        if hasattr(self, 'pg_plot') and self.pg_plot:
            try:
                self.pg_plot.close()
            except Exception:
                pass
        if hasattr(self, 'canvas') and self.canvas:
            try:
                self.canvas.draw_idle = lambda: None
            except Exception:
                pass
        if self.ros:
            self.ros.stop()
        super().closeEvent(event)


# ==============================================================================
# Entry Point
# ==============================================================================


def main(args=None):
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ and 'cv2' in os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    app = QApplication(sys.argv)

    ros_worker = MappingRosWorker()
    ros_worker.init_ros()

    # Best Practice: Register atexit and signal handlers to guarantee child process cleanup
    atexit.register(ros_worker.kill_explorer_instances)
    atexit.register(ros_worker.stop)

    def _sig_handler(signum, frame):
        app.quit()

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    ros_thread = threading.Thread(target=ros_worker.spin, daemon=True)
    ros_thread.start()

    window = MappingDashboardWindow(ros_worker)
    window.show()

    ret = app.exec_()
    ros_worker.kill_explorer_instances()
    ros_worker.stop()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
