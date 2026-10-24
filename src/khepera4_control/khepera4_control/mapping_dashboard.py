#!/usr/bin/env python3
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
  * Real-time Khepera IV footprint, heading vector, trajectory path history
  * IR close-range (orange) and Ultrasonic medium-range (cyan) sensor beam overlay
- Map Persistence Memory: Save, Load, Clear map services
- Complete tactile click visual feedback (:hover, :pressed) on all UI buttons
"""

import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrow, Polygon
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPalette
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QRadioButton, QScrollArea, QSlider, QSplitter,
    QTabWidget, QVBoxLayout, QWidget
)

from geometry_msgs.msg import Twist
from khepera4_control.common import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger


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
    padding: 3px 8px;
    color: #f8fafc;
    font-weight: bold;
}
QComboBox:hover {
    border-color: #38bdf8;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    color: #e2e8f0;
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

    def __init__(self):
        super().__init__()
        self.node: Optional[Node] = None
        self.cmd_pub = None
        self.explorer_enable_pub = None
        self.save_map_cli = None
        self.load_map_cli = None
        self.clear_map_cli = None
        self.recenter_cli = None
        self.set_exp_cli = None
        self.toggle_exp_cli = None
        self.is_running = True
        self.last_fused_odom_time = 0.0

    def _get_now(self) -> float:
        if self.node and self.node.context.ok():
            return self.node.get_clock().now().nanoseconds * 1e-9
        return time.time()

    def init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('khepera4_mapping_dashboard_node')

        # 1. Direct Single-Robot Command Publisher
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.explorer_enable_pub = self.node.create_publisher(Bool, '/explorer/enable', 10)

        # 2. Service Clients for Spatial Persistence & Autonomy & Recenter
        self.save_map_cli = self.node.create_client(Trigger, '/save_map')
        self.load_map_cli = self.node.create_client(Trigger, '/load_map')
        self.clear_map_cli = self.node.create_client(Trigger, '/clear_map')
        self.recenter_cli = self.node.create_client(Trigger, '/recenter_robot')
        self.set_exp_cli = self.node.create_client(SetBool, '/set_explorer')
        self.toggle_exp_cli = self.node.create_client(Trigger, '/toggle_explorer')

        # 3. Subscriptions (Direct single-robot topics + fallbacks)
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.node.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.node.create_subscription(Odometry, '/khepera4/odom', self._fused_odom_cb, 20)
        self.node.create_subscription(Imu, '/imu/data', self._imu_cb, 30)
        self.node.create_subscription(Imu, '/khepera4/imu/data', self._imu_cb, 30)
        self.node.create_subscription(OccupancyGrid, '/map', lambda m: self.map_received.emit(m), qos_latched)
        self.node.create_subscription(OccupancyGrid, '/elevation_map', lambda m: self.elevation_received.emit(m), qos_latched)
        self.node.create_subscription(OccupancyGrid, '/surface_map', lambda m: self.surface_received.emit(m), qos_latched)
        self.node.create_subscription(LaserScan, '/khepera4/scan', lambda m: self.scan_received.emit(m), 10)
        self.node.create_subscription(Bool, '/cliff_alert', lambda m: self.cliff_alert_received.emit(m.data), 10)
        self.node.create_subscription(Bool, '/khepera4/turned_over', lambda m: self.turned_over_received.emit(m.data), 10)
        self.node.create_subscription(Bool, '/explorer/enable', lambda m: self.explorer_state_received.emit(m.data), qos_latched)

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

    def set_auto_explore(self, enable: bool):
        """Call /set_explorer service and publish to /explorer/enable."""
        if self.node and self.explorer_enable_pub and self.node.context.ok():
            bmsg = Bool()
            bmsg.data = bool(enable)
            self.explorer_enable_pub.publish(bmsg)

        def _call():
            if not self.set_exp_cli or not self.node or not self.node.context.ok():
                return
            if not self.set_exp_cli.wait_for_service(timeout_sec=1.2):
                if self.toggle_exp_cli and self.toggle_exp_cli.wait_for_service(timeout_sec=0.5):
                    self.trigger_service(self.toggle_exp_cli, "Toggle Explorer")
                else:
                    self.feedback_received.emit("[Warning] Explorer Service Offline")
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

    def spin(self):
        while self.is_running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def stop(self):
        self.is_running = False
        if self.node:
            self.publish_cmd(0.0, 0.0)


# ==============================================================================
# Main Mapping Dashboard Window
# ==============================================================================
class MappingDashboardWindow(QMainWindow):
    """
    Dedicated Single-Robot Environment Mapping & Exploration Dashboard.
    """

    def __init__(self, ros_worker: Optional[MappingRosWorker] = None):
        super().__init__()
        self.ros = ros_worker

        # Robot State & Telemetry
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_v = 0.0
        self.odom_w = 0.0
        self.has_received_odom = False
        self.robot_path: List[Tuple[float, float]] = []

        # IMU & Attitude
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_az = 9.81
        self.is_cliff_alert = False
        self.is_turned_over = False

        # Autonomy State
        self.is_auto_exploring = False

        # Teleoperation Limits & Keyboard State
        self.max_linear_speed = 0.15   # m/s
        self.max_angular_speed = 1.0   # rad/s
        self.active_cmd_v = 0.0
        self.active_cmd_w = 0.0
        self.keys_pressed = set()
        self.dpad_direction: Optional[str] = None

        # Map Layers Caches
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_elevation: Optional[OccupancyGrid] = None
        self.latest_surface: Optional[OccupancyGrid] = None
        self.latest_scan: Optional[LaserScan] = None

        # Canvas View Settings
        self.map_layer_mode = 'obstacles'   # 'obstacles', 'elevation', 'surface'
        self.show_sensor_beams = True
        self.show_trajectory = True
        self.show_grid = True
        self.zoom_level = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0

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

        # Teleop control timer (20 Hz loop for smooth keyboard/D-Pad response)
        self.teleop_timer = QTimer(self)
        self.teleop_timer.timeout.connect(self.update_teleop_loop)
        self.teleop_timer.start(50)

        # Plot redraw timer (10 Hz)
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_map_plot)
        self.plot_timer.start(100)

        # Initial plot draw
        self.update_map_plot()

    def init_ui(self):
        self.setWindowTitle("Khepera IV Mapping & Spatial Exploration Station")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 660)
        self.setStyleSheet(MODERN_STYLE)

        # Central Splitter Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ----------------------------------------------------------------------
        # Left Control & Telemetry Panel (Wrapped in ScrollArea for responsiveness)
        # ----------------------------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(380)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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

        # 3. Autonomous Exploration GroupBox
        explore_group = QGroupBox("Exploration Autonomy")
        exp_layout = QVBoxLayout(explore_group)
        exp_layout.setSpacing(5)

        self.btn_auto_explore = QPushButton("▶ Start Auto Explore")
        self.btn_auto_explore.setObjectName("auto_explore_idle")
        self.btn_auto_explore.setCursor(Qt.PointingHandCursor)
        self.btn_auto_explore.clicked.connect(self.on_toggle_auto_explore)

        self.lbl_explore_status = QLabel("Mode: Manual Control (Click button to start exploration)")
        self.lbl_explore_status.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.lbl_explore_status.setWordWrap(True)

        exp_layout.addWidget(self.btn_auto_explore)
        exp_layout.addWidget(self.lbl_explore_status)
        left_layout.addWidget(explore_group)

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

        # Arena Center Respawn Button (Recenter Robot rightside up)
        self.btn_recenter = QPushButton("🔄 Recenter Robot (Respawn Upright)")
        self.btn_recenter.setObjectName("recenter_btn")
        self.btn_recenter.setCursor(Qt.PointingHandCursor)
        self.btn_recenter.clicked.connect(self.on_recenter_clicked)
        hud_layout.addWidget(self.btn_recenter)

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

        self.lbl_map_toast = QLabel("[Ready] Persistence services online.")
        self.lbl_map_toast.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #f59e0b; font-family: monospace; font-size: 9px; padding: 4px;"
        )
        self.lbl_map_toast.setWordWrap(True)
        persist_layout.addWidget(self.lbl_map_toast)

        left_layout.addWidget(persist_group)

        left_layout.addStretch()
        scroll_area.setWidget(left_container)
        main_layout.addWidget(scroll_area)

        # ----------------------------------------------------------------------
        # Right Center Panel: Prominent Real-Time Map Display
        # ----------------------------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 0, 2, 0)
        right_layout.setSpacing(6)

        # Map Toolbar & Layer Selector
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(2, 2, 2, 2)
        tb_layout.setSpacing(8)

        lbl_layer = QLabel("Map Layer:")
        lbl_layer.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.combo_layer = QComboBox()
        self.combo_layer.addItems([
            "2D Obstacle Map (/map)",
            "2.5D Elevation Map (/elevation_map)",
            "Surface & Cliff Map (/surface_map)"
        ])
        self.combo_layer.currentIndexChanged.connect(self.on_layer_changed)

        self.chk_sensor_rays = QCheckBox("Sensor Rays")
        self.chk_sensor_rays.setChecked(True)
        self.chk_sensor_rays.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_sensor_rays.toggled.connect(self.on_sensor_rays_toggled)

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

        tb_layout.addWidget(lbl_layer)
        tb_layout.addWidget(self.combo_layer)
        tb_layout.addWidget(self.chk_sensor_rays)
        tb_layout.addWidget(self.chk_trajectory)
        tb_layout.addWidget(self.chk_grid)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_zoom_in)
        tb_layout.addWidget(self.btn_zoom_out)
        tb_layout.addWidget(self.btn_fit)
        right_layout.addWidget(toolbar)

        # Matplotlib Canvas for High-Performance Spatial Map Rendering
        self.figure = Figure(figsize=(8, 7), facecolor='#0b0f19')
        self.canvas = FigureCanvas(self.figure)
        self.ax_map = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.07)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(right_panel, stretch=1)

    # --------------------------------------------------------------------------
    # Teleoperation Speed Controls & Slider Slots
    # --------------------------------------------------------------------------
    def on_v_slider_changed(self, val: int):
        self.max_linear_speed = val / 100.0
        self.lbl_v_val.setText(f"{self.max_linear_speed:.2f} m/s")

    def on_w_slider_changed(self, val: int):
        self.max_angular_speed = val / 100.0
        self.lbl_w_val.setText(f"{self.max_angular_speed:.2f} rad/s")

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
            self.ros.publish_cmd(0.0, 0.0)
        self.lbl_teleop_cmd.setText("Cmd: v = +0.00 m/s | ω = +0.00 rad/s [STOPPED]")

    def keyPressEvent(self, event: QKeyEvent):
        if not event.isAutoRepeat():
            key = event.key()
            self.keys_pressed.add(key)
            if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                       Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
                       Qt.Key_Space):
                if self.is_auto_exploring:
                    self.on_toggle_auto_explore()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
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

    # --------------------------------------------------------------------------
    # Autonomous Exploration Slots
    # --------------------------------------------------------------------------
    def on_toggle_auto_explore(self):
        new_state = not self.is_auto_exploring
        self.is_auto_exploring = new_state
        if self.ros:
            self.ros.set_auto_explore(new_state)

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
        if self.ros:
            self.ros.trigger_service(self.ros.clear_map_cli, "Clear Map")

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

        if not self.robot_path or math.hypot(x - self.robot_path[-1][0], y - self.robot_path[-1][1]) > 0.03:
            self.robot_path.append((x, y))
            if len(self.robot_path) > 3000:
                self.robot_path.pop(0)

    def on_imu_received(self, pitch: float, roll: float, az: float):
        self.imu_pitch = pitch
        self.imu_roll = roll
        self.imu_az = az
        self.lbl_pitch_val.setText(f"{pitch:+.1f}°")
        self.lbl_roll_val.setText(f"{roll:+.1f}°")

        slope = math.hypot(pitch, roll)
        if (az < 1.0) or (slope > 70.0):
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
        self.on_turned_over_received(False)
        if self.ros:
            self.ros.recenter_robot()

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

    def on_elevation_received(self, msg: OccupancyGrid):
        self.latest_elevation = msg

    def on_surface_received(self, msg: OccupancyGrid):
        self.latest_surface = msg

    def on_scan_received(self, msg: LaserScan):
        self.latest_scan = msg

    # --------------------------------------------------------------------------
    # Map Toolbar Slots
    # --------------------------------------------------------------------------
    def on_layer_changed(self, idx: int):
        modes = ['obstacles', 'elevation', 'surface']
        self.map_layer_mode = modes[idx] if idx < len(modes) else 'obstacles'
        self.update_map_plot()

    def on_sensor_rays_toggled(self, checked: bool):
        self.show_sensor_beams = checked
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
        self.ax_map.clear()
        self.ax_map.set_facecolor('#0f172a')

        if self.show_grid:
            self.ax_map.grid(True, color='#1e293b', linestyle='--', linewidth=0.6, alpha=0.8)
        else:
            self.ax_map.grid(False)

        self.ax_map.tick_params(colors='#64748b', labelsize=8)
        for spine in self.ax_map.spines.values():
            spine.set_color('#1e293b')

        self.ax_map.set_xlabel("X [meters]", color='#94a3b8', fontsize=8, labelpad=2)
        self.ax_map.set_ylabel("Y [meters]", color='#94a3b8', fontsize=8, labelpad=2)

        layer_title_map = {
            'obstacles': "2D Probabilistic SLAM Obstacle Grid (/map)",
            'elevation': "2.5D Terrain Slope Elevation Map (/elevation_map)",
            'surface': "Surface Classification & Cliff Hazard Map (/surface_map)"
        }
        self.ax_map.set_title(
            f"Khepera IV Spatial SLAM — {layer_title_map.get(self.map_layer_mode, '')}",
            color='#38bdf8', fontsize=10, fontweight='bold', pad=8
        )

        has_rendered_grid = False

        # 1. Render 2D Obstacle Map Layer
        if self.map_layer_mode == 'obstacles' and self.latest_map is not None:
            try:
                info = self.latest_map.info
                w, h = info.width, info.height
                res = info.resolution
                ox = info.origin.position.x
                oy = info.origin.position.y
                data = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))
                rgba = np.zeros((h, w, 4), dtype=np.float32)

                # Free cells (0): clean dark arena floor
                free_mask = (data == 0)
                rgba[free_mask] = [0.12, 0.18, 0.28, 0.85]

                # Occupied cells (100): vibrant electric cyan
                occ_mask = (data == 100)
                rgba[occ_mask] = [0.00, 0.96, 0.83, 0.98]

                extent = [ox, ox + w * res, oy, oy + h * res]
                self.ax_map.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=1)
                has_rendered_grid = True
            except Exception:
                pass

        # 2. Render 2.5D Elevation Map Layer
        elif self.map_layer_mode == 'elevation' and self.latest_elevation is not None:
            try:
                info = self.latest_elevation.info
                w, h = info.width, info.height
                res = info.resolution
                ox = info.origin.position.x
                oy = info.origin.position.y
                data = np.array(self.latest_elevation.data, dtype=np.int8).reshape((h, w))
                rgba = np.zeros((h, w, 4), dtype=np.float32)

                visited = (data >= 0)
                slopes = np.clip(data[visited], 0, 30) / 30.0
                # Green (flat) -> Yellow (moderate) -> Red (steep)
                rgba[visited, 0] = np.clip(slopes * 2.0, 0.0, 1.0)
                rgba[visited, 1] = np.clip(2.0 * (1.0 - slopes), 0.0, 1.0)
                rgba[visited, 2] = 0.2
                rgba[visited, 3] = 0.85

                extent = [ox, ox + w * res, oy, oy + h * res]
                self.ax_map.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=1)
                has_rendered_grid = True
            except Exception:
                pass

        # 3. Render Surface & Cliff Hazard Map Layer
        elif self.map_layer_mode == 'surface' and self.latest_surface is not None:
            try:
                info = self.latest_surface.info
                w, h = info.width, info.height
                res = info.resolution
                ox = info.origin.position.x
                oy = info.origin.position.y
                data = np.array(self.latest_surface.data, dtype=np.int8).reshape((h, w))
                rgba = np.zeros((h, w, 4), dtype=np.float32)

                # Visited roughness (0 - 85): Soft purple-blue
                rough = (data >= 0) & (data < 100)
                rgba[rough] = [0.35, 0.40, 0.65, 0.80]

                # Cliff drops (100): High-intensity hazard red
                cliffs = (data == 100)
                rgba[cliffs] = [1.00, 0.08, 0.25, 0.98]

                extent = [ox, ox + w * res, oy, oy + h * res]
                self.ax_map.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=1)
                has_rendered_grid = True
            except Exception:
                pass

        # Draw Arena Outline if no map loaded yet
        if not has_rendered_grid:
            arena_rect = plt.Rectangle((-3.0, -3.0), 6.0, 6.0, fill=True,
                                       facecolor='#111827', edgecolor='#334155',
                                       linestyle='--', linewidth=1.2, alpha=0.9, zorder=1)
            self.ax_map.add_patch(arena_rect)
            self.ax_map.text(0, 0, "Awaiting Map Stream from /map\n(Run khepera4_mapper or launch mapping.launch.py)",
                             ha='center', va='center', color='#64748b', fontsize=11, fontweight='bold', zorder=2)

        # 4. Draw Explored Trajectory History
        if self.show_trajectory and len(self.robot_path) > 1:
            px, py = zip(*self.robot_path)
            self.ax_map.plot(px, py, color='#38bdf8', linewidth=1.6, alpha=0.75, linestyle='-', zorder=3)

        # 5. Draw Multi-Sensor Scan Rays (IR close-range in orange, US medium-range in cyan)
        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
        if self.show_sensor_beams and self.latest_scan is not None:
            try:
                ranges = self.latest_scan.ranges
                num_rays = len(ranges)
                angle_min = self.latest_scan.angle_min
                angle_inc = self.latest_scan.angle_increment
                step = max(1, num_rays // 40)

                for i in range(0, num_rays, step):
                    r = ranges[i]
                    if math.isnan(r) or r < 0.005:
                        continue
                    r_clamped = min(r, 2.0)
                    beam_angle = ryaw + angle_min + i * angle_inc
                    hit_x = rx + r_clamped * math.cos(beam_angle)
                    hit_y = ry + r_clamped * math.sin(beam_angle)

                    color = '#f59e0b' if r < 0.25 else '#06b6d4'
                    self.ax_map.plot([rx, hit_x], [ry, hit_y], color=color, linewidth=0.8, alpha=0.45, zorder=4)
                    if r < 1.95:
                        self.ax_map.plot(hit_x, hit_y, '.', color=color, markersize=3.5, alpha=0.85, zorder=4)
            except Exception:
                pass

        # 6. Draw Khepera IV True-Scale Footprint (Radius: 0.070 m = 140 mm diameter)
        khep_circle = Circle((rx, ry), 0.070, facecolor='#0284c7', edgecolor='#7dd3fc',
                             linewidth=2.0, alpha=0.85, zorder=5)
        self.ax_map.add_patch(khep_circle)

        # Heading Vector Arrow
        arrow_len = 0.16
        hx = rx + arrow_len * math.cos(ryaw)
        hy = ry + arrow_len * math.sin(ryaw)
        self.ax_map.annotate(
            '', xy=(hx, hy), xytext=(rx, ry),
            arrowprops=dict(arrowstyle="-|>", color='#ffffff', lw=2.2, mutation_scale=14),
            zorder=6
        )
        self.ax_map.plot(rx, ry, 'o', color='#ffffff', markersize=3, zorder=7)

        # 7. Apply View Bounds & Zoom
        base_half_span = 3.5 * self.zoom_level
        self.ax_map.set_xlim(self.view_offset_x - base_half_span, self.view_offset_x + base_half_span)
        self.ax_map.set_ylim(self.view_offset_y - base_half_span, self.view_offset_y + base_half_span)
        self.ax_map.set_aspect('equal', adjustable='box')

        # 8. Canvas Overlay HUD Text (Pose & System Summary)
        odom_source = "FUSED (/khepera4/odom)" if (self.ros and self.ros.using_fused_odom) else "RAW (/odom)"
        hud_text = f"Pose: X: {rx:+.2f} m | Y: {ry:+.2f} m | Yaw: {math.degrees(ryaw):+.1f}°\n" \
                   f"Speed: {self.odom_v:+.2f} m/s | Odom: {odom_source} | Path: {len(self.robot_path)} pts"
        self.ax_map.text(
            0.02, 0.02, hud_text, transform=self.ax_map.transAxes,
            color='#e2e8f0', fontsize=8.5, fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#0b0f19', edgecolor='#334155', alpha=0.85),
            zorder=8
        )

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

        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self.ros:
            self.ros.stop()
        super().closeEvent(event)


# ==============================================================================
# Entry Point
# ==============================================================================
def main(args=None):
    app = QApplication(sys.argv)

    ros_worker = MappingRosWorker()
    ros_worker.init_ros()

    ros_thread = threading.Thread(target=ros_worker.spin, daemon=True)
    ros_thread.start()

    window = MappingDashboardWindow(ros_worker)
    window.show()

    ret = app.exec_()
    ros_worker.stop()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
