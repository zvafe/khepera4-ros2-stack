#!/usr/bin/env python3
"""
Khepera IV Control Station & Educational Dashboard
Course: Bazele Sistemelor Automate (BSA)

Interactive GUI providing:
- Mode A: Manual Video Game Drive (WASD)
- Mode B: Closed-Loop Step Response / Setpoint Regulation
- Mode C: Trajectory Tracking (Square / Circular Orbit)
- 3D Gazebo Goal Beacon Teleportation
- 1-2-3 Visual Explainer of Robot Brain Dynamics
- Dynamic Error Calculator & Step-Response Metrics (tr, ts, sigma%, est)
- Embedded Real-Time 2D Arena Map and Error Curves
"""

import sys
import os
import math
import subprocess
import threading
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QRadioButton,
    QButtonGroup, QGroupBox, QLineEdit, QSplitter, QFrame,
    QTabWidget, QProgressBar, QDoubleSpinBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QKeyEvent

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool
from khepera4_control.common import euler_from_quaternion, normalize_angle


# Direct Python Gazebo Transport for Instant Beacon Teleportation
try:
    import gz.transport13 as gz_transport
    from gz.msgs10.pose_pb2 import Pose as GzPose
    from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
    _GZ_NODE = gz_transport.Node()
    _HAS_GZ_TRANSPORT = True
except Exception:
    _HAS_GZ_TRANSPORT = False
    _GZ_NODE = None


def set_gazebo_goal_marker(x: float, y: float, z: float = 0.005):
    """Instantly teleport the visual 3D goal marker inside Gazebo."""
    def _call():
        if _HAS_GZ_TRANSPORT and _GZ_NODE:
            try:
                req = GzPose()
                req.name = 'goal_marker'
                req.position.x = float(x)
                req.position.y = float(y)
                req.position.z = float(z)
                rep = GzBoolean()
                success, rep = _GZ_NODE.request('/world/khepera4_world/set_pose', req, GzPose, GzBoolean, 800)
                if success and rep.data:
                    return
            except Exception:
                pass

        # Fallback to gz service CLI with proper timeout (2.5s)
        req_str = f'name: "goal_marker", position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}'
        cmd = [
            'gz', 'service', '-s', '/world/khepera4_world/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000', '--req', req_str
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
        except Exception:
            pass

    threading.Thread(target=_call, daemon=True).start()


class SmoothSpinBox(QDoubleSpinBox):
    """Double spinbox supporting mouse wheel scrolling by step and keyboard arrows."""
    def __init__(self, step=0.05, decimals=2, min_val=-20.0, max_val=20.0, parent=None):
        super().__init__(parent)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setRange(min_val, max_val)
        self.setKeyboardTracking(False)
        self.setToolTip("Scroll mouse wheel or click ▲ / ▼ arrows on right to step by ±0.05 m")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.stepBy(1)
        elif delta < 0:
            self.stepBy(-1)
        event.accept()


# ==============================================================================
# ROS 2 Worker Node
# ==============================================================================
# ==============================================================================
# ROS 2 Worker Node (Multi-Robot Fleet & Swarm Aware)
# ==============================================================================
class RosWorker(QObject):
    odom_received = pyqtSignal(float, float, float, float, float)  # active robot: x, y, yaw, v, w
    multi_odom_received = pyqtSignal(str, float, float, float, float, float)  # robot_name, x, y, yaw, v, w
    swarm_status_received = pyqtSignal(str)
    map_received = pyqtSignal(object)
    elevation_received = pyqtSignal(object)
    scan_received = pyqtSignal(object)
    imu_received = pyqtSignal(float, float, float)  # pitch_deg, roll_deg, az
    cliff_alert_received = pyqtSignal(bool)
    map_feedback_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.node = None
        self.cmd_pub = None
        self.cmd_pubs = {}
        self.swarm_cmd_pub = None
        self.save_map_cli = None
        self.clear_map_cli = None
        self.load_map_cli = None
        self.toggle_exp_cli = None
        self.set_exp_cli = None
        self.is_running = True

    def init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('khepera4_dashboard_node')

        # Publishers for robot namespaces
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_pubs['khepera_1'] = self.node.create_publisher(Twist, '/khepera_1/cmd_vel', 10)
        self.cmd_pubs['khepera_2'] = self.node.create_publisher(Twist, '/khepera_2/cmd_vel', 10)
        self.cmd_pubs['khepera_3'] = self.node.create_publisher(Twist, '/khepera_3/cmd_vel', 10)
        self.swarm_cmd_pub = self.node.create_publisher(String, '/swarm/cmd', 10)

        # Service clients for mapping and exploration
        self.save_map_cli = self.node.create_client(Trigger, '/save_map')
        self.clear_map_cli = self.node.create_client(Trigger, '/clear_map')
        self.load_map_cli = self.node.create_client(Trigger, '/load_map')
        self.toggle_exp_cli = self.node.create_client(Trigger, '/toggle_explorer')
        self.set_exp_cli = self.node.create_client(SetBool, '/set_explorer')

        # Subscriptions for 3-robot fleet odometry
        self.node.create_subscription(Odometry, '/khepera_1/odom', lambda m: self._multi_odom_cb(m, 'khepera_1'), 10)
        self.node.create_subscription(Odometry, '/khepera_2/odom', lambda m: self._multi_odom_cb(m, 'khepera_2'), 10)
        self.node.create_subscription(Odometry, '/khepera_3/odom', lambda m: self._multi_odom_cb(m, 'khepera_3'), 10)
        self.node.create_subscription(String, '/swarm/status', self._swarm_status_cb, 10)

        # Sensor and Mapping Subscriptions
        self.node.create_subscription(OccupancyGrid, '/map', lambda m: self.map_received.emit(m), 1)
        self.node.create_subscription(OccupancyGrid, '/elevation_map', lambda m: self.elevation_received.emit(m), 1)
        self.node.create_subscription(LaserScan, '/khepera4/scan', lambda m: self.scan_received.emit(m), 1)
        self.node.create_subscription(Imu, '/imu/data', self._imu_cb, 1)
        self.node.create_subscription(Bool, '/cliff_alert', lambda m: self.cliff_alert_received.emit(m.data), 1)

    def _multi_odom_cb(self, msg: Odometry, name: str):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        self.multi_odom_received.emit(name, pos.x, pos.y, yaw, twist.linear.x, twist.angular.z)

    def _swarm_status_cb(self, msg: String):
        self.swarm_status_received.emit(msg.data)

    def _imu_cb(self, msg: Imu):
        orient = msg.orientation
        roll, pitch, _ = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        self.imu_received.emit(math.degrees(pitch), math.degrees(roll), msg.linear_acceleration.z)

    def trigger_service(self, client, action_name: str):
        import time
        def _call():
            if not client or not self.node or not self.node.context.ok():
                return
            if not client.wait_for_service(timeout_sec=1.5):
                self.map_feedback_received.emit(f"⚠️ {action_name}: Service offline")
                return
            req = Trigger.Request()
            future = client.call_async(req)
            t0 = time.time()
            while not future.done() and (time.time() - t0 < 3.0):
                time.sleep(0.05)
            if future.done():
                try:
                    res = future.result()
                    self.map_feedback_received.emit(f"{'✅' if res.success else '⚠️'} {res.message}")
                except Exception as e:
                    self.map_feedback_received.emit(f"❌ Error: {e}")
            else:
                self.map_feedback_received.emit(f"⏱️ {action_name} timed out")
        threading.Thread(target=_call, daemon=True).start()

    def set_explorer_enabled(self, enabled: bool):
        import time
        def _call():
            if not self.set_exp_cli or not self.node or not self.node.context.ok():
                return
            if not self.set_exp_cli.wait_for_service(timeout_sec=1.5):
                self.map_feedback_received.emit("⚠️ Explorer service offline")
                return
            req = SetBool.Request()
            req.data = bool(enabled)
            future = self.set_exp_cli.call_async(req)
            t0 = time.time()
            while not future.done() and (time.time() - t0 < 3.0):
                time.sleep(0.05)
            if future.done():
                try:
                    res = future.result()
                    self.map_feedback_received.emit(f"🤖 Explorer: {res.message}")
                except Exception as e:
                    self.map_feedback_received.emit(f"❌ Error: {e}")
            else:
                self.map_feedback_received.emit("⏱️ Explorer request timed out")
        threading.Thread(target=_call, daemon=True).start()

    def publish_cmd(self, v: float, w: float, target_robot: str = 'khepera_1'):
        if self.node and self.node.context.ok():
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            if target_robot == 'all':
                for pub in self.cmd_pubs.values():
                    pub.publish(msg)
            elif target_robot in self.cmd_pubs:
                self.cmd_pubs[target_robot].publish(msg)
            else:
                self.cmd_pub.publish(msg)

    def publish_swarm_cmd(self, cmd_text: str):
        if self.node and self.swarm_cmd_pub and self.node.context.ok():
            msg = String()
            msg.data = str(cmd_text)
            self.swarm_cmd_pub.publish(msg)

    def spin(self):
        while self.is_running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def stop(self):
        self.is_running = False
        if self.node:
            self.publish_cmd(0.0, 0.0, target_robot='all')


# ==============================================================================
# Main Dashboard Window
# ==============================================================================
class KheperaDashboard(QMainWindow):
    def __init__(self, ros_worker: RosWorker):
        super().__init__()
        self.ros = ros_worker
        self.setWindowTitle("Khepera IV Control Station & Educational Dashboard (BSA)")
        self.resize(1300, 850)
        self.setMinimumSize(1100, 750)

        # Robot Physical Constants (K-Team Khepera IV)
        self.WHEEL_RADIUS = 0.021    # m
        self.WHEEL_BASE = 0.1054     # m

        # State Variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0

        # Mode: 'A' (WASD), 'B' (Step/Goal), 'C' (Trajectory)
        self.current_mode = 'A'

        # Key state for WASD
        self.keys_pressed = set()
        self.max_lin_speed = 0.15   # m/s
        self.max_ang_speed = 1.2    # rad/s

        # Mode B & C Goals
        self.goal_x = 0.5
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.waypoints: List[Tuple[float, float, float]] = []
        self.waypoint_idx = 0
        self.is_navigating = False

        # Feedback Gains (Cascade / Lyapunov)
        self.k_rho = 0.8
        self.k_alpha = 2.0
        self.k_beta = -0.4
        self.k_yaw = 1.5
        self.pos_tol = 0.02
        self.yaw_tol = 0.05

        # Telemetry & Performance History
        self.history_time = []
        self.history_x = []
        self.history_y = []
        self.history_rho = []
        self.nav_start_time = None
        self.initial_target_dist = 0.0
        self.start_nav_pos = (0.0, 0.0)

        # Multi-Robot Fleet State
        self.active_robot = 'khepera_1'
        self.fleet = {
            'khepera_1': {'role': 'base_link', 'color': '#2979ff', 'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'v': 0.0, 'w': 0.0, 'spawn': (0.0, 0.0, 0.0), 'trail_x': [], 'trail_y': []},
            'khepera_2': {'role': 'extender',  'color': '#ff9100', 'x': 0.6, 'y': 0.0, 'yaw': 0.0, 'v': 0.0, 'w': 0.0, 'spawn': (0.6, 0.0, 0.0), 'trail_x': [], 'trail_y': []},
            'khepera_3': {'role': 'explorer',  'color': '#00e676', 'x': 1.2, 'y': 0.0, 'yaw': 0.0, 'v': 0.0, 'w': 0.0, 'spawn': (1.2, 0.0, 0.0), 'trail_x': [], 'trail_y': []},
        }

        # Multi-Sensor Mapping & Spatial Memory State
        self.latest_map = None
        self.latest_elevation = None
        self.latest_scan = None
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.imu_az = 9.81
        self.is_cliff_alert = False
        self.map_layer_mode = 'obstacles'
        self.show_sensor_rays = True
        self.is_auto_exploring = False
        self.map_feedback_msg = "Map Engine: Ready"

        # Connect ROS signals
        self.ros.multi_odom_received.connect(self.on_multi_odom)
        self.ros.swarm_status_received.connect(self.on_swarm_status)
        self.ros.map_received.connect(self.on_map_received)
        self.ros.elevation_received.connect(self.on_elevation_received)
        self.ros.scan_received.connect(self.on_scan_received)
        self.ros.imu_received.connect(self.on_imu_received)
        self.ros.cliff_alert_received.connect(self.on_cliff_alert_received)
        self.ros.map_feedback_received.connect(self.on_map_feedback_received)

        # UI Setup
        self.init_ui()

        # High-rate control & UI update timer (50 Hz)
        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.on_control_tick)
        self.control_timer.start(20)  # 20ms = 50 Hz

        # Plot refresh timer (10 Hz)
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(100)

        # Place initial visual beacon in Gazebo
        set_gazebo_goal_marker(self.goal_x, self.goal_y)

        # Initial plot draw
        self.update_plots()

    # --------------------------------------------------------------------------
    # UI Building
    # --------------------------------------------------------------------------
    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #16191f; }
            QWidget { color: #e6e8ec; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox {
                border: 1px solid #2d3340;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                background-color: #1e222b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #4da3ff;
            }
            QPushButton {
                background-color: #2b3342;
                border: 1px solid #3c4659;
                border-radius: 5px;
                padding: 8px 14px;
                font-weight: bold;
                color: #f8fafc;
            }
            QPushButton:hover { background-color: #3b4559; border-color: #4da3ff; }
            QPushButton:pressed {
                background-color: #1d6ed8;
                border-color: #93c5fd;
                padding-top: 10px;
                padding-bottom: 6px;
            }
            QRadioButton { font-weight: bold; font-size: 13px; spacing: 8px; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QLineEdit {
                background-color: #12151a;
                border: 1px solid #3c4659;
                border-radius: 4px;
                padding: 4px 8px;
                color: #00ffcc;
                font-family: monospace;
            }
            QDoubleSpinBox {
                background-color: #12151a;
                border: 1px solid #3c4659;
                border-radius: 4px;
                padding: 5px 6px;
                color: #00ffcc;
                font-family: monospace;
                font-size: 13px;
                font-weight: bold;
            }
            QDoubleSpinBox:hover { border-color: #4da3ff; }
            QDoubleSpinBox:focus { border-color: #00e676; }
            QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 24px;
                background-color: #242b38;
                border-left: 1px solid #3c4659;
                border-bottom: 1px solid #3c4659;
                border-top-right-radius: 4px;
            }
            QDoubleSpinBox::up-button:hover { background-color: #3b465c; }
            QDoubleSpinBox::up-arrow {
                width: 0; height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 6px solid #00ffcc;
            }
            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 24px;
                background-color: #242b38;
                border-left: 1px solid #3c4659;
                border-bottom-right-radius: 4px;
            }
            QDoubleSpinBox::down-button:hover { background-color: #3b465c; }
            QDoubleSpinBox::down-arrow {
                width: 0; height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #00ffcc;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2b3342;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4da3ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #4da3ff;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Header Bar
        header = QHBoxLayout()
        title_label = QLabel("🤖 Khepera IV Control Station & Educational Dynamics Dashboard")
        title_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        header.addWidget(title_label)

        # 2. Multi-Robot Fleet & Swarm Operations Bar
        fleet_bar = QHBoxLayout()
        lbl_target = QLabel("🎯 Active Target Robot:")
        lbl_target.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_target.setStyleSheet("color: #4da3ff;")
        fleet_bar.addWidget(lbl_target)

        self.combo_robot = QComboBox()
        self.combo_robot.addItems([
            "khepera_1 [base_link]",
            "khepera_2 [extender]",
            "khepera_3 [explorer]",
            "All Robots (Broadcast)"
        ])
        self.combo_robot.setStyleSheet("""
            QComboBox {
                background-color: #242933;
                border: 2px solid #4da3ff;
                border-radius: 5px;
                padding: 4px 10px;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                min-width: 190px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e222b;
                selection-background-color: #4da3ff;
                color: #ffffff;
            }
        """)
        self.combo_robot.currentIndexChanged.connect(self.on_active_robot_changed)
        fleet_bar.addWidget(self.combo_robot)

        fleet_bar.addSpacing(15)

        # Swarm quick coordination buttons
        btn_relay = QPushButton("🛰️ Chain Relay")
        btn_relay.setToolTip("Extender positions dynamically as midpoint tether between Base and Explorer")
        btn_relay.setStyleSheet("""
            QPushButton { background-color: #1b4d3e; border: 1px solid #00e676; color: #00ffcc; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #246854; border-color: #69f0ae; }
            QPushButton:pressed { background-color: #12382c; border-color: #a7f3d0; padding-top: 8px; padding-bottom: 4px; }
        """)
        btn_relay.clicked.connect(lambda: self.ros.publish_swarm_cmd("mode:relay"))
        fleet_bar.addWidget(btn_relay)

        btn_triangle = QPushButton("🔺 Triangle Formation")
        btn_triangle.setStyleSheet("""
            QPushButton { background-color: #3e2723; border: 1px solid #ff9100; color: #ffe082; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #5d3a34; border-color: #ffb74d; }
            QPushButton:pressed { background-color: #2b1b18; border-color: #ffe082; padding-top: 8px; padding-bottom: 4px; }
        """)
        btn_triangle.clicked.connect(lambda: self.ros.publish_swarm_cmd("mode:triangle"))
        fleet_bar.addWidget(btn_triangle)

        btn_return = QPushButton("🏠 Return to Base")
        btn_return.setStyleSheet("""
            QPushButton { background-color: #1a237e; border: 1px solid #2979ff; color: #bbdefb; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #283593; border-color: #82b1ff; }
            QPushButton:pressed { background-color: #121858; border-color: #bbdefb; padding-top: 8px; padding-bottom: 4px; }
        """)
        btn_return.clicked.connect(lambda: self.ros.publish_swarm_cmd("return_to_base"))
        fleet_bar.addWidget(btn_return)

        btn_halt = QPushButton("⏸️ Halt Swarm")
        btn_halt.setStyleSheet("""
            QPushButton { background-color: #b71c1c; border: 1px solid #ff5252; color: #ffffff; border-radius: 5px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #c62828; border-color: #ff8a80; }
            QPushButton:pressed { background-color: #7f0000; border-color: #ffcdd2; padding-top: 8px; padding-bottom: 4px; }
        """)
        btn_halt.clicked.connect(lambda: self.ros.publish_swarm_cmd("halt"))
        fleet_bar.addWidget(btn_halt)

        fleet_bar.addStretch()

        self.lbl_tether_status = QLabel("Tether: Base <-> Extender <-> Explorer Active")
        self.lbl_tether_status.setStyleSheet("color: #00e676; font-size: 11px; font-weight: bold;")
        fleet_bar.addWidget(self.lbl_tether_status)

        main_layout.addLayout(fleet_bar)

        # 3. Mode Selector Bar
        mode_box = QGroupBox("Select Driving Mode")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setSpacing(20)

        self.btn_mode_a = QRadioButton("🚗 Option A: Manual Video Game Drive (WASD)")
        self.btn_mode_b = QRadioButton("🎯 Option B: Closed-Loop Step / Setpoint Regulation")
        self.btn_mode_c = QRadioButton("🔄 Option C: Autonomous Multi-Waypoint Trajectory")

        self.btn_mode_a.setChecked(True)
        self.btn_mode_a.setStyleSheet("color: #00e676;")
        self.btn_mode_b.setStyleSheet("color: #ff9100;")
        self.btn_mode_c.setStyleSheet("color: #00b0ff;")

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.btn_mode_a)
        mode_group.addButton(self.btn_mode_b)
        mode_group.addButton(self.btn_mode_c)

        self.btn_mode_a.toggled.connect(lambda: self.switch_mode('A'))
        self.btn_mode_b.toggled.connect(lambda: self.switch_mode('B'))
        self.btn_mode_c.toggled.connect(lambda: self.switch_mode('C'))

        mode_layout.addWidget(self.btn_mode_a)
        mode_layout.addWidget(self.btn_mode_b)
        mode_layout.addWidget(self.btn_mode_c)
        mode_layout.addStretch()

        self.lbl_odom_status = QLabel("● Simulation Connected (Odom Active)")
        self.lbl_odom_status.setStyleSheet("color: #00e676; font-size: 11px; font-weight: bold;")
        mode_layout.addWidget(self.lbl_odom_status)

        main_layout.addWidget(mode_box)

        # 3. Middle Area (Split between Controls & Visual Explainer / Plots)
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter, 1)

        # --- Left Panel: Driving Controls (Tabbed or Stacked by Mode) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(10)

        self.mode_panel_a = self.build_mode_a_panel()
        self.mode_panel_b = self.build_mode_b_panel()
        self.mode_panel_c = self.build_mode_c_panel()

        left_layout.addWidget(self.mode_panel_a)
        left_layout.addWidget(self.mode_panel_b)
        left_layout.addWidget(self.mode_panel_c)
        self.mode_panel_b.hide()
        self.mode_panel_c.hide()

        # Dynamic Error Calculator & Metrics Box
        self.error_calc_box = self.build_error_calculator_box()
        left_layout.addWidget(self.error_calc_box)
        left_layout.addStretch()

        content_splitter.addWidget(left_widget)

        # --- Right Panel: Brain Explainer (1-2-3) & Live Graphs ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1-2-3 Brain Explainer
        brain_box = self.build_brain_explainer_box()
        right_layout.addWidget(brain_box)

        # Live Real-Time Matplotlib Canvas (Top-Down Arena Map & Error Curve)
        plot_box = self.build_plot_box()
        right_layout.addWidget(plot_box, 1)

        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([460, 840])

    # --------------------------------------------------------------------------
    # Panel Builders
    # --------------------------------------------------------------------------
    def build_mode_a_panel(self) -> QGroupBox:
        box = QGroupBox("Option A: Interactive Keyboard Car Control")
        layout = QVBoxLayout(box)

        info = QLabel("Click this window to focus, then drive with <b>W A S D</b> keys like a car in a video game!")
        info.setWordWrap(True)
        info.setStyleSheet("color: #a4b3c6; font-size: 11px;")
        layout.addWidget(info)

        # WASD Virtual Key Visualizer
        key_grid = QGridLayout()
        key_grid.setAlignment(Qt.AlignCenter)

        self.key_w_lbl = QLabel("W\nForward")
        self.key_a_lbl = QLabel("A\nLeft")
        self.key_s_lbl = QLabel("S\nReverse")
        self.key_d_lbl = QLabel("D\nRight")

        for lbl in (self.key_w_lbl, self.key_a_lbl, self.key_s_lbl, self.key_d_lbl):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(70, 48)
            lbl.setStyleSheet("border: 2px solid #3c4659; border-radius: 6px; background-color: #242933; font-weight: bold;")

        key_grid.addWidget(self.key_w_lbl, 0, 1)
        key_grid.addWidget(self.key_a_lbl, 1, 0)
        key_grid.addWidget(self.key_s_lbl, 1, 1)
        key_grid.addWidget(self.key_d_lbl, 1, 2)
        layout.addLayout(key_grid)

        # Speed Sliders
        sl_layout = QGridLayout()
        sl_layout.addWidget(QLabel("Max Linear Speed (m/s):"), 0, 0)
        self.lbl_v_slider = QLabel(f"{self.max_lin_speed:.2f}")
        self.lbl_v_slider.setStyleSheet("color: #00ffcc; font-weight: bold;")
        sl_layout.addWidget(self.lbl_v_slider, 0, 1)

        self.slider_v = QSlider(Qt.Horizontal)
        self.slider_v.setRange(5, 30)  # 0.05 to 0.30 m/s
        self.slider_v.setValue(int(self.max_lin_speed * 100))
        self.slider_v.valueChanged.connect(self.on_slider_v_changed)
        sl_layout.addWidget(self.slider_v, 1, 0, 1, 2)

        sl_layout.addWidget(QLabel("Max Angular Speed (rad/s):"), 2, 0)
        self.lbl_w_slider = QLabel(f"{self.max_ang_speed:.1f}")
        self.lbl_w_slider.setStyleSheet("color: #00ffcc; font-weight: bold;")
        sl_layout.addWidget(self.lbl_w_slider, 2, 1)

        self.slider_w = QSlider(Qt.Horizontal)
        self.slider_w.setRange(5, 25)  # 0.5 to 2.5 rad/s
        self.slider_w.setValue(int(self.max_ang_speed * 10))
        self.slider_w.valueChanged.connect(self.on_slider_w_changed)
        sl_layout.addWidget(self.slider_w, 3, 0, 1, 2)

        layout.addLayout(sl_layout)

        # Commanded Speeds Readout
        self.lbl_a_cmd = QLabel("Commanded: v = 0.00 m/s | w = 0.00 rad/s")
        self.lbl_a_cmd.setStyleSheet("color: #00e676; font-family: monospace; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.lbl_a_cmd)

        return box

    def build_mode_b_panel(self) -> QGroupBox:
        box = QGroupBox("Option B: Closed-Loop Step / Setpoint Regulation")
        layout = QVBoxLayout(box)

        desc = QLabel("Command an exact position setpoint. The outer-loop controller calculates tracking errors and decelerates smoothly onto the target.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a4b3c6; font-size: 11px;")
        layout.addWidget(desc)

        # Quick Presets (X and Y Step Buttons + Snap Button)
        preset_box = QFrame()
        preset_box.setStyleSheet("background-color: #171c26; border: 1px solid #283142; border-radius: 6px; padding: 4px;")
        pl = QVBoxLayout(preset_box)
        pl.setContentsMargins(6, 4, 6, 4)
        pl.setSpacing(4)

        x_row = QHBoxLayout()
        x_lbl = QLabel("<b>X Steps:</b>")
        x_lbl.setStyleSheet("color: #38bdf8; font-size: 11px;")
        x_row.addWidget(x_lbl)
        btn_xp05 = QPushButton("+0.5 m")
        btn_xm05 = QPushButton("-0.5 m")
        btn_xp10 = QPushButton("+1.0 m")
        btn_xp05.clicked.connect(lambda: self.adjust_setpoint(dx=0.5))
        btn_xm05.clicked.connect(lambda: self.adjust_setpoint(dx=-0.5))
        btn_xp10.clicked.connect(lambda: self.adjust_setpoint(dx=1.0))
        for b in (btn_xp05, btn_xm05, btn_xp10):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            x_row.addWidget(b)
        pl.addLayout(x_row)

        y_row = QHBoxLayout()
        y_lbl = QLabel("<b>Y Steps:</b>")
        y_lbl.setStyleSheet("color: #a855f7; font-size: 11px;")
        y_row.addWidget(y_lbl)
        btn_yp05 = QPushButton("+0.5 m")
        btn_ym05 = QPushButton("-0.5 m")
        btn_yp10 = QPushButton("+1.0 m")
        btn_yp05.clicked.connect(lambda: self.adjust_setpoint(dy=0.5))
        btn_ym05.clicked.connect(lambda: self.adjust_setpoint(dy=-0.5))
        btn_yp10.clicked.connect(lambda: self.adjust_setpoint(dy=1.0))
        for b in (btn_yp05, btn_ym05, btn_yp10):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            y_row.addWidget(b)
        pl.addLayout(y_row)

        btn_snap = QPushButton("🎯 Snap Target to Current Robot Position")
        btn_snap.setStyleSheet("background-color: #1f2937; border: 1px dashed #4b5563; font-size: 11px; padding: 5px;")
        btn_snap.clicked.connect(self.snap_setpoint_to_robot)
        pl.addWidget(btn_snap)

        layout.addWidget(preset_box)

        # Custom Inputs with SmoothSpinBox
        inp_grid = QGridLayout()
        inp_grid.setVerticalSpacing(6)

        # Tip on scroll & arrow usage
        tip_lbl = QLabel("🖱️ <i>Scroll wheel or click ▲ / ▼ to step by ±0.05 m:</i>")
        tip_lbl.setStyleSheet("color: #00ffcc; font-size: 11px;")
        inp_grid.addWidget(tip_lbl, 0, 0, 1, 2)

        inp_grid.addWidget(QLabel("Goal X [m]:"), 1, 0)
        self.spin_x = SmoothSpinBox(step=0.05, decimals=2, min_val=-20.0, max_val=20.0)
        self.spin_x.setValue(self.goal_x)
        self.spin_x.valueChanged.connect(self.on_setpoint_spin_changed)
        inp_grid.addWidget(self.spin_x, 1, 1)

        inp_grid.addWidget(QLabel("Goal Y [m]:"), 2, 0)
        self.spin_y = SmoothSpinBox(step=0.05, decimals=2, min_val=-20.0, max_val=20.0)
        self.spin_y.setValue(self.goal_y)
        self.spin_y.valueChanged.connect(self.on_setpoint_spin_changed)
        inp_grid.addWidget(self.spin_y, 2, 1)

        inp_grid.addWidget(QLabel("Goal Yaw [deg]:"), 3, 0)
        self.spin_yaw = SmoothSpinBox(step=5.0, decimals=1, min_val=-180.0, max_val=180.0)
        self.spin_yaw.setValue(0.0)
        self.spin_yaw.valueChanged.connect(self.on_setpoint_spin_changed)
        inp_grid.addWidget(self.spin_yaw, 3, 1)

        layout.addLayout(inp_grid)

        # Action Buttons
        act_layout = QHBoxLayout()
        self.btn_exec_b = QPushButton("▶ Execute Step Setpoint")
        self.btn_exec_b.setStyleSheet("""
            QPushButton { background-color: #00897b; border: 1px solid #26a69a; color: white; border-radius: 5px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #009688; border-color: #80cbc4; }
            QPushButton:pressed { background-color: #004d40; border-color: #b2dfdb; padding-top: 10px; padding-bottom: 6px; }
        """)
        self.btn_exec_b.clicked.connect(self.start_mode_b_nav)

        self.btn_stop_b = QPushButton("⏹ Stop / Abort")
        self.btn_stop_b.setStyleSheet("""
            QPushButton { background-color: #d32f2f; border: 1px solid #ef5350; color: white; border-radius: 5px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #e53935; border-color: #e57373; }
            QPushButton:pressed { background-color: #b71c1c; border-color: #ffcdd2; padding-top: 10px; padding-bottom: 6px; }
        """)
        self.btn_stop_b.clicked.connect(self.stop_navigation)

        act_layout.addWidget(self.btn_exec_b)
        act_layout.addWidget(self.btn_stop_b)
        layout.addLayout(act_layout)

        beacon_note = QLabel("📍 <b>Live 3D Indicator:</b> Changing values immediately teleports the 3D beacon in Gazebo and updates the mini-map.")
        beacon_note.setWordWrap(True)
        beacon_note.setStyleSheet("color: #ffb74d; font-size: 11px;")
        layout.addWidget(beacon_note)

        return box

    def build_mode_c_panel(self) -> QGroupBox:
        box = QGroupBox("Option C: Autonomous Trajectory Path Following")
        layout = QVBoxLayout(box)

        desc = QLabel("Navigate multi-waypoint geometric paths (square or circular orbit) testing differential steering and non-holonomic curvature.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a4b3c6; font-size: 11px;")
        layout.addWidget(desc)

        traj_btns = QHBoxLayout()
        btn_square = QPushButton("⬛ 0.8m Square Path")
        btn_circle = QPushButton("⭕ 0.8m Circular Orbit")
        btn_square.clicked.connect(self.setup_square_trajectory)
        btn_circle.clicked.connect(self.setup_circle_trajectory)
        traj_btns.addWidget(btn_square)
        traj_btns.addWidget(btn_circle)
        layout.addLayout(traj_btns)

        self.lbl_traj_status = QLabel("Trajectory: Inactive")
        self.lbl_traj_status.setStyleSheet("color: #00b0ff; font-weight: bold;")
        layout.addWidget(self.lbl_traj_status)

        act_c = QHBoxLayout()
        self.btn_start_c = QPushButton("▶ Start Trajectory")
        self.btn_start_c.setStyleSheet("""
            QPushButton { background-color: #1565c0; border: 1px solid #1e88e5; color: white; border-radius: 5px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #1976d2; border-color: #64b5f6; }
            QPushButton:pressed { background-color: #0d47a1; border-color: #bbdefb; padding-top: 10px; padding-bottom: 6px; }
        """)
        self.btn_start_c.clicked.connect(self.start_mode_c_nav)

        self.btn_stop_c = QPushButton("⏹ Stop Trajectory")
        self.btn_stop_c.setStyleSheet("""
            QPushButton { background-color: #d32f2f; border: 1px solid #ef5350; color: white; border-radius: 5px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #e53935; border-color: #e57373; }
            QPushButton:pressed { background-color: #b71c1c; border-color: #ffcdd2; padding-top: 10px; padding-bottom: 6px; }
        """)
        self.btn_stop_c.clicked.connect(self.stop_navigation)

        act_c.addWidget(self.btn_start_c)
        act_c.addWidget(self.btn_stop_c)
        layout.addLayout(act_c)

        return box

    def build_error_calculator_box(self) -> QGroupBox:
        box = QGroupBox("Dynamic Error Calculator && Performance Indicators")
        layout = QVBoxLayout(box)

        grid = QGridLayout()
        grid.addWidget(QLabel("Distance Error (ρ):"), 0, 0)
        self.lbl_err_rho = QLabel("0.000 m")
        self.lbl_err_rho.setStyleSheet("color: #ff5252; font-family: monospace; font-weight: bold; font-size: 13px;")
        grid.addWidget(self.lbl_err_rho, 0, 1)

        grid.addWidget(QLabel("Heading Error (α):"), 1, 0)
        self.lbl_err_alpha = QLabel("0.0 °")
        self.lbl_err_alpha.setStyleSheet("color: #ffab40; font-family: monospace; font-weight: bold;")
        grid.addWidget(self.lbl_err_alpha, 1, 1)

        grid.addWidget(QLabel("Steady-State Error (est):"), 2, 0)
        self.lbl_err_est = QLabel("-- m")
        self.lbl_err_est.setStyleSheet("color: #00e676; font-family: monospace; font-weight: bold;")
        grid.addWidget(self.lbl_err_est, 2, 1)

        grid.addWidget(QLabel("Rise Time (tr, 10%-90%):"), 3, 0)
        self.lbl_tr = QLabel("-- s")
        self.lbl_tr.setStyleSheet("color: #40c4ff; font-family: monospace; font-weight: bold;")
        grid.addWidget(self.lbl_tr, 3, 1)

        grid.addWidget(QLabel("Settling Time (ts, 2%):"), 4, 0)
        self.lbl_ts = QLabel("-- s")
        self.lbl_ts.setStyleSheet("color: #e040fb; font-family: monospace; font-weight: bold;")
        grid.addWidget(self.lbl_ts, 4, 1)

        grid.addWidget(QLabel("Maximum Overshoot (σ%):"), 5, 0)
        self.lbl_overshoot = QLabel("-- %")
        self.lbl_overshoot.setStyleSheet("color: #ffd740; font-family: monospace; font-weight: bold;")
        grid.addWidget(self.lbl_overshoot, 5, 1)

        layout.addLayout(grid)
        return box

    def build_brain_explainer_box(self) -> QGroupBox:
        box = QGroupBox("Inside the Robot's Brain: 3-Step Closed-Loop Transformation")
        layout = QHBoxLayout(box)
        layout.setSpacing(8)

        # Step 1 Card: Perception
        card1 = QFrame()
        card1.setStyleSheet("background-color: #202633; border: 1px solid #2d384d; border-radius: 6px; padding: 6px;")
        l1 = QVBoxLayout(card1)
        h1 = QLabel("<b>1. PERCEPTION</b>")
        h1.setStyleSheet("color: #4da3ff;")
        l1.addWidget(h1)
        txt1 = QLabel("Wheel encoders measure ticks at 50 Hz. Dead-reckoning updates robot state:")
        txt1.setWordWrap(True)
        txt1.setStyleSheet("font-size: 10px; color: #a4b3c6;")
        l1.addWidget(txt1)
        self.lbl_brain_state = QLabel("X: 0.00 m\nY: 0.00 m\nθ: 0.0 °\nv: 0.00 m/s")
        self.lbl_brain_state.setStyleSheet("color: #00ffcc; font-family: monospace; font-size: 11px;")
        l1.addWidget(self.lbl_brain_state)
        layout.addWidget(card1)

        # Step 2 Card: Error Calculation
        card2 = QFrame()
        card2.setStyleSheet("background-color: #202633; border: 1px solid #2d384d; border-radius: 6px; padding: 6px;")
        l2 = QVBoxLayout(card2)
        h2 = QLabel("<b>2. ERROR DYNAMICS</b>")
        h2.setStyleSheet("color: #ff9100;")
        l2.addWidget(h2)
        txt2 = QLabel("Transforms Cartesian error into polar/robot-centric coordinates:")
        txt2.setWordWrap(True)
        txt2.setStyleSheet("font-size: 10px; color: #a4b3c6;")
        l2.addWidget(txt2)
        self.lbl_brain_err = QLabel("ρ = √(Δx²+Δy²)\nα = atan2(Δy,Δx) - θ\nβ = -θ - α + θ_goal")
        self.lbl_brain_err.setStyleSheet("color: #ffb74d; font-family: monospace; font-size: 11px;")
        l2.addWidget(self.lbl_brain_err)
        layout.addWidget(card2)

        # Step 3 Card: Cascade Control Law
        card3 = QFrame()
        card3.setStyleSheet("background-color: #202633; border: 1px solid #2d384d; border-radius: 6px; padding: 6px;")
        l3 = QVBoxLayout(card3)
        h3 = QLabel("<b>3. CASCADE CONTROL</b>")
        h3.setStyleSheet("color: #00e676;")
        l3.addWidget(h3)
        txt3 = QLabel("Lyapunov controller generates chassis speeds and wheel actuator rates:")
        txt3.setWordWrap(True)
        txt3.setStyleSheet("font-size: 10px; color: #a4b3c6;")
        l3.addWidget(txt3)
        self.lbl_brain_cmd = QLabel("v* = Kρ · ρ · cos(α)\nω* = Kα · α + Kβ · β\nω_R, ω_L = (v ± ½Lω)/r")
        self.lbl_brain_cmd.setStyleSheet("color: #69f0ae; font-family: monospace; font-size: 11px;")
        l3.addWidget(self.lbl_brain_cmd)
        layout.addWidget(card3)

        return box

    def build_plot_box(self) -> QGroupBox:
        box = QGroupBox("Real-Time Multi-Sensor Arena Map & Dynamic Step-Response Curves")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 1. Multi-Sensor & Attitude HUD Banner
        hud_bar = QHBoxLayout()
        hud_bar.setSpacing(12)

        self.lbl_attitude = QLabel("📐 Pitch: 0.0° | Roll: 0.0° [Flat Floor]")
        self.lbl_attitude.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 11px; background-color: #12161f; padding: 4px 8px; border-radius: 4px; border: 1px solid #2d3340;")
        hud_bar.addWidget(self.lbl_attitude)

        self.lbl_cliff = QLabel("🛡️ Cliff: SAFE")
        self.lbl_cliff.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px; background-color: #12161f; padding: 4px 8px; border-radius: 4px; border: 1px solid #2d3340;")
        hud_bar.addWidget(self.lbl_cliff)

        self.lbl_map_toast = QLabel("💾 Map: Awaiting Scan")
        self.lbl_map_toast.setStyleSheet("color: #ffd54f; font-weight: bold; font-size: 11px; background-color: #12161f; padding: 4px 8px; border-radius: 4px; border: 1px solid #2d3340;")
        hud_bar.addWidget(self.lbl_map_toast)

        hud_bar.addStretch()
        layout.addLayout(hud_bar)

        # 2. Interactive Mapping, Layers & Persistence Controls
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(8)

        lbl_layer = QLabel("Layer:")
        lbl_layer.setStyleSheet("color: #a4b3c6; font-size: 11px;")
        ctrl_bar.addWidget(lbl_layer)

        self.combo_map_layer = QComboBox()
        self.combo_map_layer.addItems([
            "🗺️ 2D Obstacle Map (/map)",
            "🏔️ 2.5D Slope Map (/elevation_map)",
            "🛰️ Fleet Trajectory Only"
        ])
        self.combo_map_layer.setStyleSheet("""
            QComboBox {
                background-color: #242933; border: 1px solid #4da3ff;
                border-radius: 4px; padding: 2px 8px; color: #ffffff;
                font-size: 11px; font-weight: bold;
            }
        """)
        self.combo_map_layer.currentIndexChanged.connect(self.on_map_layer_changed)
        ctrl_bar.addWidget(self.combo_map_layer)

        self.chk_sensor_rays = QCheckBox("📡 Sensor Rays")
        self.chk_sensor_rays.setChecked(True)
        self.chk_sensor_rays.setStyleSheet("color: #4da3ff; font-weight: bold; font-size: 11px;")
        self.chk_sensor_rays.toggled.connect(self.on_sensor_rays_toggled)
        ctrl_bar.addWidget(self.chk_sensor_rays)

        ctrl_bar.addSpacing(10)

        # Autonomous Exploration Toggle Button
        self.btn_auto_explore = QPushButton("🤖 Auto Explore")
        self.btn_auto_explore.setToolTip("Toggle autonomous Braitenberg wandering and arena coverage")
        self.btn_auto_explore.setStyleSheet("""
            QPushButton { background-color: #2a3a52; border: 1px solid #4da3ff; color: #80d8ff; font-size: 11px; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
            QPushButton:hover { background-color: #3b4e6d; border-color: #82b1ff; }
            QPushButton:pressed { background-color: #1a283a; border-color: #bbdefb; padding-top: 6px; padding-bottom: 2px; }
        """)
        self.btn_auto_explore.clicked.connect(self.on_toggle_auto_explore)
        ctrl_bar.addWidget(self.btn_auto_explore)

        # Map Persistence Buttons
        self.btn_save_map = QPushButton("💾 Save Map")
        self.btn_save_map.setToolTip("Persist current explored map to maps/khepera_map (.png, .pgm, .yaml)")
        self.btn_save_map.setStyleSheet("""
            QPushButton { background-color: #1b4d3e; border: 1px solid #00e676; color: #69f0ae; font-size: 11px; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
            QPushButton:hover { background-color: #246854; border-color: #b9f6ca; }
            QPushButton:pressed { background-color: #12382c; border-color: #a7f3d0; padding-top: 6px; padding-bottom: 2px; }
        """)
        self.btn_save_map.clicked.connect(self.on_save_map_clicked)
        ctrl_bar.addWidget(self.btn_save_map)

        self.btn_load_map = QPushButton("📂 Load")
        self.btn_load_map.setToolTip("Reload saved map from disk into robot memory")
        self.btn_load_map.setStyleSheet("""
            QPushButton { background-color: #2b3342; border: 1px solid #a4b3c6; color: #ffffff; font-size: 11px; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
            QPushButton:hover { background-color: #3c4659; border-color: #cfd8dc; }
            QPushButton:pressed { background-color: #1a202c; border-color: #ffffff; padding-top: 6px; padding-bottom: 2px; }
        """)
        self.btn_load_map.clicked.connect(self.on_load_map_clicked)
        ctrl_bar.addWidget(self.btn_load_map)

        self.btn_clear_map = QPushButton("🔄 Clear")
        self.btn_clear_map.setToolTip("Reset the exploration grid and memory")
        self.btn_clear_map.setStyleSheet("""
            QPushButton { background-color: #3e2723; border: 1px solid #ff9100; color: #ffab40; font-size: 11px; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
            QPushButton:hover { background-color: #5d3a34; border-color: #ffb74d; }
            QPushButton:pressed { background-color: #2b1b18; border-color: #ffe082; padding-top: 6px; padding-bottom: 2px; }
        """)
        self.btn_clear_map.clicked.connect(self.on_clear_map_clicked)
        ctrl_bar.addWidget(self.btn_clear_map)

        layout.addLayout(ctrl_bar)

        # 3. Embed Matplotlib Figure Canvas
        self.figure = Figure(figsize=(7, 4), facecolor='#16191f')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax_map = self.figure.add_subplot(1, 2, 1)
        self.ax_err = self.figure.add_subplot(1, 2, 2)
        self.figure.tight_layout(pad=2.0)

        return box

    # --------------------------------------------------------------------------
    # Mode Switching & Keyboard Handling
    # --------------------------------------------------------------------------
    def switch_mode(self, mode: str):
        self.current_mode = mode
        self.stop_navigation()

        self.mode_panel_a.setVisible(mode == 'A')
        self.mode_panel_b.setVisible(mode == 'B')
        self.mode_panel_c.setVisible(mode == 'C')

        if mode == 'A':
            # Option A: Full manual pilot control. Halt autonomous swarm behaviors
            # so the user can freely position khepera_2 or any other robot without tether fighting!
            self.ros.publish_swarm_cmd("halt")
            self.lbl_tether_status.setText("Swarm: MANUAL / WASD Mode (Full Pilot Control)")
            self.lbl_tether_status.setStyleSheet("color: #80d8ff; font-size: 11px; font-weight: bold;")

        if mode == 'B':
            try:
                self.goal_x = self.spin_x.value()
                self.goal_y = self.spin_y.value()
                self.goal_yaw = math.radians(self.spin_yaw.value())
                self.waypoints = [(self.goal_x, self.goal_y, self.goal_yaw)]
                self.waypoint_idx = 0
            except Exception:
                pass
            set_gazebo_goal_marker(self.goal_x, self.goal_y)
            self.update_plots()

    def on_slider_v_changed(self, val):
        self.max_lin_speed = val / 100.0
        self.lbl_v_slider.setText(f"{self.max_lin_speed:.2f}")

    def on_slider_w_changed(self, val):
        self.max_ang_speed = val / 10.0
        self.lbl_w_slider.setText(f"{self.max_ang_speed:.1f}")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Space):
            if self.current_mode == 'A':
                # Reassure manual control: halt autonomous coordinator so robot responds directly
                self.ros.publish_swarm_cmd("halt")
            self.keys_pressed.add(key)
            self.update_key_labels()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if key in self.keys_pressed:
            self.keys_pressed.discard(key)
            self.update_key_labels()
        super().keyReleaseEvent(event)

    def update_key_labels(self):
        active_style = "border: 2px solid #00e676; border-radius: 6px; background-color: #004d40; color: #00ffcc; font-weight: bold;"
        idle_style = "border: 2px solid #3c4659; border-radius: 6px; background-color: #242933; color: #e6e8ec; font-weight: bold;"

        self.key_w_lbl.setStyleSheet(active_style if Qt.Key_W in self.keys_pressed else idle_style)
        self.key_a_lbl.setStyleSheet(active_style if Qt.Key_A in self.keys_pressed else idle_style)
        self.key_s_lbl.setStyleSheet(active_style if Qt.Key_S in self.keys_pressed else idle_style)
        self.key_d_lbl.setStyleSheet(active_style if Qt.Key_D in self.keys_pressed else idle_style)

    # --------------------------------------------------------------------------
    # Multi-Robot Fleet & Swarm Handlers
    # --------------------------------------------------------------------------
    def on_active_robot_changed(self, idx: int):
        mapping = {
            0: 'khepera_1',
            1: 'khepera_2',
            2: 'khepera_3',
            3: 'all'
        }
        self.active_robot = mapping.get(idx, 'khepera_1')
        self.history_x.clear()
        self.history_y.clear()
        self.history_rho.clear()
        self.history_time.clear()
        self.nav_start_time = None
        target_name = 'khepera_1' if self.active_robot == 'all' else self.active_robot
        if target_name in self.fleet:
            r = self.fleet[target_name]
            self.on_odom(r['x'], r['y'], r['yaw'], r['v'], r['w'])

    def on_multi_odom(self, name: str, x: float, y: float, yaw: float, v: float, w: float):
        if name in self.fleet:
            spawn = self.fleet[name].get('spawn', (0.0, 0.0, 0.0))
            cos_s = math.cos(spawn[2])
            sin_s = math.sin(spawn[2])
            gx = spawn[0] + (x * cos_s - y * sin_s)
            gy = spawn[1] + (x * sin_s + y * cos_s)
            gyaw = normalize_angle(spawn[2] + yaw)

            self.fleet[name]['x'] = gx
            self.fleet[name]['y'] = gy
            self.fleet[name]['yaw'] = gyaw
            self.fleet[name]['v'] = v
            self.fleet[name]['w'] = w
            self.fleet[name]['trail_x'].append(gx)
            self.fleet[name]['trail_y'].append(gy)
            if len(self.fleet[name]['trail_x']) > 400:
                self.fleet[name]['trail_x'].pop(0)
                self.fleet[name]['trail_y'].pop(0)

            # Update active robot representation ONLY if this odom is from active target robot
            target_name = 'khepera_1' if self.active_robot == 'all' else self.active_robot
            if name == target_name:
                self.on_odom(gx, gy, gyaw, v, w)

    def on_swarm_status(self, json_str: str):
        try:
            data = json.loads(json_str)
            d12 = data.get('d_base_extender', 0.0)
            d23 = data.get('d_extender_explorer', 0.0)
            conn = data.get('chain_connected', True)
            mode = data.get('mode', 'idle')
            sig = data.get('chain_signal_pct', 100.0)
            if mode == 'idle':
                status_text = f"Swarm: MANUAL / STANDBY | d(1-2): {d12:.2f}m | d(2-3): {d23:.2f}m | Link: {sig:.0f}%"
                color = "#80d8ff"
            elif mode == 'relay':
                status_text = f"Swarm: CHAIN RELAY ACTIVE | Tether: {sig:.0f}% | d(1-2): {d12:.2f}m | d(2-3): {d23:.2f}m"
                color = "#00e676" if conn else "#ff5252"
            else:
                status_text = f"Swarm: {mode.upper()} | d(1-2): {d12:.2f}m | d(2-3): {d23:.2f}m"
                color = "#00e676" if conn else "#ff5252"
            self.lbl_tether_status.setText(status_text)
            self.lbl_tether_status.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Odometry & Control Loops
    # --------------------------------------------------------------------------
    def on_odom(self, x: float, y: float, yaw: float, v: float, w: float):
        self.robot_x = x
        self.robot_y = y
        self.robot_yaw = yaw
        self.robot_v = v
        self.robot_w = w

        # Update Brain Step 1
        deg_yaw = math.degrees(yaw)
        active_role = "All (Broadcast)" if self.active_robot == 'all' else self.fleet.get(self.active_robot, {}).get('role', 'base_link')
        self.lbl_brain_state.setText(
            f"Target: {self.active_robot} [{active_role}]\n"
            f"X: {x:+.3f} m\nY: {y:+.3f} m\nθ: {deg_yaw:+.1f} °\nv: {v:+.2f} m/s"
        )

        # Record history for plotting
        self.history_x.append(x)
        self.history_y.append(y)
        if len(self.history_x) > 400:
            self.history_x.pop(0)
            self.history_y.pop(0)

    def on_control_tick(self):
        if self.current_mode == 'A':
            self.handle_mode_a_control()
        elif self.current_mode in ('B', 'C') and self.is_navigating:
            self.handle_closed_loop_step()

    def handle_mode_a_control(self):
        v = 0.0
        w = 0.0

        if Qt.Key_W in self.keys_pressed:
            v += self.max_lin_speed
        if Qt.Key_S in self.keys_pressed:
            v -= self.max_lin_speed
        if Qt.Key_A in self.keys_pressed:
            w += self.max_ang_speed
        if Qt.Key_D in self.keys_pressed:
            w -= self.max_ang_speed
        if Qt.Key_Space in self.keys_pressed:
            v = 0.0
            w = 0.0

        self.ros.publish_cmd(v, w, target_robot=self.active_robot)
        self.lbl_a_cmd.setText(f"Commanded [{self.active_robot}]: v = {v:+.2f} m/s | w = {w:+.2f} rad/s")

        # Update Brain Step 3
        w_r = (v + 0.5 * self.WHEEL_BASE * w) / self.WHEEL_RADIUS
        w_l = (v - 0.5 * self.WHEEL_BASE * w) / self.WHEEL_RADIUS
        self.lbl_brain_cmd.setText(
            f"v* = {v:+.2f} m/s\nω* = {w:+.2f} rad/s\nω_R = {w_r:+.1f} rad/s\nω_L = {w_l:+.1f} rad/s"
        )

    def adjust_setpoint(self, dx: float = 0.0, dy: float = 0.0):
        """Increment or decrement setpoint cumulatively and update fields."""
        new_x = round(self.spin_x.value() + dx, 2)
        new_y = round(self.spin_y.value() + dy, 2)
        self.spin_x.setValue(new_x)
        self.spin_y.setValue(new_y)

    def snap_setpoint_to_robot(self):
        """Snap setpoint inputs to current robot coordinates."""
        self.spin_x.setValue(round(self.robot_x, 2))
        self.spin_y.setValue(round(self.robot_y, 2))
        self.spin_yaw.setValue(round(math.degrees(self.robot_yaw), 1))

    def on_setpoint_spin_changed(self):
        """Triggered whenever user scrolls wheel, clicks arrows, types, or presses step buttons."""
        self.goal_x = self.spin_x.value()
        self.goal_y = self.spin_y.value()
        self.goal_yaw = math.radians(self.spin_yaw.value())
        self.waypoints = [(self.goal_x, self.goal_y, self.goal_yaw)]
        self.waypoint_idx = 0
        set_gazebo_goal_marker(self.goal_x, self.goal_y)
        self.update_plots()

    def start_mode_b_nav(self):
        self.goal_x = self.spin_x.value()
        self.goal_y = self.spin_y.value()
        self.goal_yaw = math.radians(self.spin_yaw.value())

        self.waypoints = [(self.goal_x, self.goal_y, self.goal_yaw)]
        self.waypoint_idx = 0
        self.start_nav_pos = (self.robot_x, self.robot_y)
        self.initial_target_dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        self.history_time.clear()
        self.history_rho.clear()
        self.nav_start_time = None
        self.is_navigating = True
        set_gazebo_goal_marker(self.goal_x, self.goal_y)
        self.update_plots()

    def setup_square_trajectory(self):
        cx, cy = self.robot_x, self.robot_y
        self.waypoints = [
            (cx + 0.8, cy + 0.0, 0.0),
            (cx + 0.8, cy + 0.8, math.pi / 2.0),
            (cx + 0.0, cy + 0.8, math.pi),
            (cx + 0.0, cy + 0.0, -math.pi / 2.0)
        ]
        self.waypoint_idx = 0
        self.lbl_traj_status.setText("Square Trajectory Ready (4 Waypoints, 0.8m x 0.8m)")
        if self.waypoints:
            set_gazebo_goal_marker(self.waypoints[0][0], self.waypoints[0][1])

    def setup_circle_trajectory(self):
        R = 0.4
        cx = self.robot_x
        cy = self.robot_y + R
        self.waypoints = []
        N = 16
        for i in range(N):
            theta = -math.pi / 2.0 + (2.0 * math.pi * (i + 1) / N)
            wx = cx + R * math.cos(theta)
            wy = cy + R * math.sin(theta)
            wyaw = normalize_angle(theta + math.pi / 2.0)
            self.waypoints.append((wx, wy, wyaw))
        self.waypoint_idx = 0
        self.lbl_traj_status.setText("Circular Orbit Ready (16 Waypoints, R = 0.4m)")
        if self.waypoints:
            set_gazebo_goal_marker(self.waypoints[0][0], self.waypoints[0][1])

    def start_mode_c_nav(self):
        if not self.waypoints:
            self.setup_square_trajectory()
        self.waypoint_idx = 0
        self.start_nav_pos = (self.robot_x, self.robot_y)
        self.initial_target_dist = math.hypot(self.waypoints[0][0] - self.robot_x, self.waypoints[0][1] - self.robot_y)
        self.history_time.clear()
        self.history_rho.clear()
        self.nav_start_time = None
        self.is_navigating = True
        set_gazebo_goal_marker(self.waypoints[0][0], self.waypoints[0][1])

    def stop_navigation(self):
        self.is_navigating = False
        self.ros.publish_cmd(0.0, 0.0, target_robot=self.active_robot)

    def handle_closed_loop_step(self):
        if not self.waypoints or self.waypoint_idx >= len(self.waypoints):
            self.stop_navigation()
            return

        gx, gy, gyaw = self.waypoints[self.waypoint_idx]
        dx = gx - self.robot_x
        dy = gy - self.robot_y
        rho = math.hypot(dx, dy)
        alpha = normalize_angle(math.atan2(dy, dx) - self.robot_yaw)
        beta = normalize_angle(-self.robot_yaw - alpha + gyaw)

        # Update Brain Step 2 & Dynamic Error Calculator
        self.lbl_err_rho.setText(f"{rho:.3f} m")
        self.lbl_err_alpha.setText(f"{math.degrees(alpha):.1f} °")
        self.lbl_brain_err.setText(
            f"ρ = {rho:.3f} m\nα = {math.degrees(alpha):+.1f} °\nβ = {math.degrees(beta):+.1f} °"
        )

        # Time tracking for step response
        if self.nav_start_time is None:
            self.nav_start_time = len(self.history_time) * 0.02
        t = len(self.history_time) * 0.02
        self.history_time.append(t)
        self.history_rho.append(rho)

        # Outer Loop Controller Law
        v_max = 0.22
        w_max = 1.8

        if rho > self.pos_tol:
            if abs(alpha) > math.pi / 2.0:
                v = 0.0
                w = float(max(-w_max, min(w_max, self.k_alpha * alpha)))
            else:
                v = float(max(-v_max, min(v_max, self.k_rho * rho)))
                w = float(max(-w_max, min(w_max, self.k_alpha * alpha + self.k_beta * beta)))
        else:
            yaw_err = normalize_angle(gyaw - self.robot_yaw)
            if abs(yaw_err) > self.yaw_tol:
                v = 0.0
                w = float(max(-w_max, min(w_max, self.k_yaw * yaw_err)))
            else:
                # Target waypoint reached!
                self.waypoint_idx += 1
                if self.waypoint_idx < len(self.waypoints):
                    next_w = self.waypoints[self.waypoint_idx]
                    set_gazebo_goal_marker(next_w[0], next_w[1])
                    if self.current_mode == 'C':
                        self.lbl_traj_status.setText(f"Waypoint {self.waypoint_idx+1}/{len(self.waypoints)} Active -> ({next_w[0]:.2f}, {next_w[1]:.2f})")
                    return
                else:
                    # All waypoints reached!
                    self.stop_navigation()
                    self.compute_final_metrics()
                    if self.current_mode == 'C':
                        self.lbl_traj_status.setText("Trajectory Completed Successfully!")
                    return

        self.ros.publish_cmd(v, w, target_robot=self.active_robot)

        # Update Brain Step 3
        w_r = (v + 0.5 * self.WHEEL_BASE * w) / self.WHEEL_RADIUS
        w_l = (v - 0.5 * self.WHEEL_BASE * w) / self.WHEEL_RADIUS
        self.lbl_brain_cmd.setText(
            f"v* = {v:+.2f} m/s\nω* = {w:+.2f} rad/s\nω_R = {w_r:+.1f} rad/s\nω_L = {w_l:+.1f} rad/s"
        )

    def compute_final_metrics(self):
        """Compute standard second-order performance indicators from Charter."""
        if not self.history_rho or self.initial_target_dist <= 0.0:
            return

        target = self.initial_target_dist
        times = self.history_time
        disps = [target - r for r in self.history_rho]

        # Overshoot
        max_disp = max(disps) if disps else 0.0
        overshoot = max(0.0, (max_disp - target) / target * 100.0)

        # Steady-State Error
        e_st = abs(disps[-1] - target) if disps else 0.0

        # Rise time (10% to 90%)
        t_10 = next((t for t, s in zip(times, disps) if s >= 0.10 * target), None)
        t_90 = next((t for t, s in zip(times, disps) if s >= 0.90 * target), None)
        t_rise = (t_90 - t_10) if (t_10 is not None and t_90 is not None) else None

        # Settling time (2% band)
        band = 0.02 * target
        settled_indices = [i for i, s in enumerate(disps) if abs(s - target) > band]
        t_settling = times[settled_indices[-1]] if settled_indices else times[0]

        self.lbl_overshoot.setText(f"{overshoot:.2f} %")
        self.lbl_err_est.setText(f"{e_st:.4f} m")
        if t_rise is not None:
            self.lbl_tr.setText(f"{t_rise:.2f} s")
        self.lbl_ts.setText(f"{t_settling:.2f} s")

    # --------------------------------------------------------------------------
    # Multi-Sensor Mapping & Spatial Memory Slots
    # --------------------------------------------------------------------------
    def on_map_received(self, msg: OccupancyGrid):
        self.latest_map = msg

    def on_elevation_received(self, msg: OccupancyGrid):
        self.latest_elevation = msg

    def on_scan_received(self, msg: LaserScan):
        self.latest_scan = msg

    def on_imu_received(self, pitch: float, roll: float, az: float):
        self.imu_pitch = pitch
        self.imu_roll = roll
        self.imu_az = az
        surf_type = "Flat Floor"
        if abs(pitch) > 6.0 or abs(roll) > 6.0:
            surf_type = f"Inclined Ramp ({max(abs(pitch), abs(roll)):.1f}°)"
        elif abs(az - 9.81) > 1.2:
            surf_type = "Rough Terrain"
        self.lbl_attitude.setText(f"📐 Pitch: {pitch:+.1f}° | Roll: {roll:+.1f}° [{surf_type}]")

    def on_cliff_alert_received(self, alert: bool):
        self.is_cliff_alert = alert
        if alert:
            self.lbl_cliff.setText("🚨 CLIFF DETECTED!")
            self.lbl_cliff.setStyleSheet("color: #ffffff; background-color: #d50000; font-weight: bold; font-size: 11px; padding: 4px 8px; border-radius: 4px;")
        else:
            self.lbl_cliff.setText("🛡️ Cliff: SAFE")
            self.lbl_cliff.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px; background-color: #12161f; padding: 4px 8px; border-radius: 4px; border: 1px solid #2d3340;")

    def on_map_feedback_received(self, msg: str):
        self.map_feedback_msg = msg
        self.lbl_map_toast.setText(msg)

    def on_map_layer_changed(self, idx: int):
        layers = ['obstacles', 'elevation', 'off']
        self.map_layer_mode = layers[idx] if idx < len(layers) else 'obstacles'
        self.update_plots()

    def on_sensor_rays_toggled(self, checked: bool):
        self.show_sensor_rays = checked
        self.update_plots()

    def on_toggle_auto_explore(self):
        self.is_auto_exploring = not self.is_auto_exploring
        self.ros.set_explorer_enabled(self.is_auto_exploring)
        if self.is_auto_exploring:
            self.btn_auto_explore.setText("⏹️ Stop Explore")
            self.btn_auto_explore.setStyleSheet("""
                QPushButton { background-color: #b71c1c; border: 1px solid #ff5252; color: #ffffff; font-size: 11px; font-weight: bold; border-radius: 4px; padding: 4px 8px; }
                QPushButton:hover { background-color: #d32f2f; border-color: #ff8a80; }
                QPushButton:pressed { background-color: #7f0000; border-color: #ffcdd2; padding-top: 6px; padding-bottom: 2px; }
            """)
        else:
            self.btn_auto_explore.setText("🤖 Auto Explore")
            self.btn_auto_explore.setStyleSheet("""
                QPushButton { background-color: #2a3a52; border: 1px solid #4da3ff; color: #80d8ff; font-size: 11px; border-radius: 4px; padding: 4px 8px; font-weight: bold; }
                QPushButton:hover { background-color: #3b4e6d; border-color: #82b1ff; }
                QPushButton:pressed { background-color: #1a283a; border-color: #bbdefb; padding-top: 6px; padding-bottom: 2px; }
            """)

    def on_save_map_clicked(self):
        self.lbl_map_toast.setText("💾 Saving map files...")
        self.ros.trigger_service(self.ros.save_map_cli, "Save Map")

    def on_clear_map_clicked(self):
        self.lbl_map_toast.setText("🔄 Clearing map...")
        self.ros.trigger_service(self.ros.clear_map_cli, "Clear Map")

    def on_load_map_clicked(self):
        self.lbl_map_toast.setText("📂 Loading saved map...")
        self.ros.trigger_service(self.ros.load_map_cli, "Load Map")

    # --------------------------------------------------------------------------
    # Matplotlib Plot Refresh
    # --------------------------------------------------------------------------
    def update_plots(self):
        # 1. Arena Map (Multi-Robot Swarm View & Spatial Map)
        self.ax_map.clear()
        self.ax_map.set_facecolor('#141820')
        self.ax_map.grid(True, color='#2c3342', linestyle='--', linewidth=0.5, alpha=0.6)
        self.ax_map.set_title("Khepera IV Live Spatial Map & Sensor Suite", color='#ffffff', fontsize=10, fontweight='bold')
        self.ax_map.set_xlabel("X [m]", color='#a4b3c6', fontsize=8)
        self.ax_map.set_ylabel("Y [m]", color='#a4b3c6', fontsize=8)
        self.ax_map.tick_params(colors='#8f9ba8', labelsize=8)

        # Render 2D Occupancy Grid Layer
        if self.map_layer_mode == 'obstacles' and self.latest_map is not None:
            try:
                info = self.latest_map.info
                w, h = info.width, info.height
                res = info.resolution
                ox = info.origin.position.x
                oy = info.origin.position.y
                data = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))
                rgba = np.zeros((h, w, 4), dtype=np.float32)
                # Free space (0): clean dark arena floor
                free_mask = (data == 0)
                rgba[free_mask] = [0.18, 0.22, 0.30, 0.75]
                # Occupied space (100): bright cyan obstacle walls
                occ_mask = (data == 100)
                rgba[occ_mask] = [0.0, 0.95, 0.85, 0.95]
                extent = [ox, ox + w * res, oy, oy + h * res]
                self.ax_map.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=1)
            except Exception:
                pass

        # Render 2.5D Elevation / Slope Layer
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
                rgba[visited, 0] = slopes
                rgba[visited, 1] = 1.0 - slopes * 0.7
                rgba[visited, 2] = 0.2
                rgba[visited, 3] = 0.80
                extent = [ox, ox + w * res, oy, oy + h * res]
                self.ax_map.imshow(rgba, origin='lower', extent=extent, interpolation='nearest', zorder=1)
            except Exception:
                pass

        # Draw Sensor Scan Rays radiating from Active Robot
        if self.show_sensor_rays and self.latest_scan is not None:
            try:
                active_r = self.fleet.get(self.active_robot, self.fleet['khepera_1'])
                ar_x, ar_y, ar_yaw = active_r['x'], active_r['y'], active_r['yaw']
                ranges = self.latest_scan.ranges
                num_rays = len(ranges)
                angle_min = self.latest_scan.angle_min
                angle_inc = self.latest_scan.angle_increment
                step = max(1, num_rays // 36)
                for i in range(0, num_rays, step):
                    r = ranges[i]
                    if math.isnan(r) or r < 0.001:
                        continue
                    r_clamped = min(r, 2.0)
                    beam_angle = ar_yaw + angle_min + i * angle_inc
                    hit_x = ar_x + r_clamped * math.cos(beam_angle)
                    hit_y = ar_y + r_clamped * math.sin(beam_angle)
                    color = '#ff9100' if r < 0.25 else '#00e5ff'
                    self.ax_map.plot([ar_x, hit_x], [ar_y, hit_y], color=color, linewidth=0.8, alpha=0.45, zorder=2)
                    if r < 1.95:
                        self.ax_map.plot(hit_x, hit_y, '.', color=color, markersize=3, alpha=0.8, zorder=2)
            except Exception:
                pass

        # Draw Communication Tether Lines between robots
        r1 = self.fleet['khepera_1']
        r2 = self.fleet['khepera_2']
        r3 = self.fleet['khepera_3']

        # Draw Extender RF Relay Safe Coverage Zone (1.0m radius comfort zone)
        rf_comfort = plt.Circle((r2['x'], r2['y']), 1.0, color='#ff9100', fill=True, alpha=0.06, linestyle='--', linewidth=0.8)
        self.ax_map.add_patch(rf_comfort)

        # Tether: base_link <-> extender
        self.ax_map.plot([r1['x'], r2['x']], [r1['y'], r2['y']], color='#4da3ff', linestyle=':', linewidth=1.2, alpha=0.7)
        mid_12_x = (r1['x'] + r2['x']) / 2.0
        mid_12_y = (r1['y'] + r2['y']) / 2.0
        d_12 = math.hypot(r2['x'] - r1['x'], r2['y'] - r1['y'])
        self.ax_map.text(mid_12_x, mid_12_y + 0.04, f"{d_12:.2f}m", color='#4da3ff', fontsize=7, alpha=0.9, ha='center')

        # Tether: extender <-> explorer
        self.ax_map.plot([r2['x'], r3['x']], [r2['y'], r3['y']], color='#ffd54f', linestyle=':', linewidth=1.2, alpha=0.7)
        mid_23_x = (r2['x'] + r3['x']) / 2.0
        mid_23_y = (r2['y'] + r3['y']) / 2.0
        d_23 = math.hypot(r3['x'] - r2['x'], r3['y'] - r2['y'])
        self.ax_map.text(mid_23_x, mid_23_y + 0.04, f"{d_23:.2f}m", color='#ffd54f', fontsize=7, alpha=0.9, ha='center')

        # Draw Each Robot in the Fleet
        for name, rdata in self.fleet.items():
            rx, ry, ryaw = rdata['x'], rdata['y'], rdata['yaw']
            color = rdata['color']
            role = rdata['role']

            # Historical trail
            if len(rdata['trail_x']) > 1:
                self.ax_map.plot(rdata['trail_x'], rdata['trail_y'], color=color, linewidth=1.2, alpha=0.5)

            # Robot body marker
            self.ax_map.plot(rx, ry, 'o', color=color, markersize=8, label=f"{name} [{role}]")

            # Heading arrow
            arrow_len = 0.12
            adx = arrow_len * math.cos(ryaw)
            ady = arrow_len * math.sin(ryaw)
            self.ax_map.arrow(rx, ry, adx, ady, head_width=0.035, head_length=0.035, fc=color, ec=color)

            # Robot label
            self.ax_map.text(rx + 0.04, ry + 0.04, f"{name}\n[{role}]", color=color, fontsize=7, fontweight='bold')

            # Highlight selected target robot with outer ring
            if name == self.active_robot or self.active_robot == 'all':
                sel_ring = plt.Circle((rx, ry), 0.10, color='#ffffff', fill=False, linestyle='-', linewidth=1.4, alpha=0.9)
                self.ax_map.add_patch(sel_ring)

        # Draw visual goal beacon
        if self.current_mode in ('B', 'C') and self.waypoints:
            target_pt = self.waypoints[self.waypoint_idx] if self.waypoint_idx < len(self.waypoints) else self.waypoints[-1]
            self.ax_map.plot(target_pt[0], target_pt[1], '*', color='#ffd700', markersize=14, label='Gazebo Beacon')
            circle = plt.Circle((target_pt[0], target_pt[1]), 0.08, color='#00e676', fill=False, linestyle='--', linewidth=1.2)
            self.ax_map.add_patch(circle)

            # Draw planned path for mode C
            if self.current_mode == 'C':
                w_xs = [w[0] for w in self.waypoints]
                w_ys = [w[1] for w in self.waypoints]
                self.ax_map.plot(w_xs, w_ys, 'r--', alpha=0.5, label='Path')

        # Symmetric Centered Scaling for accurate 1:1 metric visualization
        xs = [r['x'] for r in self.fleet.values()]
        ys = [r['y'] for r in self.fleet.values()]
        if self.current_mode in ('B', 'C') and self.waypoints:
            target_pt = self.waypoints[self.waypoint_idx] if self.waypoint_idx < len(self.waypoints) else self.waypoints[-1]
            xs.append(target_pt[0])
            ys.append(target_pt[1])
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        max_dist = max(max(abs(x - cx) for x in xs), max(abs(y - cy) for y in ys), 0.70) + 0.35
        self.ax_map.set_xlim(cx - max_dist, cx + max_dist)
        self.ax_map.set_ylim(cy - max_dist, cy + max_dist)
        self.ax_map.set_aspect('equal', adjustable='box')
        self.ax_map.legend(loc='upper right', facecolor='#16191f', edgecolor='#2d3340', labelcolor='#ffffff', fontsize=7)

        # 2. Dynamic Error Curve ρ(t)
        self.ax_err.clear()
        self.ax_err.set_facecolor('#1a1e27')
        self.ax_err.grid(True, color='#2c3342', linestyle='--', linewidth=0.5)
        self.ax_err.set_title("Distance Error Curve: e(t) = r(t) - y(t)", color='#ffffff', fontsize=10, fontweight='bold')
        self.ax_err.set_xlabel("Time [s]", color='#a4b3c6', fontsize=8)
        self.ax_err.set_ylabel("Distance to Goal ρ [m]", color='#a4b3c6', fontsize=8)
        self.ax_err.tick_params(colors='#8f9ba8', labelsize=8)

        if len(self.history_time) > 1:
            self.ax_err.plot(self.history_time, self.history_rho, color='#ff5252', linewidth=1.8, label='Error ρ(t)')
            self.ax_err.axhline(self.pos_tol, color='#00e676', linestyle=':', label='2cm Tolerance')
            self.ax_err.legend(loc='upper right', facecolor='#16191f', edgecolor='#2d3340', labelcolor='#ffffff', fontsize=7)
        else:
            self.ax_err.set_xlim(0, 10)
            self.ax_err.set_ylim(0, 1.2)
            self.ax_err.text(5, 0.6, "Awaiting Setpoint Execution...\ne(t) dynamic curve will plot here",
                             ha='center', va='center', color='#8f9ba8', fontsize=9)

        self.canvas.draw_idle()

    def closeEvent(self, event):
        self.ros.stop()
        super().closeEvent(event)


# ==============================================================================
# Entry Point
# ==============================================================================
def main():
    app = QApplication(sys.argv)

    # Initialize ROS 2 in background thread
    ros_worker = RosWorker()
    ros_worker.init_ros()

    ros_thread = threading.Thread(target=ros_worker.spin, daemon=True)
    ros_thread.start()

    window = KheperaDashboard(ros_worker)
    window.show()

    ret = app.exec_()
    ros_worker.stop()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
