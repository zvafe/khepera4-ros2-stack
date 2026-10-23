#!/usr/bin/env python3
"""
Khepera IV Control Station & Educational Dashboard
Course: Bazele Sistemelor Automate (BSA)
Academic Lead: Prof. Ioan Dumitrache

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
    QTabWidget, QProgressBar, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QKeyEvent

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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
class RosWorker(QObject):
    odom_received = pyqtSignal(float, float, float, float, float)  # x, y, yaw, v, w

    def __init__(self):
        super().__init__()
        self.node = None
        self.cmd_pub = None
        self.is_running = True

    def init_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('khepera4_dashboard_node')
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.node.create_subscription(Odometry, '/odom', self._odom_cb, 10)

    def _odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        self.odom_received.emit(pos.x, pos.y, yaw, twist.linear.x, twist.angular.z)

    def publish_cmd(self, v: float, w: float):
        if self.node and self.cmd_pub and self.node.context.ok():
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            self.cmd_pub.publish(msg)

    def spin(self):
        while self.is_running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def stop(self):
        self.is_running = False
        if self.node:
            self.publish_cmd(0.0, 0.0)


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

        # Connect ROS signal
        self.ros.odom_received.connect(self.on_odom)

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
            }
            QPushButton:hover { background-color: #3b4559; border-color: #4da3ff; }
            QPushButton:pressed { background-color: #1d6ed8; }
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

        header.addStretch()
        charter_tag = QLabel("Course: Bazele Sistemelor Automate (BSA) | Academic Lead: Prof. I. Dumitrache")
        charter_tag.setStyleSheet("color: #8f9ba8; font-size: 11px;")
        header.addWidget(charter_tag)
        main_layout.addLayout(header)

        # 2. Mode Selector Bar
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
        self.btn_exec_b.setStyleSheet("background-color: #00897b; color: white;")
        self.btn_exec_b.clicked.connect(self.start_mode_b_nav)

        self.btn_stop_b = QPushButton("⏹ Stop / Abort")
        self.btn_stop_b.setStyleSheet("background-color: #d32f2f; color: white;")
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
        self.btn_start_c.setStyleSheet("background-color: #1565c0; color: white;")
        self.btn_start_c.clicked.connect(self.start_mode_c_nav)

        self.btn_stop_c = QPushButton("⏹ Stop Trajectory")
        self.btn_stop_c.setStyleSheet("background-color: #d32f2f; color: white;")
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
        box = QGroupBox("Real-Time Arena Mini-Map && Dynamic Error Curve")
        layout = QVBoxLayout(box)

        # Embed Matplotlib Figure
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
        self.lbl_brain_state.setText(
            f"X: {x:+.3f} m\nY: {y:+.3f} m\nθ: {deg_yaw:+.1f} °\nv: {v:+.2f} m/s"
        )

        # Record history for plotting
        now = self.control_timer.remainingTime()
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

        self.ros.publish_cmd(v, w)
        self.lbl_a_cmd.setText(f"Commanded: v = {v:+.2f} m/s | w = {w:+.2f} rad/s")

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
        self.ros.publish_cmd(0.0, 0.0)

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

        self.ros.publish_cmd(v, w)

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
    # Matplotlib Plot Refresh
    # --------------------------------------------------------------------------
    def update_plots(self):
        # 1. Arena Map
        self.ax_map.clear()
        self.ax_map.set_facecolor('#1a1e27')
        self.ax_map.grid(True, color='#2c3342', linestyle='--', linewidth=0.5)
        self.ax_map.set_title("2D Top-Down Arena View", color='#ffffff', fontsize=10, fontweight='bold')
        self.ax_map.set_xlabel("X [m]", color='#a4b3c6', fontsize=8)
        self.ax_map.set_ylabel("Y [m]", color='#a4b3c6', fontsize=8)
        self.ax_map.tick_params(colors='#8f9ba8', labelsize=8)

        # Draw trajectory history
        if len(self.history_x) > 1:
            self.ax_map.plot(self.history_x, self.history_y, color='#00b0ff', linewidth=1.5, label='Robot Trail')

        # Draw robot body
        self.ax_map.plot(self.robot_x, self.robot_y, 'o', color='#00e676', markersize=8, label='Robot')
        # Draw heading arrow
        arrow_len = 0.12
        dx = arrow_len * math.cos(self.robot_yaw)
        dy = arrow_len * math.sin(self.robot_yaw)
        self.ax_map.arrow(
            self.robot_x, self.robot_y, dx, dy,
            head_width=0.04, head_length=0.04, fc='#00e676', ec='#00e676'
        )

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

        self.ax_map.axis('equal')
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
