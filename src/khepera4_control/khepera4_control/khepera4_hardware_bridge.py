#!/usr/bin/env python3
"""
Physical Robot Hardware Bridge Node for K-Team Khepera IV Mobile Robot.
Connects the Master PC (ROS 2 Jazzy) to physical Khepera IV embedded computer (Yocto Linux)
via TCP/IP Wi-Fi socket communication.

Implements:
- K-Team ASCII Protocol (Commands D, R, N, G, V from User Manual Section 6)
- High-rate JSON streaming protocol
- Non-holonomic unicycle forward/inverse kinematic conversions (Manual Section 4.14)
- Wheel odometry integration & TF broadcasting
- Complete sensor publishing suite:
  * Raw multiarrays (/sensors/infrared, /sensors/ultrasonic, ir_sensors, us_sensors)
  * Individual LaserScans (/khepera4/ir_0..7/scan, /khepera4/us_0..4/scan, /khepera4/ground_ir_0..3/scan)
  * Unified 360-degree LaserScan (/khepera4/scan, /scan)
  * Battery telemetry (/battery, /battery_state)
  * Odometry (/odom, <robot_name>/odom)
  * IMU data (/imu/data, /khepera4/imu/data)
- Automatic network reconnection & safety watchdog
"""

import collections
import json
import math
import select
import socket
import time
from typing import List, Optional

from geometry_msgs.msg import TransformStamped, Twist
from khepera4_control.common import normalize_angle, quaternion_from_euler
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Imu, JointState, LaserScan
from std_msgs.msg import Float32MultiArray, String
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


def raw_ir_to_distance(raw_val: float) -> float:
    """
    Convert Khepera IV 10-bit IR proximity reading (0-1023) to metric distance (meters).
    0-30: No obstacle detected (ambient baseline).
    950-1023: Close physical bumper contact (~0.002m).
    """
    if raw_val <= 30.0:
        return float('inf')
    if raw_val >= 950.0:
        return 0.002
    norm = max(0.0, min(1.0, (float(raw_val) - 30.0) / 920.0))
    dist = 0.002 + 0.248 * (1.0 - math.sqrt(norm))
    return float(max(0.002, min(0.25, dist)))


def raw_ground_ir_to_distance(raw_val: float) -> float:
    """
    Convert Khepera IV ground IR reading to metric distance (meters).
    Normal level floor: high reflection (raw ~ 600-900 -> 0.005m - 0.010m).
    Cliff / drop-off: low reflection (raw < 100 -> > 0.025m cliff threshold).
    """
    if raw_val <= 60.0:
        return 0.050  # Over 0.025m threshold -> cliff detected
    norm = max(0.0, min(1.0, (float(raw_val) - 60.0) / 850.0))
    dist = 0.005 + 0.035 * (1.0 - norm)
    return float(max(0.002, min(0.05, dist)))


def raw_us_to_distance(raw_cm: float) -> float:
    """
    Convert Khepera IV ultrasonic sensor reading (cm) to metric distance (meters).
    1000 = out of range / nothing detected.
    0 = obstacle closer than 25 cm.
    25-250 = distance in cm (+/- 2cm).
    """
    if raw_cm >= 1000.0 or raw_cm < 0.0:
        return float('inf')
    if raw_cm == 0.0:
        return 0.20
    dist_m = float(raw_cm) / 100.0
    return float(max(0.02, min(2.0, dist_m)))


class Khepera4HardwareBridge(Node):
    def __init__(self):
        super().__init__('khepera4_hardware_bridge')

        # Parameters
        self.declare_parameter('robot_name', 'khepera_1')
        self.declare_parameter('robot_id', '')
        self.declare_parameter('role', 'base_link')
        self.declare_parameter('robot_ip', '192.168.1.101')
        self.declare_parameter('robot_port', 5000)
        self.declare_parameter('protocol', 'kteam_ascii')  # 'kteam_ascii' or 'json_stream'
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('mock', False)              # Mock mode for testing without hardware
        self.declare_parameter('poll_rate', 25.0)          # Loop rate (Hz)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.declare_parameter('single_robot', True)
        self.declare_parameter('odom_frame', '')
        self.declare_parameter('base_frame', '')
        self.declare_parameter('publish_scans', True)

        self.robot_name = str(self.get_parameter('robot_name').value)
        robot_id_val = str(self.get_parameter('robot_id').value).strip()
        if robot_id_val:
            self.robot_name = f"khepera_{robot_id_val}" if robot_id_val.isdigit() else robot_id_val

        self.role = str(self.get_parameter('role').value)
        self.robot_ip = str(self.get_parameter('robot_ip').value)
        self.robot_port = int(self.get_parameter('robot_port').value)
        self.protocol = str(self.get_parameter('protocol').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.mock = bool(self.get_parameter('mock').value)
        self.poll_rate = float(self.get_parameter('poll_rate').value)
        self.single_robot = bool(self.get_parameter('single_robot').value)
        self.publish_scans = bool(self.get_parameter('publish_scans').value)

        # Coordinate Frame Resolution
        odom_frame_val = str(self.get_parameter('odom_frame').value).strip()
        base_frame_val = str(self.get_parameter('base_frame').value).strip()

        if odom_frame_val:
            self.odom_frame = odom_frame_val
        elif self.single_robot:
            self.odom_frame = 'odom'
        else:
            self.odom_frame = f"{self.robot_name}/odom"

        if base_frame_val:
            self.base_frame = base_frame_val
        elif self.single_robot:
            self.base_frame = 'base_footprint'
        else:
            self.base_frame = f"{self.robot_name}/base_footprint"

        # Khepera IV Physical Constants (User Manual Section 4.14)
        self.WHEEL_RADIUS = 0.021             # 21 mm
        self.WHEEL_BASE = 0.1054              # 105.4 mm
        self.PULSES_PER_REV = 19456.0         # 128 * 38 * 4
        self.METERS_PER_PULSE = 6.78181e-6    # (42 * pi / 19456) mm = 0.00678181 mm
        self.PULSES_PER_METER = 147453.0      # pulses per meter
        self.SPEED_CONV = 1474.53             # 1 m/s = 1474.53 pulses / 10ms
        self.MAX_SPEED_PULSES = 1200          # Max speed order per manual

        # State Variables
        self.sock = None
        self._rx_buffer = ""
        self.is_connected = False
        self.last_cmd_time = time.time()
        self.target_v = 0.0
        self.target_w = 0.0
        self.last_poll_time = time.time()

        # Odometry Integration State
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_enc_l = 0
        self.last_enc_r = 0
        self.has_first_enc = False
        self.wheel_l_pos = 0.0
        self.wheel_r_pos = 0.0

        # Cached distances for 360-degree virtual scan fusion
        self.latest_ir_dist = [float('inf')] * 8
        self.latest_us_dist = [float('inf')] * 5
        self.latest_ground_dist = [0.005] * 4

        # ROS 2 Interfaces (Relative and Root topic names for complete compatibility)
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.root_cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        if self.robot_name and self.robot_name not in ['cmd_vel', 'odom']:
            self.named_cmd_sub = self.create_subscription(
                Twist, f'/{self.robot_name}/cmd_vel', self.cmd_callback, 10
            )

        # Odometry Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.root_odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.named_odom_pub = self.create_publisher(Odometry, f'/{self.robot_name}/odom', 10)

        # Joint States
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.root_joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Battery Publishers
        self.battery_pub = self.create_publisher(BatteryState, 'battery_state', 10)
        self.root_battery_pub = self.create_publisher(BatteryState, '/battery', 10)
        self.root_battery_state_pub = self.create_publisher(BatteryState, '/battery_state', 10)
        self.named_battery_pub = self.create_publisher(
            BatteryState, f'/{self.robot_name}/battery_state', 10
        )

        # Proximity IR & US Array Publishers
        self.ir_pub = self.create_publisher(Float32MultiArray, 'ir_sensors', 10)
        self.root_ir_pub = self.create_publisher(Float32MultiArray, '/sensors/infrared', 10)
        self.named_ir_pub = self.create_publisher(
            Float32MultiArray, f'/{self.robot_name}/sensors/infrared', 10
        )
        self.named_ir_sensors_pub = self.create_publisher(
            Float32MultiArray, f'/{self.robot_name}/ir_sensors', 10
        )

        self.us_pub = self.create_publisher(Float32MultiArray, 'us_sensors', 10)
        self.root_us_pub = self.create_publisher(Float32MultiArray, '/sensors/ultrasonic', 10)
        self.named_us_pub = self.create_publisher(
            Float32MultiArray, f'/{self.robot_name}/us_sensors', 10
        )

        # Individual LaserScans for khepera4_mapper
        self.ir_scan_pubs = [
            self.create_publisher(LaserScan, f'/khepera4/ir_{i}/scan', 10)
            for i in range(8)
        ]
        self.us_scan_pubs = [
            self.create_publisher(LaserScan, f'/khepera4/us_{i}/scan', 10)
            for i in range(5)
        ]
        self.ground_ir_scan_pubs = [
            self.create_publisher(LaserScan, f'/khepera4/ground_ir_{i}/scan', 10)
            for i in range(4)
        ]

        # Unified 360-degree Virtual LaserScan
        self.scan_pub = self.create_publisher(LaserScan, '/khepera4/scan', 10)
        self.root_scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.named_scan_pub = self.create_publisher(LaserScan, f'/{self.robot_name}/scan', 10)

        # Synthetic/Measured IMU Telemetry
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.khepera_imu_pub = self.create_publisher(Imu, '/khepera4/imu/data', 10)

        # Status
        self.status_pub = self.create_publisher(String, 'connection_status', 10)
        self.root_status_pub = self.create_publisher(String, '/connection_status', 10)

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # Main Polling Timer
        timer_period = 1.0 / self.poll_rate
        self.timer = self.create_timer(timer_period, self.loop_step)

        self.get_logger().info(
            f"[{self.robot_name} ({self.role})] Hardware Bridge initialized. "
            f"Target: {self.robot_ip}:{self.robot_port} (Frames: {self.odom_frame} -> {self.base_frame}, "
            f"SingleRobot: {self.single_robot}, Mock: {self.mock})"
        )

    def cmd_callback(self, msg: Twist):
        """Receive Twist command from ROS 2 controller/teleop."""
        self.target_v = msg.linear.x
        self.target_w = msg.angular.z
        self.last_cmd_time = time.time()

    def connect_socket(self):
        """Connect to physical Khepera IV TCP server with timeout."""
        if self.mock:
            self.is_connected = True
            return True

        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self._rx_buffer = ""

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.5)
            self.sock.connect((self.robot_ip, self.robot_port))
            # Keep socket non-blocking for event loop polling
            self.sock.setblocking(False)
            self.is_connected = True
            self.get_logger().info(f"[{self.robot_name}] Successfully connected to {self.robot_ip}:{self.robot_port}")
            status_msg = String()
            status_msg.data = f"CONNECTED to {self.robot_ip}:{self.robot_port}"
            self.status_pub.publish(status_msg)
            self.root_status_pub.publish(status_msg)
            return True
        except Exception as e:
            self.is_connected = False
            self.sock = None
            status_msg = String()
            status_msg.data = f"DISCONNECTED: {str(e)}"
            self.status_pub.publish(status_msg)
            self.root_status_pub.publish(status_msg)
            return False

    def _send_cmd(self, cmd: str) -> str:
        """Send an ASCII command terminated with CRLF and read the response line atomically."""
        if not self.sock or not self.is_connected:
            return ""
        try:
            payload = (cmd.strip() + "\r\n").encode('ascii')
            self.sock.sendall(payload)
            return self._read_line(timeout=0.08)
        except Exception as e:
            self.get_logger().warn(f"[{self.robot_name}] Socket communication error on '{cmd}': {e}")
            self.is_connected = False
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
            return ""

    def _read_line(self, timeout=0.08) -> str:
        """Read a complete newline-terminated line from the TCP stream using internal buffer."""
        deadline = time.time() + timeout
        while '\n' not in self._rx_buffer:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self.sock], [], [], max(0.001, remaining))
            if not ready:
                break
            try:
                chunk = self.sock.recv(1024)
                if not chunk:
                    self.get_logger().warn(f"[{self.robot_name}] TCP peer closed connection.")
                    self.is_connected = False
                    return ""
                self._rx_buffer += chunk.decode('ascii', errors='ignore')
            except (BlockingIOError, socket.timeout):
                break
            except Exception as e:
                self.get_logger().warn(f"[{self.robot_name}] Socket recv error: {e}")
                self.is_connected = False
                return ""

        if '\n' in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split('\n', 1)
            return line.strip()
        return ""

    def loop_step(self):
        """Main periodic hardware interaction loop."""
        now = time.time()
        dt = max(1e-4, now - self.last_poll_time)
        self.last_poll_time = now

        # Safety Watchdog: Zero velocity if commands stop arriving (> 0.5s)
        if (now - self.last_cmd_time) > 0.5:
            cmd_v = 0.0
            cmd_w = 0.0
        else:
            cmd_v = self.target_v
            cmd_w = self.target_w

        # Inverse Kinematics (Unicycle -> Wheel Speeds in metric and pulses)
        v_l = cmd_v - (self.WHEEL_BASE / 2.0) * cmd_w
        v_r = cmd_v + (self.WHEEL_BASE / 2.0) * cmd_w

        pulses_l = int(v_l * self.SPEED_CONV)
        pulses_r = int(v_r * self.SPEED_CONV)

        pulses_l = max(-self.MAX_SPEED_PULSES, min(self.MAX_SPEED_PULSES, pulses_l))
        pulses_r = max(-self.MAX_SPEED_PULSES, min(self.MAX_SPEED_PULSES, pulses_r))

        if self.mock:
            self._handle_mock_hardware(pulses_l, pulses_r, dt)
            return

        # Ensure active socket connection
        if not self.is_connected or self.sock is None:
            if not hasattr(self, '_last_reconnect_attempt') or (now - self._last_reconnect_attempt) > 2.0:
                self._last_reconnect_attempt = now
                self.connect_socket()
            return

        try:
            if self.protocol == 'kteam_ascii':
                self._handle_kteam_ascii(pulses_l, pulses_r, dt)
            else:
                self._handle_json_stream(cmd_v, cmd_w, dt)
        except Exception as e:
            self.get_logger().warn(f"[{self.robot_name}] Communication error: {str(e)}")
            self.is_connected = False
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

    def _handle_kteam_ascii(self, pulses_l: int, pulses_r: int, dt: float):
        """Official K-Team ASCII Protocol Implementation."""
        # 1. Send Set Speed: "D,pulses_left,pulses_right\r\n" -> Server responds "d\r\n"
        self._send_cmd(f"D,{pulses_l},{pulses_r}")

        # 2. Query Encoders: "R\r\n" -> "r,left,right\r\n"
        enc_data = self._send_cmd("R")
        if enc_data and enc_data.startswith("r,"):
            parts = enc_data.strip().split(',')
            if len(parts) >= 3:
                try:
                    enc_l = int(parts[1])
                    enc_r = int(parts[2])
                    self._update_odometry(enc_l, enc_r, dt)
                except ValueError:
                    pass

        # 3. Query Sensors periodically (IR, US, Battery)
        if not hasattr(self, '_poll_counter'):
            self._poll_counter = 0
        self._poll_counter += 1

        # Every 5 loops (~5 Hz): Query Battery & US
        if self._poll_counter % 5 == 0:
            v_data = self._send_cmd("V,0")
            if v_data and v_data.startswith("v,"):
                try:
                    mv = float(v_data.strip().split(',')[1])
                    self._publish_battery(mv / 1000.0)
                except Exception:
                    pass

            g_data = self._send_cmd("G")
            if g_data and g_data.startswith("g,"):
                try:
                    us_vals = [float(x) for x in g_data.strip().split(',')[1:]]
                    self._publish_us(us_vals)
                except Exception:
                    pass

        # Every 2 loops (~12.5 Hz): Query IR Proximity
        if self._poll_counter % 2 == 0:
            n_data = self._send_cmd("N")
            if n_data and n_data.startswith("n,"):
                try:
                    ir_vals = [float(x) for x in n_data.strip().split(',')[1:]]
                    self._publish_ir(ir_vals)
                except Exception:
                    pass

    def _handle_json_stream(self, v: float, w: float, dt: float):
        """High-Rate JSON Bidirectional Streaming Protocol."""
        req = json.dumps({'v': v, 'w': w}) + '\n'
        self.sock.sendall(req.encode('utf-8'))

        ready = select.select([self.sock], [], [], 0.05)
        if ready[0]:
            line = self.sock.recv(1024).decode('utf-8', errors='ignore')
            if line:
                data = json.loads(line.strip())
                enc_l = int(data.get('enc_l', 0))
                enc_r = int(data.get('enc_r', 0))
                self._update_odometry(enc_l, enc_r, dt)

                if 'volt' in data:
                    self._publish_battery(float(data['volt']))
                if 'ir' in data:
                    self._publish_ir(data['ir'])
                if 'us' in data:
                    self._publish_us(data['us'])

    def _handle_mock_hardware(self, pulses_l: int, pulses_r: int, dt: float):
        """Internal kinematic emulator when running in mock mode."""
        v_l = pulses_l / self.SPEED_CONV
        v_r = pulses_r / self.SPEED_CONV

        delta_l = v_l * dt
        delta_r = v_r * dt

        delta_enc_l = int(delta_l / self.METERS_PER_PULSE)
        delta_enc_r = int(delta_r / self.METERS_PER_PULSE)

        self.last_enc_l += delta_enc_l
        self.last_enc_r += delta_enc_r

        self._update_odometry(self.last_enc_l, self.last_enc_r, dt)

        if not hasattr(self, '_mock_poll_counter'):
            self._mock_poll_counter = 0
        self._mock_poll_counter += 1

        if self._mock_poll_counter % 5 == 0:
            self._publish_battery(7.4 - 0.0001 * (time.time() % 0.8))
            # 5 ultrasonic ranges in cm (clear space ahead with obstacle at 80 cm)
            self._publish_us([1000.0, 1000.0, 80.0, 1000.0, 1000.0])

        if self._mock_poll_counter % 2 == 0:
            # 12 IR values (8 ring proximity + 4 ground IR)
            self._publish_ir([20.0, 20.0, 25.0, 150.0, 120.0, 20.0, 20.0, 20.0, 750.0, 780.0, 770.0, 760.0])

    def _update_odometry(self, enc_l: int, enc_r: int, dt: float):
        """Integrate differential drive odometry from encoder ticks."""
        if not self.has_first_enc:
            self.last_enc_l = enc_l
            self.last_enc_r = enc_r
            self.has_first_enc = True
            return

        d_l_pulses = enc_l - self.last_enc_l
        d_r_pulses = enc_r - self.last_enc_r

        self.last_enc_l = enc_l
        self.last_enc_r = enc_r

        d_l = d_l_pulses * self.METERS_PER_PULSE
        d_r = d_r_pulses * self.METERS_PER_PULSE

        # Angular positions for joint states
        self.wheel_l_pos += d_l / self.WHEEL_RADIUS
        self.wheel_r_pos += d_r / self.WHEEL_RADIUS

        ds = (d_r + d_l) / 2.0
        dyaw = (d_r - d_l) / self.WHEEL_BASE

        self.odom_x += ds * math.cos(self.odom_yaw + dyaw / 2.0)
        self.odom_y += ds * math.sin(self.odom_yaw + dyaw / 2.0)
        self.odom_yaw = normalize_angle(self.odom_yaw + dyaw)

        v_lin = ds / dt
        w_ang = dyaw / dt

        # Publish nav_msgs/Odometry
        now_msg = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.odom_x
        odom.pose.pose.position.y = self.odom_y
        odom.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.odom_yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = v_lin
        odom.twist.twist.angular.z = w_ang

        self.odom_pub.publish(odom)
        self.root_odom_pub.publish(odom)
        self.named_odom_pub.publish(odom)

        # Broadcast TF: odom_frame -> base_frame
        if self.publish_tf and self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = now_msg
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.odom_x
            t.transform.translation.y = self.odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

        # Publish JointState
        js = JointState()
        js.header.stamp = now_msg
        js.name = ['left_wheel_joint', 'right_wheel_joint']
        js.position = [self.wheel_l_pos, self.wheel_r_pos]
        js.velocity = [d_l / (self.WHEEL_RADIUS * dt), d_r / (self.WHEEL_RADIUS * dt)]
        self.joint_pub.publish(js)
        self.root_joint_pub.publish(js)

        # Publish IMU telemetry
        imu = Imu()
        imu.header.stamp = now_msg
        imu.header.frame_id = self.base_frame
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        imu.angular_velocity.z = w_ang
        imu.linear_acceleration.z = 9.81  # Standard Earth gravity
        self.imu_pub.publish(imu)
        self.khepera_imu_pub.publish(imu)

    def _publish_battery(self, voltage: float):
        """Publish battery state telemetry."""
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage = float(voltage)
        # 2-cell LiPo: 8.4V = 100%, 6.6V = 0%
        msg.percentage = float(max(0.0, min(1.0, (voltage - 6.6) / (8.4 - 6.6))))
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO

        self.battery_pub.publish(msg)
        self.root_battery_pub.publish(msg)
        self.root_battery_state_pub.publish(msg)
        self.named_battery_pub.publish(msg)

    def _publish_ir(self, ir_vals: list):
        """Publish IR proximity arrays and individual LaserScans."""
        now_msg = self.get_clock().now().to_msg()

        # 1. MultiArray outputs
        msg = Float32MultiArray()
        msg.data = [float(x) for x in ir_vals]
        self.ir_pub.publish(msg)
        self.root_ir_pub.publish(msg)
        self.named_ir_pub.publish(msg)
        self.named_ir_sensors_pub.publish(msg)

        if not self.publish_scans or len(ir_vals) < 8:
            return

        # 2. Individual 8 IR Ring Proximity LaserScans
        # Map K-Team ASCII order [bl, l, fl, f, fr, r, br, b] to URDF [ir_0..ir_7]
        # ir_0: +10°  <- Front (ir[3])
        # ir_1: +45°  <- Front-Left (ir[2])
        # ir_2: +90°  <- Left (ir[1])
        # ir_3: +135° <- Back-Left (ir[0])
        # ir_4: -135° <- Back-Right (ir[6])
        # ir_5: -90°  <- Back/Right (ir[5])
        # ir_6: -45°  <- Front-Right (ir[4])
        # ir_7: -10°  <- Front-Right-Center (ir[4])
        urdf_ir_map = [3, 2, 1, 0, 6, 5, 4, 4]
        for urdf_idx, kteam_idx in enumerate(urdf_ir_map):
            raw_v = ir_vals[kteam_idx] if kteam_idx < len(ir_vals) else 0.0
            dist = raw_ir_to_distance(raw_v)
            self.latest_ir_dist[urdf_idx] = dist

            scan = LaserScan()
            scan.header.stamp = now_msg
            scan.header.frame_id = f"ir_{urdf_idx}_link"
            scan.angle_min = -0.15
            scan.angle_max = 0.15
            scan.angle_increment = 0.30
            scan.range_min = 0.002
            scan.range_max = 0.25
            scan.ranges = [dist]
            if urdf_idx < len(self.ir_scan_pubs):
                self.ir_scan_pubs[urdf_idx].publish(scan)

        # 3. Individual 4 Ground IR LaserScans (Cliff & Surface sensors)
        if len(ir_vals) >= 12:
            for g_idx in range(4):
                raw_g = ir_vals[8 + g_idx]
                dist_g = raw_ground_ir_to_distance(raw_g)
                self.latest_ground_dist[g_idx] = dist_g

                scan = LaserScan()
                scan.header.stamp = now_msg
                scan.header.frame_id = f"ground_ir_{g_idx}_link"
                scan.angle_min = -0.05
                scan.angle_max = 0.05
                scan.angle_increment = 0.10
                scan.range_min = 0.002
                scan.range_max = 0.05
                scan.ranges = [dist_g]
                if g_idx < len(self.ground_ir_scan_pubs):
                    self.ground_ir_scan_pubs[g_idx].publish(scan)

        # 4. Update Unified 360-degree Virtual LaserScan
        self._publish_virtual_scan(now_msg)

    def _publish_us(self, us_vals: list):
        """Publish ultrasonic array and individual LaserScans."""
        now_msg = self.get_clock().now().to_msg()

        # 1. MultiArray outputs
        msg = Float32MultiArray()
        msg.data = [float(x) for x in us_vals]
        self.us_pub.publish(msg)
        self.root_us_pub.publish(msg)
        self.named_us_pub.publish(msg)

        if not self.publish_scans:
            return

        # 2. Individual 5 Ultrasonic LaserScans
        for i in range(min(5, len(us_vals))):
            raw_cm = us_vals[i]
            dist_m = raw_us_to_distance(raw_cm)
            self.latest_us_dist[i] = dist_m

            scan = LaserScan()
            scan.header.stamp = now_msg
            scan.header.frame_id = f"us_{i}_link"
            scan.angle_min = -0.20
            scan.angle_max = 0.20
            scan.angle_increment = 0.40
            scan.range_min = 0.02
            scan.range_max = 2.0
            scan.ranges = [dist_m]
            if i < len(self.us_scan_pubs):
                self.us_scan_pubs[i].publish(scan)

        # 3. Update Unified 360-degree Virtual LaserScan
        self._publish_virtual_scan(now_msg)

    def _publish_virtual_scan(self, now_msg):
        """Publish 360-degree virtual LaserScan on base_footprint combining IR and US."""
        num_samples = 360
        angle_min = -math.pi
        angle_max = math.pi - (2.0 * math.pi / num_samples)
        angle_inc = 2.0 * math.pi / num_samples
        ranges = [float('inf')] * num_samples

        # 1. Fuse IR distances
        for i, mount in enumerate(IR_MOUNTS):
            dist_ir = self.latest_ir_dist[i]
            if math.isinf(dist_ir) or math.isnan(dist_ir):
                continue
            sx, sy, syaw = mount
            px = sx + dist_ir * math.cos(syaw)
            py = sy + dist_ir * math.sin(syaw)
            dist = math.hypot(px, py)
            angle = math.atan2(py, px)
            idx = int(round((angle - angle_min) / angle_inc)) % num_samples
            if dist < ranges[idx]:
                ranges[idx] = dist

        # 2. Fuse Ultrasonic distances
        for i, mount in enumerate(US_MOUNTS):
            dist_us = self.latest_us_dist[i]
            if math.isinf(dist_us) or math.isnan(dist_us):
                continue
            sx, sy, syaw = mount
            px = sx + dist_us * math.cos(syaw)
            py = sy + dist_us * math.sin(syaw)
            dist = math.hypot(px, py)
            angle = math.atan2(py, px)
            idx = int(round((angle - angle_min) / angle_inc)) % num_samples
            if dist < ranges[idx]:
                ranges[idx] = dist

        scan_msg = LaserScan()
        scan_msg.header.stamp = now_msg
        scan_msg.header.frame_id = self.base_frame
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = angle_inc
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / self.poll_rate
        scan_msg.range_min = 0.002
        scan_msg.range_max = 2.0
        scan_msg.ranges = ranges

        self.scan_pub.publish(scan_msg)
        self.root_scan_pub.publish(scan_msg)
        self.named_scan_pub.publish(scan_msg)

    def destroy_node(self):
        """Safely stop physical robot on shutdown."""
        if self.sock and self.is_connected:
            try:
                self.sock.sendall(b"D,0,0\r\n")
                time.sleep(0.05)
                self.sock.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Khepera4HardwareBridge()
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
