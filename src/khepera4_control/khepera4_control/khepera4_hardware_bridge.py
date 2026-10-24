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
- Joint states, battery telemetry, and proximity sensor publisher
- Automatic network reconnection & safety watchdog
"""

import socket
import select
import time
import math
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, BatteryState
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformBroadcaster
from khepera4_control.common import quaternion_from_euler, normalize_angle


class Khepera4HardwareBridge(Node):
    def __init__(self):
        super().__init__('khepera4_hardware_bridge')

        # Parameters
        self.declare_parameter('robot_name', 'khepera_1')
        self.declare_parameter('role', 'base_link')
        self.declare_parameter('robot_ip', '192.168.1.101')
        self.declare_parameter('robot_port', 5000)
        self.declare_parameter('protocol', 'kteam_ascii')  # 'kteam_ascii' or 'json_stream'
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('mock', False)              # Mock mode for testing without hardware
        self.declare_parameter('poll_rate', 25.0)          # Loop rate (Hz)

        self.robot_name = self.get_parameter('robot_name').value
        self.role = self.get_parameter('role').value
        self.robot_ip = self.get_parameter('robot_ip').value
        self.robot_port = int(self.get_parameter('robot_port').value)
        self.protocol = self.get_parameter('protocol').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.mock = bool(self.get_parameter('mock').value)
        self.poll_rate = float(self.get_parameter('poll_rate').value)

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

        # ROS 2 Interfaces (Relative topic names for clean namespacing)
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.battery_pub = self.create_publisher(BatteryState, 'battery_state', 10)
        self.ir_pub = self.create_publisher(Float32MultiArray, 'ir_sensors', 10)
        self.us_pub = self.create_publisher(Float32MultiArray, 'us_sensors', 10)
        self.status_pub = self.create_publisher(String, 'connection_status', 10)

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # Main Polling Timer
        timer_period = 1.0 / self.poll_rate
        self.timer = self.create_timer(timer_period, self.loop_step)

        self.get_logger().info(
            f"[{self.robot_name} ({self.role})] Hardware Bridge initialized. "
            f"Target: {self.robot_ip}:{self.robot_port} (Protocol: {self.protocol}, Mock: {self.mock})"
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

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.5)
            self.sock.connect((self.robot_ip, self.robot_port))
            self.sock.setblocking(False)
            self.is_connected = True
            self.get_logger().info(f"[{self.robot_name}] Successfully connected to {self.robot_ip}:{self.robot_port}")
            status_msg = String()
            status_msg.data = f"CONNECTED to {self.robot_ip}:{self.robot_port}"
            self.status_pub.publish(status_msg)
            return True
        except Exception as e:
            self.is_connected = False
            self.sock = None
            status_msg = String()
            status_msg.data = f"DISCONNECTED: {str(e)}"
            self.status_pub.publish(status_msg)
            return False

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
            # Attempt reconnect throttled
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
        # 1. Send Set Speed: "D,pulses_left,pulses_right\r\n"
        speed_cmd = f"D,{pulses_l},{pulses_r}\r\n".encode('ascii')
        self.sock.sendall(speed_cmd)

        # 2. Query Encoders: "R\r\n" -> "r,left,right\r\n"
        self.sock.sendall(b"R\r\n")
        time.sleep(0.005)
        enc_data = self._read_until_newline()

        if enc_data and enc_data.startswith("r,"):
            parts = enc_data.strip().split(',')
            if len(parts) >= 3:
                enc_l = int(parts[1])
                enc_r = int(parts[2])
                self._update_odometry(enc_l, enc_r, dt)

        # 3. Query Sensors periodically (IR, US, Battery)
        if not hasattr(self, '_poll_counter'):
            self._poll_counter = 0
        self._poll_counter += 1

        # Every 5 loops (~5 Hz): Query Battery & US
        if self._poll_counter % 5 == 0:
            self.sock.sendall(b"V,0\r\n")
            time.sleep(0.003)
            v_data = self._read_until_newline()
            if v_data and v_data.startswith("v,"):
                try:
                    mv = float(v_data.strip().split(',')[1])
                    self._publish_battery(mv / 1000.0)
                except Exception:
                    pass

            self.sock.sendall(b"G\r\n")
            time.sleep(0.003)
            g_data = self._read_until_newline()
            if g_data and g_data.startswith("g,"):
                try:
                    us_vals = [float(x) for x in g_data.strip().split(',')[1:]]
                    self._publish_us(us_vals)
                except Exception:
                    pass

        # Every 2 loops (~12.5 Hz): Query IR Proximity
        if self._poll_counter % 2 == 0:
            self.sock.sendall(b"N\r\n")
            time.sleep(0.003)
            n_data = self._read_until_newline()
            if n_data and n_data.startswith("n,"):
                try:
                    ir_vals = [float(x) for x in n_data.strip().split(',')[1:]]
                    self._publish_ir(ir_vals)
                except Exception:
                    pass

    def _read_until_newline(self, timeout=0.05) -> str:
        """Read ASCII line from non-blocking socket."""
        ready = select.select([self.sock], [], [], timeout)
        if not ready[0]:
            return ""
        buf = ""
        while True:
            try:
                char = self.sock.recv(1).decode('ascii', errors='ignore')
                if not char:
                    break
                buf += char
                if char == '\n':
                    break
            except Exception:
                break
        return buf

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
        self._publish_battery(7.4 - 0.0001 * time.time() % 0.8)

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
        odom.header.frame_id = f"{self.robot_name}/odom"
        odom.child_frame_id = f"{self.robot_name}/base_footprint"

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

        # Broadcast TF: <robot_name>/odom -> <robot_name>/base_footprint
        if self.publish_tf and self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = now_msg
            t.header.frame_id = f"{self.robot_name}/odom"
            t.child_frame_id = f"{self.robot_name}/base_footprint"
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

    def _publish_battery(self, voltage: float):
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage = float(voltage)
        # 2-cell LiPo: 8.4V = 100%, 6.6V = 0%
        msg.percentage = float(max(0.0, min(1.0, (voltage - 6.6) / (8.4 - 6.6))))
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        self.battery_pub.publish(msg)

    def _publish_ir(self, ir_vals: list):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in ir_vals]
        self.ir_pub.publish(msg)

    def _publish_us(self, us_vals: list):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in us_vals]
        self.us_pub.publish(msg)

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
