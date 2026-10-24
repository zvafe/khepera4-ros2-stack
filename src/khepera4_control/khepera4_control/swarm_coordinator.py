#!/usr/bin/env python3
"""
Multi-Robot Swarm Coordinator Node for Khepera IV Fleet.
Coordinates 3 robots across defined roles:
- khepera_1: base_link (Anchor / Coordinator / Reference Base)
- khepera_2: extender (RF / Wi-Fi Range Extender & Chain Relay)
- khepera_3: explorer (Frontier Scouting & Reconnaissance)

Implements:
- Dynamic Midpoint Tethering (Extender tracks midpoint between Base and Explorer)
- Multi-robot Formation Control (Triangle, Line, Column)
- RF Communication Range Monitoring & Tether Constraints
- Coordinated All-Halt and Return-to-Base behaviors
"""

import math
import time
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from khepera4_control.common import euler_from_quaternion, normalize_angle


class SwarmCoordinator(Node):
    def __init__(self):
        super().__init__('swarm_coordinator')

        # Parameters
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self.declare_parameter('max_comm_range', 2.0)    # Maximum tether distance (meters)
        self.declare_parameter('k_p_pos', 0.8)           # Relay position tracking P-gain
        self.declare_parameter('k_p_yaw', 1.5)           # Relay orientation tracking P-gain
        self.declare_parameter('v_max', 0.20)            # Maximum linear velocity (m/s)
        self.declare_parameter('w_max', 1.5)             # Maximum angular velocity (rad/s)
        self.declare_parameter('default_mode', 'idle')   # 'relay', 'triangle', 'return_to_base', 'idle'
        self.declare_parameter('khepera_1_spawn', [0.0, 0.0, 0.0])
        self.declare_parameter('khepera_2_spawn', [0.6, 0.0, 0.0])
        self.declare_parameter('khepera_3_spawn', [1.2, 0.0, 0.0])

        self.max_comm_range = float(self.get_parameter('max_comm_range').value)
        self.k_p_pos = float(self.get_parameter('k_p_pos').value)
        self.k_p_yaw = float(self.get_parameter('k_p_yaw').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.mode = self.get_parameter('default_mode').value

        s1 = self.get_parameter('khepera_1_spawn').value
        s2 = self.get_parameter('khepera_2_spawn').value
        s3 = self.get_parameter('khepera_3_spawn').value
        self.spawns = {
            'khepera_1': (float(s1[0]), float(s1[1]), float(s1[2])),
            'khepera_2': (float(s2[0]), float(s2[1]), float(s2[2])),
            'khepera_3': (float(s3[0]), float(s3[1]), float(s3[2])),
        }

        # Robot Pose States in Global Arena Frame: { 'name': {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'has_odom': False} }
        self.robots = {
            'khepera_1': {'role': 'base_link', 'x': self.spawns['khepera_1'][0], 'y': self.spawns['khepera_1'][1], 'yaw': self.spawns['khepera_1'][2], 'v': 0.0, 'w': 0.0, 'has_odom': False},
            'khepera_2': {'role': 'extender',  'x': self.spawns['khepera_2'][0], 'y': self.spawns['khepera_2'][1], 'yaw': self.spawns['khepera_2'][2], 'v': 0.0, 'w': 0.0, 'has_odom': False},
            'khepera_3': {'role': 'explorer',  'x': self.spawns['khepera_3'][0], 'y': self.spawns['khepera_3'][1], 'yaw': self.spawns['khepera_3'][2], 'v': 0.0, 'w': 0.0, 'has_odom': False},
        }

        # Subscribers for each robot's odometry
        self.odom_sub_1 = self.create_subscription(Odometry, '/khepera_1/odom', self._odom_1_cb, 10)
        self.odom_sub_2 = self.create_subscription(Odometry, '/khepera_2/odom', self._odom_2_cb, 10)
        self.odom_sub_3 = self.create_subscription(Odometry, '/khepera_3/odom', self._odom_3_cb, 10)

        # Publishers for robot velocity commands
        self.cmd_pub_1 = self.create_publisher(Twist, '/khepera_1/cmd_vel', 10)
        self.cmd_pub_2 = self.create_publisher(Twist, '/khepera_2/cmd_vel', 10)
        self.cmd_pub_3 = self.create_publisher(Twist, '/khepera_3/cmd_vel', 10)

        # Swarm command & status topics
        self.cmd_sub = self.create_subscription(String, '/swarm/cmd', self.swarm_cmd_callback, 10)
        self.status_pub = self.create_publisher(String, '/swarm/status', 10)

        # Periodic coordination loop at 20 Hz
        self.timer = self.create_timer(0.05, self.coordination_step)

        self.get_logger().info("Swarm Coordinator initialized: Managing [khepera_1/base_link, khepera_2/extender, khepera_3/explorer]")

    def _parse_odom(self, msg: Odometry, name: str):
        pos = msg.pose.pose.position
        _, _, yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        spawn = self.spawns.get(name, (0.0, 0.0, 0.0))
        cos_s = math.cos(spawn[2])
        sin_s = math.sin(spawn[2])
        # Project local odometry onto global world frame
        self.robots[name]['x'] = spawn[0] + (pos.x * cos_s - pos.y * sin_s)
        self.robots[name]['y'] = spawn[1] + (pos.x * sin_s + pos.y * cos_s)
        self.robots[name]['yaw'] = normalize_angle(spawn[2] + yaw)
        self.robots[name]['v'] = msg.twist.twist.linear.x
        self.robots[name]['w'] = msg.twist.twist.angular.z
        self.robots[name]['has_odom'] = True

    def _odom_1_cb(self, msg: Odometry):
        self._parse_odom(msg, 'khepera_1')

    def _odom_2_cb(self, msg: Odometry):
        self._parse_odom(msg, 'khepera_2')

    def _odom_3_cb(self, msg: Odometry):
        self._parse_odom(msg, 'khepera_3')

    def swarm_cmd_callback(self, msg: String):
        """Handle high-level swarm commands (e.g. 'mode:relay', 'mode:triangle', 'halt', 'return_to_base')."""
        cmd_text = msg.data.strip().lower()
        if cmd_text.startswith('mode:'):
            new_mode = cmd_text.split(':')[1].strip()
            self.mode = new_mode
            self.get_logger().info(f"Swarm mode switched to: {self.mode}")
        elif cmd_text in ('halt', 'stop'):
            self.mode = 'idle'
            self._halt_all()
            self.get_logger().info("Swarm emergency halt invoked.")
        elif cmd_text == 'return_to_base':
            self.mode = 'return_to_base'
            self.get_logger().info("Swarm command: Return to Base.")

    def _halt_all(self):
        stop_msg = Twist()
        self.cmd_pub_1.publish(stop_msg)
        self.cmd_pub_2.publish(stop_msg)
        self.cmd_pub_3.publish(stop_msg)

    def coordination_step(self):
        r1 = self.robots['khepera_1']
        r2 = self.robots['khepera_2']
        r3 = self.robots['khepera_3']

        # Compute inter-robot euclidean distances in global frame
        d_12 = math.hypot(r2['x'] - r1['x'], r2['y'] - r1['y'])
        d_23 = math.hypot(r3['x'] - r2['x'], r3['y'] - r2['y'])
        d_13 = math.hypot(r3['x'] - r1['x'], r3['y'] - r1['y'])

        # Connectivity health checks
        tether_12_ok = d_12 <= self.max_comm_range
        tether_23_ok = d_23 <= self.max_comm_range
        chain_connected = tether_12_ok and tether_23_ok

        # Execute behavioral controllers according to mode
        if self.mode == 'relay':
            # Intelligent RF Tether Relay Mode:
            # Objective: Maximize RF connectivity without unnecessary or erratic wandering.
            # If both distances shrink (Explorer is close to Base or close to Extender),
            # signal strength is high! In this zone, holding position is optimal.
            if r1['has_odom'] and r3['has_odom'] and r2['has_odom']:
                R_COMFORT = 0.95  # Safe, high-throughput RF range (meters)

                # Both links well within comfort threshold -> Hold position (no weird wandering)
                if d_13 <= R_COMFORT and d_12 <= R_COMFORT and d_23 <= R_COMFORT:
                    stop_msg = Twist()
                    self.cmd_pub_2.publish(stop_msg)
                else:
                    # Tether is stretching: target is the optimal midpoint on the Base-Explorer line
                    target_x = (r1['x'] + r3['x']) / 2.0
                    target_y = (r1['y'] + r3['y']) / 2.0
                    target_yaw = math.atan2(r3['y'] - r1['y'], r3['x'] - r1['x']) if (d_13 > 0.05) else r1['yaw']

                    dist_to_mid = math.hypot(target_x - r2['x'], target_y - r2['y'])

                    # Deadband: if Extender is already within 10cm of midpoint and both links <= 1.2m, stop
                    if dist_to_mid < 0.10 and d_12 <= 1.2 and d_23 <= 1.2:
                        stop_msg = Twist()
                        self.cmd_pub_2.publish(stop_msg)
                    else:
                        cmd_extender = self._compute_tracking_cmd(r2['x'], r2['y'], r2['yaw'], target_x, target_y, target_yaw)
                        self.cmd_pub_2.publish(cmd_extender)

        elif self.mode == 'triangle':
            # Triangle Formation with khepera_1 as apex
            if r1['has_odom'] and r2['has_odom'] and r3['has_odom']:
                d_form = 0.5
                w_form = 0.35
                base_yaw = r1['yaw']

                # Extender to left flank (-cos -sin)
                tx_2 = r1['x'] - d_form * math.cos(base_yaw) - w_form * math.sin(base_yaw)
                ty_2 = r1['y'] - d_form * math.sin(base_yaw) + w_form * math.cos(base_yaw)
                cmd_2 = self._compute_tracking_cmd(r2['x'], r2['y'], r2['yaw'], tx_2, ty_2, base_yaw)
                self.cmd_pub_2.publish(cmd_2)

                # Explorer to right flank (-cos +sin)
                tx_3 = r1['x'] - d_form * math.cos(base_yaw) + w_form * math.sin(base_yaw)
                ty_3 = r1['y'] - d_form * math.sin(base_yaw) - w_form * math.cos(base_yaw)
                cmd_3 = self._compute_tracking_cmd(r3['x'], r3['y'], r3['yaw'], tx_3, ty_3, base_yaw)
                self.cmd_pub_3.publish(cmd_3)

        elif self.mode == 'return_to_base':
            # Return both Extender and Explorer to base
            if r1['has_odom'] and r2['has_odom'] and r3['has_odom']:
                base_yaw = r1['yaw']
                # Extender returns behind base (0.35m back)
                tx_2 = r1['x'] - 0.35 * math.cos(base_yaw)
                ty_2 = r1['y'] - 0.35 * math.sin(base_yaw)
                cmd_2 = self._compute_tracking_cmd(r2['x'], r2['y'], r2['yaw'], tx_2, ty_2, base_yaw)
                self.cmd_pub_2.publish(cmd_2)

                # Explorer returns behind extender (0.70m back)
                tx_3 = r1['x'] - 0.70 * math.cos(base_yaw)
                ty_3 = r1['y'] - 0.70 * math.sin(base_yaw)
                cmd_3 = self._compute_tracking_cmd(r3['x'], r3['y'], r3['yaw'], tx_3, ty_3, base_yaw)
                self.cmd_pub_3.publish(cmd_3)

        elif self.mode in ('idle', 'halt'):
            self._halt_all()

        # Publish Swarm Status JSON with Signal Quality Percentage
        sig_12 = max(0.0, min(100.0, 100.0 * (1.0 - (d_12 / self.max_comm_range)**2)))
        sig_23 = max(0.0, min(100.0, 100.0 * (1.0 - (d_23 / self.max_comm_range)**2)))
        chain_signal = round(min(sig_12, sig_23), 1)

        status_dict = {
            'mode': self.mode,
            'chain_connected': chain_connected,
            'chain_signal_pct': chain_signal,
            'd_base_extender': round(d_12, 3),
            'd_extender_explorer': round(d_23, 3),
            'd_base_explorer': round(d_13, 3),
            'max_range': self.max_comm_range,
            'robots': {
                'khepera_1': {'role': 'base_link', 'x': round(r1['x'], 3), 'y': round(r1['y'], 3), 'yaw': round(r1['yaw'], 3)},
                'khepera_2': {'role': 'extender',  'x': round(r2['x'], 3), 'y': round(r2['y'], 3), 'yaw': round(r2['yaw'], 3)},
                'khepera_3': {'role': 'explorer',  'x': round(r3['x'], 3), 'y': round(r3['y'], 3), 'yaw': round(r3['yaw'], 3)},
            }
        }
        msg = String()
        msg.data = json.dumps(status_dict)
        self.status_pub.publish(msg)

    def _compute_tracking_cmd(self, x: float, y: float, yaw: float, tx: float, ty: float, tyaw: float) -> Twist:
        """Smooth unicycle tracking regulator with bidirectional drive to eliminate awkward spins."""
        dx = tx - x
        dy = ty - y
        rho = math.hypot(dx, dy)
        cmd = Twist()

        if rho < 0.05:
            # Within 5cm deadzone: gently align orientation
            yaw_err = normalize_angle(tyaw - yaw)
            if abs(yaw_err) > 0.15:
                cmd.angular.z = float(max(-self.w_max, min(self.w_max, 1.2 * yaw_err)))
            return cmd

        target_heading = math.atan2(dy, dx)
        alpha = normalize_angle(target_heading - yaw)

        # Allow smooth reverse driving if target is behind to avoid 180° turnaround spinning
        drive_reverse = False
        if abs(alpha) > (math.pi / 2.0):
            drive_reverse = True
            alpha = normalize_angle(alpha - math.pi if alpha > 0 else alpha + math.pi)

        # Smooth velocity curve with hyperbolic tangent
        speed = self.v_max * math.tanh(rho / 0.35) * math.cos(alpha)
        if drive_reverse:
            speed = -speed

        cmd.linear.x = float(max(-self.v_max, min(self.v_max, speed)))
        cmd.angular.z = float(max(-self.w_max, min(self.w_max, self.k_p_yaw * alpha)))

        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = SwarmCoordinator()
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
