#!/usr/bin/env python3
"""
Telemetry Logger and Step-Response Plotter for Khepera IV.
Complies with BSA Control Systems Research Charter (Section 4 & 5).
Records transient response data and outputs publication-quality figures.
"""

import os
import csv
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from khepera4_control.common import euler_from_quaternion


class TelemetryLogger(Node):
    def __init__(self):
        super().__init__('telemetry_logger')

        self.declare_parameter('output_dir', '/home/andy/Desktop/khepera4_ws/telemetry')
        self.declare_parameter('auto_save_seconds', 15.0)

        self.output_dir = self.get_parameter('output_dir').value
        os.makedirs(self.output_dir, exist_ok=True)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        self.timestamps = []
        self.x_vals = []
        self.y_vals = []
        self.yaw_vals = []
        self.v_actual = []
        self.w_actual = []
        self.v_cmd = []
        self.w_cmd = []

        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0

        self.t0 = None
        self.has_data = False

        auto_save = self.get_parameter('auto_save_seconds').value
        self.timer = self.create_timer(auto_save, self.save_telemetry)

        self.get_logger().info(f'Telemetry Logger started. Output directory: {self.output_dir}')

    def cmd_callback(self, msg: Twist):
        self.current_cmd_v = msg.linear.x
        self.current_cmd_w = msg.angular.z

    def odom_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now

        t = now - self.t0
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)

        self.timestamps.append(t)
        self.x_vals.append(pos.x)
        self.y_vals.append(pos.y)
        self.yaw_vals.append(yaw)
        self.v_actual.append(twist.linear.x)
        self.w_actual.append(twist.angular.z)
        self.v_cmd.append(self.current_cmd_v)
        self.w_cmd.append(self.current_cmd_w)
        self.has_data = True

    def save_telemetry(self):
        if not self.has_data or len(self.timestamps) < 10:
            return

        csv_path = os.path.join(self.output_dir, 'khepera4_telemetry.csv')
        plot_path = os.path.join(self.output_dir, 'khepera4_response_curves.png')

        # 1. Export CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t_sec', 'x_m', 'y_m', 'yaw_rad', 'v_actual_mps', 'w_actual_radps', 'v_cmd_mps', 'w_cmd_radps'])
            for i in range(len(self.timestamps)):
                writer.writerow([
                    f"{self.timestamps[i]:.4f}",
                    f"{self.x_vals[i]:.4f}",
                    f"{self.y_vals[i]:.4f}",
                    f"{self.yaw_vals[i]:.4f}",
                    f"{self.v_actual[i]:.4f}",
                    f"{self.w_actual[i]:.4f}",
                    f"{self.v_cmd[i]:.4f}",
                    f"{self.w_cmd[i]:.4f}",
                ])

        # 2. Render Figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # (a) 2D Trajectory
        axs[0, 0].plot(self.x_vals, self.y_vals, 'b-', label='Actual Path')
        axs[0, 0].plot(self.x_vals[0], self.y_vals[0], 'go', label='Start')
        axs[0, 0].plot(self.x_vals[-1], self.y_vals[-1], 'ro', label='Current')
        axs[0, 0].set_title('Robot Trajectory in Cartesian Frame (X vs Y)')
        axs[0, 0].set_xlabel('X [m]')
        axs[0, 0].set_ylabel('Y [m]')
        axs[0, 0].grid(True)
        axs[0, 0].axis('equal')
        axs[0, 0].legend()

        # (b) Linear Velocity Response v(t) vs v_cmd(t)
        axs[0, 1].plot(self.timestamps, self.v_cmd, 'r--', label='Command v*(t)')
        axs[0, 1].plot(self.timestamps, self.v_actual, 'b-', label='Actual v(t)')
        axs[0, 1].set_title('Linear Velocity Response: v*(t) vs v(t)')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].set_ylabel('Linear Speed [m/s]')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # (c) Angular Velocity Response w(t) vs w_cmd(t)
        axs[1, 0].plot(self.timestamps, self.w_cmd, 'r--', label='Command w*(t)')
        axs[1, 0].plot(self.timestamps, self.w_actual, 'g-', label='Actual w(t)')
        axs[1, 0].set_title('Angular Velocity Response: w*(t) vs w(t)')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].set_ylabel('Angular Speed [rad/s]')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # (d) Orientation Evolution Yaw(t)
        axs[1, 1].plot(self.timestamps, self.yaw_vals, 'm-', label='Yaw [rad]')
        axs[1, 1].set_title('Robot Heading Evolution: theta(t)')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].set_ylabel('Yaw [rad]')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)

        self.get_logger().info(f"Telemetry saved: CSV -> {csv_path}, Plot -> {plot_path}")


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_telemetry()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
