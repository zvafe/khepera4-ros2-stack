#!/usr/bin/env python3
"""
Open-Loop Benchmark and Kinematics Validation Node for Khepera IV.
Course: Bazele Sistemelor Automate (BSA)
Validates unicycle forward and inverse kinematics and diff-drive parameters.
"""

import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from khepera4_control.common import euler_from_quaternion, normalize_angle


class OpenLoopBenchmark(Node):
    def __init__(self):
        super().__init__('open_loop_benchmark')

        self.declare_parameter('maneuver', 'forward')  # 'forward', 'rotate', 'circle'
        self.declare_parameter('linear_speed', 0.15)    # m/s
        self.declare_parameter('angular_speed', 1.0)   # rad/s
        self.declare_parameter('duration', 4.0)        # seconds

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.has_initial_odom = False

        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        self.state = 'WAIT_ODOM'
        self.start_time = None

        self.get_logger().info('Open-Loop Benchmark Node initialized. Waiting for /odom...')

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)

        self.current_x = pos.x
        self.current_y = pos.y
        self.current_yaw = yaw
        self.has_initial_odom = True

    def control_loop(self):
        if not self.has_initial_odom:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.state == 'WAIT_ODOM':
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.start_yaw = self.current_yaw
            self.start_time = now
            self.state = 'RUNNING'
            maneuver = self.get_parameter('maneuver').value
            duration = self.get_parameter('duration').value
            self.get_logger().info(
                f"Starting maneuver: '{maneuver}' for {duration:.1f}s from initial pose "
                f"x={self.start_x:.3f}, y={self.start_y:.3f}, yaw={self.start_yaw:.3f} rad"
            )

        elif self.state == 'RUNNING':
            elapsed = now - self.start_time
            duration = self.get_parameter('duration').value
            maneuver = self.get_parameter('maneuver').value
            v_cmd = self.get_parameter('linear_speed').value
            w_cmd = self.get_parameter('angular_speed').value

            if elapsed < duration:
                twist = Twist()
                if maneuver == 'forward':
                    twist.linear.x = float(v_cmd)
                    twist.angular.z = 0.0
                elif maneuver == 'rotate':
                    twist.linear.x = 0.0
                    twist.angular.z = float(w_cmd)
                elif maneuver == 'circle':
                    twist.linear.x = float(v_cmd)
                    twist.angular.z = float(w_cmd)
                self.cmd_pub.publish(twist)
            else:
                # Stop robot
                self.cmd_pub.publish(Twist())
                self.state = 'DONE'
                self.print_report()

    def print_report(self):
        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y
        dist = math.hypot(dx, dy)
        dyaw = normalize_angle(self.current_yaw - self.start_yaw)
        maneuver = self.get_parameter('maneuver').value
        duration = self.get_parameter('duration').value
        v_cmd = self.get_parameter('linear_speed').value
        w_cmd = self.get_parameter('angular_speed').value

        self.get_logger().info('================ MANEUVER BENCHMARK REPORT ================')
        self.get_logger().info(f"Maneuver Type:       {maneuver}")
        self.get_logger().info(f"Commanded Duration:  {duration:.3f} s")
        self.get_logger().info(f"Final Position (X,Y):({self.current_x:.4f}, {self.current_y:.4f}) m")
        self.get_logger().info(f"Displacement (Dist): {dist:.4f} m (dx={dx:.4f}, dy={dy:.4f})")
        self.get_logger().info(f"Angular Rotation:    {dyaw:.4f} rad ({math.degrees(dyaw):.2f} deg)")

        if maneuver == 'forward':
            expected_dist = v_cmd * duration
            error = dist - expected_dist
            self.get_logger().info(f"Theoretical Distance:{expected_dist:.4f} m")
            self.get_logger().info(f"Tracking Error:      {error:.4f} m (Heading Drift: {math.degrees(dyaw):.4f} deg)")
        elif maneuver == 'rotate':
            expected_yaw = w_cmd * duration
            self.get_logger().info(f"Theoretical Rotation:{expected_yaw:.4f} rad ({math.degrees(expected_yaw):.2f} deg)")
            self.get_logger().info(f"Linear Displacement: {dist:.4f} m (Should be ~0 m)")

        self.get_logger().info('===========================================================')


def main(args=None):
    rclpy.init(args=args)
    node = OpenLoopBenchmark()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
