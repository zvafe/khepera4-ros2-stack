#!/usr/bin/env python3
"""
Closed-Loop Pose and Trajectory Controller for Khepera IV Mobile Robot.
Implements kinematic feedback regulation and cascade control hierarchy
as specified in BSA Control Systems Research Charter (Objectives 1, 2, 4).
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from khepera4_control.common import euler_from_quaternion, normalize_angle


class ClosedLoopController(Node):
    def __init__(self):
        super().__init__('closed_loop_controller')

        # Control Parameters (Lyapunov / Cascade feedback gains)
        self.declare_parameter('mode', 'step')          # 'step', 'waypoint', 'square', 'circle'
        self.declare_parameter('k_rho', 0.8)            # Linear velocity gain (outer loop)
        self.declare_parameter('k_alpha', 2.0)          # Heading steering gain
        self.declare_parameter('k_beta', -0.4)          # Goal orientation alignment gain
        self.declare_parameter('k_yaw', 1.5)            # In-place rotation gain
        self.declare_parameter('v_max', 0.25)           # Max linear speed (m/s)
        self.declare_parameter('w_max', 1.8)            # Max angular speed (rad/s)
        self.declare_parameter('pos_tolerance', 0.02)   # 2 cm tolerance
        self.declare_parameter('yaw_tolerance', 0.05)   # ~2.8 deg tolerance

        # Single goal setpoints (used in 'step' and 'waypoint' modes)
        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_v = 0.0
        self.current_w = 0.0
        self.has_odom = False

        # Waypoint list for trajectory modes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.active_goal = None
        self.state = 'INIT'

        # Telemetry / transient metrics logging
        self.history = []
        self.step_start_time = None
        self.target_dist = 0.0
        self.last_odom_warn_time = 0.0
        self.last_progress_log_time = 0.0

        # Control loop at 50 Hz
        self.timer = self.create_timer(0.02, self.control_step)
        self.get_logger().info('Khepera IV Closed-Loop Feedback Controller initialized.')

    def goal_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        _, _, yaw = euler_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        self.active_goal = (x, y, yaw)
        self.state = 'NAVIGATING'
        self.get_logger().info(f"Received new setpoint goal via /goal_pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)

        self.current_x = pos.x
        self.current_y = pos.y
        self.current_yaw = yaw
        self.current_v = twist.linear.x
        self.current_w = twist.angular.z
        self.has_odom = True

    def init_trajectory(self):
        mode = self.get_parameter('mode').value
        if mode == 'step':
            gx = self.get_parameter('goal_x').value
            gy = self.get_parameter('goal_y').value
            gyaw = self.get_parameter('goal_yaw').value
            self.waypoints = [(self.current_x + gx, self.current_y + gy, gyaw)]
            self.target_dist = math.hypot(gx, gy)
        elif mode == 'waypoint':
            gx = self.get_parameter('goal_x').value
            gy = self.get_parameter('goal_y').value
            gyaw = self.get_parameter('goal_yaw').value
            self.waypoints = [(gx, gy, gyaw)]
        elif mode == 'square':
            # 0.8m x 0.8m square path
            cx = self.current_x
            cy = self.current_y
            self.waypoints = [
                (cx + 0.8, cy + 0.0, 0.0),
                (cx + 0.8, cy + 0.8, math.pi / 2.0),
                (cx + 0.0, cy + 0.8, math.pi),
                (cx + 0.0, cy + 0.0, -math.pi / 2.0)
            ]
        elif mode == 'circle':
            # Approximate circular orbit of radius R = 0.4m with 16 waypoints
            R = 0.4
            cx = self.current_x
            cy = self.current_y + R
            self.waypoints = []
            N = 16
            for i in range(N):
                theta = -math.pi / 2.0 + (2.0 * math.pi * (i + 1) / N)
                wx = cx + R * math.cos(theta)
                wy = cy + R * math.sin(theta)
                wyaw = normalize_angle(theta + math.pi / 2.0)
                self.waypoints.append((wx, wy, wyaw))

        self.current_waypoint_idx = 0
        if self.waypoints:
            self.active_goal = self.waypoints[0]
            self.state = 'NAVIGATING'
            self.step_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(
                f"Trajectory initialized. Mode: {mode}. Total Waypoints: {len(self.waypoints)}. "
                f"First Goal: ({self.active_goal[0]:.3f}, {self.active_goal[1]:.3f})"
            )

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if not self.has_odom:
            if (now - self.last_odom_warn_time) > 3.0:
                self.get_logger().warn(
                    "Waiting for '/odom' messages from Gazebo... "
                    "Make sure simulation is running: 'ros2 launch khepera4_description khepera4_sim.launch.py'"
                )
                self.last_odom_warn_time = now
            return

        if self.state == 'INIT':
            self.init_trajectory()
            return

        if self.state == 'IDLE' or self.active_goal is None:
            self.cmd_pub.publish(Twist())
            return

        gx, gy, gyaw = self.active_goal

        dx = gx - self.current_x
        dy = gy - self.current_y
        rho = math.hypot(dx, dy)
        alpha = normalize_angle(math.atan2(dy, dx) - self.current_yaw)
        beta = normalize_angle(-self.current_yaw - alpha + gyaw)

        k_rho = self.get_parameter('k_rho').value
        k_alpha = self.get_parameter('k_alpha').value
        k_beta = self.get_parameter('k_beta').value
        k_yaw = self.get_parameter('k_yaw').value
        v_max = self.get_parameter('v_max').value
        w_max = self.get_parameter('w_max').value
        pos_tol = self.get_parameter('pos_tolerance').value
        yaw_tol = self.get_parameter('yaw_tolerance').value

        twist = Twist()

        # Phase 1: Position approaching (if not within pos_tolerance)
        if rho > pos_tol:
            # If the goal is behind the robot, reverse or turn in place
            if abs(alpha) > math.pi / 2.0:
                # Rotate towards target first to avoid driving backwards awkwardly
                twist.linear.x = 0.0
                twist.angular.z = float(max(-w_max, min(w_max, k_alpha * alpha)))
            else:
                v = k_rho * rho
                w = k_alpha * alpha + k_beta * beta
                twist.linear.x = float(max(-v_max, min(v_max, v)))
                twist.angular.z = float(max(-w_max, min(w_max, w)))

            # Periodic progress log
            if (now - self.last_progress_log_time) > 1.5:
                self.get_logger().info(
                    f"[Tracking] Pos: ({self.current_x:.2f}, {self.current_y:.2f}) -> "
                    f"Goal: ({gx:.2f}, {gy:.2f}) | Dist: {rho:.3f}m | Heading Err: {math.degrees(alpha):.1f}° | Speed: {twist.linear.x:.2f}m/s"
                )
                self.last_progress_log_time = now

        # Phase 2: Heading alignment at goal
        else:
            yaw_err = normalize_angle(gyaw - self.current_yaw)
            if abs(yaw_err) > yaw_tol:
                twist.linear.x = 0.0
                twist.angular.z = float(max(-w_max, min(w_max, k_yaw * yaw_err)))
                if (now - self.last_progress_log_time) > 1.5:
                    self.get_logger().info(
                        f"[Aligning Heading] Current Yaw: {math.degrees(self.current_yaw):.1f}° -> Goal: {math.degrees(gyaw):.1f}°"
                    )
                    self.last_progress_log_time = now
            else:
                # Waypoint reached!
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(f"Waypoint {self.current_waypoint_idx+1}/{len(self.waypoints)} reached!")
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.waypoints):
                    self.active_goal = self.waypoints[self.current_waypoint_idx]
                    self.get_logger().info(f"Heading to next waypoint: {self.active_goal}")
                else:
                    self.get_logger().info("All waypoints successfully reached! Mission complete.")
                    self.state = 'IDLE'
                    self.active_goal = None
                    self.cmd_pub.publish(twist)
                    self.compute_and_log_metrics()
                    return

        self.cmd_pub.publish(twist)

        # Log history for metrics computation
        if self.step_start_time is not None:
            t = now - self.step_start_time
            self.history.append({
                't': t,
                'x': self.current_x,
                'y': self.current_y,
                'yaw': self.current_yaw,
                'v': self.current_v,
                'w': self.current_w,
                'rho': rho
            })

    def compute_and_log_metrics(self):
        """
        Compute standard second-order performance indicators from Charter:
        - Rise time (tr)
        - Settling time (ts, 2% criterion)
        - Maximum overshoot (sigma%)
        - Steady-state error (est)
        """
        if not self.history or self.target_dist <= 0.0:
            return

        final_pos = (self.current_x, self.current_y)
        self.get_logger().info('================ CONTROL PERFORMANCE METRICS ================')
        self.get_logger().info(f"Target Displacement: {self.target_dist:.4f} m")

        # Compute displacement curve s(t)
        s0 = math.hypot(self.history[0]['x'], self.history[0]['y'])
        times = [pt['t'] for pt in self.history]
        disps = [math.hypot(pt['x'], pt['y']) - s0 for pt in self.history]

        target = self.target_dist
        # Overshoot
        max_s = max(disps)
        overshoot = max(0.0, (max_s - target) / target * 100.0)

        # Steady-state error
        e_st = abs(disps[-1] - target)

        # Rise time (10% to 90%)
        t_10 = next((t for t, s in zip(times, disps) if s >= 0.10 * target), None)
        t_90 = next((t for t, s in zip(times, disps) if s >= 0.90 * target), None)
        t_rise = (t_90 - t_10) if (t_10 is not None and t_90 is not None) else None

        # Settling time (2% band)
        band = 0.02 * target
        settled_indices = [i for i, s in enumerate(disps) if abs(s - target) > band]
        if settled_indices:
            last_outside = settled_indices[-1]
            t_settling = times[last_outside]
        else:
            t_settling = times[0]

        self.get_logger().info(f"Maximum Overshoot (sigma%): {overshoot:.2f} %")
        if t_rise is not None:
            self.get_logger().info(f"Rise Time (tr, 10%-90%):    {t_rise:.3f} s")
        self.get_logger().info(f"Settling Time (ts, 2% band): {t_settling:.3f} s")
        self.get_logger().info(f"Steady-State Error (est):   {e_st:.4f} m")
        self.get_logger().info('===========================================================')


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopController()
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
