#!/usr/bin/env python3
"""
Verification Script: Run Khepera IV Simulation in Gazebo Harmonic & verify exploration rate.
Requirement: Explores >= 80% of the 5m x 5m arena at full speed in under 1 minute (60s).
"""

import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist


class ExplorationVerifier(Node):
    def __init__(self):
        super().__init__('exploration_verifier')
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)

        self.latest_map = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.latest_cmd_v = 0.0
        self.latest_cmd_w = 0.0
        self.start_time = None
        self.reached_80_time = None
        self.max_coverage = 0.0

    def on_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_v = msg.twist.twist.linear.x
        self.robot_w = msg.twist.twist.angular.z

    def on_cmd(self, msg: Twist):
        self.latest_cmd_v = msg.linear.x
        self.latest_cmd_w = msg.angular.z

    def on_map(self, msg: OccupancyGrid):
        self.latest_map = msg
        if self.start_time is None:
            self.start_time = time.time()

    def compute_arena_coverage(self):
        if self.latest_map is None:
            return 0.0, 0, 0

        info = self.latest_map.info
        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        w = info.width
        h = info.height

        data = np.array(self.latest_map.data, dtype=np.int8).reshape((h, w))

        # Arena bounds in meters
        x_min, x_max = -2.4, 2.4
        y_min, y_max = -2.4, 2.4

        col_min = max(0, int((x_min - origin_x) / res))
        col_max = min(w - 1, int((x_max - origin_x) / res))
        row_min = max(0, int((y_min - origin_y) / res))
        row_max = min(h - 1, int((y_max - origin_y) / res))

        if col_max <= col_min or row_max <= row_min:
            return 0.0, 0, 0

        arena_slice = data[row_min:row_max + 1, col_min:col_max + 1]
        total_cells = arena_slice.size
        # Explored cells have value != -1 (i.e. free space 0..49 or occupied >= 50)
        explored_cells = int(np.sum(arena_slice != -1))
        coverage_pct = (explored_cells / float(total_cells)) * 100.0

        if coverage_pct > self.max_coverage:
            self.max_coverage = coverage_pct

        return coverage_pct, explored_cells, total_cells


def main():
    print("=== Khepera IV Exploration Benchmark ===")
    ws_dir = "/home/andy/Desktop/khepera4_ws"

    # Step 1: Clean simulation
    print("Cleaning any existing simulation processes...")
    subprocess.run(["bash", f"{ws_dir}/scripts/clean_sim.sh"], check=False)
    time.sleep(2)

    # Step 2: Launch mapping with auto_explore:=true, headless:=true, linear_speed:=0.25
    launch_cmd = [
        "ros2", "launch", "khepera4_control", "mapping.launch.py",
        "mode:=sim",
        "headless:=true",
        "auto_explore:=true",
        "linear_speed:=0.35",
        "dashboard:=false",

        "rviz:=false"
    ]
    print(f"Launching simulation: {' '.join(launch_cmd)}")
    env = os.environ.copy()
    sim_log = open('/tmp/khepera_sim_verify_launch.log', 'w')
    launch_proc = subprocess.Popen(
        launch_cmd,
        cwd=ws_dir,
        env=env,
        stdout=sim_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    rclpy.init()
    node = ExplorationVerifier()

    timeout_sec = 75.0
    start_wait = time.time()
    last_print = 0.0

    print("Waiting for simulation & mapping engine to initialize and publish first /map...")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.time()

            if node.latest_map is not None:
                elapsed = now - node.start_time
                cov, exp, tot = node.compute_arena_coverage()

                if now - last_print >= 1.0:
                    last_print = now
                    print(
                        f"[{elapsed:5.1f}s] Coverage: {cov:5.1f}% ({exp}/{tot} cells) | "
                        f"Pose: ({node.robot_x:+.2f}, {node.robot_y:+.2f}) | "
                        f"Cmd: v={node.latest_cmd_v:.2f} m/s, ω={node.latest_cmd_w:+.2f} rad/s"
                    )

                if cov >= 80.0 and node.reached_80_time is None:
                    node.reached_80_time = elapsed
                    print(f"\n🎉 TARGET ACHIEVED! 80% coverage reached in {elapsed:.2f} seconds! 🎉\n")
                    if elapsed <= 60.0:
                        print(f"SUCCESS: Requirement satisfied! (< 60.0s, achieved {elapsed:.2f}s)")
                    else:
                        print(f"WARNING: Reached 80% but took {elapsed:.2f}s (> 60s)")

                if elapsed >= timeout_sec or (node.reached_80_time is not None and elapsed >= node.reached_80_time + 3.0):
                    break
            else:
                if now - start_wait > 30.0:
                    print("ERROR: Timeout waiting for first /map message!")
                    break
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("\nShutting down simulation...")
        try:
            os.killpg(os.getpgid(launch_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        time.sleep(2)
        subprocess.run(["bash", f"{ws_dir}/scripts/clean_sim.sh"], check=False)

    print("\n=== Benchmark Summary ===")
    print(f"Max Arena Coverage: {node.max_coverage:.2f}%")
    if node.reached_80_time is not None:
        print(f"Time to 80% Coverage: {node.reached_80_time:.2f}s")
        if node.reached_80_time <= 60.0:
            print("Verdict: PASSED (Under 1 minute)")
            return 0
        else:
            print("Verdict: FAILED (Exceeded 60s limit)")
            return 1
    else:
        print("Verdict: FAILED (Did not reach 80% coverage within timeout)")
        return 2


if __name__ == '__main__':
    sys.exit(main())
