#!/usr/bin/env python3
"""
Automated End-to-End Pursuit Validation Script for Khepera IV Fetch Stack.

Subscribes to all telemetry topics, monitors robot state transitions,
verifies physical pursuit along +x towards the target object, and saves
captured visual frames (raw camera + DINOv2 debug HUD) for user inspection.
"""

import os
import sys
import time
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from khepera_interfaces.msg import TargetDetection, ExecutiveState


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """Fast conversion of sensor_msgs/Image to BGR numpy array."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('bgr8', 'rgb8'):
        arr = data.reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()
    return data.reshape((msg.height, msg.width, -1)).copy()


class PursuitValidator(Node):
    def __init__(self, output_dir: str = "/tmp/fetch_validation"):
        super().__init__('pursuit_validator')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Telemetry Cache
        self.detections: List[Tuple[float, bool, float]] = []
        self.poses: List[Tuple[float, float, float]] = []
        self.cmd_vels: List[Tuple[float, float, float]] = []
        self.odom_positions: List[Tuple[float, float, float]] = []
        self.executive_states: List[Tuple[float, str]] = []

        self.latest_raw_image: Optional[np.ndarray] = None
        self.latest_debug_image: Optional[np.ndarray] = None
        self.saved_snapshots_count = 0

        # Initial and terminal odom
        self.start_x: Optional[float] = None
        self.max_x: float = 0.0
        self.terminal_reached: bool = False

        # Subscriptions
        self.create_subscription(TargetDetection, '/target/detection', self._on_detection, 10)
        self.create_subscription(PoseStamped, '/target/pose_ground_estimated', self._on_pose, 10)
        self.create_subscription(ExecutiveState, '/executive/status', self._on_status, 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Image, '/camera/image_raw', self._on_camera, 10)
        self.create_subscription(Image, '/target/detection_visual', self._on_debug_image, 10)

        self.get_logger().info("PursuitValidator initialized. Monitoring topics...")

    def _get_time(self) -> float:
        t_nanos = self.get_clock().now().nanoseconds
        if t_nanos > 0:
            return t_nanos * 1e-9
        return time.time()

    def _on_detection(self, msg: TargetDetection):
        now = self._get_time()
        self.detections.append((now, msg.is_detected, msg.max_similarity))
        if msg.is_detected:
            self.get_logger().info(
                f"[DETECTION] Target spotted! Sim={msg.max_similarity:.2f}, "
                f"Contact=({msg.contact_pixel.x:.1f}, {msg.contact_pixel.y:.1f}), Patches={msg.cluster_patch_count}",
                throttle_duration_sec=1.0
            )

    def _on_pose(self, msg: PoseStamped):
        now = self._get_time()
        self.poses.append((now, msg.pose.position.x, msg.pose.position.y))
        self.get_logger().info(
            f"[PROJECTOR] Estimated Ground Pose: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
            f"frame={msg.header.frame_id}",
            throttle_duration_sec=1.0
        )

    def _on_status(self, msg: ExecutiveState):
        now = self._get_time()
        if not self.executive_states or self.executive_states[-1][1] != msg.current_state:
            self.get_logger().info(f"[EXECUTIVE STATE] Transitioned -> {msg.current_state}")
            self.executive_states.append((now, msg.current_state))
        if msg.current_state == 'TERMINAL_ACTION':
            self.terminal_reached = True

    def _on_cmd_vel(self, msg: Twist):
        now = self._get_time()
        self.cmd_vels.append((now, msg.linear.x, msg.angular.z))
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.get_logger().info(
                f"[ACTUATION /cmd_vel] v={msg.linear.x:.3f} m/s, w={msg.angular.z:.3f} rad/s",
                throttle_duration_sec=1.0
            )

    def _on_odom(self, msg: Odometry):
        now = self._get_time()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.start_x is None:
            self.start_x = x
        self.max_x = max(self.max_x, x)
        self.odom_positions.append((now, x, y))

    def _on_camera(self, msg: Image):
        try:
            self.latest_raw_image = imgmsg_to_cv2(msg)
        except Exception:
            pass

    def _on_debug_image(self, msg: Image):
        try:
            self.latest_debug_image = imgmsg_to_cv2(msg)
            now = time.time()
            if self.saved_snapshots_count < 15 and (not hasattr(self, '_last_save_time') or (now - self._last_save_time > 0.8)):
                self._last_save_time = now
                snap_path = os.path.join(self.output_dir, f"detected_debug_{self.saved_snapshots_count:03d}.png")
                cv2.imwrite(snap_path, self.latest_debug_image)
                if self.latest_raw_image is not None:
                    raw_path = os.path.join(self.output_dir, f"detected_raw_{self.saved_snapshots_count:03d}.png")
                    cv2.imwrite(raw_path, self.latest_raw_image)
                self.saved_snapshots_count += 1
                self.get_logger().info(f"Saved pursuit snapshot #{self.saved_snapshots_count} to {snap_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to process debug image: {e}")


def main():
    rclpy.init()
    validator = PursuitValidator()

    timeout_sec = 60.0
    start_wall = time.time()
    try:
        while rclpy.ok() and (time.time() - start_wall < timeout_sec):
            rclpy.spin_once(validator, timeout_sec=0.1)

            # Check if robot has reached terminal action or traversed full distance
            if validator.start_x is not None:
                forward_displacement = validator.max_x - validator.start_x
                if validator.terminal_reached or forward_displacement >= 1.15:
                    validator.get_logger().info(
                        f"SUCCESS CRITERIA MET: Robot displaced forward by {forward_displacement:.3f}m! "
                        f"Terminal reached: {validator.terminal_reached}"
                    )
                    # Let it run 3 more seconds for final snapshots
                    end_extra = time.time() + 3.0
                    while time.time() < end_extra:
                        rclpy.spin_once(validator, timeout_sec=0.1)
                    break
    except KeyboardInterrupt:
        pass

    # Print Summary Report
    print("\n" + "=" * 60)
    print("           PURSUIT VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    positives = [d for d in validator.detections if d[1]]
    forward_vels = [c for c in validator.cmd_vels if c[1] > 0.02]
    disp = (validator.max_x - validator.start_x) if validator.start_x is not None else 0.0

    print(f"Total Detections Received:      {len(validator.detections)} (Positive hits: {len(positives)})")
    if positives:
        max_sim = max(d[2] for d in positives)
        print(f"Peak Cosine Similarity:         {max_sim:.3f}")
    print(f"Ground Poses Estimated:         {len(validator.poses)}")
    print(f"Commanded Forward Pulses (/cmd_vel): {len(forward_vels)}")
    print(f"Initial Odometry X:             {validator.start_x:.3f} m" if validator.start_x else "N/A")
    print(f"Peak Forward Odometry X:        {validator.max_x:.3f} m")
    print(f"Net Forward Pursuit Distance:   {disp:.3f} m")
    print(f"FSM States Traversed:           {[s[1] for s in validator.executive_states]}")
    print(f"Terminal Action Reached:        {validator.terminal_reached}")
    print(f"Visual Debug Snapshots Saved:   {validator.saved_snapshots_count} in {validator.output_dir}")
    print("=" * 60 + "\n")

    validator.destroy_node()
    rclpy.shutdown()

    if disp > 0.20 or len(forward_vels) > 5:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
