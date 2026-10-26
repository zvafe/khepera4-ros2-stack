# Copyright 2026 Mobile Robotics Laboratory.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for wall sliding alignment, sensor ray occlusion stopping, and obstacle avoidance."""

import math
import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from khepera4_control.khepera4_mapper import Khepera4Mapper
from khepera4_control.reactive_explorer import ReactiveExplorer


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan(dist: float, min_range: float = 0.002, max_range: float = 0.25) -> LaserScan:
    scan = LaserScan()
    scan.header.frame_id = 'base_footprint'
    scan.range_min = float(min_range)
    scan.range_max = float(max_range)
    scan.ranges = [float(dist)]
    return scan


def test_ray_marching_stops_at_occupied_cells():
    """Verify that ray casting never clears cells behind an already occupied wall."""
    mapper = Khepera4Mapper()
    try:
        mapper.has_odom = True
        mapper.robot_x = 0.0
        mapper.robot_y = 0.0
        mapper.robot_yaw = 0.0

        # Mark an existing wall at cell col=105, row=100 (which is at x = origin_x + 105*0.03 = -3.0 + 3.15 = 0.15m)
        c_wall = 105
        r_wall = 100
        mapper.log_odds[r_wall, c_wall] = mapper.l_occ + 1.0  # Confirmed wall

        # Cells behind the wall should be unobserved (log_odds = 0)
        c_behind = 108
        r_behind = 100
        mapper.log_odds[r_behind, c_behind] = 0.0

        # Feed an ultrasonic reading that attempts to see through the wall to distance 0.30m
        # us_2 is at (0.0636, 0, 0), pointing East (+X)
        mapper.us_callback(2, create_scan(0.30, min_range=0.02, max_range=2.0))

        # Update occupancy grid
        mapper._update_occupancy_rays()

        # Wall cell must remain occupied
        assert mapper.log_odds[r_wall, c_wall] >= mapper.l_occ, "Wall was cleared by ray casting!"

        # Cells BEHIND the wall must NOT have been cleared as free space
        assert mapper.log_odds[r_behind, c_behind] == 0.0, (
            f"Cell behind wall was cleared through occlusion! log_odds={mapper.log_odds[r_behind, c_behind]}"
        )
    finally:
        mapper.destroy_node()


def test_wall_sliding_zero_perpendicular_drift():
    """Verify sliding along North perimeter wall has zero Y-drift and stays aligned."""
    mapper = Khepera4Mapper()
    try:
        mapper.has_odom = True
        mapper.use_imu_yaw = True
        mapper.has_imu = True
        mapper.imu_valid = True
        mapper.imu_yaw = math.pi  # Facing West (+X is behind, -X is forward, +Y is North/Right)
        mapper.robot_yaw = math.pi
        mapper.robot_x = 0.0
        mapper.robot_y = 2.39  # Near North wall (inner wall at 2.475, radius 0.07 -> max 2.405)

        # Right IR sensor (ir_2 or ir_1) reports contact with North wall
        mapper.ir_callback(2, create_scan(0.010))

        initial_y = mapper.robot_y

        qx, qy, qz, qw = 0.0, 0.0, 1.0, 0.0
        for step in range(1, 31):
            odom = Odometry()
            odom.pose.pose.position.x = -0.005 * step
            odom.pose.pose.position.y = 0.002 * step  # Slip pushing North
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            mapper.odom_callback(odom)

        # X must advance West
        assert mapper.robot_x < 0.0, f"Robot failed to advance West: x={mapper.robot_x}"

        # Y must NEVER drift through the North perimeter wall
        assert mapper.robot_y <= mapper.arena_max_y, f"Robot penetrated North wall: y={mapper.robot_y}"
        # Y drift must be strictly bounded under 2mm (no cumulative 45-degree skew)
        assert abs(mapper.robot_y - initial_y) < 0.02, (
            f"Robot Y position drifted uncontrollably during wall sliding: initial={initial_y}, final={mapper.robot_y}"
        )
    finally:
        mapper.destroy_node()


def test_reactive_explorer_steers_away_from_lateral_obstacles():
    """Verify Braitenberg repulsion steers AWAY from lateral obstacles."""
    explorer = ReactiveExplorer()
    try:
        explorer.enabled = True
        explorer.state = explorer.STATE_CRUISE

        # Obstacle on the RIGHT:
        # d_right = 0.15m, d_left = 2.0m, d_front = 2.0m
        explorer.ir_callback(6, create_scan(0.04))  # Front-Right
        explorer.ir_callback(7, create_scan(0.04))  # Front-Right-Center

        d_l, d_f, d_r = explorer._extract_sector_distances()
        assert d_r < d_l, "Right distance should be smaller than left"

        # Repulsion calculation: obstacle on right must command POSITIVE angular.z (steer LEFT, counter-clockwise)
        w_left = max(0.0, (explorer.avoid_dist - d_l) / explorer.avoid_dist)
        w_right = max(0.0, (explorer.avoid_dist - d_r) / explorer.avoid_dist)
        w_repulse = (w_right - w_left) * explorer.turn_speed * 1.5

        assert w_repulse > 0.0, f"Must turn LEFT (positive angular.z) to avoid right obstacle, got {w_repulse}"

        # Obstacle on the LEFT:
        explorer.latest_ir = [None] * 8
        explorer.ir_callback(1, create_scan(0.04))  # Front-Left
        explorer.ir_callback(0, create_scan(0.04))  # Front-Left-Center

        d_l, d_f, d_r = explorer._extract_sector_distances()
        assert d_l < d_r, "Left distance should be smaller than right"

        w_left = max(0.0, (explorer.avoid_dist - d_l) / explorer.avoid_dist)
        w_right = max(0.0, (explorer.avoid_dist - d_r) / explorer.avoid_dist)
        w_repulse = (w_right - w_left) * explorer.turn_speed * 1.5

        assert w_repulse < 0.0, f"Must turn RIGHT (negative angular.z) to avoid left obstacle, got {w_repulse}"
    finally:
        explorer.destroy_node()
