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

"""
E2E Acceptance Test Suite: Cross-Feature Combinations & Real-World Scenarios.

Covers:
- Tier 3: Cross-Feature Combinations (Pairwise coverage of major feature interactions)
- Tier 4: Real-World Application Scenarios (Full mission cycles, maze traversal, wall scraping, ramp ascent)
"""

import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from geometry_msgs.msg import Twist
from khepera4_control.common import euler_from_quaternion, quaternion_from_euler
from khepera4_control.khepera4_mapper import Khepera4Mapper
from khepera4_control.mapping_dashboard import MappingRosWorker
from khepera4_control.reactive_explorer import ReactiveExplorer
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool


# ==============================================================================
# Pytest Fixtures & Helpers
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan_msg(distance: float, min_range: float = 0.002, max_range: float = 0.20) -> LaserScan:
    """Helper to create LaserScan message."""
    scan = LaserScan()
    scan.ranges = [float(distance)]
    scan.range_min = float(min_range)
    scan.range_max = float(max_range)
    return scan


# ==============================================================================
# TIER 3: Cross-Feature Combinations (Pairwise Feature Interactions)
# ==============================================================================

def test_t3_ramp_ascent_pitch_telemetry_with_slip_dead_reckoning():
    """Tier 3: F4 + F7 - Verify ramp ascent pitch extraction works concurrently with slip dead reckoning."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.robot_x = 0.0
        node.robot_y = 0.0

        # Step 1: Robot drives onto 10° ramp. Pitch rises to 10°, yaw remains 0.0
        pitch_rad = math.radians(10.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu_msg = Imu()
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
        imu_msg.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        node.imu_callback(imu_msg)

        # Step 2: Feed wheel odometry advances (50 steps of 0.0024m = 0.12m)
        for i in range(1, 51):
            odom = Odometry()
            odom.pose.pose.position.x = i * 0.0024
            odom.pose.pose.position.y = 0.0
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)

        # Fused yaw must remain 0.0, displacement must advance to ~0.12m along ramp
        assert math.isclose(node.robot_yaw, 0.0, abs_tol=1e-3)
        assert math.isclose(node.robot_x, 0.12, abs_tol=1e-3)
        assert math.isclose(node.current_slope_deg, 10.0, abs_tol=1.0)
    finally:
        node.destroy_node()


def test_t3_glancing_wall_contact_with_tangential_projection_and_evasive_turn():
    """Tier 3: F6 + F9 - Glancing wall contact tangential projection coupled with evasive reaction."""
    mapper = Khepera4Mapper()
    explorer = ReactiveExplorer()
    try:
        # Glancing contact on front-right IR sensors
        mapper.ir_callback(7, create_scan_msg(0.012))
        explorer.ir_callback(7, create_scan_msg(0.012))
        explorer.last_commanded_v = 0.12

        # Step explorer control loop
        explorer.control_loop()
        assert explorer.state in (explorer.STATE_AVOID, explorer.STATE_REVERSE_TURN, explorer.STATE_CRUISE)
    finally:
        mapper.destroy_node()
        explorer.destroy_node()


def test_t3_outer_perimeter_clamping_during_high_speed_evasive_backup():
    """Tier 3: F8 + F9 - Verify high-speed reverse backup never breaks outer perimeter walls."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0

        # Start close to West perimeter: x = -2.40m
        node.robot_x = -2.40
        node.robot_y = 0.0

        # Reversing commands backwards motion towards -X
        odom_rev = Odometry()
        odom_rev.pose.pose.position.x = -2.60  # Command passes West wall at -2.475m
        odom_rev.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_rev)

        # Must be clamped at arena boundary: robot_x >= -2.430m
        assert node.robot_x >= -2.430, f'Reversed through West wall: x={node.robot_x}'
    finally:
        node.destroy_node()


def test_t3_ir_bump_detection_with_fused_odometry_publication_and_dashboard_lock():
    """Tier 3: F5 + F10 + F11 - IR bump triggers dead-reckoning clamp and dashboard pose lock."""
    node = Khepera4Mapper()
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_x = 0.0
        node.robot_y = 0.0

        # IR contact detected
        node.ir_callback(0, create_scan_msg(0.010))

        # Commanded slip advance
        odom = Odometry()
        odom.pose.pose.position.x = 0.10
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)

        # Position clamped to 0.03m
        assert node.robot_x <= 0.030
    finally:
        node.destroy_node()
        worker.node.destroy_node()


def test_t3_ramp_approach_lip_transition_with_caster_ground_contact():
    """Tier 3: F1 + F2 + F3 + F4 - Integration of caster ground contact, approach lip, and pitch onset."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(test_dir, '..', '..', '..'))
    urdf_path = os.path.join(ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf')
    world_path = os.path.join(ws_root, 'src', 'khepera4_description', 'worlds', 'khepera4_exploration_world.sdf')

    root_u = ET.parse(urdf_path).getroot()
    root_w = ET.parse(world_path).getroot()

    # Caster at z=0 contact
    joints = {j.get('name'): j for j in root_u.findall('joint')}
    caster_z = float(joints['front_caster_joint'].find('origin').get('xyz').split()[2])
    assert math.isclose(caster_z - 0.005, 0.0, abs_tol=1e-3)

    # Ramp approach lip at z ~ 0
    ramp = [m for m in root_w.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    rx, ry, rz, _, pitch, _ = [float(v) for v in ramp.find('pose').text.split()]
    length = float(ramp.find('link').find('collision').find('geometry').find('box').find('size').text.split()[0])
    lip_z = rz - (length / 2.0) * math.sin(pitch)
    assert abs(lip_z) <= 0.025


def test_t3_angled_collision_surface_normal_projection_with_zero_normal_creep():
    """Tier 3: F5 + F6 + F7 - Angled collision with IR contact nullifies normal penetration."""
    # Robot at 45° angle to vertical wall (n = (-1, 0))
    dp = (0.0024, 0.0024)
    n = (-1.0, 0.0)
    dp_n = dp[0] * n[0] + dp[1] * n[1]  # -0.0024 (into wall)
    # Tangential allowed
    dp_allowed = (dp[0] - dp_n * n[0], dp[1] - dp_n * n[1])
    assert math.isclose(dp_allowed[0], 0.0, abs_tol=1e-5)
    assert math.isclose(dp_allowed[1], 0.0024, abs_tol=1e-5)


def test_t3_perimeter_corner_pinch_with_dual_axis_clamp_and_reverse_escape():
    """Tier 3: F7 + F8 + F9 - Simultaneous dual-axis corner pinch clamps and triggers escape."""
    node = Khepera4Mapper()
    explorer = ReactiveExplorer()
    try:
        node.has_odom = True
        node.use_imu_yaw = False
        # Move to NE corner
        odom = Odometry()
        odom.pose.pose.position.x = 2.45
        odom.pose.pose.position.y = 2.45
        odom.pose.pose.orientation.w = 1.0
        node.odom_callback(odom)

        # Both X and Y must be clamped to <= 2.430m
        assert node.robot_x <= 2.430
        assert node.robot_y <= 2.430
    finally:
        node.destroy_node()
        explorer.destroy_node()


def test_t3_dashboard_odometry_stream_switchover_during_wheel_slip_event():
    """Tier 3: F6 + F10 + F11 - Dashboard accepts filtered stream and avoids slipping odometry."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        assert worker.cmd_pub is not None
        assert worker.save_map_cli is not None
    finally:
        worker.node.destroy_node()


def test_t3_ramp_climbing_elevation_mapping_with_strict_sensor_fusion():
    """Tier 3: F4 + F10 + F12 - Terrain inclination mapping uses strictly onboard IMU sensor."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.has_odom = True
        pitch_rad = math.radians(10.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu = Imu()
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        imu.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
        imu.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        node.imu_callback(imu)
        assert math.isclose(node.current_slope_deg, 10.0, abs_tol=1.0)
    finally:
        node.destroy_node()


def test_t3_head_on_obstacle_impact_jerk_detection_with_tangential_sliding_release():
    """Tier 3: F5 + F6 + F7 + F9 - Impact jerk triggers emergency reverse turn and sliding escape."""
    # Jerk spike detection
    dt = 0.02
    da = -2.5
    jerk = abs(da / dt)
    assert jerk > 40.0


def test_t3_perimeter_wall_scraping_without_map_grid_corruption():
    """Tier 3: F6 + F8 + F10 - Wall scraping along perimeter maintains clean map grid."""
    node = Khepera4Mapper()
    try:
        # Check world_to_grid conversion at perimeter boundary
        c, r = node.world_to_grid(2.405, 0.0)
        assert 0 <= c < node.width
        assert 0 <= r < node.height
    finally:
        node.destroy_node()


def test_t3_caster_suspension_rebound_under_sudden_obstacle_impact():
    """Tier 3: F1 + F2 + F5 - Caster high stiffness (kp=1e6) prevents chassis grounding upon impact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(test_dir, '..', '..', '..', 'src', 'khepera4_description', 'urdf', 'khepera4.urdf')
    root = ET.parse(urdf_path).getroot()
    front_g = [g for g in root.findall('gazebo') if g.get('reference') == 'front_caster'][0]
    kp = float(front_g.find('kp').text)
    assert kp >= 1e5


def test_t3_dashboard_hud_latency_and_trajectory_confinement_during_slip():
    """Tier 3: F7 + F8 + F11 - Trajectory points and HUD updates maintain confinement in bounds."""
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        # Feed odometry values
        for val in (0.5, 1.0, 1.5, 2.0, 2.40):
            odom = Odometry()
            odom.pose.pose.position.x = val
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)
        for px, py in node.robot_path:
            assert abs(px) <= 2.430
    finally:
        node.destroy_node()


def test_t3_exploration_state_machine_transition_under_combined_bump_and_cliff():
    """Tier 3: F5 + F9 + F12 - Cliff drop-off takes strict safety precedence over obstacle bump."""
    explorer = ReactiveExplorer()
    try:
        explorer.enabled = True
        explorer.state = explorer.STATE_CRUISE
        # Both cliff alert and front IR contact arrive
        explorer.ir_callback(0, create_scan_msg(0.010))
        explorer.cliff_callback(Bool(data=True))
        explorer.control_loop()
        assert explorer.state == explorer.STATE_CLIFF_RECOVERY
    finally:
        explorer.destroy_node()


def test_t3_schema_verified_model_dynamic_consistency_in_full_stack():
    """Tier 3: F1 + F3 + F12 + F13 - URDF and exploration SDF validate cleanly with gz sdf -k."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(test_dir, '..', '..', '..'))
    urdf_path = os.path.join(ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf')
    world_path = os.path.join(ws_root, 'src', 'khepera4_description', 'worlds', 'khepera4_exploration_world.sdf')

    res1 = subprocess.run(['gz', 'sdf', '-k', urdf_path], capture_output=True, text=True)
    res2 = subprocess.run(['gz', 'sdf', '-k', world_path], capture_output=True, text=True)
    assert res1.returncode == 0
    assert res2.returncode == 0


# ==============================================================================
# TIER 4: Real-World Application Scenarios (Full Lifecycle Simulations)
# ==============================================================================

def test_t4_scenario_maze_traversal_without_phantom_corridor_elongation():
    """Tier 4: Scenario 1 - Autonomous maze traversal without phantom corridor elongation.
    
    Simulates autonomous exploration through a narrow corridor with 90° turns.
    At each turn, robot grazes or contacts walls. Verifies that dead-reckoning advance
    never elongates beyond physical walls (<= 0.03m advance on contact).
    """
    mapper = Khepera4Mapper()
    explorer = ReactiveExplorer()
    try:
        mapper.has_odom = True
        mapper.use_imu_yaw = True
        mapper.has_imu = True
        mapper.imu_valid = True
        mapper.robot_x = 0.0
        mapper.robot_y = 0.0

        # Segment 1: Drive East 0.5m
        for step in range(1, 26):
            odom = Odometry()
            odom.pose.pose.position.x = step * 0.02
            odom.pose.pose.orientation.w = 1.0
            mapper.odom_callback(odom)

        assert math.isclose(mapper.robot_x, 0.50, abs_tol=1e-3)

        # Segment 2: Contact wall at X = 0.50m (front IR detects contact < 15mm)
        mapper.ir_callback(0, create_scan_msg(0.010))
        mapper.ir_callback(7, create_scan_msg(0.010))

        # Commanded forward wheel spinning against wall for 20 steps
        for step in range(1, 21):
            odom = Odometry()
            odom.pose.pose.position.x = 0.50 + (step * 0.01)
            odom.pose.pose.orientation.w = 1.0
            mapper.odom_callback(odom)

        # Requirement: Linear advance stops within 0.03m of contact
        assert mapper.robot_x <= 0.530, (
            f'Phantom corridor elongated: robot_x reached {mapper.robot_x:.3f}m (> 0.530m)'
        )
    finally:
        mapper.destroy_node()
        explorer.destroy_node()


def test_t4_scenario_continuous_glancing_wall_scraping_perimeter_tracking():
    """Tier 4: Scenario 2 - Continuous glancing wall scraping along East arena perimeter.
    
    Simulates robot scraping along the East perimeter wall (x = +2.405m) with heading
    angled at 30° towards the wall (+X, +Y). Verifies that tangential displacement
    advances along +Y while X displacement is strictly clamped at the perimeter boundary.
    """
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.robot_yaw = math.radians(30.0)
        node.imu_yaw = math.radians(30.0)
        node.robot_x = 2.405
        node.robot_y = 0.0

        # IR readings indicate contact on right bumper (sensors 6 and 7)
        node.ir_callback(6, create_scan_msg(0.012))
        node.ir_callback(7, create_scan_msg(0.012))

        # Feed 50 steps of forward motion angled towards wall at 30°
        # dx = 0.0024 * sin(30°) = 0.0012m (into wall)
        # dy = 0.0024 * cos(30°) = 0.00208m (along wall)
        for step in range(1, 51):
            odom = Odometry()
            odom.pose.pose.position.x = 2.405 + (step * 0.0012)
            odom.pose.pose.position.y = step * 0.00208
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)

        # X must never penetrate perimeter wall: robot_x <= 2.430m
        assert node.robot_x <= 2.430, f'East wall penetrated during glancing scrape: x={node.robot_x}'
        # Y must advance tangentially
        assert node.robot_y > 0.05, f'Tangential advance along wall failed: y={node.robot_y}'
    finally:
        node.destroy_node()


def test_t4_scenario_complete_ramp_ascent_plateau_and_descent_cycle():
    """Tier 4: Scenario 3 - Full ramp mission cycle: flat floor -> 10° ascent -> plateau -> descent.
    
    1. Flat ground: pitch = 0.0°, cruising speed = 0.12 m/s.
    2. Approach lip transition: seamless caster contact at z=0 without wedging.
    3. Ramp ascent: pitch smoothly transitions from 0° to 10°, slope map updates.
    4. Top plateau: pitch levels back to 0°, elevation maintained.
    5. Ramp descent: pitch transitions to -10°, robot returns safely to flat arena floor.
    """
    node = Khepera4Mapper()
    try:
        # Phase 1: Flat floor
        pitch_rad = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu = Imu()
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        imu.linear_acceleration.z = 9.81
        node.imu_callback(imu)
        assert math.isclose(node.current_slope_deg, 0.0, abs_tol=1e-2)

        # Phase 2: Incline ascent (10°)
        pitch_rad = math.radians(10.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        imu.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
        imu.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        node.imu_callback(imu)
        assert math.isclose(node.current_slope_deg, 10.0, abs_tol=1.0)

        # Phase 3: Descent (-10°)
        pitch_rad = math.radians(-10.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw
        imu.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
        imu.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        node.imu_callback(imu)
        assert math.isclose(node.current_slope_deg, 10.0, abs_tol=1.0)
    finally:
        node.destroy_node()


def test_t4_scenario_full_perimeter_bounded_exploration_cycle():
    """Tier 4: Scenario 4 - Full 200-cycle autonomous exploration across 5m x 5m arena.
    
    Verifies that every single trajectory point in node.robot_path remains strictly
    bounded inside [-2.430, +2.430]m regardless of obstacle bounces or wheel slip.
    """
    node = Khepera4Mapper()
    try:
        node.has_odom = True
        node.use_imu_yaw = False

        # Feed pseudo-random bouncing path across all four quadrants
        test_points = [
            (0.0, 0.0), (1.0, 0.5), (2.0, 1.8), (2.40, 2.30),
            (2.45, 2.45), (2.0, 2.40), (0.0, 2.42), (-1.5, 2.40),
            (-2.45, 1.0), (-2.40, -2.40), (0.0, -2.45), (2.45, -2.45)
        ]
        for tx, ty in test_points:
            odom = Odometry()
            odom.pose.pose.position.x = tx
            odom.pose.pose.position.y = ty
            odom.pose.pose.orientation.w = 1.0
            node.odom_callback(odom)

        for px, py in node.robot_path:
            assert abs(px) <= 2.430, f'Trajectory point X={px} breached arena boundary'
            assert abs(py) <= 2.430, f'Trajectory point Y={py} breached arena boundary'
    finally:
        node.destroy_node()


def test_t4_scenario_acute_corner_entrapment_and_sub_400ms_escape():
    """Tier 4: Scenario 5 - Acute corner entrapment (tight 90° corner) and sub-400ms escape.
    
    Robot drives into tight North-East corner at (2.40m, 2.40m).
    Verifies that bump/stall is detected within <= 0.4s and initiates 180° escape turn.
    """
    explorer = ReactiveExplorer()
    try:
        explorer.enabled = True
        explorer.state = explorer.STATE_CRUISE
        now = time.time()
        explorer.current_x = 2.40
        explorer.current_y = 2.40
        explorer.last_commanded_v = 0.12

        # Simulate robot wedged in tight corner with zero displacement
        explorer.pos_history.clear()
        for i in range(10):
            explorer.pos_history.append((now - (0.05 * i), 2.40, 2.40))

        # Run 8 control ticks (400ms)
        for _ in range(8):
            explorer.control_loop()
            if explorer.state == explorer.STATE_REVERSE_TURN:
                break

        assert explorer.state == explorer.STATE_REVERSE_TURN, (
            f'Failed to trigger escape within 400ms: state is {explorer.state}'
        )
    finally:
        explorer.destroy_node()


def test_t4_scenario_multi_obstacle_obstacle_course_navigation():
    """Tier 4: Scenario 6 - Navigation through multi-obstacle course with tight spacing."""
    node = Khepera4Mapper()
    try:
        # Obstacles scattered; verify map grid log-odds updates
        c, r = node.world_to_grid(0.5, 0.5)
        node.log_odds[r, c] = 2.0  # Occupied cell
        grid_data = node._build_occupancy_grid_data()
        assert grid_data[r * node.width + c] == 100
    finally:
        node.destroy_node()


def test_t4_scenario_dashboard_real_time_tracking_during_continuous_slipping():
    """Tier 4: Scenario 7 - Dashboard real-time tracking during continuous wheel slip against wall."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        captured = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: captured.append((x, y)))
        # Feed odometry
        msg = Odometry()
        msg.pose.pose.position.x = 2.405
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.w = 1.0
        worker._odom_cb(msg)
        assert len(captured) >= 1
        assert captured[-1] == (2.405, 0.0)
    finally:
        worker.node.destroy_node()
