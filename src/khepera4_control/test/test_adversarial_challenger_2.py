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
Adversarial Stress Test Harness — Challenger 2.

Tests:
1. Surface Normal Derivation & Tangential Slip Projection.
2. Evasive Reaction Latency under bump, stall, and packet loss.
3. Dashboard Odometry Stream Switching and timeout fallback.
"""

import math
import os
import random
import sys
import time

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from khepera4_control.common import quaternion_from_euler  # noqa: E402
from khepera4_control.khepera4_mapper import IR_MOUNTS, Khepera4Mapper  # noqa: E402
from khepera4_control.mapping_dashboard import MappingRosWorker  # noqa: E402
from khepera4_control.reactive_explorer import ReactiveExplorer  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402


# ==============================================================================
# Fixtures & Helpers
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_scan_msg(distance: float,
                    min_range: float = 0.002,
                    max_range: float = 0.25) -> LaserScan:
    """Create a single-ray LaserScan message."""
    scan = LaserScan()
    scan.ranges = [float(distance)]
    scan.range_min = float(min_range)
    scan.range_max = float(max_range)
    return scan


# ==============================================================================
# SECTION 1: Adversarial Surface Normal Derivation Stress-Tests
# ==============================================================================

def test_adv_normal_derivation_gaussian_noise():
    """Verify normal derivation under Gaussian noise (sigma=3mm, 6mm, 10mm)."""
    random.seed(42)
    node = Khepera4Mapper()
    try:
        node.robot_yaw = 0.0
        base_dist = 0.020

        for sigma in (0.003, 0.006, 0.010):
            for trial in range(10):
                for i in range(8):
                    if i in (0, 7):
                        noise = random.gauss(0.0, sigma)
                        dist = max(0.005, min(0.038, base_dist + noise))
                        node.ir_callback(i, create_scan_msg(dist))
                    else:
                        node.ir_callback(i, create_scan_msg(0.200))

                normal = node._compute_surface_normal()
                assert normal is not None, (
                    f'Failed to compute normal at sigma={sigma}, trial={trial}'
                )
                nx, ny = normal
                norm_len = math.hypot(nx, ny)
                assert math.isclose(norm_len, 1.0, abs_tol=1e-5), (
                    f'Normal not unit length: {norm_len}'
                )
                assert nx < -0.5, (
                    f'Normal x={nx} should point out of wall toward robot (-x)'
                )
    finally:
        node.destroy_node()


def test_adv_normal_derivation_pathological_sensor_noise():
    """Verify handling of pathological sensor inputs (NaN, Inf, negative, out-of-range)."""
    node = Khepera4Mapper()
    try:
        node.robot_yaw = 0.0
        node.ir_callback(0, create_scan_msg(float('nan')))
        node.ir_callback(1, create_scan_msg(float('inf')))
        node.ir_callback(2, create_scan_msg(-0.05))
        node.ir_callback(3, create_scan_msg(10.0))
        node.ir_callback(4, None)
        node.ir_callback(7, create_scan_msg(0.012))

        normal = node._compute_surface_normal()
        assert normal is not None, 'Normal computation failed despite 1 valid contact reading'
        nx, ny = normal
        assert not math.isnan(nx) and not math.isnan(ny)
        assert math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-5)
    finally:
        node.destroy_node()


def test_adv_normal_derivation_extreme_sparsity():
    """Verify surface normal calculation under extreme sensor sparsity."""
    node = Khepera4Mapper()
    try:
        node.robot_yaw = 0.0
        for active_idx in (0, 1, 2, 6, 7):
            for i in range(8):
                if i == active_idx:
                    node.ir_callback(i, create_scan_msg(0.015))
                else:
                    node.ir_callback(i, create_scan_msg(0.220))

            normal = node._compute_surface_normal()
            assert normal is not None, f'Failed with single sensor ir_{active_idx}'
            nx, ny = normal
            assert math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-5)
            _, _, syaw = IR_MOUNTS[active_idx]
            expected_nx = -math.cos(syaw)
            expected_ny = -math.sin(syaw)
            dot = nx * expected_nx + ny * expected_ny
            assert dot > 0.90, (
                f'Normal ({nx}, {ny}) diverges from expected ({expected_nx}, {expected_ny})'
            )
    finally:
        node.destroy_node()


def test_adv_normal_derivation_extreme_asymmetry():
    """Verify surface normal under asymmetric contact (left vs right flank scrape)."""
    node = Khepera4Mapper()
    try:
        node.robot_yaw = 0.0
        node.ir_callback(1, create_scan_msg(0.010))
        node.ir_callback(2, create_scan_msg(0.010))
        for i in (0, 3, 4, 5, 6, 7):
            node.ir_callback(i, create_scan_msg(0.200))

        normal_left = node._compute_surface_normal()
        assert normal_left is not None
        assert normal_left[1] < -0.3, f'Expected ny < 0 for left wall, got {normal_left[1]}'

        node.ir_callback(1, create_scan_msg(0.200))
        node.ir_callback(2, create_scan_msg(0.200))
        node.ir_callback(5, create_scan_msg(0.010))
        node.ir_callback(6, create_scan_msg(0.010))

        normal_right = node._compute_surface_normal()
        assert normal_right is not None
        assert normal_right[1] > 0.3, f'Expected ny > 0 for right wall, got {normal_right[1]}'
    finally:
        node.destroy_node()


# ==============================================================================
# SECTION 2: Glancing Collisions across Contact Angles (15°, 30°, 45°, 60°, 75°, 90°)
# ==============================================================================

@pytest.mark.parametrize('contact_angle_deg', [15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
def test_adv_glancing_collision_perpendicular_penetration_nullified(contact_angle_deg):
    """Verify perpendicular penetration is nullified and motion is tangential."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.has_odom = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.last_odom_x = 0.0
        node.last_odom_y = 0.0
        node.last_odom_yaw = 0.0
        node.cmd_linear_x = 0.12

        theta_rad = math.radians(contact_angle_deg)
        nx = -math.sin(theta_rad)
        ny = math.cos(theta_rad)

        node._compute_surface_normal = lambda: (nx, ny)
        node.ir_callback(0, create_scan_msg(0.010))

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = 0.005
        odom_msg.pose.pose.position.y = 0.000
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = 0.12

        node.odom_callback(odom_msg)

        delta_px = node.robot_x - 0.0
        delta_py = node.robot_y - 0.0

        p_perp = -(delta_px * nx + delta_py * ny)
        assert p_perp <= 1e-6, (
            f'Perpendicular penetration not nullified at {contact_angle_deg}°: '
            f'p_perp = {p_perp:.8f}m, delta_p = ({delta_px:.6f}, {delta_py:.6f})'
        )

        tx = -ny
        ty = nx
        if contact_angle_deg < 80.0:
            proj_tangent = delta_px * tx + delta_py * ty
            assert abs(proj_tangent) > 0.0, (
                f'Expected non-zero tangential displacement at {contact_angle_deg}°'
            )
            cross = abs(delta_px * ty - delta_py * tx)
            assert cross <= 1e-6, (
                f'Motion not aligned with surface tangent at {contact_angle_deg}°, cross={cross}'
            )
        else:
            assert math.hypot(delta_px, delta_py) <= 1e-5, (
                f'Head-on collision at {contact_angle_deg}° did not clamp advance: '
                f'disp={math.hypot(delta_px, delta_py)}'
            )
    finally:
        node.destroy_node()


@pytest.mark.parametrize('robot_yaw_deg', [0.0, 45.0, 90.0, 135.0, 180.0, 270.0])
def test_adv_glancing_multi_robot_yaw_projection(robot_yaw_deg):
    """Verify tangential projection across various robot orientations."""
    node = Khepera4Mapper()
    try:
        yaw_rad = math.radians(robot_yaw_deg)
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = yaw_rad
        node.has_odom = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = yaw_rad
        node.last_odom_x = 0.0
        node.last_odom_y = 0.0
        node.last_odom_yaw = yaw_rad
        node.cmd_linear_x = 0.12

        nx_world = -1.0
        ny_world = 0.0
        node._compute_surface_normal = lambda: (nx_world, ny_world)
        node.ir_callback(0, create_scan_msg(0.010))

        step = 0.005
        dx_odom = step * math.cos(yaw_rad)
        dy_odom = step * math.sin(yaw_rad)

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = dx_odom
        odom_msg.pose.pose.position.y = dy_odom
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_rad)
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = 0.12

        node.odom_callback(odom_msg)

        delta_px = node.robot_x - 0.0
        delta_py = node.robot_y - 0.0

        p_perp = -(delta_px * nx_world + delta_py * ny_world)
        assert p_perp <= 1e-6, (
            f'Penetration detected at yaw {robot_yaw_deg}°: p_perp = {p_perp:.8f}m'
        )

        if math.cos(yaw_rad) > 0.05:
            assert abs(delta_px) <= 1e-6, (
                f'Wall penetration in x at yaw {robot_yaw_deg}°: delta_px = {delta_px}'
            )
            if abs(math.sin(yaw_rad)) > 0.1:
                assert abs(delta_py) > 0.0, (
                    f'Expected tangential motion along y at yaw {robot_yaw_deg}°'
                )
        else:
            assert delta_px <= 0.0, (
                f'Expected motion away from wall at yaw {robot_yaw_deg}°, got {delta_px}'
            )
    finally:
        node.destroy_node()


def test_adv_glancing_collision_reverse_motion_unhindered():
    """Verify that reverse motion away from a contacted wall is preserved unhindered."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.has_odom = True
        node.has_imu = True
        node.imu_valid = True
        node.last_odom_x = 0.0
        node.last_odom_y = 0.0
        node.last_odom_yaw = 0.0
        node.cmd_linear_x = -0.05

        node._compute_surface_normal = lambda: (-1.0, 0.0)
        node.ir_callback(0, create_scan_msg(0.010))

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = -0.005
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = -0.05

        node.odom_callback(odom_msg)

        assert math.isclose(node.robot_x, -0.005, abs_tol=1e-5), (
            f'Reverse motion was hindered: x={node.robot_x}'
        )
        assert math.isclose(node.robot_y, 0.0, abs_tol=1e-5)
    finally:
        node.destroy_node()


def test_adv_head_on_stall_advance_stops_within_30mm():
    """Verify robot advance stops within <= 0.03m of initial contact during head-on stall."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.has_odom = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0
        node.last_odom_x = 0.0
        node.last_odom_y = 0.0
        node.last_odom_yaw = 0.0
        node.cmd_linear_x = 0.12

        node.ir_callback(0, create_scan_msg(0.020))
        node.ir_callback(7, create_scan_msg(0.020))

        start_x = node.robot_x

        for step in range(1, 101):
            raw_odom_x = step * 0.0024
            node.ir_callback(0, create_scan_msg(0.010))
            node.ir_callback(7, create_scan_msg(0.010))

            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = raw_odom_x
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.orientation.w = 1.0
            odom_msg.twist.twist.linear.x = 0.12

            node.odom_callback(odom_msg)

        total_advance = node.robot_x - start_x
        assert total_advance <= 0.030, (
            f'Head-on stall advance exceeded 0.03m limit: total_advance = {total_advance:.4f}m '
            f'(raw uncorrected wheel slip was {raw_odom_x:.4f}m)'
        )
        assert total_advance >= 0.0, f'Negative advance: {total_advance}'
    finally:
        node.destroy_node()


def test_adv_velocity_scaling_under_glancing_and_head_on():
    """Verify published twist velocity on /khepera4/odom scales with allowed motion."""
    node = Khepera4Mapper()
    try:
        published_msgs = []
        node.fused_odom_pub.publish = lambda msg: published_msgs.append(msg)

        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.has_odom = True
        node.has_imu = True
        node.imu_valid = True
        node.last_odom_x = 0.0
        node.last_odom_y = 0.0
        node.last_odom_yaw = 0.0
        node.cmd_linear_x = 0.12

        node._compute_surface_normal = lambda: (-1.0, 0.0)
        node.ir_callback(0, create_scan_msg(0.010))

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = 0.005
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = 0.12

        node.odom_callback(odom_msg)
        assert len(published_msgs) == 1
        assert math.isclose(published_msgs[-1].twist.twist.linear.x, 0.0, abs_tol=1e-5)

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        node._compute_surface_normal = lambda: (-inv_sqrt2, inv_sqrt2)
        node.last_odom_x = 0.005
        node.last_odom_y = 0.0

        odom_msg2 = Odometry()
        odom_msg2.pose.pose.position.x = 0.010
        odom_msg2.pose.pose.position.y = 0.0
        odom_msg2.pose.pose.orientation.w = 1.0
        odom_msg2.twist.twist.linear.x = 0.12

        node.odom_callback(odom_msg2)
        assert len(published_msgs) == 2
        expected_vx = 0.12 * inv_sqrt2
        actual_vx = published_msgs[-1].twist.twist.linear.x
        assert math.isclose(actual_vx, expected_vx, abs_tol=0.01), (
            f'Unexpected scaled vx: {actual_vx}'
        )
    finally:
        node.destroy_node()


# ==============================================================================
# SECTION 3: Evasive Reaction Latency Stress-Tests (<= 0.4s)
# ==============================================================================

def test_adv_evasion_latency_immediate_ir_contact():
    """Verify near-contact IR proximity switches to STATE_REVERSE_TURN in <= 0.05s."""
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.last_commanded_v = 0.12

        explorer.ir_callback(0, create_scan_msg(0.012))

        start_t = time.time()
        explorer.control_loop()
        elapsed = time.time() - start_t

        assert explorer.state == ReactiveExplorer.STATE_REVERSE_TURN
        assert elapsed < 0.05, f'Reaction took {elapsed:.4f}s > 0.05s'
        assert explorer.last_commanded_v < 0.0
    finally:
        explorer.destroy_node()


def test_adv_evasion_latency_imu_impact_shock():
    """Verify IMU deceleration/jerk shock switches to STATE_REVERSE_TURN in <= 0.05s."""
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.last_commanded_v = 0.12

        imu_msg = Imu()
        imu_msg.header.stamp.sec = 1
        imu_msg.header.stamp.nanosec = 0
        imu_msg.linear_acceleration.x = -2.5
        imu_msg.orientation.w = 1.0

        explorer.imu_callback(imu_msg)
        assert explorer.imu_impact_detected is True

        start_t = time.time()
        explorer.control_loop()
        elapsed = time.time() - start_t

        assert explorer.state == ReactiveExplorer.STATE_REVERSE_TURN
        assert elapsed < 0.05
        assert explorer.last_commanded_v < 0.0
    finally:
        explorer.destroy_node()


@pytest.mark.parametrize('collision_type', ['frontal_stall', 'corner_trap', 'pure_slip'])
def test_adv_evasion_latency_corner_and_slip_stall_under_0_4s(collision_type):
    """Verify bump/stall/slip detection switches to reverse turn within <= 0.4s."""
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.last_commanded_v = 0.12
        sim_time = time.time()

        dt = 0.05
        switched = False
        switch_time = None

        for tick in range(12):
            t_curr = sim_time + tick * dt

            if collision_type == 'frontal_stall':
                explorer.ir_callback(0, create_scan_msg(0.025))
            elif collision_type == 'corner_trap':
                explorer.ir_callback(1, create_scan_msg(0.020))
                explorer.us_callback(1, create_scan_msg(0.040, max_range=2.0))
            elif collision_type == 'pure_slip':
                explorer.ir_callback(0, create_scan_msg(0.030))

            odom = Odometry()
            odom.pose.pose.position.x = 1.0
            odom.pose.pose.position.y = 1.0
            odom.pose.pose.orientation.w = 1.0
            explorer.pos_history.append((t_curr, 1.0, 1.0))
            explorer.current_x = 1.0
            explorer.current_y = 1.0

            saved_time = time.time
            time.time = lambda: t_curr
            try:
                explorer.control_loop()
            finally:
                time.time = saved_time

            if explorer.state == ReactiveExplorer.STATE_REVERSE_TURN and not switched:
                switched = True
                switch_time = tick * dt
                break

        assert switched is True, f'Failed to switch to REVERSE_TURN under {collision_type}'
        assert switch_time <= 0.40, (
            f'Evasion latency {switch_time:.3f}s exceeded 0.4s requirement under {collision_type}'
        )
    finally:
        explorer.destroy_node()


def test_adv_evasion_latency_under_sensor_packet_loss():
    """Verify stall evasion triggers <= 0.4s under 50% sensor packet drop."""
    random.seed(99)
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.last_commanded_v = 0.12
        sim_time = time.time()
        dt = 0.05
        switched = False
        switch_time = None

        for tick in range(12):
            t_curr = sim_time + tick * dt
            if random.random() > 0.5:
                explorer.ir_callback(0, create_scan_msg(0.025))

            explorer.pos_history.append((t_curr, 0.5, 0.5))
            explorer.current_x = 0.5
            explorer.current_y = 0.5

            saved_time = time.time
            time.time = lambda: t_curr
            try:
                explorer.control_loop()
            finally:
                time.time = saved_time

            if explorer.state == ReactiveExplorer.STATE_REVERSE_TURN:
                switched = True
                switch_time = tick * dt
                break

        assert switched is True, 'Failed to escape stall under 50% packet loss'
        assert switch_time <= 0.40, f'Latency {switch_time}s exceeded 0.4s under packet loss'
    finally:
        explorer.destroy_node()


def test_adv_evasive_turn_direction_correctness():
    """Verify evasive reverse-turn steers away from contacting obstacle."""
    explorer = ReactiveExplorer()
    try:
        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.ir_callback(1, create_scan_msg(0.010))
        explorer.control_loop()
        assert explorer.state == ReactiveExplorer.STATE_REVERSE_TURN
        assert explorer.escape_turn_dir == -1.0, (
            f'Left contact must steer right (-1.0), got {explorer.escape_turn_dir}'
        )

        explorer.state = ReactiveExplorer.STATE_CRUISE
        explorer.ir_callback(1, create_scan_msg(0.200))
        explorer.ir_callback(6, create_scan_msg(0.010))
        explorer.control_loop()
        assert explorer.state == ReactiveExplorer.STATE_REVERSE_TURN
        assert explorer.escape_turn_dir == 1.0, (
            f'Right contact must steer left (+1.0), got {explorer.escape_turn_dir}'
        )
    finally:
        explorer.destroy_node()


# ==============================================================================
# SECTION 4: Dashboard Odometry Stream Switching Stress-Tests (R3)
# ==============================================================================

def test_adv_dashboard_stream_prioritization_and_isolation():
    """Verify that raw /odom is strictly isolated when /khepera4/odom is active."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        emitted_poses = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: emitted_poses.append((x, y)))

        fused_init = Odometry()
        fused_init.pose.pose.position.x = 2.40
        fused_init.pose.pose.position.y = 2.40
        fused_init.pose.pose.orientation.w = 1.0
        worker._fused_odom_cb(fused_init)
        assert worker.using_fused_odom is True

        for i in range(50):
            raw_msg = Odometry()
            raw_msg.pose.pose.position.x = 2.80 + i * 0.01
            raw_msg.pose.pose.position.y = 2.80 + i * 0.01
            raw_msg.pose.pose.orientation.w = 1.0
            worker._odom_cb(raw_msg)

            fused_msg = Odometry()
            fused_msg.pose.pose.position.x = 2.40
            fused_msg.pose.pose.position.y = 2.40
            fused_msg.pose.pose.orientation.w = 1.0
            worker._fused_odom_cb(fused_msg)

        for x, y in emitted_poses:
            assert x <= 2.405 and y <= 2.405, (
                f'Raw slipping odometry leaked into dashboard! Pose: ({x}, {y})'
            )
        assert len(emitted_poses) == 51
    finally:
        worker.node.destroy_node()


def test_adv_dashboard_cold_start_fallback_then_fused_takeover():
    """Verify clean transition from cold start raw fallback to fused priority."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        emitted_poses = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: emitted_poses.append((x, y)))

        assert worker.using_fused_odom is False
        raw_msg_1 = Odometry()
        raw_msg_1.pose.pose.position.x = 0.01
        raw_msg_1.pose.pose.position.y = 0.01
        raw_msg_1.pose.pose.orientation.w = 1.0
        worker._odom_cb(raw_msg_1)
        assert (0.01, 0.01) in emitted_poses

        fused_msg_1 = Odometry()
        fused_msg_1.pose.pose.position.x = 0.02
        fused_msg_1.pose.pose.position.y = 0.02
        fused_msg_1.pose.pose.orientation.w = 1.0
        worker._fused_odom_cb(fused_msg_1)
        assert (0.02, 0.02) in emitted_poses
        assert worker.using_fused_odom is True

        raw_msg_slip = Odometry()
        raw_msg_slip.pose.pose.position.x = 2.99
        raw_msg_slip.pose.pose.position.y = 2.99
        raw_msg_slip.pose.pose.orientation.w = 1.0
        worker._odom_cb(raw_msg_slip)
        assert (2.99, 2.99) not in emitted_poses
    finally:
        worker.node.destroy_node()


def test_adv_dashboard_stream_dropping_and_timeout_fallback():
    """Verify timeout fallback to raw /odom when /khepera4/odom stops publishing."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        emitted_poses = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: emitted_poses.append((x, y)))

        t0 = time.time()
        fused_msg = Odometry()
        fused_msg.pose.pose.position.x = 1.0
        fused_msg.pose.pose.position.y = 1.0
        fused_msg.pose.pose.orientation.w = 1.0
        worker._fused_odom_cb(fused_msg)

        raw_msg_early = Odometry()
        raw_msg_early.pose.pose.position.x = 1.1
        raw_msg_early.pose.pose.position.y = 1.1
        raw_msg_early.pose.pose.orientation.w = 1.0

        worker.last_fused_odom_time = t0 - 0.5
        worker._odom_cb(raw_msg_early)
        assert (1.1, 1.1) not in emitted_poses, 'Raw odom accepted before 1.0s timeout!'

        worker.last_fused_odom_time = t0 - 1.2
        raw_msg_late = Odometry()
        raw_msg_late.pose.pose.position.x = 1.2
        raw_msg_late.pose.pose.position.y = 1.2
        raw_msg_late.pose.pose.orientation.w = 1.0
        worker._odom_cb(raw_msg_late)

        assert (1.2, 1.2) in emitted_poses, 'Raw odom not accepted after 1.0s timeout fallback'
    finally:
        worker.node.destroy_node()


def test_adv_dashboard_instantaneous_recovery_from_fallback():
    """Verify instantaneous recovery and suppression of raw /odom when fused stream resumes."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        emitted_poses = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: emitted_poses.append((x, y)))

        worker.last_fused_odom_time = time.time() - 2.0
        assert worker.using_fused_odom is False

        raw_msg_fallback = Odometry()
        raw_msg_fallback.pose.pose.position.x = 0.5
        raw_msg_fallback.pose.pose.position.y = 0.5
        raw_msg_fallback.pose.pose.orientation.w = 1.0
        worker._odom_cb(raw_msg_fallback)
        assert (0.5, 0.5) in emitted_poses

        fused_msg = Odometry()
        fused_msg.pose.pose.position.x = 0.6
        fused_msg.pose.pose.position.y = 0.6
        fused_msg.pose.pose.orientation.w = 1.0
        worker._fused_odom_cb(fused_msg)
        assert (0.6, 0.6) in emitted_poses
        assert worker.using_fused_odom is True

        raw_msg_blocked = Odometry()
        raw_msg_blocked.pose.pose.position.x = 0.99
        raw_msg_blocked.pose.pose.position.y = 0.99
        raw_msg_blocked.pose.pose.orientation.w = 1.0
        worker._odom_cb(raw_msg_blocked)

        assert (0.99, 0.99) not in emitted_poses, (
            'Raw odom was not immediately suppressed on fused stream recovery'
        )
    finally:
        worker.node.destroy_node()


def test_adv_dashboard_high_jitter_packet_loss_survival():
    """Verify raw /odom suppression under high packet jitter on /khepera4/odom."""
    worker = MappingRosWorker()
    worker.init_ros()
    try:
        emitted_poses = []
        worker.odom_received.connect(lambda x, y, yaw, v, w: emitted_poses.append((x, y)))

        now = time.time()
        for cycle in range(5):
            worker.last_fused_odom_time = now - 0.8
            fused_msg = Odometry()
            fused_msg.pose.pose.position.x = float(cycle)
            fused_msg.pose.pose.position.y = float(cycle)
            fused_msg.pose.pose.orientation.w = 1.0
            worker._fused_odom_cb(fused_msg)

            for _ in range(10):
                raw = Odometry()
                raw.pose.pose.position.x = 99.0
                raw.pose.pose.position.y = 99.0
                raw.pose.pose.orientation.w = 1.0
                worker._odom_cb(raw)

        for x, y in emitted_poses:
            assert x != 99.0 and y != 99.0, (
                f'Raw odom penetrated during jitter cycle: ({x}, {y})'
            )
        assert len(emitted_poses) == 5
    finally:
        worker.node.destroy_node()
