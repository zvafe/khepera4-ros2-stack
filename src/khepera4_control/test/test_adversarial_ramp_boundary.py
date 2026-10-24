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
Adversarial Stress Test Suite: Ramp Climbing Physics (R2) & Boundary Clamping.

Created by Challenger 1 (challenger_1).
Adversarially challenges:
1. Mechanical clearance & caster contact dynamics across approach angles (0°, 15°, 30°, 45°).
2. Dynamic load compliance and underbody wedging margins under caster sinkage.
3. Knife-edge lip transition in SDF (verifies no vertical step > 0 at entrance).
4. Arena perimeter constraints under aggressive driving towards outer walls (x=2.5, y=2.5)
   and runaway velocities, verifying strict bounds [-2.405, +2.405] m.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from khepera4_control.khepera4_mapper import Khepera4Mapper
from khepera4_control.common import quaternion_from_euler, euler_from_quaternion


@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def get_workspace_paths():
    """Derive workspace paths for URDF and SDF files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(test_dir, '..', '..', '..'))
    urdf_path = os.path.join(ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf')
    world_sdf_path = os.path.join(ws_root, 'src', 'khepera4_description', 'worlds', 'khepera4_exploration_world.sdf')
    return urdf_path, world_sdf_path


# ==============================================================================
# SECTION 1: Adversarial 3D Kinematics — Mechanical Clearance & Oblique Approaches
# ==============================================================================

def simulate_ramp_approach_clearance(approach_angle_deg: float, caster_sinkage_m: float = 0.0):
    """
    Simulate full 3D multi-body contact equilibrium and compute minimum clearance
    for upper and lower chassis collision cylinders across ramp transition.
    """
    theta_ramp = math.radians(10.0)
    psi = math.radians(approach_angle_deg)

    # Contact points in robot frame (when touching floor at z=0):
    # Front caster (radius 5mm, center 0.0505, 0, 0.005) -> bottom z = caster_sinkage
    f_caster_pt = np.array([0.0505, 0.0, caster_sinkage_m])
    r_caster_pt = np.array([-0.0505, 0.0, caster_sinkage_m])
    l_wheel_pt = np.array([0.0, 0.0527, 0.0])
    r_wheel_pt = np.array([0.0, -0.0527, 0.0])
    contact_pts = [f_caster_pt, r_caster_pt, l_wheel_pt, r_wheel_pt]

    # Collision perimeter sample points (360 degrees)
    n_samples = 120
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    # Lower cylinder: R=50mm, bottom z=4.5mm
    lower_pts = np.array([[0.050 * np.cos(a), 0.050 * np.sin(a), 0.0045] for a in angles])
    # Upper cylinder: R=70mm, bottom z=14mm
    upper_pts = np.array([[0.070 * np.cos(a), 0.070 * np.sin(a), 0.0140] for a in angles])

    def terrain_z(X, Y):
        return np.maximum(0.0, X * np.tan(theta_ramp))

    min_clr_lower = float('inf')
    min_clr_upper = float('inf')

    # Sweep robot center along the approach heading from s = -0.12 m to +0.20 m
    for s in np.linspace(-0.12, 0.20, 160):
        x_nom = s * math.cos(psi)
        y_nom = s * math.sin(psi)

        def obj(vars_opt):
            z_c, pitch, roll = vars_opt
            cy, sy = np.cos(psi), np.sin(psi)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R_mat = Rz @ Ry @ Rx

            penalties = 0.0
            heights = []
            for cpt in contact_pts:
                p_w = np.array([x_nom, y_nom, z_c]) + R_mat @ cpt
                tz = terrain_z(p_w[0], p_w[1])
                diff = p_w[2] - tz
                if diff < 0:
                    penalties += 1e5 * (diff ** 2)
                heights.append(diff)
            return z_c + penalties + 10.0 * sum(h ** 2 for h in heights)

        res = minimize(obj, [terrain_z(x_nom, y_nom), 0.0, 0.0], method='Nelder-Mead')
        z_c, pitch, roll = res.x

        cy, sy = np.cos(psi), np.sin(psi)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        R_mat = (
            np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
            np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
            np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        )

        for pt in lower_pts:
            pw = np.array([x_nom, y_nom, z_c]) + R_mat @ pt
            clr = pw[2] - terrain_z(pw[0], pw[1])
            if clr < min_clr_lower:
                min_clr_lower = clr

        for pt in upper_pts:
            pw = np.array([x_nom, y_nom, z_c]) + R_mat @ pt
            clr = pw[2] - terrain_z(pw[0], pw[1])
            if clr < min_clr_upper:
                min_clr_upper = clr

    return min_clr_lower, min_clr_upper


@pytest.mark.parametrize('angle_deg', [0, 15, 30, 45])
def test_adversarial_mechanical_clearance_various_approach_angles(angle_deg):
    """
    Stress test approach at various angles (head-on 0°, oblique 15°, 30°, 45°).
    Verify that neither the upper bumper nor the recessed underbody ever touches the ramp.
    """
    clr_lower, clr_upper = simulate_ramp_approach_clearance(angle_deg, caster_sinkage_m=0.0)
    # Must clear terrain by at least 1.5 mm
    assert clr_lower >= 0.0018, f'At {angle_deg}°, lower underbody clearance {clr_lower*1000:.2f}mm < 1.8mm'
    assert clr_upper >= 0.0070, f'At {angle_deg}°, upper bumper clearance {clr_upper*1000:.2f}mm < 7.0mm'


@pytest.mark.parametrize('sinkage_mm', [0.2, 0.5, 1.0])
def test_adversarial_caster_dynamic_compliance_and_sinkage_margin(sinkage_mm):
    """
    Stress test dynamic load sinkage (up to 1.0 mm caster compliance).
    Verify that chassis lip cannot contact or wedge under dynamic load.
    """
    sinkage_m = sinkage_mm / 1000.0
    for angle in [0, 30, 45]:
        clr_lower, clr_upper = simulate_ramp_approach_clearance(angle, caster_sinkage_m=sinkage_m)
        assert clr_lower > 0.0010, (
            f'Under {sinkage_mm}mm sinkage at {angle}°, lower clearance {clr_lower*1000:.2f}mm is too small'
        )
        assert clr_upper > 0.0060, (
            f'Under {sinkage_mm}mm sinkage at {angle}°, upper clearance {clr_upper*1000:.2f}mm is too small'
        )


def test_adversarial_caster_dynamic_parameters_urdf():
    """Verify caster stiffness kp, damping kd, and minDepth in URDF prevent excessive sinkage."""
    urdf_path, _ = get_workspace_paths()
    root = ET.parse(urdf_path).getroot()
    gazebo_tags = [g for g in root.findall('gazebo') if g.get('reference') == 'front_caster']
    assert len(gazebo_tags) >= 1
    g = gazebo_tags[0]
    kp = float(g.find('kp').text)
    kd = float(g.find('kd').text)
    min_depth = float(g.find('minDepth').text)
    mu1 = float(g.find('mu1').text)

    # Robot weight ~ 5.3 N. Elastic sinkage = 5.3 / kp
    elastic_sinkage_m = 5.3 / kp
    assert elastic_sinkage_m <= 1e-5, f'Elastic sinkage {elastic_sinkage_m*1e6:.2f} um exceeds 10 um'
    assert min_depth <= 0.0005, f'minDepth {min_depth}m exceeds 0.5mm'
    assert mu1 <= 0.001, f'Caster rolling friction {mu1} exceeds 0.001'


# ==============================================================================
# SECTION 2: Knife-Edge Lip Transition in World SDF
# ==============================================================================

def test_adversarial_inclined_ramp_knife_edge_lip_no_step():
    """
    Verify inclined_ramp in exploration world SDF meets floor with seamless knife-edge.
    Assert entrance lip top surface height Z_lip <= 0.0 m (strictly no vertical step > 0).
    """
    _, world_sdf_path = get_workspace_paths()
    root = ET.parse(world_sdf_path).getroot()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    pose_vals = [float(v) for v in ramp.find('pose').text.split()]
    x0, y0, z0, roll, pitch, yaw = pose_vals

    col = ramp.find('link').find('collision')
    box_size = [float(v) for v in col.find('geometry').find('box').find('size').text.split()]
    length, width, thickness = box_size

    center = np.array([x0, y0, z0])
    rot = R.from_euler('xyz', [roll, pitch, yaw])

    # In local frame, entrance is at x = +length/2 because pitch > 0 rotates +X downwards
    local_entrance_top = np.array([length / 2.0, 0.0, thickness / 2.0])
    local_entrance_bot = np.array([length / 2.0, 0.0, -thickness / 2.0])

    world_entrance_top = center + rot.apply(local_entrance_top)
    world_entrance_bot = center + rot.apply(local_entrance_bot)

    # Top surface height at entrance must be <= 0.0 m (no step > 0)
    assert world_entrance_top[2] <= 0.0001, (
        f'Ramp entrance top height {world_entrance_top[2]:.4f}m forms an obstructive vertical step > 0'
    )
    # The bottom surface must be underground
    assert world_entrance_bot[2] < 0.0


def test_adversarial_raised_platform_approach_ramp_knife_edge():
    """Verify raised_platform approach_ramp meets floor with knife-edge (Z <= 0)."""
    _, world_sdf_path = get_workspace_paths()
    root = ET.parse(world_sdf_path).getroot()
    plat = [m for m in root.find('world').findall('model') if m.get('name') == 'raised_platform'][0]
    ramp_link = [l for l in plat.findall('link') if l.get('name') == 'approach_ramp_link'][0]
    pose_vals = [float(v) for v in ramp_link.find('pose').text.split()]
    x0, y0, z0, roll, pitch, yaw = pose_vals

    col = ramp_link.find('collision')
    box_size = [float(v) for v in col.find('geometry').find('box').find('size').text.split()]
    length, width, thickness = box_size

    center = np.array([x0, y0, z0])
    rot = R.from_euler('xyz', [roll, pitch, yaw])

    # West entrance edge is at x = -length/2
    local_entrance_top = np.array([-length / 2.0, 0.0, thickness / 2.0])
    world_entrance_top = center + rot.apply(local_entrance_top)

    assert world_entrance_top[2] <= 0.0001, (
        f'Platform approach ramp entrance height {world_entrance_top[2]:.4f}m forms a vertical step > 0'
    )


# ==============================================================================
# SECTION 3: Stress Testing Arena Perimeter Constraints & Clamping
# ==============================================================================

def test_adversarial_perimeter_drive_aggressively_to_corner_2_5():
    """
    Drive robot aggressively toward outer arena walls at coordinates x=2.5, y=2.5.
    Verify robot center coordinates never exceed [-2.405, +2.405] m and trajectory
    never penetrates outer walls.
    """
    mapper = Khepera4Mapper()
    try:
        # Start at (2.0, 2.0) heading diagonally at 45 deg into corner (2.5, 2.5)
        msg = Odometry()
        msg.pose.pose.position.x = 2.0
        msg.pose.pose.position.y = 2.0
        qx, qy, qz, qw = quaternion_from_euler(0, 0, np.pi / 4.0)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        mapper.odom_callback(msg)

        # Drive 300 steps with increasing runaway odom up to (3.5, 3.5)
        for i in range(1, 301):
            pos = 2.0 + (i / 300.0) * 1.5  # up to 3.5m
            msg = Odometry()
            msg.pose.pose.position.x = pos
            msg.pose.pose.position.y = pos
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.twist.twist.linear.x = 0.20
            mapper.odom_callback(msg)

            # Strict bounds enforcement at every tick
            assert mapper.robot_x <= 2.405 + 1e-7, f'Tick {i}: robot_x={mapper.robot_x} breached 2.405'
            assert mapper.robot_y <= 2.405 + 1e-7, f'Tick {i}: robot_y={mapper.robot_y} breached 2.405'
            assert mapper.robot_x >= -2.405 - 1e-7
            assert mapper.robot_y >= -2.405 - 1e-7

        # Trajectory path check
        for px, py in mapper.robot_path:
            assert -2.405 - 1e-7 <= px <= 2.405 + 1e-7, f'Path point ({px}, {py}) breached boundary'
            assert -2.405 - 1e-7 <= py <= 2.405 + 1e-7, f'Path point ({px}, {py}) breached boundary'
    finally:
        mapper.destroy_node()


@pytest.mark.parametrize('target_corner', [
    (2.5, 2.5),
    (-2.5, 2.5),
    (2.5, -2.5),
    (-2.5, -2.5),
])
def test_adversarial_perimeter_drive_all_four_corners(target_corner):
    """Stress test driving toward all 4 outer arena corners at +/-2.5m."""
    tx, ty = target_corner
    mapper = Khepera4Mapper()
    try:
        sx, sy = 0.8 * tx, 0.8 * ty
        heading = math.atan2(ty - sy, tx - sx)
        qx, qy, qz, qw = quaternion_from_euler(0, 0, heading)

        init_msg = Odometry()
        init_msg.pose.pose.position.x = sx
        init_msg.pose.pose.position.y = sy
        init_msg.pose.pose.orientation.x = qx
        init_msg.pose.pose.orientation.y = qy
        init_msg.pose.pose.orientation.z = qz
        init_msg.pose.pose.orientation.w = qw
        mapper.odom_callback(init_msg)

        for step in range(1, 101):
            t = step / 100.0
            cur_x = sx + t * (tx * 1.5 - sx)
            cur_y = sy + t * (ty * 1.5 - sy)
            msg = Odometry()
            msg.pose.pose.position.x = cur_x
            msg.pose.pose.position.y = cur_y
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            mapper.odom_callback(msg)

            assert -2.405 - 1e-7 <= mapper.robot_x <= 2.405 + 1e-7
            assert -2.405 - 1e-7 <= mapper.robot_y <= 2.405 + 1e-7
    finally:
        mapper.destroy_node()


@pytest.mark.parametrize('init_x, init_y', [
    (2.5, 2.5),
    (-2.5, 2.5),
    (3.0, 0.0),
    (-5.0, 5.0),
    (10.0, -10.0),
])
def test_adversarial_perimeter_initialization_outside_arena(init_x, init_y):
    """Verify that if initial odometry starts outside the arena, pose is immediately clamped."""
    mapper = Khepera4Mapper()
    try:
        msg = Odometry()
        msg.pose.pose.position.x = init_x
        msg.pose.pose.position.y = init_y
        mapper.odom_callback(msg)

        assert -2.405 - 1e-7 <= mapper.robot_x <= 2.405 + 1e-7, f'Failed clamping initial x: {mapper.robot_x}'
        assert -2.405 - 1e-7 <= mapper.robot_y <= 2.405 + 1e-7, f'Failed clamping initial y: {mapper.robot_y}'
        assert len(mapper.robot_path) >= 1
        assert -2.405 - 1e-7 <= mapper.robot_path[0][0] <= 2.405 + 1e-7
        assert -2.405 - 1e-7 <= mapper.robot_path[0][1] <= 2.405 + 1e-7
    finally:
        mapper.destroy_node()


def test_adversarial_perimeter_wall_scraping_tangent_preservation():
    """
    Verify that when robot contacts an outer wall (x=2.405) and drives obliquely,
    penetrating displacement is strictly zeroed while tangential travel along wall is preserved.
    """
    mapper = Khepera4Mapper()
    try:
        # Start at boundary (2.405, 0.0) heading North-East (45 deg)
        heading = np.pi / 4.0
        qx, qy, qz, qw = quaternion_from_euler(0, 0, heading)

        msg0 = Odometry()
        msg0.pose.pose.position.x = 2.405
        msg0.pose.pose.position.y = 0.0
        msg0.pose.pose.orientation.x = qx
        msg0.pose.pose.orientation.y = qy
        msg0.pose.pose.orientation.z = qz
        msg0.pose.pose.orientation.w = qw
        mapper.odom_callback(msg0)

        # Feed step that attempts to advance in X (+0.05) and Y (+0.05)
        msg1 = Odometry()
        msg1.pose.pose.position.x = 2.405 + 0.05
        msg1.pose.pose.position.y = 0.0 + 0.05
        msg1.pose.pose.orientation.x = qx
        msg1.pose.pose.orientation.y = qy
        msg1.pose.pose.orientation.z = qz
        msg1.pose.pose.orientation.w = qw
        msg1.twist.twist.linear.x = 0.10
        mapper.odom_callback(msg1)

        # X must remain clamped at 2.405, while Y advances along the wall
        assert math.isclose(mapper.robot_x, 2.405, abs_tol=1e-4), f'X breached or shifted: {mapper.robot_x}'
        assert mapper.robot_y > 0.01, f'Tangential advancement along wall failed: Y={mapper.robot_y}'
    finally:
        mapper.destroy_node()
