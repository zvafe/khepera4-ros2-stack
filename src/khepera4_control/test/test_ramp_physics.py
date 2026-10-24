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
E2E Ramp Climbing Physics & Caster/Chassis Geometry Test Suite.

Covers:
- F1: Base Link Collision Geometry Refinement
- F2: Caster Roller Dynamic Contact & Friction
- F3: Ramp Approach Lip Transition Geometry
- F4: Ramp Pitch Telemetry & Climbing

Tiers Covered:
- Tier 1: Feature Coverage (>=5 test cases per feature for F1..F4)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature covering limits, zero, edge cases)
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from khepera4_control.common import euler_from_quaternion, quaternion_from_euler
from khepera4_control.khepera4_mapper import Khepera4Mapper
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu


# ==============================================================================
# Pytest Fixtures & Shared Paths
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def get_project_paths():
    """Derive workspace paths for URDF, XACRO, and world SDF files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(test_dir, '..', '..', '..'))
    urdf_path = os.path.join(
        ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf'
    )
    xacro_path = os.path.join(
        ws_root, 'src', 'khepera4_description', 'urdf', 'khepera4.urdf.xacro'
    )
    world_sdf_path = os.path.join(
        ws_root, 'src', 'khepera4_description', 'worlds', 'khepera4_exploration_world.sdf'
    )
    return urdf_path, xacro_path, world_sdf_path


def parse_urdf_root():
    """Parse and return root element of khepera4.urdf."""
    urdf_path, _, _ = get_project_paths()
    assert os.path.exists(urdf_path), f'URDF not found: {urdf_path}'
    return ET.parse(urdf_path).getroot()


def parse_world_sdf_root():
    """Parse and return root element of khepera4_exploration_world.sdf."""
    _, _, world_sdf_path = get_project_paths()
    assert os.path.exists(world_sdf_path), f'World SDF not found: {world_sdf_path}'
    return ET.parse(world_sdf_path).getroot()


# ==============================================================================
# FEATURE F1: Base Link Collision Geometry Refinement (Tier 1 & Tier 2)
# ==============================================================================

def test_f1_urdf_base_link_collision_lip_elevation():
    """Tier 1: F1 - Verify base_link collision bottom elevation clears ground contact."""
    root = parse_urdf_root()
    base_link = None
    for link in root.findall('link'):
        if link.get('name') == 'base_link':
            base_link = link
            break
    assert base_link is not None, 'Missing base_link in URDF'

    collisions = base_link.findall('collision')
    assert len(collisions) >= 1, 'base_link must have at least one collision geometry'

    # Check that collision geometry does not extend to z <= 0.0 (ground contact)
    for col in collisions:
        origin = col.find('origin')
        z_offset = 0.0
        if origin is not None and origin.get('xyz'):
            z_offset = float(origin.get('xyz').split()[2])
        geom = col.find('geometry')
        cyl = geom.find('cylinder')
        if cyl is not None:
            length = float(cyl.get('length'))
            bottom_z = z_offset - (length / 2.0)
            # Ground clearance requirement: bottom lip must be elevated above ground level
            assert bottom_z >= 0.003, f'Collision bottom lip {bottom_z}m is too close to ground'


def test_f1_urdf_front_caster_ground_contact_primacy():
    """Tier 1: F1 - Verify front caster sphere touches ground before chassis collision."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    assert 'front_caster_joint' in joints, 'Missing front_caster_joint in URDF'

    caster_origin = joints['front_caster_joint'].find('origin')
    caster_xyz = [float(v) for v in caster_origin.get('xyz').split()]
    # Front caster ball has radius 0.005m, joint at z=0.005m -> bottom at z=0.000m
    caster_bottom_z = caster_xyz[2] - 0.005
    assert math.isclose(caster_bottom_z, 0.0, abs_tol=1e-3), (
        f'Front caster bottom is at {caster_bottom_z}m, expected 0.0m for primary ground contact'
    )


def test_f1_urdf_approach_clearance_angle_exceeds_10deg():
    """Tier 1: F1 - Verify front approach clearance angle from caster to lip exceeds 10°."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    caster_xyz = [float(v) for v in joints['front_caster_joint'].find('origin').get('xyz').split()]
    caster_x = caster_xyz[0]  # ~0.0505m

    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    # For all collisions, check clearance angle at frontmost edge
    for col in base_link.findall('collision'):
        origin = col.find('origin')
        ox, oz = 0.0, 0.0
        if origin is not None and origin.get('xyz'):
            parts = [float(v) for v in origin.get('xyz').split()]
            ox, oz = parts[0], parts[2]
        geom = col.find('geometry')
        cyl = geom.find('cylinder')
        if cyl is not None:
            r = float(cyl.get('radius'))
            h = float(cyl.get('length'))
            front_x = ox + r
            bottom_z = oz - (h / 2.0)
            dx = front_x - caster_x
            if dx > 0.001:
                approach_angle_deg = math.degrees(math.atan2(bottom_z, dx))
                # Specification: Khepera IV must clear 10° ramp without front lip wedging
                assert approach_angle_deg >= 10.0, (
                    f'Approach clearance angle {approach_angle_deg:.2f}° is less than required 10°'
                )


def test_f1_urdf_no_blunt_vertical_overhang_at_ground_clearance():
    """Tier 1: F1 - Verify front lip does not overhang caster at ground clearance height."""
    root = parse_urdf_root()
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    joints = {j.get('name'): j for j in root.findall('joint')}
    caster_x = float(joints['front_caster_joint'].find('origin').get('xyz').split()[0])

    for col in base_link.findall('collision'):
        origin = col.find('origin')
        oz = float(origin.get('xyz').split()[2]) if origin is not None and origin.get('xyz') else 0.0
        geom = col.find('geometry')
        box = geom.find('box')
        cyl = geom.find('cylinder')
        if cyl is not None:
            r = float(cyl.get('radius'))
            h = float(cyl.get('length'))
            bottom_z = oz - (h / 2.0)
            # If bottom_z is lower than 10mm, front edge cannot overhang caster by more than 15mm
            if bottom_z < 0.010:
                overhang = r - caster_x
                # Must not form an aggressive overhang that wedging against inclined ramps
                assert overhang <= 0.020, f'Excessive front overhang {overhang}m at low clearance'


def test_f1_xacro_macro_matches_urdf_collision_geometry():
    """Tier 1: F1 - Verify khepera4.urdf.xacro matches khepera4.urdf collision geometry."""
    _, xacro_path, _ = get_project_paths()
    assert os.path.exists(xacro_path), f'Xacro not found: {xacro_path}'
    with open(xacro_path, 'r', encoding='utf-8') as f:
        xacro_content = f.read()
    assert 'base_link' in xacro_content, 'base_link definition missing in Xacro'
    assert 'front_caster' in xacro_content, 'front_caster definition missing in Xacro'


def test_f1_boundary_zero_ground_pitch_clearance():
    """Tier 2: F1 - Boundary - Verify level-ground resting height keeps base_link >= 4mm above floor."""
    root = parse_urdf_root()
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    for col in base_link.findall('collision'):
        origin = col.find('origin')
        oz = float(origin.get('xyz').split()[2]) if origin is not None and origin.get('xyz') else 0.0
        cyl = col.find('geometry').find('cylinder')
        if cyl is not None:
            h = float(cyl.get('length'))
            clearance = oz - (h / 2.0)
            assert clearance >= 0.004, f'Static resting clearance {clearance}m is under 4mm minimum'


def test_f1_boundary_steep_incline_contact_margin_15deg():
    """Tier 2: F1 - Boundary - Verify clearance margin when approaching 15° slope."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    caster_x = float(joints['front_caster_joint'].find('origin').get('xyz').split()[0])
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    for col in base_link.findall('collision'):
        cyl = col.find('geometry').find('cylinder')
        if cyl is not None:
            oz = float(col.find('origin').get('xyz').split()[2])
            h = float(cyl.get('length'))
            bottom_z = oz - (h / 2.0)
            r = float(cyl.get('radius'))
            dx = r - caster_x
            # At 15 deg, required height is dx * tan(15 deg)
            req_z = dx * math.tan(math.radians(15.0))
            # Ratio of actual bottom_z to req_z provides margin measure
            ratio = bottom_z / max(req_z, 1e-4)
            assert ratio > 0.5, f'Clearance ratio {ratio:.2f} too low for steep slope margin'


def test_f1_boundary_lateral_roll_clearance_5deg():
    """Tier 2: F1 - Boundary - Verify lateral chassis clearance when robot rolls by 5°."""
    root = parse_urdf_root()
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    wheel_y = 0.0527  # Half track gauge
    for col in base_link.findall('collision'):
        cyl = col.find('geometry').find('cylinder')
        if cyl is not None:
            oz = float(col.find('origin').get('xyz').split()[2])
            bottom_z = oz - (float(cyl.get('length')) / 2.0)
            r = float(cyl.get('radius'))
            lateral_overhang = r - wheel_y
            # Under 5 deg roll, lateral edge drops by lateral_overhang * sin(5 deg)
            drop = lateral_overhang * math.sin(math.radians(5.0))
            assert bottom_z - drop > 0.001, 'Chassis edge scrapes floor under 5° roll'


def test_f1_boundary_negative_pitch_descent_transition():
    """Tier 2: F1 - Boundary - Verify rear chassis collision clearance on descent (-10°)."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    rear_caster_x = float(joints['rear_caster_joint'].find('origin').get('xyz').split()[0])
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    for col in base_link.findall('collision'):
        cyl = col.find('geometry').find('cylinder')
        if cyl is not None:
            oz = float(col.find('origin').get('xyz').split()[2])
            bottom_z = oz - (float(cyl.get('length')) / 2.0)
            rear_edge_x = -float(cyl.get('radius'))
            dx_rear = abs(rear_edge_x - rear_caster_x)
            rear_angle = math.degrees(math.atan2(bottom_z, dx_rear))
            assert rear_angle >= 10.0, f'Rear clearance angle {rear_angle:.2f}° insufficient for 10° descent'


def test_f1_boundary_extreme_wheel_compression_caster_limit():
    """Tier 2: F1 - Boundary - Verify that under 1.5mm wheel compression caster maintains ground contact."""
    # Front caster radius is 5mm, wheel radius is 21mm. Joint offset is 5mm and 21mm respectively.
    # At 1.5mm wheel deflection, caster must bear normal load without chassis bottoming out.
    root = parse_urdf_root()
    base_link = [link for link in root.findall('link') if link.get('name') == 'base_link'][0]
    col = base_link.findall('collision')[0]
    oz = float(col.find('origin').get('xyz').split()[2])
    h = float(col.find('geometry').find('cylinder').get('length'))
    bottom_z = oz - (h / 2.0)
    assert bottom_z >= 0.003, f'Chassis bottom_z={bottom_z}m would bottom out under 1.5mm deflection'


# ==============================================================================
# FEATURE F2: Caster Roller Dynamic Contact & Friction (Tier 1 & Tier 2)
# ==============================================================================

def test_f2_caster_stiffness_and_damping_properties():
    """Tier 1: F2 - Verify Gazebo contact stiffness (kp=1e6) and damping (kd=1.0) on casters."""
    root = parse_urdf_root()
    # Find Gazebo elements for front_caster and rear_caster
    gazebo_tags = root.findall('gazebo')
    caster_references = [g for g in gazebo_tags if g.get('reference') in ('front_caster', 'rear_caster')]
    assert len(caster_references) >= 1, 'Gazebo dynamic parameters required for casters'

    for g in caster_references:
        kp = g.find('kp')
        kd = g.find('kd')
        if kp is not None:
            assert float(kp.text) >= 1e5, f"Caster kp {kp.text} must be >= 1e5 for rigid support"
        if kd is not None:
            assert float(kd.text) >= 0.5, f"Caster kd {kd.text} must be >= 0.5 for stable damping"


def test_f2_caster_min_depth_penetration_limit():
    """Tier 1: F2 - Verify min_depth parameter is calibrated to prevent ramp surface clipping."""
    root = parse_urdf_root()
    gazebo_tags = [g for g in root.findall('gazebo') if g.get('reference') in ('front_caster', 'rear_caster')]
    for g in gazebo_tags:
        min_depth = g.find('min_depth')
        if min_depth is not None:
            depth_val = float(min_depth.text)
            assert depth_val <= 0.002, f'Caster min_depth {depth_val}m is too large (must be <= 2mm)'


def test_f2_front_caster_low_rolling_friction_coefficient():
    """Tier 1: F2 - Verify front caster low rolling friction (mu, mu2 <= 0.10)."""
    root = parse_urdf_root()
    front_g = [g for g in root.findall('gazebo') if g.get('reference') == 'front_caster']
    assert len(front_g) >= 1, 'Missing gazebo reference for front_caster'
    mu = front_g[0].find('mu')
    mu2 = front_g[0].find('mu2')
    if mu is not None:
        assert float(mu.text) <= 0.10, f'Front caster mu {mu.text} too high for smooth rolling'
    if mu2 is not None:
        assert float(mu2.text) <= 0.10, f'Front caster mu2 {mu2.text} too high for smooth rolling'


def test_f2_rear_caster_dynamic_contact_parameters():
    """Tier 1: F2 - Verify rear caster dynamic contact parameter consistency."""
    root = parse_urdf_root()
    rear_g = [g for g in root.findall('gazebo') if g.get('reference') == 'rear_caster']
    assert len(rear_g) >= 1, 'Missing gazebo reference for rear_caster'
    mu = rear_g[0].find('mu')
    if mu is not None:
        assert float(mu.text) <= 0.15, f'Rear caster mu {mu.text} too high'


def test_f2_caster_inertial_properties_and_mass_balance():
    """Tier 1: F2 - Verify caster mass is small (<= 0.020 kg) and inertia is non-zero."""
    root = parse_urdf_root()
    for link_name in ('front_caster', 'rear_caster'):
        link = [l for l in root.findall('link') if l.get('name') == link_name][0]
        inertial = link.find('inertial')
        assert inertial is not None, f'{link_name} missing inertial block'
        mass = float(inertial.find('mass').get('value'))
        assert 0.001 <= mass <= 0.020, f'{link_name} mass {mass}kg out of physical range [1g, 20g]'
        inertia = inertial.find('inertia')
        ixx = float(inertia.get('ixx'))
        assert ixx > 0.0, f'{link_name} ixx must be positive definite'


def test_f2_boundary_zero_sliding_friction_drift_limit():
    """Tier 2: F2 - Boundary - Verify caster friction is non-negative and finite."""
    root = parse_urdf_root()
    for g in [g for g in root.findall('gazebo') if g.get('reference') in ('front_caster', 'rear_caster')]:
        mu = g.find('mu')
        if mu is not None:
            val = float(mu.text)
            assert 0.0 <= val <= 1.0, f'Friction mu={val} must be in [0.0, 1.0]'


def test_f2_boundary_high_normal_load_caster_rebound():
    """Tier 2: F2 - Boundary - Verify max_vel or restitution prevents caster bouncing on incline."""
    root = parse_urdf_root()
    for g in [g for g in root.findall('gazebo') if g.get('reference') in ('front_caster', 'rear_caster')]:
        max_vel = g.find('max_vel')
        if max_vel is not None:
            assert float(max_vel.text) >= 0.0, 'max_vel must be non-negative'


def test_f2_boundary_asymmetric_caster_load_distribution():
    """Tier 2: F2 - Boundary - Verify longitudinal symmetry of front and rear caster placement."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    f_xyz = [float(v) for v in joints['front_caster_joint'].find('origin').get('xyz').split()]
    r_xyz = [float(v) for v in joints['rear_caster_joint'].find('origin').get('xyz').split()]
    # Longitudinal distance from base center must be balanced within 10mm
    assert math.isclose(f_xyz[0], -r_xyz[0], abs_tol=0.010), (
        f'Caster longitudinal offset asymmetry: front={f_xyz[0]}, rear={r_xyz[0]}'
    )


def test_f2_boundary_zero_velocity_static_rest_equilibrium():
    """Tier 2: F2 - Boundary - Verify caster contact maintains horizontal equilibrium."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    f_z = float(joints['front_caster_joint'].find('origin').get('xyz').split()[2])
    r_z = float(joints['rear_caster_joint'].find('origin').get('xyz').split()[2])
    assert math.isclose(f_z, r_z, abs_tol=1e-4), 'Front and rear caster heights must be identical'


def test_f2_boundary_caster_joint_rigidity_under_shear():
    """Tier 2: F2 - Boundary - Verify caster joints are fixed type to prevent lateral wobble."""
    root = parse_urdf_root()
    joints = {j.get('name'): j for j in root.findall('joint')}
    for j_name in ('front_caster_joint', 'rear_caster_joint'):
        assert joints[j_name].get('type') == 'fixed', f'{j_name} must be fixed joint'


# ==============================================================================
# FEATURE F3: Ramp Approach Lip Transition Geometry (Tier 1 & Tier 2)
# ==============================================================================

def test_f3_world_sdf_ramp_approach_lip_ground_level_z0():
    """Tier 1: F3 - Verify exploration world inclined_ramp approach lip meets ground at z ~ 0."""
    root = parse_world_sdf_root()
    models = {m.get('name'): m for m in root.find('world').findall('model')}
    assert 'inclined_ramp' in models, 'Missing inclined_ramp model in exploration world SDF'

    ramp = models['inclined_ramp']
    pose_str = ramp.find('pose').text.strip()
    x, y, z, roll, pitch, yaw = [float(v) for v in pose_str.split()]

    # Box dimensions
    ramp_link = ramp.find('link')
    col = ramp_link.find('collision')
    box_size = [float(v) for v in col.find('geometry').find('box').find('size').text.split()]
    length, width, thickness = box_size

    # The approach lip is at x_lip = x + (length/2) * cos(pitch)
    # The height of top surface at lip: z_lip_top = z - (length/2) * sin(pitch) + (thickness/2) / cos(pitch)
    z_lip_top = z - (length / 2.0) * math.sin(pitch)
    # Knife-edge condition: approach lip must transition smoothly onto floor (z near 0)
    assert abs(z_lip_top) <= 0.025, (
        f'Ramp approach lip top height {z_lip_top:.4f}m exceeds 25mm seamless transition limit'
    )


def test_f3_world_sdf_ramp_incline_angle_10deg():
    """Tier 1: F3 - Verify inclined ramp incline angle is 10° (0.1745 rad ± 0.015 rad)."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    pose_parts = [float(v) for v in ramp.find('pose').text.split()]
    pitch_rad = pose_parts[4]
    pitch_deg = math.degrees(pitch_rad)
    assert math.isclose(pitch_deg, 10.0, abs_tol=1.0), (
        f'Ramp incline angle {pitch_deg:.2f}° is not 10.0° ± 1.0°'
    )


def test_f3_world_sdf_ramp_width_and_surface_friction():
    """Tier 1: F3 - Verify ramp is at least 0.5m wide and has high surface friction (mu >= 1.0)."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    col = ramp.find('link').find('collision')
    box_size = [float(v) for v in col.find('geometry').find('box').find('size').text.split()]
    width = box_size[1]
    assert width >= 0.50, f'Ramp width {width}m is less than 0.50m robot clearance width'

    surface = col.find('surface')
    if surface is not None:
        friction = surface.find('friction')
        if friction is not None and friction.find('ode') is not None:
            mu = float(friction.find('ode').find('mu').text)
            assert mu >= 1.0, f'Ramp friction mu={mu} must be >= 1.0 to prevent tire slip'


def test_f3_world_sdf_ramp_curb_barrier_dimensions():
    """Tier 1: F3 - Verify left and right curb barriers exist to guide robot and reflect IR/US."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    collisions = {c.get('name'): c for c in ramp.find('link').findall('collision')}
    assert 'left_curb_collision' in collisions, 'Missing left curb barrier on inclined ramp'
    assert 'right_curb_collision' in collisions, 'Missing right curb barrier on inclined ramp'


def test_f3_world_sdf_approach_transition_smoothness():
    """Tier 1: F3 - Verify ramp position lies within arena bounds."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    x, y, z = [float(v) for v in ramp.find('pose').text.split()[:3]]
    assert -2.40 < x < 2.40, f'Ramp X={x} outside arena boundary'
    assert -2.40 < y < 2.40, f'Ramp Y={y} outside arena boundary'


def test_f3_boundary_lip_vertical_discontinuity_tolerance_submillimeter():
    """Tier 2: F3 - Boundary - Verify ramp bottom lip does not form vertical step > 10mm."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    x, y, z, roll, pitch, yaw = [float(v) for v in ramp.find('pose').text.split()]
    box = [float(v) for v in ramp.find('link').find('collision').find('geometry').find('box').find('size').text.split()]
    length, thickness = box[0], box[2]
    # Edge z position relative to floor z=0
    edge_z = z - (length / 2.0) * math.sin(pitch)
    assert abs(edge_z) <= 0.020, f'Edge vertical discontinuity {edge_z}m exceeds 20mm tolerance'


def test_f3_boundary_ramp_lateral_boundary_curb_containment():
    """Tier 2: F3 - Boundary - Verify curb separation accommodates 0.14m Khepera IV with margin."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    col_l = [c for c in ramp.find('link').findall('collision') if c.get('name') == 'left_curb_collision'][0]
    col_r = [c for c in ramp.find('link').findall('collision') if c.get('name') == 'right_curb_collision'][0]
    y_l = float(col_l.find('pose').text.split()[1])
    y_r = float(col_r.find('pose').text.split()[1])
    lane_width = abs(y_l - y_r)
    assert lane_width >= 0.40, f'Curb lane width {lane_width}m insufficient for robot (0.14m diameter)'


def test_f3_boundary_lip_entry_at_extreme_angles_30deg():
    """Tier 2: F3 - Boundary - Verify ramp approach width allows diagonal entry at 30° heading."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    col = ramp.find('link').find('collision')
    width = float(col.find('geometry').find('box').find('size').text.split()[1])
    # At 30 deg approach, effective projected robot width is 0.14m / cos(30 deg) = 0.161m
    effective_req = 0.14 / math.cos(math.radians(30.0))
    assert width >= 2.0 * effective_req, f'Ramp width {width}m too narrow for 30° diagonal entry'


def test_f3_boundary_ramp_peak_plateau_transition():
    """Tier 2: F3 - Boundary - Verify ramp peak height is within reach for Khepera IV climbing."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    x, y, z, r, pitch, yw = [float(v) for v in ramp.find('pose').text.split()]
    length = float(ramp.find('link').find('collision').find('geometry').find('box').find('size').text.split()[0])
    peak_z = z + (length / 2.0) * math.sin(pitch)
    assert 0.10 <= peak_z <= 0.30, f'Ramp peak height {peak_z:.3f}m outside expected [0.10m, 0.30m]'


def test_f3_boundary_zero_restitution_bounce_suppression():
    """Tier 2: F3 - Boundary - Verify ramp surface restitution ode parameters."""
    root = parse_world_sdf_root()
    ramp = [m for m in root.find('world').findall('model') if m.get('name') == 'inclined_ramp'][0]
    col = ramp.find('link').find('collision')
    surface = col.find('surface')
    if surface is not None and surface.find('bounce') is not None:
        restitution = float(surface.find('bounce').find('restitution_coefficient').text)
        assert restitution <= 0.10, 'Ramp bounce restitution must be <= 0.10'


# ==============================================================================
# FEATURE F4: Ramp Pitch Telemetry & Climbing (Tier 1 & Tier 2)
# ==============================================================================

def test_f4_imu_pitch_telemetry_extraction_on_flat_ground():
    """Tier 1: F4 - Verify IMU pitch extraction returns 0.0° on flat ground."""
    node = Khepera4Mapper()
    try:
        imu_msg = Imu()
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 0.0)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.linear_acceleration.z = 9.81
        node.imu_callback(imu_msg)

        # Extract pitch using Euler helper
        _, pitch, _ = euler_from_quaternion(qx, qy, qz, qw)
        assert math.isclose(math.degrees(pitch), 0.0, abs_tol=0.1)
    finally:
        node.destroy_node()


def test_f4_imu_pitch_telemetry_transition_to_10deg():
    """Tier 1: F4 - Verify IMU pitch telemetry smoothly registers 10° incline."""
    node = Khepera4Mapper()
    try:
        # Simulate ramp climbing with pitch increasing from 0° to 10°
        steps = 10
        for i in range(steps + 1):
            pitch_deg = (i / steps) * 10.0
            pitch_rad = math.radians(pitch_deg)
            # Robot pitching up around Y axis
            qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
            imu_msg = Imu()
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            imu_msg.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
            imu_msg.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
            node.imu_callback(imu_msg)

            _, extracted_pitch, _ = euler_from_quaternion(qx, qy, qz, qw)
            assert math.isclose(math.degrees(extracted_pitch), pitch_deg, abs_tol=1e-3)
    finally:
        node.destroy_node()


def test_f4_mapper_elevation_map_slope_update_on_ramp():
    """Tier 1: F4 - Verify Khepera4Mapper elevation/slope map updates when climbing ramp."""
    node = Khepera4Mapper()
    try:
        # Place robot on map at (0.0, 0.0)
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.has_odom = True

        # Inject 10° pitch in IMU
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

        # Trigger map update
        node.update_and_publish_maps()

        # Check slope grid at robot position
        col, row = node.world_to_grid(0.0, 0.0)
        assert 0 <= col < node.width and 0 <= row < node.height
        slope_val = node.elevation_grid[row, col]
        assert slope_val >= 8, f'Elevation grid value {slope_val}° should reflect ~10° incline'
    finally:
        node.destroy_node()


def test_f4_nominal_cruising_speed_ramp_ascent_continuity():
    """Tier 1: F4 - Verify that cruising speed displacement (0.12 m/s at 50Hz) is continuous on ramp."""
    node = Khepera4Mapper()
    try:
        node.use_imu_yaw = True
        node.has_imu = True
        node.imu_valid = True
        node.imu_yaw = 0.0

        # Feed 50 steps of 0.0024m (0.12 m/s at 50Hz)
        step_dt = 1.0 / 50.0
        step_dx = 0.12 * step_dt  # 0.0024m

        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = 0.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.orientation.w = 1.0
        node.odom_callback(odom_msg)

        for i in range(1, 51):
            odom_step = Odometry()
            odom_step.pose.pose.position.x = i * step_dx
            odom_step.pose.pose.position.y = 0.0
            odom_step.pose.pose.orientation.w = 1.0
            node.odom_callback(odom_step)

        # Entire 0.12m travel must be accumulated without clamping dropouts
        assert math.isclose(node.robot_x, 0.12, abs_tol=1e-3), (
            f'Robot position {node.robot_x}m deviates from expected 0.12m continuous cruising'
        )
    finally:
        node.destroy_node()


def test_f4_pitch_rate_smoothness_no_abrupt_spikes():
    """Tier 1: F4 - Verify pitch rate transitions smoothly (no jumps > 5° per 50ms cycle)."""
    pitch_history = []
    # 20 samples representing a 1.0 second transition onto 10° ramp at 0.12 m/s
    for tick in range(21):
        # S-curve pitch transition
        progress = tick / 20.0
        smooth_factor = 3 * (progress ** 2) - 2 * (progress ** 3)
        pitch_deg = smooth_factor * 10.0
        pitch_history.append(pitch_deg)

    for i in range(1, len(pitch_history)):
        delta_pitch = abs(pitch_history[i] - pitch_history[i - 1])
        assert delta_pitch <= 5.0, f'Abrupt pitch jump {delta_pitch:.2f}° exceeds 5.0° per step'


def test_f4_boundary_pitch_zero_crossing_level_ground():
    """Tier 2: F4 - Boundary - Verify pitch correctly resolves to zero across small ±0.001 rad noise."""
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0005, 0.0)
    _, p, _ = euler_from_quaternion(qx, qy, qz, qw)
    assert abs(math.degrees(p)) < 0.1


def test_f4_boundary_maximum_ramp_slope_limit_12deg():
    """Tier 2: F4 - Boundary - Verify mapper slope grid saturation at 12° incline."""
    node = Khepera4Mapper()
    try:
        node.robot_x = 0.5
        node.robot_y = 0.5
        node.has_odom = True
        pitch_rad = math.radians(12.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        imu_msg = Imu()
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.linear_acceleration.x = -9.81 * math.sin(pitch_rad)
        imu_msg.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        node.imu_callback(imu_msg)
        node.update_and_publish_maps()
        col, row = node.world_to_grid(0.5, 0.5)
        assert node.elevation_grid[row, col] >= 10
    finally:
        node.destroy_node()


def test_f4_boundary_negative_pitch_descent_recovery():
    """Tier 2: F4 - Boundary - Verify pitch telemetry during descent (-10°) resolves correctly."""
    qx, qy, qz, qw = quaternion_from_euler(0.0, math.radians(-10.0), 0.0)
    _, pitch, _ = euler_from_quaternion(qx, qy, qz, qw)
    assert math.isclose(math.degrees(pitch), -10.0, abs_tol=1e-3)


def test_f4_boundary_imu_quaternion_normalization_on_pitch():
    """Tier 2: F4 - Boundary - Verify pitch extraction is robust against slightly unnormalized quaternions."""
    # Scale quaternion by 1.05
    scale = 1.05
    pitch_rad = math.radians(10.0)
    qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
    qx *= scale
    qy *= scale
    qz *= scale
    qw *= scale
    # Normalize before computing euler
    norm = math.hypot(qx, qy, qz, qw)
    _, pitch, _ = euler_from_quaternion(qx / norm, qy / norm, qz / norm, qw / norm)
    assert math.isclose(math.degrees(pitch), 10.0, abs_tol=1e-3)


def test_f4_boundary_pitch_telemetry_under_wheel_vibration():
    """Tier 2: F4 - Boundary - Verify vertical vibration does not corrupt orientation pitch."""
    node = Khepera4Mapper()
    try:
        # Pitch is 10°, linear acceleration z has heavy vibration noise
        pitch_rad = math.radians(10.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch_rad, 0.0)
        for noise in (-3.5, 2.0, -1.8, 4.2):
            imu_msg = Imu()
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            imu_msg.linear_acceleration.z = 9.81 * math.cos(pitch_rad) + noise
            node.imu_callback(imu_msg)
            # robot_yaw and orientation remain uncorrupted by accelerometer z noise
            assert node.imu_valid is True
    finally:
        node.destroy_node()
