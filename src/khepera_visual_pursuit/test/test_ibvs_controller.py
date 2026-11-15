# Copyright 2026 Andrei Berteanu.
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

"""Unit tests for the IBVS kinematic controller engine."""

from khepera_visual_pursuit.ibvs_controller import IBVSController


def test_ibvs_controller_initialization() -> None:
    """Verify default initialization and gains assignment."""
    controller = IBVSController()
    assert controller.gains.v_max == 0.15
    assert controller.gains.omega_max == 0.8
    assert controller.gains.obstacle_stop_distance == 0.05
    assert controller.gains.min_confidence == 0.35
    assert controller.last_omega == 0.0


def test_ibvs_centered_target_at_desired_distance() -> None:
    """Verify zero command when target is centered and at desired scale."""
    controller = IBVSController()
    # Centered (e_x = 0.0), area = 0.15 (scale ratio = 1.0, e_scale = 0.0), confidence = 1.0
    cmd = controller.compute_control((0.0, 0.15, 1.0), obstacle_distance=1.0)

    assert abs(cmd.linear_x) < 1e-4
    assert abs(cmd.angular_z) < 1e-4
    assert not cmd.is_emergency_stop
    assert not cmd.is_recovery_sweep


def test_ibvs_angular_steering_and_saturation() -> None:
    """Verify angular velocity proportionality, sign, and clamping to omega_max."""
    controller = IBVSController()

    # Target off-center to the left (e_x = -0.3): angular_z should be positive (turn left)
    cmd_left = controller.compute_control((-0.3, 0.15, 1.0), obstacle_distance=1.0)
    assert cmd_left.angular_z > 0.0
    assert abs(cmd_left.angular_z - (1.2 * 0.3)) < 1e-3

    # Target off-center to the right (e_x = 0.3): angular_z should be negative (turn right)
    cmd_right = controller.compute_control((0.3, 0.15, 1.0), obstacle_distance=1.0)
    assert cmd_right.angular_z < 0.0
    assert abs(cmd_right.angular_z - (-1.2 * 0.3)) < 1e-3

    # Target at extreme edge (e_x = 1.0): raw omega = -1.2, clamped to omega_max = 0.8
    cmd_sat = controller.compute_control((1.0, 0.15, 1.0), obstacle_distance=1.0)
    assert cmd_sat.angular_z == -0.8


def test_ibvs_forward_velocity_clamping_and_dampening() -> None:
    """Verify linear velocity calculation, scale error, cosine dampening, and v_max."""
    controller = IBVSController()

    # Distant centered target (area = 0.01 << 0.15): scale error large, clamped to v_max
    cmd_far = controller.compute_control((0.0, 0.01, 1.0), obstacle_distance=1.0)
    assert cmd_far.linear_x == controller.gains.v_max

    # Distant off-center target (e_x = 0.8): dampened by cos(0.8)
    cmd_offcenter = controller.compute_control((0.8, 0.05, 1.0), obstacle_distance=1.0)
    # Target scale ratio: 0.15 / 0.05 = 3.0 -> e_scale = sqrt(3) - 1.0 = 0.732
    # v_raw = 0.4 * 0.732 * cos(0.8) = 0.4 * 0.732 * 0.6967 = 0.203 -> clamped to v_max = 0.15
    assert cmd_offcenter.linear_x <= controller.gains.v_max
    assert cmd_offcenter.linear_x > 0.0

    # Target closer than desired (area = 0.25 > 0.15): e_scale < 0, forward velocity clamped to 0
    cmd_close = controller.compute_control((0.0, 0.25, 1.0), obstacle_distance=1.0)
    assert cmd_close.linear_x == 0.0


def test_ibvs_recovery_sweep_on_confidence_drop() -> None:
    """Verify recovery angular sweep when tracking confidence drops below 0.35."""
    controller = IBVSController()
    controller.last_omega = 0.5  # Robot was turning left

    # Confidence drops to 0.20 < 0.35
    cmd_recover = controller.compute_control((0.0, 0.10, 0.20), obstacle_distance=1.0)
    assert cmd_recover.linear_x == 0.0
    assert cmd_recover.is_recovery_sweep
    # Reverse sweep opposite to last_omega (last was positive, reverse is negative)
    assert cmd_recover.angular_z == -controller.gains.reverse_sweep_omega


def test_ibvs_emergency_stop_on_obstacle() -> None:
    """Verify immediate emergency stop when obstacle is closer than 8 cm."""
    controller = IBVSController()

    # Obstacle at 0.05 m < 0.08 m
    cmd_stop = controller.compute_control((0.0, 0.02, 1.0), obstacle_distance=0.05)
    assert cmd_stop.linear_x == 0.0
    assert cmd_stop.angular_z == 0.0
    assert cmd_stop.is_emergency_stop


def test_ibvs_controller_reset() -> None:
    """Verify state reset."""
    controller = IBVSController()
    controller.last_omega = 0.5
    controller.last_error_x = 0.2
    controller.reset()
    assert controller.last_omega == 0.0
    assert controller.last_error_x == 0.0


def test_ibvs_lateral_obstacle_avoidance_during_pursuit() -> None:
    """Verify obstacle repulsion steers away from close lateral obstacles during pursuit."""
    controller = IBVSController()

    # Centered target, but obstacle on the left at 0.15 m (< 0.35 m avoid_dist)
    # Controller must steer to the right (negative angular_z)
    cmd_avoid_left = controller.compute_control(
        (0.0, 0.02, 1.0),
        obstacle_distance=1.0,
        obstacle_left=0.15,
        obstacle_right=1.0,
    )
    assert cmd_avoid_left.angular_z < 0.0
    assert cmd_avoid_left.linear_x > 0.0

    # Centered target, but obstacle on the right at 0.15 m (< 0.35 m avoid_dist)
    # Controller must steer to the left (positive angular_z)
    cmd_avoid_right = controller.compute_control(
        (0.0, 0.02, 1.0),
        obstacle_distance=1.0,
        obstacle_left=1.0,
        obstacle_right=0.15,
    )
    assert cmd_avoid_right.angular_z > 0.0
    assert cmd_avoid_right.linear_x > 0.0


def test_ibvs_stop_at_5cm_in_front_of_target() -> None:
    """Verify robot stops exactly 5 cm (0.05m) in front of the target."""
    controller = IBVSController()

    # 1. Approaching at 0.15 m: decelerating forward velocity
    cmd_approach = controller.compute_control((0.0, 0.05, 1.0), obstacle_distance=0.15)
    assert cmd_approach.linear_x > 0.0
    assert cmd_approach.linear_x < controller.gains.v_max

    # 2. At 5 cm (0.05 m) in front of target: full emergency stop
    cmd_5cm = controller.compute_control((0.0, 0.10, 1.0), obstacle_distance=0.05)
    assert cmd_5cm.linear_x == 0.0
    assert cmd_5cm.angular_z == 0.0
    assert cmd_5cm.is_emergency_stop is True

    # 3. Inside 5 cm (< 0.05 m): full emergency stop
    cmd_closer = controller.compute_control((0.0, 0.12, 1.0), obstacle_distance=0.03)
    assert cmd_closer.linear_x == 0.0
    assert cmd_closer.angular_z == 0.0
    assert cmd_closer.is_emergency_stop is True
