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

"""Kinematic Image-Based Visual Servoing (IBVS) unicycle controller."""

from dataclasses import dataclass
import math
from typing import Tuple

DEFAULT_KV: float = 0.4
DEFAULT_K_OMEGA: float = 1.2
DEFAULT_V_MAX: float = 0.15
DEFAULT_OMEGA_MAX: float = 0.8
DEFAULT_DESIRED_AREA_FRACTION: float = 0.15
DEFAULT_MIN_CONFIDENCE: float = 0.35
DEFAULT_REVERSE_SWEEP_OMEGA: float = 0.2
DEFAULT_OBSTACLE_STOP_DISTANCE: float = 0.05
MIN_SCALE_AREA: float = 1e-4


@dataclass(frozen=True)
class IBVSGains:
    """Configurable kinematic gains and physical constraints for IBVS."""

    k_v: float = DEFAULT_KV
    k_omega: float = DEFAULT_K_OMEGA
    v_max: float = DEFAULT_V_MAX
    omega_max: float = DEFAULT_OMEGA_MAX
    desired_area_fraction: float = DEFAULT_DESIRED_AREA_FRACTION
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    reverse_sweep_omega: float = DEFAULT_REVERSE_SWEEP_OMEGA
    obstacle_stop_distance: float = DEFAULT_OBSTACLE_STOP_DISTANCE


@dataclass(frozen=True)
class TwistCommand:
    """Differential drive velocity commands and operational safety flags."""

    linear_x: float
    angular_z: float
    is_emergency_stop: bool
    is_recovery_sweep: bool


class IBVSController:
    """Non-holonomic unicycle visual servoing kinematic control engine."""

    def __init__(self, gains: IBVSGains = IBVSGains()) -> None:
        """Initialize controller with specified tuning gains and constraints."""
        self.gains = gains
        self.last_omega: float = 0.0
        self.last_error_x: float = 0.0

    def compute_control(
        self,
        target_state: Tuple[float, float, float],
        obstacle_distance: float,
        obstacle_left: float = 10.0,
        obstacle_right: float = 10.0,
    ) -> TwistCommand:
        """
        Compute linear and angular velocity commands from tracker state and obstacles.

        :param target_state: Tuple of (norm_x, area_fraction, confidence)
        :param obstacle_distance: Minimum frontal obstacle distance in meters
        :param obstacle_left: Minimum left sector obstacle distance in meters
        :param obstacle_right: Minimum right sector obstacle distance in meters
        :return: TwistCommand with clamped velocities and status flags
        """
        norm_x, area_frac, confidence = target_state

        # Stop 5 cm in front of the target (target 5.0 ± 1.0 cm, threshold 0.051 m)
        if obstacle_distance <= (self.gains.obstacle_stop_distance + 0.001):
            return self._build_emergency_stop_command()

        if confidence < self.gains.min_confidence:
            return self._build_recovery_sweep_command()

        cmd = self._compute_visual_servoing_command(norm_x, area_frac)

        # Reactive obstacle avoidance: repulsive field from lateral obstacles
        w_repulse = 0.0
        avoid_dist = 0.35
        if obstacle_distance > 0.20 and (obstacle_left < avoid_dist or obstacle_right < avoid_dist):
            w_left = max(0.0, (avoid_dist - obstacle_left) / avoid_dist)
            w_right = max(0.0, (avoid_dist - obstacle_right) / avoid_dist)
            w_repulse = (w_right - w_left) * 1.2

        total_omega = float(
            max(-self.gains.omega_max, min(self.gains.omega_max, cmd.angular_z + w_repulse))
        )
        self.last_omega = total_omega

        # Smooth deceleration when approaching within 0.30 m of frontal target
        linear_v = cmd.linear_x
        slow_dist = 0.30
        if obstacle_distance < slow_dist:
            dist_remaining = obstacle_distance - self.gains.obstacle_stop_distance
            ratio = max(0.0, dist_remaining / max(1e-4, slow_dist - self.gains.obstacle_stop_distance))
            # Modulate speed towards terminal creep velocity (0.025 m/s) down to 5.0 cm stop
            linear_v = max(0.025, min(linear_v, 0.06 * ratio + 0.025))

        return TwistCommand(
            linear_x=float(linear_v),
            angular_z=float(total_omega),
            is_emergency_stop=False,
            is_recovery_sweep=False,
        )

    def _build_emergency_stop_command(self) -> TwistCommand:
        """Generate zero-velocity command when frontal collision danger is detected."""
        return TwistCommand(
            linear_x=0.0,
            angular_z=0.0,
            is_emergency_stop=True,
            is_recovery_sweep=False,
        )

    def _build_recovery_sweep_command(self) -> TwistCommand:
        """Generate reverse angular sweep when tracking confidence drops."""
        # Reverse sweep opposite to the last turn direction to reacquire target in FOV
        sweep_dir = -1.0 if self.last_omega >= 0.0 else 1.0
        omega_cmd = sweep_dir * self.gains.reverse_sweep_omega
        self.last_omega = omega_cmd

        return TwistCommand(
            linear_x=0.0,
            angular_z=float(omega_cmd),
            is_emergency_stop=False,
            is_recovery_sweep=True,
        )

    def _compute_visual_servoing_command(
        self,
        norm_x: float,
        area_frac: float,
    ) -> TwistCommand:
        """
        Calculate unicycle kinematic commands using decoupled proportional law.

        omega(t) = -K_omega * e_x
        v(t) = clamp(K_v * e_scale * max(0, cos(e_x)), 0, v_max)
        """
        e_x = float(norm_x)
        self.last_error_x = e_x

        # Angular velocity law with saturation
        omega_raw = -self.gains.k_omega * e_x
        omega = float(max(-self.gains.omega_max, min(self.gains.omega_max, omega_raw)))
        self.last_omega = omega

        # Scale error: e_scale = sqrt(A_target / A_current) - 1.0
        safe_area = max(MIN_SCALE_AREA, float(area_frac))
        scale_ratio = self.gains.desired_area_fraction / safe_area
        e_scale = math.sqrt(scale_ratio) - 1.0

        # Cosine heading alignment dampener: cos(e_x)
        alignment_dampener = max(0.0, math.cos(e_x))

        # Forward translation velocity law with clamping
        v_raw = self.gains.k_v * e_scale * alignment_dampener
        v = float(max(0.0, min(self.gains.v_max, v_raw)))

        return TwistCommand(
            linear_x=v,
            angular_z=omega,
            is_emergency_stop=False,
            is_recovery_sweep=False,
        )

    def reset(self) -> None:
        """Reset internal controller state history."""
        self.last_omega = 0.0
        self.last_error_x = 0.0
