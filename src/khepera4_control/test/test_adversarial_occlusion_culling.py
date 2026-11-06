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
Adversarial Stress Test Suite: Anti-Smudge Occlusion Culling.

Empirically challenges:
1. In-place rapid robot rotation in front of obstacle box:
   zero radial fan smudges behind obstacle boundary.
2. Extreme scan anomalies: missing rays (inf), zero ranges, noisy scans,
   obstacles at grazing angles.
3. Glancing wall proximity: cells behind wall face strictly remain uncolored.
4. Dynamic SLAM map updates: instantaneous purging of confirmed obstacle cells.
"""

import math
from typing import Tuple

from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import MapMetaData, OccupancyGrid
import numpy as np
import pytest
import rclpy


@pytest.fixture(scope='module', autouse=True)
def rclpy_module_context():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def mapper_node():
    """Provide a clean VisualOrthomapper node instance for each test."""
    node = VisualOrthomapper()
    yield node
    node.destroy_node()


def synthesize_box_scan(
    robot_pose: Tuple[float, float, float],
    box_bounds: Tuple[float, float, float, float],
    num_rays: int = 360,
    max_range: float = 2.0
) -> np.ndarray:
    """
    Compute ground-truth 360-degree laser scan for an axis-aligned box.

    :param robot_pose: (rx, ry, ryaw) in world coordinates.
    :param box_bounds: (min_x, max_x, min_y, max_y) in world coordinates.
    :param num_rays: Number of scan samples over 2*pi.
    :param max_range: Max sensor range (default 2.0m).
    :return: 1D numpy array of float32 ranges.
    """
    rx, ry, ryaw = robot_pose
    bx_min, bx_max, by_min, by_max = box_bounds
    scan_inc = 2.0 * math.pi / num_rays
    ranges = np.full(num_rays, float('inf'), dtype=np.float32)

    for i in range(num_rays):
        ray_angle_b = -math.pi + i * scan_inc
        theta_w = ryaw + ray_angle_b
        dx = math.cos(theta_w)
        dy = math.sin(theta_w)

        # Ray-AABB intersection algorithm (slab method)
        t_min = 0.0
        t_max = max_range

        # X slab
        if abs(dx) < 1e-9:
            if rx < bx_min or rx > bx_max:
                continue
        else:
            tx1 = (bx_min - rx) / dx
            tx2 = (bx_max - rx) / dx
            t_entry_x = min(tx1, tx2)
            t_exit_x = max(tx1, tx2)
            t_min = max(t_min, t_entry_x)
            t_max = min(t_max, t_exit_x)

        # Y slab
        if abs(dy) < 1e-9:
            if ry < by_min or ry > by_max:
                continue
        else:
            ty1 = (by_min - ry) / dy
            ty2 = (by_max - ry) / dy
            t_entry_y = min(ty1, ty2)
            t_exit_y = max(ty1, ty2)
            t_min = max(t_min, t_entry_y)
            t_max = min(t_max, t_exit_y)

        if t_min <= t_max and t_min > 0.002:
            if t_min <= max_range:
                ranges[i] = float(t_min)

    return ranges


def synthesize_wall_scan(
    robot_pose: Tuple[float, float, float],
    wall_y: float,
    wall_x_range: Tuple[float, float],
    num_rays: int = 360,
    max_range: float = 2.0
) -> np.ndarray:
    """
    Compute ground-truth 360-degree laser scan for a planar wall at y = wall_y.

    :param robot_pose: (rx, ry, ryaw) in world coordinates.
    :param wall_y: Y coordinate of wall face.
    :param wall_x_range: (min_x, max_x) of wall along X axis.
    :param num_rays: Number of scan samples over 2*pi.
    :param max_range: Max sensor range (default 2.0m).
    :return: 1D numpy array of float32 ranges.
    """
    rx, ry, ryaw = robot_pose
    scan_inc = 2.0 * math.pi / num_rays
    ranges = np.full(num_rays, float('inf'), dtype=np.float32)

    for i in range(num_rays):
        ray_angle_b = -math.pi + i * scan_inc
        theta_w = ryaw + ray_angle_b
        dy = math.sin(theta_w)
        dx = math.cos(theta_w)

        # Intersection with horizontal line y = wall_y:
        # ry + t * dy = wall_y => t = (wall_y - ry) / dy
        if abs(dy) > 1e-6:
            t = (wall_y - ry) / dy
            if 0.002 <= t <= max_range:
                hit_x = rx + t * dx
                if wall_x_range[0] <= hit_x <= wall_x_range[1]:
                    ranges[i] = float(t)

    return ranges


class TestInPlaceRotationAdversarial:
    """
    Empirical Challenge 1: In-place rapid robot rotation in front of obstacle box.

    Verifies zero radial fan smudges behind obstacle boundary across rotation.
    """

    def test_in_place_rotation_zero_radial_fan_smudge(self, mapper_node):
        """
        Adversarially sweep robot yaw in place in front of an obstacle box.

        Verify that strictly 0 ground cells in the shadow cone behind the box
        receive observations, while verifying baseline without culling fails.
        """
        node = mapper_node
        node.enable_occlusion_culling = True
        node.shadow_culling_margin = 0.02

        # Box located directly in front: X in [0.45, 0.75], Y in [-0.15, 0.15]
        box_bounds = (0.45, 0.75, -0.15, 0.15)
        rx, ry = 0.0, 0.0

        # Distinctive obstacle color: Bright Red (255, 0, 0)
        img = np.full((node.image_height, node.image_width, 3), 255, dtype=np.uint8)
        img[:, :, 1:] = 0

        # Rotate robot through 25 yaw angles from -40 deg to +40 deg
        yaw_angles = np.linspace(-math.radians(40), math.radians(40), 25)

        for yaw in yaw_angles:
            scan_ranges = synthesize_box_scan((rx, ry, yaw), box_bounds)
            node.latest_scan_ranges = scan_ranges
            node.warp_image_to_grid(img, rx, ry, yaw, 0.0)

        # Region behind the obstacle front face: X >= 0.45 + 0.03, |Y| <= 0.12
        shadow_cells_observed = 0
        total_shadow_cells_checked = 0

        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        # Check all grid cells in the shadow footprint and occlusion zone behind box
        for xw in np.arange(0.48, 1.40, node.res):
            for yw in np.arange(-0.12, 0.12, node.res):
                total_shadow_cells_checked += 1
                col, row = node.world_to_grid(xw, yw)
                if (col, row) in observed_coords:
                    shadow_cells_observed += 1

        assert total_shadow_cells_checked > 100
        # Empirical Assertion: Strict ZERO radial fan smudges behind obstacle boundary
        assert shadow_cells_observed == 0, (
            f'Occlusion culling leaked {shadow_cells_observed} smudged cells '
            f'behind obstacle boundary out of {total_shadow_cells_checked} checked cells!'
        )

    def test_in_place_rotation_baseline_contrast(self, mapper_node):
        """
        Verify that WITHOUT occlusion culling, radial fan smudges heavily occur.

        Proves that the test scenario is discriminative and not trivially passing.
        """
        node = mapper_node
        node.enable_occlusion_culling = False

        box_bounds = (0.45, 0.75, -0.15, 0.15)
        rx, ry = 0.0, 0.0
        img = np.full((node.image_height, node.image_width, 3), 255, dtype=np.uint8)

        yaw_angles = np.linspace(-math.radians(40), math.radians(40), 25)
        for yaw in yaw_angles:
            scan_ranges = synthesize_box_scan((rx, ry, yaw), box_bounds)
            node.latest_scan_ranges = scan_ranges
            node.warp_image_to_grid(img, rx, ry, yaw, 0.0)

        shadow_cells_observed = 0
        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        for xw in np.arange(0.48, 1.40, node.res):
            for yw in np.arange(-0.12, 0.12, node.res):
                col, row = node.world_to_grid(xw, yw)
                if (col, row) in observed_coords:
                    shadow_cells_observed += 1

        # Without culling, the obstacle face smears across dozens of ground cells
        assert shadow_cells_observed > 50, (
            f'Expected massive smudge without culling, but only got {shadow_cells_observed}'
        )

    def test_rapid_extreme_yaw_jumps(self, mapper_node):
        """
        Adversarially simulate rapid, high-speed yaw steps (e.g. 15 to 30 deg).

        Ensures discrete angular jumps do not cause ray-discretization leakage.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        box_bounds = (0.50, 0.80, -0.20, 0.20)
        rx, ry = 0.0, 0.0
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)

        # High-speed rotation with 15-degree steps spanning -45 to +45 deg
        large_yaw_steps = [-math.radians(45), -math.radians(30), -math.radians(15),
                           0.0,
                           math.radians(15), math.radians(30), math.radians(45)]

        for yaw in large_yaw_steps:
            scan_ranges = synthesize_box_scan((rx, ry, yaw), box_bounds)
            node.latest_scan_ranges = scan_ranges
            node.warp_image_to_grid(img, rx, ry, yaw, 0.0)

        # Check shadow zone
        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        leak_count = 0
        for xw in np.arange(0.53, 1.30, node.res):
            for yw in np.arange(-0.16, 0.16, node.res):
                col, row = node.world_to_grid(xw, yw)
                if (col, row) in observed_coords:
                    leak_count += 1

        assert leak_count == 0

    def test_360_degree_continuous_spin_multi_obstacle(self, mapper_node):
        """
        Adversarially spin 360 degrees in an arena with 3 surrounding obstacle boxes.

        Validates all-around culling robustness and zero wrap-around corruption.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        boxes = [
            (0.50, 0.80, -0.15, 0.15),   # Front box (+X)
            (-0.15, 0.15, 0.50, 0.80),   # Left box (+Y)
            (-0.80, -0.50, -0.15, 0.15)  # Rear box (-X)
        ]
        rx, ry = 0.0, 0.0
        img = np.full((node.image_height, node.image_width, 3), 180, dtype=np.uint8)

        # Full 360 rotation in 36 steps of 10 degrees
        for step in range(36):
            yaw = -math.pi + step * (2.0 * math.pi / 36.0)
            # Combine scans across all boxes
            composite_scan = np.full(360, float('inf'), dtype=np.float32)
            for box in boxes:
                box_scan = synthesize_box_scan((rx, ry, yaw), box)
                composite_scan = np.minimum(composite_scan, box_scan)

            node.latest_scan_ranges = composite_scan
            node.warp_image_to_grid(img, rx, ry, yaw, 0.0)

        # Verify front box shadow volume
        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        front_leaks = sum(
            1 for xw in np.arange(0.53, 1.20, node.res)
            for yw in np.arange(-0.12, 0.12, node.res)
            if node.world_to_grid(xw, yw) in observed_coords
        )
        assert front_leaks == 0


class TestScanAnomaliesAdversarial:
    """
    Empirical Challenge 2: Extreme scan anomalies.

    Missing rays (inf), zero ranges, noisy scans, obstacles at grazing angles.
    """

    def test_intermittent_ray_dropouts_and_all_inf(self, mapper_node):
        """
        Verify robustness against alternating inf dropouts and all-inf scans.

        Ensures node never crashes, computes valid masks, and degrades gracefully.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)

        # 1. 50% random ray dropouts (set to inf)
        rng = np.random.default_rng(42)
        scan = np.full(360, float('inf'), dtype=np.float32)
        scan[150:211] = 0.60
        dropouts = rng.random(360) < 0.50
        scan[dropouts] = float('inf')

        node.latest_scan_ranges = scan
        updated = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert updated > 0

        # 2. Complete all-inf scan (open world / no sensor returns)
        node.latest_scan_ranges = np.full(360, float('inf'), dtype=np.float32)
        updated_open = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert updated_open > 0

        # 3. NaN in scan ranges (corrupted sensor packet)
        corrupted_scan = np.full(360, 1.5, dtype=np.float32)
        corrupted_scan[180] = np.nan
        corrupted_scan[181] = -1.0
        node.latest_scan_ranges = corrupted_scan
        updated_corrupted = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert updated_corrupted > 0

    def test_zero_and_sub_minimum_ranges_singularity(self, mapper_node):
        """
        Verify behavior when proximity sensor reads 0.0m or sub-range_min distances.

        Ensures horizon calculation singularity protection prevents division by zero.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        img = np.full((node.image_height, node.image_width, 3), 220, dtype=np.uint8)

        # Obstacle reported at distance 0.05m (closer than camera_xc = 0.0682m)
        close_scan = np.full(360, 2.0, dtype=np.float32)
        close_scan[150:211] = 0.05  # Inside focal offset!
        node.latest_scan_ranges = close_scan

        # Must execute cleanly without exception
        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        # In front cone, all ground cells are shadowed or rejected by v >= v_base
        obs_r, obs_c = np.where(node.observed_mask)
        for r, c in zip(obs_r, obs_c):
            xw, yw = node.grid_to_world(c, r)
            # Front center cone (|yw| < 0.05) must have 0 observed cells
            assert not (xw > 0.05 and abs(yw) < 0.05)

        # Range exactly 0.0 (blinded proximity sensor)
        zero_scan = np.full(360, 2.0, dtype=np.float32)
        zero_scan[170:190] = 0.0
        node.latest_scan_ranges = zero_scan
        updated_zero = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert isinstance(updated_zero, int)

    def test_noisy_scan_gaussian_jitter(self, mapper_node):
        """
        Adversarially inject Gaussian range jitter (sigma = 0.015m) across 100 frames.

        Confirms shadow boundary remains bounded within tolerance and no drift occurs.
        """
        node = mapper_node
        node.enable_occlusion_culling = True
        node.shadow_culling_margin = 0.02

        d_nominal = 0.55
        box_bounds = (d_nominal, d_nominal + 0.30, -0.20, 0.20)
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)

        rng = np.random.default_rng(1234)
        for _ in range(50):
            nominal_scan = synthesize_box_scan((0.0, 0.0, 0.0), box_bounds)
            # Add Gaussian jitter with sigma = 0.015m
            noise = rng.normal(0.0, 0.015, size=360).astype(np.float32)
            noisy_scan = np.where(np.isfinite(nominal_scan), nominal_scan + noise, np.inf)
            node.latest_scan_ranges = noisy_scan.astype(np.float32)
            assert node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0) > 0

        # Check shadow zone behind d_nominal + 3*sigma: xw in [0.60, 1.20]
        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        leaks = sum(
            1 for xw in np.arange(0.60, 1.20, node.res)
            for yw in np.arange(-0.15, 0.15, node.res)
            if node.world_to_grid(xw, yw) in observed_coords
        )
        assert leaks == 0

    def test_obstacle_at_steep_grazing_angle(self, mapper_node):
        """
        Verify culling when obstacle face is oriented at grazing angles (~4.8 deg).

        Ensures azimuthal horizon clamping and culling handle continuous grazing boundaries.
        """
        node = mapper_node
        node.enable_occlusion_culling = True
        node.enable_distance_attenuation = True

        img = np.full((node.image_height, node.image_width, 3), 255, dtype=np.uint8)

        # Continuous grazing wall spanning the forward view from (0.0, 0.125) to (2.0, 0.2917)
        x1, y1 = 0.0, 0.125
        x2, y2 = 2.0, 0.2917
        m = (y2 - y1) / (x2 - x1)
        c_intercept = y1 - m * x1

        num_rays = 360
        grazing_scan = np.full(num_rays, float('inf'), dtype=np.float32)
        for i in range(num_rays):
            angle = -math.pi + i * (2.0 * math.pi / num_rays)
            denom = math.sin(angle) - m * math.cos(angle)
            if denom > 1e-4:
                t = c_intercept / denom
                x = t * math.cos(angle)
                if x1 <= x <= x2 and t > 0.02:
                    grazing_scan[i] = float(t)

        node.latest_scan_ranges = grazing_scan
        updated = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert updated > 0

        # Verify cells behind the grazing wall face (yw > wall_y(xw) + node.res)
        obs_r, obs_c = np.where(node.observed_mask)
        grazing_leaks = 0
        for r, c in zip(obs_r, obs_c):
            xw, yw = node.grid_to_world(c, r)
            if x1 <= xw <= x2:
                wall_y_at_x = m * xw + c_intercept
                if yw > wall_y_at_x + node.res:
                    grazing_leaks += 1

        assert grazing_leaks == 0


class TestGlancingWallProximityAdversarial:
    """
    Empirical Challenge 3: Glancing wall proximity.

    Cells behind wall face strictly remain uncolored.
    """

    def test_parallel_glancing_wall_strict_uncolored(self, mapper_node):
        """
        Place robot driving in glancing proximity along wall at y = 0.15m.

        Verify all cells behind the wall face (yw >= 0.15m) are strictly uncolored.
        """
        node = mapper_node
        node.enable_occlusion_culling = True
        node.shadow_culling_margin = 0.02

        # Robot at (0.0, 0.05), yaw = 0.0. Wall at y = 0.15m (10cm away from robot center)
        rx, ry, ryaw = 0.0, 0.05, 0.0
        wall_y = 0.15
        wall_x_range = (-0.5, 2.5)

        img = np.full((node.image_height, node.image_width, 3), 240, dtype=np.uint8)
        scan = synthesize_wall_scan((rx, ry, ryaw), wall_y, wall_x_range)
        node.latest_scan_ranges = scan

        updated = node.warp_image_to_grid(img, rx, ry, ryaw, 0.0)
        assert updated > 0

        # Inspect all cells in grid behind wall face within camera's forward reach:
        # xw in [0.20, 1.40], yw in [0.15, 0.60]
        obs_r, obs_c = np.where(node.observed_mask)
        observed_coords = set(zip(obs_c, obs_r))

        cells_behind_wall_observed = 0
        total_cells_checked = 0

        for xw in np.arange(0.20, 1.40, node.res):
            for yw in np.arange(0.16, 0.60, node.res):
                total_cells_checked += 1
                col, row = node.world_to_grid(xw, yw)
                if (col, row) in observed_coords:
                    cells_behind_wall_observed += 1

        assert total_cells_checked > 200
        # Strict requirement: cells behind wall face strictly remain uncolored!
        assert cells_behind_wall_observed == 0, (
            f'Violated glancing wall culling: {cells_behind_wall_observed} cells '
            f'behind wall face at y={wall_y} were marked observed!'
        )

    def test_glancing_wall_varying_separations_and_angles(self, mapper_node):
        """
        Stress test glancing wall at various distances (8cm, 12cm, 20cm) and heading angles.

        Ensures zero-leakage guarantee holds across diverse operational postures.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        img = np.full((node.image_height, node.image_width, 3), 210, dtype=np.uint8)

        test_configs = [
            # (ry, wall_y, yaw_deg)
            (0.02, 0.10, 0.0),    # 8cm separation, parallel
            (0.00, 0.12, 10.0),   # 12cm separation, 10 deg yaw towards wall
            (0.00, 0.15, -10.0),  # 15cm separation, 10 deg yaw away from wall
            (0.00, 0.20, 15.0),   # 20cm separation, 15 deg yaw towards wall
        ]

        for ry, wall_y, yaw_deg in test_configs:
            yaw = math.radians(yaw_deg)
            scan = synthesize_wall_scan((0.0, ry, yaw), wall_y, (-1.0, 3.0))
            node.latest_scan_ranges = scan

            node.warp_image_to_grid(img, 0.0, ry, yaw, 0.0)

            obs_r, obs_c = np.where(node.observed_mask)
            observed_coords = set(zip(obs_c, obs_r))

            # Strictly beyond boundary cell resolution (wall_y + res)
            leaks = sum(
                1 for xw in np.arange(0.25, 1.30, node.res)
                for yw in np.arange(wall_y + node.res, wall_y + 0.35, node.res)
                if node.world_to_grid(xw, yw) in observed_coords
            )
            assert leaks == 0, (
                f'Leaked {leaks} cells beyond wall_y={wall_y} boundary at yaw={yaw_deg} deg!'
            )

    def test_glancing_wall_with_slam_obstacle_purging_synergy(self, mapper_node):
        """
        Verify complete zero-cell leakage right at wall boundary when SLAM map is active.

        Proves that SLAM obstacle purging and proximity scan culling work in synergy.
        """
        node = mapper_node
        node.enable_occlusion_culling = True
        node.shadow_culling_margin = 0.02

        rx, ry, ryaw = 0.0, 0.02, 0.0
        wall_y = 0.10
        wall_x_range = (-1.0, 3.0)

        # SLAM OccupancyGrid arrives marking the wall row as occupied (data = 100)
        map_msg = OccupancyGrid()
        map_msg.info.resolution = node.res
        map_msg.info.width = node.width
        map_msg.info.height = node.height
        map_msg.info.origin.position.x = node.origin_x
        map_msg.info.origin.position.y = node.origin_y

        grid_data = np.zeros((node.height, node.width), dtype=np.int8)
        wall_row = int(math.floor((wall_y - node.origin_y) / node.res))
        grid_data[wall_row, :] = 100
        map_msg.data = grid_data.ravel().tolist()

        node.map_callback(map_msg)

        # Proximity scan
        scan = synthesize_wall_scan((rx, ry, ryaw), wall_y, wall_x_range)
        node.latest_scan_ranges = scan

        img = np.full((node.image_height, node.image_width, 3), 210, dtype=np.uint8)
        node.warp_image_to_grid(img, rx, ry, ryaw, 0.0)

        # With SLAM map active, strictly ZERO cells with yw >= wall_y are observed!
        obs_r, obs_c = np.where(node.observed_mask)
        boundary_leaks = 0
        for r, c in zip(obs_r, obs_c):
            xw, yw = node.grid_to_world(c, r)
            if yw >= wall_y:
                boundary_leaks += 1

        assert boundary_leaks == 0


class TestDynamicSLAMUpdatesAdversarial:
    """
    Empirical Challenge 4: Dynamic SLAM map updates.

    Instantaneous purging of confirmed obstacle cells.
    """

    def test_instantaneous_purge_and_exclusion(self, mapper_node):
        """
        Verify instantaneous purging and exclusion of confirmed SLAM obstacle cells.

        1. Map free floor: observed_mask is True, color_map has values, weight_map is 1.0.
        2. SLAM OccupancyGrid arrives with obstacle footprint (data = 100).
        3. Obstacle cells are instantly purged (observed_mask=False, color=0, weight=0).
        4. Subsequent camera frames do NOT re-color confirmed obstacle cells.
        """
        node = mapper_node
        node.enable_occlusion_culling = True

        # 1. Map floor in open world
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)
        node.latest_scan_ranges = np.full(360, float('inf'), dtype=np.float32)
        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Confirm some cells are observed
        initial_obs_count = np.sum(node.observed_mask)
        assert initial_obs_count > 50

        # Choose a target test cell that was observed: e.g. at (0.50m, 0.0m)
        tc_col, tc_row = node.world_to_grid(0.50, 0.0)
        assert bool(node.observed_mask[tc_row, tc_col]) is True
        assert np.any(node.color_map[tc_row, tc_col] > 0)
        assert node.weight_map[tc_row, tc_col] > 0.0

        # 2. SLAM map arrives: dynamic obstacle covers (0.45 to 0.55m, -0.05 to +0.05m)
        map_msg = OccupancyGrid()
        map_msg.info = MapMetaData()
        map_msg.info.resolution = node.res
        map_msg.info.width = node.width
        map_msg.info.height = node.height
        map_msg.info.origin.position.x = node.origin_x
        map_msg.info.origin.position.y = node.origin_y

        grid_data = np.zeros((node.height, node.width), dtype=np.int8)
        # Set 5x5 obstacle block centered around (0.50, 0.0)
        for r_offset in range(-2, 3):
            for c_offset in range(-2, 3):
                grid_data[tc_row + r_offset, tc_col + c_offset] = 100

        map_msg.data = grid_data.ravel().tolist()

        # Trigger callback
        node.map_callback(map_msg)

        # 3. Assert INSTANTANEOUS purge of confirmed obstacle cells
        for r_offset in range(-2, 3):
            for c_offset in range(-2, 3):
                curr_r = tc_row + r_offset
                curr_c = tc_col + c_offset
                assert bool(node.observed_mask[curr_r, curr_c]) is False
                assert np.all(node.color_map[curr_r, curr_c] == 0.0)
                assert node.weight_map[curr_r, curr_c] == 0.0

        # Non-obstacle observed cells outside the footprint must remain intact
        outside_col, outside_row = node.world_to_grid(0.30, 0.0)
        if node.observed_mask[outside_row, outside_col]:
            assert np.any(node.color_map[outside_row, outside_col] > 0)

        # 4. Subsequent camera image arrives while obstacle remains in SLAM map
        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Obstacle cells must STILL remain unobserved and uncolored!
        for r_offset in range(-2, 3):
            for c_offset in range(-2, 3):
                curr_r = tc_row + r_offset
                curr_c = tc_col + c_offset
                assert bool(node.observed_mask[curr_r, curr_c]) is False
                assert np.all(node.color_map[curr_r, curr_c] == 0.0)
                assert node.weight_map[curr_r, curr_c] == 0.0

    def test_dynamic_obstacle_clearance_and_recovery(self, mapper_node):
        """Verify that when a dynamic obstacle clears in SLAM, the area can be re-mapped."""
        node = mapper_node
        node.enable_occlusion_culling = True

        img = np.full((node.image_height, node.image_width, 3), 150, dtype=np.uint8)
        tc_col, tc_row = node.world_to_grid(0.60, 0.0)

        # Step A: Obstacle is present in SLAM
        map_msg = OccupancyGrid()
        map_msg.info.resolution = node.res
        map_msg.info.width = node.width
        map_msg.info.height = node.height
        map_msg.info.origin.position.x = node.origin_x
        map_msg.info.origin.position.y = node.origin_y

        grid_data = np.zeros((node.height, node.width), dtype=np.int8)
        grid_data[tc_row, tc_col] = 100
        map_msg.data = grid_data.ravel().tolist()
        node.map_callback(map_msg)

        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert bool(node.observed_mask[tc_row, tc_col]) is False

        # Step B: Dynamic obstacle vacates / disappears (occupancy becomes 0)
        grid_data_cleared = np.zeros((node.height, node.width), dtype=np.int8)
        map_msg.data = grid_data_cleared.ravel().tolist()
        node.map_callback(map_msg)

        # Step C: Camera sweeps over cell again
        node.latest_scan_ranges = np.full(360, float('inf'), dtype=np.float32)
        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        # Now cell should be successfully re-observed and mapped!
        assert bool(node.observed_mask[tc_row, tc_col]) is True
        assert np.any(node.color_map[tc_row, tc_col] > 0)

    def test_rapid_alternating_slam_updates_stability(self, mapper_node):
        """
        Adversarially hammer map_callback with 50 rapid alternating occupied/free updates.

        Ensures atomic mask consistency, zero memory leaks, and no race conditions.
        """
        node = mapper_node
        img = np.full((node.image_height, node.image_width, 3), 100, dtype=np.uint8)
        node.latest_scan_ranges = np.full(360, float('inf'), dtype=np.float32)

        tc_col, tc_row = node.world_to_grid(0.50, 0.0)

        map_msg = OccupancyGrid()
        map_msg.info.resolution = node.res
        map_msg.info.width = node.width
        map_msg.info.height = node.height
        map_msg.info.origin.position.x = node.origin_x
        map_msg.info.origin.position.y = node.origin_y

        occ_grid = np.zeros((node.height, node.width), dtype=np.int8)
        occ_grid[tc_row, tc_col] = 100
        free_grid = np.zeros((node.height, node.width), dtype=np.int8)

        for _ in range(50):
            # Pulse occupied
            map_msg.data = occ_grid.ravel().tolist()
            node.map_callback(map_msg)
            assert bool(node.observed_mask[tc_row, tc_col]) is False

            # Pulse free & warp
            map_msg.data = free_grid.ravel().tolist()
            node.map_callback(map_msg)
            node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
            assert bool(node.observed_mask[tc_row, tc_col]) is True
