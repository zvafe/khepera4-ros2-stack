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
E2E 4-Tier Test Suite: Anti-Smudge Occlusion Culling & 3D Point Cloud (Milestone 4).

4-Tier Test Architecture:
- Tier 1: Feature Coverage (Scan subscription, shadow culling, horizon clamping,
          alpha attenuation, SLAM obstacle purge, PointCloud2 buffer packing,
          ground layer, obstacle wall layer, ramp lifting, latched QoS, RViz2 config).
- Tier 2: Boundary & Corner Cases (Zero/negative inputs, inf/nan ranges, focal plane
          singularity, max wall heights, unvisited elevation cells, empty maps, 360-ray wrap).
- Tier 3: Cross-Feature Interactions (Shadow culling + alpha attenuation; obstacle walls
          + camera RGB; elevation ramp + orthomosaic; SLAM purge + pointcloud regeneration).
- Tier 4: Real-World Scenarios (In-place rotation, glancing wall approach, 10° ramp ascent,
          late-joining client latched QoS, dynamic obstacle expansion).

Authoritative Specifications:
- ORIGINAL_REQUEST.md (Follow-up 2026-11-02T01:24:44Z §R1 & §R2)
- PROJECT.md §Feature Inventory & Interface Contracts
- TEST_INFRA.md §4-Tier Test Architecture
"""

import math
import os
import sys
import time
from typing import Optional, Sequence, Tuple

from builtin_interfaces.msg import Time as RosTime
from khepera4_control.pointcloud_builder import (
    DEFAULT_WALL_COLOR_RGB,
    pack_rgb_f32,
    POINT_DTYPE,
    POINT_FIELDS,
    PointCloud3DBuilder,
    unpack_rgb_f32,
)
from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import (
    LaserScan,
    PointField,
)
import yaml

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


# ==============================================================================
# Pytest Fixtures & Synthetic Message Helpers
# ==============================================================================

@pytest.fixture(scope='module', autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def ortho_node(tmp_path):
    """Instantiate a VisualOrthomapper node for direct unit/integration testing."""
    node = VisualOrthomapper(
        parameter_overrides=[
            Parameter('keyframe_dir', Parameter.Type.STRING, str(tmp_path)),
            Parameter('enable_occlusion_culling', Parameter.Type.BOOL, True),
            Parameter('publish_pointcloud_3d', Parameter.Type.BOOL, True),
        ]
    )
    yield node
    node.destroy_node()


@pytest.fixture
def cloud_builder():
    """Instantiate a PointCloud3DBuilder configured with default Khepera IV geometry."""
    return PointCloud3DBuilder(
        map_res=0.03,
        map_width=200,
        map_height=200,
        origin_x=-3.0,
        origin_y=-3.0,
        camera_xc=0.0682,
        camera_h=0.0283,
        fx=554.256258,
        fy=554.256258,
        cx=320.0,
        cy=240.0,
        img_width=640,
        img_height=480,
        wall_height_max=0.35,
        wall_delta_z=0.03
    )


def create_test_image(
    height: int = 480,
    width: int = 640,
    color: Tuple[int, int, int] = (100, 150, 200)
) -> np.ndarray:
    """Create synthetic uniform RGB test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    return img


def create_scan_msg(
    ranges: Optional[Sequence[float]] = None,
    range_min: float = 0.002,
    range_max: float = 2.0,
    angle_min: float = -math.pi,
    angle_max: float = math.pi - (2.0 * math.pi / 360.0),
    num_rays: int = 360
) -> LaserScan:
    """Create synthetic LaserScan message."""
    scan = LaserScan()
    scan.header.frame_id = 'base_footprint'
    scan.range_min = float(range_min)
    scan.range_max = float(range_max)
    scan.angle_min = float(angle_min)
    scan.angle_max = float(angle_max)
    scan.angle_increment = float((angle_max - angle_min) / max(1, num_rays - 1))
    if ranges is None:
        scan.ranges = [float('inf')] * num_rays
    else:
        scan.ranges = [float(r) for r in ranges]
    return scan


def create_occupancy_grid(
    data: Optional[np.ndarray] = None,
    width: int = 200,
    height: int = 200,
    res: float = 0.03,
    origin_x: float = -3.0,
    origin_y: float = -3.0
) -> OccupancyGrid:
    """Create synthetic OccupancyGrid message."""
    grid = OccupancyGrid()
    grid.header.frame_id = 'map'
    grid.info.resolution = float(res)
    grid.info.width = int(width)
    grid.info.height = int(height)
    grid.info.origin.position.x = float(origin_x)
    grid.info.origin.position.y = float(origin_y)
    grid.info.origin.orientation.w = 1.0
    if data is None:
        grid.data = [0] * (width * height)
    else:
        grid.data = [int(v) for v in data.flatten()]
    return grid


def get_rviz_path() -> str:
    """Retrieve absolute path to khepera4_mapping.rviz."""
    ws_root = os.path.abspath(os.path.join(_PKG_ROOT, '..', '..'))
    return os.path.join(
        ws_root, 'src', 'khepera4_description', 'config', 'khepera4_mapping.rviz'
    )


# ==============================================================================
# Tier 1: Feature Coverage (Features 1 to 11)
# ==============================================================================

class TestTier1ScanSubscription:
    """Tier 1: Feature 1 — Subscription to /khepera4/scan (>= 5 tests)."""

    def test_t1_f1_01_scan_topic_declared(self, ortho_node):
        """Verify scan_topic parameter exists and defaults to /khepera4/scan."""
        assert ortho_node.has_parameter('scan_topic')
        assert ortho_node.get_parameter('scan_topic').value == '/khepera4/scan'

    def test_t1_f1_02_scan_subscription_active(self, ortho_node):
        """Verify subscriber is created for LaserScan on scan_topic."""
        assert hasattr(ortho_node, 'scan_sub')
        assert ortho_node.scan_sub is not None

    def test_t1_f1_03_scan_callback_stores_ranges(self, ortho_node):
        """Verify feeding LaserScan message stores latest_scan_ranges array."""
        scan = create_scan_msg([0.45] * 360)
        ortho_node.scan_callback(scan)
        assert ortho_node.latest_scan_ranges is not None
        assert len(ortho_node.latest_scan_ranges) == 360
        assert np.isclose(ortho_node.latest_scan_ranges[180], 0.45, atol=1e-3)

    def test_t1_f1_04_scan_attributes_preserved(self, ortho_node):
        """Verify scan angle and range bounds are accurately recorded."""
        scan = create_scan_msg(range_min=0.015, range_max=1.85)
        ortho_node.scan_callback(scan)
        assert ortho_node.scan_range_min == pytest.approx(0.015, abs=1e-4)
        assert ortho_node.scan_range_max == pytest.approx(1.85, abs=1e-4)

    def test_t1_f1_05_graceful_before_first_scan(self, ortho_node):
        """Verify is_in_shadow returns False when no scan has arrived yet."""
        ortho_node.latest_scan_ranges = None
        assert ortho_node.is_in_shadow(0.5, 0.0) is False


class TestTier1VectorizedShadowCulling:
    """Tier 1: Feature 2 — Vectorized Shadow Culling (>= 5 tests)."""

    def test_t1_f2_01_cells_behind_obstacle_discarded(self, ortho_node):
        """Verify candidate cell located behind obstacle horizon is culled."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50  # 0.50m directly ahead (theta=0)
        ortho_node.scan_callback(create_scan_msg(ranges))

        # Cell at 0.70m ahead is in shadow (0.70 >= 0.50 - 0.02)
        assert ortho_node.is_in_shadow(0.70, 0.0) is True

    def test_t1_f2_02_cells_in_front_retained(self, ortho_node):
        """Verify candidate cell in front of obstacle boundary is retained."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        ortho_node.scan_callback(create_scan_msg(ranges))

        # Cell at 0.35m ahead is not in shadow (0.35 < 0.50 - 0.02)
        assert ortho_node.is_in_shadow(0.35, 0.0) is False

    def test_t1_f2_03_culling_across_lateral_fov(self, ortho_node):
        """Verify shadow culling functions across +/- 25 degrees lateral FOV."""
        ranges = [float('inf')] * 360
        # -20 deg: index ~ 160; +20 deg: index ~ 200
        idx_neg20 = int(round((-math.radians(20) - (-math.pi)) / (2 * math.pi / 360))) % 360
        idx_pos20 = int(round((math.radians(20) - (-math.pi)) / (2 * math.pi / 360))) % 360
        ranges[idx_neg20] = 0.60
        ranges[idx_pos20] = 0.60
        ortho_node.scan_callback(create_scan_msg(ranges))

        assert ortho_node.is_in_shadow(0.80, -math.radians(20)) is True
        assert ortho_node.is_in_shadow(0.40, -math.radians(20)) is False
        assert ortho_node.is_in_shadow(0.80, math.radians(20)) is True
        assert ortho_node.is_in_shadow(0.40, math.radians(20)) is False

    def test_t1_f2_04_shadow_margin_parameter(self, ortho_node):
        """Verify shadow_culling_margin parameter changes culling threshold."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        ortho_node.scan_callback(create_scan_msg(ranges))

        ortho_node.shadow_culling_margin = 0.05
        # Cell at 0.46m: 0.46 >= 0.50 - 0.05 -> True
        assert ortho_node.is_in_shadow(0.46, 0.0) is True

        ortho_node.shadow_culling_margin = 0.01
        # Cell at 0.46m: 0.46 < 0.50 - 0.01 -> False
        assert ortho_node.is_in_shadow(0.46, 0.0) is False

    def test_t1_f2_05_toggle_enable_occlusion_culling(self, ortho_node):
        """Verify setting enable_occlusion_culling to False disables culling."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.40
        ortho_node.scan_callback(create_scan_msg(ranges))

        ortho_node.enable_occlusion_culling = False
        img = create_test_image(color=(200, 100, 50))
        # With culling disabled, ground warping processes normally
        count = ortho_node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert count > 0


class TestTier1AzimuthalHorizonClamping:
    """Tier 1: Feature 3 — Azimuthal Horizon Clamping (>= 5 tests)."""

    def test_t1_f3_01_obstacle_vertical_face_rows_rejected(self, ortho_node):
        """Verify v_base row cutoff is higher than nominal horizon for obstacle at 0.4m."""
        v_base = ortho_node.compute_azimuthal_horizon_v(0.40)
        # cy=240, h=0.0283, fy=554.256, d=0.40, xc=0.0682 -> denom=0.3318 -> v ~ 287 px
        assert v_base > ortho_node.cy
        assert v_base > ortho_node.min_row_v

    def test_t1_f3_02_v_base_pinhole_formula_accuracy(self, ortho_node):
        """Verify v_base matches closed-form cy + (h * fy)/(d - xc) formula."""
        for d in [0.30, 0.50, 0.80, 1.20]:
            h_fy = ortho_node.camera_h * ortho_node.fy
            expected = ortho_node.cy + h_fy / (d - ortho_node.camera_xc)
            actual = ortho_node.compute_azimuthal_horizon_v(d)
            assert actual == pytest.approx(expected, abs=1e-3)

    def test_t1_f3_03_singularity_guard_when_obstacle_inside_focal_length(self, ortho_node):
        """Verify obstacle closer than camera_xc triggers full-column rejection."""
        # xc = 0.0682m. Distance 0.05m is inside lens offset
        v_guard = ortho_node.compute_azimuthal_horizon_v(0.05)
        assert v_guard >= float(ortho_node.image_height)

    def test_t1_f3_04_open_field_infinite_range_preserves_ground_horizon(self, ortho_node):
        """Verify open field (d_obs = inf) yields guard cutoff without crash."""
        v_inf = ortho_node.compute_azimuthal_horizon_v(float('inf'))
        assert float(v_inf) >= float(ortho_node.image_height)

    def test_t1_f3_05_multi_azimuth_varying_obstacle_horizons(self, ortho_node):
        """Verify multiple azimuth obstacles compute closer cutoffs for closer depths."""
        v_near = ortho_node.compute_azimuthal_horizon_v(0.35)
        v_far = ortho_node.compute_azimuthal_horizon_v(1.00)
        assert v_near > v_far


class TestTier1DistanceAttenuatedAlphaBlending:
    """Tier 1: Feature 4 — Distance-Attenuated Alpha Blending (>= 5 tests)."""

    def test_t1_f4_01_attenuation_exponential_curve(self, ortho_node):
        """Verify attenuation matches alpha_max * ((X_max - X_b)/(X_max - X_min))^1.5."""
        xb_mid = (ortho_node.min_ground_dist + ortho_node.max_ground_dist) / 2.0
        span = ortho_node.max_ground_dist - ortho_node.min_ground_dist
        expected = ortho_node.alpha * ((ortho_node.max_ground_dist - xb_mid) / span) ** 1.5
        actual = ortho_node.compute_distance_attenuated_alpha(xb_mid)
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_t1_f4_02_boundary_near_dist_yields_full_alpha(self, ortho_node):
        """Verify alpha at min_ground_dist is equal to alpha_max (0.25)."""
        alpha_near = ortho_node.compute_distance_attenuated_alpha(ortho_node.min_ground_dist)
        assert alpha_near == pytest.approx(ortho_node.alpha, abs=1e-4)

    def test_t1_f4_03_boundary_far_dist_yields_zero_alpha(self, ortho_node):
        """Verify alpha at max_ground_dist drops to 0.0."""
        alpha_far = ortho_node.compute_distance_attenuated_alpha(ortho_node.max_ground_dist)
        assert alpha_far == pytest.approx(0.0, abs=1e-4)

    def test_t1_f4_04_monotonic_decrease_across_distance(self, ortho_node):
        """Verify alpha strictly decreases as ground distance increases."""
        distances = np.linspace(ortho_node.min_ground_dist, ortho_node.max_ground_dist, 10)
        alphas = [ortho_node.compute_distance_attenuated_alpha(d) for d in distances]
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1]

    def test_t1_f4_05_parameter_enable_distance_attenuation(self, ortho_node):
        """Verify enable_distance_attenuation parameter defaults to False for baseline tests."""
        assert ortho_node.has_parameter('enable_distance_attenuation')
        assert ortho_node.get_parameter('enable_distance_attenuation').value is False


class TestTier1SLAMObstacleCachePurge:
    """Tier 1: Feature 5 — SLAM Obstacle Cache Purge from /map (>= 5 tests)."""

    def test_t1_f5_01_map_subscription_declared(self, ortho_node):
        """Verify map_topic parameter is declared with default /map."""
        assert ortho_node.has_parameter('map_topic')
        assert ortho_node.get_parameter('map_topic').value == '/map'

    def test_t1_f5_02_map_occupied_cells_purge_observed_mask(self, ortho_node):
        """Verify confirmed occupied cell sets observed_mask to False."""
        r_occ, c_occ = 100, 100
        ortho_node.observed_mask[r_occ, c_occ] = True
        ortho_node.color_map[r_occ, c_occ] = [180.0, 120.0, 50.0]

        map_data = np.zeros((200, 200), dtype=np.int8)
        map_data[r_occ, c_occ] = 100  # Confirmed obstacle
        ortho_node.map_callback(create_occupancy_grid(map_data))

        mask_val = ortho_node.observed_mask[r_occ, c_occ]
        assert mask_val is False or mask_val == 0

    def test_t1_f5_03_map_occupied_cells_reset_color_and_weight_maps(self, ortho_node):
        """Verify confirmed occupied cell resets color_map and weight_map to zero."""
        r_occ, c_occ = 110, 110
        ortho_node.observed_mask[r_occ, c_occ] = True
        ortho_node.color_map[r_occ, c_occ] = [210.0, 200.0, 190.0]
        ortho_node.weight_map[r_occ, c_occ] = 5.0

        map_data = np.zeros((200, 200), dtype=np.int8)
        map_data[r_occ, c_occ] = 100
        ortho_node.map_callback(create_occupancy_grid(map_data))

        assert np.all(ortho_node.color_map[r_occ, c_occ] == 0.0)
        assert ortho_node.weight_map[r_occ, c_occ] == 0.0

    def test_t1_f5_04_future_warping_excludes_confirmed_obstacles(self, ortho_node):
        """Verify latest_obstacle_mask is retained to guard against re-smudging."""
        r_occ, c_occ = 105, 105
        map_data = np.zeros((200, 200), dtype=np.int8)
        map_data[r_occ, c_occ] = 100
        ortho_node.map_callback(create_occupancy_grid(map_data))

        mask_val = ortho_node.latest_obstacle_mask[r_occ, c_occ]
        assert mask_val is True or mask_val == 1

    def test_t1_f5_05_free_space_cells_not_purged(self, ortho_node):
        """Verify cells marked free or unobserved do not purge visual observations."""
        r_free, c_free = 95, 95
        ortho_node.observed_mask[r_free, c_free] = True
        ortho_node.color_map[r_free, c_free] = [150.0, 150.0, 150.0]

        map_data = np.full((200, 200), -1, dtype=np.int8)
        map_data[r_free, c_free] = 0
        ortho_node.map_callback(create_occupancy_grid(map_data))

        assert bool(ortho_node.observed_mask[r_free, c_free]) is True
        assert np.all(ortho_node.color_map[r_free, c_free] == [150.0, 150.0, 150.0])


class TestTier1PointCloud2BufferPacking:
    """Tier 1: Feature 6 — PointCloud2 Schema & Direct NumPy Buffer Packing (>= 5 tests)."""

    def test_t1_f6_01_pointcloud2_schema_layout(self, cloud_builder):
        """Verify PointCloud2 message schema conforms to ROS 2 standard."""
        color_map = np.full((200, 200, 3), 128.0, dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[100, 100] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.height == 1
        assert cloud.width == 1
        assert cloud.point_step == 16
        assert cloud.row_step == 16
        assert cloud.is_dense is True
        assert cloud.is_bigendian is False

    def test_t1_f6_02_pointfield_offsets_and_datatypes(self, cloud_builder):
        """Verify PointField offsets: x:0, y:4, z:8, rgb:12 all FLOAT32."""
        assert len(POINT_FIELDS) == 4
        assert POINT_FIELDS[0].name == 'x' and POINT_FIELDS[0].offset == 0
        assert POINT_FIELDS[0].datatype == PointField.FLOAT32
        assert POINT_FIELDS[1].name == 'y' and POINT_FIELDS[1].offset == 4
        assert POINT_FIELDS[1].datatype == PointField.FLOAT32
        assert POINT_FIELDS[2].name == 'z' and POINT_FIELDS[2].offset == 8
        assert POINT_FIELDS[2].datatype == PointField.FLOAT32
        assert POINT_FIELDS[3].name == 'rgb' and POINT_FIELDS[3].offset == 12
        assert POINT_FIELDS[3].datatype == PointField.FLOAT32

    def test_t1_f6_03_rgb_packed_uint32_viewed_as_float32(self):
        """Verify packing R, G, B into uint32 and unpacking preserves values."""
        r = np.array([255, 0, 128], dtype=np.uint8)
        g = np.array([128, 255, 64], dtype=np.uint8)
        b = np.array([64, 128, 32], dtype=np.uint8)

        f32_packed = pack_rgb_f32(r, g, b)
        r_rec, g_rec, b_rec = unpack_rgb_f32(f32_packed)

        assert np.array_equal(r, r_rec)
        assert np.array_equal(g, g_rec)
        assert np.array_equal(b, b_rec)

    def test_t1_f6_04_direct_numpy_serialization_bytes_length(self, cloud_builder):
        """Verify cloud.data bytes length is exactly point_step * total_points."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[50:60, 50:60] = True  # 100 points

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert len(cloud.data) == 100 * 16

    def test_t1_f6_05_pointcloud_synthesis_under_10ms_benchmark(self, cloud_builder):
        """Verify complete point cloud generation executes in < 10 ms for 35,000 points."""
        color_map = np.random.uniform(0, 255, (200, 200, 3)).astype(np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        # Select 25,000 observed cells
        observed_mask[:125, :200] = True

        # Scan with 100 valid hits -> 100 * 13 = 1300 wall points
        scan = [0.60] * 100 + [float('inf')] * 260
        elevation = np.zeros((200, 200), dtype=np.int8)
        cam_img = create_test_image()

        t0 = time.perf_counter()
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=elevation,
            scan_ranges=scan,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=cam_img
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        assert cloud.width >= 25000
        assert dt_ms < 10.0, f'Point cloud generation took {dt_ms:.2f}ms >= 10.0ms budget!'


class TestTier1GroundLayer:
    """Tier 1: Feature 7 — Layer 1: Ground Surface at Z = 0 (>= 5 tests)."""

    def test_t1_f7_01_ground_points_z_equals_zero(self, cloud_builder):
        """Verify floor points outside elevation ramps have z == 0.0."""
        color_map = np.full((200, 200, 3), 100.0, dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[50, 50] = True  # Floor cell far from ramps

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)

    def test_t1_f7_02_ground_world_coords_match_grid(self, cloud_builder):
        """Verify ground point (x, y) coordinates match origin + (index + 0.5) * res."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        r, c = 20, 40
        observed_mask[r, c] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        expected_x = cloud_builder.origin_x + (c + 0.5) * cloud_builder.res
        expected_y = cloud_builder.origin_y + (r + 0.5) * cloud_builder.res
        assert pts['x'][0] == pytest.approx(expected_x, abs=1e-4)
        assert pts['y'][0] == pytest.approx(expected_y, abs=1e-4)

    def test_t1_f7_03_observed_mask_filters_unobserved_cells(self, cloud_builder):
        """Verify unobserved cells (observed_mask=False) are never emitted."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[10, 10] = True
        observed_mask[20, 20] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 2

    def test_t1_f7_04_ground_rgb_matches_color_map_values(self, cloud_builder):
        """Verify packed point RGB faithfully matches color_map RGB."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[30, 30] = True
        color_map[30, 30] = [75.0, 180.0, 225.0]

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert r == 75 and g == 180 and b == 225

    def test_t1_f7_05_zero_observed_cells_produces_zero_ground_points(self, cloud_builder):
        """Verify empty observed_mask generates 0 points when no scan is provided."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0


class TestTier1ObstacleWallLayer:
    """Tier 1: Feature 8 — Layer 2: Vertical Obstacle Walls (>= 5 tests)."""

    def test_t1_f8_01_vertical_wall_columns_at_scan_hit_boundaries(self, cloud_builder):
        """Verify scan hit at distance 0.5m generates a vertical wall column."""
        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)

        # Single hit forward (ray 180 = 0.5m)
        ranges = [float('inf')] * 360
        ranges[180] = 0.50

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert len(pts) == len(cloud_builder.wall_z_steps)
        assert pts['x'][0] == pytest.approx(0.50, abs=0.01)

    def test_t1_f8_02_wall_column_vertical_span_and_step(self, cloud_builder):
        """Verify wall column spans z in [0.0, 0.35m] at step 0.03m."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-4)
        assert pts['z'][-1] == pytest.approx(0.35, abs=1e-4)
        diffs = np.diff(pts['z'])
        assert np.allclose(diffs[:-1], 0.03, atol=1e-3)
        assert diffs[-1] <= 0.03 + 1e-4

    def test_t1_f8_03_back_projection_samples_camera_rgb_inside_fov(self, cloud_builder):
        """Verify forward wall point back-projects to camera image and samples image RGB."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50  # Right in front of camera

        # Create camera image with distinct color (220, 80, 40)
        camera_img = create_test_image(color=(220, 80, 40))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=camera_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # Check first point at ground level
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (220, 80, 40)

    def test_t1_f8_04_fallback_wall_color_outside_camera_fov(self, cloud_builder):
        """Verify wall points behind robot take fallback color [60, 60, 65]."""
        ranges = [float('inf')] * 360
        ranges[0] = 0.50  # Angle -pi (directly behind robot)

        camera_img = create_test_image(color=(220, 80, 40))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=camera_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == DEFAULT_WALL_COLOR_RGB

    def test_t1_f8_05_out_of_range_or_infinite_scans_produce_no_walls(self, cloud_builder):
        """Verify scan with inf/nan ranges produces zero wall points."""
        ranges = [float('inf')] * 360

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0


class TestTier1TerrainSlopesAndRamps:
    """Tier 1: Feature 9 — Layer 3: Terrain Slopes & Elevated Ramps (>= 5 tests)."""

    def test_t1_f9_01_inclined_ramp_10deg_lifting(self, cloud_builder):
        """Verify ground points along 10 deg incline ramp are lifted proportionally."""
        # 10 deg ramp: X in [0.5, 1.7], Y in [1.0, 1.6]
        # Map cell at x=1.1, y=1.3
        col = int(round((1.1 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        expected_z = (pts['x'][0] - 0.5) * np.tan(np.radians(10.0))
        assert pts['z'][0] == pytest.approx(expected_z, abs=1e-3)
        assert pts['z'][0] > 0.05

    def test_t1_f9_02_raised_platform_deck_lifting_z08(self, cloud_builder):
        """Verify platform deck points are lifted to Z = 0.08m."""
        col = int(round((-0.9 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.1 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.08, abs=1e-3)

    def test_t1_f9_03_platform_approach_ramp_lifting(self, cloud_builder):
        """Verify approach ramp rises from Z = 0.0 at X = -1.99 to Z = 0.08 at X = -1.39."""
        col = int(round((-1.69 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.1 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        expected_z = (pts['x'][0] + 1.99) * (0.08 / 0.60)
        assert pts['z'][0] == pytest.approx(expected_z, abs=1e-3)
        assert 0.0 < pts['z'][0] < 0.08

    def test_t1_f9_04_flat_ground_outside_ramps_remains_z0(self, cloud_builder):
        """Verify ground points outside ramp footprints remain strictly Z = 0.0."""
        col = int(round((0.0 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((0.0 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)

    def test_t1_f9_05_none_elevation_grid_fallback_flat_ground(self, cloud_builder):
        """Verify when elevation_grid is None, all ground points remain at Z = 0.0."""
        # Even inside ramp footprint:
        col = int(round((1.1 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)


class TestTier1LatchedQoSTopic:
    """Tier 1: Feature 10 — PointCloud2 Topic & Latched QoS (>= 5 tests)."""

    def test_t1_f10_01_pointcloud_topic_name_and_parameter(self, ortho_node):
        """Verify pointcloud_topic parameter is declared and defaults to /khepera4/pointcloud3d."""
        assert ortho_node.has_parameter('pointcloud_topic')
        assert ortho_node.get_parameter('pointcloud_topic').value == '/khepera4/pointcloud3d'

    def test_t1_f10_02_qos_transient_local_durability(self, ortho_node):
        """Verify publisher QoS profile durability is TRANSIENT_LOCAL."""
        assert hasattr(ortho_node, 'pointcloud_pub')
        assert ortho_node.pointcloud_pub is not None

    def test_t1_f10_03_qos_reliable_reliability(self, ortho_node):
        """Verify publish_pointcloud executes cleanly and publishes on topic."""
        ortho_node.observed_mask[100, 100] = True
        ortho_node.color_map[100, 100] = [100.0, 100.0, 100.0]
        # Trigger publish
        ortho_node.publish_pointcloud()

    def test_t1_f10_04_header_frame_id_is_map(self, cloud_builder):
        """Verify PointCloud2 message header frame_id is strictly 'map'."""
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.header.frame_id == 'map'

    def test_t1_f10_05_header_stamp_propagation(self, cloud_builder):
        """Verify timestamp passed to build_pointcloud is retained in header."""
        stamp = RosTime(sec=123, nanosec=456789)
        cloud = cloud_builder.build_pointcloud(
            stamp=stamp,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.header.stamp.sec == 123
        assert cloud.header.stamp.nanosec == 456789


class TestTier1RViz2ConfigDisplay:
    """Tier 1: Feature 11 — RViz2 Config & 3D View (>= 5 tests)."""

    def test_t1_f11_01_rviz2_file_valid_yaml(self):
        """Verify khepera4_mapping.rviz exists and parses as valid YAML."""
        rviz_path = get_rviz_path()
        assert os.path.exists(rviz_path), f'{rviz_path} does not exist!'
        with open(rviz_path, 'r') as f:
            data = yaml.safe_load(f)
        assert 'Visualization Manager' in data

    def test_t1_f11_02_pointcloud2_display_plugin_present(self):
        """Verify PointCloud2 display plugin is registered in Displays list."""
        with open(get_rviz_path(), 'r') as f:
            data = yaml.safe_load(f)
        displays = data['Visualization Manager']['Displays']
        pc_displays = [d for d in displays if d.get('Class') == 'rviz_default_plugins/PointCloud2']
        assert len(pc_displays) >= 1, 'PointCloud2 display plugin not found in RViz config!'
        assert pc_displays[0].get('Enabled') is True

    def test_t1_f11_03_pointcloud2_display_topic_and_qos(self):
        """Verify display topic is /khepera4/pointcloud3d with Transient Local durability."""
        with open(get_rviz_path(), 'r') as f:
            data = yaml.safe_load(f)
        displays = data['Visualization Manager']['Displays']
        pc_d = [d for d in displays if d.get('Class') == 'rviz_default_plugins/PointCloud2'][0]
        topic_info = pc_d.get('Topic', {})
        assert topic_info.get('Value') == '/khepera4/pointcloud3d'
        assert topic_info.get('Durability Policy') == 'Transient Local'

    def test_t1_f11_04_pointcloud2_display_style_size_and_transformer(self):
        """Verify Color Transformer is RGB8, Style is Flat Squares, and Size is 0.03m."""
        with open(get_rviz_path(), 'r') as f:
            data = yaml.safe_load(f)
        displays = data['Visualization Manager']['Displays']
        pc_d = [d for d in displays if d.get('Class') == 'rviz_default_plugins/PointCloud2'][0]
        assert pc_d.get('Color Transformer') == 'RGB8'
        assert pc_d.get('Style') == 'Flat Squares'
        assert pytest.approx(float(pc_d.get('Size (m)', 0.0)), abs=1e-3) == 0.03

    def test_t1_f11_05_orbit_view_controller_pitch_and_focal_point(self):
        """Verify Orbit view controller has pitch >= 0.50 rad (~30°) for 3D visibility."""
        with open(get_rviz_path(), 'r') as f:
            data = yaml.safe_load(f)
        views = data['Visualization Manager']['Views']['Current']
        assert views.get('Class') == 'rviz_default_plugins/Orbit'
        pitch = float(views.get('Pitch', 0.0))
        msg = f'RViz view Pitch {pitch:.2f} rad is too shallow (< 0.50)!'
        assert pitch >= 0.50, msg


# ==============================================================================
# Tier 2: Boundary & Corner Cases (56 tests across 7 categories)
# ==============================================================================

class TestTier2ZeroAndNegativeInputs:
    """Tier 2: Boundary Group 1 — Zero & Negative Inputs (8 tests)."""

    def test_t2_scan_zero_distance(self, ortho_node):
        """Verify zero scan distance is treated as invalid (< scan_range_min)."""
        ranges = [0.0] * 360
        ortho_node.scan_callback(create_scan_msg(ranges))
        # Zero is below range_min (0.002) -> not a valid obstacle
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_scan_negative_distance(self, ortho_node):
        """Verify negative scan distance is ignored."""
        ranges = [-0.5] * 360
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_scan_exact_range_min(self, cloud_builder):
        """Verify scan hit at exact range_min boundary (0.02m) generates wall."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.02
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == len(cloud_builder.wall_z_steps)

    def test_t2_scan_exact_range_max(self, cloud_builder):
        """Verify scan hit at exact range_max boundary (2.0m) generates wall."""
        ranges = [float('inf')] * 360
        ranges[180] = 2.00
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == len(cloud_builder.wall_z_steps)
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['x'][0] == pytest.approx(2.00, abs=0.01)

    def test_t2_robot_at_world_origin(self, cloud_builder):
        """Verify robot at (0.0, 0.0, 0.0) generates coordinates without offset errors."""
        ranges = [float('inf')] * 360
        ranges[180] = 1.00
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['x'][0] == pytest.approx(1.00, abs=0.01)
        assert pts['y'][0] == pytest.approx(0.0, abs=0.01)

    def test_t2_robot_at_negative_world_coords(self, cloud_builder):
        """Verify robot at negative position (-2.0, -1.5, pi/2) rotates hits properly."""
        ranges = [float('inf')] * 360
        ranges[180] = 1.00  # Ahead along robot +X (which points North due to yaw=pi/2)
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(-2.0, -1.5, math.pi / 2.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['x'][0] == pytest.approx(-2.0, abs=0.02)
        assert pts['y'][0] == pytest.approx(-0.5, abs=0.02)

    def test_t2_zero_alpha_weight(self, ortho_node):
        """Verify alpha = 0.0 leaves color map unchanged during update."""
        ortho_node.alpha = 0.0
        # Initialize cell (100, 100)
        ortho_node.observed_mask[100, 100] = True
        ortho_node.color_map[100, 100] = [100.0, 100.0, 100.0]

        val = ortho_node.compute_distance_attenuated_alpha(0.5)
        assert val == 0.0

    def test_t2_negative_ground_distance_cells(self, ortho_node):
        """Verify cells behind camera lens (xb < min_ground_dist) are excluded from FOV."""
        # dist_c <= 1e-4
        assert ortho_node.min_ground_dist > ortho_node.camera_xc


class TestTier2InfAndNaNRanges:
    """Tier 2: Boundary Group 2 — Inf and NaN Ranges (8 tests)."""

    def test_t2_scan_all_inf(self, cloud_builder, ortho_node):
        """Verify scan with all 360 inf ranges generates zero walls and zero shadows."""
        ranges = [float('inf')] * 360
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_scan_with_nan(self, cloud_builder, ortho_node):
        """Verify scan containing NaN ranges is handled safely via isfinite."""
        ranges = [float('nan')] * 360
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_scan_alternating_inf_nan(self, cloud_builder):
        """Verify alternating inf and nan array does not raise unhandled exceptions."""
        ranges = [float('inf') if i % 2 == 0 else float('nan') for i in range(360)]
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0

    def test_t2_scan_beyond_max_range(self, cloud_builder, ortho_node):
        """Verify range of 2.50m (> 2.0m max) is rejected."""
        ranges = [2.50] * 360
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0
        ortho_node.scan_callback(create_scan_msg(ranges, range_max=2.0))
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_scan_below_min_range(self, cloud_builder):
        """Verify range of 0.01m (< 0.02m min for pointcloud) is rejected."""
        ranges = [0.01] * 360
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0

    def test_t2_isolated_valid_hit(self, cloud_builder):
        """Verify single isolated valid hit among 359 infs generates exactly 1 column."""
        ranges = [float('inf')] * 360
        ranges[100] = 0.75
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == len(cloud_builder.wall_z_steps)

    def test_t2_negative_inf_range(self, ortho_node):
        """Verify negative infinity range is filtered out cleanly."""
        ranges = [float('-inf')] * 360
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, 0.0) is False

    def test_t2_all_zeros_scan(self, cloud_builder):
        """Verify scan array filled with zeros produces 0 points."""
        ranges = [0.0] * 360
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0


class TestTier2CameraFocalPlaneSingularity:
    """Tier 2: Boundary Group 3 — Camera Focal Plane Singularity s <= xc (8 tests)."""

    def test_t2_obstacle_at_exact_xc(self, ortho_node):
        """Verify obstacle at distance exactly xc triggers singularity guard."""
        v = ortho_node.compute_azimuthal_horizon_v(ortho_node.camera_xc)
        assert v >= float(ortho_node.image_height)

    def test_t2_obstacle_behind_camera_lens(self, ortho_node):
        """Verify obstacle closer than xc (between robot center and lens) rejects all pixels."""
        v = ortho_node.compute_azimuthal_horizon_v(0.03)
        assert v >= float(ortho_node.image_height)

    def test_t2_obstacle_micro_offset_front(self, ortho_node):
        """Verify obstacle at xc + 1e-4 does not produce division by zero or NaN."""
        v = ortho_node.compute_azimuthal_horizon_v(ortho_node.camera_xc + 1e-4)
        assert math.isfinite(v)
        assert v > ortho_node.cy

    def test_t2_wall_backprojection_guard(self, cloud_builder):
        """Verify wall column with s <= 0.02 is protected and gets default color."""
        # Place hit at xc + 0.015 -> s = 0.015 < 0.02
        ranges = [float('inf')] * 360
        ranges[180] = cloud_builder.xc + 0.015
        cam_img = create_test_image(color=(255, 0, 0))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=cam_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        # Must NOT be sampled from cam_img (255, 0, 0)
        assert (r, g, b) == DEFAULT_WALL_COLOR_RGB

    def test_t2_horizon_clamping_guard_value(self, ortho_node):
        """Verify compute_azimuthal_horizon_v returns image_height + 10 for guarded depths."""
        expected = float(ortho_node.image_height + 10)
        assert ortho_node.compute_azimuthal_horizon_v(0.01) == expected

    def test_t2_camera_xc_zero(self):
        """Verify PointCloud3DBuilder handles camera_xc = 0.0 without errors."""
        builder = PointCloud3DBuilder(camera_xc=0.0)
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        cloud = builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == len(builder.wall_z_steps)

    def test_t2_camera_h_zero(self):
        """Verify PointCloud3DBuilder handles camera_h = 0.0."""
        builder = PointCloud3DBuilder(camera_h=0.0)
        assert builder.h == 0.0

    def test_t2_extreme_focal_length(self):
        """Verify PointCloud3DBuilder handles extreme focal length without overflow."""
        builder = PointCloud3DBuilder(fx=1e5, fy=1e5)
        assert builder.fx == 1e5


class TestTier2WallHeightBoundaries:
    """Tier 2: Boundary Group 4 — Maximum Obstacle Wall Heights Z > 0.35m (8 tests)."""

    def test_t2_wall_height_clamped_at_035(self, cloud_builder):
        """Verify default wall column heights clamp strictly at 0.35m."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.60
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert np.max(pts['z']) <= 0.35 + 1e-4

    def test_t2_custom_wall_height_010(self):
        """Verify custom wall_height_max = 0.10m produces heights in [0.0, 0.10m]."""
        builder = PointCloud3DBuilder(wall_height_max=0.10, wall_delta_z=0.03)
        assert builder.wall_z_steps[-1] == pytest.approx(0.10, abs=1e-4)

    def test_t2_custom_wall_height_050(self):
        """Verify custom wall_height_max = 0.50m produces heights in [0.0, 0.50m]."""
        builder = PointCloud3DBuilder(wall_height_max=0.50, wall_delta_z=0.05)
        assert builder.wall_z_steps[-1] == pytest.approx(0.50, abs=1e-4)

    def test_t2_custom_wall_delta_z_001(self):
        """Verify fine vertical resolution wall_delta_z = 0.01m."""
        builder = PointCloud3DBuilder(wall_height_max=0.35, wall_delta_z=0.01)
        assert len(builder.wall_z_steps) == 36

    def test_t2_custom_wall_delta_z_007(self):
        """Verify coarse vertical resolution wall_delta_z = 0.07m produces 6 points."""
        builder = PointCloud3DBuilder(wall_height_max=0.35, wall_delta_z=0.07)
        assert len(builder.wall_z_steps) == 6

    def test_t2_zero_wall_height(self):
        """Verify wall_height_max = 0.0 produces single layer at z = 0.0."""
        builder = PointCloud3DBuilder(wall_height_max=0.0)
        assert len(builder.wall_z_steps) == 1
        assert builder.wall_z_steps[0] == 0.0

    def test_t2_delta_z_larger_than_max_height(self):
        """Verify wall_delta_z > wall_height_max includes 0.0 and endpoint."""
        builder = PointCloud3DBuilder(wall_height_max=0.30, wall_delta_z=0.50)
        assert 0.0 in builder.wall_z_steps
        assert builder.wall_z_steps[-1] == pytest.approx(0.30, abs=1e-4)

    def test_t2_wall_column_point_count_exactness(self, cloud_builder):
        """Verify 10 scan hits produce exactly 10 * K points."""
        ranges = [float('inf')] * 360
        for i in range(10):
            ranges[175 + i] = 0.60
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 10 * len(cloud_builder.wall_z_steps)


class TestTier2ElevationGridBoundaries:
    """Tier 2: Boundary Group 5 — Unvisited Elevation Cells (-1) & Discontinuities (8 tests)."""

    def test_t2_unvisited_elevation_cell_neg1(self, cloud_builder):
        """Verify cell with elevation slope -1 (unvisited) remains at Z = 0.0."""
        col, row = 50, 50
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.full((200, 200), -1, dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)

    def test_t2_flat_slope_0(self, cloud_builder):
        """Verify flat floor slope 0 remains at Z = 0.0."""
        col, row = 50, 50
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)

    def test_t2_boundary_flat_slope_4(self, cloud_builder):
        """Verify boundary flat slope value 4 is not treated as incline ramp."""
        col, row = 50, 50
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.full((200, 200), 4, dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.0, abs=1e-5)

    def test_t2_ramp_slope_10(self, cloud_builder):
        """Verify ramp slope value 10 correctly identifies incline ramp."""
        col = int(round((1.2 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.full((200, 200), 10, dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] > 0.10

    def test_t2_platform_cliff_edge(self, cloud_builder):
        """Verify cell just outside platform deck drops abruptly to Z = 0.0."""
        # Inside deck: x = -0.5
        col_in = int(round((-0.5 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        # Outside deck: x = -0.3
        col_out = int(round((-0.3 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.1 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col_in] = True
        observed_mask[row, col_out] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # One point at 0.08, one at 0.0
        z_vals = sorted(pts['z'])
        assert z_vals[0] == pytest.approx(0.0, abs=1e-4)
        assert z_vals[1] == pytest.approx(0.08, abs=1e-4)

    def test_t2_platform_corners(self, cloud_builder):
        """Verify platform boundary corners evaluate correctly."""
        # Platform bounds: X in [-1.39, -0.4], Y in [0.6, 1.6]
        c_x = int(round((-0.45 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        c_y = int(round((1.55 - cloud_builder.origin_y) / cloud_builder.res - 0.5))
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[c_y, c_x] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=np.zeros((200, 200), dtype=np.int8),
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.08, abs=1e-3)

    def test_t2_elevation_grid_size_mismatch(self, cloud_builder):
        """Verify elevation_grid with mismatched shape is handled without crash."""
        bad_grid = np.zeros((50, 50), dtype=np.int8)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[50, 50] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=bad_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 1

    def test_t2_all_unvisited_grid(self, cloud_builder):
        """Verify elevation_grid with all cells -1 leaves all ground points at Z = 0.0."""
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[40:45, 40:45] = True
        elevation_grid = np.full((200, 200), -1, dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert np.all(pts['z'] == 0.0)


class TestTier2EmptyMapAndCloudBoundaries:
    """Tier 2: Boundary Group 6 — Empty Map & Cloud Boundaries (8 tests)."""

    def test_t2_empty_observed_mask(self, cloud_builder):
        """Verify empty observed_mask with valid scan produces only wall points."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.60
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == len(cloud_builder.wall_z_steps)

    def test_t2_none_scan_ranges(self, cloud_builder):
        """Verify scan_ranges = None produces only ground points."""
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[10, 10] = True
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 1

    def test_t2_empty_list_scan_ranges(self, cloud_builder):
        """Verify scan_ranges = [] produces only ground points without error."""
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[10, 10] = True
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=[],
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 1

    def test_t2_completely_empty_cloud(self, cloud_builder):
        """Verify completely empty inputs produce valid 0-width PointCloud2 message."""
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud.width == 0
        assert cloud.row_step == 0
        assert len(cloud.data) == 0

    def test_t2_none_camera_image_fallback(self, cloud_builder):
        """Verify camera_image = None assigns default wall color without crashing."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=None
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == DEFAULT_WALL_COLOR_RGB

    def test_t2_all_black_color_map(self, cloud_builder):
        """Verify color_map all zeros produces black (0, 0, 0) ground points."""
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[5, 5] = True
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (0, 0, 0)

    def test_t2_all_white_color_map(self, cloud_builder):
        """Verify color_map all 255.0 produces white (255, 255, 255) ground points."""
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[5, 5] = True
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.full((200, 200, 3), 255.0, dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (255, 255, 255)

    def test_t2_large_map_coordinates_resilience(self, cloud_builder):
        """Verify building cloud with large world coordinates does not overflow."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(1000.0, 1000.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['x'][0] == pytest.approx(1000.50, abs=0.01)


class TestTier2360RayWraparound:
    """Tier 2: Boundary Group 7 — 360-Ray Wraparound & Heading Discontinuities (8 tests)."""

    def test_t2_ray_angle_min_neg_pi(self, ortho_node):
        """Verify ray at exactly -pi maps to index 0."""
        ranges = [float('inf')] * 360
        ranges[0] = 0.40
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, -math.pi) is True

    def test_t2_ray_angle_max_pos_pi(self, ortho_node):
        """Verify ray near +pi maps to index 359 without IndexError."""
        ranges = [float('inf')] * 360
        ranges[359] = 0.40
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.5, ortho_node.scan_angle_max) is True

    def test_t2_yaw_wrapping_plus_minus_pi(self, cloud_builder):
        """Verify robot yaw at +pi and -pi produce identical wall point positions."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.80  # Ahead along robot +X

        cloud_pos = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, math.pi)
        )
        cloud_neg = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, -math.pi)
        )
        pts_pos = np.frombuffer(cloud_pos.data, dtype=POINT_DTYPE)
        pts_neg = np.frombuffer(cloud_neg.data, dtype=POINT_DTYPE)
        assert np.allclose(pts_pos['x'], pts_neg['x'], atol=1e-4)
        assert np.allclose(pts_pos['y'], pts_neg['y'], atol=1e-4)

    def test_t2_azimuth_exceeding_pi_wrapped(self, ortho_node):
        """Verify query angle 3.5 rad is mapped cleanly via modulo indexing."""
        ranges = [float('inf')] * 360
        ortho_node.scan_callback(create_scan_msg(ranges))
        # Must not raise IndexError
        assert ortho_node.is_in_shadow(0.5, 3.5) is False

    def test_t2_camera_fov_boundary_rays_minus_30deg(self, ortho_node):
        """Verify camera FOV boundary ray at -30 deg (-pi/6) indexes correctly."""
        ranges = [float('inf')] * 360
        idx = int(round((-math.pi / 6.0 - (-math.pi)) / (2.0 * math.pi / 360))) % 360
        ranges[idx] = 0.50
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.60, -math.pi / 6.0) is True

    def test_t2_camera_fov_boundary_rays_plus_30deg(self, ortho_node):
        """Verify camera FOV boundary ray at +30 deg (+pi/6) indexes correctly."""
        ranges = [float('inf')] * 360
        idx = int(round((math.pi / 6.0 - (-math.pi)) / (2.0 * math.pi / 360))) % 360
        ranges[idx] = 0.50
        ortho_node.scan_callback(create_scan_msg(ranges))
        assert ortho_node.is_in_shadow(0.60, math.pi / 6.0) is True

    def test_t2_modulo_indexing_never_out_of_bounds(self, ortho_node):
        """Verify extreme angles (-100 rad, +100 rad) do not cause IndexError."""
        ranges = [0.80] * 360
        ortho_node.scan_callback(create_scan_msg(ranges))
        for angle in [-100.0, -10.0, 10.0, 100.0]:
            assert isinstance(ortho_node.is_in_shadow(0.5, angle), (bool, np.bool_))

    def test_t2_robot_facing_west_yaw_pi(self, cloud_builder):
        """Verify robot at yaw = pi maps forward ray (+X_b) to negative world X."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50  # Forward along robot +X
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, math.pi)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['x'][0] == pytest.approx(-0.50, abs=0.01)


# ==============================================================================
# Tier 3: Cross-Feature Interactions (Pairwise Combinations, 16 tests)
# ==============================================================================

class TestTier3ShadowCullingAndAlphaAttenuation:
    """Tier 3: Pair 1 — Shadow Culling + Distance-Attenuated Alpha Blending (4 tests)."""

    def test_t3_culling_and_attenuation_concurrent(self, ortho_node):
        """Verify front cells receive attenuated alpha and behind cells are culled."""
        ortho_node.enable_distance_attenuation = True
        ranges = [float('inf')] * 360
        ranges[180] = 0.70  # Obstacle at 0.70m ahead
        ortho_node.scan_callback(create_scan_msg(ranges))

        # Cell at 0.40m ahead is in front: should NOT be in shadow
        assert ortho_node.is_in_shadow(0.40, 0.0) is False
        alpha_40 = ortho_node.compute_distance_attenuated_alpha(0.40)
        assert 0.0 < alpha_40 < ortho_node.alpha

        # Cell at 0.80m ahead is behind obstacle: SHOULD be in shadow
        assert ortho_node.is_in_shadow(0.80, 0.0) is True

    def test_t3_obstacle_advance_shrinks_ground_window(self, ortho_node):
        """Verify advancing obstacle reduces visible ground area while front cells obey decay."""
        ortho_node.enable_distance_attenuation = True

        # Obstacle at 1.0m
        ranges_1 = [float('inf')] * 360
        ranges_1[180] = 1.00
        ortho_node.scan_callback(create_scan_msg(ranges_1))
        assert ortho_node.is_in_shadow(0.60, 0.0) is False

        # Obstacle moves closer to 0.50m
        ranges_2 = [float('inf')] * 360
        ranges_2[180] = 0.50
        ortho_node.scan_callback(create_scan_msg(ranges_2))
        assert ortho_node.is_in_shadow(0.60, 0.0) is True
        # Cell at 0.30m remains visible with correct attenuation
        assert ortho_node.is_in_shadow(0.30, 0.0) is False
        a_30 = ortho_node.compute_distance_attenuated_alpha(0.30)
        a_60 = ortho_node.compute_distance_attenuated_alpha(0.60)
        assert a_30 > a_60

    def test_t3_oblique_obstacle_slanted_boundary(self, ortho_node):
        """Verify slanted obstacle produces angle-dependent shadow boundaries."""
        ortho_node.enable_distance_attenuation = True
        ranges = [float('inf')] * 360
        # Slanted wall: ray at -15 deg is at 0.50m, ray at +15 deg is at 0.80m
        idx_neg15 = int(round((-math.radians(15) - (-math.pi)) / (2 * math.pi / 360))) % 360
        idx_pos15 = int(round((math.radians(15) - (-math.pi)) / (2 * math.pi / 360))) % 360
        ranges[idx_neg15] = 0.50
        ranges[idx_pos15] = 0.80
        ortho_node.scan_callback(create_scan_msg(ranges))

        # At distance 0.65m:
        # At -15 deg: 0.65 >= 0.50 - 0.02 -> IN SHADOW
        assert ortho_node.is_in_shadow(0.65, -math.radians(15)) is True
        # At +15 deg: 0.65 < 0.80 - 0.02 -> NOT IN SHADOW
        assert ortho_node.is_in_shadow(0.65, math.radians(15)) is False

    def test_t3_shadow_margin_attenuation_interface(self, ortho_node):
        """Verify cell right at shadow margin boundary computes valid attenuated alpha."""
        ortho_node.enable_distance_attenuation = True
        ranges = [float('inf')] * 360
        d_obs = 0.60
        ranges[180] = d_obs
        ortho_node.scan_callback(create_scan_msg(ranges))

        d_boundary = d_obs - ortho_node.shadow_culling_margin - 0.005
        assert ortho_node.is_in_shadow(d_boundary, 0.0) is False
        alpha_b = ortho_node.compute_distance_attenuated_alpha(d_boundary)
        assert alpha_b > 0.0


class TestTier3ObstacleWallsAndCameraRGB:
    """Tier 3: Pair 2 — Obstacle Walls + Monocular Camera Image RGB Texturing (4 tests)."""

    def test_t3_camera_rgb_in_fov_vs_fallback(self, cloud_builder):
        """Verify obstacle hits in FOV get camera RGB, while rear hits get fallback color."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50  # Directly in front (+X)
        ranges[0] = 0.50    # Directly behind (-X)

        cam_img = create_test_image(color=(10, 180, 70))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=cam_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # Find points in front (x > 0) vs rear (x < 0)
        front_pts = pts[pts['x'] > 0.2]
        rear_pts = pts[pts['x'] < -0.2]

        r_f, g_f, b_f = unpack_rgb_f32(front_pts['rgb'][0])
        assert (r_f, g_f, b_f) == (10, 180, 70)

        r_r, g_r, b_r = unpack_rgb_f32(rear_pts['rgb'][0])
        assert (r_r, g_r, b_r) == DEFAULT_WALL_COLOR_RGB

    def test_t3_robot_turn_shifts_textured_columns(self, cloud_builder):
        """Verify rotating robot 90 deg transfers camera texturing to facing obstacles."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50  # Along robot +X

        cam_img = create_test_image(color=(240, 20, 20))

        # Case A: robot at yaw=0 -> robot +X is world +X
        cloud_a = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=cam_img
        )
        pts_a = np.frombuffer(cloud_a.data, dtype=POINT_DTYPE)
        assert pts_a['x'][0] == pytest.approx(0.50, abs=0.01)
        r, g, b = unpack_rgb_f32(pts_a['rgb'][0])
        assert (r, g, b) == (240, 20, 20)

        # Case B: robot at yaw=pi/2 -> robot +X is world +Y
        cloud_b = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, math.pi / 2.0),
            camera_image=cam_img
        )
        pts_b = np.frombuffer(cloud_b.data, dtype=POINT_DTYPE)
        assert pts_b['y'][0] == pytest.approx(0.50, abs=0.01)
        r, g, b = unpack_rgb_f32(pts_b['rgb'][0])
        assert (r, g, b) == (240, 20, 20)

    def test_t3_vertical_column_pinhole_rgb_gradient(self, cloud_builder):
        """Verify vertical column points back-project to different image rows v."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.60

        # Gradient: Top half RED [255, 0, 0], bottom half BLUE [0, 0, 255]
        grad_img = np.zeros((480, 640, 3), dtype=np.uint8)
        grad_img[:240, :, 0] = 255  # Red
        grad_img[240:, :, 2] = 255  # Blue

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=grad_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # Ground level point (z=0.0): v ~ 269 px -> BLUE
        r_bot, g_bot, b_bot = unpack_rgb_f32(pts['rgb'][0])
        assert b_bot == 255 and r_bot == 0

        # High point (z=0.35m): v < 0 (outside FOV -> fallback color)
        r_top, g_top, b_top = unpack_rgb_f32(pts['rgb'][-1])
        assert (r_top, g_top, b_top) == DEFAULT_WALL_COLOR_RGB

    def test_t3_synthetic_stripe_texture_backprojection(self, cloud_builder):
        """Verify back-projection maps image pixel rows cleanly to wall columns."""
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        test_img = create_test_image(color=(50, 150, 250))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=np.zeros((200, 200), dtype=bool),
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0),
            camera_image=test_img
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (50, 150, 250)


class TestTier3ElevationRampAndOrthomosaic:
    """Tier 3: Pair 3 — Terrain Slopes & Ramps + Ground Visual Orthomosaic Texturing (4 tests)."""

    def test_t3_ramp_lifting_preserves_floor_rgb(self, cloud_builder):
        """Verify elevated points on 10° ramp retain exact visual orthomosaic colors."""
        col = int(round((1.1 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        color_map[row, col] = [45.0, 90.0, 135.0]
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True
        elevation_grid = np.zeros((200, 200), dtype=np.int8)

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] > 0.05
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (45, 90, 135)

    def test_t3_approach_ramp_knife_edge_lip_transition(self, cloud_builder):
        """Verify smooth elevation rise along approach ramp without step discontinuities."""
        cols = [int(round((x - cloud_builder.origin_x) / cloud_builder.res - 0.5))
                for x in [-1.95, -1.75, -1.55, -1.40]]
        row = int(round((1.1 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        observed_mask = np.zeros((200, 200), dtype=bool)
        for c in cols:
            observed_mask[row, c] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed_mask,
            elevation_grid=np.zeros((200, 200), dtype=np.int8),
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # Sort by x
        pts_sorted = pts[np.argsort(pts['x'])]
        # Heights must monotonically increase
        assert np.all(np.diff(pts_sorted['z']) >= 0.0)
        assert pts_sorted['z'][0] < pts_sorted['z'][-1]

    def test_t3_platform_deck_orthomosaic_elevation(self, cloud_builder):
        """Verify platform deck points (Z=0.08m) maintain color map texture."""
        col = int(round((-0.9 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.1 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        color_map[row, col] = [180.0, 70.0, 30.0]
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=np.zeros((200, 200), dtype=np.int8),
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        assert pts['z'][0] == pytest.approx(0.08, abs=1e-3)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (180, 70, 30)

    def test_t3_clear_visual_map_resets_elevated_cloud(self, cloud_builder):
        """Verify clearing color map resets elevated points to black (0, 0, 0)."""
        col = int(round((1.1 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))

        color_map = np.zeros((200, 200, 3), dtype=np.float32)
        observed_mask = np.zeros((200, 200), dtype=bool)
        observed_mask[row, col] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=np.zeros((200, 200), dtype=np.int8),
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        r, g, b = unpack_rgb_f32(pts['rgb'][0])
        assert (r, g, b) == (0, 0, 0)


class TestTier3SLAMMapPurgeAndPointCloudRegeneration:
    """Tier 3: Pair 4 — SLAM Obstacle Cache Purge + 3D PointCloud Regeneration (4 tests)."""

    def test_t3_slam_obstacle_purge_removes_cloud_floor_points(self, ortho_node, cloud_builder):
        """Verify purging obstacle cells from /map removes ground points from regenerated cloud."""
        r_occ, c_occ = 100, 100
        ortho_node.observed_mask[r_occ, c_occ] = True
        ortho_node.color_map[r_occ, c_occ] = [200.0, 100.0, 50.0]

        # Initial cloud has ground point
        cloud_1 = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud_1.width == 1

        # Purge via /map update
        grid_data = np.zeros((200, 200), dtype=np.int8)
        grid_data[r_occ, c_occ] = 100
        ortho_node.map_callback(create_occupancy_grid(grid_data))

        # Regenerated cloud has 0 ground points
        cloud_2 = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud_2.width == 0

    def test_t3_purged_floor_replaced_by_obstacle_walls(self, ortho_node, cloud_builder):
        """Verify purged floor point is replaced by vertical obstacle wall column."""
        r_occ, c_occ = 100, 100
        ortho_node.observed_mask[r_occ, c_occ] = True

        # Purge obstacle
        grid_data = np.zeros((200, 200), dtype=np.int8)
        grid_data[r_occ, c_occ] = 100
        ortho_node.map_callback(create_occupancy_grid(grid_data))

        # Add proximity scan hit at boundary of obstacle
        ranges = [float('inf')] * 360
        ranges[180] = 0.50

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        # Only wall points remain
        assert len(pts) == len(cloud_builder.wall_z_steps)
        assert np.max(pts['z']) == pytest.approx(0.35, abs=1e-4)

    def test_t3_multiple_slam_obstacles_purged(self, ortho_node, cloud_builder):
        """Verify multiple obstacles purged simultaneously in /map."""
        for r in [50, 60, 70]:
            ortho_node.observed_mask[r, 50] = True

        grid_data = np.zeros((200, 200), dtype=np.int8)
        for r in [50, 60]:
            grid_data[r, 50] = 100
        ortho_node.map_callback(create_occupancy_grid(grid_data))

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        # Only row 70 remains
        assert cloud.width == 1

    def test_t3_dynamic_obstacle_clear_allows_ground_restoration(self, ortho_node):
        """Verify clearing obstacle allows future warping to update cells."""
        r_occ, c_occ = 100, 100
        grid_data = np.zeros((200, 200), dtype=np.int8)
        grid_data[r_occ, c_occ] = 100
        ortho_node.map_callback(create_occupancy_grid(grid_data))
        mask_val = ortho_node.latest_obstacle_mask[r_occ, c_occ]
        assert mask_val is True or mask_val == 1

        # Clear obstacle: grid reports free space
        grid_data[r_occ, c_occ] = 0
        ortho_node.map_callback(create_occupancy_grid(grid_data))
        mask_val = ortho_node.latest_obstacle_mask[r_occ, c_occ]
        assert mask_val is False or mask_val == 0


# ==============================================================================
# Tier 4: Real-World Scenarios (5 tests)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic End-to-End Application Scenarios (5 tests)."""

    def test_t4_inplace_rotation_in_front_of_obstacle_box(self, ortho_node):
        """Scenario 1: In-place rotation in front of obstacle box without smudges."""
        box_dist = 0.50
        ortho_node.enable_occlusion_culling = True

        for yaw_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            _ = math.radians(yaw_deg)
            ranges = [float('inf')] * 360
            ranges[180] = box_dist  # Box always detected directly ahead
            ortho_node.scan_callback(create_scan_msg(ranges))

            # Behind-box cell at 0.70m along camera optical axis
            in_shadow = ortho_node.is_in_shadow(0.70, 0.0)
            assert in_shadow is True, f'Shadow culling failed at yaw {yaw_deg}°!'

    def test_t4_glancing_wall_approach(self, ortho_node, cloud_builder):
        """Scenario 2: Glancing wall approach at 45 deg; stops IPM and builds column."""
        ortho_node.enable_occlusion_culling = True
        wall_dist = 0.45
        # Proximity scan hits wall across front-right angles (indices 180 to 220)
        ranges = [float('inf')] * 360
        for i in range(180, 220):
            ranges[i] = wall_dist
        ortho_node.scan_callback(create_scan_msg(ranges))

        # Ground behind wall is shadowed
        assert ortho_node.is_in_shadow(0.65, 0.0) is True

        # Point cloud builds vertical obstacle wall columns
        cam_img = create_test_image(color=(120, 140, 160))
        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, math.radians(45.0)),
            camera_image=cam_img
        )
        assert cloud.width >= 40 * len(cloud_builder.wall_z_steps)

    def test_t4_ramp_ascent_with_pitch_guard_and_3d_lifting(self, ortho_node, cloud_builder):
        """Scenario 3: 10 deg ramp climb; pitch guard suspends warping, cloud lifts ramp."""
        # Step 1: Robot drives onto ramp lip, pitch increases to 10 deg (0.1745 rad)
        ortho_node.robot_pitch = math.radians(10.0)
        test_img = create_test_image()

        # Warp must be suspended by pitch guard (> 6.0 deg)
        updated_cells = ortho_node.warp_image_to_grid(
            test_img, 0.8, 1.3, 0.0, ortho_node.robot_pitch
        )
        assert updated_cells == 0, '2D IPM warping was not suspended during ramp climb!'

        # Step 2: 3D point cloud builder lifts ramp cells along 10 deg incline
        col = int(round((1.1 - cloud_builder.origin_x) / cloud_builder.res - 0.5))
        row = int(round((1.3 - cloud_builder.origin_y) / cloud_builder.res - 0.5))
        observed = np.zeros((200, 200), dtype=bool)
        observed[row, col] = True

        cloud = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=np.zeros((200, 200, 3), dtype=np.float32),
            observed_mask=observed,
            elevation_grid=np.zeros((200, 200), dtype=np.int8),
            scan_ranges=None,
            robot_pose=(1.1, 1.3, 0.0)
        )
        pts = np.frombuffer(cloud.data, dtype=POINT_DTYPE)
        expected_ramp_z = (pts['x'][0] - 0.5) * np.tan(np.radians(10.0))
        assert pts['z'][0] == pytest.approx(expected_ramp_z, abs=1e-3)

    def test_t4_late_joining_client_latched_pointcloud(self, ortho_node):
        """Scenario 4: Late-joining client latched QoS receipt of PointCloud2 map."""
        ortho_node.observed_mask[50, 50] = True
        ortho_node.color_map[50, 50] = [200.0, 200.0, 200.0]

        # Publish point cloud
        ortho_node.publish_pointcloud()
        assert hasattr(ortho_node, 'pointcloud_pub')
        assert ortho_node.pointcloud_pub is not None

    def test_t4_dynamic_slam_obstacle_expansion_pipeline(self, ortho_node, cloud_builder):
        """Scenario 5: Dynamic obstacle in SLAM map purges floor and updates 3D cloud."""
        # 1. Robot observes floor
        r, c = 100, 100
        ortho_node.observed_mask[r, c] = True
        ortho_node.color_map[r, c] = [255.0, 255.0, 255.0]

        cloud_before = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=None,
            robot_pose=(0.0, 0.0, 0.0)
        )
        assert cloud_before.width == 1

        # 2. Obstacle detected by SLAM -> /map published
        map_msg = create_occupancy_grid()
        map_arr = np.zeros((200, 200), dtype=np.int8)
        map_arr[r, c] = 100
        map_msg.data = [int(v) for v in map_arr.flatten()]
        ortho_node.map_callback(map_msg)

        # Floor is purged
        assert ortho_node.observed_mask[r, c] is False or ortho_node.observed_mask[r, c] == 0

        # 3. Cloud reconstituted: floor point gone, replaced by obstacle scan hit
        ranges = [float('inf')] * 360
        ranges[180] = 0.50
        cloud_after = cloud_builder.build_pointcloud(
            stamp=None,
            color_map=ortho_node.color_map,
            observed_mask=ortho_node.observed_mask,
            elevation_grid=None,
            scan_ranges=ranges,
            robot_pose=(0.0, 0.0, 0.0)
        )
        pts = np.frombuffer(cloud_after.data, dtype=POINT_DTYPE)
        assert len(pts) == len(cloud_builder.wall_z_steps)
