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
Unit and integration tests for Anti-Smudge Occlusion Culling.

Tests vectorized shadow culling, azimuthal horizon clamping, distance-attenuated
alpha blending, SLAM obstacle purging, and 3D point cloud publisher integration
in VisualOrthomapper.
"""

import math
from unittest.mock import MagicMock

from khepera4_control.pointcloud_builder import PointCloud3DBuilder
from khepera4_control.visual_orthomapper import VisualOrthomapper
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2


@pytest.fixture
def rclpy_context():
    """Manage rclpy init and shutdown lifecycle for tests."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def orthomapper_node(rclpy_context):
    """Instantiate and clean up a VisualOrthomapper node."""
    node = VisualOrthomapper()
    yield node
    node.destroy_node()


class TestOcclusionCullingParameters:
    """Test parameter declarations and default configurations."""

    def test_default_parameters(self, orthomapper_node):
        """Verify default occlusion and pointcloud parameter values."""
        node = orthomapper_node
        assert node.get_parameter('scan_topic').value == '/khepera4/scan'
        assert node.get_parameter('map_topic').value == '/map'
        assert node.get_parameter('elevation_map_topic').value == '/elevation_map'
        assert node.get_parameter('enable_occlusion_culling').value is True
        assert node.get_parameter('shadow_culling_margin').value == pytest.approx(0.02)
        assert node.get_parameter('enable_distance_attenuation').value is False
        assert node.get_parameter('occ_threshold').value == 50
        assert node.get_parameter('publish_pointcloud_3d').value is True
        assert node.get_parameter('pointcloud_topic').value == '/khepera4/pointcloud3d'
        assert node.get_parameter('pointcloud_publish_rate').value == pytest.approx(2.0)

    def test_qos_and_subscriptions_exist(self, orthomapper_node):
        """Verify subscribers, publishers and latched QoS profile."""
        node = orthomapper_node
        assert node.scan_sub is not None
        assert node.map_sub is not None
        assert node.elevation_map_sub is not None
        assert node.pointcloud_pub is not None

        qos = node.pointcloud_pub.qos_profile
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


class TestLaserScanProcessing:
    """Test scan subscription and depth horizon caching."""

    def test_scan_callback_updates_horizon(self, orthomapper_node):
        """Verify /khepera4/scan updates latest_scan_ranges and scan attributes."""
        node = orthomapper_node
        scan_msg = LaserScan()
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi - (2.0 * math.pi / 360.0)
        scan_msg.angle_increment = 2.0 * math.pi / 360.0
        scan_msg.range_min = 0.002
        scan_msg.range_max = 2.0
        ranges = [1.5] * 360
        ranges[180] = 0.65  # Obstacle directly ahead (ray index 180 = 0 rad)
        scan_msg.ranges = ranges

        node.scan_callback(scan_msg)

        assert node.latest_scan_ranges is not None
        assert len(node.latest_scan_ranges) == 360
        assert node.latest_scan_ranges[180] == pytest.approx(0.65, abs=1e-4)
        assert node.scan_angle_min == pytest.approx(-math.pi)
        assert node.scan_angle_inc == pytest.approx(2.0 * math.pi / 360.0)


class TestVectorizedShadowCulling:
    """Test raycast shadow culling math and grid masking behavior."""

    def test_is_in_shadow_helper(self, orthomapper_node):
        """Verify is_in_shadow evaluates obstacle boundaries with margin."""
        node = orthomapper_node
        # 360 rays, obstacle at 0.80m straight ahead (index 180)
        node.latest_scan_ranges = np.full(360, 2.0, dtype=np.float32)
        node.latest_scan_ranges[180] = 0.80
        node.scan_angle_min = -math.pi
        node.scan_angle_inc = 2.0 * math.pi / 360.0
        node.scan_range_min = 0.002
        node.scan_range_max = 2.0
        node.shadow_culling_margin = 0.02

        # In front of obstacle horizon (d_cell < 0.78m)
        assert node.is_in_shadow(0.70, 0.0) is False
        assert node.is_in_shadow(0.77, 0.0) is False

        # At or behind obstacle horizon (d_cell >= 0.78m)
        assert node.is_in_shadow(0.78, 0.0) is True
        assert node.is_in_shadow(0.80, 0.0) is True
        assert node.is_in_shadow(1.10, 0.0) is True

        # Ray to the side (theta = 0.5 rad) has no close obstacle (d_obs = 2.0m)
        assert node.is_in_shadow(0.85, 0.5) is False

    def test_warp_image_culls_shadowed_cells(self, orthomapper_node):
        """Verify ground cells behind detected obstacle are excluded from mapping."""
        node = orthomapper_node
        node.enable_occlusion_culling = True
        node.shadow_culling_margin = 0.02

        # Synthesize white camera frame
        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)

        # Place an obstacle at 0.50m forward
        scan = np.full(360, float('inf'), dtype=np.float32)
        # Forward ray (index 180) and flanking forward rays in 60 deg cone (150 to 210)
        scan[150:211] = 0.50
        node.latest_scan_ranges = scan

        # Warp image from origin (0, 0, yaw=0, pitch=0)
        updated = node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)
        assert updated > 0

        # Check observed cells: candidate cells with X_b >= 0.48m should NOT be observed
        obs_rows, obs_cols = np.where(node.observed_mask)
        for r, c in zip(obs_rows, obs_cols):
            xw, yw = node.grid_to_world(c, r)
            d_cell = math.hypot(xw, yw)
            # All mapped cells must be in front of obstacle boundary
            assert d_cell < 0.48 + 0.03  # with cell resolution tolerance

    def test_warp_image_without_occlusion_culling(self, orthomapper_node):
        """Verify disabling occlusion culling allows mapping past obstacle depth."""
        node = orthomapper_node
        node.enable_occlusion_culling = False

        img = np.full((node.image_height, node.image_width, 3), 200, dtype=np.uint8)
        scan = np.full(360, float('inf'), dtype=np.float32)
        scan[150:211] = 0.50
        node.latest_scan_ranges = scan

        node.warp_image_to_grid(img, 0.0, 0.0, 0.0, 0.0)

        obs_rows, obs_cols = np.where(node.observed_mask)
        max_dist = max(math.hypot(*node.grid_to_world(c, r)) for r, c in zip(obs_rows, obs_cols))
        # With culling disabled, mapping extends up to max_ground_dist (1.50m)
        assert max_dist > 1.0


class TestAzimuthalHorizonClamping:
    """Test azimuthal horizon row cutoff and singularity guard."""

    def test_compute_azimuthal_horizon_v_nominal(self, orthomapper_node):
        """Verify nominal obstacle depth maps to expected image row."""
        node = orthomapper_node
        d_obs = 0.50
        xc = node.camera_xc  # 0.0682
        h = node.camera_h    # 0.0283
        fy = node.fy         # 554.256258
        cy = node.cy         # 240.0
        expected_v = cy + (h * fy) / (d_obs - xc)

        computed_v = node.compute_azimuthal_horizon_v(d_obs)
        assert computed_v == pytest.approx(expected_v, abs=1e-3)

    def test_compute_azimuthal_horizon_singularity_guard(self, orthomapper_node):
        """Verify singularity guard clamps v_base = image_height + 10 when d_obs <= xc."""
        node = orthomapper_node
        # Obstacle at or closer than camera lens offset (xc = 0.0682m)
        computed_v = node.compute_azimuthal_horizon_v(0.05)
        assert computed_v == pytest.approx(float(node.image_height + 10))

        # Exactly at xc
        computed_v_exact = node.compute_azimuthal_horizon_v(node.camera_xc)
        assert computed_v_exact == pytest.approx(float(node.image_height + 10))

        # Infinite / NaN distance
        computed_v_inf = node.compute_azimuthal_horizon_v(float('inf'))
        assert computed_v_inf == pytest.approx(float(node.image_height + 10))


class TestDistanceAttenuatedAlphaBlending:
    """Test distance-attenuated alpha blending calculations."""

    def test_compute_distance_attenuated_alpha_profile(self, orthomapper_node):
        """Verify alpha decreases monotonically with distance from robot."""
        node = orthomapper_node
        node.alpha = 0.25
        xmin = node.min_ground_dist  # 0.1338
        xmax = node.max_ground_dist  # 1.50

        # At closest ground distance: alpha = alpha_max = 0.25
        alpha_near = node.compute_distance_attenuated_alpha(xmin)
        assert alpha_near == pytest.approx(0.25, abs=1e-4)

        # At furthest ground distance: alpha = 0.0
        alpha_far = node.compute_distance_attenuated_alpha(xmax)
        assert alpha_far == pytest.approx(0.0, abs=1e-4)

        # Intermediate distance (0.5m)
        alpha_mid = node.compute_distance_attenuated_alpha(0.5)
        expected_mid = 0.25 * (((xmax - 0.5) / (xmax - xmin)) ** 1.5)
        assert alpha_mid == pytest.approx(expected_mid, abs=1e-4)
        assert alpha_near > alpha_mid > alpha_far

    def test_warp_with_distance_attenuation_enabled(self, orthomapper_node):
        """Verify distance attenuation updates distant cells with smaller weight."""
        node = orthomapper_node
        node.enable_distance_attenuation = True

        # First observation: pure white (240, 240, 240)
        img1 = np.full((node.image_height, node.image_width, 3), 240, dtype=np.uint8)
        node.warp_image_to_grid(img1, 0.0, 0.0, 0.0, 0.0)

        # Pick distant cell (large X_b, e.g. ~1.40m)
        obs_rows, obs_cols = np.where(node.observed_mask)
        dists = [math.hypot(*node.grid_to_world(c, r)) for r, c in zip(obs_rows, obs_cols)]
        far_idx = np.argmax(dists)
        rf, cf = obs_rows[far_idx], obs_cols[far_idx]
        xw_far, yw_far = node.grid_to_world(cf, rf)
        d_far = math.hypot(xw_far, yw_far)
        assert d_far > 1.20

        # Second observation: dark (40, 40, 40)
        img2 = np.full((node.image_height, node.image_width, 3), 40, dtype=np.uint8)
        node.warp_image_to_grid(img2, 0.0, 0.0, 0.0, 0.0)

        # Since d_far is close to 1.50m, alpha(d_far) is very small (< 0.02)
        # So cell color should remain close to 240, rather than the unattenuated 190
        val = node.color_map[rf, cf, 0]
        assert val > 220.0  # Much higher than unattenuated 190.0


class TestSLAMObstaclePurge:
    """Test immediate visual map memory purge on confirmed SLAM obstacles."""

    def test_map_callback_purges_occupied_footprint(self, orthomapper_node):
        """Verify /map callback immediately clears observed_mask, color_map, and weight_map."""
        node = orthomapper_node
        # Populate visual map with test data
        node.color_map[:, :] = 180.0
        node.observed_mask[:, :] = True
        node.weight_map[:, :] = 5.0

        # Create SLAM OccupancyGrid message with obstacle box in center
        map_msg = OccupancyGrid()
        map_msg.info.resolution = node.res
        map_msg.info.width = node.width
        map_msg.info.height = node.height
        map_msg.info.origin.position.x = node.origin_x
        map_msg.info.origin.position.y = node.origin_y

        data = np.zeros((node.height, node.width), dtype=np.int8)
        # Occupied region (>= 50) at rows 90..110, cols 90..110
        data[90:111, 90:111] = 100
        map_msg.data = data.flatten().tolist()

        node.map_callback(map_msg)

        # Verify occupied region is completely purged
        assert not np.any(node.observed_mask[90:111, 90:111])
        assert np.all(node.color_map[90:111, 90:111] == 0.0)
        assert np.all(node.weight_map[90:111, 90:111] == 0.0)

        # Verify free cells remain intact
        assert bool(node.observed_mask[0, 0]) is True
        assert node.color_map[0, 0, 0] == pytest.approx(180.0)
        assert node.weight_map[0, 0] == pytest.approx(5.0)

        # Verify latest_obstacle_mask recorded the obstacle
        assert np.all(node.latest_obstacle_mask[90:111, 90:111])
        assert not node.latest_obstacle_mask[0, 0]

    def test_warp_image_excludes_slam_obstacles(self, orthomapper_node):
        """Verify candidate cells on known SLAM obstacles are rejected during IPM warping."""
        node = orthomapper_node
        # Mark center area as known obstacle
        node.latest_obstacle_mask[95:105, 95:105] = True

        img = np.full((node.image_height, node.image_width, 3), 150, dtype=np.uint8)
        # Robot near obstacle facing towards it
        xw_obs, yw_obs = node.grid_to_world(100, 100)
        rx = xw_obs - 0.4
        ry = yw_obs
        node.warp_image_to_grid(img, rx, ry, 0.0, 0.0)

        # Mapped cells must NOT overlap the obstacle mask
        overlap = node.observed_mask & node.latest_obstacle_mask
        assert not np.any(overlap)


class TestElevationMapAndPointCloudIntegration:
    """Test /elevation_map storage and PointCloud3DBuilder publisher execution."""

    def test_elevation_map_callback(self, orthomapper_node):
        """Verify elevation map occupancy grid is correctly cached."""
        node = orthomapper_node
        msg = OccupancyGrid()
        msg.info.width = node.width
        msg.info.height = node.height
        elev_data = np.full((node.height, node.width), 8, dtype=np.int8)
        msg.data = elev_data.flatten().tolist()

        node.elevation_map_callback(msg)
        assert node.latest_elevation_grid is not None
        assert node.latest_elevation_grid.shape == (node.height, node.width)
        assert node.latest_elevation_grid[50, 50] == 8

    def test_pointcloud_builder_instantiation_and_publish(self, orthomapper_node):
        """Verify PointCloud3DBuilder creates and publishes PointCloud2 message."""
        node = orthomapper_node
        assert node.pointcloud_builder is not None
        assert isinstance(node.pointcloud_builder, PointCloud3DBuilder)

        # Mark some cells observed
        node.color_map[100, 100] = [120.0, 150.0, 200.0]
        node.observed_mask[100, 100] = True
        node.has_odom = True
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0

        # Spy on pointcloud_pub.publish
        node.pointcloud_pub.publish = MagicMock()
        node.publish_pointcloud()

        assert node.pointcloud_pub.publish.called
        published_msg = node.pointcloud_pub.publish.call_args[0][0]
        assert isinstance(published_msg, PointCloud2)
        assert published_msg.header.frame_id == node.map_frame
        assert published_msg.width > 0

    def test_publish_pointcloud_disabled_flag(self, orthomapper_node):
        """Verify publishing is suppressed when publish_pointcloud_3d is False."""
        node = orthomapper_node
        node.publish_pointcloud_3d = False
        node.pointcloud_pub.publish = MagicMock()

        node.publish_pointcloud()
        assert not node.pointcloud_pub.publish.called

    def test_clear_resets_obstacle_mask(self, orthomapper_node):
        """Verify clear() resets latest_obstacle_mask and visual memory."""
        node = orthomapper_node
        node.latest_obstacle_mask[50, 50] = True
        node.color_map[50, 50] = 200.0
        node.observed_mask[50, 50] = True

        node.clear(purge_disk=False)
        assert not np.any(node.latest_obstacle_mask)
        assert not np.any(node.observed_mask)
        assert np.all(node.color_map == 0.0)
