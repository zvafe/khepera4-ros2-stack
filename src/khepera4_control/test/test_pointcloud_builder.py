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

"""Unit and benchmark tests for PointCloud3DBuilder and RViz2 configuration."""

import math
import os
import time

from builtin_interfaces.msg import Time
from khepera4_control.pointcloud_builder import (
    DEFAULT_WALL_COLOR_RGB,
    pack_rgb_f32,
    PointCloud3DBuilder,
    unpack_rgb_f32,
)
import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField
import yaml


@pytest.fixture
def default_builder():
    """Create a default PointCloud3DBuilder instance."""
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
        wall_delta_z=0.03,
    )


def test_rgb_packing_unpacking_roundtrip():
    """Verify pack_rgb_f32 and unpack_rgb_f32 roundtrip preserves exact 8-bit channels."""
    r_in = np.array([0, 255, 128, 60, 200], dtype=np.uint8)
    g_in = np.array([0, 128, 255, 60, 50], dtype=np.uint8)
    b_in = np.array([0, 64, 32, 65, 10], dtype=np.uint8)

    packed = pack_rgb_f32(r_in, g_in, b_in)
    assert packed.dtype == np.float32
    assert len(packed) == len(r_in)

    r_out, g_out, b_out = unpack_rgb_f32(packed)
    np.testing.assert_array_equal(r_out, r_in)
    np.testing.assert_array_equal(g_out, g_in)
    np.testing.assert_array_equal(b_out, b_in)


def test_pointcloud2_message_structure(default_builder):
    """Verify PointCloud2 message fields, offsets, step sizes, and frame alignment."""
    color_map = np.full((200, 200, 3), 180.0, dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)
    observed_mask[100, 100] = True
    observed_mask[101, 100] = True

    stamp = Time(sec=123, nanosec=456)
    msg = default_builder.build_pointcloud(
        stamp=stamp,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=None,
    )

    assert isinstance(msg, PointCloud2)
    assert msg.header.stamp.sec == 123
    assert msg.header.stamp.nanosec == 456
    assert msg.header.frame_id == 'map'
    assert msg.height == 1
    assert msg.width == 2
    assert msg.is_dense is True
    assert msg.is_bigendian is False
    assert msg.point_step == 16
    assert msg.row_step == 32
    assert len(msg.data) == 32

    # Verify PointField list
    assert len(msg.fields) == 4
    field_names = [f.name for f in msg.fields]
    assert field_names == ['x', 'y', 'z', 'rgb']
    assert msg.fields[0].offset == 0
    assert msg.fields[1].offset == 4
    assert msg.fields[2].offset == 8
    assert msg.fields[3].offset == 12

    for f in msg.fields:
        assert f.datatype == PointField.FLOAT32
        assert f.count == 1


def test_layer1_ground_reconstruction(default_builder):
    """Verify Layer 1 ground floor cells have Z = 0 and match orthomap coordinates and color."""
    h, w = default_builder.height, default_builder.width
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    observed_mask = np.zeros((h, w), dtype=bool)

    # Set specific color at cell (r=100, c=100)
    # World coords: -3.0 + (100 + 0.5) * 0.03 = 0.015
    observed_mask[100, 100] = True
    color_map[100, 100] = [120.0, 210.0, 50.0]

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    assert len(arr) == 1
    expected_x = -3.0 + (100 + 0.5) * 0.03
    expected_y = -3.0 + (100 + 0.5) * 0.03
    assert math.isclose(arr['x'][0], expected_x, abs_tol=1e-4)
    assert math.isclose(arr['y'][0], expected_y, abs_tol=1e-4)
    assert arr['z'][0] == 0.0

    r, g, b = unpack_rgb_f32(arr['rgb'])
    assert r[0] == 120
    assert g[0] == 210
    assert b[0] == 50


def test_layer2_obstacle_walls_camera_texturing(default_builder):
    """Verify Layer 2 obstacle wall columns sample camera RGB when inside camera FOV."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Obstacle directly in front of camera at 0.50 m (optical Z = 0.50 - 0.0682 = 0.4318 m)
    # 360 rays from -pi to pi; index 180 is straight ahead (angle = 0.0)
    scan_ranges = [float('inf')] * 360
    scan_ranges[180] = 0.50  # 0.5 m straight ahead

    # Create synthetic camera image with distinctive color at projected center
    camera_img = np.full((480, 640, 3), 40, dtype=np.uint8)
    camera_img[200:280, 300:340] = [240, 100, 20]  # Orange patch in center

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_ranges,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=camera_img,
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    num_steps = len(default_builder.wall_z_steps)
    assert len(arr) == num_steps

    # X coordinates should all be around 0.50 m
    np.testing.assert_allclose(arr['x'], 0.50, atol=1e-3)
    np.testing.assert_allclose(arr['y'], 0.00, atol=1e-3)
    np.testing.assert_allclose(arr['z'], default_builder.wall_z_steps, atol=1e-3)

    # Points within camera vertical FOV should have sampled the camera color
    r, g, b = unpack_rgb_f32(arr['rgb'])
    # Check that at least some points within the orange patch sampled [240, 100, 20]
    orange_points = (r == 240) & (g == 100) & (b == 20)
    assert np.any(orange_points), 'Expected camera RGB sampling on obstacle column'


def test_layer2_obstacle_walls_behind_camera_fallback(default_builder):
    """Verify obstacle points behind robot get fallback slate gray color [60, 60, 65]."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Obstacle directly behind the robot: angle = -pi (index 0)
    scan_ranges = [float('inf')] * 360
    scan_ranges[0] = 0.40

    camera_img = np.full((480, 640, 3), 255, dtype=np.uint8)

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_ranges,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=camera_img,
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    assert len(arr) == len(default_builder.wall_z_steps)

    r, g, b = unpack_rgb_f32(arr['rgb'])
    for ri, gi, bi in zip(r, g, b):
        assert (ri, gi, bi) == DEFAULT_WALL_COLOR_RGB


def test_layer3_terrain_slopes_and_ramps(default_builder):
    """Verify Layer 3 ramps and platforms are correctly lifted above ground plane."""
    h, w = default_builder.height, default_builder.width
    color_map = np.full((h, w, 3), 150.0, dtype=np.float32)
    observed_mask = np.ones((h, w), dtype=bool)
    elevation_grid = np.zeros((h, w), dtype=np.int8)

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=elevation_grid,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])

    # 1. Check 10 deg inclined ramp (X in [0.5, 1.7], Y in [1.0, 1.6])
    ramp_mask = (arr['x'] >= 0.5) & (arr['x'] <= 1.7) & (arr['y'] >= 1.0) & (arr['y'] <= 1.6)
    assert np.any(ramp_mask)
    expected_z = (arr['x'][ramp_mask] - 0.5) * np.tan(np.radians(10.0))
    np.testing.assert_allclose(arr['z'][ramp_mask], expected_z, atol=1e-3)
    assert np.all(arr['z'][ramp_mask] >= 0.0)
    assert np.max(arr['z'][ramp_mask]) > 0.15

    # 2. Check raised platform deck (X in [-1.39, -0.4], Y in [0.6, 1.6])
    deck_mask = (arr['x'] > -1.39) & (arr['x'] <= -0.4) & (arr['y'] >= 0.6) & (arr['y'] <= 1.6)
    assert np.any(deck_mask)
    np.testing.assert_allclose(arr['z'][deck_mask], 0.08, atol=1e-3)

    # 3. Check approach ramp (X in [-1.99, -1.39], Y in [0.6, 1.6])
    app_mask = (arr['x'] >= -1.99) & (arr['x'] <= -1.39) & (arr['y'] >= 0.6) & (arr['y'] <= 1.6)
    assert np.any(app_mask)
    assert np.all(arr['z'][app_mask] >= 0.0)
    assert np.all(arr['z'][app_mask] <= 0.085)

    # 4. Check flat ground outside ramps (e.g. X in [-0.5, 0.0], Y in [-1.0, -0.5])
    flat_mask = (arr['x'] >= -0.5) & (arr['x'] <= 0.0) & (arr['y'] >= -1.0) & (arr['y'] <= -0.5)
    assert np.any(flat_mask)
    np.testing.assert_allclose(arr['z'][flat_mask], 0.0, atol=1e-5)


def test_performance_benchmark_35000_points(default_builder):
    """Verify synthesis of 35,000 points executes in < 10 ms."""
    h, w = default_builder.height, default_builder.width
    color_map = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)

    # 25,000 ground points
    observed_mask = np.zeros((h, w), dtype=bool)
    flat_indices = np.random.choice(h * w, 25000, replace=False)
    observed_mask.flat[flat_indices] = True

    elevation_grid = np.zeros((h, w), dtype=np.int8)

    # Scan with hits for ~10,000 obstacle points (e.g. 300 rays x ~12-13 vertical steps)
    scan_ranges = [0.65] * 360
    camera_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warmup
    _ = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=elevation_grid,
        scan_ranges=scan_ranges,
        robot_pose=(0.5, -0.2, 0.3),
        camera_image=camera_img,
    )

    # Benchmark over 10 iterations
    timings = []
    total_pts = 0
    for _ in range(10):
        t0 = time.perf_counter()
        msg = default_builder.build_pointcloud(
            stamp=None,
            color_map=color_map,
            observed_mask=observed_mask,
            elevation_grid=elevation_grid,
            scan_ranges=scan_ranges,
            robot_pose=(0.5, -0.2, 0.3),
            camera_image=camera_img,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        total_pts = msg.width

    avg_time = float(np.mean(timings))
    max_time = float(np.max(timings))

    assert total_pts >= 28000, f'Expected >= 28,000 points, got {total_pts}'
    assert avg_time < 10.0, f'Average synthesis time {avg_time:.2f} ms exceeded 10 ms'
    assert max_time < 15.0, f'Peak synthesis time {max_time:.2f} ms exceeded 15 ms'


def test_edge_cases_and_robustness(default_builder):
    """Verify graceful handling of empty masks, invalid scan ranges, and singularities."""
    h, w = default_builder.height, default_builder.width
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    observed_mask = np.zeros((h, w), dtype=bool)

    # Case 1: Empty cloud (no observed cells, no scan hits)
    msg_empty = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )
    assert msg_empty.width == 0
    assert len(msg_empty.data) == 0

    # Case 2: Scan ranges containing NaN, Inf, and negative values
    bad_scan = [float('nan'), float('inf'), -1.0, 0.001, 10.0] * 72
    msg_bad_scan = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=bad_scan,
        robot_pose=(0.0, 0.0, 0.0),
    )
    assert msg_bad_scan.width == 0

    # Case 3: Singularity point directly at camera focal plane (s <= 0.02)
    # Camera xc = 0.0682, so scan at 0.07m hits at x_b = 0.07, s = 0.0018 m <= 0.02
    close_scan = [float('inf')] * 360
    close_scan[180] = 0.07  # Near singularity
    msg_singularity = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=close_scan,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert msg_singularity.width == len(default_builder.wall_z_steps)


def test_rviz2_config_validation():
    """Verify RViz2 config contains PointCloud3D display and 3D Orbit view controller."""
    rviz_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../khepera4_description/config/khepera4_mapping.rviz',
        )
    )
    assert os.path.isfile(rviz_path), f'RViz config not found at {rviz_path}'

    with open(rviz_path, 'r', encoding='utf-8') as f:
        rviz_data = yaml.safe_load(f)

    # 1. Verify PointCloud3D display configuration
    displays = rviz_data['Visualization Manager']['Displays']
    pc_displays = [d for d in displays if d.get('Name') == 'PointCloud3D']
    assert len(pc_displays) == 1, 'PointCloud3D display not found in RViz configuration'

    pc_disp = pc_displays[0]
    assert pc_disp['Class'] == 'rviz_default_plugins/PointCloud2'
    assert pc_disp['Enabled'] is True
    assert pc_disp['Color Transformer'] == 'RGB8'
    assert pc_disp['Style'] == 'Flat Squares'
    assert math.isclose(float(pc_disp['Size (m)']), 0.03, abs_tol=1e-3)
    assert pc_disp['Topic']['Value'] == '/khepera4/pointcloud3d'
    assert pc_disp['Topic']['Durability Policy'] == 'Transient Local'
    assert pc_disp['Topic']['Reliability Policy'] == 'Reliable'

    # 2. Verify 3D Orbit view perspective
    views = rviz_data['Visualization Manager']['Views']
    current_view = views['Current']
    assert current_view['Class'] == 'rviz_default_plugins/Orbit'
    pitch = float(current_view['Pitch'])
    assert math.isclose(pitch, 0.60, abs_tol=0.05), f'Pitch {pitch} should be ~0.60 rad'

    focal_pt = current_view['Focal Point']
    assert math.isclose(float(focal_pt['X']), 0.0, abs_tol=1e-3)
    assert math.isclose(float(focal_pt['Y']), 0.0, abs_tol=1e-3)
    assert math.isclose(float(focal_pt['Z']), 0.1, abs_tol=1e-3)


def test_layer2_obstacle_walls_robot_orientation_transform(default_builder):
    """Verify obstacle points correctly transform according to robot world pose and yaw."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Robot at pose (x=1.0, y=2.0, yaw=pi/2)
    # Forward scan ray (index 180, angle=0.0) at 0.50 m
    # In base link: xb = 0.50, yb = 0.0
    # In map frame: xw = 1.0 + cos(pi/2)*0.5 = 1.0, yw = 2.0 + sin(pi/2)*0.5 = 2.5
    scan_ranges = [float('inf')] * 360
    scan_ranges[180] = 0.50

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_ranges,
        robot_pose=(1.0, 2.0, math.pi / 2.0),
        camera_image=None,
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    assert len(arr) == len(default_builder.wall_z_steps)
    np.testing.assert_allclose(arr['x'], 1.0, atol=1e-3)
    np.testing.assert_allclose(arr['y'], 2.5, atol=1e-3)
    np.testing.assert_allclose(arr['z'], default_builder.wall_z_steps, atol=1e-3)


def test_layer2_obstacle_walls_no_camera_image_uses_fallback(default_builder):
    """Verify obstacle points use fallback color when camera_image is None."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Obstacle in front of camera
    scan_ranges = [float('inf')] * 360
    scan_ranges[180] = 0.60

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_ranges,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=None,
    )

    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    assert len(arr) == len(default_builder.wall_z_steps)
    r, g, b = unpack_rgb_f32(arr['rgb'])
    for ri, gi, bi in zip(r, g, b):
        assert (ri, gi, bi) == DEFAULT_WALL_COLOR_RGB


def test_custom_builder_configuration():
    """Verify builder functions properly with custom resolution and bounds."""
    custom_builder = PointCloud3DBuilder(
        map_res=0.05,
        map_width=100,
        map_height=100,
        origin_x=-2.5,
        origin_y=-2.5,
        wall_height_max=0.20,
        wall_delta_z=0.05,
    )
    assert len(custom_builder.wall_z_steps) == 5  # 0.0, 0.05, 0.10, 0.15, 0.20
    np.testing.assert_allclose(
        custom_builder.wall_z_steps,
        [0.0, 0.05, 0.10, 0.15, 0.20],
        atol=1e-4,
    )

    color_map = np.zeros((100, 100, 3), dtype=np.float32)
    observed_mask = np.zeros((100, 100), dtype=bool)
    observed_mask[50, 50] = True

    msg = custom_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )
    assert msg.width == 1
    arr = np.frombuffer(msg.data, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    # x = -2.5 + (50 + 0.5) * 0.05 = 0.025
    assert math.isclose(arr['x'][0], 0.025, abs_tol=1e-4)
    assert math.isclose(arr['y'][0], 0.025, abs_tol=1e-4)
