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
Adversarial Stress Test & Benchmark Suite for PointCloud3DBuilder and PointCloud2.

Empirically verifies:
1. High-load execution time benchmark with 50,000+ points (< 10 ms strict requirement).
2. RGB byte unpacking and color channel integrity under extreme values and gradients.
3. Multi-layer elevation continuity along 10° ramp and platform boundary without coordinate drift.
4. RViz2 YAML configuration validation (PointField alignment, Orbit view parameters).
"""

import math
import os
import time

from builtin_interfaces.msg import Time as RosTime
from khepera4_control.pointcloud_builder import (
    DEFAULT_WALL_COLOR_RGB,
    pack_rgb_f32,
    POINT_DTYPE,
    POINT_FIELDS,
    PointCloud3DBuilder,
    unpack_rgb_f32,
)
import numpy as np
import pytest
from sensor_msgs.msg import PointField
import yaml


@pytest.fixture
def default_builder():
    """Create standard Khepera IV PointCloud3DBuilder."""
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


# ==============================================================================
# 1. High-Load Execution Time Benchmark (50,000+ Points)
# ==============================================================================

def test_high_load_benchmark_50k_points():
    """Benchmark PointCloud2 synthesis with 50,000+ points and assert time < 10 ms."""
    builder = PointCloud3DBuilder(map_width=300, map_height=300)
    h, w = 300, 300
    color_map = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)

    # 50,000 ground points
    observed_mask = np.zeros((h, w), dtype=bool)
    flat_indices = np.random.choice(h * w, 50000, replace=False)
    observed_mask.flat[flat_indices] = True

    elevation_grid = np.zeros((h, w), dtype=np.int8)
    scan_ranges = [0.75] * 360  # 360 rays * 13 steps = 4,680 obstacle points
    camera_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    robot_pose = (0.2, -0.4, 0.785)

    # Warmup
    builder.build_pointcloud(
        RosTime(sec=1, nanosec=0),
        color_map,
        observed_mask,
        elevation_grid,
        scan_ranges,
        robot_pose,
        camera_img,
    )

    # Benchmark over 50 iterations
    timings = []
    total_points = 0
    for _ in range(50):
        t0 = time.perf_counter()
        msg = builder.build_pointcloud(
            RosTime(sec=1, nanosec=0),
            color_map,
            observed_mask,
            elevation_grid,
            scan_ranges,
            robot_pose,
            camera_img,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        total_points = msg.width

    assert total_points >= 50000, f'Expected >= 50,000 points, got {total_points}'
    mean_ms = float(np.mean(timings))
    median_ms = float(np.median(timings))
    min_ms = float(np.min(timings))
    max_ms = float(np.max(timings))
    p95_ms = float(np.percentile(timings, 95))

    # Log statistical summary to stdout (visible in pytest -s)
    print(
        f'\n[50k Benchmark] N={total_points} pts | '
        f'Min: {min_ms:.3f} ms, Mean: {mean_ms:.3f} ms, Median: {median_ms:.3f} ms, '
        f'P95: {p95_ms:.3f} ms, Max: {max_ms:.3f} ms'
    )

    # Strict performance assertion: peak time must be strictly under 10.0 ms
    assert max_ms < 10.0, (
        f'Max synthesis time {max_ms:.3f} ms exceeded 10.0 ms limit '
        f'(mean: {mean_ms:.3f} ms, p95: {p95_ms:.3f} ms, points: {total_points})'
    )
    assert mean_ms < 5.0, f'Mean synthesis time {mean_ms:.3f} ms exceeded 5.0 ms'


def test_stress_benchmark_75k_points():
    """Stress test PointCloud2 synthesis with 74,680 points and assert time < 10 ms."""
    builder = PointCloud3DBuilder(map_width=350, map_height=350)
    h, w = 350, 350
    color_map = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)

    # 70,000 ground points + 4,680 obstacle points = 74,680 points
    observed_mask = np.zeros((h, w), dtype=bool)
    flat_indices = np.random.choice(h * w, 70000, replace=False)
    observed_mask.flat[flat_indices] = True

    elevation_grid = np.zeros((h, w), dtype=np.int8)
    scan_ranges = [0.90] * 360
    camera_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    timings = []
    total_pts = 0
    for _ in range(25):
        t0 = time.perf_counter()
        msg = builder.build_pointcloud(
            None,
            color_map,
            observed_mask,
            elevation_grid,
            scan_ranges,
            (0.0, 0.0, 0.0),
            camera_img,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        total_pts = msg.width

    assert total_pts >= 74000
    mean_ms = float(np.mean(timings))
    max_ms = float(np.max(timings))
    assert mean_ms < 8.0, f'Mean {mean_ms:.3f} ms exceeded 8.0 ms for {total_pts} points'
    assert max_ms < 10.0, f'Max {max_ms:.3f} ms exceeded 10.0 ms for {total_pts} points'


def test_extreme_stress_benchmark_100k_points():
    """Extreme stress test with 100,000 points verifying throughput and scaling."""
    builder = PointCloud3DBuilder(map_width=400, map_height=400)
    h, w = 400, 400
    color_map = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)

    # 95,320 ground points + 4,680 obstacle points = 100,000 points
    observed_mask = np.zeros((h, w), dtype=bool)
    flat_indices = np.random.choice(h * w, 95320, replace=False)
    observed_mask.flat[flat_indices] = True

    elevation_grid = np.zeros((h, w), dtype=np.int8)
    scan_ranges = [0.85] * 360
    camera_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        msg = builder.build_pointcloud(
            None,
            color_map,
            observed_mask,
            elevation_grid,
            scan_ranges,
            (1.0, -1.0, 1.57),
            camera_img,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)

    assert msg.width == 100000
    mean_ms = float(np.mean(timings))
    max_ms = float(np.max(timings))
    assert mean_ms < 10.0, f'100k points mean {mean_ms:.3f} ms exceeded 10.0 ms'
    assert max_ms < 15.0, f'100k points peak {max_ms:.3f} ms exceeded 15.0 ms'


# ==============================================================================
# 2. RGB Byte Unpacking and Color Channel Integrity
# ==============================================================================

def test_rgb_extreme_values_and_channel_isolation():
    """Verify bitwise packing/unpacking across all extreme channel combinations."""
    extremes = [0, 1, 2, 63, 64, 127, 128, 129, 254, 255]
    r_list, g_list, b_list = [], [], []
    for r_val in extremes:
        for g_val in extremes:
            for b_val in extremes:
                r_list.append(r_val)
                g_list.append(g_val)
                b_list.append(b_val)

    r = np.array(r_list, dtype=np.uint8)
    g = np.array(g_list, dtype=np.uint8)
    b = np.array(b_list, dtype=np.uint8)

    packed = pack_rgb_f32(r, g, b)
    assert packed.dtype == np.float32
    assert len(packed) == 1000

    r_out, g_out, b_out = unpack_rgb_f32(packed)
    np.testing.assert_array_equal(r_out, r, err_msg='Red channel corrupted in extreme test')
    np.testing.assert_array_equal(g_out, g, err_msg='Green channel corrupted in extreme test')
    np.testing.assert_array_equal(b_out, b, err_msg='Blue channel corrupted in extreme test')


def test_rgb_channel_isolation_no_leakage():
    """Verify modifying one color channel leaves other channels completely unaffected."""
    # Red-only activation
    r_pure = np.array([255, 128, 1], dtype=np.uint8)
    zeros = np.zeros(3, dtype=np.uint8)
    packed = pack_rgb_f32(r_pure, zeros, zeros)
    r_out, g_out, b_out = unpack_rgb_f32(packed)
    np.testing.assert_array_equal(r_out, r_pure)
    np.testing.assert_array_equal(g_out, zeros)
    np.testing.assert_array_equal(b_out, zeros)

    # Green-only activation
    g_pure = np.array([255, 128, 1], dtype=np.uint8)
    packed = pack_rgb_f32(zeros, g_pure, zeros)
    r_out, g_out, b_out = unpack_rgb_f32(packed)
    np.testing.assert_array_equal(r_out, zeros)
    np.testing.assert_array_equal(g_out, g_pure)
    np.testing.assert_array_equal(b_out, zeros)

    # Blue-only activation
    b_pure = np.array([255, 128, 1], dtype=np.uint8)
    packed = pack_rgb_f32(zeros, zeros, b_pure)
    r_out, g_out, b_out = unpack_rgb_f32(packed)
    np.testing.assert_array_equal(r_out, zeros)
    np.testing.assert_array_equal(g_out, zeros)
    np.testing.assert_array_equal(b_out, b_pure)


def test_rgb_gradients_and_alternating_bit_patterns():
    """Verify full 256-step continuous gradients and checkerboard bit patterns."""
    grad = np.arange(256, dtype=np.uint8)
    packed_diag = pack_rgb_f32(grad, grad, 255 - grad)
    r_out, g_out, b_out = unpack_rgb_f32(packed_diag)
    np.testing.assert_array_equal(r_out, grad)
    np.testing.assert_array_equal(g_out, grad)
    np.testing.assert_array_equal(b_out, 255 - grad)

    # High frequency alternating patterns: 0x55 (01010101) and 0xAA (10101010)
    p1 = np.full(100, 0x55, dtype=np.uint8)
    p2 = np.full(100, 0xAA, dtype=np.uint8)
    packed_pattern = pack_rgb_f32(p1, p2, p1)
    r_pat, g_pat, b_pat = unpack_rgb_f32(packed_pattern)
    np.testing.assert_array_equal(r_pat, p1)
    np.testing.assert_array_equal(g_pat, p2)
    np.testing.assert_array_equal(b_pat, p1)


def test_pointcloud_color_map_clipping_and_rounding(default_builder):
    """Verify out-of-range and float values in color_map are clipped and rounded cleanly."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Cell (50, 50): Negative and excessive values
    observed_mask[50, 50] = True
    color_map[50, 50] = [-50.0, 300.0, 127.6]

    # Cell (60, 60): Boundary fractional values
    observed_mask[60, 60] = True
    color_map[60, 60] = [0.4, 254.6, 128.4]

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )
    arr = np.frombuffer(msg.data, dtype=POINT_DTYPE)
    assert len(arr) == 2

    r_out, g_out, b_out = unpack_rgb_f32(arr['rgb'])
    # First point: [-50.0 -> 0, 300.0 -> 255, 127.6 -> 128]
    assert r_out[0] == 0
    assert g_out[0] == 255
    assert b_out[0] == 128

    # Second point: [0.4 -> 0, 254.6 -> 255, 128.4 -> 128]
    assert r_out[1] == 0
    assert g_out[1] == 255
    assert b_out[1] == 128


def test_camera_texturing_exact_pixel_sampling(default_builder):
    """Verify camera texture back-projection samples the exact pixel color from camera frame."""
    color_map = np.zeros((200, 200, 3), dtype=np.float32)
    observed_mask = np.zeros((200, 200), dtype=bool)

    # Place obstacle straight ahead at distance 0.50 m
    # Optical Z = 0.50 - 0.0682 = 0.4318 m
    # Pinhole projection: u = 320 - 554.256 * 0 = 320
    # For zb = 0.03 m, v = 240 + 554.256 * (0.0283 - 0.03) / 0.4318 = 237.818 -> 238
    scan_ranges = [float('inf')] * 360
    scan_ranges[180] = 0.50

    # Synthetic camera frame with unique colored pixels at exact projected pinhole location
    cam_img = np.zeros((480, 640, 3), dtype=np.uint8)
    s_dist = 0.50 - default_builder.xc
    v_val = default_builder.cy + default_builder.fy * (default_builder.h - 0.03) / s_dist
    v_proj = int(round(v_val))
    u_proj = int(round(default_builder.cx))
    cam_img[v_proj, u_proj] = [175, 88, 222]  # Violet at exact projection point

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_ranges,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=cam_img,
    )
    arr = np.frombuffer(msg.data, dtype=POINT_DTYPE)
    r_out, g_out, b_out = unpack_rgb_f32(arr['rgb'])

    # Find the point at zb = 0.03 m
    zb_idx = np.argmin(np.abs(arr['z'] - 0.03))
    assert math.isclose(arr['z'][zb_idx], 0.03, abs_tol=1e-4)
    assert (r_out[zb_idx], g_out[zb_idx], b_out[zb_idx]) == (175, 88, 222)

    # Obstacle points behind the camera should fall back to DEFAULT_WALL_COLOR_RGB
    scan_behind = [float('inf')] * 360
    scan_behind[0] = 0.40  # Directly behind (angle = -pi)
    msg_behind = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=scan_behind,
        robot_pose=(0.0, 0.0, 0.0),
        camera_image=cam_img,
    )
    arr_b = np.frombuffer(msg_behind.data, dtype=POINT_DTYPE)
    r_b, g_b, b_b = unpack_rgb_f32(arr_b['rgb'])
    for ri, gi, bi in zip(r_b, g_b, b_b):
        assert (ri, gi, bi) == DEFAULT_WALL_COLOR_RGB


# ==============================================================================
# 3. Multi-Layer Elevation Continuity & Coordinate Drift
# ==============================================================================

def test_ramp_10deg_continuity_and_slope(default_builder):
    """Verify 10° ramp has zero coordinate drift and continuous elevation along its slope."""
    h, w = default_builder.height, default_builder.width
    color_map = np.full((h, w, 3), 100.0, dtype=np.float32)
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
    arr = np.frombuffer(msg.data, dtype=POINT_DTYPE)

    # 10 deg ramp domain: X in [0.5, 1.7], Y in [1.0, 1.6]
    ramp_mask = (arr['x'] >= 0.5) & (arr['x'] <= 1.7) & (arr['y'] >= 1.0) & (arr['y'] <= 1.6)
    ramp_pts = arr[ramp_mask]
    assert len(ramp_pts) > 0

    tan_10 = math.tan(math.radians(10.0))
    expected_z = (ramp_pts['x'] - 0.5) * tan_10
    np.testing.assert_allclose(
        ramp_pts['z'],
        expected_z,
        atol=1e-4,
        err_msg='Ramp elevation does not match tan(10°) slope formula',
    )

    # Check continuity at entrance X = 0.5: height approaches 0.0
    entrance_pts = ramp_pts[ramp_pts['x'] < 0.55]
    assert len(entrance_pts) > 0
    assert np.all(entrance_pts['z'] < 0.01)

    # Verify points immediately outside ramp (X < 0.5, Y in [1.0, 1.6]) have Z = 0.0
    outside_mask = (arr['x'] >= 0.4) & (arr['x'] < 0.5) & (arr['y'] >= 1.0) & (arr['y'] <= 1.6)
    outside_pts = arr[outside_mask]
    assert len(outside_pts) > 0
    np.testing.assert_allclose(outside_pts['z'], 0.0, atol=1e-5)


def test_platform_boundary_continuity_and_knife_edge(default_builder):
    """Verify seamless continuity between approach ramp and raised platform at X = -1.39."""
    h, w = default_builder.height, default_builder.width
    color_map = np.full((h, w, 3), 100.0, dtype=np.float32)
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
    arr = np.frombuffer(msg.data, dtype=POINT_DTYPE)

    # Approach ramp: X in [-1.99, -1.39], Y in [0.6, 1.6]
    app_mask = (arr['x'] >= -1.99) & (arr['x'] <= -1.39) & (arr['y'] >= 0.6) & (arr['y'] <= 1.6)
    app_pts = arr[app_mask]

    # Raised deck: X in (-1.39, -0.4], Y in [0.6, 1.6]
    deck_mask = (arr['x'] > -1.39) & (arr['x'] <= -0.4) & (arr['y'] >= 0.6) & (arr['y'] <= 1.6)
    deck_pts = arr[deck_mask]

    assert len(app_pts) > 0
    assert len(deck_pts) > 0

    # 1. Knife-edge entrance at X = -1.99: Z should approach 0.0
    entrance_pts = app_pts[app_pts['x'] < -1.90]
    assert len(entrance_pts) > 0
    assert np.all(entrance_pts['z'] < 0.015)

    # 2. Deck height is exactly 0.08 m
    np.testing.assert_allclose(deck_pts['z'], 0.08, atol=1e-4)

    # 3. Transition continuity at X = -1.39:
    # Approach ramp peak at X near -1.39
    top_ramp_pts = app_pts[app_pts['x'] > -1.43]
    assert len(top_ramp_pts) > 0
    # Both sides of X = -1.39 boundary should agree within 0.005 m (one cell slope)
    for z_val in top_ramp_pts['z']:
        assert math.isclose(z_val, 0.08, abs_tol=0.005)

    # 4. Cliff edge: East drop-off (X > -0.4, Y in [0.6, 1.6]) drops to Z = 0.0
    cliff_east = (arr['x'] > -0.4) & (arr['x'] < -0.3) & (arr['y'] >= 0.6) & (arr['y'] <= 1.6)
    assert np.any(cliff_east)
    np.testing.assert_allclose(arr['z'][cliff_east], 0.0, atol=1e-5)


def test_coordinate_alignment_with_slam_occupancy_grid(default_builder):
    """Verify PointCloud cell coordinates exactly match SLAM grid cell centers without drift."""
    # Test across random grid cell indices (r, c)
    rng = np.random.default_rng(1234)
    rows = rng.integers(0, default_builder.height, size=50)
    cols = rng.integers(0, default_builder.width, size=50)

    observed_mask = np.zeros((default_builder.height, default_builder.width), dtype=bool)
    observed_mask[rows, cols] = True

    color_map = np.zeros((default_builder.height, default_builder.width, 3), dtype=np.float32)

    msg = default_builder.build_pointcloud(
        stamp=None,
        color_map=color_map,
        observed_mask=observed_mask,
        elevation_grid=None,
        scan_ranges=None,
        robot_pose=(0.0, 0.0, 0.0),
    )
    arr = np.frombuffer(msg.data, dtype=POINT_DTYPE)

    # Verify each synthesized point matches exact formula
    sorted_rows, sorted_cols = np.where(observed_mask)
    expected_x = (
        default_builder.origin_x +
        (sorted_cols.astype(np.float32) + 0.5) * default_builder.res
    )
    expected_y = (
        default_builder.origin_y +
        (sorted_rows.astype(np.float32) + 0.5) * default_builder.res
    )

    np.testing.assert_allclose(arr['x'], expected_x, atol=1e-5)
    np.testing.assert_allclose(arr['y'], expected_y, atol=1e-5)


# ==============================================================================
# 4. RViz2 YAML Configuration Validation
# ==============================================================================

def test_rviz2_yaml_pointfield_and_orbit_parameters():
    """Verify RViz2 config file exists, parses cleanly, and validates all parameters."""
    rviz_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../khepera4_description/config/khepera4_mapping.rviz',
        )
    )
    assert os.path.exists(rviz_path), f'RViz2 configuration not found: {rviz_path}'

    with open(rviz_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Validate Global Options
    global_opts = data['Visualization Manager']['Global Options']
    assert global_opts['Fixed Frame'] == 'map'

    # Locate PointCloud3D display
    displays = data['Visualization Manager']['Displays']
    pc_displays = [d for d in displays if d.get('Name') == 'PointCloud3D']
    assert len(pc_displays) == 1, 'Exactly one PointCloud3D display must be present'

    disp = pc_displays[0]
    assert disp['Class'] == 'rviz_default_plugins/PointCloud2'
    assert disp['Enabled'] is True
    assert disp['Color Transformer'] == 'RGB8'
    assert disp['Channel Name'] == 'rgb'
    assert disp['Position Transformer'] == 'XYZ'
    assert disp['Style'] == 'Flat Squares'
    assert math.isclose(float(disp['Size (m)']), 0.03, abs_tol=1e-3)
    assert disp['Topic']['Value'] == '/khepera4/pointcloud3d'
    assert disp['Topic']['Durability Policy'] == 'Transient Local'
    assert disp['Topic']['Reliability Policy'] == 'Reliable'

    # Validate Orbit view controller
    current_view = data['Visualization Manager']['Views']['Current']
    assert current_view['Class'] == 'rviz_default_plugins/Orbit'
    pitch = float(current_view['Pitch'])
    assert pitch >= 0.50, f'Orbit pitch {pitch} must be >= 0.50 rad for 3D perspective'
    assert math.isclose(pitch, 0.60, abs_tol=0.05)

    focal = current_view['Focal Point']
    assert math.isclose(float(focal['X']), 0.0, abs_tol=1e-3)
    assert math.isclose(float(focal['Y']), 0.0, abs_tol=1e-3)
    assert math.isclose(float(focal['Z']), 0.1, abs_tol=1e-3)

    # Verify POINT_FIELDS schema in Python matches RViz expectations
    assert len(POINT_FIELDS) == 4
    assert POINT_FIELDS[0].name == 'x' and POINT_FIELDS[0].offset == 0
    assert POINT_FIELDS[1].name == 'y' and POINT_FIELDS[1].offset == 4
    assert POINT_FIELDS[2].name == 'z' and POINT_FIELDS[2].offset == 8
    assert POINT_FIELDS[3].name == 'rgb' and POINT_FIELDS[3].offset == 12
    for f in POINT_FIELDS:
        assert f.datatype == PointField.FLOAT32
        assert f.count == 1
