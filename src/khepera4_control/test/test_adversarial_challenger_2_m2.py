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
Adversarial Stress Test Suite — Challenger 2 (Milestone M2 Gate Verification).

Verifies Milestone M2 Feature F3 (Keyframe System) & Feature F4 (Persistence):
1. Keyframe Index Zero-Padding (keyframe_XXXX.png) & Manifest JSON Schema Integrity.
2. Composite Export Layer Ordering & Blending Hierarchy:
   Visual Base -> SLAM Obstacles -> Elevation Slopes (>5 deg) -> Cliff Hazards ->
   Trajectory Path -> Landmark Gold Pins -> Robot Footprint.
3. Empty Map Persistence, Boundary Coordinates & Degenerate Grids.
4. Idempotent Repeated Clear Map Operations & Disk Lifecycle Robustness.
5. Production Node Integration: VisualOrthomapper & Khepera4Mapper Persistence.
"""

import json
import math
import os
import shutil
import sys
import tempfile

from khepera4_control.khepera4_mapper import Khepera4Mapper
from khepera4_control.visual_orthomapper import (
    generate_composite_map,
    KeyframeManager,
    VisualOrthomapper,
)
import numpy as np
from PIL import Image
import pytest
import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture(scope='module', autouse=True)
def manage_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def temp_workspace():
    """Create and cleanly destroy a temporary directory for maps and keyframes."""
    td = tempfile.mkdtemp(prefix='challenger_m2_')
    kf_dir = os.path.join(td, 'keyframes')
    yield td, kf_dir
    shutil.rmtree(td, ignore_errors=True)


# ==============================================================================
# Suite 1: Keyframe Index Zero-Padding & Manifest Schema Integrity
# ==============================================================================

class TestKeyframeZeroPaddingAndManifestIntegrity:
    """Adversarial stress-testing of keyframe naming and manifest serialization."""

    def test_keyframe_filename_4digit_zero_padding_progression(self, temp_workspace):
        """Verify filenames adhere to keyframe_XXXX.png across magnitude steps."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Generate specific IDs to verify 4-digit formatting
        for expected_id in [1, 2, 9, 10, 99, 100, 999, 1000]:
            km.next_id = expected_id
            entry = km.record_keyframe(
                dummy_img, 0.0, 0.0, 0.0, trigger_type='test'
            )
            expected_filename = f'keyframe_{expected_id:04d}.png'
            assert entry['id'] == expected_id
            assert entry['filename'] == expected_filename
            assert os.path.exists(os.path.join(kf_dir, expected_filename))

    def test_keyframe_filename_overflow_beyond_9999(self, temp_workspace):
        """Verify behavior when keyframe ID exceeds 9999 (e.g. 10000)."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 120, dtype=np.uint8)

        km.next_id = 10000
        entry = km.record_keyframe(dummy_img, 0.0, 0.0, 0.0, trigger_type='overflow_test')
        assert entry['id'] == 10000
        assert entry['filename'] == 'keyframe_10000.png'
        assert os.path.exists(os.path.join(kf_dir, 'keyframe_10000.png'))

    def test_lossless_png_encoding_integrity(self, temp_workspace):
        """Verify stored keyframe PNG preserves exact pixel RGB values losslessly."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)

        # Create known synthetic pattern
        pattern = np.zeros((480, 640, 3), dtype=np.uint8)
        pattern[100:200, 150:250] = [255, 128, 64]
        pattern[300:400, 450:550] = [10, 200, 90]

        entry = km.record_keyframe(pattern, 1.0, 2.0, 0.5, trigger_type='pattern_test')
        saved_file = os.path.join(kf_dir, entry['filename'])

        with Image.open(saved_file) as im:
            loaded_arr = np.array(im)
            assert loaded_arr.shape == (480, 640, 3)
            assert np.array_equal(loaded_arr, pattern)

    def test_manifest_schema_compliance_and_camera_pose(self, temp_workspace):
        """Verify manifest.json contains all required fields and correct forward camera pose."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(
            keyframe_dir=kf_dir,
            camera_xc=0.0682,
            camera_h=0.0283,
            hfov_deg=60.0,
            focal_length=554.256
        )
        dummy_img = np.full((480, 640, 3), 200, dtype=np.uint8)

        rx, ry, ryaw = 1.5, -0.8, math.pi / 3.0  # 60 degrees yaw
        entry = km.record_keyframe(
            dummy_img, rx, ry, ryaw, trigger_type='displacement',
            linear_vel=0.12, angular_vel=-0.05, cliff_alert=False
        )

        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        assert manifest['map_frame'] == 'map'
        assert manifest['manifest_version'] == '1.0'
        assert manifest['total_keyframes'] == 1
        assert len(manifest['keyframes']) == 1

        kf = manifest['keyframes'][0]
        assert kf['id'] == entry['id']
        assert kf['trigger_source'] == 'displacement'
        assert kf['image_file'] == entry['image_file']

        # Check camera extrinsics projection in world coordinates
        expected_cam_x = round(rx + 0.0682 * math.cos(ryaw), 4)
        expected_cam_y = round(ry + 0.0682 * math.sin(ryaw), 4)
        assert kf['camera_pose']['x'] == pytest.approx(expected_cam_x, abs=1e-4)
        assert kf['camera_pose']['y'] == pytest.approx(expected_cam_y, abs=1e-4)
        assert kf['camera_pose']['z'] == pytest.approx(0.0283, abs=1e-4)
        assert kf['camera_pose']['yaw'] == pytest.approx(round(ryaw, 4), abs=1e-4)

    def test_rapid_concurrent_manifest_writes(self, temp_workspace):
        """Verify rapid repeated writes maintain manifest integrity without corrupting JSON."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 50, dtype=np.uint8)

        count = 35
        for i in range(count):
            km.record_keyframe(
                dummy_img, float(i) * 0.1, 0.0, 0.0, trigger_type='stress'
            )

        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        assert manifest['total_keyframes'] == count
        assert len(manifest['keyframes']) == count
        assert manifest['keyframes'][-1]['id'] == count
        # Ensure temporary file was cleanly renamed and not left orphaned
        assert not os.path.exists(km.manifest_path + '.tmp')


# ==============================================================================
# Suite 2: Composite Export Layer Ordering & Blending Hierarchy
# ==============================================================================

class TestCompositeExportLayerOrdering:
    """Empirical verification of pixel layer blending order and visual precedence."""

    def test_layer_ordering_obstacles_over_slopes(self, temp_workspace):
        """Verify SLAM obstacles (log_odds > 0.5) take precedence over slopes (> 5 deg)."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'test_composite_obs_vs_slope.png')

        h, w = 200, 200
        v_grid = np.full((h, w, 3), 200, dtype=np.uint8)  # Light gray base
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        # At grid cell (100, 100), mark BOTH obstacle AND slope > 5 deg
        lo_grid[100, 100] = 2.0       # Obstacle
        elev_grid[100, 100] = 15      # 15 deg slope

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[], keyframes=[]
        )

        with Image.open(composite_path) as im:
            arr = np.array(im)
            # Row 100 -> flipped y_img = 199 - 100 = 99
            pixel = arr[99, 100]
            # Obstacle color is [20, 20, 25] (dark charcoal), slope is amber (~[220, 160, 40])
            assert pixel[0] == 20 and pixel[1] == 20 and pixel[2] == 25, (
                f'Obstacle must override slope! Got pixel {pixel}'
            )

    def test_layer_ordering_cliffs_over_slopes(self, temp_workspace):
        """Verify Cliff hazard (surface == 100) takes precedence over slopes (> 5 deg)."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'test_composite_cliff_vs_slope.png')

        h, w = 200, 200
        v_grid = np.full((h, w, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        # At cell (80, 80), mark slope AND cliff hazard
        elev_grid[80, 80] = 12       # Slope 12 deg
        surf_grid[80, 80] = 100      # Cliff hazard

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[], keyframes=[]
        )

        with Image.open(composite_path) as im:
            arr = np.array(im)
            y_img = 199 - 80
            pixel = arr[y_img, 80]
            # Cliff hazard color is [230, 25, 25] (crimson red)
            assert pixel[0] == 230 and pixel[1] == 25 and pixel[2] == 25, (
                f'Cliff hazard must override slope! Got pixel {pixel}'
            )

    def test_layer_ordering_path_over_raster_layers(self, temp_workspace):
        """Verify robot trajectory polyline is drawn on top of obstacle, slope, and cliff cells."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'test_composite_path_over_raster.png')

        h, w = 200, 200
        v_grid = np.full((h, w, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        # Fill horizontal band with obstacles
        lo_grid[100, 90:110] = 2.0  # Obstacle line

        # Draw a vertical path through this band from (0.0, -0.5) to (0.0, 0.5)
        # In grid: origin=[-3.0, -3.0], res=0.03 -> (0.0, 0.0) is (100, 100)
        path = [(0.0, -0.3), (0.0, 0.3)]

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=path, keyframes=[]
        )

        with Image.open(composite_path) as im:
            arr = np.array(im)
            # Intersection cell at (col=100, row=100) -> y_img = 99
            pixel = arr[99, 100]
            # Trajectory color is (30, 140, 255) (cyan/blue). It must NOT be obstacle [20, 20, 25]
            assert pixel[2] > 200 and pixel[0] < 100, (
                f'Trajectory line must be visible over obstacle! Got {pixel}'
            )

    def test_layer_ordering_gold_pins_over_path(self, temp_workspace):
        """Verify landmark gold pin marker is drawn on top of the trajectory line."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'test_composite_pin_over_path.png')

        h, w = 200, 200
        v_grid = np.full((h, w, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        # Path passes through (0.0, 0.0)
        path = [(-0.5, 0.0), (0.5, 0.0)]

        # Place keyframe pin exactly at (0.0, 0.0)
        keyframes = [{
            'id': 1,
            'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        }]

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=path, keyframes=keyframes
        )

        with Image.open(composite_path) as im:
            arr = np.array(im)
            # Pin center is at (100, 99). Marker is filled with gold (245, 158, 11)
            pin_pixel = arr[99, 100]
            assert pin_pixel[0] == 245 and pin_pixel[1] == 158 and pin_pixel[2] == 11, (
                f'Pin marker must be visible over path! Got {pin_pixel}'
            )


# ==============================================================================
# Suite 3: Empty Map Operations & Persistence Robustness
# ==============================================================================

class TestEmptyMapOperationsAndRobustness:
    """Stress-test edge cases: empty maps, out-of-bounds coordinates, and directory handling."""

    def test_save_composite_map_with_all_empty_inputs(self, temp_workspace):
        """Verify generate_composite_map succeeds with completely uninitialized grids."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'empty_composite.png')

        h, w = 200, 200
        empty_visual = np.zeros((h, w, 3), dtype=np.uint8)
        empty_lo = np.zeros((h, w), dtype=np.float32)
        empty_elev = np.full((h, w), -1, dtype=np.int8)
        empty_surf = np.full((h, w), -1, dtype=np.int8)

        out = generate_composite_map(
            composite_path, empty_visual, empty_lo, empty_elev, empty_surf,
            robot_path=[], keyframes=[], current_pose=None
        )

        assert os.path.exists(out)
        with Image.open(out) as im:
            assert im.size == (200, 200)
            assert im.mode == 'RGB'

    def test_save_composite_map_creates_nonexistent_subdirectories(self, temp_workspace):
        """Verify automatic creation of nested directories if target path does not exist."""
        td, _ = temp_workspace
        nested_path = os.path.join(td, 'nested', 'deep', 'map', 'khepera_visual_map.png')

        h, w = 200, 200
        empty_visual = np.full((h, w, 3), 180, dtype=np.uint8)
        empty_lo = np.zeros((h, w), dtype=np.float32)
        empty_elev = np.full((h, w), -1, dtype=np.int8)
        empty_surf = np.full((h, w), -1, dtype=np.int8)

        out = generate_composite_map(
            nested_path, empty_visual, empty_lo, empty_elev, empty_surf,
            robot_path=[], keyframes=[]
        )
        assert os.path.exists(out)

    def test_save_composite_map_with_out_of_bounds_poses(self, temp_workspace):
        """Verify out-of-bounds trajectory and keyframe coordinates do not crash."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'out_of_bounds_test.png')

        h, w = 200, 200
        v_grid = np.full((h, w, 3), 180, dtype=np.uint8)
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        # Trajectory far outside grid bounds (e.g. 50 meters away)
        path = [(50.0, 50.0), (-50.0, -50.0), (100.0, -100.0)]
        # Keyframe far outside bounds
        keyframes = [
            {'id': 1, 'pose': {'x': 1000.0, 'y': 1000.0, 'yaw': 0.0}},
            {'id': 2, 'pose': {'x': -500.0, 'y': -500.0, 'yaw': 0.0}}
        ]
        # Current pose out of bounds
        cur_pose = (99.0, 99.0, 0.0)

        out = generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=path, keyframes=keyframes, current_pose=cur_pose
        )
        assert os.path.exists(out)

    def test_save_composite_map_with_2d_grayscale_visual_grid(self, temp_workspace):
        """Verify 2D grayscale visual array is converted to 3-channel RGB without error."""
        td, _ = temp_workspace
        composite_path = os.path.join(td, 'grayscale_visual_test.png')

        h, w = 200, 200
        gray_grid = np.full((h, w), 128, dtype=np.uint8)
        lo_grid = np.zeros((h, w), dtype=np.float32)
        elev_grid = np.full((h, w), -1, dtype=np.int8)
        surf_grid = np.full((h, w), -1, dtype=np.int8)

        out = generate_composite_map(
            composite_path, gray_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[], keyframes=[]
        )
        assert os.path.exists(out)
        with Image.open(out) as im:
            assert im.mode == 'RGB'


# ==============================================================================
# Suite 4: Repeated Clear Map Lifecycle & Race Conditions
# ==============================================================================

class TestRepeatedClearMapOperations:
    """Verify idempotency, lifecycle persistence, and keyframe ID reset upon clear."""

    def test_repeated_clear_is_idempotent(self, temp_workspace):
        """Verify calling clear() 10 times consecutively does not raise errors."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 100, dtype=np.uint8)

        # Populate with 5 keyframes
        for i in range(5):
            km.record_keyframe(dummy_img, float(i), 0.0, 0.0, trigger_type='test')

        # Clear 10 times in a row
        for _ in range(10):
            km.clear()
            assert km.next_id == 1
            assert len(km.keyframes) == 0

        # Manifest must reflect 0 keyframes
        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        assert manifest['total_keyframes'] == 0
        assert manifest['keyframes'] == []

    def test_clear_when_directory_contains_foreign_files(self, temp_workspace):
        """Verify clear() only deletes PNG files and does not crash on foreign files."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 100, dtype=np.uint8)

        km.record_keyframe(dummy_img, 0.0, 0.0, 0.0, trigger_type='test')

        # Add a foreign text file and a subdirectory
        foreign_txt = os.path.join(kf_dir, 'notes.txt')
        with open(foreign_txt, 'w', encoding='utf-8') as f:
            f.write('Do not delete me')
        sub_dir = os.path.join(kf_dir, 'subfolder')
        os.makedirs(sub_dir, exist_ok=True)

        km.clear()

        # PNG must be gone, foreign text and subfolder must remain unharmed
        assert not os.path.exists(os.path.join(kf_dir, 'keyframe_0001.png'))
        assert os.path.exists(foreign_txt)
        assert os.path.exists(sub_dir)

    def test_save_clear_remap_full_mission_lifecycle(self, temp_workspace):
        """Verify complete operational mission cycle: map -> save -> clear -> remap -> save."""
        td, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 150, dtype=np.uint8)

        # 1. First mission: 3 keyframes
        km.record_keyframe(dummy_img, 0.0, 0.0, 0.0, trigger_type='init')
        km.record_keyframe(dummy_img, 0.4, 0.0, 0.0, trigger_type='displacement')
        km.record_keyframe(dummy_img, 0.8, 0.0, 0.0, trigger_type='displacement')
        assert len(km.keyframes) == 3

        # Save mission 1 composite
        m1_map = os.path.join(td, 'mission1_map.png')
        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)
        generate_composite_map(
            m1_map, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[(0.0, 0.0), (0.8, 0.0)], keyframes=km.keyframes
        )
        assert os.path.exists(m1_map)

        # 2. Clear mission 1
        km.clear()
        assert len(km.keyframes) == 0

        # 3. Second mission: Record new keyframes, verifying ID resets to 1
        e_fresh = km.record_keyframe(dummy_img, 0.0, 0.0, 0.0, trigger_type='init')
        assert e_fresh['id'] == 1
        assert e_fresh['filename'] == 'keyframe_0001.png'
        assert os.path.exists(os.path.join(kf_dir, 'keyframe_0001.png'))


# ==============================================================================
# Suite 5: ROS 2 Node Integration & Service Callbacks
# ==============================================================================

class TestRosNodeIntegrationAndServices:
    """Integration verification of VisualOrthomapper and Khepera4Mapper persistence services."""

    def test_visual_orthomapper_pin_keyframe_callback_without_data(self, temp_workspace):
        """Verify /pin_keyframe fails gracefully if odometry or camera frame is missing."""
        td, kf_dir = temp_workspace
        node = VisualOrthomapper(
            parameter_overrides=[
                Parameter('keyframe_dir', Parameter.Type.STRING, kf_dir),
            ]
        )
        try:
            req = Trigger.Request()
            resp = Trigger.Response()
            out = node.pin_keyframe_callback(req, resp)
            assert out.success is False
            assert 'not yet available' in out.message
        finally:
            node.destroy_node()

    def test_visual_orthomapper_pin_keyframe_callback_success(self, temp_workspace):
        """Verify /pin_keyframe captures snapshot when odometry and camera frame are present."""
        td, kf_dir = temp_workspace
        node = VisualOrthomapper(
            parameter_overrides=[
                Parameter('keyframe_dir', Parameter.Type.STRING, kf_dir),
            ]
        )
        try:
            node.has_odom = True
            node.robot_x = 0.5
            node.robot_y = -0.5
            node.robot_yaw = 1.0
            node.latest_camera_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

            req = Trigger.Request()
            resp = Trigger.Response()
            out = node.pin_keyframe_callback(req, resp)
            assert out.success is True
            assert 'Pinned Keyframe #1' in out.message
            assert os.path.exists(os.path.join(kf_dir, 'keyframe_0001.png'))
        finally:
            node.destroy_node()

    def test_khepera4_mapper_save_and_clear_map_callbacks(self, temp_workspace):
        """Verify Khepera4Mapper save_map_callback and clear_map_callback integration."""
        td, kf_dir = temp_workspace
        mapper = Khepera4Mapper(
            parameter_overrides=[
                Parameter('map_save_dir', Parameter.Type.STRING, td),
            ]
        )
        try:
            # 1. Trigger save_map on empty mapper
            req = Trigger.Request()
            resp = Trigger.Response()
            save_out = mapper.save_map_callback(req, resp)
            assert save_out.success is True

            # Check that all four persistent files are created
            assert os.path.exists(os.path.join(td, 'khepera_map.pgm'))
            assert os.path.exists(os.path.join(td, 'khepera_map.yaml'))
            assert os.path.exists(os.path.join(td, 'khepera_map.png'))
            assert os.path.exists(os.path.join(td, 'khepera_visual_map.png'))

            # 2. Trigger clear_map
            clear_resp = Trigger.Response()
            clear_out = mapper.clear_map_callback(req, clear_resp)
            assert clear_out.success is True

            # Verify manifest was reset to empty
            manifest_file = os.path.join(td, 'keyframes', 'manifest.json')
            assert os.path.exists(manifest_file)
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            assert manifest_data['total_keyframes'] == 0
            assert manifest_data['keyframes'] == []
        finally:
            mapper.destroy_node()

    def test_khepera4_mapper_save_map_with_corrupted_manifest(self, temp_workspace):
        """Verify Khepera4Mapper save_map recovers gracefully if manifest.json is corrupted."""
        td, kf_dir = temp_workspace
        mapper = Khepera4Mapper(
            parameter_overrides=[
                Parameter('map_save_dir', Parameter.Type.STRING, td),
            ]
        )
        try:
            os.makedirs(kf_dir, exist_ok=True)
            manifest_file = os.path.join(kf_dir, 'manifest.json')
            with open(manifest_file, 'w', encoding='utf-8') as f:
                f.write('{"total_keyframes": INVALID_JSON_TRUNCATED')

            req = Trigger.Request()
            resp = Trigger.Response()
            save_out = mapper.save_map_callback(req, resp)
            # save_map must not crash; it logs a warning and exports composite map
            assert save_out.success is True
            assert os.path.exists(os.path.join(td, 'khepera_visual_map.png'))
        finally:
            mapper.destroy_node()

    def test_keyframe_debouncing_and_angle_wrap_oracle(self, temp_workspace):
        """Verify cliff debouncing and orientation wrap-around directly on KeyframeManager."""
        _, kf_dir = temp_workspace
        km = KeyframeManager(keyframe_dir=kf_dir)
        dummy_img = np.full((480, 640, 3), 100, dtype=np.uint8)

        # 1. Cliff debouncing:
        t0 = 1000.0
        km.record_keyframe(
            dummy_img, 0.0, 0.0, 0.0, trigger_type='cliff_hazard', current_time=t0
        )

        # Same spot after 0.5s: blocked by cooldown (dt < 1.0)
        assert km.should_trigger_cliff(0.01, 0.0, current_time=t0 + 0.5) is False
        # Same spot after 2.0s: blocked by distance (dist < 0.05m)
        assert km.should_trigger_cliff(0.01, 0.0, current_time=t0 + 2.0) is False
        # 0.1m away after 0.5s: blocked by cooldown (dt < 1.0)
        assert km.should_trigger_cliff(0.10, 0.0, current_time=t0 + 0.5) is False
        # 0.1m away after 1.5s: passes both cooldown and distance!
        assert km.should_trigger_cliff(0.10, 0.0, current_time=t0 + 1.5) is True

        # 2. Angle wrap-around:
        km.last_kf_yaw = 3.10  # ~177.6 deg
        # New yaw is -3.10 rad (~ -177.6 deg). Angular difference is ~4.7 deg < 35 deg
        assert km.should_trigger_orientation(-3.10, threshold_deg=35.0) is False
        # New yaw is 2.40 rad. Angular difference is 0.70 rad (~40 deg >= 35 deg)
        assert km.should_trigger_orientation(2.40, threshold_deg=35.0) is True
