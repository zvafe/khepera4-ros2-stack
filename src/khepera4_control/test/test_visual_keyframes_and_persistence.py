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
E2E Test Suite for Keyframe System and Map Persistence.

4-Tier Opaque-Box Hierarchy:
- Tier 1: Feature Coverage (Displacement Trigger, Yaw Trigger, Cliff Hazard Trigger,
          Manual Trigger, Keyframe PNG Storage, JSON Manifest Schema,
          Composite Map Export, Clear Map Reset)
- Tier 2: Boundary & Corner Cases (Exact Thresholds 0.349m vs 0.351m, 34.9 deg vs 35.1 deg,
          Angle Wrap-Around at +/- pi, Hazard Cooldown Suppression,
          Empty Keyframe List, Missing Directory on Clear)
- Tier 3: Pairwise Combinations (Keyframe Trigger During Active Warping,
          Cliff Alert During Teleop, Clear Followed by Save)
- Tier 4: Real-World Scenarios (Hazard Inspection & Landmark Pinning Mission,
          Save-Clear-Reload Full Lifecycle)
- Tier 5: ROS 2 Service Registration & Node Integration

Authoritative Specifications:
- ORIGINAL_REQUEST.md §R2, §R4
- PROJECT.md §Feature Inventory (F3, F4)
- survey_specs.md §2 (Keyframe Schema), §3 (Clear Behavior), §4 (Composite Blending)
"""

import json
import math
import os
import shutil
import tempfile
import time

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


# ==============================================================================
# Fixtures
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
def temp_map_dir():
    """Create and cleanly destroy a temporary directory for map and keyframe tests."""
    td = tempfile.mkdtemp()
    kf_dir = os.path.join(td, 'keyframes')
    yield td, kf_dir
    shutil.rmtree(td, ignore_errors=True)


# ==============================================================================
# Tier 1: Feature Coverage (F3 & F4)
# ==============================================================================

class TestTier1KeyframeAndPersistenceFeatures:
    """Tier 1: Core behavior of Keyframe Engine (F3) and Map Persistence (F4)."""

    def test_f3_displacement_trigger_exceeds_threshold(self, temp_map_dir):
        """F3: Advancing by >= 0.35m triggers keyframe capture."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        # First position (0, 0)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Displaced by 0.36m (> 0.35m)
        assert km.should_trigger_displacement(0.36, 0.0, threshold=0.35) is True

    def test_f3_displacement_trigger_below_threshold(self, temp_map_dir):
        """F3: Moving less than 0.35m does NOT trigger keyframe."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Displaced by 0.30m (< 0.35m)
        assert km.should_trigger_displacement(0.30, 0.0, threshold=0.35) is False

    def test_f3_orientation_trigger_exceeds_threshold(self, temp_map_dir):
        """F3: Heading rotation by >= 35 deg triggers keyframe."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Rotated by 36 deg (0.628 rad)
        assert km.should_trigger_orientation(math.radians(36.0), threshold_deg=35.0) is True

    def test_f3_orientation_trigger_below_threshold(self, temp_map_dir):
        """F3: Heading rotation of < 35 deg does NOT trigger keyframe."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Rotated by 25 deg (< 35 deg)
        assert km.should_trigger_orientation(math.radians(25.0), threshold_deg=35.0) is False

    def test_f3_cliff_hazard_trigger_alert(self, temp_map_dir):
        """F3: Event /cliff_alert triggers emergency snapshot with 'cliff_hazard' tag."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        entry = km.record_keyframe(
            img, 1.2, 0.5, 0.0, trigger_type='cliff_hazard', cliff_alert=True
        )
        assert entry['trigger'] == 'cliff_hazard'
        assert entry['telemetry']['cliff_alert'] is True
        assert os.path.exists(os.path.join(kf_dir, entry['filename']))

    def test_f3_user_manual_pin_trigger(self, temp_map_dir):
        """F3: User manual pin records snapshot with 'user_manual' tag."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        entry = km.record_keyframe(img, 0.5, -0.5, 1.57, trigger_type='user_manual')
        assert entry['trigger'] == 'user_manual'
        assert entry['id'] == 1

    def test_f3_user_manual_pin_sequential_ids(self, temp_map_dir):
        """F3: Consecutive manual pins increment ID sequentially (1, 2, 3)."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        p1 = km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='user_manual')
        p2 = km.record_keyframe(img, 0.1, 0.1, 0.0, trigger_type='user_manual')
        p3 = km.record_keyframe(img, 0.2, 0.2, 0.0, trigger_type='user_manual')

        assert p1['id'] == 1
        assert p2['id'] == 2
        assert p3['id'] == 3
        assert len(km.keyframes) == 3

    def test_f3_manifest_top_level_required_keys(self, temp_map_dir):
        """F3: Manifest JSON contains map_frame, total_keyframes, last_updated, and keyframes."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        assert manifest['map_frame'] == 'map'
        assert manifest['total_keyframes'] == 0
        assert 'last_updated' in manifest
        assert isinstance(manifest['keyframes'], list)

    def test_f4_save_map_exports_composite_png(self, temp_map_dir):
        """F4: Composite map export generates valid maps/khepera_visual_map.png."""
        td, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        # Record a sample keyframe
        img = np.full((480, 640, 3), 100, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Build composite map
        composite_path = os.path.join(td, 'khepera_visual_map.png')
        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)
        path = [(0.0, 0.0), (0.5, 0.5)]

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, path, km.keyframes
        )

        assert os.path.exists(composite_path)
        with Image.open(composite_path) as im:
            assert im.size == (200, 200)
            assert im.mode == 'RGB'

    def test_f4_clear_map_purges_keyframes_and_manifest(self, temp_map_dir):
        """F4: Clear map purges images on disk and resets manifest to empty array."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')
        km.record_keyframe(img, 0.4, 0.0, 0.0, trigger_type='displacement')

        assert len(km.keyframes) == 2
        pngs_before = [f for f in os.listdir(kf_dir) if f.endswith('.png')]
        assert len(pngs_before) == 2

        # Trigger clear
        km.clear()
        assert len(km.keyframes) == 0
        pngs_after = [f for f in os.listdir(kf_dir) if f.endswith('.png')]
        assert len(pngs_after) == 0

        # Read manifest: must be empty
        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        assert manifest['total_keyframes'] == 0
        assert manifest['keyframes'] == []


# ==============================================================================
# Tier 2: Boundary & Corner Cases (F3 & F4)
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary conditions for keyframe triggering, cooldowns, and schema."""

    def test_f3_displacement_boundary_349_vs_351(self, temp_map_dir):
        """Boundary: Exactly 0.349m (no trigger) vs 0.351m (triggers)."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        assert km.should_trigger_displacement(0.349, 0.0, threshold=0.35) is False
        assert km.should_trigger_displacement(0.351, 0.0, threshold=0.35) is True

    def test_f3_orientation_boundary_349_vs_351(self, temp_map_dir):
        """Boundary: Exactly 34.9 deg (no trigger) vs 35.1 deg (triggers)."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        assert km.should_trigger_orientation(math.radians(34.9), threshold_deg=35.0) is False
        assert km.should_trigger_orientation(math.radians(35.1), threshold_deg=35.0) is True

    def test_f3_orientation_angle_wrap_around_boundary(self, temp_map_dir):
        """Boundary: Crossing pi boundary (+3.10 rad to -3.10 rad) has delta=0.08 rad."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 3.10, trigger_type='init')

        # -3.10 rad: true angular distance is 2 * (pi - 3.10) ~ 0.083 rad (~4.7 deg < 35 deg)
        assert km.should_trigger_orientation(-3.10, threshold_deg=35.0) is False

    def test_f3_cliff_hazard_cooldown_suppression(self, temp_map_dir):
        """Boundary: Second cliff alert within 0.8s and 0.02m is suppressed by cooldown."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        t0 = 100.0
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 1.0, 0.0, 0.0, trigger_type='cliff_hazard', current_time=t0)

        # 0.8s later at distance 0.02m: suppressed
        assert km.should_trigger_cliff(1.02, 0.0, current_time=t0 + 0.8) is False

    def test_f3_cliff_hazard_cooldown_expiration(self, temp_map_dir):
        """Boundary: Second cliff alert after 1.2s and 0.06m passes cooldown."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        t0 = 100.0
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, 1.0, 0.0, 0.0, trigger_type='cliff_hazard', current_time=t0)

        # 1.2s later at distance 0.06m: permitted
        assert km.should_trigger_cliff(1.06, 0.0, current_time=t0 + 1.2) is True

    def test_f4_save_map_when_no_keyframes_or_images(self, temp_map_dir):
        """Corner: /save_map called with 0 keyframes succeeds and exports base map."""
        td, _ = temp_map_dir
        composite_path = os.path.join(td, 'khepera_visual_map.png')
        v_grid = np.full((200, 200, 3), 150, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        # 0 keyframes, empty path
        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, robot_path=[], keyframes=[]
        )
        assert os.path.exists(composite_path)

    def test_f4_clear_map_when_directory_does_not_exist(self):
        """Corner: /clear_map called before keyframe_dir created handles gracefully."""
        td = tempfile.mkdtemp()
        try:
            non_existent_kf = os.path.join(td, 'does_not_exist', 'keyframes')
            km = KeyframeManager(non_existent_kf)
            km.clear()  # Must not crash
            assert km.next_id == 1
        finally:
            shutil.rmtree(td, ignore_errors=True)


# ==============================================================================
# Tier 1 Schema & File Integrity (F3 & F4)
# ==============================================================================

class TestTier1SchemaAndFileIntegrity:
    """Tier 1: Validation of JSON manifest schema and lossless PNG files."""

    def test_f3_keyframe_filename_zero_padded_sequence(self, temp_map_dir):
        """Schema: Sequential filenames keyframe_0001.png, keyframe_0002.png."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        e1 = km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')
        e2 = km.record_keyframe(img, 0.4, 0.0, 0.0, trigger_type='displacement')

        assert e1['filename'] == 'keyframe_0001.png'
        assert e2['filename'] == 'keyframe_0002.png'

    def test_f3_manifest_schema_pose_fields_complete(self, temp_map_dir):
        """Schema: Manifest pose contains x, y, z, yaw, yaw_deg, pitch_deg, roll_deg."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        entry = km.record_keyframe(img, 1.25, -0.75, math.pi / 4.0, trigger_type='displacement')
        pose = entry['pose']

        assert 'x' in pose and pose['x'] == pytest.approx(1.25)
        assert 'y' in pose and pose['y'] == pytest.approx(-0.75)
        assert 'z' in pose and pose['z'] == 0.0
        assert 'yaw' in pose and pose['yaw'] == pytest.approx(math.pi / 4.0, abs=1e-3)
        assert 'yaw_deg' in pose and pose['yaw_deg'] == pytest.approx(45.0, abs=0.1)
        assert 'pitch_deg' in pose
        assert 'roll_deg' in pose

    def test_f3_manifest_schema_telemetry_fields_complete(self, temp_map_dir):
        """Schema: Manifest telemetry contains linear_velocity, angular_velocity, cliff_alert."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        entry = km.record_keyframe(
            img, 0.0, 0.0, 0.0, trigger_type='user_manual',
            linear_vel=0.12, angular_vel=0.05, cliff_alert=False
        )
        tel = entry['telemetry']
        assert tel['linear_velocity'] == pytest.approx(0.12)
        assert tel['angular_velocity'] == pytest.approx(0.05)
        assert tel['cliff_alert'] is False

    def test_f3_manifest_schema_camera_metadata_complete(self, temp_map_dir):
        """Schema: Manifest camera_metadata contains width=640, height=480, hfov=60.0."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        entry = km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')
        cam = entry['camera_metadata']
        assert cam['width'] == 640
        assert cam['height'] == 480
        assert cam['hfov_deg'] == 60.0
        assert cam['optical_frame'] == 'camera_optical_frame'

    def test_f3_manifest_atomic_save_and_reload(self, temp_map_dir):
        """Integrity: Manifest is valid JSON readable after each write."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        for i in range(5):
            km.record_keyframe(img, float(i) * 0.4, 0.0, 0.0, trigger_type='displacement')

        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        assert manifest['total_keyframes'] == 5
        assert len(manifest['keyframes']) == 5
        assert manifest['keyframes'][-1]['id'] == 5

    def test_f4_composite_blending_layer_separation(self, temp_map_dir):
        """F4: Blended composite reflects obstacles, slopes, and cliffs distinctly."""
        td, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        composite_path = os.path.join(td, 'khepera_visual_map.png')

        # Neutral ground visual map
        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        # Set specific feature cells
        lo_grid[50, 50] = 2.0      # Obstacle at (50, 50)
        elev_grid[80, 80] = 10     # Slope at (80, 80)
        surf_grid[120, 120] = 100  # Cliff at (120, 120)

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, robot_path=[], keyframes=km.keyframes
        )

        with Image.open(composite_path) as im:
            arr = np.array(im)
            # Image is vertically flipped: row y_img = 199 - row
            # Obstacle cell (row 50 -> y_img 149)
            obs_pixel = arr[199 - 50, 50]
            assert obs_pixel[0] < 50 and obs_pixel[1] < 50  # Dark charcoal

            # Cliff cell (row 120 -> y_img 79)
            cliff_pixel = arr[199 - 120, 120]
            assert cliff_pixel[0] > 200 and cliff_pixel[1] < 50  # Vibrant red

    def test_f4_composite_map_multiple_keyframe_pins(self, temp_map_dir):
        """F4: Blended composite renders multiple keyframe pins without overlap distortion."""
        td, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        composite_path = os.path.join(td, 'khepera_visual_map.png')

        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        km.record_keyframe(img, -1.0, -1.0, 0.0, trigger_type='init')
        km.record_keyframe(img, 0.0, 0.0, 1.57, trigger_type='displacement')
        km.record_keyframe(img, 1.0, 1.0, 3.14, trigger_type='user_manual')

        v_grid = np.full((200, 200, 3), 180, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, robot_path=[], keyframes=km.keyframes
        )
        assert os.path.exists(composite_path)
        with Image.open(composite_path) as im:
            assert im.size == (200, 200)


# ==============================================================================
# Tier 3: Pairwise Combinations (F3 & F4)
# ==============================================================================

class TestTier3PairwiseCombinations:
    """Tier 3: Multi-feature interactions between Keyframes and Persistence."""

    def test_pairwise_cliff_alert_during_motion(self, temp_map_dir):
        """Pairwise: Cliff hazard alert received while robot moving at linear velocity."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        img = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Trigger cliff alert while v = 0.15 m/s
        entry = km.record_keyframe(
            img, 1.5, 0.2, 0.1, trigger_type='cliff_hazard',
            linear_vel=0.15, cliff_alert=True
        )
        assert entry['trigger'] == 'cliff_hazard'
        assert entry['telemetry']['linear_velocity'] == 0.15
        assert entry['telemetry']['cliff_alert'] is True

    def test_pairwise_clear_followed_by_immediate_save(self, temp_map_dir):
        """Pairwise: Clear map followed immediately by save map exports valid empty map."""
        td, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)
        composite_path = os.path.join(td, 'khepera_visual_map.png')

        # Add keyframe
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # Clear
        km.clear()

        # Save immediately
        v_grid = np.zeros((200, 200, 3), dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, robot_path=[], keyframes=km.keyframes
        )
        assert os.path.exists(composite_path)
        with open(km.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        assert manifest['total_keyframes'] == 0


# ==============================================================================
# Tier 4: Real-World Scenarios (F3 & F4)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end hazard inspection and save-clear-reload mission cycles."""

    def test_tier4_hazard_inspection_and_photo_pinning_scenario(self, temp_map_dir):
        """Scenario: Approach cliff, trigger hazard keyframe, and export composite."""
        td, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        # 1. Start at origin
        img = np.full((480, 640, 3), 150, dtype=np.uint8)
        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # 2. Travel along path toward cliff
        km.record_keyframe(img, 0.4, 0.0, 0.0, trigger_type='displacement')
        km.record_keyframe(img, 0.8, 0.0, 0.0, trigger_type='displacement')

        # 3. Detect cliff hazard at X = 1.0m
        t_hazard = time.time()
        km.record_keyframe(
            img, 1.0, 0.0, 0.0, trigger_type='cliff_hazard',
            current_time=t_hazard, cliff_alert=True
        )

        assert len(km.keyframes) == 4
        assert km.keyframes[-1]['trigger'] == 'cliff_hazard'

        # 4. Export composite visual map
        composite_path = os.path.join(td, 'khepera_visual_map.png')
        v_grid = np.full((200, 200, 3), 180, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid[100, 133] = 100  # Mark cliff on grid
        path = [(0.0, 0.0), (0.4, 0.0), (0.8, 0.0), (1.0, 0.0)]

        generate_composite_map(
            composite_path, v_grid, lo_grid, elev_grid,
            surf_grid, path, km.keyframes
        )

        assert os.path.exists(composite_path)

    def test_tier4_save_clear_reload_lifecycle(self, temp_map_dir):
        """Scenario: Full operational cycle of mapping, saving, clearing, and remap."""
        _, kf_dir = temp_map_dir
        km = KeyframeManager(kf_dir)

        # Mission 1: 3 keyframes
        img = np.full((480, 640, 3), 100, dtype=np.uint8)
        for i in range(3):
            km.record_keyframe(img, float(i) * 0.4, 0.0, 0.0, trigger_type='displacement')
        assert len(km.keyframes) == 3

        # Clear mission 1
        km.clear()
        assert len(km.keyframes) == 0
        assert len(os.listdir(kf_dir)) == 1  # Only manifest.json remains

        # Mission 2: Fresh keyframes start with ID #1
        e_fresh = km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')
        assert e_fresh['id'] == 1
        assert e_fresh['filename'] == 'keyframe_0001.png'


# ==============================================================================
# Tier 5: ROS 2 Service Registration & Node Integration
# ==============================================================================

class TestTier5VisualOrthomapperServices:
    """Validate ROS 2 service endpoint advertising on VisualOrthomapper node."""

    def test_visual_orthomapper_services_registered(self):
        """Verify node advertises /pin_keyframe, /record_keyframe, and /save_visual_map."""
        if not rclpy.ok():
            rclpy.init()
        node = VisualOrthomapper()
        try:
            srv_names = [s.srv_name for s in node.services]
            assert '/pin_keyframe' in srv_names
            assert '/record_keyframe' in srv_names
            assert '/save_visual_map' in srv_names
            assert '/clear_visual_map' in srv_names
        finally:
            node.destroy_node()

    def test_record_keyframe_service_trigger(self, temp_map_dir):
        """Invoke /record_keyframe service callback directly and verify capture."""
        _, kf_dir = temp_map_dir
        if not rclpy.ok():
            rclpy.init()
        node = VisualOrthomapper(
            parameter_overrides=[
                Parameter('keyframe_dir', Parameter.Type.STRING, kf_dir)
            ]
        )
        try:
            node.has_odom = True
            node.robot_x = 0.5
            node.robot_y = -0.5
            node.robot_yaw = 0.0
            node.latest_camera_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

            req = Trigger.Request()
            resp = Trigger.Response()
            resp = node.pin_keyframe_callback(req, resp)

            assert resp.success is True
            assert 'Pinned Keyframe #1' in resp.message
            assert node.keyframe_count == 1
            assert os.path.exists(os.path.join(kf_dir, 'keyframe_0001.png'))
        finally:
            node.destroy_node()

    def test_save_visual_map_service_trigger(self, temp_map_dir):
        """Invoke /save_visual_map service callback directly and verify composite export."""
        _, kf_dir = temp_map_dir
        if not rclpy.ok():
            rclpy.init()
        node = VisualOrthomapper(
            parameter_overrides=[
                Parameter('keyframe_dir', Parameter.Type.STRING, kf_dir)
            ]
        )
        try:
            node.has_odom = True
            node.robot_x = 0.0
            node.robot_y = 0.0
            node.robot_yaw = 0.0

            req = Trigger.Request()
            resp = Trigger.Response()
            resp = node.save_map_callback(req, resp)

            assert resp.success is True
            assert 'Composite visual map saved' in resp.message
            assert os.path.exists('maps/khepera_visual_map.png')
        finally:
            node.destroy_node()
            if os.path.exists('maps/khepera_visual_map.png'):
                try:
                    os.remove('maps/khepera_visual_map.png')
                except OSError:
                    pass
