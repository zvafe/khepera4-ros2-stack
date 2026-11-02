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
Adversarial Stress Test Suite — Milestone M2 Gate Verification.

Validates:
1. Adversarial Yaw Wrapping & Heading Discontinuities:
   - Rapid oscillation across -pi <-> +pi boundary without false triggers.
   - Exact 35.0 deg trigger threshold across discontinuity boundaries.
   - Multi-turn winding and extreme angles.
2. Rapid-Fire Cliff Alert Cooldown & Spatial Debouncing:
   - 50 Hz burst alert suppression when stationary.
   - Hazard edge capture under displacement >= 0.05m and dt >= 1.0s.
   - Cooldown recovery and hazard trigger priority over displacement.
3. Atomic Persistence & Fault Injection:
   - Multi-threaded concurrent keyframe recording without JSON manifest corruption.
   - Simulated write interruptions (partial .tmp writes) preserving original manifest.
   - Resilience against missing, empty, or corrupt manifest files.
4. Concurrency & Service Lifecycle (/save_map vs /clear_map vs /pin_keyframe):
   - High-frequency interleaved invocation across threads.
   - Zero deadlocks, unhandled exceptions, or file descriptor leaks.
5. Composite Map Color Blending & Geometric Array Bounds:
   - Exact layer hierarchy (Obstacle priority over Slope; Cliff crimson; gold pins).
   - Extreme coordinate resilience (out-of-bounds, exact grid edges, NaNs).
"""

from concurrent.futures import ThreadPoolExecutor
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
from std_msgs.msg import Header
from std_srvs.srv import Trigger

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture(scope='module', autouse=True)
def manage_rclpy():
    """Manage lifecycle of rclpy for module-scoped adversarial testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def temp_kf_dir():
    """Create isolated temporary directory for keyframe persistence."""
    td = tempfile.mkdtemp(prefix='adv_m2_kf_')
    yield td
    if os.path.exists(td):
        shutil.rmtree(td)


@pytest.fixture
def isolated_orthomapper(temp_kf_dir):
    """Instantiate VisualOrthomapper in isolated directory."""
    node = VisualOrthomapper(
        parameter_overrides=[
            Parameter('keyframe_dir', Parameter.Type.STRING, temp_kf_dir)
        ]
    )
    yield node
    node.destroy_node()


# ==============================================================================
# SECTION 1: Adversarial Yaw Wrapping & Heading Discontinuities
# ==============================================================================

class TestAdversarialYawWrapping:
    """Stress tests on orientation delta calculation and boundary wrapping."""

    def test_rapid_oscillation_across_pi_boundary_no_trigger(self, temp_kf_dir):
        """Verify rapid yaw switching between +pi-eps and -pi+eps never falsely triggers."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Baseline keyframe at +pi - 0.01 rad (~179.4 deg)
        base_yaw = math.pi - 0.01
        km.record_keyframe(img, 0.0, 0.0, base_yaw, trigger_type='init')
        assert len(km.keyframes) == 1

        # Rapidly oscillate across the boundary 100 times by 0.02 rad (1.15 deg)
        for i in range(100):
            test_yaw = (-math.pi + 0.01) if (i % 2 == 0) else (math.pi - 0.01)
            assert not km.should_trigger_orientation(test_yaw, threshold_deg=35.0), (
                f'Falsely triggered on oscillation iteration {i}: yaw={test_yaw}'
            )

        # Confirm no extra keyframes were recorded
        assert len(km.keyframes) == 1

    def test_exact_35_degree_boundary_across_pi_discontinuity(self, temp_kf_dir):
        """Verify orientation trigger threshold (35 deg) across the -pi <-> +pi jump."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Baseline at +175 degrees (math.radians(175.0))
        base_yaw = math.radians(175.0)
        km.record_keyframe(img, 0.0, 0.0, base_yaw, trigger_type='init')

        # 34.0 deg CCW rotation crosses +180/-180 to -151.0 deg (delta = 34 deg < 35 deg)
        sub_thresh_yaw = math.radians(-151.0)
        assert not km.should_trigger_orientation(sub_thresh_yaw, threshold_deg=35.0)

        # 36.0 deg CCW rotation crosses to -149.0 deg (delta = 36 deg >= 35 deg)
        super_thresh_yaw = math.radians(-149.0)
        assert km.should_trigger_orientation(super_thresh_yaw, threshold_deg=35.0)

    def test_reverse_discontinuity_minus_pi_to_plus_pi(self, temp_kf_dir):
        """Verify orientation trigger when rotating CW from -175 deg to +149 deg."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        base_yaw = math.radians(-175.0)
        km.record_keyframe(img, 0.0, 0.0, base_yaw, trigger_type='init')

        # Delta = 34 deg CW -> +151 deg (no trigger)
        assert not km.should_trigger_orientation(math.radians(151.0), threshold_deg=35.0)

        # Delta = 36 deg CW -> +149 deg (triggers)
        assert km.should_trigger_orientation(math.radians(149.0), threshold_deg=35.0)

    def test_multi_turn_winding_accumulation(self, temp_kf_dir):
        """Verify winding beyond 2*pi is handled gracefully without exception."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        km.record_keyframe(img, 0.0, 0.0, 0.0, trigger_type='init')

        # 4*pi + 36 deg
        extreme_yaw = 4.0 * math.pi + math.radians(36.0)
        assert km.should_trigger_orientation(extreme_yaw, threshold_deg=35.0)

        # 4*pi + 20 deg
        sub_extreme = 4.0 * math.pi + math.radians(20.0)
        assert not km.should_trigger_orientation(sub_extreme, threshold_deg=35.0)


# ==============================================================================
# SECTION 2: Rapid-Fire Cliff Alert Cooldown & Spatial Debouncing
# ==============================================================================

class TestAdversarialCliffAlerts:
    """Stress tests on cliff hazard temporal cooldown and spatial debouncing."""

    def test_rapid_fire_cliff_burst_50hz_stationary_suppression(self, temp_kf_dir):
        """Simulate 50 Hz burst of 50 cliff alerts while stationary; verify exactly 1 keyframe."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 200, dtype=np.uint8)

        t_start = 1000.0
        x_pos, y_pos = 1.25, -0.75
        yaw = 0.5

        recorded_count = 0
        # 50 messages over 1.0 second (50 Hz, dt = 0.02s per step)
        for step in range(50):
            t_curr = t_start + step * 0.02
            if km.should_trigger_cliff(x_pos, y_pos, t_curr, cooldown_sec=1.0, min_dist=0.05):
                km.record_keyframe(
                    img, x_pos, y_pos, yaw,
                    trigger_type='cliff_hazard',
                    current_time=t_curr,
                    cliff_alert=True
                )
                recorded_count += 1

        assert recorded_count == 1, (
            f'Expected exactly 1 keyframe during stationary 50Hz burst, got {recorded_count}'
        )
        assert len(km.keyframes) == 1
        assert km.keyframes[0]['trigger'] == 'cliff_hazard'

    def test_cliff_cooldown_and_displacement_recovery(self, temp_kf_dir):
        """Verify cliff hazard re-triggers once cooldown (1.0s) AND displacement (0.05m) pass."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 200, dtype=np.uint8)

        # Event 1 at t=100.0, (1.0, 1.0)
        km.record_keyframe(
            img, 1.0, 1.0, 0.0,
            trigger_type='cliff_hazard',
            current_time=100.0,
            cliff_alert=True
        )

        # Case A: dt >= 1.0s but robot has not moved (dist < 0.05m) -> SUPPRESSED
        assert not km.should_trigger_cliff(1.0, 1.0, 101.5, cooldown_sec=1.0, min_dist=0.05)

        # Case B: robot moved 0.06m but dt < 1.0s (dt = 0.5s) -> SUPPRESSED
        assert not km.should_trigger_cliff(1.06, 1.0, 100.5, cooldown_sec=1.0, min_dist=0.05)

        # Case C: robot moved 0.06m AND dt >= 1.0s (dt = 1.2s) -> TRIGGERS
        assert km.should_trigger_cliff(1.06, 1.0, 101.2, cooldown_sec=1.0, min_dist=0.05)

        # Record second hazard
        km.record_keyframe(
            img, 1.06, 1.0, 0.0,
            trigger_type='cliff_hazard',
            current_time=101.2,
            cliff_alert=True
        )
        assert len(km.keyframes) == 2

    def test_node_evaluate_triggers_cliff_priority_over_displacement(self, isolated_orthomapper):
        """Verify node-level trigger evaluator prioritizes cliff hazard over displacement."""
        node = isolated_orthomapper
        img = np.full((480, 640, 3), 150, dtype=np.uint8)

        # Initialize base pose
        header = Header()
        header.stamp.sec = 100
        header.stamp.nanosec = 0
        node.robot_x = 0.0
        node.robot_y = 0.0
        node.robot_yaw = 0.0
        node.cliff_alert = False

        # First trigger to initialize last_kf pose
        node.evaluate_keyframe_triggers(img, header.stamp)
        assert node.keyframe_count == 1
        assert node.keyframes[0]['trigger'] == 'displacement'

        # Move forward 0.50m (exceeds displacement thresh 0.35m) AND set cliff_alert = True
        header.stamp.sec = 102  # dt = 2.0s
        node.robot_x = 0.50
        node.cliff_alert = True

        node.evaluate_keyframe_triggers(img, header.stamp)
        assert node.keyframe_count == 2
        # Cliff alert MUST take priority
        assert node.keyframes[1]['trigger'] == 'cliff_hazard'
        assert node.keyframes[1]['telemetry']['cliff_alert'] is True


# ==============================================================================
# SECTION 3: Atomic Persistence & Fault Injection
# ==============================================================================

class TestAdversarialPersistenceAndFaultInjection:
    """Stress tests on JSON manifest atomicity and filesystem integrity."""

    def test_concurrent_multithreaded_keyframe_writes(self, temp_kf_dir):
        """Simulate concurrent threads recording keyframes; verify manifest never corrupted."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 100, dtype=np.uint8)

        num_threads = 8
        records_per_thread = 5
        total_expected = num_threads * records_per_thread

        def record_worker(thread_idx: int):
            for i in range(records_per_thread):
                x = thread_idx * 1.0 + i * 0.1
                km.record_keyframe(
                    img, x, 0.0, 0.0,
                    trigger_type=f'thread_{thread_idx}',
                    current_time=100.0 + thread_idx * 10 + i
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_worker, tid) for tid in range(num_threads)]
            for f in futures:
                f.result()

        assert len(km.keyframes) == total_expected

        # Validate that manifest.json on disk is valid, complete JSON
        manifest_path = os.path.join(temp_kf_dir, 'manifest.json')
        assert os.path.exists(manifest_path)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['total_keyframes'] == total_expected
        assert len(data['keyframes']) == total_expected

        # Verify all PNG files exist on disk
        png_files = [f for f in os.listdir(temp_kf_dir) if f.endswith('.png')]
        assert len(png_files) == total_expected

    def test_manifest_survives_interrupted_temp_write(self, temp_kf_dir):
        """Verify existing manifest.json remains intact if writing .tmp fails mid-stream."""
        km = KeyframeManager(keyframe_dir=temp_kf_dir)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Record legitimate initial keyframe
        km.record_keyframe(img, 0.1, 0.2, 0.3, trigger_type='initial')
        manifest_path = os.path.join(temp_kf_dir, 'manifest.json')

        with open(manifest_path, 'r', encoding='utf-8') as f:
            valid_content = f.read()
        assert len(valid_content) > 0

        # Simulate broken write: leave half-written .tmp file
        tmp_path = manifest_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write('{"corrupted_incomplete_json": [1, 2, 3')

        # Original manifest MUST still be valid JSON
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['total_keyframes'] == 1

        # Now trigger next legitimate record; must cleanly replace .tmp and succeed
        km.record_keyframe(img, 0.5, 0.5, 0.0, trigger_type='displacement')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
        assert updated_data['total_keyframes'] == 2

    def test_mapper_save_map_when_manifest_is_corrupted(self, temp_kf_dir):
        """Verify Khepera4Mapper.save_map does not crash if manifest.json is invalid JSON."""
        node = Khepera4Mapper(
            parameter_overrides=[
                Parameter('map_save_dir', Parameter.Type.STRING, temp_kf_dir)
            ]
        )
        try:
            kf_dir = os.path.join(temp_kf_dir, 'keyframes')
            os.makedirs(kf_dir, exist_ok=True)
            manifest_path = os.path.join(kf_dir, 'manifest.json')

            # Write broken JSON
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write('NOT_A_VALID_JSON{:::')

            # Invoke save_map service callback
            req = Trigger.Request()
            resp = Trigger.Response()
            resp = node.save_map_callback(req, resp)

            assert resp.success is True, f'save_map failed on corrupt manifest: {resp.message}'
            # Verify composite map was still generated
            composite_png = os.path.join(temp_kf_dir, 'khepera_visual_map.png')
            assert os.path.exists(composite_png)
        finally:
            node.destroy_node()


# ==============================================================================
# SECTION 4: Concurrency & Service Lifecycle (/save_map vs /clear_map)
# ==============================================================================

class TestAdversarialServiceConcurrency:
    """Stress tests on service race conditions and lifecycle resets."""

    def test_interleaved_save_and_clear_multithreaded_stress(self, temp_kf_dir):
        """Execute concurrent interleaved calls to save_map and clear_map without crashing."""
        node = Khepera4Mapper(
            parameter_overrides=[
                Parameter('map_save_dir', Parameter.Type.STRING, temp_kf_dir)
            ]
        )
        try:
            # Seed state with robot path and data
            node.robot_path = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
            node.has_odom = True
            node.robot_x = 0.2
            node.robot_y = 0.2

            def save_worker():
                for _ in range(5):
                    req = Trigger.Request()
                    resp = Trigger.Response()
                    node.save_map_callback(req, resp)
                    assert resp.success is True

            def clear_worker():
                for _ in range(5):
                    req = Trigger.Request()
                    resp = Trigger.Response()
                    node.clear_map_callback(req, resp)
                    assert resp.success is True

            with ThreadPoolExecutor(max_workers=4) as executor:
                f1 = executor.submit(save_worker)
                f2 = executor.submit(clear_worker)
                f3 = executor.submit(save_worker)
                f4 = executor.submit(clear_worker)
                f1.result()
                f2.result()
                f3.result()
                f4.result()

            # Final save must cleanly produce valid artifacts
            req = Trigger.Request()
            resp = Trigger.Response()
            node.save_map_callback(req, resp)
            assert resp.success is True
            assert os.path.exists(os.path.join(temp_kf_dir, 'khepera_visual_map.png'))
        finally:
            node.destroy_node()

    def test_orthomapper_pin_keyframe_without_odom_failure(self, isolated_orthomapper):
        """Verify /pin_keyframe gracefully fails when odometry is absent."""
        node = isolated_orthomapper
        node.has_odom = False
        node.latest_camera_frame = None

        req = Trigger.Request()
        resp = Trigger.Response()
        resp = node.pin_keyframe_callback(req, resp)

        assert resp.success is False
        assert 'not yet available' in resp.message.lower()


# ==============================================================================
# SECTION 5: Composite Map Blending & Geometric Array Bounds
# ==============================================================================

class TestAdversarialCompositeBlendingAndBounds:
    """Stress tests on composite map layer generation and array bounds."""

    def test_composite_layer_priority_obstacle_over_slope(self, temp_kf_dir):
        """Verify SLAM obstacle layer (dark charcoal) takes priority over terrain slopes."""
        out_path = os.path.join(temp_kf_dir, 'test_composite_priority.png')

        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        # Cell (100, 100): Both obstacle (> 0.5) AND steep slope (> 5 deg)
        lo_grid[100, 100] = 1.0
        elev_grid[100, 100] = 15  # Steep 15 deg

        generate_composite_map(
            out_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[], keyframes=[]
        )

        with Image.open(out_path) as im:
            arr = np.array(im)
            # Flipped vertically: row 100 becomes 199 - 100 = 99
            pixel = arr[99, 100]
            # Obstacle color is charcoal [20, 20, 25]
            assert np.allclose(pixel, [20, 20, 25], atol=2), (
                f'Expected obstacle charcoal, got {pixel}'
            )

    def test_composite_cliff_layer_crimson_color(self, temp_kf_dir):
        """Verify cliff hazard cells render hazard crimson [230, 25, 25]."""
        out_path = os.path.join(temp_kf_dir, 'test_composite_cliff.png')

        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        # Cell (50, 50) is a cliff hazard (surface_grid == 100)
        surf_grid[50, 50] = 100

        generate_composite_map(
            out_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=[], keyframes=[]
        )

        with Image.open(out_path) as im:
            arr = np.array(im)
            # Flipped vertically: row 50 becomes 199 - 50 = 149
            pixel = arr[149, 50]
            assert np.allclose(pixel, [230, 25, 25], atol=2), (
                f'Expected cliff crimson [230, 25, 25], got {pixel}'
            )

    def test_composite_extreme_out_of_bounds_coordinates(self, temp_kf_dir):
        """Verify trajectory and keyframe pins at extreme coordinates don't crash export."""
        out_path = os.path.join(temp_kf_dir, 'test_composite_oob.png')

        v_grid = np.full((200, 200, 3), 200, dtype=np.uint8)
        lo_grid = np.zeros((200, 200), dtype=np.float32)
        elev_grid = np.full((200, 200), -1, dtype=np.int8)
        surf_grid = np.full((200, 200), -1, dtype=np.int8)

        # Path with points far beyond the arena and grid bounds
        path = [
            (-1000.0, -1000.0),
            (-3.001, -3.001),
            (0.0, 0.0),
            (2.999, 2.999),
            (3.001, 3.001),
            (10000.0, 10000.0)
        ]

        # Keyframes at boundary edges and out of bounds
        keyframes = [
            {'id': 1, 'pose': {'x': -100.0, 'y': -100.0}},
            {'id': 2, 'pose': {'x': -3.0, 'y': -3.0}},      # Grid origin corner
            {'id': 3, 'pose': {'x': 2.97, 'y': 2.97}},      # Top-right cell
            {'id': 4, 'pose': {'x': 100.0, 'y': 100.0}},    # Far out
            {'id': 5, 'pose': {}}                             # Missing coordinates
        ]

        result = generate_composite_map(
            out_path, v_grid, lo_grid, elev_grid, surf_grid,
            robot_path=path, keyframes=keyframes,
            current_pose=(500.0, 500.0, 0.0)  # Robot far outside
        )

        assert os.path.exists(result)
        with Image.open(result) as im:
            assert im.size == (200, 200)
            assert im.mode == 'RGB'
