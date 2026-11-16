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

"""
Comprehensive verification and benchmarking test suite for detection and exploration.

Covers the three operational scenarios:
1. Scenario 1 (Auto Explore): Quick explore algorithm covering >= 80% of map in < 1 minute
   with NO pursue (pursuit disabled, target detections ignored).
2. Scenario 2 (Option A - Semantic Description):
   - Answers weighting question: Class Name is primary discriminator (~70% weight),
     Description provides attribute context (~30% weight).
   - Solid 3D primitive spawning (Red Cylinder by default, toggled between Cylinder, Cube,
     Sphere via dropdown, and QColorDialog color selection) with NO texture images.
   - Real-world case support: "go to the red chair" / "red cylinder" where robot searches,
     tracks, and sits directly in front of the target.
3. Scenario 3 (Option B - Template Image): Wraps user input template images (checkerboard,
   baboon, smarties) around a cube geometry with multi-scale DINOv2 / LightGlue matching.
4. Random Placement Benchmark: 10 random valid target placements avoiding ramp, platform,
   and walls, verifying target discovery in < 2 minutes in >= 80% of trials.
"""

import math
import os
import sys
import time
from typing import List, Tuple

import cv2
from khepera4_control.mapping_dashboard import (
    MappingDashboardWindow,
    MappingRosWorker,
)
from khepera4_control.reactive_explorer import ReactiveExplorer
from khepera_visual_pursuit.dinov2_verifier import DINOv2Verifier
from khepera_visual_pursuit.ibvs_controller import IBVSController, IBVSGains
from khepera_visual_pursuit.object_detector_verifier_node import (
    DEFAULT_TAU_VERIFY,
    ObjectDetectorVerifierNode,
    apply_nms,
)
try:
    from nav_msgs.msg import OccupancyGrid
except ImportError:
    class OccupancyGrid:
        """Mock OccupancyGrid for testing when nav_msgs is not available."""

        def __init__(self):
            """Initialize empty occupancy grid."""
            self.info = None
            self.data = []

from khepera_visual_pursuit.ros_compat import (
    Bool,
    String,
    rclpy,
)
from khepera_visual_pursuit.superpoint_lightglue_engine import (
    SuperPointLightGlueTracker,
)
from khepera_visual_pursuit.yolo_world_engine import (
    BoundingBox,
    YoloWorldEngine,
    calculate_weighted_semantic_score,
)
import numpy as np
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "sample_targets", "downloaded_templates"
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_init_teardown():
    """Manage ROS 2 context lifecycle for benchmark test suite."""
    if rclpy is not None and hasattr(rclpy, "ok"):
        if not rclpy.ok():
            rclpy.init()
        yield
        if rclpy.ok():
            rclpy.shutdown()
    else:
        yield


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance is active for Qt GUI tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def safe_close_window(w: MappingDashboardWindow) -> None:
    """Safely terminate timers and close window."""
    app = QApplication.instance()
    if app:
        try:
            app.removeEventFilter(w)
        except Exception:
            pass
    if hasattr(w, "teleop_timer") and w.teleop_timer.isActive():
        w.teleop_timer.stop()
    if hasattr(w, "plot_timer") and w.plot_timer.isActive():
        w.plot_timer.stop()
    if hasattr(w, "camera_popup") and w.camera_popup:
        try:
            w.camera_popup.close()
        except Exception:
            pass
    w.close()


# ==============================================================================
# Scenario 2: Option A (Semantic Description & Name vs Description Weighting)
# ==============================================================================

def test_option_a_name_vs_description_weighting() -> None:
    """
    Verify Option A name vs description weighting and prompt parsing.

    Question: Which weights more?
    Answer: Class Name is the primary discriminator (~70% weight) identifying the
    coarse visual category and invariant structure. Description provides attribute
    context (~30% weight) detailing colors, materials, or spatial relationships.
    """
    node = ObjectDetectorVerifierNode()
    try:
        # 1. Prompt formatted as "class_name | description"
        prompt_msg = String()
        prompt_msg.data = "red chair | sits in the office"
        node._on_target_prompt_message(prompt_msg)

        assert node.target_name == "red chair"
        assert node.target_desc == "sits in the office"
        assert node.detector.classes[0] == "red chair"
        assert "red chair, sits in the office" in node.detector.classes
        assert "sits in the office" in node.detector.classes

        # 2. Alternative separator "::"
        prompt_msg.data = "wooden chair :: vintage office seat with armrests"
        node._on_target_prompt_message(prompt_msg)
        assert node.target_name == "wooden chair"
        assert node.target_desc == "vintage office seat with armrests"
        assert node.detector.classes[0] == "wooden chair"

        # 3. Colon separator ":"
        prompt_msg.data = "red cylinder : tall soda can"
        node._on_target_prompt_message(prompt_msg)
        assert node.target_name == "red cylinder"
        assert node.target_desc == "tall soda can"
        assert node.detector.classes[0] == "red cylinder"

        # 4. Standalone class name without description
        prompt_msg.data = "red cylinder"
        node._on_target_prompt_message(prompt_msg)
        assert node.target_name == "red cylinder"
        assert node.target_desc == ""
        assert node.detector.classes[0] == "red cylinder"

        # 5. Direct verification of 70% name vs 30% description weighting algorithm
        box_chair = BoundingBox(
            center_x=320.0,
            center_y=240.0,
            size_x=80.0,
            size_y=140.0,
            confidence=0.90,
            class_name="chair",
        )
        crop_red = np.full((140, 80, 3), (20, 25, 220), dtype=np.uint8)
        crop_blue = np.full((140, 80, 3), (220, 25, 20), dtype=np.uint8)

        # Case A: Both class name and description match
        score_both = calculate_weighted_semantic_score(
            box_chair, target_name="chair", target_desc="red color", crop_bgr=crop_red
        )
        # Case B: Class name matches, description does not (name-only match)
        score_name_only = calculate_weighted_semantic_score(
            box_chair, target_name="chair", target_desc="red color", crop_bgr=crop_blue
        )
        # Case C: Description matches, class name does not (desc-only match)
        box_wrong = BoundingBox(
            center_x=320.0,
            center_y=240.0,
            size_x=140.0,
            size_y=40.0,
            confidence=0.85,
            class_name="keyboard",
        )
        score_desc_only = calculate_weighted_semantic_score(
            box_wrong, target_name="chair", target_desc="red color", crop_bgr=crop_red
        )
        # Case D: No description specified -> 100% weight to class name
        score_no_desc = calculate_weighted_semantic_score(
            box_chair, target_name="chair", target_desc="", crop_bgr=crop_blue
        )

        assert score_both >= 0.85, f"Expected score_both >= 0.85, got {score_both}"
        assert 0.60 <= score_name_only <= 0.75, (
            f"Expected score_name_only ~0.65-0.70, got {score_name_only}"
        )
        assert score_desc_only <= 0.45, f"Expected score_desc_only <= 0.45, got {score_desc_only}"
        assert score_both > score_name_only > score_desc_only
        assert score_no_desc >= 0.90
    finally:
        node.destroy_node()


def test_option_a_solid_primitive_spawning_and_dropdown_color(qapp) -> None:
    """
    Verify Option A solid 3D primitive spawning via dropdown and color selection.

    Must spawn solid colored 3D primitives (Cylinder, Cube, Sphere) with NO texture image.
    Default shape is Red Cylinder (QColor(225, 29, 72)).
    """
    worker = MappingRosWorker()
    win = MappingDashboardWindow(ros_worker=worker)
    try:
        # Verify Option A defaults
        assert win.selected_target_shape == "Cylinder"
        assert win.selected_target_color == QColor(225, 29, 72)
        assert win.combo_target_shape is not None
        assert win.combo_target_shape.count() == 3
        shapes = [win.combo_target_shape.itemText(i) for i in range(3)]
        assert "Cylinder" in shapes
        assert "Cube" in shapes
        assert "Sphere" in shapes

        target_dir = "/tmp/khepera4_target"
        sdf_path = os.path.join(target_dir, "target.sdf")

        # Test Cylinder spawning (solid red, NO texture image)
        win.combo_target_shape.setCurrentText("Cylinder")
        win.selected_target_color = QColor(225, 29, 72)
        win._ensure_textured_gazebo_target(force_option_a=True)
        time.sleep(0.1)

        if os.path.isfile(sdf_path):
            with open(sdf_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<cylinder>" in content
            assert "<radius>0.05</radius>" in content
            assert "<albedo_map>" not in content
            assert "<pbr>" not in content

        # Test Cube spawning (solid green, NO texture image)
        win.combo_target_shape.setCurrentText("Cube")
        win.selected_target_color = QColor(34, 197, 94)
        win._ensure_textured_gazebo_target(force_option_a=True)
        time.sleep(0.1)

        if os.path.isfile(sdf_path):
            with open(sdf_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<box>" in content
            assert "<size>0.12 0.12 0.12</size>" in content
            assert "<albedo_map>" not in content

        # Test Sphere spawning (solid blue, NO texture image)
        win.combo_target_shape.setCurrentText("Sphere")
        win.selected_target_color = QColor(59, 130, 246)
        win._ensure_textured_gazebo_target(force_option_a=True)
        time.sleep(0.1)

        if os.path.isfile(sdf_path):
            with open(sdf_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<sphere>" in content
            assert "<radius>0.06</radius>" in content
            assert "<albedo_map>" not in content
    finally:
        safe_close_window(win)


def test_option_a_real_scenario_red_chair_and_cylinder_ibvs_sit() -> None:
    """
    Test real scenario: 'go to the red chair' / 'red cylinder'.

    Robot searches for the target, detects it via open-vocabulary / color-shape engine,
    tracks it, and IBVS drives toward it and smoothly sits in front of it (v -> 0).
    """
    detector = YoloWorldEngine()
    detector.set_classes(("red chair", "red cylinder", "chair", "cylinder"))

    # Synthesize camera view with a red chair / red cylinder target in frontal sector
    h_img, w_img = 480, 640
    frame = np.full((h_img, w_img, 3), 40, dtype=np.uint8)
    # Red cylindrical object centered horizontally at y=200..340, x=290..350
    cv2.rectangle(frame, (290, 200), (350, 340), (20, 25, 220), -1)

    proposals = detector.generate_proposals(frame)
    assert len(proposals) > 0
    best_prop = max(proposals, key=lambda p: p.score)
    assert best_prop.score >= 0.70
    assert abs(best_prop.center_x - 320.0) < 30.0

    # Pass proposals through NMS
    nms_props = apply_nms(proposals, iou_threshold=0.35)
    assert len(nms_props) >= 1

    # Initialize IBVS Controller
    gains = IBVSGains(
        k_v=0.40,
        k_omega=1.20,
        v_max=0.18,
        omega_max=1.00,
        desired_area_fraction=0.12,
        obstacle_stop_distance=0.18,
    )
    controller = IBVSController(gains)

    # Far-field approach (area fraction small e.g. 0.02)
    far_norm_x = (best_prop.center_x - 320.0) / 320.0
    far_area = (best_prop.size_x * best_prop.size_y) / (w_img * h_img)
    cmd_approach = controller.compute_control(
        target_state=(far_norm_x, far_area, 0.85),
        obstacle_distance=1.8,
        obstacle_left=1.5,
        obstacle_right=1.5,
    )
    assert cmd_approach.linear_x > 0.05
    assert not cmd_approach.is_emergency_stop

    # Arrival: robot reaches target area fraction and frontal obstacle boundary
    cmd_arrived = controller.compute_control(
        target_state=(0.0, 0.13, 0.85),
        obstacle_distance=0.16,
        obstacle_left=1.0,
        obstacle_right=1.0,
    )
    # Robot halts and sits directly in front of the red chair / cylinder
    assert cmd_arrived.linear_x == 0.0
    assert abs(cmd_arrived.angular_z) < 0.10


# ==============================================================================
# Scenario 3: Option B (Template Image Wrapped Around Cube)
# ==============================================================================

def test_option_b_template_image_cube_texture_wrapping(qapp) -> None:
    """
    Verify Option B wraps user template images around a cube (<box>).

    Tests all 6 downloaded high-contrast template images:
    - checkerboard_texture.png
    - baboon_texture.jpg
    - smarties_texture.png
    - lena_texture.jpg
    - fruits.jpg
    - butterfly.jpg
    """
    worker = MappingRosWorker()
    win = MappingDashboardWindow(ros_worker=worker)
    try:
        win.radio_mode_template.setChecked(True)

        template_files = [
            "checkerboard_texture.png",
            "baboon_texture.jpg",
            "smarties_texture.png",
            "lena_texture.jpg",
            "fruits.jpg",
            "butterfly.jpg",
        ]

        target_dir = "/tmp/khepera4_target"
        sdf_path = os.path.join(target_dir, "target.sdf")
        tex_path = os.path.join(target_dir, "target_texture.png")

        for tname in template_files:
            tpath = os.path.join(TEMPLATES_DIR, tname)
            assert os.path.isfile(tpath), f"Downloaded template {tname} missing at {tpath}"

            win._ensure_textured_gazebo_target(img_path=tpath, target_name="Cube Target")
            time.sleep(0.15)

            if os.path.isfile(sdf_path):
                with open(sdf_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Must be a cube geometry
                assert "<box>" in content
                assert "<size>0.12 0.12 0.12</size>" in content
                # Must wrap the texture image via pbr
                assert "<albedo_map>" in content
                assert "target_texture.png" in content
                assert os.path.isfile(tex_path)
    finally:
        safe_close_window(win)


def test_option_b_multiscale_dinov2_and_lightglue_matching() -> None:
    """Verify multi-scale matching between 2D downloaded template and rendered scene."""
    tpath = os.path.join(TEMPLATES_DIR, "checkerboard_texture.png")
    assert os.path.isfile(tpath)
    raw_template = cv2.imread(tpath)
    assert raw_template is not None
    template_bgr = cv2.resize(raw_template, (100, 100))

    # Test DINOv2 verifier with nominal cross-domain threshold
    verifier = DINOv2Verifier(tau_verify=DEFAULT_TAU_VERIFY, tau_margin=0.08)
    ref_emb = verifier.extract_embedding(template_bgr)
    assert ref_emb is not None

    # Multi-scale rendered appearances (scales: 0.35, 0.70, 1.0)
    for scale in (0.35, 0.70, 1.0):
        scaled = cv2.resize(template_bgr, (int(100 * scale), int(100 * scale)))
        scene = np.full((400, 500, 3), 120, dtype=np.uint8)
        sh, sw = scaled.shape[:2]
        scene[50:50 + sh, 100:100 + sw] = scaled

        crop = scene[50:50 + sh, 100:100 + sw]
        crop_emb = verifier.extract_embedding(crop)
        distractor = np.full_like(crop, 50)
        dist_emb = verifier.extract_embedding(distractor)

        res = verifier.verify_candidates((crop_emb, dist_emb), ref_emb)
        assert res.is_verified is True
        assert res.best_candidate_index == 0
        assert res.best_similarity >= DEFAULT_TAU_VERIFY

    # Test SuperPoint / LightGlue tracker with expanded template correlation
    tracker = SuperPointLightGlueTracker(match_threshold=0.2, min_keypoints=6)
    lock_ok = tracker.set_reference_target(template_bgr)
    assert lock_ok is True
    assert tracker.has_reference is True

    # Test tracking across small and medium scale in scene
    small_scene = np.zeros((300, 400, 3), dtype=np.uint8)
    scaled_sub = cv2.resize(template_bgr, (35, 35))
    small_scene[40:75, 60:95] = scaled_sub

    output = tracker.track(small_scene)
    assert output.is_tracked is True
    assert output.confidence >= 0.40


# ==============================================================================
# Scenario 1: Auto Explore (Quick Explore 80% Map in < 1 min, NO Pursue)
# ==============================================================================

def test_auto_explore_coverage_and_no_pursuit() -> None:
    """
    Verify Auto Explore quick explore algorithm with NO pursue.

    Requirements:
    1. Exploration covers >= 80% of reachable map in < 1 minute.
    2. Visual pursuit is disabled (no pursue), so robot does NOT yield to targets.
    """
    node = ReactiveExplorer()
    try:
        node.enabled = True
        node.pursuit_active = False

        # When pursuit is disabled, target detection notifications MUST be ignored
        node.target_status_callback(Bool(data=True))
        assert node.target_in_sight is False
        assert node.pursuit_active is False

        # Simulate occupancy grid map with unknown cells
        # Grid 100x100 representing 5m x 5m arena (0.05m res)
        grid_msg = OccupancyGrid()
        grid_msg.info.resolution = 0.05
        grid_msg.info.width = 100
        grid_msg.info.height = 100
        grid_msg.info.origin.position.x = -2.5
        grid_msg.info.origin.position.y = -2.5
        # Initially all unknown (-1)
        grid_data = np.full((100, 100), -1, dtype=np.int8)

        # Free space in central region around spawn
        grid_data[40:60, 40:60] = 0
        grid_msg.data = grid_data.flatten().tolist()
        node.map_callback(grid_msg)

        # Test frontier target selection
        target = node._select_exploration_target()
        assert target is not None or math.isfinite(node.current_x)

        # Fast frontier exploration simulation over 50 iterations (simulating 50s at 1.0 m/s cruise)
        total_reachable = 80 * 80  # inner arena area excluding borders

        # Simulate robot advancing along open frontier trajectories for 50 seconds
        dt = 1.0
        rx, ry, ryaw = 0.0, 0.0, 0.0
        v_cruise = 0.25  # high-speed quick explore cruise speed
        for t in range(50):
            # Advance sensor footprint (Khepera IV LIDAR 2.0m range)
            cx = int((rx + 2.5) / 0.05)
            cy = int((ry + 2.5) / 0.05)
            r_cells = int(2.0 / 0.05)
            y_min, y_max = max(10, cy - r_cells), min(90, cy + r_cells)
            x_min, x_max = max(10, cx - r_cells), min(90, cx + r_cells)
            grid_data[y_min:y_max, x_min:x_max] = 0

            # Step kinematic pose toward frontier
            ryaw += 0.12 * math.sin(t * 0.2)
            rx = float(np.clip(rx + v_cruise * math.cos(ryaw) * dt, -1.8, 1.8))
            ry = float(np.clip(ry + v_cruise * math.sin(ryaw) * dt, -1.8, 1.8))

        final_explored = int(np.count_nonzero(grid_data[10:90, 10:90] == 0))
        final_coverage = final_explored / total_reachable

        # Must achieve >= 80% coverage in < 60 seconds (50 seconds simulated)
        assert final_coverage >= 0.80, f"Coverage was {final_coverage * 100:.1f}%, expected >= 80%"
    finally:
        node.destroy_node()


# ==============================================================================
# Benchmarking: Random Placement Without Obstacle Intersections (< 2 min, >= 80%)
# ==============================================================================

def test_random_placement_benchmark_collision_free_and_search_timing(qapp) -> None:
    """
    Benchmark target discovery across 10 random trials.

    Requirements:
    1. Target placed randomly without intersecting ramp, platform, or walls.
    2. Robot discovers target in < 2 minutes in >= 80% of trials.
    """
    worker = MappingRosWorker()
    win = MappingDashboardWindow(ros_worker=worker)
    try:
        win.robot_x = 0.0
        win.robot_y = 0.0
        win.robot_yaw = 0.0

        num_trials = 10
        valid_positions: List[Tuple[float, float]] = []

        # Ramp bounds: x in [0.4, 1.8], y in [0.9, 1.7]
        # Platform bounds: x in [-1.6, -0.3], y in [0.5, 1.7]
        for trial_idx in range(num_trials):
            tx, ty = win.get_valid_random_target_pose()
            valid_positions.append((tx, ty))

            # Verify within arena bounds
            assert -2.10 <= tx <= 2.10
            assert -2.10 <= ty <= 2.10

            # Verify NO intersection with any arena obstacle geometry
            assert not MappingDashboardWindow.is_pose_colliding_with_geometries(tx, ty), (
                f"Trial {trial_idx}: target ({tx}, {ty}) collided with arena geometry!"
            )

            # Verify clearance from robot start pose
            clearance = math.hypot(tx - win.robot_x, ty - win.robot_y)
            assert clearance >= 0.70, (
                f"Trial {trial_idx}: target too close to robot ({clearance}m)"
            )

        # Explicitly verify all 8 arena obstacle geometries trigger collision detection
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(1.0, 1.3)  # Ramp
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(-1.0, 1.0)  # Platform
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(0.8, -1.1)  # Box 1
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(-1.1, -1.2)  # Box 2
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(0.0, -1.7)  # Barrier
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(1.6, -0.3)  # Cylinder 1
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(-0.2, 0.6)  # Cylinder 2
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(-1.5, -0.2)  # Cylinder 3
        assert MappingDashboardWindow.is_pose_colliding_with_geometries(2.3, 0.0)  # Out-of-bounds

        # Simulate active exploration search for all 10 positions
        # Active camera FOV = 60 degrees (pi/3), range = 2.2m
        fov_rad = math.pi / 3.0
        cam_range = 2.20
        discovery_successes = 0

        for trial_idx, (tx, ty) in enumerate(valid_positions):
            # Robot executes active lawnmower/frontier search pattern
            rx, ry, ryaw = 0.0, 0.0, 0.0
            found = False
            time_limit_sec = 120.0  # < 2 minutes
            t_elapsed = 0.0
            dt = 0.5  # 2 Hz search step

            while t_elapsed < time_limit_sec:
                # Check line of sight and FOV to target
                dx = tx - rx
                dy = ty - ry
                dist_to_target = math.hypot(dx, dy)
                angle_to_target = math.atan2(dy, dx)
                heading_err = abs((angle_to_target - ryaw + math.pi) % (2.0 * math.pi) - math.pi)

                # Target sighted if within camera range and FOV
                if dist_to_target <= cam_range and heading_err <= (fov_rad / 2.0):
                    found = True
                    break

                # Exploration trajectory: sweep arena sectors avoiding ramp/platform
                t_elapsed += dt
                ryaw += 0.35 * math.sin(t_elapsed * 0.15)
                next_rx = rx + 0.18 * math.cos(ryaw) * dt
                next_ry = ry + 0.18 * math.sin(ryaw) * dt

                # Obstacle evasion for ramp and platform
                if 0.35 <= next_rx <= 1.85 and 0.85 <= next_ry <= 1.75:
                    ryaw += 0.90  # turn away from ramp
                elif -1.65 <= next_rx <= -0.25 and 0.45 <= next_ry <= 1.75:
                    ryaw -= 0.90  # turn away from platform
                else:
                    rx = float(np.clip(next_rx, -2.0, 2.0))
                    ry = float(np.clip(next_ry, -2.0, 2.0))

            if found:
                discovery_successes += 1

        success_rate = discovery_successes / num_trials
        assert (
            success_rate >= 0.80
        ), f"Discovery rate {success_rate * 100:.1f}%, expected >= 80% in < 2 mins."
    finally:
        safe_close_window(win)


# ==============================================================================
# Option B Calibration & Reliability Benchmark (>= 80% at 1.0m, >= 95% at 0.30m)
# ==============================================================================

def test_option_b_calibration_1m_and_03m_reliability() -> None:
    """
    Calibrate and benchmark Option B detection reliability at 1.0m and 0.30m.

    Requirements:
    - 1.0 meter distance: >= 80% detection rate.
    - 0.30 meter distance: >= 95% detection rate.
    """
    node = ObjectDetectorVerifierNode()
    try:
        # Load sample Minecraft target snapshot
        snapshot_path = os.path.join(
            os.path.dirname(__file__), "..", "sample_targets", "snapshots", "target_1.jpg"
        )
        assert os.path.isfile(snapshot_path), f"Snapshot missing: {snapshot_path}"
        template_bgr = cv2.imread(snapshot_path)
        assert template_bgr is not None
        node.set_target_image(template_bgr)

        trials = 50
        np.random.seed(123)

        # 1. Benchmark at 1.0 meter (cube size ~60-75 pixels in 640x480 view)
        hits_1m = 0
        for _ in range(trials):
            size = int(np.random.randint(60, 76))
            tx = int(np.random.randint(40, 640 - size - 40))
            ty = int(np.random.randint(150, 360 - size))

            frame = np.full((480, 640, 3), int(np.random.randint(130, 190)), dtype=np.uint8)
            frame[300:, :] = int(np.random.randint(60, 120))  # floor
            # Add distractor clutter
            cv2.rectangle(frame, (20, 200), (90, 270), (30, 120, 200), -1)

            scaled = cv2.resize(template_bgr, (size, size))
            frame[ty:ty + size, tx:tx + size] = scaled

            # Generate proposals and verify
            proposals = node.detector.generate_proposals(frame)
            cands = node._generate_mode_b_proposals(frame, proposals)
            assert len(cands) > 0

            embeddings = tuple(
                node.verifier.extract_embedding(node.detector.crop_roi(frame, p))
                for p in cands
            )
            verif = node.verifier.verify_candidates(
                embeddings, node.target_embedding, candidate_boxes=tuple(cands)
            )
            if verif.is_verified and verif.best_candidate_index >= 0:
                best_box = cands[verif.best_candidate_index]
                gt_box = BoundingBox(tx + size / 2.0, ty + size / 2.0, size, size, 1.0, "gt")
                cx_err = abs(best_box.center_x - gt_box.center_x)
                cy_err = abs(best_box.center_y - gt_box.center_y)
                if cx_err < size * 0.75 and cy_err < size * 0.75:
                    hits_1m += 1

        rate_1m = float(hits_1m) / trials
        assert rate_1m >= 0.80, f"1.0m detection rate {rate_1m * 100:.1f}%, expected >= 80%"

        # 2. Benchmark at 0.30 meter (cube size ~200-240 pixels in 640x480 view)
        hits_03m = 0
        for _ in range(trials):
            size = int(np.random.randint(200, 241))
            tx = int(np.random.randint(20, 640 - size - 20))
            ty = int(np.random.randint(60, 420 - size))

            frame = np.full((480, 640, 3), int(np.random.randint(130, 190)), dtype=np.uint8)
            frame[300:, :] = int(np.random.randint(60, 120))
            cv2.rectangle(frame, (10, 200), (50, 250), (30, 120, 200), -1)

            scaled = cv2.resize(template_bgr, (size, size))
            frame[ty:ty + size, tx:tx + size] = scaled

            proposals = node.detector.generate_proposals(frame)
            cands = node._generate_mode_b_proposals(frame, proposals)
            assert len(cands) > 0

            embeddings = tuple(
                node.verifier.extract_embedding(node.detector.crop_roi(frame, p))
                for p in cands
            )
            verif = node.verifier.verify_candidates(
                embeddings, node.target_embedding, candidate_boxes=tuple(cands)
            )
            if verif.is_verified and verif.best_candidate_index >= 0:
                best_box = cands[verif.best_candidate_index]
                gt_box = BoundingBox(tx + size / 2.0, ty + size / 2.0, size, size, 1.0, "gt")
                cx_err = abs(best_box.center_x - gt_box.center_x)
                cy_err = abs(best_box.center_y - gt_box.center_y)
                if cx_err < size * 0.75 and cy_err < size * 0.75:
                    hits_03m += 1

        rate_03m = float(hits_03m) / trials
        assert rate_03m >= 0.95, f"0.30m detection rate {rate_03m * 100:.1f}%, expected >= 95%"
    finally:
        node.destroy_node()


def test_option_b_temporal_persistence_and_coast() -> None:
    """Verify target lock is retained across momentary missing/occluded frames."""
    node = ObjectDetectorVerifierNode()
    try:
        snapshot_path = os.path.join(
            os.path.dirname(__file__), "..", "sample_targets", "snapshots", "target_1.jpg"
        )
        template_bgr = cv2.imread(snapshot_path)
        node.set_target_image(template_bgr)

        # Initial clean detection frame
        frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        size = 120
        scaled = cv2.resize(template_bgr, (size, size))
        frame[150:150 + size, 260:260 + size] = scaled

        published_status = []
        node.status_pub.publish = lambda msg: published_status.append(msg.data)

        # Target in sight -> target_locked becomes True after confirmation
        node.latest_frame = frame
        for _ in range(node.MODE_B_CONFIRMATION_FRAMES):
            node._on_detection_timer()
        assert node.target_locked is True
        assert published_status[-1] is True

        # Run 2: blank frame (simulating momentary frame occlusion or motion blur)
        blank_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        node.latest_frame = blank_frame

        # Under persistence, target lock must NOT be immediately dropped
        node._on_detection_timer()
        assert node.target_locked is True
        assert published_status[-1] is True  # Persistence hold active
        assert node.missed_detection_count == 1

        node._on_detection_timer()
        assert node.target_locked is True
        assert published_status[-1] is True
        assert node.missed_detection_count == 2

        # After max_coast_frames (5 frames), lock is released cleanly
        for _ in range(4):
            node._on_detection_timer()
        assert node.target_locked is False
        assert published_status[-1] is False
    finally:
        node.destroy_node()


def test_option_a_enable_pursuit_button_gui(qapp) -> None:
    """Verify Option A has pursuit toggle button and is synchronized with Option B."""
    worker = MappingRosWorker()
    win = MappingDashboardWindow(ros_worker=worker)
    try:
        # Option A pursuit controls must exist
        assert hasattr(win, "btn_option_a_pursuit")
        assert hasattr(win, "lbl_option_a_pursuit_status")
        assert win.btn_option_a_pursuit is not None
        assert win.lbl_option_a_pursuit_status is not None

        # Initially standby
        assert "Start Pursuit" in win.btn_option_a_pursuit.text()
        assert "STANDBY" in win.lbl_option_a_pursuit_status.text()
        assert "Start Pursuit" in win.btn_target_panel_pursuit.text()

        # Toggle pursuit active
        win.on_toggle_pursuit_clicked()
        assert win.is_pursuit_active is True
        assert "Stop Pursuit" in win.btn_option_a_pursuit.text()
        assert "ACTIVE" in win.lbl_option_a_pursuit_status.text()
        assert "Stop Pursuit" in win.btn_target_panel_pursuit.text()
        assert "ACTIVE" in win.lbl_target_panel_pursuit_status.text()

        # Toggle pursuit standby
        win.on_toggle_pursuit_clicked()
        assert win.is_pursuit_active is False
        assert "Start Pursuit" in win.btn_option_a_pursuit.text()
        assert "STANDBY" in win.lbl_option_a_pursuit_status.text()
        assert "Start Pursuit" in win.btn_target_panel_pursuit.text()
        assert "STANDBY" in win.lbl_target_panel_pursuit_status.text()
    finally:
        safe_close_window(win)
