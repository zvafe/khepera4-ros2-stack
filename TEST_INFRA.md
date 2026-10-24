# Test Infrastructure Specification: Khepera IV Ramp Climbing, Slip Detection & SLAM Consistency

**Document Version:** 1.0.0  
**Author:** E2E Test Writer (`test_writer_e2e`)  
**Workspace:** `/home/andy/Desktop/khepera4_ws`  
**Reference Document:** `PROJECT.md` / `ORIGINAL_REQUEST.md`  

---

## 1. Overview & Test Objectives

The Khepera IV Verification & Testing Track implements an opaque-box, multi-tier test harness to comprehensively validate all mechanical, dynamic, sensor fusion, mapping, and dashboard consistency requirements. The test suite operates without synthetic shortcuts (e.g. cheat GPS or synthetic lidar), verifying that the physical sensor suite (8 IR, 5 US, 4 Ground IR, 6-DOF IMU) accurately drives dead reckoning, ramp ascent, slip compensation, perimeter wall clamping, and visual dashboard monitoring.

---

## 2. Test Execution Environment

- **Operating System:** Linux (Ubuntu 24.04 LTS / x86_64)
- **Middleware:** ROS 2 Jazzy Jalisco (`/opt/ros/jazzy`)
- **Simulator:** Gazebo Harmonic (Gazebo Sim 8.15.0)
- **Language / Runtime:** Python 3.12.3
- **Test Runner:** Pytest 7.4.4 (`pytest-3`) with `launch-testing`, `launch-testing-ros`, and `colcon`
- **Execution Command:**
  ```bash
  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  pytest-3 -m "not linter" src/khepera4_control/test/
  ```

---

## 3. Four-Tier Test Case Design Methodology

The test suite enforces a rigorous 4-tier testing hierarchy:

### Tier 1: Feature Coverage
- **Criteria:** $\ge 5$ test cases per feature for features **F1 through F13** (minimum 65 test cases).
- **Scope:** Verifies the nominal operating contract and happy path for every feature defined in `PROJECT.md`.
- **Authoritative Source:** Interface specifications, geometry dimensions, sensor counts, and topic contracts.

### Tier 2: Boundary & Corner Cases
- **Criteria:** $\ge 5$ test cases per feature for features **F1 through F13** (minimum 65 test cases).
- **Scope:** Verifies limits, zero-crossing, edge cases, negative/overflow conditions, sensor saturations, and mathematical singularities.
- **Authoritative Source:** Coordinate extrema ($\pm 2.405\text{ m}$ to $\pm 2.430\text{ m}$), zero-velocity drift limits, acute corner normals, and high-frequency sample jitter.

### Tier 3: Cross-Feature Combinations
- **Criteria:** Pairwise and multi-way coverage of major feature interactions (15 test cases).
- **Scope:** Verifies concurrent behaviors where multiple subsystems interact (e.g. ramp pitch telemetry combined with wheel slip dead reckoning, glancing wall contact with tangential projection and autonomous evasive maneuvering).

### Tier 4: Real-World Application Scenarios
- **Criteria:** $\ge 5$ realistic end-to-end application scenarios (7 scenarios implemented).
- **Scope:** Simulates complete operational mission profiles:
  1. Maze traversal without phantom corridor elongation.
  2. Continuous glancing wall scraping with perimeter tangential tracking.
  3. Complete ramp ascent, plateau navigation, and descent cycle.
  4. Full perimeter bounded exploration cycle across all 4 arena quadrants.
  5. Acute corner entrapment (tight 90° corner) and sub-400ms escape.
  6. Cluttered multi-obstacle course navigation.
  7. Dashboard real-time tracking during continuous slipping.

---

## 4. Test Suite File Organization & Feature Mapping

| Test File | Features Covered | Tiers | Test Count | Description |
|---|---|---|---|---|
| `test_ramp_physics.py` | **F1, F2, F3, F4** | Tier 1, Tier 2 | **40** | Ramp climbing physics, URDF collision profile elevation, caster rolling friction & contact parameters (`kp=1e6`, `kd=1.0`), world SDF $10^\circ$ ramp geometry with $z=0$ lip, and IMU pitch telemetry extraction. |
| `test_bump_slip_detection.py` | **F5, F6, F7, F8, F9** | Tier 1, Tier 2 | **50** | IR contact (< 0.015m) & IMU jerk bump detection, surface normal computation, tangential slip projection, head-on stall advance clamp ($\le 0.03\text{ m}$), arena perimeter wall clamping ($[-2.405, +2.405]\text{ m}$), and autonomous evasive reverse-turn within $\le 0.4\text{ s}$. |
| `test_dashboard_consistency.py` | **F10, F11, F12, F13** | Tier 1, Tier 2 | **40** | Mapper odometry publication on `/khepera4/odom`, dashboard odometry source prioritization over raw `/odom` with 1.0s timeout, strict physical sensor suite compliance (8 IR, 5 US, 4 Ground IR, 1 IMU), and schema validation (`gz sdf -k`). |
| `test_e2e_acceptance.py` | **F1..F13 Interactions** | Tier 3, Tier 4 | **22** | 15 Cross-feature combinations + 7 Real-world application scenarios evaluating full integration of ramp climbing, slip compensation, perimeter containment, and dashboard consistency. |
| `test_mapper_fusion.py` | Baseline Fusion | Baseline | **7** | Existing sensor alignment, IMU heading fusion, reverse motion, and map persistence tests. |
| `test_reactive_explorer.py` | Baseline Explorer | Baseline | **2** | Existing ground IR sector integration and corner stall escape tests. |

**Total Suite Size:** **161 active test cases** (3 linter tests deselected via `-m "not linter"`).

---

## 5. Opaque-Box Test Harness Design

The test harness exercises the robot system strictly as an opaque box:
1. **Sensor Stimulation:** Injects realistic ROS 2 messages (`sensor_msgs/LaserScan`, `sensor_msgs/Imu`, `nav_msgs/Odometry`, `std_msgs/Bool`) into public node callbacks.
2. **Observable Interception:** Evaluates public state observables (`node.robot_x`, `node.robot_y`, `node.robot_yaw`, `node.robot_path`, `/khepera4/odom` topic messages, `/cmd_vel` velocity commands, `node.state`).
3. **Model & Schema Verification:** Inspects URDF and SDF XML trees with `xml.etree.ElementTree` to verify physical parameters, collision shapes, clearance angles, and executes `gz sdf -k` subprocess validation.
4. **Isolated Execution:** Tests set up independent node instances and clean up via `node.destroy_node()`, running in $< 3.0\text{ s}$ without hanging or multi-threading deadlocks.
