# Project: Khepera IV Ramp Climbing, Slip Detection & SLAM Consistency

## Architecture
The system consists of:
1. **Physical & Simulation Layer (`src/khepera4_description`)**:
   - `urdf/khepera4.urdf` and `urdf/khepera4.urdf.xacro`: Khepera IV mechanical geometry, base_link collision bounds, caster friction and contact dynamics, wheel differential drive.
   - `worlds/khepera4_exploration_world.sdf`: Gazebo Harmonic exploration arena ($5\text{ m} \times 5\text{ m}$ walled arena), $10^\circ$ inclined test ramp, obstacles.
2. **Control & Estimation Layer (`src/khepera4_control`)**:
   - `khepera4_mapper.py`: Occupancy grid mapping, IMU-fused dead reckoning, IR/US proximity-based surface normal computation, tangential slip projection, perimeter wall clamping, publishing filtered odometry on `/khepera4/odom`.
   - `reactive_explorer.py`: Autonomous exploration state machine, low-latency bump/stall detection ($\le 0.4\text{ s}$), evasive backup-and-turn maneuvers, wall following.
3. **Visualization & Dashboard Layer (`src/khepera4_control`)**:
   - `mapping_dashboard.py`: Real-time Matplotlib-based UI displaying occupancy grid, robot trajectory, sensor telemetry, subscribing preferentially to `/khepera4/odom`.
4. **Verification & Testing Track (`src/khepera4_control/test/`)**:
   - Comprehensive multi-tier test suite verifying ramp climbing physics, slip projection math, bump detection latency, perimeter boundary enforcement, and dashboard consistency.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Base Link Collision Geometry Refinement | Recess/elevate lower front collision lip of `base_link` so it does not overhang caster at ground level; eliminate blunt vertical lip | M1 | Survey R2 |
| F2 | Caster Roller Dynamic Contact & Friction | Calibrate passive caster spheres (`kp=1e6`, `kd=1.0`, `min_depth=0.0005`, low rolling friction) for primary ground contact | M1 | Survey R2 |
| F3 | Ramp Approach Lip Transition Geometry | Seamless knife-edge transition ($z=0$) in `khepera4_exploration_world.sdf` without vertical step | M1 | Survey R2 |
| F4 | Ramp Pitch Telemetry & Climbing | Ensure robot climbs $10^\circ$ ramp at $0.12\text{ m/s}$ with smooth IMU pitch $0^\circ \to 10^\circ$ | M1 | Survey R2 / AC |
| F5 | Onboard IR Proximity & IMU Impact Bump Detection | 8 IR sensors (<0.015m) + IMU acceleration/jerk ($||j|| > 40\text{ m/s}^3$ or $a_x < -1.2\text{ m/s}^2$) collision sensing | M2 | Survey R1 |
| F6 | Surface Normal & Tangential Slip Projection | Compute surface normal $\hat{n}_{world}$ from IR ring, project allowed motion tangentially along $\hat{t}_{world}$, nullify normal penetration | M2 | Survey R1 |
| F7 | Head-on Stall & Slip Clamping | Clamp dead reckoning advance to $\le 0.03\text{ m}$ of contact during head-on stall, eliminating phantom forward travel | M2 | Survey R1 / AC |
| F8 | Outer Arena Perimeter Clamping | Prevent robot trajectory and estimated pose from penetrating outer arena perimeter walls ($5\text{ m} \times 5\text{ m}$, bound $[-2.405, +2.405]\text{ m}$) | M2 | Survey R1 / R3 |
| F9 | Low-Latency Autonomous Evasion | In `reactive_explorer.py`, trigger evasive backup-and-turn maneuver within $\le 0.4\text{ s}$ of bump/stall/slip | M2 | Survey R1 / AC |
| F10 | Mapper Odometry Publication | Publish filtered, slip-projected, wall-clamped pose and odometry from `khepera4_mapper.py` on `/khepera4/odom` | M3 | Survey R3 |
| F11 | Dashboard Odometry Source Prioritization | In `mapping_dashboard.py`, prioritize `/khepera4/odom` over raw uncorrected `/odom` to eliminate visual wall penetration | M3 | Survey R3 |
| F12 | Strict Physical Sensor Compliance | Use strictly the Khepera IV physical hardware suite (8 IR, 5 US, 4 Ground IR, 6-DOF IMU) without synthetic sensors | M4 | AC |
| F13 | Schema Validation & Unit Test Pass | Both URDF and SDF validate with `gz sdf -k`, all unit tests in `src/khepera4_control/test/` pass without regressions (`pytest-3`) | M4 | AC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Test harness, Tiers 1-4 test suite covering all features F1-F13, publication of `TEST_READY.md` | none | DONE |
| M1 | Ramp Climbing Physics & Geometry (R2) | `khepera4.urdf`, `khepera4.urdf.xacro`, `khepera4_exploration_world.sdf` (Features F1, F2, F3, F4) | none | DONE |
| M2 | Bump & Slip Detection (R1) | `khepera4_mapper.py`, `reactive_explorer.py` (Features F5, F6, F7, F8, F9) | none | DONE |
| M3 | Dashboard & SLAM Consistency (R3) | `khepera4_mapper.py` (`/khepera4/odom`), `mapping_dashboard.py` (Features F10, F11) | M2 interface | DONE |
| M4 | Final Milestone: E2E Pass & Coverage Hardening | 100% E2E test pass (Phase 1) + Adversarial hardening (Phase 2) (Features F12, F13) | M1, M2, M3, E2E | DONE |

## Interface Contracts
### `khepera4_mapper.py` ↔ `mapping_dashboard.py`
- Topic: `/khepera4/odom`
  - Message Type: `nav_msgs/msg/Odometry`
  - Content: Header with frame `odom` (or `map`), child_frame_id `base_link`, PoseWithCovariance representing wall-clamped, slip-projected robot pose $(x, y, yaw)$, Twist representing filtered velocity.
  - Priority: `mapping_dashboard.py` accepts both `/odom` and `/khepera4/odom`, but preferentially uses `/khepera4/odom` when received within the last 1.0 second.

### `khepera4_mapper.py` ↔ Simulation & Sensors
- Subscriptions:
  - `/imu/data`: `sensor_msgs/msg/Imu` (orientation quaternion, linear acceleration for jerk computation)
  - `/khepera4/ir_{0..7}/scan`: `sensor_msgs/msg/LaserScan` (ranges for proximity contact < 0.015m and surface normal calculation)
  - `/khepera4/us_{0..4}/scan`: `sensor_msgs/msg/LaserScan` (long range lookahead)
  - `/odom`: `nav_msgs/msg/Odometry` (raw diff-drive wheel increments)
  - `/cmd_vel`: `geometry_msgs/msg/Twist` (commanded velocities)

### `reactive_explorer.py` ↔ Simulation & Sensors
- Bump/slip event detection within $\le 0.4\text{ s}$ switches state directly from forward cruising (`STATE_FORWARD`, `STATE_WALL_FOLLOW`) to `STATE_REVERSE_TURN`.

## Code Layout
- Exclusive Write Ownership:
  - Sub-Orchestrator M1 / Workers M1:
    * `src/khepera4_description/urdf/khepera4.urdf`
    * `src/khepera4_description/urdf/khepera4.urdf.xacro`
    * `src/khepera4_description/worlds/khepera4_exploration_world.sdf`
  - Sub-Orchestrator M2 / Workers M2:
    * `src/khepera4_control/khepera4_control/khepera4_mapper.py`
    * `src/khepera4_control/khepera4_control/reactive_explorer.py`
  - Sub-Orchestrator M3 / Workers M3:
    * `src/khepera4_control/khepera4_control/mapping_dashboard.py`
    * `src/khepera4_control/khepera4_control/khepera4_mapper.py` (only for odometry publisher interface)
  - E2E Testing Orchestrator / Test Writers:
    * `src/khepera4_control/test/test_ramp_physics.py`
    * `src/khepera4_control/test/test_bump_slip_detection.py`
    * `src/khepera4_control/test/test_dashboard_consistency.py`
    * `src/khepera4_control/test/test_e2e_acceptance.py`
    * `TEST_INFRA.md`, `TEST_READY.md`
