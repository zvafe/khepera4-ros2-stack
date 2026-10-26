# AI Coding Agent & Vibecoder Context Specification (`AGENT_NOTES.md`)
**Repository:** `khepera4_ws` | **Primary Branch:** `strong-coffee`  
**Target Environment:** ROS 2 Jazzy Jalisco on Ubuntu 24.04 LTS (Noble Numbat) & Gazebo Harmonic 8.15  
**Purpose:** Ground future AI coding assistants, automated subagents, and human developers to prevent hallucinated patterns and preserve critical invariants.

---

## 1. Directory Ontology & Workspace Structure

```text
khepera4_ws/
├── .devcontainer/devcontainer.json      # VS Code Dev Container spec for ROS 2 Jazzy & Gazebo Harmonic
├── .env.example                         # Canonical environment configuration template (ROS_DOMAIN_ID=0)
├── .gitignore                           # Git ignore rules for build, pycache, binaries, and bags
├── Makefile                             # Optional top-level build helper
├── run.sh                               # Universal interactive Docker launcher (--native, --web, --mapping, --multi)
│
├── docs/                                # Consolidated documentation standard
│   ├── quickstart_onboarding.md         # Prerequisites, native Linux, Windows WSL2/WSLg, build & test guide
│   ├── docker_and_simulation.md         # Docker Compose, Native GUI, Web noVNC, GPU flags, display troubleshooting
│   ├── multi_robot_and_hardware.md      # Swarm fleet roles, C socket daemon, systemd, fleet_manager.sh
│   ├── control_systems_charter.md       # BSA academic theory, DC motor models, Lyapunov kinematics
│   ├── testing_and_verification.md      # 4-tier test harness, 223-test catalog, execution commands
│   ├── AGENT_NOTES.md                   # THIS FILE: AI agent context, invariants, and known edge cases
│   ├── resources/                       # Static PDF manuals and hardware specifications
│   │   └── khepera4_user_manual_3.x.pdf # K-Team Khepera IV User Manual Rev 3.0
│   └── archive/                         # Preserved sprint backlogs and historical requirements
│       ├── PROJECT_RAMP_SLIP_SPRINT.md  # Preserved ramp/slip sprint milestones (M1–M4)
│       ├── ORIGINAL_REQUEST_RAMP_SLIP.md# Historical prompt trace
│       └── TEST_READY_LEGACY_161.md     # Historical 161-test run snapshot
│
├── khepera4_guide/                      # Standalone client-side HTML5/JS educational simulator
│   ├── COLCON_IGNORE                    # Marker preventing colcon build package discovery warnings
│   ├── index.html, simulator.js, style.css
│   └── README.md
│
├── maps/                                # SLAM persistence directory
│   ├── khepera_map.pgm                  # Default 2D occupancy grid raster
│   └── khepera_map.yaml                 # Standard ROS map server metadata header
│
├── physical_robot/                      # Embedded physical robot deployment stack
│   ├── fleet_manager.sh                 # Master PC fleet administration tool (ping, deploy, start, camera)
│   └── onboard_daemon/                  # Embedded C socket daemon for Yocto Linux
│       ├── Makefile                     # Native and mock build targets
│       ├── khepera4_server.service      # Systemd service autostart unit
│       └── khepera4_tcp_server.c        # Low-latency C socket server translating ASCII to libkhepera
│
└── src/
    ├── khepera4_control/                # Core ROS 2 Python nodes, controllers, SLAM, and test suites
    │   ├── package.xml, setup.py, setup.cfg
    │   ├── khepera4_control/
    │   │   ├── __init__.py
    │   │   ├── common.py                # Euler/quaternion math & angle normalization utilities
    │   │   ├── closed_loop_controller.py# Lyapunov unicycle cascade controller (rho, alpha, beta)
    │   │   ├── dashboard.py             # Qt5 BSA educational & multi-robot fleet dashboard
    │   │   ├── khepera4_hardware_bridge.py # Master PC TCP socket bridge to physical Khepera IV
    │   │   ├── khepera4_mapper.py       # 2D/2.5D SLAM, virtual scan fusion, slip projection, persistence
    │   │   ├── mapping_dashboard.py     # Qt5 dedicated single-robot SLAM & exploration station
    │   │   ├── open_loop_benchmark.py   # Kinematic validation benchmark for timed unicycle maneuvers
    │   │   ├── reactive_explorer.py     # Autonomous Braitenberg obstacle avoidance & stall escape FSM
    │   │   ├── swarm_coordinator.py     # 3-robot fleet coordination (midpoint relay tether, triangle)
    │   │   └── telemetry_logger.py      # Automated CSV & 4-panel step-response dynamic plotter
    │   ├── launch/
    │   │   ├── benchmark.launch.py      # Launches open-loop benchmark and telemetry logger
    │   │   ├── closed_loop.launch.py    # Headless setpoint regulation test
    │   │   ├── control_station.launch.py# Single-robot educational control dashboard
    │   │   ├── mapping.launch.py        # Primary SLAM, exploration & mapping dashboard pipeline
    │   │   ├── multi_robot_physical.launch.py # Bridges 3 physical robots via hardware bridge
    │   │   └── multi_robot_station.launch.py  # Primary multi-robot fleet launcher (sim/physical/mock)
    │   └── test/                        # 223 automated unit, adversarial, and physics tests
    │
    └── khepera4_description/            # Gazebo Harmonic simulation assets and robot descriptions
        ├── CMakeLists.txt, package.xml
        ├── config/khepera4_mapping.rviz # Default RViz2 visualizer configuration
        ├── launch/
        │   ├── khepera4_sim.launch.py   # Single-robot Gazebo Harmonic simulation launcher
        │   └── khepera4_multi_sim.launch.py # 3-robot Gazebo Harmonic simulation launcher
        ├── meshes/                      # Wavefront OBJ/MTL visual and collision meshes (chassis, caster, wheel)
        ├── urdf/
        │   ├── khepera4.urdf.xacro      # Parameterized multi-robot Xacro model (PRIMARY)
        │   └── khepera4.urdf            # Static single-robot fallback URDF
        └── worlds/
            ├── khepera4_exploration_world.sdf # Primary world with 10° ramp, cliffs, and obstacles
            └── khepera4_world.sdf             # Flat arena baseline world
```

---

## 2. Critical Architectural Invariants ("Do Not Break")

1. **TF2 Frame Tree Alignment (`map -> odom -> base_footprint`):**
   - In standard ROS SLAM, `khepera4_mapper.py` must publish $T(map \to odom) = T(map \to base) \cdot T(odom \to base)^{-1}$.
   - **NEVER broadcast a static identity transform for `map -> odom`.** A static identity transform causes RViz and laser scan projections to use slipping raw odometry, reintroducing wall-collision visual glitches and penetration.
2. **Stream Prioritization in Navigation & Dashboards:**
   - Always prioritize `/khepera4/odom` (slip-projected, wall-clamped odometry) over uncorrected `/odom`.
   - If `/khepera4/odom` drops out for $> 1.0\text{ s}$, nodes may fall back to `/odom`, but must suppress raw poses extending beyond arena walls ($|x| > 2.405\text{ m}$ or $|y| > 2.405\text{ m}$).
3. **Sensor Modality Contracts:**
   - In Gazebo Harmonic simulation, proximity sensors are published as discrete `sensor_msgs/msg/LaserScan` messages:
     * 8 Proximity IR: `/khepera4/ir_{0..7}/scan`
     * 5 Ultrasonic Sonar: `/khepera4/us_{0..4}/scan`
     * 4 Ground Cliff IR: `/khepera4/ground_ir_{0..3}/scan`
   - Virtual 360° Scan: `/khepera4/scan` (`sensor_msgs/msg/LaserScan`).
   - Physical Hardware Bridge: `/ir_sensors` and `/us_sensors` (`std_msgs/msg/Float32MultiArray`).
   - **DO NOT hallucinate `sensor_msgs/msg/Range`.** It is not used in this stack.
4. **Multi-Robot Role Naming Convention:**
   - Robot 1: `khepera_1` (`role: base_link`, Anchor)
   - Robot 2: `khepera_2` (`role: extender`, Chain Relay)
   - Robot 3: `khepera_3` (`role: explorer`, Frontier Scout)
   - **DO NOT use the legacy designation `'ranger'`.** It breaks coordinate frame resolution in `swarm_coordinator.py` and `dashboard.py`.
5. **Port Allocation Rules:**
   - Port `5000`: Physical robot motor/sensor TCP daemon (`khepera4_tcp_server`).
   - Port `5001`: Physical robot GStreamer camera streaming.
   - Port `6080`: Web browser noVNC virtual desktop.
   - **NEVER BIND TO PORT 1947.** Reserved globally for the Sentinel HASP license manager.

---

## 3. "Do Not Touch" Files vs. Extension Points

### 🔒 "Do Not Touch" Files (Calibrated Core)
- `src/khepera4_description/urdf/khepera4.urdf.xacro`: Caster contact parameters ($\text{kp}=10^6, \mu=0.0001, \text{minDepth}=0.0005$), chamfered two-tier lower chassis collision geometry, and wheel drive scaling ($0.021\text{ m}$ radius, $0.1054\text{ m}$ wheelbase). Altering these breaks ramp climbing and unicycle kinematics.
- `src/khepera4_description/worlds/khepera4_exploration_world.sdf`: Knife-edge ramp entry lip ($z=0$ at entry point) and 1000 Hz DART physics configuration (`<max_step_size>0.001</max_step_size>`).
- `src/khepera4_control/khepera4_control/common.py`: Verified quaternion/Euler conversion and angle normalization routines.

### 🛠️ Safe Extension Points
- `src/khepera4_control/khepera4_control/swarm_coordinator.py`: Adding new formation modes (e.g. line, orbit, leader-follower) via the `/swarm/cmd` dispatch handler.
- `src/khepera4_control/khepera4_control/reactive_explorer.py`: Integrating frontier exploration algorithms or custom escape heuristics into the state machine.
- `src/khepera4_control/khepera4_control/khepera4_mapper.py`: Adding loop-closure detection, graph SLAM optimization, or 3D voxel grids.
- `docs/`: Adding new documentation or tutorials.

---

## 4. Known Edge Cases & Defenses

1. **Orphaned Gazebo Background Processes:**
   - Gazebo Harmonic sometimes leaves orphaned `gz-sim` background daemons running after hard crashes or aborted tests, blocking DDS discovery domains.
   - *Fix:* Always run `pkill -f "gz sim"` if newly launched worlds fail to render.
2. **Hardcoded User Home Paths:**
   - Never write hardcoded paths like `/home/andy/...`. Always use `os.path.join(os.getcwd(), ...)` or read from environment variables (e.g., `KHEPERA_TELEMETRY_DIR`).
3. **Linter Failures during `colcon test`:**
   - Pytest is pre-configured via `setup.cfg` to ignore trivial whitespace/docstring warnings (`-m "not linter"`). When running tests programmatically, always ensure `-m "not linter"` is honored.
4. **C Daemon Signed Char Promotion:**
   - In `khepera4_tcp_server.c`, raw ADC byte arrays MUST be typed as `unsigned char` or cast before bitwise shifting:
     `ir[i] = (raw_ir[i*2] | ((unsigned int)raw_ir[i*2 + 1] << 8));`
     Signed `char` promotion introduces negative distance values on ARM targets.
5. **Thin-Wall Ray Carving:**
   - Arena walls are only 5 cm thick. In `khepera4_mapper.py`, free-space rays from Bresenham raycasting must check `if self.log_odds[row, col] < self.l_occ:` before subtracting `l_free` to prevent carving cavities through solid obstacles during close-range wall scrapes.
