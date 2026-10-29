# Khepera IV Master Command Reference & Simulation Guide
**Repository:** `khepera4_ws` | **ROS Distro:** ROS 2 Jazzy Jalisco | **Simulator:** Gazebo Harmonic (GZ Sim 8) | **Embedded OS:** Poky Yocto Linux

This document is the comprehensive, indexed, and copy-pasteable Master Command Reference for running all Khepera IV simulations, benchmarking stacks, teleoperation modes, and physical robot hardware bridges across all three execution targets:
1. **Native Host ROS 2 Jazzy** (Ubuntu 24.04 LTS / local installation)
2. **Docker Native GUI** (GPU / WSLg / X11 pass-through)
3. **Docker Web noVNC** (HTML5 in-browser virtual desktop on port 6080)

---

## 📑 Master Index & Table of Contents

- [⚡ 1. Fast-Track One-Liners (Quick Launch by Execution Mode)](#-1-fast-track-one-liners-quick-launch-by-execution-mode)
  - [1.1 Mode A: Native Host ROS 2 Jazzy](#11-mode-a-native-host-ros-2-jazzy)
  - [1.2 Mode B: Docker Native GUI](#12-mode-b-docker-native-gui)
  - [1.3 Mode C: Docker Web noVNC (Browser)](#13-mode-c-docker-web-novnc-browser)
- [🛠️ 2. Environment Setup, Sourcing & Build Commands](#️-2-environment-setup-sourcing--build-commands)
  - [2.1 Standard Rebuild (`--symlink-install`)](#21-standard-rebuild---symlink-install)
  - [2.2 Selective Package Build](#22-selective-package-build)
  - [2.3 Clean Rebuild (Purge Build Cache)](#23-clean-rebuild-purge-build-cache)
  - [2.4 Required Environment Variables](#24-required-environment-variables)
  - [2.5 Automated Verification Test Suite (227 Tests)](#25-automated-verification-test-suite-227-tests)
- [💻 3. Execution Modes Breakdown & Syntax Comparison](#-3-execution-modes-breakdown--syntax-comparison)
- [🚀 4. Complete Simulation Stack Commands](#-4-complete-simulation-stack-commands)
  - [4.1 Stack 1: Single Robot Baseline Simulation](#41-stack-1-single-robot-baseline-simulation)
  - [4.2 Stack 2: Closed-Loop Pole Placement & Lyapunov Regulation](#42-stack-2-closed-loop-pole-placement--lyapunov-regulation)
  - [4.3 Stack 3: Open-Loop Step-Response & Kinematics Benchmark](#43-stack-3-open-loop-step-response--kinematics-benchmark)
  - [4.4 Stack 4: Multi-Sensor 2D/2.5D SLAM & Frontier Exploration](#44-stack-4-multi-sensor-2d25d-slam--frontier-exploration)
  - [4.5 Stack 5: Interactive Teleoperation Modes](#45-stack-5-interactive-teleoperation-modes)
  - [4.6 Stack 6: 3-Robot Swarm Coordination Fleet](#46-stack-6-3-robot-swarm-coordination-fleet)
- [🤖 5. Physical Hardware Deployment (Single Robot in Lab & Multi-Robot Fleet)](#-5-physical-hardware-deployment-single-robot-in-lab--multi-robot-fleet)
  - [5.1 Lab Network Setup & Wi-Fi Pairing](#51-lab-network-setup--wi-fi-pairing)
  - [5.2 Onboard C Socket Daemon Deployment](#52-onboard-c-socket-daemon-deployment)
  - [5.3 Connecting to One Real Robot in the Lab (Step-by-Step Terminal Sequence)](#53-connecting-to-one-real-robot-in-the-lab-step-by-step-terminal-sequence)
  - [5.4 Multi-Robot Physical Fleet Deployment](#54-multi-robot-physical-fleet-deployment)
- [📡 6. Key Runtime Services, Diagnostics & Topic Inspection](#-6-key-runtime-services-diagnostics--topic-inspection)
  - [6.1 Map Memory & Persistence Services](#61-map-memory--persistence-services)
  - [6.2 Autonomous Exploration Control Services](#62-autonomous-exploration-control-services)
  - [6.3 Real-Time Topic Rates & Telemetry Monitoring](#63-real-time-topic-rates--telemetry-monitoring)
- [🧹 7. Process Cleanup & Simulation Reset](#-7-process-cleanup--simulation-reset)

---

## ⚡ 1. Fast-Track One-Liners (Quick Launch by Execution Mode)

### 1.1 Mode A: Native Host ROS 2 Jazzy
Rebuilds the workspace, exports the Gazebo Harmonic model paths, and launches the full SLAM & Autonomous Exploration Station with RViz:
```bash
source /opt/ros/jazzy/setup.bash && \
colcon build --symlink-install && \
source install/setup.bash && \
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH} && \
ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true
```

### 1.2 Mode B: Docker Native GUI
Runs the mapping stack inside the pre-built Docker container with GPU / WSLg / X11 pass-through windowing:
```bash
./run.sh --native --mapping
```

### 1.3 Mode C: Docker Web noVNC (Browser)
Runs the simulation inside Docker and streams the full desktop GUI to your web browser at `http://localhost:6080`:
```bash
./run.sh --web --mapping
```
*Direct link:* [http://localhost:6080/vnc.html?autoconnect=true&resize=remote](http://localhost:6080/vnc.html?autoconnect=true&resize=remote)

---

## 🛠️ 2. Environment Setup, Sourcing & Build Commands

### 2.1 Standard Rebuild (`--symlink-install`)
Using `--symlink-install` allows you to modify Python controllers, launch scripts, URDF/Xacro descriptions, and parameters without requiring a rebuild after each edit:

```bash
# 1. Source ROS 2 Jazzy Underlay
source /opt/ros/jazzy/setup.bash

# 2. Build All Packages with Symlinks
colcon build --symlink-install

# 3. Source Workspace Overlay
source install/setup.bash
```

### 2.2 Selective Package Build
Accelerates compile times by building only the specific package being developed:
```bash
# Build description package (URDF/Xacro/Worlds/Meshes/RViz configs)
colcon build --packages-select khepera4_description --symlink-install

# Build control and mapping package (Controllers, SLAM, Dashboards, Hardware Bridge)
colcon build --packages-select khepera4_control --symlink-install
```

### 2.3 Clean Rebuild (Purge Build Cache)
Recommended if CMake caches, setup files, or mesh export links become stale:
```bash
# Remove build, install, and log caches
rm -rf build/ install/ log/

# Source and rebuild from scratch
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 2.4 Required Environment Variables
Ensure Gazebo Harmonic resolves the Khepera IV DART physics model, meshes, and exploration SDF worlds:
```bash
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH}
```

### 2.5 Automated Verification Test Suite (227 Tests)
Run the automated opaque-box test harness validating kinematics, Lyapunov stability, SLAM mapping, and hardware socket protocols:
```bash
# Run via colcon
colcon test --packages-select khepera4_control
colcon test-result --verbose

# Or run directly via pytest
python3 -m pytest src/khepera4_control/test/ -v
```

---

## 💻 3. Execution Modes Breakdown & Syntax Comparison

Every launch command in this repository can be run in any of the three execution environments:

| Mode | Target Machine | Graphics Pipeline | How to Run Arbitrary Commands |
|---|---|---|---|
| **Native Host** | Native Ubuntu 24.04 LTS with ROS 2 Jazzy | Native OpenGL / Mesa | `ros2 launch <pkg> <launch_file> [args...]` |
| **Docker Native GUI** | High-performance Linux or Windows 11 WSLg | X11 / Wayland pass-through | `./run.sh --native ros2 launch <pkg> <launch_file> [args...]` |
| **Docker Web noVNC** | Any OS, macOS, Chromebook, low-spec laptops | Virtual Framebuffer + noVNC | `./run.sh --web ros2 launch <pkg> <launch_file> [args...]` |

---

## 🚀 4. Complete Simulation Stack Commands

### 4.1 Stack 1: Single Robot Baseline Simulation
Launches the baseline differential-drive Khepera IV model inside Gazebo Harmonic with active DART physics, ground friction, and sensor bridges.

#### Native Host ROS 2:
```bash
# Gazebo Harmonic with GUI window
ros2 launch khepera4_description khepera4_sim.launch.py

# Headless mode (server-only, saves CPU/GPU resources)
ros2 launch khepera4_description khepera4_sim.launch.py headless:=true

# Load custom SDF world
ros2 launch khepera4_description khepera4_sim.launch.py \
    world:=$(pwd)/src/khepera4_description/worlds/khepera4_exploration_world.sdf
```

#### Docker Native GUI:
```bash
./run.sh --native ros2 launch khepera4_description khepera4_sim.launch.py
```

#### Docker Web noVNC:
```bash
./run.sh --web ros2 launch khepera4_description khepera4_sim.launch.py
```

---

### 4.2 Stack 2: Closed-Loop Pole Placement & Lyapunov Regulation
Implements the non-linear polar coordinate transformation $(\rho, \alpha, \beta)$ and Lyapunov feedback regulation law:
$$v = k_\rho \rho \cos\alpha, \quad \omega = k_\alpha \alpha + k_\beta \beta + \frac{v}{\rho}\sin\alpha\left(1 - \frac{\beta}{\alpha}\right)$$
Complies with classical 2nd-order control theory damping ($\zeta = \frac{\sqrt{2}}{2} \approx 0.707$) and pole placement criteria ($k_\rho > 0, k_\alpha > 0, k_\beta < 0, k_\alpha - k_\rho > 0$).

#### Native Host ROS 2:
```bash
# 1. Start baseline simulation in background or first terminal:
ros2 launch khepera4_description khepera4_sim.launch.py headless:=false

# 2. In second terminal, launch closed-loop controller in Step mode (default goal: x=1.0, y=0.0):
ros2 launch khepera4_control closed_loop.launch.py mode:=step goal_x:=1.5 goal_y:=1.0 goal_yaw:=1.57

# 3. Trajectory Tracking Modes:
# Waypoint sequence tracking:
ros2 launch khepera4_control closed_loop.launch.py mode:=waypoint goal_x:=2.0 goal_y:=1.0

# Continuous Square trajectory (1m side):
ros2 launch khepera4_control closed_loop.launch.py mode:=square

# Continuous Circular orbit (radius 1m):
ros2 launch khepera4_control closed_loop.launch.py mode:=circle
```

#### Docker Native GUI:
```bash
./run.sh --native ros2 launch khepera4_control closed_loop.launch.py mode:=step goal_x:=1.5 goal_y:=1.0
```

#### Docker Web noVNC:
```bash
./run.sh --web ros2 launch khepera4_control closed_loop.launch.py mode:=step goal_x:=1.5 goal_y:=1.0
```

| Parameter | Default | Units | Description |
|---|:---:|:---:|---|
| `mode` | `step` | string | Controller mode: `step`, `waypoint`, `square`, `circle` |
| `goal_x` | `1.0` | m | Target setpoint Euclidean X coordinate |
| `goal_y` | `0.0` | m | Target setpoint Euclidean Y coordinate |
| `goal_yaw` | `0.0` | rad | Target setpoint heading orientation |

---

### 4.3 Stack 3: Open-Loop Step-Response & Kinematics Benchmark
Applies precision open-loop timed pulses to validate unicycle kinematics, calculate wheel slip, and log transient step-response performance metrics (Rise time $t_r$, Settling time $t_s$, Overshoot $\sigma\%$, Steady-state error $e_{st}$).

#### Native Host ROS 2:
```bash
# 1. Forward Linear Step Pulse (v = 0.15 m/s for 4.0 seconds)
ros2 launch khepera4_control benchmark.launch.py maneuver:=forward linear_speed:=0.15 duration:=4.0

# 2. In-Place Pure Rotation Pulse (w = 1.0 rad/s for pi seconds = 180° turn)
ros2 launch khepera4_control benchmark.launch.py maneuver:=rotate angular_speed:=1.0 duration:=3.1415

# 3. Circular Maneuver Benchmark (v = 0.15 m/s, w = 0.75 rad/s for 8.0 seconds)
ros2 launch khepera4_control benchmark.launch.py maneuver:=circle linear_speed:=0.15 angular_speed:=0.75 duration:=8.0
```

#### Docker Native GUI:
```bash
./run.sh --native ros2 launch khepera4_control benchmark.launch.py maneuver:=forward linear_speed:=0.15 duration:=4.0
```

#### Docker Web noVNC:
```bash
./run.sh --web ros2 launch khepera4_control benchmark.launch.py maneuver:=forward linear_speed:=0.15 duration:=4.0
```

Telemetry and transient step metrics are automatically recorded to `telemetry/benchmark_telemetry.csv`.

---

### 4.4 Stack 4: Multi-Sensor 2D/2.5D SLAM & Frontier Exploration
Full cyber-physical autonomy pipeline: Gazebo Harmonic simulation world, 360° fused proximity scan (`/khepera4/scan`), 2D probabilistic log-odds grid (`/map`), 2.5D elevation grid (`/elevation_map`), surface roughness & cliff alerts (`/surface_map`, `/cliff_alert`), autonomous reactive explorer, and Qt control dashboard.

#### Native Host ROS 2:
```bash
# A. Full Autonomous Exploration Stack (Gazebo GUI + RViz2 + Qt Dashboard + Autonomous Explorer)
ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true dashboard:=true

# B. Manual Operator SLAM (Interactive driving via Qt D-Pad / WASD, no autonomous wanderer)
ros2 launch khepera4_control mapping.launch.py auto_explore:=false dashboard:=true rviz:=true

# C. Headless Mapping (Server-only Gazebo — minimal CPU/GPU load for CI/CD or low-spec systems)
ros2 launch khepera4_control mapping.launch.py headless:=true auto_explore:=true dashboard:=true rviz:=true

# D. Minimal Headless (Pure CLI / Service Testing)
ros2 launch khepera4_control mapping.launch.py headless:=true auto_explore:=true dashboard:=false rviz:=false

# E. Custom Exploration Arena SDF
ros2 launch khepera4_control mapping.launch.py \
    world:=$(pwd)/src/khepera4_description/worlds/khepera4_exploration_world.sdf \
    auto_explore:=true rviz:=true
```

#### Docker Native GUI:
```bash
./run.sh --native --mapping
```

#### Docker Web noVNC:
```bash
./run.sh --web --mapping
```

---

### 4.5 Stack 5: Interactive Teleoperation Modes

Multiple methods are available to command the Khepera IV velocity interface (`/cmd_vel`):

#### Method 1: Keyboard Teleoperation Node
```bash
# Drive single robot in simulation
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Remap to specific robot namespace (e.g., khepera_1 on physical robot or swarm)
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/khepera_1/cmd_vel
```
*Keys: `i` = Forward, `k` = Stop, `,` = Reverse, `j` = Turn Left, `l` = Turn Right, `u`/`o`/`m`/`.` = Curved Arcs.*

#### Method 2: Educational Control Station (Mode A: WASD Drive)
Launches the BSA student station with real-time wheel speed meters, strip chart plots, and WASD keys:
```bash
# Native Host
ros2 launch khepera4_control control_station.launch.py

# Docker Native GUI
./run.sh --native ros2 launch khepera4_control control_station.launch.py

# Docker Web noVNC
./run.sh --web ros2 launch khepera4_control control_station.launch.py
```

#### Method 3: Direct One-Shot ROS 2 Command Publication
```bash
# Drive forward at 0.15 m/s
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Spin in place counter-clockwise at 1.0 rad/s
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}"

# Full emergency halt
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

### 4.6 Stack 6: 3-Robot Swarm Coordination Fleet
Coordinates Anchor (`khepera_1`), Relay (`khepera_2`), and Scout (`khepera_3`) with dynamic RF midpoint tethering and triangle formations.

#### Native Host ROS 2:
```bash
# Simulated 3-Robot Fleet with Qt Swarm Dashboard
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim gui:=true

# Headless 3-Robot Swarm Simulation
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim gui:=true headless:=true

# Mock Hardware Swarm (Runs internal kinematic emulator, no Gazebo physics)
ros2 launch khepera4_control multi_robot_station.launch.py mode:=mock gui:=true
```

#### Docker Native GUI:
```bash
./run.sh --native --multi
```

#### Docker Web noVNC:
```bash
./run.sh --web --multi
```

---

## 🤖 5. Physical Hardware Deployment (Single Robot in Lab & Multi-Robot Fleet)

This section provides the exact terminal procedure to connect from your Master PC to **one real physical Khepera IV mobile robot** in the laboratory.

```
+------------------------------------------------------------------------------------+
|                         MASTER PC / CLASSROOM WORKSTATION                          |
|                                                                                    |
|  [Qt Dashboard]       [RViz2 3D Viewer]        [teleop_twist_keyboard]             |
|          │                    │                            │                       |
|          ▼                    ▼                            ▼                       |
|  /khepera_1/odom       /khepera_1/scan              /khepera_1/cmd_vel             |
|          │                    │                            │                       |
|          └────────────────────┼────────────────────────────┘                       |
|                               ▼                                                    |
|            +--------------------------------------+                                |
|            |       khepera4_hardware_bridge       |                                |
|            |  - Translates Twist to motor pulses  |                                |
|            |  - Reads encoders & proximity IR/US  |                                |
|            +--------------------------------------+                                |
+───────────────────────────────┬────────────────────────────────────────────────────+
                                │ Wi-Fi TCP Socket (Port 5000)
                                ▼
+────────────────────────────────────────────────────────────────────────────────────+
|                     PHYSICAL KHEPERA IV ROBOT (192.168.1.101)                      |
|                                                                                    |
|            +--------------------------------------+                                |
|            |     khepera4_tcp_server (Native C)   |                                |
|            |  - libkhepera & dsPIC33 I2C driver   |                                |
|            |  - Hardware 250ms motor watchdog     |                                |
|            +--------------------------------------+                                |
+────────────────────────────────────────────────────────────────────────────────────+
```

### 5.1 Lab Network Setup & Wi-Fi Pairing
1. **Network Router:** Connect Master PC and physical robot to a dedicated Wi-Fi router or hotspot (do not use university guest Wi-Fi due to AP client isolation).
2. **Subnet IP Allocation:**
   - Master PC: `192.168.1.50` (or dynamic DHCP)
   - Physical Robot 1: `192.168.1.101`
   - TCP Socket Port: `5000`
3. **Verify Network Ping:**
   ```bash
   ping -c 3 192.168.1.101
   ```
   *Or probe the entire fleet:*
   ```bash
   ./physical_robot/fleet_manager.sh ping
   ```

---

### 5.2 Onboard C Socket Daemon Deployment
The physical Khepera IV runs Poky Yocto Linux with an embedded ARM Cortex-A8 processor. The lightweight C daemon [`khepera4_tcp_server.c`](file:///home/andy/Desktop/khepera4_ws/physical_robot/onboard_daemon/khepera4_tcp_server.c) directly commands motors and queries encoders and proximity sensors via `libkhepera`.

#### Automated One-Click Deployment via `fleet_manager.sh`:
```bash
# Deploys C source, compiles on robot with onboard GCC, and enables systemd service
./physical_robot/fleet_manager.sh deploy

# Query daemon status across the fleet
./physical_robot/fleet_manager.sh status
```

#### Manual Deployment via SSH (if deploying to a single robot):
```bash
# 1. Copy files to robot root home
scp physical_robot/onboard_daemon/khepera4_tcp_server.c \
    physical_robot/onboard_daemon/Makefile \
    root@192.168.1.101:/home/root/

scp physical_robot/onboard_daemon/khepera4_server.service \
    root@192.168.1.101:/lib/systemd/system/

# 2. SSH into robot and build native binary
ssh root@192.168.1.101 "cd /home/root && make clean && make && chmod +x khepera4_tcp_server"

# 3. Enable and start systemd autostart service
ssh root@192.168.1.101 "systemctl daemon-reload && systemctl enable khepera4_server && systemctl restart khepera4_server"
```

---

### 5.3 Connecting to One Real Robot in the Lab (Step-by-Step Terminal Sequence)

Follow this 5-step sequence when working with **one real robot** on the lab floor:

#### Terminal 1: Launch the Hardware Bridge Node
Connects to the robot's TCP socket, converts ROS 2 velocity commands into K-Team motor pulses, and broadcasts odometry and TF frames:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

ros2 run khepera4_control khepera4_hardware_bridge --ros-args \
    -p robot_name:=khepera_1 \
    -p role:=base_link \
    -p robot_ip:=192.168.1.101 \
    -p robot_port:=5000 \
    -p publish_tf:=true
```

#### Terminal 2: Echo & Monitor Real Sensor Topics
Verify that live odometry and sensor data are actively streaming over Wi-Fi:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# Check active topics published by hardware bridge
ros2 topic list | grep khepera_1

# Echo live wheel odometry (x, y, orientation yaw)
ros2 topic echo /khepera_1/odom

# Echo 8 infrared proximity distance readings (meters)
ros2 topic echo /khepera_1/ir_sensors

# Echo 5 ultrasonic sensor distance readings (meters)
ros2 topic echo /khepera_1/us_sensors

# Echo battery telemetry (voltage in Volts, percentage)
ros2 topic echo /khepera_1/battery_state

# Echo wheel joint states (encoder positions in radians)
ros2 topic echo /khepera_1/joint_states
```

#### Terminal 3: Launch Interactive Control / Mapping Dashboard
Provides a graphical UI with live speed gauges, battery health, and tactile D-pad teleoperation:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# Launch the general control dashboard
ros2 run khepera4_control control_dashboard

# Or launch the dedicated 2D/2.5D mapping dashboard
ros2 run khepera4_control mapping_dashboard
```

#### Terminal 4: View Real-Time Robot Model & Map in RViz2
Visualizes the physical robot's TF transforms, odometry trace, and real-time sensor scans:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

rviz2 -d $(ros2 pkg prefix --share khepera4_description)/config/khepera4_mapping.rviz
```

#### Terminal 5: Drive Real Robot via Keyboard Teleop
Safely command the real robot across the floor:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/khepera_1/cmd_vel
```
> [!NOTE]
> **Safety Watchdog:** The onboard C daemon incorporates a **250 ms hardware timeout**. If no velocity packet is received from the Master PC within 250 ms (e.g. Wi-Fi dropped or teleop exited), the motors immediately brake to a halt.

---

### 5.4 Multi-Robot Physical Fleet Deployment
To launch the entire 3-robot formation fleet with hardware bridges and the formation swarm coordinator:
```bash
# Verify all 3 units are powered on and pingable
./physical_robot/fleet_manager.sh ping

# Launch multi-robot physical station with Qt Swarm GUI
ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical gui:=true
```

---

## 📡 6. Key Runtime Services, Diagnostics & Topic Inspection

Run these commands in an active terminal (`source /opt/ros/jazzy/setup.bash && source install/setup.bash`):

### 6.1 Map Memory & Persistence Services
```bash
# Save current SLAM map to disk (generates maps/khepera_map.pgm, .yaml, and .png)
ros2 service call /save_map std_srvs/srv/Trigger "{}"

# Reload previously saved map from disk into memory
ros2 service call /load_map std_srvs/srv/Trigger "{}"

# Reset map memory and clear occupancy grid
ros2 service call /clear_map std_srvs/srv/Trigger "{}"

# Recenter robot and reset tip-over recovery state
ros2 service call /recenter_robot std_srvs/srv/Trigger "{}"
```

### 6.2 Autonomous Exploration Control Services
```bash
# Toggle autonomous exploration on / off dynamically
ros2 service call /toggle_explorer std_srvs/srv/Trigger "{}"

# Explicitly enable autonomous explorer
ros2 service call /set_explorer std_srvs/srv/SetBool "{data: true}"

# Explicitly disable autonomous explorer (halt autonomous drive)
ros2 service call /set_explorer std_srvs/srv/SetBool "{data: false}"
```

### 6.3 Real-Time Topic Rates & Telemetry Monitoring
```bash
# Monitor fused odometry publication rate (~50 Hz expected)
ros2 topic hz /khepera4/odom

# Monitor 360° virtual LaserScan rate (~10 Hz expected)
ros2 topic hz /khepera4/scan

# Monitor occupancy grid publication rate (~3 Hz expected)
ros2 topic hz /map

# Monitor onboard camera stream rate (~30 Hz expected)
ros2 topic hz /camera/image_raw

# Inspect current robot Euclidean pose
ros2 topic echo /khepera4/odom --field pose.pose

# Monitor cliff edge alert and rollover hazard flags
ros2 topic echo /cliff_alert
ros2 topic echo /khepera4/turned_over
```

---

## 🧹 7. Process Cleanup & Simulation Reset

If simulation processes, Gazebo daemons, or ROS nodes remain bound in the background from previous sessions:

```bash
# Option 1: Use the workspace cleanup utility script
./scripts/clean_sim.sh

# Option 2: Terminate all simulation, bridge, and UI processes directly
pkill -9 -f "gz sim|ros_gz|robot_state_publisher|parameter_bridge|rviz2|khepera4|mapping_dashboard|reactive_explorer" 2>/dev/null || true
```

For containerized sessions:
```bash
# Purge running Docker containers and remove orphaned networks
docker compose -f docker/docker-compose.yml down --remove-orphans
```
