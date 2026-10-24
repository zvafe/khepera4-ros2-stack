# BRT K4: Khepera IV ROS 2 Jazzy & Gazebo Harmonic Stack

[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy_Jalisco-343434?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic_8.15-FF6F00?logo=gazebo&logoColor=white)](https://gazebosim.org/docs/harmonic)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04_LTS-E95420?logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/24.04/)
[![Repository](https://img.shields.io/badge/GitHub-zvafe%2Fkhepera4--ros2--stack-blue?logo=github)](https://github.com/zvafe/khepera4-ros2-stack)

An academic and industrial-grade cyber-physical simulation and control stack for the **K-Team Khepera IV** differential-drive mobile robot. Built natively on **Ubuntu 24.04 LTS (Noble Numbat)**, **ROS 2 Jazzy Jalisco**, and **Gazebo Harmonic (v8.15+)**.

Project Codename: **BRT K4**  
Official Repository: [`zvafe/khepera4-ros2-stack`](https://github.com/zvafe/khepera4-ros2-stack)  
Academic Charter: *Bazele Sistemelor Automate (BSA)*, Faculty of Automatic Control and Computers, POLITEHNICA Bucharest.

---

## 📑 Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture & Computation Graph](#system-architecture--computation-graph)
3. [Multi-Team Integration & Modular Architecture](#multi-team-integration--modular-architecture)
   - [1. Core Modeling & Closed-Loop Control (Andrei Berteanu)](#1-core-modeling--closed-loop-control-andrei-berteanu)
   - [2. Robotic Arm Team](#2-robotic-arm-team)
   - [3. Swarm Team](#3-swarm-team)
   - [4. Neural Network / AI Team](#4-neural-network--ai-team)
4. [Running the Simulation](#running-the-simulation)
   - [Interactive Launcher (`./run.sh`)](#interactive-launcher-runsh)
   - [Mode A: Native GUI (WSLg / GPU Pass-Through)](#mode-a-native-gui-wslg--gpu-pass-through)
   - [Mode B: Web Browser (noVNC for Low-Spec / Academic PCs)](#mode-b-web-browser-novnc-for-low-spec--academic-pcs)
   - [Native Host Execution (No Docker)](#native-host-execution-no-docker)
   - [Interactive Educational Guide](#interactive-educational-guide)
5. [Configuration via `.env` (Port Conflict Safeguards)](#configuration-via-env-port-conflict-safeguards)
6. [Physical Robot Deployment (Offboard ROS 2 Bridge)](#physical-robot-deployment-offboard-ros-2-bridge)
7. [Directory Structure](#directory-structure)
8. [Git Workflow & Upstream Collaboration](#git-workflow--upstream-collaboration)

---

## 🔍 Executive Summary

The **BRT K4** stack establishes a bridge between continuous-time automation control theory and modern cyber-physical robotics. It provides:
- **First-Principles Kinematic & Dynamic Modeling:** Rigorous unicycle kinematics paired with DC motor electromechanical dynamics and Coulomb/viscous friction modeling.
- **Hierarchical Closed-Loop Control:** Inner-loop motor regulation coupled to outer-loop non-linear posture tracking with Lyapunov stability criteria.
- **High-Fidelity Virtual Prototyping:** Realistic Gazebo Harmonic simulation with exact physical dimensions ($r = 0.021\text{ m}, L = 0.1054\text{ m}, m = 0.540\text{ kg}$), passive low-friction casters ($\mu = 0.0001$), ground drive-wheel contact ($\mu = 1.0$), and calibrated sensors (CMOS camera, 6-DOF IMU, 8 IR sensors, 5 ultrasonic transceivers).
- **Multi-Modal Execution:** Flexible deployment through native Linux desktops, Windows 11 WSLg GPU acceleration, or zero-install HTML5 web browser interfaces via noVNC.

---

## 🏛️ System Architecture & Computation Graph

The simulation leverages bidirectional communication between ROS 2 Jazzy and Gazebo Harmonic managed via `ros_gz_bridge`:

```
+---------------------------------------------------------------------------------------+
|                                    ROS 2 COMPUTATION GRAPH                             |
|                                                                                       |
|   +--------------------------+       /cmd_vel       +-----------------------------+   |
|   |  closed_loop_controller  | -------------------> |    ros_gz_bridge Node       |   |
|   |   (Lyapunov / Cascade)   |                      | (ros_gz_sim parameter_bridge) | |
|   +--------------------------+                      +-----------------------------+   |
|                ^                                                   |                  |
|                |                                                   | gz.msgs.Twist    |
|                | /odom                                             v                  |
|   +--------------------------+                      +-----------------------------+   |
|   |     telemetry_logger     |                      |       GAZEBO HARMONIC       |   |
|   |  (e(t), tr, ts, sigma%)  |                      |                             |   |
|   +--------------------------+                      |  Model: 'khepera4'          |   |
|                ^                                    |  DiffDrive System Plugin    |   |
|                |                                    |  JointStatePublisher Plugin |   |
|   +--------------------------+                      |  DART Physics Engine        |   |
|   +--------------------------+                      |  Sensors Plugin (Ogre2)     |   |
|   |    control_dashboard     |                      |  DART Physics Engine        |   |
|   |  (Qt Real-Time Telemetry)|                      +-----------------------------+   |
|   +--------------------------+                                     |                  |
|                ^                                                   | gz.msgs.Odometry |
|                |                                                   | gz.msgs.Image    |
|   +--------------------------+                                     | gz.msgs.IMU      |
|   |     khepera4_mapper      | <-----------------------------------+ gz.msgs.LaserScan|
|   | (Occupancy / Elevation / |                                     v                  |
|   |   Surface / Persistence) |                                                        |
|   +--------------------------+                                                        |
|                                                                                       |
|   Topics Bridged:                                                                     |
|   - /clock                      [rosgraph_msgs/msg/Clock] <-------+                   |
|   - /cmd_vel                    [geometry_msgs/msg/Twist]                             |
|   - /odom                       [nav_msgs/msg/Odometry]   <-------+                   |
|   - /joint_states               [sensor_msgs/msg/JointState] <----+                   |
|   - /camera/image_raw           [sensor_msgs/msg/Image]   <-------+                   |
|   - /imu/data                   [sensor_msgs/msg/Imu]     <-------+                   |
|   - /tf                         [tf2_msgs/msg/TFMessage]  <-------+                   |
|   - /khepera4/us_0..4/scan      [sensor_msgs/msg/LaserScan] (5 Sonar Transducers) <---+
|   - /khepera4/ir_0..7/scan      [sensor_msgs/msg/LaserScan] (8 IR Proximity Sensors) <+
|   - /khepera4/ground_ir_0..3/scan [sensor_msgs/msg/LaserScan] (4 Cliff/Line Sensors) <+
+---------------------------------------------------------------------------------------+
```

---

## 🤝 Multi-Team Integration & Modular Architecture

BRT K4 is engineered as an extensible modular robotics platform designed for parallel development across interdisciplinary engineering teams.

```
                    +-------------------------------------+
                    |       BRT K4 Base Robot Core        |
                    |   (URDF / Meshes / Gazebo World)    |
                    +-------------------------------------+
                                       |
        +------------------+-----------+-----------+------------------+
        |                  |                       |                  |
        v                  v                       v                  v
+---------------+  +---------------+       +---------------+  +---------------+
|  Core Control |  |  Robotic Arm  |       |   Swarm Team  |  | AI / Neural   |
|  & BSA Theory |  |  Integration  |       |  Multi-Robot  |  | Vision & RL   |
| (A. Berteanu) |  | Modular Mount |       |  Namespacing  |  | Perception    |
+---------------+  +---------------+       +---------------+  +---------------+
```

### 1. Core Modeling & Closed-Loop Control (Andrei Berteanu)
- **Theoretical Foundations:** Complies with the *Bazele Sistemelor Automate (BSA)* academic charter led by Prof. Ioan Dumitrache.
- **Actuator Transfer Function ($H_m(s)$):** Derives continuous-time DC motor dynamics:
  $$u(t) = R_a i_a(t) + L_a \frac{d i_a(t)}{dt} + K_e \omega_m(t)$$
  $$J_{eq} \frac{d \omega_m(t)}{dt} + b_{eq} \omega_m(t) = K_t i_a(t) - \tau_L(t)$$
  Neglecting small armature inductance ($L_a \approx 0$):
  $$H_m(s) = \frac{\Omega_m(s)}{U(s)} = \frac{K_m}{\tau_m s + 1}$$
- **Cascade Feedback Structure:**
  - *Inner Loop ($\sim 100\text{ Hz}$):* Direct wheel speed feedback $(\omega_R, \omega_L)$ rejecting friction and battery voltage degradation.
  - *Outer Loop ($\sim 20\text{ Hz}$):* Non-linear posture regulator transforming Cartesian tracking errors $(e_x, e_y, e_\theta)$ into polar coordinates $(\rho, \alpha, \beta)$:
    $$\rho = \sqrt{\Delta x^2 + \Delta y^2}, \quad \alpha = -\theta + \text{atan2}(\Delta y, \Delta x), \quad \beta = -\theta - \alpha + \theta_{goal}$$
    Control Law:
    $$v = k_\rho \rho \cos\alpha, \quad \omega = k_\alpha \alpha + k_\beta \beta$$
- **Stability & Performance Metrics:** Evaluated against optimal second-order damping ratio $\zeta = \frac{\sqrt{2}}{2} \approx 0.707$ for an overshoot $\sigma\% \le 4.3\%$, with phase margin $\gamma_k \ge 60^\circ$ and real-time transient telemetry logging ($t_r, t_s, \sigma\%, e_{st}$).

### 2. Robotic Arm Team (Modular Upper Chassis Interface)
- **Mechanical Mounting Interface:** The upper shell of the Khepera IV (`base_link` at $Z = 0.0538\text{ m}$) provides dedicated rigid attachment frames (`arm_base_mount_joint` and `arm_mount_link`).
- **Inertial & Payload Specifications:**
  - Maximum recommended arm payload: $300\text{ g}$ (maintains total weight below $850\text{ g}$).
  - Dynamic Center of Mass (CoM): The base inertia tensor ($I_{xx} = I_{yy} = 0.000693\text{ kg}\cdot\text{m}^2, I_{zz} = 0.001127\text{ kg}\cdot\text{m}^2$) is parametrized to automatically account for dynamic end-effector reaction torques.
- **Integration Points:**
  - URDF extension point via `<xacro:include filename="khepera4_arm.xacro" />`.
  - Joint trajectory controller integration bridging MoveIt 2 with the mobile base.

### 3. Swarm Fleet Team (3-Robot Multi-Agent Architecture)
- **Dedicated 3-Robot Fleet Architecture:**
  - Standardized multi-robot roles with isolated namespaces and TF coordinate trees:
    - **`khepera_1` (`base_link`):** Base anchor, coordinator, and origin reference frame.
    - **`khepera_2` (`extender`):** Dynamic chain relay tracking the midpoint between Base and Explorer to maintain Wi-Fi connectivity.
    - **`khepera_3` (`explorer`):** Autonomous scout exploring the frontier under tether range constraints.
  - Isolated topic trees:
    - `/<robot_name>/cmd_vel` (`geometry_msgs/msg/Twist`)
    - `/<robot_name>/odom` (`nav_msgs/msg/Odometry`)
    - `/<robot_name>/joint_states` (`sensor_msgs/msg/JointState`)
    - `/<robot_name>/camera/image_raw` (`sensor_msgs/msg/Image`)
    - `/<robot_name>/imu/data` (`sensor_msgs/msg/Imu`)
    - `/<robot_name>/battery_state` (`sensor_msgs/msg/BatteryState`)
  - **TF Prefixing & Global Registration:**
    - Each robot publishes scoped frames: `<robot_name>/odom` $\to$ `<robot_name>/base_footprint` $\to$ `<robot_name>/base_link` $\to$ `<robot_name>/<role>`.
    - All 3 robots register into the unified global `world` frame for simultaneous multi-robot tracking in RViz2 and Gazebo.
  - **Single Command Launch:**
    ```bash
    ./run.sh --multi          # Launch 3-robot Gazebo sim + coordinator + dashboard
    ./run.sh --multi --web    # Launch 3-robot stack in HTML5 web browser
    ```
  - **Comprehensive Operations Manual:** Full hardware and simulation deployment instructions are documented in [`MULTI_ROBOT_MANUAL.md`](MULTI_ROBOT_MANUAL.md).

### 4. Neural Network / AI Team (Standardized AI Interfaces)
- **Observation Space (Standard ROS 2 Topics):**
  | Modality | Topic | Message Type | Specifications |
  | :--- | :--- | :--- | :--- |
  | **RGB Vision** | `/camera/image_raw` | `sensor_msgs/msg/Image` | WVGA ($640 \times 480$), 30 FPS, $60^\circ$ HFOV |
  | **Inertial IMU** | `/imu/data` | `sensor_msgs/msg/Imu` | 6-DOF (Linear Accel + Angular Vel) @ 100 Hz |
  | **Odometry** | `/odom` | `nav_msgs/msg/Odometry` | Ground-truth state estimation @ 50 Hz |
  | **IR Proximity Ring** | `/ir_sensors` | `sensor_msgs/msg/Range` | 8 bumper sensors configured radially |
  | **Ultrasonic Sonar** | `/us_sensors` | `sensor_msgs/msg/Range` | 5 ultrasonic distance transceivers |
  | **Floor Ground IR**| `/ground_sensors`| `sensor_msgs/msg/Range` | 4 downward cliff/line-tracking sensors |
- **Action Space:**
  - Standard velocity control: `geometry_msgs/msg/Twist` on `/cmd_vel` ($v \in [-0.25, 0.25]\text{ m/s}$, $\omega \in [-1.8, 1.8]\text{ rad/s}$).
- **AI Framework Compatibility:** Drop-in compatibility with Gymnasium environments, PyTorch reinforcement learning pipelines (PPO, SAC), Isaac Lab bridges, and Vision-Language-Action (VLA) navigation models.

### 5. Multi-Sensor Perception & Autonomous Mapping Engine
- **Full Physical Sensor Suite Virtualization:**
  - **8 IR Ring Sensors:** High-resolution close-proximity detection ($0.002\text{ m} - 0.25\text{ m}$) configured around the blue chassis bumper at $45^\circ$ angular intervals.
  - **5 Ultrasonic Sonar Transceivers:** Medium-range acoustic ranging ($0.20\text{ m} - 2.0\text{ m}$) covering front and lateral sectors ($-90^\circ, -45^\circ, 0^\circ, +45^\circ, +90^\circ$).
  - **4 Underbody Ground IR Sensors:** Downward-looking proximity ($0.001\text{ m} - 0.06\text{ m}$) for table edge / cliff detection ($d > 0.025\text{ m}$) and floor surface tracking.
  - **6-DOF IMU:** High-frequency linear acceleration and gyro angular rates on `imu_link`.
- **Fused 360° Virtual LaserScan (`/khepera4/scan`):**
  - Seamlessly combines the 8 IR and 5 Ultrasonic beams into a 360-sample radial scan with exact kinematic offsets.
- **Probabilistic 2D Occupancy Grid (`/map`):**
  - High-performance Bresenham 2D ray casting with log-odds belief updates ($l_{free} = 0.40, l_{occ} = 0.85$, $0.03\text{ m}$ resolution, $200 \times 200$ cells).
- **2.5D Terrain Elevation & Inclination Map (`/elevation_map`):**
  - Dynamically computes robot pitch and roll angles from the IMU gravity vector, mapping slopes and ramps across traversed cells.
- **Surface & Cliff Hazard Map (`/surface_map`, `/cliff_alert`):**
  - Detects floor drop-offs and pits, triggering emergency stop/backup maneuvers, while evaluating vertical vibration variance ($\sigma(a_z)$) for terrain roughness.
- **Persistent Map Memory:**
  - Supports on-demand and automatic map persistence: exports standard ROS `khepera_map.pgm` + `khepera_map.yaml` and high-resolution colored visualization `khepera_map.png` via `/save_map`, with `/load_map` and `/clear_map` services.
- **Autonomous Reactive Explorer (`reactive_explorer`):**
  - Braitenberg potential-field obstacle avoidance, harmonic wander steering, and cliff safety rollback.
- **Single Command Launch:**
  ```bash
  ./run.sh --mapping          # Launch simulation + mapping engine + dashboard
  ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true
  ```

---

## 🚀 Running the Simulation

BRT K4 supports seamless execution across all computing platforms through a unified launcher script.

### Interactive Launcher (`./run.sh`)

Ensure executable permissions are set:
```bash
chmod +x run.sh
./run.sh
```

On first boot, if `.env` does not yet exist, `./run.sh` interactively prompts you to choose the default mode and remembers your selection:
```text
======================================================================
  BRT K4: Khepera IV Simulation Launcher (ROS 2 Jazzy & Harmonic)
======================================================================
No configuration found. Please select your default simulation mode:
  1) Capable PC - Native GUI (WSLg / GPU pass-through)
  2) Web Browser - Quick demo (noVNC on http://localhost:6080)
Select [1-2]: 
```

#### Command-Line Flags
You can bypass prompts or override default settings anytime using CLI flags:
```bash
./run.sh --mapping  # Launch Multi-Sensor Exploration & Persistent 2D/2.5D Mapping
./run.sh --multi    # Launch 3-robot swarm fleet simulation and control station
./run.sh --native   # Launch directly in Native GUI mode (WSLg / X11 GPU)
./run.sh --web      # Launch directly in Web Browser mode (noVNC)
./run.sh --build    # Rebuild Docker container images
./run.sh --config   # Reconfigure default simulation mode and web port
./run.sh --help     # Display complete usage cheat sheet
```

---

### Mode A: Native GUI (WSLg / GPU Pass-Through)
*Recommended for capable desktop workstations, Linux laptops, and Windows 11 with WSLg.*

- Mounts host `/tmp/.X11-unix` and GPU device nodes (`/dev/dri`).
- Provides 60+ FPS native hardware-accelerated rendering with direct Gazebo Harmonic graphical UI and Qt dashboard windows.

Launch command:
```bash
./run.sh --native
```

---

### Mode B: Web Browser (noVNC for Low-Spec / Academic PCs)
*Recommended for student laptops, low-spec systems without dedicated GPUs, remote headless servers, or quick classroom presentations.*

- Runs a lightweight virtual display server (`Xvfb`), window manager (`fluxbox`), VNC server (`x11vnc`), and HTML5 WebSocket bridge (`novnc`).
- Accessible from any web browser without installing any software or graphics drivers.

Launch command:
```bash
./run.sh --web
```

Open in your browser:
```text
http://localhost:6080
```
*(Or your custom configured port, e.g., `http://localhost:8085`)*

---

### Native Host Execution (No Docker)

If you have ROS 2 Jazzy and Gazebo Harmonic installed natively on Ubuntu 24.04:

```bash
# 1. Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# 2. Build the workspace
cd ~/Desktop/khepera4_ws
colcon build --symlink-install
source install/setup.bash

# 3. Launch Gazebo Simulation & ROS 2 Bridge
ros2 launch khepera4_description khepera4_sim.launch.py

# 4. In a separate terminal, launch the Closed-Loop Controller
source install/setup.bash
ros2 launch khepera4_control closed_loop.launch.py mode:=step goal_x:=1.0 goal_y:=0.5

# 5. In a separate terminal, launch the Control Station Dashboard
source install/setup.bash
ros2 launch khepera4_control control_station.launch.py
```

---

### 📚 Interactive Educational Guide

An interactive, browser-based standalone educational simulator is included in the workspace:
- Explore differential kinematics, instantaneous center of curvature (ICC), and wheel speed dynamics live in 2D.
- Drag damping ratio ($\zeta$) and natural frequency ($\omega_n$) sliders to synthesize step responses and analyze Bode plots.
- Drive the robot in a 2D canvas arena or test waypoint guidance with real-time error plotting.

Open directly in Chrome or Firefox:
```bash
xdg-open khepera4_guide/index.html
```
Or start a local web server:
```bash
python3 -m http.server 8080 -d khepera4_guide/
# Navigate to http://localhost:8080
```

---

## ⚙️ Configuration via `.env` (Port Conflict Safeguards)

Configuration is managed via `.env` in the workspace root. A tracked template is provided in [`.env.example`](.env.example):

```bash
# ==============================================================================
# BRT K4 Khepera IV - Environment Configuration
# ==============================================================================

# Simulation default mode: 'native' (WSLg / GPU) or 'web' (noVNC HTML5)
KHEPERA_DEFAULT_MODE=native

# Web Browser noVNC port (Default: 6080)
# IMPORTANT: Avoid port 1947 to prevent conflicts with Sentinel HASP License Manager!
KHEPERA_WEB_PORT=6080

# ROS 2 Domain ID for multi-robot / multi-machine DDS isolation
ROS_DOMAIN_ID=42
```

> [!WARNING]
> **Avoid Port 1947 (Sentinel HASP / SafeNet Conflict):**  
> Port `1947` is strictly reserved by Sentinel HASP / SafeNet License Manager (frequently active in academic engineering labs for software licensing like MATLAB, SolidWorks, or ANSYS). The default noVNC port is set to `6080`. If `6080` is in use, modify `KHEPERA_WEB_PORT` in your `.env` file (e.g. to `6081` or `8085`).

---

## 🤖 Physical Robot Deployment (Offboard ROS 2 Bridge)

The identical control software developed in this repository deploys onto the physical K-Team Khepera IV robot through an **offboard distributed ROS 2 bridge architecture**:

```
+-----------------------------------+       Wi-Fi / DDS      +----------------------------------+
|    Host Workstation / Offboard    | <====================> |       Physical Khepera IV        |
|                                   |                        |                                  |
|  - closed_loop_controller (ROS 2) |                        |  - Embedded ARM Cortex-A8 (Linux)|
|  - telemetry_logger               |                        |  - libkhepera C Motor Control    |
|  - control_dashboard / Navigation |                        |  - libkhepera Sensor Transceivers|
|  - offboard_khepera_bridge_node   |                        |  - TCP / UDP Hardware Socket Srv |
+-----------------------------------+                        +----------------------------------+
```

- **Zero-Code Sim-to-Real Portability:** The high-level ROS 2 controller node publishes standard `/<robot_name>/cmd_vel` (`Twist`) and subscribes to `/<robot_name>/odom` (`Odometry`). The simulation and physical hardware expose identical ROS 2 message signatures.
- **Embedded C Daemon (`khepera4_tcp_server.c`):** Runs onboard each robot's Gumstix Overo computer (Yocto Linux 1.8+), directly interacting with dsPIC33 via `libkhepera`. Implements the official K-Team ASCII command set (`D`, `R`, `N`, `G`, `V`, `B`, `Z`) alongside a fail-safe 500ms safety watchdog.
- **Master PC Fleet Manager (`fleet_manager.sh`):**
  ```bash
  ./physical_robot/fleet_manager.sh ping      # Ping and test network latency to all 3 robots
  ./physical_robot/fleet_manager.sh deploy    # Automatically compile & deploy daemon to robots
  ./physical_robot/fleet_manager.sh start     # Start daemon on all 3 robots
  ./physical_robot/fleet_manager.sh status    # Check daemon and battery status
  ```
- **Master PC ROS 2 Bridge Launch:**
  ```bash
  ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical
  ```
- **Sim-to-Real Compensation:** Closed-loop outer regulation with Lyapunov polar error compensation actively rejects real-world floor wheel slip, mechanical deadband, and motor battery voltage decay. Detailed hardware operations are documented in [`MULTI_ROBOT_MANUAL.md`](MULTI_ROBOT_MANUAL.md).

---

## 📂 Directory Structure

```text
khepera4_ws/
├── .devcontainer/
│   └── devcontainer.json            # VS Code remote container configuration
├── docker/
│   ├── Dockerfile                   # Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic + noVNC
│   ├── docker-compose.yml           # Services: khepera-native & khepera-web
│   ├── entrypoint.sh                # Environment initialization & GZ_SIM_RESOURCE_PATH setup
│   └── start-web.sh                 # Headless Xvfb + fluxbox + x11vnc + noVNC startup
├── khepera4_guide/
│   ├── index.html                   # Interactive educational dashboard
│   ├── simulator.js                 # 2D unicycle kinematics & Laplace synthesizer
│   ├── style.css                    # UI styles & dark mode theme
│   └── README.md                    # Educational guide documentation
├── physical_robot/                  # Physical Khepera IV Hardware & Fleet Tools
│   ├── fleet_manager.sh             # Master PC fleet utility (ping, deploy, start, stream)
│   └── onboard_daemon/              # Embedded daemon running on physical robots
│       ├── khepera4_tcp_server.c    # C socket server implementing K-Team protocol & watchdog
│       ├── Makefile                 # Native Khepera IV & Poky cross-compilation Makefile
│       └── khepera4_server.service  # Systemd autostart unit file
├── src/
│   ├── khepera4_control/            # ROS 2 Python Control, Hardware & Swarm Package
│   │   ├── khepera4_control/
│   │   │   ├── closed_loop_controller.py   # Lyapunov polar error closed-loop controller
│   │   │   ├── dashboard.py                # Qt-based live control & multi-robot telemetry station
│   │   │   ├── khepera4_hardware_bridge.py # Master PC TCP socket driver for physical robots
│   │   │   ├── swarm_coordinator.py        # Swarm chain relay & formation coordinator
│   │   │   ├── telemetry_logger.py         # CSV & metrics recorder for step responses
│   │   │   ├── open_loop_benchmark.py      # Open-loop calibration routine
│   │   │   └── common.py                   # Math utilities & coordinate transforms
│   │   └── launch/
│   │       ├── multi_robot_station.launch.py   # Unified 3-robot launcher (sim or physical)
│   │       ├── multi_robot_physical.launch.py  # 3-robot hardware bridge launch
│   │       ├── multi_robot_control.launch.py   # 3-robot closed-loop controller launch
│   │       ├── closed_loop.launch.py           # Single-robot controller execution launch
│   │       ├── control_station.launch.py       # Single-robot simulation + dashboard launch
│   │       └── benchmark.launch.py             # Benchmark routine launch
│   └── khepera4_description/        # ROS 2 Gazebo Model Package
│       ├── meshes/                  # 3D meshes (chassis, wheels, casters)
│       ├── urdf/
│       │   ├── khepera4.urdf.xacro  # Parameterized multi-robot Xacro model
│       │   └── khepera4.urdf        # Complete kinematic, dynamic & sensor URDF
│       ├── worlds/
│       │   └── khepera4_world.sdf   # Gazebo Harmonic physics world
│       └── launch/
│           ├── khepera4_multi_sim.launch.py  # 3-robot Gazebo Harmonic simulation launch
│           └── khepera4_sim.launch.py        # Single-robot simulation launch
├── .env.example                     # Environment template with fleet IP configuration
├── .gitignore                       # Clean Git tracking configuration
├── khepera_iv_control_systems_research_charter.md # Academic control research paper charter
├── Khepera IV User Manual 3.x.pdf   # Official K-Team hardware & software manual
├── MULTI_ROBOT_MANUAL.md            # Comprehensive multi-robot fleet & hardware operations guide
├── README.md                        # Project documentation (this file)
└── run.sh                           # Interactive multi-modal launcher (with --multi support)
```

---

## 🛠️ Git Workflow & Upstream Collaboration

This repository is maintained under the GitHub organization at:
**[https://github.com/zvafe/khepera4-ros2-stack](https://github.com/zvafe/khepera4-ros2-stack)**

### Pushing Your Changes Upstream

To push your initialized repository and commit upstream to `zvafe/khepera4-ros2-stack`:

```bash
# Verify remote configuration
git remote -v

# Push main branch to upstream GitHub
git push -u origin main
```

If you authenticate using GitHub Personal Access Tokens (PAT) or SSH keys:
```bash
# Optional: Switch to SSH remote if configured
git remote set-url origin git@github.com:zvafe/khepera4-ros2-stack.git
git push -u origin main
```

---

## 📄 License & Credits
- **Authors:** Andrei Berteanu (Core Control & Modeling) 

