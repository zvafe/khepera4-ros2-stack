# BRT K4: Khepera IV Autonomous Robotics & Multi-Agent Swarm Stack

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy_Jalisco-343434.svg?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic_8.15-orange.svg?logo=gazebo&logoColor=white)](https://gazebosim.org/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04_LTS-E95420.svg?logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/noble/)
[![Tests Passing](https://img.shields.io/badge/Verification-223%20Passed-brightgreen.svg)](#-verification--automated-tests)
[![License Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An enterprise-grade, cyber-physical robotics research platform for the **K-Team Khepera IV** mobile robot. Built on **ROS 2 Jazzy** and **Gazebo Harmonic**, this repository unifies single-robot closed-loop Lyapunov cascade control, multi-sensor 2D/2.5D SLAM, autonomous frontier exploration, and a synchronized 3-robot formation fleet with real-time hardware bridges for physical robot deployment.

---

## 🚀 30-Second Quickstart

### Option 1: Universal Docker Launcher (Recommended)
Launch instantly without configuring local ROS 2 dependencies:
```bash
./run.sh
```
* **Single Robot 2D/2.5D SLAM & Exploration:** `./run.sh --mapping`
* **3-Robot Swarm Coordination Fleet:** `./run.sh --multi`
* **Native GUI (GPU / WSLg / X11 pass-through):** `./run.sh --native`
* **Web Browser (noVNC HTML5 virtual desktop on port 6080):** `./run.sh --web`

### Option 2: Native Host Execution (Linux / WSL2)
```bash
# 1. Source ROS 2 underlay
source /opt/ros/jazzy/setup.bash

# 2. Build workspace and source overlay
colcon build --symlink-install
source install/setup.bash

# 3. Export Gazebo resource paths
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH}

# 4. Launch the SLAM & Autonomous Exploration Station
ros2 launch khepera4_control mapping.launch.py auto_explore:=true
```

> 📖 **Full Command Reference:** For all launch permutations, arguments, headless flags, and runtime service inspection, see [`SIMULATION_COMMANDS.md`](file:///home/andy/Desktop/khepera4_ws/SIMULATION_COMMANDS.md).

---

## 🖥️ Primary Stacks & Verified Launch Reference

| Pipeline / Stack | Execution Command | Key Parameters & Flags | Subsystem Description |
|---|---|---|---|
| **2D/2.5D SLAM & Exploration** | `ros2 launch khepera4_control mapping.launch.py` | `auto_explore:=true\|false`<br>`dashboard:=true\|false`<br>`rviz:=true\|false`<br>`headless:=true\|false` | Single-robot spatial mapping pipeline with probabilistic occupancy `/map`, 2.5D elevation `/elevation_map`, cliff hazard warning `/cliff_alert`, and tactile D-Pad teleop. |
| **3-Robot Swarm Station** | `ros2 launch khepera4_control multi_robot_station.launch.py` | `mode:=sim\|physical\|mock`<br>`gui:=true\|false`<br>`headless:=true\|false` | Coordinates Anchor (`khepera_1`), Relay (`khepera_2`), and Scout (`khepera_3`) with live RF midpoint tethering and triangle formations. |
| **Educational Control Station** | `ros2 launch khepera4_control control_station.launch.py` | `headless:=true\|false` | Interactive BSA student station for Mode A (WASD), Mode B (Lyapunov setpoint step response), Mode C (trajectories). |
| **Physical Fleet Connect** | `ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical` | `gui:=true` | Connects master PC directly to physical robot IP addresses (`192.168.1.101..103`) via TCP command sockets. |
| **Kinematic Validation Benchmark** | `ros2 launch khepera4_control benchmark.launch.py` | `maneuver:=circle`<br>`linear_speed:=0.15` | Validates unicycle kinematics and calculates physical odometry drift across timed pulses. |

---

## 🏛️ System Architecture & Computational Flow

```
                                      +------------------------------------+
                                      |        Gazebo Harmonic Sim /       |
                                      |      Physical Hardware Bridge      |
                                      +------------------------------------+
                                              │                │
                       /odom (raw wheel odometry)              │ /imu/data, /khepera4/ir_*/scan,
                                              │                │ /khepera4/us_*/scan, ground_ir_*
                                              ▼                ▼
+-----------------------+   cmd_vel   +------------------------------------+
|   ReactiveExplorer    | ----------> |          Khepera4Mapper            |
|   (Braitenberg avoid, |             |  (Slip projection, Wall clamping,  |
|   Corner stall escape)|             |  Bresenham log-odds, Elevation map)|
+-----------------------+             +------------------------------------+
       ▲       ▲                                 │                │
       │       │                       /khepera4/odom             │ /map, /elevation_map,
       │   /cliff_alert,                         │                │ /surface_map, /khepera4/scan
       │   /khepera4/turned_over                 ▼                ▼
       │                               +------------------------------------+
       +------------------------------ |     MappingDashboard (Qt5/MPL)     |
              /explorer/enable         +------------------------------------+

[ Multi-Robot Swarm Subsystem (multi_robot_station.launch.py) ]
  /khepera_1/odom, /khepera_2/odom, /khepera_3/odom
                 │
                 ▼
     +-----------------------+     /khepera_*/cmd_vel     +-----------------------------------+
     |    SwarmCoordinator   | ------------------------>  | ClosedLoopController (per robot)  |
     | (Midpoint RF tether,  |                            | (Cascade Lyapunov feedback        |
     | Triangle formation)   |                            |  regulation: rho, alpha, beta)    |
     +-----------------------+                            +-----------------------------------+
```

### Verified Topic Contracts
* **Velocity Commands:** `/cmd_vel` or `/<robot_name>/cmd_vel` (`geometry_msgs/msg/Twist`)
* **Raw Wheel Odometry:** `/odom` or `/<robot_name>/odom` (`nav_msgs/msg/Odometry`)
* **Corrected / Slip-Projected Odometry:** `/khepera4/odom` (`nav_msgs/msg/Odometry`)
* **Fused Proximity Scan (360°):** `/khepera4/scan` (`sensor_msgs/msg/LaserScan`)
* **2D Obstacle Occupancy Grid:** `/map` (`nav_msgs/msg/OccupancyGrid`, transient-local QoS)
* **2.5D Slope Elevation Grid:** `/elevation_map` (`nav_msgs/msg/OccupancyGrid`, transient-local QoS)
* **Surface Roughness & Cliff Grid:** `/surface_map` (`nav_msgs/msg/OccupancyGrid`, transient-local QoS)
* **Inertial Measurement:** `/imu/data` or `/<robot_name>/imu/data` (`sensor_msgs/msg/Imu`)
* **Hazard Alerts:** `/cliff_alert` (`std_msgs/msg/Bool`), `/khepera4/turned_over` (`std_msgs/msg/Bool`)
* **Fleet Telemetry:** `/swarm/status` (`std_msgs/msg/String`, JSON), `/swarm/cmd` (`std_msgs/msg/String`)

---

## 📚 Unified Documentation Portal

Comprehensive documentation is organized within the [`docs/`](file:///home/andy/Desktop/khepera4_ws/docs) directory:

1. 📖 **[Developer Onboarding & Quickstart Guide](file:///home/andy/Desktop/khepera4_ws/docs/quickstart_onboarding.md):** Complete prerequisites, Linux & Windows WSL2/WSLg setup, workspace build, parameter reference, and smoke testing.
2. 🐳 **[Docker & Containerized Simulation Guide](file:///home/andy/Desktop/khepera4_ws/docs/docker_and_simulation.md):** Detailed Docker Compose lifecycle, GPU hardware acceleration, Native GUI vs noVNC Web, and display forwarding.
3. 🤖 **[Multi-Robot Architecture & Hardware Manual](file:///home/andy/Desktop/khepera4_ws/docs/multi_robot_and_hardware.md):** Swarm fleet topology, TF2 tree, C socket daemon for Yocto Linux, systemd autostart, and `fleet_manager.sh`.
4. 📐 **[Control Systems Research Charter](file:///home/andy/Desktop/khepera4_ws/docs/control_systems_charter.md):** BSA academic control theory, unicycle kinematics, Lyapunov cascade regulation, DC motor models, and transient metrics.
5. 🧪 **[Verification & Automated Testing Guide](file:///home/andy/Desktop/khepera4_ws/docs/testing_and_verification.md):** 4-tier testing hierarchy, 223-test catalog, opaque-box methodology, and test execution commands.
6. 🧠 **[AI Agent & Vibecoder Context Specification](file:///home/andy/Desktop/khepera4_ws/docs/AGENT_NOTES.md):** Critical architectural invariants, directory ontology, "Do Not Touch" files, and known edge cases to ground future AI sessions.

---

## 🧪 Verification & Automated Tests

The repository maintains an automated opaque-box test harness comprising **223 tests** across 10 specialized modules:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
colcon test --packages-select khepera4_control
colcon test-result --verbose
```
```text
Summary: 1 package finished [10.9s]
Summary: 223 tests, 0 errors, 0 failures, 0 skipped
```

Or run via pytest directly:
```bash
python3 -m pytest src/khepera4_control/test/ -v
```

---

## 👥 Authors & Academic Context

* **Laboratory:** Mobile Robotics Laboratory, Department of Automatic Control and Systems Engineering
* **Course:** Bazele Sistemelor Automate (BSA)
* **Lead Maintainer:** (`zvafe`) - `andrei@brt-labs.tech`
* **Hardware Manufacturer:** K-Team S.A., Vallorbe, Switzerland
