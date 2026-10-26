# Developer Onboarding & Quickstart Guide
**Project:** BRT K4 Khepera IV ROS 2 Stack  
**Branch:** `strong-coffee` | **ROS Distro:** ROS 2 Jazzy Jalisco | **Simulator:** Gazebo Harmonic 8.15  
**Target Operating Systems:** Linux (Ubuntu 24.04 LTS native) & Windows 11 (WSL2 + WSLg)

---

## 1. Prerequisites & Host Platform Setup

### 1.1 Linux Native (Ubuntu 24.04 LTS Noble Numbat)

1. **Install ROS 2 Jazzy Jalisco Desktop:**
   Follow the [Official ROS 2 Jazzy Installation Guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html). Ensure desktop packages are installed:
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       ros-jazzy-desktop \
       ros-jazzy-ros-gz \
       ros-jazzy-rviz2 \
       ros-jazzy-xacro \
       python3-colcon-common-extensions \
       python3-pyqt5 \
       python3-matplotlib \
       python3-numpy \
       python3-pytest
   ```

2. **Gazebo Harmonic (GZ Sim 8):**
   Included with `ros-jazzy-ros-gz`. Verify installation:
   ```bash
   gz sim --version
   # Expected: Gazebo Sim, version 8.x.x
   ```

3. **X11 Display Permissions (GUI and Visualizations):**
   Ensure local applications and containers can connect to your X11 display:
   ```bash
   xhost +local:root
   ```

---

### 1.2 Windows 11 (WSL2 + WSLg) Setup

Modern Windows 11 supports hardware-accelerated Linux GUI windows via **WSLg**.

1. **Verify WSL2 and Update:**
   Open PowerShell as Administrator:
   ```powershell
   wsl --status
   wsl --update
   ```

2. **Crucial Filesystem Rule (DO NOT CLONE ON `C:\`):**
   > [!CAUTION]
   > **Always clone into the WSL2 ext4 filesystem** (e.g., `/home/<username>/...`).
   > **Never** build ROS 2 workspaces inside `/mnt/c/...`. Building over the Windows NTFS mount causes 10x slower builds, broken symlinks (`--symlink-install` failure), and file-lock corruption.

3. **Install Ubuntu 24.04 in WSL2:**
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

4. **Launch WSL2 Terminal & Follow Linux Steps:**
   Inside your WSL2 shell:
   ```bash
   cd /home/$USER
   git clone <repo_url> khepera4_ws
   cd khepera4_ws
   ```
   WSLg automatically exposes `/tmp/.X11-unix` and `/dev/dri`, allowing Gazebo Harmonic and PyQt5 dashboards to open native hardware-accelerated windows on your Windows desktop.

---

## 2. Workspace Build & Environment Setup

Follow these exact commands to build and source the workspace:

```bash
# 1. Source ROS 2 base underlay
source /opt/ros/jazzy/setup.bash

# 2. Build the workspace using symlink-install
colcon build --symlink-install

# 3. Source the local workspace overlay
source install/setup.bash

# 4. Set Gazebo Resource Path for Khepera IV meshes and SDF worlds
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH}
```

---

## 3. Verified ROS 2 Launch Reference

All launch files are parameterized and tested. Use the table below for exact CLI syntax:

| Stack / Pipeline | CLI Launch Command | Key Arguments & Flags | Notes |
|---|---|---|---|
| **2D/2.5D SLAM & Exploration** | `ros2 launch khepera4_control mapping.launch.py` | `auto_explore:=true\|false`<br>`dashboard:=true\|false`<br>`rviz:=true\|false`<br>`headless:=true\|false` | Primary single-robot exploration pipeline with SLAM, tactile D-Pad, and spatial rasterization. |
| **3-Robot Swarm Station** | `ros2 launch khepera4_control multi_robot_station.launch.py` | `mode:=sim\|physical\|mock`<br>`gui:=true\|false`<br>`headless:=true\|false` | Spawns/connects Anchor (`khepera_1`), Extender (`khepera_2`), and Explorer (`khepera_3`). |
| **Educational Control Station** | `ros2 launch khepera4_control control_station.launch.py` | `headless:=true\|false` | Single-robot BSA educational GUI for Mode A (WASD), Mode B (setpoint step), Mode C (trajectories). |
| **Physical Fleet Connect** | `ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical` | `gui:=true` | Connects master PC to physical robot IPs via TCP sockets. |
| **Lyapunov Controller CLI** | `ros2 launch khepera4_control closed_loop.launch.py` | `goal_x:=1.0`<br>`goal_y:=1.0`<br>`goal_yaw:=0.0` | Headless closed-loop cascade step regulation test with telemetry logging. |
| **Kinematic Benchmark** | `ros2 launch khepera4_control benchmark.launch.py` | `maneuver:=circle`<br>`linear_speed:=0.15` | Open-loop unicycle kinematic validation and drift measurement. |

---

## 4. Running the Verification Test Suite

The workspace features **223 unit, adversarial, and physics simulation tests** covering all subsystems:

### Standard Colcon Test Execution:
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon test --packages-select khepera4_control
colcon test-result --verbose
```
*Expected Result: `Summary: 223 tests, 0 errors, 0 failures, 0 skipped` in ~10.5 seconds.*

### Direct Pytest Execution:
To run tests directly with detailed progress output:
```bash
python3 -m pytest src/khepera4_control/test/ -v
```

### Running Targeted Test Modules:
```bash
# Verify ramp approach physics & lip clearance (40 tests):
python3 -m pytest src/khepera4_control/test/test_ramp_physics.py -v

# Verify bump & tangential slip projection (50 tests):
python3 -m pytest src/khepera4_control/test/test_bump_slip_detection.py -v

# Verify stream prioritization and dashboard consistency (40 tests):
python3 -m pytest src/khepera4_control/test/test_dashboard_consistency.py -v
```

---

## 5. Topic & Service Smoke Testing

Once a launch file is active, verify system health using these diagnostic commands:

```bash
# List active topics across namespaces
ros2 topic list

# Verify SLAM map publishing (3 Hz transient local)
ros2 topic hz /map
ros2 topic echo /map --once

# Inspect corrected vs raw odometry
ros2 topic echo /khepera4/odom --field pose.pose.position
ros2 topic echo /odom --field pose.pose.position

# Inspect 360° virtual laser scan
ros2 topic echo /khepera4/scan --field header

# Check multi-robot swarm status telemetry
ros2 topic echo /swarm/status

# Test map persistence service triggers
ros2 service call /save_map std_srvs/srv/Trigger
ros2 service call /recenter_robot std_srvs/srv/Trigger
```

---

## 6. Troubleshooting Common Pitfalls

1. **"Cannot connect to X server :0"**:
   - Run `xhost +local:root` on the host.
   - On Wayland (Ubuntu 24.04 default), ensure XWayland is active: `echo $DISPLAY` (usually `:0` or `:1`).
2. **Meshes / Wheels Invisible in Gazebo**:
   - Ensure `GZ_SIM_RESOURCE_PATH` is set to the package directory.
3. **Orphaned Gazebo Processes**:
   - If a new simulation fails to spawn, stale Gazebo daemons may hold DDS ports. Clean them up:
     ```bash
     pkill -f "gz sim"
     ```
