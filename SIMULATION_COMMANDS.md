# Khepera IV ROS 2 Simulation & Environment Guide
**Repository:** `khepera4_ws` | **ROS Distro:** ROS 2 Jazzy Jalisco | **Simulator:** Gazebo Harmonic (GZ Sim 8)

This guide provides the complete command reference for rebuilding the workspace using `colcon build --symlink-install` and launching all verified single-robot, multi-robot, SLAM, and benchmarking simulation stacks.

---

## ⚡ 1. Fast Track: Rebuild & Run One-Liner

```bash
source /opt/ros/jazzy/setup.bash && \
colcon build --symlink-install && \
source install/setup.bash && \
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH} && \
ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true
```

---

## 🛠️ 2. Environment Rebuilding & Sourcing Workflow

### 2.1 Standard Rebuild (`--symlink-install`)
Using `--symlink-install` allows you to edit Python scripts, launch files, URDFs, and configurations without rebuilding every time:

```bash
# 1. Source ROS 2 Jazzy Underlay
source /opt/ros/jazzy/setup.bash

# 2. Build All Packages with Symlinks
colcon build --symlink-install

# 3. Source Workspace Overlay
source install/setup.bash
```

### 2.2 Selective Package Build
Build only specific packages to accelerate compile times:
```bash
# Build only description package (URDF/Xacro/Worlds/Meshes)
colcon build --packages-select khepera4_description --symlink-install

# Build only control and mapping package
colcon build --packages-select khepera4_control --symlink-install
```

### 2.3 Clean Rebuild (Purge Build Cache)
If launch files, meshes, or setup configuration become stale or unaligned:
```bash
# Remove build and install caches
rm -rf build/ install/ log/

# Source and rebuild from scratch
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 2.4 Required Environment Variables
To ensure Gazebo Harmonic resolves robot meshes, materials, and custom world SDFs:
```bash
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/khepera4_description:$(pwd)/install/khepera4_description/share:${GZ_SIM_RESOURCE_PATH}
```

### 2.5 Run Automated Verification Tests
Run the 226-test verification suite to confirm node integrity, kinematics, and mapping logic:
```bash
colcon test --packages-select khepera4_control
colcon test-result --verbose
```

---

## 🚀 3. Complete ROS 2 Simulation Launch Commands

### 3.1 Stack 1: 2D/2.5D SLAM & Autonomous Exploration
Launches Gazebo Harmonic exploration arena, Khepera IV model, sensor bridges, `khepera4_mapper` (occupancy grid + 2.5D elevation), and Qt control dashboard.

#### A. Full Stack (Gazebo GUI + RViz2 + Qt Dashboard + Autonomous Explorer)
```bash
ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true
```

#### B. Operator Mode (Interactive Manual Teleop via Qt D-Pad / WASD)
```bash
ros2 launch khepera4_control mapping.launch.py auto_explore:=false dashboard:=true rviz:=true
```

#### C. Headless Simulation (Server-Only, No Gazebo Window — Saves CPU/GPU)
```bash
ros2 launch khepera4_control mapping.launch.py headless:=true auto_explore:=true dashboard:=true rviz:=true
```

#### D. Minimal Headless (Pure CLI / Service Testing)
```bash
ros2 launch khepera4_control mapping.launch.py headless:=true auto_explore:=true dashboard:=false rviz:=false
```

#### E. Custom World File
```bash
ros2 launch khepera4_control mapping.launch.py \
    world:=$(pwd)/src/khepera4_description/worlds/khepera4_world.sdf \
    auto_explore:=true
```

| Argument | Default | Values | Description |
|---|---|---|---|
| `auto_explore` | `false` | `true`, `false` | Starts autonomous Braitenberg frontier explorer |
| `rviz` | `false` | `true`, `false` | Launches RViz2 with pre-configured SLAM display layout |
| `dashboard` | `true` | `true`, `false` | Launches Qt Spatial Exploration Station UI |
| `headless` | `false` | `true`, `false` | Runs Gazebo server without rendering window |
| `world` | *exploration world* | Path to SDF | World environment file to load in Gazebo |

---

### 3.2 Stack 2: 3-Robot Swarm Coordination Fleet
Coordinates Anchor (`khepera_1`), Relay (`khepera_2`), and Scout (`khepera_3`) with dynamic RF midpoint tethering and triangle formations.

#### A. Simulated Swarm with Fleet Dashboard
```bash
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim gui:=true
```

#### B. Headless Swarm Simulation
```bash
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim gui:=true headless:=true
```

#### C. Mock Hardware Swarm (No Gazebo — Internal Kinematic Emulator)
```bash
ros2 launch khepera4_control multi_robot_station.launch.py mode:=mock gui:=true
```

| Argument | Default | Values | Description |
|---|---|---|---|
| `mode` | `sim` | `sim`, `physical`, `mock` | Execution backend for fleet |
| `gui` | `true` | `true`, `false` | Launches Qt Swarm Control Station |
| `headless` | `false` | `true`, `false` | Runs Gazebo headless in `sim` mode |

---

### 3.3 Stack 3: Educational Control Station (BSA Student Lab)
Interactive station supporting Mode A (Manual WASD), Mode B (Lyapunov Setpoint Step Response), and Mode C (Continuous Trajectory Tracking):

```bash
ros2 launch khepera4_control control_station.launch.py
```
* **Headless:** `ros2 launch khepera4_control control_station.launch.py headless:=true`

---

### 3.4 Stack 4: Standalone Base Simulation & Spawning

#### Single Robot in Gazebo Harmonic
```bash
ros2 launch khepera4_description khepera4_sim.launch.py
```
* **Headless:** `ros2 launch khepera4_description khepera4_sim.launch.py headless:=true`

#### 3-Robot Fleet in Gazebo Harmonic
```bash
ros2 launch khepera4_description khepera4_multi_sim.launch.py
```
* **Headless:** `ros2 launch khepera4_description khepera4_multi_sim.launch.py headless:=true`

---

### 3.5 Stack 5: Kinematic Validation & Benchmarking

#### Open-Loop Precision Kinematics Benchmark
```bash
ros2 launch khepera4_control benchmark.launch.py maneuver:=circle linear_speed:=0.15 duration:=10.0
```

#### Closed-Loop Lyapunov Setpoint Regulation
```bash
ros2 launch khepera4_control closed_loop.launch.py target_x:=1.5 target_y:=1.0 target_theta:=1.57
```

---

## 📡 4. Key Runtime Services & Inspection Commands

Run these in a second sourced terminal (`source /opt/ros/jazzy/setup.bash && source install/setup.bash`):

### 4.1 Map Memory & Persistence Services
```bash
# Save current map to disk (generates maps/khepera_map.pgm, .yaml, and .png)
ros2 service call /save_map std_srvs/srv/Trigger "{}"

# Reload previously saved map from disk
ros2 service call /load_map std_srvs/srv/Trigger "{}"

# Reset map memory and clear grid
ros2 service call /clear_map std_srvs/srv/Trigger "{}"

# Respawn / upright recovery if robot tipped over
ros2 service call /recenter_robot std_srvs/srv/Trigger "{}"
```

### 4.2 Autonomous Exploration Control Services
```bash
# Toggle autonomous exploration on/off
ros2 service call /toggle_explorer std_srvs/srv/Trigger "{}"

# Explicitly enable or disable explorer
ros2 service call /set_explorer std_srvs/srv/SetBool "{data: true}"
ros2 service call /set_explorer std_srvs/srv/SetBool "{data: false}"
```

### 4.3 Smoke Testing & Topic Monitoring
```bash
# Check fused odometry stream rate (~50 Hz)
ros2 topic hz /khepera4/odom

# Check 360° proximity virtual scan rate (~10 Hz)
ros2 topic hz /khepera4/scan

# Check occupancy grid publication rate (~3 Hz)
ros2 topic hz /map

# Monitor onboard camera feed rate (~30 Hz)
ros2 topic hz /camera/image_raw

# Inspect current robot pose
ros2 topic echo /khepera4/odom --field pose.pose

# Monitor cliff drop-off and turnover alert states
ros2 topic echo /cliff_alert
ros2 topic echo /khepera4/turned_over
```

---

## 🧹 5. Process Cleanup & Reset

If simulation processes or ROS nodes remain bound in the background from previous sessions:

```bash
# Option 1: Quick Cleanup Script
./scripts/clean_sim.sh

# Option 2: Direct one-liner
pkill -9 -f "gz sim|ros_gz|robot_state_publisher|parameter_bridge|rviz2|khepera4|mapping_dashboard|reactive_explorer" 2>/dev/null || true
```

