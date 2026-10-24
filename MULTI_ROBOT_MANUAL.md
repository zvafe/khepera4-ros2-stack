# Khepera IV Multi-Robot Architecture & Operations Manual
## Fleet Coordination: `khepera_1/base_link`, `khepera_2/extender`, `khepera_3/ranger`

**Course / Research Charter:** Bazele Sistemelor Automate (BSA)  
**Hardware Platform:** K-Team Khepera IV (Gumstix Overo FireSTORM-Y, Yocto Linux 1.8+, Kernel 3.18)  
**Simulation Platform:** Gazebo Harmonic (v8.15+) & ROS 2 Jazzy Jalisco (Ubuntu 24.04 LTS)  
**Manual Reference:** *Khepera IV User Manual Revision 3.0* (K-Team S.A., Vallorbe, Switzerland)

---

## 1. Executive Summary & Fleet Structure

This project extends the **BRT K4** stack from a single-robot system into a **synchronized 3-robot autonomous fleet**. It bridges mathematical unicycle kinematics and cascade closed-loop control with real-world physical robot coordination operated from a central **Master PC**.

### 1.1 Fleet Naming & Role Specialization

The 3 robots are configured with specialized roles and coordinate frames:

| Robot Name | Role Name | Primary Frame | Visual Accent | Primary Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| `khepera_1` | `base_link` | `khepera_1/base_link` | Cobalt Blue (`#2979ff`) | **Anchor / Coordinator / Base Station:** Acts as the reference origin and fixed/master anchor for swarm maneuvers. |
| `khepera_2` | `extender` | `khepera_2/extender` | Solar Orange (`#ff9100`) | **RF / Wi-Fi Range Extender & Chain Relay:** Dynamically tracks the midpoint between Base and ranger to maintain continuous wireless connectivity. |
| `khepera_3` | `ranger` | `khepera_3/ranger` | Emerald Green (`#00e676`)| **Frontier Scout & Reconnaissance:** Moves forward into unknown territory while respecting tether communication constraints with the Extender. |

---

## 2. Distributed Computation Graph & TF Hierarchy

### 2.1 ROS 2 Topic Architecture

Each robot operates within its own dedicated ROS 2 namespace, preventing any message collisions:

```
[ Master PC (ROS 2 Jazzy) ]
  ├── /khepera_1/cmd_vel          [geometry_msgs/msg/Twist]
  ├── /khepera_1/odom             [nav_msgs/msg/Odometry]
  ├── /khepera_1/joint_states     [sensor_msgs/msg/JointState]
  ├── /khepera_1/camera/image_raw [sensor_msgs/msg/Image]
  ├── /khepera_1/imu/data         [sensor_msgs/msg/Imu]
  ├── /khepera_1/battery_state    [sensor_msgs/msg/BatteryState]
  ├── /khepera_1/ir_sensors       [std_msgs/msg/Float32MultiArray]
  │
  ├── /khepera_2/cmd_vel          (extender namespace)
  ├── /khepera_2/odom
  ├── ...
  │
  ├── /khepera_3/cmd_vel          (ranger namespace)
  ├── /khepera_3/odom
  ├── ...
  │
  ├── /swarm/cmd                  [std_msgs/msg/String]   (relay, triangle, halt, return_to_base)
  └── /swarm/status               [std_msgs/msg/String]   (live inter-robot distances, connectivity)
```

### 2.2 Global Coordinate Frame (TF2) Hierarchy

To enable multi-robot navigation and shared SLAM/map representations, all 3 robots are connected to a unified `world` reference frame:

```
                                  world
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │ (x=0.0)                  │ (x=0.6)                  │ (x=1.2)
         ▼                          ▼                          ▼
   khepera_1/odom             khepera_2/odom             khepera_3/odom
         │                          │                          │
         ▼                          ▼                          ▼
khepera_1/base_footprint   khepera_2/base_footprint   khepera_3/base_footprint
         │                          │                          │
         ▼                          ▼                          ▼
  khepera_1/base_link        khepera_2/base_link        khepera_3/base_link
         │                          │                          │
   (Role Frame)               (Role Frame)               (Role Frame)
  khepera_1/base_link        khepera_2/extender         khepera_3/ranger
```

---

## 3. High-Fidelity Simulation in Gazebo Harmonic

### 3.1 Parameterized Model Description (`khepera4.urdf.xacro`)

The robot model is generated dynamically via Xacro, accepting:
- `robot_name`: Instance namespace (`khepera_1`, `khepera_2`, `khepera_3`).
- `role`: Role frame designation (`base_link`, `extender`, `ranger`).
- `color_rgb`: Visual badge material color for Gazebo and RViz2.

### 3.2 Spawning in Gazebo

All 3 robots spawn simultaneously in `worlds/khepera4_world.sdf` with non-interfering initial poses:
- `khepera_1`: $(x=0.0, y=0.0, z=0.01, \text{yaw}=0.0)$
- `khepera_2`: $(x=0.6, y=0.0, z=0.01, \text{yaw}=0.0)$
- `khepera_3`: $(x=1.2, y=0.0, z=0.01, \text{yaw}=0.0)$

Each robot's `gz-sim-diff-drive-system` plugin is scoped to its own topic namespace (e.g. `/<robot_name>/cmd_vel`, `/<robot_name>/odom`), publishing transforms to `/tf` with appropriate frame prefixes.

### 3.3 Running Multi-Robot Simulation

#### Unified Launcher:
```bash
./run.sh --multi           # Native GUI (WSLg / GPU pass-through)
./run.sh --multi --web     # Web Browser Mode (noVNC at http://localhost:6080)
```

#### Native ROS 2 Launch:
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Option 1: Full Station (Simulation + Swarm Coordinator + Dashboard)
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim

# Option 2: Simulation Only (Headless or GUI)
ros2 launch khepera4_description khepera4_multi_sim.launch.py headless:=false
```

---

## 4. Physical Robot Architecture & Master PC Operations

*(Derived directly from Khepera IV User Manual Revision 3.0)*

### 4.1 Physical Hardware Specifications (Manual Chapters 3 & 4)

- **Processor:** Gumstix Overo FireSTORM-Y (TI DM3730 ARM Cortex-A8 @ 800 MHz, 512MB RAM).
- **Operating System:** Yocto Linux (Poky 1.8, Linux kernel 3.18, 32-bit ARMv7-A hard-float).
- **Wireless Networking:** Built-in Wi-Fi 802.11 b/g (Interface `wlan0`).
- **Low-Level Microcontroller:** Microchip dsPIC33 controlling motors and sensors at 100 Hz.
- **Drive System:** 2 coreless DC motors with 38:1 total gear reduction ($1.3\text{ W}$ usable mechanical power).
- **Wheel Geometry:** Diameter $42\text{ mm}$ ($r = 0.021\text{ m}$), Wheelbase $L = 0.1054\text{ m}$.
- **Encoder Resolution:** $19,456\text{ pulses}$ per full wheel revolution ($147.453\text{ pulses/mm}$).

### 4.2 Mathematical Kinematics & Conversions (Manual Section 4.14)

From Section 4.14.1 of the Khepera IV User Manual, the relation between metric velocity ($v_{\text{metric}}$ in mm/s) and dsPIC speed units ($v_{\text{pulses}}$ per 10ms control cycle) is:

$$v_{\text{metric}}\ [\text{mm/s}] = 0.678181 \cdot v_{\text{pulses}}$$
$$v_{\text{pulses}} = \frac{v_{\text{metric}}\ [\text{mm/s}]}{0.678181} \approx 1.47453 \cdot v_{\text{metric}}\ [\text{mm/s}] = 1474.53 \cdot v_{\text{metric}}\ [\text{m/s}]$$

Given chassis linear velocity $v$ (m/s) and angular velocity $\omega$ (rad/s):
$$v_L = v - \frac{L}{2}\omega, \quad v_R = v + \frac{L}{2}\omega$$
$$\text{speed}_L = \text{clamp}\Big(\text{round}(v_L \cdot 1474.53), -1200, 1200\Big)$$
$$\text{speed}_R = \text{clamp}\Big(\text{round}(v_R \cdot 1474.53), -1200, 1200\Big)$$

Encoder position conversion to meters (Manual Section 4.14.3):
$$\text{distance } [\text{m}] = \text{pulses} \cdot \frac{42 \cdot \pi \times 10^{-3}}{19456} = \text{pulses} \cdot 6.78181 \times 10^{-6}\text{ m}$$

---

## 5. Embedded Onboard Daemon (`khepera4_tcp_server`)

To achieve zero-lag, deterministic control from the Master PC over Wi-Fi, each physical Khepera IV runs an embedded C daemon located at:
`physical_robot/onboard_daemon/khepera4_tcp_server.c`

### 5.1 Communication Protocols Supported

1. **Official K-Team ASCII Protocol (Manual Chapter 6):**
   - `D,speed_l,speed_r\r\n` : Sets motor speed in pulses/10ms with PID.
   - `R\r\n` : Reads 32-bit wheel encoder positions (`r,left,right\r\n`).
   - `N\r\n` : Reads 12 proximity infrared sensors (`n,bl,l,fl,f,fr,r,br,b,gl,gfl,gfr,gr\r\n`).
   - `G\r\n` : Reads 5 ultrasonic transceivers in cm (`g,us0,us1,us2,us3,us4\r\n`).
   - `V,0\r\n` : Reads 2-cell LiPo battery voltage in mV (`v,voltage\r\n`).
   - `S\r\n` / `T\r\n` : Reads 3D gyroscope and accelerometer data.
   - `Z\r\n` : Microcontroller reset / emergency halt.

2. **Safety Watchdog (Fail-Safe):**
   - If the Master PC loses Wi-Fi connection or crashes for $> 500\text{ ms}$, the onboard server automatically sets motor speeds to zero (`kh4_set_speed(0, 0, dsPic)`), preventing runaway robots.

### 5.2 Autostart via Systemd

The daemon installs as a system service:
`/lib/systemd/system/khepera4_server.service`
ensuring it boots automatically when the physical Khepera IV is powered on.

---

## 6. Master PC Hardware Bridge (`khepera4_hardware_bridge`)

Running on the Master PC, the hardware bridge node:
1. Opens a non-blocking TCP socket to `ROBOT_IP:5000`.
2. Subscribes to `/<robot_name>/cmd_vel` (`geometry_msgs/msg/Twist`).
3. Formulates inverse kinematic motor pulse orders and sends them over Wi-Fi.
4. Queries encoders at $25\text{--}50\text{ Hz}$ and computes odometry $(x, y, \theta)$.
5. Publishes `/<robot_name>/odom`, `/<robot_name>/joint_states`, and broadcasts TF.
6. Handles automatic socket reconnection in case of Wi-Fi packet drops.

---

## 7. Swarm Fleet Management Tool (`fleet_manager.sh`)

An all-in-one management script is provided for the Master PC operator:

```bash
# Location:
./physical_robot/fleet_manager.sh [COMMAND] [ARGS]
```

### Available Commands:

```bash
# 1. Test ping and network latency to all 3 physical robots
./physical_robot/fleet_manager.sh ping

# 2. Automatically compile and deploy the TCP daemon & systemd service over SSH/SCP
./physical_robot/fleet_manager.sh deploy

# 3. Check daemon process and battery status across the entire fleet
./physical_robot/fleet_manager.sh status

# 4. Start the onboard daemons on all 3 robots
./physical_robot/fleet_manager.sh start

# 5. Emergency stop all robots and stop daemons
./physical_robot/fleet_manager.sh stop

# 6. Open an interactive root SSH shell into a specific robot (1, 2, or 3)
./physical_robot/fleet_manager.sh ssh 1   # SSH into khepera_1 (base_link)
./physical_robot/fleet_manager.sh ssh 2   # SSH into khepera_2 (extender)
./physical_robot/fleet_manager.sh ssh 3   # SSH into khepera_3 (ranger)

# 7. Stream live WVGA camera video from a specific robot (Manual Section 9.3)
./physical_robot/fleet_manager.sh stream-camera 3
```

---

## 8. Swarm Coordination & Formation Control

The `swarm_coordinator` node runs on the Master PC and manages cooperative behaviors:

### 8.1 Chain Relay Mode (Midpoint Tethering)
In exploration missions, wireless signal strength degrades rapidly over distance. The Extender (`khepera_2`) acts as an active relay:
$$\mathbf{p}_{\text{extender}}^* = \frac{\mathbf{p}_{\text{base}} + \mathbf{p}_{\text{ranger}}}{2}$$
$$\theta_{\text{extender}}^* = \text{atan2}(y_{\text{ranger}} - y_{\text{base}}, x_{\text{ranger}} - x_{\text{base}})$$

Closed-loop feedback steers `khepera_2` to maintain this exact midpoint. If the distance exceeds `max_comm_range` ($2.0\text{ m}$), a tether alert is triggered.

### 8.2 Triangle Formation
Maintains an equilateral V-formation centered on `khepera_1` (`base_link`), suitable for synchronized sweeping and formation driving.

### 8.3 Return to Base
Commands both `khepera_2` and `khepera_3` to autonomously return to the immediate vicinity of `khepera_1`.

---

## 9. Interactive Control Station & Telemetry Dashboard

The Qt dashboard (`dashboard.py`) provides real-time visualization and multi-robot operations:

1. **Active Robot Selector:** Dropdown to direct WASD manual teleoperation or closed-loop step setpoints to `khepera_1`, `khepera_2`, `khepera_3`, or broadcast to `All`.
2. **2D Swarm Arena View:**
   - Displays all 3 robots with color-coded bodies and heading arrows.
   - Plots distinct historical trajectory trails.
   - Visualizes communication tethers between Base $\leftrightarrow$ Extender $\leftrightarrow$ ranger with live distance badges ($d_{12}, d_{23}$).
   - Highlights the currently active robot with a bright selection ring.
3. **One-Click Swarm Action Buttons:**
   - 🛰️ **Chain Relay:** Activates autonomous midpoint tethering.
   - 🔺 **Triangle Formation:** Deploys triangular swarm formation.
   - 🏠 **Return to Base:** Recalls Extender and ranger to Base.
   - ⏸️ **Halt Swarm:** Instant zero-velocity stop for the entire fleet.

---

## 10. Master PC Launch Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Launch 3-Robot Sim (GUI)** | `ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim` |
| **Launch 3-Robot Sim (Web)** | `./run.sh --multi --web` |
| **Connect to 3 Real Robots** | `ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical` |
| **Test Fleet (Mock Hardware)**| `ros2 launch khepera4_control multi_robot_station.launch.py mode:=mock` |
| **Launch Controllers Only** | `ros2 launch khepera4_control multi_robot_control.launch.py` |
| **Ping All 3 Robots** | `./physical_robot/fleet_manager.sh ping` |
| **Deploy Daemon to Robots** | `./physical_robot/fleet_manager.sh deploy` |
