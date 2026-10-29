# Single Physical Khepera IV Hardware Guide: Connection, Sensors, Mapping & Teleoperation

**Project:** Khepera IV Mobile Robot ROS 2 System  
**Target Hardware:** K-Team Khepera IV (Gumstix Overo FireSTORM-Y, Yocto Linux, dsPIC33)  
**Host System:** Lab Workstation / Personal PC (Ubuntu 24.04 LTS / ROS 2 Jazzy Jalisco)  
**Reference:** *K-Team Khepera IV User Manual Rev 3.0*

---

## 1. Physical Hardware Overview

The Khepera IV differential drive robot communicates with the Master PC over a dedicated TCP Wi-Fi socket connection:
- **Embedded Daemon:** `khepera4_tcp_server` running on the robot listening on TCP port `5000` (or custom port).
- **Protocol:** K-Team ASCII Protocol (`D` for motor speed, `R` for encoders, `N` for 12 IR proximity sensors, `G` for 5 ultrasonic transceivers, `V` for battery voltage).
- **Safety Watchdog:** Halts motors if no command arrives within 500 ms.
- **Hardware Bridge:** `khepera4_hardware_bridge` running on the Master PC, translating ROS 2 `/cmd_vel` into motor pulses and streaming sensor telemetry into standard ROS 2 message topics.

```
+-----------------------------------------------------------------------------------------+
|                                    MASTER PC / ROS 2 JAZZY                              |
|                                                                                         |
|  +--------------------+    +--------------------+    +-------------------------------+  |
|  | mapping_dashboard  |    |   khepera4_mapper  |    |      RViz2 3D Visualizer      |  |
|  | (Teleop & Metrics) |    |  (2D / 2.5D SLAM)  |    | (Occupancy Grid & TF Display) |  |
|  +---------┬----------+    +---------┬----------+    +───────────────┬───────────────+  |
|            │                         │                               │                  |
|            ▼                         ▼                               ▼                  |
|        /cmd_vel            /khepera4/ir_* /us_*            /map, /scan, /odom, /tf      |
|            │                         │                               │                  |
|            └─────────────────────────┼───────────────────────────────┘                  |
|                                      │                                                  |
|                                      ▼                                                  |
|                    +───────────────────────────────────+                                |
|                    |     khepera4_hardware_bridge      |                                |
|                    |   (ROS 2 Python Hardware Client)  |                                |
|                    +─────────────────┬─────────────────+                                |
+──────────────────────────────────────┼──────────────────────────────────────────────────+
                                       │ Wi-Fi Socket (TCP Port 5000)
                                       ▼
                     +───────────────────────────────────+
                     |       PHYSICAL KHEPERA IV         |
                     |     [khepera4_tcp_server (C)]     |
                     |                                   |
                     |  - TI OMAP3530 ARM Cortex-A8      |
                     |  - dsPIC33 Coprocessor via I2C    |
                     |  - 2-cell 7.4V LiPo Battery       |
                     |  - 8 IR + 4 Ground IR + 5 US      |
                     |  - 500ms Motor Safety Watchdog    |
                     +───────────────────────────────────+
```

---

## 2. Pre-Flight Hardware Inspection Checklist

Before powering on or commanding the physical robot:
1. **Battery Level:**
   - Khepera IV uses a 2-cell 7.4V nominal LiPo battery (discharge cutoff: 6.6V, full charge: 8.4V).
   - If voltage is below **6.6V**, charge the robot on its docking station before running motors; accelerating on low battery causes brownout restarts.
2. **Wheels & Casters:**
   - Ensure the rubber drive tires and the front/rear ball casters are free of lint and debris.
3. **Power On:**
   - Flip the toggle power switch on the rear panel. Wait ~25 seconds for the Poky Linux kernel to boot (signaled by a startup beep).

---

## 3. Wi-Fi Pairing & Finding the Robot IP

> [!IMPORTANT]
> **Avoid Campus Wi-Fi Isolation:** Campus or guest networks (e.g. `eduroam`) block peer-to-peer TCP communication (port 5000) between devices. Always connect both your PC and the Khepera IV to a **dedicated lab router**, mobile hotspot, or private switch.

1. **Find Robot IP:**
   - If a serial console / micro-USB cable is connected:
     ```bash
     ifconfig wlan0
     ```
   - From your PC (scanning the lab subnet `192.168.1.0/24`):
     ```bash
     ./physical_robot/fleet_manager.sh ping
     ```
   - Alternatively:
     ```bash
     nmap -sn 192.168.1.0/24
     ```
2. **Save IP to Environment File:**
   In `khepera4_ws/.env`:
   ```bash
   KHEPERA_1_IP=192.168.1.101
   KHEPERA_DAEMON_PORT=5000
   ```

---

## 4. Deploying & Starting the Onboard Daemon

1. **One-Click Deployment via `fleet_manager.sh`:**
   ```bash
   ./physical_robot/fleet_manager.sh deploy
   ```
   *Transfers `khepera4_tcp_server.c` to `/home/root/`, compiles natively with `libkhepera`, and enables the `khepera4_server.service` systemd autostart service.*

2. **Check Daemon Status:**
   ```bash
   ./physical_robot/fleet_manager.sh status
   ```

3. **Manual Netcat Socket Sanity Check (Optional):**
   ```bash
   nc 192.168.1.101 5000
   ```
   Type:
   - `V,0` -> Returns `v,7450` (7.45V)
   - `R` -> Returns `r,0,0` (encoder ticks)
   - `N` -> Returns `n,bl,l,fl,f,fr,r,br,b,gl,gfl,gfr,gr` (12 IR sensor readings)
   - `G` -> Returns `g,us0,us1,us2,us3,us4` (5 ultrasonic distances in cm)
   - Press `Ctrl+C` to disconnect.

---

## 5. Launching the Physical Robot System

### Option A: Complete Single Physical Robot Launch (Recommended)
Launches the hardware bridge, URDF robot state publisher, mapping engine, and Qt dashboard:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch khepera4_control single_robot_physical.launch.py \
    robot_ip:=192.168.1.101 \
    dashboard:=true \
    mapper:=true \
    rviz:=false
```

### Option B: Using `mapping.launch.py` in Physical Mode
```bash
ros2 launch khepera4_control mapping.launch.py \
    mode:=physical \
    robot_ip:=192.168.1.101 \
    dashboard:=true \
    rviz:=true
```

### Option C: Standalone Hardware Bridge (For Custom Scripts)
```bash
ros2 run khepera4_control khepera4_hardware_bridge --ros-args \
    -p robot_ip:=192.168.1.101 \
    -p robot_port:=5000 \
    -p single_robot:=true \
    -p publish_tf:=true
```

---

## 6. Sensor Topics & Telemetry Verification

Open another terminal to inspect published telemetry:

```bash
source install/setup.bash

# 1. Check active topics
ros2 topic list

# 2. View Wheel Odometry (updates when wheels turn)
ros2 topic echo /odom

# 3. View 12 IR Proximity & Ground Values (0-1023)
ros2 topic echo /sensors/infrared

# 4. View 5 Ultrasonic Distances (cm)
ros2 topic echo /sensors/ultrasonic

# 5. View Unified 360° Virtual LaserScan (360 samples)
ros2 topic echo /scan

# 6. View Battery Telemetry (Voltage & Remaining %)
ros2 topic echo /battery

# 7. View 2D Occupancy Grid Map Metadata
ros2 topic echo /map --once
```

---

## 7. Teleoperating the Physical Robot

### Method 1: Using the Mapping Dashboard GUI
- Launch with `dashboard:=true`.
- The GUI features interactive directional arrow buttons (Forward, Reverse, Left, Right, Stop) and keyboard WASD controls.
- Speed sliders allow fine-tuning linear velocity ($0.02 - 0.20 \text{ m/s}$) and angular turn rate ($0.2 - 1.5 \text{ rad/s}$).

### Method 2: Standard Keyboard Teleoperation
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel
```
- `i`: Drive forward
- `j` / `l`: Turn left / right
- `,`: Drive backward
- `k`: Emergency halt

> [!NOTE]
> **Safety Watchdog:** If keyboard input ceases for more than 500 ms, the hardware bridge and onboard daemon automatically command $0 \text{ m/s}$, safely stopping the physical robot.

---

## 8. Visualizing and Saving the Map

1. **RViz2 Visualization:**
   ```bash
   ros2 run rviz2 rviz2 -d src/khepera4_description/config/khepera4_mapping.rviz
   ```
   - **Fixed Frame:** `map` (or `odom`)
   - Displays live robot model (`base_footprint`), laser rays (`/scan`), and the probabilistic occupancy grid (`/map`).

2. **Save Map to Disk:**
   Once exploration is complete:
   ```bash
   ros2 service call /save_map std_srvs/srv/Trigger
   ```
   - Automatically saves high-resolution YAML, PGM, and PNG map files into the `maps/` directory.
