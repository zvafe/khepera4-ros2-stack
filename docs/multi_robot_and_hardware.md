# Multi-Robot Architecture & Physical Hardware Operations Manual
**Project:** BRT K4 Khepera IV Autonomous Fleet  
**Roles:** `khepera_1/base_link` (Anchor), `khepera_2/extender` (Relay), `khepera_3/explorer` (Scout)  
**Embedded Target:** Gumstix Overo FireSTORM-Y (Cortex-A8, Yocto Linux 1.8+, Kernel 3.18, dsPIC33 coprocessor)  
**Hardware Reference:** *Khepera IV User Manual Revision 3.0* (K-Team S.A., Vallorbe, Switzerland)

---

## 1. Multi-Robot Fleet Architecture & Roles

The BRT K4 multi-robot framework coordinates an autonomous **3-robot formation fleet** operated either in high-fidelity Gazebo Harmonic simulation or on physical mobile robots via TCP/IP.

```
+-----------------------------------------------------------------------------------+
|                                 MASTER PC / ROS 2                                 |
|                                                                                   |
|  +--------------------+    +-----------------------+    +----------------------+  |
|  | khepera_1 (Anchor) |    |  khepera_2 (Extender) |    |  khepera_3 (Explorer)|  |
|  | Role: base_link    |    |  Role: extender       |    |  Role: explorer      |  |
|  | Cobalt Blue        |    |  Solar Orange         |    |  Emerald Green       |  |
|  +--------------------+    +-----------------------+    +----------------------+  |
|            │                           │                           │              |
|            ▼                           ▼                           ▼              |
|  /khepera_1/cmd_vel          /khepera_2/cmd_vel          /khepera_3/cmd_vel       |
|  /khepera_1/odom             /khepera_2/odom             /khepera_3/odom          |
|            │                           │                           │              |
+────────────┼───────────────────────────┼───────────────────────────┼──────────────+
             │                           │                           │
   TCP: 192.168.1.101:5000     TCP: 192.168.1.102:5000     TCP: 192.168.1.103:5000
             │                           │                           │
             ▼                           ▼                           ▼
      [ Physical K4 #1 ]          [ Physical K4 #2 ]          [ Physical K4 #3 ]
```

### 1.1 Standardized Fleet Roles

| Robot Name | Role Designation | Primary Frame | Visual Color | Operational Responsibility |
|---|---|---|---|---|
| `khepera_1` | `base_link` | `khepera_1/base_link` | Cobalt Blue (`#2979ff`) | **Anchor & Coordinator:** Acts as the stationary or coordinated reference origin for the formation. |
| `khepera_2` | `extender` | `khepera_2/extender` | Solar Orange (`#ff9100`) | **RF / Wi-Fi Range Relay:** Dynamically regulates its pose to the midpoint between Anchor and Explorer to maintain an active wireless link. |
| `khepera_3` | `explorer` | `khepera_3/explorer` | Emerald Green (`#00e676`) | **Frontier Scout:** Navigates into unknown terrain while respecting the RF tether distance limit from the Extender. |

*(Note: The legacy designation `'ranger'` has been permanently retired in favor of `'explorer'` across all code, launch files, and TF transforms).*

---

## 2. Distributed Computation Graph & TF2 Hierarchy

### 2.1 Scoped ROS 2 Namespaces
Each robot is fully isolated within its own namespace:
- Control: `/<robot_name>/cmd_vel` (`geometry_msgs/msg/Twist`)
- Wheel Odometry: `/<robot_name>/odom` (`nav_msgs/msg/Odometry`)
- Joint Encoders: `/<robot_name>/joint_states` (`sensor_msgs/msg/JointState`)
- Camera: `/<robot_name>/camera/image_raw` (`sensor_msgs/msg/Image`)
- IMU: `/<robot_name>/imu/data` (`sensor_msgs/msg/Imu`)
- Battery: `/<robot_name>/battery_state` (`sensor_msgs/msg/BatteryState`)
- Fleet Inter-Robot Status: `/swarm/status` (`std_msgs/msg/String`, JSON telemetry)
- Fleet Control Directives: `/swarm/cmd` (`std_msgs/msg/String`)

### 2.2 Global Coordinate Frame (TF2) Hierarchy
All robots connect to a unified `world` reference frame:
```text
                              world
                                │
        ┌───────────────────────┼───────────────────────┐
        │ (x=0.0)               │ (x=0.6)               │ (x=1.2)
        ▼                       ▼                       ▼
  khepera_1/odom          khepera_2/odom          khepera_3/odom
        │                       │                       │
        ▼                       ▼                       ▼
khepera_1/base_footprint khepera_2/base_footprint khepera_3/base_footprint
        │                       │                       │
        ▼                       ▼                       ▼
  khepera_1/base_link     khepera_2/base_link     khepera_3/base_link
        │                       │                       │
        ▼                       ▼                       ▼
  (Role Frame)            (Role Frame)            (Role Frame)
  khepera_1/base_link     khepera_2/extender      khepera_3/explorer
```

---

## 3. Physical Robot Deployment Architecture

### 3.1 Embedded Hardware Architecture
- **CPU:** TI OMAP3530 (ARM Cortex-A8 @ 720 MHz), 512 MB RAM, 512 MB NAND Flash.
- **Coprocessor:** Microchip dsPIC33F digital signal controller managing motor PWM, optical encoders, 8 IR proximity sensors, 5 ultrasonic transceivers, and battery ADC.
- **Operating System:** Poky Yocto Linux 1.8+, Linux Kernel 3.18.
- **Library:** `libkhepera` (C runtime library interfacing with I2C, SPI, and GPIO buses).

---

### 3.2 Onboard C Daemon (`khepera4_tcp_server.c`)
The physical robot runs a lightweight C socket daemon [`khepera4_tcp_server`](file:///home/andy/Desktop/khepera4_ws/physical_robot/onboard_daemon/khepera4_tcp_server.c) that translates TCP ASCII packets from the Master PC into native `libkhepera` calls:

- **Network Port:** TCP `5000`
- **Protocol:** ASCII text commands terminated with `\r\n`
- **Motor Command:** `D,left_pulse,right_pulse\r\n` (Response: `d\r\n`)
- **Odometry Request:** `R\r\n` (Response: `r,left_ticks,right_ticks\r\n`)
- **Proximity IR:** `N\r\n` (Response: `n,val0,val1,...,val11\r\n`)
- **Ultrasonic Sonar:** `G\r\n` (Response: `g,us0,us1,us2,us3,us4\r\n`)
- **Battery Status:** `V,0\r\n` (Response: `v,millivolts\r\n`)
- **Safety Watchdog:** Automatically halts motors if no command arrives within 250 ms.

### 3.3 Systemd Service Autostart
The daemon is managed on Yocto Linux via systemd:
```bash
# Service definition: physical_robot/onboard_daemon/khepera4_server.service
# Copied to /etc/systemd/system/khepera4_server.service on the robot
systemctl enable khepera4_server.service
systemctl start khepera4_server.service
```

---

## 4. Master PC Fleet Management Tool (`fleet_manager.sh`)

The script [`physical_robot/fleet_manager.sh`](file:///home/andy/Desktop/khepera4_ws/physical_robot/fleet_manager.sh) automates multi-robot network administration from the Master PC.

### 4.1 Subcommands

```bash
# 1. Ping test all 3 physical robots
./physical_robot/fleet_manager.sh ping

# 2. Deploy daemon sources via SCP and trigger remote compilation on robots
./physical_robot/fleet_manager.sh deploy

# 3. Start khepera4_server service on all robots
./physical_robot/fleet_manager.sh start

# 4. Check systemd daemon status on all robots
./physical_robot/fleet_manager.sh status

# 5. Stop daemons across the fleet
./physical_robot/fleet_manager.sh stop

# 6. Interactive SSH into a specific robot (1, 2, or 3)
./physical_robot/fleet_manager.sh ssh 3

# 7. Stream video from onboard camera (Port 5001)
./physical_robot/fleet_manager.sh stream-camera 1
```

*(Note: GStreamer camera streaming operates on port **5001** to prevent conflicts with the motor command daemon on port 5000).*

---

## 5. Sim-to-Real Launching

Switch between simulation and physical hardware transparently using the `mode` parameter:

```bash
# High-Fidelity Simulation in Gazebo Harmonic:
ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim

# Physical Hardware Deployment (Connects to K1_IP, K2_IP, K3_IP):
ros2 launch khepera4_control multi_robot_station.launch.py mode:=physical

# Mock Protocol Testing (Emulates socket communication locally):
ros2 launch khepera4_control multi_robot_station.launch.py mode:=mock
```
