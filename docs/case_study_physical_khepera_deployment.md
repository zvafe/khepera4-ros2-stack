# Case Study & Field Manual: Physical Khepera IV Deployment with ROS 2 & Python

**Project:** BRT K4 Khepera IV Autonomous Mobile Fleet  
**Author / Lab:** BRT Labs & Robotics Systems Engineering  
**Target Hardware:** K-Team Khepera IV Mobile Robots (TI OMAP3530 ARM Cortex-A8, Yocto Linux 1.8+, dsPIC33 coprocessor)  
**Host Environment:** Classroom Lab Workstation / Personal PC (Ubuntu 24.04 LTS / Docker DevContainer, ROS 2 Jazzy Jalisco)  
**Related Reference:** *Khepera IV User Manual Rev 3.0* (K-Team S.A.)  

---

## Executive Summary & Educational Case Study

### 1. The Challenge: Sim-to-Real Deployment in an Academic Lab

In academic and research robotics laboratories, students and researchers frequently develop, simulate, and test advanced control algorithms (Lyapunov regulation, frontier exploration, SLAM, multi-robot formation control) in high-fidelity simulation environments like **Gazebo Harmonic**. 

However, transitioning these algorithms from a simulation workspace to **physical Khepera IV robots** on the lab floor poses a fundamental systems architecture challenge:

```
+---------------------------------------------------------------------------------------+
|                                COMPUTATION & ARCHITECTURE MISMATCH                    |
+---------------------------------------------------------------------------------------+
|  Host Workstation (Classroom / Friend's PC)  |  Khepera IV Mobile Robot (Physical)     |
|  - Modern x86_64 / arm64 architecture         |  - TI OMAP3530 ARM Cortex-A8 (ARMv7 32-bit)|
|  - Ubuntu 24.04 LTS (Noble Numbat)           |  - Poky Yocto Linux 1.8+ (Kernel 3.18)  |
|  - ROS 2 Jazzy Jalisco (glibc 2.39, Py 3.12)  |  - Embedded RAM: 512 MB, NAND: 512 MB   |
|  - Python 3.12, NumPy, PyQt5, SciPy, TF2     |  - No native ROS 2 Jazzy binary packages|
+---------------------------------------------------------------------------------------+
```

Attempting to compile and run modern ROS 2 Jazzy directly on the robot's embedded 720 MHz ARMv7 processor will quickly exhaust memory, overheat the Gumstix module, or fail due to library incompatibilities (glibc 2.21 vs 2.39).

### 2. The Solution: Distributed Master-Robot Architecture

To solve this mismatch, the BRT K4 stack utilizes a **Distributed Master-Robot Architecture**:

```
+------------------------------------------------------------------------------------+
|                         CLASSROOM LAB PC / FRIEND'S WORKSTATION                    |
|                                                                                    |
|   +-----------------------+   +----------------------+   +---------------------+   |
|   | Autonomous Controller |   |  khepera4_mapper     |   | Swarm Coordinator   |   |
|   | / Reactive Explorer   |   |  (2D / 2.5D SLAM)    |   | (Formation Engine)  |   |
|   +-----------┬-----------+   +----------┬-----------+   +----------┬----------+   |
|               │                          │                          │              |
|               ▼                          ▼                          ▼              |
|       /khepera_1/cmd_vel         /khepera_2/cmd_vel         /khepera_3/cmd_vel     |
|               │                          │                          │              |
|               ▼                          ▼                          ▼              |
|   +----------------------------------------------------------------------------+   |
|   |                  khepera4_hardware_bridge (ROS 2 Python)                   |   |
|   |  - Converts Twist (v, w) -> Left/Right Motor Speeds (pulses/10ms)           |   |
|   |  - Integrates Wheel Encoders -> nav_msgs/Odometry & TF2 (/odom -> base)    |   |
|   |  - Decodes Proximity IR & Sonar -> std_msgs/Float32MultiArray              |   |
|   |  - Monitors Battery Voltage & Network Watchdog                             |   |
|   +----------------------------------------------------------------------------+   |
+───────────────────────────────────────┬────────────────────────────────────────────+
                                        │ High-Speed Lab Wi-Fi (TCP Port 5000)
       ┌────────────────────────────────┼────────────────────────────────┐
       │ (192.168.1.101:5000)           │ (192.168.1.102:5000)           │ (192.168.1.103:5000)
       ▼                                ▼                                ▼
+──────────────────────────+  +──────────────────────────+  +──────────────────────────+
| Physical Khepera IV #1   |  | Physical Khepera IV #2   |  | Physical Khepera IV #3   |
| [khepera4_tcp_server (C)]|  | [khepera4_tcp_server (C)]|  | [khepera4_tcp_server (C)]|
|                          |  |                          |  |                          |
|  - Native libkhepera     |  |  - Native libkhepera     |  |  - Native libkhepera     |
|  - dsPIC33 I2C Driver    |  |  - dsPIC33 I2C Driver    |  |  - dsPIC33 I2C Driver    |
|  - 250ms Motor Watchdog  |  |  - 250ms Motor Watchdog  |  |  - 250ms Motor Watchdog  |
+──────────────────────────+  +──────────────────────────+  +──────────────────────────+
```

1. **Master Workstation:** Executes all high-level ROS 2 nodes, trajectory generators, SLAM, TF2 transforms, and Qt dashboards.
2. **Physical Khepera IV:** Executes a lightweight, native C daemon (`khepera4_tcp_server`), which directly controls the motors and reads sensors via `libkhepera` and the dsPIC33 coprocessor.
3. **Bridge Layer:** A dedicated ROS 2 Python node (`khepera4_hardware_bridge.py`) maintains an active, non-blocking TCP socket to the robot, streaming motor setpoints and parsing odometry and proximity telemetry at 25 Hz.

---

## Step-by-Step Lab Execution Manual

Follow these exact steps when sitting down at a classroom lab computer or a friend's PC.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DEPLOYMENT WORKFLOW                             │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│    Phase 1      │     Phase 2     │     Phase 3     │        Phase 4        │
│ Workstation     │ Network Pairing │ Daemon Deploy   │ ROS 2 Launch &        │
│ Environment     │ & Ping Check    │ & Verification  │ Fleet Control         │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
```

---

### Phase 1: Workstation Setup (Classroom Computer / Friend's PC)

When logging onto a new or shared computer, you have two options depending on what is installed:

#### Option A: Docker / DevContainer (Recommended for shared/guest PCs)
If the computer has Docker installed, you do not need to install ROS 2 or any dependencies natively:

```bash
# 1. Clone or copy your workspace to the PC
cd ~/Desktop
git clone <your-repo-url> khepera4_ws
cd khepera4_ws

# 2. Check Docker installation
docker --version
docker compose version

# 3. Launch the containerized ROS 2 environment with GUI support
# (Allows access to Qt Control Dashboard and RViz2)
./run.sh --mapping
```
*(Tip: If on Windows WSL2 or a headless classroom PC, `./run.sh --web` opens a virtual desktop in any web browser at `http://localhost:6080`)*.

#### Option B: Native ROS 2 Jazzy on Ubuntu 24.04
If the classroom PC is running native Ubuntu 24.04 with ROS 2 Jazzy:

```bash
# 1. Source ROS 2 Jazzy environment
source /opt/ros/jazzy/setup.bash

# 2. Navigate to workspace root
cd ~/Desktop/khepera4_ws

# 3. Install missing ROS and system dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y
sudo apt-get install -y python3-pyqt5 python3-scipy python3-matplotlib net-tools iputils-ping

# 4. Build workspace with symlink install
colcon build --symlink-install --packages-select khepera4_description khepera4_control
source install/setup.bash
```

---

### Phase 2: Networking & Wi-Fi Pairing

Physical Khepera IV robots connect via Wi-Fi (802.11 b/g).

> [!WARNING]
> **University Wi-Fi "Client Isolation" Trap:**
> Campus Wi-Fi networks (such as `eduroam` or campus guest portals) almost universally enable **AP Client Isolation**. This security feature prevents wireless devices on the same network from communicating directly with each other, completely blocking TCP socket connections on port 5000.
> **Solution:** Always connect both the Master PC and the Khepera robots to a **dedicated lab router**, an unmanaged wireless switch, or a smartphone 5GHz/2.4GHz Wi-Fi hotspot where client isolation is disabled.

#### 1. Network Subnet Architecture
Configure your lab network to the standard Khepera fleet subnet:
- **Subnet:** `192.168.1.0/24` (Subnet mask: `255.255.255.0`)
- **Gateway / Router:** `192.168.1.1`
- **Master PC (Classroom Computer):** `192.168.1.50` (or dynamic DHCP)
- **Robot 1 (`khepera_1`):** `192.168.1.101`
- **Robot 2 (`khepera_2`):** `192.168.1.102`
- **Robot 3 (`khepera_3`):** `192.168.1.103`

#### 2. Configure Local Environment File
In the root of `khepera4_ws`, check or create your `.env` file:
```bash
# Ensure .env has the correct IP addresses
cat << 'EOF' > .env
KHEPERA_1_IP=192.168.1.101
KHEPERA_2_IP=192.168.1.102
KHEPERA_3_IP=192.168.1.103
KHEPERA_DAEMON_PORT=5000
KHEPERA_CAMERA_PORT=5001
ROS_DOMAIN_ID=0
EOF
```

#### 3. Test Network Reachability
Run the automated fleet connectivity probe:
```bash
./physical_robot/fleet_manager.sh ping
```

Expected output:
```text
==================================================================
  Khepera IV Multi-Robot Fleet Manager (3 Physical Units)         
==================================================================
 Fleet Configuration:
   1) khepera_1 [base_link] -> 192.168.1.101:5000
   2) khepera_2 [extender]  -> 192.168.1.102:5000
   3) khepera_3 [explorer]  -> 192.168.1.103:5000
==================================================================
Testing network reachability & latency...
 -> khepera_1  (base_link) at 192.168.1.101   : ONLINE (2.41 ms)
 -> khepera_2  (extender ) at 192.168.1.102   : ONLINE (3.10 ms)
 -> khepera_3  (explorer ) at 192.168.1.103   : ONLINE (2.85 ms)
```

---

### Phase 3: Physical Robot Preparation & Daemon Deployment

#### 1. Hardware Inspection Checklist
Before running any code on the hardware:
1. **Battery Level:** Check battery voltage via Khepera LCD or serial terminal. The robot uses a 2-cell 7.4V LiPo battery. If voltage drops below **6.6V**, charge the robot on its charging cradle. Low voltage will cause brownout resets when motors accelerate.
2. **Wheel & Caster Cleanliness:** Check the two drive wheels and the front/rear ball casters. Remove any hair or dust particles that create asymmetric rolling resistance.
3. **Power On:** Flip the power switch on the Khepera IV rear panel. Wait ~25 seconds for the Poky Yocto Linux system to boot (indicated by the LED status and beep sequence).

#### 2. Automatic One-Click Deployment
The `fleet_manager.sh` script automates cross-copying the C source code, compiling it natively on each robot using the robot's onboard GCC and `libkhepera`, and configuring the systemd background service:

```bash
./physical_robot/fleet_manager.sh deploy
```

What this script executes under the hood on each robot:
1. Transfers `khepera4_tcp_server.c`, `Makefile`, and `khepera4_server.service` via `scp`.
2. Connects via SSH and triggers:
   ```bash
   cd /home/root && make clean && make
   ```
3. Registers the systemd autostart daemon:
   ```bash
   systemctl daemon-reload
   systemctl enable khepera4_server
   systemctl restart khepera4_server
   ```

#### 3. Verify Daemon Execution
Check daemon process status across the fleet:
```bash
./physical_robot/fleet_manager.sh status
```

Expected output:
```text
 -> khepera_1   (192.168.1.101 ): DAEMON ACTIVE
 -> khepera_2   (192.168.1.102 ): DAEMON ACTIVE
 -> khepera_3   (192.168.1.103 ): DAEMON ACTIVE
```

#### 4. Quick Manual Socket Sanity Check (Optional)
If you wish to test a robot manually without ROS 2, open a raw TCP socket connection from your terminal using `nc` (netcat):
```bash
nc 192.168.1.101 5000
```
Type the following K-Team ASCII commands (press Enter after each):
- `V,0` -> Returns battery millivolts: `v,7420` (7.42 V)
- `R` -> Returns encoder ticks: `r,0,0`
- `D,50,50` -> Drives wheels forward slowly for 250 ms: `d`
- `D,0,0` -> Stops motors: `d`
- Type `Ctrl+C` to exit.

---

### Phase 4: Running ROS 2 Control & Teleoperation

#### Test 1: Single Robot Teleop & Sensor Echo
To verify communication with a single robot (`khepera_1`):

**Terminal 1:** Launch the Hardware Bridge node:
```bash
source install/setup.bash
ros2 run khepera4_control khepera4_hardware_bridge --ros-args \
    -p robot_name:=khepera_1 \
    -p robot_ip:=192.168.1.101 \
    -p robot_port:=5000 \
    -p publish_tf:=true
```

**Terminal 2:** Verify odometry and sensors are publishing:
```bash
source install/setup.bash

# Check published topics
ros2 topic list | grep khepera_1

# View odometry updates (moving wheels by hand will update x, y, yaw)
ros2 topic echo /khepera_1/odom

# View 8 infrared proximity distance values
ros2 topic echo /khepera_1/ir_sensors

# View battery telemetry
ros2 topic echo /khepera_1/battery_state
```

**Terminal 3:** Teleoperate the physical robot using keyboard:
```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/khepera_1/cmd_vel
```
*Press `i` to drive forward, `j`/`l` to turn, `k` to halt.* Notice that if you release keys, the robot stops within 500 ms due to the safety watchdog.

---

### Phase 5: Autonomous Exploration, SLAM & Multi-Robot Swarm

#### 1. Multi-Robot Physical Fleet Launch
To launch the entire 3-robot formation fleet with hardware bridges and the formation swarm coordinator:

```bash
source install/setup.bash

ros2 launch khepera4_control multi_robot_station.launch.py \
    mode:=physical \
    gui:=true
```

This launch sequence:
1. Spawns 3 `khepera4_hardware_bridge` nodes connecting to `192.168.1.101`, `.102`, and `.103`.
2. Connects all robots to the unified `world` frame via `tf2_ros`.
3. Starts `swarm_coordinator`, maintaining dynamic midpoint formation:
   - `khepera_1`: Anchor (`base_link`)
   - `khepera_2`: RF Relay Extender (`extender`)
   - `khepera_3`: Frontier Explorer Scout (`explorer`)
4. Launches the Qt5 Multi-Robot Control Dashboard displaying live speeds, battery percentages, and formation states.

#### 2. Physical Single-Robot SLAM & Mapping
To build a 2D occupancy grid and 2.5D elevation map using physical IR and sonar sensors:

```bash
ros2 launch khepera4_control mapping.launch.py \
    dashboard:=true \
    rviz:=true
```
*(In hardware mode, map updates are driven by the physical Khepera's calibrated sensor fusion engine).*

---

## Writing Custom Python Scripts for Physical Khepera IV

Below is a production-ready template for writing custom ROS 2 Python scripts that command the physical Khepera IV.

Save this script as `my_khepera_controller.py`:

```python
#!/usr/bin/env python3
"""
Custom Autonomous Controller for Physical Khepera IV Robot.
Compatible with khepera4_hardware_bridge via ROS 2 Twist interface.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32MultiArray
import math

class KheperaPhysicalController(Node):
    def __init__(self):
        super().__init__('khepera_physical_controller')

        # Parameter: Target robot namespace (e.g. khepera_1)
        self.declare_parameter('robot_name', 'khepera_1')
        self.robot_name = self.get_parameter('robot_name').value

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        self.ir_sub = self.create_subscription(Float32MultiArray, f'/{self.robot_name}/ir_sensors', self.ir_callback, 10)
        self.battery_sub = self.create_subscription(BatteryState, f'/{self.robot_name}/battery_state', self.battery_callback, 10)

        # State Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.front_ir_distance = 1.0
        self.battery_voltage = 7.4

        # Control Loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"Initialized Custom Controller for physical {self.robot_name}")

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def ir_callback(self, msg: Float32MultiArray):
        # Khepera IV has 8 proximity IR sensors. Sensors 2 and 3 face forward.
        if len(msg.data) >= 8:
            self.front_ir_distance = min(msg.data[2], msg.data[3])

    def battery_callback(self, msg: BatteryState):
        self.battery_voltage = msg.voltage
        if self.battery_voltage < 6.7:
            self.get_logger().warn(f"BATTERY CRITICAL: {self.battery_voltage:.2f}V! Please dock the robot.")

    def control_loop(self):
        cmd = Twist()

        # Emergency battery stop
        if self.battery_voltage < 6.6:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # Simple reactive obstacle avoidance:
        # If obstacle detected ahead (< 15 cm or sensor threshold), rotate in place
        if self.front_ir_distance > 500.0:  # High IR reading indicates close obstacle
            cmd.linear.x = 0.0
            cmd.angular.z = 0.8  # Turn right (rad/s)
        else:
            cmd.linear.x = 0.10  # Cruise forward at 10 cm/s
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Publish zero velocity on shutdown."""
        halt = Twist()
        self.cmd_pub.publish(halt)

def main(args=None):
    rclpy.init(args=args)
    node = KheperaPhysicalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
```

To run this custom script on the lab PC:
```bash
python3 my_khepera_controller.py --ros-args -p robot_name:=khepera_1
```

---

## Safety, Emergency Halting & Diagnostics Playbook

### 1. Emergency Stop (E-Stop) Procedures
If a robot loses control or heads toward a table edge:
1. **Software Fleet E-Stop (Fastest):**
   ```bash
   ./physical_robot/fleet_manager.sh stop
   ```
2. **Terminal ROS 2 Zero Override:**
   ```bash
   ros2 topic pub --once /khepera_1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
   ```
3. **Physical E-Stop:** Pick up the robot from behind its drive wheels or immediately flip the rear power toggle switch to `OFF`.

### 2. Diagnostics & Troubleshooting Matrix

| Symptom | Probable Cause | Diagnostic Command / Fix |
|---|---|---|
| `./fleet_manager.sh ping` reports `OFFLINE` | Wi-Fi disconnected or wrong IP in `.env` | Run `ip route` on PC; verify robot is on same Wi-Fi SSID; verify IP on robot LCD screen. |
| `Hardware Bridge` outputs `Connection refused` | Daemon not running on robot | Run `./physical_robot/fleet_manager.sh start` or SSH in: `ssh root@<IP>` then run `pidof khepera4_tcp_server`. |
| Motors stutter or stop every 0.5s | Safety watchdog triggering due to dropped Wi-Fi packets | Check Wi-Fi signal strength (`iwconfig` on robot); ensure control loop publishes `cmd_vel` at $\ge 10\text{ Hz}$. |
| Robot suddenly reboots when accelerating | Battery voltage collapse ($< 6.5\text{ V}$) | Monitor `/battery_state`; place robot on charging dock until voltage exceeds $7.8\text{ V}$. |
| Odometry drifts rapidly during turns | Wheel slip or dirty caster wheels | Clean metal casters; check tire rubber adhesion; calibrate `WHEEL_BASE` ($0.1054\text{ m}$) in `khepera4_hardware_bridge.py`. |
| Permission denied during `fleet_manager.sh deploy` | SSH key or password mismatch | Connect via `ssh root@<IP>`; default password on Poky Yocto is empty (just press Enter). |

---

## Final Verification Checklist Before Running Tests

- [ ] Host computer connected to dedicated lab Wi-Fi router (No campus client isolation).
- [ ] Robot battery voltage verified $\ge 7.0\text{ V}$.
- [ ] Robots respond to `./physical_robot/fleet_manager.sh ping`.
- [ ] Daemons running (`./physical_robot/fleet_manager.sh status` reports `DAEMON ACTIVE`).
- [ ] ROS 2 workspace built and sourced (`source install/setup.bash`).
- [ ] E-stop command tested and ready on a separate terminal.
