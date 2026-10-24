1. All ROS 2 Commands to Run the Project (Native Linux Boot)

Before running any ROS 2 command in the project root (/home/andy/Desktop/khepera4_ws), always make sure the ROS 2 Jazzy environment and your workspace overlay are sourced:

bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

(Tip: If you build changes, re-run colcon build --symlink-install followed by source install/setup.bash.)

A. Main Launch Commands (ros2 launch)
1. Dedicated Single-Robot Mapping & Autonomous Exploration Stack

Runs Gazebo Harmonic with the obstacle course (khepera4_exploration_world.sdf), 17-sensor simulation, drift-free IMU yaw mapping engine, and the dedicated mapping dashboard:

bash
ros2 launch khepera4_control mapping.launch.py auto_explore:=true rviz:=true

Available Arguments:

auto_explore:=true|false (default: false — starts with manual teleop; true starts autonomous Braitenberg wander).
rviz:=true|false (default: false — launch RViz2 alongside the dashboard).
dashboard:=true|false (default: true — opens mapping_dashboard.py).
use_imu_yaw:=true|false (default: true — fused drift-free IMU yaw to prevent wheel slip skew).
2. Multi-Robot Fleet Swarm Control Station (3 Robots)

Spawns the 3 robots (khepera_1 [Base], khepera_2 [Relay], khepera_3 [Explorer]) in a clean empty arena (khepera4_world.sdf) with the fleet dashboard and dynamic swarm coordinator:

bash
ros2 launch khepera4_control multi_robot_station.launch.py

Available Arguments:

sim:=true|false (default: true — runs in Gazebo Harmonic; false runs without spawning sim).
rviz:=true|false (default: true — pre-configured RViz showing all 3 robot coordinate frames).
dashboard:=true|false (default: true — launches dashboard.py).
fleet_mode:=idle|relay|triangle (default: idle).
3. Single-Robot Closed-Loop Trajectory Control Station

Launches the single-robot simulation with the Lyapunov non-linear trajectory controller and control dashboard:

bash
ros2 launch khepera4_control control_station.launch.py
4. Automated Trajectory Benchmark

Runs automated trajectory validation (Eight-shape, Square, Circle) and outputs benchmark metrics to CSV:

bash
ros2 launch khepera4_control benchmark.launch.py
5. Raw Gazebo Harmonic Simulations (Without Dashboards)
Single Robot Simulation Only:
bash
ros2 launch khepera4_description khepera4_sim.launch.py
3-Robot Fleet Simulation Only:
bash
ros2 launch khepera4_description khepera4_multi_sim.launch.py
6. Physical 3-Robot Hardware Bridge

Launches hardware bridges connecting over Wi-Fi/Ethernet to the physical Khepera IV robots:

bash
ros2 launch khepera4_control multi_robot_physical.launch.py

B. Standalone Node Execution (ros2 run)

You can run individual nodes independently (e.g., in separate terminal tabs):

Node Purpose	Command
Dedicated Mapping Dashboard	ros2 run khepera4_control mapping_dashboard
Multi-Robot Fleet Dashboard	ros2 run khepera4_control control_dashboard
Swarm Coordinator		ros2 run khepera4_control swarm_coordinator
Persistent Grid Mapper		ros2 run khepera4_control khepera4_mapper
Autonomous Reactive Explorer	ros2 run khepera4_control reactive_explorer
Trajectory Controller		ros2 run khepera4_control closed_loop_controller
Hardware Bridge Node		ros2 run khepera4_control khepera4_hardware_bridge --ros-args -p robot_name:=khepera_1 -p robot_ip:=192.168.1.101

C. Physical Hardware Management Commands

Manage and inspect physical robots using the fleet utility script:

bash
# Check Wi-Fi ping and latency to all 3 physical robots
./physical_robot/fleet_manager.sh ping
# Check daemon health and battery voltages on physical robots	
./physical_robot/fleet_manager.sh status
# Start/Stop the low-level serial daemon on the physical robots
./physical_robot/fleet_manager.sh start
./physical_robot/fleet_manager.sh stop



D. Helpful Runtime ROS 2 Commands
Save Persistent Map to Disk via CLI:
bash
ros2 service call /khepera4_mapper/save_map std_srvs/srv/Trigger "{}"
Set Swarm Formation via CLI:
bash
ros2 topic pub --once /swarm/cmd std_msgs/msg/String "{data: 'mode:relay'}"
ros2 topic pub --once /swarm/cmd std_msgs/msg/String "{data: 'mode:triangle'}"
ros2 topic pub --once /swarm/cmd std_msgs/msg/String "{data: 'halt'}"
Echo Sensor Data:
bash
ros2 topic echo /imu/data
ros2 topic echo /khepera4/scan
ros2 topic echo /swarm/status



2. All Possible Ways of Running the Project
┌────────────────────────────────────────────────────────────────────────┐
│                        HOW TO RUN THE PROJECT                          │
├───────────────────┬────────────────────────────────────────────────────┤
│ 1. Native Linux   │ Run colcon build and ros2 launch directly in bash. │
│    (Recommended)  │ Fastest performance, direct GPU rendering, zero    │
│                   │ container overhead.                                │
├───────────────────┼────────────────────────────────────────────────────┤
│ 2. Linux Docker   │ Run ./run.sh [--native | --web | --multi | --map]   │
│    (Isolated)     │ Reproducible isolated environment with X11/GPU.   │
├───────────────────┼────────────────────────────────────────────────────┤
│ 3. Windows Docker │ Run .\run.ps1 -Native or .\run.ps1 -Web            │
│    (Cross-Dev)    │ Cross-platform testing via WSLg or noVNC browser.  │
├───────────────────┼────────────────────────────────────────────────────┤
│ 4. VS Code        │ One-click via Tasks (Ctrl+Shift+P) or Debugger     │
│    (IDE-Integrated)│ (F5) with pre-configured tasks and launch files.  │
└───────────────────┴────────────────────────────────────────────────────┘
