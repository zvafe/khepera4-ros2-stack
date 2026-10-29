#!/usr/bin/env bash
# ==============================================================================
# Khepera IV Interactive Control & Simulation Hub
# ==============================================================================

set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${WS_ROOT}/scripts"

# Terminal Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

clear_screen() {
    clear 2>/dev/null || printf "\033c"
}

clean_environment() {
    echo -e "${YELLOW}>> Cleaning environment and stopping all processes...${NC}"
    "${SCRIPTS_DIR}/clean_sim.sh"
    echo -e "${GREEN}✓ Environment is clean.${NC}"
}

setup_environment() {
    echo -e "${YELLOW}>> Setting up workspace environment...${NC}"
    if [ ! -f "${WS_ROOT}/.env" ] && [ -f "${WS_ROOT}/.env.example" ]; then
        cp "${WS_ROOT}/.env.example" "${WS_ROOT}/.env"
        echo -e "${GREEN}✓ Created .env configuration from example.${NC}"
    fi

    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        echo -e "${CYAN}>> Native ROS 2 Jazzy detected. Sourcing and compiling...${NC}"
        # shellcheck disable=SC1091
        source /opt/ros/jazzy/setup.bash
        cd "${WS_ROOT}"
        colcon build --symlink-install --packages-select khepera4_description khepera4_control
        # shellcheck disable=SC1091
        source "${WS_ROOT}/install/setup.bash"
        echo -e "${GREEN}✓ Workspace built and sourced successfully.${NC}"
    else
        echo -e "${CYAN}>> Native ROS 2 not found. Ensuring Docker container is ready...${NC}"
        "${WS_ROOT}/run.sh" --build
        echo -e "${GREEN}✓ Docker environment built and ready.${NC}"
    fi
}

ask_execution_platform() {
    # If running with native ROS 2 installed, user can choose Native or Docker
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        echo ""
        echo -e "${BOLD}Select Execution Environment:${NC}"
        echo " 1) Native Host (Direct ROS 2 & Gazebo on Ubuntu)"
        echo " 2) Docker Container - Native GUI (WSLg / GPU pass-through)"
        echo " 3) Docker Container - Web Browser (noVNC on port 6080)"
        read -r -p "Enter environment [1-3] (default: 1): " env_choice
        ENV_CHOICE="${env_choice:-1}"
    else
        echo ""
        echo -e "${BOLD}Select Docker Display Mode:${NC}"
        echo " 1) Native GUI (X11 / WSLg window)"
        echo " 2) Web Browser (HTML5 noVNC at http://localhost:6080)"
        read -r -p "Enter mode [1-2] (default: 1): " env_choice
        if [ "${env_choice}" = "2" ]; then
            ENV_CHOICE="3"
        else
            ENV_CHOICE="2"
        fi
    fi
}

dispatch_ros2_cmd() {
    local cmd="$*"
    clean_environment

    echo -e "${GREEN}>> Executing: ${CYAN}${cmd}${NC}"
    echo ""

    if [ "${ENV_CHOICE}" = "1" ]; then
        # shellcheck disable=SC1091
        source /opt/ros/jazzy/setup.bash
        if [ -f "${WS_ROOT}/install/setup.bash" ]; then
            # shellcheck disable=SC1091
            source "${WS_ROOT}/install/setup.bash"
        fi
        eval "${cmd}"
    elif [ "${ENV_CHOICE}" = "2" ]; then
        "${WS_ROOT}/run.sh" --native ${cmd}
    elif [ "${ENV_CHOICE}" = "3" ]; then
        echo -e "${YELLOW}>> Access web desktop at: http://localhost:6080/vnc.html?autoconnect=true&resize=remote${NC}"
        "${WS_ROOT}/run.sh" --web ${cmd}
    fi
}

# ==============================================================================
# Menus
# ==============================================================================

menu_single_robot() {
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}  SINGLE ROBOT: Simulation, Mapping, Benchmarking & Controls       ${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    echo " 1) 2D/2.5D Mapping Arena with Dashboard (Simulation)"
    echo " 2) 2D/2.5D Mapping Arena with Dashboard + RViz2 (Simulation)"
    echo " 3) Autonomous Reactive Exploration Arena (Auto-Explorer + Map)"
    echo " 4) Closed-Loop Position Control: Step to (x=1.0, y=0.5)"
    echo " 5) Closed-Loop Position Control: Waypoint Sequence"
    echo " 6) Kinematic Step-Response Benchmark (Forward 1m maneuver)"
    echo " 7) Pure 3D Gazebo Simulation Baseline (Default World)"
    echo " 8) Back to Main Menu"
    echo ""
    read -r -p "Select option [1-8]: " choice

    case "$choice" in
        1)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control mapping.launch.py dashboard:=true rviz:=false"
            ;;
        2)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control mapping.launch.py dashboard:=true rviz:=true"
            ;;
        3)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control mapping.launch.py auto_explore:=true dashboard:=true rviz:=true"
            ;;
        4)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control closed_loop.launch.py mode:=step goal_x:=1.0 goal_y:=0.5"
            ;;
        5)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control closed_loop.launch.py mode:=waypoint"
            ;;
        6)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control benchmark.launch.py maneuver:=forward speed:=0.15 duration:=6.67"
            ;;
        7)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_description khepera4_sim.launch.py"
            ;;
        8)
            return
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
}

menu_swarm_fleet() {
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}  SWARM / MULTI-ROBOT FLEET SIMULATION (3 Kheperas)               ${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    echo " 1) 3-Robot Formation Swarm + Multi-Robot Dashboard (Gazebo 3D)"
    echo " 2) 3-Robot Formation Swarm (Headless Simulation - Low Resource)"
    echo " 3) Back to Main Menu"
    echo ""
    read -r -p "Select option [1-3]: " choice

    case "$choice" in
        1)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control multi_robot_station.launch.py mode:=sim gui:=true"
            ;;
        2)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control multi_robot_station.launch.py mode:=headless gui:=true"
            ;;
        3)
            return
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
}

menu_mock_physical() {
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}  MOCK MODE (Hardware Bridge Emulation Without Gazebo / Real Robot)${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    echo " 1) Mock Single Robot: Hardware Bridge + Mapper + Dashboard"
    echo " 2) Mock Single Robot: Hardware Bridge + Mapper + Dashboard + RViz2"
    echo " 3) Mock 3-Robot Swarm: Fleet Station + Multi-Robot Dashboard"
    echo " 4) Standalone Mock Hardware Bridge (CLI only)"
    echo " 5) Back to Main Menu"
    echo ""
    read -r -p "Select option [1-5]: " choice

    case "$choice" in
        1)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control single_robot_physical.launch.py mock:=true dashboard:=true rviz:=false"
            ;;
        2)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control single_robot_physical.launch.py mock:=true dashboard:=true rviz:=true"
            ;;
        3)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control multi_robot_station.launch.py mode:=mock gui:=true"
            ;;
        4)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 run khepera4_control khepera4_hardware_bridge --ros-args -p mock:=true -p single_robot:=true"
            ;;
        5)
            return
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
}

menu_physical_lab() {
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}  PHYSICAL KHEPERA IV ROBOT (Classroom Lab Wi-Fi Connection)       ${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    read -r -p "Enter Khepera Robot IP (default: 192.168.1.101): " robot_ip_input
    ROBOT_IP="${robot_ip_input:-192.168.1.101}"

    echo ""
    echo " 1) Test Ping & Network Reachability to ${ROBOT_IP}"
    echo " 2) Connect Single Robot: Bridge + Mapper + Dashboard"
    echo " 3) Connect Single Robot: Bridge + Mapper + Dashboard + RViz2"
    echo " 4) Launch Keyboard Teleoperation (/cmd_vel)"
    echo " 5) Save Active 2D Map to maps/"
    echo " 6) Back to Main Menu"
    echo ""
    read -r -p "Select option [1-6]: " choice

    case "$choice" in
        1)
            ping -c 3 "${ROBOT_IP}" || echo -e "${RED}Robot not reachable at ${ROBOT_IP}. Check lab Wi-Fi/router.${NC}"
            read -r -p "Press [Enter] to continue..."
            ;;
        2)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control single_robot_physical.launch.py robot_ip:=${ROBOT_IP} dashboard:=true rviz:=false"
            ;;
        3)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 launch khepera4_control single_robot_physical.launch.py robot_ip:=${ROBOT_IP} dashboard:=true rviz:=true"
            ;;
        4)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel"
            ;;
        5)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 service call /save_map std_srvs/srv/Trigger"
            read -r -p "Press [Enter] to continue..."
            ;;
        6)
            return
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
}

menu_utilities() {
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}  STANDALONE TOOLS, RViz2 & DASHBOARDS                            ${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    echo " 1) Open Standalone RViz2 (with Mapping & TF Profile)"
    echo " 2) Open Standalone Mapping Dashboard"
    echo " 3) Open Standalone Multi-Robot Fleet Dashboard"
    echo " 4) Launch Keyboard Teleoperation"
    echo " 5) Run All Automated Unit & Acceptance Tests (pytest)"
    echo " 6) Back to Main Menu"
    echo ""
    read -r -p "Select option [1-6]: " choice

    case "$choice" in
        1)
            ask_execution_platform
            dispatch_ros2_cmd "rviz2 -d \$(ros2 pkg prefix --share khepera4_description)/config/khepera4_mapping.rviz"
            ;;
        2)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 run khepera4_control mapping_dashboard"
            ;;
        3)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 run khepera4_control control_dashboard"
            ;;
        4)
            ask_execution_platform
            dispatch_ros2_cmd "ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel"
            ;;
        5)
            ask_execution_platform
            dispatch_ros2_cmd "pytest src/khepera4_control/test/ -q"
            read -r -p "Press [Enter] to continue..."
            ;;
        6)
            return
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
}

# ==============================================================================
# Main Interactive Loop
# ==============================================================================

while true; do
    clear_screen
    echo -e "${CYAN}==================================================================${NC}"
    echo -e "${BOLD}       🤖 BRT K4 KHEPERA IV SIMULATION & CONTROL CENTER           ${NC}"
    echo -e "${CYAN}==================================================================${NC}"
    echo -e " ${BOLD}Simulation & Hardware Modes:${NC}"
    echo "   1) 🏎️  Single Robot Simulation (Gazebo, Mapping, Closed-Loop, Benchmarks)"
    echo "   2) 👥  Swarm Fleet Simulation (3 Robots, Midpoint Formation)"
    echo "   3) 🧪  Mock Mode (Hardware Emulation without Gazebo)"
    echo "   4) 🔌  Physical Khepera IV Robot (Lab Wi-Fi Connection)"
    echo ""
    echo -e " ${BOLD}Tools, Dashboards & Maintenance:${NC}"
    echo "   5) 🎛️  Dashboards, RViz2 & Keyboard Teleop"
    echo "   6) 🧹  Clean Environment (Terminate lingering Gazebo / ROS 2 nodes)"
    echo "   7) ⚙️   Setup / Rebuild Environment (Colcon / Docker image)"
    echo ""
    echo "   0) Exit"
    echo -e "${CYAN}==================================================================${NC}"
    read -r -p "Select an option [0-7]: " main_choice

    case "$main_choice" in
        1) menu_single_robot ;;
        2) menu_swarm_fleet ;;
        3) menu_mock_physical ;;
        4) menu_physical_lab ;;
        5) menu_utilities ;;
        6)
            clean_environment
            read -r -p "Press [Enter] to continue..."
            ;;
        7)
            setup_environment
            read -r -p "Press [Enter] to continue..."
            ;;
        0)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please enter 0-7.${NC}"
            sleep 1
            ;;
    esac
done
