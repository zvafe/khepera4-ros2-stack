#!/usr/bin/env bash
# ==============================================================================
# Clean and terminate all orphaned or lingering Gazebo, ROS 2, and Khepera simulation nodes
# ==============================================================================

set -e

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Stopping Gazebo Harmonic, ROS 2 bridges, RViz2, and simulation nodes..."

# Helper: safely kill matching processes excluding the current script and parent shell
safe_kill() {
    local pattern="$1"
    # Find PIDs matching pattern, exclude self and parent PID
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null | grep -v "$$" | grep -v "$PPID" || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
}

# 1. Kill specific simulation processes
safe_kill "gz sim"
safe_kill "ros_gz"
safe_kill "parameter_bridge"
safe_kill "robot_state_publisher"
safe_kill "rviz2"
safe_kill "mapping_dashboard"
safe_kill "reactive_explorer"
safe_kill "closed_loop_controller"
safe_kill "open_loop_benchmark"
safe_kill "teleop_twist_keyboard"
safe_kill "khepera4_hardware_bridge"
safe_kill "khepera4_mapper"
safe_kill "swarm_coordinator"

# 2. Stop any running Docker simulation containers
if command -v docker >/dev/null 2>&1 && [ -f "${WS_ROOT}/docker/docker-compose.yml" ]; then
    docker compose -f "${WS_ROOT}/docker/docker-compose.yml" down --remove-orphans >/dev/null 2>&1 || true
fi

sleep 0.5
echo "✓ All simulation processes and containers successfully cleaned and terminated."
