#!/usr/bin/env bash
# ==============================================================================
# Clean and terminate all orphaned or lingering Gazebo, ROS 2, and Khepera processes
# ==============================================================================

set -e

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Stopping all Gazebo Harmonic, ROS 2 bridges, RViz2, and Khepera nodes..."

# 1. Kill native processes
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ros_gz" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "khepera4" 2>/dev/null || true
pkill -9 -f "mapping_dashboard" 2>/dev/null || true
pkill -9 -f "reactive_explorer" 2>/dev/null || true
pkill -9 -f "closed_loop_controller" 2>/dev/null || true
pkill -9 -f "teleop_twist_keyboard" 2>/dev/null || true

# 2. Stop any running Docker simulation containers
if command -v docker >/dev/null 2>&1 && [ -f "${WS_ROOT}/docker/docker-compose.yml" ]; then
    docker compose -f "${WS_ROOT}/docker/docker-compose.yml" down --remove-orphans >/dev/null 2>&1 || true
fi

sleep 1
echo "✓ All simulation processes and containers successfully cleaned and terminated."
