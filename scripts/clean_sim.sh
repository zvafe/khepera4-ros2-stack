#!/usr/bin/env bash
# Clean and terminate all orphaned or lingering Gazebo, ROS 2, and Khepera processes
echo "Stopping all Gazebo Harmonic, ROS 2 bridges, RViz2, and Khepera nodes..."
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ros_gz" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "khepera4" 2>/dev/null || true
pkill -9 -f "mapping_dashboard" 2>/dev/null || true
pkill -9 -f "reactive_explorer" 2>/dev/null || true
sleep 1
echo "All simulation processes successfully terminated."
