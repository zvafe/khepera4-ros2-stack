#!/usr/bin/env bash
set -e

# Source ROS 2 Jazzy core environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
fi

# Ensure host workspace path compatibility for symlinked installs
if [ ! -e "/home/andy/Desktop/khepera4_ws" ]; then
    mkdir -p /home/andy/Desktop 2>/dev/null || true
    ln -s /workspace /home/andy/Desktop/khepera4_ws 2>/dev/null || true
fi

# Source Khepera IV colcon workspace if built
if [ -f "/workspace/install/setup.bash" ]; then
    source "/workspace/install/setup.bash"
fi

# Configure Gazebo Harmonic resource discovery paths
export GZ_SIM_RESOURCE_PATH="/workspace/install/khepera4_description/share:/workspace/src/khepera4_description/meshes:/workspace/src/khepera4_description/worlds:/workspace/src:${GZ_SIM_RESOURCE_PATH:-}"

# Backwards-compatibility aliases for ign and gazebo
export IGN_GAZEBO_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="${GZ_SIM_RESOURCE_PATH}:${GAZEBO_MODEL_PATH:-}"

exec "$@"
