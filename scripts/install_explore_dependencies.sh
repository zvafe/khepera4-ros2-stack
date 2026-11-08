#!/usr/bin/env bash
# ==============================================================================
# Install ROS 2 Jazzy Nav2 & SLAM Toolbox dependencies and setup explore_lite
# ==============================================================================
set -e

echo ">> Installing ROS 2 Jazzy Nav2 & SLAM Toolbox packages..."
sudo apt-get update
sudo apt-get install -y \
    ros-jazzy-slam-toolbox \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-nav2-costmap-2d

WS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
if [ ! -d "${WS_SRC}/m-explore-ros2" ]; then
    echo ">> Cloning m-explore-ros2 (explore_lite) into src/..."
    git clone https://github.com/robo-friends/m-explore-ros2.git "${WS_SRC}/m-explore-ros2"
else
    echo ">> m-explore-ros2 already present at ${WS_SRC}/m-explore-ros2."
fi

echo ">> Dependencies ready. Please build the workspace with:"
echo "   colcon build --symlink-install"
