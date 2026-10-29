#!/usr/bin/env bash
# ==============================================================================
# One-Click Environment Setup & Build Script for Khepera IV
# ==============================================================================

set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================================================================="
echo " Setting up Khepera IV Workspace Environment"
echo "=================================================================="

# 1. Ensure .env exists
if [ ! -f "${WS_ROOT}/.env" ] && [ -f "${WS_ROOT}/.env.example" ]; then
    cp "${WS_ROOT}/.env.example" "${WS_ROOT}/.env"
    echo "✓ Initialized .env configuration file."
fi

# 2. Setup based on host environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo ">> Configuring Native ROS 2 Jazzy workspace..."
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    cd "${WS_ROOT}"
    colcon build --symlink-install --packages-select khepera4_description khepera4_control
    # shellcheck disable=SC1091
    source "${WS_ROOT}/install/setup.bash"
    echo "✓ Native build complete and sourced."
else
    echo ">> Native ROS 2 not installed. Building Docker container image..."
    cd "${WS_ROOT}"
    "${WS_ROOT}/run.sh" --build
    echo "✓ Docker container built and ready."
fi

echo ""
echo ">> Environment setup complete! You can now run:"
echo "   ./menu.sh"
echo "=================================================================="
