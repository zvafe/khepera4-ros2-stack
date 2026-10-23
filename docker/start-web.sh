#!/usr/bin/env bash
set -e

# Cleanup stale X11 lock files from previous runs
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true

export DISPLAY=:1
export RESOLUTION=${RESOLUTION:-1920x1080x24}

echo "[start-web] =========================================================="
echo "[start-web] Starting Khepera IV Web Simulation Stack"
echo "[start-web] Display: ${DISPLAY} | Resolution: ${RESOLUTION}"
echo "[start-web] =========================================================="

# 1. Start Xvfb virtual framebuffer X server
echo "[start-web] Launching Xvfb virtual X server..."
Xvfb ${DISPLAY} -screen 0 ${RESOLUTION} -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to become fully operational
echo "[start-web] Waiting for Xvfb to initialize..."
for i in {1..30}; do
    if xdpyinfo -display ${DISPLAY} >/dev/null 2>&1; then
        echo "[start-web] Xvfb is ready on ${DISPLAY}."
        break
    fi
    sleep 0.2
done

# 2. Start Fluxbox window manager
echo "[start-web] Launching Fluxbox window manager..."
fluxbox &
FLUXBOX_PID=$!

# 3. Start x11vnc VNC server (listening on 127.0.0.1:5900)
echo "[start-web] Launching x11vnc on 127.0.0.1:5900..."
x11vnc -display ${DISPLAY} \
       -nopw \
       -listen 127.0.0.1 \
       -rfbport 5900 \
       -xkb \
       -ncache 10 \
       -ncache_cr \
       -forever \
       -shared \
       -bg

# 4. Start websockify / noVNC web proxy (listening on container port 80)
echo "[start-web] Launching websockify (port 80 -> 127.0.0.1:5900)..."
websockify --web /usr/share/novnc 80 127.0.0.1:5900 &
WEBSOCKIFY_PID=$!

# Source ROS 2 and workspace environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
fi
if [ -f "/workspace/install/setup.bash" ]; then
    source "/workspace/install/setup.bash"
fi
export GZ_SIM_RESOURCE_PATH="/workspace/install/khepera4_description/share:/workspace/src/khepera4_description/meshes:/workspace/src/khepera4_description/worlds:/workspace/src:${GZ_SIM_RESOURCE_PATH:-}"

echo "[start-web] Stack is fully running! Open browser at http://localhost:${KHEPERA_WEB_PORT:-6080}"

# Graceful cleanup on container stop
cleanup() {
    echo "[start-web] Stopping all services..."
    kill -TERM "$WEBSOCKIFY_PID" "$FLUXBOX_PID" "$XVFB_PID" 2>/dev/null || true
    pkill x11vnc 2>/dev/null || true
    wait
    exit 0
}
trap cleanup SIGTERM SIGINT

# 5. Handle command execution or launch default interactive xterm console
if [ "$#" -gt 0 ]; then
    echo "[start-web] Running user command: $@"
    "$@" &
    CMD_PID=$!
    wait "$CMD_PID"
else
    echo "[start-web] Launching default control terminal in virtual desktop..."
    xterm -geometry 115x35+50+50 -title "Khepera IV ROS 2 Control Console" -hold -e "
        source /opt/ros/jazzy/setup.bash
        if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi
        export GZ_SIM_RESOURCE_PATH=\"/workspace/install/khepera4_description/share:/workspace/src/khepera4_description/meshes:/workspace/src/khepera4_description/worlds:/workspace/src:\$GZ_SIM_RESOURCE_PATH\"
        echo '=================================================================='
        echo '  BRT K4 - Khepera IV ROS 2 Jazzy & Gazebo Harmonic Simulation'
        echo '=================================================================='
        echo ' Quick Start Commands:'
        echo '   1. Build Workspace:     colcon build --symlink-install'
        echo '   2. Launch Simulation:   ros2 launch khepera4_description khepera4_sim.launch.py'
        echo '   3. Control Station GUI: ros2 launch khepera4_control control_station.launch.py'
        echo '   4. Keyboard Teleop:     ros2 run teleop_twist_keyboard teleop_twist_keyboard'
        echo '   5. Closed-loop Test:    ros2 launch khepera4_control closed_loop.launch.py'
        echo '   6. Run Benchmark:       ros2 launch khepera4_control benchmark.launch.py'
        echo '=================================================================='
        echo ' Right-click on the desktop anytime to open the Fluxbox menu.'
        echo '=================================================================='
        exec bash
    " &
    wait "$WEBSOCKIFY_PID"
fi
