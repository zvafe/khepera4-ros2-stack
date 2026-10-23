FROM osrf/ros:jazzy-desktop

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=jazzy \
    QT_X11_NO_MITSHM=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install Gazebo Harmonic bridge, PyQt5, GUI tools, noVNC stack, and development utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Gazebo Harmonic & ROS 2 packages
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-interfaces \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro \
    ros-jazzy-rviz2 \
    ros-jazzy-teleop-twist-keyboard \
    # Python dependencies for control station & telemetry
    python3-colcon-common-extensions \
    python3-pip \
    python3-pyqt5 \
    python3-matplotlib \
    python3-numpy \
    # OpenGL / Mesa hardware acceleration and tools
    mesa-utils \
    mesa-utils-bin \
    libgl1-mesa-dri \
    libgl1 \
    libglx0 \
    libegl1 \
    libglu1-mesa \
    # Virtual display & VNC / noVNC web stack
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    x11-utils \
    x11-xserver-utils \
    xterm \
    # General utilities
    bash-completion \
    curl \
    wget \
    git \
    nano \
    net-tools \
    procps \
    psmisc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Configure noVNC root index redirect to autoconnect with remote-resize enabled
RUN mkdir -p /usr/share/novnc && \
    echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=true&resize=remote"><title>Khepera IV Simulation</title></head><body><p>Redirecting to <a href="vnc.html?autoconnect=true&resize=remote">Khepera IV Simulation Desktop</a>...</p></body></html>' > /usr/share/novnc/index.html

# Configure custom Fluxbox menu for Khepera IV
RUN mkdir -p /root/.fluxbox && cat << 'EOF' > /root/.fluxbox/menu
[begin] (Khepera IV)
  [exec] (Interactive Terminal) {xterm -geometry 110x35}
  [separator]
  [exec] (Launch Gazebo Simulation) {bash -c "source /opt/ros/jazzy/setup.bash && [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash; ros2 launch khepera4_description khepera4_sim.launch.py"}
  [exec] (Launch Control Station GUI) {bash -c "source /opt/ros/jazzy/setup.bash && [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash; ros2 launch khepera4_control control_station.launch.py"}
  [exec] (Launch RViz2) {bash -c "source /opt/ros/jazzy/setup.bash; rviz2"}
  [exec] (Keyboard Teleop) {xterm -geometry 80x24 -title "Teleop" -e "bash -c 'source /opt/ros/jazzy/setup.bash; ros2 run teleop_twist_keyboard teleop_twist_keyboard'"}
  [separator]
  [restart] (Restart Window Manager)
  [exit] (Exit)
[end]
EOF

# Copy entrypoint and web launcher scripts
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/start-web.sh /start-web.sh
RUN chmod +x /entrypoint.sh /start-web.sh

# Setup environment sourcing in root bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> /root/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=\"/workspace/install/khepera4_description/share:/workspace/src/khepera4_description/meshes:/workspace/src/khepera4_description/worlds:/workspace/src:\$GZ_SIM_RESOURCE_PATH\"" >> /root/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
