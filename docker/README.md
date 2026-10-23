# Khepera IV (BRT K4) Docker & Simulation Environment Guide

This repository contains the complete Dockerized environment for the **Khepera IV Differential Drive Robot** project running on **ROS 2 Jazzy Jalisco** and **Gazebo Harmonic (gz-sim 8)** on **Ubuntu 24.04 (Noble Numbat)**.

---

## 🚀 Quick Start

The repository includes an interactive launcher script `run.sh` located in the workspace root.

### First-Time Launch

Run the launcher from the project root:
```bash
./run.sh
```

On first execution (when `.env` does not exist), you will be prompted:
```text
Select default simulation mode:
 1) Capable PC - Native GUI (WSLg / GPU pass-through)
 2) Web Browser - Quick demo (noVNC on http://localhost:6080)
Enter choice [1 or 2] (default: 1): 
```
Your selection will be saved into `.env` so subsequent runs will start immediately without re-prompting.

### Command-Line Flags

You can bypass defaults and trigger specific modes or actions using CLI flags:

| Command | Description |
|---|---|
| `./run.sh` | Launch with default configuration from `.env` |
| `./run.sh --native` | Force Native GUI mode (GPU / X11 / WSLg pass-through) |
| `./run.sh --web` | Force Web Browser mode (noVNC on port 6080) |
| `./run.sh --build` | Rebuild Docker container image |
| `./run.sh --config` | Re-run interactive setup wizard and update `.env` |
| `./run.sh --help` | Display command usage and flag descriptions |
| `./run.sh [MODE] [CMD...]` | Pass explicit ROS 2 commands directly to the container |

*Example of executing a command directly:*
```bash
./run.sh --native ros2 launch khepera4_description khepera4_sim.launch.py
```

---

## 🖥️ Simulation Modes Comparison

| Feature | `khepera-native` | `khepera-web` |
|---|---|---|
| **Target Hardware** | High-performance PCs, Linux Desktops, Windows 11 with WSLg | Low-spec laptops, macOS, college lab PCs, quick demos |
| **Display Server** | Host X11 / Wayland server | Virtual Xvfb with Fluxbox WM |
| **Access Method** | Native host windows | Web browser (HTML5 noVNC) |
| **GPU Acceleration** | Full hardware GPU pass-through (NVIDIA / Intel / AMD / WSLg) | Software rasterization (llvmpipe) or host vGPU |
| **Port Forwarding** | Not required (host network) | `${KHEPERA_WEB_PORT:-6080}:80` |
| **Client Requirements** | X11/WSLg display server | Any modern web browser |

---

## 🌐 Web Browser Mode (`khepera-web`)

In Web mode, a self-contained virtual desktop (Xvfb + Fluxbox + x11vnc + websockify) runs inside the container and streams live to your browser over WebSocket.

- **Web Portal URL:** [http://localhost:6080](http://localhost:6080)
- **Direct Auto-Connect URL:** [http://localhost:6080/vnc.html?autoconnect=true&resize=remote](http://localhost:6080/vnc.html?autoconnect=true&resize=remote)

### Custom Port Configuration

The web port is configured via the `.env` file in the workspace root:
```env
KHEPERA_WEB_PORT=6080
```

> [!WARNING]
> **Strict Port Restriction: Do NOT use Port 1947.**
> Port `1947` is reserved globally for the **Sentinel HASP / SafeNet Inc. License Manager**, commonly installed on academic and engineering workstations for CAD / MATLAB licenses. Binding to port 1947 will fail or corrupt university network licensing daemons. `run.sh` automatically checks and prevents binding to port 1947.

---

## 🪟 Windows 11 Setup Guide (WSL2 & Docker Desktop)

### Prerequisites

1. **Windows 11 (Build 22000 or higher)**: Provides native **WSLg** (Windows Subsystem for Linux GUI).
2. **WSL2 with Ubuntu 24.04 or 22.04**:
   ```powershell
   wsl --status
   wsl --update
   ```
3. **Docker Desktop for Windows**:
   - In Docker Desktop Settings -> **General**, ensure **Use the WSL 2 based engine** is checked.
   - In Settings -> **Resources** -> **WSL Integration**, enable integration with your default Ubuntu distribution.

### Running under WSL2

Always clone and run the workspace **inside the WSL2 filesystem** (e.g. `/home/<user>/...`), **not** on the Windows drive `/mnt/c/...`:

```bash
# Inside WSL2 Ubuntu terminal:
cd /home/<user>/
git clone <repo_url> khepera4_ws
cd khepera4_ws

# Launch in Native GUI mode (WSLg opens Gazebo windows directly on your Windows desktop)
./run.sh --native
```

WSLg automatically exposes `/tmp/.X11-unix` and `/dev/dri`, allowing Gazebo Harmonic and PyQt5 GUI tools to display hardware-accelerated windows alongside Windows applications.

If you encounter display issues on Windows, switch to Web mode:
```bash
./run.sh --web
```
Then open Google Chrome, Edge, or Firefox in Windows and navigate to `http://localhost:6080`.

---

## 🐧 Linux Setup Guide (Ubuntu / Debian / Fedora)

### Prerequisites

1. **Docker Engine & Docker Compose v2**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   sudo usermod -aG docker $USER
   # Log out and log back in for group changes to take effect
   ```

2. **Native X11 Permissions**:
   Before launching Native GUI mode, allow local connections to X11:
   ```bash
   xhost +local:root
   ```
   *(The `run.sh` script executes this automatically).*

3. **NVIDIA GPU Acceleration (Optional)**:
   If your machine has an NVIDIA GPU, install the NVIDIA Container Toolkit:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

---

## 📦 VS Code Dev Containers

This workspace includes a ready-to-use Dev Container configuration at `.devcontainer/devcontainer.json`.

1. Install Visual Studio Code and the **Dev Containers** extension (`ms-vscode-remote.remote-containers`).
2. Open the `/home/andy/Desktop/khepera4_ws` folder in VS Code.
3. Press `F1` (or `Ctrl+Shift+P`), type:
   ```text
   Dev Containers: Reopen in Container
   ```
4. VS Code will build and attach to the container. The environment comes pre-configured with:
   - **ROS 2 Tooling**: `ms-iot.vscode-ros`
   - **Python & Pylance**: Pre-configured with ROS 2 and workspace python packages
   - **C/C++ Tools**: IntelliSense and debugging
   - **URDF & XML Tools**: Syntax highlighting and model validation

---

## 🤖 Common Commands Inside the Container

Once inside the container shell (via `./run.sh --native` or the browser terminal in `./run.sh --web`):

### 1. Build Workspace
```bash
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Gazebo Harmonic Simulation
```bash
ros2 launch khepera4_description khepera4_sim.launch.py
```
*Headless simulation mode (physics server only):*
```bash
ros2 launch khepera4_description khepera4_sim.launch.py headless:=true
```

### 3. Launch Khepera IV Control Station GUI
```bash
ros2 launch khepera4_control control_station.launch.py
```
Provides real-time odometry plots, orientation telemetry, velocity sliders, and quick launch triggers.

### 4. Keyboard Teleoperation
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 5. Closed-Loop Controller
```bash
ros2 launch khepera4_control closed_loop.launch.py
```

### 6. Open-Loop Step Response Benchmark
```bash
ros2 launch khepera4_control benchmark.launch.py
```

### 7. Telemetry Logger
```bash
ros2 run khepera4_control telemetry_logger
```

---

## 🔧 Troubleshooting

### 1. "Cannot connect to X server :0" in Native Mode
- Run `xhost +local:root` on your host machine before starting Docker.
- In WSL2, verify that WSLg is running by checking `ls -la /tmp/.X11-unix`.

### 2. Port Conflict on Port 6080
- Check what process is using port 6080: `lsof -i :6080`
- Change `KHEPERA_WEB_PORT` in `.env` to another port, e.g. `KHEPERA_WEB_PORT=8080`.
- Remember to avoid port `1947`.

### 3. Gazebo Harmonic Mesh / World Not Found
- Verify that `GZ_SIM_RESOURCE_PATH` is sourced:
  ```bash
  echo $GZ_SIM_RESOURCE_PATH
  ```
  It should contain `/workspace/install/khepera4_description/share` and `/workspace/src`.
  The `/entrypoint.sh` sets this automatically.

### 4. Low Frame Rates in Web Mode
- Software rendering (`llvmpipe`) is CPU-based. In `docker-compose.yml`, adjust `shm_size: "2gb"` if you experience crashes.
- Lower the virtual screen resolution in `.env` if your CPU is constrained:
  ```env
  RESOLUTION=1280x720x24
  ```
