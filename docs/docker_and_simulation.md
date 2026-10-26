# Docker & Containerized Simulation Guide
**Project:** BRT K4 Khepera IV ROS 2 Stack  
**ROS Distro:** ROS 2 Jazzy Jalisco | **Simulator:** Gazebo Harmonic (GZ Sim 8)  
**Base Image:** Ubuntu 24.04 LTS (Noble Numbat)

---

## 1. Prerequisites & Container Runtime Installation

### 1.1 Linux Native (Ubuntu / Debian)

Install Docker Engine and the official Docker Compose v2 plugin:

```bash
# 1. Set up Docker's official GPG key & repository
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 2. Install Docker Engine, CLI, and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 3. Add current user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER
# NOTE: Log out and log back in, or run 'newgrp docker' to apply group changes
```

**Verify Installation:**
```bash
docker --version
# Expected: Docker version 26.x or newer
docker compose version
# Expected: Docker Compose version v2.x.x
```

### 1.2 Windows 11 (Docker Desktop + WSL2)

1. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. In Docker Desktop Settings:
   - **General:** Check `Use the WSL 2 based engine`.
   - **Resources -> WSL Integration:** Enable integration with your default Ubuntu distribution (`Ubuntu-24.04`).
3. Inside your WSL2 terminal, verify Docker availability:
   ```bash
   docker compose version
   ```

---

## 2. Universal Interactive Launcher (`run.sh`)

The repository includes a turnkey bash launcher [`run.sh`](file:///home/andy/Desktop/khepera4_ws/run.sh) located in the workspace root. It manages environment configuration, port checks, display routing, and container lifecycle.

### 2.1 First-Time Interactive Wizard
Run from the workspace root:
```bash
./run.sh
```
When executed without a `.env` file, the script presents an interactive configuration wizard:
```text
Select default simulation mode:
 1) Capable PC - Native GUI (WSLg / GPU pass-through)
 2) Web Browser - Quick demo (noVNC on http://localhost:6080)
Enter choice [1 or 2] (default: 1): 
```
Your choice is stored in `.env` for future runs.

### 2.2 Complete Command-Line Flag Reference

| Command Flag | Description | Execution Target |
|---|---|---|
| `./run.sh` | Launch using the default mode defined in `.env` | Active profile in `.env` |
| `./run.sh --mapping` | Launch single-robot 2D/2.5D SLAM & Exploration stack | `mapping.launch.py` with dashboard |
| `./run.sh --multi` | Launch 3-robot autonomous fleet station in Gazebo | `multi_robot_station.launch.py` |
| `./run.sh --native` | Force Native GUI mode (GPU / X11 / WSLg windowing) | Profile `native` |
| `./run.sh --web` | Force Web Browser mode (noVNC HTML5 virtual desktop) | Profile `web` |
| `./run.sh --build` | Force rebuild of Docker container images without cache | Image build stage |
| `./run.sh --config` | Re-run interactive configuration wizard and rewrite `.env` | Config wizard |
| `./run.sh --help` | Display detailed flag syntax and examples | Help output |
| `./run.sh [MODE] [CMD...]`| Execute explicit commands directly inside the container | Container shell |

*Example: Execute tests directly in Docker:*
```bash
./run.sh --native colcon test --packages-select khepera4_control
```

---

## 3. Simulation Modes Breakdown

| Characteristic | `khepera-native` (Native GUI) | `khepera-web` (Web Browser) |
|---|---|---|
| **Target Machines** | High-performance Linux PCs, Windows 11 with WSLg | macOS, low-spec laptops, institutional lab PCs |
| **Window System** | Host X11 / Wayland server | In-container virtual Xvfb + Fluxbox |
| **Display Forwarding** | `/tmp/.X11-unix` and `/dev/dri` | WebSocket HTML5 noVNC (`http://localhost:6080`) |
| **GPU Acceleration** | Hardware acceleration (NVIDIA / Intel / AMD / WSLg) | Software rasterization (`llvmpipe`) or host vGPU |
| **Network Mode** | `network_mode: "host"` (Native DDS discovery) | Port mapping `${KHEPERA_WEB_PORT:-6080}:80` |
| **Display Latency** | Near-zero (native host window rendering) | ~20–40 ms (WebSocket streaming) |

---

## 4. Web Browser Mode (noVNC) Details

In Web mode, the container initializes a virtual X server, Fluxbox window manager, x11vnc, and websockify.

- **Main Web Interface:** [http://localhost:6080](http://localhost:6080)
- **Direct Auto-Connect:** [http://localhost:6080/vnc.html?autoconnect=true&resize=remote](http://localhost:6080/vnc.html?autoconnect=true&resize=remote)

### Port Allocation & Safety Warning
The web port is configured via `.env`:
```env
KHEPERA_WEB_PORT=6080
```

> [!WARNING]
> **Strict Port Restriction: NEVER bind to Port 1947.**
> Port `1947` is globally reserved for the **Sentinel HASP / SafeNet License Manager** (used by university CAD, MATLAB, and engineering software). Binding to port 1947 can disrupt campus license servers. `run.sh` contains automated guards preventing binding to port 1947.

---

## 5. Hardware Acceleration & Display Troubleshooting

### 5.1 NVIDIA GPU Passthrough on Linux
Install the NVIDIA Container Toolkit to enable CUDA and hardware-accelerated OpenGL in Docker:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 5.2 X11 Permissions ("Cannot connect to X server :0")
If a GUI fails to display on Linux:
```bash
xhost +local:root
```
On Wayland systems (Ubuntu 24.04 default), ensure XWayland is active and check display:
```bash
echo $DISPLAY
```

### 5.3 Low Frame Rates in Web Mode
If CPU usage is high or simulation drops frames in Web mode:
1. Launch Gazebo in headless mode:
   ```bash
   ./run.sh --native ros2 launch khepera4_control mapping.launch.py headless:=true
   ```
2. Adjust the virtual screen resolution in `.env`:
   ```env
   RESOLUTION=1280x720x24
   ```

---

## 6. VS Code Dev Containers

The workspace includes a turnkey Dev Container configuration in [`.devcontainer/devcontainer.json`](file:///home/andy/Desktop/khepera4_ws/.devcontainer/devcontainer.json).

1. Install Visual Studio Code and the **Dev Containers** extension (`ms-vscode-remote.remote-containers`).
2. Open the repository root directory in VS Code.
3. Press `F1` and select:
   ```text
   Dev Containers: Reopen in Container
   ```
4. The environment automatically equips you with ROS 2 language servers, Pylance, C/C++ debugging, and URDF syntax highlighters.
