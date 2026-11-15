# Khepera IV Physical Robot Deployment & Swarm Provisioning Suite

This directory contains the operational toolchain for programming, provisioning, and controlling single and multi-robot physical Khepera IV swarms over USB serial and Wi-Fi.

---

## 1. Quick Start: USB Serial Provisioning (`khepera_serial_provisioner.py`)

Used for **first-time unboxing, lab setup, hardware recovery, and swarm identity assignment**. Connect the mini-USB diagnostic cable from your PC to the Khepera IV (COMx on Windows, `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux).

```bash
# Launch interactive menu:
./physical_robot/khepera_serial_provisioner.py

# Or via fleet manager:
./physical_robot/fleet_manager.sh serial

# Or via interactive menu:
./menu.sh -> Option 4 -> Option 6
```

### Capabilities:
- **Zero-Network Code Deployment:** Streams and compiles `khepera4_tcp_server.c`, `khepera4_camera_streamer.sh`, and `autorun.sh` over pure serial (Base64), with SysVinit autostart on boot.
- **2.4 GHz Wi-Fi Provisioning:** Scans local SSIDs, configures `wpa_supplicant`, triggers DHCP, and syncs the assigned IP to the workstation `.env`.
- **Swarm Identity Assignment:** Configures robot swarm index (`khepera_1`, `khepera_2`, `khepera_3`), swarm role (`base_link`, `extender`, `explorer`), robot hostname, and port assignments.
- **Hardware Self-Test & Diagnostics:** Battery voltage verification, storage check, and 1-second motor spin with encoder/sensor echo.
- **Interactive Console:** Full VT100 pass-through root shell (`Ctrl+]` to exit).

### Non-Interactive Command-Line Examples:
```bash
# Push code over serial without prompting:
./physical_robot/khepera_serial_provisioner.py --port /dev/ttyUSB0 --deploy

# Provision Wi-Fi and assign as Swarm Robot 2 (extender):
./physical_robot/khepera_serial_provisioner.py --port /dev/ttyUSB0 \
    --wifi-ssid "MyHotspot" --wifi-pass "secret123" \
    --swarm-id 2 --swarm-role extender
```

---

## 2. Multi-Robot Fleet Manager (`fleet_manager.sh`)

Used for **over-the-air (OTA) Wi-Fi operations across 3 Khepera IV robots** simultaneously:
- `khepera_1`: Anchor (`base_link`)
- `khepera_2`: RF Relay (`extender`)
- `khepera_3`: Frontier Scout (`explorer`)

```bash
# Check connectivity and ping latency across all robots:
./physical_robot/fleet_manager.sh ping

# Deploy latest daemons and streamer over SSH:
./physical_robot/fleet_manager.sh deploy

# Start daemons on all robots:
./physical_robot/fleet_manager.sh start

# Stop / Emergency halt all robots:
./physical_robot/fleet_manager.sh stop

# Open root SSH shell to Robot 1, 2, or 3:
./physical_robot/fleet_manager.sh ssh 1

# Receive live GStreamer video feed from Robot 1:
./physical_robot/fleet_manager.sh stream-camera 1
```
