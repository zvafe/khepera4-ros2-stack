#!/usr/bin/env bash
# ==============================================================================
# One-Click Physical Khepera IV Visual Pursuit Launcher
# ==============================================================================
# Usage:
#   ./run_physical_pursuit.sh [ROBOT_IP] [--mock] [extra ros2 launch args...]
#
# Examples:
#   ./run_physical_pursuit.sh                        # Auto-detects IP from hotspot/ARP
#   ./run_physical_pursuit.sh 10.42.0.45            # Direct IP connect
#   ./run_physical_pursuit.sh --mock                 # Dry-run emulation without hardware
# ==============================================================================

set -e

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_DIR"

# Source ROS 2 Jazzy environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace artifacts
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
else
    echo "⚠️  Workspace not built. Building required packages..."
    colcon build --symlink-install --packages-select khepera4_control khepera_visual_pursuit
    source "$WS_DIR/install/setup.bash"
fi

ROBOT_IP=""
MOCK="false"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)
            MOCK="true"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_physical_pursuit.sh [ROBOT_IP] [--mock] [extra args...]"
            exit 0
            ;;
        *)
            if [[ -z "$ROBOT_IP" && "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                ROBOT_IP="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# If no IP provided and not mock, attempt auto-discovery from local hotspot / ARP cache
if [[ -z "$ROBOT_IP" && "$MOCK" == "false" ]]; then
    echo "🔍 Scanning local hotspot and ARP cache for active Khepera IV on port 5000..."
    DETECTED_IP=$(python3 -c "
import socket, os

candidates = []
if os.path.isfile('/proc/net/arp'):
    with open('/proc/net/arp', 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.split()
            if len(parts) >= 4 and parts[2] != '0x0':
                ip = parts[0].strip()
                if ip not in ('0.0.0.0', '255.255.255.255') and not ip.startswith('127.'):
                    candidates.append(ip)

for base in ('10.42.0.', '192.168.137.', '192.168.1.'):
    for host in range(2, 60):
        c = f'{base}{host}'
        if c not in candidates:
            candidates.append(c)

found = None
for ip in candidates:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.08)
        if s.connect_ex((ip, 5000)) == 0:
            found = ip
            s.close()
            break
        s.close()
    except Exception:
        pass

if found:
    print(found)
" 2>/dev/null || true)

    if [[ -n "$DETECTED_IP" ]]; then
        ROBOT_IP="$DETECTED_IP"
        echo "✅ Detected Khepera IV robot at IP: $ROBOT_IP"
    else
        ROBOT_IP="192.168.1.101"
        echo "ℹ️  No robot detected on port 5000. Defaulting to: $ROBOT_IP"
    fi
fi

if [[ -z "$ROBOT_IP" ]]; then
    ROBOT_IP="192.168.1.101"
fi

echo "==================================================================="
echo "🚀 Launching Physical Khepera IV Visual Pursuit Stack"
echo "   Target Robot IP : $ROBOT_IP"
echo "   Hardware Mock   : $MOCK"
echo "   Mode            : Physical Deployment (TCP 5000 Control, 5001 Camera)"
echo "==================================================================="

exec ros2 launch khepera_visual_pursuit physical_visual_pursuit.launch.py \
    robot_ip:="$ROBOT_IP" \
    mock:="$MOCK" \
    "${EXTRA_ARGS[@]}"
