#!/usr/bin/env bash
# ==============================================================================
# Khepera IV Multi-Robot Fleet Manager & Deployment Utility
# Master PC Operations Script for 3 Physical Khepera IV Robots:
# - khepera_1: base_link
# - khepera_2: extender
# - khepera_3: explorer
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load IPs from .env if present
if [ -f "${WS_ROOT}/.env" ]; then
    set -a
    source "${WS_ROOT}/.env"
    set +a
fi

K1_IP="${KHEPERA_1_IP:-192.168.1.101}"
K2_IP="${KHEPERA_2_IP:-192.168.1.102}"
K3_IP="${KHEPERA_3_IP:-192.168.1.103}"
DAEMON_PORT="${KHEPERA_DAEMON_PORT:-5000}"

ROBOTS=(
    "khepera_1:base_link:${K1_IP}"
    "khepera_2:extender:${K2_IP}"
    "khepera_3:explorer:${K3_IP}"
)

print_header() {
    echo "=================================================================="
    echo "  Khepera IV Multi-Robot Fleet Manager (3 Physical Units)         "
    echo "=================================================================="
    echo " Fleet Configuration:"
    echo "   1) khepera_1 [base_link] -> ${K1_IP}:${DAEMON_PORT}"
    echo "   2) khepera_2 [extender]  -> ${K2_IP}:${DAEMON_PORT}"
    echo "   3) khepera_3 [explorer]  -> ${K3_IP}:${DAEMON_PORT}"
    echo "=================================================================="
}

cmd_ping() {
    print_header
    echo "Testing network reachability & latency..."
    for entry in "${ROBOTS[@]}"; do
        IFS=':' read -r name role ip <<< "$entry"
        printf " -> %-10s (%-9s) at %-15s : " "$name" "$role" "$ip"
        if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then
            rtt=$(ping -c 1 -W 1 "$ip" | grep 'time=' | sed -e 's/.*time=\(.*\) ms/\1/')
            echo -e "\033[32mONLINE\033[0m (${rtt} ms)"
        else
            echo -e "\033[31mOFFLINE / UNREACHABLE\033[0m"
        fi
    done
}

cmd_deploy() {
    print_header
    echo "Deploying onboard daemon & systemd service to robots..."
    local src_dir="${SCRIPT_DIR}/onboard_daemon"

    for entry in "${ROBOTS[@]}"; do
        IFS=':' read -r name role ip <<< "$entry"
        echo "------------------------------------------------------------------"
        echo "Deploying to ${name} (${ip})..."
        if ! ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then
            echo "[WARNING] ${name} (${ip}) is unreachable. Skipping."
            continue
        fi

        echo " 1. Copying source files..."
        scp -o ConnectTimeout=3 -o BatchMode=no "${src_dir}/khepera4_tcp_server.c" "${src_dir}/Makefile" root@"${ip}":/home/root/
        scp -o ConnectTimeout=3 -o BatchMode=no "${src_dir}/khepera4_server.service" root@"${ip}":/lib/systemd/system/

        echo " 2. Compiling binary on robot with libkhepera..."
        ssh -o ConnectTimeout=5 root@"${ip}" "cd /home/root && make clean && make && chmod +x khepera4_tcp_server"

        echo " 3. Enabling systemd autostart service..."
        ssh -o ConnectTimeout=5 root@"${ip}" "systemctl daemon-reload && systemctl enable khepera4_server && systemctl restart khepera4_server"
        echo "[SUCCESS] ${name} deployed and daemon started!"
    done
}

cmd_start() {
    print_header
    echo "Starting onboard daemons on all physical robots..."
    for entry in "${ROBOTS[@]}"; do
        IFS=':' read -r name role ip <<< "$entry"
        echo "Starting daemon on ${name} (${ip})..."
        ssh -o ConnectTimeout=3 root@"${ip}" "systemctl start khepera4_server || /home/root/khepera4_tcp_server 5000 &" || true
    done
}

cmd_stop() {
    print_header
    echo "Stopping onboard daemons and halting all physical robots..."
    for entry in "${ROBOTS[@]}"; do
        IFS=':' read -r name role ip <<< "$entry"
        echo "Stopping daemon on ${name} (${ip})..."
        ssh -o ConnectTimeout=3 root@"${ip}" "systemctl stop khepera4_server; killall khepera4_tcp_server 2>/dev/null" || true
    done
}

cmd_status() {
    print_header
    echo "Querying daemon process & battery status..."
    for entry in "${ROBOTS[@]}"; do
        IFS=':' read -r name role ip <<< "$entry"
        printf " -> %-10s (%-15s): " "$name" "$ip"
        if ! ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then
            echo -e "\033[31mOFFLINE\033[0m"
            continue
        fi
        status_out=$(ssh -o ConnectTimeout=3 root@"${ip}" "pidof khepera4_tcp_server >/dev/null && echo 'RUNNING' || echo 'STOPPED'")
        if [ "$status_out" = "RUNNING" ]; then
            echo -e "\033[32mDAEMON ACTIVE\033[0m"
        else
            echo -e "\033[33mDAEMON INACTIVE\033[0m"
        fi
    done
}

cmd_ssh() {
    local target="${1:-1}"
    case "$target" in
        1|khepera_1) ssh root@"${K1_IP}" ;;
        2|khepera_2) ssh root@"${K2_IP}" ;;
        3|khepera_3) ssh root@"${K3_IP}" ;;
        *) echo "Usage: ./fleet_manager.sh ssh [1|2|3]"; exit 1 ;;
    esac
}

cmd_stream_camera() {
    local target="${1:-1}"
    local target_ip="${K1_IP}"
    case "$target" in
        1|khepera_1) target_ip="${K1_IP}" ;;
        2|khepera_2) target_ip="${K2_IP}" ;;
        3|khepera_3) target_ip="${K3_IP}" ;;
    esac

    echo "Starting GStreamer camera receiver from ${target_ip}:5000..."
    echo "Ensure GStreamer sender is started on robot: 'gst-launch v4l2src device=/dev/video6 ! ... tcpserversink port=5000'"
    gst-launch-1.0 tcpclientsrc host="${target_ip}" port=5000 ! multipartdemux ! jpegdec ! videoconvert ! autovideosink || \
        vlc "tcp://${target_ip}:5000"
}

case "${1:-}" in
    ping) cmd_ping ;;
    deploy) cmd_deploy ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    ssh) cmd_ssh "${2:-1}" ;;
    stream-camera|camera) cmd_stream_camera "${2:-1}" ;;
    *)
        print_header
        echo "Usage: $0 {ping|deploy|start|stop|status|ssh [1-3]|stream-camera [1-3]}"
        echo ""
        echo "Commands:"
        echo "  ping           Check network latency to all 3 robots"
        echo "  deploy         Cross-compile/build and install TCP daemon onto robots"
        echo "  start          Start daemon on all robots via SSH"
        echo "  stop           Halt all robots and stop daemon"
        echo "  status         Check daemon process status on each robot"
        echo "  ssh [1-3]      Open interactive root SSH shell into chosen robot"
        echo "  stream-camera  Open live camera stream from chosen robot"
        exit 1
        ;;
esac
