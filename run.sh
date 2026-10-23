#!/usr/bin/env bash
set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${WS_ROOT}/.env"
ENV_EXAMPLE="${WS_ROOT}/.env.example"
COMPOSE_FILE="${WS_ROOT}/docker/docker-compose.yml"

print_help() {
    cat << EOF
Khepera IV (BRT K4) Simulation Launcher

Usage:
  ./run.sh [OPTIONS] [COMMAND...]

Options:
  --native        Launch high-performance Native GUI mode (WSLg / GPU pass-through)
  --web           Launch Web Browser mode (noVNC on http://localhost:\${KHEPERA_WEB_PORT:-6080})
  --build         Build or rebuild the Docker container image
  --config        Re-run interactive mode setup and update .env
  --help, -h      Display this help message

Examples:
  ./run.sh                                    Launch default configured mode
  ./run.sh --web                              Launch browser-based simulation
  ./run.sh --native                           Launch native X11/Wayland GUI
  ./run.sh --build                            Rebuild Docker image
  ./run.sh --native ros2 launch khepera4_description khepera4_sim.launch.py
EOF
}

configure_interactive() {
    local default_port="${KHEPERA_WEB_PORT:-6080}"
    if [ "$default_port" = "1947" ]; then
        echo "[WARNING] Port 1947 detected. This port conflicts with Sentinel HASP / SafeNet License Manager. Resetting to 6080."
        default_port=6080
    fi

    echo "=================================================================="
    echo "  BRT K4: Khepera IV ROS 2 Jazzy & Gazebo Harmonic Simulation"
    echo "=================================================================="
    echo "Select default simulation mode:"
    echo " 1) Capable PC - Native GUI (WSLg / GPU pass-through)"
    echo " 2) Web Browser - Quick demo (noVNC on http://localhost:${default_port})"

    local choice="1"
    if [ -t 0 ]; then
        read -r -p "Enter choice [1 or 2] (default: 1): " user_choice
        choice="${user_choice:-1}"
    elif read -r user_choice; then
        choice="${user_choice:-1}"
    else
        echo "Non-interactive environment detected without input. Using default (1 - Native GUI)."
        choice="1"
    fi

    local selected_mode="native"
    if [ "$choice" = "2" ] || [ "$choice" = "web" ]; then
        selected_mode="web"
    fi

    cat << EOF > "${ENV_FILE}"
# ==============================================================================
# BRT K4: Khepera IV ROS 2 Jazzy Simulation Configuration
# ==============================================================================
KHEPERA_DEFAULT_MODE=${selected_mode}
KHEPERA_WEB_PORT=${default_port}
COMPOSE_PROFILES=${selected_mode}
RESOLUTION=1920x1080x24
ROS_DOMAIN_ID=0
EOF

    echo "Configuration saved to ${ENV_FILE}:"
    echo "  Default Mode: ${selected_mode}"
    echo "  Web Port:     ${default_port}"
    echo "=================================================================="
}

# 1. Parse command line flags
CLI_MODE=""
DO_BUILD=false
DO_CONFIG=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)
            CLI_MODE="native"
            shift
            ;;
        --web)
            CLI_MODE="web"
            shift
            ;;
        --build)
            DO_BUILD=true
            shift
            ;;
        --config)
            DO_CONFIG=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 2. Check for first boot or explicit reconfiguration
if [ ! -f "${ENV_FILE}" ] || [ "$DO_CONFIG" = true ]; then
    configure_interactive
    if [ "$DO_CONFIG" = true ] && [ "$DO_BUILD" = false ] && [ -z "$CLI_MODE" ] && [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
        exit 0
    fi
fi

# Load variables from .env
# shellcheck disable=SC1090
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

# Safeguard against conflicting port 1947
if [ "${KHEPERA_WEB_PORT}" = "1947" ]; then
    echo "[WARNING] Port 1947 conflicts with Sentinel HASP / SafeNet License Manager! Re-routing to 6080."
    KHEPERA_WEB_PORT=6080
fi

# 3. Handle Docker build request
if [ "$DO_BUILD" = true ]; then
    echo "Building Khepera IV Docker container images..."
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" build
    if [ -z "$CLI_MODE" ] && [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
        echo "Build completed successfully."
        exit 0
    fi
fi

# Ensure image exists; if not, build it automatically
if ! docker image inspect khepera4:latest >/dev/null 2>&1; then
    echo "Image 'khepera4:latest' not found. Performing initial build..."
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" build
fi

# 4. Determine execution mode
TARGET_MODE="${CLI_MODE:-${KHEPERA_DEFAULT_MODE:-native}}"

# 5. Launch according to mode
if [ "$TARGET_MODE" = "native" ]; then
    echo "=================================================================="
    echo " Starting Khepera IV in Native GUI Mode (WSLg / GPU Pass-through)"
    echo "=================================================================="
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:root >/dev/null 2>&1 || xhost + >/dev/null 2>&1 || true
    fi

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        exec docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" run --rm khepera-native "${EXTRA_ARGS[@]}"
    else
        exec docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" run --rm khepera-native bash
    fi

elif [ "$TARGET_MODE" = "web" ]; then
    PORT="${KHEPERA_WEB_PORT:-6080}"
    echo "=================================================================="
    echo " Starting Khepera IV in Web Browser Mode (noVNC)"
    echo " Access simulation at:"
    echo "   Web Portal:  http://localhost:${PORT}"
    echo "   Auto-connect: http://localhost:${PORT}/vnc.html?autoconnect=true&resize=remote"
    echo "=================================================================="

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        exec docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" run --rm --service-ports khepera-web "${EXTRA_ARGS[@]}"
    else
        exec docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up khepera-web
    fi
else
    echo "Error: Unknown simulation mode '${TARGET_MODE}'. Use --native or --web."
    exit 1
fi
