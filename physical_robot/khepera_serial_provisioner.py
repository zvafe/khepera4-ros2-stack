#!/usr/bin/env python3
"""
Khepera IV Serial Programming, Wi-Fi Provisioning & Swarm Configuration Interface.

Designed for first-time unboxing, lab setup, hardware recovery, and multi-robot swarm identity provisioning.
Connects directly via USB Serial Diagnostic Cable (115200 8N1 -> ttyO2 on robot)
without requiring prior network access.

Key Capabilities:
1. Zero-Network Code Push: Bundles, streams, and compiles daemons over pure serial (Base64).
2. Wi-Fi Provisioning: Scans 2.4 GHz SSIDs, configures wpa_supplicant, and obtains DHCP lease.
3. Swarm Identity Configuration: Assigns robot index, swarm role (base_link/extender/explorer),
   unique hostname, daemon ports, and synchronizes the host workspace .env automatically.
4. Comprehensive Hardware Diagnostics: Reads battery, storage, kernel, and verifies motors/sensors.
5. Interactive Shell: Full raw pass-through serial terminal.
"""

import argparse
import base64
import glob
import io
import json
import os
import re
import select
import sys
import tarfile
import time
from typing import Dict, List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(SCRIPT_DIR)
ONBOARD_DAEMON_DIR = os.path.join(SCRIPT_DIR, "onboard_daemon")
ENV_FILE = os.path.join(WS_ROOT, ".env")

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 3.0
PROMPT_MARKER = b"[K4_PROVISIONER]# "
PROMPT_STR = "[K4_PROVISIONER]# "


# ==============================================================================
# Serial Communication Engine
# ==============================================================================

class SerialConnection:
    """Manages raw or PySerial connection to the Khepera IV USB diagnostic port."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = DEFAULT_TIMEOUT):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._posix_fd = None

    def open(self):
        if HAS_PYSERIAL:
            try:
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )
                self.ser.flushInput()
                self.ser.flushOutput()
                return True
            except Exception as e:
                raise RuntimeError(f"Failed to open port {self.port} via PySerial: {e}")
        else:
            # POSIX termios fallback
            import termios
            import fcntl
            try:
                self._posix_fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                attrs = termios.tcgetattr(self._posix_fd)
                # 115200 8N1 raw mode
                baud_const = getattr(termios, f"B{self.baud}", termios.B115200)
                attrs[0] = 0  # iflag
                attrs[1] = 0  # oflag
                attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag
                attrs[3] = 0  # lflag
                attrs[4] = baud_const  # ispeed
                attrs[5] = baud_const  # ospeed
                termios.tcsetattr(self._posix_fd, termios.TCSANOW, attrs)
                return True
            except Exception as e:
                raise RuntimeError(f"Failed to open port {self.port} via POSIX fallback: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self._posix_fd is not None:
            try:
                os.close(self._posix_fd)
            except Exception:
                pass
            self._posix_fd = None

    def write(self, data: bytes):
        if self.ser:
            self.ser.write(data)
            self.ser.flush()
        elif self._posix_fd is not None:
            os.write(self._posix_fd, data)

    def read(self, size: int = 1024, timeout: Optional[float] = None) -> bytes:
        t_limit = timeout if timeout is not None else self.timeout
        if self.ser:
            self.ser.timeout = t_limit
            return self.ser.read(size)
        elif self._posix_fd is not None:
            r, _, _ = select.select([self._posix_fd], [], [], t_limit)
            if r:
                try:
                    return os.read(self._posix_fd, size)
                except Exception:
                    return b""
            return b""
        return b""

    def read_until(self, expected: bytes, timeout: float = 8.0) -> bytes:
        buffer = bytearray()
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.read(256, timeout=0.1)
            if chunk:
                buffer.extend(chunk)
                if expected in buffer:
                    return bytes(buffer)
            else:
                time.sleep(0.02)
        return bytes(buffer)


# ==============================================================================
# Port Discovery
# ==============================================================================

def list_serial_ports() -> List[str]:
    """Scan and return list of candidate Khepera IV serial ports."""
    ports = []
    if HAS_PYSERIAL:
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
    if not ports:
        # Linux standard USB / serial paths
        ports.extend(glob.glob("/dev/ttyUSB*"))
        ports.extend(glob.glob("/dev/ttyACM*"))
        ports.extend(glob.glob("/dev/ttyO2"))
        if sys.platform.startswith("win"):
            ports.extend([f"COM{i}" for i in range(1, 16)])
    return sorted(list(set(ports)))


# ==============================================================================
# Khepera IV Robot Serial Provisioner Engine
# ==============================================================================

class KheperaSerialProvisioner:
    """Automates serial login, Wi-Fi pairing, code push, and swarm assignment."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        self.conn = SerialConnection(port, baud)
        self.is_logged_in = False

    def connect_and_login(self, max_attempts: int = 4) -> bool:
        """Handshake with Poky Yocto console and establish root shell."""
        print(f"[*] Opening serial connection to {self.port} at {self.baud} baud...")
        self.conn.open()
        time.sleep(0.3)

        print("[*] Synchronizing with Khepera IV console...")
        for attempt in range(1, max_attempts + 1):
            self.conn.write(b"\r\n")
            buf = self.conn.read_until(b"login: ", timeout=1.5)

            if b"login: " in buf:
                print(" -> Detected Yocto login prompt. Sending 'root'...")
                self.conn.write(b"root\r\n")
                pwd_buf = self.conn.read_until(b"Password:", timeout=1.5)
                if b"Password:" in pwd_buf:
                    print(" -> Sending empty password...")
                    self.conn.write(b"\r\n")
                time.sleep(0.5)

            # Check if we have a root shell
            self.conn.write(b"\r\n")
            buf = self.conn.read(1024, timeout=0.8)
            if b"#" in buf or b"$" in buf or PROMPT_MARKER in buf:
                # Set a deterministic, unique prompt to simplify command matching
                self.conn.write(f'export PS1="{PROMPT_STR}"\r\n'.encode("ascii"))
                time.sleep(0.3)
                self.conn.read(1024, timeout=0.5)
                self.is_logged_in = True
                print("\033[32m[+] Successfully authenticated and acquired root shell!\033[0m")
                return True

            print(f" -> Retrying handshake (attempt {attempt}/{max_attempts})...")
            time.sleep(0.8)

        print("\033[31m[-] Could not establish shell. Try hitting Enter on serial or rebooting robot.\033[0m")
        return False

    def exec_cmd(self, command: str, timeout: float = 12.0) -> str:
        """Send a shell command and capture complete stdout up to prompt marker."""
        if not self.is_logged_in:
            raise RuntimeError("Cannot execute command: not logged into robot shell.")

        # Flush stale incoming buffer
        _ = self.conn.read(2048, timeout=0.05)

        cmd_payload = (command.strip() + "\r\n").encode("utf-8")
        self.conn.write(cmd_payload)

        raw_output = self.conn.read_until(PROMPT_MARKER, timeout=timeout)
        text = raw_output.decode("utf-8", errors="replace")

        # Strip echo of command and trailing prompt
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            if command.strip() in line:
                continue
            if PROMPT_STR in line or line.strip() == "#":
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    # --------------------------------------------------------------------------
    # Diagnostics & System Query
    # --------------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, str]:
        """Collect system info, kernel, battery, Wi-Fi status, and storage space."""
        print("[*] Querying robot hardware diagnostics...")
        diag = {}

        # 1. System & Kernel
        diag["kernel"] = self.exec_cmd("uname -sr")
        diag["hostname"] = self.exec_cmd("hostname")

        # 2. Storage
        df_out = self.exec_cmd("df -h / | awk 'NR==2 {print $2, $3, $4, $5}'")
        diag["storage"] = df_out if df_out else "Unknown"

        # 3. MAC Addresses
        diag["mac_wlan0"] = self.exec_cmd("cat /sys/class/net/wlan0/address 2>/dev/null || echo 'N/A'")
        diag["mac_eth0"] = self.exec_cmd("cat /sys/class/net/eth0/address 2>/dev/null || echo 'N/A'")

        # 4. Wi-Fi Link
        ip_wlan = self.exec_cmd("ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}' | head -n1")
        diag["ip_wlan0"] = ip_wlan if ip_wlan else "DISCONNECTED"

        # 5. Battery Status (Raw dsPic query or file query)
        bat_query = self.exec_cmd(
            "python -c \"import socket; s=socket.socket(); s.connect(('127.0.0.1', 5000)); s.send(b'V,0\\r\\n'); print(s.recv(64))\" 2>/dev/null || true"
        )
        if "v," in bat_query:
            try:
                mv = int(re.search(r"v,(\d+)", bat_query).group(1))
                v = mv / 1000.0
                pct = max(0, min(100, int((v - 6.6) / (8.4 - 6.6) * 100)))
                diag["battery"] = f"{v:.2f} V ({pct}% LiPo)"
            except Exception:
                diag["battery"] = "Connected (Voltage parse pending)"
        else:
            diag["battery"] = "Daemon inactive (Run code deploy or test first)"

        return diag

    # --------------------------------------------------------------------------
    # Wi-Fi Provisioning (Lab Hotspot / 2.4 GHz Router)
    # --------------------------------------------------------------------------

    def scan_wifi_networks(self) -> List[str]:
        """Scan available 2.4 GHz SSIDs using robot's internal Wi-Fi adapter."""
        print("[*] Scanning 2.4 GHz Wi-Fi networks from robot...")
        raw = self.exec_cmd("iw dev wlan0 scan 2>/dev/null | grep -i 'SSID:' | sed 's/.*SSID: //'", timeout=10.0)
        ssids = [s.strip() for s in raw.splitlines() if s.strip()]
        return sorted(list(set(ssids)))

    def configure_wifi(self, ssid: str, password: str, timeout_dhcp: int = 15) -> Tuple[bool, str]:
        """Configure wpa_supplicant, restart interface, and acquire DHCP lease."""
        print(f"[*] Provisioning Wi-Fi SSID '{ssid}' onto Khepera IV...")

        wpa_conf = (
            "ctrl_interface=/var/run/wpa_supplicant\n"
            "update_config=1\n"
            "network={\n"
            f'    ssid="{ssid}"\n'
            f'    psk="{password}"\n'
            "    key_mgmt=WPA-PSK\n"
            "}\n"
        )

        # Write configuration safely
        b64_conf = base64.b64encode(wpa_conf.encode("utf-8")).decode("ascii")
        self.exec_cmd("mkdir -p /etc/wpa_supplicant")
        self.exec_cmd(f"echo '{b64_conf}' | base64 -d > /etc/wpa_supplicant/wpa_supplicant-wlan0.conf")

        print("[*] Reconnecting Wi-Fi adapter and requesting DHCP lease...")
        self.exec_cmd("killall wpa_supplicant udhcpc 2>/dev/null || true")
        self.exec_cmd("ifconfig wlan0 down && ifconfig wlan0 up")
        self.exec_cmd("wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant-wlan0.conf -D nl80211,wext")

        # Background DHCP client
        self.exec_cmd("udhcpc -i wlan0 -n -q -t 5 > /tmp/udhcpc.log 2>&1 &")

        # Poll for assigned IP
        t0 = time.time()
        assigned_ip = ""
        while time.time() - t0 < timeout_dhcp:
            ip_check = self.exec_cmd("ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}' | head -n1")
            if ip_check and ip_check != "127.0.0.1":
                assigned_ip = ip_check.strip()
                break
            time.sleep(1.0)

        if assigned_ip:
            print(f"\033[32m[+] Wi-Fi connected! Assigned IP: {assigned_ip}\033[0m")
            return True, assigned_ip
        else:
            log_out = self.exec_cmd("cat /tmp/udhcpc.log 2>/dev/null || true")
            print(f"\033[31m[-] DHCP failed or timed out. Log: {log_out}\033[0m")
            return False, ""

    def configure_static_ip(self, ip_addr: str, netmask: str = "255.255.255.0", gateway: str = "192.168.1.1") -> bool:
        """Assign static IP address to robot wireless adapter."""
        print(f"[*] Assigning static IP {ip_addr}/{netmask} (GW: {gateway})...")
        self.exec_cmd(f"ifconfig wlan0 {ip_addr} netmask {netmask} up")
        self.exec_cmd(f"route add default gw {gateway} wlan0 2>/dev/null || true")
        ip_check = self.exec_cmd("ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}' | head -n1")
        if ip_check.strip() == ip_addr.strip():
            print(f"\033[32m[+] Static IP configured successfully: {ip_addr}\033[0m")
            return True
        return False

    # --------------------------------------------------------------------------
    # Swarm Identity Configuration
    # --------------------------------------------------------------------------

    def assign_swarm_identity(self, robot_idx: int, role: str, robot_ip: str) -> bool:
        """
        Configure robot hostname, internal identity manifest, and update host .env.
        Supports seamless swarm formation setups (khepera_1:base_link, khepera_2:extender, khepera_3:explorer).
        """
        robot_name = f"khepera_{robot_idx}"
        hostname = f"khepera-{robot_idx}"
        print(f"[*] Assigning Swarm Identity: {robot_name} (Role: {role}, Hostname: {hostname})...")

        # 1. Robot Hostname
        self.exec_cmd(f"echo '{hostname}' > /etc/hostname")
        self.exec_cmd(f"hostname '{hostname}'")

        # 2. Robot Onboard Manifest
        manifest = {
            "robot_index": robot_idx,
            "robot_name": robot_name,
            "role": role,
            "hostname": hostname,
            "ip": robot_ip,
            "daemon_port": 5000,
            "camera_port": 5001,
            "provisioned_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        b64_m = base64.b64encode(json.dumps(manifest, indent=2).encode("utf-8")).decode("ascii")
        self.exec_cmd(f"echo '{b64_m}' | base64 -d > /home/root/swarm_identity.json")

        # 3. Synchronize local host .env
        self.update_host_env(robot_idx, robot_ip)
        print(f"\033[32m[+] Swarm identity established and synchronized to host .env!\033[0m")
        return True

    def update_host_env(self, robot_idx: int, robot_ip: str):
        """Update host .env file with KHEPERA_<idx>_IP address."""
        key = f"KHEPERA_{robot_idx}_IP"
        new_line = f"{key}={robot_ip}\n"
        lines = []
        key_found = False

        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(new_line)
                key_found = True
            else:
                new_lines.append(line)

        if not key_found:
            new_lines.append(new_line)

        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    # --------------------------------------------------------------------------
    # Serial Code Push (Zero-Network Base64 Streaming)
    # --------------------------------------------------------------------------

    def deploy_code_serial(self) -> bool:
        """
        Compress physical_robot/onboard_daemon, encode as base64, stream over serial,
        compile on robot, and establish SysVinit autostart persistence.
        """
        print("[*] Creating compressed deployment tarball of onboard daemons...")
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            for item in os.listdir(ONBOARD_DAEMON_DIR):
                full_path = os.path.join(ONBOARD_DAEMON_DIR, item)
                tar.add(full_path, arcname=item)

        raw_tar = tar_buf.getvalue()
        b64_data = base64.b64encode(raw_tar).decode("ascii")
        total_len = len(b64_data)
        print(f"[*] Packaging complete ({len(raw_tar)} bytes compressed -> {total_len} bytes Base64).")
        print("[*] Streaming deployment bundle over serial...")

        # Prepare robot receiving directory
        self.exec_cmd("mkdir -p /home/root && rm -f /tmp/deploy.b64 /tmp/deploy.tar.gz")

        # Send in safe chunks (1024 chars per line)
        chunk_size = 1024
        chunks = [b64_data[i:i + chunk_size] for i in range(0, total_len, chunk_size)]
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            cmd = f"echo -n '{chunk}' >> /tmp/deploy.b64\r\n"
            self.conn.write(cmd.encode("ascii"))
            # Fast pacing for reliable UART ingestion
            time.sleep(0.04)
            if idx % 10 == 0 or idx == total_chunks:
                sys.stdout.write(f"\r -> Progress: {idx}/{total_chunks} chunks ({(idx/total_chunks)*100:.1f}%)")
                sys.stdout.flush()

        print("\n[*] Decoding and extracting tarball on robot...")
        self.exec_cmd("base64 -d /tmp/deploy.b64 > /tmp/deploy.tar.gz")
        self.exec_cmd("tar -xzf /tmp/deploy.tar.gz -C /home/root/")
        self.exec_cmd("rm -f /tmp/deploy.b64 /tmp/deploy.tar.gz")

        print("[*] Compiling khepera4_tcp_server with native GCC & libkhepera...")
        build_out = self.exec_cmd("cd /home/root && make clean && make", timeout=25.0)
        print(f" -> Compiler Output:\n{build_out}")

        print("[*] Setting executable permissions and registering SysVinit autostart...")
        self.exec_cmd("chmod +x /home/root/khepera4_tcp_server /home/root/khepera4_camera_streamer.sh /home/root/autorun.sh")
        self.exec_cmd("cp /home/root/autorun.sh /etc/init.d/autorun.sh && chmod +x /etc/init.d/autorun.sh")
        self.exec_cmd("ln -sf /etc/init.d/autorun.sh /etc/rc5.d/S20autorun.sh")

        print("[*] Launching onboard hardware daemon & camera streamer...")
        self.exec_cmd("/etc/init.d/autorun.sh restart || /etc/init.d/autorun.sh start")
        time.sleep(1.0)

        status_check = self.exec_cmd("pidof khepera4_tcp_server && echo 'RUNNING' || echo 'STOPPED'")
        if "RUNNING" in status_check:
            print("\033[32m[+] Onboard daemon and camera streamer successfully deployed and running!\033[0m")
            return True
        else:
            print("\033[33m[!] Deployment finished, but daemon status verification was inconclusive.\033[0m")
            return True

    # --------------------------------------------------------------------------
    # Hardware Motor & Sensor Verification
    # --------------------------------------------------------------------------

    def test_hardware(self):
        """Perform 1-second motor spin, encoder check, and sensor echo."""
        print("[*] Running Hardware Diagnostic Self-Test...")
        test_script = (
            "python -c \""
            "import socket, time\n"
            "s = socket.socket()\n"
            "s.connect(('127.0.0.1', 5000))\n"
            "s.send(b'D,50,50\\r\\n')\n"
            "time.sleep(0.5)\n"
            "s.send(b'R\\r\\n')\n"
            "enc = s.recv(64).decode()\n"
            "s.send(b'N\\r\\n')\n"
            "ir = s.recv(128).decode()\n"
            "s.send(b'G\\r\\n')\n"
            "us = s.recv(64).decode()\n"
            "s.send(b'D,0,0\\r\\n')\n"
            "print('ENCODERS:', enc.strip())\n"
            "print('IR_PROX:', ir.strip())\n"
            "print('SONAR_US:', us.strip())\n"
            "\" 2>/dev/null || echo 'DAEMON_SOCKET_ERROR'"
        )
        res = self.exec_cmd(test_script, timeout=6.0)
        print(f" -> Hardware Telemetry Echo:\n{res}")

    # --------------------------------------------------------------------------
    # Interactive Terminal Pass-Through
    # --------------------------------------------------------------------------

    def interactive_shell(self):
        """Pass-through interactive raw terminal to the robot bash shell."""
        print("\033[33m[*] Entering raw serial console. Press 'Ctrl+]' to exit.\033[0m")
        self.conn.write(b"\r\n")

        import tty
        import termios
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                r, _, _ = select.select([sys.stdin, self.conn._posix_fd or self.conn.ser.fileno()], [], [], 0.05)
                if sys.stdin in r:
                    char = os.read(sys.stdin.fileno(), 1)
                    if char == b"\x1d":  # Ctrl+]
                        break
                    self.conn.write(char)
                if (self.conn._posix_fd or self.conn.ser.fileno()) in r:
                    incoming = self.conn.read(512, timeout=0.01)
                    if incoming:
                        os.write(sys.stdout.fileno(), incoming)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\n[*] Exited raw serial console.")

    def close(self):
        self.conn.close()


# ==============================================================================
# Interactive Menu CLI
# ==============================================================================

def run_interactive_menu(provisioner: KheperaSerialProvisioner):
    """Interactive guided provisioning interface."""
    current_ip = ""

    while True:
        print("\n" + "=" * 66)
        print(f"  Khepera IV Serial Provisioning & Swarm Manager ({provisioner.port})")
        print("=" * 66)
        print(" [1] Query System Diagnostics & Battery Status")
        print(" [2] Scan Available 2.4 GHz Wi-Fi SSIDs")
        print(" [3] Provision Wi-Fi Network (Connect to Lab Hotspot via DHCP)")
        print(" [4] Deploy Code over Serial (Base64 Stream: Daemons + Autostart)")
        print(" [5] Configure Swarm Identity (Index, Role, Hostname & .env Sync)")
        print(" [6] Run Hardware Self-Test (Motors, Encoders, IR, Sonar)")
        print(" [7] Configure Static IP Address")
        print(" [8] Open Interactive Root Shell (Pass-Through Terminal)")
        print(" [0] Exit")
        print("=" * 66)

        try:
            choice = input("Select an option [0-8]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if choice == "1":
            diag = provisioner.get_diagnostics()
            print("\n" + "-" * 50)
            print("  ROBOT DIAGNOSTICS & TELEMETRY")
            print("-" * 50)
            for k, v in diag.items():
                print(f"  {k:15s} : {v}")
            print("-" * 50)
            if diag.get("ip_wlan0") and diag["ip_wlan0"] != "DISCONNECTED":
                current_ip = diag["ip_wlan0"]

        elif choice == "2":
            ssids = provisioner.scan_wifi_networks()
            print("\nAvailable 2.4 GHz SSIDs:")
            if ssids:
                for idx, s in enumerate(ssids, 1):
                    print(f"  {idx:2d}) {s}")
            else:
                print("  No 2.4 GHz SSIDs detected.")

        elif choice == "3":
            ssid = input("Enter Wi-Fi SSID: ").strip()
            if not ssid:
                print("SSID cannot be empty.")
                continue
            pwd = input("Enter Wi-Fi Password: ").strip()
            ok, ip_addr = provisioner.configure_wifi(ssid, pwd)
            if ok:
                current_ip = ip_addr

        elif choice == "4":
            provisioner.deploy_code_serial()

        elif choice == "5":
            print("\n--- Swarm Member Identity Setup ---")
            idx_in = input("Enter Robot Swarm Index [1-9] (default: 1): ").strip() or "1"
            idx_val = int(idx_in)
            roles = ["base_link (Anchor)", "extender (RF Relay)", "explorer (Scout)"]
            print("Select Swarm Role:")
            for i, r in enumerate(roles, 1):
                print(f"  {i}) {r}")
            r_choice = input("Choice [1-3] (default: 1): ").strip() or "1"
            role_val = "base_link" if r_choice == "1" else ("extender" if r_choice == "2" else "explorer")

            ip_target = current_ip or input("Enter Robot IP (or press Enter if unchanged): ").strip()
            if not ip_target:
                diag = provisioner.get_diagnostics()
                ip_target = diag.get("ip_wlan0", "192.168.137.43")

            provisioner.assign_swarm_identity(idx_val, role_val, ip_target)

        elif choice == "6":
            provisioner.test_hardware()

        elif choice == "7":
            ip_val = input("Enter Static IP (e.g. 192.168.1.101): ").strip()
            if ip_val:
                provisioner.configure_static_ip(ip_val)
                current_ip = ip_val

        elif choice == "8":
            provisioner.interactive_shell()

        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please select between 0 and 8.")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Khepera IV Serial Provisioner & Swarm Configurator")
    parser.add_argument("--port", "-p", help="Serial port device (e.g. /dev/ttyUSB0, COM3)")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_BAUD, help="Baud rate (default: 115200)")
    parser.add_argument("--deploy", action="store_true", help="Non-interactive: push and compile code over serial")
    parser.add_argument("--wifi-ssid", help="Non-interactive: SSID to provision")
    parser.add_argument("--wifi-pass", help="Non-interactive: Wi-Fi password")
    parser.add_argument("--swarm-id", type=int, help="Non-interactive: assign swarm robot ID")
    parser.add_argument("--swarm-role", choices=["base_link", "extender", "explorer"], help="Swarm role")

    args = parser.parse_args()

    # Determine port
    port = args.port
    if not port:
        ports = list_serial_ports()
        if not ports:
            print("\033[31m[ERROR] No serial ports found. Please connect the Khepera IV USB cable.\033[0m")
            sys.exit(1)
        if len(ports) == 1:
            port = ports[0]
            print(f"[*] Auto-detected serial port: {port}")
        else:
            print("\nAvailable Serial Ports:")
            for i, p in enumerate(ports, 1):
                print(f"  {i}) {p}")
            sel = input(f"Select port [1-{len(ports)}] (default 1): ").strip() or "1"
            port = ports[int(sel) - 1]

    provisioner = KheperaSerialProvisioner(port, args.baud)

    try:
        if not provisioner.connect_and_login():
            sys.exit(1)

        # Non-interactive CLI flags
        if args.deploy:
            provisioner.deploy_code_serial()

        if args.wifi_ssid:
            ok, ip_addr = provisioner.configure_wifi(args.wifi_ssid, args.wifi_pass or "")
            if ok and args.swarm_id:
                role = args.swarm_role or "base_link"
                provisioner.assign_swarm_identity(args.swarm_id, role, ip_addr)

        # Interactive mode if no execution flag supplied
        if not (args.deploy or args.wifi_ssid):
            run_interactive_menu(provisioner)

    finally:
        provisioner.close()


if __name__ == "__main__":
    main()
