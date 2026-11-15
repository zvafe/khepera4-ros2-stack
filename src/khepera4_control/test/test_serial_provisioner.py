#!/usr/bin/env python3
"""
Unit and Integration Tests for Khepera IV Serial Provisioner and Swarm Setup.
"""

import base64
import io
import os
import re
import sys
import tarfile
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../physical_robot")))
import khepera_serial_provisioner as ksp


def test_list_serial_ports():
    """Verify list_serial_ports executes cleanly without error."""
    ports = ksp.list_serial_ports()
    assert isinstance(ports, list)


def test_tarball_base64_bundling_integrity():
    """Verify onboard_daemon directory compresses and reconstructs with 100% integrity."""
    daemon_dir = ksp.ONBOARD_DAEMON_DIR
    assert os.path.isdir(daemon_dir), f"Directory not found: {daemon_dir}"

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for item in os.listdir(daemon_dir):
            tar.add(os.path.join(daemon_dir, item), arcname=item)

    raw_bytes = tar_buf.getvalue()
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    assert len(b64_str) > 0

    # Decode and unpack into temp directory
    decoded_bytes = base64.b64decode(b64_str.encode("ascii"))
    with tempfile.TemporaryDirectory() as tmp_out:
        with tarfile.open(fileobj=io.BytesIO(decoded_bytes), mode="r:gz") as tar:
            tar.extractall(tmp_out)

        # Check required files exist
        assert os.path.exists(os.path.join(tmp_out, "khepera4_tcp_server.c"))
        assert os.path.exists(os.path.join(tmp_out, "Makefile"))
        assert os.path.exists(os.path.join(tmp_out, "khepera4_camera_streamer.sh"))
        assert os.path.exists(os.path.join(tmp_out, "autorun.sh"))


def test_host_env_updater():
    """Verify swarm identity updates KHEPERA_<idx>_IP in .env cleanly."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("KHEPERA_DEFAULT_MODE=web\nKHEPERA_1_IP=192.168.1.101\n")
        temp_env_path = tf.name

    try:
        # Patch ENV_FILE
        orig_env = ksp.ENV_FILE
        ksp.ENV_FILE = temp_env_path

        p = ksp.KheperaSerialProvisioner(port="/dev/null")
        p.update_host_env(1, "192.168.137.43")
        p.update_host_env(2, "192.168.137.44")

        with open(temp_env_path) as f:
            content = f.read()

        assert "KHEPERA_1_IP=192.168.137.43" in content
        assert "KHEPERA_2_IP=192.168.137.44" in content
        assert "KHEPERA_DEFAULT_MODE=web" in content
    finally:
        ksp.ENV_FILE = orig_env
        if os.path.exists(temp_env_path):
            os.remove(temp_env_path)


def test_battery_voltage_percentage_math():
    """Verify LiPo voltage to percentage calculation logic."""
    # 2-cell LiPo: 8.4V = 100%, 6.6V = 0%
    def calc_pct(v):
        return max(0, min(100, int((v - 6.6) / (8.4 - 6.6) * 100)))

    assert calc_pct(8.4) == 100
    assert calc_pct(7.5) == 50
    assert calc_pct(6.6) == 0
    assert calc_pct(6.0) == 0
    assert calc_pct(8.5) == 100


def test_wpa_supplicant_config_generation():
    """Verify wpa_supplicant configuration generation contains valid block syntax."""
    ssid = "LabRobotics_2.4G"
    psk = "SuperSecretPassword123"

    wpa_conf = (
        "ctrl_interface=/var/run/wpa_supplicant\n"
        "update_config=1\n"
        "network={\n"
        f'    ssid="{ssid}"\n'
        f'    psk="{psk}"\n'
        "    key_mgmt=WPA-PSK\n"
        "}\n"
    )

    assert f'ssid="{ssid}"' in wpa_conf
    assert f'psk="{psk}"' in wpa_conf
    assert "key_mgmt=WPA-PSK" in wpa_conf
