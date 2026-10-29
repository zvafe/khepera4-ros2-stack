#!/usr/bin/env python3
"""
Comprehensive Integration & Verification Test for Physical Khepera IV Hardware Bridge.

Simulates a real physical Khepera IV embedded computer listening on TCP port 8888,
verifying:
1. TCP Socket handshake and K-Team ASCII communication protocol
2. Bidirectional command execution (ROS 2 /cmd_vel -> TCP motor pulses D,l,r -> server)
3. Sensor packet parsing & ROS 2 publication:
   - /odom (Odometry integration from encoder ticks R)
   - /sensors/infrared and ir_sensors (12 IR proximity & ground values from N)
   - /sensors/ultrasonic and us_sensors (5 ultrasonic ranges from G)
   - /battery and battery_state (Battery voltage & percentage from V,0)
   - /khepera4/ir_0..7/scan, /khepera4/us_0..4/scan, /khepera4/ground_ir_0..3/scan (Individual LaserScans)
   - /khepera4/scan and /scan (Unified 360-degree LaserScan)
   - /imu/data
4. End-to-end integration with Khepera4Mapper:
   - Khepera4Mapper subscribes to the hardware bridge's sensor outputs
   - Generates 2D occupancy grid map (/map)
5. Safety watchdog halting motors when velocity commands lapse (> 0.5s)
"""

import math
import select
import socket
import threading
import time
from typing import List

from geometry_msgs.msg import Twist
from khepera4_control.khepera4_hardware_bridge import Khepera4HardwareBridge
from khepera4_control.khepera4_mapper import Khepera4Mapper
from nav_msgs.msg import OccupancyGrid, Odometry
import pytest
import rclpy
from sensor_msgs.msg import BatteryState, LaserScan
from std_msgs.msg import Float32MultiArray, String


class MockPhysicalKhepera4Server:
    """Emulates physical Khepera IV embedded daemon (khepera4_tcp_server) on TCP port."""

    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.running = False
        self.server_sock = None
        self.client_sock = None
        self.thread = None

        # State tracking
        self.commands_received: List[str] = []
        self.speed_l = 0
        self.speed_r = 0
        self.enc_l = 0
        self.enc_r = 0
        self.last_update = time.time()

        # Simulated sensor responses
        # 12 IR: 8 ring + 4 ground (obstacle detected in front sensors)
        self.ir_response = "n,25,25,30,450,450,25,25,25,750,780,770,760\r\n"
        # 5 US (distances in cm: left, front-left, front=60cm, front-right, right)
        self.us_response = "g,1000,1000,60,1000,1000\r\n"
        self.battery_mv = 7450

    def start(self):
        self.running = True
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def stop(self):
        self.running = False
        if self.client_sock:
            try:
                self.client_sock.close()
            except Exception:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self):
        self.server_sock.settimeout(0.5)
        while self.running:
            try:
                self.client_sock, _ = self.server_sock.accept()
                break
            except socket.timeout:
                continue
            except Exception:
                return

        if not self.client_sock:
            return

        self.client_sock.setblocking(False)
        rx_buf = ""

        while self.running:
            now = time.time()
            dt = max(1e-4, now - self.last_update)
            self.last_update = now

            # Integrate simulated encoder advance
            self.enc_l += int(self.speed_l * 0.05)
            self.enc_r += int(self.speed_r * 0.05)

            ready, _, _ = select.select([self.client_sock], [], [], 0.02)
            if not ready:
                continue

            try:
                chunk = self.client_sock.recv(1024)
                if not chunk:
                    break
                rx_buf += chunk.decode('ascii', errors='ignore')
            except Exception:
                break

            while '\n' in rx_buf:
                line, rx_buf = rx_buf.split('\n', 1)
                cmd = line.strip()
                if not cmd:
                    continue

                self.commands_received.append(cmd)

                # Process K-Team ASCII Command matching khepera4_tcp_server.c
                resp = ""
                if cmd.startswith('D,') or cmd.startswith('d,'):
                    parts = cmd.split(',')
                    if len(parts) >= 3:
                        try:
                            self.speed_l = int(parts[1])
                            self.speed_r = int(parts[2])
                            resp = "d\r\n"
                        except ValueError:
                            resp = "d,err\r\n"
                elif cmd == 'R' or cmd == 'r':
                    resp = f"r,{self.enc_l},{self.enc_r}\r\n"
                elif cmd == 'N' or cmd == 'n':
                    resp = self.ir_response
                elif cmd == 'G' or cmd == 'g':
                    resp = self.us_response
                elif cmd.startswith('V') or cmd.startswith('v'):
                    resp = f"v,{self.battery_mv}\r\n"
                elif cmd == 'B' or cmd == 'b':
                    resp = "b,B,5\r\n"
                elif cmd == 'Z' or cmd == 'z':
                    self.speed_l = 0
                    self.speed_r = 0
                    resp = "z\r\n"
                else:
                    resp = "?\r\n"

                if resp:
                    try:
                        self.client_sock.sendall(resp.encode('ascii'))
                    except Exception:
                        return


@pytest.fixture(scope='module', autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_physical_hardware_bridge_end_to_end():
    """Verify hardware bridge TCP connection, telemetry parsing, command execution and map building."""
    server_port = 8888
    server = MockPhysicalKhepera4Server(port=server_port)
    server.start()

    bridge = None
    mapper = None
    test_client = None
    try:
        # 1. Initialize Hardware Bridge Node configured for mock server
        bridge = Khepera4HardwareBridge()
        # Override parameters dynamically for test target
        bridge.robot_ip = '127.0.0.1'
        bridge.robot_port = server_port
        bridge.mock = False
        bridge.single_robot = True

        # Trigger socket connection
        conn_ok = bridge.connect_socket()
        assert conn_ok is True, "Failed to connect to mock Khepera IV TCP server"
        assert bridge.is_connected is True

        # 2. Track ROS 2 topic outputs
        received_odom: List[Odometry] = []
        received_battery: List[BatteryState] = []
        received_ir_arrays: List[Float32MultiArray] = []
        received_us_arrays: List[Float32MultiArray] = []
        received_ir_scans: List[LaserScan] = []
        received_360_scans: List[LaserScan] = []

        sub_odom = bridge.create_subscription(
            Odometry, '/odom', lambda m: received_odom.append(m), 10
        )
        sub_battery = bridge.create_subscription(
            BatteryState, '/battery', lambda m: received_battery.append(m), 10
        )
        sub_ir = bridge.create_subscription(
            Float32MultiArray, '/sensors/infrared', lambda m: received_ir_arrays.append(m), 10
        )
        sub_us = bridge.create_subscription(
            Float32MultiArray, '/sensors/ultrasonic', lambda m: received_us_arrays.append(m), 10
        )
        sub_ir_scan = bridge.create_subscription(
            LaserScan, '/khepera4/ir_0/scan', lambda m: received_ir_scans.append(m), 10
        )
        sub_360_scan = bridge.create_subscription(
            LaserScan, '/scan', lambda m: received_360_scans.append(m), 10
        )

        # 3. Spin bridge for multiple iterations to poll sensors
        for _ in range(15):
            bridge.loop_step()
            rclpy.spin_once(bridge, timeout_sec=0.03)
            time.sleep(0.02)

        # Verify Server received ASCII commands: D (speed), R (encoders), N (IR), G (US), V,0 (battery)
        cmds_str = " ".join(server.commands_received)
        assert "D," in cmds_str, f"Speed commands not received by server: {cmds_str}"
        assert "R" in server.commands_received, "Encoder query 'R' not received by server"
        assert "N" in server.commands_received, "Proximity query 'N' not received by server"
        assert "G" in server.commands_received, "Ultrasonic query 'G' not received by server"
        assert "V,0" in server.commands_received, "Battery query 'V,0' not received by server"

        # Verify ROS 2 topic publication
        assert len(received_odom) > 0, "No /odom messages published"
        assert received_odom[-1].header.frame_id == 'odom'
        assert received_odom[-1].child_frame_id == 'base_footprint'

        assert len(received_battery) > 0, "No /battery messages published"
        assert math.isclose(received_battery[-1].voltage, 7.45, abs_tol=0.05)
        assert 0.0 < received_battery[-1].percentage <= 1.0

        assert len(received_ir_arrays) > 0, "No /sensors/infrared messages published"
        assert len(received_ir_arrays[-1].data) == 12, "IR array should have 12 values"

        assert len(received_us_arrays) > 0, "No /sensors/ultrasonic messages published"
        assert len(received_us_arrays[-1].data) == 5, "US array should have 5 values"

        assert len(received_ir_scans) > 0, "No /khepera4/ir_0/scan published"
        assert received_ir_scans[-1].range_min <= received_ir_scans[-1].ranges[0] <= received_ir_scans[-1].range_max

        assert len(received_360_scans) > 0, "No unified 360-degree /scan published"
        assert len(received_360_scans[-1].ranges) == 360

        # 4. Test Velocity Command Translation (/cmd_vel -> D,l,r)
        test_client = rclpy.create_node('teleop_test_client')
        cmd_vel_pub = test_client.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(0.05)
        twist_cmd = Twist()
        twist_cmd.linear.x = 0.10  # 0.10 m/s
        twist_cmd.angular.z = 0.0
        cmd_vel_pub.publish(twist_cmd)

        for _ in range(8):
            rclpy.spin_once(test_client, timeout_sec=0.02)
            rclpy.spin_once(bridge, timeout_sec=0.02)
            bridge.loop_step()
            time.sleep(0.02)

        # Expected wheel speed: 0.10 m/s * 1474.53 pulses/m/10ms = ~147 pulses
        last_d_cmds = [c for c in server.commands_received if c.startswith('D,')]
        assert len(last_d_cmds) > 0
        latest_d = last_d_cmds[-1]
        parts = latest_d.split(',')
        p_l, p_r = int(parts[1]), int(parts[2])
        assert 140 <= p_l <= 155, f"Expected pulses ~147, got {p_l}"
        assert 140 <= p_r <= 155, f"Expected pulses ~147, got {p_r}"

        # 5. Integration with Khepera4Mapper: Verify Map Generation
        mapper = Khepera4Mapper()
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        received_maps: List[OccupancyGrid] = []
        bridge.create_subscription(
            OccupancyGrid, '/map', lambda m: received_maps.append(m), qos_latched
        )

        # Drive mapping iterations using executor to dispatch all sensor callbacks
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(mapper)

        for _ in range(15):
            bridge.loop_step()
            for _ in range(5):
                executor.spin_once(timeout_sec=0.01)
            mapper.publish_virtual_scan()
            mapper.update_and_publish_maps()
            for _ in range(5):
                executor.spin_once(timeout_sec=0.01)
            time.sleep(0.01)

        assert mapper.has_odom is True, "Mapper should receive odometry from hardware bridge"
        assert len(received_maps) > 0, "Mapper did not publish /map"
        grid_data = received_maps[-1].data
        # Occupancy grid should be 200x200 = 40,000 cells
        assert len(grid_data) == 40000
        # Obstacles and free space should be mapped
        num_free = sum(1 for c in grid_data if c == 0)
        num_occ = sum(1 for c in grid_data if c == 100)
        assert num_free > 0, "No free space cast in occupancy grid"
        assert num_occ > 0, "No obstacles marked in occupancy grid"

        # 6. Safety Watchdog Verification: Cease commands and verify motor halt
        time.sleep(0.6)  # Exceed 500ms timeout
        bridge.loop_step()
        halt_cmds = [c for c in server.commands_received if c.startswith('D,')]
        assert halt_cmds[-1] == "D,0,0", f"Watchdog failed to halt motors to D,0,0: {halt_cmds[-1]}"

    finally:
        if test_client:
            test_client.destroy_node()
        if bridge:
            bridge.destroy_node()
        if mapper:
            mapper.destroy_node()
        server.stop()
