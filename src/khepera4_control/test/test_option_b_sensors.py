# Copyright 2026 Mobile Robotics Laboratory.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Option B sensor integration: beam dispersion, NaN blind spots, cliff cloud, TF."""

import math

from geometry_msgs.msg import Quaternion
from khepera4_control.khepera4_mapper import Khepera4Mapper
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import LaserScan
from sensor_msgs_py import point_cloud2


@pytest.fixture(scope="module", autouse=True)
def init_rclpy():
    """Manage lifecycle of rclpy for module-scoped testing."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_virtual_scan_blind_spots_and_beam_dispersion():
    """Verify virtual scan uses 0.0 (< range_min) for unobserved angles and fans out US cones."""
    node = Khepera4Mapper()

    published_scans = []
    node.scan_pub.publish = lambda msg: published_scans.append(msg)

    # Provide a single US reading at index 2 (front sensor, 0 deg) at 1.0 meter
    us_scan = LaserScan()
    us_scan.header.stamp = node.get_clock().now().to_msg()
    us_scan.header.frame_id = 'khepera4/us_2'
    us_scan.angle_min = 0.0
    us_scan.angle_max = 0.0
    us_scan.angle_increment = 0.0
    us_scan.range_min = 0.20
    us_scan.range_max = 2.0
    us_scan.ranges = [1.0]
    node.us_callback(2, us_scan)

    node.publish_virtual_scan()

    assert len(published_scans) == 1
    scan = published_scans[0]
    assert len(scan.ranges) == 360
    assert scan.range_min == 0.05
    assert scan.range_max == 3.5

    # The rear hemisphere (indices around 0 and 359, which correspond to -pi and +pi)
    # MUST be 0.0 (< range_min) so Nav2 and SLAM Toolbox discard them without NaN pointer errors.
    assert scan.ranges[0] == 0.0, "Rear blind spot angle must be 0.0 (< range_min)"
    assert scan.ranges[359] == 0.0, "Rear blind spot angle must be 0.0 (< range_min)"

    # Front sensor (us_2 at 0 rad) is near index 180 (-pi + 180 * (2*pi/360) = 0)
    # Dispersion cone (+/- 10 deg) covers multiple indices around 180
    valid_front_ranges = [r for r in scan.ranges[175:186] if r > 0.0]
    assert len(valid_front_ranges) >= 5, "US cone dispersion must produce multiple rays"
    for r in valid_front_ranges:
        assert 0.95 <= r <= 1.15, f"Rays must reflect physical obstacle distance: {r}"

    node.destroy_node()


def test_cliff_pointcloud_publishing():
    """Verify that ground IR drop-off detection triggers /khepera4/cliff_cloud."""
    node = Khepera4Mapper()

    cliff_clouds = []
    node.cliff_cloud_pub.publish = lambda msg: cliff_clouds.append(msg)

    # Trigger cliff on ground_ir_0 (front-left outer)
    cliff_scan = LaserScan()
    cliff_scan.header.stamp = node.get_clock().now().to_msg()
    cliff_scan.range_min = 0.001
    cliff_scan.range_max = 0.05
    cliff_scan.ranges = [0.035]  # > cliff_threshold (0.025m)

    node.ground_ir_callback(0, cliff_scan)

    assert node.cliff_detected is True
    assert len(cliff_clouds) == 1
    cloud = cliff_clouds[0]
    assert cloud.header.frame_id == 'base_footprint'

    # Read points from pointcloud
    points = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))
    assert len(points) == 1
    pt = points[0]
    assert abs(pt[0] - 0.0610) < 1e-3  # ground_ir_0 x offset
    assert abs(pt[1] - 0.0200) < 1e-3  # ground_ir_0 y offset

    node.destroy_node()


def test_publish_odom_tf():
    """Verify publish_odom_tf broadcasts odom -> base_footprint TF with IMU yaw."""
    node = Khepera4Mapper(parameter_overrides=[
        rclpy.parameter.Parameter('publish_odom_tf', value=True),
        rclpy.parameter.Parameter('publish_map_tf', value=False)
    ])
    assert node.publish_odom_tf is True

    # Mock an incoming odometry message
    odom_msg = Odometry()
    odom_msg.header.stamp = node.get_clock().now().to_msg()
    odom_msg.pose.pose.position.x = 0.5
    odom_msg.pose.pose.position.y = -0.3
    odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    # Also simulate IMU heading of 0.25 rad
    node.has_imu = True
    node.imu_valid = True
    node.imu_yaw = 0.25

    captured_tfs = []
    node.tf_broadcaster.sendTransform = lambda tf: captured_tfs.append(tf)

    node.odom_callback(odom_msg)

    assert len(captured_tfs) >= 1
    odom_tf = captured_tfs[0]
    assert odom_tf.header.frame_id == 'odom'
    assert odom_tf.child_frame_id == 'base_footprint'
    assert abs(odom_tf.transform.translation.x - 0.5) < 1e-3
    assert abs(odom_tf.transform.translation.y - (-0.3)) < 1e-3

    # Orientation in TF must reflect IMU yaw (0.25 rad -> z = sin(0.125), w = cos(0.125))
    expected_qz = math.sin(0.125)
    assert abs(odom_tf.transform.rotation.z - expected_qz) < 1e-3

    node.destroy_node()


def test_cliff_thin_wall_stamping_upright():
    """Verify ground IR cliff stamps a 1-cell thin barrier tangent to edge when upright."""
    node = Khepera4Mapper()
    node.has_odom = True
    node.robot_x = 0.0
    node.robot_y = 0.0
    node.robot_yaw = 0.0
    node.robot_pitch = 0.0  # Upright

    cliff_alerts = []
    published_maps = []
    node.cliff_alert_pub.publish = lambda msg: cliff_alerts.append(msg)
    node.map_pub.publish = lambda msg: published_maps.append(msg)

    # Trigger cliff on ground_ir_1 (front-left inner)
    cliff_scan = LaserScan()
    cliff_scan.header.stamp = node.get_clock().now().to_msg()
    cliff_scan.range_min = 0.001
    cliff_scan.range_max = 0.15
    cliff_scan.ranges = [0.08]  # > cliff_threshold

    node.ground_ir_callback(1, cliff_scan)

    assert node.cliff_detected is True
    assert len(cliff_alerts) == 1
    assert cliff_alerts[0].data is True

    # Check stamped cells in surface_grid
    stamped_cells = [(r, c) for r in range(node.height) for c in range(node.width)
                     if node.surface_grid[r, c] == 100]
    assert len(stamped_cells) > 0, "Thin edge wall must be stamped in surface_grid"
    for r, c in stamped_cells:
        assert node.log_odds[r, c] == node.l_max
        assert node.hit_count[r, c] == 255

    # Check occupancy grid data converts confirmed cliffs to 100 (lethal)
    grid_data = node._build_occupancy_grid_data()
    for r, c in stamped_cells:
        idx = r * node.width + c
        assert grid_data[idx] == 100

    assert len(published_maps) == 1
    node.destroy_node()


def test_cliff_no_stamping_when_pitched_or_fallen():
    """Verify ground IR does NOT stamp barrier when robot has pitched/fallen off deck."""
    node = Khepera4Mapper()
    node.has_odom = True
    node.robot_x = 0.0
    node.robot_y = 0.0
    node.robot_yaw = 0.0
    node.robot_pitch = -0.25  # Pitched down (> 0.12 rad threshold)

    cliff_alerts = []
    node.cliff_alert_pub.publish = lambda msg: cliff_alerts.append(msg)

    cliff_scan = LaserScan()
    cliff_scan.header.stamp = node.get_clock().now().to_msg()
    cliff_scan.range_min = 0.001
    cliff_scan.range_max = 0.15
    cliff_scan.ranges = [0.08]

    node.ground_ir_callback(1, cliff_scan)

    # Alert still triggers
    assert node.cliff_detected is True
    assert len(cliff_alerts) == 1
    assert cliff_alerts[0].data is True

    # But NO cells stamped in surface_grid (protects lower arena floor)
    stamped_cells = [(r, c) for r in range(node.height) for c in range(node.width)
                     if node.surface_grid[r, c] == 100]
    assert len(stamped_cells) == 0, "Must not stamp cells on floor when robot is pitched down"

    node.destroy_node()
