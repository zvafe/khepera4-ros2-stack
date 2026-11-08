"""Unit tests for SplatArray serialization in GaussianMapperNode."""

import rclpy
from std_msgs.msg import Header

from activesplat_ros.gaussian_mapper_node import GaussianMapperNode


def test_build_splat_array_serialization() -> None:
    """Verify that _build_splat_array constructs conforming gsplat_msgs."""
    if not rclpy.ok():
        rclpy.init()

    node = GaussianMapperNode()
    try:
        header = Header()
        header.frame_id = "odom"

        # Points: [x, y, z, r_norm, g_norm, b_norm, rgb_packed]
        sample_points = [
            [1.0, 2.0, 0.5, 0.9, 0.2, 0.1, 0.0],
            [1.05, 2.05, 0.52, 0.85, 0.22, 0.12, 0.0],
        ]

        splat_array = node._build_splat_array(header, sample_points)
        assert len(splat_array.splats) == 2

        splat_0 = splat_array.splats[0]
        assert splat_0.pose.header.frame_id == "odom"
        assert abs(splat_0.pose.pose.pose.position.x - 1.0) < 1e-4
        assert abs(splat_0.pose.pose.pose.position.y - 2.0) < 1e-4
        assert abs(splat_0.pose.pose.pose.position.z - 0.5) < 1e-4
        assert splat_0.opacity == 255
        assert splat_0.harmonics_deg == 0
        assert len(splat_0.spherical_harmonics) == 3
        # Top-left 3x3 diagonal covariance must be positive
        assert splat_0.pose.pose.covariance[0] > 0.0
        assert splat_0.pose.pose.covariance[7] > 0.0
        assert splat_0.pose.pose.covariance[14] > 0.0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
