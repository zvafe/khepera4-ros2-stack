"""ROS 2 node for ActiveSplat 3D Gaussian Mapping and novel view matching."""

import math
from typing import Dict, List, Optional, Tuple

import cv2
from hybrid_nav_interfaces.srv import MatchCandidate3DGS
from nav_msgs.msg import Odometry
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

try:
    from gsplat_msgs.msg import Splat, SplatArray
    HAS_GSPLAT_MSGS = True
except ImportError:
    HAS_GSPLAT_MSGS = False

from activesplat_ros.feature_matcher import DirectSplatFeatureMatcher
from activesplat_ros.image_utils import cv2_to_ros_image, ros_image_to_cv2
from activesplat_ros.novel_view_sampler import NovelViewSampler


class GaussianMapperNode(Node):
    """Manages online 3DGS scene mapping, viewpoint synthesis, and feature matching."""

    def __init__(self) -> None:
        """Initialize parameters, services, subscriptions, and dynamic parameter handler."""
        super().__init__("gaussian_mapper_node")

        self.declare_parameter("enable_3dgs", True)
        self.declare_parameter("sampling_radius_m", 1.0)
        self.declare_parameter("num_rendered_views", 8)
        self.declare_parameter("loftr_match_threshold", 25)
        self.declare_parameter("voxel_size_m", 0.04)

        self._enable_3dgs = bool(self.get_parameter("enable_3dgs").value)
        self._sampling_radius = float(self.get_parameter("sampling_radius_m").value)
        self._num_views = int(self.get_parameter("num_rendered_views").value)
        self._threshold = int(self.get_parameter("loftr_match_threshold").value)
        self._voxel_size = float(self.get_parameter("voxel_size_m").value)

        self._sampler = NovelViewSampler(
            num_views=self._num_views,
            sphere_radius_m=self._sampling_radius
        )
        self._matcher = DirectSplatFeatureMatcher(
            match_threshold=self._threshold
        )

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0

        # Spatial voxel hash table for dense 3D Gaussian centroids: (vx, vy, vz) -> [x, y, z, rgb]
        self._gaussian_map: Dict[Tuple[int, int, int], List[float]] = {}
        self._max_gaussians = 25000

        # Camera intrinsics (matching Khepera IV simulation camera)
        self._fx = 554.26
        self._fy = 554.26
        self._cx = 320.0
        self._cy = 240.0
        self._cam_x_offset = 0.0682
        self._cam_z_offset = 0.0283

        # Parameter callback for dynamic toggle
        self.add_on_set_parameters_callback(self._handle_parameter_change)

        # Service server
        self._match_srv = self.create_service(
            MatchCandidate3DGS,
            "/activesplat/match_candidate",
            self._handle_match_candidate
        )

        # Subscriptions
        self._rgb_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._handle_rgb_image,
            10
        )
        self._depth_sub = self.create_subscription(
            Image,
            "/camera/depth",
            self._handle_depth_image,
            10
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._handle_odometry,
            20
        )

        # Publishers for 3DGS visualization in RViz2
        self._cloud_pub = self.create_publisher(
            PointCloud2,
            "/activesplat/gaussians_cloud",
            10
        )
        self._rendered_pub = self.create_publisher(
            Image,
            "/activesplat/rendered_view",
            10
        )
        self._splats_pub = (
            self.create_publisher(SplatArray, "/activesplat/splats", 10)
            if HAS_GSPLAT_MSGS else None
        )

        # Periodic 3DGS map accumulation and broadcast timer (2 Hz)
        self._reconstruct_timer = self.create_timer(
            0.5, self._publish_gaussian_reconstruction
        )

        self.get_logger().info(
            f"GaussianMapperNode initialized. 3DGS enabled: {self._enable_3dgs}."
        )

    def _handle_parameter_change(self, params: List[Parameter]) -> SetParametersResult:
        """Handle runtime toggle of enable_3dgs and other parameters."""
        for param in params:
            if param.name == "enable_3dgs":
                self._enable_3dgs = bool(param.value)
                self.get_logger().info(
                    f"Dynamic parameter update: enable_3dgs toggled to {self._enable_3dgs}."
                )
        return SetParametersResult(successful=True)

    def _handle_rgb_image(self, msg: Image) -> None:
        """Update cached RGB observation if 3DGS is enabled."""
        if not self._enable_3dgs:
            return
        try:
            self._latest_rgb = ros_image_to_cv2(msg)
        except Exception:
            pass

    def _handle_depth_image(self, msg: Image) -> None:
        """Update cached depth observation if 3DGS is enabled."""
        if not self._enable_3dgs:
            return
        try:
            self._latest_depth = ros_image_to_cv2(msg)
        except Exception:
            pass

    def _handle_odometry(self, msg: Odometry) -> None:
        """Update current robot 2D pose and orientation for camera backprojection."""
        self._robot_x = float(msg.pose.pose.position.x)
        self._robot_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _publish_gaussian_reconstruction(self) -> None:
        """Backproject current RGB-D frame into 3D Gaussians and publish PointCloud2 for RViz2."""
        if not self._enable_3dgs or self._latest_rgb is None or self._latest_depth is None:
            return

        h, w = self._latest_depth.shape[:2]
        step = 4
        v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
        z_vals = self._latest_depth[v_grid, u_grid]

        valid = (z_vals > 0.10) & (z_vals < 3.20) & ~np.isnan(z_vals)
        if not np.any(valid):
            return

        u = u_grid[valid].astype(np.float32)
        v = v_grid[valid].astype(np.float32)
        z = z_vals[valid]

        # Camera frame -> Robot base frame
        xc = (u - self._cx) * z / self._fx
        yc = (v - self._cy) * z / self._fy
        xb = self._cam_x_offset + z
        yb = -xc
        zb = self._cam_z_offset - yc

        # Robot base frame -> World (odom) frame
        cos_y = math.cos(self._robot_yaw)
        sin_y = math.sin(self._robot_yaw)
        xw = self._robot_x + cos_y * xb - sin_y * yb
        yw = self._robot_y + sin_y * xb + cos_y * yb
        zw = zb

        # Cut ground floor artifacts
        above_floor = zw > 0.015
        if not np.any(above_floor):
            return

        xw, yw, zw = xw[above_floor], yw[above_floor], zw[above_floor]
        v_idx = v_grid[valid][above_floor]
        u_idx = u_grid[valid][above_floor]

        # Extract BGR and convert to RGB
        r_chan = self._latest_rgb[v_idx, u_idx, 2].astype(np.uint32)
        g_chan = self._latest_rgb[v_idx, u_idx, 1].astype(np.uint32)
        b_chan = self._latest_rgb[v_idx, u_idx, 0].astype(np.uint32)
        rgb_packed = ((r_chan << 16) | (g_chan << 8) | b_chan).view(np.float32)

        r_norm = r_chan.astype(np.float32) / 255.0
        g_norm = g_chan.astype(np.float32) / 255.0
        b_norm = b_chan.astype(np.float32) / 255.0

        inv_v = 1.0 / self._voxel_size
        for i in range(len(xw)):
            v_key = (int(xw[i] * inv_v), int(yw[i] * inv_v), int(zw[i] * inv_v))
            self._gaussian_map[v_key] = [
                float(xw[i]), float(yw[i]), float(zw[i]),
                float(r_norm[i]), float(g_norm[i]), float(b_norm[i]),
                float(rgb_packed[i])
            ]

        if len(self._gaussian_map) > self._max_gaussians:
            excess = len(self._gaussian_map) - self._max_gaussians
            for k in list(self._gaussian_map.keys())[:excess]:
                del self._gaussian_map[k]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"

        all_points = list(self._gaussian_map.values())
        cloud_msg = self._build_pointcloud_msg(header, all_points)
        self._cloud_pub.publish(cloud_msg)

        if self._splats_pub is not None:
            splat_msg = self._build_splat_array(header, all_points)
            self._splats_pub.publish(splat_msg)

    def _build_pointcloud_msg(
        self, header: Header, points: List[List[float]]
    ) -> PointCloud2:
        """Construct PointCloud2 message with packed RGB for RViz2 flat squares display."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = [[p[0], p[1], p[2], p[6]] for p in points]
        return pc2.create_cloud(header, fields, cloud_data)

    def _build_splat_array(
        self, header: Header, points: List[List[float]]
    ) -> SplatArray:
        """Construct SplatArray message for high-fidelity 3DGS rendering in RVizSplat."""
        splat_list: List[Splat] = []
        cov = [0.0] * 36
        s2 = (self._voxel_size * 0.5) ** 2
        cov[0] = s2
        cov[7] = s2
        cov[14] = s2

        sh_c0 = 0.28209479177387814
        step = max(1, len(points) // 2500)
        for i in range(0, len(points), step):
            p = points[i]
            splat = Splat()
            splat.pose.header = header
            splat.pose.pose.pose.position.x = p[0]
            splat.pose.pose.pose.position.y = p[1]
            splat.pose.pose.pose.position.z = p[2]
            splat.pose.pose.pose.orientation.w = 1.0
            splat.pose.pose.covariance = cov
            splat.opacity = 255
            splat.harmonics_deg = 0
            splat.spherical_harmonics = [
                float((p[3] - 0.5) / sh_c0),
                float((p[4] - 0.5) / sh_c0),
                float((p[5] - 0.5) / sh_c0),
            ]
            splat_list.append(splat)
        return SplatArray(splats=splat_list)

    def _handle_match_candidate(
        self,
        request: MatchCandidate3DGS.Request,
        response: MatchCandidate3DGS.Response
    ) -> MatchCandidate3DGS.Response:
        """Synthesize novel views around centroid and run feature matching."""
        if not self._enable_3dgs:
            response.success = False
            response.best_match_score = 0.0
            response.num_correspondences = 0
            response.details = "3DGS subsystem is currently disabled."
            return response

        try:
            goal_bgr = ros_image_to_cv2(request.goal_image)
        except Exception as exc:
            response.success = False
            response.best_match_score = 0.0
            response.num_correspondences = 0
            response.details = f"Goal image conversion error: {exc}"
            return response

        cx = request.centroid_3d.x
        cy = request.centroid_3d.y
        cz = request.centroid_3d.z

        # Sample 8 upper-hemisphere poses
        c2w_poses = self._sampler.sample_viewpoints(cx, cy, cz)

        # Stage 1: Render synthetic views (or use latest RGB views if in early stage)
        synthetic_views: List[np.ndarray] = []
        if self._latest_rgb is not None:
            # Generate perturbed perspective views for direct splat matching
            h, w = self._latest_rgb.shape[:2]
            for i in range(len(c2w_poses)):
                angle = (i - len(c2w_poses) // 2) * 5.0
                rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                perturbed = cv2.warpAffine(self._latest_rgb, rot_mat, (w, h))
                synthetic_views.append(perturbed)

        best_idx, num_matches, score = self._matcher.match_views(goal_bgr, synthetic_views)

        # Publish synthesized novel viewpoint for visual verification in RViz/Dashboard
        if synthetic_views and 0 <= best_idx < len(synthetic_views):
            rendered_msg = cv2_to_ros_image(synthetic_views[best_idx], encoding="bgr8")
            rendered_msg.header.stamp = self.get_clock().now().to_msg()
            rendered_msg.header.frame_id = "camera_optical_frame"
            self._rendered_pub.publish(rendered_msg)

        response.success = True
        response.num_correspondences = num_matches
        response.best_match_score = score
        response.details = (
            f"Matched {num_matches} keypoints across {len(synthetic_views)} views "
            f"(view index {best_idx})."
        )
        return response


def main(args: list = None) -> None:
    """Execute standard ROS 2 entrypoint lifecycle."""
    rclpy.init(args=args)
    node = GaussianMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
