"""Coordinate lifting, SE(2) compounding, and point cloud seeding for Khepera IV."""

import math
from typing import Dict, List, Tuple

import numpy as np


class CoordinateLifter:
    """
    Transforms 2D SLAM poses and monocular depth maps into georeferenced 3D point clouds.

    Handles SE(2) -> SE(3) compounding, ROS optical -> OpenGL conversion, and voxel fusion.
    """

    def __init__(
        self,
        camera_x_offset_m: float = 0.0682,
        camera_z_offset_m: float = 0.0283,
        camera_pitch_rad: float = 0.0,
        fx: float = 554.256258,
        fy: float = 554.256258,
        cx: float = 320.0,
        cy: float = 240.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """
        Initialize lifter with robot extrinsics and camera intrinsics.

        :param camera_x_offset_m: Forward camera displacement from base_footprint (0.0682 m).
        :param camera_z_offset_m: Vertical ground clearance height h_floor (0.0283 m).
        :param camera_pitch_rad: Downward camera pitch angle in radians (0.0 rad).
        :param fx: Horizontal focal length in pixels.
        :param fy: Vertical focal length in pixels.
        :param cx: Optical center x-coordinate in pixels.
        :param cy: Optical center y-coordinate in pixels.
        :param width: Image width in pixels.
        :param height: Image height in pixels.
        """
        self.xc = float(camera_x_offset_m)
        self.zc = float(camera_z_offset_m)
        self.pitch = float(camera_pitch_rad)

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

        # 3x3 Intrinsic matrix K
        self.k_matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # Static transform from base_footprint to camera_optical_frame
        # P_base = R_base_cam_ros * P_cam_opt + T_base_cam
        # Optical +X (Right)   -> Base -Y: column 0 is [ 0, -1,  0]^T
        # Optical +Y (Down)    -> Base -Z: column 1 is [ 0,  0, -1]^T
        # Optical +Z (Forward) -> Base +X: column 2 is [ 1,  0,  0]^T
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        pitch_base = np.array(
            [[cos_p, 0.0, -sin_p], [0.0, 1.0, 0.0], [sin_p, 0.0, cos_p]],
            dtype=np.float32,
        )

        r_base_to_opt = pitch_base @ np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )

        self.t_base_cam_ros = np.eye(4, dtype=np.float32)
        self.t_base_cam_ros[:3, :3] = r_base_to_opt
        self.t_base_cam_ros[:3, 3] = np.array([self.xc, 0.0, self.zc], dtype=np.float32)

        # Conversion matrix from ROS optical convention to OpenGL convention:
        # ROS optical: +X right, +Y down, +Z forward
        # OpenGL:      +X right, +Y up,   +Z back
        self.opt_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    def compute_pose_matrices(
        self, x: float, y: float, yaw: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SE(3) pose matrices in ROS optical and OpenGL camera conventions.

        :param x: Robot planar position x in map frame.
        :param y: Robot planar position y in map frame.
        :param yaw: Robot heading orientation in radians.
        :return: (T_world_cam_ros, T_cam_opengl, viewmatrix_opengl)
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # SE(2) -> SE(3) base pose in map frame
        t_map_base = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0, x],
                [sin_yaw, cos_yaw, 0.0, y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Compounding: T_world_cam_ros = T_map_base * T_base_cam
        t_world_cam_ros = t_map_base @ self.t_base_cam_ros

        # Conversion to OpenGL convention: T_cam_opengl = T_world_cam_ros * diag(1, -1, -1, 1)
        t_cam_opengl = t_world_cam_ros @ self.opt_to_opengl

        # Viewmatrix is world-to-camera: V = inv(T_cam_opengl)
        viewmatrix_opengl = np.linalg.inv(t_cam_opengl).astype(np.float32)

        return t_world_cam_ros, t_cam_opengl, viewmatrix_opengl

    @staticmethod
    def get_gsplat_viewmatrix(t_world_cam_ros: np.ndarray) -> np.ndarray:
        """
        Compute gsplat rasterization world-to-camera matrix.

        gsplat uses OpenCV / ROS optical camera convention (+X right, +Y down, +Z forward).
        :param t_world_cam_ros: (4, 4) camera-to-world transform in ROS convention.
        :return: (4, 4) world-to-camera matrix for gsplat rasterization.
        """
        return np.linalg.inv(t_world_cam_ros).astype(np.float32)

    def clamp_ground_plane(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Enforce analytical ground plane constraint on downward camera rays.

        For pixels pointing towards the floor, distance is clamped to prevent
        rays from penetrating beneath the ground plane (Z >= 0).
        :param depth_map: (H, W) metric depth map in meters.
        :return: (H, W) ground-plane clamped depth map.
        """
        h, w = depth_map.shape[:2]
        v_grid = np.arange(h, dtype=np.float32)[:, None]
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        denom = cos_p * (v_grid - self.cy) / self.fy + sin_p
        floor_mask = denom > 1e-5

        z_floor_max = np.full((h, 1), np.inf, dtype=np.float32)
        z_floor_max[floor_mask] = self.zc / denom[floor_mask]

        clamped = depth_map.copy()
        clamped[floor_mask[:, 0], :] = np.minimum(
            depth_map[floor_mask[:, 0], :],
            z_floor_max[floor_mask[:, 0], :],
        )
        return clamped

    def unproject_depth_map(
        self,
        depth_map: np.ndarray,
        image_rgb: np.ndarray,
        t_world_cam_ros: np.ndarray,
        d_min: float = 0.05,
        d_max: float = 3.50,
        subsample_step: int = 1,
        min_z: float = -0.02,
        max_z: float = 3.00,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project calibrated depth pixels to 3D world coordinates with bounds filtering.

        :param depth_map: (H, W) float32 calibrated metric depth map in meters.
        :param image_rgb: (H, W, 3) uint8 RGB image.
        :param t_world_cam_ros: (4, 4) camera-to-world transform in ROS convention.
        :param d_min: Minimum valid depth in meters.
        :param d_max: Maximum valid depth in meters.
        :param subsample_step: Pixel subsampling stride.
        :param min_z: Minimum allowable world Z coordinate (subterranean filter).
        :param max_z: Maximum allowable world Z coordinate (sky/ceiling filter).
        :return: (points_world_Nx3, colors_rgb_Nx3)
        """
        clamped_depth = self.clamp_ground_plane(depth_map)
        h, w = clamped_depth.shape[:2]
        v_grid, u_grid = np.mgrid[0:h:subsample_step, 0:w:subsample_step]
        z_vals = clamped_depth[v_grid, u_grid]

        valid = (z_vals >= d_min) & (z_vals <= d_max) & np.isfinite(z_vals)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        u = u_grid[valid].astype(np.float32)
        v = v_grid[valid].astype(np.float32)
        z = z_vals[valid].astype(np.float32)

        # Camera optical coordinates (+X right, +Y down, +Z forward)
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        z_cam = z
        pts_cam = np.vstack((x_cam, y_cam, z_cam))  # (3, N)

        # Rigid transformation to map frame
        r_mat = t_world_cam_ros[:3, :3]
        t_vec = t_world_cam_ros[:3, 3:4]
        pts_world = (r_mat @ pts_cam + t_vec).T.astype(np.float32)

        # Discard subterranean points and open sky / ceiling floaters
        valid_z = (pts_world[:, 2] >= min_z) & (pts_world[:, 2] <= max_z)
        if not np.any(valid_z):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        pts_world = pts_world[valid_z]
        v_sub = v_grid[valid][valid_z]
        u_sub = u_grid[valid][valid_z]
        colors = (image_rgb[v_sub, u_sub].astype(np.float32) / 255.0)

        return pts_world, colors

    @staticmethod
    def voxel_downsample(
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.015,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse points and colors into a uniform spatial voxel grid.

        :param points: (N, 3) float32 3D world coordinates.
        :param colors: (N, 3) float32 normalized RGB colors.
        :param voxel_size: Grid cell dimension in meters (default 0.015 m = 1.5 cm).
        :return: (fused_points, fused_colors)
        """
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        voxel_indices = np.floor(points / voxel_size).astype(np.int64)

        # Spatial hash map grouping points by voxel index
        voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}
        for idx in range(len(points)):
            key = (
                int(voxel_indices[idx, 0]),
                int(voxel_indices[idx, 1]),
                int(voxel_indices[idx, 2]),
            )
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(idx)

        fused_pts = []
        fused_cols = []
        for key, indices in voxel_dict.items():
            fused_pts.append(np.mean(points[indices], axis=0))
            fused_cols.append(np.mean(colors[indices], axis=0))

        return (
            np.asarray(fused_pts, dtype=np.float32),
            np.asarray(fused_cols, dtype=np.float32),
        )
