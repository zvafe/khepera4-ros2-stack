"""Coordinate lifting, SE(2) compounding, and point cloud seeding for Khepera IV."""

import math
from typing import Dict, Optional, Tuple

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

        # Precomputed analytical floor depth ceiling: z_floor_max = zc / denom
        v_coords = np.arange(self.height, dtype=np.float32)[:, None]
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        denom = cos_p * (v_coords - self.cy) / self.fy + sin_p
        self._floor_mask = denom > 1e-5
        self._z_floor_max = np.full((self.height, 1), np.inf, dtype=np.float32)
        self._z_floor_max[self._floor_mask] = self.zc / denom[self._floor_mask]

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

        # Preallocated ray grids cache: (h, w, step) -> (v_grid, u_grid, u_norm, v_norm)
        self._grid_cache: Dict[
            Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def _get_grid(
        self, h: int, w: int, step: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fetch or create cached pixel coordinates and normalized ray directions."""
        key = (h, w, step)
        if key not in self._grid_cache:
            v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
            u_norm = ((u_grid - self.cx) / self.fx).astype(np.float32)
            v_norm = ((v_grid - self.cy) / self.fy).astype(np.float32)
            self._grid_cache[key] = (v_grid, u_grid, u_norm, v_norm)
        return self._grid_cache[key]

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
        if depth_map.shape[0] == self.height:
            return np.minimum(depth_map, self._z_floor_max)

        h = depth_map.shape[0]
        v_grid = np.arange(h, dtype=np.float32)[:, None]
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        denom = cos_p * (v_grid - self.cy) / self.fy + sin_p
        floor_mask = denom > 1e-5

        z_floor_max = np.full((h, 1), np.inf, dtype=np.float32)
        z_floor_max[floor_mask] = self.zc / denom[floor_mask]
        return np.minimum(depth_map, z_floor_max)

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
        bounds_x: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
        is_preclamped: bool = False,
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
        :param bounds_x: Optional (min_x, max_x) arena boundary filter.
        :param bounds_y: Optional (min_y, max_y) arena boundary filter.
        :param is_preclamped: Whether ground plane was already clamped.
        :return: (points_world_Nx3, colors_rgb_Nx3)
        """
        clamped_depth = depth_map if is_preclamped else self.clamp_ground_plane(depth_map)
        h, w = clamped_depth.shape[:2]
        v_grid, u_grid, u_norm, v_norm = self._get_grid(h, w, subsample_step)
        z_vals = clamped_depth[0:h:subsample_step, 0:w:subsample_step]

        valid = (z_vals >= d_min) & (z_vals <= d_max) & np.isfinite(z_vals)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        z = z_vals[valid].astype(np.float32)

        # Camera optical coordinates (+X right, +Y down, +Z forward)
        pts_cam = np.empty((3, len(z)), dtype=np.float32)
        pts_cam[0] = u_norm[valid] * z
        pts_cam[1] = v_norm[valid] * z
        pts_cam[2] = z

        # Rigid transformation to map frame
        r_mat = t_world_cam_ros[:3, :3]
        t_vec = t_world_cam_ros[:3, 3:4]
        pts_world = (r_mat @ pts_cam + t_vec).T.astype(np.float32)

        # Discard subterranean points and open sky / ceiling floaters
        valid_mask = (pts_world[:, 2] >= min_z) & (pts_world[:, 2] <= max_z)
        if bounds_x is not None:
            valid_mask &= (pts_world[:, 0] >= bounds_x[0]) & (pts_world[:, 0] <= bounds_x[1])
        if bounds_y is not None:
            valid_mask &= (pts_world[:, 1] >= bounds_y[0]) & (pts_world[:, 1] <= bounds_y[1])

        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        pts_world = pts_world[valid_mask]
        v_sub = v_grid[valid][valid_mask]
        u_sub = u_grid[valid][valid_mask]
        colors = (image_rgb[v_sub, u_sub].astype(np.float32) / 255.0)

        return pts_world, colors

    @staticmethod
    def generate_floor_grid(
        bounds_x: Tuple[float, float] = (-2.4, 2.4),
        bounds_y: Tuple[float, float] = (-2.4, 2.4),
        floor_z: float = 0.0,
        floor_color: Tuple[float, float, float] = (0.85, 0.85, 0.87),
        step: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate continuous floor plane seed Gaussians to ensure 100% gap-free floor geometry.

        :param bounds_x: (min_x, max_x) floor extents in map frame.
        :param bounds_y: (min_y, max_y) floor extents in map frame.
        :param floor_z: Elevation of the ground plane (0.0m).
        :param floor_color: Default floor RGB color in [0, 1].
        :param step: Grid spacing in meters (default 0.03m = 3cm).
        :return: (floor_points_Nx3, floor_colors_Nx3)
        """
        xs = np.arange(bounds_x[0], bounds_x[1] + 1e-4, step, dtype=np.float32)
        ys = np.arange(bounds_y[0], bounds_y[1] + 1e-4, step, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        flat_z = np.full_like(flat_x, floor_z, dtype=np.float32)

        floor_pts = np.column_stack([flat_x, flat_y, flat_z])
        floor_cols = np.tile(
            np.array(floor_color, dtype=np.float32), (len(floor_pts), 1)
        )
        return floor_pts, floor_cols

    @staticmethod
    def voxel_downsample(
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.015,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse points and colors into a uniform spatial voxel grid using GPU acceleration.

        :param points: (N, 3) float32 3D world coordinates.
        :param colors: (N, 3) float32 normalized RGB colors.
        :param voxel_size: Grid cell dimension in meters (default 0.015 m = 1.5 cm).
        :return: (fused_points, fused_colors)
        """
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pts_t = torch.as_tensor(points, device=device, dtype=torch.float32)
        cols_t = torch.as_tensor(colors, device=device, dtype=torch.float32)

        voxel_coords = torch.floor(pts_t / voxel_size).to(torch.int64)
        _, inv_idx = torch.unique(voxel_coords, dim=0, return_inverse=True)
        n_voxels = int(inv_idx.max().item()) + 1

        counts = torch.bincount(inv_idx, minlength=n_voxels).unsqueeze(-1).float()
        sum_pts = torch.zeros(n_voxels, 3, device=device).scatter_add_(
            0, inv_idx.unsqueeze(-1).expand(-1, 3), pts_t
        )
        sum_cols = torch.zeros(n_voxels, 3, device=device).scatter_add_(
            0, inv_idx.unsqueeze(-1).expand(-1, 3), cols_t
        )

        fused_pts = (sum_pts / counts).cpu().numpy()
        fused_cols = (sum_cols / counts).cpu().numpy()
        return fused_pts, fused_cols
