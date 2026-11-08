"""Standard Inria / PlayCanvas SuperSplat binary PLY exporter for 3D Gaussian Splatting."""

import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import torch


class SuperSplatPlyExporter:
    """
    Exports trained 3D Gaussian parameters to a standard binary PLY format.

    Compatible with PlayCanvas SuperSplat (https://superspl.at/editor).
    """

    @staticmethod
    def export_ply(
        output_path: str,
        means: np.ndarray,
        scales: np.ndarray,
        quats: np.ndarray,
        opacities: np.ndarray,
        sh_dc: np.ndarray,
        sh_rest: Optional[np.ndarray] = None,
        is_log_scale: bool = True,
        is_logit_opacity: bool = True,
        align_supersplat_y_up: bool = False,
        bounds_x: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
        bounds_z: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Serialize 3D Gaussians to binary little-endian PLY file.

        :param output_path: Destination path (.ply).
        :param means: (N, 3) Gaussian centers (x, y, z).
        :param scales: (N, 3) Gaussian radii or log-scales.
        :param quats: (N, 4) Normalized quaternions [w, x, y, z].
        :param opacities: (N,) or (N, 1) Opacities in [0, 1] or logit space.
        :param sh_dc: (N, 3) 0th order Spherical Harmonics (diffuse color).
        :param sh_rest: Optional (N, K) higher order SH coefficients.
        :param is_log_scale: If False, applies np.log(scales).
        :param is_logit_opacity: If False, applies inverse sigmoid logit(alpha).
        :param align_supersplat_y_up: Transform ROS Z-up frame into SuperSplat Y-up frame.
        :return: Absolute path to written PLY file.
        """
        # Convert Torch tensors if needed
        if isinstance(means, torch.Tensor):
            means = means.detach().cpu().numpy()
        if isinstance(scales, torch.Tensor):
            scales = scales.detach().cpu().numpy()
        if isinstance(quats, torch.Tensor):
            quats = quats.detach().cpu().numpy()
        if isinstance(opacities, torch.Tensor):
            opacities = opacities.detach().cpu().numpy()
        if isinstance(sh_dc, torch.Tensor):
            sh_dc = sh_dc.detach().cpu().numpy()
        if sh_rest is not None and isinstance(sh_rest, torch.Tensor):
            sh_rest = sh_rest.detach().cpu().numpy()

        n_pts = len(means)
        if n_pts == 0:
            raise ValueError("Cannot export empty Gaussian field (0 points).")

        means = np.asarray(means, dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32)
        quats = np.asarray(quats, dtype=np.float32)
        opacities = np.asarray(opacities, dtype=np.float32).reshape(n_pts, 1)
        sh_dc = np.asarray(sh_dc, dtype=np.float32).reshape(n_pts, 3)

        # Boundary and ceiling clipping in ROS coordinates
        if bounds_x is not None or bounds_y is not None or bounds_z is not None:
            mask = np.ones(n_pts, dtype=bool)
            if bounds_x is not None:
                mask &= (means[:, 0] >= bounds_x[0]) & (means[:, 0] <= bounds_x[1])
            if bounds_y is not None:
                mask &= (means[:, 1] >= bounds_y[0]) & (means[:, 1] <= bounds_y[1])
            if bounds_z is not None:
                mask &= (means[:, 2] >= bounds_z[0]) & (means[:, 2] <= bounds_z[1])
            if np.any(mask):
                means = means[mask]
                scales = scales[mask]
                quats = quats[mask]
                opacities = opacities[mask]
                sh_dc = sh_dc[mask]
                if sh_rest is not None:
                    sh_rest = sh_rest[mask]
                n_pts = len(means)

        # Ensure floor plane Gaussians (|Z| <= 0.02 in ROS frame) maintain solid visible opacity
        floor_mask = np.abs(means[:, 2]) <= 0.02
        if np.any(floor_mask):
            if is_logit_opacity:
                opacities[floor_mask] = np.maximum(opacities[floor_mask], 0.85)
            else:
                opacities[floor_mask] = np.maximum(opacities[floor_mask], 0.70)

        # Transform ROS Z-up frame to WebGL/SuperSplat Y-up frame if requested
        if align_supersplat_y_up:
            # SuperSplat applies default [0, 0, 180] Euler rotation on import.
            # We map ROS (+X forward, +Y left, +Z up) such that after SuperSplat import:
            # X_view = -Y_ros (Right), Y_view = +Z_ros (Up), Z_view = -X_ros (Depth)
            # rot_mat maps [X, Y, Z] -> [Y, -Z, -X], which R_z(180) flips to [-Y, Z, -X].
            rot_mat = np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            )
            r_align = ScipyRotation.from_matrix(rot_mat)
            means = (rot_mat @ means.T).T

            # Normalize and rotate quaternions
            quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
            quats = np.where(
                quat_norms > 1e-7, quats / quat_norms, np.array([[1.0, 0.0, 0.0, 0.0]])
            )
            quats_xyzw = quats[:, [1, 2, 3, 0]]
            r_orig = ScipyRotation.from_quat(quats_xyzw)
            r_new = r_align * r_orig
            new_xyzw = r_new.as_quat()
            quats = new_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

        # Ensure scales are in log-space
        if not is_log_scale:
            scales = np.log(np.maximum(scales, 1e-7))

        # Ensure opacities are in logit-space
        if not is_logit_opacity:
            clipped_alpha = np.clip(opacities, 1e-4, 1.0 - 1e-4)
            opacities = np.log(clipped_alpha / (1.0 - clipped_alpha))

        # Normalize rotation quaternions [w, x, y, z]
        quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = np.where(quat_norms > 1e-7, quats / quat_norms, np.array([[1.0, 0.0, 0.0, 0.0]]))

        normals = np.zeros((n_pts, 3), dtype=np.float32)

        # Flatten sh_rest if present
        if sh_rest is not None and sh_rest.size > 0:
            sh_rest = sh_rest.reshape(n_pts, -1).astype(np.float32)
            n_rest = sh_rest.shape[1]
        else:
            sh_rest = None
            n_rest = 0

        # Fast contiguous buffer serialization (zero-copy memory layout)
        total_cols = 17 + n_rest
        flat_buffer = np.empty((n_pts, total_cols), dtype=np.float32)
        flat_buffer[:, 0:3] = means
        flat_buffer[:, 3:6] = normals
        flat_buffer[:, 6:9] = sh_dc
        col_idx = 9
        if n_rest > 0 and sh_rest is not None:
            flat_buffer[:, col_idx : col_idx + n_rest] = sh_rest
            col_idx += n_rest
        flat_buffer[:, col_idx : col_idx + 1] = opacities
        flat_buffer[:, col_idx + 1 : col_idx + 4] = scales
        flat_buffer[:, col_idx + 4 : col_idx + 8] = quats
        ply_bytes = flat_buffer.tobytes()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n_pts}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
        ]

        if n_rest > 0:
            for idx in range(n_rest):
                header_lines.append(f"property float f_rest_{idx}")

        header_lines.extend([
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header\n",
        ])

        header_str = "\n".join(header_lines)

        with open(output_path, "wb") as f_out:
            f_out.write(header_str.encode("ascii"))
            f_out.write(ply_bytes)

        return os.path.abspath(output_path)
