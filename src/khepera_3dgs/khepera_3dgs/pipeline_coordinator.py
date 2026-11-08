"""End-to-end 3D Gaussian Splatting reconstruction pipeline coordinator for Khepera IV."""

import logging
import os
import time
from typing import Callable, List, Optional

import numpy as np
import torch

from .coordinate_lifting import CoordinateLifter
from .gsplat_trainer import GaussianModel, GsplatTrainer
from .metric_depth import MetricDepthEstimator
from .ply_exporter import SuperSplatPlyExporter
from .spatial_decimation import KeyframeData, SpatialDecimator

LOGGER = logging.getLogger("khepera_3dgs.coordinator")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds into a clean human-readable string.

    :param seconds: Total duration in seconds.
    :return: Formatted string (e.g. '42.1s', '1m 24s', '1h 05m').
    """
    if seconds < 0:
        return "0.0s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {rem_secs:02d}s"
    hours = int(minutes // 60)
    rem_mins = int(minutes % 60)
    return f"{hours}h {rem_mins:02d}m"


class ReconstructionPipeline:
    """
    Coordinates spatial decimation, metric depth, 3D lifting, gsplat training, and PLY export.

    Produces a drift-free 3D radiance field exported directly for PlayCanvas SuperSplat.
    """

    def __init__(
        self,
        output_dir: str = "output/khepera_3dgs",
        checkpoint_path: Optional[str] = None,
        camera_height_m: float = 0.0283,
        camera_x_offset_m: float = 0.0682,
        camera_pitch_rad: float = 0.0,
        fx: float = 554.256258,
        fy: float = 554.256258,
        cx: float = 320.0,
        cy: float = 240.0,
        width: int = 640,
        height: int = 480,
        voxel_size_m: float = 0.015,
        num_iterations: int = 2000,
        enable_floor_scale: bool = False,
        align_supersplat_y_up: bool = True,
        max_height_m: float = 3.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the complete reconstruction pipeline.

        :param output_dir: Directory to save all artifacts (.ply, depths, keyframes).
        :param checkpoint_path: Optional path to Depth-Anything-V2 metric checkpoint.
        :param camera_height_m: Ground clearance height h_floor (0.0283 m).
        :param camera_x_offset_m: Camera forward offset (0.0682 m).
        :param camera_pitch_rad: Downward camera pitch in radians (0.0 rad).
        :param fx: Horizontal focal length in pixels.
        :param fy: Vertical focal length in pixels.
        :param cx: Optical center x in pixels.
        :param cy: Optical center y in pixels.
        :param width: Native image width in pixels.
        :param height: Native image height in pixels.
        :param voxel_size_m: Voxel downsampling leaf size (0.015 m = 1.5 cm).
        :param num_iterations: Training optimization steps (2,000 to 5,000).
        :param enable_floor_scale: Enable floor ROI depth scaling (default False).
        :param align_supersplat_y_up: Rotate exported PLY to WebGL/SuperSplat Y-up frame.
        :param max_height_m: Maximum ceiling/arena height in meters (default 3.0m).
        :param device: Execution device ('cuda' or 'cpu').
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device(device)
        self.num_iterations = int(num_iterations)
        self.voxel_size = float(voxel_size_m)
        self.align_supersplat_y_up = bool(align_supersplat_y_up)
        self.max_height = float(max_height_m)
        self.width = int(width)
        self.height = int(height)

        # 1. Spatial Decimator
        self.decimator = SpatialDecimator()

        # 2. Metric Depth Estimator
        self.depth_estimator = MetricDepthEstimator(
            checkpoint_path=checkpoint_path,
            camera_height_m=camera_height_m,
            focal_length_y=fy,
            principal_point_y=cy,
            enable_floor_scale=enable_floor_scale,
            device=device,
        )

        # 3. Coordinate Lifter
        self.lifter = CoordinateLifter(
            camera_x_offset_m=camera_x_offset_m,
            camera_z_offset_m=camera_height_m,
            camera_pitch_rad=camera_pitch_rad,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
        )

    def run_reconstruction(
        self,
        keyframes: List[KeyframeData],
        images_rgb: List[np.ndarray],
        ply_filename: str = "supersplat_khepera.ply",
        depth_maps: Optional[List[np.ndarray]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute full pipeline from keyframe selection to trained SuperSplat PLY.

        :param keyframes: List of selected KeyframeData records.
        :param images_rgb: List of corresponding RGB image arrays (H, W, 3).
        :param ply_filename: Name of the output PLY artifact.
        :param depth_maps: Optional precomputed/sensor depth maps (H, W) in meters.
        :param progress_callback: Optional callback for streaming status and ETA logs.
        :return: Path to the generated SuperSplat PLY file.
        """
        n_kf = len(keyframes)
        if n_kf == 0:
            raise ValueError("No keyframes provided for reconstruction.")

        def _report(msg: str) -> None:
            LOGGER.info(msg)
            if progress_callback is not None:
                progress_callback(msg)

        _report(f"Starting 3DGS pipeline with {n_kf} keyframes...")
        t_start = time.time()

        # Phase 1 & 2: Metric Depth & Ground Plane Calibration
        if depth_maps is not None and len(depth_maps) == n_kf:
            _report(f"[Phase 1/4] Using {len(depth_maps)} provided sensor depth maps...")
            resolved_depth_maps = depth_maps
        else:
            _report("[Phase 1/4] Estimating metric depth via Depth-Anything-V2...")
            t_depth_start = time.time()

            def _depth_progress(done: int, total: int) -> None:
                el_depth = time.time() - t_depth_start
                eta_depth = (el_depth / done) * (total - done) if done < total else 0.0
                _report(
                    f"[Phase 1/4] Depth: {done}/{total} ({done / total * 100:.1f}%) | "
                    f"Elapsed: {format_duration(el_depth)} | "
                    f"ETA: {format_duration(eta_depth)}"
                )

            resolved_depth_maps = self.depth_estimator.process_batch(
                images_rgb,
                output_dir=self.output_dir,
                frame_ids=[kf.id for kf in keyframes],
                batch_size=4,
                progress_callback=_depth_progress,
            )

        # Phase 3: Coordinate Lifting & Initial Point Cloud Seeding
        LOGGER.info("[Phase 2/4] Backprojecting and seeding fused point cloud...")
        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        viewmats_list: List[np.ndarray] = []

        # Clamp ground plane on all depth maps for physically consistent floor geometry
        clamped_depth_maps = [
            self.lifter.clamp_ground_plane(d) for d in resolved_depth_maps
        ]

        for kf, img, depth in zip(keyframes, images_rgb, clamped_depth_maps):
            t_world_cam_ros, t_cam_opengl, _ = self.lifter.compute_pose_matrices(
                kf.x, kf.y, kf.yaw
            )
            # Use ROS optical world-to-camera viewmatrix for gsplat rasterization
            viewmatrix_ros = self.lifter.get_gsplat_viewmatrix(t_world_cam_ros)
            viewmats_list.append(viewmatrix_ros)

            pts_w, cols = self.lifter.unproject_depth_map(
                depth,
                img,
                t_world_cam_ros,
                d_min=0.08,
                d_max=6.00 if self.max_height > 1.0 else 3.50,
                max_z=self.max_height,
                subsample_step=2,
                is_preclamped=True,
            )
            if len(pts_w) > 0:
                all_pts.append(pts_w)
                all_cols.append(cols)

        if not all_pts:
            raise RuntimeError("Failed to backproject any valid 3D points from depth maps.")

        merged_pts = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols)
        LOGGER.info("Raw backprojected points: %d", len(merged_pts))

        fused_pts, fused_cols = self.lifter.voxel_downsample(
            merged_pts, merged_cols, voxel_size=self.voxel_size
        )
        LOGGER.info(
            "Voxel downsampling (%.3fm) resulted in %d seed Gaussians.",
            self.voxel_size,
            len(fused_pts),
        )

        # Scale iterations to ensure adequate gradient steps per keyframe
        total_iters = max(self.num_iterations, n_kf * 25)

        # Phase 4: Standalone gsplat Training with SplaTAM Depth Supervision
        _report(
            f"[Phase 3/4] Optimizing 3D radiance field via gsplat "
            f"({total_iters} steps, {n_kf} views)..."
        )
        gaussian_model = GaussianModel(device=self.device)
        gaussian_model.initialize_from_pointcloud(
            fused_pts, fused_cols, voxel_size_m=self.voxel_size
        )

        trainer = GsplatTrainer(
            gaussian_model=gaussian_model,
            densify_grad_thresh=0.0002,
            prune_opacity_thresh=0.08,
            w_depth=1.0,
            max_z=self.max_height,
        )

        # Prepare PyTorch training tensors
        gt_tensors = [
            torch.tensor(img / 255.0, dtype=torch.float32, device=self.device).unsqueeze(0)
            for img in images_rgb
        ]
        depth_tensors = [
            torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(0)
            for d in clamped_depth_maps
        ]
        viewmat_tensors = [
            torch.tensor(vm, dtype=torch.float32, device=self.device).unsqueeze(0)
            for vm in viewmats_list
        ]
        k_tensor = torch.tensor(
            self.lifter.k_matrix, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        t_train_start = time.time()

        # Training loop
        for step in range(1, total_iters + 1):
            # 1. Splatfacto Exponential LR Decay for Centroids
            trainer.update_learning_rates(step, total_iters)

            kf_idx = step % n_kf
            gt_img = gt_tensors[kf_idx]
            gt_d = depth_tensors[kf_idx]
            vm = viewmat_tensors[kf_idx]

            loss, l1, ssim, d_loss = trainer.train_step(
                gt_image=gt_img,
                viewmat=vm,
                k_mat=k_tensor,
                width=self.width,
                height=self.height,
                gt_depth=gt_d,
            )

            # 2. Splatfacto Anti-Floater Opacity Reset (clears transient ceiling/sky floaters)
            if (
                step in (int(total_iters * 0.3), int(total_iters * 0.6))
                and step <= total_iters - 300
            ):
                trainer.reset_opacities(reset_val=0.01)
                _report(
                    f"Step {step}: Splatfacto Opacity Reset to 0.01 (purging floater clouds)"
                )

            # 3. Adaptive density control: clone/split every 250 steps up to total_iters - 300
            if step % 250 == 0 and step <= total_iters - 300:
                n_clone, n_split = trainer.densify_and_clone_or_split(
                    max_scale_thresh=0.03, max_gaussians=250000
                )
                if n_clone > 0 or n_split > 0:
                    _report(
                        f"Step {step}: Densification: cloned {n_clone}, split {n_split} "
                        f"(total: {gaussian_model.num_gaussians} Gaussians)"
                    )

            # 4. Prune transparent and out-of-bounds splats every 150 iterations
            if step % 150 == 0:
                n_pruned = trainer.prune_low_opacity(
                    max_scale_thresh=0.08,
                    max_z=self.max_height,
                )
                if n_pruned > 0:
                    _report(
                        f"Step {step}: Pruned {n_pruned} low-opacity/floater Gaussians "
                        f"(current total: {gaussian_model.num_gaussians})"
                    )

            if step % 50 == 0 or step == total_iters or step == 10:
                step_elapsed = time.time() - t_train_start
                sec_per_step = step_elapsed / step
                eta_sec = (total_iters - step) * sec_per_step
                pct = (step / total_iters) * 100.0
                _report(
                    f"[Step {step}/{total_iters} ({pct:.1f}%)] "
                    f"Elapsed: {format_duration(step_elapsed)} | "
                    f"ETA: {format_duration(eta_sec)} | "
                    f"Loss: {loss:.4f} | Depth: {d_loss:.4f} | "
                    f"Splats: {gaussian_model.num_gaussians}"
                )

        # Final cleanup prune of low opacity and boundary floaters
        final_pruned = trainer.prune_low_opacity(
            prune_opacity_thresh=0.08,
            max_scale_thresh=0.08,
            max_z=self.max_height,
        )
        if final_pruned > 0:
            LOGGER.info("Final cleanup pruned %d out-of-bounds/floater Gaussians", final_pruned)

        # Phase 5: SuperSplat PLY Export
        _report("[Phase 4/4] Exporting model to SuperSplat-compatible binary PLY...")
        ply_path = os.path.join(self.output_dir, ply_filename)
        SuperSplatPlyExporter.export_ply(
            output_path=ply_path,
            means=gaussian_model.means,
            scales=gaussian_model.scales,
            quats=gaussian_model.quats,
            opacities=gaussian_model.opacities,
            sh_dc=gaussian_model.sh0,
            is_log_scale=True,
            is_logit_opacity=True,
            align_supersplat_y_up=self.align_supersplat_y_up,
        )

        elapsed = time.time() - t_start
        file_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        _report(
            f"Reconstruction completed in {format_duration(elapsed)}. "
            f"Exported {ply_path} ({file_size_mb:.2f} MB, "
            f"{gaussian_model.num_gaussians} Gaussians)."
        )

        return ply_path
