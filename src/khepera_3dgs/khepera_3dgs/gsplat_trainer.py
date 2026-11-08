"""Standalone PyTorch training engine for 3D Gaussian Splatting using gsplat."""

import logging
import math
from typing import Optional, Tuple

import gsplat
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("khepera_3dgs.gsplat_trainer")


def compute_ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    channel: int = 3,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two image tensors in [0, 1].

    :param img1: (1, H, W, C) or (1, C, H, W) float32 tensor.
    :param img2: (1, H, W, C) or (1, C, H, W) float32 tensor.
    :param window_size: Kernel dimension (default 11).
    :param channel: Number of color channels (default 3).
    :return: Scalar SSIM value in [0, 1].
    """
    # Ensure shape is (B, C, H, W)
    if img1.shape[-1] == channel:
        img1 = img1.permute(0, 3, 1, 2)
    if img2.shape[-1] == channel:
        img2 = img2.permute(0, 3, 1, 2)

    sigma = 1.5
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
        device=img1.device,
    )
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


class GaussianModel:
    """Encapsulates trainable 3D Gaussian parameters."""

    def __init__(self, device: torch.device) -> None:
        """Initialize empty parameter buffers on device."""
        self.device = device
        self._means: Optional[torch.nn.Parameter] = None
        self._scales: Optional[torch.nn.Parameter] = None
        self._quats: Optional[torch.nn.Parameter] = None
        self._opacities: Optional[torch.nn.Parameter] = None
        self._sh0: Optional[torch.nn.Parameter] = None

    @property
    def means(self) -> torch.Tensor:
        """Gaussian centroids (N, 3)."""
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        """Gaussian log scales (N, 3)."""
        return self._scales

    @property
    def quats(self) -> torch.Tensor:
        """Gaussian rotation quaternions (N, 4)."""
        return self._quats

    @property
    def opacities(self) -> torch.Tensor:
        """Gaussian raw logit opacities (N,)."""
        return self._opacities

    @property
    def sh0(self) -> torch.Tensor:
        """0th order spherical harmonics (N, 3)."""
        return self._sh0

    @property
    def num_gaussians(self) -> int:
        """Return current population count."""
        return len(self._means) if self._means is not None else 0

    def initialize_from_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        k_neighbors: int = 3,
    ) -> None:
        """
        Initialize Gaussian parameters from fused 3D points.

        :param points: (N, 3) float32 coordinates in world frame.
        :param colors: (N, 3) float32 normalized RGB in [0, 1].
        :param k_neighbors: Number of nearest neighbors for initial scale.
        """
        n_pts = len(points)
        if n_pts == 0:
            raise ValueError("Cannot initialize GaussianModel with zero points.")

        # Means
        means_t = torch.tensor(points, dtype=torch.float32, device=self.device)

        # 3-NN distance for initial scale estimation (bounded to avoid gigantic splats)
        if n_pts > k_neighbors:
            kdtree = KDTree(points)
            dists, _ = kdtree.query(points, k=k_neighbors + 1)
            mean_dists = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.03)
        else:
            mean_dists = np.full(n_pts, 0.015, dtype=np.float32)

        log_scales = np.log(np.tile(mean_dists[:, None], (1, 3)))
        scales_t = torch.tensor(log_scales, dtype=torch.float32, device=self.device)

        # Identity rotations [w=1, x=0, y=0, z=0]
        quats_np = np.zeros((n_pts, 4), dtype=np.float32)
        quats_np[:, 0] = 1.0
        quats_t = torch.tensor(quats_np, dtype=torch.float32, device=self.device)

        # Initial opacities: logit(0.5) = 0.0 for crisp surface boundaries (SplaTAM standard)
        init_alpha = 0.50
        logit_alpha = float(math.log(init_alpha / (1.0 - init_alpha)))
        opacities_t = torch.full((n_pts,), logit_alpha, dtype=torch.float32, device=self.device)

        # Convert RGB to DC Spherical Harmonics
        # C_0 = (RGB - 0.5) / 0.28209479177387814
        c0 = 0.28209479177387814
        sh0_np = (colors - 0.5) / c0
        sh0_t = torch.tensor(sh0_np, dtype=torch.float32, device=self.device)

        # Make parameters
        self._means = torch.nn.Parameter(means_t.requires_grad_(True))
        self._scales = torch.nn.Parameter(scales_t.requires_grad_(True))
        self._quats = torch.nn.Parameter(quats_t.requires_grad_(True))
        self._opacities = torch.nn.Parameter(opacities_t.requires_grad_(True))
        self._sh0 = torch.nn.Parameter(sh0_t.requires_grad_(True))

        LOGGER.info("Initialized %d Gaussians on %s", n_pts, self.device)


class GsplatTrainer:
    """
    Standalone training engine optimizing 3D Gaussians with frozen camera poses.

    Computes SplaTAM-grade joint Photometric (L1 + SSIM) and Dense Depth supervision
    with adaptive density control.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        lr_means: float = 1.6e-4,
        lr_scales: float = 5.0e-3,
        lr_quats: float = 1.0e-3,
        lr_opacities: float = 5.0e-2,
        lr_sh: float = 2.5e-3,
        ssim_lambda: float = 0.20,
        w_depth: float = 1.0,
        densify_grad_thresh: float = 0.0002,
        prune_opacity_thresh: float = 0.08,
        min_z: float = -0.02,
        max_z: float = 3.00,
    ) -> None:
        """
        Initialize trainer and parameter optimizers.

        :param gaussian_model: GaussianModel container.
        :param lr_means: Learning rate for spatial means.
        :param lr_scales: Learning rate for log scales.
        :param lr_quats: Learning rate for rotations.
        :param lr_opacities: Learning rate for logit opacities.
        :param lr_sh: Learning rate for spherical harmonics.
        :param ssim_lambda: Weight for SSIM loss (default 0.20).
        :param w_depth: Weight for dense depth loss (SplaTAM standard 1.0).
        :param densify_grad_thresh: Viewspace gradient threshold for cloning/splitting.
        :param prune_opacity_thresh: Opacity threshold below which Gaussians are pruned.
        :param min_z: Minimum allowable map Z coordinate (subterranean floor filter).
        :param max_z: Maximum allowable map Z coordinate (ceiling height filter).
        """
        self.model = gaussian_model
        self.device = gaussian_model.device
        self.lambda_ssim = float(ssim_lambda)
        self.w_depth = float(w_depth)
        self.grad_thresh = float(densify_grad_thresh)
        self.prune_thresh = float(prune_opacity_thresh)
        self.min_z = float(min_z)
        self.max_z = float(max_z)

        # Gradient accumulator for adaptive density control
        self._grad_accum = torch.zeros(self.model.num_gaussians, device=self.device)
        self._grad_counts = torch.zeros(
            self.model.num_gaussians, dtype=torch.int32, device=self.device
        )

        param_groups = [
            {"params": [self.model.means], "lr": lr_means, "name": "means"},
            {"params": [self.model.scales], "lr": lr_scales, "name": "scales"},
            {"params": [self.model.quats], "lr": lr_quats, "name": "quats"},
            {"params": [self.model.opacities], "lr": lr_opacities, "name": "opacities"},
            {"params": [self.model.sh0], "lr": lr_sh, "name": "sh0"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)

    def train_step(
        self,
        gt_image: torch.Tensor,
        viewmat: torch.Tensor,
        k_mat: torch.Tensor,
        width: int,
        height: int,
        gt_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Execute single forward-backward optimization step with joint RGB+Depth loss.

        :param gt_image: (1, H, W, 3) ground truth RGB in [0, 1].
        :param viewmat: (1, 4, 4) world-to-camera matrix (requires_grad=False).
        :param k_mat: (1, 3, 3) camera intrinsic matrix (requires_grad=False).
        :param width: Image width in pixels.
        :param height: Image height in pixels.
        :param gt_depth: Optional (1, H, W) or (1, H, W, 1) metric depth in meters.
        :return: (total_loss, l1_loss, ssim_loss, depth_loss)
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Normalized quaternions
        quats_norm = F.normalize(self.model.quats, p=2, dim=-1)

        # Activation functions
        scales = torch.exp(self.model.scales)
        opacities = torch.sigmoid(self.model.opacities)

        # Convert SH DC to RGB: RGB = SH0 * 0.28209479 + 0.5
        c0 = 0.28209479177387814
        colors = torch.clamp(self.model.sh0 * c0 + 0.5, 0.0, 1.0)

        # Rasterize via gsplat in RGB+ED mode for joint Expected Depth rendering
        renders, alphas, meta = gsplat.rasterization(
            means=self.model.means,
            quats=quats_norm,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=k_mat,
            width=width,
            height=height,
            render_mode="RGB+ED",
        )

        pred_img = renders[..., :3]  # (1, H, W, 3)
        pred_depth = renders[..., 3]  # (1, H, W)

        # Photometric losses
        l1_loss = F.l1_loss(pred_img, gt_image)
        ssim_val = compute_ssim_loss(pred_img, gt_image)
        rgb_loss = (1.0 - self.lambda_ssim) * l1_loss + self.lambda_ssim * (1.0 - ssim_val)

        # Dense Depth loss (SplaTAM formulation)
        depth_loss = torch.tensor(0.0, device=self.device)
        if gt_depth is not None:
            if gt_depth.ndim == 4 and gt_depth.shape[-1] == 1:
                target_depth = gt_depth.squeeze(-1)
            else:
                target_depth = gt_depth
            valid_mask = (
                (target_depth > 0.05)
                & (target_depth <= 3.50)
                & torch.isfinite(target_depth)
            )
            # Mask out rays pointing above ceiling bounds (z_world > max_z)
            v_rows = torch.arange(height, device=self.device).view(1, height, 1)
            cy = k_mat[0, 1, 2]
            fy = k_mat[0, 1, 1]
            sky_thresh = ((self.max_z - 0.0283) * fy) / torch.clamp(cy - v_rows, min=1.0)
            sky_mask = (v_rows < cy) & (target_depth > sky_thresh)
            valid_mask = valid_mask & (~sky_mask)

            if valid_mask.any():
                depth_diff = torch.abs(pred_depth[valid_mask] - target_depth[valid_mask])
                depth_loss = depth_diff.mean()

        total_loss = rgb_loss + self.w_depth * depth_loss
        total_loss.backward()

        # Track viewspace gradients for adaptive density control
        if self.model.means.grad is not None:
            with torch.no_grad():
                grad_norm = torch.norm(self.model.means.grad[:, :2], dim=-1)
                self._grad_accum += grad_norm
                self._grad_counts += 1

        self.optimizer.step()

        return (
            float(total_loss.item()),
            float(l1_loss.item()),
            float(ssim_val.item()),
            float(depth_loss.item()),
        )

    def prune_low_opacity(
        self,
        prune_opacity_thresh: Optional[float] = None,
        max_scale_thresh: float = 0.12,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> int:
        """
        Prune Gaussians with low opacity, excessive scale, or outside vertical bounds.

        :param prune_opacity_thresh: Custom opacity cutoff (default self.prune_thresh).
        :param max_scale_thresh: Max allowable scale dimension in meters.
        :param min_z: Minimum allowable map Z coordinate (default self.min_z).
        :param max_z: Maximum allowable map Z coordinate (default self.max_z).
        :return: Number of pruned Gaussians.
        """
        thresh = float(prune_opacity_thresh) if prune_opacity_thresh else self.prune_thresh
        z_min = float(min_z) if min_z is not None else self.min_z
        z_max = float(max_z) if max_z is not None else self.max_z
        with torch.no_grad():
            cur_opacities = torch.sigmoid(self.model.opacities)
            cur_scales = torch.exp(self.model.scales)
            max_scale, _ = torch.max(cur_scales, dim=-1)
            z_pos = self.model.means[:, 2]

            keep_mask = (
                (cur_opacities >= thresh)
                & (max_scale <= max_scale_thresh)
                & (z_pos >= z_min)
                & (z_pos <= z_max)
            )
            n_pruned = int((~keep_mask).sum().item())

            if n_pruned > 0 and keep_mask.sum() > 50:
                self._apply_pruning_mask(keep_mask)

        return n_pruned

    def densify_and_clone_or_split(
        self,
        max_scale_thresh: float = 0.03,
        max_gaussians: int = 250000,
    ) -> Tuple[int, int]:
        """
        Adaptive density control: clone under-reconstructed and split over-reconstructed splats.

        Identifies Gaussians with positional gradient magnitude above grad_thresh.
        If max scale <= max_scale_thresh: clone (duplicate Gaussian).
        If max scale > max_scale_thresh: split into two smaller Gaussians (scale / 1.6).

        :param max_scale_thresh: Maximum scale threshold to distinguish clone vs split.
        :param max_gaussians: Upper limit on total Gaussian count to prevent OOM.
        :return: (num_cloned, num_split)
        """
        n_current = self.model.num_gaussians
        if n_current >= max_gaussians or n_current == 0:
            return 0, 0

        with torch.no_grad():
            avg_grads = self._grad_accum / torch.clamp(self._grad_counts.float(), min=1.0)
            high_grad_mask = (avg_grads >= self.grad_thresh) & (self._grad_counts > 0)

            if not high_grad_mask.any():
                self._grad_accum.zero_()
                self._grad_counts.zero_()
                return 0, 0

            cur_scales = torch.exp(self.model.scales)
            max_cur_scales, _ = torch.max(cur_scales, dim=-1)

            clone_mask = high_grad_mask & (max_cur_scales <= max_scale_thresh)
            split_mask = high_grad_mask & (max_cur_scales > max_scale_thresh)

            n_clone = int(clone_mask.sum().item())
            n_split = int(split_mask.sum().item())

            available = max_gaussians - n_current
            if n_clone + n_split > available:
                ratio = available / float(n_clone + n_split + 1e-6)
                rand_clone = torch.rand(n_current, device=self.device) < ratio
                rand_split = torch.rand(n_current, device=self.device) < ratio
                clone_mask = clone_mask & rand_clone
                split_mask = split_mask & rand_split
                n_clone = int(clone_mask.sum().item())
                n_split = int(split_mask.sum().item())

            if n_clone == 0 and n_split == 0:
                self._grad_accum.zero_()
                self._grad_counts.zero_()
                return 0, 0

            new_means_list = [self.model.means.detach().clone()]
            new_scales_list = [self.model.scales.detach().clone()]
            new_quats_list = [self.model.quats.detach().clone()]
            new_opacities_list = [self.model.opacities.detach().clone()]
            new_sh0_list = [self.model.sh0.detach().clone()]

            if n_clone > 0:
                new_means_list.append(self.model.means[clone_mask].detach().clone())
                new_scales_list.append(self.model.scales[clone_mask].detach().clone())
                new_quats_list.append(self.model.quats[clone_mask].detach().clone())
                new_opacities_list.append(self.model.opacities[clone_mask].detach().clone())
                new_sh0_list.append(self.model.sh0[clone_mask].detach().clone())

            if n_split > 0:
                split_means = self.model.means[split_mask].detach()
                split_scales = self.model.scales[split_mask].detach() - math.log(1.6)
                split_quats = self.model.quats[split_mask].detach().clone()
                split_opacities = self.model.opacities[split_mask].detach().clone()
                split_sh0 = self.model.sh0[split_mask].detach().clone()

                stds = torch.exp(split_scales)
                offsets = torch.randn_like(split_means) * stds * 0.5
                child1_means = split_means + offsets
                child2_means = split_means - offsets

                new_means_list[0][split_mask] = child1_means
                new_scales_list[0][split_mask] = split_scales

                new_means_list.append(child2_means)
                new_scales_list.append(split_scales)
                new_quats_list.append(split_quats)
                new_opacities_list.append(split_opacities)
                new_sh0_list.append(split_sh0)

            all_means = torch.cat(new_means_list, dim=0).requires_grad_(True)
            all_scales = torch.cat(new_scales_list, dim=0).requires_grad_(True)
            all_quats = torch.cat(new_quats_list, dim=0).requires_grad_(True)
            all_opacities = torch.cat(new_opacities_list, dim=0).requires_grad_(True)
            all_sh0 = torch.cat(new_sh0_list, dim=0).requires_grad_(True)

            self.model._means = torch.nn.Parameter(all_means)
            self.model._scales = torch.nn.Parameter(all_scales)
            self.model._quats = torch.nn.Parameter(all_quats)
            self.model._opacities = torch.nn.Parameter(all_opacities)
            self.model._sh0 = torch.nn.Parameter(all_sh0)

            self._reset_optimizer_and_grad_buffers()

            return n_clone, n_split

    def _apply_pruning_mask(self, keep_mask: torch.Tensor) -> None:
        """Update model parameters and optimizer state after pruning."""
        new_means = self.model.means[keep_mask].detach().clone().requires_grad_(True)
        new_scales = self.model.scales[keep_mask].detach().clone().requires_grad_(True)
        new_quats = self.model.quats[keep_mask].detach().clone().requires_grad_(True)
        new_opacities = self.model.opacities[keep_mask].detach().clone().requires_grad_(True)
        new_sh0 = self.model.sh0[keep_mask].detach().clone().requires_grad_(True)

        self.model._means = torch.nn.Parameter(new_means)
        self.model._scales = torch.nn.Parameter(new_scales)
        self.model._quats = torch.nn.Parameter(new_quats)
        self.model._opacities = torch.nn.Parameter(new_opacities)
        self.model._sh0 = torch.nn.Parameter(new_sh0)

        self._reset_optimizer_and_grad_buffers()

    def _reset_optimizer_and_grad_buffers(self) -> None:
        """Reset parameter optimizers and gradient tracking buffers."""
        param_groups = [
            {"params": [self.model.means], "lr": 1.6e-4, "name": "means"},
            {"params": [self.model.scales], "lr": 5.0e-3, "name": "scales"},
            {"params": [self.model.quats], "lr": 1.0e-3, "name": "quats"},
            {"params": [self.model.opacities], "lr": 5.0e-2, "name": "opacities"},
            {"params": [self.model.sh0], "lr": 2.5e-3, "name": "sh0"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)

        self._grad_accum = torch.zeros(self.model.num_gaussians, device=self.device)
        self._grad_counts = torch.zeros(
            self.model.num_gaussians, dtype=torch.int32, device=self.device
        )
