"""End-to-end integration tests for standalone gsplat training and reconstruction."""

import os
import tempfile
import numpy as np
from plyfile import PlyData
import pytest
import torch

from khepera_3dgs.gsplat_trainer import GaussianModel, GsplatTrainer
from khepera_3dgs.pipeline_coordinator import ReconstructionPipeline
from khepera_3dgs.scripts_runner import generate_synthetic_tour


def test_gsplat_trainer_step():
    """Verify single optimization step with pose freezing and photometric loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(device=device)

    pts = np.random.uniform(-0.5, 0.5, (100, 3)).astype(np.float32)
    cols = np.random.uniform(0.1, 0.9, (100, 3)).astype(np.float32)
    model.initialize_from_pointcloud(pts, cols)

    trainer = GsplatTrainer(gaussian_model=model)

    gt_image = torch.rand(1, 480, 640, 3, device=device)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    k_mat = torch.tensor(
        [[[554.26, 0.0, 320.0], [0.0, 554.26, 240.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    gt_depth = torch.full((1, 480, 640), 1.5, dtype=torch.float32, device=device)

    # Ensure viewmat and k_mat are frozen
    assert not viewmat.requires_grad
    assert not k_mat.requires_grad

    loss, l1, ssim, d_loss = trainer.train_step(
        gt_image=gt_image,
        viewmat=viewmat,
        k_mat=k_mat,
        width=640,
        height=480,
        gt_depth=gt_depth,
    )

    assert loss > 0.0
    assert l1 >= 0.0
    assert 0.0 <= ssim <= 1.0
    assert d_loss >= 0.0

    # Ensure Gaussian parameter gradients are populated
    assert model.means.grad is not None
    assert model.scales.grad is not None
    assert model.quats.grad is not None
    assert model.opacities.grad is not None
    assert model.sh0.grad is not None


def test_end_to_end_reconstruction_pipeline():
    """Run full pipeline on synthetic trajectory and verify exported SuperSplat PLY."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = ReconstructionPipeline(
            output_dir=tmp_dir,
            voxel_size_m=0.03,  # Coarser for fast test
            num_iterations=20,  # Quick test iterations
        )

        keyframes, images = generate_synthetic_tour(num_keyframes=4)

        ply_file = pipeline.run_reconstruction(
            keyframes=keyframes,
            images_rgb=images,
            ply_filename="test_output.ply",
        )

        assert os.path.exists(ply_file)
        assert os.path.getsize(ply_file) > 1000

        ply_data = PlyData.read(ply_file)
        assert len(ply_data["vertex"]) > 0
        props = [p.name for p in ply_data["vertex"].properties]
        assert "x" in props and "rot_0" in props and "opacity" in props


def test_adaptive_densification_and_pruning():
    """Verify adaptive cloning, splitting, and opacity pruning in GsplatTrainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(device=device)

    pts = np.random.uniform(-0.5, 0.5, (30, 3)).astype(np.float32)
    cols = np.random.uniform(0.1, 0.9, (30, 3)).astype(np.float32)
    model.initialize_from_pointcloud(pts, cols)

    initial_count = model.num_gaussians
    trainer = GsplatTrainer(gaussian_model=model, densify_grad_thresh=0.0001)

    # Artificially populate gradients above threshold
    trainer._grad_accum += 1.0
    trainer._grad_counts += 1

    n_clone, n_split = trainer.densify_and_clone_or_split(max_scale_thresh=0.05)
    assert n_clone + n_split > 0
    assert model.num_gaussians > initial_count

    # Test pruning
    n_pruned = trainer.prune_low_opacity(prune_opacity_thresh=0.99)
    assert n_pruned > 0

