"""CLI entry point for Khepera IV 3DGS training and reconstruction."""

import argparse
import logging
import sys
from typing import List, Tuple

import cv2
import numpy as np

from .pipeline_coordinator import ReconstructionPipeline
from .rosbag_processor import RosbagKeyframeExtractor
from .spatial_decimation import KeyframeData, SpatialDecimator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("khepera_3dgs.cli")


def generate_synthetic_tour(
    num_keyframes: int = 24,
    arena_size_m: float = 2.4,
) -> Tuple[List[KeyframeData], List[np.ndarray]]:
    """
    Generate synthetic circular exploration trajectory with textured floor and colored walls.

    Enables testing and verifying coordinate lifting, depth calibration, and 3DGS training.

    :param num_keyframes: Number of viewpoints to synthesize.
    :param arena_size_m: Arena bounding dimension in meters.
    :return: (keyframes_list, images_rgb_list)
    """
    keyframes = []
    images = []

    radius = 0.70
    width, height = 640, 480

    for i in range(num_keyframes):
        angle = (2.0 * np.pi * i) / num_keyframes
        x = float(radius * np.cos(angle))
        y = float(radius * np.sin(angle))
        yaw = float(angle + np.pi / 2.0)  # Tangent heading facing forward

        # Synthesize ground and wall texture image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Floor (lower half) with checkered grid
        for v in range(240, 480):
            for u in range(0, 640):
                tile_u = (u // 40) % 2
                tile_v = (v // 30) % 2
                color = 180 if (tile_u ^ tile_v) else 140
                img[v, u] = [color, color, int(color * 0.9)]

        # Wall (upper half) with distinct visual features (color gradient and pillars)
        col_r = int(127 + 127 * np.sin(angle))
        col_g = int(127 + 127 * np.cos(angle))
        col_b = int(200)
        img[:240, :] = [col_r, col_g, col_b]

        # Draw visual fiducial markers on walls
        cv2.putText(
            img,
            f"KHEPERA IV ARENA POS {i}",
            (120, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        kf = KeyframeData(
            id=i,
            timestamp=float(i * 0.5),
            x=x,
            y=y,
            yaw=yaw,
            linear_vel=0.10,
            angular_vel=0.05,
            laplacian_var=120.0,
        )

        keyframes.append(kf)
        images.append(img)

    return keyframes, images


def main_cli() -> None:
    """Parse command line arguments and execute 3DGS reconstruction."""
    parser = argparse.ArgumentParser(
        description="Khepera IV Decoupled 3D Gaussian Splatting Reconstruction Pipeline"
    )
    parser.add_argument("--bag", type=str, default="", help="Path to ROS 2 rosbag (.db3 or .mcap)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/khepera_3dgs",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to Depth-Anything-V2 metric checkpoint (.pth)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Training optimization steps (default: 2000)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Voxel downsampling leaf size (default: 0.015m)",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default="supersplat_khepera.ply",
        help="Exported SuperSplat PLY filename",
    )
    parser.add_argument(
        "--synthetic-test",
        action="store_true",
        help="Run verification test on synthetic multi-view trajectory",
    )
    parser.add_argument(
        "--num-keyframes",
        type=int,
        default=24,
        help="Number of viewpoints to synthesize (default: 24)",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=3.0,
        help="Maximum ceiling height filter in meters (default 3.0m)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess saved keyframe photos from disk without Gazebo simulation",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="",
        help="Path to directory containing saved keyframes (images/ and manifest.json)",
    )

    args = parser.parse_args()

    pipeline = ReconstructionPipeline(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        voxel_size_m=args.voxel_size,
        num_iterations=args.iterations,
        max_height_m=args.max_height,
    )

    depth_maps = None

    if args.reprocess or args.load_dir:
        target_dir = args.load_dir if args.load_dir else args.output_dir
        LOGGER.info("Reprocessing saved photos from %s...", target_dir)
        try:
            keyframes, images, loaded_depths = SpatialDecimator.load_dataset(target_dir)
        except Exception as e:
            LOGGER.error("Failed to load dataset from %s: %s", target_dir, e)
            sys.exit(1)

        if not keyframes:
            LOGGER.error("No valid keyframes found in %s", target_dir)
            sys.exit(1)

        LOGGER.info(
            "Successfully loaded %d raw keyframe photos from %s", len(keyframes), target_dir
        )
        valid_depths = [d for d in loaded_depths if d is not None]
        if len(valid_depths) >= len(keyframes) * 0.75:
            last_valid = valid_depths[0]
            depth_maps = [
                d if d is not None else last_valid
                for d in loaded_depths
            ]

    elif args.bag:
        LOGGER.info("Processing bag %s...", args.bag)
        extractor = RosbagKeyframeExtractor(args.bag)
        samples = extractor.extract_synchronized_samples()
        if not samples:
            LOGGER.error("No valid samples extracted from bag.")
            sys.exit(1)

        decimator = SpatialDecimator()
        for t, img, x, y, yaw, v_lin, v_ang in samples:
            decimator.add_frame(t, img, x, y, yaw, linear_vel=v_lin, angular_vel=v_ang)

        keyframes = decimator.keyframes
        images = decimator.images
        LOGGER.info("Gating selected %d keyframes from bag.", len(keyframes))

    elif args.synthetic_test or not args.bag:
        LOGGER.info(
            "Generating synthetic verification tour (%d viewpoints)...", args.num_keyframes
        )
        keyframes, images = generate_synthetic_tour(num_keyframes=args.num_keyframes)

    ply_path = pipeline.run_reconstruction(
        keyframes=keyframes,
        images_rgb=images,
        ply_filename=args.export_name,
        depth_maps=depth_maps,
    )

    LOGGER.info("SUCCESS! Reconstruction complete: %s", ply_path)
    print("\n========================================================")
    print(" SuperSplat 3DGS Model Ready for Browser Visualization:")
    print(f" File: {ply_path}")
    print(" Test immediately in: https://superspl.at/editor")
    print("========================================================\n")


if __name__ == "__main__":
    main_cli()
