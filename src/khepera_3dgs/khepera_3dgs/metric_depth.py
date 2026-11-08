"""Metric depth estimation and ground-plane calibration using Depth-Anything-V2."""

import logging
import os
import sys
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger("khepera_3dgs.metric_depth")


class MetricDepthEstimator:
    """
    Infers metric depth using Depth-Anything-V2-Metric (Hypersim) in FP16 on CUDA.

    Calibrates scale using the Khepera IV physical ground clearance (h_floor = 0.0283 m).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        encoder: str = "vits",
        max_depth: float = 20.0,
        camera_height_m: float = 0.0283,
        focal_length_y: float = 554.256258,
        principal_point_y: float = 240.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        enable_floor_scale: bool = False,
    ) -> None:
        """
        Initialize depth model and calibration geometry.

        :param checkpoint_path: Path to Depth-Anything-V2 metric checkpoint (.pth).
        :param encoder: Architecture variant ('vits', 'vitb', 'vitl').
        :param max_depth: Maximum metric range in meters.
        :param camera_height_m: Ground clearance height h_floor (0.0283m for Khepera IV).
        :param focal_length_y: Camera intrinsic fy.
        :param principal_point_y: Camera intrinsic cy.
        :param device: Inference device ('cuda' or 'cpu').
        :param use_fp16: Enable FP16 half-precision for speed.
        :param enable_floor_scale: Enable floor ROI scale calibration (disabled by default
                                   for Depth-Anything-V2 Metric to preserve true room scale).
        """
        self.encoder = encoder
        self.max_depth = float(max_depth)
        self.h_floor = float(camera_height_m)
        self.fy = float(focal_length_y)
        self.cy = float(principal_point_y)
        self.enable_floor_scale = bool(enable_floor_scale)
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")

        self.model: Optional[torch.nn.Module] = None
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str]) -> None:
        """Load Depth-Anything-V2 model with checkpoint."""
        if checkpoint_path is None:
            default_ckpt = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../third_party/Depth-Anything-V2/checkpoints/"
                f"depth_anything_v2_metric_hypersim_{self.encoder}.pth",
            )
            if os.path.exists(default_ckpt):
                checkpoint_path = os.path.abspath(default_ckpt)

        # Attempt to import DepthAnythingV2 from third_party
        search_paths = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../third_party/Depth-Anything-V2/metric_depth",
            ),
            "/home/andy/Desktop/khepera4_ws/third_party/Depth-Anything-V2/metric_depth",
        ]
        for p in search_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.append(os.path.abspath(p))

        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }
            cfg = model_configs.get(self.encoder, model_configs["vits"])
            self.model = DepthAnythingV2(**{**cfg, "max_depth": self.max_depth})

            if checkpoint_path and os.path.exists(checkpoint_path):
                LOGGER.info("Loading Depth-Anything-V2 weights from %s", checkpoint_path)
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
            else:
                LOGGER.warning("No checkpoint found at %s. Model uninitialized.", checkpoint_path)

            self.model = self.model.to(self.device).eval()
            if self.use_fp16:
                self.model = self.model.half()
            LOGGER.info("DepthAnythingV2 initialized on %s (FP16=%s)", self.device, self.use_fp16)
        except Exception as exc:
            LOGGER.warning("Could not initialize DepthAnythingV2: %s", exc)
            self.model = None

    @torch.no_grad()
    def estimate_raw_depth(self, image_rgb: np.ndarray, input_size: int = 518) -> np.ndarray:
        """
        Predict uncalibrated metric depth map in meters.

        :param image_rgb: RGB input image (H, W, 3) uint8.
        :param input_size: Spatial dimension for neural inference.
        :return: (H, W) float32 depth map in meters.
        """
        if self.model is None:
            # Fallback planar estimation for mock or unequipped environments
            h, w = image_rgb.shape[:2]
            v_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
            delta_v = np.maximum(v_coords - self.cy, 1.0)
            ground_z = (self.fy * self.h_floor) / delta_v
            depth = np.tile(ground_z, (1, w))
            return np.clip(depth, 0.05, 5.0)

        raw_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Custom FP16 inference pipeline avoiding extra conversions
        h, w = raw_bgr.shape[:2]
        from torchvision.transforms import Compose
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        norm_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
        tensor_img = transform({"image": norm_rgb})["image"]
        tensor = torch.from_numpy(tensor_img).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()

        pred = self.model(tensor)
        pred = torch.nn.functional.interpolate(
            pred[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
        return pred.float().cpu().numpy()

    def estimate_raw_depth_batch(
        self,
        images_rgb: List[np.ndarray],
        input_size: int = 518,
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        """
        Predict uncalibrated metric depth maps in batches with GPU preprocessing.

        :param images_rgb: List of (H, W, 3) uint8 RGB images.
        :param input_size: Spatial dimension for neural inference.
        :param batch_size: Chunk size for GPU batching (default 4).
        :return: List of (H, W) float32 metric depth maps in meters.
        """
        if not images_rgb:
            return []

        if self.model is None:
            return [self.estimate_raw_depth(img, input_size) for img in images_rgb]

        h, w = images_rgb[0].shape[:2]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        depth_results: List[np.ndarray] = []
        n_total = len(images_rgb)

        for start_idx in range(0, n_total, batch_size):
            chunk = images_rgb[start_idx : start_idx + batch_size]
            t_chunk = [
                torch.from_numpy(img).permute(2, 0, 1).to(self.device).float() / 255.0
                for img in chunk
            ]
            batch = torch.stack(t_chunk, dim=0)
            batch = (batch - mean) / std
            batch = torch.nn.functional.interpolate(
                batch, (input_size, input_size), mode="bilinear", align_corners=False
            )
            if self.use_fp16:
                batch = batch.half()

            with torch.no_grad():
                preds = self.model(batch)
                preds = torch.nn.functional.interpolate(
                    preds[:, None], (h, w), mode="bilinear", align_corners=True
                )[:, 0]

            preds_np = preds.float().cpu().numpy()
            for idx in range(len(chunk)):
                depth_results.append(preds_np[idx])

        return depth_results

    def calibrate_ground_plane(
        self,
        raw_depth: np.ndarray,
        bottom_ratio: float = 0.20,
        center_span_ratio: float = 0.60,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Calibrate metric depth scale using ground clearance of Khepera IV.

        Extracts the bottom ROI corresponding to the immediate floor plane in front of the robot:
        scale = h_floor / z_pred_floor (or expected geometric floor distance)
        D_calibrated = scale * D_raw

        :param raw_depth: (H, W) raw metric depth in meters.
        :param bottom_ratio: Vertical fraction from image bottom (default 0.20 = 20%).
        :param center_span_ratio: Central horizontal fraction to avoid side obstacles.
        :return: (calibrated_depth, scale_factor, median_predicted_floor_depth)
        """
        h, w = raw_depth.shape[:2]
        v_start = int(h * (1.0 - bottom_ratio))
        u_margin = int(w * (1.0 - center_span_ratio) / 2.0)
        u_end = w - u_margin

        floor_roi = raw_depth[v_start:h, u_margin:u_end]
        valid_floor = floor_roi[(floor_roi > 0.02) & np.isfinite(floor_roi)]

        if len(valid_floor) == 0:
            scale_factor = 1.0
            z_pred_floor = self.h_floor
            return raw_depth.copy(), scale_factor, z_pred_floor

        z_pred_floor = float(np.median(valid_floor))

        if not self.enable_floor_scale:
            # Depth-Anything-V2 Metric Hypersim is already trained for metric outputs.
            # Extreme grazing angles with 2.8cm ground clearance fall outside near-plane bounds.
            return raw_depth.copy(), 1.0, z_pred_floor

        # Expected optical distance along Z for points at row v on a flat plane Z_world = 0:
        # v - cy = fy * (h_floor / Z) -> Z_expected = fy * h_floor / (v - cy)
        v_indices = np.arange(v_start, h, dtype=np.float32)
        v_diff = np.maximum(v_indices - self.cy, 1.0)
        expected_floor_z = float(np.median((self.fy * self.h_floor) / v_diff))

        # Safeguard: if expected floor distance is <0.25m (low clearance puck robot),
        # standard monocular depth networks cannot resolve 7cm depth accurately.
        if expected_floor_z < 0.25:
            LOGGER.warning(
                "Ground clearance h_floor=%.4fm is too low for stable floor ROI scaling "
                "(expected Z=%.3fm < 0.25m). Preserving native metric depth scale.",
                self.h_floor,
                expected_floor_z,
            )
            return raw_depth.copy(), 1.0, z_pred_floor

        if expected_floor_z > 0.0 and z_pred_floor > 0.0:
            computed_scale = expected_floor_z / z_pred_floor
            # Constrain fine-tuning scale factor within safe bounded window
            scale_factor = float(np.clip(computed_scale, 0.75, 1.35))
        else:
            scale_factor = 1.0

        calibrated_depth = np.clip(raw_depth * scale_factor, 0.02, self.max_depth)
        return calibrated_depth.astype(np.float32), scale_factor, z_pred_floor

    def process_frame(
        self,
        image_rgb: np.ndarray,
        output_dir: Optional[str] = None,
        frame_id: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Run end-to-end depth estimation, ground plane calibration, and save millimeter PNG / NPY.

        :param image_rgb: (H, W, 3) RGB uint8 image.
        :param output_dir: Directory to save depth artifacts.
        :param frame_id: Frame index for naming.
        :return: (calibrated_depth_meters, scale_factor)
        """
        raw_depth = self.estimate_raw_depth(image_rgb)
        calibrated_depth, scale_factor, _ = self.calibrate_ground_plane(raw_depth)

        if output_dir is not None:
            depth_dir = os.path.join(output_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)

            # 16-bit millimeter PNG: d_mm = round(D * 1000)
            depth_mm = np.clip(np.round(calibrated_depth * 1000.0), 0, 65535).astype(np.uint16)
            png_path = os.path.join(depth_dir, f"depth_{frame_id:04d}.png")
            cv2.imwrite(png_path, depth_mm)

            # Raw float32 array
            npy_path = os.path.join(depth_dir, f"depth_{frame_id:04d}.npy")
            np.save(npy_path, calibrated_depth)

        return calibrated_depth, scale_factor

    def process_batch(
        self,
        images_rgb: List[np.ndarray],
        output_dir: Optional[str] = None,
        frame_ids: Optional[List[int]] = None,
        batch_size: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Run batched depth estimation and calibration across multiple keyframes.

        :param images_rgb: List of (H, W, 3) RGB uint8 images.
        :param output_dir: Optional directory to save depth artifacts.
        :param frame_ids: Optional list of integer IDs for artifact naming.
        :param batch_size: Inference batch size.
        :param progress_callback: Optional callback fn(completed_count, total_count).
        :return: List of (H, W) calibrated float32 depth maps in meters.
        """
        n_total = len(images_rgb)
        if n_total == 0:
            return []

        if frame_ids is None:
            frame_ids = list(range(n_total))

        depth_dir = None
        if output_dir is not None:
            depth_dir = os.path.join(output_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)

        calibrated_depths: List[np.ndarray] = []
        for start_idx in range(0, n_total, batch_size):
            chunk_imgs = images_rgb[start_idx : start_idx + batch_size]
            chunk_ids = frame_ids[start_idx : start_idx + batch_size]
            raw_chunk = self.estimate_raw_depth_batch(chunk_imgs, batch_size=batch_size)

            for raw_depth, f_id in zip(raw_chunk, chunk_ids):
                calib_depth, _, _ = self.calibrate_ground_plane(raw_depth)
                calibrated_depths.append(calib_depth)

                if depth_dir is not None:
                    depth_mm = np.clip(np.round(calib_depth * 1000.0), 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_dir, f"depth_{f_id:04d}.png"), depth_mm)
                    np.save(os.path.join(depth_dir, f"depth_{f_id:04d}.npy"), calib_depth)

            if progress_callback is not None:
                progress_callback(len(calibrated_depths), n_total)

        return calibrated_depths
