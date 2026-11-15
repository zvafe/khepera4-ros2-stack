# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-speed sparse correspondence tracking utilizing SuperPoint and LightGlue."""

from dataclasses import dataclass
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from lightglue import LightGlue, SuperPoint
    import torch

    LIGHTGLUE_AVAILABLE = True
except Exception:
    LIGHTGLUE_AVAILABLE = False

NOMINAL_MATCH_THRESHOLD: float = 0.2
MIN_KEYPOINTS_THRESHOLD: int = 4
FLOOR_CLIP_FRACTION: float = 0.15
TARGET_TOP_FRACTION: float = 1.0
MAX_COAST_FRAMES: int = 15
DEFAULT_DELTA_T: float = 1.0 / 30.0
RANSAC_REPROJ_THRESHOLD: float = 3.0
BIDIRECTIONAL_CONSISTENCY_THRESHOLD: float = 3.0


@dataclass(frozen=True)
class TrackingOutput:
    """Tracking result containing spatial position, bounding polygon, and confidence."""

    is_tracked: bool
    is_fallback: bool
    normalized_centroid: Tuple[float, float]
    area_fraction: float
    confidence: float
    keypoint_count: int
    polygon_points: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class GeometricVerificationResult:
    """Outcome of bidirectional RANSAC geometric consistency check."""

    is_valid: bool
    inlier_count: int
    mean_match_score: float
    transform_matrix: Optional[np.ndarray]
    inlier_ref_points: Tuple[Tuple[float, float], ...]
    inlier_curr_points: Tuple[Tuple[float, float], ...]


def enhance_contrast_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE on the L-channel of LAB space."""
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def create_floor_mask(
    height: int,
    width: int,
    clip_fraction: float = FLOOR_CLIP_FRACTION,
) -> np.ndarray:
    """Create binary mask excluding the lower floor region from feature detection."""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    cutoff_row = max(0, int(height * (1.0 - clip_fraction)))
    mask[cutoff_row:, :] = 0
    return mask


def create_target_top_mask(
    height: int,
    width: int,
    top_fraction: float = TARGET_TOP_FRACTION,
) -> np.ndarray:
    """Create mask restricting keypoints to top portion of target bounding box."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cutoff_row = min(height, max(1, int(height * top_fraction)))
    mask[:cutoff_row, :] = 255
    return mask


class KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking 2D bounding boxes."""

    def __init__(self, delta_t: float = DEFAULT_DELTA_T) -> None:
        """Initialize constant velocity state transition matrices."""
        self.delta_t = delta_t
        self.is_initialized: bool = False
        self.coast_count: int = 0
        self.state = np.zeros((6, 1), dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64) * 100.0

        self.transition_matrix = np.array(
            [
                [1.0, 0.0, delta_t, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, delta_t, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.process_noise = np.diag([1.0, 1.0, 1000.0, 1000.0, 1.0, 1.0])
        self.measurement_noise = np.diag([2.0, 2.0, 8.0, 8.0])

    def initialize(self, initial_box: Tuple[float, float, float, float]) -> None:
        """Initialize state from observed bounding box (center_x, center_y, width, height)."""
        x_c, y_c, width, height = initial_box
        self.state = np.array([[x_c], [y_c], [0.0], [0.0], [width], [height]], dtype=np.float64)
        self.covariance = np.diag([10.0, 10.0, 5000.0, 5000.0, 10.0, 10.0])
        self.is_initialized = True
        self.coast_count = 0

    def predict(self) -> Tuple[float, float, float, float]:
        """Project current state vector and error covariance forward in time."""
        if not self.is_initialized:
            return (0.0, 0.0, 0.0, 0.0)

        self.state = np.dot(self.transition_matrix, self.state)
        self.covariance = (
            np.dot(
                np.dot(self.transition_matrix, self.covariance),
                self.transition_matrix.T,
            )
            + self.process_noise
        )
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[4, 0]),
            float(self.state[5, 0]),
        )

    def update(self, measurement: Tuple[float, float, float, float]) -> None:
        """Correct projected state estimate using incoming box measurement."""
        if not self.is_initialized:
            self.initialize(measurement)
            return

        z = np.array(
            [[measurement[0]], [measurement[1]], [measurement[2]], [measurement[3]]],
            dtype=np.float64,
        )
        residual = z - np.dot(self.measurement_matrix, self.state)
        innovation_cov = (
            np.dot(
                np.dot(self.measurement_matrix, self.covariance),
                self.measurement_matrix.T,
            )
            + self.measurement_noise
        )
        kalman_gain = np.dot(
            np.dot(self.covariance, self.measurement_matrix.T),
            np.linalg.inv(innovation_cov),
        )
        self.state = self.state + np.dot(kalman_gain, residual)
        identity = np.eye(6, dtype=np.float64)
        cov_factor = identity - np.dot(kalman_gain, self.measurement_matrix)
        self.covariance = np.dot(cov_factor, self.covariance)
        self.coast_count = 0

    def increment_coast(self) -> int:
        """Increment count of consecutive predicted frames without measurements."""
        self.coast_count += 1
        return self.coast_count

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self.is_initialized = False
        self.coast_count = 0
        self.state.fill(0.0)
        self.covariance = np.eye(6, dtype=np.float64) * 100.0


def _check_cycle_consistency(
    pts_ref: np.ndarray,
    affine_fwd: np.ndarray,
    affine_bwd: np.ndarray,
) -> np.ndarray:
    """Evaluate forward-backward cycle consistency residual error."""
    pts_hom = np.hstack([pts_ref, np.ones((len(pts_ref), 1), dtype=np.float32)])
    pts_fwd = np.dot(affine_fwd, pts_hom.T).T
    pts_fwd_hom = np.hstack([pts_fwd, np.ones((len(pts_fwd), 1), dtype=np.float32)])
    pts_cycle = np.dot(affine_bwd, pts_fwd_hom.T).T
    cycle_errors = np.linalg.norm(pts_cycle - pts_ref, axis=1)
    return cycle_errors <= BIDIRECTIONAL_CONSISTENCY_THRESHOLD


def filter_bidirectional_ransac(
    pts_ref: np.ndarray,
    pts_curr: np.ndarray,
    scores: np.ndarray,
) -> GeometricVerificationResult:
    """Filter correspondences via planar homography RANSAC with affine fallback."""
    empty_res = GeometricVerificationResult(
        is_valid=False, inlier_count=0, mean_match_score=0.0,
        transform_matrix=None, inlier_ref_points=(), inlier_curr_points=(),
    )
    if len(pts_ref) < 3 or len(pts_curr) < 3 or np.sum(scores >= NOMINAL_MATCH_THRESHOLD) < 3:
        return empty_res

    mask_score = scores >= NOMINAL_MATCH_THRESHOLD
    pts_r, pts_c, scores_filt = pts_ref[mask_score], pts_curr[mask_score], scores[mask_score]
    if len(pts_r) < 3:
        return empty_res

    # 1. Primary: Planar homography with robust RANSAC (USAC_MAGSAC)
    if len(pts_r) >= 4:
        try:
            h_mat, mask = cv2.findHomography(
                pts_r, pts_c, cv2.USAC_MAGSAC, RANSAC_REPROJ_THRESHOLD
            )
            if h_mat is not None and mask is not None:
                inliers = mask.ravel().astype(bool)
                inlier_count = int(np.sum(inliers))
                if inlier_count >= 4:
                    valid_r = pts_r[inliers]
                    valid_c = pts_c[inliers]
                    return GeometricVerificationResult(
                        is_valid=True,
                        inlier_count=inlier_count,
                        mean_match_score=float(np.mean(scores_filt[inliers])),
                        transform_matrix=h_mat,
                        inlier_ref_points=tuple(tuple(float(c) for c in pt) for pt in valid_r),
                        inlier_curr_points=tuple(tuple(float(c) for c in pt) for pt in valid_c),
                    )
        except Exception as exc:
            logger.debug("findHomography failed (%s), falling back to affine.", exc)

    # 2. Fallback: 2D Partial Affine when < 4 points or homography fails
    af_fwd, in_fwd = cv2.estimateAffinePartial2D(
        pts_r, pts_c, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )
    if af_fwd is not None and in_fwd is not None:
        inliers = in_fwd.ravel().astype(bool)
        inlier_count = int(np.sum(inliers))
        if inlier_count >= 3:
            valid_r = pts_r[inliers]
            valid_c = pts_c[inliers]
            return GeometricVerificationResult(
                is_valid=True,
                inlier_count=inlier_count,
                mean_match_score=float(np.mean(scores_filt[inliers])),
                transform_matrix=af_fwd,
                inlier_ref_points=tuple(tuple(float(c) for c in pt) for pt in valid_r),
                inlier_curr_points=tuple(tuple(float(c) for c in pt) for pt in valid_c),
            )

    return empty_res


def image_to_tensor(image_bgr: np.ndarray, device: str) -> "torch.Tensor":
    """Convert BGR numpy image to normalized RGB PyTorch tensor with pinned memory."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose((2, 0, 1)))
    tensor_cpu = torch.from_numpy(chw)

    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        try:
            return tensor_cpu.pin_memory().to(target_device, non_blocking=True).float() / 255.0
        except Exception:
            return tensor_cpu.to(target_device).float() / 255.0
    return tensor_cpu.to(target_device).float() / 255.0


class SuperPointLightGlueTracker:
    """Correspondence tracker combining feature matching with Kalman filtering."""

    def __init__(
        self,
        match_threshold: float = NOMINAL_MATCH_THRESHOLD,
        min_keypoints: int = MIN_KEYPOINTS_THRESHOLD,
        device: str = "auto",
    ) -> None:
        """Initialize tracker with operational thresholds and Kalman filter."""
        self.match_threshold = match_threshold
        self.min_keypoints = min_keypoints
        if device == "auto":
            self.device = "cuda:0" if (LIGHTGLUE_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        if self.device == "cpu" and LIGHTGLUE_AVAILABLE:
            torch.set_num_threads(min(2, os.cpu_count() or 1))

        self._is_cuda: bool = False
        if LIGHTGLUE_AVAILABLE and torch.cuda.is_available():
            self._is_cuda = torch.device(self.device).type == "cuda"

        self.kalman = KalmanBoxTracker()
        self._sift = cv2.SIFT_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._sp_extractor = None
        self._lg_matcher = None

        if LIGHTGLUE_AVAILABLE:
            try:
                dev = torch.device(self.device)
                self._sp_extractor = SuperPoint(max_num_keypoints=512).eval().to(dev)
                self._lg_matcher = LightGlue(features="superpoint").eval().to(dev)
                logger.info("Initialized SuperPoint + LightGlue on %s", dev)
            except Exception as exc:
                logger.warning("LightGlue initialization failed (%s). Using SIFT.", exc)

        self._ref_sp_feats: Optional[dict] = None
        self._ref_keypoints: Optional[np.ndarray] = None
        self._ref_descriptors: Optional[np.ndarray] = None
        self._ref_size: Tuple[float, float] = (0.0, 0.0)
        self._last_confidence: float = 0.0
        self._ref_template_bgr: Optional[np.ndarray] = None
        self.has_reference: bool = False

    def reset(self) -> None:
        """Clear reference template and reset tracking state."""
        self._ref_sp_feats = None
        self._ref_keypoints = None
        self._ref_descriptors = None
        self._ref_template_bgr = None
        self._ref_size = (0.0, 0.0)
        self._last_confidence = 0.0
        self.has_reference = False
        self.kalman.reset()

    def set_reference_target(self, template_bgr: np.ndarray) -> bool:
        """Extract reference keypoints or template from candidate target image."""
        if template_bgr is None or template_bgr.size == 0:
            return False

        h, w = template_bgr.shape[:2]
        if h < 8 or w < 8:
            return False

        cutoff = max(1, int(h * TARGET_TOP_FRACTION))
        top_crop = template_bgr[:cutoff, :]
        if np.count_nonzero(top_crop) == 0:
            logger.warning("Target reference rejected: no visual pixels in upper region.")
            return False

        self._ref_template_bgr = template_bgr.copy()
        self._ref_size = (float(w), float(h))
        self._ref_sp_feats = None

        # 1. Primary GPU Path: SuperPoint keypoint and descriptor extraction
        if LIGHTGLUE_AVAILABLE and self._sp_extractor is not None:
            try:
                enhanced = enhance_contrast_clahe(template_bgr)
                t_tensor = image_to_tensor(enhanced, self.device)
                with torch.inference_mode():
                    if self._is_cuda:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            feats = self._sp_extractor.extract(t_tensor)
                    else:
                        feats = self._sp_extractor.extract(t_tensor)
                    kps = feats["keypoints"][0]
                    cutoff_row = float(h * TARGET_TOP_FRACTION)
                    valid_mask = kps[:, 1] < cutoff_row
                    valid_count = int(valid_mask.sum().item())
                    if valid_count >= self.min_keypoints:
                        self._ref_sp_feats = {
                            "keypoints": feats["keypoints"][:, valid_mask, :],
                            "descriptors": feats["descriptors"][:, valid_mask, :],
                            "keypoint_scores": feats["keypoint_scores"][:, valid_mask],
                            "image_size": feats["image_size"],
                        }
                        self._ref_keypoints = kps[valid_mask].cpu().numpy().astype(np.float32)
                        self.has_reference = True
                        self.kalman.reset()
                        self._last_confidence = 1.0
                        logger.info(
                            "Locked target using GPU SuperPoint on %s (%d keypoints).",
                            self.device,
                            valid_count,
                        )
            except torch.cuda.OutOfMemoryError as oom_exc:
                logger.error(
                    "CUDA OOM during SuperPoint reference lock: %s. Falling back to SIFT.", oom_exc
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._ref_sp_feats = None
            except Exception as exc:
                logger.warning(
                    "GPU SuperPoint reference extraction failed (%s). Falling back to SIFT.", exc
                )
                self._ref_sp_feats = None

        # 2. Extract CPU SIFT descriptors (for secondary fallback or standalone)
        enhanced = enhance_contrast_clahe(template_bgr)
        target_mask = create_target_top_mask(h, w, TARGET_TOP_FRACTION)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        kps_sift, descs_sift = self._sift.detectAndCompute(gray, target_mask)

        if (
            kps_sift is not None
            and len(kps_sift) >= self.min_keypoints
            and descs_sift is not None
        ):
            self._ref_keypoints = np.array([kp.pt for kp in kps_sift], dtype=np.float32)
            self._ref_descriptors = descs_sift
            self.has_reference = True
            self.kalman.reset()
            self._last_confidence = 1.0
            return True

        if self._ref_sp_feats is not None:
            return True

        logger.info("Textureless target: engaging template correlation tracking fallback.")
        self._ref_keypoints = None
        self._ref_descriptors = None
        self.has_reference = True
        self.kalman.reset()
        self._last_confidence = 0.90
        return True

    def track(self, frame_bgr: np.ndarray) -> TrackingOutput:
        """Track reference target across incoming camera frame."""
        if frame_bgr is None or frame_bgr.size == 0 or not self.has_reference:
            return self._build_empty_output()

        h_frame, w_frame = frame_bgr.shape[:2]

        # 1. Primary GPU Path: SuperPoint + LightGlue Tensor Matching
        if (
            LIGHTGLUE_AVAILABLE
            and self._sp_extractor is not None
            and self._lg_matcher is not None
            and self._ref_sp_feats is not None
        ):
            try:
                enhanced = enhance_contrast_clahe(frame_bgr)
                t_frame = image_to_tensor(enhanced, self.device)
                with torch.inference_mode():
                    if self._is_cuda:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            curr_feats = self._sp_extractor.extract(t_frame)
                    else:
                        curr_feats = self._sp_extractor.extract(t_frame)

                    kps_curr = curr_feats["keypoints"][0]
                    cutoff_floor = float(h_frame * (1.0 - FLOOR_CLIP_FRACTION))
                    valid_mask = kps_curr[:, 1] < cutoff_floor
                    if int(valid_mask.sum().item()) >= self.min_keypoints:
                        filt_curr_feats = {
                            "keypoints": curr_feats["keypoints"][:, valid_mask, :],
                            "descriptors": curr_feats["descriptors"][:, valid_mask, :],
                            "keypoint_scores": curr_feats["keypoint_scores"][:, valid_mask],
                            "image_size": curr_feats["image_size"],
                        }
                        if self._is_cuda:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                matches_dict = self._lg_matcher(
                                    {"image0": self._ref_sp_feats, "image1": filt_curr_feats}
                                )
                        else:
                            matches_dict = self._lg_matcher(
                                {"image0": self._ref_sp_feats, "image1": filt_curr_feats}
                            )
                        match_indices = matches_dict["matches"][0]
                        if len(match_indices) >= self.min_keypoints:
                            scores = matches_dict["scores"][0].cpu().numpy().astype(np.float32)
                            pts_r = (
                                self._ref_sp_feats["keypoints"][0][match_indices[:, 0]]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            pts_c = (
                                filt_curr_feats["keypoints"][0][match_indices[:, 1]]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            verif = filter_bidirectional_ransac(pts_r, pts_c, scores)
                            if verif.is_valid and verif.inlier_count >= self.min_keypoints:
                                return self._build_tracked_output(verif, w_frame, h_frame)
            except torch.cuda.OutOfMemoryError as oom_exc:
                logger.error(
                    "CUDA OOM in SuperPoint/LightGlue tracking: %s. Falling back to SIFT.", oom_exc
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning("GPU SuperPoint/LightGlue track failed (%s). Falling back.", exc)

        # 2. Secondary CPU Path: SIFT + BFMatcher Fallback
        if self._ref_descriptors is not None and self._ref_keypoints is not None:
            enhanced = enhance_contrast_clahe(frame_bgr)
            floor_mask = create_floor_mask(h_frame, w_frame, FLOOR_CLIP_FRACTION)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            kps, descs = self._sift.detectAndCompute(gray, floor_mask)

            if kps is not None and len(kps) >= self.min_keypoints and descs is not None:
                pts_curr = np.array([kp.pt for kp in kps], dtype=np.float32)
                matched_r, matched_c, scores = self._match_descriptors(
                    self._ref_descriptors, descs, pts_curr
                )
                verif = filter_bidirectional_ransac(matched_r, matched_c, scores)
                if verif.is_valid and verif.inlier_count >= self.min_keypoints:
                    return self._build_tracked_output(verif, w_frame, h_frame)

        # 3. Tertiary Path: Template Correlation Fallback
        if self._ref_template_bgr is not None:
            output = self._track_with_template_correlation(frame_bgr, w_frame, h_frame)
            if output.is_tracked:
                return output

        return self._handle_tracking_failure(w_frame, h_frame)

    def _track_with_template_correlation(
        self,
        frame_bgr: np.ndarray,
        width: int,
        height: int,
    ) -> TrackingOutput:
        """Track textureless target using multi-scale color template correlation."""
        tpl = self._ref_template_bgr
        th, tw = tpl.shape[:2]
        if th >= height or tw >= width or th < 4 or tw < 4:
            return self._build_empty_output()

        best_val = -1.0
        best_loc = (0, 0)
        best_size = (tw, th)

        for scale in (0.10, 0.14, 0.18, 0.25, 0.35, 0.50, 0.70, 0.85, 1.0, 1.15):
            sw = int(tw * scale)
            sh = int(th * scale)
            if sw >= width or sh >= height or sw < 6 or sh < 6:
                continue
            scaled_tpl = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_LINEAR)
            res = cv2.matchTemplate(frame_bgr, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = float(max_val)
                best_loc = max_loc
                best_size = (sw, sh)

        if best_val < 0.40:
            return self._build_empty_output()

        sw, sh = best_size
        c_x = float(best_loc[0] + sw / 2.0)
        c_y = float(best_loc[1] + sh / 2.0)

        self.kalman.predict()
        self.kalman.update((c_x, c_y, float(sw), float(sh)))

        norm_x = float(np.clip((c_x - width / 2.0) / (width / 2.0), -1.0, 1.0))
        norm_y = float(np.clip((c_y - height / 2.0) / (height / 2.0), -1.0, 1.0))
        area_frac = float(np.clip((sw * sh) / (width * height), 0.0, 1.0))
        conf = float(np.clip(best_val, 0.0, 1.0))

        half_w, half_h = sw / 2.0, sh / 2.0
        corners = (
            (c_x - half_w, c_y - half_h),
            (c_x + half_w, c_y - half_h),
            (c_x + half_w, c_y + half_h),
            (c_x - half_w, c_y + half_h),
        )

        self._last_confidence = conf
        return TrackingOutput(
            is_tracked=True,
            is_fallback=False,
            normalized_centroid=(norm_x, norm_y),
            area_fraction=area_frac,
            confidence=conf,
            keypoint_count=max(self.min_keypoints, int(conf * 15)),
            polygon_points=corners,
        )

    def _match_descriptors(
        self,
        ref_descs: np.ndarray,
        curr_descs: np.ndarray,
        pts_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute matching pairs and normalized similarity scores using ratio test."""
        matches = self._matcher.knnMatch(ref_descs, curr_descs, k=2)
        pts_r, pts_c, scores = [], [], []

        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < 0.8 * n.distance:
                if (self._ref_keypoints is not None and
                        m.queryIdx < len(self._ref_keypoints) and
                        m.trainIdx < len(pts_curr)):
                    pts_r.append(self._ref_keypoints[m.queryIdx])
                    pts_c.append(pts_curr[m.trainIdx])
                    score = max(0.0, min(1.0, 1.0 - (m.distance / 400.0)))
                    scores.append(score)

        if not pts_r:
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_pts, empty_pts.copy(), np.zeros((0,), dtype=np.float32)

        return (
            np.array(pts_r, dtype=np.float32),
            np.array(pts_c, dtype=np.float32),
            np.array(scores, dtype=np.float32),
        )

    def _build_tracked_output(
        self,
        verif: GeometricVerificationResult,
        width: int,
        height: int,
    ) -> TrackingOutput:
        """Construct tracking output from verified geometric transform."""
        w_ref, h_ref = self._ref_size
        ref_corners = np.array(
            [[0.0, 0.0], [w_ref, 0.0], [w_ref, h_ref], [0.0, h_ref]],
            dtype=np.float32,
        )
        if verif.transform_matrix is not None and verif.transform_matrix.shape == (3, 3):
            projected = cv2.perspectiveTransform(
                ref_corners.reshape(-1, 1, 2), verif.transform_matrix
            )
            curr_corners = projected.reshape(-1, 2)
        else:
            hom_corners = np.hstack([ref_corners, np.ones((4, 1), dtype=np.float32)])
            curr_corners = np.dot(verif.transform_matrix, hom_corners.T).T

        if not np.all(np.isfinite(curr_corners)):
            return self._handle_tracking_failure(width, height)

        c_x = float(np.mean(curr_corners[:, 0]))
        c_y = float(np.mean(curr_corners[:, 1]))
        w_box = float(np.max(curr_corners[:, 0]) - np.min(curr_corners[:, 0]))
        h_box = float(np.max(curr_corners[:, 1]) - np.min(curr_corners[:, 1]))

        self.kalman.predict()
        self.kalman.update((c_x, c_y, w_box, h_box))

        norm_x = float(np.clip((c_x - width / 2.0) / (width / 2.0), -1.0, 1.0))
        norm_y = float(np.clip((c_y - height / 2.0) / (height / 2.0), -1.0, 1.0))
        area_frac = float(np.clip((w_box * h_box) / (width * height), 0.0, 1.0))
        conf = float(
            np.clip(
                (verif.inlier_count / max(self.min_keypoints, 10)) * verif.mean_match_score,
                0.0,
                1.0,
            )
        )
        self._last_confidence = conf

        corners_tuple = tuple(tuple(float(v) for v in pt) for pt in curr_corners)
        return TrackingOutput(
            is_tracked=True,
            is_fallback=False,
            normalized_centroid=(norm_x, norm_y),
            area_fraction=area_frac,
            confidence=conf,
            keypoint_count=verif.inlier_count,
            polygon_points=corners_tuple,
        )

    def _handle_tracking_failure(self, width: int, height: int) -> TrackingOutput:
        """Execute Kalman prediction fallback when correspondences drop below minimum."""
        if self.kalman.is_initialized and self.kalman.coast_count < MAX_COAST_FRAMES:
            pred_box = self.kalman.predict()
            self.kalman.increment_coast()
            c_x, c_y, w_box, h_box = pred_box

            norm_x = float(np.clip((c_x - width / 2.0) / (width / 2.0), -1.0, 1.0))
            norm_y = float(np.clip((c_y - height / 2.0) / (height / 2.0), -1.0, 1.0))
            area_frac = float(np.clip((w_box * h_box) / (width * height), 0.0, 1.0))

            conf = float(max(0.05, self._last_confidence * (0.85**self.kalman.coast_count)))
            half_w, half_h = w_box / 2.0, h_box / 2.0
            corners = (
                (c_x - half_w, c_y - half_h),
                (c_x + half_w, c_y - half_h),
                (c_x + half_w, c_y + half_h),
                (c_x - half_w, c_y + half_h),
            )
            return TrackingOutput(
                is_tracked=True,
                is_fallback=True,
                normalized_centroid=(norm_x, norm_y),
                area_fraction=area_frac,
                confidence=conf,
                keypoint_count=0,
                polygon_points=corners,
            )

        return self._build_empty_output()

    def _build_empty_output(self) -> TrackingOutput:
        """Generate default empty tracking outcome."""
        return TrackingOutput(
            is_tracked=False,
            is_fallback=False,
            normalized_centroid=(0.0, 0.0),
            area_fraction=0.0,
            confidence=0.0,
            keypoint_count=0,
            polygon_points=(),
        )
