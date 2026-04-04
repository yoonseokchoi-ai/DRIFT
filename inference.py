"""
DRIFT Inference Script
======================

Inference for DRIFT (Difficulty-aware Rectified Flows for Through-plane MRI SR):
  Stage 1: APN - Anatomical Projection Network (Sec. 3.3)
  Stage 2: Rectified Flow with PAD-based AIS adaptive stepping (Sec. 3.6)

Features:
- PAD (Physics-Aware Difficulty) based adaptive ODE stepping
- Protocol conditioning with τ = 1/T encoding
- Velocity and CETA output saving for paper figures
- Single volume or batch evaluation modes

Usage:
    # Single volume inference
    python inference/inference_drift_2d.py \
        --stage1-ckpt /path/to/stage1.ckpt \
        --stage2-ckpt /path/to/stage2.ckpt \
        --input /path/to/lr_volume.nii.gz \
        --output-dir /path/to/output \
        --t-lr 4.0 --t-hr 0.7

    # Stage 2 only (if input is already coarse prediction)
    python inference/inference_drift_2d.py \
        --stage2-ckpt /path/to/stage2.ckpt \
        --input /path/to/coarse.nii.gz \
        --output-dir /path/to/output \
        --stage2-only

    # With velocity output saving (for paper figures)
    python inference/inference_drift_2d.py \
        --stage2-ckpt /path/to/stage2.ckpt \
        --input /path/to/volume.nii.gz \
        --output-dir /path/to/output \
        --save-velocity \
        --save-intermediate-steps

    # With CETA velocity comparison
    python inference/inference_drift_2d.py \
        --stage2-ckpt /path/to/stage2.ckpt \
        --input /path/to/volume.nii.gz \
        --output-dir /path/to/output \
        --save-ceta --ceta-alt-thickness 5.0
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import time

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add drift package root to path (must come before project root to avoid name conflicts)
current_dir = os.path.dirname(os.path.abspath(__file__))
drift_root = os.path.abspath(current_dir)
if drift_root not in sys.path:
    sys.path.insert(0, drift_root)

from models.drift_2d import Stage1APNLightning, Stage2RFLightning
from dataset.drift_2d_dataset import PrecomputedNPY2DDataset, DRIFT2DDataModuleV2

# Use skimage for metrics (more stable than torchmetrics)
try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available, using fallback metrics")


# =============================================================================
# In-house Real LR Dataset (Pre-degraded by MRI scanner)
# =============================================================================

class InhouseRealLRDataset:
    """
    In-house real LR dataset.
    LR slices are pre-degraded (nearest-upsampled from real thick-slice MRI).
    HR slices are from GT volume (FLAIR only, T2 has no GT).

    Returns dict compatible with run_batch_evaluation loop:
      - x_lr: (1, H, W) tensor in [-1, 1]
      - y_hr: (1, H, W) tensor in [-1, 1] or None
      - protocol: (2,) tensor [T_lr, T_hr]
      - filename, subject, plane, blur_axis_2d
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,  # e.g., 'flair_st4mm', 't2_st7mm'
        max_slices: Optional[int] = None,
    ):
        self.data_path = Path(data_dir) / dataset_name
        self.lr_dir = self.data_path / "lr"
        self.hr_dir = self.data_path / "hr"
        self.has_gt = self.hr_dir.exists()

        # Load metadata
        meta_path = self.data_path / "metadata.json"
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.thick_mm = self.meta['thick_mm']
        self.inplane_mm = self.meta['inplane_mm']
        self.scale_factor = self.meta['scale_factor']
        self.contrast = self.meta['contrast']

        # Discover LR files
        self.lr_files = sorted([f for f in self.lr_dir.glob('*.npy')])
        if max_slices:
            self.lr_files = self.lr_files[:max_slices]

        # Build slice metadata lookup
        self.slice_meta = {}
        for s in self.meta.get('slices', []):
            self.slice_meta[s['filename']] = s

        print(f"InhouseRealLRDataset: {dataset_name}")
        print(f"  LR slices: {len(self.lr_files)}")
        print(f"  GT available: {self.has_gt}")
        print(f"  Protocol: T_lr={self.thick_mm}mm -> T_hr={self.inplane_mm}mm")
        print(f"  Scale factor: {self.scale_factor:.2f}x")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Dict:
        lr_path = self.lr_files[idx]
        filename = lr_path.name

        # Load LR slice (float16 [0,1] -> float32 [-1,1])
        lr_slice = np.load(lr_path).astype(np.float32)
        lr_min, lr_max = lr_slice.min(), lr_slice.max()
        if lr_max > lr_min:
            lr_slice = (lr_slice - lr_min) / (lr_max - lr_min)
        lr_slice = lr_slice * 2 - 1
        lr_tensor = torch.from_numpy(lr_slice).unsqueeze(0)  # (1, H, W)

        result = {
            'x_lr': lr_tensor,
            'filename': filename,
            'protocol': torch.tensor([self.thick_mm, self.inplane_mm]),
        }

        # Get blur axis from metadata
        meta = self.slice_meta.get(filename, {})
        result['blur_axis_2d'] = meta.get('blur_axis_2d', 1)
        result['plane'] = meta.get('plane', 'unknown')
        result['subject'] = meta.get('subject', 'unknown')

        # Load HR if available
        if self.has_gt:
            hr_path = self.hr_dir / filename
            if hr_path.exists():
                hr_slice = np.load(hr_path).astype(np.float32)
                hr_min, hr_max = hr_slice.min(), hr_slice.max()
                if hr_max > hr_min:
                    hr_slice = (hr_slice - hr_min) / (hr_max - hr_min)
                hr_slice = hr_slice * 2 - 1
                result['y_hr'] = torch.from_numpy(hr_slice).unsqueeze(0)
            else:
                result['y_hr'] = None
        else:
            result['y_hr'] = None

        return result


# =============================================================================
# Utility Functions
# =============================================================================

def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return data + affine."""
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: str):
    """Save numpy array as NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)
    print(f"Saved: {filepath}")


def normalize_volume(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize volume to [-1, 1] range."""
    data_min, data_max = data.min(), data.max()
    data_norm = (data - data_min) / (data_max - data_min + 1e-8)
    data_norm = data_norm * 2 - 1
    return data_norm, data_min, data_max


def denormalize_volume(data: np.ndarray, orig_min: float, orig_max: float) -> np.ndarray:
    """Denormalize volume from [-1, 1] back to original range."""
    data = (data + 1) / 2
    data = data * (orig_max - orig_min) + orig_min
    return data


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Compute PSNR using skimage or fallback."""
    try:
        if SKIMAGE_AVAILABLE:
            return skimage_psnr(target, pred, data_range=data_range)
        else:
            mse = np.mean((pred - target) ** 2)
            if mse == 0:
                return float('inf')
            return 10 * np.log10(data_range ** 2 / mse)
    except Exception as e:
        print(f"PSNR computation error: {e}")
        return float('nan')


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Compute SSIM using skimage or fallback."""
    if SKIMAGE_AVAILABLE:
        try:
            # For 2D images, skimage may need channel_axis=None
            if pred.ndim == 2:
                return skimage_ssim(target, pred, data_range=data_range)
            else:
                return skimage_ssim(target, pred, data_range=data_range, channel_axis=-1)
        except Exception as e:
            print(f"SSIM computation error: {e}")
            return float('nan')
    else:
        return 0.0  # Fallback: no SSIM without skimage


def compute_metrics(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> Dict[str, float]:
    """Compute PSNR and SSIM metrics."""
    return {
        'psnr': compute_psnr(pred, target, data_range),
        'ssim': compute_ssim(pred, target, data_range),
    }


# =============================================================================
# DRIFT 2D Inference Class
# =============================================================================

class DRIFT2DInference:
    """
    DRIFT 2D Inference Pipeline.

    Supports:
    - Stage 1 only (regression)
    - Stage 2 only (rectified flow on pre-computed coarse)
    - Full pipeline (Stage 1 -> Stage 2)
    - Velocity output saving for visualization
    - CETA velocity comparison for ablation
    """

    def __init__(
        self,
        stage1_ckpt: Optional[str] = None,
        stage2_ckpt: Optional[str] = None,
        device: torch.device = None,
        num_inference_steps: int = 8,
        use_adaptive_steps: bool = True,
        stage2_only: bool = False,
    ):
        """
        Initialize DRIFT inference.

        Args:
            stage1_ckpt: Path to Stage 1 checkpoint
            stage2_ckpt: Path to Stage 2 checkpoint
            device: Torch device
            num_inference_steps: Number of ODE steps (if not adaptive)
            use_adaptive_steps: Use PAD-based adaptive stepping
            stage2_only: Skip Stage 1, input is already coarse prediction
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_inference_steps = num_inference_steps
        self.use_adaptive_steps = use_adaptive_steps
        self.stage2_only = stage2_only

        # Load Stage 1 (optional if stage2_only)
        self.stage1 = None
        if stage1_ckpt is not None and not stage2_only:
            print(f"Loading Stage 1 from: {stage1_ckpt}")
            self.stage1 = Stage1APNLightning.load_from_checkpoint(
                stage1_ckpt, map_location=self.device, weights_only=False
            )
            self.stage1.eval()
            self.stage1.freeze()

        # Load Stage 2 (optional, or required if stage2_only)
        self.stage2 = None
        if stage2_ckpt is not None:
            print(f"Loading Stage 2 from: {stage2_ckpt}")
            self.stage2 = Stage2RFLightning.load_from_checkpoint(
                stage2_ckpt, map_location=self.device, weights_only=False
            )
            self.stage2.eval()
            self.stage2.freeze()
            # Override inference steps if specified
            if num_inference_steps != self.stage2.num_inference_steps:
                print(f"  Overriding num_inference_steps: {self.stage2.num_inference_steps} -> {num_inference_steps}")
            self.stage2.use_adaptive_steps = use_adaptive_steps

        # Mode info
        if stage2_only:
            print(f"Mode: Stage 2 only (Coarse -> Stage 2 -> HR)")
        elif self.stage2 is not None:
            print(f"Mode: Full pipeline (LR -> Stage 1 -> Stage 2 -> HR)")
        else:
            print(f"Mode: Stage 1 only (LR -> Stage 1 -> HR)")

        print(f"Device: {self.device}")
        print(f"Inference steps: {num_inference_steps} (adaptive={use_adaptive_steps})")

        # Current protocol (set per inference call)
        self.current_protocol = None

    def compute_pad(self, t_lr: float, t_hr: float) -> float:
        """Compute PAD (Physics-Aware Difficulty)."""
        return 1.0 - t_hr / max(t_lr, t_hr)

    def compute_adaptive_steps(self, t_lr: float, t_hr: float) -> int:
        """Compute adaptive step count based on PAD."""
        if self.stage2 is None:
            return self.num_inference_steps

        pad = self.compute_pad(t_lr, t_hr)
        max_steps = getattr(self.stage2, 'max_steps', 10)
        min_steps = getattr(self.stage2, 'min_steps', 2)

        steps = int(round(max_steps * pad))
        steps = max(min_steps, min(steps, max_steps))
        return steps

    @torch.no_grad()
    def run_stage1(
        self,
        x_lr: torch.Tensor,
        protocol: torch.Tensor,
    ) -> torch.Tensor:
        """Run Stage 1 regression."""
        if self.stage1 is None:
            return x_lr
        return self.stage1.model(x_lr, protocol)

    @torch.no_grad()
    def run_stage2_single_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        protocol: torch.Tensor,
    ) -> torch.Tensor:
        """Run single Stage 2 velocity prediction."""
        if self.stage2 is None:
            return torch.zeros_like(x)
        return self.stage2.rf_model(x, t, protocol)

    @torch.no_grad()
    def run_stage2_ode(
        self,
        y_coarse: torch.Tensor,
        protocol: torch.Tensor,
        num_steps: Optional[int] = None,
        return_intermediate: bool = False,
        return_velocities: bool = False,
    ) -> Dict:
        """
        Run Stage 2 ODE integration (Rectified Flow).

        Args:
            y_coarse: Coarse prediction from Stage 1
            protocol: [T_lr, T_hr] protocol tensor
            num_steps: Number of ODE steps (None = use adaptive)
            return_intermediate: Return intermediate states
            return_velocities: Return velocity at each step

        Returns:
            Dict with 'output', optionally 'intermediate', 'velocities', 'num_steps'
        """
        if self.stage2 is None:
            return {'output': y_coarse, 'num_steps': 0}

        # Determine number of steps
        if num_steps is None:
            if self.use_adaptive_steps:
                t_lr = protocol[0, 0].item()
                t_hr = protocol[0, 1].item()
                num_steps = self.compute_adaptive_steps(t_lr, t_hr)
            else:
                num_steps = self.num_inference_steps

        # ODE integration with Euler method
        dt = 1.0 / num_steps
        x = y_coarse.clone()

        intermediate = [x.clone()] if return_intermediate else None
        velocities = [] if return_velocities else None

        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.stage2.rf_model(x, t, protocol)
            x = x + v * dt

            if return_intermediate:
                intermediate.append(x.clone())
            if return_velocities:
                velocities.append(v.clone())

        result = {
            'output': x,
            'num_steps': num_steps,
        }

        if return_intermediate:
            result['intermediate'] = intermediate
        if return_velocities:
            result['velocities'] = velocities

        return result

    @torch.no_grad()
    def compute_ceta_velocities(
        self,
        y_coarse: torch.Tensor,
        y_coarse_alt: torch.Tensor,
        protocol: torch.Tensor,
        protocol_alt: torch.Tensor,
        t_eval: float = 0.0,
    ) -> Dict:
        """
        Compute CETA velocities for visualization.

        CETA (Consistent Endpoint Trajectory Alignment) ensures that
        different starting points (different T_lr) converge to the same endpoint.

        Args:
            y_coarse: Coarse prediction from T_lr
            y_coarse_alt: Coarse prediction from T_lr_alt
            protocol: [T_lr, T_hr]
            protocol_alt: [T_lr_alt, T_hr]
            t_eval: Time point for velocity evaluation

        Returns:
            Dict with velocities and estimated endpoints
        """
        if self.stage2 is None:
            return {}

        t = torch.full((y_coarse.shape[0],), t_eval, device=y_coarse.device)

        # Velocities from both starting points
        v = self.stage2.rf_model(y_coarse, t, protocol)
        v_alt = self.stage2.rf_model(y_coarse_alt, t, protocol_alt)

        # Estimated endpoints (one-step estimation)
        # In rectified flow: endpoint = start + v * (1 - t)
        remaining_time = 1.0 - t_eval
        endpoint_est = y_coarse + v * remaining_time
        endpoint_est_alt = y_coarse_alt + v_alt * remaining_time

        return {
            'velocity': v,
            'velocity_alt': v_alt,
            'endpoint_est': endpoint_est,
            'endpoint_est_alt': endpoint_est_alt,
            'endpoint_diff': (endpoint_est - endpoint_est_alt).abs(),
            't_eval': t_eval,
        }

    @torch.no_grad()
    def infer_slice(
        self,
        x_lr: torch.Tensor,
        t_lr: float,
        t_hr: float,
        num_steps: Optional[int] = None,
        return_intermediate: bool = False,
        return_velocities: bool = False,
        return_coarse: bool = False,
    ) -> Dict:
        """
        Run full inference on a single slice.

        Args:
            x_lr: (B, 1, H, W) LR input
            t_lr: LR slice thickness in mm
            t_hr: Target HR slice thickness in mm
            num_steps: Override number of ODE steps
            return_intermediate: Return intermediate ODE states
            return_velocities: Return velocity at each step
            return_coarse: Return Stage 1 coarse output

        Returns:
            Dict with results
        """
        x_lr = x_lr.to(self.device)
        B = x_lr.shape[0]

        # Protocol tensor
        protocol = torch.tensor([[t_lr, t_hr]], device=self.device).expand(B, -1)

        # Stage 1: Coarse prediction
        if self.stage2_only:
            y_coarse = x_lr  # Input is already coarse
        else:
            y_coarse = self.run_stage1(x_lr, protocol)

        # Stage 2: ODE refinement
        stage2_result = self.run_stage2_ode(
            y_coarse, protocol,
            num_steps=num_steps,
            return_intermediate=return_intermediate,
            return_velocities=return_velocities,
        )

        result = {
            'output': stage2_result['output'],
            'num_steps': stage2_result['num_steps'],
            't_lr': t_lr,
            't_hr': t_hr,
            'pad': self.compute_pad(t_lr, t_hr),
        }

        if return_coarse:
            result['coarse'] = y_coarse
        if return_intermediate:
            result['intermediate'] = stage2_result.get('intermediate')
        if return_velocities:
            result['velocities'] = stage2_result.get('velocities')

        return result

    @torch.no_grad()
    def infer_slice_sliding_window(
        self,
        x_lr: torch.Tensor,
        t_lr: float,
        t_hr: float,
        patch_size: int = 128,
        overlap: int = 32,
        num_steps: Optional[int] = None,
        return_coarse: bool = False,
        return_intermediate: bool = False,
        return_velocities: bool = False,
        batch_patches: int = 16,  # Number of patches to process in parallel
    ) -> Dict:
        """
        Run inference on a full slice using sliding window with overlap.

        OPTIMIZED: Processes multiple patches in parallel for better GPU utilization.

        This is useful when the model was trained on patches but you want
        to process full-resolution images.

        Args:
            x_lr: (1, 1, H, W) LR input (full resolution)
            t_lr: LR slice thickness in mm
            t_hr: Target HR slice thickness in mm
            patch_size: Size of each patch (e.g., 128)
            overlap: Overlap between adjacent patches (e.g., 32)
            num_steps: Override number of ODE steps
            return_coarse: Return Stage 1 coarse output
            return_intermediate: Return intermediate ODE states
            return_velocities: Return velocity at each step
            batch_patches: Number of patches to process in parallel (default: 16)

        Returns:
            Dict with aggregated results
        """
        x_lr = x_lr.float().to(self.device)
        _, _, H, W = x_lr.shape
        stride = patch_size - overlap

        # Compute number of patches needed
        n_patches_h = max(1, (H - patch_size) // stride + 1)
        n_patches_w = max(1, (W - patch_size) // stride + 1)

        # Adjust to cover the entire image
        # If image is smaller than patch_size, pad it
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)

        if pad_h > 0 or pad_w > 0:
            x_lr = F.pad(x_lr, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, H_padded, W_padded = x_lr.shape
        else:
            H_padded, W_padded = H, W

        # Recalculate patches after padding
        n_patches_h = max(1, (H_padded - patch_size) // stride + 1)
        n_patches_w = max(1, (W_padded - patch_size) // stride + 1)

        # Ensure we cover the entire image
        positions_h = [i * stride for i in range(n_patches_h)]
        positions_w = [i * stride for i in range(n_patches_w)]

        # Add final position if not covered
        if positions_h[-1] + patch_size < H_padded:
            positions_h.append(H_padded - patch_size)
        if positions_w[-1] + patch_size < W_padded:
            positions_w.append(W_padded - patch_size)

        # Create list of all patch positions
        all_positions = [(h, w) for h in positions_h for w in positions_w]
        num_patches = len(all_positions)

        # Determine number of steps for buffer allocation
        if num_steps is None:
            if self.use_adaptive_steps:
                expected_steps = self.compute_adaptive_steps(t_lr, t_hr)
            else:
                expected_steps = self.num_inference_steps
        else:
            expected_steps = num_steps

        # Initialize output and weight maps
        output_sum = torch.zeros((1, 1, H_padded, W_padded), device=self.device)
        weight_sum = torch.zeros((1, 1, H_padded, W_padded), device=self.device)

        if return_coarse:
            coarse_sum = torch.zeros((1, 1, H_padded, W_padded), device=self.device)

        # Initialize velocity accumulators (one per ODE step)
        if return_velocities:
            velocity_sums = [torch.zeros((1, 1, H_padded, W_padded), device=self.device)
                            for _ in range(expected_steps)]

        # Initialize intermediate state accumulators (num_steps + 1 states: initial + after each step)
        if return_intermediate:
            intermediate_sums = [torch.zeros((1, 1, H_padded, W_padded), device=self.device)
                                for _ in range(expected_steps + 1)]

        # Create Gaussian weight for smooth blending in overlap regions
        weight_patch = self._create_gaussian_weight(patch_size, device=self.device)

        all_steps = []

        # Process patches in batches for better GPU utilization
        for batch_start in range(0, num_patches, batch_patches):
            batch_end = min(batch_start + batch_patches, num_patches)
            batch_positions = all_positions[batch_start:batch_end]
            current_batch_size = len(batch_positions)

            # Extract all patches in this batch
            patches = torch.stack([
                x_lr[:, :, h:h+patch_size, w:w+patch_size].squeeze(0)
                for h, w in batch_positions
            ], dim=0)  # (batch_size, 1, patch_size, patch_size)

            # Protocol tensor for batch (ensure float32)
            protocol = torch.tensor([[t_lr, t_hr]], device=self.device, dtype=torch.float32).expand(current_batch_size, -1)

            # Run Stage 1 on batch
            if self.stage2_only:
                y_coarse_batch = patches
            else:
                y_coarse_batch = self.run_stage1(patches, protocol)

            # Run Stage 2 ODE on batch
            stage2_result = self._run_stage2_ode_batch(
                y_coarse_batch, protocol,
                num_steps=num_steps,
                return_intermediate=return_intermediate,
                return_velocities=return_velocities,
            )

            output_batch = stage2_result['output']
            actual_steps = stage2_result['num_steps']
            all_steps.append(actual_steps)

            # Scatter results back to output map with Gaussian weighting
            for idx, (h_pos, w_pos) in enumerate(batch_positions):
                # Accumulate output with Gaussian weighting
                output_sum[:, :, h_pos:h_pos+patch_size, w_pos:w_pos+patch_size] += output_batch[idx:idx+1] * weight_patch
                weight_sum[:, :, h_pos:h_pos+patch_size, w_pos:w_pos+patch_size] += weight_patch

                if return_coarse:
                    coarse_sum[:, :, h_pos:h_pos+patch_size, w_pos:w_pos+patch_size] += y_coarse_batch[idx:idx+1] * weight_patch

                # Accumulate velocities with Gaussian weighting
                if return_velocities and stage2_result.get('velocities'):
                    for step_idx, vel_batch in enumerate(stage2_result['velocities']):
                        if step_idx < len(velocity_sums):
                            velocity_sums[step_idx][:, :, h_pos:h_pos+patch_size, w_pos:w_pos+patch_size] += vel_batch[idx:idx+1] * weight_patch

                # Accumulate intermediate states with Gaussian weighting
                if return_intermediate and stage2_result.get('intermediate'):
                    for step_idx, inter_batch in enumerate(stage2_result['intermediate']):
                        if step_idx < len(intermediate_sums):
                            intermediate_sums[step_idx][:, :, h_pos:h_pos+patch_size, w_pos:w_pos+patch_size] += inter_batch[idx:idx+1] * weight_patch

        # Normalize by weights (before cropping to handle edges properly)
        output = output_sum / (weight_sum + 1e-8)

        if return_coarse:
            coarse = coarse_sum / (weight_sum + 1e-8)

        if return_velocities:
            velocities = [v / (weight_sum + 1e-8) for v in velocity_sums]

        if return_intermediate:
            intermediate = [s / (weight_sum + 1e-8) for s in intermediate_sums]

        # Remove padding (after normalization)
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]
            if return_coarse:
                coarse = coarse[:, :, :H, :W]
            if return_velocities:
                velocities = [v[:, :, :H, :W] for v in velocities]
            if return_intermediate:
                intermediate = [s[:, :, :H, :W] for s in intermediate]

        result = {
            'output': output,
            'num_steps': int(np.mean(all_steps)) if all_steps else 0,
            't_lr': t_lr,
            't_hr': t_hr,
            'pad': self.compute_pad(t_lr, t_hr),
            'num_patches': num_patches,
        }

        if return_coarse:
            result['coarse'] = coarse
        if return_velocities:
            result['velocities'] = velocities
        if return_intermediate:
            result['intermediate'] = intermediate

        return result

    @torch.no_grad()
    def _run_stage2_ode_batch(
        self,
        y_coarse: torch.Tensor,
        protocol: torch.Tensor,
        num_steps: Optional[int] = None,
        return_intermediate: bool = False,
        return_velocities: bool = False,
    ) -> Dict:
        """
        Run Stage 2 ODE integration on a BATCH of patches.

        This is optimized for processing multiple patches in parallel.

        Args:
            y_coarse: (B, 1, H, W) Coarse predictions from Stage 1
            protocol: (B, 2) Protocol tensors [T_lr, T_hr]
            num_steps: Number of ODE steps (None = use adaptive)
            return_intermediate: Return intermediate states
            return_velocities: Return velocity at each step

        Returns:
            Dict with 'output', optionally 'intermediate', 'velocities', 'num_steps'
        """
        if self.stage2 is None:
            return {'output': y_coarse, 'num_steps': 0}

        B = y_coarse.shape[0]

        # Determine number of steps (same for all patches in batch)
        if num_steps is None:
            if self.use_adaptive_steps:
                t_lr = protocol[0, 0].item()
                t_hr = protocol[0, 1].item()
                num_steps = self.compute_adaptive_steps(t_lr, t_hr)
            else:
                num_steps = self.num_inference_steps

        # ODE integration with Euler method
        dt = 1.0 / num_steps
        x = y_coarse.clone()

        intermediate = [x.clone()] if return_intermediate else None
        velocities = [] if return_velocities else None

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device)
            v = self.stage2.rf_model(x, t, protocol)
            x = x + v * dt

            if return_intermediate:
                intermediate.append(x.clone())
            if return_velocities:
                velocities.append(v.clone())

        result = {
            'output': x,
            'num_steps': num_steps,
        }

        if return_intermediate:
            result['intermediate'] = intermediate
        if return_velocities:
            result['velocities'] = velocities

        return result

    def _create_gaussian_weight(self, size: int, sigma: float = None, device: torch.device = None) -> torch.Tensor:
        """
        Create 2D Gaussian weight for smooth patch blending.

        Higher weight in center, lower at edges for seamless merging.
        """
        if sigma is None:
            sigma = size / 4.0  # Standard choice for smooth falloff

        coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
        gauss_1d = torch.exp(-coords**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)  # Outer product
        gauss_2d = gauss_2d / gauss_2d.max()  # Normalize to [0, 1]

        return gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    @torch.no_grad()
    def infer_volume(
        self,
        lr_volume: np.ndarray,
        t_lr: float,
        t_hr: float,
        slice_axis: int = 0,
        batch_size: int = 16,
        num_steps: Optional[int] = None,
        save_velocity: bool = False,
        save_intermediate: bool = False,
        use_sliding_window: bool = False,
        patch_size: int = 128,
        overlap: int = 32,
        batch_patches: int = 16,  # Number of patches to process in parallel
    ) -> Dict:
        """
        Run inference on full 3D volume (slice by slice).

        Args:
            lr_volume: (D, H, W) LR volume (normalized to [-1, 1])
            t_lr: LR slice thickness in mm
            t_hr: Target HR slice thickness in mm
            slice_axis: Axis to process (0=axial, 1=coronal, 2=sagittal)
            batch_size: Batch size for processing
            num_steps: Override number of ODE steps
            save_velocity: Save velocity outputs
            save_intermediate: Save intermediate ODE states
            use_sliding_window: Use sliding window for each slice (for patch-trained models)
            patch_size: Patch size for sliding window (default: 128)
            overlap: Overlap between patches (default: 32)
            batch_patches: Number of patches to process in parallel (default: 16)

        Returns:
            Dict with SR volume and optional velocity/intermediate data
        """
        # Get slice access functions based on axis
        if slice_axis == 0:
            num_slices = lr_volume.shape[0]
            get_slice = lambda v, i: v[i, :, :]
            set_slice = lambda v, i, s: v.__setitem__(i, s)
        elif slice_axis == 1:
            num_slices = lr_volume.shape[1]
            get_slice = lambda v, i: v[:, i, :]
            set_slice = lambda v, i, s: v.__setitem__((slice(None), i, slice(None)), s)
        else:
            num_slices = lr_volume.shape[2]
            get_slice = lambda v, i: v[:, :, i]
            set_slice = lambda v, i, s: v.__setitem__((slice(None), slice(None), i), s)

        pad = self.compute_pad(t_lr, t_hr)
        expected_steps = self.compute_adaptive_steps(t_lr, t_hr) if self.use_adaptive_steps else (num_steps or self.num_inference_steps)

        print(f"\nDRIFT Volume Inference:")
        print(f"  Input shape: {lr_volume.shape}")
        print(f"  Slice axis: {slice_axis}, Num slices: {num_slices}")
        print(f"  Protocol: T_lr={t_lr:.2f}mm -> T_hr={t_hr:.2f}mm")
        print(f"  PAD (difficulty): {pad:.3f}")
        print(f"  Expected steps: {expected_steps}")
        print(f"  Adaptive: {self.use_adaptive_steps}")
        if use_sliding_window:
            print(f"  Sliding window: patch_size={patch_size}, overlap={overlap}, batch_patches={batch_patches}")

        # Initialize output arrays
        sr_volume = np.zeros_like(lr_volume)

        if save_velocity:
            # Store final velocity for each slice
            velocity_volume = np.zeros_like(lr_volume)

        if save_intermediate:
            # Store intermediate states: (num_steps+1, D, H, W)
            intermediate_volumes = [np.zeros_like(lr_volume) for _ in range(expected_steps + 1)]

        all_steps = []
        total_time = 0

        # Sliding window mode: process one slice at a time
        if use_sliding_window:
            for i in tqdm(range(num_slices), desc="Processing (sliding window)"):
                slice_2d = get_slice(lr_volume, i)
                slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

                start_time = time.time()
                result = self.infer_slice_sliding_window(
                    slice_tensor, t_lr, t_hr,
                    patch_size=patch_size,
                    overlap=overlap,
                    num_steps=num_steps,
                    return_coarse=False,
                    return_intermediate=save_intermediate,
                    return_velocities=save_velocity,
                    batch_patches=batch_patches,
                )
                total_time += time.time() - start_time

                all_steps.append(result['num_steps'])

                # Store SR output
                sr_slice = result['output'].squeeze().cpu().numpy()
                set_slice(sr_volume, i, sr_slice)

                # Store velocity (last step)
                if save_velocity and result.get('velocities'):
                    vel_slice = result['velocities'][-1].squeeze().cpu().numpy()
                    set_slice(velocity_volume, i, vel_slice)

                # Store intermediate states
                if save_intermediate and result.get('intermediate'):
                    for step_idx, inter in enumerate(result['intermediate']):
                        if step_idx < len(intermediate_volumes):
                            inter_slice = inter.squeeze().cpu().numpy()
                            set_slice(intermediate_volumes[step_idx], i, inter_slice)

                # Clear GPU memory
                torch.cuda.empty_cache()

        # Normal batch mode
        else:
            for start_idx in tqdm(range(0, num_slices, batch_size), desc="Processing"):
                end_idx = min(start_idx + batch_size, num_slices)

                # Gather batch
                batch_slices = []
                for i in range(start_idx, end_idx):
                    slice_2d = get_slice(lr_volume, i)
                    batch_slices.append(torch.from_numpy(slice_2d).unsqueeze(0))

                batch_tensor = torch.stack(batch_slices, dim=0).to(self.device)  # (B, 1, H, W)

                # Inference
                start_time = time.time()
                result = self.infer_slice(
                    batch_tensor, t_lr, t_hr,
                    num_steps=num_steps,
                    return_intermediate=save_intermediate,
                    return_velocities=save_velocity,
                )
                total_time += time.time() - start_time

                all_steps.append(result['num_steps'])

                # Store SR output
                sr_batch = result['output'].squeeze(1).cpu().numpy()
                for j, i in enumerate(range(start_idx, end_idx)):
                    set_slice(sr_volume, i, sr_batch[j])

                # Store velocity (last step)
                if save_velocity and result.get('velocities'):
                    vel_batch = result['velocities'][-1].squeeze(1).cpu().numpy()
                    for j, i in enumerate(range(start_idx, end_idx)):
                        set_slice(velocity_volume, i, vel_batch[j])

            # Store intermediate states
            if save_intermediate and result.get('intermediate'):
                for step_idx, inter in enumerate(result['intermediate']):
                    inter_batch = inter.squeeze(1).cpu().numpy()
                    for j, i in enumerate(range(start_idx, end_idx)):
                        set_slice(intermediate_volumes[step_idx], i, inter_batch[j])

            # Clean up GPU memory after each batch
            del batch_tensor, result
            torch.cuda.empty_cache()

        # Build result
        output = {
            'sr_volume': sr_volume,
            'num_slices': num_slices,
            'avg_steps': np.mean(all_steps),
            'total_time': total_time,
            'time_per_slice': total_time / num_slices,
            't_lr': t_lr,
            't_hr': t_hr,
            'pad': pad,
        }

        if save_velocity:
            output['velocity_volume'] = velocity_volume

        if save_intermediate:
            output['intermediate_volumes'] = intermediate_volumes

        print(f"\nInference complete:")
        print(f"  Average steps: {output['avg_steps']:.1f}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per slice: {output['time_per_slice']*1000:.1f}ms")

        return output


# =============================================================================
# Batch Evaluation
# =============================================================================

def create_evaluation_dataset(
    data_dir: str,
    split: str,
    t_lr: float,
    t_hr: float,
    num_samples: Optional[int] = None,
) -> PrecomputedNPY2DDataset:
    """Create dataset for evaluation using the same class as training."""
    dataset = PrecomputedNPY2DDataset(
        data_path=data_dir,
        split=split,
        lr_thickness_range=(t_lr, t_lr),  # Fixed T_lr for evaluation
        tgt_thickness_range=(t_hr, t_hr),  # Fixed T_hr for evaluation
        patch_size=(128, 128),
        slice_profile='slr',
        degradation_mode='legacy',
        max_slices=num_samples,
        use_ceta=False,
    )
    return dataset


def run_batch_evaluation(
    stage1_ckpt: Optional[str],
    stage2_ckpt: str,
    data_dir: str,
    split: str,
    t_lr: float,
    t_hr: float,
    device: torch.device,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    num_steps: Optional[int] = None,
    adaptive: bool = True,
    sequential: bool = False,
    save_velocity: bool = False,
    save_intermediate: bool = False,
    save_ceta: bool = False,
    t_lr_alt: float = 5.0,
    max_steps: Optional[int] = None,  # Override max_steps for adaptive stepping
    mode: str = 'patch',  # 'patch' or 'sliding_window'
    patch_size: int = 128,
    overlap: int = 32,
    batch_patches: int = 16,  # Number of patches to process in parallel
    native_resolution: Optional[float] = None,  # HCP=0.7, MIND=0.9, IDEAS=1.0
    real_lr_dataset: Optional[str] = None,  # In-house real LR dataset name (e.g., 'flair_st4mm')
) -> Dict:
    """
    Run batch evaluation on test set.

    Args:
        mode: 'patch' for patch-based inference, 'sliding_window' for full image reconstruction
        patch_size: Patch size for sliding window mode
        overlap: Overlap between patches for sliding window mode
        batch_patches: Number of patches to process in parallel (default: 16)
    """
    print(f"\n[DEBUG] run_batch_evaluation called")
    print(f"[DEBUG] data_dir={data_dir}, split={split}")
    print(f"[DEBUG] output_dir={output_dir}")
    print(f"[DEBUG] mode={mode}, patch_size={patch_size}, overlap={overlap}, batch_patches={batch_patches}")
    print(f"[DEBUG] save_velocity={save_velocity}, save_intermediate={save_intermediate}, save_ceta={save_ceta}")

    data_path = Path(data_dir)

    print(f"\nDRIFT Batch Evaluation:")
    print(f"  Stage 1: {stage1_ckpt}")
    print(f"  Stage 2: {stage2_ckpt}")
    if real_lr_dataset:
        print(f"  Data: {data_dir}/{real_lr_dataset} (real LR)")
    else:
        print(f"  Data: {data_dir}/{split}")
        print(f"  Protocol: T_lr={t_lr:.2f}mm -> T_hr={t_hr:.2f}mm")
    print(f"  Sequential: {sequential}")
    if save_ceta:
        print(f"  CETA comparison: T_lr_alt={t_lr_alt:.2f}mm")

    # Create inference pipeline
    print(f"\n[DEBUG] Creating inference pipeline...")
    try:
        pipeline = DRIFT2DInference(
            stage1_ckpt=stage1_ckpt,
            stage2_ckpt=stage2_ckpt,
            device=device,
            num_inference_steps=num_steps or 8,
            use_adaptive_steps=adaptive,
        )

        # Override max_steps if specified
        if max_steps is not None and pipeline.stage2 is not None:
            print(f"  Overriding max_steps: {pipeline.stage2.max_steps} -> {max_steps}")
            pipeline.stage2.max_steps = max_steps

        print(f"[DEBUG] Pipeline created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'num_samples': 0}

    # Create dataset
    is_real_lr = real_lr_dataset is not None
    print(f"\n[DEBUG] Creating evaluation dataset (mode={mode}, real_lr={is_real_lr})...")
    try:
        if is_real_lr:
            # In-house real LR dataset (pre-degraded by MRI scanner)
            dataset = InhouseRealLRDataset(
                data_dir=data_dir,
                dataset_name=real_lr_dataset,
                max_slices=num_samples,
            )
            # Override t_lr/t_hr from dataset metadata
            t_lr = dataset.thick_mm
            t_hr = dataset.inplane_mm
            # Real LR always uses sliding_window (full image, no synthetic patches)
            mode = 'sliding_window'
            print(f"  Real LR mode: T_lr={t_lr}mm -> T_hr={t_hr}mm")
        else:
            # Standard synthetic evaluation using PrecomputedNPY2DDataset
            # For sliding_window mode, we need full images, not patches
            if mode == 'sliding_window':
                eval_patch_size = None  # Return full images (HCP: 320x320, MIND/IDEAS: 256x256)
            else:
                eval_patch_size = (128, 128)  # Default patch size for patch mode

            dataset = PrecomputedNPY2DDataset(
                data_path=data_dir,
                split=split,
                lr_thickness_range=(t_lr, t_lr),  # Fixed T_lr for evaluation
                tgt_thickness_range=(t_hr, t_hr),  # Fixed T_hr for evaluation
                patch_size=eval_patch_size,
                slice_profile='slr',
                degradation_mode='legacy',
                max_slices=num_samples,
                use_ceta=False,
                native_resolution=native_resolution,  # HCP=0.7, MIND=0.9, IDEAS=1.0
            )
            print(f"  Mode: {mode}, patch_size: {eval_patch_size} (None = full image)")
        print(f"  Found {len(dataset)} samples")
    except Exception as e:
        print(f"ERROR: Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'num_samples': 0}

    if len(dataset) == 0:
        print(f"ERROR: No samples found in dataset")
        return {'error': 'No samples found', 'num_samples': 0}

    # Create output directory for samples
    # Auto-generate output_dir for real LR if not specified
    if not output_dir and is_real_lr and stage2_ckpt:
        # Extract run_id from ckpt path: .../DRIFT-SR-2D/2q64bpox/checkpoints/...
        ckpt_parts = Path(stage2_ckpt).parts
        run_id = None
        for i, part in enumerate(ckpt_parts):
            if part == 'checkpoints' and i > 0:
                run_id = ckpt_parts[i - 1]
                break
        if run_id:
            output_dir = f"/ssd3/yoonseok/project/DRIFT-SR-2D-Inhouse/{run_id}/{real_lr_dataset}"

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        samples_path = out_path / "samples"
        samples_path.mkdir(exist_ok=True)
        print(f"  Output directory: {out_path}")
        print(f"  Samples directory: {samples_path}")
    else:
        subfolder_name = f"_inference_tlr{t_lr:.1f}_thr{t_hr:.1f}"
        out_path = None
        samples_path = None

    all_psnr, all_ssim, all_steps = [], [], []
    all_ceta_diff = []
    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get sample from dataset (same as training validation)
        data = dataset[idx]

        # Extract tensors - dataset returns tensors, add batch dimension
        x_lr = data['x_lr'].unsqueeze(0).float()  # (1, 1, H, W)

        # Handle GT: real LR dataset may have y_hr=None
        has_gt_sample = data.get('y_hr') is not None
        if has_gt_sample:
            y_hr = data['y_hr'].numpy().astype(np.float32)  # (1, H, W) -> numpy
            if y_hr.ndim == 3:
                y_hr = y_hr.squeeze(0)  # (H, W)
        else:
            y_hr = None

        # Get protocol from dataset
        protocol = data['protocol']
        sample_t_lr = protocol[0].item()
        sample_t_hr = protocol[1].item()

        # Get sample name
        if is_real_lr:
            # InhouseRealLRDataset: use filename directly
            sample_name = data['filename'].replace('.npy', '')
        elif hasattr(dataset, 'slice_files') and hasattr(dataset, 'file_format'):
            if dataset.file_format == 'hcp' or dataset.file_format == 'mind':
                _, modality, filename, extra_info = dataset.slice_files[idx]
                sample_name = filename.replace('.npy', '')
            elif dataset.file_format == 'brats21':
                _, modality, filename = dataset.slice_files[idx]
                sample_name = filename.replace('.npy', '')
            else:
                _, _, filename = dataset.slice_files[idx]
                sample_name = filename.replace('.npy', '')
        else:
            sample_name = f"sample_{idx:05d}"

        # Debug first sample
        if idx == 0:
            print(f"\n[DEBUG] First sample: {sample_name}")
            print(f"  x_lr shape: {x_lr.shape}, dtype: {x_lr.dtype}")
            print(f"  x_lr range: [{x_lr.min():.3f}, {x_lr.max():.3f}]")
            if y_hr is not None:
                print(f"  y_hr shape: {y_hr.shape}, dtype: {y_hr.dtype}")
                print(f"  y_hr range: [{y_hr.min():.3f}, {y_hr.max():.3f}]")
            else:
                print(f"  y_hr: None (no GT)")
            print(f"  protocol (T_lr, T_hr): [{sample_t_lr:.2f}, {sample_t_hr:.2f}]")
            print(f"  Data keys: {list(data.keys())}")

        # Inference
        start_time = time.time()
        if mode == 'sliding_window':
            result = pipeline.infer_slice_sliding_window(
                x_lr, sample_t_lr, sample_t_hr,
                patch_size=patch_size,
                overlap=overlap,
                num_steps=num_steps,
                return_coarse=save_ceta,
                return_intermediate=save_intermediate,
                return_velocities=save_velocity,
                batch_patches=batch_patches,
            )
        else:
            result = pipeline.infer_slice(
                x_lr, sample_t_lr, sample_t_hr,
                num_steps=num_steps,
                return_intermediate=save_intermediate,
                return_velocities=save_velocity,
                return_coarse=save_ceta,
            )
        inference_time = time.time() - start_time

        # Compute metrics
        y_pred = result['output'].squeeze().cpu().numpy().astype(np.float32)

        # Debug first sample output
        if idx == 0:
            print(f"  y_pred shape: {y_pred.shape}")
            print(f"  y_pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
            print(f"  y_pred has NaN: {np.isnan(y_pred).any()}")
            print(f"  num_steps: {result['num_steps']}")

        # Debug: check for NaN in output
        if np.isnan(y_pred).any():
            print(f"Warning: NaN detected in prediction for {sample_name}")
            print(f"  x_lr range: [{x_lr.min():.3f}, {x_lr.max():.3f}]")
            print(f"  y_pred has {np.isnan(y_pred).sum()} NaN values")
            metrics = {'psnr': float('nan'), 'ssim': float('nan')}
        elif y_hr is not None and y_pred.shape != y_hr.shape:
            print(f"Warning: Shape mismatch for {sample_name}: pred={y_pred.shape}, hr={y_hr.shape}")
            metrics = {'psnr': float('nan'), 'ssim': float('nan')}
        elif y_hr is not None:
            metrics = compute_metrics(y_pred, y_hr)
        else:
            metrics = None  # No GT available (qualitative only)

        if metrics is not None:
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
        all_steps.append(result['num_steps'])

        sample_result = {
            'sample': sample_name,
            'steps': result['num_steps'],
            'time': inference_time,
            't_lr': sample_t_lr,
            't_hr': sample_t_hr,
        }
        if metrics is not None:
            sample_result['psnr'] = metrics['psnr']
            sample_result['ssim'] = metrics['ssim']

        # Save sample outputs to samples/ folder
        if samples_path:
            # For real LR: organize by subject for easier comparison
            if is_real_lr:
                subject = data.get('subject', 'unknown')
                subj_dir = samples_path / subject
                subj_dir.mkdir(parents=True, exist_ok=True)
                save_dir = subj_dir
            else:
                save_dir = samples_path

            # Save final SR output
            np.save(save_dir / f"{sample_name}_sr.npy", y_pred)
            # Save HR ground truth
            if y_hr is not None:
                np.save(save_dir / f"{sample_name}_hr.npy", y_hr)
            # Save LR input
            np.save(save_dir / f"{sample_name}_lr.npy", x_lr.squeeze().cpu().numpy())
            # Save Stage 1 coarse output if available
            if result.get('coarse') is not None:
                np.save(save_dir / f"{sample_name}_stage1.npy", result['coarse'].squeeze().cpu().numpy())

        # Save velocity
        if save_velocity and samples_path and result.get('velocities'):
            vel_path = save_dir / f"{sample_name}_velocity.npy"
            velocities_np = [v.squeeze().cpu().numpy() for v in result['velocities']]
            np.save(vel_path, {'velocities': velocities_np, 't_lr': sample_t_lr, 't_hr': sample_t_hr})

        # Save intermediate states
        if save_intermediate and samples_path and result.get('intermediate'):
            inter_path = save_dir / f"{sample_name}_intermediate.npy"
            intermediate_np = [s.squeeze().cpu().numpy() for s in result['intermediate']]
            np.save(inter_path, {'intermediate': intermediate_np, 't_lr': sample_t_lr, 't_hr': sample_t_hr})

        # CETA comparison
        if save_ceta and result.get('coarse') is not None:
            # Run Stage 1 with alternative thickness
            protocol_alt = torch.tensor([[t_lr_alt, sample_t_hr]], device=device).float()
            y_coarse_alt = pipeline.run_stage1(x_lr.to(device), protocol_alt)

            # Compute CETA velocities
            protocol_tensor = torch.tensor([[sample_t_lr, sample_t_hr]], device=device).float()
            ceta_result = pipeline.compute_ceta_velocities(
                result['coarse'], y_coarse_alt,
                protocol_tensor, protocol_alt,
                t_eval=0.0
            )

            if ceta_result:
                endpoint_diff = ceta_result['endpoint_diff'].mean().item()
                all_ceta_diff.append(endpoint_diff)
                sample_result['ceta_endpoint_diff'] = endpoint_diff

                # Save CETA data to samples/{sample_name}_ceta/ folder
                if samples_path:
                    ceta_folder = samples_path / f"{sample_name}_ceta"
                    ceta_folder.mkdir(exist_ok=True)
                    # Save individual CETA components
                    np.save(ceta_folder / "velocity.npy", ceta_result['velocity'].squeeze().cpu().numpy())
                    np.save(ceta_folder / "velocity_alt.npy", ceta_result['velocity_alt'].squeeze().cpu().numpy())
                    np.save(ceta_folder / "endpoint_est.npy", ceta_result['endpoint_est'].squeeze().cpu().numpy())
                    np.save(ceta_folder / "endpoint_est_alt.npy", ceta_result['endpoint_est_alt'].squeeze().cpu().numpy())
                    np.save(ceta_folder / "endpoint_diff.npy", ceta_result['endpoint_diff'].squeeze().cpu().numpy())
                    # Save metadata
                    ceta_meta = {
                        't_lr': sample_t_lr,
                        't_lr_alt': t_lr_alt,
                        't_hr': sample_t_hr,
                        'endpoint_diff_mean': endpoint_diff,
                    }
                    np.save(ceta_folder / "metadata.npy", ceta_meta)

        results.append(sample_result)

        # Clear GPU memory in sequential mode
        if sequential:
            torch.cuda.empty_cache()

    # Summary - filter out NaN values for proper statistics
    valid_psnr = [p for p in all_psnr if not np.isnan(p)]
    valid_ssim = [s for s in all_ssim if not np.isnan(s)]

    num_failed = len(all_psnr) - len(valid_psnr)
    if num_failed > 0:
        print(f"\nWarning: {num_failed} samples had NaN metrics (skipped in summary)")

    all_times = [r['time'] for r in results]

    # Build summary with averages at the top, per-sample results at the bottom
    summary = {
        'model': 'DRIFT',
        'num_samples': len(results),
        't_lr': t_lr,
        't_hr': t_hr,
    }

    if is_real_lr:
        summary['real_lr_dataset'] = real_lr_dataset
        summary['has_gt'] = bool(valid_psnr)

    # Metrics averages (right after basic info)
    if valid_psnr:
        summary['mean_psnr'] = float(np.mean(valid_psnr))
        summary['std_psnr'] = float(np.std(valid_psnr))
        summary['mean_ssim'] = float(np.mean(valid_ssim))
        summary['std_ssim'] = float(np.std(valid_ssim))
        summary['num_valid'] = len(valid_psnr)
        summary['num_failed'] = num_failed

    if save_ceta and all_ceta_diff:
        summary['mean_ceta_endpoint_diff'] = float(np.mean(all_ceta_diff))
        summary['std_ceta_endpoint_diff'] = float(np.std(all_ceta_diff))
        summary['t_lr_alt'] = t_lr_alt

    summary['mean_steps'] = float(np.mean(all_steps))
    summary['mean_time'] = float(np.mean(all_times))
    summary['total_time'] = float(np.sum(all_times))

    # Config details
    summary['adaptive'] = adaptive
    summary['mode'] = mode
    summary['patch_size'] = patch_size if mode == 'sliding_window' else None
    summary['overlap'] = overlap if mode == 'sliding_window' else None
    summary['stage1_ckpt'] = stage1_ckpt
    summary['stage2_ckpt'] = stage2_ckpt

    # Per-sample results at the bottom
    summary['results'] = results

    print(f"\n{'='*50}")
    if valid_psnr:
        print(f"DRIFT Results ({summary['num_valid']}/{summary['num_samples']} valid samples)")
    else:
        print(f"DRIFT Results ({summary['num_samples']} samples, qualitative only)")
    print(f"Mode: {mode}" + (f" (patch_size={patch_size}, overlap={overlap})" if mode == 'sliding_window' else ""))
    print(f"{'='*50}")
    if valid_psnr:
        print(f"PSNR: {summary['mean_psnr']:.2f} +/- {summary['std_psnr']:.2f} dB")
        print(f"SSIM: {summary['mean_ssim']:.4f} +/- {summary['std_ssim']:.4f}")
    else:
        print(f"No GT available (qualitative evaluation only)")
    print(f"Avg Steps: {summary['mean_steps']:.1f}")
    if save_ceta and all_ceta_diff:
        print(f"CETA Endpoint Diff: {summary['mean_ceta_endpoint_diff']:.6f} +/- {summary['std_ceta_endpoint_diff']:.6f}")
    print(f"{'='*50}")

    if out_path:
        results_file = out_path / "drift_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {results_file}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DRIFT 2D Inference",
        epilog="Priority: CLI args > config file > defaults. "
               "Example: python inference/inference_drift_2d.py --config config/inference_mind.yaml --split test --t-lr 3.0",
    )

    # Config file (dataset-specific defaults)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to inference config YAML (e.g., config/inference_mind.yaml). '
                             'Config values are used as defaults; CLI args override them.')

    # Model checkpoints
    parser.add_argument('--stage1-ckpt', type=str, default=None,
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--stage2-ckpt', type=str, default=None,
                        help='Path to Stage 2 checkpoint')
    parser.add_argument('--stage2-only', action='store_true',
                        help='Skip Stage 1, input is already coarse prediction')

    # Input/Output
    parser.add_argument('--input', type=str,
                        help='Path to input NIfTI volume (single volume mode)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (batch mode)')
    parser.add_argument('--split', type=str, default=None,
                        help='Data split (batch mode): train, val, test')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    # Protocol (slice thickness)
    parser.add_argument('--t-lr', type=float, default=None,
                        help='LR slice thickness in mm')
    parser.add_argument('--t-hr', type=float, default=None,
                        help='Target HR slice thickness in mm')
    parser.add_argument('--native-resolution', type=float, default=None,
                        help='Native voxel resolution in mm (HCP=0.7, MIND=0.9, IDEAS=1.0). '
                             'Auto-detected if not set.')

    # Processing
    parser.add_argument('--slice-axis', type=int, default=0,
                        choices=[0, 1, 2],
                        help='Axis with degradation (0=axial, 1=coronal, 2=sagittal)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for slice processing')
    parser.add_argument('--num-samples', type=int,
                        help='Number of samples for batch evaluation')

    # Inference mode (for batch evaluation)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['patch', 'sliding_window'],
                        help='Inference mode: patch (evaluate on patches) or sliding_window (full image reconstruction)')

    # Sliding window settings (for both single volume and batch evaluation)
    parser.add_argument('--sliding-window', action='store_true',
                        help='Use sliding window inference for single volume (equivalent to --mode sliding_window for batch)')
    parser.add_argument('--patch-size', type=int, default=None,
                        help='Patch size for sliding window')
    parser.add_argument('--overlap', type=int, default=None,
                        help='Overlap between patches for sliding window')
    parser.add_argument('--batch-patches', type=int, default=None,
                        help='Number of patches to process in parallel (default: 16)')

    # ODE stepping
    parser.add_argument('--num-steps', type=int,
                        help='Fixed number of ODE steps (overrides adaptive)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use PAD-based adaptive stepping')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum ODE steps for adaptive mode')

    # Velocity/CETA output saving (for paper figures)
    parser.add_argument('--save-velocity', action='store_true',
                        help='Save velocity field outputs')
    parser.add_argument('--save-intermediate-steps', action='store_true',
                        help='Save intermediate ODE states')
    parser.add_argument('--save-ceta', action='store_true',
                        help='Compute and save CETA velocity comparison')
    parser.add_argument('--t-lr-alt', type=float, default=None,
                        help='Alternative T_lr for CETA comparison')

    # In-house real LR dataset
    parser.add_argument('--real-lr-dataset', type=str, default=None,
                        help='In-house real LR dataset name (e.g., flair_st4mm, t2_st7mm). '
                             'Uses pre-degraded LR slices instead of synthetic degradation.')

    # Processing mode
    parser.add_argument('--sequential', action='store_true',
                        help='Process samples sequentially (saves memory, slower)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load config file and apply as defaults (CLI args take priority)
    # Priority: CLI args > config file > hardcoded defaults
    # ---------------------------------------------------------------
    config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"Loaded inference config: {args.config}")

    # Config key -> (args attribute, hardcoded default)
    CONFIG_DEFAULTS = {
        'stage1_ckpt':       ('stage1_ckpt',       None),
        'stage2_ckpt':       ('stage2_ckpt',       None),
        'data_dir':          ('data_dir',          None),
        't_hr':              ('t_hr',              None),
        'native_resolution': ('native_resolution', None),
        'mode':              ('mode',              'patch'),
        'patch_size':        ('patch_size',        128),
        'overlap':           ('overlap',           32),
        'batch_patches':     ('batch_patches',     16),
        'max_steps':         ('max_steps',         10),
        't_lr_alt':          ('t_lr_alt',          5.0),
        'adaptive':          ('adaptive',          False),
        'output_dir':        ('output_dir',        None),
        'real_lr_dataset':   ('real_lr_dataset',   None),
    }

    for cfg_key, (attr, hardcoded_default) in CONFIG_DEFAULTS.items():
        cli_val = getattr(args, attr)
        if cli_val is None or (isinstance(cli_val, bool) and not cli_val):
            # CLI not explicitly set -> use config value, else hardcoded default
            cfg_val = config.get(cfg_key, hardcoded_default)
            setattr(args, attr, cfg_val)

    # Validate required fields for batch mode
    if args.split and not args.data_dir:
        parser.error("--data-dir is required for batch mode. "
                     "Use --config to load dataset-specific defaults (e.g., --config config/inference_mind.yaml)")
    if args.real_lr_dataset and not args.data_dir:
        parser.error("--data-dir is required for real LR dataset mode.")

    if config:
        print(f"  Dataset config applied:")
        print(f"    data_dir:          {args.data_dir}")
        print(f"    native_resolution: {args.native_resolution}")
        print(f"    t_hr:              {args.t_hr}")
        print(f"    stage1_ckpt:       {args.stage1_ckpt}")
        print(f"    stage2_ckpt:       {args.stage2_ckpt}")
        print(f"    mode:              {args.mode}")
        print(f"    adaptive:          {args.adaptive}")
        if args.real_lr_dataset:
            print(f"    real_lr_dataset:   {args.real_lr_dataset}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine adaptive mode
    adaptive = args.adaptive or (args.num_steps is None)

    if args.input:
        # Single volume inference
        print(f"\n=== Single Volume Inference ===")
        print(f"Input: {args.input}")
        print(f"Protocol: T_lr={args.t_lr}mm -> T_hr={args.t_hr}mm")

        # Create pipeline
        pipeline = DRIFT2DInference(
            stage1_ckpt=args.stage1_ckpt,
            stage2_ckpt=args.stage2_ckpt,
            device=device,
            num_inference_steps=args.num_steps or 8,
            use_adaptive_steps=adaptive,
            stage2_only=args.stage2_only,
        )

        # Override max_steps if specified
        if args.max_steps and pipeline.stage2:
            pipeline.stage2.max_steps = args.max_steps

        # Load volume
        lr_data, affine = load_nifti(args.input)
        lr_norm, data_min, data_max = normalize_volume(lr_data)

        # Run inference
        result = pipeline.infer_volume(
            lr_norm, args.t_lr, args.t_hr,
            slice_axis=args.slice_axis,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            save_velocity=args.save_velocity,
            save_intermediate=args.save_intermediate_steps,
            use_sliding_window=args.sliding_window,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_patches=args.batch_patches,
        )

        # Save outputs
        if args.output_dir:
            out_path = Path(args.output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            stem = Path(args.input).stem.replace('.nii', '')

            # SR volume
            sr_data = denormalize_volume(result['sr_volume'], data_min, data_max)
            save_nifti(sr_data, affine, str(out_path / f"sr_{stem}.nii.gz"))

            # Velocity volume
            if args.save_velocity and 'velocity_volume' in result:
                save_nifti(result['velocity_volume'], affine, str(out_path / f"velocity_{stem}.nii.gz"))

            # Intermediate volumes
            if args.save_intermediate_steps and 'intermediate_volumes' in result:
                for step_idx, inter_vol in enumerate(result['intermediate_volumes']):
                    inter_data = denormalize_volume(inter_vol, data_min, data_max)
                    save_nifti(inter_data, affine, str(out_path / f"step{step_idx:02d}_{stem}.nii.gz"))

            # Info JSON
            info = {k: v for k, v in result.items() if not isinstance(v, np.ndarray) and not isinstance(v, list)}
            info_file = out_path / f"info_{stem}.json"
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Info saved to: {info_file}")

    elif args.split or args.real_lr_dataset:
        # Batch evaluation mode (synthetic NPY or real LR)
        eval_mode = 'sliding_window' if args.sliding_window else args.mode

        run_batch_evaluation(
            stage1_ckpt=args.stage1_ckpt,
            stage2_ckpt=args.stage2_ckpt,
            data_dir=args.data_dir,
            split=args.split,
            t_lr=args.t_lr,
            t_hr=args.t_hr,
            device=device,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            adaptive=adaptive,
            sequential=args.sequential,
            save_velocity=args.save_velocity,
            save_intermediate=args.save_intermediate_steps,
            save_ceta=args.save_ceta,
            t_lr_alt=args.t_lr_alt,
            max_steps=args.max_steps,
            mode=eval_mode,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_patches=args.batch_patches,
            native_resolution=args.native_resolution,
            real_lr_dataset=args.real_lr_dataset,
        )

    else:
        parser.error("Either --input, --split, or --real-lr-dataset must be specified")


if __name__ == "__main__":
    main()
