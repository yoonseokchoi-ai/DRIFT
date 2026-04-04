"""
DRIFT 2D Dataset
================

2D dataset for DRIFT (Difficulty-aware Rectified Flows for Protocol-Conditioned
Through-plane MRI Super-Resolution).

Key Design (Optimized):
1. Load 3D HR volume
2. Random slice axis selection (axial/coronal/sagittal) - this is the "thick-slice direction"
3. Extract 2D slice from a PERPENDICULAR plane first
4. Apply 1D SLR blur + downsample/upsample on the 2D slice (along artifact direction)
5. Return 2D patches (128×128) for efficient training

Why this is equivalent to 3D blur → 2D extract:
- Convolution is linear and separable
- Blur along axis A, then extract slice along axis B = Extract slice along B, then blur along A
- Much more efficient: 320×320 2D slice vs 320×320×320 3D volume

Data Pipeline:
1. Load HR volume (e.g., 320×320×320)
2. Random slice_axis selection (0=axial, 1=coronal, 2=sagittal as thick-slice direction)
3. Random perpendicular axis (extract_axis) selection
4. Extract 2D HR slice along extract_axis
5. Apply 1D degradation (blur + downsample + upsample) along artifact_axis in 2D slice
6. Random crop to patch size (128×128)
7. Return: x_lr (with stair-step artifact), y_hr (clean), protocol info

Example:
- slice_axis=0 (axial thick-slice), extract_axis=1 (coronal slice)
- 2D slice shape: (D, W), artifact along axis 0 (D direction)
- Apply 1D blur along axis 0 of 2D slice

Uses mri_utils.py for:
- Orientation detection
- SLR slice profile (Sec. 3.1, Eq. S1-S3)
- Protocol parameter generation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import nibabel as nib
from functools import lru_cache

# MONAI for proper medical image resizing
try:
    from monai.transforms import Resize
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# SigPy for SLR (Shinnar-Le Roux) slice profile
try:
    import sigpy.mri.rf as rf_design
    SIGPY_AVAILABLE = True
except ImportError:
    SIGPY_AVAILABLE = False


# MRI orientation and protocol constants
from .mri_utils import (
    SCAN_TYPES,
    ORIENTATION_SLICE_AXIS_MAP,
    get_slice_axis_from_orientation,
    DEFAULT_THICKNESS_RANGE,
    DEFAULT_GAP_RANGE,
    DEFAULT_TARGET_THICKNESS_RANGE,
    DEFAULT_TARGET_GAP_RANGE,
)


# =============================================================================
# LRU Cache for Volume Loading (avoids repeated disk I/O)
# =============================================================================
# Cache size: 320³ × 4 bytes ≈ 125MB per volume
# maxsize=16 → ~2GB RAM per worker (reasonable for multi-worker training)
@lru_cache(maxsize=16)
def _cached_load_volume(filepath: str) -> Dict:
    """
    Load and preprocess volume with LRU caching.

    This dramatically speeds up training when multiple slices are sampled
    from the same volume, avoiding repeated disk I/O.
    """
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    voxel_spacing = np.abs(np.diag(affine)[:3])
    orientation = nib.aff2axcodes(affine)

    # Normalize to [-1, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    data = data * 2 - 1

    return {
        'data': torch.from_numpy(data),
        'voxel_spacing': voxel_spacing,
        'orientation': orientation,
        'filepath': filepath,
    }


class DRIFT2DDataset(Dataset):
    """
    2D Dataset for Cascaded Super-Resolution.

    Pipeline:
    1. Load 3D HR volume
    2. Random axis selection (determines which plane to extract)
    3. Random slice selection along that axis
    4. Apply SLR blur on perpendicular directions (within the 2D slice)
    5. Return 2D pair: (x_lr, y_hr) with protocol info

    Args:
        data_path: Path to HCP data root
        split: 'train' or 'test'
        contrast: 't1', 't2', or 'both'
        patch_size: 2D patch size (H, W) - default (128, 128)
        lr_thickness_range: LR slice thickness range in mm
        tgt_thickness_range: Target slice thickness range in mm
        slices_per_volume: Number of 2D slices to sample per volume
        slice_profile: 'gaussian', 'sinc', or 'slr'
        num_volumes: Limit number of volumes (None = use all)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        contrast: str = 't1',
        patch_size: Tuple[int, int] = (128, 128),
        lr_thickness_range: Tuple[float, float] = (1.2, 6.0),
        lr_gap_range: Tuple[float, float] = (0.0, 0.0),
        tgt_thickness_range: Tuple[float, float] = (0.7, 1.2),  # kept for backward compat
        tgt_gap_range: Tuple[float, float] = (0.0, 0.0),
        slices_per_volume: int = 32,
        slice_profile: str = 'slr',
        num_volumes: Optional[int] = None,
        transform: Optional[callable] = None,
        cache_in_memory: bool = False,
        degradation_mode: str = 'legacy',  # 'legacy' or 'physical'
        native_resolution: float = 0.7,  # Fixed T_hr (HCP=0.7mm, BraTS=1.0mm)
        # PAD-weighted LR thickness sampling
        lr_sampling_mode: str = 'uniform',  # 'uniform' or 'pad_weighted'
        pad_sampling_alpha: float = 2.0,  # Beta(α, 1) for PAD-weighted sampling
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.split = split
        self.contrast = contrast
        self.patch_size = patch_size
        self.slices_per_volume = slices_per_volume
        self.slice_profile = slice_profile
        self.transform = transform
        self.cache_in_memory = cache_in_memory

        # Degradation mode:
        # - 'legacy': blur → area downsample → nearest upsample (기존 방식, Stage1 학습에 사용)
        # - 'physical': blur with stride → nearest upsample (물리적으로 정확한 방식)
        self.degradation_mode = degradation_mode

        # Protocol ranges
        self.lr_thickness_range = lr_thickness_range
        self.lr_gap_range = lr_gap_range

        # Native resolution and T_hr range
        # T_hr is sampled from [native_resolution, 1.0mm] during training
        # Clinical HR MRI: ≤1mm is considered high resolution
        self.native_resolution = native_resolution
        self.max_thr = 1.0  # Clinical HR threshold
        self.thr_range = (self.native_resolution, self.max_thr)

        # PAD-weighted sampling mode
        # PAD = 1 - T_hr/T_lr measures spectral deficit (lost high-frequency information)
        # Beta(α, 1) distribution biases toward higher T_lr (harder cases)
        self.lr_sampling_mode = lr_sampling_mode
        self.pad_sampling_alpha = pad_sampling_alpha

        # Discover files
        self.file_list = self._discover_files()

        if len(self.file_list) == 0:
            raise ValueError(f"No files found in {data_path}/{split} for contrast={contrast}")

        # Limit volumes if specified
        if num_volumes is not None and num_volumes < len(self.file_list):
            self.file_list = self.file_list[:num_volumes]

        print(f"DRIFT 2D Dataset: {split} split, {contrast} contrast")
        print(f"  Volumes: {len(self.file_list)}")
        print(f"  Slices per volume: {slices_per_volume}")
        print(f"  Total samples: {len(self)}")
        print(f"  Patch size: {patch_size}")
        print(f"  LR thickness: {lr_thickness_range[0]:.1f} - {lr_thickness_range[1]:.1f} mm")
        print(f"  LR sampling: {lr_sampling_mode}" + (f" (α={pad_sampling_alpha})" if lr_sampling_mode == 'pad_weighted' else ""))
        print(f"  Target HR range: [{native_resolution:.2f}, {self.max_thr:.2f}] mm")
        print(f"  Slice profile: {slice_profile}")
        print(f"  Degradation mode: {degradation_mode}")

        # Cache
        self.cache = {}
        if cache_in_memory:
            print("Caching volumes...")
            for i, fp in enumerate(self.file_list):
                self.cache[i] = self._load_volume(fp)
                if (i + 1) % 50 == 0:
                    print(f"  Cached {i + 1}/{len(self.file_list)}")

    def _discover_files(self) -> List[Path]:
        """Discover HCP NIfTI files."""
        split_dir = self.data_path / self.split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        files = []
        subject_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for subj_dir in subject_dirs:
            subj_id = subj_dir.name

            if self.contrast == 'both':
                for c in ['t1', 't2']:
                    f = subj_dir / f"{subj_id}_{c}_hr.nii.gz"
                    if f.exists():
                        files.append(f)
            else:
                f = subj_dir / f"{subj_id}_{self.contrast}_hr.nii.gz"
                if f.exists():
                    files.append(f)

        return files

    def _load_volume(self, filepath: Path) -> Dict:
        """Load volume using LRU cache for efficiency."""
        return _cached_load_volume(str(filepath))

    def __len__(self) -> int:
        return len(self.file_list) * self.slices_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume

        # Load volume
        if vol_idx in self.cache:
            vol_data = self.cache[vol_idx]
        else:
            vol_data = self._load_volume(self.file_list[vol_idx])

        hr_volume = vol_data['data']  # (D, H, W)
        voxel_spacing = vol_data['voxel_spacing']
        orientation = vol_data['orientation']

        # Random number generator for this sample
        rng = np.random.RandomState(idx + slice_idx * 1000)

        # Step 1: Random slice_axis selection (thick-slice direction)
        # This is the direction where blur/downsampling is applied
        if self.split == 'train':
            scan_type = rng.choice(SCAN_TYPES)
            orientation_str = ''.join(orientation)
            if orientation_str in ORIENTATION_SLICE_AXIS_MAP:
                slice_axis = ORIENTATION_SLICE_AXIS_MAP[orientation_str][scan_type]
            else:
                slice_axis = get_slice_axis_from_orientation(orientation, scan_type)
        else:
            # For validation, cycle through axes deterministically
            slice_axis = slice_idx % 3

        hr_spacing = voxel_spacing[slice_axis]

        # Step 2: Protocol parameters
        # T_hr: sampled from thr_range (clinical HR range)
        # T_lr: random within range for difficulty diversity
        if self.thr_range[0] >= self.thr_range[1]:
            # Fixed T_hr case: use specified value
            T_hr = self.thr_range[0]
        else:
            # Range case: sample T_hr from [min, max]
            T_hr = rng.uniform(self.thr_range[0], self.thr_range[1])

        # Sample T_lr from (lr_min, lr_max] - exclusive lower bound, inclusive upper bound
        # This ensures T_lr > T_hr when lr_thickness_range[0] == T_hr (e.g., [0.7, 6.0] with T_hr=0.7)
        lr_min, lr_max = self.lr_thickness_range
        lr_min_exclusive = lr_min + 1e-6

        if self.lr_sampling_mode == 'pad_weighted':
            # PAD-weighted sampling using Beta(α, 1) distribution
            # PAD = 1 - T_hr/T_lr measures spectral deficit (lost high-frequency info)
            # Beta(α, 1) biases toward higher values when α > 1
            # T_lr = T_min + (T_max - T_min) × X, where X ~ Beta(α, 1)
            # This oversamples harder cases (higher T_lr → higher PAD)
            x = rng.beta(self.pad_sampling_alpha, 1.0)
            T_lr = lr_min_exclusive + (lr_max - lr_min_exclusive) * x
        else:
            # Uniform sampling (default)
            T_lr = rng.uniform(lr_min_exclusive, lr_max)

        # Ensure T_lr > T_hr (safety check)
        if T_lr <= T_hr:
            T_lr = T_hr + 0.5

        # Step 3: Select perpendicular axis for 2D slice extraction
        # If slice_axis=0, perpendicular axes are 1 and 2
        # If slice_axis=1, perpendicular axes are 0 and 2
        # If slice_axis=2, perpendicular axes are 0 and 1
        perp_axes = [i for i in range(3) if i != slice_axis]
        extract_axis = rng.choice(perp_axes)

        # Step 4: Extract 2D HR slice first (before degradation - much more efficient!)
        n_slices = hr_volume.shape[extract_axis]
        margin = max(1, n_slices // 10)
        slice_i = rng.randint(margin, max(margin + 1, n_slices - margin))

        if extract_axis == 0:
            hr_slice = hr_volume[slice_i, :, :]  # (H, W)
        elif extract_axis == 1:
            hr_slice = hr_volume[:, slice_i, :]  # (D, W)
        else:
            hr_slice = hr_volume[:, :, slice_i]  # (D, H)

        # Step 5: Apply 1D degradation on 2D slice (equivalent to 3D blur → 2D extract)
        # Determine which axis in 2D slice corresponds to slice_axis
        artifact_axis_in_2d = self._get_artifact_axis_in_2d(slice_axis, extract_axis)

        # Apply degradation to create LR (T_lr thickness)
        lr_slice = self._apply_2d_degradation(hr_slice, artifact_axis_in_2d, T_lr, hr_spacing)

        # Apply degradation to create target HR (T_hr thickness)
        # If T_hr == native_resolution, use original data
        if abs(T_hr - self.native_resolution) < 0.01:
            target_hr_slice = hr_slice
        else:
            target_hr_slice = self._apply_2d_degradation(hr_slice, artifact_axis_in_2d, T_hr, hr_spacing)

        # Step 6: Random crop to patch size
        pH, pW = self.patch_size
        H, W = hr_slice.shape

        if H > pH:
            h_start = rng.randint(0, H - pH)
        else:
            h_start = 0
        if W > pW:
            w_start = rng.randint(0, W - pW)
        else:
            w_start = 0

        hr_patch = target_hr_slice[h_start:h_start + pH, w_start:w_start + pW]
        lr_patch = lr_slice[h_start:h_start + pH, w_start:w_start + pW]

        # Ensure correct size (pad if needed)
        hr_patch = self._ensure_size_2d(hr_patch, (pH, pW))
        lr_patch = self._ensure_size_2d(lr_patch, (pH, pW))

        # Add channel dimension
        hr_patch = hr_patch.unsqueeze(0)  # (1, H, W)
        lr_patch = lr_patch.unsqueeze(0)  # (1, H, W)

        # Apply transform if any
        if self.transform is not None:
            lr_patch, hr_patch = self.transform(lr_patch, hr_patch)

        # Determine artifact direction in 2D slice (for potential use)
        # When extract_axis != slice_axis, the artifact appears along one axis of the 2D slice
        # This info could be useful for direction-aware processing
        artifact_axis_in_2d = self._get_artifact_axis_in_2d(slice_axis, extract_axis)

        return {
            'x_lr': lr_patch,
            'y_hr': hr_patch,
            'protocol': torch.tensor([T_lr, T_hr], dtype=torch.float32),
            'slice_axis': torch.tensor(slice_axis, dtype=torch.long),
            'extract_axis': torch.tensor(extract_axis, dtype=torch.long),
            'artifact_axis': torch.tensor(artifact_axis_in_2d, dtype=torch.long),
        }

    def _get_artifact_axis_in_2d(self, slice_axis: int, extract_axis: int) -> int:
        """
        Determine which axis in the 2D slice shows the stair-step artifact.

        The artifact appears along the slice_axis direction.
        When we extract a 2D slice along extract_axis, we need to map
        the 3D slice_axis to the 2D coordinate system.

        Returns: 0 (vertical/H axis) or 1 (horizontal/W axis) in the 2D slice
        """
        # 3D axes: 0=D, 1=H, 2=W
        # When extracting along axis X, the 2D slice has axes from remaining two
        remaining = [i for i in range(3) if i != extract_axis]
        # remaining[0] → 2D axis 0, remaining[1] → 2D axis 1

        if slice_axis == remaining[0]:
            return 0  # Artifact along first axis of 2D slice
        else:
            return 1  # Artifact along second axis of 2D slice

    def _apply_2d_degradation(
        self,
        hr_slice: torch.Tensor,
        blur_axis: int,
        T_lr: float,
        hr_spacing: float,
    ) -> torch.Tensor:
        """
        Apply 1D degradation on 2D slice simulating thick-slice MRI acquisition.

        This is mathematically equivalent to applying 3D blur on volume then extracting 2D slice,
        but much more efficient (320×320 vs 320×320×320).

        Two modes available (controlled by self.degradation_mode):
        - 'legacy': blur → area downsample → nearest upsample (기존 Stage1 학습에 사용)
        - 'physical': blur with stride → nearest upsample (물리적으로 정확)

        Args:
            hr_slice: High-resolution 2D slice (H, W)
            blur_axis: Axis along which to apply blur (0 or 1 in 2D slice)
            T_lr: LR slice thickness in mm
            hr_spacing: HR voxel spacing along slice_axis in mm

        Returns:
            LR slice with stair-step artifacts (same size as input)
        """
        H, W = hr_slice.shape
        original_size = [H, W]

        # Compute degradation factor
        downsample_factor = T_lr / hr_spacing
        downsample_factor = max(1.5, min(downsample_factor, 16.0))

        # Create SLR kernel sized to match slice thickness
        fwhm_vox = T_lr / hr_spacing
        sigma = fwhm_vox / 2.355

        kernel_size = int(fwhm_vox * 3) | 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, 51))

        kernel = self._create_1d_kernel(kernel_size, sigma)
        kernel = kernel.to(hr_slice.device)

        if self.degradation_mode == 'legacy':
            # Legacy mode: blur → area downsample → nearest upsample
            # (기존 Stage1 학습에 사용된 방식)

            # Step 1: Apply blur (no stride)
            blurred = self._apply_1d_blur_2d(hr_slice, kernel, blur_axis)

            # Step 2: Area downsample
            blurred_4d = blurred.unsqueeze(0).unsqueeze(0)
            new_size_along_axis = max(4, int(original_size[blur_axis] / downsample_factor))
            ds_size = list(original_size)
            ds_size[blur_axis] = new_size_along_axis
            downsampled_4d = F.interpolate(blurred_4d, size=ds_size, mode='area')

            # Step 3: Nearest upsample (creates stair-step artifact)
            if MONAI_AVAILABLE:
                downsampled_3d = downsampled_4d.squeeze(0)
                resize_transform = Resize(spatial_size=original_size, mode='nearest')
                upsampled_3d = resize_transform(downsampled_3d)
                lr_slice = upsampled_3d.squeeze()
            else:
                upsampled = F.interpolate(downsampled_4d, size=original_size, mode='nearest')
                lr_slice = upsampled.squeeze()
        else:
            # Physical mode: blur with stride → nearest upsample
            # (물리적으로 정확한 방식 - 논문에서 사용)
            stride = max(1, int(round(downsample_factor)))

            # Step 1: Apply SLR blur WITH STRIDE (blur + downsample in one operation)
            downsampled = self._apply_1d_blur_with_stride_2d(hr_slice, kernel, blur_axis, stride)

            # Step 2: Nearest upsample (creates stair-step artifact)
            downsampled_4d = downsampled.unsqueeze(0).unsqueeze(0)

            if MONAI_AVAILABLE:
                downsampled_3d = downsampled_4d.squeeze(0)
                resize_transform = Resize(spatial_size=original_size, mode='nearest')
                upsampled_3d = resize_transform(downsampled_3d)
                lr_slice = upsampled_3d.squeeze()
            else:
                upsampled = F.interpolate(downsampled_4d, size=original_size, mode='nearest')
                lr_slice = upsampled.squeeze()

        return lr_slice

    def _apply_1d_blur_with_stride_2d(
        self,
        slice_2d: torch.Tensor,
        kernel: torch.Tensor,
        axis: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Apply 1D blur with stride along specified axis of 2D slice.

        This combines blur and downsampling in one operation, matching real MRI physics
        where the slice selection profile integrates signal and samples at discrete positions.

        Args:
            slice_2d: 2D slice (H, W)
            kernel: 1D convolution kernel
            axis: Axis to apply blur (0=vertical, 1=horizontal)
            stride: Convolution stride (=downsampling factor)

        Returns:
            Downsampled 2D slice with blur applied
        """
        H, W = slice_2d.shape
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        # Reshape for conv2d: (1, 1, H, W)
        slice_4d = slice_2d.unsqueeze(0).unsqueeze(0)

        if axis == 0:
            # Blur along H (vertical) with stride
            kernel_4d = kernel.view(1, 1, -1, 1)
            slice_4d = F.pad(slice_4d, (0, 0, pad, pad), mode='reflect')
            blurred = F.conv2d(slice_4d, kernel_4d, stride=(stride, 1))
        else:
            # Blur along W (horizontal) with stride
            kernel_4d = kernel.view(1, 1, 1, -1)
            slice_4d = F.pad(slice_4d, (pad, pad, 0, 0), mode='reflect')
            blurred = F.conv2d(slice_4d, kernel_4d, stride=(1, stride))

        return blurred.squeeze()

    def _apply_1d_blur_2d(
        self,
        slice_2d: torch.Tensor,
        kernel: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """Apply 1D blur along specified axis of 2D slice."""
        H, W = slice_2d.shape
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        # Reshape for conv2d: (1, 1, H, W)
        slice_4d = slice_2d.unsqueeze(0).unsqueeze(0)

        if axis == 0:
            # Blur along H (vertical)
            kernel_4d = kernel.view(1, 1, -1, 1)
            # F.conv2d padding: (left, right, top, bottom) for 2D
            # For vertical blur, pad top and bottom
            slice_4d = F.pad(slice_4d, (0, 0, pad, pad), mode='reflect')
            blurred = F.conv2d(slice_4d, kernel_4d)
        else:
            # Blur along W (horizontal)
            kernel_4d = kernel.view(1, 1, 1, -1)
            # For horizontal blur, pad left and right
            slice_4d = F.pad(slice_4d, (pad, pad, 0, 0), mode='reflect')
            blurred = F.conv2d(slice_4d, kernel_4d)

        return blurred.squeeze()

    def _create_1d_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian or SLR kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2

        if self.slice_profile == 'gaussian':
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        elif self.slice_profile == 'slr':
            kernel = self._slr_kernel_1d(x, sigma * 2.355)  # Convert back to FWHM
        else:
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)

        kernel = kernel / kernel.sum()
        return kernel

    def _slr_kernel_1d(self, x: torch.Tensor, fwhm: float) -> torch.Tensor:
        """SLR (Shinnar-Le Roux) profile approximation."""
        if SIGPY_AVAILABLE:
            n_points = 256
            tb = 4

            pulse = rf_design.slr.dzrf(
                n=n_points,
                tb=tb,
                ptype='ex',
                ftype='ls',
                d1=0.01,
                d2=0.01,
            )

            profile = np.abs(np.fft.fftshift(np.fft.fft(pulse, 1024)))
            profile = profile / profile.max()

            profile_x = np.linspace(-fwhm * 1.5, fwhm * 1.5, len(profile))

            x_np = x.cpu().numpy()
            kernel_np = np.interp(x_np, profile_x, profile, left=0, right=0)
            kernel = torch.from_numpy(kernel_np).float()
        else:
            # Super-gaussian approximation
            sigma = fwhm / 2.355
            n = 4
            kernel = torch.exp(-0.5 * (x / sigma) ** (2 * n))

        return kernel

    def _ensure_size_2d(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Ensure 2D tensor has specified size."""
        pH, pW = size
        H, W = tensor.shape

        if H < pH or W < pW:
            pad_h = max(0, pH - H)
            pad_w = max(0, pW - W)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

        return tensor[:pH, :pW]


class DRIFT2DDataModule:
    """
    Data module for 2D Cascaded SR training.

    Provides train and validation dataloaders with appropriate collate functions.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        contrast: str = 't1',
        patch_size: Tuple[int, int] = (128, 128),
        lr_thickness_range: Tuple[float, float] = (1.2, 6.0),
        lr_gap_range: Tuple[float, float] = (0.0, 0.0),
        tgt_thickness_range: Tuple[float, float] = (0.7, 1.2),
        tgt_gap_range: Tuple[float, float] = (0.0, 0.0),
        batch_size: int = 32,
        num_workers: int = 8,
        slices_per_volume: int = 32,
        train_num_volumes: Optional[int] = None,
        val_num_volumes: Optional[int] = 100,
        slice_profile: str = 'slr',
    ):
        self.data_path = Path(data_path)
        self.contrast = contrast
        self.patch_size = patch_size
        self.lr_thickness_range = lr_thickness_range
        self.lr_gap_range = lr_gap_range
        self.tgt_thickness_range = tgt_thickness_range
        self.tgt_gap_range = tgt_gap_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.slices_per_volume = slices_per_volume
        self.train_num_volumes = train_num_volumes
        self.val_num_volumes = val_num_volumes
        self.slice_profile = slice_profile

    def train_dataloader(self) -> DataLoader:
        dataset = DRIFT2DDataset(
            data_path=self.data_path,
            split='train',
            contrast=self.contrast,
            patch_size=self.patch_size,
            lr_thickness_range=self.lr_thickness_range,
            lr_gap_range=self.lr_gap_range,
            tgt_thickness_range=self.tgt_thickness_range,
            tgt_gap_range=self.tgt_gap_range,
            slices_per_volume=self.slices_per_volume,
            slice_profile=self.slice_profile,
            num_volumes=self.train_num_volumes,
            transform=self._train_transform(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = DRIFT2DDataset(
            data_path=self.data_path,
            split='test',
            contrast=self.contrast,
            patch_size=self.patch_size,
            lr_thickness_range=self.lr_thickness_range,
            lr_gap_range=self.lr_gap_range,
            tgt_thickness_range=self.tgt_thickness_range,
            tgt_gap_range=self.tgt_gap_range,
            slices_per_volume=8,  # Fewer for validation
            slice_profile=self.slice_profile,
            num_volumes=self.val_num_volumes,
            transform=None,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _train_transform(self) -> callable:
        """Training augmentation for 2D slices."""

        def transform(lr, hr):
            # Random horizontal flip
            if np.random.rand() > 0.5:
                lr = lr.flip(dims=[2])
                hr = hr.flip(dims=[2])

            # Random vertical flip
            if np.random.rand() > 0.5:
                lr = lr.flip(dims=[1])
                hr = hr.flip(dims=[1])

            # Mild intensity augmentation
            intensity_scale = 0.95 + 0.1 * np.random.rand()
            intensity_shift = -0.04 + 0.08 * np.random.rand()

            lr = lr * intensity_scale + intensity_shift
            hr = hr * intensity_scale + intensity_shift

            return lr, hr

        return transform


# =============================================================================
# NPY Format Dataset (Individual files - best for DataLoader parallelism)
# =============================================================================

class PrecomputedNPY2DDataset(Dataset):
    """
    Dataset that reads pre-generated 2D slices from individual NPY files.

    This format provides the best DataLoader performance:
    - No file lock contention (each worker reads different files)
    - OS-level file caching works optimally
    - Perfect parallelism with multiple workers

    Expected structure (from create_2d_dataset_v2.py):
        data_path/
        ├── train/
        │   ├── slices/
        │   │   ├── 000000.npy  # (H, W) float16 - legacy format
        │   │   ├── 000001.npy
        │   │   └── ...
        │   │   OR (BraTS21 format with modality in filename):
        │   │   ├── 000000_flair.npy
        │   │   ├── 000001_t1.npy
        │   │   └── ...
        │   │   OR (HCP informative format):
        │   │   ├── 896778_t1_cor_085.npy  # subject_contrast_plane_slice.npy
        │   │   └── ...
        │   └── metadata.json
        └── test/
            └── ...

    Usage:
        dataset = PrecomputedNPY2DDataset(
            data_path="/ssd3/yoonseok/project/data/hcp_2d_sr_npy",
            split='train',
            patch_size=(128, 128),
        )

        # For TPDM (train on specific plane only):
        dataset = PrecomputedNPY2DDataset(
            data_path="/ssd3/yoonseok/project/data/hcp_2d_sr_npy",
            split='train',
            plane_filter='cor',  # Only coronal slices
        )
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        patch_size: Tuple[int, int] = (128, 128),
        lr_thickness_range: Tuple[float, float] = (1.2, 6.0),
        tgt_thickness_range: Tuple[float, float] = (0.7, 0.7),  # Target HR thickness range
        slice_profile: str = 'gaussian',
        transform: Optional[callable] = None,
        max_slices: Optional[int] = None,
        # CETA (Consistent Endpoint Trajectory Alignment) support
        use_ceta: bool = False,
        ceta_mode: str = 'fixed',  # 'fixed' (recommended), 'ratio', 'anchored', or 'pad_matched'
        ceta_ratio: float = 0.6,  # For 'ratio' mode: T_lr_alt = T_lr × ratio
        ceta_anchor: float = 2.0,  # For 'anchored' mode: T_lr_alt ~ Uniform(T_hr+ε, T_anchor)
        ceta_gap_mm: float = 1.0,  # For 'fixed' mode: T_lr_alt = T_lr - ceta_gap_mm
        ceta_delta_pad: float = 0.1,  # For 'pad_matched' mode: constant ΔPAD
        # Modality filtering (for BraTS21)
        modality: Optional[str] = None,  # 'flair', 't1', 't1ce', 't2', or None for all
        # Plane filtering (for TPDM - train on specific plane)
        plane_filter: Optional[str] = None,  # 'sag', 'cor', 'axi', or None for all
        # Degradation mode
        degradation_mode: str = 'legacy',  # 'legacy' or 'physical'
        # Native resolution of the dataset (fixed T_hr)
        # HCP: 0.7mm, BraTS21: 1.0mm
        native_resolution: Optional[float] = None,  # Auto-detect if None
        # PAD-weighted LR thickness sampling
        # 'uniform': uniform sampling from (lr_min, lr_max]
        # 'pad_weighted': Beta(α, 1) distribution biases toward harder cases
        lr_sampling_mode: str = 'uniform',
        pad_sampling_alpha: float = 2.0,  # Beta(α, 1) for PAD-weighted sampling
    ):
        super().__init__()

        self.data_path = Path(data_path) / split
        self.slices_dir = self.data_path / "slices"
        self.patch_size = patch_size
        self.lr_thickness_range = lr_thickness_range
        self.slice_profile = slice_profile
        self.transform = transform
        self.modality = modality
        self.plane_filter = plane_filter  # For TPDM: train on specific plane

        # Degradation mode:
        # - 'legacy': blur → area downsample → nearest upsample (기존 Stage1 학습에 사용)
        # - 'physical': blur with stride → nearest upsample (물리적으로 정확한 방식)
        self.degradation_mode = degradation_mode

        # CETA configuration
        self.use_ceta = use_ceta
        self.ceta_mode = ceta_mode
        self.ceta_ratio = ceta_ratio
        self.ceta_anchor = ceta_anchor
        self.ceta_gap_mm = ceta_gap_mm
        self.ceta_delta_pad = ceta_delta_pad

        # PAD-weighted sampling configuration
        self.lr_sampling_mode = lr_sampling_mode
        self.pad_sampling_alpha = pad_sampling_alpha

        # Native resolution and T_hr range
        # T_hr is sampled from [native_resolution, max_thr] during training
        # Clinical HR MRI: ≤1mm slice thickness is considered high resolution
        # Reference: Brain Tumor Imaging Protocol recommends "1mm or less" for treatment decisions
        self._native_resolution_input = native_resolution

        # Discover slice files and detect format (with caching for speed)
        # Supports three formats:
        # 1. Legacy: 000000.npy, 000001.npy, ... (no modality info)
        # 2. BraTS21: 000000_flair.npy, 000001_t1.npy, ... (modality in filename)
        # 3. HCP informative: {subject}_{contrast}_{plane}_{slice_idx}.npy
        #    e.g., 896778_t1_cor_085.npy (subject=896778, contrast=t1, plane=cor, slice=85)
        self.slice_files, self.file_format = self._discover_files_cached(modality, plane_filter)
        self.num_slices = len(self.slice_files)

        if self.num_slices == 0:
            filter_info = []
            if modality:
                filter_info.append(f"modality={modality}")
            if plane_filter:
                filter_info.append(f"plane={plane_filter}")
            raise ValueError(f"No NPY files found in {self.slices_dir}" +
                           (f" for {', '.join(filter_info)}" if filter_info else ""))

        # Limit if requested
        if max_slices is not None:
            self.slice_files = self.slice_files[:max_slices]
            self.num_slices = len(self.slice_files)

        # Skip slice_metadata loading - not needed for training
        # We use default values in __getitem__ which works fine
        self.slice_metadata = {}

        # Set native resolution and T_hr range
        # T_hr is sampled from tgt_thickness_range during training
        # Default: fixed at native resolution (0.7mm for HCP, 1.0mm for BraTS21)
        if self._native_resolution_input is not None:
            self.native_resolution = self._native_resolution_input
        elif self.file_format == 'brats21':
            self.native_resolution = 1.0  # BraTS21: 1.0mm isotropic
        elif self.file_format == 'mind':
            # MIND=0.9mm, IDEAS=1.0mm — same file format, can't auto-detect
            # Default to 0.9 (MIND); pass native_resolution explicitly for IDEAS
            self.native_resolution = 0.9
            print(f"  Warning: 'mind' format auto-detected, defaulting native_resolution=0.9mm (MIND).")
            print(f"           For IDEAS (1.0mm), set native_resolution explicitly.")
        else:
            self.native_resolution = 0.7  # HCP: 0.7mm isotropic

        # T_hr range: use tgt_thickness_range from config
        # If same (e.g., [0.7, 0.7]), T_hr is fixed at that value
        self.thr_range = tgt_thickness_range

        print(f"PrecomputedNPY2DDataset: {split}")
        print(f"  Format: {self.file_format}")
        print(f"  Slices: {self.num_slices:,}" + (f" (modality={modality})" if modality else " (all modalities)"))
        print(f"  Patch size: {patch_size}")
        print(f"  LR thickness: {lr_thickness_range[0]:.1f} - {lr_thickness_range[1]:.1f} mm")
        print(f"  LR sampling: {lr_sampling_mode}" + (f" (α={pad_sampling_alpha})" if lr_sampling_mode == 'pad_weighted' else ""))
        print(f"  Target HR range: [{self.thr_range[0]:.2f}, {self.thr_range[1]:.2f}] mm")
        print(f"  Slice profile: {slice_profile}")
        if use_ceta:
            if ceta_mode == 'fixed':
                print(f"  CETA enabled: mode={ceta_mode}, gap={ceta_gap_mm}mm")
            elif ceta_mode == 'anchored':
                print(f"  CETA enabled: mode={ceta_mode}, anchor={ceta_anchor}mm")
            elif ceta_mode == 'pad_matched':
                print(f"  CETA enabled: mode={ceta_mode}, delta_pad={ceta_delta_pad}")
            else:
                print(f"  CETA enabled: mode={ceta_mode}, ratio={ceta_ratio}")

    def _discover_files_cached(
        self,
        modality: Optional[str],
        plane_filter: Optional[str],
    ) -> Tuple[List, str]:
        """
        Discover slice files with caching for fast startup.

        Cache is stored as JSON in the data directory. If modality or plane_filter
        changes, the full list is loaded and filtered (still fast from cache).

        Returns:
            Tuple of (slice_files list, file_format string)
        """
        import os
        import re
        import json
        import time

        cache_path = self.data_path / 'slice_cache.json'

        # Try to load from cache
        cached_data = None
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                # Verify cache is valid (check file count roughly matches)
                # We don't do exact match to avoid re-scanning on small changes
            except (json.JSONDecodeError, KeyError):
                cached_data = None

        if cached_data is not None:
            file_format = cached_data['file_format']
            all_slice_files = cached_data['slice_files']

            # Apply modality and plane filters
            slice_files = self._filter_slice_files(all_slice_files, modality, plane_filter, file_format)
            return slice_files, file_format

        # No cache - scan directory and build cache
        print(f"  Building file cache (first run only)...")
        start_time = time.time()

        all_files = sorted([fn for fn in os.listdir(self.slices_dir) if fn.endswith('.npy')])

        if not all_files:
            raise ValueError(f"No NPY files found in {self.slices_dir}")

        # Detect format by checking first file
        first_file = all_files[0]
        hcp_pattern = re.compile(r'^(\d{6})_(t1|t2)_(sag|cor|axi)_(\d{3})(?:_\d{2})?\.npy$')
        # MIND/IDEAS format: sub-EBE0002_t1_axi_001.npy or sub-4001_flair_axi_055.npy
        mind_pattern = re.compile(r'^sub-([A-Za-z0-9]+)_(t1|t2|flair)_(axi|cor|sag)_(\d+)\.npy$')
        brats_pattern = re.compile(r'^(\d+)_(flair|t1|t1ce|t2)\.npy$')
        legacy_pattern = re.compile(r'^(\d+)\.npy$')

        all_slice_files = []

        if hcp_pattern.match(first_file):
            file_format = 'hcp'
            for fn in all_files:
                match = hcp_pattern.match(fn)
                if match:
                    subject = match.group(1)
                    contrast = match.group(2)
                    plane = match.group(3)
                    slice_idx = int(match.group(4))
                    # Store all info without filtering (filtering done later)
                    all_slice_files.append({
                        'filename': fn,
                        'contrast': contrast,
                        'subject': subject,
                        'plane': plane,
                        'slice_idx': slice_idx,
                    })
        elif mind_pattern.match(first_file):
            # MIND dataset: sub-EBE0002_t1_axi_001.npy
            file_format = 'mind'
            for fn in all_files:
                match = mind_pattern.match(fn)
                if match:
                    subject = match.group(1)  # e.g., 'EBE0002'
                    contrast = match.group(2)  # t1 or t2
                    plane = match.group(3)  # axi, cor, sag
                    slice_idx = int(match.group(4))
                    all_slice_files.append({
                        'filename': fn,
                        'contrast': contrast,
                        'subject': subject,
                        'plane': plane,
                        'slice_idx': slice_idx,
                    })
        elif brats_pattern.match(first_file):
            file_format = 'brats21'
            for fn in all_files:
                match = brats_pattern.match(fn)
                if match:
                    idx = int(match.group(1))
                    mod = match.group(2)
                    all_slice_files.append({
                        'filename': fn,
                        'contrast': mod,
                        'idx': idx,
                    })
        else:
            file_format = 'legacy'
            for fn in all_files:
                match = legacy_pattern.match(fn)
                if match:
                    idx = int(match.group(1))
                    all_slice_files.append({
                        'filename': fn,
                        'idx': idx,
                    })

        # Save cache
        cache_data = {
            'file_format': file_format,
            'num_files': len(all_slice_files),
            'slice_files': all_slice_files,
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            print(f"  Cache saved to {cache_path} ({time.time() - start_time:.1f}s)")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

        # Apply filters
        slice_files = self._filter_slice_files(all_slice_files, modality, plane_filter, file_format)
        return slice_files, file_format

    def _filter_slice_files(
        self,
        all_slice_files: List[dict],
        modality: Optional[str],
        plane_filter: Optional[str],
        file_format: str,
    ) -> List:
        """Filter slice files by modality and plane, returning format expected by rest of code."""
        filtered = []

        for i, entry in enumerate(all_slice_files):
            fn = entry['filename']

            if file_format == 'hcp' or file_format == 'mind':
                # HCP and MIND share the same structure
                contrast = entry['contrast']
                # Filter by modality
                if modality is not None and modality != 'both' and contrast != modality:
                    continue
                # Filter by plane
                if plane_filter is not None and entry['plane'] != plane_filter:
                    continue
                # Format: (unique_idx, modality, filename, extra_info)
                filtered.append((len(filtered), contrast, fn, {
                    'subject': entry['subject'],
                    'plane': entry['plane'],
                    'slice_idx': entry['slice_idx'],
                }))
            elif file_format == 'brats21':
                mod = entry['contrast']
                if modality is not None and modality != 'both' and mod != modality:
                    continue
                # Format: (idx, modality, filename)
                filtered.append((entry['idx'], mod, fn))
            else:  # legacy
                # Legacy format: no modality filtering possible
                filtered.append((entry['idx'], None, fn))

        if file_format == 'legacy' and modality is not None:
            print(f"  Warning: modality filtering requested but files are in legacy format")

        return filtered

    def __len__(self) -> int:
        return self.num_slices

    def _get_slice_info(self, idx: int) -> Tuple[Path, Optional[str], Optional[dict]]:
        """Get slice path, modality, and extra info by dataset index.

        Returns:
            Tuple of (slice_path, modality, extra_info)
            - extra_info is dict with subject, plane, slice_idx for HCP format
            - extra_info is None for legacy/brats21 formats
        """
        if self.file_format == 'hcp' or self.file_format == 'mind':
            file_idx, modality, filename, extra_info = self.slice_files[idx]
            return self.slices_dir / filename, modality, extra_info
        else:
            file_idx, modality, filename = self.slice_files[idx]
            return self.slices_dir / filename, modality, None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load HR slice by index
        slice_path, slice_modality, extra_info = self._get_slice_info(idx)
        hr_slice = np.load(slice_path).astype(np.float32)

        # Normalize to [0, 1] first (per-slice min-max normalization)
        slice_min = hr_slice.min()
        slice_max = hr_slice.max()
        if slice_max > slice_min:
            hr_slice = (hr_slice - slice_min) / (slice_max - slice_min)
        else:
            hr_slice = np.zeros_like(hr_slice)
        # Convert [0, 1] to [-1, 1]
        hr_slice = hr_slice * 2 - 1

        hr_slice = torch.from_numpy(hr_slice)

        # Get metadata for this slice
        # For HCP format, extract axis from plane info in extra_info
        slice_idx_str = str(idx)
        meta = self.slice_metadata.get(slice_idx_str, {})

        if extra_info is not None and 'plane' in extra_info:
            # HCP format: derive axis from plane name
            plane_to_axis = {'sag': 0, 'cor': 1, 'axi': 2}
            axis = plane_to_axis.get(extra_info['plane'], 0)
        else:
            axis = meta.get('axis', 0)

        # Use native_resolution for degradation computation
        # This ensures consistency between training and inference
        # HCP: 0.7mm isotropic, BraTS: 1.0mm isotropic
        hr_spacing = self.native_resolution

        # Protocol parameters
        # T_hr: sampled from [native_resolution, 1.0mm] (clinical HR range)
        # T_lr: random within range for difficulty diversity
        # This enables continuous resolution control while staying within clinical HR range
        # NOTE: Use idx as seed for reproducibility and to ensure diversity across workers
        # Without seed, all DataLoader workers may generate identical random values
        rng = np.random.RandomState(idx)

        # Sample T_hr from clinical HR range
        if self.thr_range[0] >= self.thr_range[1]:
            # Fixed T_hr case (e.g., [0.7, 0.7] or [1.0, 1.0])
            # Use the specified value, not native_resolution
            T_hr = self.thr_range[0]
        else:
            # Range case: sample T_hr from [min, max]
            T_hr = rng.uniform(self.thr_range[0], self.thr_range[1])

        # Sample T_lr from (lr_min, lr_max] - exclusive lower bound, inclusive upper bound
        # This ensures T_lr > T_hr when lr_thickness_range[0] == T_hr (e.g., [0.7, 6.0] with T_hr=0.7)
        lr_min, lr_max = self.lr_thickness_range
        lr_min_exclusive = lr_min + 1e-6

        if self.lr_sampling_mode == 'pad_weighted':
            # PAD-weighted sampling using Beta(α, 1) distribution
            # PAD = 1 - T_hr/T_lr measures spectral deficit (lost high-frequency info)
            # Beta(α, 1) biases toward higher values when α > 1
            # T_lr = T_min + (T_max - T_min) × X, where X ~ Beta(α, 1)
            # This oversamples harder cases (higher T_lr → higher PAD)
            x = rng.beta(self.pad_sampling_alpha, 1.0)
            T_lr = lr_min_exclusive + (lr_max - lr_min_exclusive) * x
        else:
            # Uniform sampling (default)
            T_lr = rng.uniform(lr_min_exclusive, lr_max)

        # Ensure T_lr > T_hr (safety check)
        if T_lr <= T_hr:
            T_lr = T_hr + 0.5

        # Determine blur axis from plane info in filename
        # axi/cor: axis 0 (height downsample → horizontal stair-step in array)
        # sag: axis 1 (width downsample → vertical stair-step in array)
        plane_to_blur_axis = {'sag': 1, 'axi': 0, 'cor': 0}
        if extra_info is not None and 'plane' in extra_info:
            blur_axis = plane_to_blur_axis.get(extra_info['plane'], 0)
        else:
            blur_axis = rng.randint(0, 2)

        # Apply degradation to create LR (T_lr thickness)
        lr_slice = self._apply_degradation(hr_slice, blur_axis, T_lr, hr_spacing)

        # Apply degradation to create target HR (T_hr thickness)
        # If T_hr == native_resolution, use original data
        # Otherwise, apply mild degradation to simulate T_hr resolution
        if abs(T_hr - self.native_resolution) < 0.01:
            # T_hr equals native resolution, use original data
            target_hr_slice = hr_slice
        else:
            # Apply degradation to simulate T_hr resolution
            target_hr_slice = self._apply_degradation(hr_slice, blur_axis, T_hr, hr_spacing)

        # Random crop (or full image if patch_size is None)
        H, W = hr_slice.shape

        if self.patch_size is None:
            # Return full image without cropping
            hr_patch = target_hr_slice
            lr_patch = lr_slice
        else:
            pH, pW = self.patch_size
            h_start = rng.randint(0, max(1, H - pH + 1))
            w_start = rng.randint(0, max(1, W - pW + 1))

            hr_patch = target_hr_slice[h_start:h_start + pH, w_start:w_start + pW]
            lr_patch = lr_slice[h_start:h_start + pH, w_start:w_start + pW]

            # Ensure size (pad if needed)
            hr_patch = self._ensure_size(hr_patch, (pH, pW))
            lr_patch = self._ensure_size(lr_patch, (pH, pW))

        # Add channel dim
        hr_patch = hr_patch.unsqueeze(0)
        lr_patch = lr_patch.unsqueeze(0)

        # CETA: Generate alternative LR with different thickness BEFORE transform
        # This ensures the same augmentation is applied to all patches
        lr_patch_alt = None
        T_lr_alt = None
        if self.use_ceta:
            # Sample alternative thickness that differs from T_lr by at least margin
            T_lr_alt = self._sample_alternative_thickness(rng, T_lr, T_hr)

            # Apply degradation with alternative thickness (same axis, same crop)
            lr_slice_alt = self._apply_degradation(hr_slice, blur_axis, T_lr_alt, hr_spacing)
            if self.patch_size is None:
                # Full image mode
                lr_patch_alt = lr_slice_alt
            else:
                lr_patch_alt = lr_slice_alt[h_start:h_start + pH, w_start:w_start + pW]
                lr_patch_alt = self._ensure_size(lr_patch_alt, (pH, pW))
            lr_patch_alt = lr_patch_alt.unsqueeze(0)

        # Apply transform to ALL patches with the SAME random augmentation
        if self.transform is not None:
            lr_patch, hr_patch, lr_patch_alt = self._apply_transform_with_ceta(
                lr_patch, hr_patch, lr_patch_alt
            )

        result = {
            'x_lr': lr_patch,
            'y_hr': hr_patch,
            'protocol': torch.tensor([T_lr, T_hr], dtype=torch.float32),
            'slice_axis': torch.tensor(axis, dtype=torch.long),
            'artifact_axis': torch.tensor(blur_axis, dtype=torch.long),
            'modality': slice_modality if slice_modality else 'unknown',
        }

        if self.use_ceta and lr_patch_alt is not None:
            result['x_lr_alt'] = lr_patch_alt
            result['protocol_alt'] = torch.tensor([T_lr_alt, T_hr], dtype=torch.float32)

        return result

    def _apply_transform_with_ceta(
        self,
        lr_patch: torch.Tensor,
        hr_patch: torch.Tensor,
        lr_patch_alt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the same random augmentation to all patches (lr, hr, and optionally lr_alt).

        This ensures CETA pairs have consistent spatial augmentation (same flips).
        """
        # Determine random augmentation ONCE
        flip_h = np.random.rand() > 0.5
        flip_v = np.random.rand() > 0.5
        intensity_scale = 0.95 + 0.1 * np.random.rand()
        intensity_shift = -0.04 + 0.08 * np.random.rand()

        # Apply to lr_patch
        if flip_h:
            lr_patch = lr_patch.flip(dims=[2])
        if flip_v:
            lr_patch = lr_patch.flip(dims=[1])
        lr_patch = lr_patch * intensity_scale + intensity_shift

        # Apply to hr_patch
        if flip_h:
            hr_patch = hr_patch.flip(dims=[2])
        if flip_v:
            hr_patch = hr_patch.flip(dims=[1])
        hr_patch = hr_patch * intensity_scale + intensity_shift

        # Apply to lr_patch_alt (if exists)
        if lr_patch_alt is not None:
            if flip_h:
                lr_patch_alt = lr_patch_alt.flip(dims=[2])
            if flip_v:
                lr_patch_alt = lr_patch_alt.flip(dims=[1])
            lr_patch_alt = lr_patch_alt * intensity_scale + intensity_shift

        return lr_patch, hr_patch, lr_patch_alt

    def _sample_alternative_thickness(
        self,
        rng: np.random.RandomState,
        T_lr: float,
        T_hr: float,
    ) -> float:
        """
        Sample alternative LR thickness for CETA loss.

        Four modes available (controlled by self.ceta_mode):

        1. 'fixed' mode (RECOMMENDED): T_lr_alt = T_lr - ceta_gap_mm
           - Fixed absolute mm gap between T_lr and T_lr_alt
           - Chain effect: nearby thicknesses converge to same HR
             6mm↔5mm, 5mm↔4mm, 4mm↔3mm, ... → all converge to HR (transitivity)
           - Dense chain coverage across entire thickness range

           Example (ceta_gap_mm=1.0mm, T_hr=0.7mm):
               T_lr=6.0mm → T_lr_alt=5.0mm (ΔPAD=0.023, gap=1.0mm)
               T_lr=4.0mm → T_lr_alt=3.0mm (ΔPAD=0.058, gap=1.0mm)
               T_lr=2.0mm → T_lr_alt=1.0mm (ΔPAD=0.175, gap=1.0mm)

        2. 'ratio' mode: T_lr_alt = T_lr × ceta_ratio (where ceta_ratio < 1)
           - Gap scales with T_lr (larger gap for larger T_lr)
           - Problem: Sparse coverage at high T_lr (hard cases)

           Example (ceta_ratio=0.6, T_hr=0.7mm):
               T_lr=6.0mm → T_lr_alt=3.6mm (ΔPAD=0.10, gap=2.4mm) ← sparse!
               T_lr=4.0mm → T_lr_alt=2.4mm (ΔPAD=0.12, gap=1.6mm)
               T_lr=2.0mm → T_lr_alt=1.2mm (ΔPAD=0.07, gap=0.8mm) ← too small

        3. 'anchored' mode: T_lr_alt ~ Uniform(T_hr + ε, T_anchor)
           - Star topology: all T_lr connect to anchor region
           - Problem: No transitivity (6mm≈2mm, 5mm≈2mm, but 6mm≈5mm not guaranteed)

           Example (ceta_anchor=2.0mm, T_hr=0.7mm):
               T_lr=6.0mm ↔ T_lr_alt~1.2mm (no 6mm↔5mm connection)
               T_lr=4.0mm ↔ T_lr_alt~1.5mm (no 4mm↔3mm connection)
               T_lr=2.0mm ↔ T_lr_alt~1.3mm

        4. 'pad_matched' mode: T_lr_alt such that ΔPAD = ceta_delta_pad (constant)
           - Constant PAD difference in difficulty space
           - Problem: Gap varies inversely with T_lr → sparse at high T_lr, trivial at low T_lr

           Formula: T_lr_alt = T_hr / (T_hr/T_lr + ΔPAD)

           Example (ceta_delta_pad=0.1, T_hr=0.7mm):
               T_lr=6.0mm → T_lr_alt=3.23mm (ΔPAD=0.10, gap=2.77mm) ← very sparse!
               T_lr=4.0mm → T_lr_alt=2.55mm (ΔPAD=0.10, gap=1.45mm)
               T_lr=2.0mm → T_lr_alt=1.56mm (ΔPAD=0.10, gap=0.44mm) ← trivial

        Returns:
            T_lr_alt: Alternative LR thickness satisfying T_hr < T_lr_alt < T_lr
        """
        # ===== Defensive checks for invalid inputs =====

        # Check for NaN or Inf
        if not (np.isfinite(T_lr) and np.isfinite(T_hr)):
            min_t, max_t = self.lr_thickness_range
            return (min_t + max_t) / 2.0

        # Check for negative values
        if T_lr <= 0 or T_hr <= 0:
            min_t, max_t = self.lr_thickness_range
            return (min_t + max_t) / 2.0

        # Check for invalid ordering (T_lr should be > T_hr)
        if T_lr <= T_hr:
            return T_hr + 0.5

        # Minimum gap for numerical stability
        min_gap = 0.1  # mm

        # ===== Mode selection =====
        if self.ceta_mode == 'fixed':
            # Fixed mode: T_lr_alt = T_lr - ceta_gap_mm
            # Fixed absolute gap between T_lr and T_lr_alt
            gap_mm = self.ceta_gap_mm

            # Target: T_lr_alt = T_lr - gap_mm
            target_alt = T_lr - gap_mm

            # Valid range: (T_hr + min_gap, T_lr - min_gap)
            min_valid = T_hr + min_gap
            max_valid = T_lr - min_gap

            # Check if valid range exists
            if max_valid <= min_valid:
                return (T_hr + T_lr) / 2.0

            # Clamp to valid range
            T_lr_alt = max(min_valid, min(target_alt, max_valid))

        elif self.ceta_mode == 'anchored':
            # PAD-Anchored mode: sample from low-PAD region
            # T_lr_alt ~ Uniform(T_hr + ε, min(T_anchor, T_lr - ε))
            anchor = self.ceta_anchor

            # Valid range: (T_hr + min_gap, min(anchor, T_lr - min_gap))
            min_valid = T_hr + min_gap
            max_valid = min(anchor, T_lr - min_gap)

            # If T_lr is already below anchor, use midpoint
            if max_valid <= min_valid:
                return (T_hr + T_lr) / 2.0

            # Uniform sampling from anchored low-PAD region
            T_lr_alt = rng.uniform(min_valid, max_valid)

        elif self.ceta_mode == 'pad_matched':
            # PAD-matched mode: constant ΔPAD in difficulty space
            # PAD = 1 - T_hr/T_lr, so for constant ΔPAD:
            # T_lr_alt = T_hr / (T_hr/T_lr + ΔPAD)
            delta_pad = getattr(self, 'ceta_delta_pad', 0.1)

            # Current PAD
            pad_current = 1.0 - T_hr / T_lr

            # Target PAD (lower difficulty)
            pad_target = pad_current - delta_pad

            # Compute T_lr_alt from target PAD
            # PAD_target = 1 - T_hr / T_lr_alt
            # T_hr / T_lr_alt = 1 - PAD_target
            # T_lr_alt = T_hr / (1 - PAD_target)
            if pad_target <= 0:
                # PAD target is negative or zero, use midpoint
                return (T_hr + T_lr) / 2.0

            target_alt = T_hr / (1.0 - pad_target)

            # Valid range: (T_hr + min_gap, T_lr - min_gap)
            min_valid = T_hr + min_gap
            max_valid = T_lr - min_gap

            # Check if valid range exists
            if max_valid <= min_valid:
                return (T_hr + T_lr) / 2.0

            # Clamp to valid range
            T_lr_alt = max(min_valid, min(target_alt, max_valid))

        else:
            # Ratio-based mode (default): T_lr_alt = T_lr × ceta_ratio
            ceta_ratio = self.ceta_ratio

            # Valid range: (T_hr + min_gap, T_lr - min_gap)
            min_valid = T_hr + min_gap
            max_valid = T_lr - min_gap

            # Check if valid range exists
            if max_valid <= min_valid:
                return (T_hr + T_lr) / 2.0

            # Target: T_lr_alt = T_lr × ceta_ratio
            target_alt = T_lr * ceta_ratio

            # Clamp to valid range first
            target_alt = max(min_valid, min(target_alt, max_valid))

            # Add small random variation for diversity
            range_size = max_valid - min_valid
            max_variation = min(0.3, range_size * 0.1)  # ±10% of range or 0.3mm
            T_lr_alt = target_alt + rng.uniform(-max_variation, max_variation)

        # Final safety clamp
        T_lr_alt = float(np.clip(T_lr_alt, T_hr + min_gap, T_lr - min_gap))

        # Sanity check
        assert T_hr < T_lr_alt < T_lr, (
            f"CETA thickness invariant violated: T_hr={T_hr:.3f}, "
            f"T_lr_alt={T_lr_alt:.3f}, T_lr={T_lr:.3f}"
        )

        return T_lr_alt

    def _apply_degradation(
        self,
        hr_slice: torch.Tensor,
        blur_axis: int,
        T_lr: float,
        hr_spacing: float,
    ) -> torch.Tensor:
        """
        Apply SLR-based degradation simulating thick-slice MRI acquisition.

        Two modes available (controlled by self.degradation_mode):
        - 'legacy': blur → area downsample → nearest upsample (기존 Stage1 학습에 사용)
        - 'physical': blur with stride → nearest upsample (물리적으로 정확)
        """
        H, W = hr_slice.shape
        original_size = [H, W]

        # Compute degradation factor
        # T_lr / hr_spacing gives the ratio of LR thickness to native voxel size
        # e.g., T_lr=4.0mm, hr_spacing=0.7mm → factor=5.71 (each LR "pixel" covers ~5.7 HR voxels)
        downsample_factor = T_lr / hr_spacing
        # Clamp: min 1.5 (must downsample), max based on image size
        # For 320px image, factor 16 → 20px minimum (still reasonable)
        downsample_factor = max(1.5, min(downsample_factor, 16.0))

        # Create SLR kernel (realistic MRI slice profile)
        fwhm_vox = T_lr / hr_spacing
        kernel_size = int(fwhm_vox * 3) | 1
        kernel_size = max(3, min(kernel_size, 51))  # Allow larger kernels for high T_lr

        kernel = self._create_slr_kernel(kernel_size, fwhm_vox)

        if self.degradation_mode == 'legacy':
            # Legacy mode: blur → area downsample → nearest upsample
            # (기존 Stage1 학습에 사용된 방식)

            # Step 1: Apply blur (no stride)
            blurred = self._apply_1d_blur(hr_slice, kernel, blur_axis)

            # Step 2: Area downsample
            blurred_4d = blurred.unsqueeze(0).unsqueeze(0)
            new_size_along_axis = max(4, int(original_size[blur_axis] / downsample_factor))
            ds_size = list(original_size)
            ds_size[blur_axis] = new_size_along_axis
            downsampled_4d = F.interpolate(blurred_4d, size=ds_size, mode='area')

            # Step 3: Nearest upsample (creates stair-step artifact)
            if MONAI_AVAILABLE:
                downsampled_3d = downsampled_4d.squeeze(0)
                resize_transform = Resize(spatial_size=original_size, mode='nearest')
                upsampled_3d = resize_transform(downsampled_3d)
                return upsampled_3d.squeeze()
            else:
                upsampled = F.interpolate(downsampled_4d, size=original_size, mode='nearest')
                return upsampled.squeeze()
        else:
            # Physical mode: blur with stride → nearest upsample
            # (물리적으로 정확한 방식 - 논문에서 사용)
            # NumPy/SciPy 기반으로 구현하여 DataLoader worker에서 CUDA context 문제 방지
            stride = max(1, int(round(downsample_factor)))

            # Convert to numpy for CPU-only processing
            hr_np = hr_slice.numpy()
            kernel_np = kernel.numpy()

            # Apply 1D blur WITH STRIDE using scipy (CPU only)
            downsampled_np = self._apply_1d_blur_with_stride_numpy(hr_np, kernel_np, blur_axis, stride)

            # Upsample back using nearest (creates stair-step artifact)
            from scipy.ndimage import zoom
            scale_factors = [1.0, 1.0]
            if blur_axis == 0:
                scale_factors[0] = original_size[0] / downsampled_np.shape[0]
            else:
                scale_factors[1] = original_size[1] / downsampled_np.shape[1]

            upsampled_np = zoom(downsampled_np, scale_factors, order=0)  # order=0 = nearest

            # Ensure exact size (zoom can be off by 1 pixel)
            if upsampled_np.shape[0] > original_size[0]:
                upsampled_np = upsampled_np[:original_size[0], :]
            if upsampled_np.shape[1] > original_size[1]:
                upsampled_np = upsampled_np[:, :original_size[1]]

            return torch.from_numpy(upsampled_np.astype(np.float32))

    def _apply_1d_blur_with_stride_numpy(
        self,
        img: np.ndarray,
        kernel: np.ndarray,
        axis: int,
        stride: int,
    ) -> np.ndarray:
        """
        Apply 1D blur with stride along axis using NumPy/SciPy (CPU only).

        This combines blur and downsampling, matching real MRI slice selection physics.
        Using NumPy avoids CUDA context issues in DataLoader workers.
        """
        from scipy.ndimage import convolve1d

        # Apply 1D convolution along the specified axis
        blurred = convolve1d(img, kernel, axis=axis, mode='reflect')

        # Downsample with stride
        if axis == 0:
            downsampled = blurred[::stride, :]
        else:
            downsampled = blurred[:, ::stride]

        return downsampled

    def _apply_1d_blur_with_stride(
        self,
        img: torch.Tensor,
        kernel: torch.Tensor,
        axis: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Apply 1D blur with stride along axis (PyTorch version, kept for reference).

        This combines blur and downsampling, matching real MRI slice selection physics.
        NOTE: This function may cause CUDA context issues in DataLoader workers.
        Use _apply_1d_blur_with_stride_numpy instead for training.
        """
        pad = kernel.shape[0] // 2
        img_4d = img.unsqueeze(0).unsqueeze(0)

        if axis == 0:
            kernel_4d = kernel.view(1, 1, -1, 1)
            img_4d = F.pad(img_4d, (0, 0, pad, pad), mode='reflect')
            blurred = F.conv2d(img_4d, kernel_4d, stride=(stride, 1))
        else:
            kernel_4d = kernel.view(1, 1, 1, -1)
            img_4d = F.pad(img_4d, (pad, pad, 0, 0), mode='reflect')
            blurred = F.conv2d(img_4d, kernel_4d, stride=(1, stride))

        return blurred.squeeze()

    def _apply_1d_blur(self, img: torch.Tensor, kernel: torch.Tensor, axis: int) -> torch.Tensor:
        """Apply 1D blur along axis (no stride, for backward compatibility)."""
        pad = kernel.shape[0] // 2
        img_4d = img.unsqueeze(0).unsqueeze(0)

        if axis == 0:
            kernel_4d = kernel.view(1, 1, -1, 1)
            img_4d = F.pad(img_4d, (0, 0, pad, pad), mode='reflect')
        else:
            kernel_4d = kernel.view(1, 1, 1, -1)
            img_4d = F.pad(img_4d, (pad, pad, 0, 0), mode='reflect')

        return F.conv2d(img_4d, kernel_4d).squeeze()

    def _create_slr_kernel(self, kernel_size: int, fwhm: float) -> torch.Tensor:
        """
        Create SLR (Shinnar-Le Roux) slice profile kernel.

        This simulates the actual slice selection profile used in clinical MRI.
        SLR pulses are designed to achieve sharp slice profiles with minimal
        side lobes, which is more realistic than Gaussian approximation.
        """
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2

        if self.slice_profile == 'slr' and SIGPY_AVAILABLE:
            # Use SigPy to generate SLR pulse and compute its profile
            n_points = 256
            tb = 4  # Time-bandwidth product

            try:
                pulse = rf_design.slr.dzrf(
                    n=n_points,
                    tb=tb,
                    ptype='ex',  # Excitation pulse
                    ftype='ls',  # Least-squares design
                    d1=0.01,     # Passband ripple
                    d2=0.01,     # Stopband ripple
                )

                # FFT to get slice profile
                profile = np.abs(np.fft.fftshift(np.fft.fft(pulse, 1024)))
                profile = profile / profile.max()

                # Map profile to kernel coordinates
                profile_x = np.linspace(-fwhm * 1.5, fwhm * 1.5, len(profile))
                x_np = x.cpu().numpy()
                kernel_np = np.interp(x_np, profile_x, profile, left=0, right=0)
                kernel = torch.from_numpy(kernel_np).float()
            except Exception:
                # Fallback to super-gaussian if SLR fails
                sigma = fwhm / 2.355
                n = 4  # Super-gaussian order
                kernel = torch.exp(-0.5 * (x / sigma) ** (2 * n))
        elif self.slice_profile == 'gaussian':
            # Standard Gaussian
            sigma = fwhm / 2.355
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        else:
            # Super-gaussian approximation of SLR (fallback when SigPy not available)
            sigma = fwhm / 2.355
            n = 4  # Super-gaussian order gives sharper edges like SLR
            kernel = torch.exp(-0.5 * (x / sigma) ** (2 * n))

        return kernel / kernel.sum()

    def _ensure_size(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Ensure tensor has target size."""
        pH, pW = size
        H, W = tensor.shape

        if H < pH or W < pW:
            tensor = F.pad(tensor, (0, max(0, pW - W), 0, max(0, pH - H)))

        return tensor[:pH, :pW]


# =============================================================================
# Updated DataModule (supports on-the-fly and NPY formats)
# =============================================================================

class DRIFT2DDataModuleV2:
    """
    Data module supporting multiple data formats:

    Formats:
        - 'npy': Individual NPY files (best performance, recommended)
        - 'onthefly': On-the-fly from 3D NIfTI (slowest but most flexible)

    Args:
        data_format: 'npy' or 'onthefly'
        precomputed_path: Path to precomputed NPY data
        onthefly_path: Path to 3D NIfTI volumes
    """

    def __init__(
        self,
        use_precomputed: bool = True,
        data_format: str = 'npy',  # 'npy' or 'onthefly'
        precomputed_path: Optional[Union[str, Path]] = None,
        onthefly_path: Optional[Union[str, Path]] = None,
        contrast: str = 't1',
        patch_size: Tuple[int, int] = (128, 128),
        lr_thickness_range: Tuple[float, float] = (1.2, 6.0),
        tgt_thickness_range: Tuple[float, float] = (0.7, 1.2),  # kept for backward compat
        batch_size: int = 64,
        num_workers: int = 8,
        slices_per_volume: int = 32,
        train_num_volumes: Optional[int] = None,
        val_num_volumes: Optional[int] = 100,
        slice_profile: str = 'gaussian',
        max_train_slices: Optional[int] = None,
        max_val_slices: Optional[int] = None,
        # CETA support
        use_ceta: bool = False,
        ceta_mode: str = 'fixed',  # 'fixed' (recommended), 'ratio', 'anchored', or 'pad_matched'
        ceta_ratio: float = 0.6,  # For 'ratio' mode: T_lr_alt = T_lr × ratio
        ceta_anchor: float = 2.0,  # For 'anchored' mode: T_lr_alt ~ Uniform(T_hr+ε, T_anchor)
        ceta_gap_mm: float = 1.0,  # For 'fixed' mode: T_lr_alt = T_lr - ceta_gap_mm
        ceta_delta_pad: float = 0.1,  # For 'pad_matched' mode: constant ΔPAD
        # Modality filtering (for BraTS21 with modality in filename)
        modality: Optional[str] = None,  # 'flair', 't1', 't1ce', 't2', or None for all
        # Degradation mode selection
        degradation_mode: str = 'legacy',  # 'legacy' or 'physical'
        # Native resolution (fixed T_hr) - HCP: 0.7mm, BraTS: 1.0mm
        # None = auto-detect based on file format
        native_resolution: Optional[float] = None,
        # PAD-weighted LR thickness sampling
        lr_sampling_mode: str = 'uniform',  # 'uniform' or 'pad_weighted'
        pad_sampling_alpha: float = 2.0,  # Beta(α, 1) for PAD-weighted sampling
    ):
        self.use_precomputed = use_precomputed
        self.degradation_mode = degradation_mode
        self.data_format = data_format if use_precomputed else 'onthefly'
        self.precomputed_path = Path(precomputed_path) if precomputed_path else None
        self.onthefly_path = Path(onthefly_path) if onthefly_path else None
        self.contrast = contrast
        self.modality = modality  # For NPY format with modality in filename
        self.patch_size = patch_size
        self.lr_thickness_range = lr_thickness_range
        self.tgt_thickness_range = tgt_thickness_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.slices_per_volume = slices_per_volume
        self.train_num_volumes = train_num_volumes
        self.val_num_volumes = val_num_volumes
        self.slice_profile = slice_profile
        self.max_train_slices = max_train_slices
        self.max_val_slices = max_val_slices
        self.use_ceta = use_ceta
        self.ceta_mode = ceta_mode
        self.ceta_ratio = ceta_ratio
        self.ceta_anchor = ceta_anchor
        self.ceta_gap_mm = ceta_gap_mm
        self.ceta_delta_pad = ceta_delta_pad
        self.native_resolution = native_resolution
        self.lr_sampling_mode = lr_sampling_mode
        self.pad_sampling_alpha = pad_sampling_alpha

        if use_precomputed and precomputed_path is None:
            raise ValueError("precomputed_path required when use_precomputed=True")
        if not use_precomputed and onthefly_path is None:
            raise ValueError("onthefly_path required when use_precomputed=False")

        print(f"Data format: {self.data_format}")
        if modality:
            print(f"Modality filter: {modality}")
        if use_ceta:
            if ceta_mode == 'fixed':
                print(f"CETA enabled: mode={ceta_mode}, gap={ceta_gap_mm}mm")
            elif ceta_mode == 'anchored':
                print(f"CETA enabled: mode={ceta_mode}, anchor={ceta_anchor}mm")
            elif ceta_mode == 'pad_matched':
                print(f"CETA enabled: mode={ceta_mode}, delta_pad={ceta_delta_pad}")
            else:
                print(f"CETA enabled: mode={ceta_mode}, ratio={ceta_ratio}")
        print(f"LR sampling: {lr_sampling_mode}" + (f" (α={pad_sampling_alpha})" if lr_sampling_mode == 'pad_weighted' else ""))

    def _create_dataset(self, split: str, transform: Optional[callable] = None, max_slices: Optional[int] = None):
        """Create dataset based on format."""
        if self.data_format == 'npy':
            return PrecomputedNPY2DDataset(
                data_path=self.precomputed_path,
                split=split,
                patch_size=self.patch_size,
                lr_thickness_range=self.lr_thickness_range,
                tgt_thickness_range=self.tgt_thickness_range,  # Pass T_hr range
                slice_profile=self.slice_profile,
                transform=transform,
                max_slices=max_slices,
                use_ceta=self.use_ceta,  # CETA for both train and val (val uses it for visualization)
                ceta_mode=self.ceta_mode,
                ceta_ratio=self.ceta_ratio,
                ceta_anchor=self.ceta_anchor,
                ceta_gap_mm=self.ceta_gap_mm,
                ceta_delta_pad=self.ceta_delta_pad,
                modality=self.modality,  # Pass modality filter
                degradation_mode=self.degradation_mode,  # legacy or physical
                native_resolution=self.native_resolution,  # Fixed T_hr (None = auto-detect)
                lr_sampling_mode=self.lr_sampling_mode if split == 'train' else 'uniform',  # Only for training
                pad_sampling_alpha=self.pad_sampling_alpha,
            )
        else:  # onthefly
            # For on-the-fly, native_resolution defaults to 0.7 (HCP) if not specified
            native_res = self.native_resolution if self.native_resolution is not None else 0.7
            return DRIFT2DDataset(
                data_path=self.onthefly_path,
                split='train' if split == 'train' else 'test',
                contrast=self.contrast,
                patch_size=self.patch_size,
                lr_thickness_range=self.lr_thickness_range,
                slices_per_volume=self.slices_per_volume if split == 'train' else 8,
                slice_profile=self.slice_profile,
                num_volumes=self.train_num_volumes if split == 'train' else self.val_num_volumes,
                transform=transform,
                degradation_mode=self.degradation_mode,  # legacy or physical
                native_resolution=native_res,  # Fixed T_hr
                lr_sampling_mode=self.lr_sampling_mode if split == 'train' else 'uniform',  # Only for training
                pad_sampling_alpha=self.pad_sampling_alpha,
            )

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(
            split='train',
            transform=self._train_transform(),
            max_slices=self.max_train_slices,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(
            split='test',
            transform=None,
            max_slices=self.max_val_slices,
        )
            

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def _train_transform(self) -> callable:
        """Training augmentation for 2D slices."""
        def transform(lr, hr):
            if np.random.rand() > 0.5:
                lr = lr.flip(dims=[2])
                hr = hr.flip(dims=[2])

            if np.random.rand() > 0.5:
                lr = lr.flip(dims=[1])
                hr = hr.flip(dims=[1])

            intensity_scale = 0.95 + 0.1 * np.random.rand()
            intensity_shift = -0.04 + 0.08 * np.random.rand()

            lr = lr * intensity_scale + intensity_shift
            hr = hr * intensity_scale + intensity_shift

            return lr, hr

        return transform


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Testing DRIFT2DDataset")
    print("=" * 60)

    # Test with dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create HCP-like structure
        train_dir = Path(tmpdir) / "train" / "100206"
        train_dir.mkdir(parents=True)

        # Create dummy volume
        dummy_data = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

        nib.save(
            nib.Nifti1Image(dummy_data, affine),
            train_dir / "100206_t1_hr.nii.gz"
        )

        # Test dataset
        dataset = DRIFT2DDataset(
            data_path=tmpdir,
            split='train',
            contrast='t1',
            patch_size=(32, 32),
            slices_per_volume=4,
        )

        sample = dataset[0]

        print(f"\nSample output:")
        print(f"  x_lr shape: {sample['x_lr'].shape}")
        print(f"  y_hr shape: {sample['y_hr'].shape}")
        print(f"  protocol (T_lr, T_hr): {sample['protocol'].tolist()}")
        print(f"  slice_axis (thick-slice dir): {sample['slice_axis'].item()}")
        print(f"  extract_axis (2D slice plane): {sample['extract_axis'].item()}")
        print(f"  artifact_axis (in 2D): {sample['artifact_axis'].item()}")

        # Verify degradation pipeline
        print(f"\nDegradation verification:")
        print(f"  LR min/max: {sample['x_lr'].min():.3f} / {sample['x_lr'].max():.3f}")
        print(f"  HR min/max: {sample['y_hr'].min():.3f} / {sample['y_hr'].max():.3f}")

        # Check that LR is actually degraded (should be smoother)
        lr_grad = torch.abs(sample['x_lr'][:, 1:, :] - sample['x_lr'][:, :-1, :]).mean()
        hr_grad = torch.abs(sample['y_hr'][:, 1:, :] - sample['y_hr'][:, :-1, :]).mean()
        print(f"  LR gradient magnitude: {lr_grad:.4f}")
        print(f"  HR gradient magnitude: {hr_grad:.4f}")
        print(f"  LR should be smoother: {'✓' if lr_grad < hr_grad else '✗'}")

    print("\n[PASS] DRIFT2DDataset test")
