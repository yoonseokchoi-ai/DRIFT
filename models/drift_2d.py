"""
DRIFT: Difficulty-aware Rectified Flows for Through-plane MRI Super-Resolution
================================================================================

Paper: ECCV 2026 Submission #3442

Stage 1 - APN (Anatomical Projection Network) [Sec. 3.3]:
    x_lr → z (coarse HR estimate on the structural manifold)
    - 2D U-Net with slice-thickness conditioning via τ = 1/T (Eq. 2)
    - Trained with Charbonnier + SSIM loss (Eq. 5)
    - Provides structured initialization to shorten Stage 2 trajectory

Stage 2 - Rectified Flow with velocity network v_θ [Sec. 3.4]:
    z → ỹ (refined HR output)
    - 2D U-Net with joint time + thickness conditioning via AdaGN (Eq. 3, 7)
    - Velocity matching with Huber loss and U-shaped timestep sampling (Eq. 9, 10)
    - CETA loss for thickness-consistent trajectories (Eq. 12)
    - PAD-based AIS for adaptive inference stepping (Eq. 13, 14)

Architecture reference (Sec. S4.1):
    - 4 encoder levels, symmetric decoder, channel mult (1,2,4,8)
    - AdaGN (32 groups) in every residual block
    - Zero-initialized output conv in Stage 2 (identity init)
    - large config: C=96, ~128M (Stage 1) / ~139M (Stage 2) params
"""

import math
from typing import Optional, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PyTorch Lightning
try:
    import pytorch_lightning as pl
    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False
    pl = None

# TorchMetrics
try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
    from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

# Wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# 2D Building Blocks
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time/protocol conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ProtocolEmbedding2D(nn.Module):
    """
    Slice-thickness conditioning via inverse thickness τ = 1/T (Eq. 2).

    Encodes (T_lr, T_hr) through two unshared MLPs and fuses them:
        c_i = MLP_cond([MLP_in(τ_i) || MLP_tgt(τ_hr)])

    τ proxies the effective through-plane bandwidth (Sec. 3.2).
    """

    def __init__(self, protocol_embed_dim: int = 256, out_dim: int = 512):
        super().__init__()

        self.tau_lr_embed = nn.Sequential(
            nn.Linear(1, protocol_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(protocol_embed_dim // 2, protocol_embed_dim),
        )
        self.tau_hr_embed = nn.Sequential(
            nn.Linear(1, protocol_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(protocol_embed_dim // 2, protocol_embed_dim),
        )

        self.combine = nn.Sequential(
            nn.Linear(protocol_embed_dim * 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, protocol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protocol: (B, 2) tensor with [T_lr, T_hr] in mm
        Returns:
            (B, out_dim) conditioning vector
        """
        T_lr = protocol[:, 0:1]
        T_hr = protocol[:, 1:2]

        # Convert to τ = 1/T (proportional to frequency bandwidth)
        tau_lr = 1.0 / T_lr.clamp(min=0.1)
        tau_hr = 1.0 / T_hr.clamp(min=0.1)

        emb_lr = self.tau_lr_embed(tau_lr)
        emb_hr = self.tau_hr_embed(tau_hr)

        combined = torch.cat([emb_lr, emb_hr], dim=-1)
        return self.combine(combined)


class TimeProtocolEmbedding2D(nn.Module):
    """
    Joint time-thickness conditioning for Stage 2 velocity network (Eq. 7).

        e_t = MLP_time(SinEmb(t))
        c_{t,i} = MLP_tt([e_t || c_i])

    Combines sinusoidal time embedding with τ = 1/T protocol embedding.
    Injected into residual blocks via AdaGN (Eq. 3).
    """

    def __init__(
        self,
        time_embed_dim: int = 256,
        protocol_embed_dim: int = 256,
        out_dim: int = 512,
    ):
        super().__init__()

        # Time embedding (sinusoidal, standard for diffusion/flow models)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Protocol embedding with τ = 1/T encoding
        self.protocol_embed = ProtocolEmbedding2D(
            protocol_embed_dim=protocol_embed_dim,
            out_dim=protocol_embed_dim * 2,
        )

        # Combine time and protocol
        self.combine = nn.Sequential(
            nn.Linear(time_embed_dim + protocol_embed_dim * 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor, protocol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep in [0, 1]
            protocol: (B, 2) [T_lr, T_hr] in mm
        Returns:
            (B, out_dim) conditioning vector
        """
        t_emb = self.time_mlp(t)           # (B, time_embed_dim)
        p_emb = self.protocol_embed(protocol)  # (B, protocol_embed_dim * 2)
        combined = torch.cat([t_emb, p_emb], dim=-1)
        return self.combine(combined)


class AdaGN2D(nn.Module):
    """Adaptive Group Normalization (Eq. 3): AdaGN(h, c) = γ(c) ⊙ GN(h) + β(c)."""

    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_channels = num_channels

        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        self.num_groups = max(1, num_groups)

        self.norm = nn.GroupNorm(self.num_groups, num_channels, affine=False)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_channels * 2),
        )

        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.cond_proj(cond)
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.view(-1, self.num_channels, 1, 1)
        shift = shift.view(-1, self.num_channels, 1, 1)
        return h * (1 + scale) + shift


class ResBlock2D(nn.Module):
    """2D Residual block with AdaGN conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = AdaGN2D(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGN2D(out_channels, cond_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h, cond)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


# =============================================================================
# Stage 1: Regression 2D U-Net
# =============================================================================

class APNUNet2D(nn.Module):
    """
    Anatomical Projection Network (APN) - Stage 1 (Sec. 3.3, Eq. 4).

    2D U-Net f_φ that maps LR patches to the coarse HR manifold:
        z_{p,i} = f_φ(x_{p,i}, c_i)

    Conditioned on slice-thickness via τ = 1/T (no time embedding).
    Trained with L_recon = L_Char + λ_ssim * L_SSIM (Eq. 5).
    """

    MODEL_CONFIGS = {
        'tiny': {'model_channels': 32, 'channel_mult': (1, 2, 4), 'num_res_blocks': 1},
        'small': {'model_channels': 48, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
        'base': {'model_channels': 64, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
        'large': {'model_channels': 96, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
    }

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_config: str = 'base',
        protocol_embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        if model_config not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown config: {model_config}")
        cfg = self.MODEL_CONFIGS[model_config]

        model_channels = cfg['model_channels']
        channel_mult = cfg['channel_mult']
        num_res_blocks = cfg['num_res_blocks']

        self.num_levels = len(channel_mult)
        cond_dim = protocol_embed_dim * 2

        # Protocol embedding with τ = 1/T encoding
        self.protocol_embed = ProtocolEmbedding2D(
            protocol_embed_dim=protocol_embed_dim,
            out_dim=cond_dim,
        )

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            is_last = (level == len(channel_mult) - 1)

            res_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                ch_in = ch if i == 0 else out_ch
                res_blocks.append(ResBlock2D(ch_in, out_ch, cond_dim, dropout))

            if not is_last:
                downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            else:
                downsample = None

            self.down_blocks.append(nn.ModuleDict({
                'res_blocks': res_blocks,
                'downsample': downsample,
            }))
            self.skip_channels.append(out_ch)
            ch = out_ch

        # Middle
        self.mid_block1 = ResBlock2D(ch, ch, cond_dim, dropout)
        self.mid_block2 = ResBlock2D(ch, ch, cond_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()

        for level in range(len(channel_mult) - 1, -1, -1):
            mult = channel_mult[level]
            out_ch = model_channels * mult if level > 0 else model_channels
            skip_ch = self.skip_channels[level]
            is_first = (level == len(channel_mult) - 1)

            if not is_first:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
            else:
                upsample = None

            res_blocks = nn.ModuleList()
            ch_in = ch + skip_ch
            for i in range(num_res_blocks + 1):
                if i > 0:
                    ch_in = out_ch
                res_blocks.append(ResBlock2D(ch_in, out_ch, cond_dim, dropout))

            self.up_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
            }))
            ch = out_ch

        # Output
        num_groups = min(32, ch)
        while ch % num_groups != 0:
            num_groups -= 1
        self.norm_out = nn.GroupNorm(max(1, num_groups), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, protocol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input
            protocol: (B, 2) [T_lr, T_hr]
        Returns:
            (B, C, H, W) output
        """
        cond = self.protocol_embed(protocol)
        h = self.conv_in(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            for res_block in down_block['res_blocks']:
                h = res_block(h, cond)
            skips.append(h)
            if down_block['downsample'] is not None:
                h = down_block['downsample'](h)

        # Middle
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # Decoder
        for up_block in self.up_blocks:
            if up_block['upsample'] is not None:
                h = up_block['upsample'](h)

            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)

            h = torch.cat([h, skip], dim=1)

            for res_block in up_block['res_blocks']:
                h = res_block(h, cond)

        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)

        return out


# =============================================================================
# Stage 2: RF Velocity 2D U-Net
# =============================================================================

class RFVelocityUNet2D(nn.Module):
    """
    Rectified Flow velocity network v_θ - Stage 2 (Sec. 3.4, Eq. 8).

    2D U-Net that predicts the velocity field along the straight path:
        v_{p,i}(t) = v_θ(s_{p,i}(t); c_{t,i})

    where s(t) = (1-t)z + ty is the interpolated state (Eq. 6).
    Joint time + thickness conditioning via AdaGN (Eq. 7).
    Output conv is zero-initialized so initial v_θ ≈ 0 (identity refinement).
    """

    MODEL_CONFIGS = {
        'tiny': {'model_channels': 32, 'channel_mult': (1, 2, 4), 'num_res_blocks': 1},
        'small': {'model_channels': 48, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
        'base': {'model_channels': 64, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
        'large': {'model_channels': 96, 'channel_mult': (1, 2, 4, 8), 'num_res_blocks': 2},
    }

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_config: str = 'base',
        time_embed_dim: int = 256,
        protocol_embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        if model_config not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown config: {model_config}")
        cfg = self.MODEL_CONFIGS[model_config]

        model_channels = cfg['model_channels']
        channel_mult = cfg['channel_mult']
        num_res_blocks = cfg['num_res_blocks']

        self.num_levels = len(channel_mult)
        cond_dim = time_embed_dim + protocol_embed_dim * 2

        # Time + Protocol embedding with τ = 1/T encoding
        self.cond_embed = TimeProtocolEmbedding2D(
            time_embed_dim=time_embed_dim,
            protocol_embed_dim=protocol_embed_dim,
            out_dim=cond_dim,
        )

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            is_last = (level == len(channel_mult) - 1)

            res_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                ch_in = ch if i == 0 else out_ch
                res_blocks.append(ResBlock2D(ch_in, out_ch, cond_dim, dropout))

            if not is_last:
                downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            else:
                downsample = None

            self.down_blocks.append(nn.ModuleDict({
                'res_blocks': res_blocks,
                'downsample': downsample,
            }))
            self.skip_channels.append(out_ch)
            ch = out_ch

        # Middle
        self.mid_block1 = ResBlock2D(ch, ch, cond_dim, dropout)
        self.mid_block2 = ResBlock2D(ch, ch, cond_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()

        for level in range(len(channel_mult) - 1, -1, -1):
            mult = channel_mult[level]
            out_ch = model_channels * mult if level > 0 else model_channels
            skip_ch = self.skip_channels[level]
            is_first = (level == len(channel_mult) - 1)

            if not is_first:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
            else:
                upsample = None

            res_blocks = nn.ModuleList()
            ch_in = ch + skip_ch
            for i in range(num_res_blocks + 1):
                if i > 0:
                    ch_in = out_ch
                res_blocks.append(ResBlock2D(ch_in, out_ch, cond_dim, dropout))

            self.up_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
            }))
            ch = out_ch

        # Output (zero-initialized for velocity)
        num_groups = min(32, ch)
        while ch % num_groups != 0:
            num_groups -= 1
        self.norm_out = nn.GroupNorm(max(1, num_groups), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Zero-initialize output
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, protocol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input
            t: (B,) timestep in [0, 1]
            protocol: (B, 2) [T_lr, T_hr]
        Returns:
            (B, C, H, W) velocity
        """
        cond = self.cond_embed(t, protocol)
        h = self.conv_in(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            for res_block in down_block['res_blocks']:
                h = res_block(h, cond)
            skips.append(h)
            if down_block['downsample'] is not None:
                h = down_block['downsample'](h)

        # Middle
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # Decoder
        for up_block in self.up_blocks:
            if up_block['upsample'] is not None:
                h = up_block['upsample'](h)

            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)

            h = torch.cat([h, skip], dim=1)

            for res_block in up_block['res_blocks']:
                h = res_block(h, cond)

        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)

        return out


# =============================================================================
# Loss Functions
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Robust L1)."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))


class SSIM2DLoss(nn.Module):
    """
    2D SSIM Loss using torchmetrics.

    Uses torchmetrics.functional.structural_similarity_index_measure for
    robust and well-tested SSIM computation.
    """

    def __init__(self, window_size: int = 11, data_range: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range  # [-1, 1] → range = 2.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: (B, C, H, W) predicted image in [-1, 1]
            target: (B, C, H, W) target image in [-1, 1]

        Returns:
            Scalar SSIM loss (1 - SSIM)
        """
        if TORCHMETRICS_AVAILABLE:
            # Use torchmetrics SSIM (well-tested, GPU optimized)
            ssim_val = ssim_metric(
                pred,
                target,
                data_range=self.data_range,
                kernel_size=self.window_size,
            )
            return 1.0 - ssim_val
        else:
            # Fallback to manual implementation
            return self._manual_ssim_loss(pred, target)

    def _manual_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fallback manual SSIM implementation."""
        C = pred.shape[1]

        # Create Gaussian window
        window = self._create_2d_gaussian_window(self.window_size).to(pred.device, dtype=pred.dtype)
        window = window.expand(C, 1, -1, -1)

        K1, K2 = 0.01, 0.03
        L = self.data_range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target

        sigma_pred_sq = sigma_pred_sq.clamp(min=0)
        sigma_target_sq = sigma_target_sq.clamp(min=0)

        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

        ssim_map = numerator / (denominator + 1e-8)
        return 1.0 - ssim_map.mean()

    def _create_2d_gaussian_window(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Visualization
# =============================================================================

def _create_2d_comparison_figure(
    x_lr: torch.Tensor,
    y_pred: torch.Tensor,
    y_hr: torch.Tensor,
    t_lr: float = None,
    t_hr: float = None,
) -> np.ndarray:
    """Create 1x4 comparison figure for 2D images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    if t_lr is not None and t_hr is not None:
        fig.suptitle(f'LR: {t_lr:.2f}mm → Target: {t_hr:.2f}mm', fontsize=14, fontweight='bold')

    titles = ['LR Input', 'Predicted', 'Ground Truth', '|Pred - GT|']

    # Convert to numpy and normalize
    lr_img = x_lr[0].cpu().numpy()
    pred_img = y_pred[0].cpu().numpy()
    hr_img = y_hr[0].cpu().numpy()

    lr_img = np.clip((lr_img + 1) / 2, 0, 1)
    pred_img = np.clip((pred_img + 1) / 2, 0, 1)
    hr_img = np.clip((hr_img + 1) / 2, 0, 1)
    diff_img = np.abs(pred_img - hr_img)

    for col, (img, title) in enumerate(zip([lr_img, pred_img, hr_img, diff_img], titles)):
        if col < 3:
            axes[col].imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[col].imshow(img, cmap='hot', vmin=0, vmax=0.5)
        axes[col].set_title(title, fontsize=12, fontweight='bold')
        axes[col].axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)

    return img


def _create_cascaded_2d_comparison_figure(
    x_lr: torch.Tensor,
    y_coarse: torch.Tensor,
    y_pred: torch.Tensor,
    y_hr: torch.Tensor,
    t_lr: float = None,
    t_hr: float = None,
    # CETA outputs (optional)
    ceta_data: Dict = None,
) -> np.ndarray:
    """
    Create comparison figure for cascaded 2D.

    If ceta_data is provided, creates 2-row figure:
    - Row 1: Primary trajectory (x_lr → y_coarse → y_pred → y_hr)
    - Row 2: Alternative trajectory (x_lr_alt → y_coarse_alt → y_est_alt) + velocity comparison

    Otherwise creates 1x6 figure (original behavior).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ceta_data is not None and 'x_lr_alt' in ceta_data:
        # Extended figure with CETA outputs (2 rows)
        fig, axes = plt.subplots(2, 7, figsize=(28, 8))

        # Title with both protocols
        t_lr_alt = ceta_data.get('t_lr_alt', 0.0)
        fig.suptitle(
            f'Primary: {t_lr:.2f}mm → {t_hr:.2f}mm  |  CETA Alt: {t_lr_alt:.2f}mm → {t_hr:.2f}mm',
            fontsize=14, fontweight='bold'
        )

        # Convert primary outputs to numpy (squeeze channel dim for imshow)
        lr_img = np.clip((x_lr.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        coarse_img = np.clip((y_coarse.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        pred_img = np.clip((y_pred.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        hr_img = np.clip((y_hr.squeeze().cpu().numpy() + 1) / 2, 0, 1)

        # Convert CETA outputs to numpy (squeeze channel dim for imshow)
        lr_alt_img = np.clip((ceta_data['x_lr_alt'].squeeze().cpu().numpy() + 1) / 2, 0, 1)
        coarse_alt_img = np.clip((ceta_data['y_coarse_alt'].squeeze().cpu().numpy() + 1) / 2, 0, 1)
        y_est_img = np.clip((ceta_data['y_est'].squeeze().cpu().numpy() + 1) / 2, 0, 1)
        y_est_alt_img = np.clip((ceta_data['y_est_alt'].squeeze().cpu().numpy() + 1) / 2, 0, 1)
        v_pred_img = ceta_data['v_pred'].squeeze().cpu().numpy()  # velocity can be negative
        v_pred_alt_img = ceta_data['v_pred_alt'].squeeze().cpu().numpy()

        # Compute differences
        coarse_diff = np.abs(coarse_img - hr_img)
        final_diff = np.abs(pred_img - hr_img)
        endpoint_diff = np.abs(y_est_img - y_est_alt_img)  # CETA endpoint consistency

        # Row 1: Primary trajectory
        row1_titles = ['LR Input', 'Stage1 (Coarse)', 'Stage2 (Final)', 'Ground Truth',
                       '|Coarse-GT|', '|Final-GT|', 'y_est (ŷ_A)']
        row1_images = [lr_img, coarse_img, pred_img, hr_img, coarse_diff, final_diff, y_est_img]

        for col, (img, title) in enumerate(zip(row1_images, row1_titles)):
            if col < 4 or col == 6:  # regular images
                axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:  # difference maps
                axes[0, col].imshow(img, cmap='hot', vmin=0, vmax=0.5)
            axes[0, col].set_title(title, fontsize=11, fontweight='bold')
            axes[0, col].axis('off')

        # Row 2: CETA alternative trajectory
        row2_titles = ['LR_alt', 'Coarse_alt', 'v_pred', 'v_pred_alt',
                       '|v_pred-v_alt|', '|ŷ_A - ŷ_B|', 'y_est_alt (ŷ_B)']
        v_diff = np.abs(v_pred_img - v_pred_alt_img)
        row2_images = [lr_alt_img, coarse_alt_img, v_pred_img, v_pred_alt_img,
                       v_diff, endpoint_diff, y_est_alt_img]

        for col, (img, title) in enumerate(zip(row2_images, row2_titles)):
            if col in [0, 1, 6]:  # LR_alt, Coarse_alt, y_est_alt
                axes[1, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            elif col in [2, 3]:  # velocity fields (can be +/-)
                vmax = max(np.abs(v_pred_img).max(), np.abs(v_pred_alt_img).max(), 0.5)
                axes[1, col].imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            else:  # difference maps (col 4, 5)
                axes[1, col].imshow(img, cmap='hot', vmin=0, vmax=0.3)
            axes[1, col].set_title(title, fontsize=11, fontweight='bold')
            axes[1, col].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        plt.close(fig)
        return img

    else:
        # Figure with velocity field (1x8 if v_pred available, else 1x6)
        # Check if velocity is in ceta_data (passed as v_pred for non-CETA case)
        v_pred_img = None
        if ceta_data is not None and 'v_pred' in ceta_data:
            v_pred_img = ceta_data['v_pred'].squeeze().cpu().numpy()

        num_cols = 8 if v_pred_img is not None else 6
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        if t_lr is not None and t_hr is not None:
            fig.suptitle(f'LR: {t_lr:.2f}mm → Target: {t_hr:.2f}mm', fontsize=14, fontweight='bold')

        # Convert to numpy and normalize (squeeze channel dim for imshow)
        lr_img = np.clip((x_lr.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        coarse_img = np.clip((y_coarse.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        pred_img = np.clip((y_pred.squeeze().cpu().numpy() + 1) / 2, 0, 1)
        hr_img = np.clip((y_hr.squeeze().cpu().numpy() + 1) / 2, 0, 1)

        coarse_diff = np.abs(coarse_img - hr_img)
        final_diff = np.abs(pred_img - hr_img)

        if v_pred_img is not None:
            # 8 columns: add velocity and estimated endpoint
            y_est_img = np.clip((coarse_img + (v_pred_img + 1) / 2 - 0.5), 0, 1)  # Approx visualization
            titles = ['LR Input', 'Stage1 (Coarse)', 'v_pred (velocity)', 'y_est (ŷ)',
                      'Stage2 (Final)', 'Ground Truth', '|Coarse-GT|', '|Final-GT|']

            # Compute y_est properly: coarse + v in original scale
            # v_pred is in [-1, 1] or similar, y_est = coarse + v
            coarse_raw = (x_lr[0].cpu().numpy() + 1) / 2 if y_coarse is None else (y_coarse[0].cpu().numpy() + 1) / 2
            y_est_raw = coarse_raw + v_pred_img  # velocity adds to coarse
            y_est_img = np.clip(y_est_raw, 0, 1)

            images = [lr_img, coarse_img, v_pred_img, y_est_img, pred_img, hr_img, coarse_diff, final_diff]

            for col, (img, title) in enumerate(zip(images, titles)):
                if col in [0, 1, 3, 4, 5]:  # regular images
                    axes[col].imshow(img, cmap='gray', vmin=0, vmax=1)
                elif col == 2:  # velocity field (can be +/-)
                    vmax = max(np.abs(v_pred_img).max(), 0.5)
                    axes[col].imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                else:  # difference maps
                    axes[col].imshow(img, cmap='hot', vmin=0, vmax=0.5)
                axes[col].set_title(title, fontsize=12, fontweight='bold')
                axes[col].axis('off')
        else:
            # Original 6 columns (no velocity)
            titles = ['LR Input', 'Stage1 (Coarse)', 'Stage2 (Final)', 'Ground Truth', '|Coarse-GT|', '|Final-GT|']
            images = [lr_img, coarse_img, pred_img, hr_img, coarse_diff, final_diff]

            for col, (img, title) in enumerate(zip(images, titles)):
                if col < 4:
                    axes[col].imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    axes[col].imshow(img, cmap='hot', vmin=0, vmax=0.5)
                axes[col].set_title(title, fontsize=12, fontweight='bold')
                axes[col].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        plt.close(fig)

        return img


# =============================================================================
# Stage 1: Lightning Module
# =============================================================================

if PL_AVAILABLE:

    class Stage1APNLightning(pl.LightningModule):
        """
        Stage 1: APN (Anatomical Projection Network) Lightning Module (Sec. 3.3).

        Trains f_φ to project LR patches onto the coarse HR manifold.
        Loss: L_recon = L_Char + λ_ssim * L_SSIM (Eq. 5).
        """

        def __init__(
            self,
            model_config: str = 'base',
            in_channels: int = 1,
            protocol_embed_dim: int = 256,
            charbonnier_epsilon: float = 1e-3,
            ssim_weight: float = 0.5,
            ssim_window_size: int = 11,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.0,
            warmup_steps: int = 1000,
            max_steps: int = 100000,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.warmup_steps = warmup_steps
            self.max_steps = max_steps
            self.ssim_weight = ssim_weight

            # Model with τ = 1/T protocol encoding
            self.model = APNUNet2D(
                in_channels=in_channels,
                out_channels=in_channels,
                model_config=model_config,
                protocol_embed_dim=protocol_embed_dim,
            )

            # Losses
            self.charbonnier = CharbonnierLoss(epsilon=charbonnier_epsilon)
            self.ssim_loss = SSIM2DLoss(window_size=ssim_window_size, data_range=2.0)  # [-1, 1] range

            self.val_outputs = []

        def forward(self, x: torch.Tensor, protocol: torch.Tensor) -> torch.Tensor:
            return self.model(x, protocol)

        def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            x_lr = batch['x_lr']
            y_hr = batch['y_hr']
            protocol = batch['protocol']

            y_pred = self.model(x_lr, protocol)

            char_loss = self.charbonnier(y_pred, y_hr)

            # Only compute SSIM loss if weight > 0 (saves computation)
            if self.ssim_weight > 0:
                ssim_loss = self.ssim_loss(y_pred, y_hr)
                total_loss = char_loss + self.ssim_weight * ssim_loss
                self.log('train/ssim_loss', ssim_loss, on_step=False, on_epoch=True, sync_dist=True)
            else:
                total_loss = char_loss

            self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/charbonnier', char_loss, on_step=False, on_epoch=True, sync_dist=True)

            return total_loss

        def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
            x_lr = batch['x_lr']
            y_hr = batch['y_hr']
            protocol = batch['protocol']

            with torch.no_grad():
                y_pred = self.model(x_lr, protocol)

            if TORCHMETRICS_AVAILABLE:
                psnr = psnr_metric(y_pred, y_hr, data_range=2.0)
                ssim = ssim_metric(y_pred, y_hr, data_range=2.0)
            else:
                mse = F.mse_loss(y_pred, y_hr)
                psnr = 10 * torch.log10(4.0 / mse)
                ssim = torch.tensor(0.0, device=y_pred.device)

            # Select which batch to log based on epoch (for diverse visualization)
            log_batch_idx = self.current_epoch % 10  # Cycle through first 10 batches
            should_log_images = (batch_idx == log_batch_idx)

            output = {
                'psnr': psnr,
                'ssim': ssim,
                'x_lr': x_lr[:1].detach() if should_log_images else None,
                'y_pred': y_pred[:1].detach() if should_log_images else None,
                'y_hr': y_hr[:1].detach() if should_log_images else None,
                'protocol': protocol[:1].detach() if should_log_images else None,
            }
            self.val_outputs.append(output)
            return output

        def on_validation_epoch_end(self):
            if not self.val_outputs:
                return

            avg_psnr = torch.stack([x['psnr'] for x in self.val_outputs]).mean()
            avg_ssim = torch.stack([x['ssim'] for x in self.val_outputs]).mean()

            self.log('val_psnr', avg_psnr, prog_bar=True, sync_dist=True)
            self.log('val_ssim', avg_ssim, prog_bar=True, sync_dist=True)

            # Log images - find the output with images (selected by epoch-based batch idx)
            if WANDB_AVAILABLE and self.logger is not None:
                # Find output with images, fallback to first available if target batch doesn't exist
                img_output = None
                for output in self.val_outputs:
                    if output.get('x_lr') is not None:
                        img_output = output
                        break

                if img_output is not None:
                    try:
                        t_lr = img_output['protocol'][0, 0].item()
                        t_hr = img_output['protocol'][0, 1].item()

                        img = _create_2d_comparison_figure(
                            img_output['x_lr'][0], img_output['y_pred'][0], img_output['y_hr'][0],
                            t_lr=t_lr, t_hr=t_hr
                        )

                        if hasattr(self.logger, 'experiment'):
                            self.logger.experiment.log({
                                'val/images': wandb.Image(
                                    img,
                                    caption=f'Epoch {self.current_epoch} | PSNR: {avg_psnr:.2f}'
                                ),
                            }, commit=False)
                    except Exception as e:
                        print(f"Failed to log images: {e}")

            self.val_outputs.clear()

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            def lr_lambda(step):
                if step < self.warmup_steps:
                    # Use (step + 1) to ensure LR > 0 at step=0
                    # step=0 → (1/warmup_steps), step=warmup_steps-1 → 1.0
                    return (step + 1) / max(1, self.warmup_steps)
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }


# =============================================================================
# Stage 2: Lightning Module
# =============================================================================

if PL_AVAILABLE:

    class Stage2RFLightning(pl.LightningModule):
        """
        Stage 2: Difficulty-Aware Rectified Flow Lightning Module (Sec. 3.4-3.6).

        Trains v_θ to match the target velocity u_{p,i} = y_p - z_{p,i} (Eq. 6, 9).
        Uses frozen Stage 1 APN to generate coarse prediction z.

        Key components:
        - L_RF: Huber velocity matching with U-shaped timestep sampling (Eq. 9, 10)
        - CETA: Consistent Endpoint Trajectory Alignment (Sec. 3.5, Eq. 11-12)
            ỹ_{p,k}(t) = z_{p,k} + v_{p,k}(t),  L_CETA = ||ỹ_{p,i} - ỹ_{p,j}||²
            Proximal pairs: T_j = T_i - ΔT, ΔT = 1mm
        - PAD: Physics-Aware Difficulty (Sec. 3.6, Eq. 13)
            PAD(T_i, T_hr) = 1 - T_hr/T_i
        - AIS: Adaptive Integration Scheduler (Eq. 14)
            N = clamp(⌊N_max × PAD⌉, N_min, N_max)
        """

        def __init__(
            self,
            stage1_ckpt: str = None,
            freeze_stage1: bool = True,
            model_config: str = 'base',
            in_channels: int = 1,
            time_embed_dim: int = 256,
            protocol_embed_dim: int = 256,
            num_inference_steps: int = 4,
            loss_type: str = 'huber',
            use_u_shaped_sampling: bool = True,
            u_shape_power: float = 2.0,

            # CETA (Consistent Endpoint Trajectory Alignment)
            use_ceta: bool = False,
            ceta_weight: float = 0.1,
            ceta_thickness_margin: float = 1.0,  # minimum thickness difference for dual sampling
            ceta_fixed_t: float = None,  # If set, use this fixed timestep for CETA instead of flow loss's random t

            # Adaptive Stepping Config with PAD (Physics-Aware Difficulty)
            use_adaptive_steps: bool = False,  # Enable adaptive stepping based on PAD
            use_pad: bool = True,  # Use PAD formula (recommended, no hyperparams)
            max_steps: int = 10,  # Maximum allowed steps (N_max in PAD formula)
            min_steps: int = 2,  # Minimum allowed steps (numerical stability)

            # Legacy SFI config (for ablation study comparison)
            use_sfi: bool = False,  # Deprecated: use_pad is recommended
            base_steps: int = 4,  # Legacy: base steps for SFI
            sfi_alpha: float = 0.5,  # Legacy: physical penalty weight
            max_thickness_ref: float = 6.0,  # Legacy: reference max thickness
            difficulty_mode: str = 'multiplicative',  # Legacy: SFI mode
            additive_beta: float = 1.0,  # Legacy: additive mode weight

            # Training
            learning_rate: float = 5e-5,
            weight_decay: float = 0.0,
            warmup_steps: int = 1000,
            max_steps_training: int = 100000,

            # Ablation: disable protocol conditioning for Stage 2 RF
            # When True, Stage 2 RF receives a fixed protocol (mean of training range)
            # Stage 1 (frozen) still receives the real protocol
            disable_protocol: bool = False,

            # Epoch-wise image saving for ablation study visualization
            save_epoch_images: bool = False,
            save_epoch_interval: int = 10,
            save_epoch_dir: str = None,
            fixed_val_sample_idx: int = 0,  # Which validation batch to use for consistent visualization
        ):
            super().__init__()
            self.save_hyperparameters()

            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.warmup_steps = warmup_steps
            self.max_steps_training = max_steps_training
            self.loss_type = loss_type
            self.num_inference_steps = num_inference_steps

            self.use_u_shaped_sampling = use_u_shaped_sampling
            self.u_shape_power = u_shape_power

            # Ablation flag
            self.disable_protocol = disable_protocol

            # CETA configuration
            self.use_ceta = use_ceta
            self.ceta_weight = ceta_weight
            self.ceta_thickness_margin = ceta_thickness_margin
            self.ceta_fixed_t = ceta_fixed_t

            # Epoch-wise image saving configuration
            self.save_epoch_images = save_epoch_images
            self.save_epoch_interval = save_epoch_interval
            self.save_epoch_dir = save_epoch_dir
            self.fixed_val_sample_idx = fixed_val_sample_idx

            # PAD (Physics-Aware Difficulty) based adaptive stepping
            self.use_adaptive_steps = use_adaptive_steps
            self.use_pad = use_pad
            self.max_steps = max_steps  # N_max in PAD formula
            self.min_steps = min_steps

            # Legacy SFI config (for ablation study)
            self.use_sfi = use_sfi
            self.base_steps = base_steps
            self.sfi_alpha = sfi_alpha
            self.max_thickness_ref = max_thickness_ref
            self.difficulty_mode = difficulty_mode
            self.additive_beta = additive_beta

            # Load Stage 1
            if stage1_ckpt is not None:
                self.stage1 = Stage1APNLightning.load_from_checkpoint(
                    stage1_ckpt, map_location='cpu', weights_only=False
                )

                if freeze_stage1:
                    self.stage1.freeze()
                    for param in self.stage1.parameters():
                        param.requires_grad = False
            else:
                self.stage1 = None

            # Stage 2 RF model with τ = 1/T protocol encoding
            self.rf_model = RFVelocityUNet2D(
                in_channels=in_channels,
                out_channels=in_channels,
                model_config=model_config,
                time_embed_dim=time_embed_dim,
                protocol_embed_dim=protocol_embed_dim,
            )

            self.val_outputs = []

        def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
            if self.use_u_shaped_sampling:
                u = torch.rand(batch_size, device=device)
                t = 0.5 - 0.5 * torch.sign(u - 0.5) * (torch.abs(2 * u - 1) ** (1.0 / self.u_shape_power))
                return t.clamp(min=1e-5, max=1.0 - 1e-5)
            return torch.rand(batch_size, device=device)

        def compute_pad_difficulty(self, T_lr: torch.Tensor, T_hr: torch.Tensor) -> torch.Tensor:
            """
            Compute PAD (Physics-Aware Difficulty) based on Spectral Deficit.

            PAD = 1 - T_hr/T_lr  (Normalized spectral deficit)

            Physical Interpretation:
            - Based on Nyquist-Shannon theorem: max frequency bandwidth ∝ 1/T
            - Spectral deficit = (1/T_hr - 1/T_lr) / (1/T_hr) = 1 - T_hr/T_lr
            - Measures fraction of high-frequency information lost

            Boundary Conditions:
            - T_lr = T_hr → PAD = 0 (no information loss, no difficulty)
            - T_lr → ∞ → PAD → 1 (100% information loss, maximum difficulty)

            Properties:
            - Bounded in [0, 1): mathematically elegant
            - Hyperparameter-free: no α, β, T_max needed
            - Unified: same formula for training sampling and inference stepping
            """
            return 1.0 - T_hr / T_lr.clamp(min=T_hr)

        def compute_sfi_adaptive_steps(
            self,
            protocol: torch.Tensor,
        ) -> Tuple[int, Dict[str, float]]:
            """
            Compute adaptive step count using PAD or legacy SFI formula.

            PAD (Physics-Aware Difficulty) - Recommended:
            ----------------------------------------------
            PAD = 1 - T_hr/T_lr  (Spectral Deficit Ratio)
            N_steps = round(N_max × PAD)

            Based on Nyquist-Shannon theorem: the fraction of lost frequency
            bandwidth directly determines how many refinement steps are needed.

            Examples (T_hr=0.7mm, N_max=10):
            - T_lr=0.7mm → PAD=0.00 → N=0 (identity, no refinement needed)
            - T_lr=1.4mm → PAD=0.50 → N=5 (moderate refinement)
            - T_lr=3.5mm → PAD=0.80 → N=8 (significant refinement)
            - T_lr=6.0mm → PAD=0.88 → N=9 (near-maximum refinement)

            Legacy SFI (for ablation study):
            ---------------------------------
            SFI = log2(T_lr/T_hr) × (1 + α × T_lr/T_max)
            N_steps = base_steps × max(1, SFI)

            Args:
                protocol: (B, 2+) with [T_lr, T_hr, ...]

            Returns:
                num_steps: Computed number of ODE steps
                metrics: Dict with diagnostic information
            """
            T_lr = protocol[:, 0]  # LR thickness (large, e.g., 5mm)
            T_hr = protocol[:, 1]  # HR thickness (small, e.g., 0.7mm)

            if self.use_pad:
                # PAD (Physics-Aware Difficulty) - Recommended
                # Spectral deficit ratio: fraction of lost high-frequency information
                pad = self.compute_pad_difficulty(T_lr, T_hr)
                max_pad = pad.max().item()

                # Map PAD [0, 1) to steps [min_steps, max_steps]
                # N = round(N_max × PAD), clamped to [min_steps, max_steps]
                num_steps = int(round(self.max_steps * max_pad))
                num_steps = max(self.min_steps, min(num_steps, self.max_steps))

                metrics = {
                    'pad': pad.mean().item(),
                    'max_pad': max_pad,
                    'difficulty': max_pad,  # Alias for compatibility
                    'difficulty_mode': 'pad',
                }
            else:
                # Legacy SFI (for ablation study comparison)
                ratio = T_lr / T_hr.clamp(min=0.1)
                info_gain = torch.log2(ratio.clamp(min=1.0))
                physical_penalty = 1.0 + self.sfi_alpha * (T_lr / self.max_thickness_ref)

                if self.difficulty_mode == 'multiplicative':
                    difficulty = info_gain * physical_penalty
                elif self.difficulty_mode == 'additive':
                    difficulty = info_gain + self.additive_beta * physical_penalty
                elif self.difficulty_mode == 'info_gain_only':
                    difficulty = info_gain
                elif self.difficulty_mode == 'physical_penalty_only':
                    difficulty = physical_penalty
                else:
                    raise ValueError(f"Unknown difficulty_mode: {self.difficulty_mode}")

                max_difficulty = difficulty.max().item()
                num_steps = int(self.base_steps * max(1.0, max_difficulty))
                num_steps = max(self.min_steps, min(num_steps, self.max_steps))

                metrics = {
                    'info_gain': info_gain.mean().item(),
                    'physical_penalty': physical_penalty.mean().item(),
                    'difficulty': difficulty.mean().item(),
                    'max_difficulty': max_difficulty,
                    'difficulty_mode': self.difficulty_mode,
                }

            return num_steps, metrics

        def compute_velocity(self, x_t: torch.Tensor, t: torch.Tensor, protocol: torch.Tensor) -> torch.Tensor:
            if self.disable_protocol:
                # Ablation: replace protocol with fixed mean value
                # Stage 2 RF cannot distinguish different degradation levels
                fixed_protocol = torch.tensor(
                    [[3.35, 0.7]], device=protocol.device, dtype=protocol.dtype
                ).expand_as(protocol)
                return self.rf_model(x_t, t, fixed_protocol)
            return self.rf_model(x_t, t, protocol)

        def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            x_lr = batch['x_lr']
            y_hr = batch['y_hr']
            protocol = batch['protocol']
            B = x_lr.shape[0]
            device = x_lr.device

            # Get Stage 1 output
            with torch.no_grad():
                if self.stage1 is not None:
                    y_coarse = self.stage1(x_lr, protocol)
                else:
                    y_coarse = x_lr

            # RF training
            t = self.sample_timesteps(B, device)
            t_expand = t.view(B, 1, 1, 1)

            x_t = (1 - t_expand) * y_coarse + t_expand * y_hr
            v_target = y_hr - y_coarse

            v_pred = self.compute_velocity(x_t, t, protocol)

            if self.loss_type == 'huber':
                flow_loss = F.smooth_l1_loss(v_pred, v_target, beta=0.1)
            else:
                flow_loss = F.mse_loss(v_pred, v_target)

            total_loss = flow_loss

            # CETA Loss (Consistent Endpoint Trajectory Alignment)
            ceta_data_for_logging = None
            if self.use_ceta and 'x_lr_alt' in batch:
                # Use fixed timestep for CETA if specified, otherwise reuse flow loss's random t
                if self.ceta_fixed_t is not None:
                    ceta_t = torch.full_like(t, self.ceta_fixed_t)
                    ceta_t_expand = ceta_t.view(B, 1, 1, 1)
                else:
                    ceta_t = t
                    ceta_t_expand = t_expand
                ceta_loss, ceta_data_for_logging = self._compute_ceta_loss(batch, ceta_t, ceta_t_expand, return_data=True)
                total_loss = total_loss + self.ceta_weight * ceta_loss
                self.log('train/ceta_loss', ceta_loss, on_step=False, on_epoch=True, sync_dist=True)

            self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/flow_loss', flow_loss, on_step=False, on_epoch=True, sync_dist=True)

            # Log CETA trajectory images once per epoch (at first batch)
            # Only on rank 0 to avoid duplicate logging
            if (self.use_ceta and ceta_data_for_logging is not None and
                batch_idx == 0 and self.global_rank == 0 and
                WANDB_AVAILABLE and self.logger is not None):
                try:
                    self._log_ceta_training_images(
                        x_lr=batch['x_lr'],
                        y_coarse=y_coarse,
                        y_hr=y_hr,
                        protocol=protocol,
                        ceta_data=ceta_data_for_logging,
                    )
                except Exception as e:
                    print(f"[CETA Train] Failed to log images: {e}")

            return total_loss

        def _compute_ceta_loss(
            self,
            batch: Dict[str, torch.Tensor],
            t: torch.Tensor,
            t_expand: torch.Tensor,
            return_data: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
            """
            Compute CETA (Consistent Endpoint Trajectory Alignment) loss.

            Core idea: Two different starting points (different LR thicknesses)
            should predict the same endpoint when following the flow.

            Given:
            - x_pred_A: Stage1 output from thickness T_A
            - x_pred_B: Stage1 output from thickness T_B (T_B != T_A)
            - Same HR target (same anatomy)
            - Same timestep t

            The estimated endpoints should match:
            - ŷ_A = x_pred_A + v_A
            - ŷ_B = x_pred_B + v_B
            - L_CETA = ||ŷ_A - ŷ_B||²

            This forces the vector field to predict consistent endpoints,
            converging to the same anatomy regardless of input degradation level.

            Args:
                return_data: If True, also return intermediate data for visualization
            """
            x_lr_alt = batch['x_lr_alt']  # Alternative LR with different thickness
            protocol_alt = batch['protocol_alt']  # Protocol for alternative LR
            y_hr = batch['y_hr']
            B = x_lr_alt.shape[0]

            # Get Stage 1 outputs for both versions
            with torch.no_grad():
                if self.stage1 is not None:
                    y_coarse_alt = self.stage1(x_lr_alt, protocol_alt)
                else:
                    y_coarse_alt = x_lr_alt

                # Primary coarse (already computed in main flow)
                y_coarse = self.stage1(batch['x_lr'], batch['protocol']) if self.stage1 else batch['x_lr']

            # Interpolate at same timestep t
            x_t = (1 - t_expand) * y_coarse + t_expand * y_hr
            x_t_alt = (1 - t_expand) * y_coarse_alt + t_expand * y_hr

            # Predict velocities
            v_pred = self.compute_velocity(x_t, t, batch['protocol'])
            v_pred_alt = self.compute_velocity(x_t_alt, t, protocol_alt)

            # Estimate endpoints: ŷ = x_start + v (RF equation: v = y - x)
            # At t=0, x_t = x_start, so endpoint = x_start + v = y
            y_est = y_coarse + v_pred
            y_est_alt = y_coarse_alt + v_pred_alt

            # CETA loss: endpoints should match
            ceta_loss = F.mse_loss(y_est, y_est_alt)

            if return_data:
                ceta_data = {
                    'x_lr_alt': x_lr_alt[:1].detach(),
                    'y_coarse_alt': y_coarse_alt[:1].detach(),
                    'v_pred': v_pred[:1].detach(),
                    'v_pred_alt': v_pred_alt[:1].detach(),
                    'y_est': y_est[:1].detach(),
                    'y_est_alt': y_est_alt[:1].detach(),
                    't_lr_alt': protocol_alt[0, 0].item(),
                    'endpoint_mse': ceta_loss.item(),
                }
                return ceta_loss, ceta_data
            return ceta_loss

        def _log_ceta_training_images(
            self,
            x_lr: torch.Tensor,
            y_coarse: torch.Tensor,
            y_hr: torch.Tensor,
            protocol: torch.Tensor,
            ceta_data: Dict,
        ):
            """
            Log CETA trajectory visualization during training.

            Shows:
            - Row 1: Primary trajectory (x_lr → y_coarse → y_est → y_hr)
            - Row 2: Alternative trajectory (x_lr_alt → y_coarse_alt → y_est_alt)
            - Row 3: Differences (velocity diff, endpoint diff)
            """
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            t_lr = protocol[0, 0].item()
            t_lr_alt = ceta_data['t_lr_alt']

            # Convert to numpy
            def to_img(tensor):
                return np.clip((tensor.squeeze().cpu().numpy() + 1) / 2, 0, 1)

            lr_img = to_img(x_lr[0])
            coarse_img = to_img(y_coarse[0])
            hr_img = to_img(y_hr[0])

            lr_alt_img = to_img(ceta_data['x_lr_alt'])
            coarse_alt_img = to_img(ceta_data['y_coarse_alt'])
            y_est_img = to_img(ceta_data['y_est'])
            y_est_alt_img = to_img(ceta_data['y_est_alt'])
            v_pred_img = ceta_data['v_pred'].squeeze().cpu().numpy()
            v_pred_alt_img = ceta_data['v_pred_alt'].squeeze().cpu().numpy()

            # Create figure
            fig, axes = plt.subplots(2, 6, figsize=(24, 8))

            fig.suptitle(
                f'CETA Training | Step {self.global_step} | '
                f'Primary: {t_lr:.2f}mm | Alt: {t_lr_alt:.2f}mm | '
                f'Endpoint MSE: {ceta_data["endpoint_mse"]:.6f}',
                fontsize=14, fontweight='bold'
            )

            # Row 1: Primary trajectory
            row1_titles = ['LR (Primary)', 'Coarse', 'y_est (ŷ_A)', 'GT', '|Coarse-GT|', '|ŷ_A - GT|']
            coarse_diff = np.abs(coarse_img - hr_img)
            y_est_diff = np.abs(y_est_img - hr_img)
            row1_images = [lr_img, coarse_img, y_est_img, hr_img, coarse_diff, y_est_diff]

            for col, (img, title) in enumerate(zip(row1_images, row1_titles)):
                if col < 4:
                    axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    axes[0, col].imshow(img, cmap='hot', vmin=0, vmax=0.5)
                axes[0, col].set_title(title, fontsize=11, fontweight='bold')
                axes[0, col].axis('off')

            # Row 2: Alternative trajectory + velocity comparison
            endpoint_diff = np.abs(y_est_img - y_est_alt_img)
            row2_titles = ['LR (Alt)', 'Coarse_alt', 'y_est_alt (ŷ_B)', 'v_pred', 'v_pred_alt', '|ŷ_A - ŷ_B|']
            row2_images = [lr_alt_img, coarse_alt_img, y_est_alt_img, v_pred_img, v_pred_alt_img, endpoint_diff]

            for col, (img, title) in enumerate(zip(row2_images, row2_titles)):
                if col < 3:
                    axes[1, col].imshow(img, cmap='gray', vmin=0, vmax=1)
                elif col in [3, 4]:  # velocity fields
                    vmax = max(np.abs(v_pred_img).max(), np.abs(v_pred_alt_img).max(), 0.5)
                    axes[1, col].imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                else:  # endpoint diff
                    axes[1, col].imshow(img, cmap='hot', vmin=0, vmax=0.3)
                axes[1, col].set_title(title, fontsize=11, fontweight='bold')
                axes[1, col].axis('off')

            plt.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            img_array = buf[:, :, :3].copy()
            plt.close(fig)

            # Log to wandb
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    'train/ceta_trajectory': wandb.Image(
                        img_array,
                        caption=f'Step {self.global_step} | ΔT={abs(t_lr - t_lr_alt):.2f}mm | '
                                f'Endpoint MSE={ceta_data["endpoint_mse"]:.6f}'
                    ),
                }, commit=False)
                print(f"[CETA Train] Logged trajectory at step {self.global_step} "
                      f"(T_lr={t_lr:.2f}mm vs {t_lr_alt:.2f}mm, ΔT={abs(t_lr - t_lr_alt):.2f}mm)")

        def ode_integrate(
            self,
            x_0: torch.Tensor,
            protocol: torch.Tensor,
            num_steps: Optional[int] = None,
            adaptive: bool = False,
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Integrate ODE from coarse to refined output.

            Supports two modes (for ablation study):
            1. Fixed steps: Use num_steps directly (traditional approach)
            2. Adaptive steps: Use SFI difficulty to determine step count

            Args:
                x_0: (B, C, H, W) - Coarse prediction from Stage 1
                protocol: (B, 2) - [T_lr, T_hr] in mm
                num_steps: Fixed step count (used if not adaptive)
                adaptive: Use adaptive stepping based on SFI difficulty

            Returns:
                x_1: (B, C, H, W) - Refined output
                info: Dict with diagnostic information
            """
            B = x_0.shape[0]
            device = x_0.device
            info = {}

            # Determine step count
            if adaptive and self.use_adaptive_steps:
                # Always use compute_sfi_adaptive_steps which handles both PAD and legacy SFI
                actual_steps, sfi_metrics = self.compute_sfi_adaptive_steps(protocol)
                info.update(sfi_metrics)  # Includes difficulty, difficulty_mode (pad or legacy sfi)
            else:
                actual_steps = num_steps if num_steps is not None else self.num_inference_steps

            info['num_steps'] = actual_steps
            info['adaptive'] = adaptive and self.use_adaptive_steps

            dt = 1.0 / actual_steps

            x = x_0
            for i in range(actual_steps):
                t = torch.full((B,), i / actual_steps, device=device)
                v = self.compute_velocity(x, t, protocol)
                x = x + v * dt

            return x, info

        def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
            x_lr = batch['x_lr']
            y_hr = batch['y_hr']
            protocol = batch['protocol']

            with torch.no_grad():
                if self.stage1 is not None:
                    y_coarse = self.stage1(x_lr, protocol)
                else:
                    y_coarse = x_lr

                # Use adaptive stepping if enabled
                y_pred, info = self.ode_integrate(
                    y_coarse,
                    protocol,
                    num_steps=self.num_inference_steps,
                    adaptive=self.use_adaptive_steps,
                )

            if TORCHMETRICS_AVAILABLE:
                psnr = psnr_metric(y_pred, y_hr, data_range=2.0)
                ssim = ssim_metric(y_pred, y_hr, data_range=2.0)
            else:
                mse = F.mse_loss(y_pred, y_hr)
                psnr = 10 * torch.log10(4.0 / mse)
                ssim = torch.tensor(0.0, device=y_pred.device)

            # Select which batch to log based on epoch (for diverse visualization)
            # Use epoch number to cycle through batches for varied samples
            log_batch_idx = self.current_epoch % 10  # Target batch index (0-9)
            should_log_images = (batch_idx == log_batch_idx)

            # Compute velocity at t=0 for visualization (selected batch only)
            v_pred_vis = None
            if should_log_images:
                with torch.no_grad():
                    B = x_lr.shape[0]
                    t_zero = torch.zeros(B, device=x_lr.device)
                    v_pred_vis = self.compute_velocity(y_coarse, t_zero, protocol)

            output = {
                'psnr': psnr,
                'ssim': ssim,
                'num_steps': info.get('num_steps', self.num_inference_steps),
                'x_lr': x_lr[:1].detach() if should_log_images else None,
                'y_coarse': y_coarse[:1].detach() if should_log_images else None,
                'y_pred': y_pred[:1].detach() if should_log_images else None,
                'y_hr': y_hr[:1].detach() if should_log_images else None,
                'protocol': protocol[:1].detach() if should_log_images else None,
                'info': info if should_log_images else None,
                'v_pred': v_pred_vis[:1].detach() if v_pred_vis is not None else None,
            }

            # Collect CETA outputs for visualization (selected batch only)
            # Debug: print conditions when this is the selected batch for image logging
            if should_log_images:
                print(f"[CETA Debug] epoch={self.current_epoch}, batch_idx={batch_idx}, "
                      f"use_ceta={self.use_ceta}, x_lr_alt in batch={'x_lr_alt' in batch}")
            if should_log_images and self.use_ceta and 'x_lr_alt' in batch:
                with torch.no_grad():
                    x_lr_alt = batch['x_lr_alt']
                    protocol_alt = batch['protocol_alt']

                    # Get Stage 1 output for alternative LR
                    if self.stage1 is not None:
                        y_coarse_alt = self.stage1(x_lr_alt, protocol_alt)
                    else:
                        y_coarse_alt = x_lr_alt

                    # Compute velocities at t=0 (from coarse to HR)
                    # At t=0: x_t = y_coarse, and v predicts direction to y_hr
                    B = x_lr.shape[0]
                    t_zero = torch.zeros(B, device=x_lr.device)

                    v_pred = self.compute_velocity(y_coarse, t_zero, protocol)
                    v_pred_alt = self.compute_velocity(y_coarse_alt, t_zero, protocol_alt)

                    # Estimated endpoints: ŷ = y_coarse + v
                    y_est = y_coarse + v_pred
                    y_est_alt = y_coarse_alt + v_pred_alt

                    output['ceta_data'] = {
                        'x_lr_alt': x_lr_alt[:1].detach(),
                        'y_coarse_alt': y_coarse_alt[:1].detach(),
                        'v_pred': v_pred[:1].detach(),
                        'v_pred_alt': v_pred_alt[:1].detach(),
                        'y_est': y_est[:1].detach(),
                        'y_est_alt': y_est_alt[:1].detach(),
                        't_lr_alt': protocol_alt[0, 0].item(),
                    }

            self.val_outputs.append(output)
            return output

        def on_validation_epoch_end(self):
            if not self.val_outputs:
                return

            avg_psnr = torch.stack([x['psnr'] for x in self.val_outputs]).mean()
            avg_ssim = torch.stack([x['ssim'] for x in self.val_outputs]).mean()
            avg_steps = sum(x['num_steps'] for x in self.val_outputs) / len(self.val_outputs)

            self.log('val_psnr', avg_psnr, prog_bar=True, sync_dist=True)
            self.log('val_ssim', avg_ssim, prog_bar=True, sync_dist=True)
            self.log('val/avg_steps', avg_steps, sync_dist=True)

            # Log SFI difficulty info from first batch (for ablation study)
            first_info = next((x['info'] for x in self.val_outputs if x.get('info') is not None), None)
            if first_info is not None:
                if 'difficulty' in first_info:
                    self.log('val/difficulty', first_info['difficulty'], sync_dist=True)
                if 'max_difficulty' in first_info:
                    self.log('val/max_difficulty', first_info['max_difficulty'], sync_dist=True)
                if 'info_gain' in first_info:
                    self.log('val/info_gain', first_info['info_gain'], sync_dist=True)
                if 'physical_penalty' in first_info:
                    self.log('val/physical_penalty', first_info['physical_penalty'], sync_dist=True)
                # Log difficulty mode as a string (useful for ablation comparison)
                if 'difficulty_mode' in first_info and self.current_epoch == 0:
                    mode = first_info['difficulty_mode']
                    print(f"[Stage2] Adaptive stepping: {mode.upper()}" +
                          (f" (PAD = 1 - T_hr/T_lr)" if mode == 'pad' else ""))

            # Log images (only on rank 0)
            # Find output with images (selected by epoch-based batch idx, fallback to first available)
            is_rank_zero = self.global_rank == 0
            if WANDB_AVAILABLE and self.logger is not None and is_rank_zero:
                img_output = None
                for output in self.val_outputs:
                    if output.get('x_lr') is not None:
                        img_output = output
                        break

                if img_output is not None:
                    try:
                        t_lr = img_output['protocol'][0, 0].item()
                        t_hr = img_output['protocol'][0, 1].item()
                        info = img_output.get('info', {})
                        steps_used = info.get('num_steps', self.num_inference_steps)

                        # Get CETA data if available, or create minimal data with v_pred
                        ceta_data = img_output.get('ceta_data', None)

                        # If no CETA but v_pred is available, pass it for visualization
                        if ceta_data is None and img_output.get('v_pred') is not None:
                            ceta_data = {'v_pred': img_output['v_pred']}

                        img = _create_cascaded_2d_comparison_figure(
                            img_output['x_lr'][0],
                            img_output['y_coarse'][0],
                            img_output['y_pred'][0],
                            img_output['y_hr'][0],
                            t_lr=t_lr,
                            t_hr=t_hr,
                            ceta_data=ceta_data,
                        )

                        # Build caption with CETA info if available
                        caption = f'Epoch {self.current_epoch} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | Steps: {steps_used}'
                        if ceta_data is not None:
                            t_lr_alt = ceta_data.get('t_lr_alt', 0.0)
                            caption += f' | CETA: {t_lr:.1f}mm vs {t_lr_alt:.1f}mm'

                        if hasattr(self.logger, 'experiment'):
                            self.logger.experiment.log({
                                'val/images': wandb.Image(img, caption=caption),
                            }, commit=False)
                            ceta_status = ""
                            if ceta_data is not None:
                                if 'x_lr_alt' in ceta_data:
                                    ceta_status = " (with CETA trajectory)"
                                elif 'v_pred' in ceta_data:
                                    ceta_status = " (with v_pred only)"
                            print(f"[Rank 0] Successfully logged validation image at epoch {self.current_epoch}{ceta_status}")

                        # Save JPG every N epochs for ablation study visualization
                        if self.save_epoch_images and self.current_epoch % self.save_epoch_interval == 0:
                            self._save_epoch_images_to_disk(img_output, ceta_data, avg_psnr, avg_ssim)

                    except Exception as e:
                        print(f"[Rank 0] Failed to log images: {e}")
                        import traceback
                        traceback.print_exc()

            self.val_outputs.clear()

        def _save_epoch_images_to_disk(
            self,
            img_output: Dict,
            ceta_data: Optional[Dict],
            avg_psnr: torch.Tensor,
            avg_ssim: torch.Tensor,
        ):
            """
            Save validation images to disk as JPG files for ablation study visualization.

            Saves:
            - Individual images: LR, Stage1, Final, HR, Velocity
            - If CETA enabled: alternative trajectory and endpoint difference
            - Composite comparison figure

            Directory structure:
            {save_epoch_dir}/
                epoch_{N:04d}/
                    lr.jpg
                    stage1.jpg
                    final.jpg
                    hr.jpg
                    velocity.jpg (if available)
                    diff_stage1_hr.jpg
                    diff_final_hr.jpg
                    comparison.jpg (composite figure)
                    ceta/ (if CETA enabled)
                        lr_alt.jpg
                        stage1_alt.jpg
                        velocity_alt.jpg
                        y_est.jpg
                        y_est_alt.jpg
                        endpoint_diff.jpg
            """
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image
            import os

            if self.save_epoch_dir is None:
                print(f"[Epoch {self.current_epoch}] save_epoch_dir not set, skipping disk save")
                return

            epoch_dir = Path(self.save_epoch_dir) / f'epoch_{self.current_epoch:04d}'
            epoch_dir.mkdir(parents=True, exist_ok=True)

            t_lr = img_output['protocol'][0, 0].item()
            t_hr = img_output['protocol'][0, 1].item()

            def to_img(tensor):
                """Convert tensor to [0, 1] for saving."""
                return np.clip((tensor.squeeze().cpu().numpy() + 1) / 2, 0, 1)

            def save_grayscale_jpg(img_array, path, cmap='gray'):
                """Save grayscale image as JPG."""
                plt.figure(figsize=(4, 4))
                plt.imshow(img_array, cmap=cmap, vmin=0, vmax=1)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()

            def save_diff_jpg(diff_array, path, vmax=0.5):
                """Save difference map as JPG with hot colormap."""
                plt.figure(figsize=(4, 4))
                plt.imshow(diff_array, cmap='hot', vmin=0, vmax=vmax)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()

            def save_velocity_jpg(v_array, path):
                """Save velocity field as JPG with RdBu colormap."""
                vmax = max(np.abs(v_array).max(), 0.5)
                plt.figure(figsize=(4, 4))
                plt.imshow(v_array, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()

            # Convert tensors to images
            lr_img = to_img(img_output['x_lr'][0])
            coarse_img = to_img(img_output['y_coarse'][0])
            final_img = to_img(img_output['y_pred'][0])
            hr_img = to_img(img_output['y_hr'][0])

            # Save individual images
            save_grayscale_jpg(lr_img, epoch_dir / 'lr.jpg')
            save_grayscale_jpg(coarse_img, epoch_dir / 'stage1.jpg')
            save_grayscale_jpg(final_img, epoch_dir / 'final.jpg')
            save_grayscale_jpg(hr_img, epoch_dir / 'hr.jpg')

            # Save difference maps
            diff_stage1 = np.abs(coarse_img - hr_img)
            diff_final = np.abs(final_img - hr_img)
            save_diff_jpg(diff_stage1, epoch_dir / 'diff_stage1_hr.jpg')
            save_diff_jpg(diff_final, epoch_dir / 'diff_final_hr.jpg')

            # Save velocity if available
            if ceta_data is not None and 'v_pred' in ceta_data:
                v_pred = ceta_data['v_pred'].squeeze().cpu().numpy()
                save_velocity_jpg(v_pred, epoch_dir / 'velocity.jpg')

            # Save CETA outputs if available
            if ceta_data is not None and 'x_lr_alt' in ceta_data:
                ceta_dir = epoch_dir / 'ceta'
                ceta_dir.mkdir(exist_ok=True)

                lr_alt_img = to_img(ceta_data['x_lr_alt'])
                coarse_alt_img = to_img(ceta_data['y_coarse_alt'])
                y_est_img = to_img(ceta_data['y_est'])
                y_est_alt_img = to_img(ceta_data['y_est_alt'])
                v_pred_img = ceta_data['v_pred'].squeeze().cpu().numpy()
                v_pred_alt_img = ceta_data['v_pred_alt'].squeeze().cpu().numpy()
                endpoint_diff = np.abs(y_est_img - y_est_alt_img)

                save_grayscale_jpg(lr_alt_img, ceta_dir / 'lr_alt.jpg')
                save_grayscale_jpg(coarse_alt_img, ceta_dir / 'stage1_alt.jpg')
                save_grayscale_jpg(y_est_img, ceta_dir / 'y_est.jpg')
                save_grayscale_jpg(y_est_alt_img, ceta_dir / 'y_est_alt.jpg')
                save_velocity_jpg(v_pred_img, ceta_dir / 'velocity.jpg')
                save_velocity_jpg(v_pred_alt_img, ceta_dir / 'velocity_alt.jpg')
                save_diff_jpg(endpoint_diff, ceta_dir / 'endpoint_diff.jpg', vmax=0.3)

                # Save CETA comparison figure
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                t_lr_alt = ceta_data.get('t_lr_alt', 0.0)

                fig.suptitle(
                    f'CETA Comparison | Epoch {self.current_epoch} | '
                    f'Primary: {t_lr:.2f}mm | Alt: {t_lr_alt:.2f}mm',
                    fontsize=14, fontweight='bold'
                )

                # Row 1: Primary trajectory
                axes[0, 0].imshow(lr_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 0].set_title(f'LR ({t_lr:.1f}mm)', fontweight='bold')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(coarse_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title('Stage1', fontweight='bold')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(y_est_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 2].set_title('y_est (Endpoint)', fontweight='bold')
                axes[0, 2].axis('off')

                axes[0, 3].imshow(hr_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 3].set_title('GT', fontweight='bold')
                axes[0, 3].axis('off')

                # Row 2: Alternative trajectory
                axes[1, 0].imshow(lr_alt_img, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title(f'LR Alt ({t_lr_alt:.1f}mm)', fontweight='bold')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(coarse_alt_img, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title('Stage1 Alt', fontweight='bold')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(y_est_alt_img, cmap='gray', vmin=0, vmax=1)
                axes[1, 2].set_title('y_est Alt', fontweight='bold')
                axes[1, 2].axis('off')

                axes[1, 3].imshow(endpoint_diff, cmap='hot', vmin=0, vmax=0.3)
                axes[1, 3].set_title('|y_est - y_est_alt|', fontweight='bold')
                axes[1, 3].axis('off')

                plt.tight_layout()
                plt.savefig(ceta_dir / 'comparison.jpg', dpi=150, bbox_inches='tight')
                plt.close()

            # Save main comparison figure
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            fig.suptitle(
                f'Epoch {self.current_epoch} | {t_lr:.2f}mm → {t_hr:.2f}mm | '
                f'PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}',
                fontsize=14, fontweight='bold'
            )

            # Row 1: Images
            axes[0, 0].imshow(lr_img, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title('LR Input', fontweight='bold')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(coarse_img, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('Stage1 (Coarse)', fontweight='bold')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(final_img, cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title('Final (Stage2)', fontweight='bold')
            axes[0, 2].axis('off')

            axes[0, 3].imshow(hr_img, cmap='gray', vmin=0, vmax=1)
            axes[0, 3].set_title('GT', fontweight='bold')
            axes[0, 3].axis('off')

            # Row 2: Differences and velocity
            axes[1, 0].imshow(diff_stage1, cmap='hot', vmin=0, vmax=0.5)
            axes[1, 0].set_title('|Stage1 - GT|', fontweight='bold')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(diff_final, cmap='hot', vmin=0, vmax=0.5)
            axes[1, 1].set_title('|Final - GT|', fontweight='bold')
            axes[1, 1].axis('off')

            if ceta_data is not None and 'v_pred' in ceta_data:
                v_pred = ceta_data['v_pred'].squeeze().cpu().numpy()
                vmax = max(np.abs(v_pred).max(), 0.5)
                axes[1, 2].imshow(v_pred, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[1, 2].set_title('Velocity (t=0)', fontweight='bold')
            else:
                axes[1, 2].axis('off')
            axes[1, 2].axis('off')

            # Improvement visualization
            improvement = diff_stage1 - diff_final  # Positive = Stage2 is better
            axes[1, 3].imshow(improvement, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
            axes[1, 3].set_title('Improvement (red=S2 better)', fontweight='bold')
            axes[1, 3].axis('off')

            plt.tight_layout()
            plt.savefig(epoch_dir / 'comparison.jpg', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Epoch {self.current_epoch}] Saved epoch images to {epoch_dir}")

        def configure_optimizers(self):
            params = self.rf_model.parameters()

            optimizer = torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            def lr_lambda(step):
                if step < self.warmup_steps:
                    # Use (step + 1) to ensure LR > 0 at step=0
                    # step=0 → (1/warmup_steps), step=warmup_steps-1 → 1.0
                    return (step + 1) / max(1, self.warmup_steps)
                progress = (step - self.warmup_steps) / max(1, self.max_steps_training - self.warmup_steps)
                return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Test APNUNet2D
    print("\n" + "=" * 50)
    print("Testing APNUNet2D (Stage 1)")
    print("=" * 50)

    model = APNUNet2D(model_config='tiny').to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.2f}M")

    x = torch.randn(2, 1, 128, 128).to(device)
    protocol = torch.tensor([[3.0, 0.7], [5.0, 1.0]]).to(device)

    with torch.no_grad():
        y = model(x, protocol)

    print(f"Input: {x.shape} → Output: {y.shape}")

    # Test RFVelocityUNet2D
    print("\n" + "=" * 50)
    print("Testing RFVelocityUNet2D (Stage 2)")
    print("=" * 50)

    rf_model = RFVelocityUNet2D(model_config='tiny').to(device)
    num_params = sum(p.numel() for p in rf_model.parameters()) / 1e6
    print(f"Parameters: {num_params:.2f}M")

    t = torch.rand(2).to(device)

    with torch.no_grad():
        v = rf_model(x, t, protocol)

    print(f"Input: {x.shape}, t: {t.shape} → Output: {v.shape}")
    print(f"Initial velocity magnitude: {v.abs().mean().item():.6f} (should be ~0)")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
