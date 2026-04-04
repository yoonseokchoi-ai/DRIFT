# 🌊 DRIFT: Difficulty-aware Rectified Flows for Through-plane MRI Super-Resolution

> **ECCV 2026 Submission**  
> Reconstructs isotropic MRI volumes from anisotropic thick-slice acquisitions using a two-stage rectified flow framework with physics-aware adaptive inference.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Lightning-2.5+-792ee5?logo=pytorchlightning" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python" />
</p>

---

## 📋 Overview

DRIFT addresses the **efficiency–fidelity trade-off** in through-plane MRI super-resolution:

| Stage | Module | Role | Paper Reference |
|-------|--------|------|-----------------|
| **Stage 1** | APN (Anatomical Projection Network) | Maps LR patches → coarse HR manifold | Sec. 3.3, Eq. 4–5 |
| **Stage 2** | Rectified Flow (velocity network v_θ) | Refines high-frequency details via ODE | Sec. 3.4, Eq. 6–10 |

**Key innovations:**
- 🎯 **PAD** (Physics-Aware Difficulty): metadata-driven difficulty metric from slice-thickness (Eq. 13)
- ⚡ **AIS** (Adaptive Integration Scheduler): allocates ODE steps by thickness — fewer for easy, more for hard cases (Eq. 14)
- 🔗 **CETA** (Consistent Endpoint Trajectory Alignment): enforces thickness-consistent reconstructions via proximal pairs (Eq. 11–12)

---

## 📁 Project Structure

```
drift/
├── train.py                 # 🏋️ Training script (Stage 1 & 2)
├── inference.py             # 🔬 Inference script (single volume / batch)
├── requirements.txt         # 📦 Python dependencies
├── models/
│   ├── __init__.py
│   └── drift_2d.py          # 🧠 APNUNet2D, RFVelocityUNet2D, Lightning modules
├── dataset/
│   ├── __init__.py
│   ├── mri_utils.py          # 🩺 MRI orientation & protocol constants
│   └── drift_2d_dataset.py   # 📊 Dataset & DataModule (SLR simulation)
└── config/
    ├── drift_2d_config.yaml          # HCP (0.7mm isotropic)
    ├── drift_2d_mind_config.yaml     # MIND (0.9mm isotropic)
    └── drift_2d_ideas_config.yaml    # IDEAS (1.0mm isotropic)
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yoonseokchoi-ai/DRIFT.git
cd DRIFT
```

### 2. Create conda environment

```bash
conda create -n drift python=3.11 -y
conda activate drift
```

### 3. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

> 💡 For other CUDA versions, see [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Install dependencies

```bash
pip install pytorch-lightning==2.5.0 monai==1.4.0 nibabel==5.3.2 \
    torchmetrics==1.6.1 scikit-image==0.25.0 einops==0.8.0 \
    rich==13.9.4 wandb pyyaml tqdm sigpy
```

> 📌 `sigpy` is required for SLR (Shinnar-Le Roux) slice profile simulation. If unavailable, the code falls back to a Gaussian approximation.

### 5. (Optional) Verify installation

```bash
python -c "
import sys; sys.path.insert(0, '.')
from models.drift_2d import APNUNet2D, RFVelocityUNet2D
import torch
x = torch.randn(1, 1, 128, 128)
p = torch.tensor([[4.0, 0.7]])
model = APNUNet2D(model_config='tiny')
print(f'✅ APNUNet2D output: {model(x, p).shape}')
rf = RFVelocityUNet2D(model_config='tiny')
t = torch.tensor([0.5])
print(f'✅ RFVelocityUNet2D output: {rf(x, t, p).shape}')
"
```

---

## 📊 Data Preparation

DRIFT supports **three public brain MRI datasets** with isotropic HR ground truth:

| Dataset | Resolution | Modalities | Train/Test | Volume Size |
|---------|-----------|------------|------------|-------------|
| [HCP](https://www.humanconnectome.org/) | 0.7 mm | T1w, T2w | 890 / 223 | 320×320×320 |
| [MIND](https://openneuro.org/datasets/ds006391) | 0.9 mm | T1w, T2w | 411 / 102 | 256×256×256 |
| [IDEAS](https://openneuro.org/datasets/ds004199) | 1.0 mm | T1w, FLAIR | 110 / 25 | 256×256×256 |

### Data format

DRIFT supports two data formats. Choose one and set the paths in your config.

#### Option A: Pre-extracted 2D slices (NPY, recommended ⚡)

Pre-extract 2D slices from 3D volumes to avoid repeated disk I/O during training. Each `.npy` file is a single 2D slice of shape `(H, W)` in `float32`, normalized to `[0, 1]`.

```
/your/data/path/hcp_2d_npy/          # <-- your path here
├── train/
│   └── slices/
│       ├── 100206_t1_axi_032.npy    # naming: {subject}_{modality}_{plane}_{idx}.npy
│       ├── 100206_t1_axi_034.npy
│       └── ...
└── test/
    └── slices/
        ├── 200109_t1_axi_032.npy
        └── ...
```

#### Option B: On-the-fly from 3D volumes (NIfTI)

Place 3D NIfTI volumes in a directory — slices are extracted on-the-fly during training (slower, but no preprocessing needed):

```
/your/data/path/hcp_nifti/           # <-- your path here
├── train/
│   ├── 100206_t1.nii.gz
│   ├── 100206_t2.nii.gz
│   └── ...
└── val/
    ├── 200109_t1.nii.gz
    └── ...
```

### ⚠️ Update config paths (required)

Before training, you **must** edit the data paths in your config file to match your local environment. Open `config/drift_2d_config.yaml` and update the following fields:

```yaml
data:
  use_precomputed: true                          # true for NPY (Option A), false for NIfTI (Option B)
  data_format: npy
  precomputed_path: /your/data/path/hcp_2d_npy   # ← CHANGE THIS to your NPY data directory
  data_path: /your/data/path/hcp_nifti            # ← CHANGE THIS to your NIfTI data directory (Option B)

wandb:
  save_dir: /your/output/path/                    # ← CHANGE THIS to where you want checkpoints & logs
```

> 💡 **Tip:** Similarly update `config/drift_2d_mind_config.yaml` and `config/drift_2d_ideas_config.yaml` if you use MIND or IDEAS datasets. Each config has the same `data.precomputed_path` and `data.data_path` fields to update.

---

## 🏋️ Training

### Stage 1: APN (Anatomical Projection Network)

Trains f_φ to project LR patches onto the coarse HR manifold (Sec. 3.3):

```bash
python train.py \
    --config config/drift_2d_config.yaml \
    --modality t1 \
    --stage 1
```

**Expected output:**
- Best checkpoint saved to `checkpoints/stage1_2d_best-epoch*-val_ssim*.ckpt`
- Training logs on WandB (if enabled)

### Stage 2: Rectified Flow Refinement

Trains v_θ to refine high-frequency details with frozen Stage 1 (Sec. 3.4):

```bash
python train.py \
    --config config/drift_2d_config.yaml \
    --modality t1 \
    --stage 2 \
    --stage1-ckpt /path/to/stage1_best.ckpt   # ← replace with your Stage 1 checkpoint path
```

> 📌 The Stage 1 checkpoint path is printed at the end of Stage 1 training as `Best checkpoint: ...`.

### Multi-GPU training

DRIFT uses PyTorch Lightning's DDP strategy. Configure in the YAML:

```yaml
hardware:
  accelerator: gpu
  devices: 8          # number of GPUs
  strategy: auto      # or 'ddp'
  precision: 16-mixed # mixed precision for memory efficiency
```

Or use a subset of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --config config/drift_2d_config.yaml \
    --modality t1 --stage 1
```

### Training tips

| Setting | Stage 1 (APN) | Stage 2 (RF) |
|---------|---------------|--------------|
| Learning rate | 1e-4 | 5e-5 |
| Batch size / GPU | 64 | 25 |
| Epochs | 100 | 100 |
| Key loss | Charbonnier + SSIM | Huber (velocity) + CETA |

---

## 🔬 Inference

### Single volume (NIfTI)

```bash
python inference.py \
    --stage1-ckpt /path/to/stage1.ckpt \       # ← your Stage 1 checkpoint
    --stage2-ckpt /path/to/stage2.ckpt \       # ← your Stage 2 checkpoint
    --input /path/to/thick_slice_volume.nii.gz \  # ← input NIfTI volume to super-resolve
    --output-dir /path/to/output \             # ← directory to save results
    --t-lr 5.0 \
    --t-hr 0.7 \
    --adaptive
```

| Argument | Description | Example |
|----------|-------------|---------|
| `--t-lr` | Input slice thickness in mm | `5.0` (5mm thick-slice scan) |
| `--t-hr` | Target thickness in mm (native resolution of training data) | `0.7` (HCP), `0.9` (MIND), `1.0` (IDEAS) |
| `--adaptive` | Enable PAD-based AIS adaptive stepping (recommended) | — |

### Batch evaluation on test set

```bash
python inference.py \
    --config config/drift_2d_config.yaml \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_npy_data \   # ← directory with test NPY slices
    --output-dir /path/to/results \
    --mode batch \
    --adaptive
```

### Sliding-window inference (for full-resolution slices)

```bash
python inference.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --input /path/to/volume.nii.gz \
    --output-dir /path/to/output \
    --t-lr 5.0 --t-hr 0.7 \
    --sliding-window \
    --patch-size 128 --overlap 32 \
    --adaptive
```

### Save velocity fields (for visualization)

```bash
python inference.py \
    --stage2-ckpt /path/to/stage2.ckpt \
    --input /path/to/volume.nii.gz \
    --output-dir /path/to/output \
    --t-lr 5.0 --t-hr 0.7 \
    --save-velocity \
    --save-intermediate-steps
```

---

## 🧠 Paper ↔ Code Reference

| Paper Concept | Code Class / Variable | Location |
|---------------|----------------------|----------|
| APN (f_φ) | `APNUNet2D` | `models/drift_2d.py` |
| Velocity network (v_θ) | `RFVelocityUNet2D` | `models/drift_2d.py` |
| Stage 1 Lightning | `Stage1APNLightning` | `models/drift_2d.py` |
| Stage 2 Lightning | `Stage2RFLightning` | `models/drift_2d.py` |
| AdaGN (Eq. 3) | `AdaGN2D` | `models/drift_2d.py` |
| τ = 1/T embedding (Eq. 2) | `ProtocolEmbedding2D` | `models/drift_2d.py` |
| Time + Protocol (Eq. 7) | `TimeProtocolEmbedding2D` | `models/drift_2d.py` |
| L_Char + L_SSIM (Eq. 5) | `CharbonnierLoss`, `SSIM2DLoss` | `models/drift_2d.py` |
| PAD (Eq. 13) | `compute_sfi_adaptive_steps()` | `models/drift_2d.py` |
| SLR slice profile (Eq. S1) | `drift_2d_dataset.py` | `dataset/` |

---

## ⚙️ Configuration

Key settings in `config/drift_2d_config.yaml`:

```yaml
model:
  model_config: large    # 'tiny' (5M), 'small' (12M), 'base' (25M), 'large' (50M)

stage2:
  use_ceta: true         # CETA loss (Sec. 3.5)
  ceta_weight: 1.0       # λ_ceta (Eq. 12)
  ceta_mode: fixed       # proximal gap mode
  ceta_gap_mm: 1.0       # ΔT = 1mm (Table 3)

  use_adaptive_steps: true  # PAD-based AIS (Sec. 3.6)
  max_ode_steps: 10         # N_max (Eq. 14)
  min_ode_steps: 2          # N_min

loss:
  use_u_shaped_sampling: true  # endpoint-biased timestep (Eq. 10)
  u_shape_power: 2.0           # α = 2.0 (Table S1)

data:
  patch_size: [128, 128]
  lr_thickness_range: [0.7, 6.0]  # T_i ~ U(T_hr, 6.0] mm
  slice_profile: slr               # SLR-based simulation (Sec. 3.1)
```

---

## 📄 Citation

```bibtex
@inproceedings{drift2026,
  title={DRIFT: Difficulty-aware Rectified Flows for Through-plane MRI Super-Resolution},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## 📜 License

This project is for academic research purposes. Please contact the authors for commercial use.
