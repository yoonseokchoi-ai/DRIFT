"""
DRIFT Training Script
=====================

Train DRIFT (Difficulty-aware Rectified Flows for Through-plane MRI SR):
  Stage 1: APN (x_lr → z) - Anatomical Projection Network (Sec. 3.3)
  Stage 2: RF  (z → ỹ)   - Rectified Flow refinement (Sec. 3.4)

Usage:
    # Stage 1 Training (APN)
    python drift/train.py --config drift/config/drift_2d_config.yaml \
        --modality t1 --stage 1

    # Stage 2 Training (requires Stage 1 checkpoint)
    python drift/train.py --config drift/config/drift_2d_config.yaml \
        --modality t1 --stage 2 --stage1-ckpt /path/to/stage1.ckpt
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl

# PyTorch 2.6+ requires explicit allowlisting of custom classes for checkpoint loading
# Add MONAI's TraceKeys to safe globals to allow resume from checkpoints
try:
    from monai.utils.enums import TraceKeys
    torch.serialization.add_safe_globals([TraceKeys])
except ImportError:
    pass
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Add drift package root to path (must come before project root to avoid name conflicts)
current_dir = os.path.dirname(os.path.abspath(__file__))
drift_root = os.path.abspath(current_dir)
if drift_root not in sys.path:
    sys.path.insert(0, drift_root)

from models.drift_2d import Stage1APNLightning, Stage2RFLightning
from dataset.drift_2d_dataset import DRIFT2DDataModuleV2


def main(config: dict, args):
    pl.seed_everything(config['training'].get('seed', 42))

    # Determine training stage
    stage = args.stage
    assert stage in [1, 2], f"Invalid stage: {stage}. Must be 1 or 2."

    # Reconcile GPU settings
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_gpus = [s.strip() for s in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if s.strip()]
        num_visible_gpus = len(visible_gpus)
        num_requested_gpus = config['hardware']['devices']
        if isinstance(num_requested_gpus, list):
            num_requested_gpus = len(num_requested_gpus)
        if num_requested_gpus > num_visible_gpus:
            print(f"Warning: Config requests {num_requested_gpus} GPUs, but only {num_visible_gpus} are visible.")
            config['hardware']['devices'] = num_visible_gpus

    num_gpus = config['hardware'].get('devices', 8)
    if isinstance(num_gpus, list):
        num_gpus = len(num_gpus)

    # Data module (2D) - supports NPY, HDF5, and on-the-fly
    contrast = args.modality
    use_precomputed = config['data'].get('use_precomputed', True)
    data_format = config['data'].get('data_format', 'npy')  # 'npy', 'h5', or 'onthefly'

    # Get stage2 config for CETA (needed for data module)
    stage2_config = config.get('stage2', {})

    # Determine modality filter for NPY format (BraTS21)
    # 'all' means no filtering (use all modalities)
    modality_filter = None if contrast == 'all' else contrast

    print("Initializing data module...")
    if use_precomputed:
        format_names = {'npy': 'NPY (individual files)', 'h5': 'HDF5 (chunks)'}
        print(f"  Mode: {format_names.get(data_format, data_format)}")
        data_module = DRIFT2DDataModuleV2(
            use_precomputed=True,
            data_format=data_format,
            precomputed_path=config['data']['precomputed_path'],
            onthefly_path=config['data'].get('data_path'),
            contrast=contrast,
            patch_size=tuple(config['data']['patch_size']),
            lr_thickness_range=tuple(config['data']['lr_thickness_range']),
            tgt_thickness_range=tuple(config['data']['tgt_thickness_range']),
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            slice_profile=config['data'].get('slice_profile', 'slr'),
            max_train_slices=config['data'].get('max_train_slices'),
            max_val_slices=config['data'].get('max_val_slices'),
            # CETA support (only for Stage 2)
            use_ceta=stage2_config.get('use_ceta', False) if stage == 2 else False,
            ceta_mode=stage2_config.get('ceta_mode', 'fixed'),
            ceta_ratio=stage2_config.get('ceta_ratio', 0.6),
            ceta_anchor=stage2_config.get('ceta_anchor', 2.0),
            ceta_gap_mm=stage2_config.get('ceta_gap_mm', 1.0),
            ceta_delta_pad=stage2_config.get('ceta_delta_pad', 0.1),
            # Modality filtering (for BraTS21 NPY format with modality in filename)
            modality=modality_filter,
            # Degradation mode: 'legacy' (blur→area downsample→nearest) or 'physical' (blur with stride→nearest)
            degradation_mode=config['data'].get('degradation_mode', 'legacy'),
            # PAD-weighted LR thickness sampling: 'uniform' or 'pad_weighted'
            lr_sampling_mode=config['data'].get('lr_sampling_mode', 'uniform'),
            pad_sampling_alpha=config['data'].get('pad_sampling_alpha', 2.0),
            # Native resolution (T_hr): HCP=0.7mm, BraTS/MIND/IDEAS=1.0mm
            native_resolution=config['data'].get('native_resolution', 0.7),
        )
    else:
        print("  Mode: On-the-fly (flexible)")
        data_module = DRIFT2DDataModuleV2(
            use_precomputed=False,
            onthefly_path=config['data']['data_path'],
            contrast=contrast,
            patch_size=tuple(config['data']['patch_size']),
            lr_thickness_range=tuple(config['data']['lr_thickness_range']),
            tgt_thickness_range=tuple(config['data']['tgt_thickness_range']),
            slices_per_volume=config['data'].get('slices_per_volume', 32),
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            train_num_volumes=config['data'].get('train_num_volumes'),
            val_num_volumes=config['data'].get('val_num_volumes'),
            slice_profile=config['data'].get('slice_profile', 'slr'),
            # Degradation mode
            degradation_mode=config['data'].get('degradation_mode', 'legacy'),
            # PAD-weighted LR thickness sampling: 'uniform' or 'pad_weighted'
            lr_sampling_mode=config['data'].get('lr_sampling_mode', 'uniform'),
            pad_sampling_alpha=config['data'].get('pad_sampling_alpha', 2.0),
            # Native resolution (T_hr): HCP=0.7mm, BraTS/MIND/IDEAS=1.0mm
            native_resolution=config['data'].get('native_resolution', 0.7),
        )

    # Get stage-specific settings
    stage_key = f'stage{stage}'
    stage_config = config.get(stage_key, {})

    learning_rate = float(stage_config.get('learning_rate', 1e-4))
    max_epochs = stage_config.get('max_epochs', 100)
    max_steps = stage_config.get('max_steps')

    # Calculate dynamic warmup_steps (5% of total steps)
    # Get train dataloader to compute total steps
    print("Computing training steps...")
    train_loader = data_module.train_dataloader()
    steps_per_epoch = len(train_loader) // config['training'].get('accumulate_grad_batches', 1)
    steps_per_epoch = max(1, steps_per_epoch // num_gpus) if num_gpus > 1 else steps_per_epoch

    if max_steps:
        total_steps = max_steps
    else:
        total_steps = steps_per_epoch * max_epochs

    warmup_ratio = config['training'].get('warmup_ratio', 0.05)  # Default 5%
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(100, warmup_steps)  # Minimum 100 steps

    print(f"Dynamic warmup: {warmup_ratio*100:.0f}% of {total_steps} = {warmup_steps} steps")

    # Create model based on stage
    print("Creating model...")
    if stage == 1:
        model = Stage1APNLightning(
            model_config=config['model'].get('model_config', 'base'),
            in_channels=config['model'].get('in_channels', 1),
            protocol_embed_dim=config['model'].get('protocol_embed_dim', 256),
            charbonnier_epsilon=config['loss'].get('charbonnier_epsilon', 1e-3),
            ssim_weight=config['loss'].get('ssim_weight', 0.5),
            ssim_window_size=config['loss'].get('ssim_window_size', 11),
            learning_rate=learning_rate,
            weight_decay=config['training'].get('weight_decay', 0.0),
            warmup_steps=warmup_steps,  # Dynamic warmup
            max_steps=total_steps,
        )
    else:  # stage == 2
        # stage1_ckpt is optional - if None, RF will train directly from x_lr (ablation study)
        if args.stage1_ckpt is None:
            print("=" * 60)
            print("WARNING: Training Stage 2 RF without Stage 1!")
            print("  - Input: x_lr (directly)")
            print("  - This is for ablation study to test RF-only performance")
            print("=" * 60)

        model = Stage2RFLightning(
            stage1_ckpt=args.stage1_ckpt,  # None if not provided
            freeze_stage1=stage2_config.get('freeze_stage1', True),
            model_config=config['model'].get('model_config', 'base'),
            in_channels=config['model'].get('in_channels', 1),
            time_embed_dim=config['model'].get('time_embed_dim', 256),
            protocol_embed_dim=config['model'].get('protocol_embed_dim', 256),
            num_inference_steps=config['inference'].get('num_steps', 4),
            loss_type=config['loss'].get('type', 'huber'),
            use_u_shaped_sampling=config['loss'].get('use_u_shaped_sampling', True),
            u_shape_power=config['loss'].get('u_shape_power', 2.0),
            # CETA configuration
            use_ceta=stage2_config.get('use_ceta', False),
            ceta_weight=stage2_config.get('ceta_weight', 0.1),
            ceta_fixed_t=stage2_config.get('ceta_fixed_t', None),
            # PAD (Physics-Aware Difficulty) based adaptive stepping
            use_adaptive_steps=stage2_config.get('use_adaptive_steps', False),
            use_pad=stage2_config.get('use_pad', True),  # Recommended (no hyperparams)
            max_steps=stage2_config.get('max_ode_steps', 10),  # N_max in PAD formula
            min_steps=stage2_config.get('min_ode_steps', 2),
            # Legacy SFI config (for ablation study comparison)
            use_sfi=stage2_config.get('use_sfi', False),
            base_steps=stage2_config.get('base_steps', 4),
            sfi_alpha=stage2_config.get('sfi_alpha', 0.5),
            max_thickness_ref=stage2_config.get('max_thickness_ref', 6.0),
            difficulty_mode=stage2_config.get('difficulty_mode', 'multiplicative'),
            additive_beta=stage2_config.get('additive_beta', 1.0),
            # Training
            learning_rate=learning_rate,
            weight_decay=config['training'].get('weight_decay', 0.0),
            warmup_steps=warmup_steps,  # Dynamic warmup
            max_steps_training=total_steps,
            # Ablation: disable protocol conditioning
            disable_protocol=stage2_config.get('disable_protocol', False),
            # Epoch-wise image saving for ablation study visualization
            save_epoch_images=stage2_config.get('save_epoch_images', False),
            save_epoch_interval=stage2_config.get('save_epoch_interval', 10),
            save_epoch_dir=stage2_config.get('save_epoch_dir', None),
            fixed_val_sample_idx=stage2_config.get('fixed_val_sample_idx', 0),
        )

    # Logger
    experiment_name = config['logging']['experiment_name']
    experiment_name = experiment_name.replace('<stage>', f'stage{stage}')
    experiment_name = experiment_name.replace('<modality>', contrast)
    experiment_name = experiment_name.replace('<model_config>', config['model'].get('model_config', 'base'))

    if config['logging'].get('use_wandb', False):
        logger = WandbLogger(
            project=config['logging']['project_name'],
            name=experiment_name,
            config=config,
            save_dir=config['wandb'].get('save_dir', '/ssd3/yoonseok/project/')
        )
    else:
        logger = TensorBoardLogger(
            save_dir="logs",
            name=experiment_name,
        )

    # Best checkpoint based on metric (top-k by SSIM)
    best_checkpoint_callback = ModelCheckpoint(
        filename=f"stage{stage}_2d_best" + "-{epoch:03d}-{val_ssim:.4f}",
        save_top_k=config['logging'].get('save_top_k', 3),
        monitor='val_ssim',
        mode='max',
        save_last=True,
    )

    # Periodic checkpoint (every N epochs, default 50)
    periodic_checkpoint_callback = ModelCheckpoint(
        filename=f"stage{stage}_2d_epoch" + "-{epoch:03d}-{val_ssim:.4f}",
        save_top_k=-1,  # Save all periodic checkpoints
        every_n_epochs=config['training'].get('save_every_n_epochs', 50),
        save_last=False,  # Don't overwrite last from best callback
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()

    callbacks = [best_checkpoint_callback, periodic_checkpoint_callback, lr_monitor, progress_bar]

    # Trainer
    # Note: Don't pass max_steps=-1 when using max_epochs, it causes "epoch 0/-2" issue
    trainer_kwargs = {
        'accelerator': config['hardware'].get('accelerator', 'gpu'),
        'devices': config['hardware'].get('devices', 8),
        'strategy': config['hardware'].get('strategy', 'auto'),
        'precision': config['hardware'].get('precision', '16-mixed'),
        'accumulate_grad_batches': config['training'].get('accumulate_grad_batches', 1),
        'gradient_clip_val': config['training'].get('gradient_clip_val', 1.0),
        'logger': logger,
        'callbacks': callbacks,
        'log_every_n_steps': config['logging'].get('log_every_n_steps', 50),
        'check_val_every_n_epoch': config['logging'].get('check_val_every_n_epoch', 1),
        'num_sanity_val_steps': 1,
    }

    # Use either max_epochs or max_steps, not both
    # YAML null becomes Python None, so check explicitly
    if max_steps is not None and max_steps > 0:
        trainer_kwargs['max_steps'] = max_steps
        print(f"[DEBUG] Using max_steps={max_steps}")
    else:
        trainer_kwargs['max_epochs'] = max_epochs
        print(f"[DEBUG] Using max_epochs={max_epochs}")

    trainer = pl.Trainer(**trainer_kwargs)

    # Print configuration
    print("\n" + "=" * 70)
    print(f"DRIFT 2D SR Training - Stage {stage}")
    print("=" * 70)
    print(f"Modality: {contrast}")
    print(f"Model config: {config['model'].get('model_config', 'base')}")
    print("-" * 70)
    print("Data Configuration (2D Slice-by-Slice):")
    print(f"  Data format: {data_format}")
    print(f"  Precomputed path: {config['data'].get('precomputed_path', 'N/A')}")
    print(f"  Patch size: {config['data']['patch_size']} (2D)")
    print(f"  LR thickness range: {config['data']['lr_thickness_range']} mm")
    lr_sampling_mode = config['data'].get('lr_sampling_mode', 'uniform')
    print(f"  LR sampling mode: {lr_sampling_mode}" + (f" (α={config['data'].get('pad_sampling_alpha', 2.0)})" if lr_sampling_mode == 'pad_weighted' else ""))
    print(f"  Target thickness range: {config['data']['tgt_thickness_range']} mm")
    print(f"  Slice profile: {config['data'].get('slice_profile', 'slr')}")
    print(f"  Degradation mode: {config['data'].get('degradation_mode', 'legacy')}")
    print("-" * 70)

    if stage == 1:
        print("Stage 1: Regression 2D U-Net")
        print("  Input: x_lr (2D slice) → Output: ŷ_coarse")
        print("  Loss: Charbonnier + SSIM (2D)")
        print(f"  SSIM weight: {config['loss'].get('ssim_weight', 0.5)}")
    else:
        print("Stage 2: RF Refiner 2D")
        print(f"  Stage 1 checkpoint: {args.stage1_ckpt}")
        print("  Input: ŷ_coarse → Output: y_hr")
        print("  Loss: Velocity matching (2D)")
        # Print adaptive stepping config
        if stage2_config.get('use_adaptive_steps', False):
            print("-" * 70)
            if stage2_config.get('use_pad', True):
                print("Adaptive Stepping (PAD - Physics-Aware Difficulty):")
                print(f"  PAD formula: N_steps = round(N_max × PAD), PAD = 1 - T_hr/T_lr")
                print(f"  Max ODE steps (N_max): {stage2_config.get('max_ode_steps', 10)}")
                print(f"  Min ODE steps: {stage2_config.get('min_ode_steps', 2)}")
            else:
                # Legacy SFI config (deprecated)
                print("Adaptive Stepping (SFI - Legacy, deprecated):")
                print(f"  Difficulty mode: {stage2_config.get('difficulty_mode', 'multiplicative')}")
                print(f"  Base steps: {stage2_config.get('base_steps', 4)}")
                print(f"  SFI alpha: {stage2_config.get('sfi_alpha', 0.5)}")
                print(f"  Max thickness ref: {stage2_config.get('max_thickness_ref', 6.0)} mm")
                if stage2_config.get('difficulty_mode') == 'additive':
                    print(f"  Additive beta: {stage2_config.get('additive_beta', 1.0)}")

    print("-" * 70)
    print("Training Configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Accumulate grad: {config['training'].get('accumulate_grad_batches', 1)}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_ratio*100:.0f}%)")

    # Calculate effective batch size
    effective_batch = config['training']['batch_size'] * config['training'].get('accumulate_grad_batches', 1) * num_gpus
    print(f"  Effective batch: {config['training']['batch_size']} × {config['training'].get('accumulate_grad_batches', 1)} × {num_gpus} = {effective_batch}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\nTotal parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M")
    print("=" * 70 + "\n")

    # Save config
    # if hasattr(logger, 'log_dir') and logger.log_dir is not None:
    #     config_save_dir = Path(logger.log_dir)
    # elif hasattr(logger, 'save_dir') and logger.save_dir is not None:
    #     config_save_dir = Path(logger.save_dir) / experiment_name
    # else:
    #     config_save_dir = Path(config['wandb']['save_dir']) / experiment_name

    # config_save_dir.mkdir(parents=True, exist_ok=True)
    # config_save_path = config_save_dir / 'config.yaml'
    # with open(config_save_path, 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    # print(f"Config saved to: {config_save_path}")

    # Train (reuse train_loader to avoid re-initialization)
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=args.resume,
    )

    print(f"\nBest checkpoint: {best_checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cascaded 2D SR")
    parser.add_argument("--config", type=str, default="config/cascaded_sr_2d_config.yaml")
    parser.add_argument("--modality", type=str, default="t1",
                        choices=['t1', 't2', 'both', 'flair', 't1ce', 'all'],
                        help="Modality to train on. 'all' uses all modalities (for BraTS21)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage (1=Regression, 2=RF Refiner)")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Path to Stage 1 checkpoint (required for Stage 2)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # CETA ablation arguments (override config values)
    parser.add_argument("--ceta-mode", type=str, default=None,
                        choices=['fixed', 'ratio', 'anchored', 'pad_matched'],
                        help="CETA mode (overrides config)")
    parser.add_argument("--ceta-gap-mm", type=float, default=None,
                        help="CETA gap in mm for 'fixed' mode (overrides config)")
    parser.add_argument("--ceta-ratio", type=float, default=None,
                        help="CETA ratio for 'ratio' mode (overrides config)")
    parser.add_argument("--ceta-anchor", type=float, default=None,
                        help="CETA anchor for 'anchored' mode (overrides config)")
    parser.add_argument("--ceta-delta-pad", type=float, default=None,
                        help="CETA delta PAD for 'pad_matched' mode (overrides config)")
    parser.add_argument("--use-ceta", type=str, default=None, choices=['true', 'false'],
                        help="Enable/disable CETA (overrides config)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max_epochs in config")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Override experiment name for WandB")
    parser.add_argument("--disable-protocol", action="store_true",
                        help="Ablation: disable protocol conditioning for Stage 2 RF")
    parser.add_argument("--no-u-shaped-sampling", action="store_true",
                        help="Ablation: use uniform timestep sampling instead of U-shaped")
    parser.add_argument("--ceta-weight", type=float, default=None,
                        help="Override CETA weight (lambda_ceta)")
    parser.add_argument("--ceta-fixed-t", type=float, default=None,
                        help="Use fixed timestep for CETA loss (ablation). If not set, uses flow loss's random t.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size in config")
    parser.add_argument("--max-train-slices", type=int, default=None,
                        help="Limit number of training slices")
    parser.add_argument("--max-val-slices", type=int, default=None,
                        help="Limit number of validation/test slices")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Debug mode overrides
    if args.debug:
        config['training']['batch_size'] = 2
        config['training']['accumulate_grad_batches'] = 1
        config['data']['slices_per_volume'] = 4
        config['data']['patch_size'] = [64, 64]
        config['model']['model_config'] = 'tiny'
        config['hardware']['devices'] = 1
        config['stage1']['max_epochs'] = 2
        config['stage2']['max_epochs'] = 2
        print("=" * 50)
        print("DEBUG MODE (2D)")
        print("=" * 50)

    # CETA ablation overrides (command-line > config)
    if args.ceta_mode is not None:
        config['stage2']['ceta_mode'] = args.ceta_mode
    if args.ceta_gap_mm is not None:
        config['stage2']['ceta_gap_mm'] = args.ceta_gap_mm
    if args.ceta_ratio is not None:
        config['stage2']['ceta_ratio'] = args.ceta_ratio
    if args.ceta_anchor is not None:
        config['stage2']['ceta_anchor'] = args.ceta_anchor
    if args.ceta_delta_pad is not None:
        config['stage2']['ceta_delta_pad'] = args.ceta_delta_pad
    if args.use_ceta is not None:
        config['stage2']['use_ceta'] = (args.use_ceta.lower() == 'true')
    if args.ceta_weight is not None:
        config['stage2']['ceta_weight'] = args.ceta_weight
    if args.ceta_fixed_t is not None:
        config['stage2']['ceta_fixed_t'] = args.ceta_fixed_t
    if args.max_epochs is not None:
        config['stage2']['max_epochs'] = args.max_epochs
        config['stage1']['max_epochs'] = args.max_epochs
    if args.exp_name is not None:
        config['logging']['experiment_name'] = args.exp_name

    # Batch size override
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Data size overrides
    if args.max_train_slices is not None:
        config['data']['max_train_slices'] = args.max_train_slices
    if args.max_val_slices is not None:
        config['data']['max_val_slices'] = args.max_val_slices

    # Ablation: disable protocol conditioning
    if args.disable_protocol:
        config['stage2']['disable_protocol'] = True
    # Ablation: uniform timestep sampling
    if args.no_u_shaped_sampling:
        config['loss']['use_u_shaped_sampling'] = False

    main(config, args)
