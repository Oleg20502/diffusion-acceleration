#!/usr/bin/env python
"""
Main training script for HSIVI on Colored MNIST
Adapted from https://github.com/longinYu/HSIVI

Usage:
    python -m hsivi_train.train --config default
    python -m hsivi_train.train --config fast --n_train_iters 5000
    python -m hsivi_train.train --config custom --config_path ./my_config.json
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hsivi_train.config import HSIVIConfig, get_default_config, get_fast_config, get_high_quality_config
from hsivi_train.hsivi_trainer import HSIVITrainer
from utils.dataset_h5 import H5ImagesDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train HSIVI model on Colored MNIST')
    
    # Config options
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast', 'high_quality', 'custom'],
                       help='Configuration preset to use')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to custom config JSON file')
    
    # Override options
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--workdir', type=str, default=None,
                       help='Override working directory')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained diffusion model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Training overrides
    parser.add_argument('--n_train_iters', type=int, default=None,
                       help='Override number of training iterations')
    parser.add_argument('--training_batch_size', type=int, default=None,
                       help='Override training batch size')
    parser.add_argument('--n_discrete_steps', type=int, default=None,
                       help='Override number of discrete steps (NFE + 1)')
    parser.add_argument('--phi_learning_rate', type=float, default=None,
                       help='Override phi learning rate')
    parser.add_argument('--f_learning_rate', type=float, default=None,
                       help='Override f learning rate')
    parser.add_argument('--f_learning_times', type=int, default=None,
                       help='Override f network updates per phi update')
    parser.add_argument('--fid_every', type=int, default=None,
                       help='Calculate FID every N steps (0 to disable)')
    parser.add_argument('--fid_num_samples', type=int, default=None,
                       help='Number of samples for FID calculation')
    
    # Model options
    parser.add_argument('--phi_base_dim', type=int, default=None,
                       help='Override phi network base dimension')
    parser.add_argument('--f_base_dim', type=int, default=None,
                       help='Override f network base dimension')
    parser.add_argument('--independent_log_gamma', action='store_true',
                       help='Use independent log_gamma per layer')
    parser.add_argument('--image_gamma', action='store_true',
                       help='Use non-isotropic (pixel-wise) gamma')
    parser.add_argument('--skip_type', type=str, choices=['uniform', 'quad'],
                       default=None, help='Override skip type')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision training')
    
    return parser.parse_args()


def load_pretrained_epsilon(checkpoint_path: str, config: HSIVIConfig, device: str):
    """Load pretrained epsilon (noise prediction) network."""
    from diffusion.ddpm import Unet, GaussianDiffusion
    
    print(f"Loading pretrained model from {checkpoint_path}")
    
    # Create model with same architecture as training
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4),
        channels=config.channels,
        flash_attn=False
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size=config.image_size,
        timesteps=config.diffusion_timesteps,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if 'model' in checkpoint:
        diffusion.load_state_dict(checkpoint['model'])
    elif 'ema' in checkpoint:
        # Try loading from EMA
        from ema_pytorch import EMA
        ema = EMA(diffusion, beta=0.995, update_every=10)
        ema.load_state_dict(checkpoint['ema'])
        diffusion = ema.ema_model
    else:
        diffusion.load_state_dict(checkpoint)
    
    # Return just the UNet model (epsilon predictor)
    return diffusion.model


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    if args.config == 'custom' and args.config_path:
        config = HSIVIConfig.load(args.config_path)
    elif args.config == 'fast':
        config = get_fast_config()
    elif args.config == 'high_quality':
        config = get_high_quality_config()
    else:
        config = get_default_config()
    
    # Apply overrides
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.workdir:
        config.workdir = args.workdir
    if args.pretrained_model:
        config.pretrained_model = args.pretrained_model
    if args.n_train_iters:
        config.n_train_iters = args.n_train_iters
    if args.training_batch_size:
        config.training_batch_size = args.training_batch_size
    if args.n_discrete_steps:
        config.n_discrete_steps = args.n_discrete_steps
    if args.phi_learning_rate:
        config.phi_learning_rate = args.phi_learning_rate
    if args.f_learning_rate:
        config.f_learning_rate = args.f_learning_rate
    if args.f_learning_times:
        config.f_learning_times = args.f_learning_times
    if args.fid_every is not None:
        config.fid_every = args.fid_every
    if args.fid_num_samples:
        config.fid_num_samples = args.fid_num_samples
    if args.phi_base_dim:
        config.phi_base_dim = args.phi_base_dim
    if args.f_base_dim:
        config.f_base_dim = args.f_base_dim
    if args.independent_log_gamma:
        config.independent_log_gamma = True
    if args.image_gamma:
        config.image_gamma = True
    if args.skip_type:
        config.skip_type = args.skip_type
    if args.seed:
        config.seed = args.seed
    if args.no_mixed_precision:
        config.mixed_precision = False
    
    print("=" * 60)
    print("HSIVI Training Configuration")
    print("=" * 60)
    print(f"Data directory: {config.data_dir}")
    print(f"Working directory: {config.workdir}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Channels: {config.channels}")
    print(f"Discrete steps (NFE+1): {config.n_discrete_steps}")
    print(f"Training iterations: {config.n_train_iters}")
    print(f"Batch size: {config.training_batch_size}")
    print(f"Phi LR: {config.phi_learning_rate}")
    print(f"F LR: {config.f_learning_rate}")
    print(f"F updates per phi: {config.f_learning_times}")
    print(f"Skip type: {config.skip_type}")
    print(f"FID every: {config.fid_every} steps" if config.fid_every > 0 else "FID: disabled")
    print(f"FID samples: {config.fid_num_samples}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load pretrained model if specified
    pretrained_epsilon = None
    if config.pretrained_model and os.path.exists(config.pretrained_model):
        pretrained_epsilon = load_pretrained_epsilon(
            config.pretrained_model, config, device
        )
    
    # Create trainer
    trainer = HSIVITrainer(
        config=config,
        pretrained_epsilon=pretrained_epsilon,
        device=device
    )
    
    # Create dataloader
    print(f"Loading dataset from {config.data_dir}")
    dataset = H5ImagesDataset(config.data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True
    )
    
    # Train
    trainer.train(dataloader, resume_path=args.resume)


if __name__ == '__main__':
    main()

