#!/usr/bin/env python
"""
Sampling script for trained HSIVI model
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from hsivi_train.config import HSIVIConfig
from hsivi_train.hsivi_trainer import HSIVITrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained HSIVI model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON (defaults to checkpoint dir)')
    parser.add_argument('--output_dir', type=str, default='./hsivi_samples',
                       help='Output directory for samples')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--save_trajectory', action='store_true',
                       help='Save intermediate sampling trajectory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load config
    checkpoint_dir = Path(args.checkpoint).parent.parent
    if args.config:
        config_path = args.config
    else:
        config_path = checkpoint_dir / 'config.json'
    
    if os.path.exists(config_path):
        config = HSIVIConfig.load(config_path)
        print(f"Loaded config from {config_path}")
    else:
        print("Config not found, using defaults")
        config = HSIVIConfig()
    
    # Create trainer (without pretrained epsilon - not needed for sampling)
    trainer = HSIVITrainer(
        config=config,
        pretrained_epsilon=None,
        device=device
    )
    
    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    all_samples = []
    num_batches = math.ceil(args.num_samples / args.batch_size)
    
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - len(all_samples))
        
        if args.save_trajectory:
            samples, trajectory = trainer.sample(
                batch_size=batch_size,
                return_trajectory=True
            )
            
            # Save trajectory
            traj_dir = output_dir / 'trajectory'
            traj_dir.mkdir(exist_ok=True)
            for t_idx, t_samples in enumerate(trajectory):
                save_image(
                    t_samples,
                    traj_dir / f'batch_{i}_step_{t_idx}.png',
                    nrow=int(math.sqrt(batch_size))
                )
        else:
            samples = trainer.sample(batch_size=batch_size)
        
        all_samples.append(samples)
        print(f"Generated batch {i+1}/{num_batches}")
    
    # Concatenate and save
    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    
    # Save grid
    nrow = int(math.sqrt(args.num_samples))
    save_image(
        all_samples,
        output_dir / 'samples_grid.png',
        nrow=nrow
    )
    
    # Save individual samples
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(all_samples):
        save_image(sample, individual_dir / f'sample_{i:04d}.png')
    
    print(f"Saved {args.num_samples} samples to {output_dir}")
    print(f"Grid saved to {output_dir / 'samples_grid.png'}")


if __name__ == '__main__':
    main()

