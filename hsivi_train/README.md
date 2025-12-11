# HSIVI Training for Colored MNIST

This folder contains the implementation of **Hierarchical Semi-Implicit Variational Inference (HSIVI)** for diffusion model acceleration, adapted from the [original HSIVI repository](https://github.com/longinYu/HSIVI).

> **Paper**: Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration (NeurIPS 2023)

## Overview

HSIVI accelerates diffusion model sampling by learning a hierarchical semi-implicit variational distribution that approximates the reverse diffusion process. Instead of requiring 1000 steps like standard DDPM, HSIVI can generate samples in as few as 5-10 steps.

## Files Structure

```
hsivi_train/
├── __init__.py          # Package exports
├── config.py            # Configuration classes
├── hsivi_trainer.py     # Main trainer class
├── train.py             # Training entry point
├── sample.py            # Sampling utility
├── run_colored_mnist.sh # Full training script
├── run_fast_test.sh     # Quick test script
└── models/
    ├── __init__.py
    ├── phi_net.py       # Phi network (generator)
    └── f_net.py         # F network (discriminator)
```

## Quick Start

### 1. Fast Test (verify setup)

Run a quick test to make sure everything works:

```bash
cd /home/okashurin/okashurin/diffusion-acceleration
bash hsivi_train/run_fast_test.sh
```

Or with Python directly:

```bash
python -m hsivi_train.train --config fast --n_train_iters 1000
```

### 2. Full Training

Train the full HSIVI model:

```bash
bash hsivi_train/run_colored_mnist.sh
```

Or customize training parameters:

```bash
python -m hsivi_train.train \
    --config default \
    --data_dir ./data \
    --workdir ./work_dir/hsivi_experiment \
    --pretrained_model ./ckpts/model-5.pt \
    --n_discrete_steps 11 \
    --n_train_iters 100000
```

### 3. Generate Samples

After training, generate samples:

```bash
python -m hsivi_train.sample \
    --checkpoint ./work_dir/hsivi_colored_mnist/checkpoints/latest.pt \
    --num_samples 64 \
    --output_dir ./hsivi_samples
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_discrete_steps` | 11 | Number of discrete timesteps (NFE + 1) |
| `n_train_iters` | 100000 | Training iterations |
| `training_batch_size` | 64 | Batch size |
| `phi_learning_rate` | 1.6e-5 | Phi network learning rate |
| `f_learning_rate` | 8e-5 | F network learning rate |
| `f_learning_times` | 20 | F updates per phi update |
| `skip_type` | "quad" | Timestep selection strategy |
| `image_gamma` | True | Use pixel-wise variance |

## Configuration Presets

- **default**: Standard configuration for colored MNIST
- **fast**: Quick testing with reduced model and iterations
- **high_quality**: Larger model for best quality

```python
from hsivi_train.config import get_default_config, get_fast_config, get_high_quality_config

config = get_fast_config()  # For testing
config = get_default_config()  # For normal training
config = get_high_quality_config()  # For best quality
```

## Programmatic Usage

```python
import torch
from torch.utils.data import DataLoader
from hsivi_train import HSIVITrainer, HSIVIConfig
from utils.dataset_h5 import H5ImagesDataset

# Load config
config = HSIVIConfig(
    n_discrete_steps=11,
    n_train_iters=100000,
    workdir='./work_dir/my_experiment'
)

# Load pretrained diffusion model (optional but recommended)
from diffusion.ddpm import Unet, GaussianDiffusion
model = Unet(dim=64, dim_mults=(1,2,4))
diffusion = GaussianDiffusion(model, image_size=28, timesteps=1000)
checkpoint = torch.load('./ckpts/model-5.pt')
diffusion.load_state_dict(checkpoint['model'])
pretrained_epsilon = diffusion.model

# Create trainer
trainer = HSIVITrainer(
    config=config,
    pretrained_epsilon=pretrained_epsilon,
    device='cuda'
)

# Train
dataset = H5ImagesDataset('./data')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
trainer.train(dataloader)

# Sample
samples = trainer.sample(batch_size=16)
```

## Expected Results

With default settings (NFE=10):
- Training time: ~4-6 hours on single GPU
- Sample quality: FID comparable to DDIM with 10 steps
- Speedup: ~100x faster than full DDPM

## References

```bibtex
@inproceedings{
yu2023hierarchical,
title={Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration},
author={Longlin Yu and Tianyu Xie and Yu Zhu and Tong Yang and Xiangyu Zhang and Cheng Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=ghIBaprxsV}
}
```

