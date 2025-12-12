"""
HSIVI Trainer for Diffusion Model Acceleration
Adapted from https://github.com/longinYu/HSIVI

Implements Hierarchical Semi-Implicit Variational Inference for
accelerating diffusion model sampling.
"""

import os
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("FID metric not available. Install with: pip install torchmetrics[image]")

from .config import HSIVIConfig
from .models.phi_net import PhiNetWithGamma
from .models.f_net import FNet, FNetSimple


def get_beta_schedule(schedule_type: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Get beta schedule for diffusion process."""
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    elif schedule_type == 'cosine':
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    elif schedule_type == 'sigmoid':
        t = torch.linspace(0, timesteps, timesteps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(-3 / 1).sigmoid()
        v_end = torch.tensor(3 / 1).sigmoid()
        alphas_cumprod = (-((t * 6 - 3) / 1).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")


def get_skip_timesteps(n_steps: int, total_timesteps: int, skip_type: str) -> torch.Tensor:
    """Get timesteps for discrete sampling based on skip type."""
    if skip_type == 'uniform':
        skip = total_timesteps // n_steps
        timesteps = torch.arange(0, total_timesteps, skip)
    elif skip_type == 'quad':
        timesteps = torch.linspace(0, np.sqrt(total_timesteps * 0.8), n_steps) ** 2
        timesteps = timesteps.long()
    else:
        raise ValueError(f"Unknown skip type: {skip_type}")
    
    # Ensure we have exactly n_steps and include 0
    timesteps = torch.unique(timesteps)
    if len(timesteps) > n_steps:
        timesteps = timesteps[:n_steps]
    
    return timesteps.long()


class HSIVITrainer:
    """
    Trainer for Hierarchical Semi-Implicit Variational Inference.
    
    This trainer implements the HSIVI algorithm for diffusion model
    acceleration, training phi networks that transform noise through
    hierarchical conditional distributions.
    """
    
    def __init__(
        self,
        config: HSIVIConfig,
        pretrained_epsilon: Optional[nn.Module] = None,
        device: str = 'cuda'
    ):
        """
        Initialize HSIVI Trainer.
        
        Args:
            config: HSIVI configuration
            pretrained_epsilon: Pretrained epsilon (noise) prediction network
            device: Device to use for training
        """
        self.config = config
        self.device = device
        
        # Setup directories
        self.workdir = Path(config.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / 'samples').mkdir(exist_ok=True)
        (self.workdir / 'checkpoints').mkdir(exist_ok=True)
        
        # Save config
        config.save(self.workdir / 'config.json')
        
        # Setup diffusion parameters
        self._setup_diffusion()
        
        # Setup networks
        self._setup_networks(pretrained_epsilon)
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup training state
        self.step = 0
        self.best_loss = float('inf')
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        
        # TensorBoard logging
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = self.workdir / 'tensorboard'
            log_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging to {log_dir}")
    
    def _setup_diffusion(self):
        """Setup diffusion process parameters."""
        config = self.config
        
        # Beta schedule
        betas = get_beta_schedule(
            config.beta_schedule,
            config.diffusion_timesteps,
            config.beta_start,
            config.beta_end
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (convert to float32 for compatibility with mixed precision)
        self.betas = betas.float().to(self.device)
        self.alphas_cumprod = alphas_cumprod.float().to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.float().to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float().to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).float().to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float().to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float().to(self.device)
        
        # Get discrete timesteps
        self.discrete_timesteps = get_skip_timesteps(
            config.n_discrete_steps,
            config.diffusion_timesteps,
            config.skip_type
        ).to(self.device)
        
        print(f"Discrete timesteps: {self.discrete_timesteps.cpu().numpy()}")
    
    def _setup_networks(self, pretrained_epsilon: Optional[nn.Module]):
        """Setup phi and f networks."""
        config = self.config
        
        # Phi network (generator)
        self.phi_net = PhiNetWithGamma(
            image_size=config.image_size,
            channels=config.channels,
            base_dim=config.phi_base_dim,
            dim_mults=config.phi_dim_mults,
            time_emb_dim=config.phi_time_emb_dim,
            num_res_blocks=config.phi_num_res_blocks,
            attn_resolutions=config.phi_attn_resolutions,
            dropout=config.phi_dropout,
            n_discrete_steps=config.n_discrete_steps,
            image_gamma=config.image_gamma,
            independent_log_gamma=config.independent_log_gamma
        ).to(self.device)
        
        # F network (discriminator/critic)
        if config.f_simple:
            self.f_net = FNetSimple(
                image_size=config.image_size,
                channels=config.channels,
                base_dim=config.f_base_dim,
                time_emb_dim=config.f_time_emb_dim,
                use_spectral_norm=config.f_use_spectral_norm
            ).to(self.device)
        else:
            self.f_net = FNet(
                image_size=config.image_size,
                channels=config.channels,
                base_dim=config.f_base_dim,
                dim_mults=config.f_dim_mults,
                time_emb_dim=config.f_time_emb_dim,
                num_res_blocks=config.f_num_res_blocks,
                use_spectral_norm=config.f_use_spectral_norm
            ).to(self.device)
        
        # Pretrained epsilon network (frozen)
        self.epsilon_net = pretrained_epsilon
        if self.epsilon_net is not None:
            self.epsilon_net.to(self.device)
            self.epsilon_net.eval()
            for param in self.epsilon_net.parameters():
                param.requires_grad = False
        
        # Print parameter counts
        phi_params = sum(p.numel() for p in self.phi_net.parameters() if p.requires_grad)
        f_params = sum(p.numel() for p in self.f_net.parameters() if p.requires_grad)
        print(f"Phi network parameters: {phi_params:,}")
        print(f"F network parameters: {f_params:,}")
    
    def _setup_optimizers(self):
        """Setup optimizers for phi and f networks."""
        config = self.config
        
        self.phi_optimizer = AdamW(
            self.phi_net.parameters(),
            lr=config.phi_learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        self.f_optimizer = AdamW(
            self.f_net.parameters(),
            lr=config.f_learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # Store base learning rates for warmup
        self.phi_base_lr = config.phi_learning_rate
        self.f_base_lr = config.f_learning_rate
        self.warmup_steps = config.warmup_steps
    
    def _get_warmup_lr(self, base_lr: float, step: int) -> float:
        """Calculate learning rate with linear warmup."""
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return base_lr
        return base_lr * (step / self.warmup_steps)
    
    def _update_learning_rate(self):
        """Update learning rate based on current step (warmup)."""
        phi_lr = self._get_warmup_lr(self.phi_base_lr, self.step)
        f_lr = self._get_warmup_lr(self.f_base_lr, self.step)
        
        for param_group in self.phi_optimizer.param_groups:
            param_group['lr'] = phi_lr
        
        for param_group in self.f_optimizer.param_groups:
            param_group['lr'] = f_lr
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) - forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from epsilon prediction."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * eps
    
    def get_target_score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get target score function from pretrained epsilon network.
        
        If no pretrained network, returns standard score estimate.
        """
        if self.epsilon_net is not None:
            with torch.no_grad():
                # Get epsilon prediction
                t_float = t.float() / self.config.diffusion_timesteps
                eps = self.epsilon_net(x_t, t_float)
                # Score = -epsilon / sqrt(1 - alpha_cumprod)
                sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                score = -eps / sqrt_one_minus_alpha
                return score
        else:
            # Without pretrained model, we can't compute exact score
            # This mode requires data-driven training
            return None
    
    def compute_phi_loss(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for phi network update.
        
        Implements the HSIVI objective: minimize the KL divergence
        between the learned conditional and the target.
        """
        # Ensure float32 for mixed precision compatibility
        x_t = x_t.float()
        
        batch_size = x_t.shape[0]
        t = self.discrete_timesteps[t_idx].expand(batch_size)
        t_prev = self.discrete_timesteps[t_idx - 1].expand(batch_size) if t_idx > 0 else torch.zeros_like(t)
        
        # Generate sample from phi
        x_tm1_mean = self.phi_net(x_t.float(), t)
        log_gamma = self.phi_net.get_log_gamma(layer_idx if self.config.independent_log_gamma else None)
        std = torch.exp(0.5 * log_gamma)
        
        # Sample with reparameterization
        noise = torch.randn_like(x_tm1_mean)
        x_tm1 = x_tm1_mean + std * noise
        
        # F network output
        f_output = self.f_net(x_t.float(), x_tm1.float(), t)
        
        # HSIVI loss = E[f(x_t, x_{t-1})] + variance penalty
        # We want to maximize f for generated samples (minimize negative)
        loss_f = -f_output.mean()
        
        # Variance regularization (entropy term)
        # log_gamma contributes to entropy: H = 0.5 * log(2 * pi * e * var)
        entropy_term = 0.5 * log_gamma.mean()
        
        # Total loss
        loss = loss_f - 0.01 * entropy_term  # Small weight on entropy
        
        metrics = {
            'phi_loss': loss.item(),
            'f_output': f_output.mean().item(),
            'log_gamma': log_gamma.mean().item(),
            'std': std.mean().item()
        }
        
        return loss, metrics
    
    def compute_f_loss(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        layer_idx: int,
        real_x_tm1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for f network update.
        
        The f network is trained to distinguish between samples from
        the target distribution and samples from the phi network.
        """
        # Ensure float32 for mixed precision compatibility
        x_t = x_t.float()
        
        batch_size = x_t.shape[0]
        t = self.discrete_timesteps[t_idx].expand(batch_size)
        t_prev = self.discrete_timesteps[t_idx - 1].expand(batch_size) if t_idx > 0 else torch.zeros_like(t)
        
        # Generate fake sample from phi (detached)
        with torch.no_grad():
            x_tm1_mean = self.phi_net(x_t.float(), t)
            log_gamma = self.phi_net.get_log_gamma(layer_idx if self.config.independent_log_gamma else None)
            std = torch.exp(0.5 * log_gamma)
            noise = torch.randn_like(x_tm1_mean)
            fake_x_tm1 = x_tm1_mean + std * noise
        
        # F network on fake samples
        f_fake = self.f_net(x_t.float(), fake_x_tm1.float(), t)
        
        # Generate "real" sample from target distribution
        # If we have access to score function (pretrained model), use it
        if real_x_tm1 is not None:
            f_real = self.f_net(x_t.float(), real_x_tm1.float(), t)
        else:
            # Use DDPM posterior as target
            # This is an approximation when we don't have real samples
            with torch.no_grad():
                if self.epsilon_net is not None:
                    t_float = t.float()
                    eps = self.epsilon_net(x_t.float(), t_float)
                    x0_pred = self.predict_x0_from_eps(x_t, t, eps)
                    
                    # DDPM posterior mean
                    alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                    alpha_tm1 = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
                    beta_t = self.betas[t].view(-1, 1, 1, 1)
                    
                    posterior_mean = (
                        torch.sqrt(alpha_tm1) * beta_t / (1 - alpha_t) * x0_pred +
                        torch.sqrt(1 - beta_t) * (1 - alpha_tm1) / (1 - alpha_t) * x_t
                    )
                    posterior_var = beta_t * (1 - alpha_tm1) / (1 - alpha_t)
                    
                    real_x_tm1 = posterior_mean + torch.sqrt(posterior_var) * torch.randn_like(x_t)
                else:
                    # Without pretrained model, use noisy version of x_t
                    real_x_tm1 = x_t + 0.1 * torch.randn_like(x_t)
            
            f_real = self.f_net(x_t.float(), real_x_tm1.float(), t)
        
        # Wasserstein loss (critic)
        loss = f_fake.mean() - f_real.mean()
        
        # Gradient penalty for stability
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device, dtype=torch.float32)
        interpolated = (alpha * real_x_tm1 + (1 - alpha) * fake_x_tm1).float()
        interpolated.requires_grad_(True)
        
        f_interp = self.f_net(x_t.float(), interpolated.float(), t)
        grad = torch.autograd.grad(
            outputs=f_interp.sum(),
            inputs=interpolated,
            create_graph=True
        )[0]
        grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
        gp_loss = ((grad_norm - 1) ** 2).mean()
        
        loss = loss + 10.0 * gp_loss
        
        metrics = {
            'f_loss': loss.item(),
            'f_fake': f_fake.mean().item(),
            'f_real': f_real.mean().item(),
            'gp': gp_loss.item()
        }
        
        return loss, metrics
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of training images
        
        Returns:
            Dictionary of metrics
        """
        config = self.config
        batch = batch.to(self.device).float()  # Ensure float32
        batch_size = batch.shape[0]
        
        # Normalize to [-1, 1]
        if batch.max() > 1.0:
            batch = batch * 2 - 1
        
        all_metrics = {}
        
        # Sample random layer/timestep
        n_layers = config.n_discrete_steps - 1
        layer_idx = torch.randint(0, n_layers, (1,)).item()
        t_idx = n_layers - layer_idx  # Reverse order (T -> 0)
        
        # Get x_t by adding noise to batch
        t = self.discrete_timesteps[t_idx].expand(batch_size)
        noise = torch.randn_like(batch)
        x_t = self.q_sample(batch, t, noise)
        
        # Update F network multiple times
        for _ in range(config.f_learning_times):
            self.f_optimizer.zero_grad()
            
            if config.mixed_precision:
                with autocast('cuda'):
                    f_loss, f_metrics = self.compute_f_loss(x_t, t_idx, layer_idx)
                self.scaler.scale(f_loss).backward()
                self.scaler.unscale_(self.f_optimizer)
                torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), config.grad_clip)
                self.scaler.step(self.f_optimizer)
                self.scaler.update()
            else:
                f_loss, f_metrics = self.compute_f_loss(x_t, t_idx, layer_idx)
                f_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), config.grad_clip)
                self.f_optimizer.step()
        
        all_metrics.update(f_metrics)
        
        # Update Phi network
        self.phi_optimizer.zero_grad()
        
        if config.mixed_precision:
            with autocast('cuda'):
                phi_loss, phi_metrics = self.compute_phi_loss(x_t, t_idx, layer_idx)
            self.scaler.scale(phi_loss).backward()
            self.scaler.unscale_(self.phi_optimizer)
            torch.nn.utils.clip_grad_norm_(self.phi_net.parameters(), config.grad_clip)
            self.scaler.step(self.phi_optimizer)
            self.scaler.update()
        else:
            phi_loss, phi_metrics = self.compute_phi_loss(x_t, t_idx, layer_idx)
            phi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.phi_net.parameters(), config.grad_clip)
            self.phi_optimizer.step()
        
        all_metrics.update(phi_metrics)
        all_metrics['layer_idx'] = layer_idx
        all_metrics['t_idx'] = t_idx
        
        return all_metrics
    
    @torch.no_grad()
    def sample(self, batch_size: int = 64, return_trajectory: bool = False) -> torch.Tensor:
        """
        Generate samples using trained phi network.
        
        Args:
            batch_size: Number of samples to generate
            return_trajectory: Whether to return full sampling trajectory
        
        Returns:
            Generated samples
        """
        self.phi_net.eval()
        
        # Start from pure noise (ensure float32)
        x = torch.randn(batch_size, self.config.channels, 
                       self.config.image_size, self.config.image_size,
                       device=self.device, dtype=torch.float32)
        
        trajectory = [x] if return_trajectory else None
        
        # Iterate through discrete timesteps in reverse
        n_steps = len(self.discrete_timesteps)
        for i in range(n_steps - 1, 0, -1):
            t = self.discrete_timesteps[i].expand(batch_size)
            layer_idx = n_steps - 1 - i
            
            # Get phi prediction
            x_mean = self.phi_net(x.float(), t)
            
            if i > 1:  # Add noise except for last step
                log_gamma = self.phi_net.get_log_gamma(
                    layer_idx if self.config.independent_log_gamma else None
                )
                std = torch.exp(0.5 * log_gamma)
                x = x_mean + std * torch.randn_like(x)
            else:
                x = x_mean
            
            if return_trajectory:
                trajectory.append(x)
        
        self.phi_net.train()
        
        # Unnormalize to [0, 1]
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        
        if return_trajectory:
            trajectory = [(t + 1) / 2 for t in trajectory]
            return x, trajectory
        
        return x
    
    @torch.no_grad()
    def compute_fid(self, dataloader: DataLoader, num_samples: Optional[int] = None) -> float:
        """
        Compute FID score between generated samples and real data.
        
        Args:
            dataloader: DataLoader for real images
            num_samples: Number of samples to use (default: config.fid_num_samples)
        
        Returns:
            FID score
        """
        if not FID_AVAILABLE:
            print("FID metric not available. Skipping FID calculation.")
            return float('nan')
        
        num_samples = num_samples or self.config.fid_num_samples
        batch_size = self.config.sampling_batch_size
        
        print(f"Computing FID with {num_samples} samples...")
        
        # Initialize FID metric
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        
        # Process real images
        num_real_processed = 0
        for batch in tqdm(dataloader, desc="Processing real images", leave=False):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(self.device)
            
            # Convert to uint8 [0, 255]
            if batch.max() <= 1.0:
                batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8)
            else:
                batch_uint8 = batch.clamp(0, 255).to(torch.uint8)
            
            fid.update(batch_uint8, real=True)
            num_real_processed += batch.shape[0]
            
            if num_real_processed >= num_samples:
                break
        
        # Generate and process fake images
        num_fake_processed = 0
        num_batches = math.ceil(num_samples / batch_size)
        
        for _ in tqdm(range(num_batches), desc="Generating samples for FID", leave=False):
            curr_batch_size = min(batch_size, num_samples - num_fake_processed)
            samples = self.sample(batch_size=curr_batch_size)
            
            # Convert to uint8 [0, 255]
            samples_uint8 = (samples * 255).clamp(0, 255).to(torch.uint8)
            fid.update(samples_uint8, real=False)
            
            num_fake_processed += curr_batch_size
        
        # Compute FID
        fid_score = fid.compute().item()
        
        print(f"FID Score: {fid_score:.2f}")
        return fid_score
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'phi_net': self.phi_net.state_dict(),
            'f_net': self.f_net.state_dict(),
            'phi_optimizer': self.phi_optimizer.state_dict(),
            'f_optimizer': self.f_optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        path = self.workdir / 'checkpoints' / f'{name}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint['step']
        self.phi_net.load_state_dict(checkpoint['phi_net'])
        self.f_net.load_state_dict(checkpoint['f_net'])
        self.phi_optimizer.load_state_dict(checkpoint['phi_optimizer'])
        self.f_optimizer.load_state_dict(checkpoint['f_optimizer'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"Loaded checkpoint from {path} at step {self.step}")
    
    def train(self, dataloader: DataLoader, resume_path: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader for training data
            resume_path: Path to checkpoint to resume from
        """
        config = self.config
        
        if resume_path is not None:
            self.load_checkpoint(resume_path)
        
        # Training loop
        pbar = tqdm(total=config.n_train_iters, initial=self.step, desc='Training HSIVI')
        data_iter = iter(dataloader)
        
        running_metrics = {}
        
        while self.step < config.n_train_iters:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            # Training step
            metrics = self.train_step(batch)
            
            # Update running metrics
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = []
                running_metrics[k].append(v)
            
            self.step += 1
            
            # Update learning rate (warmup)
            self._update_learning_rate()
            
            # Logging
            if self.step % config.log_every == 0:
                avg_metrics = {k: np.mean(v) for k, v in running_metrics.items()}
                log_str = f"Step {self.step}: " + ", ".join([
                    f"{k}={v:.4f}" for k, v in avg_metrics.items()
                    if not k.endswith('_idx')
                ])
                pbar.set_description(log_str)
                
                # TensorBoard logging
                if self.writer is not None:
                    for k, v in avg_metrics.items():
                        if not k.endswith('_idx'):
                            self.writer.add_scalar(f'train/{k}', v, self.step)
                    
                    # Log learning rates
                    self.writer.add_scalar('lr/phi', self.phi_optimizer.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('lr/f', self.f_optimizer.param_groups[0]['lr'], self.step)
                
                running_metrics = {}
            
            # Sampling
            if self.step % config.sample_every == 0:
                samples = self.sample(config.num_samples)
                nrow = int(math.sqrt(config.num_samples))
                save_image(
                    samples,
                    self.workdir / 'samples' / f'step_{self.step:06d}.png',
                    nrow=nrow
                )
                
                # Log samples to TensorBoard
                if self.writer is not None:
                    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(0, 1))
                    self.writer.add_image('samples/generated', grid, self.step)
            
            # FID calculation
            if config.fid_every > 0 and self.step % config.fid_every == 0:
                fid_score = self.compute_fid(dataloader)
                
                # Log FID to TensorBoard
                if self.writer is not None and not math.isnan(fid_score):
                    self.writer.add_scalar('eval/fid', fid_score, self.step)
                
                # Track best FID and save best model
                if not math.isnan(fid_score) and fid_score < self.best_loss:
                    self.best_loss = fid_score
                    self.save_checkpoint('best')
                    print(f"New best FID: {fid_score:.2f}")
            
            # Saving
            if self.step % config.save_every == 0:
                self.save_checkpoint('latest')
                self.save_checkpoint(f'step_{self.step:06d}')
            
            pbar.update(1)
        
        # Final save
        self.save_checkpoint('final')
        pbar.close()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        print("Training complete!")
    
    def log_model_info(self):
        """Log model architecture and parameter histograms to TensorBoard."""
        if self.writer is None:
            return
        
        # Log parameter histograms
        for name, param in self.phi_net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'phi_net/{name}', param.data, self.step)
        
        for name, param in self.f_net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'f_net/{name}', param.data, self.step)
    
    def close(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

