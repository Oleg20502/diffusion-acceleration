"""
F Network (Discriminator) for HSIVI
Adapted from https://github.com/longinYu/HSIVI

The f network acts as a discriminator/critic in the variational inference
framework, helping to tighten the variational bound.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Sinusoidal timestep embedding."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


class SpectralNorm(nn.Module):
    """Spectral normalization wrapper."""
    
    def __init__(self, module, name='weight', n_power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self._make_params()
    
    def _make_params(self):
        weight = getattr(self.module, self.name)
        height = weight.data.shape[0]
        width = weight.view(height, -1).shape[1]
        
        u = weight.new_empty(height).normal_(0, 1)
        v = weight.new_empty(width).normal_(0, 1)
        u = F.normalize(u, dim=0)
        v = F.normalize(v, dim=0)
        
        self.module.register_buffer(self.name + '_u', u)
        self.module.register_buffer(self.name + '_v', v)
    
    def _update_vectors(self):
        weight = getattr(self.module, self.name)
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        
        height = weight.shape[0]
        weight_mat = weight.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0)
                u = F.normalize(torch.mv(weight_mat, v), dim=0)
            
            setattr(self.module, self.name + '_u', u)
            setattr(self.module, self.name + '_v', v)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return sigma
    
    def forward(self, *args, **kwargs):
        if self.training:
            sigma = self._update_vectors()
            weight = getattr(self.module, self.name)
            setattr(self.module, self.name, weight / sigma)
        
        return self.module(*args, **kwargs)


def spectral_norm(module, name='weight'):
    """Apply spectral normalization to a module."""
    return nn.utils.spectral_norm(module, name)


class ResBlockF(nn.Module):
    """Residual block for F network with spectral normalization."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, use_spectral_norm=True):
        super().__init__()
        
        norm_fn = spectral_norm if use_spectral_norm else (lambda x: x)
        
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = norm_fn(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            norm_fn(nn.Linear(time_emb_dim, out_channels * 2))
        )
        
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = norm_fn(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        if in_channels != out_channels:
            self.skip_conv = norm_fn(nn.Conv2d(in_channels, out_channels, 1))
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t_emb = self.time_mlp(t_emb)
        t_emb = rearrange(t_emb, 'b c -> b c 1 1')
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels, use_spectral_norm=True):
        super().__init__()
        norm_fn = spectral_norm if use_spectral_norm else (lambda x: x)
        self.conv = norm_fn(nn.Conv2d(channels, channels, 3, stride=2, padding=1))
    
    def forward(self, x):
        return self.conv(x)


class FNet(nn.Module):
    """
    F Network (Discriminator/Critic) for HSIVI.
    
    This network estimates the density ratio or acts as a critic
    in the semi-implicit variational inference framework.
    
    Args:
        image_size: Size of the image (assumes square images)
        channels: Number of image channels (3 for RGB)
        base_dim: Base channel dimension
        dim_mults: Channel multipliers for each resolution
        time_emb_dim: Dimension of time embedding
        num_res_blocks: Number of residual blocks per resolution
        use_spectral_norm: Whether to use spectral normalization
        output_dim: Output dimension (1 for scalar output)
    """
    
    def __init__(
        self,
        image_size=28,
        channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4),
        time_emb_dim=256,
        num_res_blocks=2,
        use_spectral_norm=True,
        output_dim=1
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.base_dim = base_dim
        
        norm_fn = spectral_norm if use_spectral_norm else (lambda x: x)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            norm_fn(nn.Linear(base_dim, time_emb_dim)),
            nn.SiLU(),
            norm_fn(nn.Linear(time_emb_dim, time_emb_dim))
        )
        
        # Input projection (takes concatenated x_t and x_{t-1})
        self.input_conv = norm_fn(nn.Conv2d(channels * 2, base_dim, 3, padding=1))
        
        # Build encoder
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        curr_res = image_size
        for i in range(len(dim_mults)):
            in_ch = dims[i]
            out_ch = dims[i + 1]
            
            for j in range(num_res_blocks):
                self.blocks.append(
                    ResBlockF(in_ch if j == 0 else out_ch, out_ch, time_emb_dim, use_spectral_norm)
                )
            
            if i < len(dim_mults) - 1:
                self.downsamples.append(Downsample(out_ch, use_spectral_norm))
                curr_res //= 2
            else:
                self.downsamples.append(nn.Identity())
        
        # Final layers
        final_dim = dims[-1]
        final_res = curr_res
        
        self.final_norm = nn.GroupNorm(min(32, final_dim), final_dim)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.final_linear = norm_fn(nn.Linear(final_dim, output_dim))
    
    def forward(self, x_t, x_tm1, t):
        """
        Forward pass.
        
        Args:
            x_t: Sample at time t [B, C, H, W]
            x_tm1: Sample at time t-1 [B, C, H, W]
            t: Timestep tensor [B]
        
        Returns:
            Scalar output [B, 1]
        """
        # Concatenate inputs
        x = torch.cat([x_t, x_tm1], dim=1)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, self.base_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder blocks
        block_idx = 0
        for i, downsample in enumerate(self.downsamples):
            for j in range(2):  # num_res_blocks = 2
                h = self.blocks[block_idx](h, t_emb)
                block_idx += 1
            h = downsample(h)
        
        # Final
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_pool(h)
        h = h.view(h.shape[0], -1)
        h = self.final_linear(h)
        
        return h


class FNetSimple(nn.Module):
    """
    Simplified F Network for smaller images like MNIST.
    
    Uses a simpler architecture suitable for 28x28 images.
    """
    
    def __init__(
        self,
        image_size=28,
        channels=3,
        base_dim=64,
        time_emb_dim=128,
        use_spectral_norm=True
    ):
        super().__init__()
        
        self.base_dim = base_dim  # Store for use in forward
        norm_fn = spectral_norm if use_spectral_norm else (lambda x: x)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Convolutional layers
        self.conv1 = norm_fn(nn.Conv2d(channels * 2, base_dim, 4, 2, 1))  # 14x14
        self.conv2 = norm_fn(nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1))  # 7x7
        self.conv3 = norm_fn(nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1))  # 3x3
        
        # Time projection
        self.time_proj = norm_fn(nn.Linear(time_emb_dim, base_dim * 4))
        
        # Final layers
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            norm_fn(nn.Linear(base_dim * 4, base_dim * 2)),
            nn.SiLU(),
            norm_fn(nn.Linear(base_dim * 2, 1))
        )
    
    def forward(self, x_t, x_tm1, t):
        """
        Forward pass.
        
        Args:
            x_t: Sample at time t [B, C, H, W]
            x_tm1: Sample at time t-1 [B, C, H, W]
            t: Timestep tensor [B]
        
        Returns:
            Scalar output [B, 1]
        """
        # Concatenate inputs
        x = torch.cat([x_t, x_tm1], dim=1)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, self.base_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Convolutions
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        
        # Add time embedding
        t_proj = self.time_proj(t_emb)
        h = h + t_proj.view(-1, h.shape[1], 1, 1)
        
        # Final output
        return self.final(h)

