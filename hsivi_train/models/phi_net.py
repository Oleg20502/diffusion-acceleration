"""
Phi Network for HSIVI
Adapted from https://github.com/longinYu/HSIVI

The phi network generates samples in a hierarchical manner,
transforming noise through a series of conditional distributions.
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


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
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
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (heads c) h w -> b heads (h w) c', heads=self.num_heads)
        k = rearrange(k, 'b (heads c) h w -> b heads (h w) c', heads=self.num_heads)
        v = rearrange(v, 'b (heads c) h w -> b heads (h w) c', heads=self.num_heads)
        
        attn = torch.einsum('bhic,bhjc->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjc->bhic', attn, v)
        
        out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', h=h, w=w)
        return x + self.proj(out)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class PhiNet(nn.Module):
    """
    Phi Network for HSIVI.
    
    This network transforms noise samples hierarchically through
    conditional distributions parameterized by neural networks.
    
    Args:
        image_size: Size of the image (assumes square images)
        channels: Number of image channels (3 for RGB)
        base_dim: Base channel dimension
        dim_mults: Channel multipliers for each resolution
        time_emb_dim: Dimension of time embedding
        num_res_blocks: Number of residual blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        image_size=28,
        channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4),
        time_emb_dim=256,
        num_res_blocks=2,
        attn_resolutions=(14,),
        dropout=0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.base_dim = base_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(channels, base_dim, 3, padding=1)
        
        # Build encoder
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        curr_res = image_size
        for i in range(len(dim_mults)):
            in_ch = dims[i]
            out_ch = dims[i + 1]
            
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(in_ch if j == 0 else out_ch, out_ch, time_emb_dim, dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            self.down_blocks.append(blocks)
            
            if i < len(dim_mults) - 1:
                self.down_samples.append(Downsample(out_ch))
                curr_res //= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle block
        mid_ch = dims[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        
        # Build decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i in reversed(range(len(dim_mults))):
            out_ch = dims[i]
            in_ch = dims[i + 1]
            
            if i < len(dim_mults) - 1:
                self.up_samples.append(Upsample(in_ch))
            else:
                self.up_samples.append(nn.Identity())
            
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                skip_ch = dims[i + 1] if j == 0 else out_ch
                blocks.append(ResBlock(in_ch + skip_ch if j == 0 else out_ch, out_ch, time_emb_dim, dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            self.up_blocks.append(blocks)
            curr_res *= 2
        
        # Output projection
        self.out_norm = nn.GroupNorm(min(32, base_dim), base_dim)
        self.out_conv = nn.Conv2d(base_dim, channels, 3, padding=1)
        
        # Log gamma for variance (learnable per layer or shared)
        self.log_gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, t, noise=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (noisy image) [B, C, H, W]
            t: Timestep tensor [B]
            noise: Optional noise input for stochastic layers
        
        Returns:
            Output tensor [B, C, H, W]
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.base_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.input_conv(x)
        hs = [h]
        
        # Encoder
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            hs.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            h = upsample(h)
            h = torch.cat([h, hs.pop()], dim=1)
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
    
    def get_log_gamma(self):
        """Get log gamma value for variance."""
        return self.log_gamma


class PhiNetWithGamma(nn.Module):
    """
    Extended Phi Network with learnable gamma parameters for HSIVI.
    
    This version supports both isotropic and non-isotropic (image-dependent)
    gamma values for the conditional distributions.
    """
    
    def __init__(
        self,
        image_size=28,
        channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4),
        time_emb_dim=256,
        num_res_blocks=2,
        attn_resolutions=(14,),
        dropout=0.1,
        n_discrete_steps=11,
        image_gamma=True,
        independent_log_gamma=False
    ):
        super().__init__()
        
        self.phi_net = PhiNet(
            image_size=image_size,
            channels=channels,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_emb_dim=time_emb_dim,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout
        )
        
        self.n_discrete_steps = n_discrete_steps
        self.image_gamma = image_gamma
        self.independent_log_gamma = independent_log_gamma
        
        # Learnable log gamma parameters
        if image_gamma:
            # Non-isotropic: gamma per pixel
            if independent_log_gamma:
                # Independent gamma for each layer
                self.log_gamma = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, channels, image_size, image_size))
                    for _ in range(n_discrete_steps - 1)
                ])
            else:
                # Shared gamma across layers
                self.log_gamma = nn.Parameter(torch.zeros(1, channels, image_size, image_size))
        else:
            # Isotropic: single gamma value
            if independent_log_gamma:
                self.log_gamma = nn.ParameterList([
                    nn.Parameter(torch.zeros(1))
                    for _ in range(n_discrete_steps - 1)
                ])
            else:
                self.log_gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, t, noise=None):
        return self.phi_net(x, t, noise)
    
    def get_log_gamma(self, layer_idx=None):
        """Get log gamma for a specific layer or all layers."""
        if self.independent_log_gamma:
            if layer_idx is not None:
                return self.log_gamma[layer_idx]
            return self.log_gamma
        return self.log_gamma
    
    def sample_from_conditional(self, x, t, layer_idx=None):
        """
        Sample from the conditional distribution q(x_{t-1} | x_t).
        
        Args:
            x: Current sample x_t
            t: Current timestep
            layer_idx: Layer index for gamma selection
        
        Returns:
            Sampled x_{t-1}
        """
        mean = self.forward(x, t)
        log_gamma = self.get_log_gamma(layer_idx)
        std = torch.exp(0.5 * log_gamma)
        
        if self.training:
            noise = torch.randn_like(mean)
            return mean + std * noise
        return mean

