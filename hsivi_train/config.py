"""
Configuration for HSIVI Training
Adapted from https://github.com/longinYu/HSIVI
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import json
from pathlib import Path


@dataclass
class HSIVIConfig:
    """Configuration for HSIVI training on colored MNIST."""
    
    # Data settings
    data_dir: str = "./data"
    image_size: int = 28
    channels: int = 3
    
    # Model architecture - Phi Network
    phi_base_dim: int = 64
    phi_dim_mults: Tuple[int, ...] = (1, 2, 4)
    phi_num_res_blocks: int = 2
    phi_attn_resolutions: Tuple[int, ...] = (14,)
    phi_dropout: float = 0.1
    phi_time_emb_dim: int = 256
    
    # Model architecture - F Network  
    f_base_dim: int = 64
    f_dim_mults: Tuple[int, ...] = (1, 2, 4)
    f_num_res_blocks: int = 2
    f_use_spectral_norm: bool = True
    f_time_emb_dim: int = 256
    f_simple: bool = True  # Use simplified F network for MNIST
    
    # HSIVI specific settings
    n_discrete_steps: int = 11  # NFE + 1
    independent_log_gamma: bool = False  # Independent vs shared log_gamma
    image_gamma: bool = True  # Non-isotropic (pixel-wise) vs isotropic gamma
    skip_type: str = "quad"  # 'uniform' or 'quad' for timestep selection
    
    # Training settings
    n_train_iters: int = 100000
    training_batch_size: int = 64
    phi_learning_rate: float = 1.6e-5
    f_learning_rate: float = 8e-5
    f_learning_times: int = 20  # F network updates per phi update
    
    # Optimizer settings
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    
    # EMA settings
    ema_decay: float = 0.9999
    ema_update_every: int = 1
    
    # Pretrained model path (for diffusion model's epsilon network)
    pretrained_model: Optional[str] = None
    
    # Diffusion settings
    diffusion_timesteps: int = 1000
    beta_schedule: str = "linear"  # 'linear', 'cosine', 'sigmoid'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Logging and saving
    workdir: str = "./work_dir/hsivi_colored_mnist"
    save_every: int = 5000
    sample_every: int = 1000
    log_every: int = 100
    num_samples: int = 64
    
    # Evaluation
    testing_batch_size: int = 64
    sampling_batch_size: int = 64
    fid_num_samples: int = 10000
    fid_every: int = 10000  # Calculate FID every N steps (0 to disable)
    
    # Hardware
    n_gpus_per_node: int = 1
    seed: int = 42
    mixed_precision: bool = True
    
    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            k: list(v) if isinstance(v, tuple) else v 
            for k, v in self.__dict__.items()
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'HSIVIConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples
        tuple_keys = ['phi_dim_mults', 'phi_attn_resolutions', 'f_dim_mults']
        for key in tuple_keys:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        
        return cls(**config_dict)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.skip_type in ['uniform', 'quad'], \
            f"skip_type must be 'uniform' or 'quad', got {self.skip_type}"
        assert self.beta_schedule in ['linear', 'cosine', 'sigmoid'], \
            f"beta_schedule must be 'linear', 'cosine', or 'sigmoid', got {self.beta_schedule}"
        assert self.n_discrete_steps >= 2, \
            f"n_discrete_steps must be >= 2, got {self.n_discrete_steps}"


def get_default_config() -> HSIVIConfig:
    """Get default configuration for colored MNIST."""
    return HSIVIConfig()


def get_fast_config() -> HSIVIConfig:
    """Get fast training configuration for testing."""
    return HSIVIConfig(
        n_train_iters=10000,
        save_every=2000,
        sample_every=500,
        log_every=50,
        n_discrete_steps=6,
        phi_base_dim=32,
        f_base_dim=32,
    )


def get_high_quality_config() -> HSIVIConfig:
    """Get high quality configuration for best results."""
    return HSIVIConfig(
        n_train_iters=200000,
        n_discrete_steps=11,
        phi_base_dim=96,
        f_base_dim=96,
        phi_dim_mults=(1, 2, 4, 8),
        f_dim_mults=(1, 2, 4, 8),
        training_batch_size=128,
        f_learning_times=30,
    )

