"""
Utility script to train lightweight GANs for seeding ES-DDPM.

Supports:
- DCGAN (fast, minimal; good for 28x28/32x32 color MNIST)
- StyleGAN2 (via stylegan2_pytorch) if you have time / deps

Examples:
DCGAN on H5 dataset:
  python train_gans.py --arch dcgan --data-h5 ./data --epochs 20 --batch-size 256

StyleGAN2 (requires `stylegan2_pytorch` installed) on folder of PNGs:
  python train_gans.py --arch stylegan2 --data-folder ./data/training --epochs 10
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_h5 import H5ImagesDataset


class DCGANGenerator(nn.Module):
    """
    Генератор DCGAN для выхода 28x28.
    """
    def __init__(self, z_dim: int = 128, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),             # (B, 256, 7, 7)
            nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0),   # (B, 256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),   # (B, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),  # (B, C, 28, 28)
            nn.Tanh(),
        )


    # def __init__(self, z_dim: int = 128, channels: int = 3):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(z_dim, 256 * 7 * 7),
    #         nn.BatchNorm1d(256 * 7 * 7),
    #         nn.ReLU(inplace=True),
    #         nn.Unflatten(1, (256, 7, 7)),             # (B, 256, 7, 7)
    #         nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),   # (B, 128, 7, 7)
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # (B, 64, 14, 14)
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # (B, 32, 28, 28)
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),  # (B, C, 28, 28)
    #         nn.Tanh(),
    #     )

    # def __init__(self, z_dim: int = 128, channels: int = 3):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(z_dim, 256 * 7 * 7),
    #         nn.BatchNorm1d(256 * 7 * 7),
    #         nn.ReLU(inplace=True),
    #         nn.Unflatten(1, (256, 7, 7)),
    #         nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 7x7 -> 14x14
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 14x14 -> 28x28
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
    #         nn.Tanh(),
    #     )

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def get_unique_exp_dir(base: Path, name: str) -> Path:
    candidate = base / name
    if not candidate.exists():
        return candidate
    i = 1
    while (base / f"{name}_{i}").exists():
        i += 1
    return base / f"{name}_{i}"


def ensure_unique_dir(path: Path) -> Path:
    """
    Return a path that does not exist. If `path` exists, append _i suffix.
    """
    if not path.exists():
        return path
    i = 1
    base = path
    while (base.parent / f"{base.name}_{i}").exists():
        i += 1
    return base.parent / f"{base.name}_{i}"


def build_h5_loader(data_dir: str, batch_size: int, image_size: int, num_workers: int = 0):
    assert image_size == 28, "DCGAN generator outputs 28x28; set --image-size 28 for H5 data."
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
        ]
    )
    ds = H5ImagesDataset(data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def build_folder_loader(folder: str, batch_size: int, image_size: int, num_workers: int = 0):
    from torchvision.datasets import ImageFolder

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    ds = ImageFolder(folder, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def train_dcgan(
    data_loader: DataLoader,
    epochs: int,
    z_dim: int,
    lr_g: float,
    lr_d: float,
    d_steps: int,
    g_steps: int,
    label_smooth: float,
    disc_noise_std: float,
    device: torch.device,
    sample_dir: Path,
    ckpt_path: Path,
    ema_decay: float = 0.0,
    log_dir: Optional[Path] = None,
    load_ckpt: Optional[str] = None,    
):
    G = DCGANGenerator(z_dim=z_dim).to(device)
    D = DCGANDiscriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    if load_ckpt is not None:
        ckpt = torch.load(load_ckpt, map_location=device)
        arch_in = ckpt.get("arch", "dcgan")
        if arch_in != "dcgan":
            raise ValueError(f"Checkpoint arch {arch_in} does not match expected dcgan")
        if "z_dim" in ckpt and ckpt["z_dim"] != z_dim:
            raise ValueError(f"Checkpoint z_dim {ckpt['z_dim']} != current z_dim {z_dim}")
        if "G" in ckpt:
            G.load_state_dict(ckpt["G"])
        if "D" in ckpt:
            D.load_state_dict(ckpt["D"])
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    if load_ckpt is not None:
        if "opt_G" in ckpt:
            opt_G.load_state_dict(ckpt["opt_G"])
        if "opt_D" in ckpt:
            opt_D.load_state_dict(ckpt["opt_D"])
    criterion = nn.BCEWithLogitsLoss()

    ema_G = None
    if ema_decay > 0:
        ema_G = DCGANGenerator(z_dim=z_dim).to(device)
        ema_G.load_state_dict(G.state_dict())

    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        loss_d_hist: List[float] = []
        loss_g_hist: List[float] = []
    best_g_loss = float("inf")
    best_ckpt_path = ckpt_path.parent / "dcgan_best.pt"

    # save initial samples to inspect initialization
    with torch.no_grad():
        z = torch.randn(64, z_dim, device=device)
        init_samples = G(z)
        init_samples = (init_samples + 1) * 0.5
        save_image(init_samples, sample_dir / "dcgan_samples_init.png", nrow=8)

    for epoch in range(epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                real = batch[0]
            else:
                real = batch
            real = real.to(device)
            real = real * 2 - 1  # [-1,1]

            # Train D (possibly multiple steps)
            loss_D = torch.tensor(0.0, device=device)
            for _ in range(d_steps):
                z = torch.randn(real.size(0), z_dim, device=device)
                fake = G(z).detach()

                if disc_noise_std > 0:
                    noise_real = torch.randn_like(real) * disc_noise_std
                    noise_fake = torch.randn_like(fake) * disc_noise_std
                    real_in = real + noise_real
                    fake_in = fake + noise_fake
                else:
                    real_in = real
                    fake_in = fake

                D_real = D(real_in).squeeze()
                D_fake = D(fake_in).squeeze()

                real_labels = torch.ones_like(D_real) * label_smooth
                fake_labels = torch.zeros_like(D_fake)

                loss_D_step = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
                opt_D.zero_grad()
                loss_D_step.backward()
                opt_D.step()
                loss_D = loss_D_step

            # Train G (possibly multiple steps)
            loss_G = torch.tensor(0.0, device=device)
            for _ in range(g_steps):
                z = torch.randn(real.size(0), z_dim, device=device)
                fake = G(z)
                D_fake = D(fake).squeeze()
                loss_G_step = criterion(D_fake, torch.ones_like(D_fake))
                opt_G.zero_grad()
                loss_G_step.backward()
                opt_G.step()
                loss_G = loss_G_step

            if ema_G is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_G.parameters(), G.parameters()):
                        p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            pbar.set_postfix({"loss_D": loss_D.item(), "loss_G": loss_G.item()})
            if log_dir is not None:
                loss_d_hist.append(loss_D.item())
                loss_g_hist.append(loss_G.item())

            # track best generator loss per batch
            current_g_loss = loss_G.item()
            if current_g_loss < best_g_loss:
                best_g_loss = current_g_loss
                torch.save(
                    {
                        "arch": "dcgan",
                        "z_dim": z_dim,
                        "G": (ema_G if ema_G is not None else G).state_dict(),
                        "D": D.state_dict(),
                        "opt_G": opt_G.state_dict(),
                        "opt_D": opt_D.state_dict(),
                        "best_loss_G": best_g_loss,
                        "step": epoch * len(data_loader) + (_ + 1),
                        "epoch": epoch + 1,
                    },
                    best_ckpt_path,
                )

        # save samples (EMA for inference if available)
        with torch.no_grad():
            z = torch.randn(64, z_dim, device=device)
            # diff_max = (z[0] - z[1]).abs().max().item()
            # print(f"Max difference between two random z vectors: {diff_max}")
            samples = (ema_G if ema_G is not None else G)(z)
            samples = (samples + 1) * 0.5
            save_image(samples, sample_dir / f"dcgan_samples_epoch{epoch+1}.png", nrow=8)

    if log_dir is not None:
        log_txt = log_dir / "losses.csv"
        with log_txt.open("w", encoding="utf-8") as f:
            f.write("step,loss_D,loss_G\n")
            for i, (ld, lg) in enumerate(zip(loss_d_hist, loss_g_hist)):
                f.write(f"{i},{ld},{lg}\n")

        plt.figure(figsize=(8, 4))
        plt.plot(loss_d_hist, label="loss_D")
        plt.plot(loss_g_hist, label="loss_G")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "loss_plot.png", dpi=150)
        plt.close()

    ckpt_dir = ckpt_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing = list(ckpt_dir.glob("dcgan_*.pt"))
    max_idx = 0
    for p in existing:
        try:
            idx = int(p.stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    next_idx = max_idx + 1
    save_path = ckpt_dir / f"dcgan_{next_idx}.pt"

    torch.save(
        {
            "arch": "dcgan",
            "z_dim": z_dim,
            "G": (ema_G if ema_G is not None else G).state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        },
        save_path,
    )
    print(f"DCGAN training done. Saved to {save_path}")


def train_stylegan2(folder: str, image_size: int, batch_size: int, epochs: int, results_dir: Path, load_ckpt: Optional[str] = None):
    """
    Lightweight wrapper around stylegan2_pytorch.Trainer (optional dependency).
    Data must be a folder of images.
    """
    try:
        from stylegan2_pytorch import Trainer
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError(
            "stylegan2_pytorch is not installed. Install via `pip install stylegan2_pytorch` to enable this mode."
        ) from exc

    trainer = Trainer(
        data=folder,
        image_size=image_size,
        results_dir=str(results_dir),
        batch_size=batch_size,
        num_workers=4,
        aug_prob=0.15,  # light ADA-style prob
    )
    # Trainer handles its own loop; epochs approximated via steps
    steps = epochs * 1000
    trainer.train(steps=steps, load_from_checkpoint=load_ckpt)
    print(f"StyleGAN2 training done. Checkpoints in {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GANs (DCGAN, StyleGAN2) for ES-DDPM seeding.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml; CLI overrides config.")
    parser.add_argument("--arch", type=str, choices=["dcgan", "stylegan2"], default="dcgan")
    parser.add_argument("--data-h5", type=str, default=str(ROOT / "data"), help="Dir with training.h5/json (for DCGAN).")
    parser.add_argument("--data-folder", type=str, default=str(ROOT / "data" / "training"), help="Folder of images (for StyleGAN2).")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--lrG", type=float, default=2e-4, help="Learning rate for generator.")
    parser.add_argument("--lrD", type=float, default=2e-4, help="Learning rate for discriminator.")
    parser.add_argument("--d-steps", type=int, default=1, help="Discriminator steps per batch.")
    parser.add_argument("--g-steps", type=int, default=1, help="Generator steps per batch.")
    parser.add_argument("--label-smooth", type=float, default=0.95, help="Real label smoothing (0-1].")
    parser.add_argument("--disc-noise-std", type=float, default=0.01, help="Gaussian noise std added to D inputs.")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--exp-name", type=str, default="default", help="Experiment name for saving outputs.")
    parser.add_argument("--results", type=str, default=None, help="Override samples dir; otherwise experiments/<name>/_i/samples")
    parser.add_argument("--ckpt", type=str, default=None, help="Override ckpt dir; otherwise experiments/<name>/_i/ckpts")
    parser.add_argument("--log-root", type=str, default=None, help="Override logs dir; otherwise experiments/<name>/_i/logs")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Path to load initial checkpoint from.")
    args = parser.parse_args()

    # config loading: load yaml, apply where arg is at default; CLI overrides
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            default_v = parser.get_default(k)
            if getattr(args, k) == default_v:
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_base = ROOT / "experiments"
    exp_base.mkdir(parents=True, exist_ok=True)
    exp_name = args.exp_name
    exp_dir = ensure_unique_dir(get_unique_exp_dir(exp_base, exp_name))
    exp_dir.mkdir(parents=True, exist_ok=True)

    results_dir = ensure_unique_dir(Path(args.results)) if args.results else ensure_unique_dir(exp_dir / "samples")
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ensure_unique_dir(Path(args.ckpt).parent) if args.ckpt else ensure_unique_dir(exp_dir / "ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / "dcgan.pt"

    log_root = ensure_unique_dir(Path(args.log_root)) if args.log_root else ensure_unique_dir(exp_dir / "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = log_root
    # save config snapshot
    config_out = exp_dir / "config.yaml"
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True, allow_unicode=True)

    if args.arch == "dcgan":
        loader = build_h5_loader(args.data_h5, args.batch_size, args.image_size)
        train_dcgan(
            loader,
            epochs=args.epochs,
            z_dim=args.z_dim,
            lr_g=args.lrG,
            lr_d=args.lrD,
            d_steps=args.d_steps,
            g_steps=args.g_steps,
            label_smooth=args.label_smooth,
            disc_noise_std=args.disc_noise_std,
            device=device,
            sample_dir=results_dir,
            ckpt_path=ckpt_path,
            ema_decay=args.ema_decay,
            log_dir=run_dir,
            load_ckpt=args.load_ckpt,
        )
    else:
        train_stylegan2(
            folder=args.data_folder,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            results_dir=results_dir,
            load_ckpt=args.load_ckpt,
        )


if __name__ == "__main__":
    main()

