"""
Train a lightweight convolutional VAE for image generation (28x28 by default).

Examples:
  python train_vae.py --data-h5 ./data --epochs 20 --batch-size 256
  python train_vae.py --dataset folder --data-folder ./data/training --image-size 32
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Type
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_h5 import H5ImagesDataset


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


def get_unique_exp_dir(base: Path, name: str) -> Path:
    candidate = base / name
    if not candidate.exists():
        return candidate
    i = 1
    while (base / f"{name}_{i}").exists():
        i += 1
    return base / f"{name}_{i}"


def build_h5_loader(data_dir: str, batch_size: int, image_size: int, num_workers: int = 0):
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


class ConvVAE(nn.Module):
    def __init__(self, z_dim: int = 64, channels: int = 3):
        super().__init__()
        # Увеличенная глубина энкодера (больше слоёв на каждом разрешении)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # дополнительный слой 7 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

        self.fc_decode = nn.Linear(z_dim, self.flat_dim)
        # Симметричный декодер с дополнительными слоями на 7x7, 14x14 и 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 28 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), 128, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ConvVAEDeep(nn.Module):
    """
    Более широкий/глубокий вариант (~5-6x параметров относительно ConvVAE).
    """

    def __init__(self, z_dim: int = 128, channels: int = 3):
        super().__init__()
        # Энкодер: больше каналов и один дополнительный блок на 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # доп. слой 7 -> 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = 512 * 7 * 7
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

        self.fc_decode = nn.Linear(z_dim, self.flat_dim)
        # Декодер симметричный и расширенный
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 28 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), 512, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def train_vae(
    data_loader: DataLoader,
    epochs: int,
    z_dim: int,
    lr: float,
    beta: float,
    device: torch.device,
    sample_dir: Path,
    ckpt_path: Path,
    log_dir: Optional[Path] = None,
    channels: int = 3,
    load_ckpt: Optional[str] = None,
    fid_every: int = 0,
    fid_samples: int = 1024,
    fid_batch_size: int = 256,
    model_cls: Type[nn.Module] = ConvVAE,
    model_arch: str = "base",
    y_max: Optional[float] = None,
):
    model = model_cls(z_dim=z_dim, channels=channels).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def write_model_info():
        if log_dir is None:
            return
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info_path = log_dir / "model.txt"
        with info_path.open("w", encoding="utf-8") as f:
            f.write("Model architecture:\n")
            f.write(repr(model))
            f.write("\n\n")
            f.write(f"Total parameters: {total_params}\n")
            f.write(f"Trainable parameters: {trainable_params}\n")

    start_epoch = 0
    best_loss = float("inf")
    if load_ckpt is not None:
        ckpt = torch.load(load_ckpt, map_location=device)
        if ckpt.get("arch") != "vae":
            raise ValueError(f"Checkpoint arch {ckpt.get('arch')} does not match expected vae")
        ckpt_arch = ckpt.get("model_arch", "base")
        if ckpt_arch != model_arch:
            raise ValueError(f"Checkpoint model_arch {ckpt_arch} != current {model_arch}")
        if "z_dim" in ckpt and ckpt["z_dim"] != z_dim:
            raise ValueError(f"Checkpoint z_dim {ckpt['z_dim']} != current z_dim {z_dim}")
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))

    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        loss_total_hist: List[float] = []
        loss_recon_hist: List[float] = []
        loss_kl_hist: List[float] = []
        fid_hist: List[Tuple[int, float]] = []
        write_model_info()

    # initial samples
    with torch.no_grad():
        z = torch.randn(64, z_dim, device=device)
        init_samples = model.decode(z)
        save_image(init_samples, sample_dir / "vae_samples_init.png", nrow=8)

    def write_loss_history():
        if log_dir is None:
            return
        log_txt = log_dir / "losses.csv"
        with log_txt.open("w", encoding="utf-8") as f:
            f.write("step,loss_total,loss_recon,loss_kl\n")
            for i, (lt, lr_, lk) in enumerate(zip(loss_total_hist, loss_recon_hist, loss_kl_hist)):
                f.write(f"{i},{lt},{lr_},{lk}\n")

    def save_loss_plot():
        if log_dir is None or not loss_total_hist:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(loss_total_hist, label="loss_total")
        plt.plot(loss_recon_hist, label="loss_recon")
        plt.plot(loss_kl_hist, label="loss_kl")
        plt.xlabel("step")
        plt.ylabel("loss")
        if y_max is not None:
            plt.ylim(top=y_max)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "loss_plot.png", dpi=150)
        plt.close()

    def write_fid_history():
        if log_dir is None or not fid_hist:
            return
        fid_txt = log_dir / "fid.csv"
        with fid_txt.open("w", encoding="utf-8") as f:
            f.write("epoch,fid\n")
            for ep, val in fid_hist:
                f.write(f"{ep},{val}\n")

    def save_fid_plot():
        if log_dir is None or not fid_hist:
            return
        epochs_arr = [ep for ep, _ in fid_hist]
        fid_arr = [v for _, v in fid_hist]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs_arr, fid_arr, marker="o", label="FID")
        plt.xlabel("epoch")
        plt.ylabel("FID (lower is better)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "fid_plot.png", dpi=150)
        plt.close()

    def compute_fid():
        if fid_every <= 0:
            return None
        prev_mode = model.training
        model.eval()
        try:
            weights = Inception_V3_Weights.IMAGENET1K_V1
            inception = inception_v3(weights=weights, aux_logits=True, transform_input=False)
            inception.fc = nn.Identity()
            inception.to(device)
            inception.eval()

            # Mean/std: use transforms() if meta missing (older torchvision may not have mean/std in meta)
            if "mean" in weights.meta and "std" in weights.meta:
                mean_vals = weights.meta["mean"]
                std_vals = weights.meta["std"]
            else:
                tr = weights.transforms()
                mean_vals = tr.mean
                std_vals = tr.std
            mean = torch.tensor(mean_vals, device=device).view(1, 3, 1, 1)
            std = torch.tensor(std_vals, device=device).view(1, 3, 1, 1)

            def get_feats_from_imgs(imgs: torch.Tensor) -> torch.Tensor:
                imgs = imgs.clone()
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
                imgs = (imgs - mean) / std
                with torch.no_grad():
                    out = inception(imgs)
                # out may be InceptionOutputs(logits, aux_logits) when aux_logits=True
                if isinstance(out, (tuple, list)):
                    feats = out[0]
                elif isinstance(out, InceptionOutputs):
                    feats = out.logits
                else:
                    feats = out
                return feats

            real_feats = []
            real_seen = 0
            for batch in data_loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device)
                real_feats.append(get_feats_from_imgs(imgs))
                real_seen += imgs.size(0)
                if real_seen >= fid_samples:
                    break
            if not real_feats:
                return None
            real_feats = torch.cat(real_feats, dim=0)[:fid_samples]

            fake_feats = []
            fake_seen = 0
            while fake_seen < fid_samples:
                cur_bs = min(fid_batch_size, fid_samples - fake_seen)
                z = torch.randn(cur_bs, z_dim, device=device)
                with torch.no_grad():
                    fake_imgs = model.decode(z)
                fake_feats.append(get_feats_from_imgs(fake_imgs))
                fake_seen += cur_bs
            fake_feats = torch.cat(fake_feats, dim=0)[:fid_samples]

            def feats_to_stats(feats: torch.Tensor):
                feats_np = feats.cpu().numpy()
                mu = np.mean(feats_np, axis=0)
                sigma = np.cov(feats_np, rowvar=False)
                return mu, sigma

            mu_r, sigma_r = feats_to_stats(real_feats)
            mu_f, sigma_f = feats_to_stats(fake_feats)

            def sqrtm_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                prod = a.dot(b)
                vals, vecs = np.linalg.eigh(prod)
                vals = np.clip(vals, 0, None)
                sqrt_vals = np.sqrt(vals)
                return (vecs * sqrt_vals).dot(vecs.T)

            covmean = sqrtm_product(sigma_r, sigma_f)
            diff = mu_r - mu_f
            fid_value = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
            fid_value = float(np.real(fid_value))
            return fid_value
        finally:
            model.train(prev_mode)

    global_step = 0
    for epoch in range(start_epoch, epochs):
        epoch_total = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta)
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            epoch_total += float(total_loss.item())
            pbar.set_postfix({"loss": total_loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()})

            if log_dir is not None:
                global_step += 1
                loss_total_hist.append(float(total_loss.item()))
                loss_recon_hist.append(float(recon_loss.item()))
                loss_kl_hist.append(float(kl_loss.item()))

        # end of epoch bookkeeping
        denom = max(1, len(data_loader))
        avg_epoch_loss = epoch_total / denom
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "arch": "vae",
                    "model_arch": model_arch,
                    "z_dim": z_dim,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "beta": beta,
                },
                ckpt_path.parent / "vae_best.pt",
            )

        with torch.no_grad():
            z = torch.randn(64, z_dim, device=device)
            samples = model.decode(z)
            save_image(samples, sample_dir / f"vae_samples_epoch{epoch+1}.png", nrow=8)

        # persist loss history each epoch (overwrite the same file)
        write_loss_history()
        # save intermediate loss plot each epoch
        save_loss_plot()

        # compute and log FID every fid_every epochs
        if fid_every > 0 and (epoch + 1) % fid_every == 0:
            fid_val = compute_fid()
            if fid_val is not None and log_dir is not None:
                fid_hist.append((epoch + 1, fid_val))
                write_fid_history()
                save_fid_plot()

    # save logs
    if log_dir is not None:
        save_loss_plot()
        write_fid_history()
        save_fid_plot()

    torch.save(
        {
            "arch": "vae",
            "model_arch": model_arch,
            "z_dim": z_dim,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epochs,
            "beta": beta,
            "best_loss": best_loss,
        },
        ckpt_path,
    )
    print(f"VAE training done. Saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a convolutional VAE for image generation.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml; CLI overrides config.")
    parser.add_argument("--dataset", type=str, choices=["h5", "folder"], default="h5")
    parser.add_argument("--data-h5", type=str, default=str(ROOT / "data"), help="Dir with training.h5/json (for H5 dataset).")
    parser.add_argument("--data-folder", type=str, default=str(ROOT / "data" / "training"), help="Folder of images (for folder dataset).")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight.")
    parser.add_argument("--channels", type=int, default=3, help="Input channels.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default="vae_default", help="Experiment name for saving outputs.")
    parser.add_argument("--results", type=str, default=None, help="Override samples dir; otherwise experiments/<name>/_i/samples")
    parser.add_argument("--ckpt", type=str, default=None, help="Override ckpt path; otherwise experiments/<name>/_i/ckpts/vae.pt")
    parser.add_argument("--log-root", type=str, default=None, help="Override logs dir; otherwise experiments/<name>/_i/logs")
    parser.add_argument("--load-ckpt", type=str, default=None, help="Path to load initial checkpoint from.")
    parser.add_argument("--fid-every", type=int, default=0, help="Compute FID every N epochs; 0 disables.")
    parser.add_argument("--fid-samples", type=int, default=1024, help="Number of samples for FID (per real/fake).")
    parser.add_argument("--fid-batch-size", type=int, default=256, help="Batch size for FID forward pass.")
    parser.add_argument("--vae-arch", type=str, choices=["base", "deep"], default="base", help="Choose VAE architecture.")
    parser.add_argument("--y-max", type=float, default=None, help="Clamp loss plot upper y-limit.")
    args = parser.parse_args()

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
    exp_dir = ensure_unique_dir(get_unique_exp_dir(exp_base, args.exp_name))
    exp_dir.mkdir(parents=True, exist_ok=True)

    results_dir = ensure_unique_dir(Path(args.results)) if args.results else ensure_unique_dir(exp_dir / "samples")
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ensure_unique_dir(Path(args.ckpt).parent) if args.ckpt else ensure_unique_dir(exp_dir / "ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / "vae.pt"

    log_root = ensure_unique_dir(Path(args.log_root)) if args.log_root else ensure_unique_dir(exp_dir / "logs")
    log_root.mkdir(parents=True, exist_ok=True)

    config_out = exp_dir / "config.yaml"
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True, allow_unicode=True)

    if args.dataset == "h5":
        loader = build_h5_loader(args.data_h5, args.batch_size, args.image_size, num_workers=args.num_workers)
    else:
        loader = build_folder_loader(args.data_folder, args.batch_size, args.image_size, num_workers=args.num_workers)

    model_cls = ConvVAE if args.vae_arch == "base" else ConvVAEDeep

    train_vae(
        loader,
        epochs=args.epochs,
        z_dim=args.z_dim,
        lr=args.lr,
        beta=args.beta,
        device=device,
        sample_dir=results_dir,
        ckpt_path=ckpt_path,
        log_dir=log_root,
        channels=args.channels,
        load_ckpt=args.load_ckpt,
        fid_every=args.fid_every,
        fid_samples=args.fid_samples,
        fid_batch_size=args.fid_batch_size,
        model_cls=model_cls,
        model_arch=args.vae_arch,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()

