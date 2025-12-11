import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scr.train_gans import DCGANGenerator


def main():
    parser = argparse.ArgumentParser(description="Sample batches from DCGAN checkpoint.")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "ckpts" / "dcgan.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to generate.")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "dcgan_samples"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    z_dim = ckpt["z_dim"]

    G = DCGANGenerator(z_dim=z_dim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(args.batches):
            z = torch.randn(args.batch_size, z_dim, device=device)
            imgs = G(z)  # [-1,1]
            imgs = (imgs + 1) * 0.5  # [0,1]
            save_image(imgs, out_dir / f"dcgan_batch_{i:04d}.png", nrow=int(args.batch_size ** 0.5))


if __name__ == "__main__":
    main()