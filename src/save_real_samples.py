import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_h5 import H5ImagesDataset


def main():
    parser = argparse.ArgumentParser(description="Save a grid of real images from H5 dataset.")
    parser.add_argument("--data-h5", type=str, default=str(ROOT / "data"), help="Dir with training.h5/json.")
    parser.add_argument("--image-size", type=int, default=32, help="Resize real images to this size.")
    parser.add_argument("--num", type=int, default=64, help="Number of images to save.")
    parser.add_argument("--out", type=str, default=str(ROOT / "real_samples.png"), help="Output image path.")
    parser.add_argument("--nrow", type=int, default=8, help="Grid row size for save_image.")
    args = parser.parse_args()

    tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size))])
    ds = H5ImagesDataset(args.data_h5, transform=tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.num, shuffle=True, num_workers=0, pin_memory=False)

    batch = next(iter(dl))
    imgs = batch[: args.num]
    # imgs already in [0,1]; save_image expects [0,1]
    save_image(imgs, args.out, nrow=args.nrow)
    print(f"Saved {imgs.shape[0]} images to {args.out}")


if __name__ == "__main__":
    main()

