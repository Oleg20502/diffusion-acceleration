import os, json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_DIR = "/home/jovyan/chugunov/UNet/UNet/data/training"  
OUT_H5  = os.path.join("/home/jovyan/chugunov/UNet/UNet/data", "training.h5")
OUT_JSON = os.path.join("/home/jovyan/chugunov/UNet/UNet/data", "training.json")
IMAGE_SIZE = 28

exts = ['.jpg', '.jpeg', '.png']
paths = [p for ext in exts for p in Path(DATA_DIR).glob(f"**/*{ext}")]

hf = h5py.File(OUT_H5, 'w')
index = {}

for p in tqdm(paths):
    rel = str(p.relative_to(DATA_DIR))  
    img = Image.open(p).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.array(img)           
    arr = arr.transpose(2, 0, 1)    
    hf.create_dataset(rel, data=arr, compression="gzip", compression_opts=4)
    index[rel] = {"shape": arr.shape}

hf.close()

with open(OUT_JSON, "w") as f:
    json.dump(index, f)