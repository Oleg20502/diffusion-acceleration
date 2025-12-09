# datasets_h5.py
import io, json, os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_h5_file(hf, path):
    return np.array(hf[path])


class H5ImagesDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "training.h5")
        self.h5_json_path = os.path.join(self.data_dir, "training.json")

        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            h5_json = json.load(f)

        self.filelist = sorted(h5_json.keys())

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fname = self.filelist[index]
        img = load_h5_file(self.h5f, fname)
        img = torch.from_numpy(img).float() / 255.0  
        return img

    def __del__(self):
        try:
            self.h5f.close()
        except Exception:
            pass