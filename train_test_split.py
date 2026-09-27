import os
import glob
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SatellitePatchDataset(Dataset):
    def __init__(self, in_paths, tg_paths):
        self.inputs, self.targets = [], []
        for i in range(len(in_paths)):
            with rasterio.open(in_paths[i]) as s:
                d = s.read().astype(np.float32)
                if (d.max() - d.min()) > 0: d = (d - d.min()) / (d.max() - d.min())
                self.inputs.append(torch.tensor(d))
            with rasterio.open(tg_paths[i]) as s:
                d = s.read().astype(np.float32)
                if (d.max() - d.min()) > 0: d = (d - d.min()) / (d.max() - d.min())
                self.targets.append(torch.tensor(d))

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

def build_pipeline_splits(base_dir, experiment):
    if experiment == "COLORIZE":
        in_dir, target_dir = os.path.join(base_dir, "TIR_100m_p256_s128"), os.path.join(base_dir, "RGB_100m_p256_s128")
        in_suffix, tgt_suffix = "_TIR_100m_patch_", "_RGB_100m_patch_"
    else:
        in_dir, target_dir = os.path.join(base_dir, "TIR_200m_p256_s128"), os.path.join(base_dir, "TIR_100m_p512_s256")
        in_suffix, tgt_suffix = "_TIR_200m_patch_", "_TIR_100m_patch_"

    in_files = sorted(glob.glob(os.path.join(in_dir, "*.tif")))
    paired_in, paired_tgt = [], []
    for f in in_files:
        filename = os.path.basename(f)
        if in_suffix not in filename: continue
        parts = filename.split(in_suffix)
        expected_name = f"{parts[0]}{tgt_suffix}{parts[1]}"
        target_path = os.path.join(target_dir, expected_name)
        if os.path.exists(target_path):
            paired_in.append(f)
            paired_tgt.append(target_path)
            
    tr_in, te_in, tr_tg, te_tg = train_test_split(paired_in, paired_tgt, test_size=0.30, random_state=42)
    va_in, ts_in, va_tg, ts_tg = train_test_split(te_in, te_tg, test_size=0.50, random_state=42)
    return (tr_in, tr_tg), (va_in, va_tg), (ts_in, ts_tg)