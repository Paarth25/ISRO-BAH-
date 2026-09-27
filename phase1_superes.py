import os
import glob
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import rasterio

# Force clear out any residual GPU clutter before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
BASE_DIR = r"C:\Users\vegeta\Desktop\isro\ps10\models\dataset"

# TOGGLE BETWEEN YOUR TWO MODELS HERE:
# "SUPER_RES"  -> TIR 200m (256x256) to TIR 100m (512x512) [Model 1]
# "COLORIZE"   -> TIR 100m (256x256) to RGB 100m (256x256) [Model 2]
EXPERIMENT = "SUPER_RES" 

BATCH_SIZE = 4 if EXPERIMENT == "SUPER_RES" else 16
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=== GPU Pipeline Initialized ===")
print(f"Running Task: {EXPERIMENT} | Batch Size: {BATCH_SIZE}\n")

# =====================================================================
# 2. DATASET SPLITS AND ALIGNMENT PIPELINE
# =====================================================================
class LoaderDataset(Dataset):
    def __init__(self, in_p, tg_p):
        self.inputs, self.targets = [], []
        print(f"Loading {len(in_p)} elements directly into System RAM...")
        for i in range(len(in_p)):
            with rasterio.open(in_p[i]) as s:
                d = s.read().astype(np.float32)
                if (d.max() - d.min()) > 0: 
                    d = (d - d.min()) / (d.max() - d.min())
                self.inputs.append(torch.tensor(d))
            with rasterio.open(tg_p[i]) as s:
                d = s.read().astype(np.float32)
                if (d.max() - d.min()) > 0: 
                    d = (d - d.min()) / (d.max() - d.min())
                self.targets.append(torch.tensor(d))

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

def get_splits():
    if EXPERIMENT == "COLORIZE":
        in_dir = os.path.join(BASE_DIR, "TIR_100m_p256_s128")
        target_dir = os.path.join(BASE_DIR, "RGB_100m_p256_s128")
        in_suffix = "_TIR_100m_patch_"
        tgt_suffix = "_RGB_100m_patch_"
    else:
        in_dir = os.path.join(BASE_DIR, "TIR_200m_p256_s128")
        target_dir = os.path.join(BASE_DIR, "TIR_100m_p512_s256")
        in_suffix = "_TIR_200m_patch_"
        tgt_suffix = "_TIR_100m_patch_"

    in_files = sorted(glob.glob(os.path.join(in_dir, "*.tif")))
    paired_in, paired_tgt = [], []
    
    for f in in_files:
        filename = os.path.basename(f)
        if in_suffix not in filename: continue
        parts = filename.split(in_suffix)
        
        base_scene = parts[0]
        coord_id = parts[1]
        
        expected_name = f"{base_scene}{tgt_suffix}{coord_id}"
        target_path = os.path.join(target_dir, expected_name)
        
        if os.path.exists(target_path):
            paired_in.append(f)
            paired_tgt.append(target_path)

    print(f"--- Alignment Metrics for {EXPERIMENT} ---")
    print(f"Total matched pairs found on disk: {len(paired_in)}")
    
    tr_in, te_in, tr_tg, te_tg = train_test_split(paired_in, paired_tgt, test_size=0.30, random_state=42)
    va_in, ts_in, va_tg, ts_tg = train_test_split(te_in, te_tg, test_size=0.50, random_state=42)
    print(f"Split results -> Train: {len(tr_in)} | Val: {len(va_in)} | Test: {len(ts_in)}")
    return (tr_in, tr_tg), (va_in, va_tg), (ts_in, ts_tg)

# =====================================================================
# 3. NEURAL NETWORK ARCHITECTURES
# =====================================================================
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetStandard(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.bottleneck(self.pool3(s3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.final(d1)

class UNetUpscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial_upscale = nn.ConvTranspose2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.base_unet = UNetStandard(in_channels=32, out_channels=out_channels)
    def forward(self, x):
        return self.base_unet(self.initial_upscale(x))

# =====================================================================
# 4. TRAINING ENGINE EXECUTION WITH ACCUMULATORS
# =====================================================================
if __name__ == "__main__":
    train_paths, val_paths, _ = get_splits()
    
    print("\n--- Initializing Train Loader ---")
    train_loader = DataLoader(LoaderDataset(*train_paths), batch_size=BATCH_SIZE, shuffle=True)
    
    print("\n--- Initializing Validation Loader ---")
    val_loader = DataLoader(LoaderDataset(*val_paths), batch_size=BATCH_SIZE, shuffle=False)
    
    if EXPERIMENT == "COLORIZE":
        model = UNetStandard(in_channels=1, out_channels=3).to(DEVICE)
    else:
        model = UNetUpscale(in_channels=1, out_channels=1).to(DEVICE)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')
    
    # Simple arrays to save performance arrays for post-processing analysis later
    history_train_loss = []
    history_val_loss = []

    print(f"\nModel bound to hardware. Starting {EPOCHS} Epoch loops...")
    sys.stdout.flush()
    
    for epoch in range(EPOCHS):
        # --- TRAINING CYCLE ---
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type):
                loss = criterion(model(x), y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item() * x.size(0)
            
        mean_train_mse = t_loss / len(train_loader.dataset)
        history_train_loss.append(mean_train_mse)

        # --- VALIDATION CYCLE ---
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type):
                    loss = criterion(model(x), y)
                v_loss += loss.item() * x.size(0)
                
        mean_val_mse = v_loss / len(val_loader.dataset)
        history_val_loss.append(mean_val_mse)
            
        # Clean, decoupled error logging output
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train MSE: {mean_train_mse:.6f} | Val MSE: {mean_val_mse:.6f}")
        sys.stdout.flush()
    
    # Save network states cleanly
    torch.save(model.state_dict(), f"phase1_{EXPERIMENT}.pth")
    print("\nTraining finished. Weights stored successfully.")