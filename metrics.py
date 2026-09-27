import numpy as np
import torch

def calculate_psnr(mse):
    if mse == 0: return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_batch_ssim(preds, targets):
    p = torch.clamp(preds, 0.0, 1.0).detach().cpu().numpy()
    t = torch.clamp(targets, 0.0, 1.0).detach().cpu().numpy()
    ssim_scores = []
    for b in range(p.shape[0]):
        scores_per_channel = []
        for c in range(p.shape[1]):
            img1, img2 = p[b, c], t[b, c]
            mu1, mu2 = img1.mean(), img2.mean()
            sigma1, sigma2 = img1.var(), img2.var()
            covariance = np.mean((img1 - mu1) * (img2 - mu2))
            
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim_num = (2 * mu1 * mu2 + C1) * (2 * covariance + C2)
            ssim_den = (sigma1 + sigma2 + C2) * (mu1**2 + mu2**2 + C1)
            
            if ssim_den > 1e-12:
                scores_per_channel.append(ssim_num / ssim_den)
            else:
                scores_per_channel.append(1.0 if np.allclose(img1, img2, atol=1e-4) else 0.0)
        ssim_scores.append(np.mean(scores_per_channel))
    return np.mean(ssim_scores)

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_batch_fid(pred_tensors, target_tensors, device):
    """
    Computes FID score for a batch of predictions and targets.
    Expects tensors in range [0, 1] with shape [B, 3, H, W].
    """
    # Force check channels - FID requires exactly 3 channels (RGB)
    if pred_tensors.shape[1] != 3 or target_tensors.shape[1] != 3:
        return 0.0 # Return fallback 0 if called during grayscale SUPER_RES
        
    # Initialize FID metric on the active device
    # Using a minor feature dimension reduction (64) keeps it incredibly fast and lightweight
    fid = FrechetInceptionDistance(feature=64).to(device)
    
    # Scale float [0, 1] tensors to uint8 [0, 255] images
    preds_uint8 = torch.clamp(pred_tensors * 255, 0, 255).to(torch.uint8)
    targets_uint8 = torch.clamp(target_tensors * 255, 0, 255).to(torch.uint8)
    
    # Update real and fake states
    fid.update(targets_uint8, real=True)
    fid.update(preds_uint8, real=False)
    
    # Compute final score
    fid_score = fid.compute().item()
    return fid_score