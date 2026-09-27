import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import rasterio

# Pipeline Infrastructure Imports
import config
from train_test_split import build_pipeline_splits, SatellitePatchDataset
from metrics import calculate_psnr, calculate_batch_ssim , calculate_batch_fid
from models import AttentionUNetStandard, AttentionUNetUpscale, ResNet34UNet, SwinIRLight, UNetStandard, UNetUpscale

def instantiate_model():
    out_c = 3 if config.EXPERIMENT == "COLORIZE" else 1
    is_color = (config.EXPERIMENT == "COLORIZE")
    is_superres = (config.EXPERIMENT == "SUPER_RES") # Add this flag!
    
    if config.MODEL_TYPE == "UNET":
        if config.EXPERIMENT == "COLORIZE":
            return UNetStandard(in_channels=1, out_channels=out_c).to(config.DEVICE)
        return UNetUpscale(in_channels=1, out_channels=out_c).to(config.DEVICE)
        
    elif config.MODEL_TYPE == "ATTENTION_UNET":
        if is_color: 
            return AttentionUNetStandard(in_channels=1, out_channels=out_c).to(config.DEVICE)
        return AttentionUNetUpscale(in_channels=1, out_channels=out_c).to(config.DEVICE)
        
    elif config.MODEL_TYPE == "RESNET_UNET":
        # ---> FIX THIS LINE HERE <---
        # Super-res needs pre_upscale=True, Colorize needs pre_upscale=False
        return ResNet34UNet(in_channels=1, out_channels=out_c, pre_upscale=is_superres).to(config.DEVICE)
        
    elif config.MODEL_TYPE == "SWIN_IR":
        return SwinIRLight(in_channels=1, out_channels=out_c, pre_upscale=is_color).to(config.DEVICE)
        
    else:
        raise ValueError(f"Unknown architecture selector token: {config.MODEL_TYPE}")
if __name__ == "__main__":
    print(f"Initializing Engine | Task: {config.EXPERIMENT} | Architecture: {config.MODEL_TYPE}")
    
    # 1. Dataset Generation Stage
    train_paths, val_paths, test_paths = build_pipeline_splits(config.BASE_DIR, config.EXPERIMENT)
    train_loader = DataLoader(SatellitePatchDataset(*train_paths), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SatellitePatchDataset(*val_paths), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SatellitePatchDataset(*test_paths), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Model Blueprint Injection
    model = instantiate_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda' if config.DEVICE.type == 'cuda' else 'cpu')
    
    best_val_loss = float('inf')
    
    # Open our logs channel inside outputs/logs/
    with open(config.TEXT_LOG_PATH, "w") as log_file:
        log_file.write(f"Timestamped Training Execution for {config.MODEL_TYPE} - {config.EXPERIMENT}\n")
        log_file.write("Epoch,Train_MSE,Val_MSE\n")
        
        # 3. Training and Validation Engine Loop
        print(f"Beginning optimization loop ({config.EPOCHS} Epochs)...")
        for epoch in range(config.EPOCHS):
            model.train()
            t_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=config.DEVICE.type):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                t_loss += loss.item() * x.size(0)
                
            mean_train_loss = t_loss / len(train_loader.dataset)
            
            # Validation Step
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        loss = criterion(model(x), y)
                    v_loss += loss.item() * x.size(0)
            
            mean_val_loss = v_loss / len(val_loader.dataset)
            
            # Save stats to log file
            log_file.write(f"{epoch+1},{mean_train_loss:.6f},{mean_val_loss:.6f}\n")
            print(f"Epoch {epoch+1:02d}/{config.EPOCHS:02d} | Train MSE: {mean_train_loss:.6f} | Val MSE: {mean_val_loss:.6f}")
            
            # CHECKPOINT RULE: Save weights to best path if validation improves
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model.state_dict(), config.BEST_WEIGHTS_PATH)
                print(f" --> [SAVED BEST] Checkpoint stored at {config.BEST_WEIGHTS_PATH}")
                
            sys.stdout.flush()

    # Save the absolute final epoch state to final path
    torch.save(model.state_dict(), config.FINAL_WEIGHTS_PATH)
    print(f"[COMPLETED TRAINING] Final operational parameters archived at {config.FINAL_WEIGHTS_PATH}")

    # 4. Independent Holdout Test Set Phase Running Natively
    print("\n--- Deploying Best Performing Weights onto Unseen Test Set ---")
    model.load_state_dict(torch.load(config.BEST_WEIGHTS_PATH, map_location=config.DEVICE))
    model.eval()
    
    test_loss, test_ssim = 0.0, 0.0
    inference_latencies = []
    
    # Initialize FID metric if running Colorization
    if config.EXPERIMENT == "COLORIZE":
        from torchmetrics.image.fid import FrechetInceptionDistance
        # Using a 64-feature embedding ensures fast calculation on satellite patches
        fid_metric = FrechetInceptionDistance(feature=64).to(config.DEVICE)
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            start = time.time()
            with torch.amp.autocast(device_type=config.DEVICE.type):
                outputs = model(x)
            if config.DEVICE.type == 'cuda': torch.cuda.synchronize()
            inference_latencies.append((time.time() - start) / x.size(0))
            
            test_loss += criterion(outputs, y).item() * x.size(0)
            test_ssim += calculate_batch_ssim(outputs, y) * x.size(0)
            
            # Accumulate FID features per batch for Colorization
            if config.EXPERIMENT == "COLORIZE":
                preds_uint8 = torch.clamp(outputs * 255, 0, 255).to(torch.uint8)
                targets_uint8 = torch.clamp(y * 255, 0, 255).to(torch.uint8)
                fid_metric.update(targets_uint8, real=True)
                fid_metric.update(preds_uint8, real=False)

    final_test_mse = test_loss / len(test_loader.dataset)
    final_test_ssim = test_ssim / len(test_loader.dataset)
    final_test_psnr = calculate_psnr(final_test_mse)
    
    # Compute final global FID score
    final_test_fid = 0.0
    if config.EXPERIMENT == "COLORIZE":
        final_test_fid = fid_metric.compute().item()
    
    # --- PRINT OUT TO TERMINAL ---
    print(f"\n=================== FINAL METRICS BENCHMARK CARD ===================")
    print(f" Experiment Track:                 {config.EXPERIMENT}")
    print(f" Evaluated Engine Backbone:        {config.MODEL_TYPE}")
    print(f" Unseen Test Mean Squared Error:   {final_test_mse:.6f}") 
    print(f" Target Restoration PSNR Score:    {final_test_psnr:.2f} dB")
    print(f" Structural Consistency (SSIM):    {final_test_ssim * 100:.2f}%")
    if config.EXPERIMENT == "COLORIZE":
        print(f" Fréchet Inception Distance (FID): {final_test_fid:.2f}") # <-- FID terminal printout
    print(f" Hardware Frame Compute Latency:   {torch.tensor(inference_latencies).mean()*1000:.2f} ms")
    print(f"====================================================================")
    import matplotlib.pyplot as plt
    import pandas as pd
    def generate_and_save_plots():
        print("\n--- Generating Performance Visualization Graphs ---")
        if not os.path.exists(config.TEXT_LOG_PATH):
            print(f"[WARNING] Log file not found at {config.TEXT_LOG_PATH}. Skipping plots.")
            return
            
        # Fix: Read only the lines belonging to the 15 epochs to bypass appended test logs
        try:
            data = pd.read_csv(config.TEXT_LOG_PATH, nrows=15) # Restricts parsing to training epochs
            data.columns = data.columns.str.strip()
            
            epochs = data.iloc[:, 0]    
            train_mse = data.iloc[:, 1] 
            val_mse = data.iloc[:, 2]   
        except Exception as e:
            print(f"[ERROR] Pandas failed to read training rows safely: {e}")
            return

        # 1. Plot Standard Training vs Validation Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_mse, label='Train MSE', color='blue', linewidth=2)
        plt.plot(epochs, val_mse, label='Validation MSE', color='red', linestyle='--', linewidth=2)
        plt.title(f'{config.MODEL_TYPE} - Training vs Validation Performance ({config.EXPERIMENT})')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        loss_curve_path = os.path.join(config.FIGURES_DIR, f"{config.MODEL_PREFIX}_loss_curve.png")
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED GRAPH] Loss curve saved to: {loss_curve_path}")
        
        # 2. Plot the GAP/DIFFERENCE Curve
        plt.figure(figsize=(10, 5))
        loss_difference = train_mse - val_mse
        plt.plot(epochs, loss_difference, label='Train-Val Gap', color='purple', linewidth=2)
        plt.axhline(0, color='black', linestyle=':', alpha=0.5)
        plt.title(f'{config.MODEL_TYPE} - Generalization Gap over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Delta Loss (Train - Val)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        gap_curve_path = os.path.join(config.FIGURES_DIR, f"{config.MODEL_PREFIX}_generalization_gap.png")
        plt.savefig(gap_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED GRAPH] Gap analysis curve saved to: {gap_curve_path}")


    generate_and_save_plots() 
    
    # --- APPEND TEST RESULTS TO LOG FILE ---
    try:
        with open(config.TEXT_LOG_PATH, "a") as log_file:
            log_file.write("\n--- Holdout Test Set Final Results ---\n")
            log_file.write(f"Test_MSE,Test_PSNR,Test_SSIM,Test_FID\n")
            log_file.write(f"{final_test_mse:.6f},{final_test_psnr:.2f},{final_test_ssim * 100:.2f},{final_test_fid:.2f}\n")
        print(f"[LOG UPDATED] Test metrics appended successfully to {config.TEXT_LOG_PATH}")
    except Exception as log_error:
        print(f"[WARNING] Could not append test metrics to log file: {log_error}")



