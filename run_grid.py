import os
import sys
import subprocess
import time

# Get the absolute directory where run_grid.py itself lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def update_config_file(experiment, model_type, lr, batch_size, epochs, weight_decay):
    """
    Reads config.py, updates the core task and model types, and explicitly injects 
    the active hyperparameter combination.
    """
    config_path = os.path.join(SCRIPT_DIR, "config.py")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not locate config.py at {config_path}")
        
    with open(config_path, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EXPERIMENT ="):
            new_lines.append(f'EXPERIMENT = "{experiment}"\n')
        elif stripped.startswith("MODEL_TYPE ="):
            new_lines.append(f'MODEL_TYPE = "{model_type}"\n')
        elif stripped.startswith("LEARNING_RATE ="):
            new_lines.append(f'LEARNING_RATE = {lr}\n')
        elif stripped.startswith("BATCH_SIZE ="):
            new_lines.append(f'BATCH_SIZE = {batch_size}\n')
        elif stripped.startswith("EPOCHS ="):
            new_lines.append(f'EPOCHS = {epochs}\n')
        elif stripped.startswith("WEIGHT_DECAY ="):
            new_lines.append(f'WEIGHT_DECAY = {weight_decay}\n')
        else:
            new_lines.append(line)
            
    with open(config_path, "w") as f:
        f.writelines(new_lines)

def main():
    # Target Scope: Strictly Colorization on ResNet-UNet
    target_experiment = "COLORIZE"
    target_model = "RESNET_UNET"
    
    # Hyperparameter Grid Arrays
    learning_rates = [1e-4, 5e-4, 3e-4]
    batch_sizes = [4, 8, 16]
    epochs_options = [30, 50]
    weight_decays = [0, 1e-5, 1e-4]
    
    # Calculate complete space dimensions
    total_runs = len(learning_rates) * len(batch_sizes) * len(epochs_options) * len(weight_decays)
    
    print(f"=========================================================================")
    print(f"LAUNCHING TARGETED HYPERPARAMETER SEARCH GRID")
    print(f"Task Focus:   {target_experiment}")
    print(f"Architecture: {target_model}")
    print(f"Total Combinations to Sweep: {total_runs} Independent Training Runs")
    print(f"=========================================================================")
    
    run_counter = 1
    start_time = time.time()
    main_py_path = os.path.join(SCRIPT_DIR, "main.py")
    
    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in epochs_options:
                for wd in weight_decays:
                    print(f"\n" + "-"*75)
                    print(f"Executing Search Combination {run_counter}/{total_runs}")
                    print(f"LR: {lr} | Batch Size: {bs} | Epochs: {epochs} | Weight Decay: {wd}")
                    print("-"*75)
                    
                    # 1. Dynamically rewrite configuration values on disk
                    try:
                        update_config_file(target_experiment, target_model, lr, bs, epochs, wd)
                        print(f"[CONFIG REWRITTEN] Settings successfully locked in config.py")
                    except Exception as e:
                        print(f"[ERROR] Failed to update config values: {e}")
                        continue
                    
                    # 2. Invoke the untouched main.py pipeline inside its own sandbox process
                    print(f"Spawning native pipeline training instance...")
                    sys.stdout.flush()
                    
                    process = subprocess.run(
                        [sys.executable, main_py_path],
                        cwd=SCRIPT_DIR,
                        capture_output=False,
                        text=True
                    )
                    
                    if process.returncode == 0:
                        print(f"[SUCCESS] Multi-epoch trial concluded cleanly.")
                    else:
                        print(f"[CRITICAL FAILURE] Trial instance crashed with exit code: {process.returncode}")
                        
                    run_counter += 1
                    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n" + "#"*80)
    print(f"GRID SEARCH COMPLETE. Processed {total_runs} combinations in {elapsed_minutes:.2f} minutes.")
    print(f"Inspect your logs and generated figures directories to extract your optimal hyperparameters!")
    print("#"*80)

if __name__ == "__main__":
    main()