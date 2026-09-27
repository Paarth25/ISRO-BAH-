import os
import sys
import subprocess
import time

# Get the script's absolute home directory directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def rewrite_config_for_superres(model_architecture):
    """
    Overwrites configuration properties safely by placing core task locks
    at the VERY TOP of the file so they are declared before paths reference them.
    """
    config_path = os.path.join(SCRIPT_DIR, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.py within structural runtime layout context: {config_path}")
        
    with open(config_path, "r") as f:
        lines = f.readlines()
        
    # Clean out any old/existing declarations of these variables anywhere in the file
    target_skips = ["EXPERIMENT =", "MODEL_TYPE =", "EPOCHS =", "BATCH_SIZE =", "LEARNING_RATE =", "# --- SUPER RESOLUTION GRID"]
    cleaned_lines = [line for line in lines if not any(line.strip().startswith(k) for k in target_skips)]
        
    # Build the header block containing the fresh grid properties
    fresh_header = [
        "# --- SUPER RESOLUTION GRID SEARCH GENERATED ASSIGNMENTS ---\n",
        'EXPERIMENT = "SUPER_RES"\n',
        f'MODEL_TYPE = "{model_architecture}"\n',
        'EPOCHS = 20\n',
        'BATCH_SIZE = 8\n',
        'LEARNING_RATE = 2e-4\n',
        "# ---------------------------------------------------------\n\n"
    ]
    
    # Combine header first, then the remaining config logic (imports, paths, etc.)
    final_lines = fresh_header + cleaned_lines
    
    with open(config_path, "w") as f:
        f.writelines(final_lines)
def main():
    # Targeted comparison spectrum
    candidate_models = ["SRRESNET", "RESNET_UNET", "SWIN_IR"]
    
    print("="*80)
    print("        LAUNCHING METRIC GRID RUNNER: SUPER-RESOLUTION BENCHMARKS        ")
    print(f"Candidate Evaluation Queue: {candidate_models}")
    print("="*80)
    
    start_wall_time = time.time()
    main_py_path = os.path.join(SCRIPT_DIR, "main.py")
    
    for idx, model_name in enumerate(candidate_models, start=1):
        print(f"\n[{idx}/{len(candidate_models)}] Testing Architecture Target: {model_name}")
        print("-" * 60)
        
        # 1. Update system parameters config layout
        try:
            rewrite_config_for_superres(model_name)
            print(f"[STAGE-01] config.py locked on SUPER_RES + {model_name}")
        except Exception as err:
            print(f"[FATAL CONFIG CONFIGURATION ASSIGNMENT BLOCKED]: {err}")
            continue
            
        # 2. Run pipeline execution process sequence
        print(f"[STAGE-02] Forking execution loop context...")
        sys.stdout.flush()
        
        process_handle = subprocess.run(
            [sys.executable, main_py_path],
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True
        )
        
        if process_handle.returncode == 0:
            print(f"[SUCCESS] {model_name} processing routine finished normally.")
        else:
            print(f"[CRITICAL ABORT] Variant thread crashed for structure: {model_name}. Exit Code: {process_handle.returncode}")
            
    total_elapsed = (time.time() - start_wall_time) / 60
    print("\n" + "#"*80)
    print(f"GRID CONCLUDED SUCCESSFULLY. Super-Resolution architectures swept in {total_elapsed:.2f} mins.")
    print("Review generated figures directory layouts to pick the ultimate model track champion!")
    print("#"*80)

if __name__ == "__main__":
    main()