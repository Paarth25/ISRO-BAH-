import os
import torch

# Central Dataset Directory Path
BASE_DIR = r"C:\Users\vegeta\Desktop\isro\ps10\dataset"

# Pipeline Swappers
EXPERIMENT = "COLORIZE"
MODEL_TYPE = "RESNET_UNET"

# Core Hyperparameters
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- EXPANDED FOLDER ROUTING SYSTEM ---
# 1. Weights Directories
WEIGHTS_SUBDIR = os.path.join("weights", EXPERIMENT.lower())
os.makedirs(WEIGHTS_SUBDIR, exist_ok=True)

MODEL_PREFIX = MODEL_TYPE.lower().replace("_unet", "") # converts "RESNET_UNET" -> "resnet"
BEST_WEIGHTS_PATH = os.path.join(WEIGHTS_SUBDIR, f"{MODEL_PREFIX}_best.pth")
FINAL_WEIGHTS_PATH = os.path.join(WEIGHTS_SUBDIR, f"{MODEL_PREFIX}_final.pth")

# 2. Outputs Directories
OUTPUTS_DIR = "outputs"
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

for folder in [PREDICTIONS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Path to log files
TEXT_LOG_PATH = os.path.join(LOGS_DIR, f"{EXPERIMENT.lower()}_{MODEL_PREFIX}_training.log")