from pathlib import Path
import torch

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
VAL_FRACTION = 0.20

CLS_WEIGHT = 1.0
SEG_WEIGHT = 1.0

# ReduceLROnPlateau settings
LR_FACTOR = 0.5
LR_PATIENCE = 2
LR_MIN = 1e-5
LR_THRESHOLD = 1e-3

DATA_ROOT = Path("./data")
OUTPUT_ROOT = Path("./outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
