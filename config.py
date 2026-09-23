"""Central configuration for the satellite change-detection dashboard."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import torch

ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

for folder in (MODELS_DIR, OUTPUTS_DIR, DATA_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------
# Your new checkpoint is directly inside DL/
CHECKPOINT_PATH = ROOT / "change_detector.pth"

# Fallback if you still keep a copy inside models/
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = MODELS_DIR / "change_detector.pth"

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
TEST_METRICS_JSON = OUTPUTS_DIR / "test_metrics.json"
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------
# GOOGLE EARTH ENGINE
# ---------------------------------------------------------
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "")
GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT", "")
GEE_PRIVATE_KEY_FILE = os.getenv("GEE_PRIVATE_KEY_FILE", "")

# ---------------------------------------------------------
# SENTINEL-2
# ---------------------------------------------------------
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

RGB_BANDS = [
    "B4",
    "B3",
    "B2",
]

NIR_BAND = "B8"

NATIVE_RESOLUTION_M = 10

DEFAULT_MAX_CLOUD = 20

DEFAULT_SEARCH_WINDOW_DAYS = 30

THUMB_SIZE = 512

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256

MODEL_CONFIG = {
    "architecture": "SiameseUNet",
    "in_channels": 3,
    "base_channels": 32,
    "img_size": 256,
}

# ---------------------------------------------------------
# CHANGE DETECTION
# ---------------------------------------------------------
#
# Start at 0.10 because your previous 0.50 threshold
# produced a completely black prediction.
#
# This DOES NOT modify or retrain the model.
#
DEFAULT_THRESHOLD = 0.10

MIN_REGION_PIXELS = 4

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------
NOMINATIM_USER_AGENT = (
    "india-change-detection-dashboard"
)