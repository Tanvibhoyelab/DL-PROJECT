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
RESOURCES_DIR = ROOT / "resources"
CITY_CACHE_DIR = DATA_DIR / "city_cache"

for folder in (MODELS_DIR, OUTPUTS_DIR, DATA_DIR, REPORTS_DIR, CITY_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------
CHECKPOINT_PATH = ROOT / "change_detector.pth"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = MODELS_DIR / "change_detector.pth"

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
TEST_METRICS_JSON = OUTPUTS_DIR / "test_metrics.json"
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / "confusion_matrix.png"
QUALITATIVE_PNG = OUTPUTS_DIR / "qualitative_examples.png"

# ---------------------------------------------------------
# LOCATIONS
# ---------------------------------------------------------
INDIA_DISTRICTS_JSON = RESOURCES_DIR / "india_districts.json"
NOMINATIM_USER_AGENT = "india-change-detection-dashboard"

MIN_AOI_RADIUS_KM = 1.0
MAX_AOI_RADIUS_KM = 20.0
DEFAULT_AOI_RADIUS_KM = 3.0

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
S2_CLOUD_PROBABILITY_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
S2_START_DATE = "2017-03-28"

RGB_BANDS = ["B4", "B3", "B2"]
NIR_BAND = "B8"

NATIVE_RESOLUTION_M = 10
DEFAULT_MAX_CLOUD = 20
DEFAULT_SEARCH_WINDOW_DAYS = 45
THUMB_SIZE = 512

# Surface-reflectance stretch used to turn Sentinel-2 L2A into an 8-bit
# true-colour image. Both dates always use the same stretch, otherwise the
# before/after pair would differ for purely cosmetic reasons.
S2_VIS_MIN = 0
S2_VIS_MAX = 3000
S2_VIS_GAMMA = 1.1

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 256

MODEL_CONFIG = {
    "architecture": "SiameseUNet",
    "in_channels": 3,
    "base_channels": 32,
    "img_size": IMAGE_SIZE,
}

# ---------------------------------------------------------
# CHANGE DETECTION
# ---------------------------------------------------------
# Decision threshold on the change probability. Training tunes this on the
# validation split and stores the tuned value in the checkpoint; this is only
# the fallback used when a checkpoint carries no tuned threshold.
DEFAULT_THRESHOLD = 0.5

MIN_REGION_PIXELS = 12

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
