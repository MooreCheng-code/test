"""Global configuration for bankruptcy prediction project."""

from pathlib import Path

# Paths (relative to this config file's directory)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
for directory in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed
SEED = 42

# Data Split & Cross-Validation
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3  # Can be adjusted to 5 if computation time allows

# Evaluation Metrics Parameters
TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]
ALERT_RATIOS = [1.5, 3, 5, 10]  # Used to report recall at given alert multiplier
DEFAULT_TARGET_RECALL = 0.95
TARGET_COL = "Bankrupt?"

# Feature Selection (None indicates using all features; can be replaced by core feature list)
FEATURES = None

# MVP Evaluation & Operational Constants
TOP_K_PERCENT = 0.10
MIN_BATCH_SIZE_FOR_TOP_K = 25
COST_FN = 20.0
COST_FP = 1.0
COST_FN_SENSITIVITY_LIST = [10.0, 20.0, 50.0]
MAX_MISSING_FEATURE_RATIO = 0.20
MVP_PASS_PR_AUC_MULTIPLIER = 3.0
MVP_PASS_TOP10_RECALL = 0.50

