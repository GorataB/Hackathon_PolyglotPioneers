"""
Central configuration for the forecasting project.

Only project settings belong here.
Anything that can be discovered from the dataset
should NOT be hard-coded.
"""

from pathlib import Path

# ----------------------------------------------------
# Project Directories
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "Data_Raw"

DATA_CLEAN_DIR = PROJECT_ROOT / "Data_Clean"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

LOGS_DIR = PROJECT_ROOT / "logs"

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

DATASET_NAME = "prediction_data.csv"

DATASET_PATH = DATA_CLEAN_DIR / DATASET_NAME

TARGET_COLUMN = "FAO_23014"

DATE_COLUMNS = [
    "date",
    "year_month",
]

NON_FEATURE_COLUMNS = [
    "first",
    "last",
]

# ----------------------------------------------------
# Data Split
# ----------------------------------------------------

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15

# ----------------------------------------------------
# Sequence
# ----------------------------------------------------

SEQUENCE_LENGTH = 12

# ----------------------------------------------------
# Training
# ----------------------------------------------------

BATCH_SIZE = 32

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

EPOCHS = 100

PATIENCE = 15

GRADIENT_CLIP = 1.0

# ----------------------------------------------------
# Model
# ----------------------------------------------------

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.20

# ----------------------------------------------------
# Reproducibility
# ----------------------------------------------------

RANDOM_SEED = 42

# ----------------------------------------------------
# Artifacts
# ----------------------------------------------------

FEATURE_SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"

TARGET_SCALER_FILE = MODELS_DIR / "target_scaler.joblib"

FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.joblib"

METADATA_FILE = MODELS_DIR / "metadata.json"

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

BEST_MODEL_FILE = CHECKPOINT_DIR / "best_lstm_model.pth"

LAST_CHECKPOINT_FILE = CHECKPOINT_DIR / "last_checkpoint.pth"

TRAINING_HISTORY_FILE = MODELS_DIR / "training_history.json"

TENSORBOARD_DIR = PROJECT_ROOT / "runs"
# ----------------------------------------------------
# Ensure Required Directories Exist
# ----------------------------------------------------

for directory in (
    DATA_CLEAN_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    CHECKPOINT_DIR,
    TENSORBOARD_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

DEVICE = "auto"
MODEL_NAME = "LSTM Forecasting"