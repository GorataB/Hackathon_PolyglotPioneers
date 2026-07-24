"""
logger.py

Central logging utility for the forecasting project.

Features
--------
- Console logging
- File logging
- Automatic log directory creation
- Prevents duplicate handlers
- Shared logger across all modules
"""

import logging
from pathlib import Path

from src.config.config import LOGS_DIR

# ----------------------------------------------------
# Create logs directory
# ----------------------------------------------------

LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "forecasting_pipeline.log"

# ----------------------------------------------------
# Logger Factory
# ----------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a project logger.

    Parameters
    ----------
    name : str
        Usually __name__.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -----------------------------
    # Console Handler
    # -----------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    # -----------------------------
    # File Handler
    # -----------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    # -----------------------------
    # Attach handlers
    # -----------------------------

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger