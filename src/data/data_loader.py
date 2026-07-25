"""
data_loader.py

Loads and validates the cleaned forecasting dataset.

Responsibilities
----------------
- Load cleaned dataset
- Validate required columns
- Inspect dataset quality
- Sort chronologically
- Dynamically discover feature columns
"""

from pathlib import Path

import pandas as pd

from src.config.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    DATE_COLUMNS,
    NON_FEATURE_COLUMNS,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and validates the cleaned modelling dataset.
    """

    def __init__(self, dataset_path: Path = DATASET_PATH):

        self.dataset_path = Path(dataset_path)

        self.data = None

        self.feature_columns = None

    # ----------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the cleaned dataset.
        """

        logger.info("Loading dataset...")

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        logger.info(
            f"Dataset loaded successfully "
            f"({df.shape[0]} rows × {df.shape[1]} columns)"
        )

        self.data = df

        return df

    # ----------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------

    def validate(self) -> None:
        """
        Validate dataset integrity.
        """

        if self.data is None:
            raise RuntimeError("Dataset has not been loaded.")

        logger.info("Validating dataset...")

        # Target column
        if TARGET_COLUMN not in self.data.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' is missing."
            )

        # Duplicate rows
        duplicate_rows = self.data.duplicated().sum()

        if duplicate_rows > 0:

            logger.warning(
                f"Found {duplicate_rows} duplicate rows."
            )

        # Missing values
        missing = self.data.isnull().sum()

        missing = missing[missing > 0]

        if len(missing):

            logger.warning(
                "Missing values detected:"
            )

            for column, value in missing.items():

                logger.warning(
                    f"{column}: {value}"
                )

        else:

            logger.info("No missing values detected.")

    # ----------------------------------------------------
    # SORT DATA
    # ----------------------------------------------------

    def sort(self) -> None:
        """
        Sort dataset chronologically if a date column exists.
        """

        if self.data is None:
            raise RuntimeError("Dataset has not been loaded.")

        for column in DATE_COLUMNS:

            if column in self.data.columns:

                logger.info(
                    f"Sorting by '{column}'."
                )

                self.data = self.data.sort_values(
                    column
                ).reset_index(drop=True)

                return

        logger.warning(
            "No recognised date column found."
        )

    # ----------------------------------------------------
    # FEATURE DISCOVERY
    # ----------------------------------------------------

        # ----------------------------------------------------
    # FEATURE DISCOVERY
    # ----------------------------------------------------

    def discover_features(self):
        """
        Automatically determine model input features.

        Excludes:
        - date columns
        - non-feature columns
        - target column
        - FAO predictor columns

        This ensures the LSTM uses the same exogenous variables
        as the ARIMAX model for a fair comparison.
        """

        if self.data is None:
            raise RuntimeError("Dataset has not been loaded.")

        excluded = (
            set(DATE_COLUMNS)
            | set(NON_FEATURE_COLUMNS)
            | {TARGET_COLUMN}
            | {"FAO_23012", "FAO_23013"}
        )

        self.feature_columns = [

            column

            for column in self.data.columns

            if column not in excluded

        ]

        logger.info(
            f"Feature columns: {self.feature_columns}"
        )

        return self.feature_columns

    # ----------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------

    def summary(self):
        """
        Log a summary of the dataset.
        """

        logger.info("=" * 60)

        logger.info("Dataset Summary")

        logger.info("=" * 60)

        logger.info(f"Rows      : {len(self.data)}")

        logger.info(f"Columns   : {len(self.data.columns)}")

        logger.info(
            f"Features  : {len(self.feature_columns)}"
        )

        logger.info(f"Target    : {TARGET_COLUMN}")

        logger.info("=" * 60)

    # ----------------------------------------------------
    # COMPLETE PIPELINE
    # ----------------------------------------------------

    def run(self):
        """
        Execute the complete loading pipeline.
        """

        self.load()

        self.validate()

        self.sort()

        self.discover_features()

        self.summary()

        return self.data.copy(), self.feature_columns