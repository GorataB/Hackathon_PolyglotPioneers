"""
preprocessor.py

Leakage-free preprocessing pipeline for the forecasting project.

Responsibilities
----------------
1. Split the cleaned dataset chronologically
2. Fit feature and target scalers using TRAINING data only
3. Transform validation and test data using the fitted scalers
4. Save preprocessing artifacts
5. Save preprocessing metadata

This module intentionally DOES NOT:
- load datasets
- create LSTM sequences
- train models

Author: Polyglot Pioneers
"""

from __future__ import annotations
from src.data.data_types import ProcessedData

import json

import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

from src.config.config import (
    FEATURE_NAMES_FILE,
    FEATURE_SCALER_FILE,
    METADATA_FILE,
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
    TARGET_SCALER_FILE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------------


class Preprocessor:
    """
    Performs leakage-free preprocessing.
    """

    def __init__(self):

        self.feature_scaler = StandardScaler()

        self.target_scaler = StandardScaler()

    # --------------------------------------------------------------

    def split_dataset(
        self,
        dataframe,
        feature_columns,
    ):
        """
        Split the dataframe chronologically.
        """

        logger.info("Splitting dataset chronologically...")

        if round(
            TRAIN_RATIO
            + VALIDATION_RATIO
            + TEST_RATIO,
            2,
        ) != 1.00:

            raise ValueError(
                "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1."
            )

        total_rows = len(dataframe)

        train_end = int(total_rows * TRAIN_RATIO)

        validation_end = (
            train_end
            + int(total_rows * VALIDATION_RATIO)
        )

        train_df = dataframe.iloc[:train_end].copy()

        validation_df = dataframe.iloc[
            train_end:validation_end
        ].copy()

        test_df = dataframe.iloc[
            validation_end:
        ].copy()

        logger.info(
            f"Train rows      : {len(train_df)}"
        )

        logger.info(
            f"Validation rows : {len(validation_df)}"
        )

        logger.info(
            f"Test rows       : {len(test_df)}"
        )

        return (
            train_df,
            validation_df,
            test_df,
        )

    # --------------------------------------------------------------

    def fit_scalers(
        self,
        train_df,
        feature_columns,
    ):
        """
        Fit scalers ONLY using training data.
        """

        logger.info(
            "Fitting feature scaler on training data..."
        )

        self.feature_scaler.fit(
            train_df[feature_columns]
        )

        logger.info(
            "Fitting target scaler on training data..."
        )

        self.target_scaler.fit(
            train_df[[TARGET_COLUMN]]
        )

    # --------------------------------------------------------------

    def transform(
        self,
        dataframe,
        feature_columns,
    ):
        """
        Transform one dataset partition.
        """

        X = self.feature_scaler.transform(
            dataframe[feature_columns]
        )

        y = self.target_scaler.transform(
            dataframe[[TARGET_COLUMN]]
        ).flatten()

        return (
            X.astype(np.float32),
            y.astype(np.float32),
        )

    # --------------------------------------------------------------

    def save_artifacts(
        self,
        feature_columns,
    ):
        """
        Save preprocessing artifacts.
        """

        logger.info(
            "Saving preprocessing artifacts..."
        )

        dump(
            self.feature_scaler,
            FEATURE_SCALER_FILE,
        )

        dump(
            self.target_scaler,
            TARGET_SCALER_FILE,
        )

        dump(
            feature_columns,
            FEATURE_NAMES_FILE,
        )

    # --------------------------------------------------------------

    def save_metadata(
        self,
        processed_data: ProcessedData,
    ):
        """
        Save metadata required for inference.
        """

        metadata = {

            "target_column": TARGET_COLUMN,

            "sequence_length": SEQUENCE_LENGTH,

            "feature_count":
                len(processed_data.feature_names),

            "feature_names":
                processed_data.feature_names,

            "train_rows":
                len(processed_data.X_train),

            "validation_rows":
                len(processed_data.X_validation),

            "test_rows":
                len(processed_data.X_test),

        }

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        logger.info("Metadata saved.")

    # --------------------------------------------------------------

    def run(
        self,
        dataframe,
        feature_columns,
    ) -> ProcessedData:
        """
        Execute the preprocessing pipeline.
        """

        (
            train_df,
            validation_df,
            test_df,
        ) = self.split_dataset(
            dataframe,
            feature_columns,
        )

        self.fit_scalers(
            train_df,
            feature_columns,
        )

        X_train, y_train = self.transform(
            train_df,
            feature_columns,
        )

        X_validation, y_validation = self.transform(
            validation_df,
            feature_columns,
        )

        X_test, y_test = self.transform(
            test_df,
            feature_columns,
        )

        self.save_artifacts(
            feature_columns,
        )

        processed = ProcessedData(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
            feature_names=feature_columns,
        )

        self.save_metadata(processed)

        logger.info(
            "Preprocessing pipeline completed successfully."
        )

        return processed