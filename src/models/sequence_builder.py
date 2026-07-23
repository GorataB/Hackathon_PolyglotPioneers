"""
sequence_builder.py

Prepares time-series data for the LSTM model.

Responsibilities
----------------
1. Load prediction_data.csv
2. Select features and target
3. Scale features
4. Save preprocessing artifacts
5. Create sliding-window sequences
6. Split into train/validation/test datasets
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds LSTM-ready sequences from the processed dataset.
    """

    def __init__(
        self,
        data_path="Data_Clean/prediction_data.csv",
        target_column="FAO_23014",
        sequence_length=12,
        artifacts_dir="models",
    ):
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.sequence_length = sequence_length

        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)

        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load processed dataset.
        """

        logger.info("Loading dataset...")

        df = pd.read_csv(self.data_path)

        logger.info(f"Dataset loaded: {df.shape}")

        return df

    def prepare_features(self, df):
        """
        Separate features from target.
        """

        logger.info("Preparing features...")

        drop_columns = [
            "year_month",
            "date",
            "first",
            "last",
            self.target_column,
        ]

        feature_columns = [
            column
            for column in df.columns
            if column not in drop_columns
        ]

        X = df[feature_columns]

        y = df[self.target_column]

        logger.info(f"Selected {len(feature_columns)} features.")

        return X, y, feature_columns

    def scale_features(self, X):
        """
        Standardise features and save scaler.
        """

        logger.info("Scaling features...")

        X_scaled = self.scaler.fit_transform(X)

        scaler_path = self.artifacts_dir / "scaler.joblib"

        dump(self.scaler, scaler_path)

        logger.info(f"Scaler saved to {scaler_path}")

        return X_scaled

    def save_feature_names(self, feature_names):
        """
        Save feature names.
        """

        feature_path = self.artifacts_dir / "feature_names.joblib"

        dump(feature_names, feature_path)

        logger.info(f"Feature names saved to {feature_path}")

    def create_sequences(self, X, y):
        """
        Convert data into LSTM sliding windows.
        """

        logger.info("Creating sequences...")

        X_sequences = []
        y_sequences = []

        for i in range(len(X) - self.sequence_length):

            X_sequences.append(
                X[i:i + self.sequence_length]
            )

            y_sequences.append(
                y.iloc[i + self.sequence_length]
            )

        X_sequences = np.asarray(
            X_sequences,
            dtype=np.float32,
        )

        y_sequences = np.asarray(
            y_sequences,
            dtype=np.float32,
        )

        logger.info(
            f"Created {len(X_sequences)} sequences."
        )

        return X_sequences, y_sequences

    def split_data(
        self,
        X,
        y,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    ):
        """
        Split into Train / Validation / Test.
        """

        logger.info("Splitting dataset...")

        if round(
            train_size + validation_size + test_size,
            2,
        ) != 1.00:
            raise ValueError(
                "Train + Validation + Test must equal 1."
            )

        X_train, X_remaining, y_train, y_remaining = (
            train_test_split(
                X,
                y,
                train_size=train_size,
                shuffle=False,
            )
        )

        validation_fraction = (
            validation_size /
            (validation_size + test_size)
        )

        (
            X_validation,
            X_test,
            y_validation,
            y_test,
        ) = train_test_split(
            X_remaining,
            y_remaining,
            train_size=validation_fraction,
            shuffle=False,
        )

        logger.info(
            f"""
Train: {X_train.shape}
Validation: {X_validation.shape}
Test: {X_test.shape}
"""
        )

        return (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        )

    def build(self, split=True):
        """
        Execute complete preprocessing pipeline.
        """

        logger.info("Starting preprocessing pipeline...")

        df = self.load_data()

        X, y, feature_names = self.prepare_features(df)

        self.save_feature_names(feature_names)

        X_scaled = self.scale_features(X)

        X_sequences, y_sequences = self.create_sequences(
            X_scaled,
            y,
        )

        if not split:

            logger.info("Pipeline complete.")

            return (
                X_sequences,
                y_sequences,
                feature_names,
            )

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        ) = self.split_data(
            X_sequences,
            y_sequences,
        )

        logger.info("Pipeline complete.")

        return (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            feature_names,
        )