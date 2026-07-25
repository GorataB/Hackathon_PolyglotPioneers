"""
sequence_builder.py

Creates LSTM-ready sliding window sequences.

Responsibilities
----------------
1. Convert scaled feature arrays into sequences
2. Align targets with each sequence
3. Preserve chronological order

This module assumes preprocessing has already been completed.
"""

from __future__ import annotations

import numpy as np

from src.config.config import SEQUENCE_LENGTH
from src.data.data_types import ProcessedData, SequenceData
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SequenceBuilder:
    """
    Builds sliding-window sequences for the LSTM model.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.sequence_length = sequence_length

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert one dataset partition into LSTM sequences.
        """

        logger.info(
            f"Creating sequences (window={self.sequence_length})..."
        )

        if len(X) <= self.sequence_length:
            raise ValueError(
                "Sequence length is larger than the available samples."
            )

        X_sequences = []
        y_sequences = []
        date_sequences = []

        for i in range(len(X) - self.sequence_length):

            X_sequences.append(
                X[i:i + self.sequence_length]
            )

            y_sequences.append(
                y[i + self.sequence_length]
            )

            date_sequences.append(
                dates[i + self.sequence_length]
            )

        X_sequences = np.asarray(
            X_sequences,
            dtype=np.float32,
        )

        y_sequences = np.asarray(
            y_sequences,
            dtype=np.float32,
        )

        date_sequences = np.asarray(
            date_sequences,
        )

        logger.info(
            f"Generated {len(X_sequences)} sequences."
        )

        logger.info(
            f"Sequence shape: {X_sequences.shape}"
        )

        return (
            X_sequences,
            y_sequences,
            date_sequences,
        )

    def build(
        self,
        processed: ProcessedData,
    ) -> SequenceData:
        """
        Build train, validation and test sequence datasets.
        """

        logger.info(
            "Building LSTM sequence datasets..."
        )

        (
            X_train,
            y_train,
            train_dates,
        ) = self.create_sequences(
            processed.X_train,
            processed.y_train,
            processed.train_dates,
        )

        (
            X_validation,
            y_validation,
            validation_dates,
        ) = self.create_sequences(
            processed.X_validation,
            processed.y_validation,
            processed.validation_dates,
        )

        (
            X_test,
            y_test,
            test_dates,
        ) = self.create_sequences(
            processed.X_test,
            processed.y_test,
            processed.test_dates,
        )

        logger.info(
            f"y_train_seq min={y_train.min()}, max={y_train.max()}"
        )

        logger.info(
            f"y_val_seq min={y_validation.min()}, max={y_validation.max()}"
        )

        logger.info(
            f"y_test_seq min={y_test.min()}, max={y_test.max()}"
        )

        logger.info(
            f"y_val_seq NaNs={np.isnan(y_validation).sum()}, Infs={np.isinf(y_validation).sum()}"
        )

        sequence_data = SequenceData(
            X_train=X_train,
            y_train=y_train,

            X_validation=X_validation,
            y_validation=y_validation,

            X_test=X_test,
            y_test=y_test,

            train_dates=train_dates,
            validation_dates=validation_dates,
            test_dates=test_dates,

            feature_names=processed.feature_names,

            input_size=X_train.shape[2],

            sequence_length=self.sequence_length,
        )

        logger.info(
            "Sequence generation completed successfully."
        )

        logger.info(
            f"Input size: {sequence_data.input_size}"
        )

        return sequence_data