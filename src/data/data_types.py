"""
Shared data containers used throughout the forecasting pipeline.

Keeping these classes in one place prevents circular imports and
provides a single source of truth for the pipeline interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessedData:
    """
    Output from the preprocessing stage.
    """

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray

    feature_names: list[str]


@dataclass
class SequenceData:
    """
    Output from the sequence generation stage.
    """

    X_train: np.ndarray
    y_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    feature_names: list[str]

    input_size: int

    sequence_length: int