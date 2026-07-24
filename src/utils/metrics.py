"""
metrics.py

Evaluation metrics used throughout the forecasting project.

These metrics are intentionally implemented using NumPy instead of
scikit-learn to avoid unnecessary dependencies inside the training
loop and to make them reusable across the Trainer, Predictor,
Evaluator and Dashboard.

Author: Polyglot Pioneers
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    """

    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return 0.0

    return float(
        1 - (ss_res / ss_tot)
    )


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all regression metrics.

    Returns
    -------
    dict
    """

    return {

        "mae": mae(
            y_true,
            y_pred,
        ),

        "mse": mse(
            y_true,
            y_pred,
        ),

        "rmse": rmse(
            y_true,
            y_pred,
        ),

        "r2": r2(
            y_true,
            y_pred,
        ),
    }