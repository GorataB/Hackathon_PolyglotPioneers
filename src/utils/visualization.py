"""
visualization.py

Visualization utilities for model predictions and uncertainty.

Author: Polyglot Pioneers
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    lower_interval: np.ndarray,
    upper_interval: np.ndarray,
    save_path: str = "models/figures/uncertainty_forecast.png",
):
    """
    Plot actual values, predictions and 95% prediction intervals.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth values.

    predictions : np.ndarray
        Mean Monte Carlo predictions.

    lower_interval : np.ndarray
        Lower prediction interval.

    upper_interval : np.ndarray
        Upper prediction interval.

    save_path : str
        Location to save the figure.
    """

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 6))

    x = np.arange(len(targets))

    plt.plot(
        x,
        targets,
        label="Actual",
        linewidth=2,
    )

    plt.plot(
        x,
        predictions,
        label="Prediction",
        linewidth=2,
    )

    plt.fill_between(
        x,
        lower_interval,
        upper_interval,
        alpha=0.25,
        label="95% Prediction Interval",
    )

    plt.title("LSTM Forecast with Monte Carlo Dropout Uncertainty")

    plt.xlabel("Test Sample")
    plt.ylabel("Scaled Target")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved uncertainty plot to: {save_path}")