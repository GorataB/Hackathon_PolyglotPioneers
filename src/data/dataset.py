"""
dataset.py

PyTorch Dataset implementation for the forecasting project.

Responsibilities
----------------
- Wrap sequence data in a PyTorch Dataset
- Validate dimensions
- Provide samples to DataLoader

Author: Polyglot Pioneers
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForecastDataset(Dataset):
    """
    Dataset used by the LSTM trainer.

    Parameters
    ----------
    X : numpy.ndarray
        Shape:
            (samples, sequence_length, features)

    y : numpy.ndarray
        Shape:
            (samples,)
    """

    def __init__(self, X, y):

        if len(X) != len(y):
            raise ValueError(
                "Feature and target sample counts do not match."
            )

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.as_tensor(
            y,
            dtype=torch.float32,
        )

        logger.info(
            f"Dataset created with {len(self.X)} samples."
        )

    def __len__(self):
        """
        Number of samples.
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        Retrieve one training sample.
        """
        return self.X[index], self.y[index]

    @property
    def input_size(self):
        """
        Number of model input features.
        """
        return self.X.shape[2]

    @property
    def sequence_length(self):
        """
        Sequence/window length.
        """
        return self.X.shape[1]