"""
lstm_model.py

Defines the LSTM forecasting model.

Architecture
------------
Input
    ↓
LSTM
    ↓
Dropout
    ↓
Fully Connected Layer
    ↓
Prediction
"""

import logging

import torch
import torch.nn as nn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory network for time-series forecasting.
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

        logger.info(
            "LSTM Model Initialised | "
            f"Input={input_size}, Hidden={hidden_size}, "
            f"Layers={num_layers}, Dropout={dropout}"
        )

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : Tensor
            Shape:
            (batch_size, sequence_length, input_features)

        Returns
        -------
        Tensor
            Shape:
            (batch_size,)
        """

        lstm_output, _ = self.lstm(x)

        last_hidden = lstm_output[:, -1, :]

        last_hidden = self.dropout(last_hidden)

        prediction = self.fc(last_hidden)

        return prediction.squeeze(-1)