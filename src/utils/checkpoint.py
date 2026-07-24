"""
checkpoint.py

Utility functions for saving and loading training artifacts.

Responsibilities
----------------
- Save best model
- Save latest checkpoint
- Load checkpoint
- Save training history
- Resume training state

Author: Polyglot Pioneers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Handles saving and loading of training checkpoints.
    """

    @staticmethod
    def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch: int,
        validation_loss: float,
        filepath: Path | str,
        history: dict | None = None,
    ) -> None:
        """
        Save a training checkpoint.

        Parameters
        ----------
        model
            PyTorch model.

        optimizer
            Optimizer used during training.

        scheduler
            Learning rate scheduler.

        epoch
            Current epoch.

        validation_loss
            Validation loss for this checkpoint.

        filepath
            Destination checkpoint path.

        history
            Optional training history.
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "history": history,
        }

        torch.save(
            checkpoint,
            filepath,
        )

        logger.info(
            f"Checkpoint saved to {filepath}"
        )

    @staticmethod
    def load_checkpoint(
        model,
        optimizer,
        scheduler,
        filepath: Path | str,
        device,
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Parameters
        ----------
        model
            PyTorch model.

        optimizer
            Optimizer.

        scheduler
            Learning rate scheduler.

        filepath
            Checkpoint location.

        device
            CPU / CUDA / MPS device.

        Returns
        -------
        dict
            Entire checkpoint dictionary.
        """

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {filepath}"
            )

        checkpoint = torch.load(
            filepath,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if (
            scheduler is not None
            and scheduler_state is not None
        ):
            scheduler.load_state_dict(
                scheduler_state
            )

        logger.info(
            f"Checkpoint loaded from {filepath}"
        )

        return checkpoint

    @staticmethod
    def save_history(
        history: dict,
        filepath: Path | str,
    ) -> None:
        """
        Save training history as JSON.

        Parameters
        ----------
        history
            Dictionary containing training metrics.

        filepath
            JSON destination.
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        logger.info(
            f"Training history saved to {filepath}"
        )

    @staticmethod
    def load_history(
        filepath: Path | str,
    ) -> dict:
        """
        Load previously saved training history.

        Parameters
        ----------
        filepath
            JSON history file.

        Returns
        -------
        dict
            Training history.
        """

        filepath = Path(filepath)

        if not filepath.exists():

            logger.info(
                "Training history not found."
            )

            return {}

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            history = json.load(file)

        logger.info(
            f"Training history loaded from {filepath}"
        )

        return history