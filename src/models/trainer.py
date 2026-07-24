"""
trainer.py

Production trainer for the LSTM forecasting model.

Responsibilities
----------------
- Device management
- Reproducibility
- Dataset preparation
- DataLoader creation
- Model training
- Validation
- Checkpoint management
- TensorBoard logging
- Evaluation
- Prediction

Author: Polyglot Pioneers
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.models.uncertainty import (
    confidence_interval,
    monte_carlo_predict,
    prediction_interval,
)

from tqdm import tqdm

from src.config.config import (
    BATCH_SIZE,
    BEST_MODEL_FILE,
    LAST_CHECKPOINT_FILE,
    TRAINING_HISTORY_FILE,
    TENSORBOARD_DIR,
    EPOCHS,
    GRADIENT_CLIP,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    DROPOUT,
    PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
)

from src.data.data_types import SequenceData
from src.data.dataset import ForecastDataset
from src.utils.visualization import plot_prediction_intervals

from src.models.lstm_model import LSTMModel

from src.utils.metrics import evaluate
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import get_logger

logger = get_logger(__name__)
class Trainer:
    """
    Trainer for the forecasting LSTM.
    """

    def __init__(
        self,
        sequence_data: SequenceData,
    ):

        logger.info("Initialising trainer...")

        self.sequence_data = sequence_data

        self.device = self._get_device()

        self._set_seed(RANDOM_SEED)

        logger.info(
            f"Training device: {self.device}"
        )

        # -------------------------------------
        # TensorBoard
        # -------------------------------------

        self.writer = SummaryWriter(
            log_dir=TENSORBOARD_DIR
        )

        logger.info(
            f"SequenceData y_train: "
            f"min={sequence_data.y_train.min()}, "
            f"max={sequence_data.y_train.max()}"
        )

        logger.info(
            f"SequenceData y_validation: "
            f"min={sequence_data.y_validation.min()}, "
            f"max={sequence_data.y_validation.max()}"
        )

        logger.info(
            f"SequenceData y_test: "
            f"min={sequence_data.y_test.min()}, "
            f"max={sequence_data.y_test.max()}"
        )

        # -------------------------------------
        # Dataset
        # -------------------------------------

        self.train_dataset = ForecastDataset(
            sequence_data.X_train,
            sequence_data.y_train,
        )

        self.validation_dataset = ForecastDataset(
            sequence_data.X_validation,
            sequence_data.y_validation,
        )

        self.test_dataset = ForecastDataset(
            sequence_data.X_test,
            sequence_data.y_test,
        )

        # -------------------------------------
        # DataLoaders
        # -------------------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        # -------------------------------------
        # Model
        # -------------------------------------

        self.model = LSTMModel(
            input_size=sequence_data.input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(self.device)

        # -------------------------------------
        # Optimizer
        # -------------------------------------

        self.optimizer = Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # -------------------------------------
        # Scheduler
        # -------------------------------------

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # -------------------------------------
        # Loss
        # -------------------------------------

        self.criterion = nn.MSELoss()

        # -------------------------------------
        # Training state
        # -------------------------------------

        self.current_epoch = 0

        self.best_epoch = 0

        self.best_validation_loss = float("inf")

        self.early_stopping_counter = 0

        self.epochs = EPOCHS

        self.patience = PATIENCE

        self.history = {

            "train_loss": [],

            "validation_loss": [],

            "mae": [],

            "rmse": [],

            "r2": [],

            "learning_rate": [],

        }

        logger.info(
            "Trainer initialised successfully."
        )
            # =====================================================
    # Utilities
    # =====================================================

    @staticmethod
    def _set_seed(seed: int):

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _get_device():

        if torch.cuda.is_available():

            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):

            logger.warning(
                "MPS backend detected; falling back to CPU because "
                "it can produce unstable training results on this project."
            )

            return torch.device("cpu")

        return torch.device("cpu")

    def _current_learning_rate(self):

        return self.optimizer.param_groups[0]["lr"]
        # =====================================================
    # Training
    # =====================================================

    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average training loss.
        """

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            leave=False,
        )

        for inputs, targets in progress:

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            if torch.isnan(outputs).any():
                raise RuntimeError(
                    "NaN values detected in model outputs during validation."
                )

            if torch.isinf(outputs).any():
                raise RuntimeError(
                    "Infinite values detected in model outputs during validation."
                )

            loss = self.criterion(outputs, targets)

            if torch.isnan(loss):
                raise RuntimeError(
                    "Validation loss became NaN."
                )

            if torch.isinf(loss):
                raise RuntimeError(
                    "Validation loss became infinite."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                GRADIENT_CLIP,
            )

            self.optimizer.step()

            running_loss += (
                loss.item() * inputs.size(0)
            )

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        epoch_loss = (
            running_loss
            / len(self.train_dataset)
        )

        return epoch_loss
        # =====================================================
    # Validation
    # =====================================================

    @torch.no_grad()
    def _validate_epoch(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Validate the model for one epoch.
        """

        self.model.eval()

        running_loss = 0.0

        predictions = []

        targets = []

        for inputs, labels in self.validation_loader:

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(inputs)

            loss = self.criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * inputs.size(0)
            )

            predictions.append(
                outputs.cpu().numpy()
            )

            targets.append(
                labels.cpu().numpy()
            )

        validation_loss = (
            running_loss
            / len(self.validation_dataset)
        )

        predictions = np.concatenate(
            predictions,
            axis=0,
        )

        targets = np.concatenate(
            targets,
            axis=0,
        )

        return (
            validation_loss,
            predictions,
            targets,
        )
        # =====================================================
    # TensorBoard
    # =====================================================

    def _log_tensorboard(
        self,
        train_loss: float,
        validation_loss: float,
        metrics: dict,
    ) -> None:
        """
        Log metrics to TensorBoard.
        """

        epoch = self.current_epoch + 1

        self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Validation",
            validation_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Metrics/MAE",
            metrics["mae"],
            epoch,
        )

        self.writer.add_scalar(
            "Metrics/MSE",
            metrics["mse"],
            epoch,
        )

        self.writer.add_scalar(
            "Metrics/RMSE",
            metrics["rmse"],
            epoch,
        )

        self.writer.add_scalar(
            "Metrics/R2",
            metrics["r2"],
            epoch,
        )

        self.writer.add_scalar(
            "Learning Rate",
            self._current_learning_rate(),
            epoch,
        )
            # =====================================================
    # Scheduler
    # =====================================================

    def _update_scheduler(
        self,
        validation_loss: float,
    ) -> None:
        """
        Update the learning rate scheduler.
        """

        previous_lr = self._current_learning_rate()

        self.scheduler.step(
            validation_loss
        )

        current_lr = self._current_learning_rate()

        if previous_lr != current_lr:

            logger.info(

                "Learning rate changed "

                f"from {previous_lr:.6f} "

                f"to {current_lr:.6f}"

            )
                # =====================================================
    # Public API
    # =====================================================

    def train(self) -> dict:
        """
        Train the model.

        Returns
        -------
        dict
            Training history.
        """

        logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.epochs):

            self.current_epoch = epoch

            train_loss = self._train_epoch()

            validation_loss, predictions, targets = (
                self._validate_epoch()
            )

            predictions = np.nan_to_num(
                predictions,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            targets = np.nan_to_num(
                targets,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            metrics = evaluate(
                targets,
                predictions,
            )

            self.history["train_loss"].append(train_loss)
            self.history["validation_loss"].append(validation_loss)
            self.history["mae"].append(metrics["mae"])
            self.history["rmse"].append(metrics["rmse"])
            self.history["r2"].append(metrics["r2"])
            self.history["learning_rate"].append(
                self._current_learning_rate()
            )

            self._log_tensorboard(
                train_loss,
                validation_loss,
                metrics,
            )

            self._update_scheduler(
                validation_loss
            )

            # Save latest checkpoint

            CheckpointManager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                validation_loss=validation_loss,
                filepath=LAST_CHECKPOINT_FILE,
                history=self.history,
            )

            # Save best checkpoint

            if (
                epoch == 0
                or validation_loss < self.best_validation_loss
            ):

                self.best_validation_loss = validation_loss
                self.best_epoch = epoch

                self.early_stopping_counter = 0

                CheckpointManager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    validation_loss=validation_loss,
                    filepath=BEST_MODEL_FILE,
                    history=self.history,
                )

            else:

                self.early_stopping_counter += 1

            CheckpointManager.save_history(
                self.history,
                TRAINING_HISTORY_FILE,
            )

            logger.info(

                f"Epoch {epoch + 1}/{self.epochs} | "

                f"Train={train_loss:.6f} | "

                f"Validation={validation_loss:.6f} | "

                f"MAE={metrics['mae']:.6f} | "

                f"RMSE={metrics['rmse']:.6f} | "

                f"R²={metrics['r2']:.4f}"

            )

            if self.early_stopping_counter >= self.patience:

                logger.info(
                    "Early stopping triggered."
                )

                break

        logger.info(
            f"Training finished. Best epoch: {self.best_epoch + 1}"
        )

        return self.history

    def resume_training(self) -> bool:
        """
        Resume training from the latest checkpoint.

        Returns
        -------
        bool
            True if a checkpoint was loaded.
        """

        try:

            checkpoint = CheckpointManager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                filepath=LAST_CHECKPOINT_FILE,
                device=self.device,
            )

            self.current_epoch = checkpoint["epoch"] + 1
            self.best_validation_loss = checkpoint["validation_loss"]

            if checkpoint.get("history") is not None:
                self.history = checkpoint["history"]

            logger.info(
                f"Resuming from epoch {self.current_epoch}"
            )

            return True

        except FileNotFoundError:

            logger.info(
                "No previous checkpoint found."
            )

            return False

    def load_best_model(self) -> None:
        """
        Load the best saved model.
        """

        if not Path(BEST_MODEL_FILE).exists():

            logger.warning(
                "Best checkpoint not found. "
                "Using current model weights."
            )

            return

        CheckpointManager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filepath=BEST_MODEL_FILE,
            device=self.device,
        )

        logger.info(
            "Best model loaded."
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate the best model using the test set.
        """

        self.model.eval()

        predictions = []
        targets = []
        lower_intervals = []
        upper_intervals = []

        for inputs, labels in self.test_loader:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            mean, std, samples = monte_carlo_predict(
                self.model,
                inputs,
                n_samples=100,
            )

            lower_ci, upper_ci = confidence_interval(
                mean,
                std,
                n_samples=100,
            )

            lower_pi, upper_pi = prediction_interval(
                mean,
                std,
            )

            if len(predictions) == 0:
                logger.info(f"Mean shape: {mean.shape}")
                logger.info(f"Std shape: {std.shape}")
                logger.info(f"Samples shape: {samples.shape}")

                logger.info(
                    f"Prediction: {mean[0]:.4f}"
                )
                logger.info(
                    f"95% Confidence Interval: [{lower_ci[0]:.4f}, {upper_ci[0]:.4f}]"
                )
                logger.info(
                    f"95% Prediction Interval: [{lower_pi[0]:.4f}, {upper_pi[0]:.4f}]"
                )

            predictions.append(mean)
            lower_intervals.append(lower_pi)
            upper_intervals.append(upper_pi)

            targets.append(
                labels.cpu().numpy()
            )

        predictions = np.concatenate(
            predictions,
            axis=0,
        )

        targets = np.concatenate(
            targets,
            axis=0,
        )

        lower_intervals = np.concatenate(
            lower_intervals,
            axis=0,
        )

        upper_intervals = np.concatenate(
            upper_intervals,
            axis=0,
        )

        plot_prediction_intervals(
            targets=targets.flatten(),
            predictions=predictions.flatten(),
            lower_interval=lower_intervals.flatten(),
            upper_interval=upper_intervals.flatten(),
        )

        logger.info(
            f"Predictions: min={predictions.min():.4f}, "
            f"max={predictions.max():.4f}, "
            f"mean={predictions.mean():.4f}"
        )

        logger.info(
            f"Targets: min={targets.min():.4f}, "
            f"max={targets.max():.4f}, "
            f"mean={targets.mean():.4f}"
        )

        logger.info(f"First 10 predictions: {predictions[:10].flatten()}")
        logger.info(f"First 10 targets: {targets[:10].flatten()}")

        logger.info(
            f"Prediction correlation = "
            f"{np.corrcoef(predictions.flatten(), targets.flatten())[0,1]:.4f}"
        )

        metrics = evaluate(
            targets,
            predictions,
        )

        logger.info(
            f"Test metrics: {metrics}"
        )

        return metrics

    @torch.no_grad()
    def predict(
        self,
        sequences: np.ndarray,
    ) -> np.ndarray:
        """
        Predict using trained model.
        """

        self.model.eval()

        sequences = torch.as_tensor(
            sequences,
            dtype=torch.float32,
            device=self.device,
        )

        predictions = self.model(
            sequences
        )

        return (
            predictions
            .cpu()
            .numpy()
        )

    def close(self) -> None:
        """
        Close TensorBoard writer.
        """

        self.writer.close()