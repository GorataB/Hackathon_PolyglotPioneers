"""
test_trainer.py

Integration test for the LSTM Trainer.

Run:

    python3 test_trainer.py
"""

import json
from pathlib import Path

import torch

from src.models.sequence_builder import SequenceBuilder
from src.models.lstm_model import LSTMModel
from src.models.trainer import Trainer


def main():

    # ----------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------

    builder = SequenceBuilder()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_names,
    ) = builder.build()

    # ----------------------------------------------------
    # Create Model
    # ----------------------------------------------------

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    # ----------------------------------------------------
    # Create Trainer
    # ----------------------------------------------------

    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        patience=5,
    )

    history = trainer.train(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # ----------------------------------------------------
    # Output Summary
    # ----------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Finished")
    print("=" * 60)

    print("\nHistory Keys")
    print(history.keys())

    print("\nFinal Training Loss")
    print(history["train_loss"][-1])

    print("\nFinal Validation Loss")
    print(history["validation_loss"][-1])

    print("\nFinal Learning Rate")
    print(history["learning_rate"][-1])

    print("\nBest Epoch")
    print(trainer.best_epoch)

    # ----------------------------------------------------
    # Check Artifacts
    # ----------------------------------------------------

    checkpoint = Path("models/best_lstm_model.pth")
    history_file = Path("models/training_history.json")

    print("\nCheckpoint Exists")
    print(checkpoint.exists())

    print("\nTraining History Exists")
    print(history_file.exists())

    # ----------------------------------------------------
    # Verify History File
    # ----------------------------------------------------

    if history_file.exists():

        with open(history_file) as f:
            saved_history = json.load(f)

        print("\nHistory File Keys")
        print(saved_history.keys())

        print("\nHistory Length")

        print(
            len(saved_history["train_loss"]),
            len(saved_history["validation_loss"]),
            len(saved_history["learning_rate"]),
        )

    # ----------------------------------------------------
    # Verify Checkpoint
    # ----------------------------------------------------

    if checkpoint.exists():

        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
        )

        print("\nCheckpoint Keys")

        print(checkpoint_data.keys())

        print("\nCheckpoint Epoch")

        print(checkpoint_data["epoch"])

        print("\nCheckpoint Validation Loss")

        print(checkpoint_data["validation_loss"])

    # ----------------------------------------------------
    # Resume Test
    # ----------------------------------------------------

    print("\nTesting Checkpoint Loading...")

    resumed_model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    resumed_trainer = Trainer(
        resumed_model,
    )

    start_epoch = resumed_trainer.load_checkpoint()

    print("Resume Epoch:", start_epoch)

    # ----------------------------------------------------
    # Assertions
    # ----------------------------------------------------

    assert checkpoint.exists()
    assert history_file.exists()

    assert trainer.best_epoch > 0

    assert len(history["train_loss"]) > 0
    assert len(history["validation_loss"]) > 0
    assert len(history["learning_rate"]) > 0

    assert (
        len(history["train_loss"])
        == len(history["validation_loss"])
        == len(history["learning_rate"])
    )

    assert start_epoch > 0

    print("\n" + "=" * 60)
    print("All Trainer Tests Passed")
    print("=" * 60)


if __name__ == "__main__":
    main()