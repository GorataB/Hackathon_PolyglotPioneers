import numpy as np
import torch


def confidence_interval(mean, std, n_samples, confidence=0.95):
    """
    Confidence interval for the estimated mean prediction.
    """
    z = 1.96 if confidence == 0.95 else 1.0
    margin = z * (std / np.sqrt(n_samples))
    return mean - margin, mean + margin


def prediction_interval(mean, std, confidence=0.95):
    """
    Prediction interval for future observations.
    """
    z = 1.96 if confidence == 0.95 else 1.0
    margin = z * std
    return mean - margin, mean + margin


def enable_dropout(model):
    """
    Enable dropout layers during inference.
    Leaves BatchNorm layers (if any) in eval mode.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def monte_carlo_predict(model, inputs, n_samples=100):
    """
    Perform Monte Carlo Dropout inference.

    Args:
        model: Trained PyTorch model.
        inputs: Input tensor.
        n_samples: Number of stochastic forward passes.

    Returns:
        mean_prediction
        std_prediction
        all_predictions
    """
    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            prediction = model(inputs)
            predictions.append(prediction.cpu().numpy())

    predictions = np.array(predictions)

    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)

    return mean_prediction, std_prediction, predictions