import numpy as np

from src.utils.metrics import evaluate


def persistence_baseline(y: np.ndarray) -> dict:
    """
    Predict the next value using the previous value.
    """

    y = np.asarray(y).reshape(-1)

    y_true = y[1:]
    y_pred = y[:-1]

    metrics = evaluate(y_true, y_pred)

    print("\n===== Persistence Baseline =====")

    for k, v in metrics.items():
        print(f"{k.upper():5}: {v:.6f}")

    return metrics