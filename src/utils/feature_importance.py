"""
feature_importance.py

Computes feature importance using a Random Forest Regressor.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def feature_importance_report(
    df: pd.DataFrame,
    feature_columns,
    target_column,
):
    """
    Train a Random Forest and display feature importance.
    """

    X = df[feature_columns]
    y = df[target_column]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    importance = (
        pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)
    print(importance)
    print("=" * 70 + "\n")

    return importance