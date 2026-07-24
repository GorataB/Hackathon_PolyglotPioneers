"""
correlation.py

Computes correlations between all input features and the target.
"""

import pandas as pd


def correlation_report(df: pd.DataFrame, target: str):
    """
    Print correlations between all numeric columns and the target.
    """

    correlations = (
        df.corr(numeric_only=True)[target]
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 70)
    print("CORRELATION OF FEATURES WITH TARGET")
    print("=" * 70)

    print(correlations)

    print("=" * 70 + "\n")

    return correlations