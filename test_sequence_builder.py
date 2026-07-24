"""
test_sequence_builder.py

Tests the complete preprocessing pipeline.

Run:

    python3 test_sequence_builder.py
"""

from pathlib import Path

from joblib import load

from src.models.sequence_builder import SequenceBuilder


def main():

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

    print("\n" + "=" * 60)
    print("Sequence Builder Results")
    print("=" * 60)

    print("\nTraining Sequences")
    print(X_train.shape)

    print("\nValidation Sequences")
    print(X_validation.shape)

    print("\nTesting Sequences")
    print(X_test.shape)

    print("\nTraining Targets")
    print(y_train.shape)

    print("\nValidation Targets")
    print(y_validation.shape)

    print("\nTesting Targets")
    print(y_test.shape)

    print("\nFeature Names")
    print(feature_names)

    # ----------------------------------------------------
    # Verify saved artifacts
    # ----------------------------------------------------

    print("\n" + "=" * 60)
    print("Artifact Checks")
    print("=" * 60)

    artifacts = [
        "models/feature_scaler.joblib",
        "models/target_scaler.joblib",
        "models/feature_names.joblib",
    ]

    for artifact in artifacts:

        exists = Path(artifact).exists()

        print(f"{artifact:<35} {exists}")

    # ----------------------------------------------------
    # Load artifacts
    # ----------------------------------------------------

    feature_scaler = load(
        "models/feature_scaler.joblib"
    )

    target_scaler = load(
        "models/target_scaler.joblib"
    )

    saved_feature_names = load(
        "models/feature_names.joblib"
    )

    print("\nFeature Scaler")
    print(type(feature_scaler).__name__)

    print("\nTarget Scaler")
    print(type(target_scaler).__name__)

    print("\nSaved Feature Names")
    print(saved_feature_names)

    # ----------------------------------------------------
    # Dataset integrity
    # ----------------------------------------------------

    print("\n" + "=" * 60)
    print("Dataset Integrity")
    print("=" * 60)

    print(f"Number of Features : {X_train.shape[2]}")
    print(f"Sequence Length    : {X_train.shape[1]}")
    print(f"Training Samples   : {len(X_train)}")
    print(f"Validation Samples : {len(X_validation)}")
    print(f"Testing Samples    : {len(X_test)}")

    # ----------------------------------------------------
    # Scaling diagnostics
    # ----------------------------------------------------

    print("\n" + "=" * 60)
    print("Scaling Diagnostics")
    print("=" * 60)

    print(f"Mean of X_train : {X_train.mean():.6f}")
    print(f"Std of X_train  : {X_train.std():.6f}")

    print(f"Mean of y_train : {y_train.mean():.6f}")
    print(f"Std of y_train  : {y_train.std():.6f}")

    # ----------------------------------------------------
    # Assertions
    # ----------------------------------------------------

    assert X_train.shape[2] == len(feature_names)

    assert X_train.shape[1] == builder.sequence_length

    assert len(X_train) > 0

    assert len(X_validation) > 0

    assert len(X_test) > 0

    assert Path(
        "models/feature_scaler.joblib"
    ).exists()

    assert Path(
        "models/target_scaler.joblib"
    ).exists()

    assert Path(
        "models/feature_names.joblib"
    ).exists()

    print("\n" + "=" * 60)
    print("All Sequence Builder Tests Passed")
    print("=" * 60)


if __name__ == "__main__":
    main()