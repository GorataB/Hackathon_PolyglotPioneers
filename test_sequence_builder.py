from pathlib import Path

from src.models.sequence_builder import SequenceBuilder

print("=" * 70)
print("Sequence Builder Test")
print("=" * 70)

builder = SequenceBuilder(sequence_length=12)

(
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test,
    feature_names,
) = builder.build()

print()

print("Dataset Shapes")
print("-" * 70)

print(f"Training     : {X_train.shape}")
print(f"Validation   : {X_validation.shape}")
print(f"Testing      : {X_test.shape}")

print()

print("Target Shapes")
print("-" * 70)

print(f"Training     : {y_train.shape}")
print(f"Validation   : {y_validation.shape}")
print(f"Testing      : {y_test.shape}")

print()

print("Feature Names")
print("-" * 70)

print(feature_names)

print()

print("First Training Sequence Shape")
print("-" * 70)

print(X_train[0].shape)

print()

print("First Target")
print("-" * 70)

print(y_train[0])

print()

print("Checking Saved Artifacts")
print("-" * 70)

artifacts = [
    "models/scaler.joblib",
    "models/feature_names.joblib",
]

for artifact in artifacts:

    exists = Path(artifact).exists()

    print(f"{artifact:<35} {'✅ Found' if exists else '❌ Missing'}")

print()

print("=" * 70)
print("Sequence Builder Test Complete")
print("=" * 70)