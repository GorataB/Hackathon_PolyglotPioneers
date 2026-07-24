import torch

from src.models.lstm_model import LSTMModel

SEQUENCE_LENGTH = 12
NUM_FEATURES = 6
BATCH_SIZE = 32

model = LSTMModel(
    input_size=NUM_FEATURES,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
)

print("=" * 60)
print("LSTM Model Test")
print("=" * 60)

dummy_input = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
)

print("Input Shape")
print(dummy_input.shape)

prediction = model(dummy_input)

print()

print("Prediction Shape")
print(prediction.shape)

print()

print("Sample Predictions")
print(prediction[:5])

print()

print("Model Summary")
print(model)

print()

print("Total Parameters")

total = sum(
    p.numel()
    for p in model.parameters()
)

print(total)