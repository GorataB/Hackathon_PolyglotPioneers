import types

import torch

from src.models.trainer import Trainer


def test_get_device_falls_back_to_cpu_for_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        types.SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    trainer = Trainer.__new__(Trainer)

    assert trainer._get_device() == torch.device("cpu")
