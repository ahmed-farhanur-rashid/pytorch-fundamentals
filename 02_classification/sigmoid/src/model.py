import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    """
    Binary classification model.

    Architecture:
        input (13 features) → Linear(13, 1) → Sigmoid → probability (0 to 1)

    Sigmoid squashes any number into the range [0, 1]:
        σ(z) = 1 / (1 + e^(-z))

    If output > 0.5 → predict class 1 (has heart disease)
    If output ≤ 0.5 → predict class 0 (no heart disease)
    """

    def __init__(self, input_size=13):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
