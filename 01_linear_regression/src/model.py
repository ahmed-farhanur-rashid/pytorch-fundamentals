import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    """
    Architecture:
        input (N features)  →  nn.Linear  →  output (1 value)

    nn.Linear(in, out) creates a fully-connected layer that:
      - holds a weight matrix of shape (out, in)
      - holds a bias vector of shape (out,)
      - computes:  output = input @ weight.T + bias

    By wrapping it in nn.Module we get:
      - model.parameters()  → lets the optimizer find all weights/biases
      - model.state_dict()  → lets us save and load the model
      - model(x)            → automatically calls forward(x)
      - model.train() / model.eval() → switches training/evaluation mode
    """


    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.detach()

    def get_bias(self) -> torch.Tensor:
        return self.linear.bias.detach()
