import torch
import torch.nn as nn

class SoftMax(nn.Module):
    """
        Architecture:
            input (N features) → nn.Linear → logits (N, C) → Softmax → probabilities (N, C)

        Converts raw logits into a valid probability distribution using softmax:
            softmax(z_i) = exp(z_i) / Σ exp(z_j)

        Each output is in [0, 1] and all outputs sum to 1, giving a probability
        for each class.  The class with the highest probability is the prediction.

        Note: CrossEntropyLoss applies LogSoftmax internally, so do NOT use
        softmax in forward() when training with CrossEntropyLoss — only apply
        it during inference when you need actual probability values.
    """


    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.detach()

    def get_bias(self) -> torch.Tensor:
        return self.linear.bias.detach()
