import torch.nn as nn


"""
    Returns PyTorch's built-in Mean Squared Error loss function.

    Parameters
    ----------
    reduction : how to combine per-sample losses into one number
        "mean"  → average all squared errors  (standard, what we use)
        "sum"   → sum all squared errors       (total loss)
        "none"  → return one loss per sample   (useful for debugging)

    Alt way to write it:

        def get_loss_fn(reduction: str = "mean") -> nn.Module:
            return nn.MSELoss(reduction=reduction)
    
    If no argument is passed inside, the sensible default is mean.
    Which is what is implemented below.
"""


def get_loss_fn():
    return nn.MSELoss()
