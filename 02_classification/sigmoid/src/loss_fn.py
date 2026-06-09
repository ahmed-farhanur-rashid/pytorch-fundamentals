import torch.nn as nn


def get_loss_fn():
    """
    Binary Cross Entropy Loss.

    Used for binary classification (two classes: 0 or 1).

    It measures how far the predicted probability is from the true label:
      - If true=1 and pred=0.9 → low loss (good prediction)
      - If true=1 and pred=0.1 → high loss (bad prediction)

    Formula: -[y * log(p) + (1-y) * log(1-p)]
    """
    return nn.BCELoss()
