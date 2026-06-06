import torch.nn as nn


"""
    Returns PyTorch's CrossEntropyLoss — the standard loss function for classification.

    Why CrossEntropyLoss instead of MSELoss?
    ────────────────────────────────────────
    MSE measures the Euclidean distance between predicted and target values.
    That works for regression (where targets are continuous numbers), but breaks
    down for classification because class labels are arbitrary integers, not
    ordered magnitudes.

    Example with 3 classes {cat=0, dog=1, bird=2}:
      - True label: cat (0)
      - Model A predicts class 2 (bird) → MSE sees |2 - 0| = 2  (large error)
      - Model B predicts class 1 (dog)  → MSE sees |1 - 0| = 1  (smaller error)

    MSE thinks Model B is "closer" to correct, but there is no meaningful
    ordering — bird is not "twice as wrong" as dog. All wrong answers are
    equally wrong.

    CrossEntropyLoss avoids this by operating on probability distributions,
    not integer magnitudes. It:
      1. Applies LogSoftmax to the raw logits → converts them to log-probabilities
      2. Picks the log-probability of the true class (NLLLoss)
      3. Negates it — so higher probability for the correct class = lower loss

    This means the loss penalises the model for being confident in the wrong
    class and rewards it for concentrating probability mass on the correct one,
    regardless of how the classes are numbered.

    Input contract
    ──────────────
    logits : Tensor of shape (N, C) — raw, unnormalised scores for each class
    target : Tensor of shape (N,)    — integer class indices in [0, C)

    Returns
    ───────
    Scalar loss (mean reduction by default).
"""


def get_loss_fn():
    return nn.CrossEntropyLoss()
