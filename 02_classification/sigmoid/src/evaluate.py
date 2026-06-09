import torch
import numpy as np


def evaluate_on_test(model, test_loader):
    """
    Evaluates the model on the test set.

    Returns:
        all_labels → true labels (numpy array)
        all_preds  → predicted labels 0 or 1 (numpy array)
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            predicted = (y_hat > 0.5).float()

            all_labels.extend(y.numpy())
            all_preds.extend(predicted.numpy())

    return np.array(all_labels).flatten(), np.array(all_preds).flatten()
