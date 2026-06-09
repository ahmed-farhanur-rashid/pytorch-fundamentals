import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def _make_dir(save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, save_path: str):
    _make_dir(save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train", color="#2196F3")
    ax1.plot(val_losses, label="Validation", color="#FF5722", linestyle="--")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label="Train", color="#2196F3")
    ax2.plot(val_accs, label="Validation", color="#FF5722", linestyle="--")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, accuracy=None, save_path: str = ""):
    _make_dir(save_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    title = "Confusion Matrix"
    if accuracy is not None:
        title += f" — Accuracy: {accuracy:.2%}"
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path: str):
    """
    Shows which features the model considers most important.
    Positive weight = pushes toward disease (class 1)
    Negative weight = pushes toward no disease (class 0)
    """
    _make_dir(save_path)

    weights = model.linear.weight.detach().numpy().flatten()

    sorted_idx = np.argsort(np.abs(weights))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_weights = weights[sorted_idx]

    colors = ['#FF5722' if w > 0 else '#2196F3' for w in sorted_weights]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_names, sorted_weights, color=colors)
    ax.set_xlabel("Weight")
    ax.set_title("Feature Importance (Red = Disease, Blue = No Disease)")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
