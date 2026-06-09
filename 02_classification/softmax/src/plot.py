import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _make_dir(save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ──────────────────────────────────────────────────────────
# Loss & Accuracy
# ──────────────────────────────────────────────────────────

def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, save_path: str):
    """
    How to call:
    plot_loss_and_accuracy(
        train_losses = [0.9, 0.7, 0.5],
        val_losses   = [1.0, 0.8, 0.6],
        train_accs   = [0.6, 0.75, 0.85],
        val_accs     = [0.55, 0.70, 0.80],
        save_path    = "../visualization/loss_acc.png"
    )
    """
    _make_dir(save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(train_losses, label="Train",      color="#2196F3")
    ax1.plot(val_losses,   label="Validation", color="#FF5722", linestyle="--")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(train_accs, label="Train",      color="#2196F3")
    ax2.plot(val_accs,   label="Validation", color="#FF5722", linestyle="--")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ──────────────────────────────────────────────────────────
# Train vs Validation Accuracy (side by side)
# ──────────────────────────────────────────────────────────

def plot_train_val_accuracy(train_accs, val_accs, save_path: str):
    """
    Plots train and validation accuracy side by side as separate subplots.

    How to call:
    plot_train_val_accuracy(
        train_accs = [0.6, 0.75, 0.85],
        val_accs   = [0.55, 0.70, 0.80],
        save_path  = "../visualization/train_val_acc.png"
    )
    """
    _make_dir(save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Train Accuracy
    ax1.plot(train_accs, color="#2196F3", marker='o', markersize=4)
    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)

    # Validation Accuracy
    ax2.plot(val_accs, color="#FF5722", marker='o', markersize=4)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ──────────────────────────────────────────────────────────
# Model Weights (what the model learned per class)
# ──────────────────────────────────────────────────────────

def plot_parameters(model, save_path: str):
    """
    Visualizes the learned weights for each of the 10 digit classes.
    Red = positive weight (pushes toward class)
    Blue = negative weight (pushes away from class)

    How to call:
    plot_parameters(
        model     = model,
        save_path = "../visualization/weights.png"
    )
    """
    _make_dir(save_path)

    W     = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        ax.set_xlabel(f"class: {i}")
        ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Learned Weights per Class", fontsize=13)
    plt.savefig(save_path)
    plt.close()

# ──────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, save_path: str):
    """
    cm = confusion matrix from sklearn.metrics.confusion_matrix

    How to call:
    plot_confusion_matrix(
        cm        = confusion_matrix(y_true, y_pred),
        save_path = "../visualization/confusion_matrix.png"
    )
    """
    _make_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ──────────────────────────────────────────────────────────
# Sample Predictions
# ──────────────────────────────────────────────────────────

def plot_sample_predictions(images, labels, preds, save_path: str, n=10):
    """
    Shows n sample images with their true vs predicted labels.
    Green title = correct, Red title = wrong.

    How to call:
    plot_sample_predictions(
        images    = images,    # tensor of shape (N, 1, 28, 28)
        labels    = labels,    # true labels
        preds     = preds,     # predicted labels
        save_path = "../visualization/sample_predictions.png"
    )
    """
    _make_dir(save_path)

    fig, axes = plt.subplots(1, n, figsize=(15, 2))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        correct = labels[i] == preds[i]
        color   = 'green' if correct else 'red'
        ax.set_title(f"T:{labels[i]}\nP:{preds[i]}", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()