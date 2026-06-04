import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_losses(losses: dict, title: str, save_path: str):
    """
    How to call:
    plot_losses(
        losses    = {
            "train"      : best_train_loss,
            "validation" : best_validation_loss
        },
        title     = "Best Model Loss",
        save_path = "../visualization/loss.png"
    )
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for label, values in losses.items():
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_lr_search_summary(learning_rates, train_errors, val_errors, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogx(learning_rates, train_errors,
                label="Train loss",      color="#2196F3", marker='o')
    ax.semilogx(learning_rates, val_errors,
                label="Validation loss", color="#FF5722", marker='s', linestyle="--")

    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Final MSE Loss")
    ax.set_title("LR Search — Final Loss per Learning Rate")
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions(y_actual, y_hat, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_actual, y_hat, alpha=0.6, color="#2196F3",
               edgecolors="white", linewidths=0.5, label="Predictions")

    min_val = min(y_actual.min(), y_hat.min())
    max_val = max(y_actual.max(), y_hat.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="#FF5722", linestyle="--", linewidth=1.5, label="Perfect fit")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()