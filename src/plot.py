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
    """
    How to call:
    plot_lr_search_summary(
        learning_rates = LR_CANDIDATES,
        train_errors   = [train_losses_across_lrs[lr][-1] for lr in LR_CANDIDATES],
        val_errors     = [val_losses_across_lrs[lr][-1]   for lr in LR_CANDIDATES],
        save_path      = "../visualization/lr_search.png"
    )
    """
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