import torch
from tqdm import tqdm


def train(model, train_loader, loss_fn, optimizer, epochs=100, val_loader=None):
    """
    Trains the model and tracks loss and accuracy.

    For binary classification, predictions are compared at threshold 0.5:
      - model outputs 0.73 → predict 1 (since 0.73 > 0.5)
      - model outputs 0.21 → predict 0 (since 0.21 ≤ 0.5)
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (y_hat > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    val_loss += loss.item()
                    predicted = (y_hat > 0.5).float()
                    val_correct += (predicted == y).sum().item()
                    val_total += y.size(0)

            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_correct / val_total)

    return model, train_losses, train_accs, val_losses, val_accs
