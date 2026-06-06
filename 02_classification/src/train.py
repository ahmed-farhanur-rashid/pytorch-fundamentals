import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

def train(model, train_loader, loss_fn, optimizer, epoch: int = 100):
    """
    Trains the model and tracks loss and accuracy over epochs.

    Returns:
        model        → trained model
        train_losses → loss per epoch
        train_accs   → accuracy per epoch
    """
    train_losses = []
    train_accs   = []
    train_dataset_length = len(train_loader)

    for i in tqdm(range(epoch), desc="Training"):
        model.train()

        loss_per_epoch = 0.0
        correct        = 0
        total          = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss  = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            predicted  = torch.argmax(y_hat, dim=1)
            correct   += (predicted == y).sum().item()
            total     += y.size(0)

        train_losses.append(loss_per_epoch / train_dataset_length)
        train_accs.append(correct / total)

    return model, train_losses, train_accs