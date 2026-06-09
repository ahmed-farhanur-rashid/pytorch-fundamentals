import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

def train(model, train_loader, loss_fn, optimizer, epoch: int = 30, validation_loader=None):
    """
    Trains the model and tracks loss and accuracy over epochs.

    Returns:
        model        → trained model
        train_losses → loss per epoch
        train_accs   → accuracy per epoch
    """
    train_losses = []
    train_accs   = []

    validation_losses = []
    validation_accs   = []

    for i in tqdm(range(epoch), desc="Training"):
        model.train()

        loss_per_epoch = 0.0
        correct        = 0
        total          = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x.view(x.size(0), -1))
            loss  = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            predicted  = torch.argmax(y_hat, dim=1)
            correct   += (predicted == y).sum().item()
            total     += y.size(0)

        train_losses.append(loss_per_epoch / len(train_loader))
        train_accs.append(correct / total)

        if(validation_loader is not None):
            model.eval()
            val_loss_per_epoch = 0.0
            val_correct        = 0
            val_total          = 0

            with torch.no_grad():
                for x, y in validation_loader:
                    y_hat = model(x.view(x.size(0), -1))
                    loss  = loss_fn(y_hat, y)

                    val_loss_per_epoch += loss.item()

                    predicted  = torch.argmax(y_hat, dim=1)
                    val_correct   += (predicted == y).sum().item()
                    val_total     += y.size(0)

            validation_losses.append(val_loss_per_epoch / len(validation_loader))
            validation_accs.append(val_correct / val_total)

    return model, train_losses, train_accs, validation_losses, validation_accs