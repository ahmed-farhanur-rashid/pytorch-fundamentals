import torch


def evaluate(model, data_loader, loss_fn):
    model.eval()
    evaluation_loss = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            y_hat = model(x)                # forward pass
            loss = loss_fn(y_hat, y)        # calculate loss
            evaluation_loss += loss.item()  # append loss

    model.train()
    return evaluation_loss / len(data_loader)