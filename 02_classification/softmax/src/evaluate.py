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


def evaluate_on_test(model, test_loader):
    model.eval()
    all_y     = []
    all_y_hat = []

    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            all_y.append(y)
            all_y_hat.append(y_hat)

    # concatenates them into a single tensor and converts it to numpy
    # this is important as tensor creates 2D torch array arranged by batches. 
    all_y     = torch.cat(all_y).numpy()
    all_y_hat = torch.cat(all_y_hat).numpy()

    return all_y, all_y_hat