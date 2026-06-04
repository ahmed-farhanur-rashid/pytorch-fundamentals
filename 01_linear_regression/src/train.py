import torch
from evaluate import evaluate


def train(model, train_loader, loss_fn, optimizer, epoch: int=100, validation_loader=None):
    train_losses = []
    validation_losses = []

    for i in range(epoch):
        model.train()
        loss_per_epoch = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()           # clear gradients
            y_hat = model(x)                # forward pass
            loss = loss_fn(y_hat, y)        # compute loss
            loss.backward()                 # backward pass
            optimizer.step()                # update weight
            loss_per_epoch += loss.item()   # log loss

        train_losses.append(loss_per_epoch / len(train_loader))

        if validation_loader:
            evaluated_loss = evaluate(model, validation_loader, loss_fn)
            validation_losses.append(evaluated_loss)
    
    return model, train_losses, validation_losses
