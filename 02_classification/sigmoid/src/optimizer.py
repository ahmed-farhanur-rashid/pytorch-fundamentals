import torch.optim as optim


def get_optimizer(model, lr=0.1):
    return optim.SGD(model.parameters(), lr=lr)
