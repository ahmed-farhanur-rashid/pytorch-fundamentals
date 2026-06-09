import torch.optim as optim


def get_optimizer(model, lr=0.01):
    """
        Returns an SGD optimizer for the given model.
        lr: step size for weight updates.
    """
    return optim.SGD(model.parameters(), lr=lr)