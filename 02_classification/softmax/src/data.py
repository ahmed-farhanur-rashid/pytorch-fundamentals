import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

def load_mnist(root='./data'):
    """
        Loads the MNIST dataset and applies necessary transformations.
        MNIST dataset itself is a class hence no class inheritance was used.

        MNIST dataset:
          - 60,000 training images
          - 10,000 test images
          - 28x28 grayscale images
          - 10 classes (digits 0-9)

        transforms.ToTensor() does two things:
          - converts image to tensor
          - normalizes pixel values from [0, 255] to [0.0, 1.0]
    """
    train_dataset = dsets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = dsets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    return train_dataset, test_dataset


# ──────────────────────────────────────────────────────────
# Dataset Splitter
# ──────────────────────────────────────────────────────────

"""
    Not needed for MNIST since it already has separate training and test sets.
"""


# ──────────────────────────────────────────────────────────
# Data Loader
# ──────────────────────────────────────────────────────────

def create_data_loader(dataset, batch_size=32, shuffle=False):
    """
        Wraps dataset in a DataLoader for mini-batch iteration.

        batch_size = 1            → Stochastic Gradient Descent (SGD)
        1 < batch_size < total N  → Mini-Batch Gradient Descent
        batch_size = total N      → Batch (Full) Gradient Descent

        shuffle=True on training prevents the model from learning
        sample order instead of features.
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
