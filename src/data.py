import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


# ──────────────────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────────────────

class DiabetesDataset(Dataset):
    """
    The dataset has:
      - 442 patients
      - 10 input features: age, sex, bmi, blood pressure, and 6 blood serum values
      - 1 target: a number (25–346) representing disease progression one year later

    To be a valid PyTorch Dataset, a class MUST define:
      __len__     → returns how many samples there are
      __getitem__ → returns sample number `idx` as (X, y)
    """
    
    def __init__(self, csv_path: str, normalize_target: bool = True):
        df = pd.read_csv(csv_path)

        X = df.drop(columns=["target"]).values.astype(np.float32)
        y = df["target"].values.astype(np.float32).reshape(-1, 1)
        
        # Output is 1 column, hence we transform y into column vector/matrix

        self.target_min = float(y.min())
        self.target_max = float(y.max())

        # min-max normalization
        if normalize_target:
            y = (y - self.target_min) / (self.target_max - self.target_min)

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.n_features = self.X.shape[1]
    
    def __len__(self):
        return len(self.X)

    def __getItem__(self, idx):
        return self.X[idx], self.y[idx]

    def denormalize(self, normalized_y):
        return normalized_y * (self.target_max - self.target_min) + self.target_min


# ──────────────────────────────────────────────────────────
# Dataset Splitter
# ──────────────────────────────────────────────────────────

def split_dataset(dataset, train_split=0.8, test_split=0.1, validation_split=0.1, seed=42):

    n            = len(dataset)
    train_n      = int(n * train_split)
    validation_n = int(n * validation_split)
    test_n       = int(n * test_split)

    generator = torch.Generator().manual_seed(seed)
    train_ds, validation_ds, test_ds = random_split(
        dataset, [train_n, validation_n, test_n], generator
    )
    return train_ds, validation_ds, test_ds


# ──────────────────────────────────────────────────────────
# Dataset Data
# ──────────────────────────────────────────────────────────

def create_data_loader(dataset, batch_size=32, shuffle=False):
    """
    Wrap each split in a DataLoader so we can iterate over mini-batches.

    batch_size controls how many samples the model sees per weight update:
      batch_size = 1            → Stochastic Gradient Descent (SGD)
      1 < batch_size < total N  → Mini-Batch Gradient Descent
      batch_size = total N      → Batch (Full) Gradient Descent

    shuffle=True on the training loader prevents the model from
    learning the order of samples instead of their features.
    """

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)