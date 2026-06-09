import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def load_heart_disease():
    """
    Loads the Heart Disease dataset from UCI.

    Predicts whether a patient has heart disease based on 13 features.
    Target: 0 = no disease, 1 = has disease.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    df = pd.read_csv(url, names=column_names, na_values='?')
    df = df.dropna()

    # Convert target: 0 = no disease, 1-4 = disease
    df['target'] = (df['target'] > 0).astype(float)

    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.float32).reshape(-1, 1)

    return X, y, df.columns[:-1].tolist()


def create_splits(X, y, train_ratio=0.8, seed=42):
    """Splits into train, validation, and test sets."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        scaler
    )


def create_data_loader(dataset, batch_size=32, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
