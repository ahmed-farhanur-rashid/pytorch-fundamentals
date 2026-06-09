import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from data import load_heart_disease, create_splits, create_data_loader
from model import Sigmoid
from loss_fn import get_loss_fn
from optimizer import get_optimizer
from train import train
from evaluate import evaluate_on_test
from plot import (plot_loss_and_accuracy,
                  plot_confusion_matrix,
                  plot_feature_importance)

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────

X, y, feature_names = load_heart_disease()
train_set, val_set, test_set, scaler = create_splits(X, y)

train_loader = create_data_loader(train_set, 32, shuffle=True)
val_loader = create_data_loader(val_set, 32, shuffle=False)
test_loader = create_data_loader(test_set, 32, shuffle=False)

# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────

model = Sigmoid(input_size=13)
loss_fn = get_loss_fn()
optimizer = get_optimizer(model, lr=0.1)

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

model, train_losses, train_accs, val_losses, val_accs = train(
    model, train_loader, loss_fn, optimizer,
    epochs=100, val_loader=val_loader
)

# ──────────────────────────────────────────────────────────
# Evaluate on Test Set
# ──────────────────────────────────────────────────────────

all_labels, all_preds = evaluate_on_test(model, test_loader)
test_acc = (all_labels == all_preds).mean()

# ──────────────────────────────────────────────────────────
# Visualize
# ──────────────────────────────────────────────────────────

os.makedirs(os.path.join(ROOT, "visualization"), exist_ok=True)

plot_loss_and_accuracy(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accs=train_accs,
    val_accs=val_accs,
    save_path=os.path.join(ROOT, "visualization", "loss_acc.png")
)

plot_confusion_matrix(
    cm=confusion_matrix(all_labels, all_preds),
    accuracy=test_acc,
    save_path=os.path.join(ROOT, "visualization", "confusion_matrix.png")
)

plot_feature_importance(
    model=model,
    feature_names=feature_names,
    save_path=os.path.join(ROOT, "visualization", "feature_importance.png")
)
