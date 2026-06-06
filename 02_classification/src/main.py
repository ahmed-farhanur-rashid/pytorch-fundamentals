import torch
import os
from sklearn.metrics import confusion_matrix

from data      import load_mnist, create_data_loader
from model     import SoftMax
from loss_fn   import get_loss_fn
from optimizer import get_optimizer
from train     import train
from visualize import (
    plot_loss_and_accuracy,
    plot_parameters,
    plot_confusion_matrix,
    plot_sample_predictions
)

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cpu")

# ──────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────

train_dataset, test_dataset = load_mnist()
train_data_loader           = create_data_loader(train_dataset, 64, True)
test_data_loader            = create_data_loader(test_dataset,  64, False)

# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────

model     = SoftMax(input_size=28*28, output_size=10).to(device)
loss_fn   = get_loss_fn()
optimizer = get_optimizer(model)

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

model, train_losses, train_accs = train(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    epoch=25
)

# ──────────────────────────────────────────────────────────
# Evaluate on Test Set
# ──────────────────────────────────────────────────────────

model.eval()

all_preds  = []
all_labels = []
all_images = []

with torch.no_grad():
    for x, y in test_data_loader:
        x, y      = x.to(device), y.to(device)
        y_hat     = model(x.view(x.size(0), -1))
        predicted = torch.argmax(y_hat, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_images.extend(x.cpu())

# ──────────────────────────────────────────────────────────
# Visualize
# ──────────────────────────────────────────────────────────

plot_loss_and_accuracy(
    train_losses = train_losses,
    train_accs   = train_accs,
    save_path    = "./visualization/loss_acc.png"
)

plot_parameters(
    model     = model,
    save_path = "./visualization/weights.png"
)

plot_confusion_matrix(
    cm        = confusion_matrix(all_labels, all_preds),
    save_path = "./visualization/confusion_matrix.png"
)

plot_sample_predictions(
    images    = all_images,
    labels    = all_labels,
    preds     = all_preds,
    save_path = "./visualization/sample_predictions.png",
    n         = 10
)